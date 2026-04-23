from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Literal
import os

import numpy as np
from .domain import Domain
from .conversion import ConversionParams

from .reactions import HybridReactionSystem
from .update_hybrid_state import HybridState
from .pde import laplacian_1d, rk4_step
from .diffusion import SSADiffusion
from .gillespie import gillespie_draw
from .mass_calculations import MassProjector
from .results import SimulationResults

# Make sure these are imported from wherever they live in your package
# from .regime_utils import combined_mass, sufficient_pde_concentration_mask

PDETermsFn = Callable[[np.ndarray, Dict[str, float]], np.ndarray]
CDRemovalMode = Literal["standard", "probabilistic"]
# signature: pde_terms(C, rates) -> reaction-only dC/dt, shape (n_species, Npde)


@dataclass
class SRCMEngine:
    reactions: HybridReactionSystem
    pde_reaction_terms: PDETermsFn
    diffusion_rates: Dict[str, float]         # macroscopic diffusion D per species
    domain: Domain
    conversion: ConversionParams
    reaction_rates: Dict[str, float]          # dict passed into propensity lambdas + pde_terms
    cd_removal_mode: CDRemovalMode = "standard"

    def __post_init__(self):
        # basic validations
        if set(self.reactions.species) != set(self.diffusion_rates.keys()):
            raise ValueError("diffusion_rates keys must match reactions.species")
        if self.domain.K <= 1 and self.domain.boundary == "zero-flux":
            raise ValueError("zero-flux with K<=1 is degenerate (needs at least 2 compartments)")
        if self.cd_removal_mode not in ("standard", "probabilistic"):
            raise ValueError("cd_removal_mode must be 'standard' or 'probabilistic'")

        # Precompute PDE Laplacian once
        self._L = laplacian_1d(self.domain)

        # Precompute SSA jump rates per species: D / h^2
        jump_rates = {}
        for sp in self.reactions.species:
            D = float(self.diffusion_rates[sp])
            if D < 0:
                raise ValueError("diffusion rates must be >= 0")
            jump_rates[sp] = D / (self.domain.h ** 2)

        self._diffusion = SSADiffusion(species=self.reactions.species, jump_rates=jump_rates)

        # Useful mappings
        self._sp_to_idx = {sp: i for i, sp in enumerate(self.reactions.species)}
        
        #Mass calculatiion class
        self.MassProj = MassProjector(pde_multiple=self.domain.pde_multiple,dx=self.domain.dx, h = self.domain.h)
        
    # ------------------------------------------------------------------
    # PDE RHS: diffusion + reaction terms
    # ------------------------------------------------------------------
    def pde_rhs(self, C: np.ndarray, t: float) -> np.ndarray:
        """
        C shape: (n_species, Npde)
        returns dC/dt shape: (n_species, Npde)
        """
        n_species, Npde = C.shape
        out = np.zeros_like(C, dtype=float)

        # Diffusion part: (D/dx^2) * L @ C_s
        dx2 = self.domain.dx ** 2
        for sp, s_idx in self._sp_to_idx.items():
            D = float(self.diffusion_rates[sp])
            out[s_idx, :] = (D / dx2) * (self._L @ C[s_idx, :])

        # Reaction part (macroscopic)
        out += self.pde_reaction_terms(C, self.reaction_rates)
        return out

    # ------------------------------------------------------------------
    # Propensity building
    # ------------------------------------------------------------------
    def build_propensity_vector(
        self,
        state: HybridState,
        pde_mass: np.ndarray,
        DC_mask: np.ndarray,
        CD_mask: np.ndarray,
        sufficient_mask: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build propensity vector for:
        - diffusion blocks: n_species * K
        - CD blocks:        n_species * K
        - DC blocks:        n_species * K
        - hybrid blocks:    n_hybrid * K

        IMPORTANT (SRCM admissibility):
        Any event that REMOVES continuous mass (CD or hybrid with C_<sp> < 0)
        is only allowed if sufficient_mask[sp, i] == 1 in that compartment,
        unless cd_removal_mode == "probabilistic" for CD events.

        INPUT:
            state: HybridState container
            pde_mass: Integrated PDE mass per compartment, shape (n_species, K)
            DC_mask:  Mask (n_species, K), 1 means disc -> cont allowed
            CD_mask:  Mask (n_species, K), 1 means cont -> disc allowed
            sufficient_mask: Mask (n_species, K), 1 if ALL PDE cells in that
                             compartment have concentration >= (1/h), else 0
        """
        state.assert_consistent(self.domain)

        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)

        n_blocks = 3 * n_species + n_hybrid
        n_total = n_blocks * K

        if out is None:
            a = np.zeros(n_total, dtype=float)
        else:
            if out.shape != (n_total,):
                raise ValueError(f"out must have shape {(n_total,)}")
            a = out
            a.fill(0.0)

        # ------------------------------------------------------------------
        # 0..(n_species-1): SSA diffusion
        # ------------------------------------------------------------------
        self._diffusion.fill_propensities(a, state.ssa, K)

        # ------------------------------------------------------------------
        # n_species..(2*n_species-1): CD (PDE -> SSA)
        # ------------------------------------------------------------------
        for s_idx in range(n_species):
            block = n_species + s_idx
            start = block * K
            end = start + K

            available_mass = np.maximum(pde_mass[s_idx, :], 0.0)
            gamma = self.conversion.rate_for(s_idx)

            if self.cd_removal_mode == "probabilistic":
                # Relaxed gating: if in CD regime, allow even when sufficient_mask fails
                cd_allowed = (CD_mask[s_idx, :] == 1)
            else:
                # Standard gating: require enough concentration everywhere
                cd_allowed = (CD_mask[s_idx, :] == 1) & (sufficient_mask[s_idx, :] == 1)

            a[start:end] = gamma * available_mass * cd_allowed

        # ------------------------------------------------------------------
        # (2*n_species)..(3*n_species-1): DC (SSA -> PDE)
        # ------------------------------------------------------------------
        for s_idx in range(n_species):
            block = 2 * n_species + s_idx
            start = block * K
            end = start + K

            dc_allowed = (DC_mask[s_idx, :] == 1)
            gamma = self.conversion.rate_for(s_idx)
            a[start:end] = gamma * state.ssa[s_idx, :] * dc_allowed

        # ------------------------------------------------------------------
        # Hybrid reactions
        #
        # If hr consumes continuous mass (any C_<sp> delta < 0), only allow it
        # when sufficient_mask[sp, i] == 1 for all such species sp.
        # Also disallow if pde_mass is negative.
        # ------------------------------------------------------------------
        base_block = 3 * n_species
        h = float(self.domain.h)
        species = self.reactions.species

        for rxn_idx, hr in enumerate(self.reactions.hybrid_reactions):
            block = base_block + rxn_idx
            start = block * K

            consumed_species = [
                key.split("_", 1)[1]
                for key, delta in hr.state_change.items()
                if key.startswith("C_") and delta < 0
            ]
            consumes_C = len(consumed_species) > 0

            for i in range(K):
                if consumes_C:
                    ok = True
                    for sp in consumed_species:
                        s_idx = self._sp_to_idx[sp]

                        if sufficient_mask[s_idx, i] == 0:
                            ok = False
                            break

                        if pde_mass[s_idx, i] < 0.0:
                            ok = False
                            break

                    if not ok:
                        a[start + i] = 0.0
                        continue

                D_local = {sp: int(state.ssa[self._sp_to_idx[sp], i]) for sp in species}
                C_local = {sp: float(pde_mass[self._sp_to_idx[sp], i]) for sp in species}

                val = float(hr.propensity(D_local, C_local, self.reaction_rates, h))
                a[start + i] = val if val > 0.0 else 0.0

        return a

    # ------------------------------------------------------------------
    # CD event helpers
    # ------------------------------------------------------------------
    def _apply_cd_standard(
        self,
        state: HybridState,
        species_idx: int,
        comp: int,
    ) -> None:
        """
        Standard CD removal:
        add one SSA particle and remove one particle mass uniformly
        through the existing state helper.
        """
        state.add_discrete(species_idx, comp, +1)
        state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)

    # def _apply_cd_probabilistic(
    #     self,
    #     state: HybridState,
    #     species_idx: int,
    #     comp: int,
    #     rng: np.random.Generator,
    # ) -> None:
    #     """
    #     Probabilistic CD removal:
    #     - if compartment has mass Y_k < 1, clear all PDE mass and create one
    #       SSA particle with probability Y_k
    #     - if Y_k >= 1, remove exactly one particle mass from the PDE slice
    #       using either uniform removal or the fallback redistribution scheme
    #     """
    #     P = self.domain.pde_multiple
    #     start_index = comp * P
    #     final_index = start_index + P
    #     slice_ = state.pde[species_idx, start_index:final_index]

    #     dx = self.domain.dx
    #     h = self.domain.h
    #     drop = 1.0 / h

    #     Y_k = float(slice_.sum() * dx)

    #     # Safety for tiny negative roundoff
    #     if Y_k <= 0.0:
    #         slice_[:] = np.maximum(slice_, 0.0)
    #         return

    #     # Case 1: less than one particle worth of mass
    #     if Y_k < 1.0:
    #         p = min(max(Y_k, 0.0), 1.0)
    #         slice_[:] = 0.0
    #         if float(rng.random()) < p:
    #             state.add_discrete(species_idx, comp, +1)
    #         return

    #     # Case 2: at least one particle worth of mass
    #     # try simple uniform removal first
    #     if np.all(slice_ >= drop):
    #         state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)
    #         state.add_discrete(species_idx, comp, +1)
    #         return

    #     # Fallback: zero out smaller cells and remove the rest from larger cells
    #     remaining = 1.0
    #     active = np.ones(P, dtype=bool)

    #     while remaining > 1e-12:
    #         n_active = int(np.sum(active))
    #         if n_active == 0:
    #             raise RuntimeError("Not enough PDE mass to remove one particle")

    #         share = remaining / n_active
    #         changed = False

    #         for j in range(P):
    #             if not active[j]:
    #                 continue

    #             removable = float(slice_[j] * dx)

    #             if removable <= share + 1e-12:
    #                 remaining -= removable
    #                 slice_[j] = 0.0
    #                 active[j] = False
    #                 changed = True

    #         if changed:
    #             continue

    #         # every active cell can support the equal share
    #         slice_[active] -= share / dx
    #         remaining = 0.0

    #     state.add_discrete(species_idx, comp, +1)




    # def _apply_cd_probabilistic(
    #     self,
    #     state: HybridState,
    #     species_idx: int,
    #     comp: int,
    #     rng: np.random.Generator,
    # ) -> None:
    #     P = self.domain.pde_multiple
    #     start = comp * P
    #     end = start + P
    #     slice_ = state.pde[species_idx, start:end]
    #     dx = self.domain.dx
    #     h = self.domain.h

    #     Y_k = slice_.sum() * dx   # total mass in compartment

    #     # Safety for tiny negative roundoff
    #     if Y_k <= 0.0:
    #         slice_[:] = 0.0
    #         return

    #     # Case 1: less than one particle -> probabilistic conversion or discard
    #     if Y_k < 1.0:
    #         p = min(max(Y_k, 0.0), 1.0)
    #         slice_[:] = 0.0
    #         if rng.random() < p:
    #             state.add_discrete(species_idx, comp, +1)
    #         return

    #     # Case 2: at least one particle worth of mass
    #     # Try uniform slab removal (fast path for well‑mixed compartments)
    #     drop = 1.0 / h
    #     if np.all(slice_ >= drop - 1e-12):
    #         state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)
    #         state.add_discrete(species_idx, comp, +1)
    #         return

    #     # Fallback: proportional scaling (preserves shape)
    #     alpha = (Y_k - 1.0) / Y_k   # factor to remove exactly one particle
    #     slice_[:] *= alpha
    #     state.add_discrete(species_idx, comp, +1)


    #THis only removes if less than 1./h
    # def _apply_cd_probabilistic(
    #     self,
    #     state: HybridState,
    #     species_idx: int,
    #     comp: int,
    #     rng: np.random.Generator,
    # ) -> None:
    #     P = self.domain.pde_multiple
    #     start = comp * P
    #     end = start + P
    #     slice_ = state.pde[species_idx, start:end]

    #     dx = self.domain.dx
    #     h = self.domain.h
    #     drop = 1.0 / h

    #     Y_k = float(slice_.sum() * dx)

    #     # tiny roundoff protection
    #     if Y_k <= 0.0:
    #         slice_[:] = np.maximum(slice_, 0.0)
    #         return

    #     # Only new feature: probabilistic handling of sub-particle mass
    #     if Y_k < 1.0:
    #         p = min(max(Y_k, 0.0), 1.0)
    #         slice_[:] = 0.0
    #         if rng.random() < p:
    #             state.add_discrete(species_idx, comp, +1)
    #         return

    #     # For Y_k >= 1, keep the old behaviour exactly
    #     if np.all(slice_ >= drop - 1e-12):
    #         state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)
    #         state.add_discrete(species_idx, comp, +1)
    #         return

    #     # old fallback
    #     remaining = 1.0
    #     active = np.ones(P, dtype=bool)

    #     while remaining > 1e-12:
    #         n_active = int(np.sum(active))
    #         if n_active == 0:
    #             raise RuntimeError("Not enough PDE mass to remove one particle")

    #         share = remaining / n_active
    #         changed = False

    #         for j in range(P):
    #             if not active[j]:
    #                 continue

    #             removable = float(slice_[j] * dx)

    #             if removable <= share + 1e-12:
    #                 remaining -= removable
    #                 slice_[j] = 0.0
    #                 active[j] = False
    #                 changed = True

    #         if changed:
    #             continue

    #         slice_[active] -= share / dx
    #         remaining = 0.0

    #     state.add_discrete(species_idx, comp, +1)

    def _apply_cd_probabilistic(
        self,
        state: HybridState,
        species_idx: int,
        comp: int,
        rng: np.random.Generator,
    ) -> None:
        P = self.domain.pde_multiple
        start_index = comp * P
        slice_ = state.pde[species_idx, start_index:start_index + P]
        dx = self.domain.dx
        h = self.domain.h

        Y_k = float(slice_.sum() * dx)

        if Y_k >= 1.0:
            # Only allow standard removal if every cell has >= 1/h
            # If any cell is below 1/h, block the event entirely
            if np.all(slice_ >= 1.0 / h):
                state.add_discrete(species_idx, comp, +1)
                state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)
            return

        # Sub-particle regime: absorb stochastically, always clear the slice
        if Y_k > 0.0 and float(rng.random()) < Y_k:
            state.add_discrete(species_idx, comp, +1)
        slice_[:] = 0.0


    def apply_event(
        self,
        idx: int,
        state: HybridState,
        rng: np.random.Generator,
        pde_mass: np.ndarray,
    ) -> None:
        """
        Apply the event at flat propensity index `idx` to `state` in-place.

        Block layout (per compartment, length K):
          diffusion blocks:  [0 .. n_species-1]
          CD blocks:         [n_species .. 2*n_species-1]
          DC blocks:         [2*n_species .. 3*n_species-1]
          hybrid blocks:     [3*n_species .. 3*n_species+n_hybrid-1]

        Notes
        -----
        - For diffusion we use a third uniform u3 to pick direction.
        - For CD/DC we add/remove exactly 1 particle, and adjust PDE slice by ±(1/h).
        - For hybrid reactions we use hr.state_change:
            'D_<sp>' updates SSA
            'C_<sp>' updates PDE slice by ±1 particle mass
        """
        state.assert_consistent(self.domain)

        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)

        if idx < 0:
            return
        if idx >= (3 * n_species + n_hybrid) * K:
            raise ValueError("idx out of range for propensity vector")

        block = idx // K
        comp = idx % K

        # -------------------- Diffusion --------------------
        if block < n_species:
            species_idx = block
            u3 = float(rng.random())
            self._diffusion.apply_move(state.ssa, self.domain, species_idx, comp, u=u3)
            return

        # -------------------- CD: PDE -> SSA --------------------
        if block < 2 * n_species:
            species_idx = block - n_species

            if self.cd_removal_mode == "probabilistic":
                self._apply_cd_probabilistic(state, species_idx, comp, rng)
            else:
                self._apply_cd_standard(state, species_idx, comp)
            return

        # -------------------- DC: SSA -> PDE --------------------
        if block < 3 * n_species:
            species_idx = block - 2 * n_species

            state.add_discrete(species_idx, comp, -1)
            state.add_continuous_particle_mass(self.domain, species_idx, comp, +1)
            return

        # -------------------- Hybrid reactions --------------------
        rxn_idx = block - 3 * n_species
        hr = self.reactions.hybrid_reactions[rxn_idx]

        for key, delta in hr.state_change.items():
            prefix, sp = key.split("_", 1)
            species_idx = self._sp_to_idx[sp]

            if prefix == "D":
                state.add_discrete(species_idx, comp, int(delta))
            elif prefix == "C":
                state.add_continuous_particle_mass(self.domain, species_idx, comp, int(delta))
            else:
                raise ValueError(f"Unknown state_change key prefix '{prefix}'")

    def _simulate_interval(
        self,
        state: HybridState,
        t0: float,
        t1: float,
        rng: np.random.Generator,
        propensity: np.ndarray,
        cumulative: np.ndarray,
    ) -> None:
        """
        Simulate SSA/conversion/hybrid events in [t0, t1) holding PDE fixed.
        Updates `state` in-place.
        """
        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)

        while t0 < t1:
            # masses + masks
           


            comb, pde_mass = self.MassProj.get_combined_total_mass(ssa_counts=state.ssa, pde_conc=state.pde)
            DC_mask = self.conversion.DC_mask(comb)
            CD_mask = self.conversion.CD_mask(comb)

            sufficient = self.MassProj.get_high_concentration_mask(pde_conc=state.pde)

            # propensities
            self.build_propensity_vector(
                state,
                pde_mass,
                DC_mask,
                CD_mask,
                sufficient,
                out=propensity,
            )

            # DEBUG: decode first negative propensity
            if np.any(propensity < 0):
                j = int(np.where(propensity < 0)[0][0])
                block = j // K
                comp = j % K

                n_blocks = 3 * n_species + n_hybrid

                print("\n" + "=" * 80)
                print("NEGATIVE PROPENSITY")
                print(f"t0={t0}, t1={t1}")
                print(f"flat idx={j}, value={float(propensity[j])}")
                print(f"block={block}/{n_blocks-1}, compartment={comp}/{K-1}")

                if block < n_species:
                    print("Section: diffusion")
                    print("species:", self.reactions.species[block])

                elif block < 2 * n_species:
                    print("Section: CD (PDE -> SSA)")
                    print("species:", self.reactions.species[block - n_species])
                    print("pde_mass local:", pde_mass[:, comp])
                    print("sufficient local:", sufficient[:, comp])
                    print("CD_mask local:", CD_mask[:, comp])

                elif block < 3 * n_species:
                    print("Section: DC (SSA -> PDE)")
                    print("species:", self.reactions.species[block - 2 * n_species])
                    print("ssa local:", state.ssa[:, comp])
                    print("DC_mask local:", DC_mask[:, comp])

                else:
                    rxn_idx = block - 3 * n_species
                    hr = self.reactions.hybrid_reactions[rxn_idx]
                    print("Section: hybrid reaction")
                    print("rxn_idx:", rxn_idx)
                    print("label:", hr.label)
                    print("state_change:", hr.state_change)
                    if getattr(hr, "description", None):
                        print("desc:", hr.description)

                    species = self.reactions.species
                    D_local = {sp: int(state.ssa[self._sp_to_idx[sp], comp]) for sp in species}
                    C_local = {sp: float(pde_mass[self._sp_to_idx[sp], comp]) for sp in species}
                    print("D_local:", D_local)
                    print("C_local:", C_local)

                print("SSA local:", state.ssa[:, comp])
                s = comp * self.domain.pde_multiple
                e = s + self.domain.pde_multiple
                print("PDE slice mins:", np.min(state.pde[:, s:e], axis=1))
                print("=" * 80 + "\n")

                raise ValueError("Negative propensity detected (decoded above)")

            # Gillespie draw
            tau, idx = gillespie_draw(propensity, rng, cumulative=cumulative)

            if not np.isfinite(tau):
                return

            if t0 + tau < t1:
                t0 += float(tau)
                self.apply_event(idx, state, rng, pde_mass=pde_mass)
            else:
                return

    def run(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        seed: int = 0,
    ):
        """
        Run one simulation and return full time series results.
        """

        if time <= 0:
            raise ValueError("time must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        rng = np.random.default_rng(seed)

        n_species = self.reactions.n_species
        K = self.domain.K
        Npde = self.domain.n_pde

        if initial_pde is None:
            initial_pde = np.zeros((n_species, Npde), dtype=float)

        state = HybridState(
            ssa=initial_ssa.copy().astype(int),
            pde=initial_pde.copy().astype(float),
        )
        state.assert_consistent(self.domain)

        # time grid (include t=0)
        n_steps = int(np.floor(time / dt)) + 1
        tvec = np.arange(n_steps, dtype=float) * dt

        # output arrays
        ssa_out = np.zeros((n_species, K, n_steps), dtype=int)
        pde_out = np.zeros((n_species, Npde, n_steps), dtype=float)

        # record initial
        ssa_out[:, :, 0] = state.ssa
        pde_out[:, :, 0] = state.pde

        # allocate propensity + cumulative once
        n_hybrid = len(self.reactions.hybrid_reactions)
        n_blocks = 3 * n_species + n_hybrid
        propensity = np.zeros(n_blocks * K, dtype=float)
        cumulative = np.empty_like(propensity)

        # main loop over PDE ticks
        for n in range(n_steps - 1):
            t0 = tvec[n]
            t1 = tvec[n + 1]

            # stochastic evolution on [t0, t1) with PDE frozen
            self._simulate_interval(state, t0, t1, rng, propensity, cumulative)

            # deterministic PDE step to t1
            state.pde = rk4_step(state.pde, t=t0, dt=dt, rhs=self.pde_rhs)

            # record at t1
            ssa_out[:, :, n + 1] = state.ssa
            pde_out[:, :, n + 1] = state.pde

        return SimulationResults(
            time=tvec,
            ssa=ssa_out,
            pde=pde_out,
            domain=self.domain,
            species=self.reactions.species,
        )

    def run_repeats(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        repeats: int,
        seed: int = 0,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
        prefer: str = "processes",
        progress: bool = True,
    ):
        """
        Run multiple independent simulations and return mean SSA/PDE time series.

        parallel=True uses joblib to spread repeats across CPU cores.
        n_jobs=-1 uses all available cores.
        """
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        # First run locally for shapes/time vector
        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        ssa_sum = res0.ssa.astype(float)
        pde_sum = res0.pde.astype(float)

        if repeats == 1:
            return SimulationResults(
                time=res0.time,
                ssa=ssa_sum,
                pde=pde_sum,
                domain=self.domain,
                species=self.reactions.species,
            )

        # -----------------------
        # Serial repeats
        # -----------------------
        if not parallel:
            iterator = range(1, repeats)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SRCM repeats",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                ssa_sum += res_r.ssa
                pde_sum += res_r.pde

        # -----------------------
        # Parallel repeats (joblib)
        # -----------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            if progress:
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    tqdm = None
                    progress = False

            def one(r: int):
                res = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                return res.ssa.astype(float), res.pde.astype(float)

            tasks = (delayed(one)(r) for r in range(1, repeats))

            if n_jobs == -1:
                n_workers = os.cpu_count() or 1
            else:
                n_workers = n_jobs

            print(f"Running SRCM repeats in parallel on {n_workers} core(s)")

            par = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")
            results_iter = par(tasks)

            if progress and tqdm is not None:
                results_iter = tqdm(
                    results_iter,
                    total=repeats - 1,
                    desc="SRCM repeats",
                    unit="run",
                    dynamic_ncols=True,
                )

            for ssa_r, pde_r in results_iter:
                ssa_sum += ssa_r
                pde_sum += pde_r

        ssa_mean = ssa_sum / repeats
        pde_mean = pde_sum / repeats

        return SimulationResults(
            time=res0.time,
            ssa=ssa_mean,
            pde=pde_mean,
            domain=self.domain,
            species=self.reactions.species,
        )

    def run_trajectories(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        repeats: int,
        seed: int = 0,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
        prefer: str = "processes",
        progress: bool = True,
    ):
        """
        Run multiple independent simulations and return individual SSA/PDE trajectories.

        Returns arrays with leading dimension = repeats.
        """
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        # First run for shapes + shared time vector
        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        time_vec = res0.time

        # Pre-allocate (fast + consistent shapes)
        ssa_all = np.empty((repeats,) + res0.ssa.shape, dtype=float)
        pde_all = np.empty((repeats,) + res0.pde.shape, dtype=float)

        ssa_all[0] = res0.ssa.astype(float)
        pde_all[0] = res0.pde.astype(float)

        if repeats == 1:
            return SimulationResults(
                time=time_vec,
                ssa=ssa_all,
                pde=pde_all,
                domain=self.domain,
                species=self.reactions.species,
            )

        # -----------------------
        # Serial
        # -----------------------
        if not parallel:
            iterator = range(1, repeats)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SRCM trajectories",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                ssa_all[r] = res_r.ssa.astype(float)
                pde_all[r] = res_r.pde.astype(float)

        # -----------------------
        # Parallel
        # -----------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            if progress:
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    tqdm = None
                    progress = False

            def one(r: int):
                res = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                return r, res.ssa.astype(float), res.pde.astype(float)

            tasks = (delayed(one)(r) for r in range(1, repeats))
            results_iter = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(tasks)

            if progress and tqdm is not None:
                results_iter = tqdm(
                    results_iter,
                    total=repeats - 1,
                    desc="SRCM trajectories",
                    unit="run",
                    dynamic_ncols=True,
                )

            for r, ssa_r, pde_r in results_iter:
                ssa_all[r] = ssa_r
                pde_all[r] = pde_r

        return SimulationResults(
            time=time_vec,
            ssa=ssa_all,
            pde=pde_all,
            domain=self.domain,
            species=self.reactions.species,
        )

    def run_repeats_final(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        repeats: int,
        seed: int = 0,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
        prefer: str = "processes",
        progress: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Run multiple independent simulations and return ONLY the final frame from each repeat.

        Returns
        -------
        final_ssa : np.ndarray
            Shape (repeats, n_species, K), dtype int
        final_pde : np.ndarray
            Shape (repeats, n_species, Npde), dtype float
        t_final : float
            Final time recorded (typically floor(time/dt)*dt)
        """
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        # One run to get shapes + final time
        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        n_species, K, _ = res0.ssa.shape
        _, Npde, _ = res0.pde.shape
        t_final = float(res0.time[-1])

        final_ssa = np.empty((repeats, n_species, K), dtype=int)
        final_pde = np.empty((repeats, n_species, Npde), dtype=float)

        final_ssa[0] = res0.ssa[:, :, -1]
        final_pde[0] = res0.pde[:, :, -1]

        if repeats == 1:
            if save_path is not None:
                np.savez_compressed(
                    save_path,
                    final_ssa=final_ssa,
                    final_pde=final_pde,
                    t_final=t_final,
                    species=np.array(self.reactions.species, dtype=object),
                )
            return final_ssa, final_pde, t_final

        # -----------------------
        # Serial
        # -----------------------
        if not parallel:
            iterator = range(1, repeats)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SRCM repeats (final only)",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                final_ssa[r] = res_r.ssa[:, :, -1]
                final_pde[r] = res_r.pde[:, :, -1]

        # -----------------------
        # Parallel
        # -----------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            if progress:
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    tqdm = None
                    progress = False

            def one(r: int):
                res = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                return r, res.ssa[:, :, -1].astype(int), res.pde[:, :, -1].astype(float)

            tasks = (delayed(one)(r) for r in range(1, repeats))
            par = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")
            results_iter = par(tasks)

            if progress and tqdm is not None:
                results_iter = tqdm(
                    results_iter,
                    total=repeats - 1,
                    desc="SRCM repeats (final only)",
                    unit="run",
                    dynamic_ncols=True,
                )

            for r, ssa_last, pde_last in results_iter:
                final_ssa[r] = ssa_last
                final_pde[r] = pde_last

        if save_path is not None:
            np.savez_compressed(
                save_path,
                final_ssa=final_ssa,
                final_pde=final_pde,
                t_final=t_final,
                species=np.array(self.reactions.species, dtype=object),
            )

        return final_ssa, final_pde, t_final