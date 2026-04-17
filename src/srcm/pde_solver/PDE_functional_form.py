from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from ..hybrid_solver import Domain, SimulationResults


PurePDERHS = Callable[..., tuple[np.ndarray, ...] | np.ndarray]


@dataclass
class PDEEngine:
    """
    Deterministic pure-PDE solver for 1D reaction-diffusion systems.

    Notes
    -----
    - Supports 1 or 2 species.
    - Returns SimulationResults in the same convention as SRCM:
        ssa: (S, K, T)   [filled with zeros for pure PDE mode]
        pde: (S, Npde, T)
    - The PDE reaction function here should represent the FULL deterministic PDE
      system, not the hybrid split PDE terms.
    """

    species: list[str]
    pde_reaction_terms: PurePDERHS
    diffusion_rates: Dict[str, float]
    domain: Domain
    reaction_rates: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("species must be non-empty")
        if len(self.species) > 2:
            raise ValueError("PDEEngine currently supports at most 2 species")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be unique")

        if set(self.species) != set(self.diffusion_rates.keys()):
            raise ValueError("diffusion_rates keys must match species")

        for sp, D in self.diffusion_rates.items():
            if float(D) < 0:
                raise ValueError(f"Diffusion rate for {sp!r} must be >= 0")

        self._sp_to_idx = {sp: i for i, sp in enumerate(self.species)}
        self._L = self._build_laplacian_1d(self.domain)

    # ------------------------------------------------------------------
    # spatial operator
    # ------------------------------------------------------------------
    @staticmethod
    def _build_laplacian_1d(domain: Domain) -> np.ndarray:
        N = domain.n_pde
        if N <= 2:
            raise ValueError("Pure PDE requires n_pde > 2")

        A = np.zeros((N, N), dtype=float)

        if domain.boundary == "periodic":
            for i in range(N):
                A[i, i] = -2.0
                A[i, (i - 1) % N] = 1.0
                A[i, (i + 1) % N] = 1.0
            return A

        if domain.boundary == "zero-flux":
            for i in range(N):
                if i == 0:
                    A[i, i] = -1.0
                    A[i, i + 1] = 1.0
                elif i == N - 1:
                    A[i, i] = -1.0
                    A[i, i - 1] = 1.0
                else:
                    A[i, i] = -2.0
                    A[i, i - 1] = 1.0
                    A[i, i + 1] = 1.0
            return A

        raise ValueError(f"Unknown boundary {domain.boundary!r}")

    # ------------------------------------------------------------------
    # rhs
    # ------------------------------------------------------------------
    def pde_rhs(self, C: np.ndarray, t: float) -> np.ndarray:
        """
        Parameters
        ----------
        C : np.ndarray
            Shape (n_species, Npde)

        Returns
        -------
        np.ndarray
            Shape (n_species, Npde)
        """
        n_species, Npde = C.shape
        expected = (len(self.species), self.domain.n_pde)
        if C.shape != expected:
            raise ValueError(f"C must have shape {expected}, got {C.shape}")

        out = np.zeros_like(C, dtype=float)

        dx2 = self.domain.dx ** 2
        for sp, s_idx in self._sp_to_idx.items():
            D = float(self.diffusion_rates[sp])
            out[s_idx, :] = (D / dx2) * (self._L @ C[s_idx, :])

        args = [C[i] for i in range(n_species)]
        reaction_part = self.pde_reaction_terms(*args, self.reaction_rates)

        if isinstance(reaction_part, np.ndarray):
            reaction_arr = reaction_part.astype(float, copy=False)
        else:
            reaction_arr = np.array(tuple(reaction_part), dtype=float)

        if reaction_arr.shape != C.shape:
            raise ValueError(
                f"pde_reaction_terms returned shape {reaction_arr.shape}, expected {C.shape}"
            )

        out += reaction_arr
        return out

    # ------------------------------------------------------------------
    # rk4 step
    # ------------------------------------------------------------------
    def rk4_step(self, C: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.pde_rhs(C, t)
        k2 = self.pde_rhs(C + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.pde_rhs(C + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.pde_rhs(C + dt * k3, t + dt)
        return C + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        initial_pde: np.ndarray,
        time: float,
        dt: float,
    ) -> SimulationResults:
        """
        Run one deterministic PDE simulation.

        Parameters
        ----------
        initial_pde : np.ndarray
            Shape (n_species, Npde)
        time : float
        dt : float
        """
        if time <= 0:
            raise ValueError("time must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        n_species = len(self.species)
        K = self.domain.K
        Npde = self.domain.n_pde

        initial_pde = np.asarray(initial_pde, dtype=float)
        if initial_pde.shape != (n_species, Npde):
            raise ValueError(
                f"initial_pde must have shape {(n_species, Npde)}, got {initial_pde.shape}"
            )

        n_steps = int(np.floor(time / dt)) + 1
        tvec = np.arange(n_steps, dtype=float) * dt

        pde_out = np.zeros((n_species, Npde, n_steps), dtype=float)
        ssa_out = np.zeros((n_species, K, n_steps), dtype=float)

        C = initial_pde.copy()
        pde_out[:, :, 0] = C

        for n in range(n_steps - 1):
            C = self.rk4_step(C, tvec[n], dt)
            pde_out[:, :, n + 1] = C

        return SimulationResults(
            time=tvec,
            ssa=ssa_out,
            pde=pde_out,
            domain=self.domain,
            species=list(self.species),
        )