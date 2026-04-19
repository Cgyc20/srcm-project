from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from .reaction import SSAReaction
from ..hybrid_solver import Domain, SimulationResults


@dataclass
class SSAEngine:
    """
    Pure SSA backend aligned with SRCM conventions.

    Public output conventions
    -------------------------
    Single / mean run:
        ssa shape = (n_species, K, T)

    Trajectories:
        ssa shape = (R, n_species, K, T)

    Pure SSA has no PDE field, so returned SimulationResults use:
        pde = None
    """

    reaction_system: SSAReaction
    diffusion_rates: dict[str, float]
    domain: Domain

    def __post_init__(self):
        self.species = list(self.reaction_system.species_list)
        self.n_species = len(self.species)
        self.species_index = self.reaction_system.species_index
        self.stoichiometric_matrix = self.reaction_system.stoichiometric_matrix
        self.number_of_reactions = self.reaction_system.number_of_reactions
        self.reaction_set = self.reaction_system.reaction_set

        if set(self.species) != set(self.diffusion_rates.keys()):
            raise ValueError("diffusion_rates keys must match reaction_system.species_list")

        self.K = int(self.domain.K)
        self.L = float(self.domain.length)
        self.h = float(self.domain.h)
        self.boundary_conditions = str(self.domain.boundary)

        if self.boundary_conditions not in ("periodic", "zero-flux"):
            raise ValueError("boundary must be 'periodic' or 'zero-flux'")

        self.jump_rate_list = [
            float(self.diffusion_rates[sp]) / (self.h ** 2)
            for sp in self.species
        ]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _timevector(self, time: float, dt: float) -> np.ndarray:
        if time <= 0:
            raise ValueError("time must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        return np.arange(0.0, time, dt, dtype=float)

    def _initial_tensor(self, time: float, dt: float, initial_ssa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal tensor ordering remains (T, S, K) for the Gillespie loop,
        but all public outputs are converted to (S, K, T).
        """
        tvec = self._timevector(time, dt)

        if initial_ssa.shape != (self.n_species, self.K):
            raise ValueError(
                f"initial_ssa must have shape {(self.n_species, self.K)}, got {initial_ssa.shape}"
            )

        tensor = np.zeros((len(tvec), self.n_species, self.K), dtype=int)
        tensor[0, :, :] = initial_ssa.astype(int, copy=False)
        return tvec, tensor

    def _propensity_calculation(
        self,
        frame: np.ndarray,              # shape (S, K)
        propensity_vector: np.ndarray,  # shape (S*K + R*K,)
    ) -> np.ndarray:
        if frame.shape != (self.n_species, self.K):
            raise ValueError(
                f"frame must have shape {(self.n_species, self.K)}, got {frame.shape}"
            )

        # diffusion blocks
        # for species_index in range(self.n_species):
        #     start_idx = species_index * self.K
        #     end_idx = (species_index + 1) * self.K
        #     propensity_vector[start_idx:end_idx] = (
        #         self.jump_rate_list[species_index] * frame[species_index, :] * 2.0
        #     )

        for species_index in range(self.n_species):
            start_idx = species_index * self.K
            end_idx = (species_index + 1) * self.K
            n = frame[species_index, :]

            if self.boundary_conditions == "periodic":
                propensity_vector[start_idx:end_idx] = (
                    2.0 * self.jump_rate_list[species_index] * n
                )

            else:  # zero-flux
                if self.K == 1:
                    propensity_vector[start_idx:end_idx] = 0.0
                else:
                    rates = 2.0 * self.jump_rate_list[species_index] * n
                    rates[0] = self.jump_rate_list[species_index] * n[0]
                    rates[-1] = self.jump_rate_list[species_index] * n[-1]
                    propensity_vector[start_idx:end_idx] = rates
                
        # reaction blocks
        for i, reaction in enumerate(self.reaction_set):
            start = self.K * self.n_species + i * self.K
            end = start + self.K

            reaction_type = reaction["reaction_type"]
            reactant_indices = reaction["reactant_indices"]
            rate = reaction["reaction_rate"]

            if reaction_type == "zero_order":
                propensity_vector[start:end] = rate * self.h

            elif reaction_type == "first_order":
                idx = reactant_indices[0]
                propensity_vector[start:end] = rate * frame[idx, :]

            elif reaction_type == "second_order":
                if len(reactant_indices) == 1:
                    idx = reactant_indices[0]
                    propensity_vector[start:end] = (
                        rate * frame[idx, :] * (frame[idx, :] - 1) / self.h
                    )
                else:
                    idx1, idx2 = reactant_indices
                    propensity_vector[start:end] = (
                        rate * frame[idx1, :] * frame[idx2, :] / self.h
                    )
            else:
                raise ValueError(f"Unknown reaction type {reaction_type}")

        return propensity_vector

    def _ssa_loop(
        self,
        time: float,
        dt: float,
        initial_ssa: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        tvec : (T,)
        tensor : (T, S, K)   internal ordering only
        """
        tvec, tensor = self._initial_tensor(time, dt, initial_ssa)
        current_frame = initial_ssa.copy().astype(int)

        propensity_vector = np.zeros(
            self.K * self.n_species + self.K * self.number_of_reactions,
            dtype=float,
        )

        t = 0.0

        while t < time:
            propensity_vector = self._propensity_calculation(current_frame, propensity_vector)

            # alpha0 = float(np.sum(propensity_vector))
            # if alpha0 <= 0.0:
            #     break

            alpha0 = float(np.sum(propensity_vector))
            if alpha0 <= 0.0:
                ind_now = int(t / dt)
                for time_index in range(ind_now + 1, len(tvec)):
                    tensor[time_index, :, :] = current_frame
                break

            r1, r2, r3 = rng.random(3)
            tau = (1.0 / alpha0) * np.log(1.0 / r1)

            alpha_cum = np.cumsum(propensity_vector)
            index = int(np.searchsorted(alpha_cum, r2 * alpha0))
            compartment_index = index % self.K

            old_time = t
            t += tau

            if index < self.n_species * self.K:
                # diffusion event
                species_index = index // self.K

                if self.boundary_conditions == "periodic":
                    if r3 < 0.5:
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, (compartment_index - 1) % self.K] += 1
                    else:
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, (compartment_index + 1) % self.K] += 1

                else:  # zero-flux
                    if r3 < 0.5:
                        # left attempt
                        if compartment_index == 0:
                            # reflect right
                            if self.K > 1:
                                current_frame[species_index, compartment_index] -= 1
                                current_frame[species_index, compartment_index + 1] += 1
                        elif compartment_index == self.K - 1:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index - 1] += 1
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index - 1] += 1
                    else:
                        # right attempt
                        if compartment_index == self.K - 1:
                            if self.K > 1:
                                current_frame[species_index, compartment_index] -= 1
                                current_frame[species_index, compartment_index - 1] += 1
                        elif compartment_index == 0:
                            if self.K > 1:
                                current_frame[species_index, compartment_index] -= 1
                                current_frame[species_index, compartment_index + 1] += 1
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index + 1] += 1

            else:
                # reaction event
                reaction_index = (index - self.n_species * self.K) // self.K
                stoich = self.stoichiometric_matrix[:, reaction_index]
                current_frame[:, compartment_index] += stoich

            # record at regular dt intervals
            ind_before = int(old_time / dt)
            ind_after = int(t / dt)

            if ind_after > ind_before:
                for time_index in range(ind_before + 1, min(ind_after + 1, len(tvec))):
                    tensor[time_index, :, :] = current_frame

        return tvec, tensor

    def _resolve_n_jobs(self, n_jobs: int) -> int:
        cpu = os.cpu_count() or 1
        if n_jobs == -1:
            return cpu
        if n_jobs <= 0:
            raise ValueError("n_jobs must be a positive int or -1")
        return min(int(n_jobs), cpu)

    def _run_one_repeat(
        self,
        initial_ssa: np.ndarray,
        time: float,
        dt: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        return self._ssa_loop(time=time, dt=dt, initial_ssa=initial_ssa, rng=rng)

    # ------------------------------------------------------------------
    # public API aligned with SRCMEngine
    # ------------------------------------------------------------------
    def run(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        seed: int = 0,
    ) -> SimulationResults:
        """
        Pure SSA ignores initial_pde.
        """
        tvec, tensor_tsk = self._run_one_repeat(
            initial_ssa=initial_ssa,
            time=float(time),
            dt=float(dt),
            seed=int(seed),
        )

        # convert (T, S, K) -> (S, K, T)
        ssa_out = np.transpose(tensor_tsk, (1, 2, 0))

        return SimulationResults(
            time=tvec,
            ssa=ssa_out,
            pde=None,
            domain=self.domain,
            species=self.species,
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
    ) -> SimulationResults:
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        ssa_sum = res0.ssa.astype(float)

        if repeats == 1:
            return SimulationResults(
                time=res0.time,
                ssa=ssa_sum,
                pde=None,
                domain=self.domain,
                species=self.species,
            )

        if not parallel:
            iterator = range(1, repeats)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SSA repeats",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                ssa_sum += res_r.ssa

        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            tasks = (
                delayed(self._run_one_repeat)(
                    initial_ssa=initial_ssa,
                    time=float(time),
                    dt=float(dt),
                    seed=seed + r,
                )
                for r in range(1, repeats)
            )

            results_iter = Parallel(
                n_jobs=self._resolve_n_jobs(n_jobs),
                prefer=prefer,
                return_as="generator",
            )(tasks)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    results_iter = tqdm(
                        results_iter,
                        total=repeats - 1,
                        desc="SSA repeats",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for _, tensor_tsk in results_iter:
                ssa_sum += np.transpose(tensor_tsk, (1, 2, 0))

        ssa_mean = ssa_sum / repeats

        return SimulationResults(
            time=res0.time,
            ssa=ssa_mean,
            pde=None,
            domain=self.domain,
            species=self.species,
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
    ) -> SimulationResults:
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        time_vec = res0.time

        # output shape: (R, S, K, T)
        ssa_all = np.empty((repeats,) + res0.ssa.shape, dtype=float)
        ssa_all[0] = res0.ssa.astype(float)

        if repeats == 1:
            return SimulationResults(
                time=time_vec,
                ssa=ssa_all,
                pde=None,
                domain=self.domain,
                species=self.species,
            )

        if not parallel:
            iterator = range(1, repeats)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SSA trajectories",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                ssa_all[r] = res_r.ssa.astype(float)

        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            tasks = (
                delayed(self._run_one_repeat)(
                    initial_ssa=initial_ssa,
                    time=float(time),
                    dt=float(dt),
                    seed=seed + r,
                )
                for r in range(1, repeats)
            )

            results_iter = Parallel(
                n_jobs=self._resolve_n_jobs(n_jobs),
                prefer=prefer,
                return_as="generator",
            )(tasks)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    results_iter = tqdm(
                        results_iter,
                        total=repeats - 1,
                        desc="SSA trajectories",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r, (_, tensor_tsk) in enumerate(results_iter, start=1):
                ssa_all[r] = np.transpose(tensor_tsk, (1, 2, 0)).astype(float)

        return SimulationResults(
            time=time_vec,
            ssa=ssa_all,
            pde=None,
            domain=self.domain,
            species=self.species,
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
        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        n_species, K, _ = res0.ssa.shape
        t_final = float(res0.time[-1])

        final_ssa = np.empty((repeats, n_species, K), dtype=int)
        final_ssa[0] = res0.ssa[:, :, -1].astype(int)

        # pure SSA has no PDE
        final_pde = np.zeros((repeats, n_species, self.domain.n_pde), dtype=float)

        if repeats == 1:
            if save_path is not None:
                np.savez_compressed(
                    save_path,
                    final_ssa=final_ssa,
                    final_pde=final_pde,
                    t_final=t_final,
                    species=np.array(self.species, dtype=object),
                )
            return final_ssa, final_pde, t_final

        if not parallel:
            iterator = range(1, repeats)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SSA repeats (final only)",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                final_ssa[r] = res_r.ssa[:, :, -1].astype(int)

        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            tasks = (
                delayed(self._run_one_repeat)(
                    initial_ssa=initial_ssa,
                    time=float(time),
                    dt=float(dt),
                    seed=seed + r,
                )
                for r in range(1, repeats)
            )

            results_iter = Parallel(
                n_jobs=self._resolve_n_jobs(n_jobs),
                prefer=prefer,
                return_as="generator",
            )(tasks)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    results_iter = tqdm(
                        results_iter,
                        total=repeats - 1,
                        desc="SSA repeats (final only)",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass

            for r, (_, tensor_tsk) in enumerate(results_iter, start=1):
                final_ssa[r] = tensor_tsk[-1, :, :].astype(int)

        if save_path is not None:
            np.savez_compressed(
                save_path,
                final_ssa=final_ssa,
                final_pde=final_pde,
                t_final=t_final,
                species=np.array(self.species, dtype=object),
            )

        return final_ssa, final_pde, t_final