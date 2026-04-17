from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .domain import Domain


@dataclass(frozen=True)
class SimulationResults:
    """
    Stores time series output of an SRCM simulation.

    Shapes
    ------
    ssa : (n_species, K, n_steps)
    pde : (n_species, Npde, n_steps)
    time : (n_steps,)
    """
    time: np.ndarray
    ssa: np.ndarray
    pde: np.ndarray
    domain: Domain
    species: list[str]

    @property
    def n_species(self) -> int:
        return int(self.ssa.shape[0])

    @property
    def K(self) -> int:
        return int(self.ssa.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    def combined(self) -> np.ndarray:
        """
        Return combined grid on PDE resolution:
          combined = pde + (ssa / h) expanded over each compartment slice

        Output shape: (n_species, Npde, n_steps)
        """
        n_species, K, n_steps = self.ssa.shape
        Npde = self.domain.n_pde
        out = self.pde.copy()

        for s in range(n_species):
            for i in range(K):
                start = self.domain.starts[i]
                end = self.domain.ends[i]
                out[s, start:end, :] += (self.ssa[s, i, :] / self.domain.h)

        return out
