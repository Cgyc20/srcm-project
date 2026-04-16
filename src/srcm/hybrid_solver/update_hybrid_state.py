from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .domain import Domain


@dataclass
class HybridState:
    """
    Hybrid state container.

    Conventions
    ----------
    - ssa: (n_species, K) integer particle counts
    - pde: (n_species, Npde) float concentrations
    """
    ssa: np.ndarray
    pde: np.ndarray

    def copy(self) -> "HybridState":
        return HybridState(self.ssa.copy(), self.pde.copy())

    @property
    def n_species(self) -> int:
        return int(self.ssa.shape[0])

    def assert_consistent(self, domain: Domain) -> None:
        if self.ssa.ndim != 2:
            raise ValueError("ssa must be 2D (n_species, K)")
        if self.pde.ndim != 2:
            raise ValueError("pde must be 2D (n_species, Npde)")

        n_species, K = self.ssa.shape
        n_species2, Npde = self.pde.shape

        if n_species2 != n_species:
            raise ValueError("ssa and pde must have same n_species")
        if K != domain.K:
            raise ValueError(f"ssa K mismatch: expected {domain.K}, got {K}")
        if Npde != domain.n_pde:
            raise ValueError(f"pde Npde mismatch: expected {domain.n_pde}, got {Npde}")

    # ------------------------------------------------------------------
    # Helpers used by conversion + hybrid reaction state_change
    # ------------------------------------------------------------------
    def add_discrete(self, species_idx: int, comp_idx: int, delta: int) -> None:
        self.ssa[species_idx, comp_idx] += int(delta)

    def add_continuous_particle_mass(self, domain: Domain, species_idx: int, comp_idx: int, delta_particles: int) -> None:
        """
        Add/remove 'delta_particles' worth of mass to the PDE in a compartment.

        Convention:
          One particle corresponds to adding concentration (1/h) uniformly over
          that compartment's PDE slice (so integrated mass is 1).

        delta_particles can be negative.
        """
        s = int(domain.starts[comp_idx])
        e = int(domain.ends[comp_idx])
        self.pde[species_idx, s:e] += float(delta_particles) * domain.inv_h
