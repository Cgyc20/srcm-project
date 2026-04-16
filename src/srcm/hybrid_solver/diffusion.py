from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .domain import Domain


@dataclass(frozen=True)
class SSADiffusion:
    """
    SSA diffusion handler.

    jump_rates: dict mapping species name -> jump rate
      e.g. jump_rate = D_macro / h^2

    Conventions
    ----------
    - SSA counts array: shape (n_species, K)
    - diffusion propensities are stored in blocks:
        block 0: species 0, length K
        block 1: species 1, length K
        ...
      total diffusion entries = n_species * K
    """
    species: list[str]
    jump_rates: dict[str, float]

    def __post_init__(self):
        for s in self.species:
            if s not in self.jump_rates:
                raise ValueError(f"Missing jump rate for species '{s}'")
            if self.jump_rates[s] < 0:
                raise ValueError(f"Jump rate for '{s}' must be >= 0")

    @property
    def n_species(self) -> int:
        return len(self.species)

    def fill_propensities(self, out: np.ndarray, ssa_counts: np.ndarray, K: int) -> None:
        """
        Fill diffusion propensities into 'out' (first n_species*K entries).

        propensity(species s at compartment i) = 2 * jump_rate[s] * D_s,i
        """
        if ssa_counts.shape != (self.n_species, K):
            raise ValueError(f"ssa_counts must have shape {(self.n_species, K)}")
        if out.shape[0] < self.n_species * K:
            raise ValueError("out is too small to hold diffusion propensities")

        for s_idx, s in enumerate(self.species):
            start = s_idx * K
            end = start + K
            out[start:end] = 2.0 * self.jump_rates[s] * ssa_counts[s_idx, :]

    def apply_move(self, ssa_counts: np.ndarray, domain: Domain, species_idx: int, compartment_idx: int, u: float) -> None:
        """
        Apply one diffusion jump for a given species in a given compartment.

        Parameters
        ----------
        ssa_counts : (n_species, K) array (modified in-place)
        domain : Domain
        species_idx : int
        compartment_idx : int
        u : float in [0,1)
            Used to choose direction (u < 0.5 left, else right)
        """
        K = domain.K
        if not (0 <= species_idx < self.n_species):
            raise ValueError("species_idx out of range")
        if not (0 <= compartment_idx < K):
            raise ValueError("compartment_idx out of range")
        if not (0.0 <= u < 1.0):
            raise ValueError("u must be in [0,1)")

        # If there is no particle, do nothing (safety)
        if ssa_counts[species_idx, compartment_idx] <= 0:
            return

        move_left = (u < 0.5)

        if domain.boundary == "periodic":
            target = (compartment_idx - 1) % K if move_left else (compartment_idx + 1) % K
            ssa_counts[species_idx, compartment_idx] -= 1
            ssa_counts[species_idx, target] += 1
            return

        # zero-flux (matches your existing behavior: "blocked moves do nothing")
        if domain.boundary == "zero-flux":
            if move_left:
                # at left boundary: forced right (your current behavior)
                if compartment_idx == 0:
                    target = 1
                else:
                    target = compartment_idx - 1
                ssa_counts[species_idx, compartment_idx] -= 1
                ssa_counts[species_idx, target] += 1
            else:
                # at either boundary: blocked (do nothing)
                if compartment_idx == 0 or compartment_idx == K - 1:
                    return
                target = compartment_idx + 1
                ssa_counts[species_idx, compartment_idx] -= 1
                ssa_counts[species_idx, target] += 1
            return

        raise ValueError(f"Unknown boundary '{domain.boundary}'")
