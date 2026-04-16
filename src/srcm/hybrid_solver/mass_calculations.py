from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MassProjector:
    """
    Handles the projection and integration of mass between the 
    continuous PDE grid and the discrete SSA compartments.
    """
    pde_multiple: int  # Number of PDE cells per SSA compartment
    dx: float          # PDE grid spacing
    h: float           # SSA compartment width (usually pde_multiple * dx)

    def __post_init__(self):
        if self.pde_multiple <= 0:
            raise ValueError("pde_multiple must be > 0")
        if self.dx <= 0:
            raise ValueError("dx must be > 0")
        if self.h <= 0:
            raise ValueError("h must be > 0")

    def integrate_pde_mass(self, pde_conc: np.ndarray) -> np.ndarray:
        """
        Integrates fine-grid PDE concentrations into SSA-sized compartment masses.
        Uses a left-hand rule approximation: Mass = Σ(Concentration) * dx
        """
        if pde_conc.ndim != 2:
            raise ValueError("pde_conc must be 2D (n_species, Npde)")
        
        n_species, Npde = pde_conc.shape
        if Npde % self.pde_multiple != 0:
            raise ValueError(f"Npde ({Npde}) must be divisible by pde_multiple ({self.pde_multiple})")

        K = Npde // self.pde_multiple

        # View as blocks: (n_species, K_compartments, cells_per_compartment)
        blocks = pde_conc.reshape(n_species, K, self.pde_multiple)
        
        # Sum concentration per block and multiply by cell width
        return blocks.sum(axis=2) * self.dx

    def get_combined_total_mass(self, ssa_counts: np.ndarray, pde_conc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the total mass in each compartment by summing discrete counts 
        and integrated continuous mass.
        """
        pde_mass = self.integrate_pde_mass(pde_conc)
        
        if ssa_counts.shape != pde_mass.shape:
            raise ValueError(f"Shape mismatch: ssa_counts {ssa_counts.shape} vs pde_mass {pde_mass.shape}")

        combined = ssa_counts.astype(float, copy=False) + pde_mass
        return combined, pde_mass

    def get_high_concentration_mask(self, pde_conc: np.ndarray) -> np.ndarray:
        """
        Identifies compartments where ALL underlying PDE cells are above 
        the resolution threshold (1/h).
        """
        n_species, Npde = pde_conc.shape
        K = Npde // self.pde_multiple
        threshold = 1.0 / self.h

        # Reshape to inspect individual cells within each compartment
        cells = pde_conc.reshape(n_species, K, self.pde_multiple)

        # Logic: A compartment is "sufficiently dense" if all its cells are >= 1/h
        is_sufficient = np.all(cells >= threshold, axis=2)
        
        return is_sufficient.astype(np.int8)