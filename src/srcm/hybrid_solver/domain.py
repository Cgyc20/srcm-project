from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Domain:
    """
    1D domain shared by SSA (coarse compartments) and PDE (fine grid).

    Attributes
    ----------
    length : float
        Total domain length.
    n_ssa : int
        Number of SSA compartments (K).
    pde_multiple : int
        Number of PDE cells per SSA compartment.
    boundary : str
        'zero-flux' or 'periodic'
    """
    length: float
    n_ssa: int
    pde_multiple: int
    boundary: str = "zero-flux"

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError("Domain.length must be > 0")
        if self.n_ssa <= 0:
            raise ValueError("Domain.n_ssa must be > 0")
        if self.pde_multiple <= 0:
            raise ValueError("Domain.pde_multiple must be > 0")
        if self.boundary not in ("zero-flux", "periodic"):
            raise ValueError("Domain.boundary must be 'zero-flux' or 'periodic'")
        
    @property
    def K(self) -> int:
        """Number of SSA compartments."""
        return self.n_ssa
    
    @property
    def h(self) -> float:
        """SSA compartment size."""
        return self.length / self.n_ssa

    @property
    def inv_h(self) -> float:
        """Inverse of the compartment size"""
        return 1.0 / self.h

    @property
    def n_pde(self) -> int:
        """Number of PDE grid points."""
        return self.n_ssa * self.pde_multiple

    @property
    def dx(self) -> float:
        """PDE grid spacing."""
        return self.length / self.n_pde

    @property
    def starts(self) -> np.ndarray:
        """Start PDE index for each SSA compartment (length K)."""
        return np.arange(self.K, dtype=int) * self.pde_multiple

    @property
    def ends(self) -> np.ndarray:
        """End PDE index (exclusive) for each SSA compartment (length K)."""
        return self.starts + self.pde_multiple

    @property
    def ssa_x(self) -> np.ndarray:
        """Coordinates of SSA compartments (left edges)."""
        return np.linspace(0.0, self.length - self.h, self.K)

    @property
    def pde_x(self) -> np.ndarray:
        """Coordinates of PDE grid points."""
        return np.linspace(0.0, self.length, self.n_pde)
    

