from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class ConversionParams:
    """
    SRCM conversion parameters.

    Conventions
    ----------
    K : number of SSA compartments
    n_species : number of species (often 'M' in notes)

    Arrays use:
      SSA counts: (n_species, K)
      PDE conc:   (n_species, Npde) where Npde = K * pde_multiple
    """
    # NOTE:
    #   For backward-compatibility, `threshold` and `rate` may be provided either
    #   as scalars (single global values) or as per-species 1D arrays of length
    #   n_species. All internal computations broadcast accordingly.
    #threshold: Union[float, np.ndarray]   # mass threshold for regime decision

    #Changing this on March 25th 2026, two different thresholds

    # Changed this again on 16th April 2026 for the new package.
    rate: float | np.ndarray     # gamma conversion rate
    threshold: float | np.ndarray | None = None
    DC_threshold: float | np.ndarray | None = None #Dsicrete to continuous, higher threshold
    CD_threshold: float | np.ndarray | None = None #Continuous to discrete, lower threshold
   

    def __post_init__(self):
        # 1. Determine the values first
        if self.threshold is not None:
            final_DC = self.threshold
            final_CD = self.threshold
            is_dual = False
        elif self.DC_threshold is not None and self.CD_threshold is not None:
            final_DC = self.DC_threshold
            final_CD = self.CD_threshold
            is_dual = True
        else:
            raise ValueError("Must provide either 'threshold' OR both 'DC_threshold' and 'CD_threshold'.")

        # 2. Use object.__setattr__ to bypass the "Frozen" restriction
        object.__setattr__(self, "rate", self._validate_param(self.rate, "rate"))
        object.__setattr__(self, "DC_threshold", self._validate_param(final_DC, "DC_threshold"))
        object.__setattr__(self, "CD_threshold", self._validate_param(final_CD, "CD_threshold"))

        # 3. Validate hysteresis logic
        if is_dual:
            DC_arr = np.asarray(self.DC_threshold, dtype=float)
            CD_arr = np.asarray(self.CD_threshold, dtype=float)
            if np.any(DC_arr <= CD_arr):
                raise ValueError("DC_threshold must be > CD_threshold for all species.")
    def _validate_param(self, val, name: str) -> float | np.ndarray:
        """Helper to validate that a parameter is a non-negative scalar or 1D array."""
        if np.isscalar(val):
            if float(val) < 0:
                raise ValueError(f"ConversionParams.{name} must be >= 0")
            return float(val)
        else:
            arr = np.asarray(val, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"ConversionParams.{name} must be a scalar or 1D array")
            if np.any(arr < 0):
                raise ValueError(f"ConversionParams.{name} values must be >= 0")
            return arr
    


      
            
    def rate_for(self, s_idx: int) -> float:
        """Return conversion rate for a given species index."""
        if np.isscalar(self.rate):
            return float(self.rate)
        rate_arr = np.asarray(self.rate, dtype=float)
        return float(rate_arr[s_idx])

    def DC_threshold_for(self, s_idx: int) -> float:
        """Return discrete-to-continuous threshold for species s_idx."""
        if np.isscalar(self.DC_threshold):
            return float(self.DC_threshold)
        return float(np.asarray(self.DC_threshold, dtype=float)[s_idx])


    def CD_threshold_for(self, s_idx: int) -> float:
        """Return continuous-to-discrete threshold for species s_idx."""
        if np.isscalar(self.CD_threshold):
            return float(self.CD_threshold)
        return float(np.asarray(self.CD_threshold, dtype=float)[s_idx])


    def DC_mask(self, combined_mass: np.ndarray) -> np.ndarray:
        """
        combined_mass: (n_species, K)
        Returns: (n_species, K) int8 mask where 1 means combined_mass > DC_threshold
        """
        if combined_mass.ndim != 2:
            raise ValueError("combined_mass must be 2D (n_species, K)")

        thr = self.DC_threshold
        if np.isscalar(thr):
            return (combined_mass > float(thr)).astype(np.int8)

        thr_arr = np.asarray(thr, dtype=float)
        if thr_arr.shape != (combined_mass.shape[0],):
            raise ValueError(
                "Per-species DC_threshold must have shape (n_species,) matching combined_mass"
            )
        return (combined_mass > thr_arr[:, None]).astype(np.int8)


    def CD_mask(self, combined_mass: np.ndarray) -> np.ndarray:
        """
        combined_mass: (n_species, K)
        Returns: (n_species, K) int8 mask where 1 means combined_mass < CD_threshold
        """
        if combined_mass.ndim != 2:
            raise ValueError("combined_mass must be 2D (n_species, K)")

        thr = self.CD_threshold
        if np.isscalar(thr):
            return (combined_mass < float(thr)).astype(np.int8)

        thr_arr = np.asarray(thr, dtype=float)
        if thr_arr.shape != (combined_mass.shape[0],):
            raise ValueError(
                "Per-species CD_threshold must have shape (n_species,) matching combined_mass"
            )
        return (combined_mass < thr_arr[:, None]).astype(np.int8)

        

    def sufficient_pde_mass_mask(self, pde_conc: np.ndarray, pde_multiple: int, h: float) -> np.ndarray:
        """
        pde_conc: (n_species, Npde), Npde must be divisible by pde_multiple
        Returns: (n_species, K) int8 mask.

        Rule:
          sufficient = all fine PDE cells in the compartment satisfy C >= 1/h
        """
        if pde_conc.ndim != 2:
            raise ValueError("pde_conc must be 2D (n_species, Npde)")
        if pde_multiple <= 0:
            raise ValueError("pde_multiple must be > 0")
        if h <= 0:
            raise ValueError("h must be > 0")

        n_species, Npde = pde_conc.shape
        if Npde % pde_multiple != 0:
            raise ValueError("Npde must be divisible by pde_multiple")

        K = Npde // pde_multiple
        conc_threshold = 1.0 / h

        reshaped = pde_conc.reshape(n_species, K, pde_multiple)
        return np.all(reshaped >= conc_threshold, axis=2).astype(np.int8)
