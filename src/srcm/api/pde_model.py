from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from ..pde_solver import PDEEngine


@dataclass
class PDEResults:
    """
    Container for PDE simulation outputs.

    Attributes
    ----------
    t : np.ndarray
        Time vector, shape (Nt,)
    x : np.ndarray
        Spatial grid, shape (K,)
    species : list[str]
        Species names in order
    data : dict[str, np.ndarray]
        Mapping species name -> array of shape (Nt, K)
    """
    t: np.ndarray
    x: np.ndarray
    species: list[str]
    data: dict[str, np.ndarray]

    def __getattr__(self, name: str):
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


class PDEModel:
    """
    Thin, user-friendly wrapper around PDEEngine with a hybrid-like API.
    """

    def __init__(self, species: list[str], boundary: str = "zero-flux") -> None:
        if not isinstance(species, list) or len(species) not in (1, 2):
            raise ValueError("species must be a list of length 1 or 2")
        if len(set(species)) != len(species):
            raise ValueError("species names must be unique")
        if boundary not in ("zero-flux", "periodic"):
            raise ValueError("boundary must be 'zero-flux' or 'periodic'")

        self.species = species
        self.boundary = boundary
        self._diffusion: dict[str, float] = {}
        self._pde_func: Optional[Callable] = None

    def diffusion(self, **kwargs: float) -> None:
        for sp in self.species:
            if sp not in kwargs:
                raise ValueError(f"Missing diffusion coefficient for species '{sp}'")

        extra = set(kwargs) - set(self.species)
        if extra:
            raise ValueError(f"Unknown species in diffusion(): {sorted(extra)}")

        parsed: dict[str, float] = {}
        for sp, d in kwargs.items():
            if not isinstance(d, (int, float)) or d < 0:
                raise ValueError(f"Diffusion coefficient for '{sp}' must be non-negative")
            parsed[sp] = float(d)

        self._diffusion = parsed

    def pde(self, func: Callable) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self._pde_func = func

    def _validate_ready(self) -> None:
        if len(self._diffusion) != len(self.species):
            raise RuntimeError("Diffusion not fully specified. Call diffusion(...) first.")
        if self._pde_func is None:
            raise RuntimeError("PDE reaction function not set. Call pde(...) first.")

    def _build_engine_reaction(self, x: np.ndarray) -> Callable:
        assert self._pde_func is not None

        if len(self.species) == 1:
            def reaction_one(u: np.ndarray) -> np.ndarray:
                out = self._pde_func(u, x)
                if not isinstance(out, tuple) or len(out) != 1:
                    raise ValueError("For one species, pde(...) must return a 1-tuple: (fU,)")
                return np.asarray(out[0], dtype=float)

            return reaction_one

        def reaction_two(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            out = self._pde_func(u, v, x)
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError("For two species, pde(...) must return a 2-tuple: (fU, fV)")
            return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)

        return reaction_two

    def run(
        self,
        *,
        L: float,
        K: int,
        total_time: float,
        dt: float,
        init: dict[str, np.ndarray],
    ) -> tuple[PDEResults, dict]:
        self._validate_ready()

        if not isinstance(K, int) or K <= 2:
            raise ValueError("K must be an integer > 2")

        dx = float(L) / K
        engine = PDEEngine(L=L, dt=dt, dx=dx, boundary_type=self.boundary)

        missing = [sp for sp in self.species if sp not in init]
        if missing:
            raise ValueError(f"Missing initial condition(s) for: {missing}")

        extra = set(init) - set(self.species)
        if extra:
            raise ValueError(f"Unknown species in init: {sorted(extra)}")

        u0 = np.asarray(init[self.species[0]], dtype=float)
        if u0.shape != (K,):
            raise ValueError(f"Initial condition for '{self.species[0]}' must have shape ({K},)")

        reaction_func = self._build_engine_reaction(engine.x)

        if len(self.species) == 1:
            D = [self._diffusion[self.species[0]]]
            engine.input_system(
                reaction_func=reaction_func,
                diffusion_coefficients=D,
                u0=u0,
            )
            u_record = engine.run_simulation(total_time=total_time)

            results = PDEResults(
                t=engine.timevector.copy(),
                x=engine.x.copy(),
                species=self.species.copy(),
                data={self.species[0]: u_record},
            )

        else:
            v0 = np.asarray(init[self.species[1]], dtype=float)
            if v0.shape != (K,):
                raise ValueError(
                    f"Initial condition for '{self.species[1]}' must have shape ({K},)"
                )

            D = [self._diffusion[self.species[0]], self._diffusion[self.species[1]]]
            engine.input_system(
                reaction_func=reaction_func,
                diffusion_coefficients=D,
                u0=u0,
                v0=v0,
            )
            u_record, v_record = engine.run_simulation(total_time=total_time)

            results = PDEResults(
                t=engine.timevector.copy(),
                x=engine.x.copy(),
                species=self.species.copy(),
                data={
                    self.species[0]: u_record,
                    self.species[1]: v_record,
                },
            )

        meta = {
            "model": "pde",
            "species": self.species.copy(),
            "boundary": self.boundary,
            "L": float(L),
            "K": int(K),
            "dx": dx,
            "dt": float(dt),
            "total_time": float(total_time),
            "diffusion": self._diffusion.copy(),
        }

        return results, meta

    @staticmethod
    def save(results: PDEResults, path: str, meta: Optional[dict] = None) -> None:
        save_dict = {
            "t": np.asarray(results.t, dtype=float),
            "x": np.asarray(results.x, dtype=float),
            "species": np.asarray(results.species, dtype=object),
        }

        for sp, arr in results.data.items():
            save_dict[sp] = np.asarray(arr, dtype=float)

        if meta is not None:
            for k, v in meta.items():
                if isinstance(v, (int, float, str, bool)):
                    save_dict[f"meta_{k}"] = np.array(v)
                else:
                    save_dict[f"meta_{k}"] = np.array(v, dtype=object)

        np.savez_compressed(path, **save_dict)