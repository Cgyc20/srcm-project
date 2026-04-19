from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import inspect

import numpy as np


PDEFunc1 = Callable[[np.ndarray], np.ndarray]
PDEFunc2 = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class PDEEngine:
    """
    Simple 1D reaction-diffusion PDE solver for one or two species.

    Notes
    -----
    - Uses RK4 in time.
    - Uses a finite-difference Laplacian in space.
    - Supports:
        * one species: reaction_func(u) -> du/dt_reaction
        * two species: reaction_func(u, v) -> (du/dt_reaction, dv/dt_reaction)
    """

    L: float
    dt: float
    dx: float
    boundary_type: str = "zero-flux"

    def __post_init__(self) -> None:
        if not isinstance(self.L, (int, float)) or self.L <= 0:
            raise ValueError("L must be a positive number")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number")
        if not isinstance(self.dx, (int, float)) or self.dx <= 0:
            raise ValueError("dx must be a positive number")
        if self.boundary_type not in ("zero-flux", "periodic"):
            raise ValueError("boundary_type must be 'zero-flux' or 'periodic'")

        self.L = float(self.L)
        self.dt = float(self.dt)
        self.dx = float(self.dx)

        self.n = int(round(self.L / self.dx))
        if self.n <= 2:
            raise ValueError("Number of spatial grid points must be > 2")

        self.x = np.linspace(0.0, self.L - self.dx, self.n)

        self.model_type: Optional[str] = None
        self.reaction_func = None
        self.diffusion_coefficients: Optional[list[float]] = None
        self.u0: Optional[np.ndarray] = None
        self.v0: Optional[np.ndarray] = None

        self._lap = self._build_finite_difference_matrix()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def input_system(
        self,
        reaction_func: Callable,
        diffusion_coefficients: list[float],
        u0: np.ndarray,
        v0: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register reaction terms, diffusion coefficients, and initial data.
        """
        if not callable(reaction_func):
            raise TypeError("reaction_func must be callable")
        if not isinstance(diffusion_coefficients, list):
            raise TypeError("diffusion_coefficients must be a list")
        if len(diffusion_coefficients) not in (1, 2):
            raise ValueError("diffusion_coefficients must have length 1 or 2")
        if any((not isinstance(d, (int, float)) or d < 0) for d in diffusion_coefficients):
            raise ValueError("Diffusion coefficients must be non-negative numbers")

        u0 = np.asarray(u0, dtype=float)
        if u0.shape != (self.n,):
            raise ValueError(f"u0 must have shape ({self.n},), got {u0.shape}")

        n_args = len(inspect.signature(reaction_func).parameters)

        self.reaction_func = reaction_func
        self.diffusion_coefficients = [float(d) for d in diffusion_coefficients]
        self.u0 = u0.copy()
        self.v0 = None

        if v0 is None:
            if n_args != 1:
                raise ValueError("One-species reaction_func must accept exactly 1 argument")
            if len(self.diffusion_coefficients) != 1:
                raise ValueError("One-species model requires exactly 1 diffusion coefficient")

            test_out = reaction_func(self.u0)
            test_out = np.asarray(test_out, dtype=float)
            if test_out.shape != (self.n,):
                raise ValueError(
                    f"One-species reaction_func must return shape ({self.n},), got {test_out.shape}"
                )

            self.model_type = "one_species"
            self._Lu = (self.diffusion_coefficients[0] / self.dx**2) * self._lap
            return

        v0 = np.asarray(v0, dtype=float)
        if v0.shape != (self.n,):
            raise ValueError(f"v0 must have shape ({self.n},), got {v0.shape}")

        if n_args != 2:
            raise ValueError("Two-species reaction_func must accept exactly 2 arguments")
        if len(self.diffusion_coefficients) != 2:
            raise ValueError("Two-species model requires exactly 2 diffusion coefficients")

        test_out = reaction_func(self.u0, v0)
        if not isinstance(test_out, tuple) or len(test_out) != 2:
            raise ValueError("Two-species reaction_func must return a tuple (du, dv)")

        du_test = np.asarray(test_out[0], dtype=float)
        dv_test = np.asarray(test_out[1], dtype=float)
        if du_test.shape != (self.n,) or dv_test.shape != (self.n,):
            raise ValueError(
                f"Two-species reaction_func outputs must both have shape ({self.n},)"
            )

        self.v0 = v0.copy()
        self.model_type = "two_species"
        self._Lu = (self.diffusion_coefficients[0] / self.dx**2) * self._lap
        self._Lv = (self.diffusion_coefficients[1] / self.dx**2) * self._lap

    # ------------------------------------------------------------------
    # printing
    # ------------------------------------------------------------------
    def print_system(self) -> None:
        if self.model_type is None or self.reaction_func is None:
            print("System not defined yet.")
            return

        try:
            source = inspect.getsource(self.reaction_func).strip()
            print(source)
        except Exception:
            print("Could not inspect reaction function source.")

        if self.model_type == "one_species":
            print(f"Model: one species")
            print(f"D_u = {self.diffusion_coefficients[0]}")
            print("PDE: u_t = D_u u_xx + f(u)")
        else:
            print(f"Model: two species")
            print(f"D_u = {self.diffusion_coefficients[0]}")
            print(f"D_v = {self.diffusion_coefficients[1]}")
            print("PDE: u_t = D_u u_xx + f(u,v)")
            print("     v_t = D_v v_xx + g(u,v)")

    # ------------------------------------------------------------------
    # spatial operator
    # ------------------------------------------------------------------
    def _build_finite_difference_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=float)

        if self.boundary_type == "periodic":
            for i in range(self.n):
                A[i, i] = -2.0
                A[i, (i - 1) % self.n] = 1.0
                A[i, (i + 1) % self.n] = 1.0
            return A

        if self.boundary_type == "zero-flux":
            for i in range(self.n):
                if i == 0:
                    A[i, i] = -1.0
                    A[i, i + 1] = 1.0
                elif i == self.n - 1:
                    A[i, i] = -1.0
                    A[i, i - 1] = 1.0
                else:
                    A[i, i] = -2.0
                    A[i, i - 1] = 1.0
                    A[i, i + 1] = 1.0
            return A

        raise ValueError(f"Unknown boundary_type '{self.boundary_type}'")

    # ------------------------------------------------------------------
    # rhs
    # ------------------------------------------------------------------
    def _rhs_one_species(self, u: np.ndarray) -> np.ndarray:
        return self._Lu @ u + np.asarray(self.reaction_func(u), dtype=float)

    def _rhs_two_species(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ru, rv = self.reaction_func(u, v)
        du = self._Lu @ u + np.asarray(ru, dtype=float)
        dv = self._Lv @ v + np.asarray(rv, dtype=float)
        return du, dv

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def rk4_step(self, u: np.ndarray, v: Optional[np.ndarray] = None):
        if self.model_type == "one_species":
            k1 = self.dt * self._rhs_one_species(u)
            k2 = self.dt * self._rhs_one_species(u + 0.5 * k1)
            k3 = self.dt * self._rhs_one_species(u + 0.5 * k2)
            k4 = self.dt * self._rhs_one_species(u + k3)
            return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        if self.model_type == "two_species":
            if v is None:
                raise ValueError("Two-species model requires both u and v")

            k1u, k1v = self._rhs_two_species(u, v)
            k2u, k2v = self._rhs_two_species(u + 0.5 * self.dt * k1u, v + 0.5 * self.dt * k1v)
            k3u, k3v = self._rhs_two_species(u + 0.5 * self.dt * k2u, v + 0.5 * self.dt * k2v)
            k4u, k4v = self._rhs_two_species(u + self.dt * k3u, v + self.dt * k3v)

            u_next = u + self.dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
            v_next = v + self.dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            return u_next, v_next

        raise RuntimeError("System not initialized. Call input_system(...) first.")

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def run_simulation(self, total_time: float):
        if self.model_type is None:
            raise RuntimeError("System not initialized. Call input_system(...) first.")
        if not isinstance(total_time, (int, float)) or total_time <= 0:
            raise ValueError("total_time must be a positive number")

        total_time = float(total_time)
        #self.timevector = np.arange(0.0, total_time + 1e-12, self.dt)
        self.timevector = np.arange(0.0, total_time, self.dt) #Changed (possible bug,)
        if self.model_type == "one_species":
            u_record = np.zeros((len(self.timevector), self.n), dtype=float)
            u = self.u0.copy()
            u_record[0, :] = u

            for t_idx in range(1, len(self.timevector)):
                u = self.rk4_step(u)
                u_record[t_idx, :] = u

            return u_record

        u_record = np.zeros((len(self.timevector), self.n), dtype=float)
        v_record = np.zeros((len(self.timevector), self.n), dtype=float)

        u = self.u0.copy()
        v = self.v0.copy()

        u_record[0, :] = u
        v_record[0, :] = v

        for t_idx in range(1, len(self.timevector)):
            u, v = self.rk4_step(u, v)
            u_record[t_idx, :] = u
            v_record[t_idx, :] = v

        return u_record, v_record

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------
    def save_data(
        self,
        u_record: np.ndarray,
        v_record: Optional[np.ndarray] = None,
        filename: str = "simulation_data.npz",
    ) -> None:
        save_kwargs = {
            "u_record": np.asarray(u_record, dtype=float),
            "L": self.L,
            "n": self.n,
            "dx": self.dx,
            "dt": self.dt,
            "boundary_type": self.boundary_type,
            "x": self.x,
            "model_type": self.model_type,
        }

        if self.diffusion_coefficients is not None:
            save_kwargs["diffusion_coefficients"] = np.asarray(self.diffusion_coefficients, dtype=float)

        if hasattr(self, "timevector"):
            save_kwargs["timevector"] = self.timevector

        if self.u0 is not None:
            save_kwargs["u0"] = self.u0

        if self.v0 is not None:
            save_kwargs["v0"] = self.v0

        if v_record is not None:
            save_kwargs["v_record"] = np.asarray(v_record, dtype=float)

        np.savez_compressed(filename, **save_kwargs)