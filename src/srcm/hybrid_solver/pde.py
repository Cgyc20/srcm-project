from __future__ import annotations
from typing import Callable
import numpy as np
from .domain import Domain

RHSFn = Callable[[np.ndarray, float], np.ndarray]
# Signature: rhs(state, t) -> dstate/dt, same shape as state


def rk4_step(y: np.ndarray, t: float, dt: float, rhs: RHSFn) -> np.ndarray:
    """
    Perform one RK4 step for an ODE system y' = rhs(y, t).

    Parameters
    ----------
    y : np.ndarray
        Current state, shape arbitrary (for us: (n_species, Npde)).
    t : float
        Current time.
    dt : float
        Time step.
    rhs : callable
        Function rhs(y, t) returning dy/dt with same shape as y.

    Returns
    -------
    y_next : np.ndarray
        State at time t + dt.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def laplacian_1d(domain: Domain) -> np.ndarray:
    """
    Construct the 1D Laplacian matrix on the PDE grid.

    Boundary conditions:
      - zero-flux (Neumann)
      - periodic

    Returns
    -------
    L : np.ndarray
        Shape (Npde, Npde)
        Discrete Laplacian WITHOUT diffusion coefficient or dx^2 scaling.
    """
    N = domain.n_pde
    L = np.zeros((N, N), dtype=float)

    if domain.boundary == "zero-flux":
        # Left boundary
        L[0, 0] = -1.0
        L[0, 1] = 1.0

        # Interior points
        for i in range(1, N - 1):
            L[i, i - 1] = 1.0
            L[i, i] = -2.0
            L[i, i + 1] = 1.0

        # Right boundary
        L[N - 1, N - 2] = 1.0
        L[N - 1, N - 1] = -1.0

    elif domain.boundary == "periodic":
        for i in range(N):
            L[i, i] = -2.0
            L[i, (i - 1) % N] = 1.0
            L[i, (i + 1) % N] = 1.0

    else:
        raise ValueError(f"Unknown boundary condition '{domain.boundary}'")

    return L
