from __future__ import annotations
import numpy as np


def gillespie_draw(propensities: np.ndarray, rng: np.random.Generator, cumulative: np.ndarray | None = None):
    """
    Standard Gillespie draw.

    Parameters
    ----------
    propensities : np.ndarray
        1D array of nonnegative propensities (a_j).
    rng : np.random.Generator
        Numpy random generator.
    cumulative : np.ndarray | None
        Optional preallocated array to store cumulative sum (same shape as propensities).
        If provided, will be written in-place.

    Returns
    -------
    tau : float
        Time to next event. If total propensity = 0, returns np.inf.
    idx : int
        Index of event channel. If total propensity = 0, returns -1.
    """
    if propensities.ndim != 1:
        raise ValueError("propensities must be a 1D array")

    if np.any(propensities < 0):
        j = int(np.where(propensities < 0)[0][0])
        raise ValueError(f"propensities must be nonnegative: idx={j}, val={propensities[j]}")


    a0 = float(np.sum(propensities))
    if a0 == 0.0:
        return np.inf, -1

    u1 = rng.random()
    u2 = rng.random()

    tau = (1.0 / a0) * np.log(1.0 / u1)

    if cumulative is None:
        cumulative = np.cumsum(propensities)
    else:
        if cumulative.shape != propensities.shape:
            raise ValueError("cumulative must have same shape as propensities")
        np.cumsum(propensities, out=cumulative)

    idx = int(np.searchsorted(cumulative, u2 * a0))
    # safety clamp (rare floating edge cases)
    # if idx >= propensities.size:
    #     idx = propensities.size - 1

    return tau, idx
