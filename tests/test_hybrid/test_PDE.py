import numpy as np
from srcm import rk4_step, laplacian_1d, Domain


def test_rk4_exponential_decay():
    def rhs(y, t):
        return -y

    y0 = np.array([1.0])
    t0 = 0.0
    dt = 0.1

    y1 = rk4_step(y0, t0, dt, rhs)

    expected = np.exp(-dt)
    assert np.allclose(y1[0], expected, atol=1e-7)


def test_rk4_shape_preserved():
    def rhs(y, _):
        return np.ones_like(y)

    y0 = np.zeros((3, 10))
    y1 = rk4_step(y0, 0.0, 0.5, rhs)

    assert y1.shape == (3, 10)
    assert np.allclose(y1, 0.5)  # since y' = 1



def test_laplacian_zero_flux_small():
    dom = Domain(length=1.0, n_ssa=1, pde_multiple=5, boundary="zero-flux")
    L = laplacian_1d(dom)

    expected = np.array([
        [-1,  1,  0,  0,  0],
        [ 1, -2,  1,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  1, -2,  1],
        [ 0,  0,  0,  1, -1],
    ], dtype=float)

    assert np.array_equal(L, expected)


def test_laplacian_periodic_small():
    dom = Domain(length=1.0, n_ssa=1, pde_multiple=5, boundary="periodic")
    L = laplacian_1d(dom)

    expected = np.array([
        [-2,  1,  0,  0,  1],
        [ 1, -2,  1,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  1, -2,  1],
        [ 1,  0,  0,  1, -2],
    ], dtype=float)

    assert np.array_equal(L, expected)
