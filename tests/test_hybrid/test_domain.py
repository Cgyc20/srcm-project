import numpy as np
from srcm import Domain



def test_domain_basic_properties():
    dom = Domain(length=10.0, n_ssa=100, pde_multiple=5)

    assert dom.K == 100
    assert np.isclose(dom.h, 0.1)
    assert dom.n_pde == 500
    assert np.isclose(dom.dx, 0.02)



def test_domain_starts_ends():
    dom = Domain(length=10.0, n_ssa=4, pde_multiple=3)

    assert np.all(dom.starts == np.array([0, 3, 6, 9]))
    assert np.all(dom.ends == np.array([3, 6, 9, 12]))


def test_domain_invalid_boundary():
    import pytest
    with pytest.raises(ValueError):
        Domain(length=10.0, n_ssa=10, pde_multiple=2, boundary="nonsense")

