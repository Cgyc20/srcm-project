import numpy as np
from srcm import Domain, SSADiffusion



def test_diffusion_propensity_filling():
    dom = Domain(length=1.0, n_ssa=4, pde_multiple=1, boundary="periodic")
    diff = SSADiffusion(species=["U", "V"], jump_rates={"U": 2.0, "V": 3.0})

    ssa = np.array([
        [1, 2, 0, 4],   # U
        [0, 1, 1, 1],   # V
    ], dtype=int)

    out = np.zeros(2 * dom.K)
    diff.fill_propensities(out, ssa, dom.K)

    # propensity = 2 * jump_rate * count
    expected = np.array([
        2 * 2.0 * np.array([1, 2, 0, 4]),   # U block
        2 * 3.0 * np.array([0, 1, 1, 1]),   # V block
    ]).reshape(-1)

    assert np.allclose(out, expected)


def test_apply_move_periodic_left():
    dom = Domain(length=1.0, n_ssa=4, pde_multiple=1, boundary="periodic")
    diff = SSADiffusion(species=["U"], jump_rates={"U": 1.0})
    ssa = np.array([[0, 1, 0, 0]], dtype=int)

    diff.apply_move(ssa, dom, species_idx=0, compartment_idx=1, u=0.1)  # left
    assert np.array_equal(ssa, np.array([[1, 0, 0, 0]]))  # moved to 0


def test_apply_move_periodic_right_wrap():
    dom = Domain(length=1.0, n_ssa=4, pde_multiple=1, boundary="periodic")
    diff = SSADiffusion(species=["U"], jump_rates={"U": 1.0})
    ssa = np.array([[1, 0, 0, 0]], dtype=int)

    diff.apply_move(ssa, dom, species_idx=0, compartment_idx=0, u=0.9)  # right
    assert np.array_equal(ssa, np.array([[0, 1, 0, 0]]))


def test_apply_move_zero_flux_matches_existing_behavior():
    dom = Domain(length=1.0, n_ssa=4, pde_multiple=1, boundary="zero-flux")
    diff = SSADiffusion(species=["U"], jump_rates={"U": 1.0})

    # At compartment 0, "left" forces move right to 1 (your current logic)
    ssa = np.array([[1, 0, 0, 0]], dtype=int)
    diff.apply_move(ssa, dom, species_idx=0, compartment_idx=0, u=0.1)  # left
    assert np.array_equal(ssa, np.array([[0, 1, 0, 0]]))

    # At compartment 0, "right" is blocked (does nothing)
    ssa = np.array([[1, 0, 0, 0]], dtype=int)
    diff.apply_move(ssa, dom, species_idx=0, compartment_idx=0, u=0.9)  # right
    assert np.array_equal(ssa, np.array([[1, 0, 0, 0]]))
