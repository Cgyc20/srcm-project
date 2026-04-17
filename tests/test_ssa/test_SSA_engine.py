from __future__ import annotations

import numpy as np
import pytest

from srcm import Domain, SSAReaction, SSAEngine


def make_reaction_system_one_species_zero_order():
    rs = SSAReaction()
    rs.add_reaction({}, {"A": 1}, 1.0)
    return rs


def make_reaction_system_two_species_first_order():
    rs = SSAReaction()
    rs.add_reaction({"A": 1}, {"B": 1}, 0.5)
    return rs


def make_reaction_system_two_species_mixed():
    rs = SSAReaction()
    rs.add_reaction({}, {"A": 1}, 1.0)              # zero-order
    rs.add_reaction({"A": 1}, {"B": 1}, 0.3)        # first-order
    rs.add_reaction({"A": 1, "B": 1}, {}, 0.4)      # second-order
    return rs


def make_domain(K=2, L=10.0, boundary="periodic"):
    return Domain(length=float(L), n_ssa=int(K), pde_multiple=1, boundary=boundary)


def test_engine_initialises_correctly():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0, boundary="periodic")

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    assert engine.n_species == 2
    assert engine.K == 2
    assert engine.species == ["A", "B"]


def test_engine_requires_matching_diffusion_keys():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    with pytest.raises(ValueError, match="diffusion_rates keys must match"):
        SSAEngine(
            reaction_system=rs,
            diffusion_rates={"A": 0.01},   # missing B
            domain=domain,
        )


def test_timevector_length():
    rs = make_reaction_system_one_species_zero_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01},
        domain=domain,
    )

    tvec = engine._timevector(time=5.0, dt=1.0)
    assert np.allclose(tvec, np.array([0., 1., 2., 3., 4., 5.]))


def test_initial_tensor_shape_and_initial_conditions():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=3, L=9.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    init = np.array([[10, 5, 1], [20, 15, 2]], dtype=int)
    tvec, tensor = engine._initial_tensor(time=5.0, dt=1.0, initial_ssa=init)

    assert tensor.shape == (len(tvec), 2, 3)   # internal order (T,S,K)
    assert np.array_equal(tensor[0], init)


def test_initial_tensor_rejects_wrong_shape():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=3, L=9.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    bad_init = np.array([10, 5, 1], dtype=int)

    with pytest.raises(ValueError, match="initial_ssa must have shape"):
        engine._initial_tensor(time=5.0, dt=1.0, initial_ssa=bad_init)


def test_jump_rates_are_correct():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.02, "B": 0.04},
        domain=domain,
    )

    expected = [0.02 / (engine.h ** 2), 0.04 / (engine.h ** 2)]
    assert np.allclose(engine.jump_rate_list, expected)


def test_propensity_rejects_wrong_frame_shape():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    bad_frame = np.zeros((2, 3), dtype=int)
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    with pytest.raises(ValueError, match="frame must have shape"):
        engine._propensity_calculation(bad_frame, propensity)


def test_zero_order_diffusion_propensity_block():
    rs = make_reaction_system_one_species_zero_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01},
        domain=domain,
    )

    frame = np.array([[10, 1]], dtype=int)
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    out = engine._propensity_calculation(frame, propensity)

    jump_rate = engine.jump_rate_list[0]
    assert out[0] == pytest.approx(jump_rate * 10 * 2.0)
    assert out[1] == pytest.approx(jump_rate * 1 * 2.0)


def test_zero_order_reaction_propensity_block():
    rs = make_reaction_system_one_species_zero_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01},
        domain=domain,
    )

    frame = np.array([[10, 1]], dtype=int)
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    out = engine._propensity_calculation(frame, propensity)

    start = engine.K * engine.n_species
    expected = 1.0 * engine.h
    assert out[start + 0] == pytest.approx(expected)
    assert out[start + 1] == pytest.approx(expected)


def test_first_order_reaction_propensity_block():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    frame = np.array([[10, 5], [0, 0]], dtype=int)
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    out = engine._propensity_calculation(frame, propensity)

    start = engine.K * engine.n_species
    assert out[start + 0] == pytest.approx(0.5 * 10)
    assert out[start + 1] == pytest.approx(0.5 * 5)


def test_second_order_reaction_propensity_block():
    rs = SSAReaction()
    rs.add_reaction({"A": 2}, {"B": 1}, 0.2)

    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    frame = np.array([[10, 5], [0, 0]], dtype=int)
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    out = engine._propensity_calculation(frame, propensity)

    start = engine.K * engine.n_species
    assert out[start + 0] == pytest.approx(0.2 * 10 * 9 / engine.h)
    assert out[start + 1] == pytest.approx(0.2 * 5 * 4 / engine.h)


def test_mixed_propensities_multiple_compartments():
    rs = make_reaction_system_two_species_mixed()
    domain = make_domain(K=6, L=30.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    frame = np.array(
        [
            [10, 5, 15, 10, 5, 3],
            [0, 0, 0, 1, 2, 3],
        ],
        dtype=int,
    )
    propensity = np.zeros(engine.K * engine.n_species + engine.K * engine.number_of_reactions)

    out = engine._propensity_calculation(frame, propensity)

    # diffusion blocks
    jump_A = engine.jump_rate_list[0]
    jump_B = engine.jump_rate_list[1]

    for i in range(engine.K):
        assert out[i] == pytest.approx(jump_A * frame[0, i] * 2.0)
        assert out[engine.K + i] == pytest.approx(jump_B * frame[1, i] * 2.0)

    # zero-order block
    start0 = engine.K * engine.n_species
    for i in range(engine.K):
        assert out[start0 + i] == pytest.approx(1.0 * engine.h)

    # first-order block
    start1 = start0 + engine.K
    for i in range(engine.K):
        assert out[start1 + i] == pytest.approx(0.3 * frame[0, i])

    # second-order block
    start2 = start1 + engine.K
    for i in range(engine.K):
        assert out[start2 + i] == pytest.approx(0.4 * frame[0, i] * frame[1, i] / engine.h)


def test_run_returns_simulation_results_with_srcm_shape():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.02},
        domain=domain,
    )

    init = np.array([[10, 5], [1, 2]], dtype=int)

    res = engine.run(
        initial_ssa=init,
        initial_pde=None,
        time=2.0,
        dt=1.0,
        seed=0,
    )

    assert res.ssa.shape == (2, 2, 3)   # (S,K,T)
    assert res.pde is None
    assert np.array_equal(res.ssa[:, :, 0], init)


def test_run_repeats_shape_and_nonnegative():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.01},
        domain=domain,
    )

    init = np.array([[10, 5], [1, 2]], dtype=int)

    res = engine.run_repeats(
        initial_ssa=init,
        initial_pde=None,
        time=2.0,
        dt=1.0,
        repeats=10,
        seed=0,
        parallel=False,
        progress=False,
    )

    assert res.ssa.shape == (2, 2, 3)
    assert np.all(res.ssa >= 0)


def test_run_trajectories_shape():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.01},
        domain=domain,
    )

    init = np.array([[10, 5], [1, 2]], dtype=int)

    res = engine.run_trajectories(
        initial_ssa=init,
        initial_pde=None,
        time=2.0,
        dt=1.0,
        repeats=4,
        seed=0,
        parallel=False,
        progress=False,
    )

    assert res.ssa.shape == (4, 2, 2, 3)   # (R,S,K,T)
    assert res.pde is None


def test_run_repeats_final_shape():
    rs = make_reaction_system_two_species_first_order()
    domain = make_domain(K=2, L=10.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01, "B": 0.01},
        domain=domain,
    )

    init = np.array([[10, 5], [1, 2]], dtype=int)

    final_ssa, final_pde, t_final = engine.run_repeats_final(
        initial_ssa=init,
        initial_pde=None,
        time=2.0,
        dt=1.0,
        repeats=5,
        seed=0,
        parallel=False,
        progress=False,
    )

    assert final_ssa.shape == (5, 2, 2)
    assert final_pde.shape == (5, 2, domain.n_pde)
    assert t_final == pytest.approx(2.0)


def test_no_negative_counts_after_runs():
    rs = SSAReaction()
    rs.add_reaction({"A": 1}, {}, 1.0)

    domain = make_domain(K=2, L=2.0)

    engine = SSAEngine(
        reaction_system=rs,
        diffusion_rates={"A": 0.01},
        domain=domain,
    )

    init = np.array([[0, 0]], dtype=int)

    res = engine.run_repeats(
        initial_ssa=init,
        initial_pde=None,
        time=1.0,
        dt=1.0,
        repeats=5,
        seed=0,
        parallel=False,
        progress=False,
    )

    assert np.all(res.ssa >= 0)