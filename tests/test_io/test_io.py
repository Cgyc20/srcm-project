from __future__ import annotations

import numpy as np
import pytest

from srcm import Domain
from srcm import SimulationResults
from srcm import ResultsIO


def make_domain():
    return Domain(length=1.0, n_ssa=4, pde_multiple=2, boundary="periodic")


def make_timeseries_results():
    domain = make_domain()
    species = ["U", "V"]

    time = np.array([0.0, 0.1, 0.2])
    ssa = np.arange(2 * domain.K * len(time), dtype=int).reshape(2, domain.K, len(time))
    pde = np.arange(2 * domain.n_pde * len(time), dtype=float).reshape(2, domain.n_pde, len(time))

    return SimulationResults(
        time=time,
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
    )


def make_trajectory_results():
    domain = make_domain()
    species = ["U", "V"]

    time = np.array([0.0, 0.1, 0.2])
    # (R, S, K, T)
    ssa = np.arange(3 * 2 * domain.K * len(time), dtype=float).reshape(3, 2, domain.K, len(time))
    # (R, S, Npde, T)
    pde = np.arange(3 * 2 * domain.n_pde * len(time), dtype=float).reshape(3, 2, domain.n_pde, len(time))

    return SimulationResults(
        time=time,
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
    )


def test_save_and_load_timeseries(tmp_path):
    res = make_timeseries_results()
    path = tmp_path / "timeseries.npz"

    meta_in = {"run_type": "hybrid", "tag": "smoke"}

    ResultsIO.save(results=res, path=path, meta=meta_in)
    loaded, meta_out = ResultsIO.load(path)

    assert isinstance(loaded, SimulationResults)
    np.testing.assert_allclose(loaded.time, res.time)
    np.testing.assert_array_equal(loaded.ssa, res.ssa)
    np.testing.assert_allclose(loaded.pde, res.pde)

    assert loaded.species == res.species
    assert loaded.domain.length == pytest.approx(res.domain.length)
    assert loaded.domain.K == res.domain.K
    assert loaded.domain.pde_multiple == res.domain.pde_multiple
    assert loaded.domain.boundary == res.domain.boundary

    assert meta_out["run_type"] == "hybrid"
    assert meta_out["tag"] == "smoke"
    assert meta_out["result_kind"] == "timeseries"
    assert meta_out["species"] == ["U", "V"]


def test_save_and_load_trajectories(tmp_path):
    res = make_trajectory_results()
    path = tmp_path / "trajectories.npz"

    ResultsIO.save(results=res, path=path, meta={"run_type": "ensemble"})
    loaded, meta_out = ResultsIO.load(path)

    assert isinstance(loaded, SimulationResults)
    np.testing.assert_allclose(loaded.time, res.time)
    np.testing.assert_allclose(loaded.ssa, res.ssa)
    np.testing.assert_allclose(loaded.pde, res.pde)

    assert loaded.ssa.shape == res.ssa.shape
    assert loaded.pde.shape == res.pde.shape
    assert meta_out["result_kind"] == "trajectories"


def test_save_timeseries_with_none_pde_materializes_zero_pde(tmp_path):
    domain = make_domain()
    species = ["U", "V"]
    time = np.array([0.0, 0.1, 0.2])
    ssa = np.ones((2, domain.K, len(time)), dtype=int)

    res = SimulationResults(
        time=time,
        ssa=ssa,
        pde=None,
        domain=domain,
        species=species,
    )

    path = tmp_path / "timeseries_zero_pde.npz"
    ResultsIO.save(results=res, path=path)

    loaded, meta_out = ResultsIO.load(path)

    assert isinstance(loaded, SimulationResults)
    assert loaded.pde.shape == (2, domain.n_pde, len(time))
    assert np.allclose(loaded.pde, 0.0)
    assert meta_out["result_kind"] == "timeseries"


def test_save_trajectories_with_none_pde_materializes_zero_pde(tmp_path):
    domain = make_domain()
    species = ["U", "V"]
    time = np.array([0.0, 0.1, 0.2])

    # (R, S, K, T)
    ssa = np.ones((5, 2, domain.K, len(time)), dtype=float)

    res = SimulationResults(
        time=time,
        ssa=ssa,
        pde=None,
        domain=domain,
        species=species,
    )

    path = tmp_path / "traj_zero_pde.npz"
    ResultsIO.save(results=res, path=path)

    loaded, meta_out = ResultsIO.load(path)

    assert isinstance(loaded, SimulationResults)
    assert loaded.pde.shape == (5, 2, domain.n_pde, len(time))
    assert np.allclose(loaded.pde, 0.0)
    assert meta_out["result_kind"] == "trajectories"


def test_save_and_load_final_only(tmp_path):
    domain = make_domain()
    species = ["U", "V"]

    final_ssa = np.arange(7 * 2 * domain.K, dtype=int).reshape(7, 2, domain.K)
    final_pde = np.arange(7 * 2 * domain.n_pde, dtype=float).reshape(7, 2, domain.n_pde)
    t_final = 12.5

    path = tmp_path / "final_only.npz"

    ResultsIO.save(
        path=path,
        final_ssa=final_ssa,
        final_pde=final_pde,
        t_final=t_final,
        domain=domain,
        species=species,
        meta={"run_type": "final_only", "seed0": 123},
    )

    loaded, meta_out = ResultsIO.load(path)

    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(loaded["final_ssa"], final_ssa)
    np.testing.assert_allclose(loaded["final_pde"], final_pde)
    assert loaded["t_final"] == pytest.approx(t_final)

    assert loaded["species"] == species
    assert loaded["domain"].length == pytest.approx(domain.length)
    assert loaded["domain"].K == domain.K
    assert loaded["domain"].pde_multiple == domain.pde_multiple
    assert loaded["domain"].boundary == domain.boundary

    assert meta_out["run_type"] == "final_only"
    assert meta_out["seed0"] == 123
    assert meta_out["result_kind"] == "final_only"


def test_save_final_only_requires_all_inputs(tmp_path):
    domain = make_domain()
    path = tmp_path / "bad_final.npz"

    final_ssa = np.zeros((2, 2, domain.K), dtype=int)
    final_pde = np.zeros((2, 2, domain.n_pde), dtype=float)

    with pytest.raises(ValueError, match="final-only"):
        ResultsIO.save(
            path=path,
            final_ssa=final_ssa,
            final_pde=final_pde,
            t_final=1.0,
            domain=domain,
            species=None,
        )


def test_save_rejects_unsupported_ssa_shape(tmp_path):
    domain = make_domain()
    species = ["U", "V"]

    # invalid rank-2 SSA
    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    res = SimulationResults(
        time=np.array([0.0]),
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
    )

    with pytest.raises(ValueError, match="Unsupported SSA shape"):
        ResultsIO.save(results=res, path=tmp_path / "bad_shape.npz")


def test_save_from_engine_run_with_results(tmp_path):
    res = make_timeseries_results()
    path = tmp_path / "wrapper_results.npz"

    ResultsIO.save_from_engine_run(
        path=path,
        results=res,
        meta={"note": "timeseries"},
    )

    loaded, meta_out = ResultsIO.load(path)
    assert isinstance(loaded, SimulationResults)
    np.testing.assert_array_equal(loaded.ssa, res.ssa)
    assert meta_out["note"] == "timeseries"


def test_save_from_engine_run_with_final_tuple(tmp_path):
    domain = make_domain()
    species = ["U", "V"]

    final_ssa = np.ones((4, 2, domain.K), dtype=int)
    final_pde = np.ones((4, 2, domain.n_pde), dtype=float)
    t_final = 3.0

    path = tmp_path / "wrapper_final.npz"

    ResultsIO.save_from_engine_run(
        path=path,
        final_tuple=(final_ssa, final_pde, t_final),
        domain=domain,
        species=species,
        meta={"note": "final tuple"},
    )

    loaded, meta_out = ResultsIO.load(path)
    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(loaded["final_ssa"], final_ssa)
    np.testing.assert_allclose(loaded["final_pde"], final_pde)
    assert loaded["t_final"] == pytest.approx(t_final)
    assert meta_out["note"] == "final tuple"


def test_save_from_engine_run_requires_input(tmp_path):
    with pytest.raises(ValueError, match="Provide either results"):
        ResultsIO.save_from_engine_run(path=tmp_path / "empty.npz")