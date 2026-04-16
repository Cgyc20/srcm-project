import numpy as np
from srcm_engine.core import gillespie_draw


def test_gillespie_zero_propensity():
    rng = np.random.default_rng(0)
    a = np.zeros(5)
    tau, idx = gillespie_draw(a, rng)
    assert tau == np.inf
    assert idx == -1


def test_gillespie_single_channel_always_selected():
    rng = np.random.default_rng(1)
    a = np.array([0.0, 0.0, 3.0, 0.0])
    for _ in range(20):
        tau, idx = gillespie_draw(a, rng)
        assert np.isfinite(tau)
        assert idx == 2


def test_gillespie_with_preallocated_cumulative():
    rng = np.random.default_rng(2)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    cum = np.empty_like(a)

    tau, idx = gillespie_draw(a, rng, cumulative=cum)

    assert np.isfinite(tau)
    assert 0 <= idx < a.size
    # cumulative should be increasing and end at sum
    assert np.all(np.diff(cum) >= 0)
    assert np.isclose(cum[-1], np.sum(a))
