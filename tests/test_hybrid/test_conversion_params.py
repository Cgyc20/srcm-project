import numpy as np
from srcm import ConversionParams
import pytest
# March 25th 2026: THese are removed as we removed the single threshold.
# def test_exceeds_threshold_mask():
#     conv = ConversionParams(threshold=5, rate=1.0)

#     combined = np.array([[0, 5, 6],
#                           [10, 2, 5]])

#     mask = conv.exceeds_threshold_mask(combined)

#     expected = np.array([[0, 0, 1],
#                           [1, 0, 0]])

#     assert np.array_equal(mask, expected)


# def test_sufficient_pde_mass_mask():
#     conv = ConversionParams(threshold=10, rate=1.0)

#     h = 0.5
#     pde_multiple = 2

#     # shape (n_species=1, Npde=4)
#     pde = np.array([[2.0, 2.0, 1.9, 2.0]])  # threshold = 1/h = 2.0

#     mask = conv.sufficient_pde_mass_mask(pde, pde_multiple, h)

#     expected = np.array([[1, 0]])
#     assert np.array_equal(mask, expected)


def test_single_threshold_initialization():
    # Test Scalar
    conv = ConversionParams(threshold=10, rate=1.0)
    assert conv.DC_threshold == 10
    assert conv.CD_threshold == 10
    
    # Test Array (Per-species)
    thr_arr = np.array([5.0, 15.0])
    conv_arr = ConversionParams(threshold=thr_arr, rate=1.0)
    assert np.array_equal(conv_arr.DC_threshold, thr_arr)
    assert np.array_equal(conv_arr.CD_threshold, thr_arr)


def test_invalid_threshold_order_raises():
    with pytest.raises(ValueError):
        ConversionParams(
            DC_threshold=5,
            CD_threshold=6,   # invalid: CD > DC
            rate=1.0
        )


def test_DC_mask():
    conv = ConversionParams(DC_threshold=5, CD_threshold=3, rate=1.0)

    combined = np.array([
        [0, 5, 6],
        [10, 2, 5],
    ])

    mask = conv.DC_mask(combined)

    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int8)

    assert np.array_equal(mask, expected)


def test_CD_mask():
    conv = ConversionParams(DC_threshold=5, CD_threshold=3, rate=1.0)

    combined = np.array([
        [0, 3, 6],
        [10, 2, 5],
    ])

    mask = conv.CD_mask(combined)

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.int8)

    assert np.array_equal(mask, expected)


def test_sufficient_pde_mass_mask():
    conv = ConversionParams(DC_threshold=10, CD_threshold=5, rate=1.0)

    h = 0.5
    pde_multiple = 2

    # shape (n_species=1, Npde=4)
    pde = np.array([[2.0, 2.0, 1.9, 2.0]])  # threshold = 1/h = 2.0

    mask = conv.sufficient_pde_mass_mask(pde, pde_multiple, h)

    expected = np.array([[1, 0]], dtype=np.int8)
    assert np.array_equal(mask, expected)