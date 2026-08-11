"""Tests for hat.station_mapping.metrics."""

import numpy as np
import pytest

from hat.station_mapping.metrics import mae, mape, mse, mspe, rmse, zero


class TestZero:
    def test_scalar_val_1d_grid(self):
        result = zero(5.0, np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, 0.0)

    def test_1d_val_2d_grid(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = zero(np.array([10.0]), grid)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, 0.0)


class TestMAE:
    def test_identical_values(self):
        result = mae(np.array([5.0]), np.array([5.0]))
        np.testing.assert_almost_equal(result, 0.0)

    def test_simple_difference(self):
        result = mae(np.array([5.0]), np.array([3.0]))
        np.testing.assert_almost_equal(result, 2.0)

    def test_multi_element(self):
        # val=[1,2,3] vs grid=[4,5,6] → mean(|[-3,-3,-3]|, axis=0)
        # With wrapper: val becomes (3,1), grid becomes (1,3) → broadcast to (3,3)
        # Actually: val (3,) → (3,1), grid (3,) → (1,3)
        # result shape is (3,) from mean over axis=0
        result = mae(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        # Each grid element compared against all val elements:
        # grid[0]=4: |1-4|=3, |2-4|=2, |3-4|=1 → mean=2
        # grid[1]=5: |1-5|=4, |2-5|=3, |3-5|=2 → mean=3
        # grid[2]=6: |1-6|=5, |2-6|=4, |3-6|=3 → mean=4
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scalar_val(self):
        result = mae(5.0, np.array([3.0, 7.0, 5.0]))
        np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0, 0.0]))


class TestMSE:
    def test_simple(self):
        result = mse(np.array([0.0]), np.array([3.0]))
        np.testing.assert_almost_equal(result, 9.0)

    def test_identical(self):
        result = mse(np.array([7.0]), np.array([7.0]))
        np.testing.assert_almost_equal(result, 0.0)


class TestRMSE:
    def test_simple(self):
        result = rmse(np.array([0.0]), np.array([3.0]))
        np.testing.assert_almost_equal(result, 3.0)

    def test_identical(self):
        result = rmse(np.array([5.0]), np.array([5.0]))
        np.testing.assert_almost_equal(result, 0.0)


class TestMAPE:
    def test_simple(self):
        # |(10 - 12) / 10| = 0.2
        result = mape(np.array([10.0]), np.array([12.0]))
        np.testing.assert_almost_equal(result, 0.2)

    def test_zero_denominator_guard(self):
        # When val is near zero, denominator clipped to 1e-8
        result = mape(np.array([0.0]), np.array([1.0]))
        assert np.isfinite(result).all()

    def test_identical(self):
        result = mape(np.array([10.0]), np.array([10.0]))
        np.testing.assert_almost_equal(result, 0.0)


class TestMSPE:
    def test_simple(self):
        # ((10 - 12) / 10)^2 = 0.04
        result = mspe(np.array([10.0]), np.array([12.0]))
        np.testing.assert_almost_equal(result, 0.04)

    def test_zero_denominator_guard(self):
        result = mspe(np.array([0.0]), np.array([1.0]))
        assert np.isfinite(result).all()


class TestMetricWrapper:
    """Test broadcasting behavior of metric_wrapper."""

    def test_scalar_val_1d_grid(self):
        # scalar val + 1D grid → result shape matches grid length
        result = mae(5.0, np.array([3.0, 7.0, 5.0]))
        assert result.shape == (3,)

    def test_1d_val_2d_grid(self):
        # 1D val + 2D grid → result shape is grid.shape[1:]
        grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        val = np.array([3.0, 3.0])  # (2,)
        result = mae(val, grid)
        assert result.shape == (3,)
