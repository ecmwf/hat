"""Tests for hat.station_mapping.station_mapping.StationMapping."""

import numpy as np
import pytest

from hat.station_mapping.station_mapping import StationMapping


@pytest.fixture
def regular_grid():
    """10x10 grid with lat 0-9, lon 0-9."""
    lats = np.arange(10, dtype=float)
    lons = np.arange(10, dtype=float)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return grid_lat, grid_lon


@pytest.fixture
def known_metric_grid():
    """Grid where cell (3,3) has lowest metric value."""
    grid = np.full((10, 10), 100.0)
    grid[3, 3] = 10.0  # best cell
    return grid


class TestConductMappingNoMetric:
    """Without metric, should map to nearest grid cell."""

    def test_maps_to_nearest(self, regular_grid):
        grid_lat, grid_lon = regular_grid
        config = {"max_search_distance": 5, "metric_error_func": "zero", "distance_error_func": "mae"}
        sm = StationMapping(config)

        station_lats = np.array([3.2])
        station_lons = np.array([7.1])

        indxs, indys, cindxs, cindys, errors = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric=None, grid_metric=None
        )

        # Nearest cell to (3.2, 7.1) is (3, 7)
        assert cindxs[0] == 3
        assert cindys[0] == 7

    def test_maps_to_minimum_distance(self, regular_grid):
        grid_lat, grid_lon = regular_grid
        config = {"max_search_distance": 5, "metric_error_func": "zero", "distance_error_func": "mae", "lambda": 1.0}
        sm = StationMapping(config)

        station_lats = np.array([5.0])
        station_lons = np.array([5.0])

        indxs, indys, _, _, errors = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric=None, grid_metric=None
        )

        # Exact grid point → error should be 0
        assert indxs[0] == 5
        assert indys[0] == 5
        np.testing.assert_almost_equal(errors[0], 0.0)


class TestConductMappingWithMetric:
    """With metric, should balance metric error and distance."""

    def test_lambda_zero_picks_best_metric(self, regular_grid, known_metric_grid):
        grid_lat, grid_lon = regular_grid
        config = {"max_search_distance": 5, "metric_error_func": "mae", "distance_error_func": "mae", "lambda": 0}
        sm = StationMapping(config)

        # Station at (5,5), metric value 10 → best match is cell (3,3) with value 10
        station_lats = np.array([5.0])
        station_lons = np.array([5.0])
        station_metric = np.array([10.0])

        indxs, indys, _, _, _ = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric, known_metric_grid
        )

        assert indxs[0] == 3
        assert indys[0] == 3

    def test_high_lambda_prefers_distance(self, regular_grid, known_metric_grid):
        grid_lat, grid_lon = regular_grid
        config = {"max_search_distance": 5, "metric_error_func": "mae", "distance_error_func": "mae", "lambda": 1000}
        sm = StationMapping(config)

        # With very high lambda, distance dominates → should pick nearest
        station_lats = np.array([5.0])
        station_lons = np.array([5.0])
        station_metric = np.array([10.0])

        indxs, indys, cindxs, cindys, _ = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric, known_metric_grid
        )

        assert indxs[0] == 5
        assert indys[0] == 5


class TestMaxMinError:
    def test_max_error_reverts_to_nearest(self, regular_grid, known_metric_grid):
        grid_lat, grid_lon = regular_grid
        # All errors are >= 90 (|station_metric - grid|), set max_error very low
        config = {
            "max_search_distance": 5,
            "metric_error_func": "mae",
            "distance_error_func": "zero",
            "lambda": 0,
            "max_error": 0.001,
        }
        sm = StationMapping(config)

        station_lats = np.array([5.0])
        station_lons = np.array([5.0])
        station_metric = np.array([999.0])  # far from all grid values

        indxs, indys, cindxs, cindys, _ = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric, known_metric_grid
        )

        # Should revert to nearest
        assert indxs[0] == cindxs[0]
        assert indys[0] == cindys[0]

    def test_min_error_accepts_nearest(self, regular_grid):
        grid_lat, grid_lon = regular_grid
        # Grid metric at (5,5) = station_metric → error = 0 ≤ min_error
        metric_grid = np.full((10, 10), 100.0)
        metric_grid[5, 5] = 50.0
        metric_grid[3, 3] = 50.0  # same value elsewhere

        config = {
            "max_search_distance": 5,
            "metric_error_func": "mae",
            "distance_error_func": "zero",
            "lambda": 0,
            "min_error": 1.0,
        }
        sm = StationMapping(config)

        station_lats = np.array([5.0])
        station_lons = np.array([5.0])
        station_metric = np.array([50.0])  # exact match at nearest

        indxs, indys, cindxs, cindys, _ = sm.conduct_mapping(
            station_lats, station_lons, grid_lat, grid_lon, station_metric, metric_grid
        )

        # nearest cell error ≤ min_error → accepts nearest
        assert indxs[0] == 5
        assert indys[0] == 5


class TestMultipleStations:
    def test_maps_multiple(self, regular_grid):
        grid_lat, grid_lon = regular_grid
        config = {"max_search_distance": 5, "metric_error_func": "zero", "distance_error_func": "mae", "lambda": 1.0}
        sm = StationMapping(config)

        station_lats = np.array([1.0, 5.0, 8.0])
        station_lons = np.array([1.0, 5.0, 8.0])

        indxs, indys, _, _, errors = sm.conduct_mapping(station_lats, station_lons, grid_lat, grid_lon)

        assert len(indxs) == 3
        np.testing.assert_array_equal(indxs, [1, 5, 8])
        np.testing.assert_array_equal(indys, [1, 5, 8])
