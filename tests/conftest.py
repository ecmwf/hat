"""Shared fixtures for HAT tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_grid_coords():
    """10x10 grid with regular lat/lon spacing."""
    lats = np.linspace(40, 50, 10)
    lons = np.linspace(0, 10, 10)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return grid_lat, grid_lon


@pytest.fixture
def simple_metric_grid():
    """10x10 grid with known metric values."""
    rng = np.random.default_rng(42)
    return rng.uniform(100, 1000, size=(10, 10))


@pytest.fixture
def sample_station_df():
    """Small DataFrame mimicking station input."""
    return pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "lat": [42.0, 45.0, 48.0],
            "lon": [2.0, 5.0, 8.0],
            "upstream_area": [500.0, 1000.0, 750.0],
        }
    )
