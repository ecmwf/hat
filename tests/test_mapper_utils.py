"""Tests for mapper utility functions."""

import numpy as np
import pandas as pd
import pytest

from hat.station_mapping.mapper import light_zero_color, outputs_to_df


class TestOutputsToDf:
    def test_adds_columns(self, sample_station_df):
        df = sample_station_df.copy()
        grid_lat = np.arange(10, dtype=float).reshape(10, 1) * np.ones((1, 10))
        grid_lon = np.ones((10, 1)) * np.arange(10, dtype=float).reshape(1, 10)

        indx = np.array([2, 5, 8])
        indy = np.array([1, 4, 7])
        cindx = np.array([2, 5, 8])
        cindy = np.array([1, 4, 7])
        errors = np.array([0.1, 0.2, 0.3])

        result = outputs_to_df(df, indx, indy, cindx, cindy, errors, grid_lat, grid_lon, (10, 10), filename=None)

        assert "opt_x_index" in result.columns
        assert "opt_y_index" in result.columns
        assert "opt_error" in result.columns
        assert "opt_1d_index" in result.columns
        np.testing.assert_array_equal(result["opt_x_index"].values, indx)
        np.testing.assert_array_equal(result["opt_error"].values, errors)

    def test_writes_csv(self, sample_station_df, tmp_path):
        df = sample_station_df.copy()
        grid_lat = np.arange(10, dtype=float).reshape(10, 1) * np.ones((1, 10))
        grid_lon = np.ones((10, 1)) * np.arange(10, dtype=float).reshape(1, 10)

        indx = np.array([2, 5, 8])
        indy = np.array([1, 4, 7])
        cindx = np.array([2, 5, 8])
        cindy = np.array([1, 4, 7])
        errors = np.array([0.1, 0.2, 0.3])

        outfile = tmp_path / "output.csv"
        outputs_to_df(df, indx, indy, cindx, cindy, errors, grid_lat, grid_lon, (10, 10), filename=str(outfile))

        assert outfile.exists()
        loaded = pd.read_csv(outfile)
        assert len(loaded) == 3
        assert "opt_x_index" in loaded.columns


class TestLightZeroColor:
    def test_first_entry_transparent(self):
        result = light_zero_color("Viridis")
        assert result[0][0] == 0.0
        assert result[0][1] == "rgba(0,0,0,0)"

    def test_length_matches_base(self):
        from plotly.colors import get_colorscale

        base = get_colorscale("Viridis")
        result = light_zero_color("Viridis")
        assert len(result) == len(base)

    def test_custom_zero_color(self):
        result = light_zero_color("Viridis", zero_color="white")
        assert result[0][1] == "white"
