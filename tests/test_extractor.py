"""Unit tests for the extractor function."""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch

from hat.extract_timeseries.extractor import extractor


@pytest.fixture
def dummy_grid_data():
    """4x5 grid, 2 timesteps, temperature variable."""
    lats = np.array([40.0, 41.0, 42.0, 43.0])
    lons = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    temperature_values = np.array(
        [
            # t=0: 2024-01-01
            [
                [10.0, 11.0, 12.0, 13.0, 14.0],  # lat=40
                [15.0, 16.0, 17.0, 18.0, 19.0],  # lat=41
                [20.0, 21.0, 22.0, 23.0, 24.0],  # lat=42
                [25.0, 26.0, 27.0, 28.0, 29.0],  # lat=43
            ],
            # t=1: 2024-01-02
            [
                [30.0, 31.0, 32.0, 33.0, 34.0],  # lat=40
                [35.0, 36.0, 37.0, 38.0, 39.0],  # lat=41
                [40.0, 41.0, 42.0, 43.0, 44.0],  # lat=42
                [45.0, 46.0, 47.0, 48.0, 49.0],  # lat=43
            ],
        ]
    )

    list_of_dicts = []

    list_of_dicts.append(
        {
            "values": temperature_values[0].flatten(),
            "param": "temperature",
            "date": 20240101,
            "time": 0,
            "distinctLatitudes": lats.tolist(),
            "distinctLongitudes": lons.tolist(),
        }
    )

    list_of_dicts.append(
        {
            "values": temperature_values[1].flatten(),
            "param": "temperature",
            "date": 20240102,
            "time": 0,
            "distinctLatitudes": lats.tolist(),
            "distinctLongitudes": lons.tolist(),
        }
    )

    return list_of_dicts


@pytest.fixture
def station_dataframe():
    """2 stations with both index and coordinate columns."""
    return pd.DataFrame(
        {
            "station_id": ["STATION_A", "STATION_B"],
            "opt_x_index": [1, 2],
            "opt_y_index": [2, 3],
            "opt_x_coord": [41.1, 41.9],  # offset to test nearest-neighbor
            "opt_y_coord": [12.2, 13.1],
        }
    )


@pytest.fixture
def station_csv_file(station_dataframe, tmp_path):
    """Write station DataFrame to temporary CSV."""
    csv_path = tmp_path / "stations.csv"
    station_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.mark.parametrize(
    "mapping_config",
    [
        {"index": {"x": "opt_x_index", "y": "opt_y_index"}},
        {"coords": {"x": "opt_x_coord", "y": "opt_y_coord"}},
    ],
    ids=["index", "coords"],
)
def test_extractor_with_temperature(dummy_grid_data, station_csv_file, mapping_config):
    """Test extraction with both index and coords mapping."""
    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            **mapping_config,
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
    }

    result_ds = extractor(config)

    assert isinstance(result_ds, xr.Dataset)
    assert "temperature" in result_ds.data_vars
    assert "station" in result_ds.dims
    assert len(result_ds.station) == 2
    assert list(result_ds.station.values) == ["STATION_A", "STATION_B"]

    time_dim = "time" if "time" in result_ds.dims else "forecast_reference_time"
    assert len(result_ds[time_dim]) == 2

    # Station A: indices [1,2] -> values [17.0, 37.0]
    # Station B: indices [2,3] -> values [23.0, 43.0]
    np.testing.assert_allclose(result_ds["temperature"].sel(station="STATION_A").values, [17.0, 37.0])
    np.testing.assert_allclose(result_ds["temperature"].sel(station="STATION_B").values, [23.0, 43.0])


def test_extractor_with_station_filter(dummy_grid_data, tmp_path):
    """Test station filtering."""
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S2", "S3"],
            "opt_x_index": [1, 2, 1],
            "opt_y_index": [2, 3, 3],
            "network": ["primary", "secondary", "primary"],
        }
    )
    csv_path = tmp_path / "stations.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "station": {
            "file": str(csv_path),
            "name": "station_id",
            "filter": "network == 'primary'",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
    }

    result_ds = extractor(config)

    assert len(result_ds.station) == 2
    assert list(result_ds.station.values) == ["S1", "S3"]
    assert result_ds["temperature"].sel(station="S1").values[0] == 17.0
    assert result_ds["temperature"].sel(station="S3").values[0] == 18.0


def test_extractor_rejects_both_index_and_coords(dummy_grid_data, station_csv_file):
    """Test that providing both index and coords raises ValueError."""
    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
            "coords": {"x": "opt_x_coord", "y": "opt_y_coord"},  # Both provided
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValueError, match="must use either 'index' or 'coords', not both"):
        extractor(config)


def test_extractor_with_output_file(dummy_grid_data, station_csv_file, tmp_path):
    """Test output file writing."""
    output_file = tmp_path / "output.nc"

    config = {
        "station": {
            "file": station_csv_file,
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {
                "list-of-dicts": {
                    "list_of_dicts": dummy_grid_data,
                }
            },
            "coords": {
                "x": "latitude",
                "y": "longitude",
            },
        },
        "output": {
            "file": str(output_file),
        },
    }

    result_ds = extractor(config)

    assert output_file.exists()

    loaded_ds = xr.open_dataset(output_file)
    assert "temperature" in loaded_ds.data_vars
    assert "station" in loaded_ds.dims
    assert len(loaded_ds.station) == 2

    xr.testing.assert_allclose(result_ds["temperature"], loaded_ds["temperature"])
    xr.testing.assert_equal(result_ds.station, loaded_ds.station)

    loaded_ds.close()


def test_extractor_with_empty_stations(dummy_grid_data, tmp_path):
    """Test that extractor raises clear error for empty station list."""
    empty_csv = tmp_path / "empty_stations.csv"
    pd.DataFrame(columns=["station_id", "opt_x_index", "opt_y_index"]).to_csv(empty_csv, index=False)

    config = {
        "station": {
            "file": str(empty_csv),
            "name": "station_id",
            "index": {"x": "opt_x_index", "y": "opt_y_index"},
        },
        "grid": {
            "source": {"list-of-dicts": {"list_of_dicts": dummy_grid_data}},
            "coords": {"x": "latitude", "y": "longitude"},
        },
    }

    with pytest.raises(ValueError, match="No stations found"):
        extractor(config)


@patch("earthkit.data.from_source")
def test_extractor_gribjump(mock_from_source, tmp_path):
    """Test gribjump path: verifies ranges computation and earthkit call."""

    # Mock returns object with to_xarray() that returns minimal dataset
    mock_source = Mock()
    mock_source.to_xarray.return_value = xr.Dataset(
        {"temperature": xr.DataArray([[15.0, 25.0], [35.0, 45.0]], dims=["index", "time"])}
    )
    mock_from_source.return_value = mock_source

    # Station CSV with index_1d (includes duplicate to test deduplication)
    csv_file = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "name": ["S1", "S2", "S3"],
            "idx": [100, 200, 100],  # S1 and S3 share index 100
        }
    ).to_csv(csv_file, index=False)

    config = {
        "station": {"file": str(csv_file), "name": "name", "index_1d": "idx"},
        "grid": {"source": {"gribjump": {"request": {"class": "od", "expver": "0001", "stream": "oper"}}}},
    }

    result = extractor(config)

    # Verify earthkit.data.from_source was called correctly
    mock_from_source.assert_called_once_with(
        "gribjump", request={"class": "od", "expver": "0001", "stream": "oper"}, ranges=[(100, 101), (200, 201)]
    )

    # Verify output
    assert len(result.station) == 3
    assert list(result.station.values) == ["S1", "S2", "S3"]
