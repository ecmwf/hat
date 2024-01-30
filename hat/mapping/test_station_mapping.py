import numpy as np
import pandas as pd
import pytest

from hat.mapping.station_mapping import process_station_data

# Test data
station = {
    "lat_col": "latitude",
    "lon_col": "longitude",
    "station_name_col": "station_name",
    "csv_variable": "area",
    "manual_lat_col": "manual_latitude",
    "manual_lon_col": "manual_longitude",
    "manual_area": "manual_area",
}
latitudes = np.array([0, 1, 2, 3])
longitudes = np.array([10, 20, 30, 40])
nc_data = np.array([[100, 200, 300, 400], [500, 600, 700, 800]])
max_neighboring_cells = 3
min_area_diff = 5.0
max_area_diff = 10.0
cell_size = 1.0

def test_process_station_data():
    expected_result = {
        "station_name": "Station A",
        "station_lat": 1.5,
        "station_lon": 25.0,
        "station_area": 150.0,
        "near_grid_lat_idx": 1,
        "near_grid_lon_idx": 2,
        "near_grid_lat": 1,
        "near_grid_lon": 30,
        "near_grid_area": 700.0,
        "near_grid_polygon": "polygon",
        "optimum_grid_lat_idx": 1,
        "optimum_grid_lon_idx": 2,
        "optimum_grid_lat": 1,
        "optimum_grid_lon": 30,
        "optimum_grid_area": 700.0,
        "optimum_area_diff": 0.0,
        "optimum_distance_km": 0.0,
        "optimum_grid_polygon": "polygon",
        "manual_lat": 1.5,
        "manual_lon": 25.0,
        "manual_lat_idx": 1,
        "manual_lon_idx": 2,
        "manual_area": 150.0,
    }

    result = process_station_data(
        station=station,
        latitudes=latitudes,
        longitudes=longitudes,
        nc_data=nc_data,
        max_neighboring_cells=max_neighboring_cells,
        min_area_diff=min_area_diff,
        max_area_diff=max_area_diff,
        cell_size=cell_size,
    )

    assert result == expected_result

def test_process_station_data_with_missing_data():
    station_with_missing_data = {
        "lat_col": "latitude",
        "lon_col": "longitude",
        "station_name_col": "station_name",
        "csv_variable": None,
        "manual_lat_col": "manual_latitude",
        "manual_lon_col": "manual_longitude",
        "manual_area": "manual_area",
    }

    expected_result = {
        "station_name": "Station B",
        "station_lat": 2.5,
        "station_lon": 35.0,
        "station_area": np.nan,
        "near_grid_lat_idx": 2,
        "near_grid_lon_idx": 3,
        "near_grid_lat": 2,
        "near_grid_lon": 40,
        "near_grid_area": np.nan,
        "near_grid_polygon": "polygon",
        "optimum_grid_lat_idx": 2,
        "optimum_grid_lon_idx": 3,
        "optimum_grid_lat": 2,
        "optimum_grid_lon": 40,
        "optimum_grid_area": np.nan,
        "optimum_area_diff": np.nan,
        "optimum_distance_km": np.nan,
        "optimum_grid_polygon": "polygon",
        "manual_lat": 2.5,
        "manual_lon": 35.0,
        "manual_lat_idx": 2,
        "manual_lon_idx": 3,
        "manual_area": 250.0,
    }

    result = process_station_data(
        station=station_with_missing_data,
        latitudes=latitudes,
        longitudes=longitudes,
        nc_data=nc_data,
        max_neighboring_cells=max_neighboring_cells,
        min_area_diff=min_area_diff,
        max_area_diff=max_area_diff,
        cell_size=cell_size,
    )

    assert result == expected_result