import numpy as np
import pandas as pd
import pytest
from geopy.distance import geodesic

from hat.mapping.station_mapping import (
    calculate_area_diff_percentage,
    calculate_distance,
    calculate_distance_cells,
    create_grid_polygon,
    find_best_matching_grid,
    get_grid_index,
    process_station_data,
)


# Test for get_grid_index function
def test_get_grid_index():
    latitudes = np.array([0, 1, 2, 3, 4])
    longitudes = np.array([0, 1, 2, 3, 4])
    lat, lon = 2.5, 2.5
    lat_idx, lon_idx = get_grid_index(lat, lon, latitudes, longitudes)
    assert lat_idx == 2 and lon_idx == 2


# Test for calculate_distance function
def test_calculate_distance():
    lat1, lon1 = 0, 0
    lat2, lon2 = 1, 1
    expected_distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    assert pytest.approx(distance) == expected_distance


# Test for calculate_distance_cells function
def test_calculate_distance_cells():
    distance = calculate_distance_cells(0, 0, 3, 4)
    assert distance == 5  # Assuming a 3-4-5 right triangle


# Test for calculate_area_diff_percentage function
def test_calculate_area_diff_percentage():
    eval_value = 80
    ref_value = 100
    expected_diff_percentage = 20
    diff_percentage = calculate_area_diff_percentage(eval_value, ref_value)
    assert diff_percentage == expected_diff_percentage


# Test for find_best_matching_grid function
def test_find_best_matching_grid():
    lat, lon = 2.5, 2.5
    latitudes = np.array([0, 1, 2, 3, 4])
    longitudes = np.array([0, 1, 2, 3, 4])
    nc_data = np.full((5, 5), 1)  # initialise mock data with all value of 1
    nc_data[3, 3] = 0.5  # set one value to 0.5 as the optimum grid value
    csv_value = 0.5
    max_neighboring_cells = 1
    max_area_diff = 10
    lat_idx, lon_idx = find_best_matching_grid(
        lat,
        lon,
        latitudes,
        longitudes,
        nc_data,
        csv_value,
        max_neighboring_cells,
        max_area_diff,
    )
    assert lat_idx == 2 and lon_idx == 2


# Test for create_grid_polygon function
def test_create_grid_polygon():
    lat, lon = 2.5, 2.5
    cell_size = 1
    polygon = create_grid_polygon(lat, lon, cell_size)
    assert polygon.bounds == (2, 2, 3, 3)  # Check if the polygon bounds are as expected


# Mock function to simulate netCDF data access
def mock_nc_data(lat_idx, lon_idx):
    """Simulate netCDF data access"""
    return np.random.rand()


@pytest.fixture
def mock_station():
    """Provides a mock station dictionary"""
    return {
        "lat_col": 2.5,  # Assuming latitude
        "lon_col": 2.5,  # Assuming longitude
        "station_name_col": "Test Station",  # Station name
        "csv_variable": 0.5,  # Mock variable of ups. area
    }


@pytest.fixture
def mock_latitudes_longitudes():
    """Provides mock latitudes and longitudes arrays"""
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def mock_nc_data():
    """Provides mock netCDF (nc_data) array with
    integer values ranging from 1 to 5."""
    # Generates a 5x5 grid with values from 1 to 5 (inclusive)
    mock_nc = np.random.randint(1, 6, size=(5, 5))
    mock_nc[3, 3] = 0.51  # Set neighboring value as optimum grid
    return mock_nc


def test_process_station_data(mock_station, mock_latitudes_longitudes, mock_nc_data):
    max_neighboring_cells = 1
    min_area_diff = 10
    max_area_diff = 20
    cell_size = 1

    # Unpack mock latitudes and longitudes
    latitudes, longitudes = mock_latitudes_longitudes

    processed_data = process_station_data(
        mock_station,
        latitudes,
        longitudes,
        mock_nc_data,
        max_neighboring_cells,
        min_area_diff,
        max_area_diff,
        "lat_col",
        "lon_col",
        "station_name_col",
        "csv_variable",
        cell_size,
        None,  # Assuming no manual mapping in this test
        None,
        None,
    )

    # Assert the structure of the returned data
    assert "station_name" in processed_data, "Station name is missing"
    assert isinstance(
        processed_data["station_name"], str
    ), "Station name should be a string"

    assert "station_lat" in processed_data, "Station latitude is missing"
    assert isinstance(
        processed_data["station_lat"], float
    ), "Station latitude should be a float"

    assert "station_lon" in processed_data, "Station longitude is missing"
    assert isinstance(
        processed_data["station_lon"], float
    ), "Station longitude should be a float"

    assert "station_area" in processed_data, "Station area is missing"
    assert isinstance(
        processed_data["station_area"], float
    ), "Station area should be a float"

    assert "near_grid_lat" in processed_data, "Near grid latitude is missing"
    assert isinstance(
        processed_data["near_grid_lat"], float
    ), "Near grid latitude should be a float"

    assert "near_grid_lon" in processed_data, "Near grid longitude is missing"
    assert isinstance(
        processed_data["near_grid_lon"], float
    ), "Near grid longitude should be a float"

    assert "near_grid_area" in processed_data, "Near grid area is missing"
    assert isinstance(
        processed_data["near_grid_area"], float
    ), "Near grid area should be a float"

    assert "near_area_diff" in processed_data, "Near area difference is missing"
    assert isinstance(
        processed_data["near_area_diff"], float
    ), "Near area difference should be a float"

    assert "optimum_grid_lat" in processed_data, "Optimum grid latitude is missing"
    assert isinstance(
        processed_data["optimum_grid_lat"], float
    ), "Optimum grid latitude should be a float"

    assert "optimum_grid_lon" in processed_data, "Optimum grid longitude is missing"
    assert isinstance(
        processed_data["optimum_grid_lon"], float
    ), "Optimum grid longitude should be a float"

    assert "optimum_grid_area" in processed_data, "Optimum grid area is missing"
    assert isinstance(
        processed_data["optimum_grid_area"], float
    ), "Optimum grid area should be a float"

    assert "optimum_area_diff" in processed_data, "Optimum area difference is missing"
    assert isinstance(
        processed_data["optimum_area_diff"], float
    ), "Optimum area difference should be a float"

    assert (
        "optimum_distance_km" in processed_data
    ), "Optimum distance in kilometers is missing"
    assert isinstance(
        processed_data["optimum_distance_km"], float
    ), "Optimum distance should be a float"

    # If manually mapping variables are used in your test, include them in assertions
    if "manual_lat" in processed_data:
        assert isinstance(
            processed_data["manual_lat"], float
        ), "Manual latitude should be a float or NaN"

    if "manual_lon" in processed_data:
        assert isinstance(
            processed_data["manual_lon"], float
        ), "Manual longitude should be a float or NaN"

    if "manual_area" in processed_data:
        assert isinstance(
            processed_data["manual_area"], float
        ), "Manual area should be a float or NaN"

    # Assert for grid index values if they are part of the returned data structure
    if "near_grid_lat_idx" in processed_data:
        assert isinstance(
            processed_data["near_grid_lat_idx"], int
        ), "Near grid latitude index should be an integer"

    if "near_grid_lon_idx" in processed_data:
        assert isinstance(
            processed_data["near_grid_lon_idx"], int
        ), "Near grid longitude index should be an integer"

    if "optimum_grid_lat_idx" in processed_data:
        assert isinstance(
            processed_data["optimum_grid_lat_idx"], int
        ), "Optimum grid latitude index should be an integer"

    if "optimum_grid_lon_idx" in processed_data:
        assert isinstance(
            processed_data["optimum_grid_lon_idx"], int
        ), "Optimum grid longitude index should be an integer"
