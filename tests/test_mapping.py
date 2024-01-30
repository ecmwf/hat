import numpy as np
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
    assert lat_idx == 3 and lon_idx == 3


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
    nc_data = np.full((5, 5), 0.5)  # Fixed mock data
    csv_value = 0.5
    max_neighboring_cells = 1
    max_area_diff = 10
    lat_idx, lon_idx = find_best_matching_grid(lat, lon, latitudes, longitudes,
                                               nc_data, csv_value,
                                               max_neighboring_cells, max_area_diff)
    assert lat_idx == 3 and lon_idx == 3


# Test for create_grid_polygon function
def test_create_grid_polygon():
    lat, lon = 2.5, 2.5
    cell_size = 1
    polygon = create_grid_polygon(lat, lon, cell_size)
    assert polygon.bounds == (2, 2, 3, 3)  # Check if the polygon bounds are as expected


# Test for process_station_data function
def test_process_station_data():
    station = {'lat_col': 2.5, 'lon_col': 2.5, 'csv_variable': 0.5}
    latitudes = np.array([0, 1, 2, 3, 4])
    longitudes = np.array([0, 1, 2, 3, 4])
    nc_data = np.full((5, 5), 0.5)  # Fixed mock data
    cell_size = 1
    processed_data = process_station_data(station, latitudes, longitudes,
                                          nc_data, 2, 10, 20, 'lat_col', 'lon_col',
                                          'station_name_col', 'csv_variable', cell_size,
                                          None, None, None)
    assert 'station_name' in processed_data and 'near_grid_lat' in processed_data
