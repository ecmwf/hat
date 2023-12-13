#!/usr/bin/env python3

import argparse
import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from netCDF4 import Dataset
from numpy.ma import is_masked
from geopy.distance import geodesic

def get_grid_index(lat, lon, latitudes, longitudes):
    """Find the index of the nearest grid cell to the given lat/lon."""
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return lat_idx, lon_idx

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance in kilometers between two points."""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def area_diff_percentage(old_value, new_value):
    """Calculate the area difference as a percentage."""
    if old_value == 0:
        return np.nan  # Avoid division by zero
    return ((new_value - old_value) / old_value) * 100

def find_best_matching_grid(lat, lon, latitudes, longitudes, nc_data, csv_value, max_neighboring_cells, max_area_diff):
    lat_idx, lon_idx = get_grid_index(lat, lon, latitudes, longitudes)
    min_diff = float('inf')
    best_match = (lat_idx, lon_idx)

    # Define the search bounds based on max_neighboring_cells
    lat_start = max(lat_idx - max_neighboring_cells, 0)
    lat_end = min(lat_idx + max_neighboring_cells + 1, len(latitudes))
    lon_start = max(lon_idx - max_neighboring_cells, 0)
    lon_end = min(lon_idx + max_neighboring_cells + 1, len(longitudes))

    # Iterate over the neighboring cells within the bounds
    for i in range(lat_start, lat_end):
        for j in range(lon_start, lon_end):
            grid_data = nc_data[i, j]
            if is_masked(grid_data):
                continue
            grid_data = float(grid_data)
            area_diff = area_diff_percentage(grid_data, csv_value)
            if abs(area_diff) < min_diff and abs(area_diff) <= max_area_diff:
                min_diff = abs(area_diff)
                best_match = (i, j)
    
    return best_match

def process_station_data(station, latitudes, longitudes, nc_data, max_neighboring_cells, max_area_diff, lat_col, lon_col, station_name_col, csv_variable):
    """Process data for a single station."""
    lat, lon = station[lat_col], station[lon_col]
    csv_data = station[csv_variable]

    # Original grid point
    lat_idx, lon_idx = get_grid_index(lat, lon, latitudes, longitudes)
    original_grid_data = nc_data[lat_idx, lon_idx]
    original_grid_data = float(original_grid_data) if not is_masked(original_grid_data) else np.nan
    original_area_diff = area_diff_percentage(original_grid_data, csv_data)

    # Best matching grid point
    new_lat_idx, new_lon_idx = find_best_matching_grid(lat, lon, latitudes, longitudes, nc_data, csv_data, max_neighboring_cells, max_area_diff)
    new_grid_data = nc_data[new_lat_idx, new_lon_idx]
    new_grid_data = float(new_grid_data) if not is_masked(new_grid_data) else np.nan
    new_area_diff = area_diff_percentage(new_grid_data, csv_data)

    return {
        'station_name': station[station_name_col],
        'station_lat': lat,
        'station_lon': lon,
        'grid_lat': latitudes[lat_idx],
        'grid_lon': longitudes[lon_idx],
        'nc_variable': original_grid_data,
        'csv_variable': csv_data,
        'original_area_diff': original_area_diff,
        'distance_km': calculate_distance(lat, lon, latitudes[lat_idx], longitudes[lon_idx]),
        'new_grid_lat': latitudes[new_lat_idx],
        'new_grid_lon': longitudes[new_lon_idx],
        'new_nc_variable': new_grid_data,
        'new_area_diff': new_area_diff,
        'new_distance_km': calculate_distance(lat, lon, latitudes[new_lat_idx], longitudes[new_lon_idx])
    }

def main(config):
    netcdf_file = config["netcdf_file"]
    nc_variable = config["nc_variable"]
    csv_file = config["csv_file"]
    station_name_col = config["station_name_col"]
    lat_col = config["lat_col"]
    lon_col = config["lon_col"]
    csv_variable = config["csv_variable"]
    max_neighboring_cells = config["max_neighboring_cells"]
    max_area_diff = config["max_area_diff"]

    # Read station CSV and filter out invalid data
    stations = pd.read_csv(csv_file)
    stations = stations[stations[csv_variable] > 0]

    # Load netCDF data
    dataset = Dataset(netcdf_file, 'r')
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    nc_data = dataset.variables[nc_variable][:]

    # Process each station
    data_list = [process_station_data(station, latitudes, longitudes, nc_data, max_neighboring_cells, max_area_diff, lat_col, lon_col, station_name_col, csv_variable) for index, station in stations.iterrows()]
    df = pd.DataFrame(data_list)

    # Create GeoDataFrames
    gdf_station_point = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['station_lon'], df['station_lat'])])
    gdf_original_grid_point = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['grid_lon'], df['grid_lat'])])
    gdf_new_grid_point = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['new_grid_lon'], df['new_grid_lat'])])
    gdf_line = gpd.GeoDataFrame(df, geometry=[LineString([(row['station_lon'], row['station_lat']), (row['grid_lon'], row['grid_lat'])]) for index, row in df.iterrows()])
    gdf_line_new = gpd.GeoDataFrame(df, geometry=[LineString([(row['station_lon'], row['station_lat']), (row['new_grid_lon'], row['new_grid_lat'])]) for index, row in df.iterrows()])

    # Saving GeoJSON files
    gdf_station_point.to_file("station.geojson", driver="GeoJSON")
    gdf_original_grid_point.to_file("original_grid_point.geojson", driver="GeoJSON")
    gdf_new_grid_point.to_file("new_grid_point.geojson", driver="GeoJSON")
    gdf_line.to_file("station2grid_line.geojson", driver="GeoJSON")
    gdf_line_new.to_file("station2grid_new_line.geojson", driver="GeoJSON")

    dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process netCDF and station data.')
    parser.add_argument('config_file', type=str, help='Path to the JSON configuration file')

    args = parser.parse_args()

    # Load configuration from the specified JSON file
    with open(args.config_file, 'r') as file:
        config = json.load(file)

    main(config)
