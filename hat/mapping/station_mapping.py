#!/usr/bin/env python3
import argparse
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from numpy.ma import is_masked
from shapely.geometry import LineString, Point, box
from shapely.wkt import loads

import hat
from hat.observations import read_station_metadata_file


def get_grid_index(lat, lon, latitudes, longitudes):
    """Find the index of the nearest grid cell to the given lat/lon."""
    lat_idx = int(np.abs(latitudes - lat)).argmin()
    lon_idx = int(np.abs(longitudes - lon)).argmin()
    return lat_idx, lon_idx


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance in kilometers between two points."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def calculate_distance_cells(lat_idx1, lon_idx1, lat_idx2, lon_idx2):
    """
    Calculate the distance between two points in terms of grid cells.

    Parameters:
    - lat_idx1, lon_idx1: Grid indices for the first point.
    - lat_idx2, lon_idx2: Grid indices for the second point.

    Returns:
    - int: Distance in terms of the number of grid cells.
    """
    lat_diff = abs(lat_idx1 - lat_idx2)
    lon_diff = abs(lon_idx1 - lon_idx2)
    return np.sqrt(lat_diff**2 + lon_diff**2)


def calculate_area_diff_percentage(eval_value, ref_value):
    """Calculate the area difference as a percentage."""
    try:
        ref_value = float(ref_value)
        eval_value = float(eval_value)
    except ValueError:
        return np.nan  # Return NaN if conversion fails

    if ref_value <= 0:
        return np.nan  # Avoid division by zero
    return ((ref_value - eval_value) / ref_value) * 100


def find_best_matching_grid(
    lat,
    lon,
    latitudes,
    longitudes,
    nc_data,
    csv_value,
    max_neighboring_cells,
    max_area_diff,
):
    lat_idx, lon_idx = get_grid_index(lat, lon, latitudes, longitudes)
    min_diff = float("inf")
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
            area_diff = calculate_area_diff_percentage(grid_data, csv_value)
            if abs(area_diff) < min_diff and abs(area_diff) <= max_area_diff:
                min_diff = abs(area_diff)
                best_match = (i, j)

    return best_match


def create_grid_polygon(lat, lon, cell_size):
    """Create a rectangular polygon around the given lat/lon based on cell size."""
    half_cell = cell_size / 2
    return box(lon - half_cell, lat - half_cell, lon + half_cell, lat + half_cell)


def process_station_data(
    station,
    latitudes,
    longitudes,
    nc_data,
    max_neighboring_cells,
    min_area_diff,
    max_area_diff,
    lat_col,
    lon_col,
    station_name_col,
    csv_variable,
    cell_size,
    manual_lat_col,
    manual_lon_col,
    manual_area,
):
    """
    Process data for a single station.

    Parameters:
        station (dict): A dictionary containing station data.
        latitudes (numpy.ndarray): Array of latitude values for the grid cells.
        longitudes (numpy.ndarray): Array of longitude values for the grid cells.
        nc_data (numpy.ndarray): Array of grid cell data.
        max_neighboring_cells (int):
            Maximum number of neighboring cells to consider
            for finding the best matching grid cell.
        min_area_diff (float): Minimum area difference percentage to consider
            for finding the best matching grid cell.
        max_area_diff (float): Maximum area difference percentage to consider
            for finding the best matching grid cell.
        lat_col (str): Column name for latitude in the station data.
        lon_col (str): Column name for longitude in the station data.
        station_name_col (str): Column name for station name in the station data.
        csv_variable (str): Column name for the variable in the station data.
        cell_size (float): Size of each grid cell in kilometers.
        manual_lat_col (str):
            Column name for manually mapped latitude in the station data.
        manual_lon_col (str):
            Column name for manually mapped longitude in the station data.
        manual_area (str):
            Column name for manually mapped area in the station data.

    Returns:
        dict: A dictionary containing processed data for the station,
        nearest grid cell, and best matching grid cell (if applicable).
    """
    lat, lon = float(station[lat_col]), float(station[lon_col])
    station_area = float(station[csv_variable]) if station[csv_variable] else np.nan

    # manually mapped variable
    if manual_area is not None:
        manual_lat = station.get(manual_lat_col, np.nan)
        manual_lon = station.get(manual_lon_col, np.nan)
        manual_lat = (
            float(manual_lat)
            if not pd.isna(manual_lat) and manual_lat != ""
            else np.nan
        )
        manual_lon = (
            float(manual_lon)
            if not pd.isna(manual_lon) and manual_lon != ""
            else np.nan
        )
        manual_lat_idx, manual_lon_idx = get_grid_index(
            manual_lat, manual_lon, latitudes, longitudes
        )
        manual_area = float(station[manual_area]) if station[manual_area] else np.nan

    else:
        manual_lat = np.nan
        manual_lon = np.nan
        manual_area = np.nan
        manual_lat_idx, manual_lon_idx = np.nan, np.nan

    # Nearest grid cell
    lat_idx, lon_idx = get_grid_index(lat, lon, latitudes, longitudes)
    near_grid_area = nc_data[lat_idx, lon_idx]
    near_grid_area = float(near_grid_area) if not is_masked(near_grid_area) else np.nan
    near_area_diff = calculate_area_diff_percentage(near_grid_area, station_area)
    near_grid_polygon = create_grid_polygon(
        latitudes[lat_idx], longitudes[lon_idx], cell_size
    )
    near_distance_km = calculate_distance(
        lat, lon, latitudes[lat_idx], longitudes[lon_idx]
    )

    # if the area difference is greater than the minimum, find the best
    # matching grid cell otherwise use the nearest grid cell
    if near_area_diff >= min_area_diff:
        # Best matching upstream area grid cell within the specified search radius
        optimum_lat_idx, optimum_lon_idx = find_best_matching_grid(
            lat,
            lon,
            latitudes,
            longitudes,
            nc_data,
            station_area,
            max_neighboring_cells,
            max_area_diff,
        )
        optimum_grid_area = nc_data[optimum_lat_idx, optimum_lon_idx]
        optimum_grid_area = (
            float(optimum_grid_area) if not is_masked(optimum_grid_area) else np.nan
        )
        optimum_area_diff = calculate_area_diff_percentage(
            optimum_grid_area, station_area
        )
        optimum_grid_polygon = create_grid_polygon(
            latitudes[optimum_lat_idx], longitudes[optimum_lon_idx], cell_size
        )
        optimum_distance_km = calculate_distance(
            lat, lon, latitudes[optimum_lat_idx], longitudes[optimum_lon_idx]
        )
    else:
        # Use the nearest grid cell as the best matching grid cell
        optimum_lat_idx, optimum_lon_idx = lat_idx, lon_idx
        optimum_grid_area = near_grid_area
        optimum_area_diff = near_area_diff
        optimum_grid_polygon = near_grid_polygon
        optimum_distance_km = near_distance_km

    return {
        # Station data
        "station_name": station[station_name_col],
        "station_lat": lat,
        "station_lon": lon,
        "station_area": station_area,
        # Near grid data
        "near_grid_lat_idx": lat_idx,
        "near_grid_lon_idx": lon_idx,
        "near_grid_lat": latitudes[lat_idx],
        "near_grid_lon": longitudes[lon_idx],
        "near_grid_area": near_grid_area,
        "near_area_diff": near_area_diff,
        "near_grid_polygon": near_grid_polygon,
        # Optimum grid from search
        "optimum_grid_lat_idx": optimum_lat_idx,
        "optimum_grid_lon_idx": optimum_lon_idx,
        "optimum_grid_lat": latitudes[optimum_lat_idx],
        "optimum_grid_lon": longitudes[optimum_lon_idx],
        "optimum_grid_area": optimum_grid_area,
        "optimum_area_diff": optimum_area_diff,
        "optimum_distance_km": optimum_distance_km,
        "optimum_grid_polygon": optimum_grid_polygon,
        # Manually mapped variable
        "manual_lat": manual_lat,
        "manual_lon": manual_lon,
        "manual_lat_idx": manual_lat_idx,
        "manual_lon_idx": manual_lon_idx,
        "manual_area": manual_area,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Station mapping tool: maps stations on provided grid."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    # Load configuration from the specified JSON file
    with open(args.config_file, "r") as file:
        config = json.load(file)
    station_mapping(config)


def station_mapping(config):
    # grid and upstream area file
    upstream_area_file = config["upstream_area_file"]

    # stations file
    csv_file = config["csv_file"]
    csv_ups_col = config["csv_ups_col"]
    station_name_col = config["csv_station_name_col"]
    lat_col = config["csv_lat_col"]
    lon_col = config["csv_lon_col"]
    stations_epsg = config.get("stations_epsg", "4326")
    stations_filter = config.get("stations_filter", "")

    # mapping parameters
    max_neighboring_cells = config["max_neighboring_cells"]
    max_area_diff = config["max_area_diff"]
    min_area_diff = config["min_area_diff"]

    # manually mapped stations - for testing
    manual_lat_col = config.get("manual_lat_col")
    manual_lon_col = config.get("manual_lon_col")
    manual_area = config.get("manual_area")

    # Read station CSV and filter out invalid data
    stations_filter += f",{csv_ups_col} > 0"
    stations = read_station_metadata_file(
        csv_file,
        [lon_col, lat_col],
        stations_epsg,
        stations_filter,
    )

    # Load netCDF data
    dataset = xr.open_dataset(upstream_area_file)
    nc_variable = hat.data.find_main_var(dataset, min_dim=2)
    nc_data = dataset[nc_variable] * 1e-6  # Convert from m^2 to km^2
    latitudes = dataset["lat"].values
    longitudes = dataset["lon"].values
    dataset.close()

    # extract cell size from coordinates
    cell_size = abs(latitudes[0] - latitudes[1])

    # Process each station and collect data in a list
    data_list = []
    for index, station in stations.iterrows():
        # print(index)
        station_data = process_station_data(
            station,
            latitudes,
            longitudes,
            nc_data,
            max_neighboring_cells,
            min_area_diff,
            max_area_diff,
            lat_col,
            lon_col,
            station_name_col,
            csv_ups_col,
            cell_size,
            manual_lat_col,
            manual_lon_col,
            manual_area,
        )
        data_list.append(station_data)
    df = pd.DataFrame(data_list)

    if config["out_directory"]:
        out_dir = config["out_directory"]
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        save_geo_dataframes(df, out_dir, cell_size)

    return df


def save_geo_dataframes(df, out_dir, cell_size):
    df["near_grid_polygon"] = df.apply(
        lambda row: create_grid_polygon(
            row["near_grid_lat"], row["near_grid_lon"], cell_size
        ),
        axis=1,
    )
    df["optimum_grid_polygon"] = df.apply(
        lambda row: create_grid_polygon(
            row["optimum_grid_lat"], row["optimum_grid_lon"], cell_size
        ),
        axis=1,
    )

    # Convert any additional geometry columns to WKT for serialization
    df["near_grid_polygon_wkt"] = df["near_grid_polygon"].apply(lambda x: x.wkt)
    df["optimum_grid_polygon_wkt"] = df["optimum_grid_polygon"].apply(lambda x: x.wkt)

    # Drop the Shapely object columns that were replaced by wkts
    df = df.drop(columns=["near_grid_polygon", "optimum_grid_polygon"])

    # Create line between station and grids
    optimum_lines = df.apply(
        lambda row: LineString(
            [
                (row["station_lon"], row["station_lat"]),
                (row["optimum_grid_lon"], row["optimum_grid_lat"]),
            ]
        ),
        axis=1,
    )

    # Create GeoDataFrames
    gdf_station_point = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df["station_lon"], df["station_lat"])]
    )
    gdf_near_grid_polygon = gpd.GeoDataFrame(
        df, geometry=df["near_grid_polygon_wkt"].apply(loads)
    )
    gdf_optimum_grid_polygon = gpd.GeoDataFrame(
        df, geometry=df["optimum_grid_polygon_wkt"].apply(loads)
    )
    gdf_line_optimum = gpd.GeoDataFrame(df, geometry=optimum_lines)

    # Save to files
    gdf_station_point.to_file(
        os.path.join(out_dir, "stations.geojson"), driver="GeoJSON"
    )
    gdf_near_grid_polygon.to_file(
        os.path.join(out_dir, "near_grid.geojson"), driver="GeoJSON"
    )
    gdf_optimum_grid_polygon.to_file(
        os.path.join(out_dir, "optimum_grid.geojson"), driver="GeoJSON"
    )
    gdf_line_optimum.to_file(
        os.path.join(out_dir, "stations2grid_optimum_line.geojson"), driver="GeoJSON"
    )
    gdf_station_point.to_csv(os.path.join(out_dir, "stations.csv"))


if __name__ == "__main__":
    main()
