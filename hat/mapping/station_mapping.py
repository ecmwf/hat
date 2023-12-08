#!/usr/bin/env python3

import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from netCDF4 import Dataset
import numpy as np
from numpy.ma import is_masked
from geopy.distance import geodesic

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance in kilometers between two points."""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def main(netcdf_file, nc_variable, csv_file, station_name_col, lat_col, lon_col, csv_variable):
    # Read station CSV and filter out invalid data
    stations = pd.read_csv(csv_file)
    stations = stations[stations[csv_variable] > 0]  # Filter out stations with csv_variable <= 0 or NaN

    # Load netCDF data
    dataset = Dataset(netcdf_file, 'r')
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    nc_data = dataset.variables[nc_variable][:]

    data_list = []

    for index, station in stations.iterrows():
        lat, lon = station[lat_col], station[lon_col]
        lat_idx = (np.abs(latitudes - lat)).argmin()
        lon_idx = (np.abs(longitudes - lon)).argmin()

        grid_data = nc_data[lat_idx, lon_idx]
        if is_masked(grid_data):
            grid_data = np.nan
        else:
            grid_data = float(grid_data)

        csv_data = station[csv_variable]
        area_difference = (csv_data - grid_data)*1e-6

        station_data = {
            'station_name': station[station_name_col],
            'station_lat': lat,
            'station_lon': lon,
            'grid_lat': latitudes[lat_idx],
            'grid_lon': longitudes[lon_idx],
            'nc_variable': grid_data,
            'csv_variable': csv_data,
            'area_difference': area_difference,
            'distance_km': calculate_distance(lat, lon, latitudes[lat_idx], longitudes[lon_idx])
        }

        data_list.append(station_data)

    df = pd.DataFrame(data_list)

    # Creating separate GeoDataFrames for grid points, stations, and lines with specific attributes
    gdf_grid_point = gpd.GeoDataFrame(df[['nc_variable', 'grid_lat', 'grid_lon']], 
                                      geometry=[Point(xy) for xy in zip(df.grid_lon, df.grid_lat)])
    gdf_station_point = gpd.GeoDataFrame(df[['station_name', 'csv_variable', 'station_lat', 'station_lon']], 
                                         geometry=[Point(xy) for xy in zip(df.station_lon, df.station_lat)])
    gdf_line = gpd.GeoDataFrame(df[['area_difference', 'distance_km', 'station_lon', 'station_lat', 'grid_lon', 'grid_lat']], 
                                geometry=[LineString([(row['station_lon'], row['station_lat']), 
                                                      (row['grid_lon'], row['grid_lat'])]) for index, row in df.iterrows()])

    gdf_grid_point.to_file("grid_point.geojson", driver="GeoJSON")
    gdf_station_point.to_file("station.geojson", driver="GeoJSON")
    gdf_line.to_file("station2grid_line.geojson", driver="GeoJSON")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process netCDF and station data.')
    parser.add_argument('netcdf_file', type=str, help='Path to the netCDF file')
    parser.add_argument('nc_variable', type=str, help='Variable name in the netCDF file')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing station data')
    parser.add_argument('station_name_col', type=str, help='Column name for station names in the CSV file')
    parser.add_argument('lat_col', type=str, help='Column name for latitude in the CSV file')
    parser.add_argument('lon_col', type=str, help='Column name for longitude in the CSV file')
    parser.add_argument('csv_variable', type=str, help='Column name for the CSV variable')
    args = parser.parse_args()

    main(args.netcdf_file, args.nc_variable, args.csv_file, args.station_name_col, args.lat_col, args.lon_col, args.csv_variable)
