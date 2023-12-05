#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from netCDF4 import Dataset


def get_grid_index(lat, lon, lat_array, lon_array):
    # Find the index of the closest latitude and longitude in the arrays
    lat_idx = np.abs(lat_array - lat).argmin()
    lon_idx = np.abs(lon_array - lon).argmin()
    return lat_idx, lon_idx

def copy_metadata_from_source(src_dataset, dst_dataset):
    # Copy global attributes
    dst_dataset.setncatts(src_dataset.__dict__)

    # Copy dimensions
    for name, dimension in src_dataset.dimensions.items():
        dst_dataset.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

    # Copy coordinate variables (like 'lat', 'lon') and their attributes
    for name, variable in src_dataset.variables.items():
        if name in ['lat', 'lon']:
            x = dst_dataset.createVariable(name, variable.datatype, variable.dimensions)
            dst_dataset[name][:] = src_dataset[name][:]
            dst_dataset[name].setncatts(src_dataset[name].__dict__)


def main(netcdf_file, variable, csv_file, station_name_col, lat_col, lon_col):
    # Read station CSV
    stations = pd.read_csv(csv_file)
    
    # Load netCDF data
    dataset = Dataset(netcdf_file, 'r')
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    data_variable = dataset.variables[variable][:]

    # Initialize an empty 2D array for the new data
    new_data_array = np.full(data_variable.shape, np.nan, dtype=data_variable.dtype)  # Fill with NaNs

    # Process each station and populate the new data array
    for index, station in stations.iterrows():
        lat, lon = station[lat_col], station[lon_col]
        lat_idx = (np.abs(latitudes - lat)).argmin()
        lon_idx = (np.abs(longitudes - lon)).argmin()

        # Extract data for the corresponding grid cell and store it in the new data array
        new_data_array[lat_idx, lon_idx] = data_variable[lat_idx, lon_idx]

    # Create a new netCDF file to save the modified data
    with Dataset("modified_data.nc", "w", format="NETCDF4") as new_dataset:
        # Copy global attributes, dimensions, and coordinate variables
        copy_metadata_from_source(dataset, new_dataset)

        # Create the modified variable
        new_var = new_dataset.createVariable(variable, data_variable.dtype, ('lat', 'lon'))
        new_var[:] = new_data_array

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process netCDF and station data.')
    parser.add_argument('netcdf_file', type=str, help='Path to the netCDF file')
    parser.add_argument('variable', type=str, help='Variable name in the netCDF file')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing station data')
    parser.add_argument('station_name_col', type=str, help='Column name for station names in the CSV file')
    parser.add_argument('lat_col', type=str, help='Column name for latitude in the CSV file')
    parser.add_argument('lon_col', type=str, help='Column name for longitude in the CSV file')
    args = parser.parse_args()

    main(args.netcdf_file, args.variable, args.csv_file, args.station_name_col, args.lat_col, args.lon_col)
