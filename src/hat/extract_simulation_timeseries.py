"""
Python module for extracting simulation timeseries from grids (i.e. rasters)
"""
from typing import List

import geopandas as gpd
import pandas as pd
import xarray as xr

from hat.config import load_package_config, valid_custom_config

# hat modules
from hat.data import find_files
from hat.geo import geopoints_to_array, latlon_coords
from hat.timeseries import all_timeseries, extract_timeseries_from_filepaths

# Read configutation files
# NOTE config files are installed as part of hat package using setup.py
# (just updating the .json files will not necessarily work
# (i.e. would require pip install -e .)
DEFAULT_CONFIG = load_package_config("timeseries.json")


def geopandas_to_xarray(station_metadata: gpd.GeoDataFrame,
                        timeseries: pd.DataFrame):
    """Convert results from geopandas geodataframe to xarray dataset"""

    # NOTE we do not use this inbuilt method..
    # ds = xr.Dataset.from_dataframe(timeseries)
    # because it can be extremely slow (5 mins) when extracting all stations
    # i.e. because it creates a separate data array for each station
    # here instead we create a single data array for all stations

    # 2D numpy array of timeseries
    arr = timeseries.to_numpy()

    # labels for the numpy array
    coords = {
        "time": list(timeseries.index),
        "station": list(timeseries.columns),
    }

    # xarray data array is essentially "just" a numpy array with labels
    da = xr.DataArray(arr,
                      dims=["time", "station"],
                      coords=coords,
                      name="simulation_timeseries")

    # parse point geometries data into lists
    lons = [point.x for point in station_metadata.geometry]
    lats = [point.y for point in station_metadata.geometry]

    # add to dataarray
    da["longitude"] = ("station", lons)
    da["latitude"] = ("station", lats)

    # create xarray data set (in this case with just one data array)
    ds = da.to_dataset()

    return ds


def extract_timeseries(
    station_metadata: gpd.GeoDataFrame,
    simulation_datadir: str = "",
    simulation_fpaths: List = [],
    config: dict = DEFAULT_CONFIG,
):
    config = valid_custom_config(config)

    if not simulation_datadir and not simulation_fpaths:
        raise TypeError(
            """extract_timeseries() missing 1 required variable:
            'simulation_datadir' or 'simulation_fpaths' """
        )

    if not simulation_fpaths:
        simulation_fpaths = find_files(
            simulation_datadir,
            file_extension=config["simulation_input_file_extension"],
            recursive=config["recursive_search"],
        )

    # infer coordinates for all grids from first the grid
    coords = latlon_coords(simulation_fpaths[0])

    # numpy array of station locations
    station_mask = geopoints_to_array(station_metadata, coords)

    # extract timeseries from files using mask
    timeseries_from_mask = extract_timeseries_from_filepaths(
        simulation_fpaths, station_mask)

    # all timeseries (i.e. handle proximal and duplicate stations)
    timeseries = all_timeseries(station_metadata, station_mask,
                                timeseries_from_mask, coords)

    # convert to xarray dataset
    timeseries = geopandas_to_xarray(station_metadata, timeseries)

    return timeseries
