from typing import List

import earthkit.data
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

""" NETCDF"""


def mask_array_np(arr, mask):
    # apply mask to data array (will broadcast mask if necessary)
    # for timeseries extraction mask.shape is (y,x) and arr.shape is (t, y, x)
    # where t is time dimension
    return arr[..., mask]


def extract_timeseries_using_mask(
    da: xr.DataArray, mask: np.ndarray, core_dims=["y", "x"]
):
    """extract timeseries using a station mask with xarray.apply_ufunc()

    - input_core_dims
      list of core dimensions on each input argument
      (needs to be same length as number of input arguments)
    - output_core_dims
      list of core dimensions on each output
      (needs to be same length as number of output variables)
    - output_dtypes
      the output data type (e.g. nc).
    - exclude_dims
      dimensions along which your function should not be applied
    - dask
      One of ['forbidden', 'allowed', 'parallelized'].
      If 'forbidden', function is computed eagerly (not lazily).
      If 'allowed', function to be computed on dask arrays.
      If 'parallelized', automatically parallelizes your function

    BUG: this approach might return less timeseries than there are stations

    """

    # dask computational graph (i.e. lazy)
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[core_dims, core_dims],
        output_core_dims=[["station"]],
        output_dtypes=[da.dtype],
        exclude_dims=set(core_dims),
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"station": int(mask.sum())}},
    )

    # extract timeseries (i.e. compute graph)
    timeseries = task.compute()

    return timeseries


def extract_timeseries_from_filepaths(fpaths: List[str], mask: np.ndarray):
    """extract timeseries from a collection of files and given a boolean mask
    to represent point locations"""

    # earthkit data file source
    fs = earthkit.data.from_source("file", fpaths)

    # xarray dataset
    ds = fs.to_xarray()

    # timeseries extraction using masking algorithm
    timeseries = extract_timeseries_using_mask(
        ds.dis, mask, core_dims=["latitude", "longitude"]
    )

    return timeseries


def station_timeseries_index(
    station: gpd.GeoDataFrame, lon_in_mask: np.ndarray, lat_in_mask: np.ndarray
):
    """Get timeseries index for a given station, required because timeseries
    can be shared between stations that are close together"""

    dx = station.geometry.x - lon_in_mask
    dy = station.geometry.y - lat_in_mask
    idx = (dx**2 + dy**2).argmin()

    return idx


def all_timeseries(
    stations: gpd.GeoDataFrame, mask, masked_timeseries: xr.DataArray, coords: dict
) -> pd.DataFrame:
    """
    Fixed bug in timeseries extraction by mask where:
    stations can be dropped if too close together (or duplicates)

    This is required so that the number of output timeseries equals
    the number of input stations provided by the user
    """
    """
    Here we get latlon of timeseries points using the same
    masking approach as in timeseries extraction.
    The idea is to ensure we preserve ordering so that we
    can re-map station_ids to latlon coordinates
    """

    # broadcast the coord arrays to same shape as mask
    lon2d, lat2d = np.meshgrid(coords["x"], coords["y"])

    # get lats and lons of True values in mask
    lat_in_mask = lat2d[mask]
    lon_in_mask = lon2d[mask]
    """
    We can now get the complete collection of timeseries
    (even where there are duplicates of proximal stations)
    """

    # table of all station timeseries
    complete_station_timeseries = {}

    for _, station in stations.iterrows():
        # station id
        station_id = station["station_id"]

        # timeseries index for a given station
        timeseries_index = station_timeseries_index(station, lon_in_mask, lat_in_mask)

        # timeseries values for a given station
        station_timeseries = masked_timeseries.sel(station=timeseries_index)

        # update complete station timeseries
        complete_station_timeseries[station_id] = station_timeseries.values.flatten()
    """
    Finally, create a pandas dataframe with a datetime index
    """

    # stationIDs and discharge values
    df = pd.DataFrame(complete_station_timeseries)

    # datetime index
    df["datetime"] = pd.to_datetime(station_timeseries.time.values)
    df = df.set_index("datetime")

    return df
