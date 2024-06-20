import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from hat.geo import get_latlon_keys


def mask_array_np(arr, mask):
    # apply mask to data array (will broadcast mask if necessary)
    # for timeseries extraction mask.shape is (y,x) and arr.shape is (t, y, x)
    # where t is time dimension
    return arr[..., mask]


def extract_timeseries_using_mask(
    da: xr.DataArray, mask: np.ndarray, station_dim="station"
):
    """extract timeseries using a station mask with xarray.apply_ufunc()

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

    core_dims = get_latlon_keys(da)

    # dask computational graph (i.e. lazy)
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[core_dims, core_dims],
        output_core_dims=[[station_dim]],
        output_dtypes=[da.dtype],
        exclude_dims=set(core_dims),
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {station_dim: int(mask.sum())},
            "allow_rechunk": True,
        },
    )

    # extract timeseries (i.e. compute graph)
    with ProgressBar(dt=10):
        timeseries = task.compute()

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


def assign_stations(
    stations: gpd.GeoDataFrame,
    mask: np.ndarray,
    da_stations: xr.DataArray,
    coords: dict,
    station_dim: str,
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
    da_with_duplicate = None
    stations_list = []
    stations_id = []
    for _, station in tqdm(stations.iterrows(), total=stations.shape[0]):
        # station id
        station_id = station[station_dim]

        # timeseries index for a given station
        timeseries_index = station_timeseries_index(station, lon_in_mask, lat_in_mask)

        # add to the list
        stations_id += [station_id]
        stations_list += [timeseries_index]

    da_with_duplicate = da_stations.sel(station=stations_list)
    da_with_duplicate = da_with_duplicate.assign_coords(station=stations_id)

    return da_with_duplicate
