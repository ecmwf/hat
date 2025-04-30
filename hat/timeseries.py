import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from hat.geo import get_latlon_keys


def mask_array_np(arr, mask):
    return arr[..., mask]


def extract_timeseries_using_mask(
    da: xr.DataArray, mask: np.ndarray, station_dim="station"
):
    # TODO: check if there is possibly a bug where approach
    # might return less timeseries than there are stations

    core_dims = get_latlon_keys(da)

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

    with ProgressBar(dt=10):
        timeseries = task.compute()

    return timeseries


def station_timeseries_index(
    station: gpd.GeoDataFrame, lon_in_mask: np.ndarray, lat_in_mask: np.ndarray
):
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
    
    # Here we get latlon of timeseries points using the same
    # masking approach as in timeseries extraction.
    # The idea is to ensure we preserve ordering so that we
    # can re-map station_ids to latlon coordinates
    lon2d, lat2d = np.meshgrid(coords["x"], coords["y"])

    lat_in_mask = lat2d[mask]
    lon_in_mask = lon2d[mask]

    # We can now get the complete collection of timeseries
    # (even where there are duplicates of proximal stations)
    stations_list = []
    stations_id = []

    for _, station in tqdm(stations.iterrows(), total=stations.shape[0]):
        station_id = station[station_dim]
        timeseries_index = station_timeseries_index(station, lon_in_mask, lat_in_mask)
        stations_id.append(station_id)
        stations_list.append(timeseries_index)

    da_with_duplicate = da_stations.sel(station=stations_list)
    da_with_duplicate = da_with_duplicate.assign_coords(station=stations_id)

    return da_with_duplicate
