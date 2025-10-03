from dask.diagnostics import ProgressBar
import pandas as pd
import xarray as xr
import numpy as np
import earthkit.data as ekd
from earthkit.hydro._readers import find_main_var

from hat import _LOGGER as logger


def process_grid_inputs(grid_config):
    src_name = list(grid_config["source"].keys())[0]
    logger.info(f"Processing grid inputs from source: {src_name}")
    logger.debug(f"Grid config: {grid_config['source'][src_name]}")
    ds = ekd.from_source(src_name, **grid_config["source"][src_name]).to_xarray(
        **grid_config.get("to_xarray_options", {})
    )
    var_name = find_main_var(ds, 3)
    da = ds[var_name]
    logger.info(f"Xarray created from source:\n{da}\n")
    gridx_colname = grid_config.get("coord_x", "lat")
    gridy_colname = grid_config.get("coord_y", "lon")
    da = da.sortby([gridx_colname, gridy_colname])
    shape = da[gridx_colname].shape[0], da[gridy_colname].shape[0]
    return da, var_name, gridx_colname, gridy_colname, shape


def construct_mask(indx, indy, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[indx, indy] = True

    flat_indices = np.ravel_multi_index((indx, indy), shape)
    _, inverse = np.unique(flat_indices, return_inverse=True)
    return mask, inverse


def create_mask_from_index(index_config, df, shape):
    logger.info(f"Creating mask {shape} from index: {index_config}")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    indx_colname = index_config.get("x", "opt_x_index")
    indy_colname = index_config.get("y", "opt_y_index")
    indx, indy = df[indx_colname].values, df[indy_colname].values
    mask, duplication_indexes = construct_mask(indx, indy, shape)
    return mask, duplication_indexes


def create_mask_from_coords(coords_config, df, gridx, gridy, shape):
    logger.info(f"Creating mask {shape} from coordinates: {coords_config}")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    x_colname = coords_config.get("x", "opt_x_coord")
    y_colname = coords_config.get("y", "opt_y_coord")
    xs = df[x_colname].values
    ys = df[y_colname].values

    diffx = np.abs(xs[:, np.newaxis] - gridx)
    indx = np.argmin(diffx, axis=1)
    diffy = np.abs(ys[:, np.newaxis] - gridy)
    indy = np.argmin(diffy, axis=1)

    mask, duplication_indexes = construct_mask(indx, indy, shape)
    return mask, duplication_indexes


def process_inputs(station_config, grid_config):
    logger.debug(f"Reading station file, {station_config}")
    df = pd.read_csv(station_config["file"])
    filters = station_config.get("filter")
    if filters is not None:
        logger.debug(f"Applying filters: {filters} to station DataFrame")
        df = df.query(filters)
    station_names = df[station_config["name"]].values

    index_config = station_config.get("index", None)
    coords_config = station_config.get("coords", None)

    if index_config is not None and coords_config is not None:
        raise ValueError("Use either index or coords, not both.")

    da, da_varname, gridx_colname, gridy_colname, shape = process_grid_inputs(grid_config)

    if index_config is not None:
        mask, duplication_indexes = create_mask_from_index(index_config, df, shape)
    elif coords_config is not None:
        mask, duplication_indexes = create_mask_from_coords(
            coords_config, df, da[gridx_colname].values, da[gridy_colname].values, shape
        )
    else:
        # default to index approach
        mask, duplication_indexes = create_mask_from_index(index_config, df, shape)

    return da, da_varname, gridx_colname, gridy_colname, mask, station_names, duplication_indexes


def mask_array_np(arr, mask):
    return arr[..., mask]


def apply_mask(da, mask, coordx, coordy):
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[(coordx, coordy), (coordx, coordy)],
        output_core_dims=[["station"]],
        output_dtypes=[da.dtype],
        exclude_dims={coordx, coordy},
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"station": int(mask.sum())},
            "allow_rechunk": True,
        },
    )
    with ProgressBar(dt=15):
        return task.compute()


def extractor(config):
    da, da_varname, gridx_colname, gridy_colname, mask, station_names, duplication_indexes = process_inputs(
        config["station"], config["grid"]
    )
    logger.info("Extracting timeseries at selected stations")
    masked_da = apply_mask(da, mask, gridx_colname, gridy_colname)
    ds = xr.Dataset({da_varname: masked_da})
    ds = ds.isel(station=duplication_indexes)
    ds["station"] = station_names
    if config.get("output", None) is not None:
        logger.info(f"Saving output to {config['output']['file']}")
        ds.to_netcdf(config["output"]["file"])
    return ds
