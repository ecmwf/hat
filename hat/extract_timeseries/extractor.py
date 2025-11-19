from dask.diagnostics.progress import ProgressBar
import pandas as pd
import xarray as xr
import numpy as np
from typing import Any
from hat.core import load_da

from hat import _LOGGER as logger


def process_grid_inputs(grid_config):
    da, var_name = load_da(grid_config, 3)
    logger.info(f"Xarray created from source:\n{da}\n")
    coord_config = grid_config.get("coords", {})
    x_dim = coord_config.get("x", "lat")
    y_dim = coord_config.get("y", "lon")
    da = da.sortby([x_dim, y_dim])
    shape = da[x_dim].shape[0], da[y_dim].shape[0]
    return da, var_name, x_dim, y_dim, shape


def construct_mask(x_indices, y_indices, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[x_indices, y_indices] = True

    flat_indices = np.ravel_multi_index((x_indices, y_indices), shape)
    _, duplication_indexes = np.unique(flat_indices, return_inverse=True)
    return mask, duplication_indexes


def create_mask_from_index(df, shape):
    logger.info(f"Creating mask {shape} from index")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    x_indices = df["x_index"].values
    y_indices = df["y_index"].values
    if np.any(x_indices < 0) or np.any(x_indices >= shape[0]) or np.any(y_indices < 0) or np.any(y_indices >= shape[1]):
        raise ValueError(
            f"Station indices out of grid bounds. Grid shape={shape}, "
            f"x_index range=({int(x_indices.min())},{int(x_indices.max())}), "
            f"y_index range=({int(y_indices.min())},{int(y_indices.max())})"
        )
    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def create_mask_from_coords(df, gridx, gridy, shape):
    logger.info(f"Creating mask {shape} from coordinates")
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    station_x = df["x_coord"].values
    station_y = df["y_coord"].values

    x_distances = np.abs(station_x[:, np.newaxis] - gridx)
    x_indices = np.argmin(x_distances, axis=1)
    y_distances = np.abs(station_y[:, np.newaxis] - gridy)
    y_indices = np.argmin(y_distances, axis=1)

    mask, duplication_indexes = construct_mask(x_indices, y_indices, shape)
    return mask, duplication_indexes


def parse_stations(station_config: dict[str, Any]) -> pd.DataFrame:
    """Read, filter, and normalize station DataFrame to canonical column names."""
    logger.debug(f"Reading station file, {station_config}")
    if "name" not in station_config:
        raise ValueError("Station config must include a 'name' key mapping to the station column")
    df = pd.read_csv(station_config["file"])
    filters = station_config.get("filter")
    if filters is not None:
        logger.debug(f"Applying filters: {filters} to station DataFrame")
        df = df.query(filters)

    if len(df) == 0:
        raise ValueError("No stations found. Check station file or filter.")

    has_index = "index" in station_config
    has_coords = "coords" in station_config
    has_index_1d = "index_1d" in station_config

    if not has_index_1d:
        if has_index and has_coords:
            raise ValueError("Station config must use either 'index' or 'coords', not both.")
        if not has_index and not has_coords:
            raise ValueError("Station config must provide either 'index' or 'coords' for station mapping.")

    renames = {}
    renames[station_config["name"]] = "station_name"

    if has_index:
        index_config = station_config["index"]
        x_col = index_config.get("x", "opt_x_index")
        y_col = index_config.get("y", "opt_y_index")
        renames[x_col] = "x_index"
        renames[y_col] = "y_index"

    if has_coords:
        coords_config = station_config["coords"]
        x_col = coords_config.get("x", "opt_x_coord")
        y_col = coords_config.get("y", "opt_y_coord")
        renames[x_col] = "x_coord"
        renames[y_col] = "y_coord"

    if has_index_1d:
        renames[station_config["index_1d"]] = "index_1d"

    df_renamed = df.rename(columns=renames)

    if has_index and ("x_index" not in df_renamed.columns or "y_index" not in df_renamed.columns):
        raise ValueError(
            "Station file missing required index columns. Expected columns to map to 'x_index' and 'y_index'."
        )
    if has_coords and ("x_coord" not in df_renamed.columns or "y_coord" not in df_renamed.columns):
        raise ValueError(
            "Station file missing required coordinate columns. Expected columns to map to 'x_coord' and 'y_coord'."
        )
    if has_index_1d and "index_1d" not in df_renamed.columns:
        raise ValueError("Station file missing required 'index_1d' column.")

    return df_renamed


def _process_gribjump(grid_config: dict[str, Any], df: pd.DataFrame) -> xr.Dataset:
    if "index_1d" not in df.columns:
        raise ValueError("Gribjump source requires 'index_1d' in station config.")

    station_names = df["station_name"].values
    unique_indices, duplication_indexes = np.unique(df["index_1d"].values, return_inverse=True)  # type: ignore[call-overload]

    # Converting indices to ranges is currently faster than using indices
    # directly. This is a problem in the earthkit-data gribjump source and will
    # be fixed there.
    ranges = [(i, i + 1) for i in unique_indices]

    gribjump_config = {
        "source": {
            "gribjump": {
                **grid_config["source"]["gribjump"],
                "ranges": ranges,
                # fetch_coords_from_fdb is currently very slow. Needs fix in
                # earthkit-data gribjump source.
                # "fetch_coords_from_fdb": True,
            }
        },
        "to_xarray_options": grid_config.get("to_xarray_options", {}),
    }

    masked_da, var_name = load_da(gribjump_config, 2)

    ds = xr.Dataset({var_name: masked_da})
    ds = ds.isel(index=duplication_indexes)
    ds = ds.rename({"index": "station"})
    ds["station"] = station_names
    return ds


def _process_regular(grid_config: dict[str, Any], df: pd.DataFrame) -> xr.Dataset:
    station_names = df["station_name"].values
    da, var_name, x_dim, y_dim, shape = process_grid_inputs(grid_config)

    use_index = "x_index" in df.columns and "y_index" in df.columns

    if use_index:
        mask, duplication_indexes = create_mask_from_index(df, shape)
    else:
        mask, duplication_indexes = create_mask_from_coords(df, da[x_dim].values, da[y_dim].values, shape)

    logger.info("Extracting timeseries at selected stations")
    masked_da = apply_mask(da, mask, x_dim, y_dim)

    ds = xr.Dataset({var_name: masked_da})
    ds = ds.isel(index=duplication_indexes)
    ds = ds.rename({"index": "station"})
    ds["station"] = station_names
    return ds


def process_inputs(station_config: dict[str, Any], grid_config: dict[str, Any]) -> xr.Dataset:
    df = parse_stations(station_config)
    if "gribjump" in grid_config.get("source", {}):
        return _process_gribjump(grid_config, df)
    return _process_regular(grid_config, df)


def mask_array_np(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[..., mask]


def apply_mask(da: xr.DataArray, mask: np.ndarray, coordx: str, coordy: str) -> xr.DataArray:
    task = xr.apply_ufunc(
        mask_array_np,
        da,
        mask,
        input_core_dims=[(coordx, coordy), (coordx, coordy)],
        output_core_dims=[["index"]],
        output_dtypes=[da.dtype],
        exclude_dims={coordx, coordy},
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"index": int(mask.sum())},
            "allow_rechunk": True,
        },
    )
    with ProgressBar(dt=15):
        return task.compute()


def extractor(config: dict[str, Any]) -> xr.Dataset:
    ds = process_inputs(config["station"], config["grid"])
    if config.get("output", None) is not None:
        logger.info(f"Saving output to {config['output']['file']}")
        ds.to_netcdf(config["output"]["file"])
    return ds
