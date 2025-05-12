import pandas as pd
import xarray as xr
from hat.data import find_main_var

from hat.station_mapping.station_mapping import StationMapping


def get_grid_inputs(grid_config):
    ds = xr.open_dataset(grid_config["file"])
    nc_variable = find_main_var(ds, min_dim=2)
    metric_grid = ds[nc_variable].values

    grid_area_coords1, grid_area_coords2 = xr.broadcast(
        ds[grid_config.get("coord_x", "lat")], ds[grid_config.get("coord_y", "lon")]
    )
    grid_area_coords1 = grid_area_coords1.values
    grid_area_coords2 = grid_area_coords2.values

    return metric_grid, grid_area_coords1, grid_area_coords2


def get_station_inputs(station_config):
    df = pd.read_csv(station_config["file"])
    station_coords1 = df[station_config["coord_x"]].values
    station_coords2 = df[station_config["coord_y"]].values
    station_metric = df[station_config["metric"]].values
    return station_metric, station_coords1, station_coords2, df


def outputs_to_df(df, indx, indy, cindx, cindy, errors, grid_area_coords1, grid_area_coords2, filename):
    df["opt_x_index"] = indx
    df["opt_y_index"] = indy
    df["near_x_index"] = cindx
    df["near_y_index"] = cindy
    df["opt_error"] = errors
    df["opt_x_coord"] = grid_area_coords1[indx, 0]
    df["opt_y_coord"] = grid_area_coords2[0, indy]
    df.to_csv(filename, index=False)


def mapper(config):
    metric_grid, grid_area_coords1, grid_area_coords2 = get_grid_inputs(config["grid"])
    station_metric, station_coords1, station_coords2, df = get_station_inputs(config["station"])
    mapping_outputs = StationMapping(config["parameters"]).conduct_mapping(
        station_coords1, station_coords2, grid_area_coords1, grid_area_coords2, station_metric, metric_grid
    )
    outputs_to_df(df, *mapping_outputs, grid_area_coords1, grid_area_coords2, filename=config["output"]["file"])
