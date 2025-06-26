import pandas as pd
import xarray as xr
import earthkit.data as ekd
import numpy as np
import plotly.express as px
from plotly.colors import get_colorscale
from earthkit.hydro.readers import find_main_var
from .station_mapping import StationMapping


def get_grid_inputs(grid_config):
    ds = ekd.from_source(*grid_config["source"]).to_xarray()
    nc_variable = find_main_var(ds, 2)
    metric_grid = ds[nc_variable].values

    coord_dict = grid_config.get("coords", None)
    coord_x = "lat" if coord_dict is None else coord_dict["x"]
    coord_y = "lon" if coord_dict is None else coord_dict["y"]
    ds = ds.sortby([coord_x, coord_y])

    grid_area_coords1, grid_area_coords2 = xr.broadcast(ds[coord_x], ds[coord_y])
    grid_area_coords1 = grid_area_coords1.values.copy()
    grid_area_coords2 = grid_area_coords2.values.copy()

    return metric_grid, grid_area_coords1, grid_area_coords2


def get_station_inputs(station_config):
    df = pd.read_csv(station_config["file"])
    coord_x = station_config["coords"]["x"]
    coord_y = station_config["coords"]["y"]
    station_coords1 = df[coord_x].values
    station_coords2 = df[coord_y].values
    station_metric = df[station_config["metric"]].values
    return station_metric, station_coords1, station_coords2, df


def apply_blacklist(blacklist_config, metric_grid, grid_area_coords1, grid_area_coords2):
    if blacklist_config is not None:
        ds = ekd.from_source(*blacklist_config["source"]).to_xarray()
        nc_variable = find_main_var(ds, 2)
        mask = ds[nc_variable].values
        metric_grid[mask] = np.nan

    return metric_grid, grid_area_coords1, grid_area_coords2


def outputs_to_df(df, indx, indy, cindx, cindy, errors, grid_area_coords1, grid_area_coords2, filename):
    df["opt_x_index"] = indx
    df["opt_y_index"] = indy
    df["near_x_index"] = cindx
    df["near_y_index"] = cindy
    df["opt_error"] = errors
    df["opt_x_coord"] = grid_area_coords1[indx, 0]
    df["opt_y_coord"] = grid_area_coords2[0, indy]
    if filename is not None:
        df.to_csv(filename, index=False)
    return df


def light_zero_color(colorscale_name="Viridis", zero_color="rgba(0,0,0,0)"):
    base = get_colorscale(colorscale_name)
    n = len(base)
    scaled = [[i / (n - 1), color] for i, (_, color) in enumerate(base)]
    scaled[0] = [0.0, zero_color]
    return scaled


def generate_summary_plots(df, plot_config):
    if plot_config is None:
        return

    distance_plot_config = plot_config.get("error", None)
    if distance_plot_config is not None:
        df["grid_offset_x"] = df["opt_x_index"] - df["near_x_index"]
        df["grid_offset_y"] = df["opt_y_index"] - df["near_y_index"]
        custom_scale = light_zero_color("Viridis")
        fig = px.density_heatmap(
            df,
            x="grid_offset_x",
            y="grid_offset_y",
            marginal_x="histogram",
            marginal_y="histogram",
            color_continuous_scale=custom_scale,
        )
        fig.write_html(distance_plot_config["file"])
        fig.show()

    error_plot_config = plot_config.get("error", None)
    if error_plot_config is not None:
        fig = px.histogram(df, x="opt_error")
        fig.write_html(error_plot_config["file"])
        fig.show()


def mapper(config):
    metric_grid, grid_area_coords1, grid_area_coords2 = get_grid_inputs(config["grid"])
    station_metric, station_coords1, station_coords2, df = get_station_inputs(config["station"])
    metric_grid, grid_area_coords1, grid_area_coords2 = apply_blacklist(
        config.get("blacklist", None), metric_grid, grid_area_coords1, grid_area_coords2
    )
    mapping_outputs = StationMapping(config["parameters"]).conduct_mapping(
        station_coords1, station_coords2, grid_area_coords1, grid_area_coords2, station_metric, metric_grid
    )
    df = outputs_to_df(df, *mapping_outputs, grid_area_coords1, grid_area_coords2, filename=config["output"]["file"] if config.get("output", None) is not None else None)
    generate_summary_plots(df, config.get("plot", None))
    return df
