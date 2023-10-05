"""
Command line tool for extracting simulation timeseries from grids
 (i.e. rasters) requested from MARS

Usage

(hat)$ extract_timeseries --help

"""

import json
import os
from typing import List

import geopandas as gpd
import typer

from hat.cli import prettyprint, title
from hat.config import timeseries_config

# hat modules
from hat.data import save_dataset_to_netcdf, read_simulation_as_xarray
from hat.extract_simulation_timeseries import DEFAULT_CONFIG, extract_timeseries
from hat.observations import read_station_metadata_file



def print_overview(config: dict, station_metadata: gpd.GeoDataFrame, simulation):
    """Print overview of relevant information for user"""

    title("Configuration", color="cyan")
    for key in config:
        if str(config[key]):
            if "path" in str(key):
                print(f"{key} = ", os.path.abspath(config[key]))
            else:
                print(f"{key} = ", config[key])

    title("Observations", color="cyan")
    print(f"number of stations = {len(station_metadata)}")

    title("Simulation", color="cyan")
    print(simulation)


def print_default_config(DEFAULT_CONFIG):
    prettyprint(
        "Showing default configuration and returning...",
        color="cyan",
        first_line_empty=True,
        last_line_empty=True,
    )
    print(json.dumps(DEFAULT_CONFIG, indent=4))
    prettyprint(
        'To run timeseries extraction do not use "--show-default-config"',
        color="cyan",
        first_line_empty=True,
        last_line_empty=True,
    )


def command_line_tool(
    config_filepath: str = typer.Option(
        "",
        help="Path to configuration file",
    ),
    show_default_config: bool = typer.Option(
        False, help="Print default configuration and exit"
    ),
):
    """Command line tool to extract simulation timeseries of river discharge
    from gridded files (grib or netcdf)"""

    # show default config and exit?
    if show_default_config:
        print_default_config(DEFAULT_CONFIG)
        return

    title("STARTING TIME SERIES EXTRACTION")

    config = timeseries_config(config_filepath)

    # read station file
    stations = read_station_metadata_file(
        config["station_metadata_filepath"],
        config['station_coordinates'],
        config['station_epsg'],
        config["station_filters"]
    )

    # read simulated data
    simulation = read_simulation_as_xarray(config['simulation'])

    print_overview(config, stations, simulation)

    # Extract time series
    timeseries = extract_timeseries(stations, simulation)
    title("Timeseries extracted")
    print(timeseries)

    save_dataset_to_netcdf(timeseries, config["simulation_output_filepath"])

    title("TIMESERIES EXTRACTION COMPLETE", background="cyan", bold=True)


def main():
    typer.run(command_line_tool)


if __name__ == "__main__":
    main()
