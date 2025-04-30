"""
Command line tool for extracting simulation timeseries from grids
 (i.e. rasters) requested from MARS

Usage

(hat)$ extract_timeseries --help

"""

import json
import os

import geopandas as gpd
import typer

from hat.config import read_config, prettyprint

# hat modules
from hat.data import read_simulation_as_xarray, save_dataset_to_netcdf
from hat.extract_simulation_timeseries import DEFAULT_CONFIG, extract_timeseries
from hat.observations import read_station_metadata_file


def title(text, **kwargs):
    """A pretty title"""
    print("")
    prettyprint(text, **kwargs)
    print("-" * len(text))


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
    config: str = typer.Option(
        "",
        help="Path to configuration file",
    ),
    show_default_config: bool = typer.Option(False, help="Print default configuration and exit"),
):
    """Command line tool to extract simulation timeseries of river discharge
    from gridded files (grib or netcdf)"""

    # show default config and exit?
    if show_default_config:
        print_default_config(DEFAULT_CONFIG)
        return

    title("STARTING TIME SERIES EXTRACTION")

    cfg = read_config(config)

    # read station file
    stations = read_station_metadata_file(
        cfg["station_metadata_filepath"],
        cfg["station_coordinates"],
        cfg["station_epsg"],
        cfg["station_filters"],
    )

    # read simulated data
    simulation = read_simulation_as_xarray(cfg["simulation"])

    print_overview(cfg, stations, simulation)

    # Extract time series
    timeseries = extract_timeseries(stations, simulation, cfg)
    title("Timeseries extracted")
    print(timeseries)

    save_dataset_to_netcdf(timeseries, cfg["simulation_output_filepath"])

    title("TIMESERIES EXTRACTION COMPLETE", background="cyan", bold=True)


def main():
    typer.run(command_line_tool)


if __name__ == "__main__":
    main()
