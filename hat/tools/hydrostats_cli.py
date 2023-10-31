from typing import List

import typer
import xarray as xr

from hat import hydrostats_functions
from hat.data import find_main_var
from hat.exceptions import UserError
from hat.filters import filter_timeseries
from hat.hydrostats import run_analysis


def check_inputs(functions, sims, obs):
    """basic sanitation check of user inputs"""

    if functions == "":
        raise UserError(
            "Name(s) of statistical function(s) required, e.g. try --functions"
        )

    if sims == "":
        raise UserError(
            "Filepath to simulation timeseries is required, e.g. try --sims "
        )

    if obs == "":
        raise UserError(
            "Filepath to observation timeseries is required, e.g. try --obs"
        )

    if not sims.endswith(".nc"):
        raise UserError(f"Simulation filepath must end with .nc was given: {sims}")

    if not obs.endswith(".nc"):
        raise UserError(f"Observation filepath must end with .nc was given: {obs}")

    return True


def parse_functions(functions_string: str) -> List:
    """parse functions string from user into a list of functions"""

    # remove white space and separate on comma
    functions = functions_string.replace(" ", "").split(",")

    # check function names are allowed
    for function in functions:
        if function.lower() not in hydrostats_functions.__all__:
            raise UserError(f"Unknown statistical function: {function}")

    return functions


def hydrostats_cli(
    functions: str = "",
    sims: str = "",
    obs: str = "",
    obs_threshold: int = 80,
    outpath="./statistics.nc",
):
    """
    Hydrological Statistics Command Line Tool

    Usage
    $ hydrostats --name $NAME --sims $SIMS --obs $OBS

    --functions = names of statistical function(s)
    --sims = filepath to simulation file
    --obs = filepath to observation file
    --obs_threshold = percentage of observations required
    """

    # basic santitation check of user inputs
    valid = check_inputs(functions, sims, obs)
    if not valid:
        return

    # parse function names
    functions = parse_functions(functions)
    if not functions:
        return

    # simulations
    sims_ds = xr.open_dataset(sims)
    var = find_main_var(sims_ds, min_dim=2)
    sims_da = sims_ds[var]

    # observations
    obs_ds = xr.open_dataset(obs)
    var = find_main_var(obs_ds, min_dim=2)
    obs_da = obs_ds[var]

    # clean timeseries
    sims_da, obs_da = filter_timeseries(sims_da, obs_da, threshold=obs_threshold)

    # calculate statistics
    statistics_ds = run_analysis(functions, sims_da, obs_da)

    # save to netcdf
    statistics_ds.to_netcdf(outpath)


def main():
    typer.run(hydrostats_cli)


if __name__ == "__main__":
    main()
