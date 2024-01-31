"""high level python api for hydrological statistics"""

from typing import List

import numpy as np
import xarray as xr

from hat import hydrostats_functions


def run_analysis(
    functions: List,
    sims_ds: xr.DataArray,
    obs_ds: xr.DataArray,
) -> xr.Dataset:
    """
    Run statistical analysis on simulation and observation timeseries
    """

    # list of stations
    stations = sims_ds.coords["station"].values

    ds = xr.Dataset()

    # For each statistical function
    for name in functions:
        # get function itself from name
        func = getattr(hydrostats_functions, name)

        # do timeseries analysis for each station
        # (using a "numpy in, numpy out" function)
        statistics = []
        for station in stations:
            sims = sims_ds.sel(station=station).to_numpy()
            obs = obs_ds.sel(station=station).to_numpy()

            stat = func(sims, obs)
            if stat is None:
                print(f"Warning! All NaNs for station {station}")
                stat = 0
            statistics += [stat]
        statistics = np.array(statistics)

        # Add the Series to the DataFrame
        ds[name] = xr.DataArray(statistics, coords={"station": stations})

    return ds
