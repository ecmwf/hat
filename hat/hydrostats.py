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
    Run statistical analysis on simulation and observation timeseries.
    """

    stations = sims_ds.coords["station"].values

    ds = xr.Dataset()

    for name in functions:
        func = getattr(hydrostats_functions, name)

        statistics = []
        for station in stations:
            sims = sims_ds.sel(station=station).to_numpy()
            obs = obs_ds.sel(station=station).to_numpy()

            stat = func(sims, obs)
            if stat is None:
                print(f"Warning! All NaNs for station {station}")
                stat = 0
            statistics.append(stat)
        statistics = np.array(statistics)

        ds[name] = xr.DataArray(statistics, coords={"station": stations})

    return ds
