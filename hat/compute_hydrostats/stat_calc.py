import earthkit.data as ekd
from hat.data import find_main_var
import numpy as np
import xarray as xr
from hat.compute_hydrostats import stats


def load_da(ds_config):
    ds = ekd.from_source(*ds_config["datasource"]).to_xarray(xarray_open_mfdataset_kwargs={"chunks": {"time": "auto"}})
    var_name = find_main_var(ds, 2)
    da = ds[var_name]
    return da


def find_valid_subset(sim_da, obs_da, sim_coords, obs_coords, new_coords):
    sim_station_colname = sim_coords.get("s", "station")
    obs_station_colname = obs_coords.get("s", "station")
    matching_stations = np.intersect1d(sim_da[sim_station_colname].values, obs_da[obs_station_colname].values)
    sim_time_colname = sim_coords.get("t", "time")
    obs_time_colname = obs_coords.get("t", "time")
    matching_times = np.intersect1d(sim_da[sim_time_colname].values, obs_da[obs_time_colname].values)

    sim_da = sim_da.sel({sim_time_colname: matching_times, sim_station_colname: matching_stations})
    obs_da = obs_da.sel({obs_time_colname: matching_times, obs_station_colname: matching_stations})

    sim_da = sim_da.rename(
        {sim_time_colname: new_coords.get("t", "time"), sim_station_colname: new_coords.get("s", "station")}
    )
    obs_da = obs_da.rename(
        {obs_time_colname: new_coords.get("t", "time"), obs_station_colname: new_coords.get("s", "station")}
    )

    return sim_da, obs_da


def stat_calc(config):
    sim_config = config["sim"]
    sim_da = load_da(config["sim"])
    obs_config = config["obs"]
    obs_da = load_da(obs_config)
    new_coords = config["output"]["coords"]
    sim_da, obs_da = find_valid_subset(sim_da, obs_da, sim_config["coords"], obs_config["coords"], new_coords)
    stat_dict = {}
    for stat in config["stats"]:
        func = getattr(stats, stat)
        stat_dict[stat] = func(sim_da, obs_da, new_coords.get("t", "time"))
    ds = xr.Dataset(stat_dict)

    return ds
