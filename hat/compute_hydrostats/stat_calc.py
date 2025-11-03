import earthkit.data as ekd
from earthkit.hydro._readers import find_main_var
import numpy as np
import xarray as xr
import scores


def load_da(ds_config):
    ds = ekd.from_source(*ds_config["source"]).to_xarray()
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
    time_dim = new_coords.get("t", "time")
    stat_dict = {}
    for stat in config["stats"]:
        parts = stat.split(".")
        func = scores
        for part in parts:
            func = getattr(func, part)
        stat_dict[stat] = func(sim_da, obs_da, reduce_dims=time_dim)
    ds = xr.Dataset(stat_dict)
    if config["output"].get("file", None) is not None:
        ds.to_netcdf(config["output"]["file"])
    return ds
