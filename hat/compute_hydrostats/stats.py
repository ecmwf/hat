import numpy as np
import xarray as xr


def bias(sim_da, obs_da, time_name):
    return (sim_da - obs_da).mean(dim=time_name, skipna=True)


def mae(sim_da, obs_da, time_name):
    return np.abs(sim_da - obs_da).mean(dim=time_name, skipna=True)


def mape(sim_da, obs_da, time_name):
    return mae(sim_da, obs_da, time_name) / np.abs(obs_da).sum(dim=time_name, skipna=True)


def mse(sim_da, obs_da, time_name):
    return ((sim_da - obs_da) ** 2).mean(dim=time_name, skipna=True)


def rmse(sim_da, obs_da, time_name):
    return np.sqrt(mse(sim_da, obs_da, time_name))


def br(sim_da, obs_da, time_name):
    # as defined in:
    # Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. Journal of hydrology, 424, 264-277.
    return sim_da.mean(dim=time_name, skipna=True) / obs_da.mean(dim=time_name, skipna=True)


def vr(sim_da, obs_da, time_name):
    # as defined in:
    # Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. Journal of hydrology, 424, 264-277.
    return (sim_da.std(dim=time_name, skipna=True) * obs_da.mean(dim=time_name, skipna=True)) / (
        obs_da.std(dim=time_name, skipna=True) * sim_da.mean(dim=time_name, skipna=True)
    )


def pc_bias(sim_da, obs_da, time_name):
    return (sim_da - obs_da).mean(dim=time_name, skipna=True) / obs_da.mean(dim=time_name, skipna=True)


def correlation(sim_da, obs_da, time_name):
    def _corr(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) < 2:
            return np.nan
        return np.corrcoef(a[mask], b[mask])[0, 1]

    return xr.apply_ufunc(
        _corr,
        sim_da,
        obs_da,
        input_core_dims=[[time_name], [time_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def kge(sim_da, obs_da, time_name):
    # as defined in:
    # Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. Journal of hydrology, 424, 264-277.
    B = br(sim_da, obs_da, time_name)
    y = vr(sim_da, obs_da, time_name)
    r = correlation(sim_da, obs_da, time_name)

    return 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)


def index_agreement(sim_da, obs_da, time_name):
    numerator = ((obs_da - sim_da) ** 2).sum(dim=time_name, skipna=True)
    mean_obs = obs_da.mean(dim=time_name, skipna=True)
    denominator = ((np.abs(sim_da - mean_obs) + np.abs(obs_da - mean_obs)) ** 2).sum(dim=time_name, skipna=True)

    return 1 - (numerator / denominator)


def nse(sim_da, obs_da, time_name):
    numerator = ((sim_da - obs_da) ** 2).sum(dim=time_name, skipna=True)
    denominator = ((obs_da - obs_da.mean(dim=time_name, skipna=True)) ** 2).sum(dim=time_name, skipna=True)

    return 1 - (numerator / denominator)
