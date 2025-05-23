import numpy as np


def bias(sim_da, obs_da, time_name):
    return (sim_da - obs_da).mean(dim=time_name, skipna=True)


def apb(sim_da, obs_da, time_name):
    return np.abs(sim_da - obs_da).sum(dim=time_name, skipna=True) / obs_da.sum(dim=time_name, skipna=True)


def apb2(sim_da, obs_da, time_name):
    return np.abs(sim_da.mean(dim=time_name, skipna=True) - obs_da.mean(dim=time_name, skipna=True)) / obs_da.mean(
        dim=time_name, skipna=True
    )


def mae(sim_da, obs_da, time_name):
    return (sim_da - obs_da).abs().mean(dim=time_name, skipna=True)


# @hydrostat
# def br(s, o):
#     """
#     Bias ratio
#     input:
#         s: simulated
#         o: observed
#     output:
#         br: bias ratio
#     """
#     return 1 - abs(np.mean(s) / np.mean(o) - 1)


# @hydrostat
# def correlation(s, o):
#     """
#     correlation coefficient
#     input:
#         s: simulated
#         o: observed
#     output:
#         correlation: correlation coefficient
#     """
#     return np.corrcoef(o, s)[0, 1]


# @hydrostat
# def kge(s, o):
#     """
#         Kling Gupta Efficiency
#         input:
#         s: simulated
#         o: observed
#     output:
#         KGE: Kling Gupta Efficiency
#     """
#     B = np.mean(s) / np.mean(o)
#     y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
#     r = np.corrcoef(o, s)[0, 1]

#     return 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)


# @hydrostat
# def index_agreement(s, o):
#     """
#         index of agreement
#         input:
#         s: simulated
#         o: observed
#     output:
#         ia: index of agreement
#     """
#     return 1 - (np.sum((o - s) ** 2)) / (
#         np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
#     )


# @hydrostat
# def ns(s, o):
#     """
#     Nash-Sutcliffe efficiency coefficient
#     input:
#         s: simulated
#         o: observed
#     output:
#         NS: Nash-Sutcliffe efficient coefficient
#     """
#     return 1 - sum((s - o) ** 2) / sum((o - np.mean(o)) ** 2)


# @hydrostat
# def nslog(s, o):
#     """
#     Nash-Sutcliffe efficiency coefficient from log-transformed data
#     input:
#         s: simulated
#         o: observed
#     output:
#         NSlog: Nash-Sutcliffe efficient coefficient from log-transformed data
#     """
#     s = np.log(s)
#     o = np.log(o)
#     return 1 - sum((s - o) ** 2) / sum((o - np.mean(o)) ** 2)


# @hydrostat
# def pc_bias(s, o):
#     """
#     Percent Bias
#     input:
#         s: simulated
#         o: observed
#     output:
#         pc_bias: percent bias
#     """
#     return 100.0 * sum(s - o) / sum(o)


# @hydrostat
# def pc_bias2(s, o):
#     """
#     Percent Bias 2
#     input:
#         s: simulated
#         o: observed
#     output:
#         apb2: absolute percent bias 2
#     """
#     return 100 * (np.mean(s) - np.mean(o)) / np.mean(o)


# @hydrostat
# def rmse(s, o):
#     """
#     Root Mean Squared Error
#     input:
#         s: simulated
#         o: observed
#     output:
#         rmses: root mean squared error
#     """
#     return np.sqrt(np.mean((s - o) ** 2))


# @hydrostat
# def rsr(s, o):
#     """
#     RMSE-observations standard deviation ratio
#     input:
#         s: simulated
#         o: observed
#     output:
#         RSR: RMSE-observations standard deviation ratio
#     """

#     rmse = np.sqrt(np.sum((s - o) ** 2))
#     stdev_obs = np.sqrt(np.sum((o - np.mean(o)) ** 2))
#     return rmse / stdev_obs


# @hydrostat
# def vr(s, o):
#     """
#         Variability ratio
#         input:
#         s: simulated
#         o: observed
#     output:
#         vr: variability ratio
#     """
#     return 1 - abs((np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o)) - 1)
