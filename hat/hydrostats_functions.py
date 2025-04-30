"""
Collection of hydrostats functions often used in hydrological sciences.

Inputs
- two equal sized numpy arrays of simulation and observation timeseries

Output
- scalar metric

Functions:
    RSR :     RMSE-observations standard deviation ratio
    br :      bias ratio
    pc_bias : percentage bias
    pc_bias2: percentage bias 2
    apb :     absolute percent bias
    apb2 :    absolute percent bias 2
    rmse :    root mean square error
    mae :     mean absolute error
    bias :    bias
    NS :      Nash Sutcliffe Coefficient
    NSlog :   Nash Sutcliffe Coefficient from log-transformed data
    correlation: correlation
    KGE:      Kling Gupta Efficiency
    vr :      variability ratio
"""

import numpy as np

from hat.hydrostats_decorators import hydrostat

# NOTE only functions in this list will be available to other hat modules
__all__ = [
    "apb",
    "apb2",
    "bias",
    "br",
    "correlation",
    "kge",
    "index_agreement",
    "mae",
    "ns",
    "nslog",
    "pc_bias",
    "pc_bias2",
    "rmse",
    "rsr",
    "vr",
]


@hydrostat
def apb(s, o):
    """
    Absolute Percent Bias
    input:
        s: simulated
        o: observed
    output:
        apb: absolute percent bias
    """
    return 100.0 * sum(abs(s - o)) / sum(o)


@hydrostat
def apb2(s, o):
    """
    Absolute Percent Bias 2
    input:
        s: simulated
        o: observed
    output:
        apb2: absolute percent bias 2
    """
    return 100 * abs(np.mean(s) - np.mean(o)) / np.mean(o)


@hydrostat
def bias(s, o):
    """
    Bias
    input:
        s: simulated
        o: observed
    output:
        bias: bias
    """
    return np.mean(s - o)


@hydrostat
def br(s, o):
    """
    Bias ratio
    input:
        s: simulated
        o: observed
    output:
        br: bias ratio
    """
    return 1 - abs(np.mean(s) / np.mean(o) - 1)


@hydrostat
def correlation(s, o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    return np.corrcoef(o, s)[0, 1]


@hydrostat
def kge(s, o):
    """
        Kling Gupta Efficiency
        input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0, 1]

    return 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)


@hydrostat
def index_agreement(s, o):
    """
        index of agreement
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    return 1 - (np.sum((o - s) ** 2)) / (
        np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    )


@hydrostat
def mae(s, o):
    """
    Mean Absolute Error
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    return np.mean(abs(s - o))


@hydrostat
def ns(s, o):
    """
    Nash-Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        NS: Nash-Sutcliffe efficient coefficient
    """
    return 1 - sum((s - o) ** 2) / sum((o - np.mean(o)) ** 2)


@hydrostat
def nslog(s, o):
    """
    Nash-Sutcliffe efficiency coefficient from log-transformed data
    input:
        s: simulated
        o: observed
    output:
        NSlog: Nash-Sutcliffe efficient coefficient from log-transformed data
    """
    s = np.log(s)
    o = np.log(o)
    return 1 - sum((s - o) ** 2) / sum((o - np.mean(o)) ** 2)


@hydrostat
def pc_bias(s, o):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    return 100.0 * sum(s - o) / sum(o)


@hydrostat
def pc_bias2(s, o):
    """
    Percent Bias 2
    input:
        s: simulated
        o: observed
    output:
        apb2: absolute percent bias 2
    """
    return 100 * (np.mean(s) - np.mean(o)) / np.mean(o)


@hydrostat
def rmse(s, o):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    return np.sqrt(np.mean((s - o) ** 2))


@hydrostat
def rsr(s, o):
    """
    RMSE-observations standard deviation ratio
    input:
        s: simulated
        o: observed
    output:
        RSR: RMSE-observations standard deviation ratio
    """

    rmse = np.sqrt(np.sum((s - o) ** 2))
    stdev_obs = np.sqrt(np.sum((o - np.mean(o)) ** 2))
    return rmse / stdev_obs


@hydrostat
def vr(s, o):
    """
        Variability ratio
        input:
        s: simulated
        o: observed
    output:
        vr: variability ratio
    """
    return 1 - abs((np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o)) - 1)
