from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt


# PLOTLY (interactive)
def plotly_timeseries(t, y):
    df = pd.DataFrame({"time": t, "discharge": y})
    return px.line(df, x="time", y="discharge", title="Discharge Timeseries")


# MATPLOTLIB (not interactive)
def plot_timeseries(t, y, jupyter=False):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(14, 6)
    ax1.plot(t, y, "dodgerblue")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("discharge", color="b")
    ax1.tick_params("y", colors="b")

    if jupyter:
        return fig

    st.write(fig)


def histogram(
        arr: Union[np.array, np.ma.MaskedArray],
        bins=10,
        clip=None,
        title="Histogram",
        figsize=(6, 4),
):
    """plot histogram of a numpy array or masked numpy array"""

    # apply mask (if one exists)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.compressed()

    # return if not numpy
    if not isinstance(arr, np.ndarray):
        print("histogram() requires a numpy array or masked numpy array")
        return

    # remove flat dimensions
    arr = arr.squeeze()

    # remove nans
    arr = arr[~np.isnan(arr)]

    # histogram range (percentile clip or minmax)
    if clip:
        histogram_range = (
            round(np.percentile(arr, clip)),
            round(np.percentile(arr, 100 - clip)),
        )
    else:
        histogram_range = (np.min(arr), np.max(arr))

    # count number of values in each bin
    counts, bins = np.histogram(arr, bins=bins, range=histogram_range)

    _ = plt.figure(figsize=figsize)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(title)

    # show plot
    plt.show()
