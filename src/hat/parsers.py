import dateutil.parser
import streamlit as st


@st.cache_data
def datetime_from_cftime(cftimes):
    """parse CFTimeIndex to python datetime,
    e.g. from a NetCDF file ds.indexes['time']"""
    return [dateutil.parser.parse(x.isoformat()) for x in cftimes]


def simulation_timeperiod(sim):
    # simulation timeperiod
    min_time = min(sim.indexes["time"])
    max_time = max(sim.indexes["time"])

    return (min_time, max_time)
