import pandas as pd
import xarray as xr


# @st.cache_data
def temporal_filter(
    _metadata,
    _observations: pd.DataFrame,
    timeperiod,
    station_id_name="station_id",
):
    """
    filter station metadata and timeseries by timeperiod

    timeperiod is a tuble of (start, end) datetime objects

    the underscores for metadata and timeseries variables tells
    ignore them when caching (i.e. only look for changes in timeperiod)
    """

    # parse time period into start and end date
    start_date, end_date = timeperiod

    # filter observation timeseries by timeperiod
    _observations = _observations[start_date:end_date]

    # filter metadata by stations with data in this time perdiod
    # (i.e. count above zero)
    count = _observations.sum()
    keep = count[count > 0]
    _metadata = _metadata[_metadata[station_id_name].isin(keep.index)]

    return (_metadata, _observations)


def calibration_station_filter(metadata, calibration_stations):
    """filter station metadata by calibration station collection"""

    if calibration_stations == "any station":
        return metadata

    return metadata[metadata[calibration_stations] == 1]


def quality_flag_filter(metadata, quality_flag):
    """filter station metadata by quality flag"""

    if quality_flag == "any station":
        return metadata

    return metadata[metadata["qflag_01deg"] == int(quality_flag)]


def drainage_area_filter(metadata, drainage_area):
    """filter station metadata by drainage area"""

    min_area, max_area = drainage_area
    metadata = metadata[metadata["DrainingArea.km2.LDD"] >= min_area]
    metadata = metadata[metadata["DrainingArea.km2.LDD"] <= max_area]

    return metadata


def apply_station_filters(metadata, timeseries, station_filters):
    """apply station filters in one go"""

    metadata, timeseries = temporal_filter(
        metadata, timeseries, station_filters["timeperiod"]
    )
    metadata = calibration_station_filter(
        metadata, station_filters["calibration_stations"]
    )
    metadata = quality_flag_filter(metadata, station_filters["quality_flag"])
    metadata = drainage_area_filter(metadata, station_filters["drainage_area"])

    return metadata, timeseries


def stations_with_discharge(obs, timeperiod, metadata):
    """only keep stations with observed discharge in this timeperiod"""

    # filter observations by simulation timeperiod
    obs = obs.sel(time=slice(*timeperiod))

    # filter station metadata
    # (only keep stations with observed discharge in this timeperiod)
    obsdis = obs.obsdis
    valid_stations = obsdis.station[obsdis.sum("time") > 0]
    valid_station_ids = [f"G{int(num):04d}" for num in valid_stations.data]
    metadata = metadata[metadata["ObsID"].isin(valid_station_ids)]

    return metadata, obsdis


def apply_filter(df: pd.DataFrame, key: str, operator: str, value: str) -> pd.DataFrame:
    """
    Apply the filter on the DataFrame based on the provided
    key, operator, and value.
    """
    operators = {
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
    }

    if key not in df.columns:
        raise ValueError(f"Key '{key}' does not exist as column name in dataframe")

    if operator not in operators:
        raise ValueError(f"Operator '{operator}' is not supported")

    filter_func = operators[operator]

    return df[filter_func(df[key], value)]


def filter_dataframe(df, filters: str):
    """
    Process the --filter option value and apply the filter on the DataFrame.
    """

    if not filters:
        return df

    # multiple filters are separated with a comma ','
    if "," in filters:
        filters = filters.split(",")
    else:
        filters = [filters]

    # apply each filter in turn
    for filter_str in filters:
        if filter_str == "":
            continue
        parts = filter_str.split()
        if len(parts) != 3:
            raise ValueError("Invalid filter format. Expected 'key operator value'.")

        key, operator, value = parts

        df = apply_filter(df, key, operator, value)

    if len(df) == 0:
        raise ValueError("There are no remaining rows (try different filters?)")

    return df


def filter_timeseries(sims_ds: xr.DataArray, obs_ds: xr.DataArray, threshold=80):
    """Clean the simulation and observation timeseries

    Only keep..

    - stations in both the observation and simulation datasets
    - observations in the same time period as the simulations
    - observations with enough valid data in this timeperiod
    - simulations that match the remaining observations

    """
    # Only keep stations in both the observation and simulation datasets
    matching_stations = sorted(
        set(sims_ds.station.values).intersection(obs_ds.station.values)
    )
    print(len(matching_stations))
    sims_ds = sims_ds.sel(station=matching_stations)
    obs_ds = obs_ds.sel(station=matching_stations)
    obs_ds = obs_ds.sel(time=sims_ds.time)

    obs_ds = obs_ds.dropna(dim="station", how="all")
    sims_ds = sims_ds.sel(station=obs_ds.station)

    # Only keep observations in the same time period as the simulations
    # obs_ds = obs_ds.where(sims_ds.time == obs_ds.time, drop=True)

    # Only keep obsevations with enough valid data in this timeperiod

    # discharge data
    print(sims_ds)
    print(obs_ds)

    # Replace negative values with NaN
    # dis = dis.where(dis >= 0)

    # # Percentage of valid discharge data at each point in time
    # valid_percent = dis.notnull().mean(dim="time") * 100

    # # Boolean index of where there is enough valid data
    # enough_observation_data = valid_percent > threshold

    # # keep where there is enough observation data
    # obs_ds = obs_ds.where(enough_observation_data, drop=True)

    # # keep simulation that match remaining observations
    # sims_ds = sims_ds.where(enough_observation_data, drop=True)

    return (sims_ds, obs_ds)
