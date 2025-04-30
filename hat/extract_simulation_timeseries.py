import geopandas as gpd
import xarray as xr

from hat.config import load_package_config, valid_custom_config
from hat.geo import geopoints_to_array, latlon_coords
from hat.timeseries import assign_stations, extract_timeseries_using_mask

# Read configutation files
# NOTE config files are installed as part of hat package using setup.py
# (just updating the .json files will not necessarily work
# (i.e. would require pip install -e .)
DEFAULT_CONFIG = load_package_config("timeseries.json")


def extract_timeseries(
    stations: gpd.GeoDataFrame,
    simulations_da: xr.DataArray,
    config: dict = DEFAULT_CONFIG,
):
    config = valid_custom_config(config)

    coords = latlon_coords(simulations_da)

    station_mask = geopoints_to_array(stations, coords)

    da_points = extract_timeseries_using_mask(simulations_da, station_mask)

    da_stations = assign_stations(
        stations,
        station_mask,
        da_points,
        coords,
        config["station_id_column_name"],
    )

    return da_stations
