import geopandas as gpd
import pandas as pd

from hat.config import load_package_config
from hat.data import is_csv, read_csv_and_cache
from hat.filters import filter_dataframe

# Read configutation files
# NOTE config files are installed as part of hat package using setup.py
# (just updating the .json files will not necessarily work
# (i.e. would require pip install -e .)
COORD_NAMES = load_package_config("river_network_coordinate_names.json")
DEFAULT_CONFIG = load_package_config("timeseries.json")
RIVER_NETWORK_COORD_NAMES = COORD_NAMES[DEFAULT_CONFIG["river_network_name"]]
EPSG = DEFAULT_CONFIG["station_epsg"]


def add_geometry_column(gdf: gpd.GeoDataFrame, coord_names):
    """add geometry column to stations geodataframe. station must have valid
    coordinates, i.e. not missing data and plot somewhere on Earth"""

    # names of x and y coords for a given river network
    x_coord_name, y_coord_name = coord_names

    # Create numeric x and y columns by converting original string columns,
    # errors='coerce' will turn non-numeric (like empty strings) to NaN
    gdf["x"] = pd.to_numeric(gdf[x_coord_name], errors="coerce")
    gdf["y"] = pd.to_numeric(gdf[y_coord_name], errors="coerce")

    # Drop rows with NaN values in either x or y columns
    gdf = gdf.dropna(subset=["x", "y"])

    # Filter rows that do not plot on Earth (e.g. -9999)
    gdf = gdf[
        (gdf["x"] >= -180) & (gdf["x"] <= 180) & (gdf["y"] >= -90) & (gdf["y"] <= 90)
    ]

    # Create a geometry column
    gdf["geometry"] = gpd.points_from_xy(gdf[x_coord_name], gdf[y_coord_name])

    return gdf


def read_station_metadata_file(
    fpath: str, coord_names: str, epsg: int = EPSG
) -> gpd.GeoDataFrame:
    """read hydrological stations from file. will cache as pickle object
    because .csv file used by the team takes 12 seconds to load"""

    if is_csv(fpath):
        gdf = read_csv_and_cache(fpath)
        gdf = add_geometry_column(gdf, coord_names)
        gdf = gdf.set_crs(epsg=epsg)
    else:
        gdf = gpd.read_file(fpath)

    return gdf


def read_station_metadata(
    station_metadata_filepath, filters="", coord_names: str = RIVER_NETWORK_COORD_NAMES
):
    # read station metadata from file
    stations = read_station_metadata_file(station_metadata_filepath, coord_names)

    # (optionally) filter the stations, e.g. 'Contintent == Europe'
    if filters:
        stations = filter_dataframe(stations, filters)

    return stations
