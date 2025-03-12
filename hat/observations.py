import geopandas as gpd
import pandas as pd

from hat.data import is_csv
from hat.filters import filter_dataframe


def add_geometry_column(gdf: gpd.GeoDataFrame, coord_names):
    """
    Adds a geometry column to a GeoDataFrame using columns specified by coord_names.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to add geometry to.
    coord_names : tuple
        Tuple of column names to use as x and y coordinates, in lat/lon order.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometry column added.
    """

    x_coord_name, y_coord_name = coord_names

    # errors='coerce' turns non-numeric to NaN
    gdf["x"] = pd.to_numeric(gdf[x_coord_name], errors="coerce")
    gdf["y"] = pd.to_numeric(gdf[y_coord_name], errors="coerce")

    gdf = gdf.dropna(subset=["x", "y"])

    # filter rows that do not plot on Earth (e.g. -9999)
    gdf = gdf[
        (gdf["x"] >= -180) & (gdf["x"] <= 180) & (gdf["y"] >= -90) & (gdf["y"] <= 90)
    ]

    gdf["geometry"] = gpd.points_from_xy(gdf[x_coord_name], gdf[y_coord_name])

    return gpd.GeoDataFrame(gdf, geometry="geometry")


def read_station_metadata_file(
    fpath: str, coord_names: str, epsg: int, filters: str = None
) -> gpd.GeoDataFrame:
    """
    read hydrological stations from file. will cache as pickle object
    because .csv file used by the team takes 12 seconds to load
    """

    try:
        if is_csv(fpath):
            gdf = gpd.read_file(fpath)
            gdf = add_geometry_column(gdf, coord_names)
            gdf = gdf.set_crs(epsg=epsg)
        else:
            gdf = gpd.read_file(fpath)
    except Exception:
        raise Exception(f"Could not open file {fpath}")

    # (optionally) filter the stations, e.g. 'Continent == Europe'
    if filters is not None:
        gdf = filter_dataframe(gdf, filters)
    return gdf
