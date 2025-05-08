import geopandas as gpd
import pandas as pd

from hat.data import is_csv, read_csv_and_cache
from hat.filters import filter_dataframe


def add_geometry_column(gdf: gpd.GeoDataFrame, coord_names, epsg):
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
    gdf = gdf[(gdf["x"] >= -180) & (gdf["x"] <= 180) & (gdf["y"] >= -90) & (gdf["y"] <= 90)]

    # Create a geometry column
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf["x"], gdf["y"]), crs=f"EPSG:{epsg}")

    return gpd.GeoDataFrame(gdf, geometry="geometry")


def read_station_metadata_file(fpath: str, coord_names: str, epsg: int, filters: str = None) -> gpd.GeoDataFrame:
    """read hydrological stations from file. will cache as pickle object
    because .csv file used by the team takes 12 seconds to load"""

    try:
        if is_csv(fpath):
            gdf = read_csv_and_cache(fpath)
            gdf = add_geometry_column(gdf, coord_names, epsg)
            gdf = gdf.set_crs(epsg=epsg)
        else:
            gdf = gpd.read_file(fpath)
    except Exception:
        raise Exception(f"Could not open file {fpath}")

    # (optionally) filter the stations, e.g. 'Contintent == Europe'
    if filters is not None:
        gdf = filter_dataframe(gdf, filters)
    return gdf
