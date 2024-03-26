# add file location to python path
import json
import os
import sys

import geopandas as gpd
import numpy as np
import shapely
from jsonschema import ValidationError, validate
from pyproj import CRS, Transformer

sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Lat lon to EFAS grid Transforer
EFAS_5KM_CRS = CRS.from_proj4(
    """ +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000
     +y_0=3210000 +ellps=GRS80 +units=m +no_defs """
)
latlon_to_efas5km = Transformer.from_crs(4326, EFAS_5KM_CRS)


def reproject_4326_to_3857(xarr):
    """reproject an xarray from epsg 4326 to 3857"""

    # EPSG 4326 is the WGS84 ellipsoid, "latlon", coordinate reference system
    crs = CRS.from_epsg(4326)

    # explicitly set CRS to EPSG:4326 (sometimes only implicit in file)
    xarr = xarr.rio.set_crs(crs)

    return xarr.rio.reproject(3857)


def reproject_efas_5km(xarr, epsg):
    """safely reprojects EFAS 5km.
    CRS was not read from file so defined from docs"""
    crs = CRS.from_proj4(
        """ +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000
         +y_0=3210000 +ellps=GRS80 +units=m +no_defs """
    )
    xarr = xarr.rio.set_crs(crs)

    return xarr.rio.reproject(epsg)


def transform_bounds(src_bounds, src_crs, dst_crs):
    """transform bounding box from source to destination crs"""

    transformer = Transformer.from_crs(src_crs, dst_crs)

    left, bottom, right, top = src_bounds

    minx, miny = transformer.transform(left, bottom)
    maxx, maxy = transformer.transform(right, top)

    return ((minx, miny), (maxx, maxy))


def shapely_to_geojson(shapely_object, fpath="data.geojson"):
    geo_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": shapely.geometry.mapping(shapely_object),
            }
        ],
    }

    with open(fpath, "w") as f:
        json.dump(geo_dict, f)


def point_in_bounds(point, bounds):
    """is point (x,y) in bounds (minx, miny, maxx, maxy)"""

    x, y = point
    minx, miny, maxx, maxy = bounds

    point = shapely.Point(x, y)
    box = shapely.box(minx, miny, maxx, maxy)
    return point.within(box)


def find_nearest(array, value):
    """nearest value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def name_of_adjusted_coord(
    coord_name: str, model: str, version_num: int, resolution: str
):
    """column name in station metadata of manually adjusted coordinate
    based on model river network"""
    return f"{model.title()}{coord_name.title()}{version_num}_{resolution}"


def name_of_adjusted_coords(river_network: str, version=1):
    """names of columns in station metadata with river network coordinates"""

    model, resolution = river_network.split("_")
    lat_name = name_of_adjusted_coord("lat", model, version, resolution)
    lon_name = name_of_adjusted_coord("lon", model, version, resolution)

    return (lon_name, lat_name)


def river_network_geometry(metadata, river_network, version=1):
    """station metadata geodataframe with river network corrected geometry"""

    if river_network == "EPSG:4326":
        return metadata

    lon_name, lat_name = name_of_adjusted_coords(river_network, version=version)

    gdf = metadata.copy(deep=True)
    gdf["geometry"] = gpd.points_from_xy(gdf[lon_name], gdf[lat_name])

    return gdf


def river_network_to_coord_names_mapping():
    """explicit mapping between river network name and coord names
    (to handle inconsistent naming conventions)"""

    return {
        "station": ("StationLon", "StationLat"),
        "cama_3min": ("CamaLon1_3min", "CamaLat1_3min"),
        "cama_6min": ("CamaLon1_6min", "CamaLat1_6min"),
        "cama_15min": ("CamaLon1_15min", "CamaLat1_15min"),
        "lisflood_5km": ("LisfloodX5k", "LisfloodY5k"),
        "lisflood_1min": ("LisfloodX.efas1min", "LisfloodY.efas1min"),
        "lisflood_3min": ("LisfloodX_3min", "LisfloodY_3min"),
    }


def river_network_to_coord_names(river_network: str = "") -> dict:
    "river network name to coordinate column names in station metadata table"

    # default is station lonlat (epsg:4326)
    if not river_network:
        return {"x": "StationLon", "y": "StationLat"}

    # mapping of rivernetwork to coord names
    river_network_to_coords = river_network_to_coord_names_mapping()

    # river network must be in mapping dict
    valid_river_networks = list(river_network_to_coords.keys())

    # return coord names
    if river_network.lower() in valid_river_networks:
        x, y = river_network_to_coords[river_network]
        return {"x": x, "y": y}
    else:
        print(f"River network '{river_network}' not in: {valid_river_networks}")


def geojson_schema():
    """GeoJSON schema definition"""

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GeoJSON schema",
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "FeatureCollection",
                    "Feature",
                    "Point",
                    "MultiPoint",
                    "LineString",
                    "MultiLineString",
                    "Polygon",
                    "MultiPolygon",
                    "GeometryCollection",
                ],
            },
            "bbox": {
                "type": "array",
                "minItems": 4,
                "maxItems": 6,
                "items": {"type": "number"},
            },
            "id": {"type": "string"},
            "geometry": {"$ref": "#/definitions/geometry"},
            "properties": {"type": "object"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "Feature"},
                        "id": {"type": "string"},
                        "geometry": {"$ref": "#/definitions/geometry"},
                        "properties": {"type": "object"},
                    },
                    "required": ["type", "properties", "geometry"],
                },
            },
            "geometries": {
                "type": "array",
                "items": {"$ref": "#/definitions/geometry"},
            },
        },
        "definitions": {
            "geometry": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "Point",
                            "MultiPoint",
                            "LineString",
                            "MultiLineString",
                            "Polygon",
                            "MultiPolygon",
                            "GeometryCollection",
                        ],
                    },
                    "coordinates": {
                        "oneOf": [
                            {"$ref": "#/definitions/pointCoordinates"},
                            {"$ref": "#/definitions/multiPointCoordinates"},
                            {"$ref": "#/definitions/lineStringCoordinates"},
                            {"$ref": "#/definitions/multiLineStringCoordinates"},
                            {"$ref": "#/definitions/polygonCoordinates"},
                            {"$ref": "#/definitions/multiPolygonCoordinates"},
                        ]
                    },
                    "geometries": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/geometry"},
                    },
                },
                "required": ["type"],
            },
            "pointCoordinates": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "number"},
            },
            "multiPointCoordinates": {
                "type": "array",
                "items": {"$ref": "#/definitions/pointCoordinates"},
            },
            "lineStringCoordinates": {
                "type": "array",
                "items": {"$ref": "#/definitions/pointCoordinates"},
            },
            "multiLineStringCoordinates": {
                "type": "array",
                "items": {"$ref": "#/definitions/lineStringCoordinates"},
            },
            "polygonCoordinates": {
                "type": "array",
                "items": {"$ref": "#/definitions/lineStringCoordinates"},
            },
            "multiPolygonCoordinates": {
                "type": "array",
                "items": {"$ref": "#/definitions/polygonCoordinates"},
            },
        },
    }


def is_valid_geojson(file_path):
    """check if a file is valid geojson"""

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        validate(data, geojson_schema())
        return True
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Invalid GeoJSON: {e}")
        return False


def geopoints_to_array(gdf, array_coords) -> np.ndarray:
    """Create 2D boolean array where points in gdf nearest to array coords

    NOTE numpy arrays convention is (row, col) which corresponds to (y,x)
    """

    # check is geopandas dataframe
    if not isinstance(gdf, gpd.GeoDataFrame):
        print("Points must be in a geopandas dataframe")
        return

    # check geometries are points
    points = gdf.geometry
    if not all(isinstance(point, shapely.Point) for point in points):
        print("Geometries must be shapely POINT")
        return

    # x and y (e.g. longitude and latitude) of georeferenced points
    point_xs = np.array([point.x for point in points])
    point_ys = np.array([point.y for point in points])

    # nearest neighbour indices of the points in the array
    x_indices = [(np.abs(array_coords["x"] - point_x)).argmin() for point_x in point_xs]
    y_indices = [(np.abs(array_coords["y"] - point_y)).argmin() for point_y in point_ys]

    # create an empty boolean array
    shape = (len(array_coords["y"]), len(array_coords["x"]))
    arr = np.full(shape, False)

    # set point elements to True
    arr[y_indices, x_indices] = True

    return arr


def geopoints_from_csv(fpath: str, lat_name: str, lon_name: str) -> gpd.GeoDataFrame:
    """Load georeferenced points from file.
    Requires name of latitude and longitude columns"""

    gdf = gpd.read_file(fpath)
    gdf["geometry"] = gpd.points_from_xy(gdf[lon_name], gdf[lat_name])

    return gdf


def get_latlon_keys(ds):
    lat_key = None
    lon_key = None
    if "lat" in ds.coords and "lon" in ds.coords:
        lat_key = "lat"
        lon_key = "lon"
    elif "latitude" in ds.coords and "longitude" in ds.coords:
        lat_key = "latitude"
        lon_key = "longitude"
    else:
        raise Exception(
            f"Lat/lon coordinates could not be detected in dataset with coords {ds.coords}"  # noqa: E501
        )

    return lat_key, lon_key


def latlon_coords(ds, names: list = []):
    """Latitude and longitude coordinates of an xarray"""

    if not names:
        lat_key, lon_key = get_latlon_keys(ds)
    else:
        assert len(names) == 2, "Must provide two names for lat and lon"
    lat_key, lon_key = names
    lat_coords = ds.coords.get(lat_key).data
    lon_coords = ds.coords.get(lon_key).data
    coords = {"x": lon_coords, "y": lat_coords}

    return coords
