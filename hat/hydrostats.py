"""high level python api for hydrological statistics"""
from typing import List

import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from branca.colormap import linear
from folium.plugins import Fullscreen

from hat import hydrostats_functions


def run_analysis(
    functions: List,
    sims_ds: xr.DataArray,
    obs_ds: xr.DataArray,
) -> xr.Dataset:
    """Run statistical analysis on simulation and observation timeseries"""

    # list of stations
    stations = sims_ds.coords["station"].values

    ds = xr.Dataset()

    # For each statistical function
    for name in functions:
        # get function itself from name
        func = getattr(hydrostats_functions, name)

        # do timeseries analysis for each station
        # (using a "numpy in, numpy out" function)
        statistics = []
        for station in stations:
            sims = sims_ds.sel(station=station).to_numpy()
            obs = obs_ds.sel(station=station).to_numpy()
           
            stat = func(sims, obs)
            if stat is None:
                print(f"Warning! All NaNs for station {station}")
                stat = 0
            statistics += [stat]
        statistics = np.array(statistics)

        # Add the Series to the DataFrame
        ds[name] = xr.DataArray(statistics, coords={'station': stations})

    return ds


def display_map(ds: xr.Dataset, name: str, minv: float = 0, maxv: float = 1):
    # xarray to geopandas
    gdf = gpd.GeoDataFrame(
        ds.to_dataframe(),
        geometry=gpd.points_from_xy(ds["longitude"], ds["latitude"]),
        crs="epsg:4326",
    )
    gdf["station_id"] = gdf.index
    gdf = gdf[~gdf[name].isnull()]

    # Create a color map
    colormap = linear.Blues_09.scale(minv, maxv)

    # Define style function
    def style_function(feature):
        property = feature["properties"][name]
        return {"fillOpacity": 0.7, "weight": 0, "fillColor": colormap(property)}

    m = folium.Map(location=[48, 5], zoom_start=5, prefer_canvas=True, tiles=None)
    _ = folium.GeoJson(
        gdf,
        marker=folium.CircleMarker(fillColor="white", fillOpacity=0.5, radius=5),
        name=name,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=["station_id", name]),
        popup=folium.GeoJsonPopup(fields=["station_id", name]),
    ).add_to(m)

    # Add the CartoDB Positron tileset as a layer
    cartodb_positron = folium.TileLayer(
        tiles="CartoDB Dark_Matter",
        name="Dark",
        overlay=False,
        control=True,
    )
    cartodb_positron.add_to(m)

    # Add the CartoDB Positron tileset as a layer
    cartodb_positron = folium.TileLayer(
        tiles="CartoDB Positron",
        name="Light",
        overlay=False,
        control=True,
    )
    cartodb_positron.add_to(m)

    # Add OpenStreetMap layer
    open_street_map = folium.TileLayer(
        tiles="OpenStreetMap",
        name="Open Street Map",
        overlay=False,
        control=True,
    )
    open_street_map.add_to(m)

    # Add the satellite layer
    esri_satellite = folium.TileLayer(
        tiles="""https://server.arcgisonline.com/ArcGIS/rest/services/
        World_Imagery/MapServer/tile/{z}/{y}/{x}""",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    )
    esri_satellite.add_to(m)

    # add controls
    folium.LayerControl().add_to(m)
    Fullscreen().add_to(m)

    return m
