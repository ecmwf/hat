"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""
import json
import plotly.graph_objs as go
import numpy as np
import ipywidgets as widgets
from ipywidgets import HTML
from ipyleaflet import Map, Marker, GeoJSON, Popup
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point
from IPython.display import display


def initialize_map(center_lat, center_lon):
    """Initialize a leaflet map centered at the provided coordinates."""
    m = Map(center=[center_lat, center_lon], zoom=10, layout=widgets.Layout(width='50%'))
    return m


def initialize_plot():
    """Initialize a plotly figure widget."""
    f = go.FigureWidget(
        layout=go.Layout(
            width=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    )
    return f


def update_plot(f, station_data, station_name):
    f.data = []
    
    # Dynamically detect variables from the provided station data
    detected_variables = list(station_data.keys())
    
    # Loop through detected variables and plot
    for var in detected_variables:
        y_data = station_data.get(var, [])
        if isinstance(y_data, np.ndarray):
            data_exists = y_data.size > 0
        elif isinstance(y_data, (list, tuple, pd.Series)):
            data_exists = len(y_data) > 0
        else:
            data_exists = False

        if data_exists:
            f.add_scatter(y=y_data, mode='lines+markers', name=var)

    # Updated to use the passed station_name
    f.layout.title = f"Time Series for : {station_name}"



def compute_center_coordinates(stations):
    """Compute the center coordinates for all stations."""
    lats = [station_data['lat'] for station_data in stations.values()]
    lons = [station_data['lon'] for station_data in stations.values()]
    center_lat = (max(lats) + min(lats)) / 2
    center_lon = (max(lons) + min(lons)) / 2
    return center_lat, center_lon


def display_geospatial(stations):
    """Display markers on a map for each station and show associated data in a plot."""

    center_lat, center_lon = compute_center_coordinates(stations)
    m = initialize_map(center_lat, center_lon)
    f = initialize_plot()

    def handle_click(station_id, lat, lon, **kwargs):
        # Passing station_id as station_name to update_plot
        update_plot(f, stations[station_id], station_id)
        m.center = [lat, lon]

    for station_id, data in stations.items():
        marker = Marker(location=[data['lat'], data['lon']], draggable=False)

        def callback(*args, station_id=station_id, lat=data['lat'], lon=data['lon'], **kwargs):
            handle_click(station_id, lat, lon, **kwargs)

        marker.on_click(callback)
        m.add_layer(marker)

    layout = widgets.HBox([m, f])
    display(layout)
    return m

    
def handle_timestamp(obj):
    """
    Custom function for handling Timestamps during JSON serialization.
    """
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def handle_geojson_click(map_object, feature, **kwargs):
    """
    Handle click events on a GeoJSON layer in an ipyleaflet map.
    
    Parameters:
    - map_object: The ipyleaflet map object to which the GeoJSON layer is added.
    - feature: The clicked feature from the GeoJSON layer.
    """
    properties = feature['properties']
    if not properties:
        return

    # Exclude 'style' and any other undesired keys from the properties
    excluded_keys = ['style']
    description_items = [(k, v) for k, v in properties.items() if v and k not in excluded_keys]
    description = '<br>'.join(f"<b>{k}</b>: {v}" for k, v in description_items)

    # Compute the centroid of the clicked feature
    geom = shape(feature['geometry'])
    centroid = geom.centroid

    if isinstance(centroid, Point):
        popup_location = (centroid.y, centroid.x)
    else:
        # If the feature has a complex geometry (e.g., MultiPolygon) and does not have a Point centroid, default to map center
        popup_location = map_object.center

    # Create a popup with the properties and add it to the map at the computed centroid location
    popup = Popup(location=popup_location, child=HTML(value=description), close_button=True, auto_close=True)
    map_object.add_layer(popup)



def add_vector_to_map(map_object, vector_file_path, fill_color="#F00", line_color="#000", opacity=0.2, line_weight = 1, ):
    """
    Add vector data (from a Shapefile or GeoJSON) to an ipyleaflet map.
    """
    # Determine the file type from its extension
    file_type = vector_file_path.split('.')[-1].lower()
    
    if file_type == 'shp':
        gdf = gpd.read_file(vector_file_path)
        geojson_content = json.loads(gdf.to_json())

    elif file_type == 'geojson':
        gdf = gpd.read_file(vector_file_path)
        geojson_content = json.loads(gdf.to_json(default=handle_timestamp))
    
    # Style function
    def geojson_style(feature):
        return {
            'fillColor': fill_color,
            'color': line_color,
            'weight': line_weight,
            'fillOpacity': opacity
        }
    
    geojson_layer = GeoJSON(data=geojson_content, style_callback=geojson_style)

    # Pass the map_object to handle_geojson_click using a lambda function
    geojson_layer.on_click(lambda feature, **kwargs: handle_geojson_click(map_object, feature, **kwargs))

    map_object.add_layer(geojson_layer)


