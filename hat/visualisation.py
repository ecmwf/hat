"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""
import json
import plotly.graph_objs as go
import numpy as np
import ipywidgets as widgets
from ipywidgets import HTML
from ipyleaflet import Map, Marker, GeoJSON, Popup, Heatmap, Rectangle
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point
from IPython.display import display
import xarray as xr
from functools import partial



class GeoMap:
    def __init__(self, center_lat, center_lon):
        """Initialize a leaflet map centered at the provided coordinates."""
        self.map = Map(center=[center_lat, center_lon], zoom=4, layout=widgets.Layout(width='50%'))
    
    def add_marker(self, lat, lon, on_click_callback):
        """Add a marker to the map."""
        marker = Marker(location=[lat, lon], draggable=False)
        marker.on_click(on_click_callback)
        self.map.add_layer(marker)
    
    def add_rectangle(self, bounds, on_click_callback):
        """Add a rectangle to the map."""
        rectangle = Rectangle(bounds=bounds, color="blue", fill_opacity=0.01, opacity=0.5, weight=1)
        rectangle.on_click(on_click_callback)
        self.map.add_layer(rectangle)
    
    def add_geojson_layer(self, geojson_content, style_callback, on_click_callback):
        """Add a GeoJSON layer to the map."""
        geojson_layer = GeoJSON(data=geojson_content, style_callback=style_callback)
        geojson_layer.on_click(on_click_callback)
        self.map.add_layer(geojson_layer)
    
    def display(self):
        """Display the map."""
        display(self.map)

    @staticmethod
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
    

    def handle_geojson_click(self, feature, **kwargs):
        """
        Handle click events on a GeoJSON layer in an ipyleaflet map.

        Parameters:
        - feature: The clicked feature from the GeoJSON layer.
        """
        properties = feature['properties']
        if not properties:
            return

        excluded_keys = ['style']
        description_items = [(k, v) for k, v in properties.items() if v and k not in excluded_keys]
        description = '<br>'.join(f"<b>{k}</b>: {v}" for k, v in description_items)

        geom = shape(feature['geometry'])
        centroid = geom.centroid

        if isinstance(centroid, Point):
            popup_location = (centroid.y, centroid.x)
        else:
            popup_location = self.map.center

        popup = Popup(location=popup_location, child=HTML(value=description), close_button=True, auto_close=True)
        self.map.add_layer(popup)
    

    def add_vector_to_map(self, vector_file_path, fill_color="#F00", line_color="#000", opacity=0.2, line_weight=1):
        """
        Add vector data (from a Shapefile or GeoJSON) to an ipyleaflet map.
        """
        file_type = vector_file_path.split('.')[-1].lower()

        if file_type == 'shp':
            gdf = gpd.read_file(vector_file_path)
            geojson_content = json.loads(gdf.to_json())
        elif file_type == 'geojson':
            gdf = gpd.read_file(vector_file_path)
            geojson_content = json.loads(gdf.to_json(default=handle_timestamp))

        def geojson_style(feature):
            return {
                'fillColor': fill_color,
                'color': line_color,
                'weight': line_weight,
                'fillOpacity': opacity
            }

        geojson_layer = GeoJSON(data=geojson_content, style_callback=geojson_style)
        geojson_layer.on_click(lambda feature, **kwargs: self.handle_geojson_click(feature, **kwargs))
        self.map.add_layer(geojson_layer)  



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

def update_plots(f, station_data, station_name):
    # Avoid resetting f.data to keep existing plot data
    # f.data = []
    
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
            # Append a new scatter plot to f.data
            new_scatter = go.Scatter(y=y_data, mode='lines+markers', name=f"{station_name}: {var}")
            f.add_trace(new_scatter)

    # Updated to use the passed station_name
    f.layout.title = f"Time Series for Multiple Locations"




def compute_center_coordinates(stations):
    """Compute the center coordinates for all stations."""
    lats = [station_data['lat'] for station_data in stations.values()]
    lons = [station_data['lon'] for station_data in stations.values()]
    center_lat = (max(lats) + min(lats)) / 2
    center_lon = (max(lons) + min(lons)) / 2
    return center_lat, center_lon


def display_geospatial(stations):
    center_lat, center_lon = compute_center_coordinates(stations)
    geo_map = GeoMap(center_lat, center_lon)
    f = GeoMap.initialize_plot()  # Changed from initialize_plot() to GeoMap.initialize_plot()
    
    def handle_click(station_id, lat, lon, **kwargs):
        update_plot(f, stations[station_id], station_id)
        geo_map.map.center = [lat, lon]

    for station_id, data in stations.items():
        def callback(*args, station_id=station_id, lat=data['lat'], lon=data['lon'], **kwargs):
            handle_click(station_id, lat, lon, **kwargs)

        geo_map.add_marker(data['lat'], data['lon'], callback)

    layout = widgets.HBox([geo_map.map, f])
    display(layout)
    return geo_map.map

    
def handle_timestamp(obj):
    """
    Custom function for handling Timestamps during JSON serialization.
    """
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



def display_geospatial_nc(ds, subsample_factor=10):
    """Display grid points on a map for each location in the NetCDF and show associated data in a plot.

    Parameters:
    - ds: The xarray dataset.
    - subsample_factor: Factor by which to subsample the grid for visualization.
    """

    # Use the mean of latitudes and longitudes as the center for initialization
    center_lat, center_lon = ds['lat'].mean().item(), ds['lon'].mean().item()

    geo_map = GeoMap(center_lat, center_lon)
    f = initialize_plot()

    # Determine the resolution of the NetCDF data
    lat_resolution = np.mean(np.diff(ds['lat'].values)) * subsample_factor
    lon_resolution = np.mean(np.diff(ds['lon'].values)) * subsample_factor

    def handle_click(lat, lon, **kwargs):
        time_series_data = ds.sel(lat=lat, lon=lon)['dis'].values
        update_plot(f, {"Discharge": time_series_data}, f"Location ({lat}, {lon})")

    for lat in ds['lat'].values[::subsample_factor]:
        for lon in ds['lon'].values[::subsample_factor]:
            # Determine bounds for the rectangle
            bounds = [(lat - lat_resolution, lon - lon_resolution), (lat + lat_resolution, lon + lon_resolution)]

            def callback(*args, lat=lat, lon=lon, **kwargs):
                handle_click(lat, lon, **kwargs)

            geo_map.add_rectangle(bounds, callback)

    layout = widgets.HBox([geo_map.map, f])
    display(layout)
    return geo_map.map

def display_geospatial_ncs(datasets, subsample_factor=10):
    """
    Display grid points on a map for each location in the NetCDF and show associated data in a plot.

    Parameters:
    - datasets: List of xarray datasets.
    - subsample_factor: Factor by which to subsample the grid for visualization.
    """
    
    # Assuming all datasets have similar latitude and longitude ranges
    center_lat, center_lon = datasets[0]['lat'].mean().item(), datasets[0]['lon'].mean().item()
    
    geo_map = GeoMap(center_lat, center_lon)
    f = GeoMap.initialize_plot()

    for ds in datasets:
        # Determine the resolution of the NetCDF data
        lat_resolution = np.mean(np.diff(ds['lat'].values)) * subsample_factor
        lon_resolution = np.mean(np.diff(ds['lon'].values)) * subsample_factor

        def handle_click(lat, lon, ds, **kwargs):
            time_series_data = ds.sel(lat=lat, lon=lon)['dis'].values
            update_plots(f, {"Discharge": time_series_data}, f"Location ({lat}, {lon})")

        for lat in ds['lat'].values[::subsample_factor]:
            for lon in ds['lon'].values[::subsample_factor]:
                # Determine bounds for the rectangle
                bounds = [(lat - lat_resolution, lon - lon_resolution), (lat + lat_resolution, lon + lon_resolution)]

                def callback(*args, lat=lat, lon=lon, ds=ds, **kwargs):
                    handle_click(lat, lon, ds, **kwargs)

                geo_map.add_rectangle(bounds, callback)

    layout = widgets.HBox([geo_map.map, f])
    display(layout)
    return geo_map.map

def update_plot2(f, station_data, station_name, lat, lon):
    # Check existing trace names
    existing_traces = [trace.name for trace in f.data]
    
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

        trace_name = f"{station_name} - {var}"
        if data_exists and trace_name not in existing_traces:
            f.add_scatter(y=y_data, mode='lines+markers', name=trace_name)

    # Update the title to include the latitude and longitude
    f.layout.title = f"Time Series for Location: {lat:.3f}, {lon:.3f}"


def display_geospatial_nc2(datasets, subsample_factor=10):
    center_lat, center_lon = datasets[0][1]['lat'].mean().item(), datasets[0][1]['lon'].mean().item()
    geo_map = GeoMap(center_lat, center_lon)
    f = GeoMap.initialize_plot()  # Create only one plot for all datasets

    lat_resolution = np.mean(np.diff(datasets[0][1]['lat'].values)) * subsample_factor
    lon_resolution = np.mean(np.diff(datasets[0][1]['lon'].values)) * subsample_factor

    def handle_click(lat, lon, **kwargs):
        f.data = []
        for ds_label, ds in datasets:
            try:
                time_series_data = ds.sel(lat=lat, lon=lon)['dis'].values
                update_plot2(f, {"Discharge": time_series_data}, ds_label, lat, lon)
            except KeyError:
                # Handle the case where the dataset does not have data for the clicked coordinates
                pass

    for lat in datasets[0][1]['lat'].values[::subsample_factor]:
        for lon in datasets[0][1]['lon'].values[::subsample_factor]:
            bounds = [(lat - lat_resolution, lon - lon_resolution), (lat + lat_resolution, lon + lon_resolution)]

            def callback(*args, lat=lat, lon=lon, **kwargs):
                handle_click(lat, lon, **kwargs)

            geo_map.add_rectangle(bounds, callback)

    layout = widgets.HBox([geo_map.map, f])
    display(layout)
    return geo_map.map


