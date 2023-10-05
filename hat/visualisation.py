"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""
import json
import plotly.graph_objs as go
import numpy as np
import os
from ipywidgets import HTML, HBox, VBox, Output, Layout, Label, GridBox
from ipyleaflet import Map, Marker, GeoJSON, Rectangle
import pandas as pd
from IPython.display import display
from IPython.display import display, clear_output


class GeoMap:
    def __init__(self, center_lat, center_lon):
        # Initialize the map widget
        self.map = Map(center=[center_lat, center_lon], zoom=5, layout=Layout(width='50%', height='500px'))
        
        # Adjust DataFrame display settings
        display(HTML("""
        <style>
            .output {
                align-items: center;
                text-align: left;
            }
        </style>
        """))
        
        pd.set_option('display.max_colwidth', None)
        
        # Initialize the output widget for time series plots and dataframe display
        self.output_widget = Output()
        self.df_output = Output()
        
        # Create the title label for the properties table
        self.feature_properties_title = Label("", layout=Layout(font_weight='bold', margin='10px 0'))
        
        # Main layout: map and output widget on top, properties table at the bottom
        self.layout = VBox([HBox([self.map, self.output_widget, self.df_output])])

   
    def add_marker(self, lat, lon, on_click_callback):
        """Add a marker to the map."""
        marker = Marker(location=[lat, lon], draggable=False, opacity = 0.4)
        marker.on_click(on_click_callback)
        self.map.add_layer(marker)
    
    def add_rectangle(self, bounds, on_click_callback):
        """Add a rectangle to the map."""
        rectangle = Rectangle(bounds=bounds, color="blue", fill_opacity=0.01, opacity=0.5, weight=1)
        rectangle.on_click(on_click_callback)
        self.map.add_layer(rectangle)
    
    def display(self):
        display(self.vbox) 

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
    

    def add_vector_to_map(self, vector_data, style=None, fill_color=None, line_color=None):
        """Add vector data to the map."""
        if style is None:
            style = {
                "stroke": True,
                "color": line_color if line_color else "#FF0000",
                "weight": 2,
                "opacity": 1,
                "fill": True,
                "fillColor": fill_color if fill_color else "#03f",
                "fillOpacity": 0.7,
            }
        
        # Load the GeoJSON data from the file
        with open(vector_data, 'r') as f:
            vector = json.load(f)

        # Create the GeoJSON layer
        geojson_layer = GeoJSON(data=vector, style=style)
    
        # Update the title label
        vector_name = os.path.basename(vector_data)
        
       
        # Define the callback to handle feature clicks
        def on_feature_click(event, feature, **kwargs):
            title = f"Feature property of the {vector_name}"

            properties = feature['properties']
            df = properties_to_dataframe(properties)

            styles = {
                "selector": "th, td",
                "props": [("text-align", "left")]
            }

            # Clear the previous output and display the new DataFrame
            with self.df_output:
                clear_output(wait=True)
                styled_df = df.style.set_caption(title).set_table_styles([styles])
                display(styled_df)
    

        # Bind the callback to the layer
        geojson_layer.on_click(on_feature_click)
        
        self.map.add_layer(geojson_layer)
        

def properties_to_dataframe(properties):
    return pd.DataFrame([properties])


def display_geospatial_nc_stations(ds, external_data_path):
    # 1. Determine center coordinates
    center_lat, center_lon = ds['latitude'].mean().item(), ds['longitude'].mean().item()
    
    # Create a GeoMap centered on the mean coordinates
    geo_map = GeoMap(center_lat, center_lon)
    
    f = GeoMap.initialize_plot()  # Initialize a plotly figure widget for the time series

    # Read the external time series data
    external_df = pd.read_csv(external_data_path, parse_dates=["Timestamp"])
    external_df.set_index('Timestamp', inplace=True)
    
    # Convert ds 'time' to datetime format for alignment with external_df
    ds_time = ds['time'].values.astype('datetime64[D]')
    
    # Define a callback to handle marker clicks
    def handle_click(station_id, **kwargs):
        f.data = []  # Clear existing data from the plot
        
        # Time series from the ds dataset
        ds_time_series_data = ds['simulation_timeseries'].sel(station=station_id).values
        update_plot_station(f, {"Simulation": ds_time_series_data}, station_id, center_lat, center_lon)
        
        # Time series from the external dataframe
        if station_id in external_df.columns:
            ext_time_series_data = external_df[station_id].reindex(ds_time).values
            update_plot_station(f, {"External Data": ext_time_series_data}, station_id, center_lat, center_lon)
    
    # 2. Add markers for each station
    for station in ds['station'].values:
        lat, lon = ds['latitude'].sel(station=station).item(), ds['longitude'].sel(station=station).item()
        
        # Define a callback for this specific station
        def callback(*args, station_id=station, **kwargs):
            handle_click(station_id, **kwargs)
        
        # Add the marker to the map
        geo_map.add_marker(lat, lon, callback)
    
    # Display the map and time series side-by-side
    layout = VBox([HBox([geo_map.map, f]), geo_map.df_output])
    display(layout)
    return geo_map



def update_plot_station(f, station_data, station_id, lat, lon):
    existing_traces = [trace.name for trace in f.data]
    for var, y_data in station_data.items():
        data_exists = len(y_data) > 0

        trace_name = f"Station {station_id} - {var}"
        if data_exists and trace_name not in existing_traces:
            f.add_scatter(y=y_data, mode='lines+markers', name=trace_name)

    # Update the title to include the station ID
    f.layout.title = f"Time Series for Station ID: {station_id}"