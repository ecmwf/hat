"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""

import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from ipywidgets import HBox, VBox, Layout, Label, Output, HTML
from ipyleaflet import Map, Marker, Rectangle, GeoJSON, Popup, AwesomeIcon, CircleMarker, LegendControl
import plotly.graph_objs as go
from hat.observations import read_station_metadata_file 
from hat.hydrostats import run_analysis
from hat.filters import filter_timeseries
from IPython.core.display import display
from IPython.display import clear_output
import time
import xarray as xr
from typing import Dict, Union
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from ipyleaflet_legend import Legend


class IPyLeaflet:
    """Visualization class for interactive map with IPyLeaflet and Plotly."""
    
    def __init__(self, center_lat: float, center_lon: float):
        # Initialize the map widget
        self.map = Map(center=[center_lat, center_lon], zoom=5, layout=Layout(width='500px', height='500px'))
              
        pd.set_option('display.max_colwidth', None)
        
        # Initialize the output widget for time series plots and dataframe display
        self.output_widget = Output()
        self.df_output = Output()
        
        # Create the title label for the properties table
        self.feature_properties_title = Label("", layout=Layout(font_weight='bold', margin='10px 0'))
        
        # Main layout: map and output widget on top, properties table at the bottom
        self.layout = VBox([HBox([self.map, self.output_widget, self.df_output])])
    
    def add_marker(self, lat: float, lon: float, on_click_callback):
        """Add a marker to the map."""
        marker = Marker(location=[lat, lon], draggable=False, opacity = 0.7)
        marker.on_click(on_click_callback)
        self.map.add_layer(marker)
    
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
    

class ThrottledClick:
    """Class to throttle click events."""
    def __init__(self, delay=1.0):
        self.delay = delay
        self.last_call = 0

    def should_process(self):
        current_time = time.time()
        if current_time - self.last_call > self.delay:
            self.last_call = current_time
            return True
        return False


class NotebookMap:
    """Main class for visualization in Jupyter Notebook."""

    def __init__(self, config: Dict, stations_metadata: str, observations: str, simulations: Dict, stats=None):
        self.config = config
        self.stations_metadata = read_station_metadata_file(
            fpath=stations_metadata,
            coord_names=config['station_coordinates'],
            epsg=config['station_epsg'],
            filters=config['station_filters']
        )
        self.station_index = config["station_id_column_name"]
        self.station_file_name = os.path.basename(stations_metadata)  # Derive the file name here
        self.stations_metadata[self.station_index] = self.stations_metadata[self.station_index].astype(str)
        self.observations = observations
        # Ensure simulations is always a dictionary
        
        self.simulations = simulations
        # Prepare sim_ds and obs_ds for statistics
        self.sim_ds = self.prepare_simulations_data()
        self.obs_ds = self.prepare_observations_data()
        self.common_id = self.find_common_station()
        print(self.common_id)
        self.stations_metadata = self.stations_metadata.loc[self.stations_metadata[self.station_index] == self.common_id]
        self.obs_ds =  self.obs_ds.sel(station = self.common_id)
        for sim, ds in  self.sim_ds.items():
            self.sim_ds[sim] = ds.sel(station=self.common_id)


        self.obs_ds =  self.obs_ds.sel(station = self.common_id)
        self.stats = stats
        self.threshold = 70 #to be opt
        self.statistics = None
        if self.stats:
            self.calculate_statistics()
        self.statistics_output = Output()

    def prepare_simulations_data(self):
        # If simulations is a string, assume it's a file path
        if isinstance(self.simulations, str):
            return xr.open_dataset(self.simulations)
        
        # If simulations is a dictionary, load data for each experiment
        elif isinstance(self.simulations, dict):
            datasets = {}
        for exp, path in self.simulations.items():
            # Expanding the tilde
            expanded_path = os.path.expanduser(path)
            
            if os.path.isfile(expanded_path):  # Check if it's a file
                ds = xr.open_dataset(expanded_path)
            elif os.path.isdir(expanded_path):  # Check if it's a directory
                # Handle the case when it's a directory; 
                # assume all .nc files in the directory need to be combined
                files = [f for f in os.listdir(expanded_path) if f.endswith('.nc')]
                ds = xr.open_mfdataset([os.path.join(expanded_path, f) for f in files], combine='by_coords')
            else:
                raise ValueError(f"Invalid path: {expanded_path}")
            datasets[exp] = ds
                
            return datasets
        
        else:
            raise TypeError("Invalid type for simulations. Expected str or dict.")
        

    def prepare_observations_data(self):
        file_extension = os.path.splitext(self.observations)[-1].lower()
        
        if file_extension == '.csv':
            obs_df = pd.read_csv(self.observations, parse_dates=["Timestamp"])
            obs_melted = obs_df.melt(id_vars="Timestamp", var_name="station", value_name="obsdis")
            
            # Convert the melted DataFrame to xarray Dataset
            obs_ds = obs_melted.set_index(["Timestamp", "station"]).to_xarray()
            obs_ds = obs_ds.rename({"Timestamp": "time"})

        elif file_extension == '.nc':
            obs_ds = xr.open_dataset(self.observations)
            
            # Check if the necessary attributes are present
            if 'obsdis' not in obs_ds or 'time' not in obs_ds.coords:
                raise ValueError("The NetCDF file does not have the expected variables or coordinates.")
            
            # Convert the 'station' coordinate to a multi-index of 'latitude' and 'longitude'
            lats = obs_ds['lat'].values
            lons = obs_ds['lon'].values
            multi_index = pd.MultiIndex.from_tuples(list(zip(lats, lons)), names=['lat', 'lon'])
            obs_ds['station'] = ('station', multi_index)
        
        else:
            raise ValueError("Unsupported file format for observations.")
        
        # Subset obs_ds based on sim_ds time values
        if isinstance(self.sim_ds, xr.Dataset):
            time_values = self.sim_ds['time'].values
        elif isinstance(self.sim_ds, dict):
            # Use the first dataset in the dictionary to determine time values
            first_dataset = next(iter(self.sim_ds.values()))
            time_values = first_dataset['time'].values
        else:
            raise ValueError("Unexpected type for self.sim_ds")

        obs_ds = obs_ds.sel(time=time_values)
        return obs_ds


    def find_common_station(self):
        ids = []
        ids += [list(self.obs_ds['station'].values)]
        print(self.obs_ds.station_id)
        ids += [list(ds['station'].values) for ds in self.sim_ds.values()]
        print(self.sim_ds)
        ids += [self.stations_metadata[self.station_index]]


        common_ids = None
        for id  in ids:
            print(id)

            if common_ids is None:
                common_ids = set(id)
            else:
                common_ids = set(id) & common_ids
            
            print(common_ids)

        return list(common_ids)      
    
    def calculate_statistics(self):
        statistics = {}
        if isinstance(self.sim_ds, xr.Dataset):
            # Single simulation dataset
            sim_ds_f, obs_ds_f = filter_timeseries(self.sim_ds, self.obs_ds, self.threshold)
            statistics["single"] = run_analysis(self.stats, sim_ds_f, obs_ds_f)
        elif isinstance(self.sim_ds, dict):
            # Dictionary of simulation datasets
            for exp, ds in self.sim_ds.items():
                sim_ds_f, obs_ds_f = filter_timeseries(ds, self.obs_ds, self.threshold)
                statistics[exp] = run_analysis(self.stats, sim_ds_f, obs_ds_f)
        self.statistics = statistics


    
   
    
    def display_dataframe_with_scroll(self, df, title=""):
        with self.geo_map.df_output:
            clear_output(wait=True)
            
            # Define styles directly within the HTML content
            table_style = """
            <style>
                .custom-table-container {
                    text-align: center;
                    margin: 0 auto;
                }
                .custom-table {
                    max-width: 1000px;
                    overflow-x: scroll;
                    overflow-y: auto;
                    display: inline-block;
                    border: 2px solid grey;
                    border-collapse: collapse;
                    width: 100%;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                }
                .custom-table th {
                    background-color: #f2f2f2;
                    text-align: center;
                }
                .custom-table tr:hover {
                    background-color: #f5f5f5;
                }
            </style>
            """
            table_html = df.to_html(classes='custom-table')
            content = f"{table_style}<div class='custom-table-container'><h3>{title}</h3>{table_html}</div>"
            
            display(HTML(content))


    # Define a callback to handle marker clicks
    def handle_click(self, station_id):
        # Use the throttler to determine if we should process the click
        if not self.throttler.should_process():
            return
        
        self.loading_label.value = "Loading..."  # Indicate that data is being loaded
        try:
            # Convert station ID to string for consistency
            station_id = str(station_id)
                    
            # Clear existing data from the plot
            self.f.data = []
                    
            # Convert ds_time to a list of string formatted dates for reindexing
            ds_time_str = [dt.isoformat() for dt in pd.to_datetime(self.ds_time)]

            # Loop over all datasets to plot them
            for ds, exp_name in zip(self.ds_list, self.simulations.keys()):
                if station_id in ds['station'].values:
                    ds_time_series_data = ds['simulation_timeseries'].sel(station=station_id).values
                    
                    # Filter out NaN values and their associated dates using the utility function
                    valid_dates_ds, valid_data_ds = filter_nan_values(ds_time_str, ds_time_series_data)
                    
                    self.f.add_trace(go.Scatter(x=valid_dates_ds, y=valid_data_ds, mode='lines', name=exp_name))
                else:
                    print(f"Station ID: {station_id} not found in dataset {exp_name}.") 

            # Time series from the obs
            if station_id in self.obs_ds['station'].values:
                obs_time_series = self.obs_ds['obsdis'].sel(station=station_id).values
                
                # Filter out NaN values and their associated dates using the utility function
                valid_dates_obs, valid_data_obs = filter_nan_values(ds_time_str, obs_time_series)
                
                self.f.add_trace(go.Scatter(x=valid_dates_obs, y=valid_data_obs, mode='lines', name='Obs. Data'))
            else:
                print(f"Station ID: {station_id} not found in obs_df. Columns are: {self.obs_ds.columns}")


            # Update the x-axis and y-axis properties
            self.f.layout.xaxis.title = 'Date'
            self.f.layout.xaxis.tickformat = '%d-%m-%Y'
            self.f.layout.yaxis.title = 'Discharge [m3/s]'    

            # Generate and display the statistics table for the clicked station
            with self.statistics_output:
                df_stats = self.generate_statistics_table(station_id)
                self.display_dataframe_with_scroll(df_stats, title="Statistics Overview")

            self.loading_label.value = ""  # Clear the loading message
            
        except Exception as e:
            print(f"Error encountered: {e}")
            self.loading_label.value = "Error encountered. Check the printed message."

        with self.geo_map.output_widget:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(self.f)

        
    def mapplot(self, colorby='kge', sim='exp1'):
        # Utilize the already prepared datasets
        if isinstance(self.sim_ds, dict):  # If there are multiple experiments
            self.ds_list = list(self.sim_ds.values())
        else:  # If there's only one dataset
            self.ds_list = [self.sim_ds]

        # Utilize the obs_ds to convert it to a dataframe (if needed elsewhere in the method)
        self.obs_df = self.obs_ds.to_dataframe().reset_index()

        self.statistics_output.clear_output()

        # Assume all datasets have the same coordinates for simplicity
        # Determine center coordinates using the first dataset in the list
        center_lat, center_lon = self.ds_list[0]['latitude'].mean().item(), self.ds_list[0]['longitude'].mean().item()

        # Create a GeoMap centered on the mean coordinates
        self.geo_map = IPyLeaflet(center_lat, center_lon)

        self.f = self.geo_map.initialize_plot()  # Initialize a plotly figure widget for the time series

        # Convert ds 'time' to datetime format for alignment with external_df
        self.ds_time = self.ds_list[0]['time'].values.astype('datetime64[D]')

        # Create a label to indicate loading
        self.loading_label = Label(value="")

        self.throttler = ThrottledClick()

        # Check if statistics were computed
        if self.statistics is None:
            raise ValueError("Statistics have not been computed. Run `calculate_statistics` first.")

        # Check if the chosen simulation exists in the statistics
        if sim not in self.statistics:
            raise ValueError(f"Simulation '{sim}' not found in computed statistics.")

        # Check if the chosen statistic (colorby) exists for the simulation
        if colorby not in self.statistics[sim].data_vars:
            raise ValueError(f"Statistic '{colorby}' not found in computed statistics for simulation '{sim}'.")
        
        # Retrieve the desired data
        stat_data = self.statistics[sim][colorby].values

        # Define a colormap (you can choose any other colormap that you like)
        colormap = plt.cm.viridis

        # Normalize the data for coloring
        norm = plt.Normalize(stat_data.min(), stat_data.max())

        # Create a GeoJSON structure from the stations_metadata DataFrame
        
        for _, row in self.stations_metadata.iterrows():
            lat, lon = row['StationLat'], row['StationLon']
            station_id = row['ObsID']
            
            # Get the index of the station in the statistics data
            station_indices = np.where(self.ds_list[0]['station'].values.astype(str) == str(station_id))[0]
            
            if len(station_indices) == 0:
                color = 'gray'
            else:
                if station_indices[0] >= len(stat_data) or np.isnan(stat_data[station_indices[0]]):
                    color = 'gray'
                else:
                    color = matplotlib.colors.rgb2hex(colormap(norm(stat_data[station_indices[0]])))
                    print(f"Station {station_id} has color {color} based on statistic value {stat_data[station_indices[0]]}")

            circle_marker = CircleMarker(location=(lat, lon), radius=5, color=color, fill_opacity=0.8)
            circle_marker.on_click(partial(self.handle_marker_click, row=row))
            self.geo_map.map.add_layer(circle_marker)

        # Add legend to your map
        
        # legend_dict = self.colormap_to_legend(stat_data, colormap)
        # print("Legend Dict:", legend_dict)
        # my_legend = Legend(legend_dict, name=colorby)
        # self.geo_map.map.add_control(my_legend)
        self.statistics_output = Output()

        # Initialize the layout only once
        # Create a new VBox for the plotly figure and the statistics output
        # plot_with_stats = VBox([self.f, self.statistics_output])

        # Modify the main layout to use the new VBox
    
        # self.layout = VBox([HBox([self.geo_map.map, self.geo_map.df_output]), self.loading_label])
        self.layout = VBox([HBox([self.geo_map.map, self.f]), self.geo_map.df_output])
        # self.layout = VBox([HBox([self.geo_map.map, self.f, self.statistics_output]), self.geo_map.df_output, self.loading_label])
        display(self.layout)


    def colormap_to_legend(self, stat_data, colormap, n_labels=5):
        """
        Convert a matplotlib colormap to a dictionary suitable for ipyleaflet-legend.
        
        Parameters:
        - colormap: A matplotlib colormap instance
        - n_labels: Number of labels/colors in the legend
        """
        values = np.linspace(0, 1, n_labels)
        colors = [matplotlib.colors.rgb2hex(colormap(value)) for value in values]
        labels = [f"{value:.2f}" for value in np.linspace(stat_data.min(), stat_data.max(), n_labels)]
        return dict(zip(labels, colors))


    def handle_marker_click(self, row, **kwargs):
        station_id = row['ObsID']
        station_name = row['StationName']

        self.handle_click(row['ObsID'])

        # Convert the station metadata to a DataFrame for display
        df = pd.DataFrame([row])

        title_plot = f"Time Series for Station ID: {station_id}, {station_name}"  
        title_table = f"Station property from metadata: {self.station_file_name}" 
        self.f.layout.title.text = title_plot  # Set title for the time series plot

        # Display the DataFrame in the df_output widget 
        self.display_dataframe_with_scroll(df, title=title_table) 
        

    def handle_geojson_click(self, feature, **kwargs):
        # Extract properties directly from the feature
        station_id = feature['properties']['station_id']
        self.handle_click(station_id)

        # Display metadata of the station
        df_station = self.stations_metadata[self.stations_metadata['ObsID'] == station_id]
        title_table = f"Station property from metadata: {self.station_file_name}"
        self.display_dataframe_with_scroll(df_station, title=title_table)
    
    
    def generate_statistics_table(self, station_id):
        # print(f"Generating statistics table for station: {station_id}")
        
        data = []
        # Loop through each simulation and get the statistics for the given station_id
    
        for exp_name, stats in self.statistics.items():
            print("Loop initiated for experiment:", exp_name)
            if str(station_id) in stats['station'].values:
                print("Available station IDs in stats:", stats['station'].values)
                row = [exp_name] + [stats[var].sel(station=station_id).values for var in stats.data_vars if var not in ['longitude', 'latitude']]
                data.append(row)
            print("Data after processing:", data)

        # Convert the data to a DataFrame for display
        columns = ['Exp. name'] + list(stats.data_vars.keys())
        columns.remove('longitude')
        columns.remove('latitude')
        statistics_df = pd.DataFrame(data, columns=columns)

        # Check if the dataframe has been generated correctly
        if statistics_df.empty:
            print(f"No statistics data found for station ID: {station_id}.")
            return pd.DataFrame()  # Return an empty dataframe

        return statistics_df


    def overlay_external_vector(self, vector_data: str, style=None, fill_color=None, line_color=None):
        """Add vector data to the map."""
        if style is None:
            style = {
                "stroke": True,
                "color": line_color if line_color else "#FF0000",
                "weight": 2,
                "opacity": 1,
                "fill": True,
                "fillColor": fill_color if fill_color else "#03f",
                "fillOpacity": 0.3,
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
            title = f"Feature property of the external vector: {vector_name}"

            properties = feature['properties']
            df = properties_to_dataframe(properties)

            # Display the DataFrame in the df_output widget using the modified method
            self.display_dataframe_with_scroll(df, title=title)

        # Bind the callback to the layer
        geojson_layer.on_click(on_feature_click)
        
        self.geo_map.map.add_layer(geojson_layer)

    @staticmethod
    def plot_station(self, station_data, station_id, lat, lon):
        existing_traces = [trace.name for trace in self.f.data]
        for var, y_data in station_data.items():
            data_exists = len(y_data) > 0

            trace_name = f"Station {station_id} - {var}"
            if data_exists and trace_name not in existing_traces:
                # Use the time data from the dataset
                x_data = station_data.get('time')
                
                if x_data is not None:
                    # Convert datetime64 array to native Python datetime objects
                    x_data = pd.to_datetime(x_data)

                self.f.add_scatter(x=x_data, y=y_data, mode='lines+markers', name=trace_name)



def properties_to_dataframe(properties: Dict) -> pd.DataFrame:
    """Convert feature properties to a DataFrame for display."""
    return pd.DataFrame([properties])


def filter_nan_values(dates, data_values):
    """
    Filters out NaN values and their associated dates.
    
    Parameters:
    - dates: List of dates.
    - data_values: List of data values corresponding to the dates.
    
    Returns:
    - valid_dates: List of dates without NaN values.
    - valid_data: List of non-NaN data values.
    """
    valid_dates = [date for date, val in zip(dates, data_values) if not np.isnan(val)]
    valid_data = [val for val in data_values if not np.isnan(val)]
    
    return valid_dates, valid_data