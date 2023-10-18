"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""

import json
import os
import time
from functools import partial
from typing import Dict

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import xarray as xr
from ipyleaflet import (
    ImageOverlay,
    CircleMarker,
    Map,
    Marker,
)

from IPython.core.display import display
from IPython.display import clear_output
from ipywidgets import HTML, HBox, Label, Layout, Output, VBox, DatePicker
from shapely.geometry import Point

from hat.filters import filter_timeseries
from hat.hydrostats import run_analysis
from hat.observations import read_station_metadata_file


class IPyLeaflet:
    """Visualization class for interactive map with IPyLeaflet and Plotly."""

    def __init__(self, bounds):

        # Initialize the map widget with the calculated center
        self.map = Map(
            layout=Layout(width= "100%", height="600px")
        )

        # Fit the map to the provided bounds
        self.map.fit_bounds(bounds)

        pd.set_option("display.max_colwidth", None)

        # Initialize the output widget for time series plots and dataframe display
        self.output_widget = Output()
        self.df_output = Output()
        self.df_stats = Output()

        # Create the title label for the properties table
        self.feature_properties_title = Label(
            "", layout=Layout(font_weight="bold", margin="10px 0")
        )


    def add_marker(self, lat: float, lon: float, on_click_callback):
        """Add a marker to the map."""
        marker = Marker(location=[lat, lon], draggable=False, opacity=0.7)
        marker.on_click(on_click_callback)
        self.map.add_layer(marker)

    @staticmethod
    def initialize_plot():
        """Initialize a plotly figure widget."""
        f = go.FigureWidget(
            layout=go.Layout(
                height=350,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
        )
        return f 

    # def initialize_statistics_table(self):
    #     # Create a placeholder dataframe with no values but with the title, header, and "Exp. name"
    #     columns = ["Exp. name", "...", "..."]  # Add the required columns
    #     data = [["", "", ""]]  # Empty values
    #     df = pd.DataFrame(data, columns=columns)
    #     self.display_dataframe_with_scroll(df, self.df_stats, title="")

    # def initialize_station_property_table(self):
    #     # Create a placeholder dataframe for station property with blank header and one blank row
    #     columns = ["...", "...", "..."]  # Add the required columns
    #     data = [["", "", ""]]  # Empty values
    #     df = pd.DataFrame(data, columns=columns)
    #     self.display_dataframe_with_scroll(df, self.df_output, title="Station Property")     


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

    def __init__(
        self,
        config: Dict,
        stations_metadata: str,
        observations: str,
        simulations: Dict,
        stats=None,
    ):
        self.config = config
        self.stations_metadata = read_station_metadata_file(
            fpath=stations_metadata,
            coord_names=config["station_coordinates"],
            epsg=config["station_epsg"],
            filters=config["station_filters"],
        )
        
        self.station_file_name = os.path.basename(stations_metadata)  # station metadata file name here
        
        # set station index
        self.station_index = config["station_id_column_name"]
        self.stations_metadata[self.station_index] = self.stations_metadata[
            self.station_index
        ].astype(str)
          

        # Prepare sim_ds and obs_ds for statistics, simuation is a directory dict {<sim name>:<directory">}
        self.observations = observations
        self.obs_var_name = config["obs_var_name"]
        self.simulations = simulations 
        self.sim_ds = self.prepare_simulations_data()
        self.obs_ds = self.prepare_observations_data()

        # retrieve statistics from the statistics netcdf input
        self.statistics = {}
        if stats:
            for name, path in stats.items():
                self.statistics[name] = xr.open_dataset(path)

        assert self.statistics.keys() == self.sim_ds.keys()


        # find common station ids between metadata, observation and simulations
        self.common_id = self.find_common_station()
        print(f"Found {len(self.common_id)} common stations")
        self.stations_metadata = self.stations_metadata.loc[
            self.stations_metadata[self.station_index].isin(self.common_id)
        ]
        self.obs_ds = self.obs_ds.sel(station=self.common_id)
        for sim, ds in self.sim_ds.items():
            self.sim_ds[sim] = ds.sel(station=self.common_id)


        # Calculate the bounding extent to display map (min/max latitudes and longitudes)
        common_stations_df = self.stations_metadata[self.stations_metadata['station_id'].isin(self.common_id)]
        min_lat = common_stations_df['y'].min()
        max_lat = common_stations_df['y'].max()
        min_lon = common_stations_df['x'].min()
        max_lon = common_stations_df['x'].max()
        self.bounds = ((min_lat, min_lon), (max_lat, max_lon))

        
    def prepare_simulations_data(self):
        """process simulations raw datasets to a standard dataframe"""

        # If simulations is a dictionary, load data for each experiment
        sim_ds = {}
        for exp, path in self.simulations.items():
            # Expanding the tilde
            expanded_path = os.path.expanduser(path)

            if os.path.isfile(expanded_path):  # Check if it's a file
                ds = xr.open_dataset(expanded_path)
            elif os.path.isdir(expanded_path):  # Check if it's a directory
                # Handle the case when it's a directory;
                # assume all .nc files in the directory need to be combined
                files = [f for f in os.listdir(expanded_path) if f.endswith(".nc")]
                ds = xr.open_mfdataset(
                    [os.path.join(expanded_path, f) for f in files], combine="by_coords"
                )
            else:
                raise ValueError(f"Invalid path: {expanded_path}")
            sim_ds[exp] = ds

        return sim_ds

    def prepare_observations_data(self):
        """process observation raw dataset to a standard dataframe"""
        file_extension = os.path.splitext(self.observations)[-1].lower()

        if file_extension == ".csv":
            obs_df = pd.read_csv(self.observations, parse_dates=["Timestamp"])
            obs_melted = obs_df.melt(
                id_vars="Timestamp", var_name="station", value_name=self.obs_var_name
            )

            # Convert the melted DataFrame to xarray Dataset
            obs_ds = obs_melted.set_index(["Timestamp", "station"]).to_xarray()
            obs_ds = obs_ds.rename({"Timestamp": "time"})

        elif file_extension == ".nc":
            obs_ds = xr.open_dataset(self.observations)

        else:
            raise ValueError("Unsupported file format for observations.")

        # Subset obs_ds based on sim_ds time values
        if isinstance(self.sim_ds, xr.Dataset):
            time_values = self.sim_ds["time"].values
        elif isinstance(self.sim_ds, dict):
            # Use the first dataset in the dictionary to determine time values
            first_dataset = next(iter(self.sim_ds.values()))
            time_values = first_dataset["time"].values
        else:
            raise ValueError("Unexpected type for self.sim_ds")

        obs_ds = obs_ds.sel(time=time_values)
        return obs_ds

    def find_common_station(self):
        """find common station between observation and simulation and station metadata"""
        ids = []
        ids += [list(self.obs_ds["station"].values)]
        ids += [list(ds["station"].values) for ds in self.sim_ds.values()]
        ids += [self.stations_metadata[self.station_index]]
        if self.statistics:
            ids += [list(ds["station"].values) for ds in self.statistics.values()]

        common_ids = None
        for id in ids:
            if common_ids is None:
                common_ids = set(id)
            else:
                common_ids = set(id) & common_ids
        return list(common_ids)

    def calculate_statistics(self):
        """in progress: to calculate statistics using the run_analysis tools -- 
        takes a long time more recommended to load the statistic file"""

        statistics = {}
        # Dictionary of simulation datasets
        for exp, ds in self.sim_ds.items():
            sim_ds_f, obs_ds_f = filter_timeseries(
                ds.dis, self.obs_ds.obsdis, self.threshold
            )
            print(sim_ds_f)
            print(obs_ds_f)
            statistics[exp] = run_analysis(self.stats, sim_ds_f, obs_ds_f)
        return statistics

    def display_dataframe_with_scroll(self, df, output, title=""):
        """to display the dataframe as a html table with a horizontal scroll bar"""
        with output:
            clear_output(wait=True)

            # Define styles directly within the HTML content
            table_style = """
            <style>
                .custom-table-container {
                    text-align: center;
                    margin: 0;
                }
                .custom-table {
                    max-width: 1000px;
                    overflow-x: scroll;
                    overflow-y: auto;
                    display: inline-block;
                    border: 2px solid grey;
                    border-collapse: collapse;
                    width: 100%;
                    margin: 0
                }
                .custom-table th, .custom-table td {
                    border: 1px solid #9E9E9E;
                    padding: 8px;
                    border-right: 2px solid black;
                    border-left: 2px solid black;
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

            title_style = "style='font-size: 18px; font-weight: bold; text-align: center;'"
            table_html = df.to_html(classes="custom-table")
            content = f"{table_style}<div class='custom-table-container'><h3 {title_style}>{title}</h3>{table_html}</div>"

            display(HTML(content))

    
    def handle_click(self, station_id):
        '''Define a callback to handle marker clicks
        it adds the plot figure'''
        # Use the throttler to determine if we should process the click
        if not self.throttler.should_process():
            return

        self.loading_label.value = "Loading..."  # Indicate that data is being loaded

        # Convert station ID to string for consistency
        station_id = str(station_id)

        # Clear existing data from the plot
        self.f.data = []

        # Convert ds_time to a list of string formatted dates for reindexing
        ds_time_str = [dt.isoformat() for dt in pd.to_datetime(self.ds_time)]

        # Loop over all simulation datasets to plot them
        for name, ds in self.sim_ds.items():
            if station_id in ds["station"].values:
                ds_time_series_data = ds["dis"].sel(station=station_id).values

                # Filter out NaN values and their associated dates using the utility function
                valid_dates_ds, valid_data_ds = filter_nan_values(
                    ds_time_str, ds_time_series_data
                )

                self.f.add_trace(
                    go.Scatter(
                        x=valid_dates_ds, y=valid_data_ds, mode="lines", name="Simulation: "+name
                    )
                )
            else:
                print(f"Station ID: {station_id} not found in dataset {name}.")

        # Time series from the obs
        if station_id in self.obs_ds["station"].values:
            obs_time_series = self.obs_ds["obsdis"].sel(station=station_id).values

            # Filter out NaN values and their associated dates using the utility function
            valid_dates_obs, valid_data_obs = filter_nan_values(
                ds_time_str, obs_time_series
            )

            self.f.add_trace(
                go.Scatter(
                    x=valid_dates_obs, y=valid_data_obs, mode="lines", name="Obs. Data"
                )
            )
        else:
            print(
                f"Station ID: {station_id} not found in obs_df. Columns are: {self.obs_ds.columns}"
            )

        # Update the x-axis and y-axis properties
        self.f.layout.xaxis.title = "Date"
        self.f.layout.xaxis.tickformat = "%d-%m-%Y"
        self.f.layout.yaxis.title = "Discharge [m3/s]"


        # Generate and display the statistics table for the clicked station
        if self.statistics:
            df_stats = self.generate_statistics_table(station_id)
            self.display_dataframe_with_scroll(
                df_stats, self.geo_map.df_stats, title="Statistics Overview"
            )

        self.loading_label.value = ""  # Clear the loading message

        with self.geo_map.output_widget:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(self.f)

    def update_plot_based_on_date(self, change):
        """Update the x-axis range of the plot based on selected dates."""
        start_date = self.start_date_picker.value.strftime('%Y-%m-%d')
        end_date = self.end_date_picker.value.strftime('%Y-%m-%d')
        
        # Update the x-axis range of the plotly figure
        self.f.update_layout(xaxis_range=[start_date, end_date])
    
    def generate_html_legend(self, colormap, min_val, max_val):
        # Convert the colormap to a list of RGB values
        rgb_values = [matplotlib.colors.rgb2hex(colormap(i)) for i in np.linspace(0, 1, 256)]
        
        # Create a gradient style using the RGB values
        gradient_style = ', '.join(rgb_values)
        gradient_html = f"""
        <div style="
            background: linear-gradient(to right, {gradient_style});
            height: 30px;
            width: 200px;
            border: 1px solid black;
        "></div>
        """
        
        # Create labels
        labels_html = f"""
        <div style="display: flex; justify-content: space-between;">
            <span>Low: {min_val:.1f}</span>
            <span>High: {max_val:.1f}</span>
        </div>
        """
        
        # Combine gradient and labels
        legend_html = gradient_html + labels_html
        
        return HTML(legend_html)


    def mapplot(self, colorby="kge", sim = None, range = None):

        #If sim / experiement name is not provided, by default it takes the first one in the dictionary of simulation list
        if sim is None: 
            sim = list(self.simulations.keys())[0]
        
        # Create an instance of IPyLeaflet with the calculated bounds
        self.geo_map = IPyLeaflet(bounds=self.bounds)

        # Initialize a plotly figure widget for the time series
        self.f = (
            self.geo_map.initialize_plot()
        )  

        # self.geo_map.initialize_statistics_table()
        # self.geo_map.initialize_station_property_table()

        # Convert ds 'time' to datetime format for alignment with external_df
        self.ds_time = self.obs_ds["time"].values.astype("datetime64[D]")

        # Create a label to indicate loading
        self.loading_label = Label(value="")

        self.throttler = ThrottledClick()

        # Check if statistics were computed
        if self.statistics is None:
            raise ValueError(
                "Statistics have not been computed. Run `calculate_statistics` first."
            )

        # Check if the chosen simulation exists in the statistics
        if sim not in self.statistics:
            raise ValueError(f"Simulation '{sim}' not found in computed statistics.")

        # Check if the chosen statistic (colorby) exists for the simulation
        if colorby not in self.statistics[sim].data_vars:
            raise ValueError(
                f"Statistic '{colorby}' not found in computed statistics for simulation '{sim}'."
            )

        # Retrieve the statistics data of simulation choice/ by default
        stat_data = self.statistics[sim][colorby]

        # Normalize the data for coloring
        if range is None:
            min_val, max_val = stat_data.values.min(), stat_data.values.max()
        else:
            min_val, max_val = range[0], range[1]
                   
        norm = plt.Normalize(min_val, max_val)
        
        #create legend widget
        colormap = plt.cm.YlGnBu
        legend_widget = self.generate_html_legend(colormap, min_val, max_val)
    
        # Create marker from stations_metadata 
        for _, row in self.stations_metadata.iterrows():
            lat, lon = row["StationLat"], row["StationLon"]
            station_id = row[self.config["station_id_column_name"]]

            if station_id in list(stat_data.station):
                #TODO should be in IpyLeaflet
                color = matplotlib.colors.rgb2hex(
                    colormap(norm(stat_data.sel(station=station_id).values))
                )
            else:
                color = "gray"

            #TODO should be in IpyLeaflet
            circle_marker = CircleMarker(
                location=(lat, lon),
                radius=7,
                color="gray",
                fill_color=color,
                fill_opacity=0.8,
                weight = 1
            )
            circle_marker.on_click(partial(self.handle_marker_click, row=row))
            self.geo_map.map.add_layer(circle_marker)

        # Add date pickers for start and end dates
        self.start_date_picker = DatePicker(description='Start Date')
        self.end_date_picker = DatePicker(description='End Date')

        # Observe changes in the date pickers to update the plot
        self.start_date_picker.observe(self.update_plot_based_on_date, names='value')
        self.end_date_picker.observe(self.update_plot_based_on_date, names='value')

        title_label = Label(
                        "Interactive Map Visualisation for Hydrological Model Performance", 
                        layout=Layout(justify_content='center'),
                        style={'font_weight': 'bold', 'font_size': '24px', 'font_family': 'Arial'}
                        )
        date_label = Label("Please select the date to accurately change the date axis of the plot")

        main_layout = Layout(justify_content='space-around',align_items='stretch',spacing='2px', width= '1000px' )
        left_layout = Layout(justify_content='space-around', align_items='center',spacing='2px', width = '40%')
        right_layout = Layout(justify_content='center',align_items='center',spacing='2px', width = '60%')
        date_picker_box = HBox([self.start_date_picker, self.end_date_picker])
        top_right_frame = VBox([self.f, date_label, date_picker_box, self.geo_map.df_stats],layout=right_layout )
        top_left_frame = VBox([self.geo_map.map, legend_widget],layout=left_layout )
        main_top_frame = HBox([top_left_frame, top_right_frame])
        layout = VBox([title_label, self.loading_label, main_top_frame, self.geo_map.df_output],layout=main_layout)
        #TODO list all object e.g. geomap (all in a dict or list), 
        
        display(layout)

    
    def handle_marker_click(self, row, **kwargs): #TODO Interactive map class
        station_id = row[self.config["station_id_column_name"]]
        station_name = row["StationName"]

        self.handle_click(station_id)

        title_plot = f"<b>Time Series for the selected station:<br>ID: {station_id}, name: {station_name}"
        self.f.update_layout(
        title_text=title_plot,
        title_font=dict(size=18, family="Arial", color="black"),
        title_x=0.5,
        title_y=0.96  # Adjust as needed to position the title appropriately
    )

        # Display the DataFrame in the df_output widget
        df = pd.DataFrame(
            [row]
        )  # Convert the station metadata to a DataFrame for display
        title_table = f"Station property from metadata: {self.station_file_name}"
        self.display_dataframe_with_scroll(
            df, self.geo_map.df_output, title=title_table
        )

    def handle_geojson_click(self, feature, **kwargs):
        # Extract properties directly from the feature
        station_id = feature["properties"]["station_id"]
        self.handle_click(station_id)

        # Display metadata of the station
        df_station = self.stations_metadata[
            self.stations_metadata["ObsID"] == station_id
        ]
        title_table = f"Station property from metadata: {self.station_file_name}"
        self.display_dataframe_with_scroll(df_station, title=title_table)

    def generate_statistics_table(self, station_id):

        data = []
        # Loop through each simulation and get the statistics for the given station_id

        for exp_name, stats in self.statistics.items():

            if station_id in stats["station"].values:
                # print("Available station IDs in stats:", stats["station"].values)
                row = [exp_name] + [
                    round(stats[var].sel(station=station_id).values.item(), 2)
                    for var in stats.data_vars
                    if var not in ["longitude", "latitude"]
                ]

                data.append(row)

        # Convert the data to a DataFrame for display
        columns = ["Exp. name"] + list(stats.data_vars.keys())
        statistics_df = pd.DataFrame(data, columns=columns)

        # Round the numerical columns to 2 decimal places
        numerical_columns = [col for col in statistics_df.columns if col != "Exp. name"]
        statistics_df[numerical_columns] = statistics_df[numerical_columns].round(2)

        # Check if the dataframe has been generated correctly
        if statistics_df.empty:
            print(f"No statistics data found for station ID: {station_id}.")
            return pd.DataFrame()  # Return an empty dataframe

        return statistics_df

    # def overlay_external_vector(
    #     self, vector_data: str, style=None, fill_color=None, line_color=None
    # ):
    #     """Add vector data to the map."""
    #     if style is None:
    #         style = {
    #             "stroke": True,
    #             "color": line_color if line_color else "#FF0000",
    #             "weight": 2,
    #             "opacity": 1,
    #             "fill": True,
    #             "fillColor": fill_color if fill_color else "#03f",
    #             "fillOpacity": 0.3,
    #         }

    #     # Load the GeoJSON data from the file
    #     with open(vector_data, "r") as f:
    #         vector = json.load(f)

    #     # Create the GeoJSON layer
    #     geojson_layer = GeoJSON(data=vector, style=style)

    #     Update the title label
    #     vector_name = os.path.basename(vector_data)

    #     # Define the callback to handle feature clicks
    #     def on_feature_click(event, feature, **kwargs):
    #         title = f"Feature property of the external vector: {vector_name}"

    #         properties = feature["properties"]
    #         df = properties_to_dataframe(properties)


    #     # Bind the callback to the layer
    #     geojson_layer.on_click(on_feature_click)

    #     self.geo_map.map.add_layer(geojson_layer)

    @staticmethod
    def plot_station(self, station_data, station_id):
        existing_traces = [trace.name for trace in self.f.data]
        for var, y_data in station_data.items():
            data_exists = len(y_data) > 0

            trace_name = f"Station {station_id} - {var}"
            if data_exists and trace_name not in existing_traces:
                # Use the time data from the dataset
                x_data = station_data.get("time")

                if x_data is not None:
                    # Convert datetime64 array to native Python datetime objects
                    x_data = pd.to_datetime(x_data)

                self.f.add_scatter(
                    x=x_data, y=y_data, mode="lines+markers", name=trace_name
                )


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