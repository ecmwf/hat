
"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""

import json
import os
import time
from typing import Dict
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import xarray as xr
from ipyleaflet import CircleMarker, Map, GeoJSON
from IPython.core.display import display
from IPython.display import clear_output
from ipywidgets import HTML, HBox, Label, Layout, Output, VBox, DatePicker
import matplotlib as mpl
from hat.observations import read_station_metadata_file


class InteractiveElements:
    def __init__(self, bounds, map_instance, statistics=None):
        self.map = self._initialize_map(bounds)
        self.loading_label = Label(value="")
        self.throttler = self._initialize_throttler()
        self.plotly_obj = PlotlyObject()
        self.table_obj = TableObject(self)
        self.statistics = statistics
        self.output_widget = Output()
        self.map_instance = map_instance
        self.statistics = map_instance.statistics


    def _initialize_map(self, bounds):
        """Initialize the map widget."""
        map_widget = Map(layout=Layout(width="100%", height="600px"))
        map_widget.fit_bounds(bounds)
        return map_widget
    
    def generate_html_legend(self, colormap, min_val, max_val):
        # Convert the colormap to a list of RGB values
        rgb_values = [mpl.colors.rgb2hex(colormap(i)) for i in np.linspace(0, 1, 256)]
        
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

    def _initialize_throttler(self, delay=1.0):
        """Initialize the throttler for click events."""
        return ThrottledClick(delay)


    def add_plotly_object(self, plotly_obj):
        """Add a PlotlyObject to the IPyLeaflet class."""
        self.plotly_obj = plotly_obj
    
    def add_table_object(self, table_obj):
        """Add a PlotlyObject to the IPyLeaflet class."""
        self.table_obj = table_obj

    def handle_marker_selection(self, feature, **kwargs):
        """Handle the selection of a marker on the map."""
        # Extract station_id from the selected feature
        station_id = feature["properties"]["station_id"]
        
        # Call the action handler with the extracted station_id
        self.handle_marker_action(station_id)
    
    def handle_marker_action(self, station_id):
        '''Define a callback to handle marker clicks and add the plot figure.'''
        
        # Check if we should process the click
        if not self.throttler.should_process():
            return

        self.loading_label.value = "Loading..."  # Indicate that data is being loaded
        station_id = str(station_id)  # Convert station ID to string for consistency
        print(f"Handling click for station: {station_id}")

        # Update the plot with simulation and observation data
        self.plotly_obj.update(station_id)

        # Generate and display the statistics table for the clicked station
        if self.statistics:
            self.table_obj.update(station_id)
            
            # Update the table in the layout
            children_list = list(self.map_instance.top_right_frame.children)
            
            # Find the index of the old table and replace it with the new table
            for i, child in enumerate(children_list):
                if isinstance(child, type(self.table_obj.stat_table_html)):
                    children_list[i] = self.table_obj.stat_table_html
                    break
            else:
                # If the old table wasn't found, append the new table
                children_list.append(self.table_obj.stat_table_html)
            
            self.map_instance.top_right_frame.children = tuple(children_list)

        self.loading_label.value = ""  # Clear the loading message

        with self.output_widget:
            clear_output(wait=True)  # Clear any previous plots or messages

        print("End of handle_marker_action")



class PlotlyObject:
    def __init__(self):
        self.figure = go.FigureWidget(
            layout=go.Layout(
                height=350,
                margin=dict(l=100),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                xaxis_title="Date",
                xaxis_tickformat="%d-%m-%Y",
                yaxis_title="Discharge [m3/s]"
            )
        )

    def update_simulation_data(self, station_id):
        for name, ds in self.sim_ds.items():
            if station_id in ds["station"].values:
                ds_time_series_data = ds["dis"].sel(station=station_id).values
                valid_dates_ds, valid_data_ds = filter_nan_values(
                    self.ds_time_str, ds_time_series_data
                )
                self._update_trace(valid_dates_ds, valid_data_ds, name)
            else:
                print(f"Station ID: {station_id} not found in dataset {name}.")

    def update_observation_data(self, station_id):
        if station_id in self.obs_ds["station"].values:
            obs_time_series = self.obs_ds["obsdis"].sel(station=station_id).values
            valid_dates_obs, valid_data_obs = filter_nan_values(
                self.ds_time_str, obs_time_series
            )
            self._update_trace(valid_dates_obs, valid_data_obs, "Obs. Data")
        else:
            print(f"Station ID: {station_id} not found in obs_df.")

    def _update_trace(self, x_data, y_data, name):
        trace_exists = any([trace.name == name for trace in self.figure.data])
        if trace_exists:
            for trace in self.figure.data:
                if trace.name == name:
                    trace.x = x_data
                    trace.y = y_data
        else:
            self.figure.add_trace(
                go.Scatter(x=x_data, y=y_data, mode="lines", name=name)
            )
        print(f"Updated plot with trace '{name}'. Total traces now: {len(self.figure.data)}")
    
    def update(self, station_id):
        self.update_simulation_data(station_id)
        self.update_observation_data(station_id)


class TableObject:
    def __init__(self, map_instance):
        self.map_instance = map_instance
        # Define the styles for the statistics table
        self.table_style = """
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
        self.stat_title_style = "style='font-size: 18px; font-weight: bold; text-align: center;'"

        

    def generate_statistics_table(self, station_id):
        data = []

        # Check if statistics is None or empty
        if not self.map_instance.statistics:
            print("No statistics data provided.")
            return pd.DataFrame()  # Return an empty dataframe

        # Loop through each simulation and get the statistics for the given station_id
        for exp_name, stats in self.map_instance.statistics.items():
            if station_id in stats["station"].values:
                row = [exp_name] + [
                    round(stats[var].sel(station=station_id).values.item(), 2)
                    for var in stats.data_vars
                    if var not in ["longitude", "latitude"]
                ]
                data.append(row)

        # Check if data has any items
        if not data:
            print(f"No statistics data found for station ID: {station_id}.")
            return pd.DataFrame()  # Return an empty dataframe

        # Convert the data to a DataFrame for display
        columns = ["Exp. name"] + list(stats.data_vars.keys())
        statistics_df = pd.DataFrame(data, columns=columns)

        # Round the numerical columns to 2 decimal places
        numerical_columns = [col for col in statistics_df.columns if col != "Exp. name"]
        statistics_df[numerical_columns] = statistics_df[numerical_columns].round(2)

        return statistics_df

    def display_dataframe_with_scroll(self, df, title=""):
        table_html = df.to_html(classes="custom-table")
        content = f"{self.table_style}<div class='custom-table-container'><h3 {self.stat_title_style}>{title}</h3>{table_html}</div>"
        return(HTML(content))

    def update(self, station_id):
        df_stats = self.generate_statistics_table(station_id)
        self.stat_table_html = self.display_dataframe_with_scroll(df_stats, title="Statistics Overview")
        print("Updated stat_table_html:", self.stat_table_html)




class InteractiveMap:
    def __init__(self, config, stations, observations, simulations, stats=None):
        self.config = config
        self.stations_metadata = read_station_metadata_file(
            fpath=stations,
            coord_names=config["station_coordinates"],
            epsg=config["station_epsg"],
            filters=config["station_filters"],
        )
        
        obs_var_name = config["obs_var_name"]

        # Use the external functions to prepare data
        self.sim_ds = prepare_simulations_data(simulations)
        self.obs_ds = prepare_observations_data(observations, self.sim_ds, obs_var_name)

        # Convert ds 'time' to datetime format for alignment with external_df
        self.ds_time = self.obs_ds["time"].values.astype("datetime64[D]")

        # set station index
        self.station_index = config["station_id_column_name"]
        self.stations_metadata[self.station_index] = self.stations_metadata[self.station_index].astype(str)

        # Retrieve statistics from the statistics netcdf input
        self.statistics = {}
        if stats:
            for name, path in stats.items():
                self.statistics[name] = xr.open_dataset(path)
        
        # Ensure the keys of self.statistics match the keys of self.sim_ds
        assert set(self.statistics.keys()) == set(self.sim_ds.keys()), "Mismatch between statistics and simulations keys."

        # find common station ids between metadata, observation and simulations
        self.common_id = self.find_common_station()

        print(f"Found {len(self.common_id)} common stations")
        self.stations_metadata = self.stations_metadata.loc[self.stations_metadata[self.station_index].isin(self.common_id)]
        self.obs_ds = self.obs_ds.sel(station=self.common_id)
        for sim, ds in self.sim_ds.items():
            self.sim_ds[sim] = ds.sel(station=self.common_id)

        # Pass the map bound to the interactive elements
        self.bounds = compute_bounds(self.stations_metadata, self.common_id, self.station_index, self.config["station_coordinates"])
        # self.interactive_elements = InteractiveElements(self.bounds)
        self.interactive_elements = InteractiveElements(self.bounds, self)

        # Pass the necessary data to the interactive elements
        self.interactive_elements.plotly_obj.sim_ds = self.sim_ds
        self.interactive_elements.plotly_obj.obs_ds = self.obs_ds
        self.interactive_elements.plotly_obj.ds_time_str = [dt.isoformat() for dt in pd.to_datetime(self.ds_time)]
        


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
    

    def _update_plot_dates(self, change):
        start_date = self.start_date_picker.value.strftime('%Y-%m-%d')
        end_date = self.end_date_picker.value.strftime('%Y-%m-%d')
        self.interactive_elements.plotly_obj.figure.update_layout(xaxis_range=[start_date, end_date])

    
    def mapplot(self, colorby="kge", sim=None, range=None, colormap=None):
    
        # Retrieve the statistics data of simulation choice/ by default
        stat_data = self.statistics[sim][colorby]
        
        # Normalize the data for coloring
        if range is None:
            min_val, max_val = stat_data.values.min(), stat_data.values.max()
        else:
            min_val, max_val = range[0], range[1]
                   
        norm = plt.Normalize(min_val, max_val)

        if colormap is None:
             colormap = plt.cm.get_cmap("viridis")

        def map_color(feature):
            station_id = feature['properties'][self.config["station_id_column_name"]]
            color = mpl.colors.rgb2hex(
                colormap(norm(stat_data.sel(station=station_id).values))
            )
            return {
                'color': 'black',
                'fillColor': color,
            }

        geo_data = GeoJSON(
            data=json.loads(self.stations_metadata.to_json()),
            style={'radius': 7, 'opacity':0.5, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.5},
            hover_style={'radius': 10, 'fillOpacity': 1},
            point_style={'radius': 5},
            style_callback=map_color,
        )


        self.legend_widget = self.interactive_elements.generate_html_legend(colormap, min_val, max_val)


        geo_data.on_click(self.interactive_elements.handle_marker_selection)
        self.interactive_elements.map.add_layer(geo_data)

        # Add date pickers for start and end dates
        self.start_date_picker = DatePicker(description='Start')
        self.end_date_picker = DatePicker(description='End')

        # Observe changes in the date pickers to update the plot
        self.start_date_picker.observe(self._update_plot_dates, names='value')
        self.end_date_picker.observe(self._update_plot_dates, names='value')

        #initialise stat table with first ID
        default_station_id = self.common_id[0]
        self.interactive_elements.table_obj.update(default_station_id)
        
        # Initialize layout elements
        self._initialize_layout_elements()

        # Display the main layout
        display(self.layout)

    
    def _initialize_layout_elements(self):
        # Title label
        self.title_label = Label(
            "Interactive Map Visualisation for Hydrological Model Performance", 
            layout=Layout(justify_content='center'),
            style={'font_weight': 'bold', 'font_size': '24px', 'font_family': 'Arial'}
        )

        # Date label
        self.date_label = Label("Please select the date to accurately change the date axis of the plot")

    
        # Layouts
        main_layout = Layout(justify_content='space-around', align_items='stretch', spacing='2px', width='1000px')
        left_layout = Layout(justify_content='space-around', align_items='center', spacing='2px', width='40%')
        right_layout = Layout(justify_content='center', align_items='center', spacing='2px', width='60%')

        # Date picker box
        self.date_picker_box = HBox([self.start_date_picker, self.end_date_picker])

        # Frames
        self.top_right_frame = VBox([self.interactive_elements.plotly_obj.figure, 
                                     self.date_label, self.date_picker_box, self.interactive_elements.table_obj.stat_table_html
                                     ], layout=right_layout)
        self.top_left_frame = VBox([self.interactive_elements.map, self.legend_widget], layout=left_layout)  # Assuming self.map is the map widget and self.legend_widget is the legend
        self.main_top_frame = HBox([self.top_left_frame, self.top_right_frame])

        # Main layout
        self.layout = VBox([self.title_label, self.interactive_elements.loading_label, self.main_top_frame], layout=main_layout)

        
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


def prepare_simulations_data(simulations):
    """process simulations raw datasets to a standard dataframe"""

    # If simulations is a dictionary, load data for each experiment
    sim_ds = {}
    for exp, path in simulations.items():
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


def prepare_observations_data(observations, sim_ds, obs_var_name):
    """process observation raw dataset to a standard dataframe"""
    file_extension = os.path.splitext(observations)[-1].lower()

    if file_extension == ".csv":
        obs_df = pd.read_csv(observations, parse_dates=["Timestamp"])
        obs_melted = obs_df.melt(
            id_vars="Timestamp", var_name="station", value_name=obs_var_name
        )

        # Convert the melted DataFrame to xarray Dataset
        obs_ds = obs_melted.set_index(["Timestamp", "station"]).to_xarray()
        obs_ds = obs_ds.rename({"Timestamp": "time"})

    elif file_extension == ".nc":
        obs_ds = xr.open_dataset(observations)

    else:
        raise ValueError("Unsupported file format for observations.")

    # Subset obs_ds based on sim_ds time values
    if isinstance(sim_ds, xr.Dataset):
        time_values = sim_ds["time"].values
    elif isinstance(sim_ds, dict):
        # Use the first dataset in the dictionary to determine time values
        first_dataset = next(iter(sim_ds.values()))
        time_values = first_dataset["time"].values
    else:
        raise ValueError("Unexpected type for sim_ds")

    obs_ds = obs_ds.sel(time=time_values)
    return obs_ds


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

def compute_bounds(stations_metadata, common_ids, station_index, coord_names):
    # Filter the metadata to only include stations with common IDs
    filtered_stations = stations_metadata[stations_metadata[station_index].isin(common_ids)]
    
    lon_column = coord_names[0]
    lat_column = coord_names[1]
    
    lons = filtered_stations[lon_column].values
    lats = filtered_stations[lat_column].values

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    return [(float(min_lat), float(min_lon)), (float(max_lat), float(max_lon))]

