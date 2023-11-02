import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.core.display import display
from IPython.display import clear_output
from ipywidgets import HTML, DatePicker, HBox, Label, Layout, Output, VBox


class ThrottledClick:
    """
    Initialize a click throttler with a given delay.
    to prevent user from swift multiple events clicking that results in crashing
    """

    def __init__(self, delay=1.0):
        self.delay = delay
        self.last_call = 0

    def should_process(self):
        """
        Determine if a click should be processed based on the delay.
        """
        current_time = time.time()
        if current_time - self.last_call > self.delay:
            self.last_call = current_time
            return True
        return False


class WidgetsManager:
    def __init__(self, widgets, index_column, loading_widget=None):
        self.widgets = widgets
        self.index_column = index_column
        self.throttler = self._initialize_throttler()
        self.loading_widget = loading_widget

    def _initialize_throttler(self, delay=1.0):
        """Initialize the throttler for click events."""
        return ThrottledClick(delay)

    def update(self, feature, **kwargs):
        """Handle the selection of a marker on the map."""

        # Check if we should process the click
        if not self.throttler.should_process():
            return

        if self.loading_widget is not None:
            self.loading_widget.value = (
                "Loading..."  # Indicate that data is being loaded
            )

        # Extract station_id from the selected feature
        metadata = feature["properties"]
        index = metadata[self.index_column]

        # update widgets
        for wgt in self.widgets.values():
            wgt.update(index, metadata)

        if self.loading_widget is not None:
            self.loading_widget.value = ""  # Clear the loading message

    def __getitem__(self, item):
        return self.widgets[item]


class Widget:
    def __init__(self, output):
        self.output = output

    def update(self, index, metadata):
        raise NotImplementedError


def filter_nan_values(dates, data_values):
    """Filters out NaN values and their associated dates."""
    valid_dates = [date for date, val in zip(dates, data_values) if not np.isnan(val)]
    valid_data = [val for val in data_values if not np.isnan(val)]

    return valid_dates, valid_data


class PlotlyWidget(Widget):
    """Plotly widget to display timeseries."""

    def __init__(self, datasets):
        self.datasets = datasets
        # initial_title = {
        #     'text': "Click on your desired station location",
        #     'y':0.9,
        #     'x':0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # }
        self.figure = go.FigureWidget(
            layout=go.Layout(
                # title = initial_title,
                height=350,
                margin=dict(l=120),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                xaxis_title="Date",
                xaxis_tickformat="%d-%m-%Y",
                yaxis_title="Discharge [m3/s]",
            )
        )
        ds_time = datasets["obs"]["time"].values.astype("datetime64[D]")
        self.ds_time_str = [dt.isoformat() for dt in pd.to_datetime(ds_time)]

        # Add date pickers for start and end dates
        self.start_date_picker = DatePicker(description="Start")
        self.end_date_picker = DatePicker(description="End")

        # Observe changes in the date pickers to update the plot
        self.start_date_picker.observe(self._update_plot_dates, names="value")
        self.end_date_picker.observe(self._update_plot_dates, names="value")

        # Date picker title
        date_label = Label(
            "Please select the date to accurately change the date axis of the plot"
        )
        date_picker_box = HBox([self.start_date_picker, self.end_date_picker])

        layout = Layout(justify_content="center", align_items="center")
        output = VBox([self.figure, date_label, date_picker_box], layout=layout)
        super().__init__(output)

    def _update_plot_dates(self, change):
        start_date = self.start_date_picker.value.strftime("%Y-%m-%d")
        end_date = self.end_date_picker.value.strftime("%Y-%m-%d")
        self.figure.update_layout(xaxis_range=[start_date, end_date])

    def update_data(self, station_id):
        """Update the simulation data for the given station ID."""
        for name, ds in self.datasets.items():
            if station_id in ds["station"].values:
                ds_time_series_data = ds.sel(station=station_id).values
                valid_dates_ds, valid_data_ds = filter_nan_values(
                    self.ds_time_str, ds_time_series_data
                )
                self._update_trace(valid_dates_ds, valid_data_ds, name)
            else:
                print(f"Station ID: {station_id} not found in dataset {name}.")

    def _update_trace(self, x_data, y_data, name):
        """Update or add a trace to the Plotly figure."""
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

    def update_title(self, metadata):
        station_id = metadata["station_id"]
        station_name = metadata["StationName"]
        updated_title = (
            f"<b>Selected station:<br>ID: {station_id}, name: {station_name}</b> "
        )
        self.figure.update_layout(
            title={
                "text": updated_title,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"color": "black", "size": 16},
            }
        )

    def update(self, index, metadata):
        """Update the overall plot with new data for the given station ID."""
        # self.update_title(metadata)
        self.update_data(index)


class HTMLTableWidget(Widget):
    def __init__(self, dataframe, title):
        """
        Initialize the table object for displaying statistics and station properties.
        """
        self.dataframe = dataframe
        self.title = title
        super().__init__(Output())

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
        self.stat_title_style = (
            "style='font-size: 18px; font-weight: bold; text-align: center;'"
        )
        # Initialize the stat_table_html and station_table_html with empty tables
        empty_df = pd.DataFrame()
        self.display_dataframe_with_scroll(empty_df, title=self.title)

    def display_dataframe_with_scroll(self, df, title=""):
        """Display a DataFrame with a scrollable view."""
        table_html = df.to_html(classes="custom-table")
        content = f"{self.table_style}<div class='custom-table-container'><h3 {self.stat_title_style}>{title}</h3>{table_html}</div>"  # noqa: E501
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(HTML(content))

    def update(self, index, metadata):
        dataframe = self.extract_dataframe(index)
        self.display_dataframe_with_scroll(dataframe, title=self.title)


class DataFrameWidget(Widget):
    def __init__(self, dataframe, title):
        """
        Initialize the table object for displaying statistics and station properties.
        """
        self.dataframe = dataframe
        self.title = title
        super().__init__(output=Output(title=self.title))

        # Initialize the stat_table_html and station_table_html with empty tables
        empty_df = pd.DataFrame()
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(empty_df)

    def update(self, index, metadata):
        dataframe = self.extract_dataframe(index)
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(dataframe)


class MetaDataWidget(HTMLTableWidget):
    def __init__(self, dataframe, station_index):
        title = "Station Metadata"
        self.station_index = station_index
        super().__init__(dataframe, title)

    def extract_dataframe(self, station_id):
        """Generate a station property table for the given station ID."""
        stations_df = self.dataframe
        selected_station_df = stations_df[stations_df[self.station_index] == station_id]

        return selected_station_df


class StatisticsWidget(HTMLTableWidget):
    def __init__(self, dataframe):
        title = "Model Performance Statistics Overview"
        super().__init__(dataframe, title)

    def extract_dataframe(self, station_id):
        """Generate a statistics table for the given station ID."""
        data = []

        # Check if statistics is None or empty
        if not self.dataframe:
            print("No statistics data provided.")
            return pd.DataFrame()  # Return an empty dataframe

        # Loop through each simulation and get the statistics for the given station_id
        for exp_name, stats in self.dataframe.items():
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
