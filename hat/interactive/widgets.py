import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.display import clear_output, display
from ipywidgets import HTML, DatePicker, HBox, Label, Layout, Output, VBox


class ThrottledClick:
    """
    Initialize a click throttler with a given delay.

    Parameters
    ----------
    delay : float, optional
        The delay in seconds between clicks. Defaults to 1.0.

    Notes
    -----
    This class is used to prevent users from rapidly clicking a button or widget
    multiple times, which can cause the application to crash or behave unexpectedly.

    Examples
    --------
    >>> click_throttler = ThrottledClick(delay=0.5)
    >>> if click_throttler.should_process():
    ...     # do something
    """

    def __init__(self, delay=1.0):
        self.delay = delay
        self.last_call = 0

    def should_process(self):
        """
        Determine if a click should be processed based on the delay.

        Returns
        -------
        bool
            True if the click should be processed, False otherwise.

        Notes
        -----
        This method should be called before processing a click event. If the
        time since the last click is greater than the delay, the method returns
        True and updates the last_call attribute. Otherwise, it returns False.
        """
        current_time = time.time()
        if current_time - self.last_call > self.delay:
            self.last_call = current_time
            return True
        return False


class WidgetsManager:
    """
    A class for managing a collection of widgets and updating following
    a user interaction, providing an index.

    Parameters
    ----------
    widgets : dict
        A dictionary of widgets to manage.
    index_column : str
        The name of the column containing the index used to update the widgets.
    loading_widget : optional
        A widget to display a loading message while data is being loaded.

    Attributes
    ----------
    widgets : dict
        A dictionary of widgets being managed.
    index_column : str
        The name of the column containing the index used to update the widgets.
    throttler : ThrottledClick
        A throttler for click events.
    loading_widget : optional
        A widget to display a loading message while data is being loaded.
    """

    def __init__(self, widgets, index_column, loading_widget=None):
        self.widgets = widgets
        self.index_column = index_column
        self.throttler = ThrottledClick()
        self.loading_widget = loading_widget

    def __getitem__(self, item):
        return self.widgets[item]

    def update(self, feature, **kwargs):
        """
        Handle the selection of a marker on the map.

        Parameters
        ----------
        feature : dict
            A dictionary containing information about the selected feature.
        **kwargs : dict
            Additional keyword arguments to pass to the widgets update method.
        """

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
            wgt.update(index, metadata, **kwargs)

        if self.loading_widget is not None:
            self.loading_widget.value = ""  # Clear the loading message


class Widget:
    """
    A base class for interactive widgets.

    Parameters
    ----------
    output : Output
        The ipywidget compatible object to display the widget's content.

    Attributes
    ----------
    output : Output
        The ipywidget compatible object to display the widget's content.

    Methods
    -------
    update(index, metadata, **kwargs)
        Update the widget's content based on the given index and metadata.
    """

    def __init__(self, output):
        self.output = output

    def update(self, index, *args, **kwargs):
        raise NotImplementedError


def _filter_nan_values(dates, data_values):
    """
    Filters out NaN values and their associated dates.
    """
    assert len(dates) == len(
        data_values
    ), "Dates and data values must be the same length."
    valid_dates = [date for date, val in zip(dates, data_values) if not np.isnan(val)]
    valid_data = [val for val in data_values if not np.isnan(val)]

    return valid_dates, valid_data


class PlotlyWidget(Widget):
    """
    A widget to display timeseries data using Plotly.

    Parameters
    ----------
    datasets : dict
        A dictionary containing the xarray timeseries datasets to be displayed.

    Attributes
    ----------
    datasets : dict
        A dictionary containing the xarray timeseries datasets to be displayed.
    figure : plotly.graph_objs._figurewidget.FigureWidget
        The Plotly figure widget.
    ds_time_str : list
        A list of strings representing the dates in the timeseries data.
    start_date_picker : DatePicker
        The date picker widget for selecting the start date.
    end_date_picker : DatePicker
        The date picker widget for selecting the end date.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.figure = go.FigureWidget(
            layout=go.Layout(
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

        self.start_date_picker = DatePicker(description="Start")
        self.end_date_picker = DatePicker(description="End")

        self.start_date_picker.observe(self._update_plot_dates, names="value")
        self.end_date_picker.observe(self._update_plot_dates, names="value")

        date_label = Label(
            "Please select the date to accurately change the date axis of the plot"
        )
        date_picker_box = HBox([self.start_date_picker, self.end_date_picker])

        layout = Layout(justify_content="center", align_items="center")
        output = VBox([self.figure, date_label, date_picker_box], layout=layout)
        super().__init__(output)

    def _update_plot_dates(self):
        """
        Updates the plot with the selected start and end dates.
        """
        start_date = self.start_date_picker.value.strftime("%Y-%m-%d")
        end_date = self.end_date_picker.value.strftime("%Y-%m-%d")
        self.figure.update_layout(xaxis_range=[start_date, end_date])

    def _update_data(self, station_id):
        """
        Updates the simulation data for the given station ID.
        """
        for name, ds in self.datasets.items():
            if station_id in ds["station"].values:
                ds_time_series_data = ds.sel(station=station_id).values
                valid_dates_ds, valid_data_ds = _filter_nan_values(
                    self.ds_time_str, ds_time_series_data
                )
                self._update_trace(valid_dates_ds, valid_data_ds, name)
            else:
                print(f"Station ID: {station_id} not found in dataset {name}.")
                return False
        return True

    def _update_trace(self, x_data, y_data, name):
        """
        Updates the plot trace for the given name with the given x and y data.
        """
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

    def _update_title(self, metadata):
        """
        Updates the plot title following the point metadata.
        """
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

    def update(self, index, *args, **kwargs):
        """
        Updates the overall plot with new data for the given index.

        Parameters
        ----------
        index : str
            The ID of the station to update the data for.
        metadata : dict
            A dictionary containing the metadata for the selected station.
        """
        return self._update_data(index)


class HTMLTableWidget(Widget):
    """
    A widget to display a pandas dataframe with the HTML format.

    Parameters
    ----------
    title : str
        The title of the table.
    """

    def __init__(self, title):
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
        self._display_dataframe_with_scroll(empty_df, title=self.title)

    def _display_dataframe_with_scroll(self, df, title=""):
        table_html = df.to_html(classes="custom-table")
        content = f"{self.table_style}<div class='custom-table-container'><h3 {self.stat_title_style}>{title}</h3>{table_html}</div>"  # noqa: E501
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(HTML(content))

    def update(self, index, *args, **kwargs):
        """
        Update the table with the dataframe as the given index.

        Parameters
        ----------
        index : int
            The index of the data to be displayed.
        metadata : dict
            The metadata associated with the data index.
        """
        dataframe = self._extract_dataframe(index)
        self._display_dataframe_with_scroll(dataframe, title=self.title)
        if dataframe.empty:
            return False
        return True


class DataFrameWidget(Widget):
    """
    A widget to display a pandas dataframe with the default pandas display
    style.

    Parameters
    ----------
    title : str
        The title of the table.
    """

    def __init__(self, title):
        self.title = title
        super().__init__(output=Output(title=self.title))

        # Initialize the stat_table_html and station_table_html with empty tables
        empty_df = pd.DataFrame()
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(empty_df)

    def update(self, index, *args, **kwargs):
        """
        Update the table with the dataframe as the given index.

        Parameters
        ----------
        index : int
            The index of the data to be displayed.
        metadata : dict
            The metadata associated with the data index.
        """
        dataframe = self._extract_dataframe(index)
        with self.output:
            clear_output(wait=True)  # Clear any previous plots or messages
            display(dataframe)
        if dataframe.empty:
            return False
        return True

    def _extract_dataframe(self, index):
        """
        Virtual method to return the object dataframe at the index.
        """
        raise NotImplementedError


class MetaDataWidget(HTMLTableWidget):
    """
    An extension of the HTMLTableWidget class to display a station metadata.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas dataframe to be displayed in the table.
    station_index : str
        Column name of the station index.
    """

    def __init__(self, dataframe, station_index):
        title = "Station Metadata"
        self.dataframe = dataframe
        self.station_index = station_index
        super().__init__(title)

    def _extract_dataframe(self, station_id):
        stations_df = self.dataframe
        selected_station_df = stations_df[stations_df[self.station_index] == station_id]
        return selected_station_df


class StatisticsWidget(HTMLTableWidget):
    """
    An extension of the HTMLTableWidget to display statistics at stations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A pandas dataframe to be displayed in the table.
    station_index : str
        Column name of the station index.
    """

    def __init__(self, statistics):
        title = "Model Performance Statistics Overview"
        self.statistics = statistics
        for stat in self.statistics.values():
            assert (
                "station" in stat.dims
            ), 'Dimension "station" not found in statistics datasets.'  # noqa: E501
        super().__init__(title)

    def _extract_dataframe(self, station_id):
        """Generate a statistics table for the given station ID."""
        data = []

        # Check if statistics is None or empty
        if not self.statistics:
            print("No statistics data provided.")
            return pd.DataFrame()  # Return an empty dataframe

        # Loop through each simulation and get the statistics for the given station_id
        for exp_name, stats in self.statistics.items():
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
