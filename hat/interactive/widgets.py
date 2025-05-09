import time
import datetime
import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.display import clear_output, display
from ipywidgets import HTML, Button, DatePicker, HBox, Label, Layout, Output, Text, VBox

import sys

sys.path.append("../../../floods-html")
from floods_html import floods_html


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

    def index(self, feature, **kwargs):
        metadata = feature["properties"]
        index = metadata[self.index_column]
        return index

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
            self.loading_widget.value = "Loading..."  # Indicate that data is being loaded

        # Extract station_id from the selected feature
        if isinstance(feature, dict):
            metadata = feature["properties"]
            index = metadata[self.index_column]
        else:
            index = feature

        # update widgets
        for wgt in self.widgets.values():
            wgt.update(index, **kwargs)

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
    assert len(dates) == len(data_values), "Dates and data values must be the same length."
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
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
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

        date_label = Label("Please select the date to accurately change the date axis of the plot")
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
                valid_dates_ds, valid_data_ds = _filter_nan_values(self.ds_time_str, ds_time_series_data)
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
            self.figure.add_trace(go.Scatter(x=x_data, y=y_data, mode="lines", name=name))

    def _update_title(self, metadata):
        """
        Updates the plot title following the point metadata.
        """
        station_id = metadata["station_id"]
        station_name = metadata["StationName"]
        updated_title = f"<b>Selected station:<br>ID: {station_id}, name: {station_name}</b> "  # noqa: E501
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
        self.stat_title_style = "style='font-size: 18px; font-weight: bold; text-align: center;'"
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
            assert "station" in stat.dims, 'Dimension "station" not found in statistics datasets.'  # noqa: E501
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


def crps(x, y):
    """
    Computes CRPS from x using y as reference,
    first x dimension must be ensembles, next dimensions can be arbitrary
    x: ensemble data (n_ens, n_points)
    y: observation/analysis data (n_points)
    returns: crps (n_points)
    REFERENCE
      Hersbach, 2000: Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems.
      Weather and Forecasting 15: 559-570.
    """

    # first sort ensemble
    x.sort(axis=0)

    # construct alpha and beta, size nens+1
    n_ens = x.shape[0]
    shape = (n_ens + 1,) + x.shape[1:]
    alpha = np.zeros(shape)
    beta = np.zeros(shape)

    # x[i+1]-x[i] and x[i]-y[i] arrays
    diffxy = x - y.reshape(1, *(y.shape))
    diffxx = x[1:] - x[:-1]  # x[i+1]-x[i], size ens-1

    # if i == 0
    alpha[0] = 0
    beta[0] = np.fmax(diffxy[0], 0)  # x(0)-y
    # if i == n_ens
    alpha[-1] = np.fmax(-diffxy[-1], 0)  # y-x(n)
    beta[-1] = 0
    # else
    alpha[1:-1] = np.fmin(diffxx, np.fmax(-diffxy[:-1], 0))  # x(i+1)-x(i) or y-x(i) or 0
    beta[1:-1] = np.fmin(diffxx, np.fmax(diffxy[1:], 0))  # 0 or x(i+1)-y or x(i+1)-x(i)

    # compute crps
    p_exp = (np.arange(n_ens + 1) / float(n_ens)).reshape(n_ens + 1, *([1] * y.ndim))
    crps = np.sum(alpha * (p_exp**2) + beta * ((1 - p_exp) ** 2), axis=0)

    return crps


class PPForecastPlotWidget(Widget):
    def __init__(self, config, stations_metadata, station_index):
        self.config = config
        self.stations_metadata = stations_metadata
        self.station_index = station_index

        self.date = config["date"]
        self.assets = config["assets"]
        obs_dir = config["observations"]
        print(config["source"])
        os.sys.path.append(config["source"])
        import plot_site_forecast as psf
        import pp_helper_functions as phf

        self.psf = psf
        self.phf = phf
        # from pp_helper_functions import (
        #     collate_plotting_data,
        #     extract_timestep,
        #     open_mod_file,
        #     open_record,
        #     prepare_obs,
        #     select_obs_quantiles,
        # )

        # observations
        self.observations = {}
        for ts in ["06", "24"]:
            obs_file = os.path.join(obs_dir, f"Qobs_nrt{ts}.csv")
            print(obs_file)
            obs_data = pd.read_csv(obs_file, index_col="Timestamp")
            obs_data.index = pd.to_datetime(obs_data.index)
            self.observations[ts] = obs_data

        # store last index
        self.index = None

        # Forecast date selector
        self.date_input = Text(
            description="Forecast Date:",
            placeholder="YYYYMMDDHH",
            value=self.date.strftime("%Y%m%d%H"),
            disabled=False,
            style={"description_width": "initial"},
            # layout=Layout(width='500px')
        )
        date_button = Button(description="Update", layout=Layout(width="100px"))
        date_button.on_click(self._update_date)

        # Add the date selector to the map
        self.date_widget = HBox([self.date_input, date_button])

        self.figure = Output()
        output = VBox(
            [self.date_widget, self.figure],
            layout=Layout(width="1000px", align_items="center"),
        )

        lower_crps_file = os.path.join(self.config["climatology"], "crps_obs_fcst.csv")
        upper_crps_file = os.path.join(self.config["climatology"], "crps_obs_mcp.csv")
        self.crps_lower = pd.read_csv(lower_crps_file, index_col="Unnamed: 0")
        self.crps_upper = pd.read_csv(upper_crps_file, index_col="Unnamed: 0")

        super().__init__(output)

    def update(self, index=None, *args, **kwargs):
        if index is None:
            index = self.index
        else:
            self.index = index

        # station metadata
        metadata = self.stations_metadata.loc[self.stations_metadata[self.station_index] == index]
        if metadata.empty:
            with self.figure:
                print(f"Station ID: {index} not found in the stations metadata.")
            return False

        fc_dir = os.path.join(
            self.config["forecast"],
            self.date.strftime("%Y%m"),
            f"PPR{self.date.strftime('%Y%m%d%H')}",
        )

        # opening the record.pickle file
        obs_data = self.phf.open_record(index, rec_path=fc_dir)

        # opening the model file in assets
        pp = self.phf.open_mod_file(os.path.join(self.assets, f"ID_{index}.pickle"))
        ts = self.phf.extract_timestep(pp)

        # get obs and valid dates from obs_data
        obs, valid_dates = self.phf.prepare_obs(obs_data["obs"], pp, self.date, ts)

        # select observed predicted distribution (i.e., forecast of the observations not the water balance
        vars_dict = pd.read_csv(os.path.join(fc_dir, f"mcp_{index}.csv"), index_col=0)
        frcst = self.phf.select_obs_quantiles(vars_dict, pp, self.date)

        pp_objects = self.phf.collate_plotting_data(
            pd.DataFrame(metadata),  # from outlets.csv
            pp,  # model file ID_{obsidian}.pickle file in the assets /ec/ws3/tc/emos/work/cems/floods/efas/assets/efas_5.0/ppData/models
            self.date,  # date time forecast
            valid_dates,  # list of dates in the forecast time period, created from create_full_list_of_dates() and prepare_obs()
            frcst,  # quantiles from the ftp file or Obs_{date}, only use the forecast lead time values, using select_obs_quantiles()
            obs,  # Observations, from record.pickle file, extracted using prepare_obs(), which also gets the valid_dates. Obs_data is obtained using station_data = phf.open_record(station_id)
            False,
            False,
            False,  # if station in fail_list, then True, False, False, otherwise False, False, False
        )

        obs_station = self.observations[f"{int(ts):02d}"][index]
        obs_dates = []
        obs_values = []
        for date in valid_dates:  # dates are backwards
            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            if date_obj in obs_station.index and date_obj >= self.date:
                obs_dates.append(date_obj)
                obs_values.append(obs_station.loc[date_obj])
        if not obs_dates:
            print("No observations available for this station, skipping observations plot")
            rt_obs = None
            fc_crps = None
        else:
            rt_obs = pd.DataFrame({"obs": obs_values}, index=obs_dates)
            frcst.index = pd.to_datetime(frcst.index.str.replace("Obs_", ""), format="%Y%m%d%H")
            fc_obs = frcst[frcst.index.isin(rt_obs.index)]
            fc_crps = crps(np.transpose(fc_obs.values), rt_obs["obs"].values)

        # Disable the numpy warning in the plotting
        np.seterr(invalid="ignore")

        with self.figure:
            clear_output(wait=True)
            self.psf.plot_site_forecast(pp_objects, display=True, rt_observations=rt_obs)
            if index in self.crps_lower:
                fc_crps = np.flip(fc_crps[1:])
                self.psf.plot_crps(self.crps_lower[str(index)], self.crps_upper[str(index)], fc_crps)
            else:
                print(f"No CRPS data available for station {index}")

    def _update_date(self, *args, **kwargs):
        """
        Updates the plot with the selected start and end dates.
        """

        self.date = datetime.strptime(self.date_input.value, "%Y%m%d%H")
        self.update()


class UpdatingHTML(Widget):
    def __init__(self, config):
        self.config = config
        self.content = "Select a point to begin."

        right_layout = Layout(
            justify_content="space-around",
            align_items="center",
            spacing="2px",
            overflow="auto",
            height="600px",
            width="100%",
            margin="0px 0px 0px 5px",
        )

        self.HTML_object = HTML(self.content, layout=right_layout)

        out = self.HTML_object
        super().__init__(out)

    def update(self, new_station, *args, **kwargs):
        json_path = self.config["json"].format(date=self.config["date"]) + new_station + f"_{self.config['date']}.json"
        svg_path = self.config["svg"].format(date=self.config["date"])
        with open(json_path, "r") as f:
            json_info = json.load(f)

        htmls = floods_html.json_to_html(json_info["data"], svg_location=svg_path)
        large_html_string = "".join(htmls)
        self.HTML_object.value = large_html_string
