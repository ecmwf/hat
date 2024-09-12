import os

import ipywidgets
import pandas as pd
import xarray as xr
from IPython.display import display

from hat.interactive.leaflet import LeafletMap, StatsColormap, PPColormap, ReportingPointsColormap
from hat.interactive.widgets import (
    MetaDataWidget,
    PlotlyWidget,
    PPForecastPlotWidget,
    StatisticsWidget,
    WidgetsManager,
    UpdatingHTML,
)
from hat.observations import read_station_metadata_file


def prepare_simulations_data(simulations, sims_var_name):
    """
    Process simulations and put then in a dictionnary of xarray data arrays.

    Parameters
    ----------
    simulations : dict
        A dictionary of paths to the simulation netCDF files, with the keys
        being the simulation names.
    sims_var_name : str
        The name of the variable in the simulation netCDF files that contains
        the simulated values.

    Returns
    -------
    dict
        A dictionary of xarray data arrays containing the simulation data.

    """
    # If simulations is a dictionary, load data for each experiment
    sim_ds = {}
    for exp, path in simulations.items():
        # Expanding the tilde
        expanded_path = os.path.expanduser(path)

        if os.path.isfile(expanded_path):  # Check if it's a file
            ds = xr.open_dataset(expanded_path)

        sim_ds[exp] = ds[sims_var_name]

    return sim_ds


def prepare_observations_data(observations, sim_ds, obs_var_name):
    """
    Process observation raw dataset to a standard xarray dataset.
    The observation dataset can be either a csv file or a netcdf file.
    The observation dataset is subsetted based on the time values of the
    simulation dataset.

    Parameters
    ----------
    observations : str
        The path to the observation netCDF file.
    sim_ds : dict or xarray.Dataset
        A dictionary of xarray datasets containing the simulation data, or a
        single xarray dataset.
    obs_var_name : str
        The name of the variable in the observation netCDF file that contains
        the observed values.

    Returns
    -------
    xarray.Dataset
        An xarray dataset containing the observation data.

    Raises
    ------
    ValueError
        If the file format of the observations file is not supported.

    """
    file_extension = os.path.splitext(observations)[-1].lower()

    if file_extension == ".csv":
        obs_df = pd.read_csv(observations, parse_dates=["Timestamp"])
        obs_melted = obs_df.melt(id_vars="Timestamp", var_name="station", value_name=obs_var_name)
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

    obs_ds = obs_ds[obs_var_name].sel(time=time_values)
    return obs_ds


def find_common_stations(station_index, stations_metadata, obs_ds, sim_ds, statistics):
    """
    Find common stations between observations, simulations and station
    metadata.

    Parameters
    ----------
    station_index : str
        The name of the column in the station metadata file that contains the
        station IDs.
    stations_metadata : pandas.DataFrame
        A pandas DataFrame containing the station metadata.
    obs_ds : xarray.Dataset
        An xarray dataset containing the observation data.
    sim_ds : dict or xarray.Dataset
        A dictionary of xarray data arrays containing the simulation data.
    statistics : dict
        A dictionary of xarray data arrays containing the statistics data.

    Returns
    -------
    list
        A list of common station IDs.

    """
    ids = []
    ids += [list(obs_ds["station"].values)]
    ids += [list(ds["station"].values) for ds in sim_ds.values()]
    ids += [stations_metadata[station_index]]
    if statistics:
        ids += [list(ds["station"].values) for ds in statistics.values()]

    common_ids = None
    for id in ids:
        if common_ids is None:
            common_ids = set(id)
        else:
            common_ids = set(id) & common_ids
    return list(common_ids)


class StationsExplorer:
    """
    Base class for the interactive stations explorer.
    Provides the stations metadata, the title label and an empty LeafletMap.
    """

    def __init__(self, config, title="Interactive stations explorer"):
        """
        Initializes an instance of the Explorer class.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the
            Explorer.
        title : str, optional
            The title of the explorer to display in the application.
        """

        self.config = config

        # Create station objects
        self.stations_metadata = read_station_metadata_file(
            fpath=config["stations"],
            coord_names=config["station_coordinates"],
            epsg=config["station_epsg"],
            filters=config["station_filters"],
        )
        self.station_index = config["station_id_column_name"]

        # Title label
        self.title_label = ipywidgets.Label(
            "Interactive Map Visualisation for Hydrological Stations",
            layout=ipywidgets.Layout(justify_content="center"),
            style={"font_weight": "bold", "font_size": "24px", "font_family": "Arial"},
        )

        # Create the main leaflet map
        self.leafletmap = LeafletMap()


class TimeSeriesExplorer(StationsExplorer):
    """
    Initialize the interactive map with configurations and data sources.
    """

    def __init__(self, config):
        """
        Initializes an instance of the Explorer class.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the
            Explorer.

        Notes
        -----
        This method initializes an instance of the Explorer class with the
        given configuration parameters.
        The configuration parameters should be provided as a dictionary with
        the following keys:

        - stations : str
            The path to the station metadata file.
        - observations : str
            The path to the observation netCDF file.
        - simulations : dict
            A dictionary of paths to the simulation netCDF files, with the keys
            being the simulation names.
        - statistics : dict, optional
            A dictionary of paths to the statistics netCDF files, with the keys
            being the simulation names.
        - station_coordinates : list of str
            The names of the columns in the station metadata file that contain
            the station coordinates.
        - station_epsg : int
            The EPSG code of the coordinate reference system used by the
            station coordinates.
        - station_filters : dict
            A dictionary of filters to apply to the station metadata file.
        - sims_var_name : str
            The name of the variable in the simulation netCDF files that
            contains the simulated values.
        - obs_var_name : str
            The name of the variable in the observation netCDF file that
            contains the observed values.
        - station_id_column_name : str
            The name of the column in the station metadata file that contains
            the station IDs.

        Raises
        ------
        AssertionError
            If there is a mismatch between the keys of the statistics netCDF
            files and the simulation netCDF files.

        """
        # Initialise base class
        title = "Interactive Map Visualisation for Hydrological Model Performance"
        super().__init__(config, title)

        self.loading_widget = ipywidgets.Label(value="")

        # Use the external functions to prepare data
        sim_ds = prepare_simulations_data(config["simulations"], config["sims_var_name"])
        obs_ds = prepare_observations_data(config["observations"], sim_ds, config["obs_var_name"])

        # set station index
        self.station_index = config["station_id_column_name"]

        # Retrieve statistics from the statistics netcdf input
        self.statistics = {}
        stats = config.get("statistics")
        if stats is not None:
            for name, path in stats.items():
                self.statistics[name] = xr.open_dataset(path)

        # Ensure the keys of self.statistics match the keys of self.sim_ds
        assert set(self.statistics.keys()) == set(sim_ds.keys()), "Mismatch between statistics and simulations keys."

        # find common station ids between metadata, observation and simulations
        common_ids = find_common_stations(
            self.station_index,
            self.stations_metadata,
            obs_ds,
            sim_ds,
            self.statistics,
        )

        print(f"Found {len(common_ids)} common stations")
        self.stations_metadata = self.stations_metadata.loc[self.stations_metadata[self.station_index].isin(common_ids)]
        obs_ds = obs_ds.sel(station=common_ids)
        for sim, ds in sim_ds.items():
            sim_ds[sim] = ds.sel(station=common_ids)

        # Create the interactive widgets
        datasets = sim_ds
        datasets["obs"] = obs_ds
        widgets = {}
        widgets["plot"] = PlotlyWidget(datasets)
        widgets["stats"] = StatisticsWidget(self.statistics)
        widgets["meta"] = MetaDataWidget(self.stations_metadata, self.station_index)
        self.widgets = WidgetsManager(widgets, config["station_id_column_name"], self.loading_widget)

    def create_frame(self):
        """
        Initialize the layout of the widgets for the map visualization.

        Returns
        -------
        ipywidgets.VBox
            A vertical box containing the layout elements for the map
            visualization.

        """
        # Layouts 2
        main_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="stretch",
            spacing="2px",
            width="1000px",
        )
        left_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="center",
            spacing="2px",
            width="40%",
        )
        right_layout = ipywidgets.Layout(
            justify_content="center",
            align_items="center",
            spacing="2px",
            width="60%",
        )

        # Frames
        top_left_frame = self.leafletmap.output(left_layout)
        top_right_frame = ipywidgets.VBox(
            [self.widgets["plot"].output, self.widgets["stats"].output],
            layout=right_layout,
        )
        main_top_frame = ipywidgets.HBox([top_left_frame, top_right_frame])

        # Main layout
        main_frame = ipywidgets.VBox(
            [self.title_label, main_top_frame, self.widgets["meta"].output],
            layout=main_layout,
        )
        return main_frame

    def plot(self, colorby=None, sim=None, limits=None, mp_colormap="viridis"):
        """
        Plot the stations markers colored by a given metric.

        Parameters
        ----------
        colorby : str, optional
            The name of the metric to color the stations by.
        sim : str, optional
            The name of the simulation to use for the metric.
        limits : list, optional
            A list of two values representing the minimum and maximum values
            for the color bar.
        mp_colormap : str, optional
            The name of the matplotlib colormap to use for the color bar.

        """
        # create colormap from statistics
        stats = None
        if self.statistics and colorby is not None and sim is not None:
            stats = self.statistics[sim][colorby]
        colormap = StatsColormap(self.config, stats, mp_colormap, limits)

        # add layer to the leaflet map
        self.leafletmap.add_geolayer(
            self.stations_metadata,
            colormap,
            self.widgets,
            self.config["station_coordinates"],
        )

        # Initialize frame elements
        frame = self.create_frame()

        # Display the main layout
        display(frame)


class PPForecastExplorer(StationsExplorer):
    def __init__(self, config):
        # Initialise base class
        title = "Interactive Map Visualisation for the PP module of the EFAS forecast"
        super().__init__(config, title)

        # Create the interactive widgets
        widgets = {}
        # Create loading widget
        self.loading_widget = ipywidgets.Label(value="")
        widgets["meta"] = MetaDataWidget(self.stations_metadata, self.station_index)
        widgets["plot"] = PPForecastPlotWidget(config["pp"], self.stations_metadata, self.station_index)
        self.widgets = WidgetsManager(widgets, config["station_id_column_name"], self.loading_widget)

    def create_frame(self):
        """
        Initialize the layout of the widgets for the map visualization.

        Returns
        -------
        ipywidgets.VBox
            A vertical box containing the layout elements for the map
            visualization.

        """
        # Layouts 2
        main_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="stretch",
            spacing="2px",
            width="1000px",
        )
        # half_layout = ipywidgets.Layout(
        #     justify_content="space-around",
        #     align_items="center",
        #     spacing="2px",
        #     width="50%",
        # )
        # left_layout = ipywidgets.Layout(
        #     justify_content="space-around",
        #     align_items="center",
        #     spacing="2px",
        #     width="40%",
        # )
        # right_layout = ipywidgets.Layout(
        #     justify_content="center", align_items="center", spacing="2px", width="60%"
        # )

        # Main layout
        main_top_frame = ipywidgets.HBox(
            [self.leafletmap.map, self.widgets["plot"].output],
        )
        main_frame = ipywidgets.VBox(
            [self.title_label, main_top_frame, self.widgets["meta"].output],
            layout=main_layout,
        )
        return main_frame

    def plot(self):
        """
        Plot the stations markers colored by a given metric.

        Parameters
        ----------
        colorby : str, optional
            The name of the metric to color the stations by.
        sim : str, optional
            The name of the simulation to use for the metric.
        limits : list, optional
            A list of two values representing the minimum and maximum values
            for the color bar.
        mp_colormap : str, optional
            The name of the matplotlib colormap to use for the color bar.

        """
        # create colormap from statistics
        colormap = PPColormap(self.config)

        # add layer to the leaflet map
        self.leafletmap.add_geolayer(
            self.stations_metadata,
            colormap,
            self.widgets,
            self.config["station_coordinates"],
        )

        # Initialize frame elements
        frame = self.create_frame()

        # Display the main layout
        display(frame)


class ReportingPointsExplorer(StationsExplorer):
    def __init__(self, config):
        # Initialise base class
        self.config = config
        title = "Interactive Map Visualisation of Reporting Points"
        config["stations"] = self.config["stations"].format(date=self.config["date"])
        super().__init__(config, title)

        # Create the interactive widgets
        widgets = {}
        # Create loading widget
        self.loading_widget = ipywidgets.Label(value="")
        widgets["html"] = UpdatingHTML(self.config)
        self.widgets = WidgetsManager(widgets, config["station_id_column_name"], self.loading_widget)

    def create_frame(self):
        """
        Initialize the layout of the widgets for the map visualization.

        Returns
        -------
        ipywidgets.VBox
            A vertical box containing the layout elements for the map
            visualization.

        """

        main_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="stretch",
            spacing="2px",
            width="1000px",
        )
        left_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="center",
            spacing="2px",
            width="50%",
        )

        top_left_frame = self.leafletmap.output(left_layout)

        # Main layout
        main_top_frame = ipywidgets.HBox(
            [top_left_frame, self.widgets["html"].output],
        )
        main_frame = ipywidgets.VBox(
            [self.title_label, main_top_frame],
            layout=main_layout,
        )
        return main_frame

    def plot(self):
        """
        Plot the stations markers colored by a given metric.

        Parameters
        ----------
        colorby : str, optional
            The name of the metric to color the stations by.
        sim : str, optional
            The name of the simulation to use for the metric.
        limits : list, optional
            A list of two values representing the minimum and maximum values
            for the color bar.
        mp_colormap : str, optional
            The name of the matplotlib colormap to use for the color bar.

        """

        # create colormap from statistics
        colormap = ReportingPointsColormap(self.config)

        # add layer to the leaflet map
        self.leafletmap.add_geolayer(
            self.stations_metadata,
            colormap,
            self.widgets,
            self.config["station_coordinates"],
        )

        # Initialize frame elements
        frame = self.create_frame()

        # Display the main layout
        display(frame)
