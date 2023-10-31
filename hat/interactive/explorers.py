import os

import ipywidgets
import pandas as pd
import xarray as xr
from IPython.core.display import display

from hat.interactive.leaflet import LeafletMap, PyleafletColormap
from hat.interactive.widgets import (
    MetaDataWidget,
    PlotlyWidget,
    StatisticsWidget,
    WidgetsManager,
)
from hat.observations import read_station_metadata_file


def prepare_simulations_data(simulations, sims_var_name):
    """
    process simulations raw datasets to a standard dataframe
    """

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
        sim_ds[exp] = ds[sims_var_name]

    return sim_ds


def prepare_observations_data(observations, sim_ds, obs_var_name):
    """
    process observation raw dataset to a standard dataframe
    """
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

    obs_ds = obs_ds[obs_var_name].sel(time=time_values)
    return obs_ds


def find_common_station(station_index, stations_metadata, statistics, sim_ds, obs_ds):
    """
    find common station between observation and simulation and station metadata
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


class TimeSeriesExplorer:
    """
    Initialize the interactive map with configurations and data sources.
    """

    def __init__(self, config, stations, observations, simulations, stats=None):
        self.config = config
        self.stations_metadata = read_station_metadata_file(
            fpath=stations,
            coord_names=config["station_coordinates"],
            epsg=config["station_epsg"],
            filters=config["station_filters"],
        )

        # Use the external functions to prepare data
        sim_ds = prepare_simulations_data(simulations, config["sims_var_name"])
        obs_ds = prepare_observations_data(observations, sim_ds, config["obs_var_name"])

        # set station index
        self.station_index = config["station_id_column_name"]

        # Retrieve statistics from the statistics netcdf input
        self.statistics = {}
        if stats:
            for name, path in stats.items():
                self.statistics[name] = xr.open_dataset(path)

        # Ensure the keys of self.statistics match the keys of self.sim_ds
        assert set(self.statistics.keys()) == set(
            sim_ds.keys()
        ), "Mismatch between statistics and simulations keys."

        # find common station ids between metadata, observation and simulations
        common_ids = find_common_station(
            self.station_index, self.stations_metadata, self.statistics, sim_ds, obs_ds
        )

        print(f"Found {len(common_ids)} common stations")
        self.stations_metadata = self.stations_metadata.loc[
            self.stations_metadata[self.station_index].isin(common_ids)
        ]
        obs_ds = obs_ds.sel(station=common_ids)
        for sim, ds in sim_ds.items():
            sim_ds[sim] = ds.sel(station=common_ids)

        # Create loading widget
        self.loading_widget = ipywidgets.Label(value="")

        # Title label
        self.title_label = ipywidgets.Label(
            "Interactive Map Visualisation for Hydrological Model Performance",
            layout=ipywidgets.Layout(justify_content="center"),
            style={"font_weight": "bold", "font_size": "24px", "font_family": "Arial"},
        )

        # Create the interactive widgets
        datasets = sim_ds
        datasets["obs"] = obs_ds
        widgets = {}
        widgets["plot"] = PlotlyWidget(datasets)
        widgets["stats"] = StatisticsWidget(self.statistics)
        widgets["meta"] = MetaDataWidget(self.stations_metadata, self.station_index)
        self.widgets = WidgetsManager(
            widgets, config["station_id_column_name"], self.loading_widget
        )

        # Create the main leaflet map
        self.leafletmap = LeafletMap()

    def create_frame(self):
        """
        Initialize the layout elements for the map visualization.
        """

        # # Layouts 1
        # main_layout = ipywidgets.Layout(
        #     justify_content='space-around',
        #     align_items='stretch',
        #     spacing='2px',
        #     width='1000px'
        # )
        # half_layout = ipywidgets.Layout(
        #     justify_content='space-around',
        #     align_items='center',
        #     spacing='2px',
        #     width='50%'
        # )

        # # Frames
        # stats_frame = ipywidgets.HBox(
        #     [self.widgets['plot'].output, self.widgets['stats'].output],
        #     # layout=main_layout
        # )
        # main_frame = ipywidgets.VBox(
        #     [
        #         self.title_label,
        #         self.loading_widget,
        #         self.leafletmap.output(main_layout),
        #         self.widgets['meta'].output, stats_frame
        #     ],
        #     layout=main_layout
        # )

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
            justify_content="center", align_items="center", spacing="2px", width="60%"
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

    def mapplot(self, colorby=None, sim=None, limits=None, mp_colormap="viridis"):
        """Plot the map with stations colored by a given metric.
        input example:
        colorby = "kge" this should be the objective functions of the statistics
        limits = [<min>, <max>] min and max values of the color bar
        mp_colormap = "viridis" colormap name to be used based on matplotlib colormap
        """
        # create colormap from statistics
        stats = None
        if self.statistics and colorby is not None and sim is not None:
            stats = self.statistics[sim][colorby]
        colormap = PyleafletColormap(self.config, stats, mp_colormap, limits)

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
