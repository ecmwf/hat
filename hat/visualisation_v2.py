"""
Python module for visualising geospatial content using jupyter notebook, 
for both spatial and temporal, e.g. netcdf, vector, raster, with time series etc
"""
import os
import pandas as pd
from typing import Dict, Union, List

import geopandas as gpd
from shapely.geometry import Point
import xarray as xr

from hat.hydrostats import run_analysis
from hat.filters import filter_timeseries

class NotebookMap:
    def __init__(self, config: Dict, stations_metadata: str, observations: str, simulations: Union[Dict, str], stats=None):
        self.config = config

        # Prepare Station Metadata
        self.stations_metadata = self.prepare_station_metadata(
            fpath=stations_metadata,
            station_id_column_name=config["station_id_column_name"],
            coord_names=config['station_coordinates'],
            epsg=config['station_epsg'],
            filters=config['station_filters']
        )

        # Prepare Observations Data
        self.observation = self.prepare_observations_data(observations)

        # Prepare Simulations Data
        self.simulations = self.prepare_simulations_data(simulations)

        # Ensure stations in obs and sims are present in metadata
        valid_stations = set(self.stations_metadata[self.config["station_id_column_name"]].values)
        
        # Filter data based on valid stations
        self.observation = self.filter_stations_by_metadata(self.observation, valid_stations)
        self.simulations = {exp: self.filter_stations_by_metadata(ds, valid_stations) for exp, ds in self.simulations.items()}

        self.stats_input = stats
        self.stat_threshold = 70 # default for now, may need to be added as option
        self.stats_output = {}
        if self.stats_input:
            self.stats_output = self.calculate_statistics()
    
    def filter_stations_by_metadata(self, ds, valid_stations):
        """Filter the stations in the dataset to only include those in valid_stations."""
        return ds.sel(station=[s for s in ds.station.values if s in valid_stations])


    def prepare_station_metadata(self, fpath: str, station_id_column_name: str, coord_names: List[str], epsg: int, filters=None) -> xr.Dataset:
        # Read the station metadata file
        df = pd.read_csv(fpath)

        # Convert to a GeoDataFrame
        geometry = [Point(xy) for xy in zip(df[coord_names[0]], df[coord_names[1]])]
        gdf = gpd.GeoDataFrame(df, crs=f"EPSG:{epsg}", geometry=geometry)
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                gdf = gdf[gdf[column] == value]
                
        return gdf
    
    def prepare_observations_data(self, observations: str) -> xr.Dataset:
        """
        Load and preprocess observations data.
        
        Parameters:
        - observations: Path to the observations data file.
        
        Returns:
        - obs_ds: An xarray Dataset containing the observations data.
        """
        file_extension = os.path.splitext(observations)[-1].lower()
        station_id_column_name = self.config.get('station_id_column_name', 'station_id_num')

        if file_extension == '.csv':
            obs_df = pd.read_csv(observations, parse_dates=["Timestamp"])
            obs_melted = obs_df.melt(id_vars="Timestamp", var_name="station", value_name="obsdis")
            
            # Convert melted DataFrame to xarray Dataset
            obs_ds = obs_melted.set_index(["Timestamp", "station"]).to_xarray()
            obs_ds = obs_ds.rename({"Timestamp": "time"})

        elif file_extension == '.nc':
            obs_ds = xr.open_dataset(observations)
            
            # Check for necessary attributes
            if 'obsdis' not in obs_ds or 'time' not in obs_ds.coords:
                raise ValueError("The NetCDF file lacks the expected variables or coordinates.")
                
            # Rename the station_id to station and set it as an index
            obs_ds = obs_ds.rename({station_id_column_name: "station"})
            obs_ds = obs_ds.set_index(station="station")
        else:
            raise ValueError("Unsupported file format for observations.")
        
        return obs_ds

    
    def prepare_simulations_data(self, simulations: Union[Dict, str]) -> Dict[str, xr.Dataset]:
        """
        Load and preprocess simulations data.
        
        Parameters:
        - simulations: Either a string path to the simulations data file or a dictionary mapping 
        experiment names to file paths.
        
        Returns:
        - datasets: A dictionary mapping experiment names to their respective xarray Datasets.
        """
        sim_ds = {}
        
        # Handle the case where simulations is a single string path
        if isinstance(simulations, str):
            sim_ds["default"] = xr.open_dataset(simulations)

        # Handle the case where simulations is a dictionary of experiment names to paths
        elif isinstance(simulations, dict):
            for exp, path in simulations.items():
                expanded_path = os.path.expanduser(path)
                
                if os.path.isfile(expanded_path):  # If it's a file
                    ds = xr.open_dataset(expanded_path)
                    
                elif os.path.isdir(expanded_path):  # If it's a directory
                    # Assume all .nc files in the directory need to be combined
                    files = [f for f in os.listdir(expanded_path) if f.endswith('.nc')]
                    ds = xr.open_mfdataset([os.path.join(expanded_path, f) for f in files], combine='by_coords')
                    
                else:
                    raise ValueError(f"Invalid path: {expanded_path}")
                
                sim_ds[exp] = ds
        else:
            raise TypeError("Expected simulations to be either str or dict.")
        
        return sim_ds


    
    def calculate_statistics(self) -> Dict[str, xr.Dataset]:
        """
        Calculate statistics for the simulations against the observations.
        
        Returns:
        - statistics: A dictionary mapping experiment names to their respective statistics xarray Datasets.
        """
        stats_output = {}
        
        if isinstance(self.simulations, xr.Dataset):
            # For a single simulation dataset
            sim_filtered, obs_filtered = filter_timeseries(self.simulations, self.observation, self.stat_threshold)
            stats_output["default"] = run_analysis(self.stats_input, sim_filtered, obs_filtered)
        elif isinstance(self.simulations, dict):
            # For multiple simulation datasets
            for exp, ds in self.simulations.items():
                # print(f"Processing experiment: {exp}")
                # print("Simulation dataset stations:", ds.station.values)
                # print("Observation dataset stations:", self.observation.station.values)

                sim_filtered, obs_filtered = filter_timeseries(ds, self.observation, self.stat_threshold)
                # stats_output[exp] = run_analysis(self.stats_input, sim_filtered, obs_filtered)

                stations_series = pd.Series(ds.station.values)
                duplicates = stations_series.value_counts().loc[lambda x: x > 1]
                print(exp, ":", duplicates)

        else:
            raise ValueError("Unexpected type for self.simulations")
        
        return stats_output

# Utility Functions

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