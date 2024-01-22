
import matplotlib.pyplot as plt
from hat.mapping.station_mapping import calculate_distance, calculate_area_diff_percentage, calculate_distance_cells
import pandas as pd
import numpy as np
import json
import os

# Function to calculate Mean Absolute Error
def calculate_mae(df, column_reference, column_evaluated):
    """
    Calculate the Mean Absolute Error (MAE) as a relative percentage between two columns in a DataFrame.

    :param df: Pandas DataFrame
    :param column_reference: Name of the reference column
    :param column_evaluated: Name of the column to be evaluated against the reference
    :return: MAE value as a percentage
    """
    mae = round((np.abs(
        (df[column_reference] - df[column_evaluated]) / df[column_reference] * 100)).mean(), 2)
    print(f"Mean Abs. Error (MAE) % between {column_reference} and {column_evaluated}: {mae}%")
    return mae


# Function to calculate Average Error
def calculate_rmse(df, column_reference, column_evaluated):
    """
    Calculate the Root Mean Square Error (RMSE) between two columns in a DataFrame.

    :param df: Pandas DataFrame
    :param column_reference: Name of the reference column
    :param column_evaluated: Name of the column to be evaluated against the reference
    :return: RMSE value
    """
    rmse = np.sqrt(((df[column_reference] - df[column_evaluated]) ** 2).mean())
    print(f"RMSE between {column_reference} and {column_evaluated}: {rmse} km2")
    return rmse


# Function to count values within a specified range
def count_within_abs_error_range(df, column_reference, column_evaluated, lower_limit, upper_limit):
    """
    Count the number of rows in a DataFrame where the absolute percentage error between two columns falls within a specified range.

    :param df: Pandas DataFrame
    :param column_reference: Name of the reference column
    :param column_evaluated: Name of the column to be evaluated against the reference
    :param abs_lower_limit: Lower limit of the absolute percentage error range
    :param abs_upper_limit: Upper limit of the absolute percentage error range
    :return: Count of rows within the specified error range
    """
    abs_error_percent = np.abs((df[column_reference] - df[column_evaluated]) / df[column_reference] * 100)
    count = ((abs_error_percent > lower_limit) & (abs_error_percent < upper_limit)).sum()
    count_all = len(df)
    print(f"Count of rows with absolute error % between {column_reference} and {column_evaluated} in the range of ({lower_limit}%, {upper_limit}%): {count} / {count_all}")
    return 


def count_perfect_mapping(df, ref_lat_col, ref_lon_col, eval_lat_col, eval_lon_col, tolerance_degrees):
    """
    Count the number of rows where the geographic distance between reference and evaluated latitude/longitude 
    pairs is within the specified tolerance distance in decimal degrees.

    :param df: Pandas DataFrame
    :param ref_lat_col: Name of the reference latitude column
    :param ref_lon_col: Name of the reference longitude column
    :param eval_lat_col: Name of the evaluated latitude column
    :param eval_lon_col: Name of the evaluated longitude column
    :param tolerance_degrees: Tolerance distance in decimal degrees
    :return: Count of rows within the specified distance
    """
    count = 0
    valid_count = 0  # Counter for rows with valid values

    for index, row in df.iterrows():
        if not pd.isna(row[ref_lat_col]) and not pd.isna(row[eval_lat_col]):
            lat_diff = abs(row[ref_lat_col] - row[eval_lat_col])
            lon_diff = abs(row[ref_lon_col] - row[eval_lon_col])
            # Check if both latitude and longitude differences are within the tolerance 
            # (tried using the haversine formula but it resulted to more rounding errors)
            # so I used the combination diff of lat and lot instead
            if lat_diff <= tolerance_degrees and lon_diff <= tolerance_degrees:
                count += 1
            valid_count += 1
    print(f"Count of perfect mapping rows with distance within {tolerance_degrees} decimal degrees: {count} / {valid_count}")
    return 

def plot_distance_histogram(df, ref_lat_col, ref_lon_col, eval_lat_col, eval_lon_col, interval, y_range=None, max_distance=None, y_scale='linear'):
    """
    Plot a histogram of distances between reference and evaluated points.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the station data.
    - ref_lat_col, ref_lon_col, eval_lat_col, eval_lon_col: Column names for latitude and longitude.
    - interval (float): Interval size for the histogram bins.
    - y_range (list): List of two elements [min, max] specifying the range of the y-axis.
    - max_distance (float): Maximum distance for the x-axis.
    - y_scale (str): Scale of the y-axis ('linear' or 'log').

    Returns:
    - matplotlib.figure.Figure: Matplotlib figure object of the histogram.
    """
    distances = df.apply(lambda row: calculate_distance(row[ref_lat_col], row[ref_lon_col], row[eval_lat_col], row[eval_lon_col]), axis=1)
    distances = distances[~np.isnan(distances)]  # Remove NaN values

    fig, ax = plt.subplots()
    if max_distance:
        max_distance = max_distance
    else:
        max_distance = distances.max()

    ax.hist(distances, bins=np.arange(0, max_distance + interval, interval), color='blue', alpha=0.7)
    ax.set_title('Histogram of Distances Between Reference and Evaluated Points')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Number of Stations')
    ax.set_xlim(0, max_distance)
    # ax.set_ylim(y_range if y_range else [0, ax.get_ylim()[1]])
    ax.set_yscale(y_scale)

    return 

def plot_distance_cells_histogram(df, ref_lat_idx_col, ref_lon_idx_col, eval_lat_idx_col, eval_lon_idx_col, interval, y_range=None, max_distance=None, y_scale='linear'):
    """
    Plot a histogram of grid distances between reference and evaluated points.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the station data.
    - ref_lat_idx_col, ref_lon_idx_col, eval_lat_idx_col, eval_lon_idx_col: Column names for grid indices of latitude and longitude.
    - interval (float): Interval size for the histogram bins.
    - y_range (list): List of two elements [min, max] specifying the range of the y-axis.
    - max_distance (float): Maximum distance for the x-axis.
    - y_scale (str): Scale of the y-axis ('linear' or 'log').

    Returns:
    - matplotlib.figure.Figure: Matplotlib figure object of the histogram.
    """
    grid_distances = df.apply(lambda row: calculate_distance_cells(row[ref_lat_idx_col], row[ref_lon_idx_col], row[eval_lat_idx_col], row[eval_lon_idx_col]), axis=1)
    grid_distances = grid_distances[~np.isnan(grid_distances)]  # Remove NaN values

    fig, ax = plt.subplots()

    if max_distance:
        max_distance = max_distance
    else:
        max_distance = grid_distances.max()

    ax.hist(grid_distances, bins=np.arange(0, max_distance + interval, interval), color='blue', alpha=0.7)
    ax.set_title('Histogram of Grid Distances Between Reference and Evaluated Points')
    ax.set_xlabel('Grid Distance (Number of Cells)')
    ax.set_ylabel('Number of Stations')
    ax.set_xlim(0, max_distance if max_distance else grid_distances.max())
    # ax.set_ylim(y_range if y_range else [0, ax.get_ylim()[1]])
    ax.set_yscale(y_scale)

    return fig

def plot_area_error_histogram(df, ref_area_col, eval_area_col, interval, y_range=None, max_area_error=None, y_scale='linear'):
    """
    Plot a histogram of area error percentage between reference and evaluated areas.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the station data.
    - ref_area_col, eval_area_col: Column names for reference and evaluated areas.
    - interval (float): Interval size for the histogram bins.
    - y_range (list): List of two elements [min, max] specifying the range of the y-axis.
    - max_area_error (float): Maximum area error for the x-axis.
    - y_scale (str): Scale of the y-axis ('linear' or 'log').

    Returns:
    - matplotlib.figure.Figure: Matplotlib figure object of the histogram.
    """
    area_diff_percentages = df.apply(lambda row: abs(calculate_area_diff_percentage(row[eval_area_col], row[ref_area_col])), axis=1)
    area_diff_percentages = area_diff_percentages.dropna()

    fig, ax = plt.subplots()
    if max_area_error:
        max_area_error = max_area_error
    else:
        max_area_error = area_diff_percentages.max()
    ax.hist(area_diff_percentages, bins=np.arange(0, max_area_error + interval, interval), color='blue', alpha=0.7)
    ax.set_title('Histogram of Area Difference Percentage')
    ax.set_xlabel('Area Difference (%)')
    ax.set_ylabel('Number of Stations')
    ax.set_xlim(0, max_area_error if max_area_error else area_diff_percentages.max())
    # ax.set_ylim(y_range if y_range else [0, ax.get_ylim()[1]])
    ax.set_yscale(y_scale)

    return fig

def update_config_file(config_path, new_max_cells, new_max_diff, out_folder_name):
    config_path = os.path.expanduser(config_path)

    with open(config_path, 'r') as file:
        config = json.load(file)

    config["max_neighboring_cells"] = new_max_cells
    config["max_area_diff"] = new_max_diff
    config["out_directory"] = out_folder_name  # Update output directory

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
