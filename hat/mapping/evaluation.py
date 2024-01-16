import matplotlib.pyplot as plt
from hat.mapping.station_mapping import calculate_distance, calculate_area_diff_percentage
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
    return count

from geopy.distance import geodesic

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hat.mapping.station_mapping import calculate_distance

def plot_distance_histogram(df, ref_lat_col, ref_lon_col, eval_lat_col, eval_lon_col, interval, y_max=None):
    """
    Plot a histogram of the distances between reference and evaluated points.

    :param df: Pandas DataFrame with the station data.
    :param ref_lat_col: Column name of the reference latitude.
    :param ref_lon_col: Column name of the reference longitude.
    :param eval_lat_col: Column name of the evaluated latitude.
    :param eval_lon_col: Column name of the evaluated longitude.
    :param interval: The interval size for the histogram bins.
    :param y_max: Maximum value for the y-axis. If None, the axis limit is determined automatically.
    :return: The matplotlib figure object.
    """
    # Calculate the distances
    distances = df.apply(lambda row: calculate_distance(row[ref_lat_col], row[ref_lon_col], row[eval_lat_col], row[eval_lon_col]), axis=1)
    distances = distances[~np.isnan(distances)]  # Remove NaN values
    
    max_distance = distances.max()

    # Create the histogram
    fig, ax = plt.subplots()
    ax.hist(distances, bins=np.arange(0, max_distance + interval, interval), color='blue', alpha=0.7)
    ax.set_title('Histogram of Distances Between Reference and Evaluated Points')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Number of Stations')
    ax.set_xlim(0, max_distance)  # Set the limit of x-axis to the maximum distance

    # Set the y-axis limit if specified
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Show the plot
    plt.show()

    return fig

def plot_area_error_histogram(df, ref_area_col, eval_area_col, interval, y_max=None, max_area_error=None):
    """
    Plot a histogram of the area error percentage between reference and evaluated areas using vectorized operations.

    :param df: Pandas DataFrame with the station data.
    :param ref_area_col: Column name of the reference area.
    :param eval_area_col: Column name of the evaluated area.
    :param interval: The interval size for the histogram bins.
    :param y_max: Maximum value for the y-axis. If None, the axis limit is determined automatically.
    :param max_area_error: Maximum area error to include in the histogram. If None, all data is included.
    :return: The matplotlib figure object.
    """
    # Vectorized calculation of area differences
    area_diff = df[ref_area_col] - df[eval_area_col]
    area_diff_percentage = np.abs((area_diff / df[ref_area_col]) * 100)

    # Handle divisions by zero or invalid values
    area_diff_percentage.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter out NaN values and limit the maximum area error if specified
    area_diff_percentage = area_diff_percentage.dropna()
    if max_area_error is not None:
        area_diff_percentage = area_diff_percentage[abs(area_diff_percentage) <= max_area_error]

    # Determine the range of the histogram
    min_error = area_diff_percentage.min()
    max_error = area_diff_percentage.max()
    bins = np.arange(min_error, max_error + interval, interval)

    # Create the histogram
    fig, ax = plt.subplots()
    ax.hist(area_diff_percentage, bins=bins, color='blue', alpha=0.7)
    ax.set_title('Histogram of Area Difference Percentage')
    ax.set_xlabel('Area Difference (%)')
    ax.set_ylabel('Number of Stations')

    # Set the x-axis limit to the range of area differences
    ax.set_xlim(min_error, max_error)

    # Set the y-axis limit if specified
    if y_max is not None:
        ax.set_ylim(0, y_max)

    plt.show()
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
