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
    mae = (np.abs(
        (df[column_reference] - df[column_evaluated]) / df[column_reference] * 100)).mean()
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
    return count


def update_config_file(config_path, new_max_cells, new_max_diff):

    config_path = os.path.expanduser(config_path)

    with open(config_path, 'r') as file:
        config = json.load(file)

    config["max_neighboring_cells"] = new_max_cells
    config["max_area_diff"] = new_max_diff

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
