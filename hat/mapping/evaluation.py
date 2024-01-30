import matplotlib.pyplot as plt
import numpy as np

from hat.mapping.station_mapping import (
    calculate_area_diff_percentage,
    calculate_distance_cells,
)


# Function to calculate Mean Absolute Error
def calculate_mae(df, column_reference, column_evaluated):
    """
    Calculate the Mean Absolute Error (MAE) as
    mean difference between reference and evaluated columns in a DataFrame.

    :param df: Pandas DataFrame
    :param column_reference: Name of the reference column
    :param column_evaluated: Name of the column to be evaluated against the reference
    :return: MAE value as km2
    """
    mae = round(
        (
            np.abs(
                (df[column_reference] - df[column_evaluated])
            )
        ).mean(),
        2,
    )
    print(
        "Mean Abs. Error (MAE) between"
        + f"{column_reference} & {column_evaluated}: {mae}km2"
    )
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
    rmse = round(np.sqrt(((df[column_reference] - df[column_evaluated]) ** 2).mean()))
    print(f"RMSE between {column_reference} and {column_evaluated}: {rmse} km2")
    return rmse


def count_and_analyze_area_distance(
    df,
    area_diff_limit,
    distance_limit,
    ref_name="manual",
    eval_name="optimum_grid",
    y_scale="log",
):
    """
    Count stations based on area difference and grid cell distance,
    and analyze distances histogram exceeding the area difference limit.

    :param df: Pandas DataFrame
    :param area_diff_limit: Upper limit of area difference percentage.
    :param distance_limit: Upper limit of grid cell distance.
    :param ref_area_col, eval_area_col:
        identification names for reference and evaluated data,
        options are 'manual', 'nearest_grid', 'optimum_grid'
    :param y_scale: Scale of the y-axis ('linear' or 'log').
    :return: Detailed messages about counts and a histogram figure
        of grid distances exceeding the area diff limit.
    """

    ref_area_col, eval_area_col = (
        ref_name + "_area",
        eval_name + "_area",
    )  # column name for reference and evaluated grid upstream area
    ref_lat_idx_col, ref_lon_idx_col = (
        ref_name + "_lat_idx",
        ref_name + "_lon_idx",
    )  # column name for reference grid indices
    eval_lat_idx_col, eval_lon_idx_col = (
        eval_name + "_lat_idx",
        eval_name + "_lon_idx",
    )  # column name for evaluated grid indices

    # Initialise counters for counting stations and distances frequencies
    count_outside_area_limit = 0
    count_inside_area_limit = 0
    count_within_distance_limit = 0
    count_outside_distance_limit = 0
    distance_freq = {}

    for index, row in df.iterrows():
        area_diff = abs(
            calculate_area_diff_percentage(row[eval_area_col], row[ref_area_col])
        )

        if area_diff <= area_diff_limit:
            grid_distance = round(
                calculate_distance_cells(
                    row[ref_lat_idx_col],
                    row[ref_lon_idx_col],
                    row[eval_lat_idx_col],
                    row[eval_lon_idx_col],
                )
            )
            distance_freq[grid_distance] = distance_freq.get(grid_distance, 0) + 1
            count_inside_area_limit += 1

            if grid_distance <= distance_limit:
                count_within_distance_limit += 1
            else:
                count_outside_distance_limit += 1

        else:
            count_outside_area_limit += 1

    # Prepare data for histogram
    distances, frequencies = zip(*distance_freq.items())
    fig, ax = plt.subplots()
    ax.bar(distances, frequencies, color="blue", alpha=0.7, width=0.9)
    ax.set_title(
        "Histogram of Distances"
        + f"Found within Acceptable Area Differences of {area_diff_limit}%"
    )
    ax.set_xlabel("Grid Distance (Number of Cells)")
    ax.set_ylabel("Frequency")
    ax.set_yscale(y_scale)
    ax.yaxis.grid(True)

    print(
        f"No of stations within {area_diff_limit}% upstream area difference margin: \n"
        + f"  Not found: {count_outside_area_limit}\n"
        + f"  Found: {count_inside_area_limit}\n"
        + f"  - Found at right location: {count_within_distance_limit}\n"
        + f"  - Found but NOT at right location: {count_outside_distance_limit}"
    )

    return fig
