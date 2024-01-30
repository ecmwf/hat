import pandas as pd
import pytest
from matplotlib.figure import Figure

from hat.mapping.evaluation import (
    calculate_mae,
    calculate_rmse,
    count_and_analyze_area_distance,
)


def test_calculate_mae():
    data = {"reference": [100, 200], "evaluated": [90, 195]}
    df = pd.DataFrame(data)
    expected_mae = 7.5  # Calculated manually
    assert calculate_mae(df, "reference", "evaluated") == pytest.approx(expected_mae)


def test_calculate_rmse():
    data = {"reference": [100, 200], "evaluated": [90, 195]}
    df = pd.DataFrame(data)
    expected_rmse = 7.91
    assert calculate_rmse(df, "reference", "evaluated") == pytest.approx(expected_rmse, abs=1e-3)


# Sample data for testing
@pytest.fixture
def sample_dataframe():
    data = {
        "manual_area": [100, 200, 300],
        "optimum_grid_area": [90, 210, 290],
        "manual_lat_idx": [0, 0, 0],
        "manual_lon_idx": [0, 1, 2],
        "optimum_grid_lat_idx": [0, 0, 3],
        "optimum_grid_lon_idx": [0, 2, 2],
    }
    df = pd.DataFrame(data)
    return df


def test_count_and_analyze_area_distance(sample_dataframe):
    area_diff_limit = 10  # 10 percent
    distance_limit = 2  # 2 grid cells
    result = count_and_analyze_area_distance(
        sample_dataframe, area_diff_limit, distance_limit, "manual", "optimum_grid", "log"
    )

    fig = result["figure"]
    assert isinstance(fig, Figure), "The function should return a matplotlib figure."

    # Calculate expected counts based on the sample data
    expected_outside_area_limit = 0  # All areas are within the 10% limit
    expected_inside_area_limit = 3  # All rows is within limit
    expected_within_distance_limit = 2  # Rows 1 and 2 are within 2 cells distance
    expected_outside_distance_limit = 1  # Row 3 is outside 2 cells distance

    # Assertions for count values
    assert result["count_outside_area_limit"] == expected_outside_area_limit
    assert result["count_inside_area_limit"] == expected_inside_area_limit
    assert result["count_within_distance_limit"] == expected_within_distance_limit
    assert result["count_outside_distance_limit"] == expected_outside_distance_limit
