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
    expected_rmse = 7.91  # Corrected manually calculated value
    assert calculate_rmse(df, "reference", "evaluated") == pytest.approx(expected_rmse)



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
    fig = count_and_analyze_area_distance(
        sample_dataframe, area_diff_limit, distance_limit
    )

    assert isinstance(fig, Figure), "The function should return a matplotlib figure."
