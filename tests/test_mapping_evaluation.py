import pytest
import pandas as pd
import matplotlib.pyplot as plt
from hat.mapping.evaluation import *

def test_calculate_mae():
    data = {'reference': [100, 200, 300], 'evaluated': [90, 195, 305]}
    df = pd.DataFrame(data)
    expected_mae = 2.33  # Calculated manually
    assert calculate_mae(df, 'reference', 'evaluated') == pytest.approx(expected_mae)

def test_calculate_rmse():
    data = {'reference': [100, 200, 300], 'evaluated': [90, 195, 305]}
    df = pd.DataFrame(data)
    expected_rmse = 5.77  # Calculated manually
    assert calculate_rmse(df, 'reference', 'evaluated') == pytest.approx(expected_rmse)

@pytest.fixture
def example_dataframe():
    return pd.DataFrame({
        'reference': [100, 200, 300, 400],
        'evaluated': [110, 190, 320, 380]
    })

def test_count_within_abs_error_range(example_dataframe):
    expected_count = 2
    result = count_within_abs_error_range(example_dataframe, 'reference', 'evaluated', 5, 10)
    assert result == expected_count

@pytest.fixture
def example_geo_dataframe():
    return pd.DataFrame({
        'ref_lat': [0, 0, 0],
        'ref_lon': [0, 0, 0],
        'eval_lat': [0, 0.01, 0.02],
        'eval_lon': [0, 0.01, 0.02]
    })

def test_count_perfect_mapping(example_geo_dataframe):
    expected_count = 1
    result = count_perfect_mapping(example_geo_dataframe, 'ref_lat', 'ref_lon', 'eval_lat', 'eval_lon', 0.01)
    assert result == expected_count

@pytest.fixture
def area_distance_df():
    return pd.DataFrame({
        'ref_area': [100, 200, 300], 
        'eval_area': [105, 195, 290],
        'ref_lat_idx': [0, 1, 2], 
        'ref_lon_idx': [0, 1, 2],
        'eval_lat_idx': [0, 2, 4], 
        'eval_lon_idx': [0, 2, 4]
    })

def test_count_within_area_and_distance(area_distance_df):
    expected_count = 1  # Only the second row fits both criteria
    result = count_within_area_and_distance(area_distance_df, 'ref_area', 'eval_area', 'ref_lat_idx', 'ref_lon_idx', 'eval_lat_idx', 'eval_lon_idx', 10, 2)
    assert result == expected_count

@pytest.fixture
def distance_histogram_df():
    return pd.DataFrame({
        'ref_lat': [0, 1, 2], 
        'ref_lon': [0, 1, 2],
        'eval_lat': [0, 1.1, 2.2], 
        'eval_lon': [0, 1.1, 2.2]
    })

def test_plot_distance_histogram(distance_histogram_df):
    fig = plot_distance_histogram(distance_histogram_df, 'ref_lat', 'ref_lon', 'eval_lat', 'eval_lon', 0.1)
    assert isinstance(fig, plt.Figure)  # Ensure a matplotlib figure is returned
