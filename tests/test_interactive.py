import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from hat.interactive import leaflet as lf
from hat.interactive import widgets as wd


class TestThrottledClick:
    def test_delay(self):
        throttler = wd.ThrottledClick(0.01)
        assert throttler.should_process() is True
        assert throttler.should_process() is False
        time.sleep(0.015)
        assert throttler.should_process() is True


class DummyWidget(wd.Widget):
    def __init__(self):
        super().__init__(output=None)
        self.index = None

    def update(self, index, metadata, **kwargs):
        self.index = index


class TestWidgetsManager:
    def test_update(self):
        dummy = DummyWidget()
        widgets = wd.WidgetsManager(widgets={"dummy": dummy}, index_column="station")
        feature = {
            "properties": {
                "station": "A",
            }
        }
        widgets.update(feature)
        assert dummy.index == "A"


def test_filter_nan():
    dates = np.arange(10)
    values = np.arange(10, dtype=float)
    values[5] = np.nan
    dates_new, values_new = wd._filter_nan_values(dates, values)
    assert len(dates_new) == 9
    assert len(values_new) == 9


class TestPlotlyWidget:
    def test_update(self):
        datasets = {
            "obs": xr.DataArray(
                [[0, 3, 6], [0, 3, 6]],
                coords={
                    "time": np.array(
                        ["2007-07-13", "2007-01-14"], dtype="datetime64[ns]"
                    ),
                    "station": [0, 1, 2],
                },
            ),
            "sim1": xr.DataArray(
                [[1, 2, 3], [1, 2, 3]],
                coords={
                    "time": np.array(
                        ["2007-07-13", "2007-01-14"], dtype="datetime64[ns]"
                    ),
                    "station": [0, 1, 2],
                },
            ),
            "sim2": xr.DataArray(
                [[4, 5, 6], [4, 5, 6]],
                coords={
                    "time": np.array(
                        ["2007-07-13", "2007-01-14"], dtype="datetime64[ns]"
                    ),
                    "station": [0, 1, 2],
                },
            ),
        }
        widget = wd.PlotlyWidget(
            datasets=datasets,
        )
        assert widget.update(1) is True  # if id found
        assert widget.update(3) is False  # if id not found


class TestMetaDataWidget:
    def test_update(self):
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [0.1, 0.2, 0.3]}
        )
        widget = wd.MetaDataWidget(df, "col2")
        assert widget.update("a") is True
        assert widget.update("f") is False


class TestStatisticsWidget:
    def test_update(self):
        datasets = {
            "sim1": xr.Dataset(
                {"data": [0, 1, 2]},
                coords={
                    "station": [0, 1, 2],
                },
            ),
            "sim2": xr.Dataset(
                {"data": [4, 5, 6]},
                coords={
                    "station": [0, 1, 2],
                },
            ),
        }
        widget = wd.StatisticsWidget(datasets)
        assert widget.update(0) is True
        assert widget.update(4) is False

    def test_fail_creation(self):
        datasets = {
            "sim1": xr.Dataset(
                {"data": [0, 1, 2]},
                coords={
                    "id": [0, 1, 2],
                },
            ),
        }
        with pytest.raises(AssertionError):
            wd.StatisticsWidget(datasets)


class TestPyleafletColormap:
    config = {
        "station_id_column_name": "station",
    }
    stats = xr.DataArray(
        [0, 1, 2],
        coords={
            "station": [0, 1, 2],
        },
    )

    def test_default_creation(self):
        lf.PyleafletColormap()

    def test_creation_with_stats(self):
        lf.PyleafletColormap(
            self.config, self.stats, colormap_style="plasma", range=[1, 2]
        )

    def test_creation_wrong_colormap(self):
        with pytest.raises(KeyError):
            lf.PyleafletColormap(
                self.config, self.stats, colormap_style="awdawd", range=[1, 2]
            )

    def test_fail_creation(self):
        config = {}
        with pytest.raises(AssertionError):
            lf.PyleafletColormap(config, self.stats)

    def test_default_style(self):
        colormap = lf.PyleafletColormap(self.config)
        style_fct = colormap.style_callback()
        style_fct(feature={})

    def test_stats_style(self):
        feature = {
            "properties": {
                "station": 2,
            }
        }
        colormap = lf.PyleafletColormap(self.config, self.stats)
        style_fct = colormap.style_callback()
        style_fct(feature)

    def test_stats_style_fail(self):
        feature = {
            "properties": {
                "station": 4,
            }
        }
        colormap = lf.PyleafletColormap(self.config, self.stats, empty_color="black")
        style_fct = colormap.style_callback()
        style = style_fct(feature)
        assert style["fillColor"] == "black"

    def test_stats_style_default(self):
        colormap = lf.PyleafletColormap(default_color="blue")
        style_fct = colormap.style_callback()
        style = style_fct({})
        assert style["fillColor"] == "blue"

    def test_legend(self):
        colormap = lf.PyleafletColormap(self.config, self.stats)
        colormap.legend()


class TestLeafletMap:
    def test_creation(self):
        lf.LeafletMap()

    def test_output(self):
        map = lf.LeafletMap()
        map.output()

    def test_add_geolayer(self):
        map = lf.LeafletMap()
        widgets = {}
        colormap = lf.PyleafletColormap()
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
        map.add_geolayer(gdf, colormap, widgets)
