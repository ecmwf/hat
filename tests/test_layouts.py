"""Tests for hat.interactive.figures — layout, styles, and figure classes."""

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import pytest

from hat.interactive.figures import (
    AIFLForecastStyles,
    DetForecastFigure,
    EnsembleForecastFigure,
    ForecastFigure,
    PlotlyTraceStyleCollection,
    deep_update,
)


# ---------------------------------------------------------------------------
# deep_update
# ---------------------------------------------------------------------------


class TestDeepUpdate:
    def test_flat_merge(self):
        assert deep_update({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        original = {"a": {"x": 1, "y": 2}, "b": 3}
        assert deep_update(original, {"a": {"y": 99}}) == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_non_overlapping_keys(self):
        assert deep_update({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overwrite_dict_with_scalar(self):
        assert deep_update({"a": {"x": 1}}, {"a": "replaced"}) == {"a": "replaced"}

    def test_overwrite_scalar_with_dict(self):
        assert deep_update({"a": 1}, {"a": {"nested": True}}) == {"a": {"nested": True}}

    def test_empty_update(self):
        assert deep_update({"a": 1}, {}) == {"a": 1}

    def test_empty_original(self):
        assert deep_update({}, {"a": 1}) == {"a": 1}

    def test_deeply_nested(self):
        original = {"a": {"b": {"c": {"d": 1}}}}
        result = deep_update(original, {"a": {"b": {"c": {"e": 2}}}})
        assert result == {"a": {"b": {"c": {"d": 1, "e": 2}}}}


# ---------------------------------------------------------------------------
# PlotlyTraceStyleCollection
# ---------------------------------------------------------------------------


class TestPlotlyTraceStyleCollection:
    def test_init_empty(self):
        coll = PlotlyTraceStyleCollection()
        assert coll.get_hat_style("anything") == {}

    def test_init_with_styles(self):
        styles = {"line1": {"color": "red"}}
        coll = PlotlyTraceStyleCollection(styles)
        assert coll.get_hat_style("line1") == {"color": "red"}

    def test_set_and_get(self):
        coll = PlotlyTraceStyleCollection()
        coll.set_hat_style("trace_a", {"mode": "lines", "width": 2})
        assert coll.get_hat_style("trace_a") == {"mode": "lines", "width": 2}

    def test_set_overwrites(self):
        coll = PlotlyTraceStyleCollection({"t": {"a": 1}})
        coll.set_hat_style("t", {"b": 2})
        assert coll.get_hat_style("t") == {"b": 2}

    def test_update_merges_nested(self):
        coll = PlotlyTraceStyleCollection({"t": {"line": {"color": "red", "width": 1}}})
        coll.update_hat_style("t", {"line": {"width": 3}})
        assert coll.get_hat_style("t") == {"line": {"color": "red", "width": 3}}

    def test_update_creates_if_missing(self):
        coll = PlotlyTraceStyleCollection()
        coll.update_hat_style("new_trace", {"mode": "markers"})
        assert coll.get_hat_style("new_trace") == {"mode": "markers"}

    def test_from_dict(self):
        styles = {"a": {"mode": "lines"}, "b": {"mode": "markers"}}
        coll = PlotlyTraceStyleCollection.from_dict(styles)
        assert coll.get_hat_style("a") == {"mode": "lines"}
        assert coll.get_hat_style("b") == {"mode": "markers"}

    def test_from_yaml(self, tmp_path):
        import yaml

        styles = {"trace1": {"color": "blue"}, "trace2": {"color": "green"}}
        f = tmp_path / "styles.yaml"
        f.write_text(yaml.dump(styles))
        coll = PlotlyTraceStyleCollection.from_yaml(f)
        assert coll.get_hat_style("trace1") == {"color": "blue"}


# ---------------------------------------------------------------------------
# AIFLForecastStyles
# ---------------------------------------------------------------------------


class TestAIFLForecastStyles:
    def test_default_styles_present(self):
        styles = AIFLForecastStyles()
        assert styles.get_hat_style("aifl")["mode"] == "lines"
        assert styles.get_hat_style("aifl")["name"] == "AIFL"
        assert "rl_2.0" in styles._hat_trace_styles
        assert "rl_5.0" in styles._hat_trace_styles
        assert "rl_20.0" in styles._hat_trace_styles

    def test_override_merges(self):
        styles = AIFLForecastStyles(hat_trace_styles={"aifl": {"line": {"color": "green"}}})
        # Should merge: mode still "lines", but color changed
        assert styles.get_hat_style("aifl")["mode"] == "lines"
        assert styles.get_hat_style("aifl")["line"]["color"] == "green"

    def test_add_custom_style(self):
        styles = AIFLForecastStyles(hat_trace_styles={"my_trace": {"mode": "markers"}})
        assert styles.get_hat_style("my_trace") == {"mode": "markers"}


# ---------------------------------------------------------------------------
# ForecastFigure
# ---------------------------------------------------------------------------


class TestForecastFigure:
    def test_is_figure_widget(self):
        fig = ForecastFigure()
        assert isinstance(fig, go.FigureWidget)

    def test_base_layout_applied(self):
        fig = ForecastFigure()
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.height == 500

    def test_layout_override(self):
        fig = ForecastFigure(layout_overrides={"height": 300, "plot_bgcolor": "gray"})
        assert fig.layout.height == 300
        assert fig.layout.plot_bgcolor == "gray"

    def test_add_hat_trace_creates_scatter(self):
        fig = ForecastFigure()
        fig.add_hat_trace("aifl", x=[1, 2, 3], y=[4, 5, 6])
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_add_hat_trace_applies_style(self):
        fig = ForecastFigure()
        fig.add_hat_trace("aifl", x=[1], y=[2])
        # AIFL default style applies line styling and mode
        assert fig.data[0].line.color is not None
        assert fig.data[0].mode == "lines"

    def test_add_hat_trace_unknown_style(self):
        fig = ForecastFigure()
        fig.add_hat_trace("unknown_key", x=[1], y=[2])
        assert len(fig.data) == 1
        assert fig.data[0].name == "unknown_key"

    def test_add_hat_trace_dedup_by_name(self):
        fig = ForecastFigure()
        # Use a style without name override to test dedup
        fig._hat_traces.set_hat_style("custom", {"mode": "lines"})
        fig.add_hat_trace("custom", x=[1, 2], y=[3, 4], name="series1")
        fig.add_hat_trace("custom", x=[5, 6], y=[7, 8], name="series1")
        assert len(fig.data) == 1

    def test_add_hat_trace_different_names(self):
        fig = ForecastFigure()
        fig._hat_traces.set_hat_style("s", {"mode": "lines"})
        fig.add_hat_trace("s", x=[1], y=[2], name="a")
        fig.add_hat_trace("s", x=[3], y=[4], name="b")
        assert len(fig.data) == 2

    def test_style_overrides_in_constructor(self):
        fig = ForecastFigure(style_overrides={"aifl": {"line": {"color": "red"}}})
        fig.add_hat_trace("aifl", x=[1], y=[2])
        assert fig.data[0].line.color == "red"


# ---------------------------------------------------------------------------
# DetForecastFigure
# ---------------------------------------------------------------------------


class TestDetForecastFigure:
    @pytest.fixture
    def dates(self):
        base = datetime(2026, 1, 1)
        return [base + timedelta(hours=i) for i in range(5)]

    def test_hat_plot_with_thresholds(self, dates):
        fig = DetForecastFigure()
        thresholds = {"rl_2.0": 100.0, "rl_5.0": 200.0, "rl_20.0": 500.0}
        fig.hat_plot(dates, [10, 20, 30, 40, 50], thresholds)
        # 3 threshold traces + 1 AIFL trace = 4
        assert len(fig.data) == 4

    def test_hat_plot_with_none_thresholds(self, dates):
        fig = DetForecastFigure()
        thresholds = {"rl_2.0": None, "rl_5.0": 200.0, "rl_20.0": None}
        fig.hat_plot(dates, [10, 20, 30, 40, 50], thresholds)
        # Only rl_5.0 + AIFL = 2
        assert len(fig.data) == 2

    def test_hat_plot_no_thresholds(self, dates):
        fig = DetForecastFigure()
        fig.hat_plot(dates, [10, 20, 30, 40, 50], thresholds=None)
        # Only AIFL trace
        assert len(fig.data) == 1

    def test_threshold_values_are_constant(self, dates):
        fig = DetForecastFigure()
        thresholds = {"rl_2.0": 42.0, "rl_5.0": None, "rl_20.0": None}
        fig.hat_plot(dates, [1, 2, 3, 4, 5], thresholds)
        # First trace is the threshold
        threshold_trace = fig.data[0]
        assert all(v == 42.0 for v in threshold_trace.y)
        assert len(threshold_trace.y) == len(dates)


# ---------------------------------------------------------------------------
# EnsembleForecastFigure
# ---------------------------------------------------------------------------


class TestEnsembleForecastFigure:
    @pytest.fixture
    def dates(self):
        base = datetime(2026, 1, 1)
        return [base + timedelta(hours=i) for i in range(10)]

    @pytest.fixture
    def ens_data(self):
        """10 timesteps x 5 members."""
        rng = np.random.default_rng(123)
        return rng.uniform(10, 100, size=(10, 5))

    def test_hovermode_closest(self):
        fig = EnsembleForecastFigure()
        assert fig.layout.hovermode == "closest"

    def test_add_ensemble_traces_count(self, dates, ens_data):
        fig = EnsembleForecastFigure()
        fig.add_ensemble_traces(dates, ens_data)
        # 2 (min-max band) + 2 (IQR band) + 5 members + 1 mean = 10
        assert len(fig.data) == 10

    def test_add_ensemble_traces_highlight_control(self, dates, ens_data):
        fig = EnsembleForecastFigure()
        fig.add_ensemble_traces(dates, ens_data, highlight_control=True)
        # Check member 0 is named "cf"
        # Members start at index 4 (after 2 min-max + 2 IQR bands)
        assert fig.data[4].name == "cf"
        assert fig.data[4].line.width == 3

    def test_hat_plot_with_thresholds(self, dates, ens_data):
        fig = EnsembleForecastFigure()
        thresholds = {"rl_2.0": 50.0, "rl_5.0": 80.0, "rl_20.0": None}
        # forecast as list of lists (transposed from ens_data)
        forecast = ens_data.tolist()
        fig.hat_plot(dates, forecast, thresholds)
        # 2 thresholds + ensemble traces (2+2+5+1=10) = 12
        assert len(fig.data) == 12

    def test_mean_trace_is_last(self, dates, ens_data):
        fig = EnsembleForecastFigure()
        fig.add_ensemble_traces(dates, ens_data)
        assert fig.data[-1].name == "Mean"
        # Verify mean values
        expected_mean = ens_data.mean(axis=1)
        np.testing.assert_array_almost_equal(fig.data[-1].y, expected_mean)
