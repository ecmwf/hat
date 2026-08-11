"""Additional tests for hat.interactive.figures — focused on edge cases."""

import pytest

from hat.interactive.figures import ForecastFigure, PlotlyTraceStyleCollection, deep_update


class TestDeepUpdateEdgeCases:
    def test_mutates_original(self):
        original = {"a": 1}
        deep_update(original, {"b": 2})
        assert original == {"a": 1, "b": 2}

    def test_list_values_replaced_not_merged(self):
        original = {"items": [1, 2, 3]}
        result = deep_update(original, {"items": [4, 5]})
        assert result == {"items": [4, 5]}

    def test_none_values(self):
        result = deep_update({"a": 1}, {"a": None})
        assert result == {"a": None}


class TestPlotlyTraceStyleCollectionEdgeCases:
    def test_multiple_updates_accumulate(self):
        coll = PlotlyTraceStyleCollection()
        coll.update_hat_style("t", {"a": 1})
        coll.update_hat_style("t", {"b": 2})
        assert coll.get_hat_style("t") == {"a": 1, "b": 2}

    def test_set_replaces_entirely(self):
        coll = PlotlyTraceStyleCollection({"t": {"a": 1, "b": 2}})
        coll.set_hat_style("t", {"c": 3})
        assert coll.get_hat_style("t") == {"c": 3}


class TestForecastFigureEdgeCases:
    def test_kwargs_override_style(self):
        fig = ForecastFigure()
        fig._hat_traces.set_hat_style("s", {"mode": "lines", "line": {"color": "red"}})
        fig.add_hat_trace("s", x=[1], y=[2], name="x", line=dict(color="blue"))
        # kwargs should override style
        assert fig.data[0].line.color == "blue"

    def test_empty_data(self):
        fig = ForecastFigure()
        fig.add_hat_trace("aifl", x=[], y=[])
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 0
