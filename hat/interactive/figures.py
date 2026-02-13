from datetime import datetime
from pathlib import Path
import yaml

import plotly.graph_objects as go


def deep_update(original: dict, update: dict) -> dict:
    # this may be part of ecpyutil some day
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(original.get(key, None), dict):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original


class PlotlyTraceStyleCollection:
    def __init__(self, hat_trace_styles=None):
        self._hat_trace_styles = hat_trace_styles or {}

    def get_hat_style(self, trace_name: str) -> dict:
        return self._hat_trace_styles.get(trace_name, {})

    def set_hat_style(self, trace_name: str, options: dict):
        self._hat_trace_styles[trace_name] = options

    def update_hat_style(self, trace_name: str, updates: dict):
        if trace_name not in self._hat_trace_styles:
            self._hat_trace_styles[trace_name] = {}
        self._hat_trace_styles[trace_name] = deep_update(self._hat_trace_styles[trace_name], updates)

    @classmethod
    def from_yaml(cls, filepath: Path | str):
        with open(filepath, "r") as f:
            yaml_config = yaml.safe_load(f)
        return cls.from_dict(yaml_config)

    @classmethod
    def from_dict(cls, layout_dict: dict):
        layout_kwargs = {k: v for k, v in layout_dict.items() if k != "hat_trace_styles"}
        hat_trace_styles = layout_dict.get("hat_trace_styles", {})
        return cls(hat_trace_styles=hat_trace_styles, **layout_kwargs)


class AIFLForecastStyles(PlotlyTraceStyleCollection):
    """pre-configured layout for AIFL deterministic forecast plots, with sensible defaults for axes, titles, etc."""

    _aifl_trace_styles = {
        "aifl": dict(
            mode="lines",
            name="AIFL",
            line=dict(color="blue", width=2.5, dash="solid"),
            legendgroup="forecasts",
            legendgrouptitle=dict(text="Forecasts"),
            hovertemplate="%{y:,.1f}",
        ),
        "rl_2.0": dict(
            mode="lines",
            line=dict(color="#E9E515", width=2.5, dash="dash"),
            legendgroup="Thresholds",
            legendgrouptitle=dict(text="Thresholds"),
            hovertemplate="%{y:,.0f}",
        ),
        "rl_5.0": dict(
            mode="lines",
            line=dict(color="#F24122", width=2.5, dash="dash"),
            legendgroup="Thresholds",
            legendgrouptitle=dict(text="Thresholds"),
            hovertemplate="%{y:,.0f}",
        ),
        "rl_20.0": dict(
            mode="lines",
            line=dict(color="#730573", width=2.5, dash="dash"),
            legendgroup="Thresholds",
            legendgrouptitle=dict(text="Thresholds"),
            hovertemplate="%{y:,.0f}",
        ),
    }

    def __init__(self, hat_trace_styles=None):
        super().__init__(hat_trace_styles=self._aifl_trace_styles)
        for trace_name, option in (hat_trace_styles or {}).items():
            self.update_hat_style(trace_name, option)


class DetForecastFigure(go.FigureWidget):
    """wrapper object to hold base plot layout and styles for common elements for a forecast plot.
    Extends the data handling notion of a normal plotly Figure

    Allow configuration/labelling of axes, titles, etc. + overriding layout defaults via configuration
    """

    base_font = dict(family="Roboto", size=12, color="rgb(82, 82, 82)")
    base_layout = dict(
        xaxis=dict(
            title="Valid Time",
            showline=True,
            showgrid=True,
            showticklabels=True,
            autorange=True,
            fixedrange=True,
            gridcolor="#c4c3c3",
            gridwidth=1,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            tickfont=base_font,
            tickformat="%d-%m-%Y",
        ),
        yaxis=dict(
            title="River discharge (m³/s)",
            showline=True,
            showgrid=True,
            gridcolor="#c4c3c3",
            gridwidth=1,
            rangemode="nonnegative",
            showticklabels=True,
            linecolor="rgb(180, 180, 180)",
            linewidth=2,
            ticks="outside",
            tickfont=base_font,
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        hovermode="x unified",
        legend=dict(
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=base_font,
            groupclick="toggleitem",
        ),
        height=500,
        title=dict(
            y=1,
            yref="paper",
            automargin=True,
            xanchor="left",
            x=0,
            yanchor="top",
            font={**base_font, "color": "black", "size": 14},
        ),
    )

    def __init__(self, *args, style_overrides=None, layout_overrides=None, **kwargs):
        self._hat_traces = AIFLForecastStyles(hat_trace_styles=style_overrides)
        layout = deep_update(self.base_layout, layout_overrides or {})
        layout = go.Layout(layout)
        super().__init__(*args, layout=layout, **kwargs)

    def add_hat_trace(self, trace_name: str, x, y, **kwargs):
        trace_style = self._hat_traces.get_hat_style(trace_name)
        trace_kwargs = {"x": x, "y": y, **trace_style, **kwargs}
        self.add_trace(go.Scatter(**trace_kwargs))

    def hat_plot(
        self, valid_dates: list[datetime], forecast: list[float], thresholds: dict[str, float | None], **kwargs
    ):
        for thres, label in zip(["rl_2.0", "rl_5.0", "rl_20.0"], ["2-yr RP", "5-yr RP", "20-yr RP"]):
            if thresholds[thres] is not None:
                yvals = [float(thresholds[thres])] * len(valid_dates)
                self.add_hat_trace(thres, x=valid_dates, y=yvals, name=label)
        self.add_hat_trace("aifl", x=valid_dates, y=forecast, name="AIFL")


class EnsembleForecastFigure(go.FigureWidget):
    """fan style of plot for ensemble forecast data"""


class EnsembleBoxPlotForecastFigure(go.FigureWidget):
    """box-plot style of plot for ensemble forecast data"""
