from datetime import datetime
from pathlib import Path
import yaml

import numpy as np

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


class ForecastFigure(go.FigureWidget):
    """base figure for forecast plots, with sensible defaults for axes, titles, etc."""

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

    def add_hat_trace(self, trace_name: str, x, y, name=None, **kwargs):
        trace_style = self._hat_traces.get_hat_style(trace_name)
        tname = name or trace_name
        trace_kwargs = {"x": x, "y": y, "name": tname, **trace_style, **kwargs}
        trace_exists = any([trace.name == tname for trace in self.data])
        if trace_exists:
            for trace in self.data:
                if trace.name == tname:
                    trace.update(trace_kwargs)
        else:
            self.add_trace(go.Scatter(**trace_kwargs))


class DetForecastFigure(ForecastFigure):
    """wrapper object to hold base plot layout and styles for common elements for a forecast plot.
    Extends the data handling notion of a normal plotly Figure

    Allow configuration/labelling of axes, titles, etc. + overriding layout defaults via configuration
    """

    def hat_plot(
        self, valid_dates: list[datetime], forecast: list[float], thresholds: dict[str, float | None], **kwargs
    ):
        if thresholds is not None:
            for thres, label in zip(["rl_2.0", "rl_5.0", "rl_20.0"], ["2-yr RP", "5-yr RP", "20-yr RP"]):
                if thresholds[thres] is not None:
                    yvals = [float(thresholds[thres])] * len(valid_dates)
                    self.add_hat_trace(thres, x=valid_dates, y=yvals, name=label)
        self.add_hat_trace("aifl", x=valid_dates, y=forecast, name="AIFL")


class EnsembleForecastFigure(ForecastFigure):
    """fan style of plot for ensemble forecast data"""

    def __init__(self, *args, style_overrides=None, layout_overrides=None, **kwargs):
        self.base_layout = deep_update(self.base_layout, {"hovermode": "closest"})
        super().__init__(*args, style_overrides=style_overrides, layout_overrides=layout_overrides, **kwargs)

    def add_ensemble_traces(self, valid_dates: list[datetime], ens_data=np.ndarray["time", "number"]):
        y_max = ens_data.max(axis=1)
        y_min = ens_data.min(axis=1)
        y_q25 = np.quantile(ens_data, 0.25, axis=1)
        y_q75 = np.quantile(ens_data, 0.75, axis=1)
        y_max = ens_data.max(axis=1)
        y_mean = ens_data.mean(axis=1)

        # Min-Max band
        self.add_trace(
            go.Scatter(
                x=valid_dates,
                y=y_min,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="_min",
            )
        )
        self.add_trace(
            go.Scatter(
                x=valid_dates,
                y=y_max,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(91,137,197,0.12)",
                name="Min-Max range",
                hoverinfo="skip",
            )
        )

        # IQR band
        self.add_trace(
            go.Scatter(
                x=valid_dates,
                y=y_q25,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="_q25",
            )
        )
        self.add_trace(
            go.Scatter(
                x=valid_dates,
                y=y_q75,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(67,111,177,0.28)",
                name="Interquartile range",
                hoverinfo="skip",
            )
        )

        # Member lines
        custom_stats = np.column_stack([y_min, y_q25, y_q75, y_mean, y_max])
        for member in range(ens_data.shape[1]):
            self.add_trace(
                go.Scatter(
                    x=valid_dates,
                    y=ens_data[:, member],
                    mode="lines+markers",
                    line=dict(color="rgba(45,86,152,0.20)", width=1),
                    marker=dict(size=0, opacity=0),
                    name=f"# {member}",
                    legendgroup="members",
                    showlegend=False,
                    customdata=custom_stats,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Member: <b>%{y:.3f}</b><br>"
                        "Min: %{customdata[0]:.3f} | Max: %{customdata[4]:.3f}<br>"
                        "IQR: %{customdata[1]:.3f} \u2013 %{customdata[2]:.3f}<br>"
                        "Mean: %{customdata[3]:.3f}<extra>%{fullData.name}</extra>"
                    ),
                    hoverlabel=dict(bgcolor="rgba(25,47,89,0.95)", font=dict(color="white")),
                )
            )

        # Mean line on top
        self.add_trace(
            go.Scatter(
                x=valid_dates,
                y=y_mean,
                mode="lines",
                line=dict(color="rgb(8,48,107)", width=3),
                name="Mean",
                customdata=custom_stats,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean: <b>%{y:.3f}</b><br>"
                    "Min: %{customdata[0]:.3f} | Max: %{customdata[4]:.3f}<br>"
                    "IQR: %{customdata[1]:.3f} \u2013 %{customdata[2]:.3f}<extra>Mean</extra>"
                ),
            )
        )

    def hat_plot(
        self, valid_dates: list[datetime], forecast: list[list[float]], thresholds: dict[str, float | None], **kwargs
    ):
        if thresholds is not None:
            for thres, label in zip(["rl_2.0", "rl_5.0", "rl_20.0"], ["2-yr RP", "5-yr RP", "20-yr RP"]):
                if thresholds[thres] is not None:
                    yvals = [float(thresholds[thres])] * len(valid_dates)
                    self.add_hat_trace(thres, x=valid_dates, y=yvals, name=label)
        self.add_ensemble_traces(valid_dates, np.array(forecast))


class EnsembleBoxPlotForecastFigure(ForecastFigure):
    """box-plot style of plot for ensemble forecast data"""
