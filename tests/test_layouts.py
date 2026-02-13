from hat.interactive.figures import StyledLayout

import pytest


FONT = dict(family="Roboto", size=12, color="rgb(82, 82, 82)")
PLOT_STYLES = {
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
        tickfont=FONT,
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
        tickfont=FONT,
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
        font=FONT,
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
        font={**FONT, "color": "black", "size": 14},
    ),
)


@pytest.fixture
def test_layout():
    return StyledLayout.from_dict(
        {
            **base_layout,
            "hat_trace_styles": PLOT_STYLES,
        }
    )


def test_layout_creation():
    layout = StyledLayout(
        **base_layout,
        hat_trace_styles=PLOT_STYLES,
    )

    assert isinstance(layout, StyledLayout)
    assert layout.xaxis.title.text == "Valid Time"
    assert layout.yaxis.title.text == "River discharge (m³/s)"

    # a plotly defaults
    assert layout.plotly_name == "layout"


def test_layout_operations(test_layout):
    layout = test_layout

    assert layout.xaxis.title.text == "Valid Time"
    assert layout.yaxis.title.text == "River discharge (m³/s)"
    assert layout.legend.font.size == FONT["size"]
    assert layout.get_hat_style("aifl")["name"] == "AIFL"
    assert not layout.get_hat_style("wrong")

    layout.update_hat_style("aifl", dict(name="Updated AIFL"))
    assert layout.get_hat_style("aifl")["name"] == "Updated AIFL"
    assert "mode" in layout.get_hat_style("aifl")

    layout.set_hat_style("aifl", dict(name="Updated AIFL"))
    assert layout.get_hat_style("aifl")["name"] == "Updated AIFL"
    assert "mode" not in layout.get_hat_style("aifl")
