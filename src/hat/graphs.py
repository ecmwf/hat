import pandas as pd
import plotly.express as px


def graph_sims_and_obs(
    sims,
    obs,
    ID,
    sims_data_name="simulation_timeseries",
    obs_data_name="obsdis",
    height=500,
    width=1200,
):
    # observations, simulations, time
    o = obs.sel(station=ID)[obs_data_name].values
    s = sims.sel(station=ID)[sims_data_name].values
    t = obs.sel(station=ID).time.values

    df = pd.DataFrame({"time": t, "simulations": s, "observations": o})
    fig = px.line(
        df,
        x="time",
        y=["simulations", "observations"],
        title="Simulations & Observations",
    )
    fig.data[0].line.color = "#34eb7d"
    fig.data[1].line.color = "#3495eb"
    fig.update_layout(height=height, width=width)
    fig.update_yaxes(title_text="discharge")
    fig.show()
