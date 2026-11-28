"""Explore the app sample: filter by gear and month, see maps and summaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import BLUE, BLUES, MONTHS, footer, load_sample

st.set_page_config(page_title="Explore the data", layout="wide")
st.title("Explore the data")
st.caption("A 50,000-row stratified sample (equal numbers per gear type), so gear counts are not real-world "
           "prevalence. Compare rates, not counts.")

df = load_sample()

# ---- filters in the sidebar ----
st.sidebar.header("Filters")
gears = sorted(df["geartype"].unique())
chosen = st.sidebar.multiselect("Gear types", gears, default=gears)
m_lo, m_hi = st.sidebar.slider("Months", 1, 12, (1, 12))
d = df[df["geartype"].isin(chosen) & df["month"].between(m_lo, m_hi)]
if d.empty:
    st.warning("No rows match these filters.")
    st.stop()

# ---- headline numbers ----
fish = d[d["any_fishing"] == 1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(d):,}")
c2.metric("Share with fishing", f"{d['any_fishing'].mean():.1%}")
c3.metric("Median hours (when fishing)", f"{fish['fishing_hours'].median():.1f}" if len(fish) else "n/a")
c4.metric("Flag states", f"{d['flag'].nunique()}")

tab_map, tab_gear, tab_month, tab_data = st.tabs(["Map", "By gear type", "By month", "Data"])

with tab_map:
    c_left, c_right = st.columns([3, 2])
    metric = c_left.radio("Map shows", ["Total fishing hours (log10 scale)", "Share of rows with fishing"], horizontal=True)
    step = c_right.select_slider("Square size (degrees)", options=[1, 2, 4], value=2,
                                 help="Smaller squares give more detail but more empty gaps, because the sample is only 50,000 rows.")
    g = d.assign(lat_bin=np.floor(d["cell_ll_lat"] / step) * step + step / 2,
                 lon_bin=np.floor(d["cell_ll_lon"] / step) * step + step / 2)
    grouped = g.groupby(["lat_bin", "lon_bin"])
    if metric.startswith("Total"):
        value, label = np.log10(1 + grouped["fishing_hours"].sum()), "log10(1 + hours)"
    else:
        value, label = grouped["any_fishing"].mean(), "Share with fishing"
    grid = value.unstack("lon_bin").sort_index()   # rows = latitude squares, columns = longitude squares

    # A plain heatmap needs no internet connection (a world-map layer would have to download coastlines).
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns, y=grid.index, colorscale=BLUES, hoverongaps=False,
        colorbar=dict(title=label, thickness=14),
        hovertemplate="Lat %{y:.0f}, Lon %{x:.0f}<br>" + label + ": %{z:.2f}<extra></extra>"))
    fig.update_xaxes(title="Longitude", range=[-180, 180], dtick=30, showgrid=False)
    fig.update_yaxes(title="Latitude", range=[-90, 90], dtick=30, showgrid=False)
    # a faint tint behind the plot makes empty squares (no data) distinguishable from low values
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(140,140,140,0.15)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each square is {step} x {step} degrees. Grey squares have no rows in the filtered sample; "
               "continents appear as grey gaps because ships do not fish on land.")

with tab_gear:
    by = d.groupby("geartype").agg(share_fishing=("any_fishing", "mean"), rows=("any_fishing", "size")).sort_values("share_fishing")
    fig = px.bar(by, x="share_fishing", y=by.index, orientation="h", labels={"share_fishing": "Share of rows with fishing", "y": ""},
                 hover_data={"rows": True})
    fig.update_traces(marker_color=BLUE)
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
    st.plotly_chart(fig, width="stretch")

with tab_month:
    mo = d.groupby("month")["any_fishing"].mean().reset_index()
    mo["Month"] = mo["month"].map(lambda m: MONTHS[m - 1][:3])
    fig = px.line(mo, x="Month", y="any_fishing", markers=True, labels={"any_fishing": "Share of rows with fishing"})
    fig.update_traces(line_color=BLUE, line_width=2)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Seasonal swings are small overall (about 29% to 34% in the full sample); regions and gear types matter more.")

with tab_data:
    cols = ["date", "cell_ll_lat", "cell_ll_lon", "flag", "geartype", "hours", "fishing_hours", "mmsi_present"]
    st.dataframe(d[cols].head(1000), width="stretch", hide_index=True)
    st.download_button("Download filtered rows (CSV)", d[cols].to_csv(index=False), "fishing_filtered.csv", "text/csv")
footer()
