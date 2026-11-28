"""Predict fishing for a chosen place, month, gear type and flag (feature set A models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (BLUE, MONTHS, assign_cluster, build_row, cluster_profile, footer, load_json,
                   load_sample, predict_fishing)

st.set_page_config(page_title="Predict fishing", layout="wide")
st.title("Predict fishing")
st.write("Choose a place, month, gear type and flag state. The models estimate **whether any fishing occurs** in that "
         "0.1 degree cell and month, and **how many hours** if it does. They do not need to know the vessels' total hours, "
         "so this is the deployable version of the model.")

df = load_sample()
PRESETS = {"East China Sea": (30.0, 125.0), "North Sea": (56.0, 3.0), "Bay of Biscay": (45.0, -5.0),
           "Gulf of Alaska": (57.0, -150.0), "Southwest Atlantic (Patagonian shelf)": (-45.0, -60.0),
           "Equatorial Pacific": (0.0, -140.0), "Custom": None}

left, right = st.columns([1, 2])
with left:
    preset = st.selectbox("Location preset", list(PRESETS))
    lat0, lon0 = PRESETS[preset] or (30.0, 125.0)
    lat = st.number_input("Latitude (degrees, south is negative)", -90.0, 90.0, lat0, step=0.1, format="%.1f")
    lon = st.number_input("Longitude (degrees, west is negative)", -180.0, 180.0, lon0, step=0.1, format="%.1f")
    month = st.selectbox("Month", range(1, 13), index=6, format_func=lambda m: MONTHS[m - 1])
    gears = sorted(df["geartype"].unique())
    gear = st.selectbox("Gear type", gears, index=gears.index("trawlers") if "trawlers" in gears else 0)
    flags = df["flag"].value_counts().index.tolist()
    flag = st.selectbox("Flag state", flags, index=flags.index("CHN") if "CHN" in flags else 0)

row = build_row(lat, lon, month, flag, gear)
p, hours = predict_fishing(row)
test = load_json("test_results.json")
thr = test["A"]["Set A (deployable)"]["threshold"] if test else 0.5

with right:
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of fishing", f"{p:.0%}", help="Share of similar cases in training that had fishing, as estimated by the model.")
    c2.metric("Hours if fishing occurs", f"{hours:.1f} h", help="Typical hours in the cell and month when fishing does occur.")
    c3.metric("Expected hours", f"{p * hours:.1f} h", help="Probability x hours. A rough ranking score, not a forecast.")
    if p >= thr:
        st.success(f"Above the model's decision threshold ({thr:.2f}): fishing is **likely** here for this gear, flag and month.")
    else:
        st.info(f"Below the model's decision threshold ({thr:.2f}): fishing is **less likely** here for this gear, flag and month.")

    # the same place, all twelve months
    rows = [build_row(lat, lon, m, flag, gear) for m in range(1, 13)]
    series = pd.DataFrame({"Month": [MONTHS[m - 1][:3] for m in range(1, 13)],
                           "Probability": [predict_fishing(r)[0] for r in rows]})
    fig = px.line(series, x="Month", y="Probability", markers=True, title="Probability of fishing across the year")
    fig.update_traces(line_color=BLUE, line_width=2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("Fishing-ground cluster")
k = assign_cluster(lat, lon, np.log1p(hours))
prof = cluster_profile()
st.write(f"If fishing occurs, this cell resembles **cluster {k}**: typically centred near latitude "
         f"{prof.loc[k, 'centre_lat']:.0f}, longitude {prof.loc[k, 'centre_lon']:.0f}, with a median of "
         f"{prof.loc[k, 'median_fishing_hours']:.1f} fishing hours. Clusters group cells by location and fishing intensity.")
with st.expander("All clusters"):
    st.dataframe(prof, width="stretch")

with st.expander("What the model assumed"):
    if not row.attrs.get("exact_fleet", True):
        st.caption(f"The sample has no {flag} vessels using {gear}; typical values for {gear} were used instead.")
    st.dataframe(row[["n_vessels", "mean_length_m", "share_self_reported", "mean_class_score"]].round(2), hide_index=True)
    st.caption("The models were trained with flag x gear averages of vessel size and reporting, so the app looks these up "
               "from the sample.")
    if test:
        st.caption(f"Test ROC-AUC on ocean blocks never seen in training: {test['A']['Set A (deployable)']['roc_auc']:.2f}. "
                   "Individual places can be much better or worse than this average.")
footer()
