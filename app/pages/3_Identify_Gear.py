"""Guess the gear type from where, when and how a fleet operated (vessel size columns are NOT used)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import BLUE, MONTHS, build_row, footer, load_json, load_model, load_sample

st.set_page_config(page_title="Identify gear type", layout="wide")
st.title("Identify the gear type")
st.write("Give the location, month, flag and activity of a fleet in one 0.1 degree cell. The model ranks the 16 gear types. "
         "It does **not** see vessel size, because those columns are averages per gear and would give the answer away.")

df = load_sample()
left, right = st.columns([1, 2])
with left:
    lat = st.number_input("Latitude", -90.0, 90.0, 56.0, step=0.1, format="%.1f")
    lon = st.number_input("Longitude", -180.0, 180.0, 3.0, step=0.1, format="%.1f")
    month = st.selectbox("Month", range(1, 13), index=2, format_func=lambda m: MONTHS[m - 1])
    flags = df["flag"].value_counts().index.tolist()
    flag = st.selectbox("Flag state", flags)
    hours = st.number_input("Total vessel hours in the cell and month", 0.0, 200000.0, 20.0, step=1.0)
    fishing = st.number_input("Of which fishing hours", 0.0, float(hours), min(10.0, float(hours)), step=1.0)
    mmsi = st.number_input("Vessels present", 1, 600, 2, step=1)

model = load_model("gear")
row = build_row(lat, lon, month, flag, df["geartype"].iloc[0], hours=hours, mmsi=mmsi, fishing_hours=fishing)
proba = model["pipeline"].predict_proba(row[model["columns"]])[0]
out = pd.DataFrame({"Gear type": model["pipeline"].classes_, "Probability": proba}).sort_values("Probability", ascending=False)

with right:
    top = out.iloc[0]
    st.metric("Most likely gear type", top["Gear type"], f"{top['Probability']:.0%} probability", delta_color="off")
    show = out.head(6).sort_values("Probability")
    fig = px.bar(show, x="Probability", y="Gear type", orientation="h", title="Six most likely gear types")
    fig.update_traces(marker_color=BLUE)
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

test = load_json("test_results.json")
if test:
    st.caption(f"On ocean blocks never seen in training the model scored a macro-F1 of {test['C']['macro_f1']:.2f} "
               "(guessing: 0.06). Several GFW gear classes overlap in meaning, so treat the top few as candidates, not a verdict.")
footer()
