"""
Shared helpers for the Streamlit app: loading data and models and building model inputs.

The app never retrains anything. It loads the models saved by notebook 03 and the
results saved by notebooks 03 and 04, and re-uses src/features.py so that a row typed
into the app is prepared exactly like a row in the training data.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402

APP_DATA = ROOT / "data" / "processed" / "fishing_app_sample.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"
BLUES = [[0, "#cde2fb"], [0.25, "#6da7ec"], [0.5, "#2a78d6"], [0.75, "#184f95"], [1, "#0d366b"]]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

CITATION = ("Data: Global Fishing Watch (2025), Global AIS-based Apparent Fishing Effort Dataset v3.0, "
            "https://doi.org/10.5281/zenodo.14982712 (CC BY-NC 4.0). 2024 data are provisional.")


def footer():
    st.divider()
    st.caption(CITATION)


# ---------- loading (cached so each file is read once) ----------
@st.cache_data
def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(APP_DATA)
    df["flag"] = df["flag"].astype(str)
    df["geartype"] = df["geartype"].astype(str)
    return F.add_targets(df)


@st.cache_resource
def load_model(name: str) -> dict:
    return joblib.load(MODELS / f"{name}.joblib")


@st.cache_data
def load_json(name: str):
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def show_figure(name: str, caption: str = ""):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# ---------- building model inputs ----------
@st.cache_data
def fleet_tables():
    """Typical vessel features for each (flag, gear) pair and for each gear.
    The models were trained with these flag x gear averages, so the app looks them up."""
    df = load_sample()
    cols = ["n_vessels", "mean_length_m", "share_self_reported", "mean_class_score"]
    return df.groupby(["flag", "geartype"])[cols].median(), df.groupby("geartype")[cols].median()


def build_row(lat, lon, month, flag, gear, hours=0.0, mmsi=1, fishing_hours=0.0):
    """One-row DataFrame with every column the models can ask for."""
    by_pair, by_gear = fleet_tables()
    exact = (flag, gear) in by_pair.index
    v = by_pair.loc[(flag, gear)] if exact else by_gear.loc[gear]
    row = pd.DataFrame([{
        "cell_ll_lat": lat, "cell_ll_lon": lon, "month": month, "flag": flag, "geartype": gear,
        "hours": hours, "mmsi_present": mmsi, "fishing_hours": fishing_hours,
        "n_vessels": v["n_vessels"], "mean_length_m": v["mean_length_m"],
        "share_self_reported": v["share_self_reported"], "mean_class_score": v["mean_class_score"],
    }])
    row = F.engineer(F.add_targets(row))
    row.attrs["exact_fleet"] = exact
    return row


def predict_fishing(row: pd.DataFrame):
    """Returns (probability of any fishing, predicted hours if fishing occurs)."""
    occ, hrs = load_model("occurrence_A"), load_model("hours_A")
    p = float(occ["pipeline"].predict_proba(row[occ["columns"]])[0, 1])
    hours = float(np.expm1(hrs["pipeline"].predict(row[hrs["columns"]])[0]))   # undo log1p
    return p, max(hours, 0.0)


def _sphere(lat, lon, log_hours):
    """Latitude/longitude on a sphere, so the date line is not a gap (as in notebook 03)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return pd.DataFrame({"x": np.cos(la) * np.cos(lo), "y": np.cos(la) * np.sin(lo),
                         "z": np.sin(la), "log_fishing_hours": np.asarray(log_hours, float)})


def assign_cluster(lat, lon, log_hours) -> int:
    c = load_model("clusters")
    return int(c["kmeans"].predict(c["scaler"].transform(_sphere([lat], [lon], [log_hours])))[0])


@st.cache_data
def cluster_profile() -> pd.DataFrame:
    """Where each fishing-ground cluster sits, summarised on the app sample."""
    df = load_sample()
    pos = df[df["any_fishing"] == 1]
    c = load_model("clusters")
    labels = c["kmeans"].predict(c["scaler"].transform(
        _sphere(pos["cell_ll_lat"], pos["cell_ll_lon"], pos["log_fishing_hours"])))
    return (pos.assign(cluster=labels).groupby("cluster")
            .agg(rows=("cluster", "size"), centre_lat=("cell_ll_lat", "median"),
                 centre_lon=("cell_ll_lon", "median"), median_fishing_hours=("fishing_hours", "median")).round(1))
