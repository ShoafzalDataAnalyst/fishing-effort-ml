"""
Shared preprocessing for the fishing-effort project.

Everything that turns the raw working sample into model inputs lives here, so the
notebooks and the Streamlit app use exactly the same steps.

Design rules (each is justified in the report):
* Targets are built from `fishing_hours` only.
* Train/test split is by 10 x 10 degree geographic blocks, not random rows,
  because neighbouring cells are strongly spatially autocorrelated.
* Imputers, scalers and encoders are fitted on the training data only (they sit
  inside a scikit-learn Pipeline, so cross-validation refits them per fold).
* Vessel features are flag x gear averages, so they identify the gear type.
  They are therefore NOT used when the task is to predict the gear type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BLOCK_DEG = 10
SEED = 42

# Vessel summary columns joined in make_sample.py
VESSEL_COLS = ["n_vessels", "mean_length_m", "mean_engine_kw", "mean_tonnage_gt",
               "share_self_reported", "mean_class_score"]

# Dropped because |rho| >= 0.90 with mean_length_m (see EDA section 11)
REDUNDANT_VESSEL_COLS = ["mean_engine_kw", "mean_tonnage_gt"]

LOCATION_TIME = ["cell_ll_lat", "cell_ll_lon", "abs_lat", "month_sin", "month_cos"]
VESSEL_KEPT = ["log_n_vessels", "mean_length_m", "share_self_reported", "mean_class_score"]
ACTIVITY = ["log_hours", "mmsi_present"]
CATEGORICAL = ["flag", "geartype"]


def load_sample(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["any_fishing"] = (df["fishing_hours"] > 0).astype(int)
    df["log_fishing_hours"] = np.log1p(df["fishing_hours"])
    df["fishing_share"] = np.where(df["hours"] > 0, df["fishing_hours"] / df["hours"], 0.0)
    return df


def add_geo_blocks(df: pd.DataFrame, deg: int = BLOCK_DEG) -> pd.DataFrame:
    df = df.copy()
    lat_b = np.floor((df["cell_ll_lat"] + 90) / deg).astype(int)
    lon_b = np.floor((df["cell_ll_lon"] + 180) / deg).astype(int)
    df["block"] = lat_b * 1000 + lon_b
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Uses no statistics from other rows, so it is
    safe to apply before the split."""
    df = df.copy()
    df["flag"] = df["flag"].astype(str)
    df["geartype"] = df["geartype"].astype(str)
    df["abs_lat"] = df["cell_ll_lat"].abs()
    # month is cyclic: December is next to January
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # heavy right skew (EDA sections 4 and 10) -> log1p
    df["log_hours"] = np.log1p(df["hours"])
    df["log_n_vessels"] = np.log1p(df["n_vessels"])
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Targets + blocks + engineered features in one call."""
    return engineer(add_geo_blocks(add_targets(df)))


def geo_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Hold out whole geographic blocks. Returns (train_idx, test_idx) as positions."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["block"]))
    return train_idx, test_idx


def feature_sets(with_activity: bool = False, with_vessel: bool = True) -> dict:
    """Column lists for a model. `with_activity` adds hours / mmsi_present."""
    num = list(LOCATION_TIME)
    if with_vessel:
        num += VESSEL_KEPT
    if with_activity:
        num += ACTIVITY
    return {"num": num, "cat": list(CATEGORICAL)}


def make_preprocessor(num_cols, cat_cols, scale: bool = True) -> ColumnTransformer:
    """Median-impute (+ scale) numeric columns; one-hot categorical columns.
    Rare categories (under 1% of rows) are pooled; unseen ones are tolerated."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(steps)
    cat_pipe = OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=0.01, sparse_output=False)
    return ColumnTransformer([("num", num_pipe, list(num_cols)),
                              ("cat", cat_pipe, list(cat_cols))])
