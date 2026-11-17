"""
Build the working sample from the raw Global Fishing Watch v3 files.

Steps
-----
1. Read the 12 monthly CSVs inside the zip, one month at a time (low memory).
2. From each month, keep up to ceil(per_gear / 12) rows per gear type, so every
   gear type ends up with about `per_gear` rows spread evenly over the year.
3. Aggregate the 2024 vessel table by flag x gear type and join it on.
4. Save the sample (CSV + Parquet) and a smaller sample for the Streamlit app.

No imputation or scaling happens here: cleaning is done (and documented) in the
notebooks so it is visible in the analysis.

Usage
-----
    python src/make_sample.py --per-gear 20000
"""
import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sample_fleet(zip_path: Path, per_gear: int, seed: int) -> pd.DataFrame:
    """Stratified sample by gear type within every month."""
    rng = np.random.default_rng(seed)
    per_gear_month = math.ceil(per_gear / 12)
    parts = []

    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise SystemExit(f"No CSV files found inside {zip_path}")
        print(f"Found {len(names)} monthly files; keeping up to "
              f"{per_gear_month:,} rows per gear type per month")

        for name in names:
            with z.open(name) as f:
                df = pd.read_csv(f)
            keep = []
            for _, positions in df.groupby("geartype").indices.items():
                k = min(len(positions), per_gear_month)
                keep.append(rng.choice(positions, size=k, replace=False))
            sampled = df.iloc[np.concatenate(keep)]
            parts.append(sampled)
            print(f"  {Path(name).name}: {len(df):>9,} rows -> {len(sampled):>7,} kept")

    return pd.concat(parts, ignore_index=True)


def vessel_summary(vessels_csv: Path, year: int) -> pd.DataFrame:
    """Summarise the vessel table by flag x gear type (the fleet table's keys)."""
    v = pd.read_csv(vessels_csv, dtype={"flag_ais": "string"}, low_memory=False)
    v = v[v["year"] == year].copy()

    # Robustly turn the True/False column into 1.0/0.0 (NaN stays NaN)
    v["self_reported"] = (
        v["self_reported_fishing_vessel"].astype("string").str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
    )

    # active_hours / fishing_hours are deliberately NOT aggregated: they are
    # outcome-like quantities and would leak the target.
    summary = (
        v.groupby(["flag_gfw", "vessel_class_gfw"])
        .agg(
            n_vessels=("mmsi", "nunique"),
            mean_length_m=("length_m_gfw", "mean"),
            mean_engine_kw=("engine_power_kw_gfw", "mean"),
            mean_tonnage_gt=("tonnage_gt_gfw", "mean"),
            share_self_reported=("self_reported", "mean"),
            mean_class_score=("vessel_class_inferred_score", "mean"),
        )
        .reset_index()
        .rename(columns={"flag_gfw": "flag", "vessel_class_gfw": "geartype"})
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw", type=Path)
    p.add_argument("--out-dir", default="data/processed", type=Path)
    p.add_argument("--year", default=2024, type=int)
    p.add_argument("--per-gear", default=20000, type=int,
                   help="target rows per gear type in the main sample")
    p.add_argument("--app-rows", default=50000, type=int,
                   help="rows in the smaller sample used by the Streamlit app")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    fleet_zip = args.raw_dir / f"fleet-monthly-csvs-10-v3-{args.year}.zip"
    vessels_csv = args.raw_dir / "fishing-vessels-v3.csv"
    for f in (fleet_zip, vessels_csv):
        if not f.exists():
            raise SystemExit(f"Missing file: {f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample = sample_fleet(fleet_zip, args.per_gear, args.seed)
    summary = vessel_summary(vessels_csv, args.year)

    joined = sample.merge(summary, on=["flag", "geartype"], how="left",
                          validate="many_to_one")
    n_unmatched = int(joined["n_vessels"].isna().sum())
    if n_unmatched:
        print(f"WARNING: {n_unmatched:,} sampled rows have no vessel match")

    # shuffle so rows are not ordered by month
    joined = joined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # smaller sample for the app (keeps all gear types represented)
    app_rows = min(args.app_rows, len(joined))
    keep = []
    rng = np.random.default_rng(args.seed)
    per_gear_app = math.ceil(app_rows / joined["geartype"].nunique())
    for _, positions in joined.groupby("geartype").indices.items():
        k = min(len(positions), per_gear_app)
        keep.append(rng.choice(positions, size=k, replace=False))
    app = joined.iloc[np.concatenate(keep)].reset_index(drop=True)

    outputs = {
        "fishing_sample.csv": lambda path: joined.to_csv(path, index=False),
        "fishing_app_sample.csv": lambda path: app.to_csv(path, index=False),
        "fishing_sample.parquet": lambda path: joined.to_parquet(path, index=False),
        "fishing_app_sample.parquet": lambda path: app.to_parquet(path, index=False),
    }
    print("\nSaving:")
    for name, writer in outputs.items():
        path = args.out_dir / name
        try:
            writer(path)
        except ImportError:
            print(f"  skipped {name} (install pyarrow for Parquet output)")
            continue
        print(f"  {path}  {path.stat().st_size / 1e6:6.1f} MB")

    print(f"\nMain sample: {len(joined):,} rows x {joined.shape[1]} columns")
    print(f"App sample : {len(app):,} rows")
    print("\nRows per gear type (main sample):")
    print(joined["geartype"].value_counts().to_string())


if __name__ == "__main__":
    main()
