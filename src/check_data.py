"""
Quick check of the 2024 Global Fishing Watch files.

Put this script in the same folder as:
  - fleet-monthly-csvs-10-v3-2024.zip
  - fishing-vessels-v3.csv
Then run:  python check_data.py
Requires:  pip install pandas
"""
import zipfile
from pathlib import Path

import pandas as pd

FLEET_ZIP = Path("fleet-monthly-csvs-10-v3-2024.zip")
VESSELS_CSV = Path("fishing-vessels-v3.csv")

# ---------- 1. Monthly fleet table ----------
frames = []
with zipfile.ZipFile(FLEET_ZIP) as z:
    names = sorted(n for n in z.namelist() if n.endswith(".csv"))
    print(f"{len(names)} CSV files inside the zip (expect 12, one per month)")
    for name in names:
        with z.open(name) as f:
            df = pd.read_csv(f)
        frames.append(df)
        print(f"  {name}: {len(df):,} rows")

fleet = pd.concat(frames, ignore_index=True)
print("\n=== FLEET (monthly, 2024) ===")
print(f"Rows: {len(fleet):,}   Columns: {fleet.shape[1]}")
print(fleet.dtypes)
print("\nMissing values per column:")
print(fleet.isna().sum())
print("\nSummary statistics:")
print(fleet[["hours", "fishing_hours", "mmsi_present"]].describe())
print(f"\nDistinct flags: {fleet['flag'].nunique()}")
print(f"Distinct gear types: {fleet['geartype'].nunique()}")
print(fleet["geartype"].value_counts())
print(f"\nRows where fishing_hours > hours (should be 0): "
      f"{(fleet['fishing_hours'] > fleet['hours']).sum():,}")
print(f"Duplicate rows: {fleet.duplicated().sum():,}")

# ---------- 2. Vessel table ----------
vessels = pd.read_csv(VESSELS_CSV)
print("\n=== VESSELS (all years) ===")
print(f"Rows: {len(vessels):,}   Columns: {vessels.shape[1]}")
v2024 = vessels[vessels["year"] == 2024]
print(f"Rows for 2024: {len(v2024):,}   Unique MMSI in 2024: {v2024['mmsi'].nunique():,}")
print("\nMissing values (2024 rows):")
print(v2024.isna().sum())

# ---------- 3. Join feasibility ----------
# The fleet table's flag/geartype come from flag_gfw / vessel_class_gfw.
agg = (
    v2024.groupby(["flag_gfw", "vessel_class_gfw"])
    .agg(n_vessels=("mmsi", "nunique"),
         mean_length_m=("length_m_gfw", "mean"),
         mean_engine_kw=("engine_power_kw_gfw", "mean"),
         mean_tonnage=("tonnage_gt_gfw", "mean"))
    .reset_index()
    .rename(columns={"flag_gfw": "flag", "vessel_class_gfw": "geartype"})
)
joined = fleet.merge(agg, on=["flag", "geartype"], how="left")
print("\n=== JOIN CHECK ===")
print(f"Joined rows: {len(joined):,} (should equal fleet rows: {len(fleet):,})")
print(f"Fleet rows with no vessel match: {joined['n_vessels'].isna().sum():,}")
