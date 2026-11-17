# Global Fishing Effort 2024: an end-to-end ML pipeline

Machine Learning and Data Analytics (6COSC017C-n), Coursework 1.
Student ID: 18503

## Project goal

Using AIS-derived apparent fishing effort for 2024, this project:

1. predicts whether any fishing occurs in a grid cell and month,
2. predicts how much fishing occurs (log fishing hours) where it does,
3. classifies the fishing gear type,
4. clusters fishing grounds,

and audits how uneven AIS coverage biases the results.

## Data

| Item | Detail |
|---|---|
| Name | Global AIS-based Apparent Fishing Effort Dataset, Version 3.0 |
| Publisher | Global Fishing Watch |
| Source | https://doi.org/10.5281/zenodo.14982712 (https://zenodo.org/records/14982712) |
| Used files | `fleet-monthly-csvs-10-v3-2024.zip` (13,838,619 rows, 10 columns) and `fishing-vessels-v3.csv` (94,457 vessels in 2024, 22 columns) |
| Licence | CC BY-NC 4.0 (non-commercial use only) |

Citation: Global Fishing Watch. 2025. Global AIS-based Apparent Fishing Effort Dataset, Version 3.0. https://doi.org/10.5281/zenodo.14982712

The code in this repository is MIT licensed. The data is **not** covered by the MIT licence; it remains CC BY-NC 4.0 and must be attributed to Global Fishing Watch.

The raw data is too large for Git. The notebooks and the app use a stratified working sample that `src/make_sample.py` builds from the raw files (see below).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

1. Download `fleet-monthly-csvs-10-v3-2024.zip` and `fishing-vessels-v3.csv` from the Zenodo record above and place them in `data/raw/`.
2. Build the sample:
   ```bash
   python src/make_sample.py --per-gear 20000
   ```
   This writes `data/processed/fishing_sample.csv` and `.parquet` (about 320,000 rows) and a smaller `fishing_app_sample.parquet` for the app.
3. Run the notebooks in `notebooks/` in order.
4. Launch the app (once built): `streamlit run app/Home.py`

## Repository structure

```
data/raw/          raw downloads (not committed)
data/processed/    working sample (committed)
notebooks/         analysis notebooks
src/               reusable code (sampling, preprocessing)
app/               multi-page Streamlit app
```

## Sampling method

Up to 20,000 rows per gear type (16 gear types), drawn evenly across the 12 months, then joined with a flag × gear-type summary of the 2024 vessel table. Stratifying by gear type is necessary because the rarest gear type has only about 35,000 of 13.8 million rows. The sample therefore does **not** preserve the natural gear-type mix; this is discussed in the report.

## Known limitations

- 2024 data is provisional (vessel classifications may change).
- AIS reception is uneven across regions and changed in 2022.
- Flag states are partly inferred from the first three digits of the vessel identifier and can be wrong.
- Apparent fishing effort is an estimate, not a catch measurement.

## Weekly milestones

See the commit history. Each week's work is committed with a message describing the milestone.
