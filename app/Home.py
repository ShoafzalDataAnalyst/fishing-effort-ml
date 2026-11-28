"""Home page of the multi-page Streamlit app. Run with:  streamlit run app/Home.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from utils import load_json, load_sample, footer

st.set_page_config(page_title="Global Fishing Effort 2024", layout="wide")

st.title("Global Fishing Effort 2024")
st.write(
    "An end-to-end machine-learning project on satellite-tracked (AIS) fishing activity. "
    "Each row of the data is one 0.1 degree ocean cell in one month, for one flag state and gear type. "
    "The models answer four questions: **will any fishing occur**, **how many hours**, "
    "**which gear type is it**, and **where are the main fishing grounds**."
)

test = load_json("test_results.json")
df = load_sample()
c1, c2, c3, c4 = st.columns(4)
c4.metric("Rows in the app sample", f"{len(df):,}")
if test:
    c1.metric("Will fishing occur? (ROC-AUC)", f"{test['A']['Set A (deployable)']['roc_auc']:.2f}",
              help="Test score on ocean blocks never used for training. 0.5 = guessing, 1.0 = perfect.")
    c2.metric("How many hours? (R-squared)", f"{test['B']['A']['r2']:.2f}",
              help="Share of the variation in log fishing hours explained, on unseen blocks.")
    c3.metric("Which gear? (macro-F1)", f"{test['C']['macro_f1']:.2f}",
              help="Average F1 over 16 gear types on unseen blocks. Guessing would score about 0.06.")
else:
    st.info("Run notebook 04 to create reports/test_results.json and show test scores here.")

st.subheader("Pages")
st.page_link("pages/1_Explore_Data.py", label="Explore the data: maps, gear types, seasons")
st.page_link("pages/2_Predict_Fishing.py", label="Predict fishing for a place, month, gear and flag")
st.page_link("pages/3_Identify_Gear.py", label="Identify the gear type from where and how a fleet operates")
st.page_link("pages/4_Model_Performance.py", label="Model performance, validation and limitations")

st.subheader("How the models were built")
st.markdown(
    """
- **Data:** Global Fishing Watch 2024 monthly fleet table (13.8 million rows), reduced to a stratified working
  sample of about 320,000 rows (about 20,000 per gear type).
- **Honest validation:** whole 10 x 10 degree ocean blocks were held out for testing, because neighbouring cells
  look alike and a random split would flatter the models.
- **Two feature sets:** the deployable set (location, season, gear, flag, fleet averages) and an upper-bound set
  that also sees total vessel hours, which already contain the fishing hours.
"""
)
st.subheader("Please keep in mind")
st.warning(
    "Apparent fishing effort is inferred from vessel positions, not measured catch. AIS reception is uneven across "
    "regions, flag states are partly inferred, and the working sample is balanced by gear type, so it does not show "
    "how common each gear is in reality. These models describe patterns in this data; they are not a guide to where "
    "fish are or to any vessel's behaviour."
)
footer()
