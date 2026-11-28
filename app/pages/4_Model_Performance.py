"""Results from notebooks 03 (cross-validation) and 04 (held-out test blocks), with limitations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import BLUE, ORANGE, footer, load_json, show_figure

st.set_page_config(page_title="Model performance", layout="wide")
st.title("Model performance")
cv, test = load_json("cv_results.json"), load_json("test_results.json")
if not (cv and test):
    st.error("reports/cv_results.json and reports/test_results.json are needed. Run notebooks 03 and 04 first.")
    st.stop()
st.write("All test scores come from **ocean blocks never used for training or tuning**. "
         "Set A is the deployable model; set B also sees total vessel hours (which include the fishing hours) and is only an upper bound.")

t_occ, t_hrs, t_gear, t_clu, t_val = st.tabs(["Will fishing occur?", "How many hours?", "Which gear?", "Fishing grounds", "Validation and fairness"])

with t_occ:
    A, B = test["A"]["Set A (deployable)"], test["A"]["Set B (+ hours, mmsi_present)"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (set A)", f"{A['roc_auc']:.2f}", help="0.5 = guessing, 1.0 = perfect")
    c2.metric("PR-AUC (set A)", f"{A['pr_auc']:.2f}", help="A random guesser scores the share of rows with fishing (about 0.34 in the test blocks).")
    c3.metric("F1 at tuned threshold", f"{A['f1_tuned']:.2f}", help=f"Threshold {A['threshold']:.2f}, chosen on training folds only.")
    comp = pd.DataFrame({"Metric": ["ROC-AUC", "PR-AUC", "F1 (tuned threshold)"] * 2,
                         "Feature set": ["A (deployable)"] * 3 + ["B (adds hours)"] * 3,
                         "Score": [A["roc_auc"], A["pr_auc"], A["f1_tuned"], B["roc_auc"], B["pr_auc"], B["f1_tuned"]]})
    fig = px.bar(comp, x="Metric", y="Score", color="Feature set", barmode="group",
                 color_discrete_map={"A (deployable)": BLUE, "B (adds hours)": ORANGE})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Set B is far stronger because total vessel hours already contain the fishing hours: that is information leakage, "
               "not a better model, so it cannot be used to predict ahead.")
    g = pd.Series(test["A_by_gear"]).sort_values().rename("ROC-AUC").to_frame()
    fig = px.bar(g, x="ROC-AUC", y=g.index, orientation="h", title="Set A: ROC-AUC by gear type", labels={"y": ""})
    fig.update_traces(marker_color=BLUE); fig.update_xaxes(range=[0.4, 0.9])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    show_figure("17_taskA_curves.png", "ROC and precision-recall curves on the test blocks")
    show_figure("18_taskA_confusion_calibration.png", "Confusion matrix and calibration (set A)")

with t_hrs:
    b = test["B"]
    tab = pd.DataFrame({"RMSE (log hours)": [b["baseline"]["rmse"], b["A"]["rmse"], b["B"]["rmse"]],
                        "MAE": [b["baseline"]["mae"], b["A"]["mae"], b["B"]["mae"]],
                        "R-squared": [b["baseline"]["r2"], b["A"]["r2"], b["B"]["r2"]]},
                       index=["Baseline (training mean)", "Set A (deployable)", "Set B (adds hours)"]).round(3)
    st.dataframe(tab, width="stretch")
    st.write(f"Set A improves on the baseline but explains only about {b['A']['r2']:.0%} of the variation: effort is hard to "
             "predict from place, season and fleet alone. In "
             f"{test['B_share_all_fishing']:.0%} of fishing rows every recorded hour was a fishing hour, so set B often just copies the total.")
    s = test["two_part_spearman"]
    st.write(f"**Both parts combined** (probability x hours), ranking all test cells: Spearman correlation with actual fishing hours "
             f"{s['A']:.2f} for set A and {s['B']:.2f} for set B.")
    show_figure("20_taskB_fit.png", "Predicted versus actual log hours")

with t_gear:
    c = test["C"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Macro-F1, 16 gear types", f"{c['macro_f1']:.2f}", help="Guessing would score about 0.06.")
    c2.metric("95% interval", f"{c['macro_f1_ci'][0]:.2f} to {c['macro_f1_ci'][1]:.2f}", help="Block bootstrap")
    c3.metric("Accuracy", f"{c['accuracy']:.0%}")
    st.write("Gear classes in the data overlap in meaning, and the model sees only place, season, flag and activity, "
             "so it is much better than guessing but far from perfect. The confusion matrix shows which types get mixed up.")
    show_figure("22_taskC_f1_by_gear.png", "F1 by gear type")
    show_figure("23_taskC_confusion.png", "Confusion matrix (% of each actual class)")
    imp = test["importance_top"]
    rows = [{"Task": {"A": "Occurrence", "B": "Hours", "C": "Gear"}[k], "Input": n, "Importance": v} for k, d in imp.items() for n, v in d.items()]
    st.write("**Inputs the models rely on most** (permutation importance: how much the score drops when one input is shuffled):")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with t_clu:
    k = cv["D_k"]
    sil = pd.DataFrame(cv["D_silhouette"]).T
    st.write(f"K-means with **k = {k}** groups the cells that have fishing by location and effort. Silhouette scores for k = 3 to 6 "
             "were all close (0.36 to 0.38), so k was also chosen for how well the groups can be described.")
    fig = px.line(sil.reset_index(names="k"), x="k", y="silhouette", markers=True)
    fig.update_traces(line_color=BLUE, line_width=2); fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    show_figure("16_cluster_maps.png", "Fishing-ground clusters (blue) against all fishing cells (grey)")

with t_val:
    rv = pd.DataFrame(test["random_vs_block"]).T.round(3)
    st.write("**Why the validation uses ocean blocks.** The same model scored higher with a random split of rows than with held-out blocks:")
    st.dataframe(rv, width="stretch")
    st.caption("A random split lets the model learn individual places; holding out whole blocks shows how it does somewhere new.")
    lat = pd.Series(test["audit_lat"]).rename("ROC-AUC").to_frame()
    flg = pd.Series(test["audit_flag"]).rename("ROC-AUC").to_frame()
    c1, c2 = st.columns([3, 2])
    for col, d, t in [(c1, lat, "By latitude band"), (c2, flg, "By flag group")]:
        fig = px.bar(d, x=d.index, y="ROC-AUC", title=t); fig.update_traces(marker_color=BLUE)
        fig.update_yaxes(range=[0.4, 1.0]); fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="")
        col.plotly_chart(fig, width="stretch")
    st.write("Performance differs by region and is lower for vessels with an `UNKNOWN` flag. These gaps are suggestive, "
             "not proof of AIS-coverage bias: regions also differ in gear mix and in how often fishing occurs.")
    st.subheader("Limitations")
    st.markdown("- The working sample is balanced by gear type, so it does not reflect real gear prevalence.\n"
                "- 2024 data are provisional; AIS reception is uneven and changed in 2022; flags are partly inferred.\n"
                "- Apparent fishing is inferred from vessel movement, not catch.\n"
                "- Vessel features are flag x gear averages, not individual vessels.\n"
                "- Test scores come from one geographic split; the confidence intervals in notebook 04 show how much they can move.")
footer()
