"""
Streamlit interactive dashboard for stdg-eval.

Launch via:
    streamlit run run_dashboard.py
or:
    stdg-eval dashboard --config configs/my_config.yaml

Dashboard structure
-------------------
Sidebar
  • Upload / path-based data loading
  • Column-type overrides
  • Global evaluation controls

Tab 1 – Individual Report
  • Dataset selector
  • Univariate distributions (CDF / bar charts)
  • Bivariate correlation heatmaps
  • Missingness overview

Tab 2 – Benchmarking Report
  • Metric weight sliders (fidelity + missingness axes)
  • Score table + radar chart
  • Rankings per axis
  • Best dataset per axis
"""

from __future__ import annotations

import io
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from stdg_eval.config import (
    DEFAULT_COMPOSITE_WEIGHTS,
    DEFAULT_FIDELITY_WEIGHTS,
    DEFAULT_MISSINGNESS_WEIGHTS,
)
from stdg_eval.evaluation.fidelity import evaluate_fidelity
from stdg_eval.evaluation.missingness import evaluate_missingness
from stdg_eval.evaluation.scoring import (
    compute_composite_score,
    compute_fidelity_score,
    compute_missingness_score,
)
from stdg_eval.utils.data_utils import detect_column_types, load_dataset, load_config
from stdg_eval.visualization import plots as P


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="stdg-eval · Synthetic Data Evaluation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #1565C0;
        padding: 10px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: bold;
        color: #1565C0;
    }
    section[data-testid="stSidebar"] {
        min-width: 320px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session-state helpers
# ===========================================================================

def _init_state():
    defaults = {
        "real_df": None,
        "synth_dfs": {},          # {name: DataFrame}
        "col_types": None,
        "fidelity_results": {},   # {name: FidelityResults}
        "missingness_results": {},  # {name: MissingnessResults}
        "evaluated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ===========================================================================
# Sidebar – data loading
# ===========================================================================

def _sidebar():
    st.sidebar.title("stdg-eval")
    st.sidebar.caption("Synthetic Data Evaluation Library")
    st.sidebar.divider()

    # ------------------------------------------------------------------
    # Data source selection
    # ------------------------------------------------------------------
    st.sidebar.subheader("1 · Load data")
    source = st.sidebar.radio("Source", ["Upload files", "Config file (YAML / .txt)"], index=0)

    real_df: Optional[pd.DataFrame] = None
    synth_dfs: Dict[str, pd.DataFrame] = {}

    if source == "Upload files":
        real_file = st.sidebar.file_uploader("Real dataset (CSV)", type=["csv"])
        synth_files = st.sidebar.file_uploader(
            "Synthetic dataset(s) (CSV)", type=["csv"], accept_multiple_files=True
        )
        if real_file:
            real_df = pd.read_csv(real_file)
        for f in synth_files or []:
            synth_dfs[f.name.replace(".csv", "")] = pd.read_csv(f)

    else:  # Config file
        cfg_file = st.sidebar.file_uploader("Config file (.yaml or .txt)", type=["yaml", "yml", "txt"])
        if cfg_file:
            try:
                import tempfile, pathlib
                suffix = pathlib.Path(cfg_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(cfg_file.read())
                    tmp_path = tmp.name
                cfg = load_config(tmp_path)
                os.unlink(tmp_path)
                real_df = pd.read_csv(cfg["real_data"])
                for entry in cfg.get("synthetic_datasets", []):
                    synth_dfs[entry["name"]] = pd.read_csv(entry["path"])
                # Column type overrides from config
                if "column_types" in cfg:
                    st.session_state["col_types"] = cfg["column_types"]
            except Exception as e:
                st.sidebar.error(f"Failed to load config: {e}")

    if real_df is not None:
        st.session_state["real_df"] = real_df
    if synth_dfs:
        st.session_state["synth_dfs"] = synth_dfs

    # ------------------------------------------------------------------
    # Column type configuration
    # ------------------------------------------------------------------
    if st.session_state["real_df"] is not None:
        st.sidebar.divider()
        st.sidebar.subheader("2 · Column types")
        with st.sidebar.expander("Override column types", expanded=False):
            df = st.session_state["real_df"]
            inferred = detect_column_types(df)
            overrides = {}
            for col in df.columns:
                default_idx = 0 if inferred.get(col) == "numerical" else 1
                choice = st.selectbox(
                    col, ["numerical", "categorical"],
                    index=default_idx, key=f"coltype_{col}"
                )
                overrides[col] = choice
            if st.button("Apply column types"):
                st.session_state["col_types"] = overrides
                st.session_state["evaluated"] = False
                st.sidebar.success("Column types updated.")

        if st.session_state["col_types"] is None:
            st.session_state["col_types"] = detect_column_types(
                st.session_state["real_df"]
            )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("3 · Evaluate")

    with st.sidebar.expander("Metric options", expanded=False):
        run_uni = st.checkbox("Univariate", value=True, key="run_uni")
        run_wd = st.checkbox("↳ Wasserstein Distance", value=True, key="run_wd", disabled=not run_uni)
        run_tvd = st.checkbox("↳ Total Variation Distance", value=True, key="run_tvd", disabled=not run_uni)

        run_bi = st.checkbox("Bivariate", value=True, key="run_bi")
        run_spearman = st.checkbox("↳ Spearman Correlation", value=True, key="run_spearman", disabled=not run_bi)
        run_contingency = st.checkbox("↳ Contingency Matrix", value=True, key="run_contingency", disabled=not run_bi)

        run_multi = st.checkbox("Multivariate", value=True, key="run_multi")
        run_cc = st.checkbox("↳ Cross-Classification", value=True, key="run_cc", disabled=not run_multi)
        run_pmse = st.checkbox("↳ Propensity MSE", value=True, key="run_pmse", disabled=not run_multi)

        run_miss = st.checkbox("Missingness", value=True, key="run_miss")
        run_miss_rate = st.checkbox("↳ Rate", value=True, key="run_miss_rate", disabled=not run_miss)
        run_miss_set = st.checkbox("↳ Pattern Distribution", value=True, key="run_miss_set", disabled=not run_miss)
        run_miss_auroc = st.checkbox("↳ Classifier AUROC", value=True, key="run_miss_auroc", disabled=not run_miss)
        run_miss_dep = st.checkbox("↳ Dependency Structure", value=True, key="run_miss_dep", disabled=not run_miss)

    run_btn = st.sidebar.button(
        "▶ Run evaluation",
        disabled=(
            st.session_state["real_df"] is None
            or not st.session_state["synth_dfs"]
        ),
        use_container_width=True,
        type="primary",
    )

    if run_btn:
        _run_evaluation(
            run_uni, run_bi, run_multi, run_miss,
            run_wd, run_tvd,
            run_spearman, run_contingency,
            run_cc, run_pmse,
            run_miss_rate, run_miss_set, run_miss_auroc, run_miss_dep,
        )


def _run_evaluation(
    run_uni, run_bi, run_multi, run_miss,
    run_wd=True, run_tvd=True,
    run_spearman=True, run_contingency=True,
    run_cc=True, run_pmse=True,
    run_miss_rate=True, run_miss_set=True, run_miss_auroc=True, run_miss_dep=True,
):
    real = st.session_state["real_df"]
    synths = st.session_state["synth_dfs"]
    col_types = st.session_state["col_types"]

    fidelity_results = {}
    missingness_results = {}

    progress = st.sidebar.progress(0, text="Evaluating…")
    n = len(synths)

    for i, (name, synth) in enumerate(synths.items()):
        progress.progress((i) / n, text=f"Evaluating {name}…")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if run_uni or run_bi or run_multi:
                res = evaluate_fidelity(
                    real, synth, col_types=col_types,
                    run_univariate=run_uni,
                    run_bivariate=run_bi,
                    run_multivariate=run_multi,
                )
                # Post-filter individual sub-metrics the user deselected
                if "univariate" in res:
                    if not run_wd:
                        res["univariate"].pop("wasserstein", None)
                    if not run_tvd:
                        res["univariate"].pop("tvd", None)
                    if not res["univariate"]:
                        del res["univariate"]
                if "bivariate" in res:
                    if not run_spearman:
                        res["bivariate"].pop("spearman", None)
                    if not run_contingency:
                        res["bivariate"].pop("contingency", None)
                    if not res["bivariate"]:
                        del res["bivariate"]
                if "multivariate" in res:
                    if not run_cc:
                        res["multivariate"].pop("cross_classification", None)
                    if not run_pmse:
                        res["multivariate"].pop("propensity_mse", None)
                    if not res["multivariate"]:
                        del res["multivariate"]
                fidelity_results[name] = res
            if run_miss:
                missingness_results[name] = evaluate_missingness(
                    real, synth, col_types=col_types,
                    run_rate=run_miss_rate,
                    run_set_distribution=run_miss_set,
                    run_classifier_auroc=run_miss_auroc,
                    run_dependency_structure=run_miss_dep,
                )

    progress.progress(1.0, text="Done.")
    time.sleep(0.5)
    progress.empty()

    st.session_state["fidelity_results"] = fidelity_results
    st.session_state["missingness_results"] = missingness_results
    st.session_state["evaluated"] = True
    st.rerun()


# ===========================================================================
# Main content helpers
# ===========================================================================

def _score_badge(label: str, score: float):
    color = "#2e7d32" if score >= 0.8 else "#f57f17" if score >= 0.5 else "#c62828"
    st.markdown(
        f'<div class="metric-card"><small>{label}</small><br>'
        f'<span class="score-badge" style="color:{color}">{score:.3f}</span></div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Shared weight controls (rendered once, before tabs)
# ===========================================================================

def _weight_controls():
    """Render weight sliders in an expander. Called once before the tab layout."""
    ss = st.session_state
    run_uni = ss.get("run_uni", True)
    run_bi = ss.get("run_bi", True)
    run_multi = ss.get("run_multi", True)
    run_miss = ss.get("run_miss", True)

    # A fidelity group only contributes if it has at least one sub-metric enabled
    uni_active = run_uni and (ss.get("run_wd", True) or ss.get("run_tvd", True))
    bi_active = run_bi and (ss.get("run_spearman", True) or ss.get("run_contingency", True))
    multi_active = run_multi and (ss.get("run_cc", True) or ss.get("run_pmse", True))
    has_fidelity = uni_active or bi_active or multi_active

    miss_rate_active = run_miss and ss.get("run_miss_rate", True)
    miss_set_active = run_miss and ss.get("run_miss_set", True)
    miss_auroc_active = run_miss and ss.get("run_miss_auroc", True)
    miss_dep_active = run_miss and ss.get("run_miss_dep", True)
    has_missingness = miss_rate_active or miss_set_active or miss_auroc_active or miss_dep_active

    with st.expander("Weighting scheme", expanded=False):
        st.caption(
            "Adjust the weights for each metric group and evaluation axis. "
            "Weights are automatically normalised to sum to 1."
        )
        weight_cols = st.columns(2)
        with weight_cols[0]:
            st.markdown("**Fidelity weights**")
            if has_fidelity:
                if uni_active:
                    st.slider("Univariate", 0.0, 1.0, DEFAULT_FIDELITY_WEIGHTS[0], 0.01, key="w_uni")
                    st.caption(_fidelity_sub_label("run_wd", "WD", "run_tvd", "TVD"))
                if bi_active:
                    st.slider("Bivariate", 0.0, 1.0, DEFAULT_FIDELITY_WEIGHTS[1], 0.01, key="w_bi")
                    st.caption(_fidelity_sub_label("run_spearman", "Spearman", "run_contingency", "Contingency"))
                if multi_active:
                    st.slider("Multivariate", 0.0, 1.0, DEFAULT_FIDELITY_WEIGHTS[2], 0.01, key="w_multi")
                    st.caption(_fidelity_sub_label("run_cc", "Cross-class", "run_pmse", "pMSE"))
            else:
                st.caption("No fidelity metrics selected.")
        with weight_cols[1]:
            st.markdown("**Missingness weights**")
            if has_missingness:
                if miss_rate_active:
                    st.slider("Rate", 0.0, 1.0, DEFAULT_MISSINGNESS_WEIGHTS[0], 0.01, key="w_rate")
                if miss_set_active:
                    st.slider("Pattern distribution", 0.0, 1.0, DEFAULT_MISSINGNESS_WEIGHTS[1], 0.01, key="w_set")
                if miss_auroc_active:
                    st.slider("Classifier AUROC", 0.0, 1.0, DEFAULT_MISSINGNESS_WEIGHTS[2], 0.01, key="w_auroc")
                if miss_dep_active:
                    st.slider("Dependency structure", 0.0, 1.0, DEFAULT_MISSINGNESS_WEIGHTS[3], 0.01, key="w_dep")
            else:
                st.caption("No missingness metrics selected.")
        st.markdown("**Composite axis weights**")
        comp_cols = st.columns(4)
        with comp_cols[0]:
            if has_fidelity:
                st.slider("Fidelity axis", 0.0, 1.0, DEFAULT_COMPOSITE_WEIGHTS[0], 0.01, key="w_fid")
            else:
                st.metric("Fidelity axis", "N/A")
        with comp_cols[1]:
            if has_missingness:
                st.slider("Missingness axis", 0.0, 1.0, DEFAULT_COMPOSITE_WEIGHTS[1], 0.01, key="w_miss")
            else:
                st.metric("Missingness axis", "N/A")
        with comp_cols[2]:
            st.metric("Utility axis", "TODO", help="Utility metrics not yet implemented.")
        with comp_cols[3]:
            st.metric("Privacy axis", "TODO", help="Privacy metrics not yet implemented.")


def _fidelity_sub_label(key_a: str, label_a: str, key_b: str, label_b: str) -> str:
    """Return a caption listing which sub-metrics are active within a fidelity group."""
    ss = st.session_state
    active = [l for k, l in [(key_a, label_a), (key_b, label_b)] if ss.get(k, True)]
    return "Includes: " + " + ".join(active) if active else ""


def _get_weights() -> tuple:
    """Read the current weight values from session state for enabled metrics only."""
    ss = st.session_state
    run_uni = ss.get("run_uni", True)
    run_bi = ss.get("run_bi", True)
    run_multi = ss.get("run_multi", True)
    run_miss = ss.get("run_miss", True)

    uni_active = run_uni and (ss.get("run_wd", True) or ss.get("run_tvd", True))
    bi_active = run_bi and (ss.get("run_spearman", True) or ss.get("run_contingency", True))
    multi_active = run_multi and (ss.get("run_cc", True) or ss.get("run_pmse", True))

    miss_rate_active = run_miss and ss.get("run_miss_rate", True)
    miss_set_active = run_miss and ss.get("run_miss_set", True)
    miss_auroc_active = run_miss and ss.get("run_miss_auroc", True)
    miss_dep_active = run_miss and ss.get("run_miss_dep", True)

    fidelity_weights = [
        ss.get("w_uni", DEFAULT_FIDELITY_WEIGHTS[0]) if uni_active else 0.0,
        ss.get("w_bi", DEFAULT_FIDELITY_WEIGHTS[1]) if bi_active else 0.0,
        ss.get("w_multi", DEFAULT_FIDELITY_WEIGHTS[2]) if multi_active else 0.0,
    ]
    miss_weights = [
        ss.get("w_rate", DEFAULT_MISSINGNESS_WEIGHTS[0]) if miss_rate_active else 0.0,
        ss.get("w_set", DEFAULT_MISSINGNESS_WEIGHTS[1]) if miss_set_active else 0.0,
        ss.get("w_auroc", DEFAULT_MISSINGNESS_WEIGHTS[2]) if miss_auroc_active else 0.0,
        ss.get("w_dep", DEFAULT_MISSINGNESS_WEIGHTS[3]) if miss_dep_active else 0.0,
    ]
    composite_weights = [
        ss.get("w_fid", DEFAULT_COMPOSITE_WEIGHTS[0]) if (uni_active or bi_active or multi_active) else 0.0,
        ss.get("w_miss", DEFAULT_COMPOSITE_WEIGHTS[1]) if (miss_rate_active or miss_set_active or miss_auroc_active or miss_dep_active) else 0.0,
    ]
    return fidelity_weights, miss_weights, composite_weights


# ===========================================================================
# Tab 1: Individual dataset report
# ===========================================================================

def _tab_individual():
    st.header("Individual Dataset Report")

    synths = st.session_state["synth_dfs"]
    if not synths:
        st.info("Upload data and run evaluation to see results here.")
        return

    selected = st.selectbox("Select synthetic dataset", list(synths.keys()))
    if not selected:
        return

    real = st.session_state["real_df"]
    synth = synths[selected]
    col_types = st.session_state["col_types"] or {}
    fid_res = st.session_state["fidelity_results"].get(selected, {})
    miss_res = st.session_state["missingness_results"].get(selected, {})

    if not fid_res and not miss_res:
        st.warning("No evaluation results found. Click **▶ Run evaluation** in the sidebar.")
        return

    # ------------------------------------------------------------------
    # Score summary row
    # ------------------------------------------------------------------
    st.subheader("Scores")
    score_cols = st.columns(3)

    fidelity_weights, miss_weights, composite_weights = _get_weights()

    f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights) if fid_res else {}
    m_scores = compute_missingness_score(miss_res, weights=miss_weights) if miss_res else {}

    with score_cols[0]:
        if f_scores:
            _score_badge("Fidelity", f_scores["overall"])
    with score_cols[1]:
        if m_scores:
            _score_badge("Missingness", m_scores["overall"])
    with score_cols[2]:
        if f_scores and m_scores:
            comp = compute_composite_score(f_scores, m_scores, weights=composite_weights)
            _score_badge("Composite", comp["composite"])

    st.divider()

    # ------------------------------------------------------------------
    # Univariate
    # ------------------------------------------------------------------
    if "univariate" in fid_res:
        with st.expander("Univariate distributions", expanded=True):
            num_cols = [c for c, t in col_types.items() if t == "numerical"]
            cat_cols = [c for c, t in col_types.items() if t == "categorical"]

            wd_res = fid_res["univariate"].get("wasserstein")
            tvd_res = fid_res["univariate"].get("tvd")

            if num_cols and wd_res:
                st.markdown("**Numerical columns — CDF comparison (Wasserstein Distance)**")
                wd_vals = wd_res.details.get("raw_distances", {})
                col_scores = wd_res.column_scores or {}

                # Sort by descending WD (most divergent first)
                sorted_num = sorted(num_cols, key=lambda c: -wd_vals.get(c, 0))

                n_cols = min(3, len(sorted_num))
                for row_start in range(0, len(sorted_num), n_cols):
                    row_cols = sorted_num[row_start: row_start + n_cols]
                    cols_st = st.columns(n_cols)
                    for col_st, col in zip(cols_st, row_cols):
                        with col_st:
                            fig = P.plot_numerical_cdf(
                                real[col], synth[col], col,
                                wasserstein_distance=wd_vals.get(col),
                                synth_label=selected,
                                synth_color=P.SYNTH_COLORS[0],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            if col in col_scores:
                                st.caption(f"Score: {col_scores[col]:.3f}")

            if cat_cols and tvd_res:
                st.markdown("**Categorical columns — frequency comparison (TVD)**")
                tvd_vals = tvd_res.details.get("tvd_values", {})
                r_freqs = tvd_res.details.get("real_frequencies", {})
                s_freqs = tvd_res.details.get("synth_frequencies", {})
                col_scores = tvd_res.column_scores or {}

                sorted_cat = sorted(cat_cols, key=lambda c: -tvd_vals.get(c, 0))

                n_cols = min(3, len(sorted_cat))
                for row_start in range(0, len(sorted_cat), n_cols):
                    row_cols = sorted_cat[row_start: row_start + n_cols]
                    cols_st = st.columns(n_cols)
                    for col_st, col in zip(cols_st, row_cols):
                        if col not in r_freqs:
                            continue
                        with col_st:
                            fig = P.plot_categorical_bars(
                                r_freqs[col], s_freqs[col], col,
                                tvd=tvd_vals.get(col),
                                synth_label=selected,
                                synth_color=P.SYNTH_COLORS[0],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            if col in col_scores:
                                st.caption(f"Score: {col_scores[col]:.3f}")

    # ------------------------------------------------------------------
    # Bivariate
    # ------------------------------------------------------------------
    if "bivariate" in fid_res:
        with st.expander("Bivariate associations", expanded=True):
            sp_res = fid_res["bivariate"].get("spearman")
            ct_res = fid_res["bivariate"].get("contingency")

            if sp_res and "columns" in sp_res.details:
                st.markdown("**Spearman correlation matrices**")
                ncols = sp_res.details["columns"]
                real_corr = pd.DataFrame(sp_res.details["real_correlation_matrix"])
                synth_corr = pd.DataFrame(sp_res.details["synth_correlation_matrix"])
                fig = P.plot_correlation_heatmaps(real_corr, synth_corr, synth_label=selected)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Mean |Δ| = {sp_res.details.get('mean_absolute_difference', 0):.4f}  |  Score: {sp_res.score:.3f}")

            if ct_res and ct_res.details.get("pair_tvds"):
                st.markdown("**Contingency table TVD per pair** (top-20 most divergent)")
                pair_tvds = ct_res.details["pair_tvds"]
                top_pairs = sorted(pair_tvds.items(), key=lambda x: -x[1])[:20]
                pair_df = pd.DataFrame(top_pairs, columns=["Pair", "TVD"])
                pair_df["Score"] = 1 - pair_df["TVD"]
                st.dataframe(pair_df.style.format({"TVD": "{:.4f}", "Score": "{:.4f}"}),
                             use_container_width=True)

    # ------------------------------------------------------------------
    # Multivariate
    # ------------------------------------------------------------------
    if "multivariate" in fid_res:
        with st.expander("Multivariate metrics", expanded=True):
            cc_res = fid_res["multivariate"].get("cross_classification")
            pmse_res = fid_res["multivariate"].get("propensity_mse")

            m_cols = st.columns(2)
            if cc_res:
                with m_cols[0]:
                    st.metric("Cross-Classification AUROC",
                              f"{cc_res.details.get('mean_auroc', 0):.4f}",
                              help="0.5 = indistinguishable; 1.0 = perfectly separable")
                    st.caption(f"Score: {cc_res.score:.3f}")
                    fold_aurocs = cc_res.details.get("fold_aurocs", [])
                    if fold_aurocs:
                        fig = go_bar_folds(fold_aurocs)
                        st.plotly_chart(fig, use_container_width=True)
            if pmse_res:
                with m_cols[1]:
                    st.metric("Propensity MSE",
                              f"{pmse_res.details.get('pmse', 0):.6f}",
                              help="Lower = better; 0 = perfect fidelity")
                    st.caption(f"Score: {pmse_res.score:.3f}")
                    st.caption(
                        f"pMSE null baseline: {pmse_res.details.get('pmse_null', 0):.6f} | "
                        f"ratio: {pmse_res.details.get('pmse_ratio', 0):.4f}"
                    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------
    if miss_res:
        with st.expander("Missingness", expanded=True):
            rate_res = miss_res.get("rate")
            set_res = miss_res.get("set_distribution")
            dep_res = miss_res.get("dependency_structure")

            if rate_res:
                st.markdown("**Per-variable missingness rates**")
                fig = P.plot_missingness_rates(
                    rate_res.details.get("real_rates", {}),
                    rate_res.details.get("synth_rates", {}),
                    synth_label=selected,
                )
                st.plotly_chart(fig, use_container_width=True)

            miss_pattern_cols = st.columns(2)
            with miss_pattern_cols[0]:
                st.markdown("**Real — missingness pattern**")
                fig = P.plot_missingness_pattern_heatmap(real, title="Real: missingness pattern")
                st.plotly_chart(fig, use_container_width=True)
            with miss_pattern_cols[1]:
                st.markdown(f"**{selected} — missingness pattern**")
                fig = P.plot_missingness_pattern_heatmap(synth, title=f"{selected}: missingness pattern")
                st.plotly_chart(fig, use_container_width=True)

            if dep_res and "columns" in dep_res.details:
                dep_cols = dep_res.details["columns"]
                if len(dep_cols) >= 2:
                    st.markdown("**Missingness dependency structure**")
                    dep_plot_cols = st.columns(2)
                    real_dep = pd.DataFrame(dep_res.details["real_correlation_matrix"])
                    synth_dep = pd.DataFrame(dep_res.details["synth_correlation_matrix"])
                    with dep_plot_cols[0]:
                        st.plotly_chart(
                            P.plot_missingness_dependency(real_dep, "Real: missingness dependency"),
                            use_container_width=True,
                        )
                    with dep_plot_cols[1]:
                        st.plotly_chart(
                            P.plot_missingness_dependency(synth_dep, f"{selected}: missingness dependency"),
                            use_container_width=True,
                        )

            if set_res:
                st.caption(
                    f"Pattern distribution TVD: {set_res.details.get('tvd', 0):.4f}  |  "
                    f"Unique patterns (real): {set_res.details.get('n_unique_real_patterns', 'n/a')}  |  "
                    f"Unique patterns ({selected}): {set_res.details.get('n_unique_synth_patterns', 'n/a')}"
                )


def go_bar_folds(fold_aurocs: list) -> "go.Figure":
    """Small bar chart for cross-validation fold AUROCs."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(fold_aurocs))],
        y=fold_aurocs,
        marker_color=P.REAL_COLOR,
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="0.5 (ideal)", annotation_position="top right")
    fig.update_layout(
        title="AUROC per CV fold",
        yaxis=dict(range=[0, 1], title="AUROC"),
        height=250,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ===========================================================================
# Tab 2: Benchmarking report
# ===========================================================================

def _tab_benchmarking():
    st.header("Benchmarking Report")

    synths = st.session_state["synth_dfs"]
    fid_all = st.session_state["fidelity_results"]
    miss_all = st.session_state["missingness_results"]

    if not synths or (not fid_all and not miss_all):
        st.info("Run evaluation first (sidebar → **▶ Run evaluation**).")
        return

    fidelity_weights, miss_weights, composite_weights = _get_weights()

    st.divider()

    # ------------------------------------------------------------------
    # Compute scores for all datasets
    # ------------------------------------------------------------------
    rows = []
    axis_scores: Dict[str, Dict[str, float]] = {}  # {name: {axis: score}}

    for name in synths:
        fid_res = fid_all.get(name, {})
        miss_res = miss_all.get(name, {})

        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights) if fid_res else {}
        m_scores = compute_missingness_score(miss_res, weights=miss_weights) if miss_res else {}
        comp = compute_composite_score(f_scores, m_scores, weights=composite_weights) if (f_scores or m_scores) else {}

        row = {"Dataset": name}
        axis_entry: Dict[str, float] = {}

        if f_scores:
            row["Fidelity"] = f_scores.get("overall", float("nan"))
            row.update({
                "  Univariate": f_scores.get("univariate", float("nan")),
                "  Bivariate": f_scores.get("bivariate", float("nan")),
                "  Multivariate": f_scores.get("multivariate", float("nan")),
            })
            axis_entry["Fidelity"] = row["Fidelity"]

        if m_scores:
            row["Missingness"] = m_scores.get("overall", float("nan"))
            axis_entry["Missingness"] = row["Missingness"]

        if comp:
            row["Composite"] = comp.get("composite", float("nan"))
            axis_entry["Composite"] = row["Composite"]

        rows.append(row)
        axis_scores[name] = axis_entry

    summary_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Score table
    # ------------------------------------------------------------------
    st.subheader("Score summary")
    st.plotly_chart(P.plot_score_table(summary_df), use_container_width=True)

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------
    st.subheader("Rankings")
    rank_tabs = st.tabs(["Composite", "Fidelity", "Missingness"])

    score_cols_map = {
        "Composite": "Composite",
        "Fidelity": "Fidelity",
        "Missingness": "Missingness",
    }

    for tab, (axis, col) in zip(rank_tabs, score_cols_map.items()):
        with tab:
            if col in summary_df.columns:
                ranked = (
                    summary_df[["Dataset", col]]
                    .dropna()
                    .sort_values(col, ascending=False)
                    .reset_index(drop=True)
                )
                ranked.index += 1  # 1-based rank
                ranked.index.name = "Rank"

                best = ranked.iloc[0]["Dataset"] if len(ranked) > 0 else "N/A"
                st.success(f"🏆 Best **{axis}** dataset: **{best}**")
                st.plotly_chart(
                    P.plot_score_bar(
                        dict(zip(ranked["Dataset"], ranked[col])),
                        title=f"{axis} scores",
                    ),
                    use_container_width=True,
                )
            else:
                st.info(f"{col} scores not available — run the relevant metrics.")

    # ------------------------------------------------------------------
    # Radar chart (multi-axis)
    # ------------------------------------------------------------------
    radar_axes = [ax for ax in ["Fidelity", "Missingness"] if ax in summary_df.columns]
    if len(radar_axes) >= 2:
        st.subheader("Multi-axis comparison")
        fig = P.plot_score_radar(
            {row["Dataset"]: {ax: row.get(ax, 0) for ax in radar_axes}
             for row in rows},
            axes=radar_axes,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Per-group breakdown
    # ------------------------------------------------------------------
    st.subheader("Per-group score breakdown")
    group_cols = ["Dataset", "  Univariate", "  Bivariate", "  Multivariate", "Missingness"]
    available = [c for c in group_cols if c in summary_df.columns]
    if available:
        st.dataframe(
            summary_df[available].style.format(
                {c: "{:.4f}" for c in available if c != "Dataset"}
            ),
            use_container_width=True,
        )


# ===========================================================================
# Main app entry point
# ===========================================================================

def run_dashboard():
    """Entry point called by ``run_dashboard.py`` and the CLI."""
    _sidebar()

    if st.session_state["real_df"] is None:
        st.title("stdg-eval · Synthetic Data Evaluation Dashboard")
        st.info(
            "👈 **Get started**: upload your real dataset and one or more synthetic "
            "datasets in the sidebar, then click **▶ Run evaluation**."
        )
        st.markdown("""
### Evaluation axes

| Axis | Status | Metrics |
|------|--------|---------|
| **Fidelity** | ✅ Available | Wasserstein Distance, TVD, Spearman Correlation, Contingency Matrix, Cross-Classification, Propensity MSE |
| **Missingness** | ✅ Available | Missingness Rate, Pattern Distribution, Classifier AUROC, Dependency Structure |
| **Utility** | 🔜 TODO | Downstream task performance |
| **Privacy** | 🔜 TODO | Disclosure risk, membership inference |
        """)
        return

    _weight_controls()

    tab1, tab2 = st.tabs(["📊 Individual Report", "🏆 Benchmarking Report"])
    with tab1:
        _tab_individual()
    with tab2:
        _tab_benchmarking()


if __name__ == "__main__":
    run_dashboard()
