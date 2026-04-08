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
from stdg_eval.utils.data_utils import detect_column_types, load_dataset, load_config, validate_column_types, eval_config_from_dict
from stdg_eval.utils.precomputed_io import load_precomputed
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
        # Precomputed bivariate/multivariate results loaded from JSON
        # {synth_name: {group: {metric_key: MetricResult}}}
        "precomputed_results": {},
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
                import hashlib, tempfile, pathlib
                cfg_bytes = cfg_file.read()
                cfg_hash = hashlib.md5(cfg_bytes).hexdigest()
                suffix = pathlib.Path(cfg_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(cfg_bytes)
                    tmp_path = tmp.name
                cfg = load_config(tmp_path)
                os.unlink(tmp_path)
                real_df = pd.read_csv(cfg["real_data"])
                for entry in cfg.get("synthetic_datasets", []):
                    synth_dfs[entry["name"]] = pd.read_csv(entry["path"])
                # Column type overrides from config
                if "column_types" in cfg:
                    st.session_state["col_types"] = cfg["column_types"]
                # Precomputed bivariate/multivariate results
                if "precomputed_results" in cfg:
                    try:
                        st.session_state["precomputed_results"] = load_precomputed(
                            cfg["precomputed_results"]
                        )
                    except Exception as exc:
                        st.sidebar.warning(f"Could not load precomputed results: {exc}")
                # Metric enable flags from config — only applied when a new config
                # file is loaded (detected via content hash). After initial load the
                # sidebar checkboxes are the source of truth and are not overwritten,
                # so the user can freely toggle metrics that were disabled in the config.
                if "metrics" in cfg and st.session_state.get("_config_hash") != cfg_hash:
                    eval_cfg = eval_config_from_dict(cfg)
                    fc = eval_cfg.fidelity
                    mc = eval_cfg.missingness
                    st.session_state["run_wd"] = fc.run_wasserstein
                    st.session_state["run_tvd"] = fc.run_tvd
                    st.session_state["run_hd"] = fc.run_hellinger
                    st.session_state["run_spearman"] = fc.run_spearman
                    st.session_state["run_contingency"] = fc.run_contingency
                    st.session_state["run_pcd"] = fc.run_pcd
                    st.session_state["run_cc"] = fc.run_auc_roc
                    st.session_state["run_pmse"] = fc.run_propensity_mse
                    st.session_state["run_crcl_rs"] = fc.run_crcl_rs
                    st.session_state["run_crcl_sr"] = fc.run_crcl_sr
                    st.session_state["run_miss_rate"] = mc.run_rate
                    st.session_state["run_miss_set"] = mc.run_set_distribution
                    st.session_state["run_miss_auroc"] = mc.run_missing_auroc
                    st.session_state["run_miss_dep"] = mc.run_dependency_structure
                st.session_state["_config_hash"] = cfg_hash
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

    # Precomputed results upload (alternative / supplement to config reference)
    with st.sidebar.expander("Precomputed results (optional)", expanded=False):
        st.caption(
            "Upload a JSON file produced by `stdg-eval precompute` to skip "
            "recomputing expensive bivariate / multivariate metrics."
        )
        pre_file = st.file_uploader(
            "Precomputed results (.json)", type=["json"], key="precomputed_upload"
        )
        if pre_file:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(pre_file.read())
                    tmp_path = tmp.name
                st.session_state["precomputed_results"] = load_precomputed(tmp_path)
                os.unlink(tmp_path)
                n_synths = len(st.session_state["precomputed_results"])
                st.success(f"Loaded precomputed results for {n_synths} dataset(s).")
            except Exception as exc:
                st.error(f"Failed to load precomputed results: {exc}")
        elif st.session_state.get("precomputed_results"):
            n_synths = len(st.session_state["precomputed_results"])
            st.info(f"Using precomputed results for {n_synths} dataset(s) (loaded from config).")
            if st.button("Clear precomputed results"):
                st.session_state["precomputed_results"] = {}
                st.rerun()

    with st.sidebar.expander("Metric options", expanded=False):
        run_uni = st.checkbox("Univariate", value=True, key="run_uni")
        run_wd = st.checkbox("↳ Wasserstein Distance", value=True, key="run_wd", disabled=not run_uni)
        run_tvd = st.checkbox("↳ Total Variation Distance", value=True, key="run_tvd", disabled=not run_uni)
        run_hd = st.checkbox("↳ Hellinger Distance", value=True, key="run_hd", disabled=not run_uni)

        run_bi = st.checkbox("Bivariate", value=True, key="run_bi")
        run_spearman = st.checkbox("↳ Spearman Correlation", value=True, key="run_spearman", disabled=not run_bi)
        run_contingency = st.checkbox("↳ Contingency Matrix", value=True, key="run_contingency", disabled=not run_bi)
        run_pcd = st.checkbox("↳ Pairwise Correlation Difference", value=True, key="run_pcd", disabled=not run_bi)

        run_multi = st.checkbox("Multivariate", value=True, key="run_multi")
        run_cc = st.checkbox("↳ AUC-ROC", value=True, key="run_cc", disabled=not run_multi)
        run_pmse = st.checkbox("↳ Propensity MSE", value=True, key="run_pmse", disabled=not run_multi)
        run_crcl_rs = st.checkbox("↳ CrCl-RS (train real, test synth)", value=True, key="run_crcl_rs", disabled=not run_multi)
        run_crcl_sr = st.checkbox("↳ CrCl-SR (train synth, test real)", value=True, key="run_crcl_sr", disabled=not run_multi)

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
            run_wd, run_tvd, run_hd,
            run_spearman, run_contingency, run_pcd,
            run_cc, run_pmse, run_crcl_rs, run_crcl_sr,
            run_miss_rate, run_miss_set, run_miss_auroc, run_miss_dep,
        )


def _run_evaluation(
    run_uni, run_bi, run_multi, run_miss,
    run_wd=True, run_tvd=True, run_hd=True,
    run_spearman=True, run_contingency=True, run_pcd=True,
    run_cc=True, run_pmse=True, run_crcl_rs=True, run_crcl_sr=True,
    run_miss_rate=True, run_miss_set=True, run_miss_auroc=True, run_miss_dep=True,
):
    real = st.session_state["real_df"]
    synths = st.session_state["synth_dfs"]
    col_types = st.session_state["col_types"]
    precomputed = st.session_state.get("precomputed_results", {})

    fidelity_results = {}
    missingness_results = {}

    # Validate column types and surface any warnings in the UI before evaluation
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_column_types(real, col_types, dataset_label="real")
        for name, synth in synths.items():
            validate_column_types(synth, col_types, dataset_label=f"synthetic ({name})")
    for w in caught:
        st.warning(str(w.message))

    progress = st.sidebar.progress(0, text="Evaluating…")
    n = len(synths)

    for i, (name, synth) in enumerate(synths.items()):
        progress.progress((i) / n, text=f"Evaluating {name}…")

        # Determine which groups are already covered by precomputed results so
        # we can skip their (potentially expensive) recomputation.
        precomp = precomputed.get(name, {})
        uni_precomputed = run_uni and bool(precomp.get("univariate"))
        bi_precomputed = run_bi and bool(precomp.get("bivariate"))
        multi_precomputed = run_multi and bool(precomp.get("multivariate"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if run_uni or run_bi or run_multi:
                res = evaluate_fidelity(
                    real, synth, col_types=col_types,
                    run_univariate=run_uni and not uni_precomputed,
                    run_bivariate=run_bi and not bi_precomputed,
                    run_multivariate=run_multi and not multi_precomputed,
                )
                # Inject precomputed groups into the result
                if uni_precomputed:
                    res.setdefault("univariate", {}).update(precomp["univariate"])
                if bi_precomputed:
                    res.setdefault("bivariate", {}).update(precomp["bivariate"])
                if multi_precomputed:
                    res.setdefault("multivariate", {}).update(precomp["multivariate"])

                # Post-filter individual sub-metrics the user deselected
                if "univariate" in res:
                    if not run_wd:
                        res["univariate"].pop("wasserstein", None)
                    if not run_tvd:
                        res["univariate"].pop("tvd", None)
                    if not run_hd:
                        res["univariate"].pop("hellinger", None)
                    if not res["univariate"]:
                        del res["univariate"]
                if "bivariate" in res:
                    if not run_spearman:
                        res["bivariate"].pop("spearman", None)
                    if not run_contingency:
                        res["bivariate"].pop("contingency", None)
                    if not run_pcd:
                        res["bivariate"].pop("pcd", None)
                    if not res["bivariate"]:
                        del res["bivariate"]
                if "multivariate" in res:
                    if not run_cc:
                        res["multivariate"].pop("auc_roc", None)
                    if not run_pmse:
                        res["multivariate"].pop("propensity_mse", None)
                    if not run_crcl_rs:
                        res["multivariate"].pop("crcl_rs", None)
                    if not run_crcl_sr:
                        res["multivariate"].pop("crcl_sr", None)
                    if not res["multivariate"]:
                        del res["multivariate"]
                fidelity_results[name] = res
            if run_miss:
                missingness_results[name] = evaluate_missingness(
                    real, synth, col_types=col_types,
                    run_rate=run_miss_rate,
                    run_set_distribution=run_miss_set,
                    run_missing_auroc=run_miss_auroc,
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
    uni_active = run_uni and (ss.get("run_wd", True) or ss.get("run_tvd", True) or ss.get("run_hd", True))
    bi_active = run_bi and (ss.get("run_spearman", True) or ss.get("run_contingency", True) or ss.get("run_pcd", True))
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
                    if ss.get("run_wd", True):
                        st.slider("↳ Wasserstein", 0.0, 1.0, 1.0, 0.01, key="w_wd_metric")
                    if ss.get("run_tvd", True):
                        st.slider("↳ TVD", 0.0, 1.0, 1.0, 0.01, key="w_tvd_metric")
                    if ss.get("run_hd", True):
                        st.slider("↳ Hellinger", 0.0, 1.0, 1.0, 0.01, key="w_hd_metric")
                if bi_active:
                    st.slider("Bivariate", 0.0, 1.0, DEFAULT_FIDELITY_WEIGHTS[1], 0.01, key="w_bi")
                    active_bi = [l for k, l in [("run_spearman", "Spearman"), ("run_contingency", "Contingency"), ("run_pcd", "PCD")] if ss.get(k, True)]
                    st.caption("Includes: " + " + ".join(active_bi) if active_bi else "")
                if multi_active:
                    st.slider("Multivariate", 0.0, 1.0, DEFAULT_FIDELITY_WEIGHTS[2], 0.01, key="w_multi")
                    st.caption(_fidelity_sub_label("run_cc", "AUC-ROC", "run_pmse", "pMSE"))
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

    uni_active = run_uni and (ss.get("run_wd", True) or ss.get("run_tvd", True) or ss.get("run_hd", True))
    bi_active = run_bi and (ss.get("run_spearman", True) or ss.get("run_contingency", True) or ss.get("run_pcd", True))
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
    univariate_metric_weights = {}
    if run_uni and ss.get("run_wd", True):
        univariate_metric_weights["wasserstein"] = ss.get("w_wd_metric", 1.0)
    if run_uni and ss.get("run_tvd", True):
        univariate_metric_weights["tvd"] = ss.get("w_tvd_metric", 1.0)
    if run_uni and ss.get("run_hd", True):
        univariate_metric_weights["hellinger"] = ss.get("w_hd_metric", 1.0)
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
    return fidelity_weights, univariate_metric_weights, miss_weights, composite_weights


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

    fidelity_weights, univariate_metric_weights, miss_weights, composite_weights = _get_weights()

    f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights, univariate_metric_weights=univariate_metric_weights) if fid_res else {}
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
            hd_res = fid_res["univariate"].get("hellinger")
            hd_vals = hd_res.details.get("hellinger_values", {}) if hd_res else {}
            hd_col_scores = hd_res.column_scores or {} if hd_res else {}

            if num_cols and (wd_res or hd_res):
                st.markdown("**Numerical columns — CDF comparison**")
                wd_vals = wd_res.details.get("raw_distances", {}) if wd_res else {}
                wd_col_scores = wd_res.column_scores or {} if wd_res else {}

                # Sort by descending WD if available, else HD
                sort_key = wd_vals if wd_vals else {c: hd_vals.get(c, 0) for c in num_cols}
                sorted_num = sorted(num_cols, key=lambda c: -sort_key.get(c, 0))

                n_cols = min(3, len(sorted_num))
                for row_start in range(0, len(sorted_num), n_cols):
                    row_cols = sorted_num[row_start: row_start + n_cols]
                    cols_st = st.columns(n_cols)
                    for col_st, col in zip(cols_st, row_cols):
                        with col_st:
                            fig = P.plot_numerical_cdf(
                                real[col], synth[col], col,
                                wasserstein_distance=wd_vals.get(col),
                                hellinger_distance=hd_vals.get(col),
                                synth_label=selected,
                                synth_color=P.SYNTH_COLORS[0],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            score_parts = []
                            if col in wd_col_scores:
                                score_parts.append(f"WD score: {wd_col_scores[col]:.3f}")
                            if col in hd_col_scores:
                                score_parts.append(f"HD score: {hd_col_scores[col]:.3f}")
                            if score_parts:
                                st.caption("  |  ".join(score_parts))

            if cat_cols and (tvd_res or hd_res):
                st.markdown("**Categorical columns — frequency comparison**")
                tvd_vals = tvd_res.details.get("tvd_values", {}) if tvd_res else {}
                r_freqs = tvd_res.details.get("real_frequencies", {}) if tvd_res else {}
                s_freqs = tvd_res.details.get("synth_frequencies", {}) if tvd_res else {}
                tvd_col_scores = tvd_res.column_scores or {} if tvd_res else {}

                sort_key = tvd_vals if tvd_vals else {c: hd_vals.get(c, 0) for c in cat_cols}
                sorted_cat = sorted(cat_cols, key=lambda c: -sort_key.get(c, 0))

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
                                hellinger_distance=hd_vals.get(col),
                                synth_label=selected,
                                synth_color=P.SYNTH_COLORS[0],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            score_parts = []
                            if col in tvd_col_scores:
                                score_parts.append(f"TVD score: {tvd_col_scores[col]:.3f}")
                            if col in hd_col_scores:
                                score_parts.append(f"HD score: {hd_col_scores[col]:.3f}")
                            if score_parts:
                                st.caption("  |  ".join(score_parts))

    # ------------------------------------------------------------------
    # Bivariate
    # ------------------------------------------------------------------
    if "bivariate" in fid_res:
        with st.expander("Bivariate associations", expanded=True):
            sp_res = fid_res["bivariate"].get("spearman")
            ct_res = fid_res["bivariate"].get("contingency")
            pcd_res = fid_res["bivariate"].get("pcd")

            col_types = st.session_state.get("col_types") or {}
            all_cols = list(col_types.keys())

            if sp_res and "columns" in sp_res.details:
                st.markdown("**Spearman correlation** (numerical columns)")
                real_corr = pd.DataFrame(sp_res.details["real_correlation_matrix"])
                synth_corr = pd.DataFrame(sp_res.details["synth_correlation_matrix"])
                fig = P.plot_correlation_heatmaps(real_corr, synth_corr, synth_label=selected)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Mean |Δ| = {sp_res.details.get('mean_absolute_difference', 0):.4f}  |  Score: {sp_res.score:.3f}")

            if ct_res and ct_res.details.get("pair_tvds"):
                st.markdown("**Contingency TVD** (categorical and mixed pairs)")
                pair_tvds = ct_res.details["pair_tvds"]
                fig = P.plot_contingency_tvd_heatmap(
                    pair_tvds, all_cols, col_types, synth_label=selected
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Mean TVD = {float(np.mean(list(pair_tvds.values()))):.4f}  |  Score: {ct_res.score:.3f}")

            if pcd_res and pcd_res.details.get("pair_differences"):
                st.markdown("**Pairwise Correlation Difference** (phi-k, all columns)")
                fig = P.plot_pcd_heatmaps(
                    pcd_res.details.get("pair_real", {}),
                    pcd_res.details.get("pair_synth", {}),
                    pcd_res.details["pair_differences"],
                    all_cols,
                    synth_label=selected,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Mean |Δ| = {pcd_res.details.get('mean_absolute_difference', 0):.4f}  |  "
                    f"Score: {pcd_res.score:.3f}  |  "
                    f"t-test p = {pcd_res.details.get('p_value', float('nan')):.4f}"
                    + ("  ✱ significant" if pcd_res.details.get("significant_difference") else "")
                )

    # ------------------------------------------------------------------
    # Multivariate
    # ------------------------------------------------------------------
    if "multivariate" in fid_res:
        with st.expander("Multivariate metrics", expanded=True):
            cc_res = fid_res["multivariate"].get("auc_roc")
            pmse_res = fid_res["multivariate"].get("propensity_mse")
            crcl_rs_res = fid_res["multivariate"].get("crcl_rs")
            crcl_sr_res = fid_res["multivariate"].get("crcl_sr")

            m_cols = st.columns(2)
            if cc_res:
                with m_cols[0]:
                    st.metric("AUC-ROC (CV mean)",
                              f"{cc_res.details.get('mean_auroc', 0):.4f}",
                              help="0.5 = indistinguishable; 1.0 = perfectly separable")
                    st.caption(f"Score: {cc_res.score:.3f}")
                    if cc_res.details.get("oob_auroc") is not None:
                        st.caption(
                            f"OOB AUROC: {cc_res.details['oob_auroc']:.4f}  |  "
                            f"OOB fidelity score: {cc_res.details['oob_fidelity_score']:.4f}"
                        )
                    n_used = cc_res.details.get("n_real_used")
                    n_total = cc_res.details.get("n_real")
                    if n_used is not None and n_used != n_total:
                        st.caption(f"Rows used (complete case): {n_used} / {n_total}")
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

            for crcl_res, mode_label in [(crcl_rs_res, "RS"), (crcl_sr_res, "SR")]:
                if not crcl_res:
                    continue
                per_var = crcl_res.details.get("per_variable", {})
                mean_ratio = crcl_res.details.get("mean_ratio")
                skipped = crcl_res.details.get("skipped", [])
                st.markdown(f"**CrCl-{mode_label}**")
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.metric(
                        f"CrCl-{mode_label} score",
                        f"{crcl_res.score:.4f}",
                        help="1 = ratio of 1.0 (perfect transfer); 0 = large deviation from 1",
                    )
                with info_cols[1]:
                    if mean_ratio is not None:
                        st.metric("Mean ratio (perf_other / perf_held)", f"{mean_ratio:.4f}",
                                  help="Ratio > 1: synthetic generalises better than held-out real; < 1: worse")
                if skipped:
                    st.caption(f"Skipped columns: {', '.join(skipped)}")
                if per_var:
                    fig = P.plot_crcl_scores(per_var, mode=mode_label, synth_label=selected)
                    st.plotly_chart(fig, use_container_width=True)
                    fig = P.plot_crcl_ratios(per_var, mode=mode_label, synth_label=selected)
                    st.plotly_chart(fig, use_container_width=True)

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

            shared_cols = [c for c in real.columns if c in synth.columns]
            miss_pattern_cols = st.columns(2)
            with miss_pattern_cols[0]:
                st.markdown("**Real — missingness pattern**")
                fig = P.plot_missingness_pattern_heatmap(real[shared_cols], title="Real: missingness pattern")
                st.plotly_chart(fig, use_container_width=True)
            with miss_pattern_cols[1]:
                st.markdown(f"**{selected} — missingness pattern**")
                fig = P.plot_missingness_pattern_heatmap(synth[shared_cols], title=f"{selected}: missingness pattern")
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

    fidelity_weights, univariate_metric_weights, miss_weights, composite_weights = _get_weights()

    st.divider()

    # ------------------------------------------------------------------
    # Compute scores for all datasets
    # ------------------------------------------------------------------
    rows = []
    axis_scores: Dict[str, Dict[str, float]] = {}  # {name: {axis: score}}

    for name in synths:
        fid_res = fid_all.get(name, {})
        miss_res = miss_all.get(name, {})

        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights, univariate_metric_weights=univariate_metric_weights) if fid_res else {}
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



# ===========================================================================
# Tab 3: Score summary report
# ===========================================================================

def _tab_score_summary():
    st.header("Score Summary Report")

    fid_all = st.session_state["fidelity_results"]
    miss_all = st.session_state["missingness_results"]
    synths = st.session_state["synth_dfs"]

    if not synths or (not fid_all and not miss_all):
        st.info("Run evaluation first (sidebar → **▶ Run evaluation**).")
        return

    fidelity_weights, univariate_metric_weights, miss_weights, composite_weights = _get_weights()
    run_names = list(synths.keys())

    # ------------------------------------------------------------------
    # Compute summary scores (same logic as benchmarking tab)
    # ------------------------------------------------------------------
    summary_rows = []
    for name in run_names:
        fid_res = fid_all.get(name, {})
        miss_res = miss_all.get(name, {})
        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights, univariate_metric_weights=univariate_metric_weights) if fid_res else {}
        m_scores = compute_missingness_score(miss_res, weights=miss_weights) if miss_res else {}
        comp = compute_composite_score(f_scores, m_scores, weights=composite_weights) if (f_scores or m_scores) else {}
        row = {"Dataset": name}
        if f_scores:
            row.update({
                "Fidelity": f_scores.get("overall"),
                "  Univariate": f_scores.get("univariate"),
                "  Bivariate": f_scores.get("bivariate"),
                "  Multivariate": f_scores.get("multivariate"),
            })
        if m_scores:
            row["Missingness"] = m_scores.get("overall")
        if comp:
            row["Composite"] = comp.get("composite")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    def _score_table(df_rows, index_col, score_cols):
        df = pd.DataFrame(df_rows).set_index(index_col)
        non_score = [c for c in df.columns if c not in score_cols]
        col_order = non_score + score_cols
        return df[col_order].style.format(
            {c: "{:.4f}" for c in score_cols}, na_rep="—"
        ).background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=1)

    # ------------------------------------------------------------------
    # Table 1 — individual metric scores
    # ------------------------------------------------------------------
    st.subheader("Individual metric scores")

    metric_label_map = {
        ("univariate", "wasserstein"): ("Wasserstein Distance", "Univariate"),
        ("univariate", "tvd"): ("Total Variation Distance", "Univariate"),
        ("univariate", "hellinger"): ("Hellinger Distance", "Univariate"),
        ("bivariate", "spearman"): ("Spearman Correlation", "Bivariate"),
        ("bivariate", "contingency"): ("Contingency Matrix", "Bivariate"),
        ("bivariate", "pcd"): ("Pairwise Correlation Difference", "Bivariate"),
        ("multivariate", "auc_roc"): ("AUC-ROC", "Multivariate"),
        ("multivariate", "propensity_mse"): ("Propensity MSE", "Multivariate"),
        ("multivariate", "crcl_rs"): ("CrCl-RS", "Multivariate"),
        ("multivariate", "crcl_sr"): ("CrCl-SR", "Multivariate"),
        ("missingness", "rate"): ("Missingness Rate", "Missingness"),
        ("missingness", "set_distribution"): ("Pattern Distribution", "Missingness"),
        ("missingness", "missing_auroc"): ("Classifier AUROC", "Missingness"),
        ("missingness", "dependency_structure"): ("Dependency Structure", "Missingness"),
    }
    metric_rows = []
    for (group, key), (label, group_label) in metric_label_map.items():
        row = {"Metric": label, "Group": group_label}
        found = False
        for name in run_names:
            res = miss_all.get(name, {}).get(key) if group == "missingness" \
                else fid_all.get(name, {}).get(group, {}).get(key)
            if res is not None:
                row[name] = round(res.score, 4)
                found = True
            else:
                row[name] = None
        if found:
            metric_rows.append(row)

    if metric_rows:
        st.dataframe(
            _score_table(metric_rows, "Metric", run_names),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Table 2 — metric group scores
    # ------------------------------------------------------------------
    st.subheader("Metric group scores")

    group_label_map = {
        "  Univariate": "Univariate (fidelity)",
        "  Bivariate": "Bivariate (fidelity)",
        "  Multivariate": "Multivariate (fidelity)",
        "Missingness": "Missingness",
    }
    group_rows = []
    for col, label in group_label_map.items():
        if col not in summary_df.columns:
            continue
        row = {"Group": label}
        for name in run_names:
            match = summary_df.loc[summary_df["Dataset"] == name, col]
            row[name] = round(float(match.values[0]), 4) if len(match) > 0 and pd.notna(match.values[0]) else None
        group_rows.append(row)

    if group_rows:
        st.dataframe(
            _score_table(group_rows, "Group", run_names),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Table 3 — axis and composite scores
    # ------------------------------------------------------------------
    st.subheader("Axis and composite scores")

    axis_label_map = {
        "Fidelity": "Fidelity",
        "Missingness": "Missingness",
        "Composite": "Composite",
    }
    axis_rows = []
    for col, label in axis_label_map.items():
        if col not in summary_df.columns:
            continue
        row = {"Axis": label}
        for name in run_names:
            match = summary_df.loc[summary_df["Dataset"] == name, col]
            row[name] = round(float(match.values[0]), 4) if len(match) > 0 and pd.notna(match.values[0]) else None
        axis_rows.append(row)

    if axis_rows:
        st.dataframe(
            _score_table(axis_rows, "Axis", run_names),
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Helper: normalise a weight dict, keeping only present keys
    # ------------------------------------------------------------------
    def _norm_dict(d: dict) -> dict:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else d

    # ------------------------------------------------------------------
    # Build per-variable table
    # ------------------------------------------------------------------
    # Normalised weights for group-level fidelity
    fid_group_names = ["univariate", "bivariate", "multivariate"]
    fid_group_w_raw = dict(zip(fid_group_names, fidelity_weights))

    # Normalised weights within univariate metrics
    uni_metric_w_norm = _norm_dict(univariate_metric_weights) if univariate_metric_weights else {}

    # Normalised weights within missingness metrics
    miss_metric_names = ["rate", "set_distribution", "missing_auroc", "dependency_structure"]
    miss_metric_w_raw = dict(zip(miss_metric_names, miss_weights))

    rows = []

    # --- Per-variable rows for WD, TVD, Hellinger ---
    # Collect all columns seen across runs
    for metric_key, metric_label, col_type_filter in [
        ("wasserstein", "Wasserstein Distance", "numerical"),
        ("tvd", "Total Variation Distance", "categorical"),
        ("hellinger", "Hellinger Distance", None),  # covers all columns
    ]:
        col_types = st.session_state["col_types"] or {}
        if col_type_filter:
            variable_cols = [c for c, t in col_types.items() if t == col_type_filter]
        else:
            variable_cols = list(col_types.keys())

        # Collect all variables seen for this metric
        all_vars: set = set()
        for name in run_names:
            fid_res = fid_all.get(name, {})
            uni = fid_res.get("univariate", {})
            res = uni.get(metric_key)
            if res and res.column_scores:
                all_vars.update(res.column_scores.keys())
        all_vars = sorted(all_vars & set(variable_cols)) or sorted(all_vars)

        uni_w_total = sum(fidelity_weights[:1]) if fidelity_weights else 0.0  # univariate group weight (raw)

        for var in all_vars:
            row = {
                "Metric": metric_label,
                "Variable": var,
            }
            # Normalised weight contribution of this metric×variable to fidelity score
            if uni_metric_w_norm and metric_key in uni_metric_w_norm:
                metric_share = uni_metric_w_norm[metric_key]
            else:
                n_active = len(uni_metric_w_norm) if uni_metric_w_norm else 1
                metric_share = 1.0 / n_active if n_active else 0.0
            row["Metric weight (in univariate)"] = f"{metric_share:.3f}"

            for name in run_names:
                fid_res = fid_all.get(name, {})
                uni = fid_res.get("univariate", {})
                res = uni.get(metric_key)
                score = res.column_scores.get(var) if (res and res.column_scores) else None
                row[name] = f"{score:.4f}" if score is not None else "—"

            rows.append(row)

    # --- Group-level summary rows for bivariate, multivariate, missingness ---
    # Normalised fidelity group weights (only present groups)
    present_fid_groups = [
        g for g in fid_group_names
        if any(g in fid_all.get(n, {}) for n in run_names)
    ]
    norm_fid_w = _norm_dict({g: fid_group_w_raw[g] for g in present_fid_groups if fid_group_w_raw.get(g, 0) > 0})

    # --- Per-pair rows for Spearman (pair score = 1 - |diff|) ---
    if any("bivariate" in fid_all.get(n, {}) and "spearman" in fid_all.get(n, {})["bivariate"] for n in run_names):
        all_pairs: set = set()
        for name in run_names:
            res = fid_all.get(name, {}).get("bivariate", {}).get("spearman")
            if res:
                all_pairs.update(res.details.get("pair_differences", {}).keys())
        for pair in sorted(all_pairs):
            col1, col2 = pair.split("|")
            row = {
                "Metric": "Spearman Correlation",
                "Variable": f"{col1} × {col2}",
                "Metric weight (in univariate)": "—",
            }
            for name in run_names:
                res = fid_all.get(name, {}).get("bivariate", {}).get("spearman")
                diff = res.details.get("pair_differences", {}).get(pair) if res else None
                score = (1.0 - diff) if diff is not None else None
                row[name] = f"{score:.4f}" if score is not None else "—"
            rows.append(row)

    # --- Per-pair rows for PCD ---
    if any("bivariate" in fid_all.get(n, {}) and "pcd" in fid_all.get(n, {})["bivariate"] for n in run_names):
        all_pairs = set()
        for name in run_names:
            res = fid_all.get(name, {}).get("bivariate", {}).get("pcd")
            if res:
                all_pairs.update(res.details.get("pair_differences", {}).keys())
        for pair in sorted(all_pairs):
            col1, col2 = pair.split("|")
            row = {
                "Metric": "Pairwise Correlation Difference",
                "Variable": f"{col1} x {col2}",
                "Metric weight (in univariate)": "—",
            }
            for name in run_names:
                res = fid_all.get(name, {}).get("bivariate", {}).get("pcd")
                diff = res.details.get("pair_differences", {}).get(pair) if res else None
                score = (1.0 - diff) if diff is not None else None
                row[name] = f"{score:.4f}" if score is not None else "—"
            rows.append(row)

    # --- Per-pair rows for Contingency (pair score = 1 - TVD) ---
    if any("bivariate" in fid_all.get(n, {}) and "contingency" in fid_all.get(n, {})["bivariate"] for n in run_names):
        all_pairs = set()
        for name in run_names:
            res = fid_all.get(name, {}).get("bivariate", {}).get("contingency")
            if res:
                all_pairs.update(res.details.get("pair_tvds", {}).keys())
        for pair in sorted(all_pairs):
            col1, col2 = pair.split("|")
            row = {
                "Metric": "Contingency TVD",
                "Variable": f"{col1} × {col2}",
                "Metric weight (in univariate)": "—",
            }
            for name in run_names:
                res = fid_all.get(name, {}).get("bivariate", {}).get("contingency")
                tvd = res.details.get("pair_tvds", {}).get(pair) if res else None
                score = (1.0 - tvd) if tvd is not None else None
                row[name] = f"{score:.4f}" if score is not None else "—"
            rows.append(row)

    # --- Group-level summary rows for multivariate ---
    for group_key, group_label, sub_key in [
        ("multivariate", "Multivariate (AUC-ROC)", "auc_roc"),
        ("multivariate", "Multivariate (Propensity MSE)", "propensity_mse"),
        ("multivariate", "Multivariate (CrCl-RS)", "crcl_rs"),
        ("multivariate", "Multivariate (CrCl-SR)", "crcl_sr"),
    ]:
        if not any(sub_key in fid_all.get(n, {}).get(group_key, {}) for n in run_names):
            continue
        row = {
            "Metric": group_label,
            "Variable": "(overall)",
            "Metric weight (in univariate)": "—",
        }
        for name in run_names:
            res = fid_all.get(name, {}).get(group_key, {}).get(sub_key)
            score = res.score if res else None
            row[name] = f"{score:.4f}" if score is not None else "—"
        rows.append(row)

    # --- Missingness metric rows ---
    miss_label_map = {
        "rate": "Missingness Rate",
        "set_distribution": "Missingness Pattern Distribution",
        "missing_auroc": "Missingness Classifier AUROC",
        "dependency_structure": "Missingness Dependency Structure",
    }
    present_miss = [
        m for m in miss_metric_names
        if any(m in miss_all.get(n, {}) for n in run_names)
    ]
    norm_miss_w = _norm_dict({m: miss_metric_w_raw[m] for m in present_miss if miss_metric_w_raw.get(m, 0) > 0})

    for m_key in present_miss:
        row = {
            "Metric": miss_label_map[m_key],
            "Variable": "(overall)",
            "Metric weight (in univariate)": "—",
        }
        for name in run_names:
            res = miss_all.get(name, {}).get(m_key)
            score = res.score if res else None
            row[name] = f"{score:.4f}" if score is not None else "—"
        rows.append(row)

    # --- Weighted score summary rows ---
    separator = {"Metric": "─" * 20, "Variable": "", "Metric weight (in univariate)": ""}
    for name in run_names:
        separator[name] = ""
    rows.append(separator)

    for name in run_names:
        fid_res = fid_all.get(name, {})
        miss_res = miss_all.get(name, {})
        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights, univariate_metric_weights=univariate_metric_weights) if fid_res else {}
        m_scores = compute_missingness_score(miss_res, weights=miss_weights) if miss_res else {}
        comp = compute_composite_score(f_scores, m_scores, weights=composite_weights) if (f_scores or m_scores) else {}

    # One summary row per aggregate score type
    norm_comp_w = _norm_dict(dict(zip(["fidelity", "missingness"], composite_weights)))
    for score_label, score_key, weight_note in [
        ("Univariate score", "univariate", f"group weight: {fidelity_weights[0]:.3f}"),
        ("Bivariate score", "bivariate", f"group weight: {fidelity_weights[1]:.3f}"),
        ("Multivariate score", "multivariate", f"group weight: {fidelity_weights[2]:.3f}"),
        ("Fidelity score (overall)", "overall_fidelity", f"composite weight: {norm_comp_w.get('fidelity', 0):.3f}"),
        ("Missingness score (overall)", "overall_missingness", f"composite weight: {norm_comp_w.get('missingness', 0):.3f}"),
        ("Composite score", "composite", ""),
    ]:
        row = {
            "Metric": score_label,
            "Variable": "",
            "Metric weight (in univariate)": weight_note,
        }
        for name in run_names:
            fid_res = fid_all.get(name, {})
            miss_res = miss_all.get(name, {})
            f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights, univariate_metric_weights=univariate_metric_weights) if fid_res else {}
            m_scores = compute_missingness_score(miss_res, weights=miss_weights) if miss_res else {}
            comp = compute_composite_score(f_scores, m_scores, weights=composite_weights) if (f_scores or m_scores) else {}

            if score_key == "composite":
                val = comp.get("composite")
            elif score_key == "overall_fidelity":
                val = f_scores.get("overall")
            elif score_key == "overall_missingness":
                val = m_scores.get("overall")
            else:
                val = f_scores.get(score_key)
            row[name] = f"{val:.4f}" if val is not None else "—"
        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    st.subheader("Per-variable and per-metric scores across runs")
    st.caption(
        "Metric weight (in univariate) shows the normalised weight of that metric "
        "within the univariate group score. Weight notes in the summary rows show "
        "how each group contributes to the final composite score."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv,
        file_name="score_summary.csv",
        mime="text/csv",
    )


# ===========================================================================
# Tab 4: Metric correlation analysis
# ===========================================================================

def _tab_metric_correlation():
    st.header("Metric Correlation Analysis")
    st.warning("TO-DO: Quality check the logic")
    st.caption(
        "How much do the different metrics agree with each other? "
        "**Across variables**: do metrics identify the same variables as well/poorly reproduced? "
        "**Across runs**: do metrics agree on which synthetic dataset is best?"
    )

    fid_all = st.session_state["fidelity_results"]
    synths = st.session_state["synth_dfs"]

    if not synths or not fid_all:
        st.info("Run evaluation first (sidebar → **▶ Run evaluation**).")
        return

    run_names = list(synths.keys())
    col_types = st.session_state["col_types"] or {}
    num_cols = [c for c, t in col_types.items() if t == "numerical"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]


    # ------------------------------------------------------------------
    # Section 1: Across-variable correlation (per run)
    # ------------------------------------------------------------------
    st.subheader("1 · Across-variable metric agreement")
    st.caption(
        "Each cell shows the Pearson correlation of two metrics' per-variable (or per-pair) "
        "scores. High correlation means both metrics agree on which variables are "
        "well/poorly reproduced."
    )

    run_sel = st.selectbox("Select run", run_names, key="mc_run_sel")
    fid_res = fid_all.get(run_sel, {})
    uni = fid_res.get("univariate", {})
    bi = fid_res.get("bivariate", {})

    # Build score vectors keyed by variable/pair
    metric_vectors: dict[str, dict[str, float]] = {}

    # Univariate metrics — per column scores
    for mkey, mlabel in [
        ("wasserstein", "Wasserstein (num)"),
        ("tvd", "TVD (cat)"),
        ("hellinger", "Hellinger"),
    ]:
        res = uni.get(mkey)
        if res and res.column_scores:
            metric_vectors[mlabel] = dict(res.column_scores)

    # Spearman — per pair score = 1 - |diff|
    sp_res = bi.get("spearman")
    if sp_res:
        pair_diffs = sp_res.details.get("pair_differences", {})
        metric_vectors["Spearman (pairs)"] = {k: 1.0 - v for k, v in pair_diffs.items()}

    # Contingency — per pair score = 1 - TVD
    ct_res = bi.get("contingency")
    if ct_res:
        pair_tvds = ct_res.details.get("pair_tvds", {})
        metric_vectors["Contingency (pairs)"] = {k: 1.0 - v for k, v in pair_tvds.items()}

    # PCD — per pair score = 1 - |diff|
    pcd_res = bi.get("pcd")
    if pcd_res:
        pcd_diffs = pcd_res.details.get("pair_differences", {})
        metric_vectors["PCD (pairs)"] = {k: 1.0 - v for k, v in pcd_diffs.items()}

    if len(metric_vectors) < 2:
        st.info("Need at least 2 metrics with per-variable scores to compute correlations.")
    else:
        # Build a DataFrame: index = union of all variables/pairs, columns = metrics
        all_keys = sorted(set(k for v in metric_vectors.values() for k in v))
        score_df = pd.DataFrame(
            {label: [vec.get(k, float("nan")) for k in all_keys] for label, vec in metric_vectors.items()},
            index=all_keys,
        )

        # Pearson correlation between metric columns (pairwise, ignoring NaN)
        corr_df = score_df.corr(method="pearson")

        st.plotly_chart(P.plot_metric_correlation_heatmap(corr_df, f"Metric agreement across variables — {run_sel}"), use_container_width=True)

        with st.expander("Raw scores per variable / pair"):
            st.dataframe(
                score_df.style.format("{:.4f}", na_rep="—"),
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Section 2: Across-run correlation (all runs)
    # ------------------------------------------------------------------
    st.subheader("2 · Across-run metric agreement")
    st.caption(
        "Each cell shows the Pearson correlation of two metrics' overall scores across runs. "
        "High correlation means both metrics agree on which synthetic dataset is best. "
        "Hellinger is split into numerical and categorical subsets so it is directly "
        "comparable to Wasserstein (numerical only) and TVD (categorical only)."
    )

    if len(run_names) < 2:
        st.info("Need at least 2 runs to compute across-run correlations.")
        return

    # Build a DataFrame: index = run names, columns = metrics (overall scores)
    # Hellinger is split into numerical and categorical subsets so that it is
    # directly comparable to WD (numerical only) and TVD (categorical only).
    run_metric_scores: dict[str, dict[str, float]] = {name: {} for name in run_names}

    for name in run_names:
        fres = fid_all.get(name, {})
        uni = fres.get("univariate", {})

        # WD — numerical columns only
        wd_res = uni.get("wasserstein")
        if wd_res is not None:
            run_metric_scores[name]["Wasserstein (num)"] = wd_res.score

        # TVD — categorical columns only
        tvd_res = uni.get("tvd")
        if tvd_res is not None:
            run_metric_scores[name]["TVD (cat)"] = tvd_res.score

        # Hellinger split by column type so comparisons are like-for-like
        hd_res = uni.get("hellinger")
        if hd_res and hd_res.column_scores:
            num_scores = [v for c, v in hd_res.column_scores.items() if col_types.get(c) == "numerical"]
            cat_scores = [v for c, v in hd_res.column_scores.items() if col_types.get(c) == "categorical"]
            if num_scores:
                run_metric_scores[name]["Hellinger (num)"] = float(np.mean(num_scores))
            if cat_scores:
                run_metric_scores[name]["Hellinger (cat)"] = float(np.mean(cat_scores))

        # Bivariate and multivariate — already type-specific by design
        for group, mkey, label in [
            ("bivariate", "spearman", "Spearman"),
            ("bivariate", "contingency", "Contingency"),
            ("bivariate", "pcd", "PCD"),
            ("multivariate", "auc_roc", "AUC-ROC"),
            ("multivariate", "propensity_mse", "Propensity MSE"),
            ("multivariate", "crcl_rs", "CrCl-RS"),
            ("multivariate", "crcl_sr", "CrCl-SR"),
        ]:
            res = fres.get(group, {}).get(mkey)
            if res is not None:
                run_metric_scores[name][label] = res.score

    run_df = pd.DataFrame(run_metric_scores).T  # rows = runs, cols = metrics
    run_df = run_df.dropna(axis=1, how="all")

    if run_df.shape[1] < 2:
        st.info("Need at least 2 metrics with overall scores to compute across-run correlations.")
        return

    run_corr = run_df.corr(method="pearson")
    st.plotly_chart(P.plot_metric_correlation_heatmap(run_corr, "Metric agreement across runs"), use_container_width=True)

    with st.expander("Overall scores per run"):
        st.dataframe(
            run_df.style.format("{:.4f}", na_rep="—"),
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
| **Fidelity** | ✅ Available | Wasserstein Distance, TVD, Spearman Correlation, Contingency Matrix, AUC-ROC, Propensity MSE |
| **Missingness** | ✅ Available | Missingness Rate, Pattern Distribution, Classifier AUROC, Dependency Structure |
| **Utility** | 🔜 TODO | Downstream task performance |
| **Privacy** | 🔜 TODO | Disclosure risk, membership inference |
        """)
        return

    _weight_controls()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Individual Report", "🏆 Benchmarking Report", "📋 Score Summary", "🔗 Metric Correlations"])
    with tab1:
        _tab_individual()
    with tab2:
        _tab_benchmarking()
    with tab3:
        _tab_score_summary()
    with tab4:
        _tab_metric_correlation()


if __name__ == "__main__":
    run_dashboard()
