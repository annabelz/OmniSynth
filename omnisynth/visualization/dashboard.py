"""
Streamlit interactive dashboard for OmniSynth.

Launch via:
    streamlit run run_dashboard.py
or:
    OmniSynth dashboard --config configs/my_config.yaml

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
from scipy.stats import rankdata

from omnisynth.config import (
    DEFAULT_COMPOSITE_WEIGHTS,
    DEFAULT_FIDELITY_WEIGHTS,
    DEFAULT_MISSINGNESS_WEIGHTS,
)
from omnisynth.evaluation.fidelity import evaluate_fidelity
from omnisynth.evaluation.missingness import evaluate_missingness
from omnisynth.evaluation.scoring import (
    compute_composite_score,
    compute_fidelity_score,
    compute_missingness_score,
)
from omnisynth.utils.data_utils import detect_column_types, load_dataset, load_config, validate_column_types, eval_config_from_dict, weights_from_dict
from omnisynth.utils.precomputed_io import load_precomputed
from omnisynth.visualization import plots as P
from omnisynth.visualization.metric_registry import (
    FIDELITY_GROUPS,
    MISSINGNESS_METRICS,
    group_is_active,
)


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OmniSynth · Synthetic Data Evaluation",
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
        "meta_eval_results": None,   # loaded from meta_eval_results path in config
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ===========================================================================
# Sidebar – data loading
# ===========================================================================

def _sidebar():
    st.sidebar.title("OmniSynth")
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
                # Meta-evaluation results
                if "meta_eval_results" in cfg:
                    try:
                        import json as _json
                        with open(cfg["meta_eval_results"]) as _f:
                            st.session_state["meta_eval_results"] = _json.load(_f)
                    except Exception as exc:
                        st.sidebar.warning(f"Could not load meta-eval results: {exc}")
                # Metric enable flags from config — only applied when a new config
                # file is loaded (detected via content hash). After initial load the
                # sidebar checkboxes are the source of truth and are not overwritten,
                # so the user can freely toggle metrics that were disabled in the config.
                if st.session_state.get("_config_hash") != cfg_hash:
                    if "metrics" in cfg:
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
                    if "weights" in cfg:
                        import numpy as _np
                        w = weights_from_dict(cfg)
                        fid_w = w.get("fidelity")
                        if fid_w:
                            arr = _np.array(fid_w, dtype=float)
                            arr = arr / arr.sum()
                            for g, v in zip(FIDELITY_GROUPS, arr):
                                st.session_state[g["w_key"]] = float(v)
                        miss_w = w.get("missingness")
                        if miss_w:
                            arr = _np.array(miss_w, dtype=float)
                            arr = arr / arr.sum()
                            for m, v in zip(MISSINGNESS_METRICS, arr):
                                st.session_state[m["w_key"]] = float(v)
                        comp_w = w.get("composite")
                        if comp_w and len(comp_w) >= 2:
                            arr = _np.array(comp_w[:2], dtype=float)
                            arr = arr / arr.sum()
                            st.session_state["w_fid"] = float(arr[0])
                            st.session_state["w_miss"] = float(arr[1])
                st.session_state["_config_hash"] = cfg_hash

                # --- Load summary note ---
                note_lines = [f"**Real dataset:** `{cfg['real_data']}`"]
                synth_entries = cfg.get("synthetic_datasets", [])
                if synth_entries:
                    note_lines.append(f"**Synthetic datasets** ({len(synth_entries)}):")
                    for entry in synth_entries:
                        note_lines.append(f"- `{entry['name']}`: `{entry['path']}`")
                precomputed_loaded = bool(st.session_state.get("precomputed_results"))
                note_lines.append(
                    f"**Precomputed results:** {'loaded' if precomputed_loaded else 'not found'}"
                )
                meta_loaded = bool(st.session_state.get("meta_eval_results"))
                note_lines.append(
                    f"**Meta-eval results:** {'loaded' if meta_loaded else 'not found'}"
                )
                st.sidebar.info("\n\n".join(note_lines))
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

    # Precomputed results status (loaded from config via precomputed_results key)
    if st.session_state.get("precomputed_results"):
        n_synths = len(st.session_state["precomputed_results"])
        st.sidebar.caption(f"Precomputed results loaded for {n_synths} dataset(s).")

    with st.sidebar.expander("Metric options", expanded=False):
        for group in FIDELITY_GROUPS:
            run_group = st.checkbox(group["label"], value=True, key=group["run_key"])
            for m in group["metrics"]:
                st.checkbox(f"↳ {m['label']}", value=True, key=m["run_key"], disabled=not run_group)

        run_miss = st.checkbox("Missingness", value=True, key="run_miss")
        for m in MISSINGNESS_METRICS:
            st.checkbox(f"↳ {m['label']}", value=True, key=m["run_key"], disabled=not run_miss)

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
        _run_evaluation()


def _run_evaluation():
    ss = st.session_state
    real = ss["real_df"]
    synths = ss["synth_dfs"]
    col_types = ss["col_types"]
    precomputed = ss.get("precomputed_results", {})

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

    run_any_fidelity = any(ss.get(g["run_key"], True) for g in FIDELITY_GROUPS)
    run_miss = ss.get("run_miss", True)

    for i, (name, synth) in enumerate(synths.items()):
        progress.progress((i) / n, text=f"Evaluating {name}…")

        precomp = precomputed.get(name, {})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if run_any_fidelity:
                # Skip groups already covered by precomputed results
                precomputed_groups = {
                    g["key"]: ss.get(g["run_key"], True) and bool(precomp.get(g["key"]))
                    for g in FIDELITY_GROUPS
                }
                res = evaluate_fidelity(
                    real, synth, col_types=col_types,
                    **{
                        f"run_{g['key']}": ss.get(g["run_key"], True) and not precomputed_groups[g["key"]]
                        for g in FIDELITY_GROUPS
                    },
                )
                # Inject precomputed groups
                for g in FIDELITY_GROUPS:
                    if precomputed_groups[g["key"]]:
                        res.setdefault(g["key"], {}).update(precomp[g["key"]])

                # Post-filter sub-metrics the user deselected
                for g in FIDELITY_GROUPS:
                    if g["key"] in res:
                        for m in g["metrics"]:
                            if not ss.get(m["run_key"], True):
                                res[g["key"]].pop(m["key"], None)
                        if not res[g["key"]]:
                            del res[g["key"]]

                fidelity_results[name] = res

            if run_miss:
                miss_precomputed = bool(precomp.get("missingness"))
                if miss_precomputed:
                    miss_res = dict(precomp["missingness"])
                    # Post-filter sub-metrics the user deselected
                    for m in MISSINGNESS_METRICS:
                        if not ss.get(m["run_key"], True):
                            miss_res.pop(m["key"], None)
                    missingness_results[name] = miss_res
                else:
                    missingness_results[name] = evaluate_missingness(
                        real, synth, col_types=col_types,
                        **{f"run_{m['key']}": ss.get(m["run_key"], True) for m in MISSINGNESS_METRICS},
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
    active_fid_groups = [g for g in FIDELITY_GROUPS if group_is_active(g, ss)]
    active_miss_metrics = [
        m for m in MISSINGNESS_METRICS
        if ss.get("run_miss", True) and ss.get(m["run_key"], True)
    ]
    has_fidelity = bool(active_fid_groups)
    has_missingness = bool(active_miss_metrics)

    with st.expander("Weighting scheme", expanded=False):
        st.caption(
            "Adjust the weights for each metric group and evaluation axis. "
            "Weights are automatically normalised to sum to 1."
        )
        weight_cols = st.columns(2)
        with weight_cols[0]:
            st.markdown("**Fidelity weights**")
            if has_fidelity:
                for g in active_fid_groups:
                    st.slider(g["label"], 0.0, 1.0, g["default_weight"], 0.01, key=g["w_key"])
                    active_subs = [m["short_label"] for m in g["metrics"] if ss.get(m["run_key"], True)]
                    st.caption("Includes: " + " + ".join(active_subs))
            else:
                st.caption("No fidelity metrics selected.")
        with weight_cols[1]:
            st.markdown("**Missingness weights**")
            if has_missingness:
                for m in active_miss_metrics:
                    st.slider(m["label"], 0.0, 1.0, m["default_weight"], 0.01, key=m["w_key"])
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


def _get_weights() -> tuple:
    """Read the current weight values from session state for enabled metrics only."""
    ss = st.session_state
    run_miss = ss.get("run_miss", True)

    fidelity_weights = [
        ss.get(g["w_key"], g["default_weight"]) if group_is_active(g, ss) else 0.0
        for g in FIDELITY_GROUPS
    ]

    miss_weights = [
        ss.get(m["w_key"], m["default_weight"]) if (run_miss and ss.get(m["run_key"], True)) else 0.0
        for m in MISSINGNESS_METRICS
    ]

    any_fid_active = any(group_is_active(g, ss) for g in FIDELITY_GROUPS)
    any_miss_active = run_miss and any(ss.get(m["run_key"], True) for m in MISSINGNESS_METRICS)

    composite_weights = [
        ss.get("w_fid", DEFAULT_COMPOSITE_WEIGHTS[0]) if any_fid_active else 0.0,
        ss.get("w_miss", DEFAULT_COMPOSITE_WEIGHTS[1]) if any_miss_active else 0.0,
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
                    std_auroc = cc_res.details.get("std_auroc")
                    auroc_label = f"{cc_res.details.get('mean_auroc', 0):.4f}"
                    if std_auroc is not None:
                        auroc_label += f" ± {std_auroc:.4f}"
                    st.metric("AUC-ROC (CV mean ± std)",
                              auroc_label,
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
                              help="Lower = better; range [0, 0.25]; 0 = perfect fidelity")
                    st.caption(f"Score: {pmse_res.score:.3f}  (= 1 − 4 × pMSE)")
                    st.caption(
                        f"Synthetic fraction: {pmse_res.details.get('c_synthetic_fraction', 0):.3f} | "
                        f"Worst case: {pmse_res.details.get('pmse_worst_case', 0.25):.2f}"
                    )
                    prop_scores = pmse_res.details.get("propensity_scores")
                    prop_labels = pmse_res.details.get("labels")
                    if prop_scores and prop_labels:
                        fig = P.plot_propensity_histogram(prop_scores, prop_labels, synth_label=selected)
                        st.plotly_chart(fig, use_container_width=True)

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
            auroc_res = miss_res.get("missing_auroc")
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

            st.markdown("**Missingness pattern frequency (UpSet)**")
            upset_cols = st.columns(2)
            with upset_cols[0]:
                fig = P.plot_missingness_upset(
                    real[shared_cols],
                    title="Real",
                    bar_color=P.REAL_COLOR,
                )
                st.plotly_chart(fig, use_container_width=True)
            with upset_cols[1]:
                fig = P.plot_missingness_upset(
                    synth[shared_cols],
                    title=selected,
                    bar_color=P.SYNTH_COLORS[0],
                    score=set_res.score if set_res else None,
                )
                st.plotly_chart(fig, use_container_width=True)

            if set_res:
                st.caption(
                    f"Pattern distribution TVD: {set_res.details.get('tvd', 0):.4f}  |  "
                    f"Unique patterns (real): {set_res.details.get('n_unique_real_patterns', 'n/a')}  |  "
                    f"Unique patterns ({selected}): {set_res.details.get('n_unique_synth_patterns', 'n/a')}"
                )

            if dep_res and "columns" in dep_res.details:
                dep_cols = dep_res.details["columns"]
                if len(dep_cols) >= 2:
                    st.markdown("**Missingness dependency structure**")
                    real_dep = pd.DataFrame(dep_res.details["real_correlation_matrix"])
                    synth_dep = pd.DataFrame(dep_res.details["synth_correlation_matrix"])
                    dep_plot_cols = st.columns(3)
                    with dep_plot_cols[0]:
                        st.plotly_chart(
                            P.plot_missingness_dependency(real_dep, "Real"),
                            use_container_width=True,
                        )
                    with dep_plot_cols[1]:
                        st.plotly_chart(
                            P.plot_missingness_dependency(synth_dep, selected),
                            use_container_width=True,
                        )
                    with dep_plot_cols[2]:
                        st.plotly_chart(
                            P.plot_missingness_dependency_diff(real_dep, synth_dep, "Difference (real − synth)"),
                            use_container_width=True,
                        )

            if auroc_res and "auroc_real" in auroc_res.details:
                st.markdown("**Missingness AUROC per variable**")
                fig = P.plot_missing_auroc(
                    auroc_res.details["auroc_real"],
                    auroc_res.details["auroc_synth"],
                    synth_label=selected,
                )
                st.plotly_chart(fig, use_container_width=True)



def go_bar_folds(fold_aurocs: list) -> "go.Figure":
    """Small bar chart for cross-validation fold AUROCs with mean ± std error band."""
    import plotly.graph_objects as go
    import numpy as np

    mean_val = float(np.mean(fold_aurocs))
    std_val = float(np.std(fold_aurocs))

    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(fold_aurocs))],
        y=fold_aurocs,
        marker_color=P.REAL_COLOR,
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="0.5 (ideal)", annotation_position="top right")
    # Mean line with ± 1 std shaded band
    fig.add_hline(
        y=mean_val,
        line_dash="solid",
        line_color="darkorange",
        line_width=1.5,
        annotation_text=f"mean={mean_val:.3f}",
        annotation_position="top left",
        annotation_font_color="darkorange",
    )
    fig.add_hrect(
        y0=mean_val - std_val,
        y1=mean_val + std_val,
        fillcolor="darkorange",
        opacity=0.12,
        line_width=0,
        annotation_text=f"±1 std ({std_val:.3f})",
        annotation_position="bottom right",
        annotation_font_color="darkorange",
        annotation_font_size=10,
    )
    fig.update_layout(
        title="AUROC per CV fold",
        yaxis=dict(range=[0, 1], title="AUROC"),
        height=280,
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

    fidelity_weights, miss_weights, composite_weights = _get_weights()
    run_names = list(synths.keys())

    # ------------------------------------------------------------------
    # Compute summary scores (same logic as benchmarking tab)
    # ------------------------------------------------------------------
    summary_rows = []
    for name in run_names:
        fid_res = fid_all.get(name, {})
        miss_res = miss_all.get(name, {})
        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights) if fid_res else {}
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
        **{
            (g["key"], m["key"]): (m["label"], g["label"])
            for g in FIDELITY_GROUPS
            for m in g["metrics"]
        },
        **{
            ("missingness", m["key"]): (m["label"], "Missingness")
            for m in MISSINGNESS_METRICS
        },
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
    fid_group_w_raw = {g["key"]: w for g, w in zip(FIDELITY_GROUPS, fidelity_weights)}

    # Equal weighting across active univariate metrics
    uni_group = next(g for g in FIDELITY_GROUPS if g["key"] == "univariate")
    n_active_uni = sum(
        1 for m in uni_group["metrics"]
        if any(m["key"] in fid_all.get(n, {}).get("univariate", {}) for n in run_names)
    )

    # Normalised weights within missingness metrics
    miss_metric_w_raw = {m["key"]: w for m, w in zip(MISSINGNESS_METRICS, miss_weights)}

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
            metric_share = 1.0 / n_active_uni if n_active_uni else 0.0
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
        g["key"] for g in FIDELITY_GROUPS
        if any(g["key"] in fid_all.get(n, {}) for n in run_names)
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
    miss_label_map = {m["key"]: f"Missingness {m['label']}" for m in MISSINGNESS_METRICS}
    present_miss = [
        m["key"] for m in MISSINGNESS_METRICS
        if any(m["key"] in miss_all.get(n, {}) for n in run_names)
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
        f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights) if fid_res else {}
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
            f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights) if fid_res else {}
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
        "**Across runs**: do metrics agree on which synthetic dataset is best? "
        "**Across variables**: do metrics identify the same variables as well/poorly reproduced?"
    )

    fid_all = st.session_state["fidelity_results"]
    miss_all = st.session_state.get("missingness_results", {})
    synths = st.session_state["synth_dfs"]

    if not synths or not fid_all:
        st.info("Run evaluation first (sidebar → **▶ Run evaluation**).")
        return

    run_names = list(synths.keys())
    col_types = st.session_state["col_types"] or {}

    # ------------------------------------------------------------------
    # Section 1: Across-run correlation (all runs)
    # ------------------------------------------------------------------
    st.subheader("1 · Across-run metric agreement")
    st.caption(
        "Each cell shows the Pearson correlation of two metrics' overall scores across runs. "
        "High correlation means both metrics agree on which synthetic dataset is best. "
        "Hellinger is split into numerical and categorical subsets so it is directly "
        "comparable to Wasserstein (numerical only) and TVD (categorical only)."
    )

    if len(run_names) < 2:
        st.info("Need at least 2 runs to compute across-run correlations.")
    else:
        fid_scores: dict[str, dict[str, float]] = {name: {} for name in run_names}
        miss_scores: dict[str, dict[str, float]] = {name: {} for name in run_names}

        for name in run_names:
            fres = fid_all.get(name, {})
            mres = miss_all.get(name, {})

            # Fidelity — driven by registry
            for group in FIDELITY_GROUPS:
                grp_results = fres.get(group["key"], {})
                for m in group["metrics"]:
                    res = grp_results.get(m["key"])
                    if res is None:
                        continue
                    if m["key"] == "hellinger" and res.column_scores:
                        num_scores = [v for c, v in res.column_scores.items() if col_types.get(c) == "numerical"]
                        cat_scores = [v for c, v in res.column_scores.items() if col_types.get(c) == "categorical"]
                        if num_scores:
                            fid_scores[name][f"{m['short_label']} (num)"] = float(np.mean(num_scores))
                        if cat_scores:
                            fid_scores[name][f"{m['short_label']} (cat)"] = float(np.mean(cat_scores))
                    else:
                        fid_scores[name][m["short_label"]] = res.score

            # Missingness — driven by registry
            for m in MISSINGNESS_METRICS:
                res = mres.get(m["key"])
                if res is not None:
                    miss_scores[name][m["label"]] = res.score

        fid_df = pd.DataFrame(fid_scores).T.dropna(axis=1, how="all")
        miss_df = pd.DataFrame(miss_scores).T.dropna(axis=1, how="all")

        corr_cols = st.columns(2)
        with corr_cols[0]:
            if fid_df.shape[1] < 2:
                st.info("Need at least 2 fidelity metrics across runs.")
            else:
                fid_corr = fid_df.corr(method="pearson")
                st.plotly_chart(P.plot_metric_correlation_heatmap(fid_corr, "Fidelity metric agreement across runs"), use_container_width=True)
                with st.expander("Fidelity scores per run"):
                    st.dataframe(fid_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

        with corr_cols[1]:
            if miss_df.shape[1] < 2:
                st.info("Need at least 2 missingness metrics across runs.")
            else:
                miss_corr = miss_df.corr(method="pearson")
                st.plotly_chart(P.plot_metric_correlation_heatmap(miss_corr, "Missingness metric agreement across runs"), use_container_width=True)
                with st.expander("Missingness scores per run"):
                    st.dataframe(miss_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

    # ------------------------------------------------------------------
    # Section 2: Across-variable correlation (per run)
    # ------------------------------------------------------------------
    st.subheader("2 · Across-variable metric agreement")
    st.caption(
        "Each cell shows the Pearson correlation of two metrics' per-variable (or per-pair) "
        "scores. High correlation means both metrics agree on which variables are "
        "well/poorly reproduced."
    )

    run_sel = st.selectbox("Select run", run_names, key="mc_run_sel")
    fid_res = fid_all.get(run_sel, {})
    uni = fid_res.get("univariate", {})
    bi = fid_res.get("bivariate", {})

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
        all_keys = sorted(set(k for v in metric_vectors.values() for k in v))
        score_df = pd.DataFrame(
            {label: [vec.get(k, float("nan")) for k in all_keys] for label, vec in metric_vectors.items()},
            index=all_keys,
        )
        corr_df = score_df.corr(method="pearson")
        st.plotly_chart(P.plot_metric_correlation_heatmap(corr_df, f"Metric agreement across variables — {run_sel}"), use_container_width=True)

        with st.expander("Raw scores per variable / pair"):
            st.dataframe(
                score_df.style.format("{:.4f}", na_rep="—"),
                use_container_width=True,
            )


# ===========================================================================
# Tab 0: Dataset description
# ===========================================================================

def _tab_dataset_description():
    ss = st.session_state
    real: pd.DataFrame = ss["real_df"]
    synths: Dict[str, pd.DataFrame] = ss["synth_dfs"]
    col_types: dict = ss["col_types"] or {}

    st.header("Dataset description")

    # --- Real dataset summary ---
    st.subheader("Real dataset")
    info_cols = st.columns(3)
    info_cols[0].metric("Observations", f"{len(real):,}")
    info_cols[1].metric("Columns", len(real.columns))
    info_cols[2].metric("Missing cells", f"{real.isnull().sum().sum():,} ({100 * real.isnull().mean().mean():.1f}%)")

    col_summary = pd.DataFrame({
        "Column": real.columns,
        "Type": [col_types.get(c, "unknown") for c in real.columns],
        "Missing (%)": [f"{100 * real[c].isnull().mean():.1f}%" for c in real.columns],
        "Unique values": [real[c].nunique() for c in real.columns],
    })
    st.dataframe(col_summary, use_container_width=True, hide_index=True)

    # --- Synthetic datasets summary ---
    st.subheader("Synthetic datasets")

    # Overview table: one row per synthetic dataset
    overview_rows = []
    for name, synth in synths.items():
        n_missing = synth.isnull().sum().sum()
        pct_missing = 100 * synth.isnull().mean().mean()
        extra_cols = [c for c in synth.columns if c not in real.columns]
        missing_from_real = [c for c in real.columns if c not in synth.columns]
        overview_rows.append({
            "Dataset": name,
            "Observations": f"{len(synth):,}",
            "Columns": len(synth.columns),
            "Missing cells": f"{n_missing:,} ({pct_missing:.1f}%)",
            "Extra cols (not in real)": ", ".join(extra_cols) if extra_cols else "—",
            "Missing cols (vs real)": ", ".join(missing_from_real) if missing_from_real else "—",
        })
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    # --- Per-synthetic per-column detail ---
    if synths:
        selected = st.selectbox(
            "Column-level detail for:", list(synths.keys()), key="desc_selected"
        )
        synth = synths[selected]
        shared_cols = [c for c in real.columns if c in synth.columns]
        detail = pd.DataFrame({
            "Column": shared_cols,
            "Type": [col_types.get(c, "unknown") for c in shared_cols],
            "Real — missing (%)": [f"{100 * real[c].isnull().mean():.1f}%" for c in shared_cols],
            f"{selected} — missing (%)": [f"{100 * synth[c].isnull().mean():.1f}%" for c in shared_cols],
            "Real — unique": [real[c].nunique() for c in shared_cols],
            f"{selected} — unique": [synth[c].nunique() for c in shared_cols],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    # --- Raw data preview ---
    st.subheader("Raw data")
    raw_selected = st.selectbox(
        "Synthetic dataset to preview:", list(synths.keys()), key="desc_raw_selected"
    ) if synths else None

    raw_cols = st.columns(2)
    with raw_cols[0]:
        st.markdown("**Real dataset** (first 10 rows)")
        st.dataframe(real.head(10), use_container_width=True)
    with raw_cols[1]:
        if raw_selected:
            st.markdown(f"**{raw_selected}** (first 10 rows)")
            st.dataframe(synths[raw_selected].head(10), use_container_width=True)


# ===========================================================================
# Tab 5: Ranking report (Yan et al. 2022)
# ===========================================================================

def _tab_ranking():
    """
    Ranking mechanism following Yan et al. (2022) — datasets ranked per metric
    (rank 1 = best score), ties receive the average of tied ranks.

    Hierarchy mirrors the scoring system:
      1. Per-metric ranks within each fidelity group → average → group rank
      2. Group ranks weighted by fidelity group weights → fidelity axis rank
      3. Per-metric ranks within missingness weighted by missingness metric weights
         → missingness axis rank
      4. Fidelity + missingness axis ranks weighted by composite weights
         → final rank score  (lower = better)

    Uses the same weight values as set in the weight controls.

    Reference: Yan C, Yan Y, Wan Z, Zhang Z, Omberg L, Guinney J, et al.
    A Multifaceted benchmarking of synthetic electronic health record generation
    models. Nat Commun. 2022 Dec 9;13(1):7609.
    doi:10.1038/s41467-022-35295-1. PMID: 36494374.
    """
    st.header("Ranking Report")
    st.caption(
        "Ranks synthetic datasets relative to each other following Yan et al. (*Nat Commun* 2022). "
        "For each metric, datasets are ranked by score (rank 1 = best; ties → average rank). "
        "Ranks are aggregated using the same hierarchical weights as the scoring system — "
        "**lower final rank score = better**."
    )

    synths = st.session_state["synth_dfs"]
    fid_all = st.session_state["fidelity_results"]
    miss_all = st.session_state["missingness_results"]

    if not synths or (not fid_all and not miss_all):
        st.info("Run evaluation first (sidebar → **▶ Run evaluation**).")
        return

    run_names = list(synths.keys())
    if len(run_names) < 2:
        st.info("Ranking requires at least 2 synthetic datasets.")
        return

    n = len(run_names)
    fidelity_weights, miss_weights, composite_weights = _get_weights()

    # ------------------------------------------------------------------
    # Helper: rank a score dict (higher score → rank 1)
    # ------------------------------------------------------------------
    def _ranks(scores_dict: Dict[str, float]) -> Dict[str, float]:
        eligible = [name for name in run_names if name in scores_dict]
        if len(eligible) < 2:
            return {}
        arr = np.array([scores_dict[name] for name in eligible])
        ranks = rankdata(-arr, method="average")
        return dict(zip(eligible, map(float, ranks)))

    # ------------------------------------------------------------------
    # Per-metric ranks + group-level average ranks (fidelity)
    # ------------------------------------------------------------------
    per_metric_ranks: Dict[str, Dict[str, float]] = {}   # display_label → {dataset: rank}
    fid_group_ranks: Dict[str, Dict[str, float]] = {}    # group_key → {dataset: avg_rank}

    for g, g_weight in zip(FIDELITY_GROUPS, fidelity_weights):
        if g_weight == 0.0:
            continue
        metric_rank_list: List[Dict[str, float]] = []
        for m in g["metrics"]:
            key = m["key"]
            scores = {
                name: fid_all[name][g["key"]][key].score
                for name in run_names
                if name in fid_all
                and g["key"] in fid_all[name]
                and key in fid_all[name][g["key"]]
            }
            if not scores:
                continue
            ranks = _ranks(scores)
            if ranks:
                label = f"{m['label']} ({g['label']})"
                per_metric_ranks[label] = ranks
                metric_rank_list.append(ranks)

        if metric_rank_list:
            # Average per-metric ranks within the group
            all_ds = set().union(*[r.keys() for r in metric_rank_list])
            fid_group_ranks[g["key"]] = {
                ds: float(np.mean([r[ds] for r in metric_rank_list if ds in r]))
                for ds in all_ds
            }

    # ------------------------------------------------------------------
    # Per-metric ranks (missingness)
    # ------------------------------------------------------------------
    miss_metric_ranks: List[Tuple[Dict[str, float], float]] = []  # (ranks, weight)

    for m, m_weight in zip(MISSINGNESS_METRICS, miss_weights):
        if m_weight == 0.0:
            continue
        key = m["key"]
        scores = {
            name: miss_all[name][key].score
            for name in run_names
            if name in miss_all and key in miss_all[name]
        }
        if not scores:
            continue
        ranks = _ranks(scores)
        if ranks:
            label = f"{m['label']} (Missingness)"
            per_metric_ranks[label] = ranks
            miss_metric_ranks.append((ranks, m_weight))

    # ------------------------------------------------------------------
    # Fidelity axis rank = weighted average of group ranks
    # ------------------------------------------------------------------
    fid_axis_ranks: Dict[str, float] = {}
    if fid_group_ranks:
        g_weight_map = {g["key"]: w for g, w in zip(FIDELITY_GROUPS, fidelity_weights)}
        for name in run_names:
            wsum, wtotal = 0.0, 0.0
            for gk, group_ranks in fid_group_ranks.items():
                w = g_weight_map.get(gk, 0.0)
                if name in group_ranks and w > 0:
                    wsum += w * group_ranks[name]
                    wtotal += w
            if wtotal > 0:
                fid_axis_ranks[name] = wsum / wtotal

    # ------------------------------------------------------------------
    # Missingness axis rank = weighted average of metric ranks
    # ------------------------------------------------------------------
    miss_axis_ranks: Dict[str, float] = {}
    if miss_metric_ranks:
        for name in run_names:
            wsum, wtotal = 0.0, 0.0
            for ranks, w in miss_metric_ranks:
                if name in ranks and w > 0:
                    wsum += w * ranks[name]
                    wtotal += w
            if wtotal > 0:
                miss_axis_ranks[name] = wsum / wtotal

    # ------------------------------------------------------------------
    # Final composite rank = weighted average of axis ranks
    # ------------------------------------------------------------------
    w_fid, w_miss = composite_weights[0], composite_weights[1]
    final_scores: Dict[str, float] = {}
    for name in run_names:
        wsum, wtotal = 0.0, 0.0
        if name in fid_axis_ranks and w_fid > 0:
            wsum += w_fid * fid_axis_ranks[name]
            wtotal += w_fid
        if name in miss_axis_ranks and w_miss > 0:
            wsum += w_miss * miss_axis_ranks[name]
            wtotal += w_miss
        if wtotal > 0:
            final_scores[name] = wsum / wtotal

    if not final_scores:
        st.info("Not enough data to compute ranks.")
        return

    # Sort datasets by final rank score (ascending = best first)
    sorted_names = [name for name, _ in sorted(final_scores.items(), key=lambda x: x[1])]

    best = sorted_names[0]
    st.success(f"Best overall dataset: **{best}**")
    st.divider()

    # ------------------------------------------------------------------
    # Bar chart (rank 1 → 1.0, rank N → 0.0 so higher bar = better)
    # ------------------------------------------------------------------
    bar_scores = (
        {name: (n - final_scores[name]) / (n - 1) for name in sorted_names}
        if n > 1 else {name: 1.0 for name in sorted_names}
    )
    fig = P.plot_score_bar(bar_scores, title="Relative ranking (higher = better)")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Fidelity rank table — MultiIndex columns (group → metric)
    # ------------------------------------------------------------------
    if fid_group_ranks:
        st.subheader("Fidelity ranks")
        st.caption(
            "Ranks per metric (1 = best). Column headers show normalised group weights. "
            "**Fidelity rank** = weighted average of group ranks."
        )

        total_fid_w = sum(fidelity_weights)
        col_tuples: List[Tuple[str, str]] = []
        col_data: Dict[Tuple[str, str], Dict[str, float]] = {}

        for g, g_weight in zip(FIDELITY_GROUPS, fidelity_weights):
            if g_weight == 0.0:
                continue
            norm_w = g_weight / total_fid_w if total_fid_w > 0 else 0.0
            g_header = f"{g['label']}  (w={norm_w:.2f})"
            for m in g["metrics"]:
                label_key = f"{m['label']} ({g['label']})"
                if label_key not in per_metric_ranks:
                    continue
                ct = (g_header, m["label"])
                col_tuples.append(ct)
                col_data[ct] = {
                    name: round(per_metric_ranks[label_key].get(name, float("nan")), 2)
                    for name in sorted_names
                }

        ct_total = ("Total", "Fidelity rank")
        col_tuples.append(ct_total)
        col_data[ct_total] = {
            name: round(fid_axis_ranks.get(name, float("nan")), 2)
            for name in sorted_names
        }

        if col_tuples:
            fid_df = pd.DataFrame(
                {ct: col_data[ct] for ct in col_tuples},
                index=sorted_names,
            )
            fid_df.columns = pd.MultiIndex.from_tuples(col_tuples)
            fid_df.index.name = "Dataset"
            st.dataframe(fid_df.style.format("{:.2f}", na_rep="—"), use_container_width=True)

    # ------------------------------------------------------------------
    # Missingness rank table — flat columns with normalised weights in header
    # ------------------------------------------------------------------
    if miss_axis_ranks:
        st.subheader("Missingness ranks")
        st.caption(
            "Ranks per metric (1 = best). Column headers show normalised metric weights. "
            "**Missingness rank** = weighted average of metric ranks."
        )

        total_miss_w = sum(w for _, w in miss_metric_ranks)
        miss_col_data: Dict[str, Dict[str, float]] = {}

        for m, m_weight in zip(MISSINGNESS_METRICS, miss_weights):
            if m_weight == 0.0:
                continue
            label_key = f"{m['label']} (Missingness)"
            if label_key not in per_metric_ranks:
                continue
            norm_w = m_weight / total_miss_w if total_miss_w > 0 else 0.0
            col_name = f"{m['label']}  (w={norm_w:.2f})"
            miss_col_data[col_name] = {
                name: round(per_metric_ranks[label_key].get(name, float("nan")), 2)
                for name in sorted_names
            }

        miss_col_data["Missingness rank"] = {
            name: round(miss_axis_ranks.get(name, float("nan")), 2)
            for name in sorted_names
        }

        miss_df = pd.DataFrame(miss_col_data, index=sorted_names)
        miss_df.index.name = "Dataset"
        st.dataframe(miss_df.style.format("{:.2f}", na_rep="—"), use_container_width=True)


# ===========================================================================
# Tab 6: Meta-evaluation report
# ===========================================================================

def _tab_meta_eval():
    import re as _re

    st.header("Meta-evaluation Report")
    st.caption(
        "Results from running the benchmark on programmatically generated noisy datasets. "
        "Each scenario is evaluated across multiple replicates; points show individual "
        "replicate scores, diamonds show mean ± std."
    )

    meta = st.session_state.get("meta_eval_results")
    if not meta:
        st.info(
            "No meta-evaluation results loaded. "
            "Add ``meta_eval_results: path/to/results.json`` to your config file, "
            "or run ``OmniSynth meta-eval --config <config>`` first."
        )
        return

    # Score keys present across all scenarios
    all_per_ds_keys: set = set()
    for data in meta.values():
        for row in data.get("per_dataset", []):
            all_per_ds_keys.update(k for k, v in row.items() if isinstance(v, float))

    score_label_map = {
        "fidelity_overall":            "Fidelity (overall)",
        "fidelity_univariate":         "Fidelity — Univariate",
        "fidelity_bivariate":          "Fidelity — Bivariate",
        "fidelity_multivariate":       "Fidelity — Multivariate",
        "missingness_overall":         "Missingness (overall)",
        "missingness_rate":            "Missingness — Rate",
        "missingness_set_distribution":"Missingness — Pattern",
        "missingness_missing_auroc":   "Missingness — AUROC",
        "missingness_dependency_structure": "Missingness — Dependency",
        "composite_score":             "Composite",
    }
    available_keys = [k for k in score_label_map if k in all_per_ds_keys]

    if not available_keys:
        st.warning("No numeric score columns found in meta-eval results.")
        return

    # ------------------------------------------------------------------
    # Summary plot — fidelity, missingness, composite on one plot
    # ------------------------------------------------------------------
    st.subheader("Summary across scenarios")
    axis_filter = st.selectbox(
        "Show scenarios for axis",
        ["All", "Fidelity", "Missingness", "Composite"],
        key="meta_summary_axis",
    )
    all_scenarios = list(meta.keys())
    if axis_filter == "Fidelity":
        filtered_scenarios = [s for s in all_scenarios if _re.match(r"^fidelity", s)]
    elif axis_filter == "Missingness":
        filtered_scenarios = [s for s in all_scenarios if _re.match(r"^missingness", s)]
    elif axis_filter == "Composite":
        filtered_scenarios = [s for s in all_scenarios if _re.match(r"^composite", s)]
    else:
        filtered_scenarios = all_scenarios
    filtered_meta = {s: meta[s] for s in filtered_scenarios}
    summary_keys = [k for k in ("fidelity_overall", "missingness_overall", "composite_score") if k in all_per_ds_keys]
    has_sample_sizes = any(
        _re.search(r"_n\d+$", k) or k.endswith("_full")
        for k in filtered_meta
    )
    if summary_keys and filtered_meta:
        if has_sample_sizes:
            fig = P.plot_meta_eval_summary_grouped(filtered_meta, summary_keys, score_label_map)
        else:
            fig = P.plot_meta_eval_summary(filtered_meta, summary_keys, score_label_map)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Per-scenario plots
    # ------------------------------------------------------------------
    st.subheader("Per-scenario breakdown")

    # Fixed top-level keys shown in every plot
    plot_keys = [k for k in ("fidelity_overall", "missingness_overall", "composite_score") if k in all_per_ds_keys]

    # Sub-metric keys shown as tables beneath each plot
    _fidelity_sub_keys = [
        "fidelity_univariate", "fidelity_bivariate", "fidelity_multivariate",
    ]
    _missingness_sub_keys = [
        "missingness_rate", "missingness_set_distribution",
        "missingness_missing_auroc", "missingness_dependency_structure",
    ]
    fidelity_sub_keys    = [k for k in _fidelity_sub_keys    if k in all_per_ds_keys]
    missingness_sub_keys = [k for k in _missingness_sub_keys if k in all_per_ds_keys]
    sub_keys = fidelity_sub_keys + missingness_sub_keys

    # Group result keys by base scenario name when sample sizes are present
    # base_scenario → {size (int or None) → per_dataset list}
    base_scenario_map: Dict[str, Dict] = {}
    for key in meta:
        m = _re.match(r"^(.+?)_n(\d+)$", key)
        if m:
            base, size = m.group(1), int(m.group(2))
        elif key.endswith("_full"):
            base, size = key[:-5], None
        else:
            base, size = key, None
        base_scenario_map.setdefault(base, {})[size] = meta[key].get("per_dataset", [])

    # Preserve config-file ordering by base scenario
    seen_bases: Dict[str, None] = {}
    for key in meta:
        m = _re.match(r"^(.+?)_n(\d+)$", key)
        if m:
            base = m.group(1)
        elif key.endswith("_full"):
            base = key[:-5]
        else:
            base = key
        seen_bases[base] = None
    ordered_bases = list(seen_bases.keys())

    fidelity_bases    = [b for b in ordered_bases if b.startswith("fidelity")]
    missingness_bases = [b for b in ordered_bases if b.startswith("missingness")]
    composite_bases   = [b for b in ordered_bases if b.startswith("composite")]
    other_bases       = [b for b in ordered_bases if b not in fidelity_bases + missingness_bases + composite_bases]

    for group_label, group in [
        ("Fidelity scenarios",    fidelity_bases),
        ("Missingness scenarios",  missingness_bases),
        ("Composite scenarios",    composite_bases),
        ("Other scenarios",        other_bases),
    ]:
        if not group:
            continue
        st.markdown(f"**{group_label}**")
        cols = st.columns(min(len(group), 3))
        for i, base in enumerate(group):
            size_results = base_scenario_map[base]
            with cols[i % 3]:
                if has_sample_sizes and len(size_results) > 1:
                    fig = P.plot_meta_eval_scenario_grouped(
                        base_scenario=base,
                        size_results=size_results,
                        score_keys=plot_keys,
                        score_labels=score_label_map,
                    )
                else:
                    # Single size — use original flat plot
                    per_ds = next(iter(size_results.values()))
                    n = len(per_ds)
                    fig = P.plot_meta_eval_scenario(
                        scenario_name=f"{base}  (n={n})",
                        per_dataset=per_ds,
                        score_keys=plot_keys,
                        score_labels=score_label_map,
                    )
                st.plotly_chart(fig, use_container_width=True)

                # Sub-metric table — one row per (metric, sample size)
                all_per_ds = [row for pd_list in size_results.values() for row in pd_list]
                present_sub = [k for k in sub_keys if any(k in row for row in all_per_ds)]
                if present_sub:
                    table_rows = []
                    for sz, per_ds in sorted(size_results.items(),
                                             key=lambda x: (x[0] is None, x[0] or 0)):
                        size_tag = "full" if sz is None else f"n={sz:,}"
                        for k in present_sub:
                            vals = [row[k] for row in per_ds if k in row]
                            if not vals:
                                continue
                            arr = np.array(vals)
                            table_rows.append({
                                "Metric": score_label_map.get(k, k),
                                "Sample size": size_tag,
                                "Mean": f"{float(np.mean(arr)):.4f}",
                                "Std": f"{float(np.std(arr)):.4f}",
                            })
                    if table_rows:
                        st.dataframe(
                            pd.DataFrame(table_rows),
                            use_container_width=True,
                            hide_index=True,
                        )


# ===========================================================================
# Main app entry point
# ===========================================================================

def run_dashboard():
    """Entry point called by ``run_dashboard.py`` and the CLI."""
    _sidebar()

    if st.session_state["real_df"] is None:
        st.title("OmniSynth · Synthetic Data Evaluation Dashboard")
        st.info(
            "👈 **Get started**: upload your real dataset and one or more synthetic "
            "datasets in the sidebar, then click **▶ Run evaluation**."
        )
        return

    _weight_controls()

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["🗂 Dataset Description", "📊 Individual Report", "🏆 Benchmarking Report", "📋 Score Summary", "🧪 Meta-evaluation"])
    with tab0:
        _tab_dataset_description()
    with tab1:
        _tab_individual()
    with tab2:
        _tab_benchmarking()
    with tab3:
        _tab_score_summary()
    with tab4:
        _tab_meta_eval()


if __name__ == "__main__":
    run_dashboard()
