"""
Metric registry for the OmniSynth dashboard.

Defines the canonical list of metrics and their UI metadata in one place.
All dashboard functions (sidebar checkboxes, weight controls, result filtering,
weight extraction) loop over these structures rather than hardcoding metric
names individually.

To add a new metric to the dashboard:
  1. Add it to the relevant group in FIDELITY_GROUPS or to MISSINGNESS_METRICS.
  2. Register it in the evaluation layer (evaluation/fidelity.py or missingness.py).
  Nothing else needs to change.

Structure
---------
FIDELITY_GROUPS  — list of fidelity group dicts, each containing a 'metrics' list.
MISSINGNESS_METRICS — flat list of missingness metric dicts.

Fidelity group fields
    key             str   Results-dict key (e.g. "univariate").
    label           str   Display label for the group checkbox / weight slider.
    run_key         str   session_state key for the group-level checkbox.
    w_key           str   session_state key for the group weight slider.
    default_weight  float Default value for the group weight slider.
    metrics         list  See fidelity sub-metric fields below.

Fidelity sub-metric fields
    key          str   Key in the results sub-dict (e.g. "wasserstein").
    label        str   Full display label used in the sidebar checkbox.
    short_label  str   Short label used in weight-control captions.
    run_key      str   session_state key for the sub-metric checkbox.

Missingness metric fields
    key             str   Key in the MissingnessResults dict.
    label           str   Display label.
    run_key         str   session_state key for the checkbox.
    w_key           str   session_state key for the weight slider.
    default_weight  float Default value for the weight slider.
"""

from __future__ import annotations

from omnisynth.config import (
    DEFAULT_FIDELITY_WEIGHTS,
    DEFAULT_MISSINGNESS_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Fidelity groups
# ---------------------------------------------------------------------------

FIDELITY_GROUPS: list[dict] = [
    {
        "key": "univariate",
        "label": "Univariate",
        "run_key": "run_uni",
        "w_key": "w_uni",
        "default_weight": DEFAULT_FIDELITY_WEIGHTS[0],
        "metrics": [
            {
                "key": "wasserstein",
                "label": "Wasserstein Distance",
                "short_label": "Wasserstein",
                "run_key": "run_wd",
            },
            {
                "key": "tvd",
                "label": "Total Variation Distance",
                "short_label": "TVD",
                "run_key": "run_tvd",
            },
            {
                "key": "hellinger",
                "label": "Hellinger Distance",
                "short_label": "Hellinger",
                "run_key": "run_hd",
            },
        ],
    },
    {
        "key": "bivariate",
        "label": "Bivariate",
        "run_key": "run_bi",
        "w_key": "w_bi",
        "default_weight": DEFAULT_FIDELITY_WEIGHTS[1],
        "metrics": [
            {
                "key": "spearman",
                "label": "Spearman Correlation",
                "short_label": "Spearman",
                "run_key": "run_spearman",
            },
            {
                "key": "contingency",
                "label": "Contingency Matrix",
                "short_label": "Contingency",
                "run_key": "run_contingency",
            },
            {
                "key": "pcd",
                "label": "Pairwise Correlation Diff",
                "short_label": "PCD",
                "run_key": "run_pcd",
            },
        ],
    },
    {
        "key": "multivariate",
        "label": "Multivariate",
        "run_key": "run_multi",
        "w_key": "w_multi",
        "default_weight": DEFAULT_FIDELITY_WEIGHTS[2],
        "metrics": [
            {
                "key": "auc_roc",
                "label": "AUC-ROC",
                "short_label": "AUC-ROC",
                "run_key": "run_cc",
            },
            {
                "key": "propensity_mse",
                "label": "Propensity MSE",
                "short_label": "pMSE",
                "run_key": "run_pmse",
            },
            {
                "key": "crcl_rs",
                "label": "CrCl-RS (train real, test synth)",
                "short_label": "CrCl-RS",
                "run_key": "run_crcl_rs",
            },
            {
                "key": "crcl_sr",
                "label": "CrCl-SR (train synth, test real)",
                "short_label": "CrCl-SR",
                "run_key": "run_crcl_sr",
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Missingness metrics
# ---------------------------------------------------------------------------

MISSINGNESS_METRICS: list[dict] = [
    {
        "key": "rate",
        "label": "Rate",
        "run_key": "run_miss_rate",
        "w_key": "w_rate",
        "default_weight": DEFAULT_MISSINGNESS_WEIGHTS[0],
    },
    {
        "key": "set_distribution",
        "label": "Pattern Distribution",
        "run_key": "run_miss_set",
        "w_key": "w_set",
        "default_weight": DEFAULT_MISSINGNESS_WEIGHTS[1],
    },
    {
        "key": "missing_auroc",
        "label": "Missing AUROC",
        "run_key": "run_miss_auroc",
        "w_key": "w_auroc",
        "default_weight": DEFAULT_MISSINGNESS_WEIGHTS[2],
    },
    {
        "key": "dependency_structure",
        "label": "Dependency Structure",
        "run_key": "run_miss_dep",
        "w_key": "w_dep",
        "default_weight": DEFAULT_MISSINGNESS_WEIGHTS[3],
    },
]

# ---------------------------------------------------------------------------
# Helpers used by the dashboard
# ---------------------------------------------------------------------------

def group_is_active(group: dict, ss: dict) -> bool:
    """Return True if the group checkbox is on and at least one sub-metric is on."""
    return ss.get(group["run_key"], True) and any(
        ss.get(m["run_key"], True) for m in group["metrics"]
    )
