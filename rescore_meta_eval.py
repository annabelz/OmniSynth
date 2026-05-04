"""
rescore_meta_eval.py — Re-score a meta-eval results JSON using a config-specified
subset of metrics and weights.

Usage
-----
    python rescore_meta_eval.py \\
        --config datasets/diabetes/diabetes_meta_eval_CC_config.yaml \\
        --output path/to/rescored_results.json

What it does
------------
1. Loads the existing results JSON at config.results_path.
2. For every per_dataset row, re-computes fidelity_overall, missingness_overall,
   and composite_score using only the metrics marked `true` in the config and
   the group/axis weights defined there.
3. Re-aggregates mean ± std per scenario from the updated per_dataset values.
4. Writes the new results to --output (the original file is never modified).

Individual per-metric scores (fidelity_wasserstein, missingness_rate, …) are
preserved unchanged in each per_dataset row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stdg_eval.meta_eval.config import load_meta_eval_config

# ---------------------------------------------------------------------------
# Metric groupings  (must match evaluate_fidelity / compute_fidelity_score)
# ---------------------------------------------------------------------------
FIDELITY_GROUPS: Dict[str, List[str]] = {
    "univariate":   ["wasserstein", "tvd", "hellinger"],
    "bivariate":    ["spearman", "contingency", "pcd"],
    "multivariate": ["auc_roc", "propensity_mse", "crcl_rs", "crcl_sr"],
}
FIDELITY_GROUP_ORDER = ["univariate", "bivariate", "multivariate"]

MISSINGNESS_METRIC_ORDER = ["rate", "set_distribution", "missing_auroc", "dependency_structure"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalise(weights: List[float]) -> List[float]:
    arr = np.array(weights, dtype=float)
    total = arr.sum()
    if total == 0:
        raise ValueError("Weight vector must not sum to zero.")
    return (arr / total).tolist()


def _recompute_fidelity(
    row: Dict,
    enabled: Dict[str, bool],
    group_weights: Optional[List[float]],
) -> Optional[float]:
    """
    Re-compute fidelity_overall from stored per-metric floats in *row*.

    Group score = mean of enabled metrics within the group.
    Overall = weighted average across groups present in *row*.
    Returns None if no metrics are available.
    """
    group_scores: Dict[str, float] = {}
    for group, metrics in FIDELITY_GROUPS.items():
        active = [
            m for m in metrics
            if enabled.get(m, True) and f"fidelity_{m}" in row
            and row[f"fidelity_{m}"] is not None
        ]
        if active:
            group_scores[group] = float(np.mean([row[f"fidelity_{m}"] for m in active]))

    if not group_scores:
        return None

    present = [g for g in FIDELITY_GROUP_ORDER if g in group_scores]

    if group_weights:
        full_w = dict(zip(FIDELITY_GROUP_ORDER, group_weights))
        w = _normalise([full_w.get(g, 1.0) for g in present])
    else:
        w = _normalise([1.0] * len(present))

    return float(np.dot([group_scores[g] for g in present], w))


def _recompute_missingness(
    row: Dict,
    enabled: Dict[str, bool],
    metric_weights: Optional[List[float]],
) -> Optional[float]:
    """
    Re-compute missingness_overall from stored per-metric floats in *row*.
    """
    active = [
        m for m in MISSINGNESS_METRIC_ORDER
        if enabled.get(m, True) and f"missingness_{m}" in row
        and row[f"missingness_{m}"] is not None
    ]
    if not active:
        return None

    if metric_weights:
        full_w = dict(zip(MISSINGNESS_METRIC_ORDER, metric_weights))
        w = _normalise([full_w.get(m, 1.0) for m in active])
    else:
        w = _normalise([1.0] * len(active))

    return float(np.dot([row[f"missingness_{m}"] for m in active], w))


def _recompute_composite(
    fidelity_overall: Optional[float],
    missingness_overall: Optional[float],
    composite_weights: Optional[List[float]],
) -> Optional[float]:
    available = [s for s in [fidelity_overall, missingness_overall] if s is not None]
    if not available:
        return None
    if fidelity_overall is None:
        return missingness_overall
    if missingness_overall is None:
        return fidelity_overall
    w = _normalise(composite_weights or [1.0, 1.0])
    return float(w[0] * fidelity_overall + w[1] * missingness_overall)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _reaggregate(combined_per_dataset: List[Dict]) -> Dict:
    """Re-build fidelity / missingness / composite aggregated blocks from per_dataset rows."""
    fid_lists: Dict[str, List[float]] = {}
    miss_lists: Dict[str, List[float]] = {}
    comp_list: List[float] = []

    for row in combined_per_dataset:
        for k, v in row.items():
            if k.startswith("fidelity_") and isinstance(v, (int, float)):
                fid_lists.setdefault(k[len("fidelity_"):], []).append(float(v))
            elif k.startswith("missingness_") and isinstance(v, (int, float)):
                miss_lists.setdefault(k[len("missingness_"):], []).append(float(v))
            elif k == "composite_score" and isinstance(v, (int, float)):
                comp_list.append(float(v))

    out: Dict = {}
    if fid_lists:
        out["fidelity"] = {k: _stats(v) for k, v in fid_lists.items()}
    if miss_lists:
        out["missingness"] = {k: _stats(v) for k, v in miss_lists.items()}
    if comp_list:
        out["composite"] = _stats(comp_list)
    return out


# ---------------------------------------------------------------------------
# Main rescoring logic
# ---------------------------------------------------------------------------

def rescore(config_path: str, output_path: str) -> None:
    cfg = load_meta_eval_config(config_path)

    results_file = Path(cfg.results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        data = json.load(f)

    # Extract enabled metrics and weights from config
    _metrics = cfg.metrics or {}
    fid_enabled: Dict[str, bool] = {
        m: bool(_metrics.get("fidelity", {}).get(m, True))
        for group in FIDELITY_GROUPS.values() for m in group
    }
    miss_enabled: Dict[str, bool] = {
        m: bool(_metrics.get("missingness", {}).get(m, True))
        for m in MISSINGNESS_METRIC_ORDER
    }

    _weights = cfg.weights or {}
    w_fidelity = _weights.get("fidelity")
    w_missingness = _weights.get("missingness")
    w_composite = _weights.get("composite")

    run_fidelity = "fidelity" in cfg.axes
    run_missingness = "missingness" in cfg.axes

    # Report configuration
    enabled_fid = [m for m, v in fid_enabled.items() if v]
    enabled_miss = [m for m, v in miss_enabled.items() if v]
    print(f"Config:  {config_path}")
    print(f"Source:  {results_file}")
    print(f"Output:  {output_path}")
    print(f"Fidelity metrics    : {', '.join(enabled_fid) if run_fidelity else '(axis disabled)'}")
    print(f"Missingness metrics : {', '.join(enabled_miss) if run_missingness else '(axis disabled)'}")
    print(f"Weights fidelity    : {w_fidelity or 'equal (default)'}")
    print(f"Weights missingness : {w_missingness or 'equal (default)'}")
    print(f"Weights composite   : {w_composite or 'equal (default)'}")
    print(f"\nRescoring {len(data)} scenario entries...")

    rescored: Dict = {}
    for key, entry in data.items():
        new_per_dataset = []
        for row in entry["per_dataset"]:
            row = dict(row)

            # Remove disabled metrics from the row
            for m, enabled in fid_enabled.items():
                if not enabled:
                    row.pop(f"fidelity_{m}", None)
            for m, enabled in miss_enabled.items():
                if not enabled:
                    row.pop(f"missingness_{m}", None)

            if run_fidelity:
                f_overall = _recompute_fidelity(row, fid_enabled, w_fidelity)
                if f_overall is not None:
                    row["fidelity_overall"] = f_overall
            else:
                f_overall = None
                row.pop("fidelity_overall", None)

            if run_missingness:
                m_overall = _recompute_missingness(row, miss_enabled, w_missingness)
                if m_overall is not None:
                    row["missingness_overall"] = m_overall
            else:
                m_overall = None
                row.pop("missingness_overall", None)

            comp = _recompute_composite(
                f_overall if run_fidelity else None,
                m_overall if run_missingness else None,
                w_composite,
            )
            if comp is not None:
                row["composite_score"] = comp

            new_per_dataset.append(row)

        new_entry = {
            "n_datasets": entry["n_datasets"],
            "sample_size": entry["sample_size"],
            "per_dataset": new_per_dataset,
        }
        new_entry.update(_reaggregate(new_per_dataset))
        rescored[key] = new_entry

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rescored, indent=2))
    print(f"Done — written to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score a meta-eval results JSON using config-specified metrics and weights."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the meta-eval YAML config file.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the rescored results JSON.",
    )
    args = parser.parse_args()
    rescore(args.config, args.output)


if __name__ == "__main__":
    main()
