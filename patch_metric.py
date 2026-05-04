"""
patch_metric.py — Re-run a single metric on existing CSVs and update the results JSON.

Only the targeted metric's value is recomputed; all other stored values are
preserved.  Overall scores (missingness_overall, fidelity_overall,
composite_score) and scenario-level aggregates (mean/std) are then
recalculated from the updated per_dataset rows.

Usage
-----
    python patch_metric.py \\
        --config datasets/diabetes/diabetes_meta_eval_CC_config.yaml \\
        --metric missing_auroc \\
        --output path/to/patched_results.json

    # Overwrite the original file in-place:
    python patch_metric.py \\
        --config datasets/diabetes/diabetes_meta_eval_CC_config.yaml \\
        --metric missing_auroc

Supported metric names
----------------------
  Missingness : rate, set_distribution, missing_auroc, dependency_structure
  Fidelity    : wasserstein, tvd, hellinger, spearman, contingency, pcd,
                auc_roc, propensity_mse, crcl_rs, crcl_sr
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stdg_eval.meta_eval.config import load_meta_eval_config
from stdg_eval.utils.data_utils import detect_column_types

# ---------------------------------------------------------------------------
# Metric registry  (name → (axis, callable that returns MetricResult))
# ---------------------------------------------------------------------------

def _build_registry():
    from stdg_eval.metrics.missingness.measures import (
        MissingnessRate,
        MissingnessSetDistribution,
        MissingnessClassifierAUROC,
        MissingnessDependencyStructure,
    )
    from stdg_eval.metrics.fidelity.univariate import (
        WassersteinDistance,
        TotalVariationDistance,
        HellingerDistance,
    )
    from stdg_eval.metrics.fidelity.bivariate import (
        SpearmanCorrelation,
        ContingencyMatrix,
        PairwiseCorrelationDifference,
    )
    from stdg_eval.metrics.fidelity.multivariate import (
        AucRoc,
        PropensityMSE,
        CrossClassificationRS,
        CrossClassificationSR,
    )

    return {
        # missingness
        "rate":                 ("missingness", MissingnessRate()),
        "set_distribution":     ("missingness", MissingnessSetDistribution()),
        "missing_auroc":        ("missingness", MissingnessClassifierAUROC()),
        "dependency_structure": ("missingness", MissingnessDependencyStructure()),
        # fidelity — univariate
        "wasserstein": ("fidelity", WassersteinDistance()),
        "tvd":         ("fidelity", TotalVariationDistance()),
        "hellinger":   ("fidelity", HellingerDistance()),
        # fidelity — bivariate
        "spearman":    ("fidelity", SpearmanCorrelation()),
        "contingency": ("fidelity", ContingencyMatrix()),
        "pcd":         ("fidelity", PairwiseCorrelationDifference()),
        # fidelity — multivariate
        "auc_roc":         ("fidelity", AucRoc()),
        "propensity_mse":  ("fidelity", PropensityMSE()),
        "crcl_rs":         ("fidelity", CrossClassificationRS()),
        "crcl_sr":         ("fidelity", CrossClassificationSR()),
    }


# ---------------------------------------------------------------------------
# Scoring helpers  (mirrored from rescore_meta_eval.py)
# ---------------------------------------------------------------------------

FIDELITY_GROUPS = {
    "univariate":   ["wasserstein", "tvd", "hellinger"],
    "bivariate":    ["spearman", "contingency", "pcd"],
    "multivariate": ["auc_roc", "propensity_mse", "crcl_rs", "crcl_sr"],
}
FIDELITY_GROUP_ORDER   = ["univariate", "bivariate", "multivariate"]
MISSINGNESS_METRIC_ORDER = ["rate", "set_distribution", "missing_auroc", "dependency_structure"]


def _normalise(weights: List[float]) -> List[float]:
    arr = np.array(weights, dtype=float)
    total = arr.sum()
    if total == 0:
        raise ValueError("Weight vector sums to zero.")
    return (arr / total).tolist()


def _recompute_fidelity(row, enabled, group_weights):
    group_scores = {}
    for group, metrics in FIDELITY_GROUPS.items():
        active = [m for m in metrics
                  if enabled.get(m, True)
                  and f"fidelity_{m}" in row
                  and row[f"fidelity_{m}"] is not None]
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


def _recompute_missingness(row, enabled, metric_weights):
    active = [m for m in MISSINGNESS_METRIC_ORDER
              if enabled.get(m, True)
              and f"missingness_{m}" in row
              and row[f"missingness_{m}"] is not None]
    if not active:
        return None
    if metric_weights:
        full_w = dict(zip(MISSINGNESS_METRIC_ORDER, metric_weights))
        w = _normalise([full_w.get(m, 1.0) for m in active])
    else:
        w = _normalise([1.0] * len(active))
    return float(np.dot([row[f"missingness_{m}"] for m in active], w))


def _recompute_composite(f_overall, m_overall, composite_weights):
    if f_overall is None and m_overall is None:
        return None
    if f_overall is None:
        return m_overall
    if m_overall is None:
        return f_overall
    w = _normalise(composite_weights or [1.0, 1.0])
    return float(w[0] * f_overall + w[1] * m_overall)


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _reaggregate(combined_per_dataset: List[Dict]) -> Dict:
    fid_lists:  Dict[str, List[float]] = {}
    miss_lists: Dict[str, List[float]] = {}
    comp_list:  List[float] = []
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
# Main patch logic
# ---------------------------------------------------------------------------

def patch(config_path: str, metric_name: str, output_path: Optional[str],
          input_path: Optional[str] = None) -> None:
    registry = _build_registry()
    if metric_name not in registry:
        raise ValueError(
            f"Unknown metric {metric_name!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    axis, metric_obj = registry[metric_name]
    col_key = f"{axis}_{metric_name}"   # e.g. "missingness_missing_auroc"

    cfg = load_meta_eval_config(config_path)
    real = pd.read_csv(cfg.input_data)
    col_types = detect_column_types(real, override=cfg.column_types)

    results_file = Path(input_path) if input_path else Path(cfg.results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file) as f:
        data = json.load(f)

    out_path = Path(output_path) if output_path else results_file

    # Scoring config
    _metrics = cfg.metrics or {}
    _weights = cfg.weights or {}
    fid_enabled  = {m: bool(_metrics.get("fidelity", {}).get(m, True))
                    for g in FIDELITY_GROUPS.values() for m in g}
    miss_enabled = {m: bool(_metrics.get("missingness", {}).get(m, True))
                    for m in MISSINGNESS_METRIC_ORDER}
    w_fidelity   = _weights.get("fidelity")
    w_missingness = _weights.get("missingness")
    w_composite  = _weights.get("composite")

    print(f"Config  : {config_path}")
    print(f"Metric  : {metric_name}  ({axis}, stored as '{col_key}')")
    print(f"Source  : {results_file}")
    print(f"Output  : {out_path}")
    print(f"\nPatching {len(data)} scenario entries...\n")

    for key, entry in data.items():
        per_dataset = entry["per_dataset"]
        sample_size = entry.get("sample_size")
        n = len(per_dataset)

        print(f"  {key}  ({n} replicates, sample_size={sample_size})", flush=True)

        for i, row in enumerate(per_dataset):
            synth_path = row.get("path")
            if not synth_path or not Path(synth_path).exists():
                print(f"    [{i+1}/{n}] SKIP — CSV not found: {synth_path}")
                continue

            synth = pd.read_csv(synth_path)

            # Reconstruct the real reference used during the original evaluation
            if sample_size is not None:
                n_rows = min(sample_size, len(real))
                real_ref = real.sample(n=n_rows, random_state=cfg.random_seed + i).reset_index(drop=True)
            else:
                real_ref = real

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = metric_obj.evaluate(real_ref, synth, col_types)

            row[col_key] = float(result.score)

            # Recompute overall scores from updated row
            if axis == "fidelity":
                f_overall = _recompute_fidelity(row, fid_enabled, w_fidelity)
                m_overall = row.get("missingness_overall")
                if f_overall is not None:
                    row["fidelity_overall"] = f_overall
            else:
                f_overall = row.get("fidelity_overall")
                m_overall = _recompute_missingness(row, miss_enabled, w_missingness)
                if m_overall is not None:
                    row["missingness_overall"] = m_overall

            comp = _recompute_composite(f_overall, m_overall, w_composite)
            if comp is not None:
                row["composite_score"] = comp

        # Re-aggregate scenario-level stats from updated rows
        entry.update(_reaggregate(per_dataset))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nDone — written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run a single metric on existing CSVs and patch the results JSON."
    )
    parser.add_argument("--config",  required=True,
                        help="Path to the meta-eval YAML config file.")
    parser.add_argument("--metric",  required=True,
                        help="Metric name to re-run (e.g. 'missing_auroc').")
    parser.add_argument("--input",   default=None,
                        help="Input results JSON. Defaults to results_path from config.")
    parser.add_argument("--output",  default=None,
                        help="Output path. Defaults to overwriting the source results file.")
    args = parser.parse_args()
    patch(args.config, args.metric, args.output, input_path=args.input)


if __name__ == "__main__":
    main()
