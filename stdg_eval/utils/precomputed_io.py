"""
Save and load precomputed fidelity metric results.

All fidelity metric groups (univariate, bivariate, multivariate) can be
precomputed once with the CLI::

    stdg-eval precompute --config my_config.yaml --output precomputed.json

and then referenced in the config::

    precomputed_results: precomputed.json

The dashboard injects these results directly and skips recomputation for
any group that is already covered.

JSON schema
-----------
{
  "<synth_name>": {
    "univariate": {
      "wasserstein": {"score": float, "details": {...}},
      "tvd":         {"score": float, "details": {...}},
      "hellinger":   {"score": float, "details": {...}}
    },
    "bivariate": {
      "spearman":    {"score": float, "details": {...}},
      "contingency": {"score": float, "details": {...}},
      "pcd":         {"score": float, "details": {...}}
    },
    "multivariate": {
      "auc_roc": {"score": float, "details": {...}},
      "propensity_mse":       {"score": float, "details": {...}}
    }
  },
  ...
}

Only the groups/metrics that were computed are present; absent keys are
simply not precomputed and will be computed at evaluation time.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict

import numpy as np

from stdg_eval.metrics.base import MetricResult


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (bool, int, str, type(None))):
        return obj
    # Fallback — try str so the file at least writes
    return str(obj)


def _from_serializable(obj: Any) -> Any:
    """Convert JSON null → float('nan') for numeric leaf values."""
    if isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    # Leave None as None — callers that expect a float will handle it
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Currently supported metric groups — extend as new axes (utility, privacy, …) are added
PRECOMPUTABLE_GROUPS = ("univariate", "bivariate", "multivariate", "missingness")


def save_precomputed(
    results: Dict[str, Dict],
    path: str | pathlib.Path,
    groups: tuple[str, ...] = PRECOMPUTABLE_GROUPS,
) -> None:
    """
    Serialise metric results for the given groups to a JSON file.

    Parameters
    ----------
    results:
        Mapping ``{synth_name: {group: {metric_key: MetricResult}}}`` covering
        any combination of metric groups (fidelity, missingness, utility, …).
    path:
        Destination file path (created or overwritten).
    groups:
        Which top-level groups to include; defaults to all known groups.
    """
    payload: Dict[str, Any] = {}

    for synth_name, result in results.items():
        payload[synth_name] = {}
        for group in groups:
            if group not in result:
                continue
            payload[synth_name][group] = {}
            for metric_key, metric_result in result[group].items():
                payload[synth_name][group][metric_key] = {
                    "metric_name": metric_result.metric_name,
                    "score": _to_serializable(metric_result.score),
                    "details": _to_serializable(metric_result.details),
                }

    path = pathlib.Path(path)
    path.write_text(json.dumps(payload, indent=2))


def load_precomputed(path: str | pathlib.Path) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
    """
    Load precomputed fidelity results from a JSON file.

    Returns
    -------
    dict
        ``{synth_name: {group: {metric_key: MetricResult}}}``
        ready to be merged into ``fidelity_results`` in the dashboard or CLI.
    """
    raw = json.loads(pathlib.Path(path).read_text())

    out: Dict[str, Dict[str, Dict[str, MetricResult]]] = {}
    for synth_name, groups in raw.items():
        out[synth_name] = {}
        for group, metrics in groups.items():
            out[synth_name][group] = {}
            for metric_key, entry in metrics.items():
                details = _from_serializable(entry.get("details", {}))
                score = entry.get("score")
                if score is None:
                    score = float("nan")
                out[synth_name][group][metric_key] = MetricResult(
                    metric_name=entry.get("metric_name", metric_key),
                    score=float(score),
                    details=details,
                )

    return out
