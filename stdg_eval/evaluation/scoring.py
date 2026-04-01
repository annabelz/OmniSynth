"""
Score aggregation functions.

All scoring functions take pre-computed MetricResult dicts (from evaluate_fidelity /
evaluate_missingness) and user-defined weight vectors, and return a float score
in [0, 1].

Weight conventions
------------------
Weights are specified as lists of non-negative floats. They are automatically
normalised to sum to 1.0, so users can pass e.g. [1, 2, 1] or [0.25, 0.5, 0.25]
interchangeably.

Fidelity weight vector:   [univariate_weight, bivariate_weight, multivariate_weight]
Missingness weight vector: [rate_weight, set_distribution_weight,
                            classifier_auroc_weight, dependency_structure_weight]
Composite weight vector:  [fidelity_weight, missingness_weight]
    TODO: extend to [fidelity, missingness, utility, privacy] once those axes
          are implemented.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from stdg_eval.config import (
    DEFAULT_COMPOSITE_WEIGHTS,
    DEFAULT_FIDELITY_WEIGHTS,
    DEFAULT_MISSINGNESS_WEIGHTS,
)
from stdg_eval.metrics.base import MetricResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_weights(weights: List[float]) -> List[float]:
    """Normalise a weight vector to sum to 1, raising on negative or all-zero."""
    arr = np.array(weights, dtype=float)
    if np.any(arr < 0):
        raise ValueError("All weights must be non-negative.")
    total = arr.sum()
    if total == 0:
        raise ValueError("Weight vector must not sum to zero.")
    return (arr / total).tolist()


def _weighted_average(scores: List[float], weights: List[float]) -> float:
    """Return the weighted average of *scores* using *weights* (already normalised)."""
    return float(np.dot(scores, weights))


def _group_score(group: Dict[str, MetricResult]) -> float:
    """Simple mean of all metric scores within a group."""
    vals = [r.score for r in group.values()]
    return float(np.mean(vals)) if vals else 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fidelity_score(
    fidelity_results: Dict[str, Dict[str, MetricResult]],
    weights: Optional[List[float]] = None,
    univariate_metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute a weighted fidelity score from :func:`evaluate_fidelity` results.

    Parameters
    ----------
    fidelity_results:
        Output of :func:`~stdg_eval.evaluation.fidelity.evaluate_fidelity`.
    weights:
        ``[univariate_weight, bivariate_weight, multivariate_weight]``.
        Defaults to ``DEFAULT_FIDELITY_WEIGHTS`` (equal weights).
        Only groups that are present in *fidelity_results* contribute.
    univariate_metric_weights:
        ``{"wasserstein": w, "tvd": w, "hellinger": w}`` weights applied within
        the univariate group. Defaults to equal weighting across present metrics.

    Returns
    -------
    dict with keys:
        ``"overall"``     — final weighted score ∈ [0, 1]
        ``"univariate"``  — mean score across univariate metrics
        ``"bivariate"``   — mean score across bivariate metrics
        ``"multivariate"``— mean score across multivariate metrics
        ``"weights_used"``— normalised weights actually applied
    """
    weights = weights or DEFAULT_FIDELITY_WEIGHTS

    group_names = ["univariate", "bivariate", "multivariate"]
    present = [g for g in group_names if g in fidelity_results]

    if not present:
        return {"overall": 1.0, "weights_used": [], "message": "No fidelity results provided."}

    # Only use weights for the groups that were actually computed
    full_weights = dict(zip(group_names, weights))
    active_weights = _normalise_weights([full_weights.get(g, 0.0) for g in present])

    def _univariate_score(group: Dict[str, MetricResult]) -> float:
        if not univariate_metric_weights:
            return _group_score(group)
        present_metrics = [k for k in group if k in univariate_metric_weights]
        if not present_metrics:
            return _group_score(group)
        w = _normalise_weights([univariate_metric_weights[k] for k in present_metrics])
        return float(np.dot([group[k].score for k in present_metrics], w))

    group_scores = {}
    for g in present:
        if g == "univariate":
            group_scores[g] = _univariate_score(fidelity_results[g])
        else:
            group_scores[g] = _group_score(fidelity_results[g])

    score_vec = [group_scores[g] for g in present]
    overall = _weighted_average(score_vec, active_weights)

    return {
        "overall": overall,
        **group_scores,
        "weights_used": dict(zip(present, active_weights)),
    }


def compute_missingness_score(
    missingness_results: Dict[str, MetricResult],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute a weighted missingness score from :func:`evaluate_missingness` results.

    Parameters
    ----------
    missingness_results:
        Output of :func:`~stdg_eval.evaluation.missingness.evaluate_missingness`.
    weights:
        ``[rate_weight, set_distribution_weight,
           classifier_auroc_weight, dependency_structure_weight]``.
        Defaults to ``DEFAULT_MISSINGNESS_WEIGHTS`` (equal weights).

    Returns
    -------
    dict with keys:
        ``"overall"``              — final weighted score ∈ [0, 1]
        ``"rate"``                 — missingness rate score
        ``"set_distribution"``     — pattern distribution score
        ``"classifier_auroc"``     — AUROC similarity score
        ``"dependency_structure"`` — dependency structure score
        ``"weights_used"``         — normalised weights actually applied
    """
    weights = weights or DEFAULT_MISSINGNESS_WEIGHTS

    metric_names = ["rate", "set_distribution", "classifier_auroc", "dependency_structure"]
    present = [m for m in metric_names if m in missingness_results]

    if not present:
        return {"overall": 1.0, "weights_used": [], "message": "No missingness results provided."}

    full_weights = dict(zip(metric_names, weights))
    active_weights = _normalise_weights([full_weights.get(m, 0.0) for m in present])

    metric_scores = {m: missingness_results[m].score for m in present}
    score_vec = [metric_scores[m] for m in present]

    overall = _weighted_average(score_vec, active_weights)

    return {
        "overall": overall,
        **metric_scores,
        "weights_used": dict(zip(present, active_weights)),
    }


def compute_composite_score(
    fidelity_score: float | Dict,
    missingness_score: float | Dict,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute a composite score combining fidelity and missingness axes.

    Parameters
    ----------
    fidelity_score:
        Either the ``"overall"`` float from :func:`compute_fidelity_score`, or
        the full dict returned by that function (the ``"overall"`` key is used).
    missingness_score:
        Either the ``"overall"`` float from :func:`compute_missingness_score`,
        or the full dict.
    weights:
        ``[fidelity_weight, missingness_weight]``.
        Defaults to ``DEFAULT_COMPOSITE_WEIGHTS`` (0.5 / 0.5).

        TODO: extend signature to ``[fidelity, missingness, utility, privacy]``
              once utility and privacy axes are implemented.

    Returns
    -------
    dict with keys:
        ``"composite"``         — weighted composite score ∈ [0, 1]
        ``"fidelity_score"``    — the fidelity component used
        ``"missingness_score"`` — the missingness component used
        ``"weights_used"``      — ``{"fidelity": w1, "missingness": w2}``
    """
    weights = weights or DEFAULT_COMPOSITE_WEIGHTS

    f_val = fidelity_score["overall"] if isinstance(fidelity_score, dict) else float(fidelity_score)
    m_val = missingness_score["overall"] if isinstance(missingness_score, dict) else float(missingness_score)

    w = _normalise_weights(weights)
    composite = float(w[0] * f_val + w[1] * m_val)

    return {
        "composite": composite,
        "fidelity_score": f_val,
        "missingness_score": m_val,
        "weights_used": {"fidelity": w[0], "missingness": w[1]},
        # TODO: add utility_score and privacy_score here once implemented
    }
