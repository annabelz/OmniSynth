"""
Top-level fidelity evaluation function.

evaluate_fidelity(real, synthetic) runs all enabled fidelity metrics and
returns a structured dict of MetricResult objects keyed by group and metric name.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from stdg_eval.config import DEFAULT_CONFIG, EvalConfig
from stdg_eval.metrics.base import MetricResult
from stdg_eval.metrics.fidelity.univariate import WassersteinDistance, TotalVariationDistance
from stdg_eval.metrics.fidelity.bivariate import SpearmanCorrelation, ContingencyMatrix
from stdg_eval.metrics.fidelity.multivariate import CrossClassification, PropensityMSE
from stdg_eval.utils.data_utils import ColumnTypes, align_columns, detect_column_types


# Type alias for the return value
FidelityResults = Dict[str, Dict[str, MetricResult]]
# Structure: {"univariate": {"wasserstein": MetricResult, "tvd": MetricResult},
#              "bivariate": {...}, "multivariate": {...}}


def evaluate_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col_types: Optional[ColumnTypes] = None,
    config: Optional[EvalConfig] = None,
    run_univariate: bool = True,
    run_bivariate: bool = True,
    run_multivariate: bool = True,
) -> FidelityResults:
    """
    Evaluate all fidelity metrics comparing *real* to *synthetic*.

    Parameters
    ----------
    real:
        Ground-truth dataset.
    synthetic:
        Synthetic dataset to evaluate.
    col_types:
        Optional mapping ``{column_name: "numerical"|"categorical"}``.
        Inferred from *real* if not provided.
    config:
        Optional :class:`~stdg_eval.config.EvalConfig` with metric-level knobs.
        Uses library defaults if not provided.
    run_univariate, run_bivariate, run_multivariate:
        Flags to selectively disable metric groups.

    Returns
    -------
    FidelityResults
        Nested dict:
        ``{"univariate": {"wasserstein": MetricResult, "tvd": MetricResult},
           "bivariate":   {"spearman": MetricResult, "contingency": MetricResult},
           "multivariate": {"cross_classification": MetricResult, "propensity_mse": MetricResult}}``
    """
    cfg = config or DEFAULT_CONFIG
    fc = cfg.fidelity

    real, synthetic, col_types = align_columns(real, synthetic, col_types)

    results: FidelityResults = {}

    # ------------------------------------------------------------------
    # Univariate
    # ------------------------------------------------------------------
    if run_univariate:
        results["univariate"] = {}

        wd_metric = WassersteinDistance()
        results["univariate"]["wasserstein"] = wd_metric.evaluate(real, synthetic, col_types)

        tvd_metric = TotalVariationDistance()
        results["univariate"]["tvd"] = tvd_metric.evaluate(real, synthetic, col_types)

    # ------------------------------------------------------------------
    # Bivariate
    # ------------------------------------------------------------------
    if run_bivariate:
        results["bivariate"] = {}

        spearman = SpearmanCorrelation()
        results["bivariate"]["spearman"] = spearman.evaluate(real, synthetic, col_types)

        contingency = ContingencyMatrix(
            max_categories=fc.contingency_max_categories
        )
        results["bivariate"]["contingency"] = contingency.evaluate(real, synthetic, col_types)

    # ------------------------------------------------------------------
    # Multivariate
    # ------------------------------------------------------------------
    if run_multivariate:
        results["multivariate"] = {}

        cc = CrossClassification(
            model=fc.propensity_mse_model,  # reuse model choice
            n_estimators=fc.cross_classification_n_estimators,
            cv_folds=fc.cross_classification_cv_folds,
            random_state=cfg.random_state,
        )
        results["multivariate"]["cross_classification"] = cc.evaluate(real, synthetic, col_types)

        pmse = PropensityMSE(
            model=fc.propensity_mse_model,
            n_estimators=fc.propensity_mse_n_estimators,
            max_iter=fc.propensity_mse_max_iter,
            random_state=cfg.random_state,
        )
        results["multivariate"]["propensity_mse"] = pmse.evaluate(real, synthetic, col_types)

    return results