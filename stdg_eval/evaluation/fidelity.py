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
from stdg_eval.metrics.fidelity.univariate import WassersteinDistance, TotalVariationDistance, HellingerDistance
from stdg_eval.metrics.fidelity.bivariate import SpearmanCorrelation, ContingencyMatrix, PairwiseCorrelationDifference
from stdg_eval.metrics.fidelity.multivariate import AucRoc, PropensityMSE, CrossClassificationRS, CrossClassificationSR
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
    verbose: bool = False,
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
           "multivariate": {"auc_roc": MetricResult, "propensity_mse": MetricResult}}``
    """
    cfg = config or DEFAULT_CONFIG
    fc = cfg.fidelity

    real, synthetic, col_types = align_columns(real, synthetic, col_types)

    results: FidelityResults = {}

    def _log(label: str) -> None:
        if verbose:
            print(f"  [fidelity] {label}", flush=True)

    # ------------------------------------------------------------------
    # Univariate
    # ------------------------------------------------------------------
    if run_univariate:
        results["univariate"] = {}

        if fc.run_wasserstein:
            _log("Wasserstein Distance")
            results["univariate"]["wasserstein"] = WassersteinDistance().evaluate(real, synthetic, col_types)
        if fc.run_tvd:
            _log("Total Variation Distance")
            results["univariate"]["tvd"] = TotalVariationDistance().evaluate(real, synthetic, col_types)
        if fc.run_hellinger:
            _log("Hellinger Distance")
            results["univariate"]["hellinger"] = HellingerDistance().evaluate(real, synthetic, col_types)

    # ------------------------------------------------------------------
    # Bivariate
    # ------------------------------------------------------------------
    if run_bivariate:
        results["bivariate"] = {}

        if fc.run_spearman:
            _log("Spearman Correlation")
            results["bivariate"]["spearman"] = SpearmanCorrelation().evaluate(real, synthetic, col_types)
        if fc.run_contingency:
            _log("Contingency Matrix")
            results["bivariate"]["contingency"] = ContingencyMatrix(
                max_categories=fc.contingency_max_categories
            ).evaluate(real, synthetic, col_types)
        if fc.run_pcd:
            _log("Pairwise Correlation Difference (phik)")
            results["bivariate"]["pcd"] = PairwiseCorrelationDifference().evaluate(real, synthetic, col_types)

    # ------------------------------------------------------------------
    # Multivariate
    # ------------------------------------------------------------------
    if run_multivariate:
        results["multivariate"] = {}

        if fc.run_auc_roc:
            _log("Cross-Classification")
            results["multivariate"]["auc_roc"] = AucRoc(
                model=fc.propensity_mse_model,
                n_estimators=fc.auc_roc_n_estimators,
                cv_folds=fc.auc_roc_cv_folds,
                random_state=cfg.random_state,
                impute=fc.auc_roc_impute,
            ).evaluate(real, synthetic, col_types)
        if fc.run_propensity_mse:
            _log("Propensity MSE")
            results["multivariate"]["propensity_mse"] = PropensityMSE(
                model=fc.propensity_mse_model,
                n_estimators=fc.propensity_mse_n_estimators,
                max_iter=fc.propensity_mse_max_iter,
                random_state=cfg.random_state,
            ).evaluate(real, synthetic, col_types)
        # if fc.run_crcl_rs:
        #     _log("CrCl-RS (train real, test synth)")
        #     results["multivariate"]["crcl_rs"] = CrossClassificationRS(
        #         test_size=fc.crcl_test_size,
        #         max_depth=fc.crcl_max_depth,
        #         random_state=cfg.random_state,
        #         impute=fc.crcl_impute,
        #     ).evaluate(real, synthetic, col_types)
        # if fc.run_crcl_sr:
        #     _log("CrCl-SR (train synth, test real)")
        #     results["multivariate"]["crcl_sr"] = CrossClassificationSR(
        #         test_size=fc.crcl_test_size,
        #         max_depth=fc.crcl_max_depth,
        #         random_state=cfg.random_state,
        #         impute=fc.crcl_impute,
        #     ).evaluate(real, synthetic, col_types)

    return results