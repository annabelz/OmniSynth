"""
Top-level missingness evaluation function.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from stdg_eval.config import DEFAULT_CONFIG, EvalConfig
from stdg_eval.metrics.base import MetricResult
from stdg_eval.metrics.missingness.measures import (
    MissingnessClassifierAUROC,
    MissingnessDependencyStructure,
    MissingnessRate,
    MissingnessSetDistribution,
)
from stdg_eval.utils.data_utils import ColumnTypes, align_columns

MissingnessResults = Dict[str, MetricResult]


def evaluate_missingness(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col_types: Optional[ColumnTypes] = None,
    config: Optional[EvalConfig] = None,
    run_rate: bool = True,
    run_set_distribution: bool = True,
    run_classifier_auroc: bool = True,
    run_dependency_structure: bool = True,
) -> MissingnessResults:
    """
    Evaluate all missingness similarity metrics comparing *real* to *synthetic*.

    Parameters
    ----------
    real:
        Ground-truth dataset (should retain original missing values — do *not*
        impute before passing in).
    synthetic:
        Synthetic dataset to evaluate.
    col_types:
        Optional column type mapping. Inferred from *real* if not provided.
    config:
        Optional :class:`~stdg_eval.config.EvalConfig`. Uses defaults otherwise.
    run_rate, run_set_distribution, run_classifier_auroc, run_dependency_structure:
        Flags to selectively disable individual metrics.

    Returns
    -------
    MissingnessResults
        Flat dict: ``{"rate": MetricResult, "set_distribution": MetricResult,
                      "classifier_auroc": MetricResult, "dependency_structure": MetricResult}``
    """
    cfg = config or DEFAULT_CONFIG
    mc = cfg.missingness

    real, synthetic, col_types = align_columns(real, synthetic, col_types)

    results: MissingnessResults = {}

    if run_rate and mc.run_rate:
        results["rate"] = MissingnessRate().evaluate(real, synthetic, col_types)

    if run_set_distribution and mc.run_set_distribution:
        results["set_distribution"] = MissingnessSetDistribution().evaluate(
            real, synthetic, col_types
        )

    if run_classifier_auroc and mc.run_classifier_auroc:
        results["classifier_auroc"] = MissingnessClassifierAUROC(
            model=mc.classifier_model,
            max_iter=mc.classifier_max_iter,
            n_estimators=mc.classifier_n_estimators,
            min_missing_rate=mc.min_missing_rate,
            random_state=cfg.random_state,
        ).evaluate(real, synthetic, col_types)

    if run_dependency_structure and mc.run_dependency_structure:
        results["dependency_structure"] = MissingnessDependencyStructure(
            method=mc.dependency_method
        ).evaluate(real, synthetic, col_types)

    return results
