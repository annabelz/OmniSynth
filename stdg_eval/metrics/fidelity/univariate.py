"""
Univariate fidelity metrics.

WassersteinDistance  — numerical columns  (1st Wasserstein / Earth Mover's Distance)
TotalVariationDistance — categorical columns
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes, get_numerical_columns, get_categorical_columns


class WassersteinDistance(BaseMetric):
    """
    1st Wasserstein distance (Earth Mover's Distance) for numerical columns.

    Normalisation
    -------------
    Raw WD is in [0, ∞) and scale-dependent. We normalise per column by the
    inter-quartile range (IQR) of the real column (with a small ε guard):

        normalised_wd = wd / (IQR_real + ε)

    The per-column score is then:

        score_col = exp(−normalised_wd)

    so that score = 1 when distributions are identical and approaches 0 for
    very large divergences.  The overall score is the mean across all numerical
    columns, weighted equally.
    """

    name = "Wasserstein Distance"
    description = (
        "1st Wasserstein distance between real and synthetic distributions for each "
        "numerical column, normalised by the real column's IQR."
    )
    axis = "fidelity"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        num_cols = get_numerical_columns(col_types)
        if not num_cols:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "No numerical columns found."},
            )

        raw_distances: Dict[str, float] = {}
        normalised_distances: Dict[str, float] = {}
        column_scores: Dict[str, float] = {}

        for col in num_cols:
            r = real[col].dropna().values.astype(float)
            s = synthetic[col].dropna().values.astype(float)

            if len(r) == 0 or len(s) == 0:
                warnings.warn(f"WassersteinDistance: column '{col}' has no non-null values — skipped.")
                continue

            wd = wasserstein_distance(r, s)
            iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
            norm_wd = wd / (iqr + 1e-8)

            raw_distances[col] = float(wd)
            normalised_distances[col] = float(norm_wd)
            column_scores[col] = float(np.exp(-norm_wd))

        if not column_scores:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "All numerical columns were empty."},
            )

        overall_score = float(np.mean(list(column_scores.values())))

        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            details={
                "raw_distances": raw_distances,
                "normalised_distances": normalised_distances,
            },
            column_scores=column_scores,
        )


class TotalVariationDistance(BaseMetric):
    """
    Total Variation Distance (TVD) for categorical columns.

    TVD = 0.5 * Σ |P(x) − Q(x)|  ∈ [0, 1]

    Per-column score = 1 − TVD, so 1 = identical distributions, 0 = disjoint.
    The overall score is the mean across all categorical columns.
    """

    name = "Total Variation Distance"
    description = (
        "Total Variation Distance between real and synthetic category frequency "
        "distributions for each categorical column."
    )
    axis = "fidelity"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        cat_cols = get_categorical_columns(col_types)
        if not cat_cols:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "No categorical columns found."},
            )

        tvd_values: Dict[str, float] = {}
        column_scores: Dict[str, float] = {}
        real_freqs: Dict[str, Dict] = {}
        synth_freqs: Dict[str, Dict] = {}

        for col in cat_cols:
            r = real[col].dropna()
            s = synthetic[col].dropna()

            if len(r) == 0 or len(s) == 0:
                warnings.warn(f"TotalVariationDistance: column '{col}' has no non-null values — skipped.")
                continue

            all_categories = set(r.unique()) | set(s.unique())

            r_freq = r.value_counts(normalize=True)
            s_freq = s.value_counts(normalize=True)

            p = np.array([r_freq.get(cat, 0.0) for cat in all_categories])
            q = np.array([s_freq.get(cat, 0.0) for cat in all_categories])

            tvd = 0.5 * float(np.sum(np.abs(p - q)))

            tvd_values[col] = tvd
            column_scores[col] = 1.0 - tvd
            real_freqs[col] = r_freq.to_dict()
            synth_freqs[col] = s_freq.to_dict()

        if not column_scores:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "All categorical columns were empty."},
            )

        overall_score = float(np.mean(list(column_scores.values())))

        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            details={
                "tvd_values": tvd_values,
                "real_frequencies": real_freqs,
                "synth_frequencies": synth_freqs,
            },
            column_scores=column_scores,
        )
