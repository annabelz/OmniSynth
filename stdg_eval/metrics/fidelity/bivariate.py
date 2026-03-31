"""
Bivariate fidelity metrics.

SpearmanCorrelation        — numerical × numerical pairwise rank correlations
ContingencyMatrix          — categorical × categorical (and mixed) associations
PairwiseCorrelationDifference — TODO
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes, get_numerical_columns, get_categorical_columns


class SpearmanCorrelation(BaseMetric):
    """
    Compare pairwise Spearman rank-correlation matrices of real vs synthetic data.

    Score
    -----
    For each column pair (i, j) in the upper triangle of the correlation matrix:

        diff_ij = |ρ_real_ij − ρ_synth_ij|

    Overall score = 1 − mean(diff_ij) across all pairs.

    ``details`` exposes both full correlation matrices and the per-pair differences.
    """

    name = "Spearman Correlation"
    description = (
        "Mean absolute difference between real and synthetic pairwise Spearman "
        "rank-correlation matrices (numerical columns only)."
    )
    axis = "fidelity"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        num_cols = get_numerical_columns(col_types)

        if len(num_cols) < 2:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "Fewer than 2 numerical columns — no pairs to compare."},
            )

        r = real[num_cols].dropna()
        s = synthetic[num_cols].dropna()

        real_corr = r.corr(method="spearman").values
        synth_corr = s.corr(method="spearman").values

        # Upper-triangle indices (excluding diagonal)
        idx = np.triu_indices(len(num_cols), k=1)
        real_upper = real_corr[idx]
        synth_upper = synth_corr[idx]
        diffs = np.abs(real_upper - synth_upper)

        # Build a labelled dict of pair differences
        pair_diffs: Dict[str, float] = {}
        for k, (i, j) in enumerate(zip(idx[0], idx[1])):
            key = f"{num_cols[i]}|{num_cols[j]}"
            pair_diffs[key] = float(diffs[k])

        overall_score = float(1.0 - np.mean(diffs))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, overall_score),
            details={
                "real_correlation_matrix": pd.DataFrame(
                    real_corr, index=num_cols, columns=num_cols
                ).to_dict(),
                "synth_correlation_matrix": pd.DataFrame(
                    synth_corr, index=num_cols, columns=num_cols
                ).to_dict(),
                "pair_differences": pair_diffs,
                "mean_absolute_difference": float(np.mean(diffs)),
                "columns": num_cols,
            },
        )


class ContingencyMatrix(BaseMetric):
    """
    Association comparison via normalised contingency tables.

    Handles three pair types:
    - categorical × categorical: normalised contingency TVD
    - numerical × categorical (mixed): bin the numerical column and compute TVD
    - numerical × numerical: skipped (handled by SpearmanCorrelation)

    Score = 1 − mean(TVD across all qualifying column pairs).
    """

    name = "Contingency Matrix"
    description = (
        "Pairwise association comparison using normalised contingency table TVD for "
        "categorical and mixed (numerical × categorical) column pairs."
    )
    axis = "fidelity"

    def __init__(self, n_bins: int = 10, max_categories: int = 30) -> None:
        self.n_bins = n_bins
        self.max_categories = max_categories

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        cat_cols = get_categorical_columns(col_types)
        num_cols = get_numerical_columns(col_types)

        pair_tvds: Dict[str, float] = {}

        # categorical × categorical pairs
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1 :]:
                tvd = self._contingency_tvd(real, synthetic, c1, c2)
                if tvd is not None:
                    pair_tvds[f"{c1}|{c2}"] = tvd

        # numerical × categorical (mixed) pairs
        for num_col in num_cols:
            for cat_col in cat_cols:
                tvd = self._mixed_tvd(real, synthetic, num_col, cat_col)
                if tvd is not None:
                    pair_tvds[f"{num_col}|{cat_col}"] = tvd

        if not pair_tvds:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "No qualifying column pairs found."},
            )

        overall_score = float(1.0 - np.mean(list(pair_tvds.values())))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, overall_score),
            details={"pair_tvds": pair_tvds},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _contingency_tvd(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> Optional[float]:
        """TVD between normalised joint frequency tables for two categorical cols."""
        # Skip high-cardinality columns
        n_unique = max(real[col1].nunique(), real[col2].nunique())
        if n_unique > self.max_categories:
            return None

        r = real[[col1, col2]].dropna()
        s = synthetic[[col1, col2]].dropna()
        if len(r) == 0 or len(s) == 0:
            return None

        all_vals1 = sorted(set(r[col1].unique()) | set(s[col1].unique()))
        all_vals2 = sorted(set(r[col2].unique()) | set(s[col2].unique()))

        r_ct = pd.crosstab(r[col1], r[col2], normalize=True)
        s_ct = pd.crosstab(s[col1], s[col2], normalize=True)

        r_ct = r_ct.reindex(index=all_vals1, columns=all_vals2, fill_value=0.0)
        s_ct = s_ct.reindex(index=all_vals1, columns=all_vals2, fill_value=0.0)

        tvd = 0.5 * float(np.sum(np.abs(r_ct.values - s_ct.values)))
        return tvd

    def _mixed_tvd(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        num_col: str,
        cat_col: str,
    ) -> Optional[float]:
        """TVD after binning the numerical column into ``n_bins`` equal-frequency bins."""
        r = real[[num_col, cat_col]].dropna()
        s = synthetic[[num_col, cat_col]].dropna()
        if len(r) == 0 or len(s) == 0:
            return None

        # Derive bin edges from real data
        _, bin_edges = pd.qcut(r[num_col], q=self.n_bins, retbins=True, duplicates="drop")
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        r_binned = pd.cut(r[num_col], bins=bin_edges)
        s_binned = pd.cut(s[num_col], bins=bin_edges)

        all_bins = sorted(set(r_binned.unique()) | set(s_binned.unique()), key=str)
        all_cats = sorted(set(r[cat_col].unique()) | set(s[cat_col].unique()), key=str)

        r_tmp = r.copy()
        s_tmp = s.copy()
        r_tmp[num_col] = r_binned
        s_tmp[num_col] = s_binned

        r_ct = pd.crosstab(r_tmp[num_col], r_tmp[cat_col], normalize=True)
        s_ct = pd.crosstab(s_tmp[num_col], s_tmp[cat_col], normalize=True)

        r_ct = r_ct.reindex(index=all_bins, columns=all_cats, fill_value=0.0)
        s_ct = s_ct.reindex(index=all_bins, columns=all_cats, fill_value=0.0)

        tvd = 0.5 * float(np.sum(np.abs(r_ct.values - s_ct.values)))
        return tvd


class PairwiseCorrelationDifference(BaseMetric):
    """
    TODO: Implement pairwise correlation difference metric.

    Planned approach:
    - Compute full pairwise correlation matrices for real and synthetic data
      using a mixed-type correlation measure (e.g., Spearman for numerical,
      Cramér's V for categorical, point-biserial for mixed pairs).
    - Score = 1 − mean absolute difference across all pairs.

    This will supersede the separate SpearmanCorrelation and ContingencyMatrix
    metrics once a unified mixed-type correlation measure is implemented.
    """

    name = "Pairwise Correlation Difference"
    description = "TODO: Mixed-type pairwise correlation difference (not yet implemented)."
    axis = "fidelity"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        raise NotImplementedError(
            "PairwiseCorrelationDifference is not yet implemented. "
            "See class docstring for the planned approach."
        )
