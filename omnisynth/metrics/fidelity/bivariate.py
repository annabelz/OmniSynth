"""
Bivariate fidelity metrics.

SpearmanCorrelation        — numerical × numerical pairwise rank correlations
ContingencyMatrix          — categorical × categorical (and mixed) associations
PairwiseCorrelationDifference — phik-based mixed-type correlation difference + t-test
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from omnisynth.metrics.base import BaseMetric, MetricResult
from omnisynth.utils.data_utils import ColumnTypes, get_numerical_columns, get_categorical_columns


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
    - numerical × categorical (mixed): bin the numerical column (Scott's rule on
      pooled real + synthetic values) and compute TVD
    - numerical × numerical: skipped (handled by SpearmanCorrelation)

    Score = 1 − mean(TVD across all qualifying column pairs).
    """

    name = "Contingency Matrix"
    description = (
        "Pairwise association comparison using normalised contingency table TVD for "
        "categorical and mixed (numerical × categorical) column pairs."
    )
    axis = "fidelity"

    def __init__(self, max_categories: int = 30) -> None:
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

    @staticmethod
    def _scott_bin_edges(real_col: pd.Series, synth_col: pd.Series) -> np.ndarray:
        """
        Bin edges for a numerical column using Scott's rule on the pooled
        (real + synthetic) non-missing values, spanning the combined range.
        """
        pooled = pd.concat([real_col, synth_col], ignore_index=True).dropna().values
        if len(pooled) < 2:
            return np.array([-np.inf, np.inf])
        std = float(np.std(pooled, ddof=1))
        if std == 0.0:
            return np.array([-np.inf, np.inf])
        h = 3.5 * std * (len(pooled) ** (-1.0 / 3.0))
        lo, hi = float(pooled.min()), float(pooled.max())
        n_bins = max(1, int(np.ceil((hi - lo) / h)))
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _mixed_tvd(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        num_col: str,
        cat_col: str,
    ) -> Optional[float]:
        """TVD after binning the numerical column using Scott's rule on pooled data."""
        r = real[[num_col, cat_col]].dropna()
        s = synthetic[[num_col, cat_col]].dropna()
        if len(r) == 0 or len(s) == 0:
            return None

        bin_edges = self._scott_bin_edges(r[num_col], s[num_col])

        r_binned = pd.cut(r[num_col], bins=bin_edges, include_lowest=True)
        s_binned = pd.cut(s[num_col], bins=bin_edges, include_lowest=True)

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
    Pairwise Correlation Difference (PCD) as per Hernandez et al. (2025).

    Uses the phi-k (phik) correlation coefficient introduced by Baak et al. to
    compute a unified correlation matrix for mixed-type data (numerical and
    categorical).  phi-k:

    - Works consistently across categorical and numerical variable pairs.
    - Captures non-linear relationships.
    - Reduces to the Pearson correlation coefficient for bivariate normal inputs.

    PCD is computed as the mean absolute difference between the upper-triangular
    (off-diagonal) entries of the real and synthetic phik correlation matrices::

        PCD = (1/n) * sum_i |Corr(X_real)_i - Corr(X_synth)_i|

    A low PCD (close to 0) indicates that the synthetic data preserves pairwise
    correlations well; a high PCD (close to 1) indicates strong disagreement.

    The metric is complemented by a one-sample Student's t-test (alpha=0.05)
    on the vector of absolute differences to determine whether the deviation
    from zero is statistically significant.

    Score = 1 - PCD  (higher is better).
    """

    name = "Pairwise Correlation Difference"
    description = (
        "Mean absolute difference between real and synthetic phik correlation "
        "matrices (mixed-type: numerical and categorical columns). "
        "Complemented by Student's t-test (alpha=0.05)."
    )
    axis = "fidelity"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        import phik as phik_lib
        from scipy.stats import ttest_1samp

        num_cols = get_numerical_columns(col_types)
        all_cols = list(col_types.keys())

        if len(all_cols) < 2:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "Fewer than 2 columns - no pairs to compare."},
            )

        # Build shared bin edges for numerical columns using Scott's rule on the
        # pooled (real + synthetic) values, so both datasets are binned on the
        # same grid and the resulting phi-k values are directly comparable.
        bins: dict = {}
        for col in num_cols:
            pooled = pd.concat([real[col], synthetic[col]], ignore_index=True).dropna().values
            if len(pooled) < 2:
                continue
            std = float(np.std(pooled, ddof=1))
            if std == 0.0:
                continue
            h = 3.49 * std * (len(pooled) ** (-1.0 / 3.0))
            lo, hi = float(pooled.min()), float(pooled.max())
            n_bins = max(1, int(np.ceil((hi - lo) / h)))
            bins[col] = np.linspace(lo, hi, n_bins + 1)

        # Compute phik correlation matrices (values in [0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real_phik = phik_lib.phik_matrix(
                real[all_cols], interval_cols=num_cols, bins=bins, verbose=False, njobs=1
            )
            synth_phik = phik_lib.phik_matrix(
                synthetic[all_cols], interval_cols=num_cols, bins=bins, verbose=False, njobs=1
            )

        # Reindex both to the same column order
        real_phik = real_phik.reindex(index=all_cols, columns=all_cols)
        synth_phik = synth_phik.reindex(index=all_cols, columns=all_cols)

        n = len(all_cols)
        idx = np.triu_indices(n, k=1)
        real_upper = real_phik.values[idx]
        synth_upper = synth_phik.values[idx]

        # Filter out NaN pairs
        valid_mask = ~(np.isnan(real_upper) | np.isnan(synth_upper))
        real_valid = real_upper[valid_mask]
        synth_valid = synth_upper[valid_mask]
        diffs_valid = np.abs(real_valid - synth_valid)

        if len(diffs_valid) == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "No valid column pairs found."},
            )

        pcd = float(np.mean(diffs_valid))

        # Student's t-test: H0 = mean difference is 0
        if len(diffs_valid) >= 2:
            t_stat, p_value = ttest_1samp(diffs_valid, 0.0)
            t_stat = float(t_stat)
            p_value = float(p_value)
            significant = p_value < 0.05
        else:
            t_stat, p_value, significant = float("nan"), float("nan"), None

        # Build labelled per-pair dicts
        pair_real: Dict[str, float] = {}
        pair_synth: Dict[str, float] = {}
        pair_diffs: Dict[str, float] = {}
        valid_indices = np.where(valid_mask)[0]
        for vi, k in enumerate(valid_indices):
            i, j = idx[0][k], idx[1][k]
            key = f"{all_cols[i]}|{all_cols[j]}"
            pair_real[key] = float(real_valid[vi])
            pair_synth[key] = float(synth_valid[vi])
            pair_diffs[key] = float(diffs_valid[vi])

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, 1.0 - pcd),
            details={
                "pcd": pcd,
                "pair_real": pair_real,
                "pair_synth": pair_synth,
                "pair_differences": pair_diffs,
                "mean_absolute_difference": pcd,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_difference": significant,
            },
        )
