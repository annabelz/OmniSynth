"""
Missingness similarity metrics.

MissingnessRate               — per-variable missing-rate comparison
MissingnessSetDistribution    — distribution over joint missingness patterns
MissingnessClassifierAUROC    — how well missingness can be predicted from other columns
MissingnessDependencyStructure — correlation of missingness indicators
"""

# TODO: double check implementation

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes


# ---------------------------------------------------------------------------
# Helper: build missingness indicator matrix
# ---------------------------------------------------------------------------

def _missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a binary DataFrame: 1 where value is missing, 0 otherwise."""
    return df.isnull().astype(int)


# ---------------------------------------------------------------------------
# MissingnessRate
# ---------------------------------------------------------------------------

class MissingnessRate(BaseMetric):
    """
    Per-variable missingness rate comparison.

    For each column, the missingness distribution is binary (missing vs. present),
    so TVD per column = |rate_real - rate_synth|.  The overall summary score is
    the TVD between the two missingness-rate bar charts:

        TVD = 0.5 * Σ |rate_real_i − rate_synth_i|

    Score = 1 − TVD, clipped to [0, 1].
    """

    name = "Missingness Rate"
    description = "TVD between real and synthetic per-column missingness rate distributions."
    axis = "missingness"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        real_rates: Dict[str, float] = {}
        synth_rates: Dict[str, float] = {}
        column_scores: Dict[str, float] = {}

        for col in real.columns:
            r_rate = float(real[col].isnull().mean())
            s_rate = float(synthetic[col].isnull().mean()) if col in synthetic.columns else 0.0

            real_rates[col] = r_rate
            synth_rates[col] = s_rate
            column_scores[col] = abs(r_rate - s_rate)

        tvd = 0.5 * float(np.sum(list(column_scores.values()))) if column_scores else 0.0

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, 1.0 - tvd),
            details={
                "real_rates": real_rates,
                "synth_rates": synth_rates,
                "tvd": tvd,
            },
            column_scores=column_scores,
        )


# ---------------------------------------------------------------------------
# MissingnessSetDistribution
# ---------------------------------------------------------------------------

class MissingnessSetDistribution(BaseMetric):
    """
    Distribution over joint missingness patterns (the "missingness set").

    Each row in a dataset can be characterised by its binary missingness pattern
    (a tuple indicating which columns are missing). This metric compares the
    distribution over patterns between real and synthetic data using TVD.

    Score = 1 − TVD(pattern_dist_real, pattern_dist_synth).

    Note: With many columns the number of possible patterns can be exponential.
    In practice, medical datasets tend to have a small number of common patterns.
    """

    name = "Missingness Set Distribution"
    description = (
        "TVD between real and synthetic distributions over joint missingness patterns "
        "(which combinations of columns are simultaneously missing)."
    )
    axis = "missingness"

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_ind = _missingness_indicators(real)
        s_ind = _missingness_indicators(synthetic)

        # Represent each row as a tuple (the pattern)
        r_patterns = r_ind.apply(tuple, axis=1).value_counts(normalize=True)
        s_patterns = s_ind.apply(tuple, axis=1).value_counts(normalize=True)

        all_patterns = set(r_patterns.index) | set(s_patterns.index)

        p = np.array([r_patterns.get(pat, 0.0) for pat in all_patterns])
        q = np.array([s_patterns.get(pat, 0.0) for pat in all_patterns])
        tvd = 0.5 * float(np.sum(np.abs(p - q)))

        # Top patterns for display
        top_r = r_patterns.head(10).to_dict()
        top_s = s_patterns.head(10).to_dict()

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, 1.0 - tvd),
            details={
                "tvd": tvd,
                "n_unique_real_patterns": len(r_patterns),
                "n_unique_synth_patterns": len(s_patterns),
                "top_real_patterns": {str(k): v for k, v in top_r.items()},
                "top_synth_patterns": {str(k): v for k, v in top_s.items()},
            },
        )


# ---------------------------------------------------------------------------
# MissingnessClassifierAUROC
# ---------------------------------------------------------------------------

class MissingnessClassifierAUROC(BaseMetric):
    """
    Missingness predictability via AUROC.

    For each column c with sufficient missingness, fit a classifier to predict
    P(column c is missing) from the observed values of all other columns.
    Compare the AUROC on real vs synthetic.

    Score = 1 − mean(|AUROC_real_c − AUROC_synth_c|) across qualifying columns.

    A large discrepancy means the synthetic data's missingness is driven by
    different predictors than in the real data (i.e., MAR mechanism differs).
    """

    name = "Missingness Classifier AUROC"
    description = (
        "Per-column AUROC for predicting missingness from other columns. "
        "Compares whether the same columns predict missingness in real vs synthetic data."
    )
    axis = "missingness"

    def __init__(
        self,
        model: str = "logistic",
        max_iter: int = 500,
        n_estimators: int = 100,
        min_missing_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.min_missing_rate = min_missing_rate
        self.random_state = random_state

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        auroc_real: Dict[str, float] = {}
        auroc_synth: Dict[str, float] = {}
        column_scores: Dict[str, float] = {}

        for target_col in real.columns:
            r_rate = real[target_col].isnull().mean()
            s_rate = synthetic[target_col].isnull().mean() if target_col in synthetic.columns else 0.0

            # Skip columns with insufficient missingness to train a classifier
            if r_rate < self.min_missing_rate or s_rate < self.min_missing_rate:
                continue
            if r_rate > 0.99 or s_rate > 0.99:
                continue  # column is almost always missing → no signal

            auc_r = self._fit_predict(real, target_col, col_types)
            auc_s = self._fit_predict(synthetic, target_col, col_types)

            if auc_r is None or auc_s is None:
                continue

            auroc_real[target_col] = auc_r
            auroc_synth[target_col] = auc_s
            column_scores[target_col] = 1.0 - abs(auc_r - auc_s)

        if not column_scores:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "No columns had sufficient missingness for classification."},
            )

        overall_score = float(np.mean(list(column_scores.values())))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, overall_score),
            details={
                "auroc_real": auroc_real,
                "auroc_synth": auroc_synth,
            },
            column_scores=column_scores,
        )

    def _fit_predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        col_types: ColumnTypes,
    ) -> Optional[float]:
        """Fit missingness classifier for *target_col* on *df*, return AUROC."""
        y = df[target_col].isnull().astype(int)
        X_df = df.drop(columns=[target_col])

        # Simple imputation + encoding
        X_parts = []
        for col in X_df.columns:
            series = X_df[col]
            if col_types.get(col) == "categorical":
                dummies = pd.get_dummies(series.astype(str).fillna("__missing__"), prefix=col)
                X_parts.append(dummies)
            else:
                filled = series.fillna(series.median())
                X_parts.append(filled.rename(col))

        if not X_parts:
            return None

        X = pd.concat(X_parts, axis=1).astype(float)

        # Need at least a few positive and negative samples
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            return None

        try:
            clf = self._build_model()
            clf.fit(X, y)
            prob = clf.predict_proba(X)[:, 1]
            return float(roc_auc_score(y, prob))
        except Exception:
            return None

    def _build_model(self):
        if self.model == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)),
        ])


# ---------------------------------------------------------------------------
# MissingnessDependencyStructure
# ---------------------------------------------------------------------------

class MissingnessDependencyStructure(BaseMetric):
    """
    Compare the correlation structure of missingness indicators.

    For each column, create a binary indicator: 1 if missing, 0 if observed.
    Compute the pairwise correlation matrix of these indicators for real and
    synthetic data. Score = 1 − mean(|corr_real − corr_synth|).

    This captures whether the *dependency* structure of missingness is preserved
    (e.g., if column A and column B tend to be missing together in the real data,
    is the same true in the synthetic data?).

    Columns that are never missing in either dataset are excluded.
    """

    name = "Missingness Dependency Structure"
    description = (
        "Pairwise correlation of missingness indicators — measures whether the same "
        "columns tend to be missing together in real vs synthetic data."
    )
    axis = "missingness"

    def __init__(self, method: str = "pearson") -> None:
        self.method = method

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_ind = _missingness_indicators(real)
        s_ind = _missingness_indicators(synthetic)

        # Keep only columns that have at least some missingness in either dataset
        has_missing = [
            col for col in real.columns
            if r_ind[col].sum() > 0 or (col in synthetic.columns and s_ind[col].sum() > 0)
        ]

        if len(has_missing) < 2:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"message": "Fewer than 2 columns with missingness — no pairs to compare."},
            )

        r_corr = r_ind[has_missing].corr(method=self.method).fillna(0).values
        s_corr = s_ind[has_missing].corr(method=self.method).fillna(0).values

        idx = np.triu_indices(len(has_missing), k=1)
        diffs = np.abs(r_corr[idx] - s_corr[idx])
        mean_diff = float(np.mean(diffs))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, 1.0 - mean_diff),
            details={
                "mean_absolute_difference": mean_diff,
                "real_correlation_matrix": pd.DataFrame(
                    r_corr, index=has_missing, columns=has_missing
                ).to_dict(),
                "synth_correlation_matrix": pd.DataFrame(
                    s_corr, index=has_missing, columns=has_missing
                ).to_dict(),
                "columns": has_missing,
            },
        )