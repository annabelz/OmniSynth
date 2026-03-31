"""
Multivariate fidelity metrics.

CrossClassification  — discriminator AUROC (can real be told apart from synthetic?)
PropensityMSE        — propensity score MSE
"""
# TODO: revise the implementation 

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# TODO: should i be imputing for fidelity or should i do complete case analysis?

def _encode_for_sklearn(df: pd.DataFrame, col_types: ColumnTypes) -> pd.DataFrame:
    """
    One-hot encode categorical columns and impute missing values so the DataFrame
    can be passed to scikit-learn estimators.
    """
    out = df.copy()
    for col, ctype in col_types.items():
        if col not in out.columns:
            continue
        if ctype == "categorical":
            out[col] = out[col].astype(str).fillna("__missing__")
        else:
            median = out[col].median()
            out[col] = out[col].fillna(median)

    # One-hot encode all object/category columns
    cat_cols = [c for c in out.columns if col_types.get(c) == "categorical"]
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, drop_first=False)

    return out.astype(float)


# ---------------------------------------------------------------------------
# CrossClassification
# ---------------------------------------------------------------------------

class CrossClassification(BaseMetric):
    """
    Train a discriminator to distinguish real from synthetic samples.

    Approach
    --------
    1. Combine real (label=0) and synthetic (label=1) samples.
    2. Train a classifier (default: Random Forest) via k-fold cross-validation.
    3. Compute mean AUROC across folds.
    4. An AUROC of 0.5 means the classifier cannot distinguish real from synthetic
       (perfect fidelity); AUROC approaching 1.0 means easy discrimination.

    Score
    -----
        score = 1 − 2 * |AUROC − 0.5|

    so score = 1 when AUROC = 0.5 and score = 0 when AUROC = 1.0.
    """

    name = "Cross-Classification"
    description = (
        "Discriminator AUROC: a classifier is trained to distinguish real from "
        "synthetic samples. Score = 1 when AUROC = 0.5 (indistinguishable)."
    )
    axis = "fidelity"

    def __init__(
        self,
        model: str = "rf",
        n_estimators: int = 100,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_enc = _encode_for_sklearn(real, col_types)
        s_enc = _encode_for_sklearn(synthetic, col_types)

        # Align columns after one-hot expansion
        all_cols = sorted(set(r_enc.columns) | set(s_enc.columns))
        r_enc = r_enc.reindex(columns=all_cols, fill_value=0)
        s_enc = s_enc.reindex(columns=all_cols, fill_value=0)

        X = pd.concat([r_enc, s_enc], ignore_index=True)
        y = np.array([0] * len(r_enc) + [1] * len(s_enc))

        if self.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        auroc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        mean_auroc = float(np.mean(auroc_scores))
        std_auroc = float(np.std(auroc_scores))
        score = float(1.0 - 2.0 * abs(mean_auroc - 0.5))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, score),
            details={
                "mean_auroc": mean_auroc,
                "std_auroc": std_auroc,
                "fold_aurocs": auroc_scores.tolist(),
                "n_real": len(real),
                "n_synthetic": len(synthetic),
            },
        )


# ---------------------------------------------------------------------------
# PropensityMSE
# ---------------------------------------------------------------------------

class PropensityMSE(BaseMetric):
    """
    Propensity score Mean Squared Error (pMSE).

    Approach
    --------
    1. Combine real (label=0) and synthetic (label=1) samples.
    2. Fit a propensity model (logistic regression or RF) to predict P(synthetic).
    3. pMSE = mean((p_i − 0.5)²) across all N+M samples.
       Under perfect fidelity, P(synthetic) ≈ 0.5 everywhere → pMSE ≈ 0.

    Normalisation
    -------------
    Following Snoke et al. (2018), we normalise by the theoretical null pMSE:

        c = n_synthetic / (n_real + n_synthetic)
        pMSE_null = c × (1 − c)     # ≈ 0.25 when n_real = n_synthetic

    Propensity scores are computed via k-fold cross-validation to avoid overfitting
    on the training data (which would inflate pMSE for flexible models).

        score = max(0, 1 − pMSE / pMSE_null)

    References
    ----------
    Snoke et al. (2018) "General and specific utility measures for synthetic data."
    """

    name = "Propensity MSE"
    description = (
        "Propensity score MSE: measures how well a classifier can assign samples to "
        "'real' vs 'synthetic'. Lower pMSE → better fidelity → higher score."
    )
    axis = "fidelity"

    def __init__(
        self,
        model: str = "logistic",
        n_estimators: int = 100,
        max_iter: int = 1000,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_enc = _encode_for_sklearn(real, col_types)
        s_enc = _encode_for_sklearn(synthetic, col_types)

        all_cols = sorted(set(r_enc.columns) | set(s_enc.columns))
        r_enc = r_enc.reindex(columns=all_cols, fill_value=0)
        s_enc = s_enc.reindex(columns=all_cols, fill_value=0)

        X = pd.concat([r_enc, s_enc], ignore_index=True).values
        y = np.array([0] * len(r_enc) + [1] * len(s_enc))

        # Theoretical null pMSE (Snoke et al. 2018)
        c = len(s_enc) / (len(r_enc) + len(s_enc))
        pmse_null = float(c * (1.0 - c))

        # Compute propensity scores via k-fold cross-validation to avoid overfitting
        clf = self._build_model()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        propensities = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

        pmse = float(np.mean((propensities - 0.5) ** 2))
        score = float(1.0 - pmse / pmse_null) if pmse_null > 1e-10 else 1.0

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, score)),
            details={
                "pmse": pmse,
                "pmse_null_theoretical": pmse_null,
                "pmse_ratio": pmse / (pmse_null + 1e-10),
                "c_synthetic_fraction": c,
            },
        )

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