"""
Multivariate fidelity metrics.

AucRoc  — discriminator AUROC (can real be told apart from synthetic?)
PropensityMSE        — propensity score MSE
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from omnisynth.metrics.base import BaseMetric, MetricResult
from omnisynth.utils.data_utils import ColumnTypes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _encode_for_sklearn(
    df: pd.DataFrame,
    col_types: ColumnTypes,
    impute: bool = False,
) -> pd.DataFrame:
    """
    One-hot encode categorical columns and prepare the DataFrame for sklearn.

    Parameters
    ----------
    impute:
        If True, impute missing values (median for numerical, ``"__missing__"``
        category for categorical) before encoding.
        If False (default), perform complete case analysis: rows with any NaN
        are dropped before encoding.
    """
    out = df.copy()

    if impute:
        for col, ctype in col_types.items():
            if col not in out.columns:
                continue
            if ctype == "categorical":
                out[col] = out[col].astype(str).fillna("__missing__")
            else:
                out[col] = out[col].fillna(out[col].median())
    else:
        out = out.dropna()

    # One-hot encode categorical columns
    cat_cols = [c for c in out.columns if col_types.get(c) == "categorical"]
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, drop_first=False)

    return out.astype(float)


# ---------------------------------------------------------------------------
# AucRoc
# ---------------------------------------------------------------------------

class AucRoc(BaseMetric):
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

    name = "AUC-ROC"
    description = (
        "Discriminator AUC-ROC: a classifier is trained to distinguish real from "
        "synthetic samples. Score = 1 when AUROC = 0.5 (indistinguishable)."
    )
    axis = "fidelity"

    def __init__(
        self,
        model: str = "rf",
        n_estimators: int = 100,
        max_depth: int = 3,
        cv_folds: int = 5,
        random_state: int = 42,
        impute: bool = False,
    ) -> None:
        self.model = model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.impute = impute

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_enc = _encode_for_sklearn(real, col_types, impute=self.impute)
        s_enc = _encode_for_sklearn(synthetic, col_types, impute=self.impute)

        # Align columns after one-hot expansion
        all_cols = sorted(set(r_enc.columns) | set(s_enc.columns))
        r_enc = r_enc.reindex(columns=all_cols, fill_value=0)
        s_enc = s_enc.reindex(columns=all_cols, fill_value=0)

        X = pd.concat([r_enc, s_enc], ignore_index=True)
        y = np.array([0] * len(r_enc) + [1] * len(s_enc))

        from sklearn.metrics import roc_auc_score

        oob_auroc = None

        if self.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True,
            )
            clf.fit(X, y)
            oob_proba = clf.oob_decision_function_[:, 1]
            oob_auroc = float(roc_auc_score(y, oob_proba))
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        # Guard: CV fails if only one class is present (e.g. all synthetic rows
        # dropped by dropna), or if the minority class has fewer samples than
        # cv_folds. Fall back to a neutral AUROC (0.5 → score = 1.0).
        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
        if len(class_counts) < 2 or min_class_count < self.cv_folds:
            mean_auroc = 0.5
            std_auroc = 0.0
            score = 1.0
            details = {
                "mean_auroc": mean_auroc,
                "std_auroc": std_auroc,
                "fold_aurocs": [],
                "n_real": len(real),
                "n_synthetic": len(synthetic),
                "oob_auroc": oob_auroc,
                "note": f"Skipped CV: only {len(class_counts)} class(es) present or minority class has {min_class_count} samples < {self.cv_folds} folds",
            }
            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
            )
        auroc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        mean_auroc = float(np.mean(auroc_scores))
        std_auroc = float(np.std(auroc_scores))
        score = float(1.0 - 2.0 * abs(mean_auroc - 0.5))

        details = {
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "fold_aurocs": auroc_scores.tolist(),
            "n_real": len(real),
            "n_synthetic": len(synthetic),
            "n_real_used": len(r_enc),
            "n_synthetic_used": len(s_enc),
            "impute": self.impute,
        }
        if oob_auroc is not None:
            details["oob_auroc"] = oob_auroc
            details["oob_error"] = float(1.0 - oob_auroc)
            details["oob_fidelity_score"] = float(1.0 - 2.0 * abs(oob_auroc - 0.5))

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, score),
            details=details,
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
    2. Fit a logistic regression classifier (default) via 5-fold cross-validation
       to predict P(synthetic) for each record.
    3. pMSE = (1/N) Σ (p̂_i − 0.5)²  following Woo et al. (2009).
       Under perfect fidelity P(synthetic) ≈ 0.5 everywhere → pMSE ≈ 0.
       Worst case: all records perfectly classified → pMSE = 0.25.

    Score
    -----
    Raw pMSE ∈ [0, 0.25] is linearly mapped to a score ∈ [1, 0]:

        score = 1 − 4 × pMSE

    so score = 1 when pMSE = 0 (indistinguishable) and score = 0 when
    pMSE = 0.25 (perfectly separable).

    References
    ----------
    Woo MJ et al. (2009) "Global measures of data utility for microdata masked
    for disclosure limitation." Journal of Privacy and Confidentiality, 1(1).

    Lautrup AD et al. (2025) "SynthEval: A Framework for Detailed Utility and
    Privacy Evaluation of Tabular Synthetic Data." Data Min Knowl Disc, 39(1).
    """

    name = "Propensity MSE"
    description = (
        "Propensity score MSE (Woo et al. 2009): measures how well a logistic "
        "regression can distinguish real from synthetic records via 5-fold CV. "
        "pMSE ∈ [0, 0.25]; score = 1 − 4 × pMSE."
    )
    axis = "fidelity"

    def __init__(
        self,
        model: str = "logistic",
        n_estimators: int = 100,
        max_iter: int = 1000,
        cv_folds: int = 5,
        random_state: int = 42,
        impute: bool = False,
    ) -> None:
        self.model = model
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.impute = impute

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        r_enc = _encode_for_sklearn(real, col_types, impute=self.impute)
        s_enc = _encode_for_sklearn(synthetic, col_types, impute=self.impute)

        all_cols = sorted(set(r_enc.columns) | set(s_enc.columns))
        r_enc = r_enc.reindex(columns=all_cols, fill_value=0)
        s_enc = s_enc.reindex(columns=all_cols, fill_value=0)

        X = pd.concat([r_enc, s_enc], ignore_index=True).values
        y = np.array([0] * len(r_enc) + [1] * len(s_enc))

        # Guard: CV fails if only one class present or minority class too small.
        # Fall back to neutral pMSE=0 → score=1.
        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
        if len(class_counts) < 2 or min_class_count < self.cv_folds:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={
                    "pmse": 0.0,
                    "pmse_worst_case": 0.25,
                    "c_synthetic_fraction": len(s_enc) / max(len(X), 1),
                    "propensity_scores": [],
                    "labels": y.tolist(),
                    "note": f"Skipped CV: only {len(class_counts)} class(es) present or minority class has {min_class_count} samples < {self.cv_folds} folds",
                },
            )

        # Compute propensity scores via k-fold cross-validation to avoid overfitting
        clf = self._build_model()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        propensities = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

        pmse = float(np.mean((propensities - 0.5) ** 2))
        # Linear mapping: pMSE=0 → score=1, pMSE=0.25 → score=0
        score = float(1.0 - 4.0 * pmse)

        return MetricResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, score)),
            details={
                "pmse": pmse,
                "pmse_worst_case": 0.25,
                "c_synthetic_fraction": len(s_enc) / len(X),
                "propensity_scores": propensities.tolist(),
                "labels": y.tolist(),
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


# ---------------------------------------------------------------------------
# Shared helpers for cross-classification metrics
# ---------------------------------------------------------------------------

def _crcl_encode_aligned(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    col_types: ColumnTypes,
    impute: bool = False,
):
    """
    Encode real and synthetic datasets into a shared feature space for a given
    target column.  Returns (X_real, y_real, X_synth, y_synth, is_categorical).

    One-hot encoding uses the union of category values from both datasets so
    that real and synthetic feature matrices share identical columns.
    Rows with NaN in the target are always dropped first.
    Feature NaNs: dropped row-wise (complete case, default) or imputed.
    Returns None if either encoded dataset has fewer than 2 rows.
    """
    feature_cols = [c for c in col_types if c != target_col
                    and c in real.columns and c in synthetic.columns]
    is_cat_target = col_types.get(target_col) == "categorical"

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[feature_cols + [target_col]].dropna(subset=[target_col])
        if not impute:
            return sub.dropna()
        for col in feature_cols:
            if col_types.get(col) == "categorical":
                sub = sub.copy()
                sub[col] = sub[col].astype(str).fillna("__missing__")
            else:
                sub = sub.copy()
                sub[col] = sub[col].fillna(sub[col].median())
        return sub

    r = _clean(real)
    s = _clean(synthetic)
    if len(r) < 2 or len(s) < 2:
        return None

    # Encode target with a shared label space
    if is_cat_target:
        all_vals = sorted(set(r[target_col].astype(str)) | set(s[target_col].astype(str)))
        le = LabelEncoder()
        le.fit(all_vals)
        y_r = le.transform(r[target_col].astype(str))
        y_s = le.transform(s[target_col].astype(str))
    else:
        y_r = r[target_col].values.astype(float)
        y_s = s[target_col].values.astype(float)

    # One-hot encode features using the union of categories from both datasets
    cat_feat_cols = [c for c in feature_cols if col_types.get(c) == "categorical"]
    r_feat = r[feature_cols].copy()
    s_feat = s[feature_cols].copy()

    if cat_feat_cols:
        r_dummies = pd.get_dummies(r_feat, columns=cat_feat_cols, drop_first=False)
        s_dummies = pd.get_dummies(s_feat, columns=cat_feat_cols, drop_first=False)
        all_cols = sorted(set(r_dummies.columns) | set(s_dummies.columns))
        r_feat = r_dummies.reindex(columns=all_cols, fill_value=0)
        s_feat = s_dummies.reindex(columns=all_cols, fill_value=0)

    return r_feat.values.astype(float), y_r, s_feat.values.astype(float), y_s, is_cat_target


def _crcl_score(y_true, y_pred, is_categorical: bool) -> float:
    """Accuracy for categorical targets, R² for numerical."""
    if is_categorical:
        return float(accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def _build_dt(is_categorical: bool, random_state: int, max_depth=None):
    if is_categorical:
        return DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    return DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)


def _crcl_run(
    train_X, train_y, other_X, other_y,
    is_cat: bool, test_size: float, random_state: int, max_depth,
):
    """
    Split train set, fit decision tree, score on held-out split and on other set.
    Returns (perf_held_out, perf_other, ratio) or None if too few classes.
    """
    if is_cat and len(np.unique(train_y)) < 2:
        return None
    try:
        X_tr, X_held, y_tr, y_held = train_test_split(
            train_X, train_y, test_size=test_size, random_state=random_state,
            stratify=train_y if is_cat else None,
        )
    except ValueError:
        # Fallback: some classes too rare to stratify — use random split
        X_tr, X_held, y_tr, y_held = train_test_split(
            train_X, train_y, test_size=test_size, random_state=random_state,
        )
    clf = _build_dt(is_cat, random_state, max_depth)
    clf.fit(X_tr, y_tr)
    perf_held = _crcl_score(y_held, clf.predict(X_held), is_cat)
    perf_other = _crcl_score(other_y, clf.predict(other_X), is_cat)
    ratio = perf_other / perf_held if abs(perf_held) > 1e-9 else float("nan")
    return perf_held, perf_other, ratio


# ---------------------------------------------------------------------------
# CrossClassification  (unified RS / SR via mode parameter)
# ---------------------------------------------------------------------------

class CrossClassification(BaseMetric):
    """
    Unified cross-classification metric supporting both CrCl-RS and CrCl-SR
    modes via the ``mode`` parameter.

    For each variable used as a target in turn, the remaining variables are
    used as predictors.  A decision tree is trained on one dataset and
    evaluated on both a held-out split of the training dataset and on the
    other dataset.  The ratio of the two performance values is computed, and
    the mean ratio across all target variables is reported.  A value close to
    1 is ideal.

    Modes
    -----
    ``"RS"`` — CrCl-RS: train on real, evaluate on held-out real and synthetic.
        Useful for assessing whether the statistical properties of the real data
        are preserved in the synthetic data.

    ``"SR"`` — CrCl-SR: train on synthetic, evaluate on held-out synthetic and real.
        Useful for assessing whether conclusions from models trained on synthetic
        data transfer to real data.

    Procedure (repeated for each variable as target)
    ------------------------------------------------
    1. Encode real and synthetic into a shared feature space (union OHE).
    2. Split the training dataset (real for RS, synthetic for SR) into
       train / held-out test sets.
    3. Train a decision tree on the training split.
    4. Evaluate on the held-out test set  → perf_held
    5. Evaluate on the other dataset      → perf_other
    6. ratio = perf_other / perf_held

    Performance metric: accuracy (categorical target), R² (numerical target).

    Score
    -----
        score = max(0, 1 − |mean_ratio − 1|)

    References
    ----------
    Goncalves A, Ray P, Soper B, Stevens J, Coyle L, Sales AP.
    Generation and evaluation of synthetic patient data.
    BMC Med Res Methodol. 2020;20(1):108.
    doi:10.1186/s12874-020-00977-1
    """

    axis = "fidelity"

    def __init__(
        self,
        mode: str = "RS",
        test_size: float = 0.3,
        max_depth: int = None,
        random_state: int = 42,
        impute: bool = False,
    ) -> None:
        if mode not in ("RS", "SR"):
            raise ValueError(f"mode must be 'RS' or 'SR', got {mode!r}")
        self.mode = mode
        self.test_size = test_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.impute = impute

    @property
    def name(self) -> str:
        return f"CrCl-{self.mode}"

    @property
    def description(self) -> str:
        if self.mode == "RS":
            return (
                "Cross-classification (train real, test real+synth): ratio of synthetic "
                "to held-out real performance per target variable (ideal = 1)."
            )
        return (
            "Cross-classification (train synth, test synth+real): ratio of real "
            "to held-out synthetic performance per target variable (ideal = 1)."
        )

    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        per_variable: dict = {}
        skipped: list = []

        for target_col in col_types:
            encoded = _crcl_encode_aligned(real, synthetic, target_col, col_types, self.impute)
            if encoded is None:
                skipped.append(target_col)
                continue
            X_r, y_r, X_s, y_s, is_cat = encoded

            # Select train/other based on mode
            if self.mode == "RS":
                train_X, train_y, other_X, other_y = X_r, y_r, X_s, y_s
                perf_keys = ("perf_real_test", "perf_synth")
            else:
                train_X, train_y, other_X, other_y = X_s, y_s, X_r, y_r
                perf_keys = ("perf_synth_test", "perf_real")

            if len(train_X) < 10:
                skipped.append(target_col)
                continue

            result = _crcl_run(
                train_X, train_y, other_X, other_y,
                is_cat, self.test_size, self.random_state, self.max_depth,
            )
            if result is None:
                skipped.append(target_col)
                continue

            perf_held, perf_other, ratio = result
            per_variable[target_col] = {
                perf_keys[0]: perf_held,
                perf_keys[1]: perf_other,
                "ratio": ratio,
                "target_type": "categorical" if is_cat else "numerical",
            }

        valid_ratios = [
            v["ratio"] for v in per_variable.values() if not np.isnan(v["ratio"])
        ]
        if not valid_ratios:
            return MetricResult(
                metric_name=self.name, score=1.0,
                details={"message": "No valid target variables.", "skipped": skipped},
            )

        mean_ratio = float(np.mean(valid_ratios))
        score = max(0.0, float(1.0 - abs(mean_ratio - 1.0)))
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "mean_ratio": mean_ratio,
                "per_variable": per_variable,
                "skipped": skipped,
                "mode": self.mode,
            },
        )


# Convenience aliases
class CrossClassificationRS(CrossClassification):
    def __init__(self, **kw):
        super().__init__(mode="RS", **kw)


class CrossClassificationSR(CrossClassification):
    def __init__(self, **kw):
        super().__init__(mode="SR", **kw)