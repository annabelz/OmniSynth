"""
Shared pytest fixtures for OmniSynth tests.

Fixtures
--------
col_types           ColumnTypes dict for the standard test schema.
real_df             Small deterministic DataFrame (no missing values).
synth_perfect       Exact copy of real_df — all metrics should score ≈ 1.0.
synth_shifted       Numerically shifted / categorically reassigned — scores should be < 1.0.
real_with_missing   real_df with controlled NaN patterns in two columns.
synth_with_missing  Synthetic counterpart with similar (but not identical) missingness.
only_numerical_df   DataFrame with only numerical columns.
only_categorical_df DataFrame with only categorical columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnisynth.utils.data_utils import ColumnTypes


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def col_types() -> ColumnTypes:
    return {
        "age":       "numerical",
        "bmi":       "numerical",
        "score":     "numerical",
        "sex":       "categorical",
        "diagnosis": "categorical",
        "smoker":    "categorical",
    }


@pytest.fixture(scope="session")
def col_types_num_only() -> ColumnTypes:
    return {"x": "numerical", "y": "numerical", "z": "numerical"}


@pytest.fixture(scope="session")
def col_types_cat_only() -> ColumnTypes:
    return {"color": "categorical", "grade": "categorical"}


# ---------------------------------------------------------------------------
# Base real DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "age":       rng.normal(50, 10, n).clip(18, 90),
        "bmi":       rng.normal(25, 4, n).clip(15, 45),
        "score":     rng.normal(100, 15, n),
        "sex":       rng.choice(["M", "F"], n),
        "diagnosis": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "smoker":    rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })


# ---------------------------------------------------------------------------
# Perfect synthetic (exact copy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synth_perfect(real_df) -> pd.DataFrame:
    """Identical to real — every metric should return score ≈ 1.0."""
    return real_df.copy()


# ---------------------------------------------------------------------------
# Shifted synthetic (deliberately different)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synth_shifted(real_df) -> pd.DataFrame:
    """
    Numerically shifted and categorically imbalanced — scores should be
    noticeably below 1.0 for distribution-sensitive metrics.
    """
    rng = np.random.default_rng(99)
    n = len(real_df)
    return pd.DataFrame({
        "age":       rng.normal(65, 5, n).clip(18, 90),   # much older
        "bmi":       rng.normal(32, 3, n).clip(15, 45),   # higher BMI
        "score":     rng.normal(70, 10, n),                # lower score
        "sex":       rng.choice(["M", "F"], n, p=[0.9, 0.1]),   # heavily imbalanced
        "diagnosis": rng.choice(["A", "B", "C"], n, p=[0.1, 0.1, 0.8]),  # inverted
        "smoker":    rng.choice(["yes", "no"], n, p=[0.9, 0.1]),  # inverted
    })


# ---------------------------------------------------------------------------
# DataFrames with missing values
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_with_missing(real_df) -> pd.DataFrame:
    """real_df with ~15 % missingness in 'bmi' and ~10 % in 'smoker'."""
    rng = np.random.default_rng(1)
    df = real_df.copy()
    n = len(df)
    df.loc[rng.choice(n, size=int(0.15 * n), replace=False), "bmi"] = np.nan
    df.loc[rng.choice(n, size=int(0.10 * n), replace=False), "smoker"] = np.nan
    return df


@pytest.fixture(scope="session")
def synth_with_missing(real_df) -> pd.DataFrame:
    """Synthetic with similar (but independently drawn) missingness rates."""
    rng = np.random.default_rng(2)
    n = len(real_df)
    df = pd.DataFrame({
        "age":       rng.normal(50, 10, n).clip(18, 90),
        "bmi":       rng.normal(25, 4, n).clip(15, 45),
        "score":     rng.normal(100, 15, n),
        "sex":       rng.choice(["M", "F"], n),
        "diagnosis": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "smoker":    rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })
    df.loc[rng.choice(n, size=int(0.14 * n), replace=False), "bmi"] = np.nan
    df.loc[rng.choice(n, size=int(0.11 * n), replace=False), "smoker"] = np.nan
    return df


@pytest.fixture(scope="session")
def synth_wrong_missing(real_df) -> pd.DataFrame:
    """Synthetic with missingness in completely different columns from real."""
    rng = np.random.default_rng(3)
    n = len(real_df)
    df = pd.DataFrame({
        "age":       rng.normal(50, 10, n).clip(18, 90),
        "bmi":       rng.normal(25, 4, n).clip(15, 45),
        "score":     rng.normal(100, 15, n),
        "sex":       rng.choice(["M", "F"], n),
        "diagnosis": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "smoker":    rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })
    # Missing in 'age' and 'score' instead of 'bmi' and 'smoker'
    df.loc[rng.choice(n, size=int(0.15 * n), replace=False), "age"] = np.nan
    df.loc[rng.choice(n, size=int(0.10 * n), replace=False), "score"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Single-type DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def only_numerical_df() -> pd.DataFrame:
    rng = np.random.default_rng(10)
    n = 100
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(5, 2, n),
        "z": rng.normal(-3, 0.5, n),
    })


@pytest.fixture(scope="session")
def only_categorical_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 100
    return pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
    })
