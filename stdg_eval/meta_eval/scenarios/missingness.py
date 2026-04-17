"""
Missingness scenarios for meta-evaluation.

Each public ``scenario_missingness_*`` function builds a *transform* — a
closure that generates one noisy variant of the input DataFrame — and
delegates the generation loop, timing, and CSV writing to
:func:`~stdg_eval.meta_eval.scenarios.base.generate_datasets`.

Available scenarios
-------------------
missingness_1 — 10 % MCAR
    Replace 10 % of observed cells with NaN at random.

missingness_2 — 20 % MCAR
    Replace 20 % of observed cells with NaN at random.

missingness_3 — 30 % MCAR
    Replace 30 % of observed cells with NaN at random.

missingness_4 — MAR bivariate
    Rows in a quartile of variable A have 50 % of their B values masked;
    cycles all ordered (A, B) pairs × 4 quartiles across datasets.

missingness_5 — MNAR self-conditioning
    Rows in a quartile of variable X have 50 % of their X values masked;
    cycles all variables × 4 quartiles across datasets.
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes
from stdg_eval.meta_eval.scenarios.base import (
    generate_datasets,
    numerical_quartile_mask,
    categorical_quartile_mask,
)


# ===========================================================================
# Transform factories
# ===========================================================================

def _mcar_transform(missing_rate: float):
    """Return a transform that masks *missing_rate* fraction of observed cells."""
    def transform(df: pd.DataFrame, rng: np.random.Generator, _col_types: ColumnTypes, _i: int) -> pd.DataFrame:
        noisy = df.copy()
        observed_positions = list(zip(*np.where(~df.isnull().values)))
        n_to_mask = int(round(missing_rate * len(observed_positions)))
        if n_to_mask > 0 and observed_positions:
            chosen = rng.choice(len(observed_positions), size=n_to_mask, replace=False)
            for pos in chosen:
                row_i, col_i = observed_positions[pos]
                noisy.iat[row_i, col_i] = np.nan
        return noisy
    return transform


def _mar_bivariate_transform(all_pairs: list):
    """Return a transform that masks 50 % of B values in one quartile of A."""
    def transform(df: pd.DataFrame, rng: np.random.Generator, col_types: ColumnTypes, i: int) -> pd.DataFrame:
        noisy = df.copy()
        pair_idx = i % len(all_pairs)
        quartile_idx = (i // len(all_pairs)) % 4
        a_col, b_col = all_pairs[pair_idx]

        if col_types[a_col] == "numerical":
            mask = numerical_quartile_mask(df[a_col], quartile_idx)
        else:
            mask = categorical_quartile_mask(df[a_col], quartile_idx)

        rows_idx = df.index[mask].tolist()
        if not rows_idx:
            return noisy

        observed_in_b = [idx for idx in rows_idx if pd.notna(df.at[idx, b_col])]
        n_mask = max(1, len(observed_in_b) // 2)
        if observed_in_b:
            chosen = rng.choice(len(observed_in_b), size=min(n_mask, len(observed_in_b)), replace=False)
            for j in chosen:
                noisy.at[observed_in_b[j], b_col] = np.nan
        return noisy
    return transform


def _mnar_self_transform(cols: list):
    """Return a transform that masks 50 % of X values in one quartile of X itself."""
    def transform(df: pd.DataFrame, rng: np.random.Generator, col_types: ColumnTypes, i: int) -> pd.DataFrame:
        noisy = df.copy()
        col_idx = i % len(cols)
        quartile_idx = (i // len(cols)) % 4
        col = cols[col_idx]

        if col_types[col] == "numerical":
            mask = numerical_quartile_mask(df[col], quartile_idx)
        else:
            mask = categorical_quartile_mask(df[col], quartile_idx)

        rows_idx = df.index[mask].tolist()
        observed_in_col = [idx for idx in rows_idx if pd.notna(df.at[idx, col])]
        n_mask = max(1, len(observed_in_col) // 2)
        if observed_in_col:
            chosen = rng.choice(len(observed_in_col), size=min(n_mask, len(observed_in_col)), replace=False)
            for j in chosen:
                noisy.at[observed_in_col[j], col] = np.nan
        return noisy
    return transform


# ===========================================================================
# Public scenario functions
# ===========================================================================

def scenario_missingness_1(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "missingness_1",
    random_seed: int = 42,
    verbose: bool = False,
) -> List[str]:
    """Missingness Scenario 1 — 10 % MCAR."""
    return generate_datasets(
        _mcar_transform(0.10),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
    )


def scenario_missingness_2(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "missingness_2",
    random_seed: int = 42,
    verbose: bool = False,
) -> List[str]:
    """Missingness Scenario 2 — 20 % MCAR."""
    return generate_datasets(
        _mcar_transform(0.20),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
    )


def scenario_missingness_3(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "missingness_3",
    random_seed: int = 42,
    verbose: bool = False,
) -> List[str]:
    """Missingness Scenario 3 — 30 % MCAR."""
    return generate_datasets(
        _mcar_transform(0.30),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
    )


def scenario_missingness_4(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "missingness_4",
    random_seed: int = 42,
    verbose: bool = False,
) -> List[str]:
    """Missingness Scenario 4 — MAR bivariate conditioning."""
    cols = [c for c in col_types if c in df.columns]
    all_pairs = [(a, b) for a, b in itertools.permutations(cols, 2)]
    if not all_pairs:
        raise ValueError("Need at least 2 columns to form variable pairs.")
    return generate_datasets(
        _mar_bivariate_transform(all_pairs),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
    )


def scenario_missingness_5(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "missingness_5",
    random_seed: int = 42,
    verbose: bool = False,
) -> List[str]:
    """Missingness Scenario 5 — MNAR self-conditioning."""
    cols = [c for c in col_types if c in df.columns]
    if not cols:
        raise ValueError("No columns available.")
    return generate_datasets(
        _mnar_self_transform(cols),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
    )


# ===========================================================================
# Registry
# ===========================================================================

MISSINGNESS_SCENARIOS: Dict[str, callable] = {
    "missingness_1": scenario_missingness_1,
    "missingness_2": scenario_missingness_2,
    "missingness_3": scenario_missingness_3,
    "missingness_4": scenario_missingness_4,
    "missingness_5": scenario_missingness_5,
}
