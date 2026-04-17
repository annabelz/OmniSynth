"""
Missingness scenarios for meta-evaluation.

Each scenario function generates ``n_datasets`` variants of an input DataFrame
with controlled missingness patterns and writes them to ``output_dir``.
All functions are reproducible: dataset *i* (0-indexed) uses seed
``random_seed + i``.

Return value: list of absolute file paths written.

Available scenarios
-------------------
missingness_1 — 10 % MCAR
    Replace 10 % of all cells (across the entire dataset) with NaN, chosen
    completely at random.

missingness_2 — 20 % MCAR
    Same as missingness_1 with 20 % of cells replaced.

missingness_3 — 30 % MCAR
    Same as missingness_1 with 30 % of cells replaced.

missingness_4 — MAR bivariate (conditioning on A, masking B)
    For each dataset, one (variable-pair, quartile) combination is selected.
    Rows in the chosen quartile of conditioning variable A are identified;
    50 % of those rows have their value of B replaced with NaN.

    Conditioning variable A:
      • Numerical/ordinal : quartile 0–3 maps to
        [−∞, Q1), [Q1, Q2), [Q2, Q3), [Q3, +∞].
      • Categorical       : sorted unique categories split into 4 equal groups;
        quartile index selects which group's rows are targeted.

    Iteration order across datasets: all pairs are cycled first (inner loop),
    quartile index advances second (outer loop).

missingness_5 — MAR univariate self-conditioning
    For each dataset, one (variable, quartile) combination is selected.
    Rows in the chosen quartile of the variable itself are identified;
    50 % of those rows have their value replaced with NaN.

    Conditioning:
      • Numerical/ordinal : same quartile logic as missingness_4.
      • Categorical       : sorted categories split into 4 equal groups.

    Iteration order: all variables are cycled first (inner loop),
    quartile index advances second (outer loop).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes


# ===========================================================================
# Shared helpers (reuse quartile-mask logic from fidelity.py without import)
# ===========================================================================

def _numerical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
    """Boolean mask: rows whose value falls in the given quartile (0–3)."""
    q = series.quantile([0.25, 0.50, 0.75]).values
    if quartile_idx == 0:
        mask = series < q[0]
    elif quartile_idx == 1:
        mask = (series >= q[0]) & (series < q[1])
    elif quartile_idx == 2:
        mask = (series >= q[1]) & (series < q[2])
    else:
        mask = series >= q[2]
    return mask.fillna(False)


def _categorical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
    """Boolean mask: rows whose category belongs to the quartile_idx-th quarter."""
    cats = sorted(series.dropna().unique().tolist(), key=str)
    n = len(cats)
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    start = sum(sizes[:quartile_idx])
    group = set(cats[start: start + sizes[quartile_idx]])
    return series.isin(group)


def _is_ordinal(series: pd.Series) -> bool:
    vals = series.dropna()
    if len(vals) == 0:
        return False
    return bool(np.all(vals.values == np.floor(vals.values)))


# ===========================================================================
# Scenarios 1–3 — MCAR at fixed rates
# ===========================================================================

def _mcar(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: Path,
    prefix: str,
    random_seed: int,
    missing_rate: float,
) -> List[str]:
    """
    Replace *missing_rate* fraction of all cells with NaN, completely at
    random (MCAR).  Cells that are already NaN are not counted toward the
    target rate — only observed cells are candidates for masking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for i in range(n_datasets):
        rng = np.random.default_rng(random_seed + i)
        noisy = df.copy()

        # Build flat index of observed cells
        observed_mask = ~df.isnull()
        observed_positions = list(zip(*np.where(observed_mask.values)))

        n_to_mask = int(round(missing_rate * len(observed_positions)))
        if n_to_mask > 0 and observed_positions:
            chosen = rng.choice(len(observed_positions), size=n_to_mask, replace=False)
            for pos in chosen:
                row_i, col_i = observed_positions[pos]
                noisy.iat[row_i, col_i] = np.nan

        out_path = output_dir / f"{prefix}_{i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))

    return paths


def scenario_missingness_1(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "missingness_1",
    random_seed: int = 42,
) -> List[str]:
    """
    Missingness Scenario 1 — 10 % MCAR.

    Replaces 10 % of all observed cells with NaN, chosen completely at random
    (Missing Completely At Random).  Dataset *i* uses seed ``random_seed + i``.
    """
    return _mcar(df, n_datasets, Path(output_dir), prefix, random_seed, 0.10)


def scenario_missingness_2(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "missingness_2",
    random_seed: int = 42,
) -> List[str]:
    """
    Missingness Scenario 2 — 20 % MCAR.

    Replaces 20 % of all observed cells with NaN, completely at random.
    Dataset *i* uses seed ``random_seed + i``.
    """
    return _mcar(df, n_datasets, Path(output_dir), prefix, random_seed, 0.20)


def scenario_missingness_3(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "missingness_3",
    random_seed: int = 42,
) -> List[str]:
    """
    Missingness Scenario 3 — 30 % MCAR.

    Replaces 30 % of all observed cells with NaN, completely at random.
    Dataset *i* uses seed ``random_seed + i``.
    """
    return _mcar(df, n_datasets, Path(output_dir), prefix, random_seed, 0.30)


# ===========================================================================
# Scenario 4 — MAR bivariate (A conditions missingness in B)
# ===========================================================================

def scenario_missingness_4(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "missingness_4",
    random_seed: int = 42,
) -> List[str]:
    """
    Missingness Scenario 4 — MAR bivariate conditioning.

    Each dataset applies one targeted missingness pattern defined by a
    (variable-pair, quartile) combination.  Conditioning variable A
    determines which rows are targeted; 50 % of those rows have their
    value of B replaced with NaN.

    Conditioning variable A:
      • Numerical/ordinal : quartile 0–3 selects
        [−∞, Q1), [Q1, Q2), [Q2, Q3), [Q3, +∞].
      • Categorical       : sorted categories split into 4 equal groups;
        quartile index selects the group.

    Iteration order: all ordered pairs (A, B) cycled first (inner loop),
    quartile index advances second (outer loop).  Ordered pairs means both
    (A→B) and (B→A) are considered, so each variable can act as either the
    conditioning variable or the target.

    Dataset *i* uses seed ``random_seed + i``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in col_types if c in df.columns]
    # Ordered pairs: both (A, B) and (B, A) so each var can condition or be masked
    all_pairs = [(a, b) for a, b in itertools.permutations(cols, 2)]
    n_pairs = len(all_pairs)

    if n_pairs == 0:
        raise ValueError("Need at least 2 columns to form variable pairs.")

    paths: List[str] = []

    for i in range(n_datasets):
        rng = np.random.default_rng(random_seed + i)

        pair_idx = i % n_pairs
        quartile_idx = (i // n_pairs) % 4

        a_col, b_col = all_pairs[pair_idx]
        a_type = col_types[a_col]

        noisy = df.copy()

        # Rows in quartile of A
        if a_type == "numerical":
            mask = _numerical_quartile_mask(df[a_col], quartile_idx)
        else:
            mask = _categorical_quartile_mask(df[a_col], quartile_idx)

        rows_idx = df.index[mask].tolist()
        if not rows_idx:
            out_path = output_dir / f"{prefix}_{i:03d}.csv"
            noisy.to_csv(out_path, index=False)
            paths.append(str(out_path.resolve()))
            continue

        # Select 50 % of those rows (only observed cells in B are candidates)
        observed_in_b = [idx for idx in rows_idx if pd.notna(df.at[idx, b_col])]
        n_mask = max(1, len(observed_in_b) // 2)
        if observed_in_b:
            chosen = rng.choice(len(observed_in_b), size=min(n_mask, len(observed_in_b)), replace=False)
            for j in chosen:
                noisy.at[observed_in_b[j], b_col] = np.nan

        out_path = output_dir / f"{prefix}_{i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))

    return paths


# ===========================================================================
# Scenario 5 — MNAR univariate self-conditioning
# ===========================================================================

def scenario_missingness_5(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "missingness_5",
    random_seed: int = 42,
) -> List[str]:
    """
    Missingness Scenario 5 — MAR univariate self-conditioning.

    Each dataset applies one targeted missingness pattern defined by a
    (variable, quartile) combination.  50 % of the rows in the chosen
    quartile of the variable have their value replaced with NaN.

    Conditioning:
      • Numerical/ordinal : quartile 0–3 selects
        [−∞, Q1), [Q1, Q2), [Q2, Q3), [Q3, +∞].
      • Categorical       : sorted categories split into 4 equal groups;
        quartile index selects the group.

    Iteration order: all variables cycled first (inner loop), quartile
    index advances second (outer loop).

    Dataset *i* uses seed ``random_seed + i``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in col_types if c in df.columns]
    n_cols = len(cols)

    if n_cols == 0:
        raise ValueError("No columns available.")

    paths: List[str] = []

    for i in range(n_datasets):
        rng = np.random.default_rng(random_seed + i)

        col_idx = i % n_cols
        quartile_idx = (i // n_cols) % 4

        col = cols[col_idx]
        ctype = col_types[col]

        noisy = df.copy()

        # Rows in quartile of the variable itself
        if ctype == "numerical":
            mask = _numerical_quartile_mask(df[col], quartile_idx)
        else:
            mask = _categorical_quartile_mask(df[col], quartile_idx)

        rows_idx = df.index[mask].tolist()

        # Only observed cells are candidates
        observed_in_col = [idx for idx in rows_idx if pd.notna(df.at[idx, col])]
        n_mask = max(1, len(observed_in_col) // 2)
        if observed_in_col:
            chosen = rng.choice(len(observed_in_col), size=min(n_mask, len(observed_in_col)), replace=False)
            for j in chosen:
                noisy.at[observed_in_col[j], col] = np.nan

        out_path = output_dir / f"{prefix}_{i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))

    return paths


# ===========================================================================
# Registry — maps scenario name → function
# ===========================================================================

MISSINGNESS_SCENARIOS: Dict[str, callable] = {
    "missingness_1": scenario_missingness_1,
    "missingness_2": scenario_missingness_2,
    "missingness_3": scenario_missingness_3,
    "missingness_4": scenario_missingness_4,
    "missingness_5": scenario_missingness_5,
}
