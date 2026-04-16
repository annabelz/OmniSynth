"""
Fidelity scenarios for meta-evaluation.

Each scenario function generates ``n_datasets`` noisy variants of an input
DataFrame and writes them to ``output_dir``.  All functions are reproducible:
dataset *i* (0-indexed) uses seed ``random_seed + i``, so re-running with the
same seed and input yields identical outputs.

Return value: list of absolute file paths written.

Available scenarios
-------------------
fidelity_1 — Low Gaussian noise, all variables
    Numerical / ordinal : N(0, 1·std) per cell; ordinal rounded to nearest int.
    Categorical         : one-hot + N(0,1) per category → argmax (noisy flip).

fidelity_2 — Low Gaussian noise, numerical/ordinal only
    Same as fidelity_1 for numerical/ordinal; categorical variables unchanged.

fidelity_3 — High Gaussian noise, all variables
    Numerical / ordinal : N(0, 2·std) per cell; ordinal rounded.
    Categorical         : pure N(0,1) per category → argmax (ignores current
                          value; effectively random reassignment).

fidelity_4 — High Gaussian noise, numerical/ordinal only
    Same noise as fidelity_3 for numerical/ordinal; categorical unchanged.

fidelity_5 — Structured bivariate noise
    For each dataset, one (variable-pair, quartile) combination is selected.
    Rows in the chosen quartile of the conditioning variable A are identified;
    50 % of those rows are selected at random and B is perturbed:
      • B numerical/ordinal : compute the local mean of the selected B values;
            if local mean > global mean → add +|N(0, std(B))| (push higher);
            else → add −|N(0, std(B))| (push lower); ordinal rounded after.
      • B categorical       : replace each selected value with a random draw
            from a randomly chosen quarter of B's categories.
    Conditioning variable A:
      • Numerical/ordinal : quartile index 0–3 maps to [Q0,Q1), [Q1,Q2),
            [Q2,Q3), [Q3,Q4] of A's distribution.
      • Categorical       : categories sorted and split into four equal groups;
            quartile index selects which group defines the row subset.
    Iteration order across datasets: all variable pairs are cycled first (inner
    loop), then quartile index advances (outer loop).  This ensures every pair
    is visited at each quartile before repeating.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes


# ===========================================================================
# Shared low-level helpers
# ===========================================================================

def _is_ordinal(series: pd.Series) -> bool:
    """True if all non-missing values are integers (no fractional part)."""
    vals = series.dropna()
    if len(vals) == 0:
        return False
    return bool(np.all(vals.values == np.floor(vals.values)))


def _add_numerical_noise(
    series: pd.Series,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> pd.Series:
    """
    Add N(0, noise_scale · std(col)) noise to a numerical column.
    NaN cells are preserved.
    """
    std = float(series.std(ddof=1)) 
    if np.isnan(std) or std == 0.0:
        return series.copy()
    noisy = series.values.copy().astype(float)
    mask = ~np.isnan(noisy)
    noisy[mask] += rng.normal(0.0, noise_scale * std, int(mask.sum()))
    return pd.Series(noisy, index=series.index, name=series.name)


def _snap_to_observed_ints(series: pd.Series, observed_ints: np.ndarray) -> pd.Series:
    """Round each value to the nearest integer in *observed_ints*; preserve NaN."""
    def _snap(v):
        if np.isnan(v):
            return v
        return float(observed_ints[int(np.argmin(np.abs(observed_ints - v)))])
    return series.map(_snap)


def _add_ordinal_noise(
    series: pd.Series,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> pd.Series:
    """
    Add N(0, noise_scale · std(col)) noise to an ordinal (integer-valued
    numerical) column, then snap each value to the nearest observed integer.
    """
    observed_ints = np.sort(np.unique(series.dropna().values.astype(int)))
    noisy = _add_numerical_noise(series, rng, noise_scale=noise_scale)
    return _snap_to_observed_ints(noisy, observed_ints)


def _add_categorical_noise_onehot(
    series: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Fidelity-1/2 style categorical noise.

    For each cell, construct a one-hot vector (current category = 1, others 0),
    add N(0, 1) noise to every element, and assign the category with the
    highest resulting score.  This gives a meaningful but moderate flip
    probability (~16 % for a binary variable).
    """
    categories = sorted(series.dropna().unique().tolist())
    if len(categories) <= 1:
        return series.copy()
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    result = series.copy().astype(object)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        base = np.zeros(k)
        base[cat_index[val]] = 1.0
        scores = base + rng.normal(0.0, 1.0, k)
        result.at[idx] = categories[int(np.argmax(scores))]
    return result


def _add_categorical_noise_random(
    series: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Fidelity-3/4 style categorical noise.

    For each cell, the starting score for each category is its empirical
    proportion in the column (e.g. a category appearing 10 % of the time
    starts at 0.1).  N(0, 1) noise is then added to every category's score
    and the argmax is assigned.  The current cell value is ignored, so the
    result is driven by the marginal distribution plus noise.
    """
    categories = sorted(series.dropna().unique().tolist())
    if len(categories) <= 1:
        return series.copy()

    # Empirical proportions as the base score for each category
    counts = series.value_counts(normalize=True)
    base = np.array([counts.get(c, 0.0) for c in categories])

    k = len(categories)
    result = series.copy().astype(object)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        scores = base + rng.normal(0.0, 1.0, k)
        result.at[idx] = categories[int(np.argmax(scores))]
    return result


# ===========================================================================
# Scenario-level helpers for fidelity_5
# ===========================================================================

def _numerical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
    """
    Boolean mask for rows whose non-NaN value of *series* falls in the given
    quartile (0 = lowest, 3 = highest).
    """
    q = series.quantile([0.25, 0.50, 0.75]).values  # Q1, Q2, Q3
    if quartile_idx == 0:
        mask = series < q[0]
    elif quartile_idx == 1:
        mask = (series >= q[0]) & (series < q[1])
    elif quartile_idx == 2:
        mask = (series >= q[1]) & (series < q[2])
    else:  # 3
        mask = series >= q[2]
    return mask.fillna(False)


def _categorical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
    """
    Boolean mask for rows whose categorical value belongs to the
    quartile_idx-th quarter of sorted unique categories.
    """
    cats = sorted(series.dropna().unique().tolist(), key=str)
    n = len(cats)
    # Split into 4 groups as evenly as possible; last group absorbs remainder
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    start = sum(sizes[:quartile_idx])
    group = set(cats[start: start + sizes[quartile_idx]])
    return series.isin(group)


def _perturb_b_numerical(
    noisy: pd.DataFrame,
    b_col: str,
    transform_idx: List,
    df: pd.DataFrame,
    rng: np.random.Generator,
    is_ordinal: bool,
) -> None:
    """
    Perturb B (numerical/ordinal) in-place for the selected rows.

    local_mean > global_mean → add +|N(0, std(B))| (push values higher)
    local_mean ≤ global_mean → add −|N(0, std(B))| (push values lower)
    Ordinal columns are snapped to the nearest observed integer after noise.
    """
    std_b = float(df[b_col].std(ddof=1))
    if np.isnan(std_b) or std_b == 0.0:
        return
    global_mean = float(df[b_col].mean())

    valid_idx = [i for i in transform_idx if pd.notna(noisy.at[i, b_col])]
    if not valid_idx:
        return

    local_mean = float(noisy.loc[valid_idx, b_col].mean())
    sign = 1.0 if local_mean > global_mean else -1.0
    noise = sign * np.abs(rng.normal(0.0, std_b, len(valid_idx)))

    for j, idx in enumerate(valid_idx):
        noisy.at[idx, b_col] = float(noisy.at[idx, b_col]) + noise[j]

    if is_ordinal:
        observed_ints = np.sort(np.unique(df[b_col].dropna().values.astype(int)))
        noisy[b_col] = _snap_to_observed_ints(noisy[b_col], observed_ints)


def _perturb_b_categorical(
    noisy: pd.DataFrame,
    b_col: str,
    transform_idx: List,
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    """
    Replace each selected B value with a random draw from a randomly chosen
    quarter of B's unique categories.
    """
    cats = sorted(df[b_col].dropna().unique().tolist(), key=str)
    n_cats = len(cats)
    if n_cats == 0:
        return
    quarter_size = max(1, n_cats // 4)
    start = int(rng.integers(0, n_cats))
    quarter = [cats[(start + k) % n_cats] for k in range(quarter_size)]

    for idx in transform_idx:
        if pd.notna(noisy.at[idx, b_col]):
            noisy.at[idx, b_col] = quarter[int(rng.integers(len(quarter)))]


# ===========================================================================
# Scenario 1 — Low Gaussian noise, all variables
# ===========================================================================

def scenario_fidelity_1(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "fidelity_1",
    random_seed: int = 42,
) -> List[str]:
    """
    Fidelity Scenario 1 — Low Gaussian noise applied to all variables.

    - **Numerical** : add N(0, std(col)) per cell.
    - **Ordinal**   : add N(0, std(col)), round to nearest observed integer.
    - **Categorical**: one-hot + N(0,1) per category → argmax (noisy flip).

    NaN cells are preserved as-is.
    Dataset *i* uses seed ``random_seed + i``.
    """
    return _apply_global_noise(
        df, n_datasets, output_dir, col_types, prefix, random_seed,
        num_scale=1.0, cat_mode="onehot",
    )


# ===========================================================================
# Scenario 2 — Low Gaussian noise, numerical/ordinal only
# ===========================================================================

def scenario_fidelity_2(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "fidelity_2",
    random_seed: int = 42,
) -> List[str]:
    """
    Fidelity Scenario 2 — Low Gaussian noise, numerical/ordinal only.

    Same as fidelity_1 for numerical/ordinal variables; categorical variables
    are left unchanged.

    - **Numerical** : add N(0, std(col)) per cell.
    - **Ordinal**   : add N(0, std(col)), round to nearest observed integer.
    - **Categorical**: unchanged.

    Dataset *i* uses seed ``random_seed + i``.
    """
    return _apply_global_noise(
        df, n_datasets, output_dir, col_types, prefix, random_seed,
        num_scale=1.0, cat_mode="none",
    )


# ===========================================================================
# Scenario 3 — High Gaussian noise, all variables
# ===========================================================================

def scenario_fidelity_3(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "fidelity_3",
    random_seed: int = 42,
) -> List[str]:
    """
    Fidelity Scenario 3 — High Gaussian noise applied to all variables.

    - **Numerical** : add N(0, 2·std(col)) per cell.
    - **Ordinal**   : add N(0, 2·std(col)), round to nearest observed integer.
    - **Categorical**: draw N(0,1) independently for all categories (no
                       one-hot base), assign argmax → effectively uniform
                       random reassignment.

    Dataset *i* uses seed ``random_seed + i``.
    """
    return _apply_global_noise(
        df, n_datasets, output_dir, col_types, prefix, random_seed,
        num_scale=2.0, cat_mode="random",
    )


# ===========================================================================
# Scenario 4 — High Gaussian noise, numerical/ordinal only
# ===========================================================================

def scenario_fidelity_4(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "fidelity_4",
    random_seed: int = 42,
) -> List[str]:
    """
    Fidelity Scenario 4 — High Gaussian noise, numerical/ordinal only.

    Same noise as fidelity_3 for numerical/ordinal; categorical unchanged.

    - **Numerical** : add N(0, 2·std(col)) per cell.
    - **Ordinal**   : add N(0, 2·std(col)), round to nearest observed integer.
    - **Categorical**: unchanged.

    Dataset *i* uses seed ``random_seed + i``.
    """
    return _apply_global_noise(
        df, n_datasets, output_dir, col_types, prefix, random_seed,
        num_scale=2.0, cat_mode="none",
    )


# ===========================================================================
# Shared engine for scenarios 1–4
# ===========================================================================

def _apply_global_noise(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str,
    random_seed: int,
    num_scale: float,
    cat_mode: str,  # "onehot" | "random" | "none"
) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for i in range(n_datasets):
        rng = np.random.default_rng(random_seed + i)
        noisy = df.copy()

        for col, ctype in col_types.items():
            if col not in df.columns:
                continue
            if ctype == "numerical":
                if _is_ordinal(df[col]):
                    noisy[col] = _add_ordinal_noise(df[col], rng, noise_scale=num_scale)
                else:
                    noisy[col] = _add_numerical_noise(df[col], rng, noise_scale=num_scale)
            elif ctype == "categorical":
                if cat_mode == "onehot":
                    noisy[col] = _add_categorical_noise_onehot(df[col], rng)
                elif cat_mode == "random":
                    noisy[col] = _add_categorical_noise_random(df[col], rng)
                # cat_mode == "none": leave unchanged

        out_path = output_dir / f"{prefix}_{i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))

    return paths


# ===========================================================================
# Scenario 5 — Structured bivariate noise
# ===========================================================================

def scenario_fidelity_5(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    col_types: ColumnTypes,
    prefix: str = "fidelity_5",
    random_seed: int = 42,
) -> List[str]:
    """
    Fidelity Scenario 5 — Structured bivariate noise.

    Each dataset applies one targeted perturbation defined by a
    (variable-pair, quartile) combination.  The conditioning variable A
    determines which rows are affected; the target variable B is perturbed
    within those rows.

    Iteration order across datasets
    --------------------------------
    Pairs are cycled first (inner loop), quartile index advances second
    (outer loop)::

        dataset 0   → pair 0,  quartile 0
        dataset 1   → pair 1,  quartile 0
        ...
        dataset P-1 → pair P-1, quartile 0
        dataset P   → pair 0,  quartile 1
        ...

    where P = number of unique column pairs (combinations of 2).

    Row selection (conditioning variable A)
    ----------------------------------------
    - **Numerical/ordinal A**: quartile 0–3 maps to
      ``[−∞, Q1)``, ``[Q1, Q2)``, ``[Q2, Q3)``, ``[Q3, +∞]``.
    - **Categorical A**: sorted unique categories split into 4 equal groups;
      quartile index selects which group's rows are targeted.

    Perturbation of 50 % of selected rows (target variable B)
    ----------------------------------------------------------
    - **B numerical/ordinal**:
        Compute the local mean of B within the 50 % selected rows.
        If local mean > global mean of B → add +|N(0, std(B))| (push higher).
        If local mean ≤ global mean of B → add −|N(0, std(B))| (push lower).
        Ordinal B is snapped to the nearest observed integer after noise.
    - **B categorical**:
        Pick a random quarter of B's unique categories; replace each selected
        cell with a random draw from that quarter.

    NaN cells in B are preserved regardless of selection.
    Each dataset *i* uses seed ``random_seed + i``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in col_types if c in df.columns]
    all_pairs = list(itertools.combinations(cols, 2))  # unordered pairs (A, B)
    n_pairs = len(all_pairs)

    if n_pairs == 0:
        raise ValueError("Need at least 2 columns to form variable pairs.")

    paths: List[str] = []

    for i in range(n_datasets):
        rng = np.random.default_rng(random_seed + i)

        # Determine which (pair, quartile) to use for this dataset
        pair_idx = i % n_pairs
        quartile_idx = (i // n_pairs) % 4

        a_col, b_col = all_pairs[pair_idx]
        a_type = col_types[a_col]
        b_type = col_types[b_col]

        noisy = df.copy()

        # --- Identify rows in the conditioning quartile of A ---
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

        # --- Select 50 % of those rows at random ---
        n_transform = max(1, len(rows_idx) // 2)
        chosen_positions = rng.choice(len(rows_idx), size=n_transform, replace=False)
        transform_idx = [rows_idx[j] for j in sorted(chosen_positions)]

        # --- Perturb B ---
        if b_type == "numerical":
            _perturb_b_numerical(
                noisy, b_col, transform_idx, df, rng,
                is_ordinal=_is_ordinal(df[b_col]),
            )
        else:
            _perturb_b_categorical(noisy, b_col, transform_idx, df, rng)

        out_path = output_dir / f"{prefix}_{i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))

    return paths


# ===========================================================================
# Registry — maps scenario name → function
# ===========================================================================

FIDELITY_SCENARIOS: Dict[str, callable] = {
    "fidelity_1": scenario_fidelity_1,
    "fidelity_2": scenario_fidelity_2,
    "fidelity_3": scenario_fidelity_3,
    "fidelity_4": scenario_fidelity_4,
    "fidelity_5": scenario_fidelity_5,
}
