"""
Fidelity scenarios for meta-evaluation.

Each public ``scenario_fidelity_*`` function builds a *transform* — a
closure that generates one noisy variant of the input DataFrame — and
delegates the generation loop, timing, and CSV writing to
:func:`~stdg_eval.meta_eval.scenarios.base.generate_datasets`.

Available scenarios
-------------------
fidelity_1 — Low Gaussian noise, all variables
    numerical/ordinal: N(0, 1·std); categorical: one-hot + N(0,1) → argmax.

fidelity_2 — Low Gaussian noise, numerical/ordinal only
    Same as fidelity_1 for numerical/ordinal; categorical unchanged.

fidelity_3 — High Gaussian noise, all variables
    numerical/ordinal: N(0, 2·std); categorical: pure N(0,1) → argmax (random
    reassignment).

fidelity_4 — High Gaussian noise, numerical/ordinal only
    Same as fidelity_3 for numerical/ordinal; categorical unchanged.

fidelity_5 — Structured bivariate noise
    One (variable-pair, quartile) perturbation per dataset; cycles all pairs ×
    4 quartiles across datasets.
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
# Low-level column-transformation helpers
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
    """Add N(0, noise_scale · std(col)) noise; NaN cells preserved."""
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
    """Add Gaussian noise then snap to nearest observed integer."""
    observed_ints = np.sort(np.unique(series.dropna().values.astype(int)))
    noisy = _add_numerical_noise(series, rng, noise_scale=noise_scale)
    return _snap_to_observed_ints(noisy, observed_ints)


def _add_categorical_noise_onehot(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    """One-hot + N(0,1) per category → argmax (moderate flip probability)."""
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


def _add_categorical_noise_random(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Empirical-proportion base + N(0,1) per category → argmax (random reassignment)."""
    categories = sorted(series.dropna().unique().tolist())
    if len(categories) <= 1:
        return series.copy()
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


def _perturb_b_numerical(
    noisy: pd.DataFrame,
    b_col: str,
    transform_idx: List,
    df: pd.DataFrame,
    rng: np.random.Generator,
    is_ordinal: bool,
) -> None:
    """Perturb B (numerical/ordinal) in-place: push toward or away from global mean."""
    std_b = float(df[b_col].std(ddof=1))
    if np.isnan(std_b) or std_b == 0.0:
        return
    global_mean = float(df[b_col].mean())
    valid_idx = [i for i in transform_idx if pd.notna(noisy.at[i, b_col])]
    if not valid_idx:
        return
    noisy[b_col] = noisy[b_col].astype(float)
    local_mean = float(noisy.loc[valid_idx, b_col].mean())
    sign = 1.0 if local_mean > global_mean else -1.0
    noise = sign * np.abs(rng.normal(0.0, std_b, len(valid_idx)))
    for j, idx in enumerate(valid_idx):
        noisy.at[idx, b_col] = noisy.at[idx, b_col] + noise[j]
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
    """Replace selected B values with a draw from a random quarter of B's categories."""
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
# Transform factories
# ===========================================================================

def _global_noise_transform(num_scale: float, cat_mode: str):
    """Return a transform that applies Gaussian noise to all columns."""
    def transform(df: pd.DataFrame, rng: np.random.Generator, col_types: ColumnTypes, i: int) -> pd.DataFrame:
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
        return noisy
    return transform


def _bivariate_noise_transform(all_pairs: list, col_types: ColumnTypes):
    """Return a transform that applies one structured (pair, quartile) perturbation."""
    def transform(df: pd.DataFrame, rng: np.random.Generator, col_types: ColumnTypes, i: int) -> pd.DataFrame:
        noisy = df.copy()
        pair_idx = i % len(all_pairs)
        quartile_idx = (i // len(all_pairs)) % 4
        a_col, b_col = all_pairs[pair_idx]
        a_type = col_types[a_col]
        b_type = col_types[b_col]

        if a_type == "numerical":
            mask = numerical_quartile_mask(df[a_col], quartile_idx)
        else:
            mask = categorical_quartile_mask(df[a_col], quartile_idx)

        rows_idx = df.index[mask].tolist()
        if not rows_idx:
            return noisy

        n_transform = max(1, len(rows_idx) // 2)
        chosen_positions = rng.choice(len(rows_idx), size=n_transform, replace=False)
        transform_idx = [rows_idx[j] for j in sorted(chosen_positions)]

        if b_type == "numerical":
            _perturb_b_numerical(noisy, b_col, transform_idx, df, rng, is_ordinal=_is_ordinal(df[b_col]))
        else:
            _perturb_b_categorical(noisy, b_col, transform_idx, df, rng)
        return noisy
    return transform


# ===========================================================================
# Public scenario functions
# ===========================================================================

def scenario_fidelity_1(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "fidelity_1",
    random_seed: int = 42,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """Fidelity Scenario 1 — Low Gaussian noise, all variables."""
    return generate_datasets(
        _global_noise_transform(num_scale=1.0, cat_mode="onehot"),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
        file_offset=file_offset,
    )


def scenario_fidelity_2(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "fidelity_2",
    random_seed: int = 42,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """Fidelity Scenario 2 — Low Gaussian noise, numerical/ordinal only."""
    return generate_datasets(
        _global_noise_transform(num_scale=1.0, cat_mode="none"),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
        file_offset=file_offset,
    )


def scenario_fidelity_3(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "fidelity_3",
    random_seed: int = 42,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """Fidelity Scenario 3 — High Gaussian noise, all variables."""
    return generate_datasets(
        _global_noise_transform(num_scale=2.0, cat_mode="random"),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
        file_offset=file_offset,
    )


def scenario_fidelity_4(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "fidelity_4",
    random_seed: int = 42,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """Fidelity Scenario 4 — High Gaussian noise, numerical/ordinal only."""
    return generate_datasets(
        _global_noise_transform(num_scale=2.0, cat_mode="none"),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
        file_offset=file_offset,
    )


def scenario_fidelity_5(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir,
    col_types: ColumnTypes,
    prefix: str = "fidelity_5",
    random_seed: int = 42,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """Fidelity Scenario 5 — Structured bivariate noise (one pair×quartile per dataset)."""
    cols = [c for c in col_types if c in df.columns]
    all_pairs = list(itertools.combinations(cols, 2))
    if not all_pairs:
        raise ValueError("Need at least 2 columns to form variable pairs.")
    return generate_datasets(
        _bivariate_noise_transform(all_pairs, col_types),
        df, n_datasets, output_dir, prefix, random_seed, col_types, verbose=verbose,
        file_offset=file_offset,
    )


# ===========================================================================
# Registry
# ===========================================================================

FIDELITY_SCENARIOS: Dict[str, callable] = {
    "fidelity_1": scenario_fidelity_1,
    "fidelity_2": scenario_fidelity_2,
    "fidelity_3": scenario_fidelity_3,
    "fidelity_4": scenario_fidelity_4,
    "fidelity_5": scenario_fidelity_5,
}
