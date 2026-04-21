"""
Composite scenarios for meta-evaluation.

Each composite scenario combines one fidelity transform and one missingness
transform, applied sequentially: fidelity noise is added first, then
missingness is layered on top of the already-perturbed dataset.

The 25 combinations are named ``composite_f{F}_m{M}`` where F ∈ {1..5} and
M ∈ {1..5}, e.g. ``composite_f1_m1`` (low Gaussian noise + 10 % MCAR).

Fidelity transforms
-------------------
f1 — Low Gaussian noise, all variables         (num_scale=1, cat=one-hot)
f2 — Low Gaussian noise, numerical only        (num_scale=1, cat=unchanged)
f3 — High Gaussian noise, all variables        (num_scale=2, cat=random)
f4 — High Gaussian noise, numerical only       (num_scale=2, cat=unchanged)
f5 — Structured bivariate noise

Missingness transforms
----------------------
m1 — 10 % MCAR
m2 — 20 % MCAR
m3 — 30 % MCAR
m4 — MAR bivariate conditioning
m5 — MNAR self-conditioning
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes
from stdg_eval.meta_eval.scenarios.base import generate_datasets, TransformFn
from stdg_eval.meta_eval.scenarios.fidelity import (
    _global_noise_transform,
    _bivariate_noise_transform,
)
from stdg_eval.meta_eval.scenarios.missingness import (
    _mcar_transform,
    _mar_bivariate_transform,
    _mnar_self_transform,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _chain(fidelity_fn: TransformFn, missingness_fn: TransformFn) -> TransformFn:
    """Return a transform that applies *fidelity_fn* then *missingness_fn*."""
    def transform(df: pd.DataFrame, rng, col_types: ColumnTypes, i: int) -> pd.DataFrame:
        noisy = fidelity_fn(df, rng, col_types, i)
        return missingness_fn(noisy, rng, col_types, i)
    return transform


def _fidelity_transform(f_idx: int, df: pd.DataFrame, col_types: ColumnTypes) -> TransformFn:
    """Build the fidelity TransformFn for scenario index *f_idx* (1–5)."""
    if f_idx == 1:
        return _global_noise_transform(num_scale=1.0, cat_mode="onehot")
    elif f_idx == 2:
        return _global_noise_transform(num_scale=1.0, cat_mode="none")
    elif f_idx == 3:
        return _global_noise_transform(num_scale=2.0, cat_mode="random")
    elif f_idx == 4:
        return _global_noise_transform(num_scale=2.0, cat_mode="none")
    elif f_idx == 5:
        cols = [c for c in col_types if c in df.columns]
        all_pairs = list(itertools.combinations(cols, 2))
        if not all_pairs:
            raise ValueError("Need at least 2 columns to form variable pairs (fidelity_5).")
        return _bivariate_noise_transform(all_pairs, col_types)
    raise ValueError(f"Unknown fidelity index {f_idx}")


def _missingness_transform(m_idx: int, df: pd.DataFrame, col_types: ColumnTypes) -> TransformFn:
    """Build the missingness TransformFn for scenario index *m_idx* (1–5)."""
    if m_idx == 1:
        return _mcar_transform(0.10)
    elif m_idx == 2:
        return _mcar_transform(0.20)
    elif m_idx == 3:
        return _mcar_transform(0.30)
    elif m_idx == 4:
        cols = [c for c in col_types if c in df.columns]
        all_pairs = [(a, b) for a, b in itertools.permutations(cols, 2)]
        if not all_pairs:
            raise ValueError("Need at least 2 columns to form variable pairs (missingness_4).")
        return _mar_bivariate_transform(all_pairs)
    elif m_idx == 5:
        cols = [c for c in col_types if c in df.columns]
        if not cols:
            raise ValueError("No columns available (missingness_5).")
        return _mnar_self_transform(cols)
    raise ValueError(f"Unknown missingness index {m_idx}")


# ===========================================================================
# Public scenario function factory
# ===========================================================================

def _make_composite_scenario(f_idx: int, m_idx: int):
    """Return a scenario function for the (f_idx, m_idx) combination."""
    name = f"composite_f{f_idx}_m{m_idx}"

    def scenario_fn(
        df: pd.DataFrame,
        n_datasets: int,
        output_dir,
        col_types: ColumnTypes,
        prefix: str = name,
        random_seed: int = 42,
        verbose: bool = False,
        file_offset: int = 0,
    ) -> List[str]:
        f"""Composite scenario {name}: fidelity_{f_idx} then missingness_{m_idx}."""
        transform = _chain(
            _fidelity_transform(f_idx, df, col_types),
            _missingness_transform(m_idx, df, col_types),
        )
        return generate_datasets(
            transform, df, n_datasets, output_dir, prefix, random_seed, col_types,
            verbose=verbose, file_offset=file_offset,
        )

    scenario_fn.__name__ = f"scenario_{name}"
    scenario_fn.__qualname__ = f"scenario_{name}"
    return scenario_fn


# ===========================================================================
# Registry  (all 25 combinations)
# ===========================================================================

COMPOSITE_SCENARIOS: Dict[str, callable] = {
    f"composite_f{f}_m{m}": _make_composite_scenario(f, m)
    for f in range(1, 6)
    for m in range(1, 6)
}
