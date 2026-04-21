"""
Shared generation engine for meta-evaluation scenarios.

All scenario functions delegate to :func:`generate_datasets`, which owns the
loop, timing, progress printing, and CSV writing.  Each scenario only needs to
supply a *transform function* with signature::

    transform_fn(df, rng, col_types, dataset_idx) -> pd.DataFrame

The returned DataFrame is the noisy variant; ``df`` itself is never mutated.

Helper masks
------------
:func:`numerical_quartile_mask` and :func:`categorical_quartile_mask` are
shared by fidelity and missingness scenarios and live here to avoid duplication.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes

TransformFn = Callable[
    [pd.DataFrame, np.random.Generator, ColumnTypes, int],
    pd.DataFrame,
]


def generate_datasets(
    transform_fn: TransformFn,
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str | Path,
    prefix: str,
    random_seed: int,
    col_types: ColumnTypes,
    verbose: bool = False,
    file_offset: int = 0,
) -> List[str]:
    """
    Generate *n_datasets* noisy variants of *df* and write each to *output_dir*.

    Parameters
    ----------
    transform_fn:
        ``(df, rng, col_types, dataset_idx) -> pd.DataFrame``
        Must return a new DataFrame; *df* is passed read-only.
    df:
        Original (real) dataset — never mutated.
    n_datasets:
        Number of datasets to generate.
    output_dir:
        Directory to write CSVs into (created if absent).
    prefix:
        Filename prefix; files are named ``{prefix}_{file_offset+i:03d}.csv``.
    random_seed:
        Dataset *i* uses seed ``random_seed + i``.
    col_types:
        Column type mapping forwarded to *transform_fn*.
    verbose:
        Print per-dataset progress and timing.
    file_offset:
        Starting index for output filenames.  Replicate *i* is written as
        ``{prefix}_{file_offset + i:03d}.csv``.  Allows multiple single-dataset
        calls to share one output directory without overwriting each other.

    Returns
    -------
    List[str]
        Absolute paths of the written CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    for i in range(n_datasets):
        if verbose:
            print(f"  Generating dataset {i + 1}/{n_datasets}...", flush=True)
        t0 = time.time()
        rng = np.random.default_rng(random_seed + i)
        noisy = transform_fn(df, rng, col_types, file_offset + i)
        out_path = output_dir / f"{prefix}_{file_offset + i:03d}.csv"
        noisy.to_csv(out_path, index=False)
        paths.append(str(out_path.resolve()))
        if verbose:
            print(f"    done in {time.time() - t0:.1f}s", flush=True)

    return paths


def numerical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
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


def categorical_quartile_mask(series: pd.Series, quartile_idx: int) -> pd.Series:
    """Boolean mask: rows whose category belongs to the quartile_idx-th quarter."""
    cats = sorted(series.dropna().unique().tolist(), key=str)
    n = len(cats)
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    start = sum(sizes[:quartile_idx])
    group = set(cats[start: start + sizes[quartile_idx]])
    return series.isin(group)
