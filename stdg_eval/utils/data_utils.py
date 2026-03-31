"""
Data loading and preprocessing utilities.

Key responsibilities:
  - Load CSV / Parquet / pickle datasets.
  - Infer or accept user-provided column type assignments (numerical vs categorical).
  - Align real and synthetic datasets to a shared column schema.
  - Load YAML / plain-text run configurations.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from stdg_eval.config import (
    CATEGORICAL_CARDINALITY_THRESHOLD,
    CATEGORICAL_MAX_UNIQUE,
)

ColumnTypes = Dict[str, str]  # column name -> "numerical" | "categorical"


def load_dataset(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a tabular dataset from a CSV, Parquet, or pickle file."""
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file format: {suffix!r}. Use CSV, Parquet, or pickle.")


def detect_column_types(
    df: pd.DataFrame,
    override: Optional[ColumnTypes] = None,
) -> ColumnTypes:
    """
    Infer column types (``"numerical"`` or ``"categorical"``) from a DataFrame.

    Inference rules (in order):
    1. If an override is provided for a column, it is used as-is.
    2. Object / string / boolean columns → categorical.
    3. Numeric columns with ≤ CATEGORICAL_MAX_UNIQUE distinct values AND
       a cardinality fraction ≤ CATEGORICAL_CARDINALITY_THRESHOLD → categorical.
    4. All remaining numeric columns → numerical.

    Parameters
    ----------
    df:
        Input DataFrame.
    override:
        Mapping ``{column_name: "numerical"|"categorical"}`` that takes precedence
        over the automatic inference for the listed columns.

    Returns
    -------
    ColumnTypes
        Mapping from each column name to its type string.
    """
    override = override or {}
    col_types: ColumnTypes = {}
    n_rows = len(df)

    for col in df.columns:
        if col in override:
            col_types[col] = override[col]
            continue

        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        if dtype == "bool" or dtype == "object" or isinstance(dtype, pd.CategoricalDtype):
            col_types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(dtype):
            frac = n_unique / max(n_rows, 1)
            if n_unique <= CATEGORICAL_MAX_UNIQUE and frac <= CATEGORICAL_CARDINALITY_THRESHOLD:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numerical"
        else:
            # Datetime, timedelta, etc. — treat as numerical for distance computations
            col_types[col] = "numerical"

    return col_types


def align_columns(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col_types: Optional[ColumnTypes] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTypes]:
    """
    Ensure real and synthetic share the same column set and infer column types.

    Columns present in one DataFrame but not the other are dropped with a warning.

    Parameters
    ----------
    real, synthetic:
        DataFrames to align.
    col_types:
        Optional pre-computed column type mapping. If not provided, types are
        inferred from *real*.

    Returns
    -------
    (real_aligned, synthetic_aligned, col_types)
    """
    shared = [c for c in real.columns if c in synthetic.columns]
    dropped_real = [c for c in real.columns if c not in synthetic.columns]
    dropped_synth = [c for c in synthetic.columns if c not in real.columns]

    if dropped_real:
        import warnings
        warnings.warn(
            f"Columns in real but not synthetic — dropped: {dropped_real}", stacklevel=2
        )
    if dropped_synth:
        import warnings
        warnings.warn(
            f"Columns in synthetic but not real — dropped: {dropped_synth}", stacklevel=2
        )

    real_a = real[shared].copy()
    synth_a = synthetic[shared].copy()

    if col_types is None:
        col_types = detect_column_types(real_a)

    # Only keep types for shared columns
    col_types = {c: col_types.get(c, "numerical") for c in shared}

    return real_a, synth_a, col_types


def get_numerical_columns(col_types: ColumnTypes) -> List[str]:
    return [c for c, t in col_types.items() if t == "numerical"]


def get_categorical_columns(col_types: ColumnTypes) -> List[str]:
    return [c for c, t in col_types.items() if t == "categorical"]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict:
    """
    Load a run configuration from a YAML file.

    Expected top-level keys
    -----------------------
    real_data : str
        Path to the real dataset CSV.
    synthetic_datasets : list of {name: str, path: str}
        One entry per synthetic dataset.
    column_types : dict, optional
        ``{column_name: "numerical"|"categorical"}`` overrides.

    Example
    -------
    .. code-block:: yaml

        real_data: data/real.csv
        synthetic_datasets:
          - name: synth1
            path: data/synth1.csv
          - name: synth2
            path: data/synth2.csv
        column_types:
          age: numerical
          sex: categorical
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with open(path) as fh:
            return yaml.safe_load(fh)
    if suffix == ".txt":
        # Plain-text format: first line = real path, remaining lines = synth paths
        lines = path.read_text().splitlines()
        lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        return {
            "real_data": lines[0],
            "synthetic_datasets": [
                {"name": f"synth{i + 1}", "path": p} for i, p in enumerate(lines[1:])
            ],
        }
    raise ValueError(f"Unsupported config format: {suffix!r}. Use YAML or .txt.")
