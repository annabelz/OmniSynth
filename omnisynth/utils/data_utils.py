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

from omnisynth.config import (
    CATEGORICAL_CARDINALITY_THRESHOLD,
    CATEGORICAL_MAX_UNIQUE,
    EvalConfig,
    FidelityConfig,
    MissingnessConfig,
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

    validate_column_types(real_a, col_types, dataset_label="real")
    validate_column_types(synth_a, col_types, dataset_label="synthetic")

    return real_a, synth_a, col_types


def validate_column_types(
    df: pd.DataFrame,
    col_types: ColumnTypes,
    dataset_label: str = "dataset",
) -> None:
    """
    Warn if column values don't match their assigned types.

    - Numerical columns must contain numeric data; warns if a column assigned
      ``"numerical"`` has a non-numeric dtype (e.g. strings).
    - Categorical columns should not be high-cardinality numeric data; warns if
      a column assigned ``"categorical"`` looks like it should be numerical.
    """
    import warnings

    for col, ctype in col_types.items():
        if col not in df.columns:
            continue
        series = df[col]

        if ctype == "numerical":
            if not pd.api.types.is_numeric_dtype(series):
                warnings.warn(
                    f"[{dataset_label}] Column '{col}' is assigned type 'numerical' "
                    f"but contains non-numeric data (dtype: {series.dtype}). "
                    "This may cause errors in metric computation.",
                    UserWarning,
                    stacklevel=3,
                )
        elif ctype == "categorical":
            if pd.api.types.is_numeric_dtype(series):
                n_unique = series.nunique(dropna=True)
                n_rows = len(series)
                frac = n_unique / max(n_rows, 1)
                if n_unique > CATEGORICAL_MAX_UNIQUE or frac > CATEGORICAL_CARDINALITY_THRESHOLD:
                    warnings.warn(
                        f"[{dataset_label}] Column '{col}' is assigned type 'categorical' "
                        f"but contains numeric data with high cardinality "
                        f"({n_unique} unique values, {frac:.1%} of rows). "
                        "Consider assigning it as 'numerical' instead.",
                        UserWarning,
                        stacklevel=3,
                    )


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
    metrics : dict, optional
        Per-metric enable flags, e.g.::

            metrics:
              wasserstein: false
              tvd: false
              hellinger: true
              spearman: true
              contingency: true
              auc_roc: true
              propensity_mse: true
              rate: true
              set_distribution: true
              missing_auroc: true
              dependency_structure: true

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
        metrics:
          wasserstein: false
          hellinger: true
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


def eval_config_from_dict(cfg: dict) -> EvalConfig:
    """
    Build an :class:`~omnisynth.config.EvalConfig` from a loaded config dict.

    Reads the optional ``metrics`` key and maps it to ``FidelityConfig`` and
    ``MissingnessConfig`` enable flags.

    Accepts both formats:
    - Nested: ``metrics: {fidelity: {wasserstein: true, ...}, missingness: {...}}``
    - Flat (legacy): ``metrics: {wasserstein: true, tvd: false, ...}``
    """
    raw_metrics = cfg.get("metrics") or {}

    # Detect nested vs flat format
    if "fidelity" in raw_metrics or "missingness" in raw_metrics:
        fid_m = raw_metrics.get("fidelity") or {}
        miss_m = raw_metrics.get("missingness") or {}
    else:
        fid_m = raw_metrics
        miss_m = raw_metrics

    fidelity = FidelityConfig(
        run_wasserstein=bool(fid_m.get("wasserstein", False)),
        run_tvd=bool(fid_m.get("tvd", False)),
        run_hellinger=bool(fid_m.get("hellinger", False)),
        run_spearman=bool(fid_m.get("spearman", False)),
        run_contingency=bool(fid_m.get("contingency", False)),
        run_pcd=bool(fid_m.get("pcd", fid_m.get("pairwise_correlation_difference", False))),
        run_auc_roc=bool(fid_m.get("auc_roc", False)),
        run_propensity_mse=bool(fid_m.get("propensity_mse", False)),
        run_crcl_rs=bool(fid_m.get("crcl_rs", False)),
        run_crcl_sr=bool(fid_m.get("crcl_sr", False)),
    )
    missingness = MissingnessConfig(
        run_rate=bool(miss_m.get("rate", False)),
        run_set_distribution=bool(miss_m.get("set_distribution", False)),
        run_missing_auroc=bool(miss_m.get("missing_auroc", False)),
        run_dependency_structure=bool(miss_m.get("dependency_structure", False)),
    )
    return EvalConfig(fidelity=fidelity, missingness=missingness)


def weights_from_dict(cfg: dict) -> dict:
    """
    Extract scoring weights from a loaded config dict.

    Expected config structure::

        weights:
          fidelity: [0.34, 0.33, 0.33]      # [univariate, bivariate, multivariate]
          missingness: [0.25, 0.25, 0.25, 0.25]  # [rate, set_distribution, missing_auroc, dependency_structure]
          composite: [0.5, 0.5]             # [fidelity, missingness]

    Returns a dict with keys ``"fidelity"``, ``"missingness"``, ``"composite"``
    (each a list of floats, or None if not specified).
    """
    raw = cfg.get("weights") or {}
    return {
        "fidelity":    raw.get("fidelity"),
        "missingness": raw.get("missingness"),
        "composite":   raw.get("composite"),
    }
