"""
Baseline scenario for meta-evaluation.

baseline — Identity transform
    The output dataset is an exact copy of the input sample.  Evaluating
    real vs. real establishes the upper-bound ceiling for all metrics and
    confirms that perfect fidelity/missingness scores are recoverable.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from omnisynth.utils.data_utils import ColumnTypes
from omnisynth.meta_eval.scenarios.base import generate_datasets


def scenario_baseline(
    df: pd.DataFrame,
    n_datasets: int,
    output_dir: str,
    col_types: ColumnTypes,
    prefix: str,
    random_seed: int,
    file_offset: int = 0,
    **_kwargs,
) -> List[str]:
    """Generate *n_datasets* identity copies of *df* (no transformation)."""

    def _identity(
        df: pd.DataFrame,
        rng: np.random.Generator,
        col_types: ColumnTypes,
        dataset_idx: int,
    ) -> pd.DataFrame:
        return df.copy()

    return generate_datasets(
        transform_fn=_identity,
        df=df,
        n_datasets=n_datasets,
        output_dir=output_dir,
        prefix=prefix,
        random_seed=random_seed,
        col_types=col_types,
        file_offset=file_offset,
    )


BASELINE_SCENARIOS: Dict[str, object] = {
    "baseline": scenario_baseline,
}
