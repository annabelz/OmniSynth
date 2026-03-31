"""
Abstract base class for all evaluation metrics.

Adding a new metric
-------------------
1. Subclass ``BaseMetric``.
2. Implement ``evaluate(real, synthetic, col_types)`` returning a ``MetricResult``.
3. Optionally override ``name`` and ``description`` class attributes.
4. Register the metric in the relevant evaluation module
   (``evaluation/fidelity.py`` or ``evaluation/missingness.py``).

MetricResult contract
---------------------
``score``
    A float in **[0, 1]** where **1 = perfect fidelity** (real ≡ synthetic for
    this metric) and **0 = worst possible**.  Every metric *must* produce a
    normalised score so that weighted aggregation is meaningful.
``details``
    Arbitrary dict of raw/intermediate values (distances, matrices, per-column
    breakdowns, …).  Used for plotting and debugging; not used in scoring.
``column_scores``
    Optional per-column scores (also in [0, 1]).  Useful for univariate metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from stdg_eval.utils.data_utils import ColumnTypes


@dataclass
class MetricResult:
    """Container returned by every metric's ``evaluate`` method."""

    metric_name: str
    score: float  # normalised [0, 1]; 1 = perfect
    details: Dict[str, Any] = field(default_factory=dict)
    column_scores: Optional[Dict[str, float]] = None  # per-column scores when applicable

    def __repr__(self) -> str:  # noqa: D105
        return f"MetricResult(metric={self.metric_name!r}, score={self.score:.4f})"


class BaseMetric(ABC):
    """Abstract base for all stdg-eval metrics."""

    #: Human-readable name used in reports and dashboard labels.
    name: str = "BaseMetric"
    #: One-sentence description shown in the dashboard tooltip.
    description: str = ""
    #: Evaluation axis this metric belongs to.
    axis: str = "fidelity"  # "fidelity" | "missingness" | "utility" | "privacy"

    @abstractmethod
    def evaluate(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        col_types: ColumnTypes,
    ) -> MetricResult:
        """
        Compare *real* and *synthetic* DataFrames and return a ``MetricResult``.

        Parameters
        ----------
        real:
            Ground-truth dataset (full or subset).
        synthetic:
            Synthetic dataset to evaluate.
        col_types:
            Mapping ``{column_name: "numerical"|"categorical"}``.

        Returns
        -------
        MetricResult
            Normalised score in [0, 1] plus raw details.
        """
        ...
