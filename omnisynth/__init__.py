"""
OmniSynth: A modular evaluation library for tabular synthetic data generation.

Evaluation axes (current and planned):
  - Fidelity    : How closely the synthetic data matches the statistical properties of real data.
  - Missingness : How faithfully missing-data patterns are reproduced.
  - Utility     : TODO — downstream task performance on synthetic vs real data.
  - Privacy     : TODO — disclosure risk / membership inference assessments.
"""

from omnisynth.evaluation.fidelity import evaluate_fidelity
from omnisynth.evaluation.missingness import evaluate_missingness
from omnisynth.evaluation.scoring import (
    compute_composite_score,
    compute_fidelity_score,
    compute_missingness_score,
)

__version__ = "0.1.0"
__all__ = [
    "evaluate_fidelity",
    "evaluate_missingness",
    "compute_fidelity_score",
    "compute_missingness_score",
    "compute_composite_score",
]
