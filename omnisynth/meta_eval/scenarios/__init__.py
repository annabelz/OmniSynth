"""
Scenario registry for meta-evaluation.

Maps scenario name strings (as used in the meta-eval config) to the
corresponding generator functions.

Adding a new scenario
---------------------
1. Write a transform factory ``_my_transform(...) -> TransformFn`` whose
   returned closure has signature
   ``(df, rng, col_types, dataset_idx) -> pd.DataFrame``.
2. Wrap it in a public ``scenario_*`` function that calls
   :func:`~omnisynth.meta_eval.scenarios.base.generate_datasets`.
3. Add the entry to the module's ``*_SCENARIOS`` dict and register it below.
"""

from omnisynth.meta_eval.scenarios.base import (
    generate_datasets,
    numerical_quartile_mask,
    categorical_quartile_mask,
    TransformFn,
)
from omnisynth.meta_eval.scenarios.baseline import BASELINE_SCENARIOS
from omnisynth.meta_eval.scenarios.fidelity import FIDELITY_SCENARIOS
from omnisynth.meta_eval.scenarios.missingness import MISSINGNESS_SCENARIOS
from omnisynth.meta_eval.scenarios.composite import COMPOSITE_SCENARIOS

SCENARIO_REGISTRY: dict = {
    **BASELINE_SCENARIOS,
    **FIDELITY_SCENARIOS,
    **MISSINGNESS_SCENARIOS,
    **COMPOSITE_SCENARIOS,
}

__all__ = [
    "SCENARIO_REGISTRY",
    "BASELINE_SCENARIOS",
    "FIDELITY_SCENARIOS",
    "MISSINGNESS_SCENARIOS",
    "COMPOSITE_SCENARIOS",
    "generate_datasets",
    "numerical_quartile_mask",
    "categorical_quartile_mask",
    "TransformFn",
]
