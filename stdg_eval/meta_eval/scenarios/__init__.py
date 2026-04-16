"""
Scenario registry for meta-evaluation.

Maps scenario name strings (as used in the meta-eval config) to the
corresponding generator functions.
"""

from stdg_eval.meta_eval.scenarios.fidelity import FIDELITY_SCENARIOS
from stdg_eval.meta_eval.scenarios.missingness import MISSINGNESS_SCENARIOS

SCENARIO_REGISTRY: dict = {
    **FIDELITY_SCENARIOS,
    **MISSINGNESS_SCENARIOS,
}

__all__ = ["SCENARIO_REGISTRY", "FIDELITY_SCENARIOS", "MISSINGNESS_SCENARIOS"]
