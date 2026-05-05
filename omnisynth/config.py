"""
Default configuration for OmniSynth.

Column type inference, metric enable/disable flags, and scoring defaults are
all centralised here so they can be overridden without touching library code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

# TODO: validate this decision with ariel?
# ---------------------------------------------------------------------------
# Column-type inference
# ---------------------------------------------------------------------------
# Columns with fewer unique values than this threshold (as a fraction of rows)
# are treated as categorical when the dtype is numeric.
CATEGORICAL_CARDINALITY_THRESHOLD: float = 0.05
# If the total number of unique values is also ≤ this hard cap, treat as categorical.
CATEGORICAL_MAX_UNIQUE: int = 20

# ---------------------------------------------------------------------------
# Fidelity scoring defaults
# ---------------------------------------------------------------------------
# Default weights: [univariate, bivariate, multivariate]
# Must sum to 1.0.
DEFAULT_FIDELITY_WEIGHTS: List[float] = [0.34, 0.33, 0.33]

# Default weights within univariate group: [numerical (wasserstein), categorical (tvd)]
# Auto-adjusted by actual column counts at runtime when set to None.
DEFAULT_UNIVARIATE_WEIGHTS: Optional[Dict[str, float]] = None

# ---------------------------------------------------------------------------
# Missingness scoring defaults
# ---------------------------------------------------------------------------
# Default weights: [rate, set_distribution, missing_auroc, dependency_structure]
# Must sum to 1.0.
DEFAULT_MISSINGNESS_WEIGHTS: List[float] = [0.25, 0.25, 0.25, 0.25]

# ---------------------------------------------------------------------------
# Composite score defaults
# ---------------------------------------------------------------------------
# Default weights: [fidelity, missingness]
# TODO: extend to [fidelity, missingness, utility, privacy] once those axes are
#       implemented.
DEFAULT_COMPOSITE_WEIGHTS: List[float] = [0.5, 0.5]

# ---------------------------------------------------------------------------
# Metric-specific settings
# ---------------------------------------------------------------------------

@dataclass
class FidelityConfig:
    """Knobs for individual fidelity metrics."""

    # Univariate — metric enable flags
    run_wasserstein: bool = True
    run_tvd: bool = True
    run_hellinger: bool = True

    # Univariate — metric settings
    wasserstein_n_bins: int = 100  # used only for plotting; SciPy computes exact WD
    tvd_normalize: bool = True     # normalise frequency distributions before TVD

    # Bivariate — metric enable flags
    run_spearman: bool = True
    run_contingency: bool = True
    run_pcd: bool = True

    # Bivariate — metric settings
    spearman_method: Literal["spearman", "pearson"] = "spearman"
    contingency_max_categories: int = 30  # skip cols with more unique values

    # Multivariate — metric enable flags
    run_auc_roc: bool = True
    run_propensity_mse: bool = True
    run_crcl_rs: bool = True
    run_crcl_sr: bool = True

    # Multivariate — metric settings
    auc_roc_n_estimators: int = 100
    auc_roc_cv_folds: int = 5
    auc_roc_impute: bool = False  # default: complete case analysis

    crcl_test_size: float = 0.3
    crcl_max_depth: int = None
    crcl_impute: bool = False
    propensity_mse_model: Literal["logistic", "rf"] = "logistic"
    propensity_mse_n_estimators: int = 100  # only used when model == "rf"
    propensity_mse_max_iter: int = 1000     # only used when model == "logistic"


@dataclass
class MissingnessConfig:
    """Knobs for individual missingness metrics."""

    # Metric enable flags
    run_rate: bool = True
    run_set_distribution: bool = True
    run_missing_auroc: bool = True
    run_dependency_structure: bool = True

    # Metric settings
    classifier_model: Literal["logistic", "rf"] = "logistic"
    classifier_max_iter: int = 500
    classifier_n_estimators: int = 100  # only used when model == "rf"
    min_missing_rate: float = 0.001    # skip columns with less missingness than this
    dependency_method: Literal["pearson", "spearman"] = "pearson"


@dataclass
class EvalConfig:
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    random_state: int = 42


# Module-level default instance – importable directly.
DEFAULT_CONFIG = EvalConfig()