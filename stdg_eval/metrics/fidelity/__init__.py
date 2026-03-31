from stdg_eval.metrics.fidelity.univariate import WassersteinDistance, TotalVariationDistance
from stdg_eval.metrics.fidelity.bivariate import SpearmanCorrelation, ContingencyMatrix
from stdg_eval.metrics.fidelity.multivariate import CrossClassification, PropensityMSE

__all__ = [
    "WassersteinDistance",
    "TotalVariationDistance",
    "SpearmanCorrelation",
    "ContingencyMatrix",
    "CrossClassification",
    "PropensityMSE",
]
