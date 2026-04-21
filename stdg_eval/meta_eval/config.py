"""
Meta-evaluation configuration.

Example YAML
------------
input_data: data/real.csv

output_dir: meta_eval/noisy/       # where generated noisy datasets are written

results_path: meta_eval/results.json

scenarios:
  - name: fidelity_1
    n_datasets: 20
  # - name: fidelity_2         # add more scenarios as they are implemented
  #   n_datasets: 10

column_types:                   # optional — auto-inferred if omitted
  age: numerical
  sex: categorical

axes:                           # optional — defaults to [fidelity, missingness]
  - fidelity
  - missingness

random_seed: 42                 # optional — default 42
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ScenarioConfig:
    """Configuration for one scenario."""
    name: str
    """Scenario identifier, e.g. ``"fidelity_1"``.  Must be in SCENARIO_REGISTRY."""
    n_datasets: int = 10
    """Number of noisy datasets to generate for this scenario."""
    params: Dict = field(default_factory=dict)
    """Optional scenario-specific parameter overrides passed as kwargs."""


@dataclass
class MetaEvalConfig:
    """Top-level meta-evaluation configuration."""
    input_data: str
    """Path to the input (real) dataset CSV."""
    output_dir: str
    """Directory where generated noisy datasets are written."""
    results_path: str
    """Path where the meta-evaluation JSON results are written."""
    scenarios: List[ScenarioConfig]
    """List of scenarios to run."""
    column_types: Optional[Dict[str, str]] = None
    """Column type overrides.  Auto-inferred if None."""
    axes: List[str] = field(default_factory=lambda: ["fidelity", "missingness"])
    """Evaluation axes to run (``"fidelity"`` and/or ``"missingness"``)."""
    random_seed: int = 42
    """Base random seed passed to all scenario functions."""
    sample_sizes: Optional[List[Optional[int]]] = None
    """Sample sizes to draw from the real dataset per replicate.
    Each entry is an integer row count, or ``None`` for the full dataset.
    When specified, each replicate independently draws a random sample of the
    given size, generates one noisy dataset from that sample, and evaluates
    against the same sample.  Results are keyed as
    ``{scenario_name}_n{size}`` (e.g. ``fidelity_1_n100``) or
    ``{scenario_name}_full`` for the full-dataset entry.
    When omitted, the full dataset is used and keys are just
    ``{scenario_name}`` (existing behaviour).
    """
    metrics: Optional[Dict[str, Dict[str, bool]]] = None
    """Per-axis metric enable/disable flags.

    Example::

        metrics:
          fidelity:
            wasserstein: true
            tvd: true
            hellinger: true
            spearman: true
            contingency: true
            pcd: true
            auc_roc: true
            propensity_mse: true
            crcl_rs: false
            crcl_sr: false
          missingness:
            rate: true
            set_distribution: true
            missing_auroc: true
            dependency_structure: true

    Omitted keys default to ``True``.  Omitting the ``metrics`` block entirely
    runs all metrics.
    """
    verbose: str = "some"
    """Verbosity level: ``"none"`` | ``"some"`` | ``"all"``.

    - ``"none"`` — no output.
    - ``"some"`` — scenario banners, per-dataset score lines, checkpoints,
      and final summaries.
    - ``"all"``  — everything in ``"some"`` plus per-dataset generation
      progress and per-metric prints.
    """


def load_meta_eval_config(path: str | Path) -> MetaEvalConfig:
    """Parse a YAML file into a :class:`MetaEvalConfig`."""
    path = Path(path).resolve()
    config_dir = path.parent

    def _resolve(p: str) -> str:
        """Resolve a path relative to the config file's directory."""
        return str((config_dir / p).resolve())

    with open(path) as f:
        raw = yaml.safe_load(f)

    scenarios = [
        ScenarioConfig(
            name=s["name"],
            n_datasets=int(s.get("n_datasets", 10)),
            params=s.get("params", {}),
        )
        for s in raw.get("scenarios", [])
    ]

    raw_sizes = raw.get("sample_sizes")
    if raw_sizes is not None:
        sample_sizes = [None if (s is None or str(s).lower() == "full") else int(s) for s in raw_sizes]
    else:
        sample_sizes = None

    return MetaEvalConfig(
        input_data=_resolve(raw["input_data"]),
        output_dir=_resolve(raw["output_dir"]),
        results_path=_resolve(raw["results_path"]),
        scenarios=scenarios,
        column_types=raw.get("column_types") or None,
        axes=raw.get("axes", ["fidelity", "missingness"]),
        random_seed=int(raw.get("random_seed", 42)),
        sample_sizes=sample_sizes,
        metrics=raw.get("metrics") or None,
        verbose=str(raw.get("verbose", "some")),
    )
