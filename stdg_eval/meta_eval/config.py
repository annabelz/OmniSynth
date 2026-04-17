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
from typing import Dict, List, Optional

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

    return MetaEvalConfig(
        input_data=raw["input_data"],
        output_dir=raw["output_dir"],
        results_path=raw["results_path"],
        scenarios=scenarios,
        column_types=raw.get("column_types") or None,
        axes=raw.get("axes", ["fidelity", "missingness"]),
        random_seed=int(raw.get("random_seed", 42)),
        verbose=str(raw.get("verbose", "some")),
    )
