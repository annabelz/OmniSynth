"""
Meta-evaluation runner.

Orchestrates the full meta-evaluation pipeline:
  1. For each scenario in the config, generate noisy datasets.
  2. Run the requested evaluation axes (fidelity, missingness) on each dataset.
  3. Aggregate scores: mean ± std across the n_datasets replicates.
  4. Return (and optionally write) a structured results dict.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stdg_eval.evaluation.fidelity import evaluate_fidelity
from stdg_eval.evaluation.missingness import evaluate_missingness
from stdg_eval.evaluation.scoring import (
    compute_fidelity_score,
    compute_missingness_score,
    compute_composite_score,
)
from stdg_eval.meta_eval.config import MetaEvalConfig
from stdg_eval.meta_eval.scenarios import SCENARIO_REGISTRY
from stdg_eval.utils.data_utils import detect_column_types


def run_meta_eval(config: MetaEvalConfig, verbose: bool = True) -> Dict:
    """
    Run a full meta-evaluation as described in *config*.

    Parameters
    ----------
    config : MetaEvalConfig
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Nested results dict, one entry per scenario::

            {
              "fidelity_1": {
                "n_datasets": 20,
                "fidelity": {
                  "overall": {"mean": 0.83, "std": 0.02},
                  "univariate": {"mean": ..., "std": ...},
                  ...
                },
                "missingness": {
                  "overall": {"mean": ..., "std": ...},
                  ...
                },
                "composite": {"mean": ..., "std": ...},
                "per_dataset": [         # one entry per replicate
                  {"fidelity_score": ..., "missingness_score": ..., "composite_score": ...},
                  ...
                ],
              },
              ...
            }
    """
    real = pd.read_csv(config.input_data)
    col_types = detect_column_types(real, override=config.column_types)

    run_fidelity = "fidelity" in config.axes
    run_missingness = "missingness" in config.axes

    all_results: Dict = {}

    for scenario_cfg in config.scenarios:
        name = scenario_cfg.name
        if name not in SCENARIO_REGISTRY:
            raise ValueError(
                f"Unknown scenario {name!r}. "
                f"Available: {sorted(SCENARIO_REGISTRY.keys())}"
            )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {name}  ({scenario_cfg.n_datasets} datasets)")
            print(f"{'='*60}")

        # ------------------------------------------------------------------
        # 1. Generate noisy datasets
        # ------------------------------------------------------------------
        scenario_dir = Path(config.output_dir) / name
        scenario_fn = SCENARIO_REGISTRY[name]
        paths = scenario_fn(
            df=real,
            n_datasets=scenario_cfg.n_datasets,
            output_dir=scenario_dir,
            col_types=col_types,
            prefix=name,
            random_seed=config.random_seed,
            **scenario_cfg.params,
        )

        if verbose:
            print(f"  Generated {len(paths)} datasets in {scenario_dir}")

        # ------------------------------------------------------------------
        # 2. Evaluate each dataset
        # ------------------------------------------------------------------
        per_dataset: List[Dict] = []
        fid_score_lists: Dict[str, List[float]] = {}   # key → list of scores
        miss_score_lists: Dict[str, List[float]] = {}
        composite_list: List[float] = []

        for i, path in enumerate(paths):
            synth = pd.read_csv(path)
            row: Dict = {"path": path}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if run_fidelity:
                    fid = evaluate_fidelity(real, synth, col_types=col_types, verbose=False)
                    f_scores = compute_fidelity_score(fid)
                else:
                    f_scores = {}

                if run_missingness:
                    miss = evaluate_missingness(real, synth, col_types=col_types, verbose=False)
                    m_scores = compute_missingness_score(miss)
                else:
                    m_scores = {}

                comp = compute_composite_score(f_scores, m_scores) if (f_scores or m_scores) else {}

            for k, v in f_scores.items():
                if v is not None:
                    fid_score_lists.setdefault(k, []).append(v)
                    row[f"fidelity_{k}"] = v

            for k, v in m_scores.items():
                if v is not None:
                    miss_score_lists.setdefault(k, []).append(v)
                    row[f"missingness_{k}"] = v

            if comp.get("composite") is not None:
                composite_list.append(comp["composite"])
                row["composite_score"] = comp["composite"]

            per_dataset.append(row)

            if verbose:
                parts = []
                if "overall" in f_scores:
                    parts.append(f"fidelity={f_scores['overall']:.4f}")
                if "overall" in m_scores:
                    parts.append(f"missingness={m_scores['overall']:.4f}")
                if comp.get("composite") is not None:
                    parts.append(f"composite={comp['composite']:.4f}")
                print(f"  [{i+1:>{len(str(len(paths)))}}/{len(paths)}] {Path(path).name}  {', '.join(parts)}")

        # ------------------------------------------------------------------
        # 3. Aggregate
        # ------------------------------------------------------------------
        def _stats(values: List[float]) -> Dict[str, float]:
            arr = np.array(values)
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        scenario_result: Dict = {
            "n_datasets": len(paths),
            "per_dataset": per_dataset,
        }

        if fid_score_lists:
            scenario_result["fidelity"] = {k: _stats(v) for k, v in fid_score_lists.items()}

        if miss_score_lists:
            scenario_result["missingness"] = {k: _stats(v) for k, v in miss_score_lists.items()}

        if composite_list:
            scenario_result["composite"] = _stats(composite_list)

        all_results[name] = scenario_result

        if verbose:
            print(f"\n  Summary for {name}:")
            if "fidelity" in scenario_result:
                ov = scenario_result["fidelity"].get("overall", {})
                print(f"    fidelity overall  mean={ov.get('mean', float('nan')):.4f}  "
                      f"std={ov.get('std', float('nan')):.4f}")
            if "missingness" in scenario_result:
                ov = scenario_result["missingness"].get("overall", {})
                print(f"    missingness overall  mean={ov.get('mean', float('nan')):.4f}  "
                      f"std={ov.get('std', float('nan')):.4f}")
            if "composite" in scenario_result:
                ov = scenario_result["composite"]
                print(f"    composite  mean={ov['mean']:.4f}  std={ov['std']:.4f}")

    return all_results


def save_meta_eval_results(results: Dict, path: str | Path) -> None:
    """Write meta-evaluation results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
