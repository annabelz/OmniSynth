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
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from omnisynth.evaluation.fidelity import evaluate_fidelity
from omnisynth.evaluation.missingness import evaluate_missingness
from omnisynth.evaluation.scoring import (
    compute_fidelity_score,
    compute_missingness_score,
    compute_composite_score,
)
from omnisynth.config import EvalConfig, FidelityConfig
from omnisynth.meta_eval.config import MetaEvalConfig
from omnisynth.meta_eval.scenarios import SCENARIO_REGISTRY
from omnisynth.utils.data_utils import detect_column_types


def _build_eval_config(metrics: dict) -> EvalConfig:
    """Build an EvalConfig from a ``metrics`` dict (fidelity/missingness sub-dicts)."""
    fid_cfg = metrics.get("fidelity", {})
    fc = FidelityConfig(
        run_wasserstein=bool(fid_cfg.get("wasserstein", True)),
        run_tvd=bool(fid_cfg.get("tvd", True)),
        run_hellinger=bool(fid_cfg.get("hellinger", True)),
        run_spearman=bool(fid_cfg.get("spearman", True)),
        run_contingency=bool(fid_cfg.get("contingency", True)),
        run_pcd=bool(fid_cfg.get("pcd", True)),
        run_auc_roc=bool(fid_cfg.get("auc_roc", True)),
        run_propensity_mse=bool(fid_cfg.get("propensity_mse", True)),
        run_crcl_rs=bool(fid_cfg.get("crcl_rs", True)),
        run_crcl_sr=bool(fid_cfg.get("crcl_sr", True)),
    )
    return EvalConfig(fidelity=fc)


def _fidelity_group_flags(metrics: dict) -> dict:
    """Return run_univariate/bivariate/multivariate kwargs derived from the metrics dict."""
    fid = metrics.get("fidelity", {})
    return {
        "run_univariate": any(fid.get(k, True) for k in ("wasserstein", "tvd", "hellinger")),
        "run_bivariate":  any(fid.get(k, True) for k in ("spearman", "contingency", "pcd")),
        "run_multivariate": any(fid.get(k, True) for k in ("auc_roc", "propensity_mse", "crcl_rs", "crcl_sr")),
    }


def _missingness_flags(metrics: dict) -> dict:
    """Return run_* kwargs for evaluate_missingness derived from the metrics dict."""
    miss = metrics.get("missingness", {})
    return {
        "run_rate":                 bool(miss.get("rate", True)),
        "run_set_distribution":     bool(miss.get("set_distribution", True)),
        "run_missing_auroc":        bool(miss.get("missing_auroc", True)),
        "run_dependency_structure": bool(miss.get("dependency_structure", True)),
    }


def run_meta_eval(
    config: MetaEvalConfig,
    verbose: Optional[str] = None,
    skip_generation: bool = False,
    generate_only: bool = False,
    merge: bool = False,
    scenarios: Optional[List[str]] = None,
) -> Dict:
    """
    Run a full meta-evaluation as described in *config*.

    Parameters
    ----------
    config : MetaEvalConfig
    verbose : str, optional
        Override ``config.verbose``.  One of ``"none"``, ``"some"``, ``"all"``.

        - ``"none"`` — no output.
        - ``"some"`` — scenario banners, per-dataset score lines, checkpoints,
          and final summaries.
        - ``"all"``  — everything in ``"some"`` plus per-dataset generation
          progress and per-metric prints.
    skip_generation : bool
        If ``True``, skip dataset generation and evaluate pre-existing CSV files
        found in ``output_dir/<result_key>/``.  Raises if no files are found.
    generate_only : bool
        If ``True``, generate noisy datasets for each scenario but skip
        evaluation.  Returns an empty dict.
    merge : bool
        If ``True``, load any existing results from ``config.results_path``
        before running and merge new results into them.  Existing scenario keys
        are preserved; keys produced by this run overwrite any prior values for
        those keys.
    scenarios : list of str, optional
        If provided, only the named scenarios are run (others in the config are
        skipped).  Names must still exist in the SCENARIO_REGISTRY.

    Returns
    -------
    dict
        Nested results dict, one entry per scenario (empty if generate_only).
    """
    verbose = config.verbose if verbose is None else verbose
    if verbose not in ("none", "some", "all"):
        raise ValueError(f"verbose must be 'none', 'some', or 'all', got {verbose!r}")

    show_some = verbose in ("some", "all")
    show_all = verbose == "all"

    real = pd.read_csv(config.input_data)
    col_types = detect_column_types(real, override=config.column_types)

    # Build evaluation kwargs from optional metrics config
    _metrics = config.metrics or {}
    eval_config = _build_eval_config(_metrics) if _metrics else None
    fid_group_flags = _fidelity_group_flags(_metrics)
    miss_flags = _missingness_flags(_metrics)

    # Scoring weights (fall back to defaults when not specified)
    _weights = config.weights or {}
    w_fidelity = _weights.get("fidelity")
    w_missingness = _weights.get("missingness")
    w_composite = _weights.get("composite")

    run_fidelity = "fidelity" in config.axes
    run_missingness = "missingness" in config.axes

    # Seed all_results from the existing file when merging
    all_results: Dict = {}
    if merge and config.results_path:
        existing = Path(config.results_path)
        if existing.exists():
            with open(existing) as _f:
                all_results = json.load(_f)
            if show_some:
                print(f"  Loaded {len(all_results)} existing scenario(s) from {existing}")

    # Filter scenarios list if --scenarios was specified
    active_scenarios = config.scenarios
    if scenarios:
        active_scenarios = [s for s in config.scenarios if s.name in scenarios]
        if show_some:
            skipped = len(config.scenarios) - len(active_scenarios)
            print(f"  Running {len(active_scenarios)} scenario(s) "
                  f"(skipping {skipped} not in filter)")

    # When sample_sizes is not specified, use a single sentinel (None = full dataset)
    # and keep the old result key format ({scenario_name}) for backwards compatibility.
    use_sample_sizes = config.sample_sizes is not None
    effective_sample_sizes: List[Optional[int]] = config.sample_sizes if use_sample_sizes else [None]

    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    total_runs = len(active_scenarios) * len(effective_sample_sizes)

    for scenario_cfg in active_scenarios:
        name = scenario_cfg.name
        if name not in SCENARIO_REGISTRY:
            raise ValueError(
                f"Unknown scenario {name!r}. "
                f"Available: {sorted(SCENARIO_REGISTRY.keys())}"
            )
        scenario_fn = SCENARIO_REGISTRY[name]

        for sample_size in effective_sample_sizes:
            # Result key: plain name for backwards compat; suffixed when sample_sizes used
            if not use_sample_sizes:
                result_key = name
            else:
                size_tag = f"n{len(real)}" if sample_size is None else f"n{sample_size}"
                result_key = f"{name}_{size_tag}"

            size_str = f"n={len(real):,} (full)" if sample_size is None else f"n={sample_size:,}"

            # Detect already-completed replicates when resuming with --merge
            existing_per_dataset: List[Dict] = []
            existing_n = 0
            if merge and result_key in all_results:
                existing_per_dataset = all_results[result_key].get("per_dataset", [])
                existing_n = len(existing_per_dataset)
                if existing_n >= scenario_cfg.n_datasets:
                    if show_some:
                        print(f"\n  Skipping {result_key!r} — already complete "
                              f"({existing_n}/{scenario_cfg.n_datasets} replicates)")
                    continue
                if show_some:
                    print(f"\n  Resuming {result_key!r} — "
                          f"{existing_n}/{scenario_cfg.n_datasets} done, "
                          f"{scenario_cfg.n_datasets - existing_n} remaining")

            if show_some:
                print(f"\n{'='*60}")
                if use_sample_sizes:
                    print(f"Scenario: {name}  |  Sample size: {size_str}  ({scenario_cfg.n_datasets} replicates)")
                else:
                    print(f"Scenario: {name}  ({scenario_cfg.n_datasets} datasets)")
                print(f"{'='*60}")
                axes_str = ", ".join(config.axes)
                print(f"  Axes       : {axes_str}")
                if run_fidelity:
                    _fid = _metrics.get("fidelity", {})
                    _fid_names = [m for m in (
                        "wasserstein", "tvd", "hellinger",
                        "spearman", "contingency", "pcd",
                        "auc_roc", "propensity_mse", "crcl_rs", "crcl_sr",
                    ) if bool(_fid.get(m, True))]
                    print(f"  Fidelity   : {', '.join(_fid_names)}")
                if run_missingness:
                    _miss = _metrics.get("missingness", {})
                    _miss_names = [m for m in (
                        "rate", "set_distribution", "missing_auroc", "dependency_structure",
                    ) if bool(_miss.get(m, True))]
                    print(f"  Missingness: {', '.join(_miss_names)}")

            # ------------------------------------------------------------------
            # 1. Generate or discover noisy datasets
            # ------------------------------------------------------------------
            # Directory layout:
            #   output_dir/{name}/              — no sample_sizes in config
            #   output_dir/{name}/{size_tag}/   — sample_sizes configured (incl. null → n{len(real)})
            if not use_sample_sizes:
                scenario_dir = Path(config.output_dir) / name
            else:
                scenario_dir = Path(config.output_dir) / name / size_tag

            if skip_generation:
                paths = sorted(str(p) for p in scenario_dir.glob("*.csv"))
                if not paths:
                    raise FileNotFoundError(
                        f"--eval-only specified but no CSV files found in {scenario_dir}"
                    )
                if use_sample_sizes and sample_size is not None:
                    # Reconstruct the per-replicate samples using the same seeds used
                    # during generation, so each noisy file is evaluated against its
                    # original sample rather than the full dataset.
                    eval_pairs: List[tuple] = []
                    for i, path in enumerate(paths):
                        n = min(sample_size, len(real))
                        real_sample = real.sample(n=n, random_state=config.random_seed + i).reset_index(drop=True)
                        eval_pairs.append((path, real_sample))
                else:
                    eval_pairs = [(p, real) for p in paths]
                if show_some:
                    print(f"  Found {len(paths)} existing datasets in {scenario_dir}")
            elif not use_sample_sizes or sample_size is None:
                # Bulk generation: generate remaining replicates from full real data
                n_remaining = scenario_cfg.n_datasets - existing_n
                paths = scenario_fn(
                    df=real,
                    n_datasets=n_remaining,
                    output_dir=scenario_dir,
                    col_types=col_types,
                    prefix=name,
                    random_seed=config.random_seed + existing_n,
                    file_offset=existing_n,
                    **scenario_cfg.params,
                )
                # Each replicate evaluates against the full real dataset
                eval_pairs = [(p, real) for p in paths]
                if show_some:
                    print(f"  Generated {len(paths)} datasets in {scenario_dir}")
            else:
                # Per-replicate sampling: draw a fresh random sample for each replicate,
                # generate one noisy dataset into the shared size directory.
                eval_pairs = []
                for i in range(existing_n, scenario_cfg.n_datasets):
                    n = min(sample_size, len(real))
                    real_sample = real.sample(n=n, random_state=config.random_seed + i).reset_index(drop=True)
                    rep_paths = scenario_fn(
                        df=real_sample,
                        n_datasets=1,
                        output_dir=scenario_dir,
                        col_types=col_types,
                        prefix=name,
                        random_seed=config.random_seed + i,
                        file_offset=i,
                        **scenario_cfg.params,
                    )
                    eval_pairs.append((rep_paths[0], real_sample))
                if show_some:
                    print(f"  Generated {len(eval_pairs)} replicates ({size_str} each) in {scenario_dir}")

            if generate_only:
                continue

            # ------------------------------------------------------------------
            # 2. Evaluate each dataset
            # ------------------------------------------------------------------
            per_dataset: List[Dict] = []
            fid_score_lists: Dict[str, List[float]] = {}
            miss_score_lists: Dict[str, List[float]] = {}
            composite_list: List[float] = []

            n_total_pairs = len(eval_pairs)
            for i, (path, real_ref) in enumerate(eval_pairs):
                synth = pd.read_csv(path)
                row: Dict = {"path": str(path), "sample_size": sample_size}

                if show_all:
                    print(f"\n  [{i+1:>{len(str(n_total_pairs))}}/{n_total_pairs}] {Path(path).name}", flush=True)

                t0 = time.monotonic()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if run_fidelity:
                        fid = evaluate_fidelity(
                            real_ref, synth, col_types=col_types,
                            config=eval_config, verbose=show_all,
                            **fid_group_flags,
                        )
                        f_scores = compute_fidelity_score(fid, weights=w_fidelity)
                    else:
                        f_scores = {}

                    if run_missingness:
                        miss = evaluate_missingness(
                            real_ref, synth, col_types=col_types,
                            config=eval_config, verbose=show_all,
                            **miss_flags,
                        )
                        m_scores = compute_missingness_score(miss, weights=w_missingness)
                    else:
                        m_scores = {}

                    comp = compute_composite_score(f_scores, m_scores, weights=w_composite) if (f_scores or m_scores) else {}
                elapsed = time.monotonic() - t0

                # Save overall fidelity score
                if isinstance(f_scores.get("overall"), float):
                    fid_score_lists.setdefault("overall", []).append(f_scores["overall"])
                    row["fidelity_overall"] = f_scores["overall"]
                # Save individual metric scores from raw fidelity results
                if run_fidelity:
                    for group_metrics in fid.values():
                        for metric_name, metric_result in group_metrics.items():
                            score = float(metric_result.score)
                            fid_score_lists.setdefault(metric_name, []).append(score)
                            row[f"fidelity_{metric_name}"] = score

                for k, v in m_scores.items():
                    if isinstance(v, float):
                        miss_score_lists.setdefault(k, []).append(v)
                        row[f"missingness_{k}"] = v

                if comp.get("composite") is not None:
                    composite_list.append(comp["composite"])
                    row["composite_score"] = comp["composite"]

                row["elapsed_s"] = round(elapsed, 2)
                per_dataset.append(row)

                if show_some:
                    parts = []
                    if "overall" in f_scores:
                        parts.append(f"fidelity={f_scores['overall']:.4f}")
                    if "overall" in m_scores:
                        parts.append(f"missingness={m_scores['overall']:.4f}")
                    if comp.get("composite") is not None:
                        parts.append(f"composite={comp['composite']:.4f}")
                    parts.append(f"time={elapsed:.1f}s")
                    print(f"  [{i+1:>{len(str(n_total_pairs))}}/{n_total_pairs}] → {', '.join(parts)}", flush=True)

            # ------------------------------------------------------------------
            # 3. Aggregate — combine with any existing replicates when resuming
            # ------------------------------------------------------------------
            combined_per_dataset = existing_per_dataset + per_dataset

            # Rebuild score lists from all replicates so aggregated stats are correct
            all_fid_lists: Dict[str, List[float]] = {}
            all_miss_lists: Dict[str, List[float]] = {}
            all_composite: List[float] = []
            for row_data in combined_per_dataset:
                for k, v in row_data.items():
                    if k.startswith("fidelity_") and isinstance(v, (int, float)):
                        all_fid_lists.setdefault(k[len("fidelity_"):], []).append(float(v))
                    elif k.startswith("missingness_") and isinstance(v, (int, float)):
                        all_miss_lists.setdefault(k[len("missingness_"):], []).append(float(v))
                    elif k == "composite_score" and isinstance(v, (int, float)):
                        all_composite.append(float(v))

            scenario_result: Dict = {
                "n_datasets": len(combined_per_dataset),
                "sample_size": sample_size,
                "per_dataset": combined_per_dataset,
            }

            if all_fid_lists:
                scenario_result["fidelity"] = {k: _stats(v) for k, v in all_fid_lists.items()}

            if all_miss_lists:
                scenario_result["missingness"] = {k: _stats(v) for k, v in all_miss_lists.items()}

            if all_composite:
                scenario_result["composite"] = _stats(all_composite)

            all_results[result_key] = scenario_result

            if config.results_path:
                save_meta_eval_results(all_results, config.results_path, merge=False)

            if show_some:
                n_done = len(all_results)
                print(f"\n  ✓ Checkpoint: {n_done}/{total_runs} runs complete "
                      f"— results written to {config.results_path}")
                print(f"\n  Summary for {result_key}:")
                if "fidelity" in scenario_result:
                    ov = scenario_result["fidelity"].get("overall") or {}
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


def save_meta_eval_results(results: Dict, path: str | Path, merge: bool = False) -> None:
    """Write meta-evaluation results to a JSON file.

    Parameters
    ----------
    results:
        New results to write.
    path:
        Destination JSON file.
    merge:
        If ``True`` and *path* already exists, load the existing file and
        update it with *results* (new keys overwrite old ones) before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.exists():
        with open(path) as _f:
            existing = json.load(_f)
        existing.update(results)
        results = existing
    path.write_text(json.dumps(results, indent=2))
