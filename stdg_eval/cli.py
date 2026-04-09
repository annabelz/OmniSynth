"""
CLI entry point for stdg-eval.

Commands
--------
stdg-eval dashboard [--config PATH] [--port PORT]
    Launch the Streamlit interactive dashboard.

stdg-eval evaluate --real PATH --synth PATH [PATH ...] [--output PATH]
    Run evaluation headlessly and write results to a JSON file.

stdg-eval precompute --config PATH --output PATH [--groups bivariate multivariate]
    Run only the expensive bivariate / multivariate fidelity metrics and save
    results to a JSON file that the dashboard can load directly, skipping
    recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict


def _cmd_dashboard(args):
    import subprocess

    dashboard_py = Path(__file__).parent.parent / "run_dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_py),
           "--server.port", str(args.port)]
    if args.config:
        # Pass config path via environment variable (picked up by dashboard.py)
        env = os.environ.copy()
        env["STDG_EVAL_CONFIG"] = str(args.config)
        subprocess.run(cmd, env=env)
    else:
        subprocess.run(cmd)


def _cmd_evaluate(args):
    import pandas as pd
    from stdg_eval.evaluation.fidelity import evaluate_fidelity
    from stdg_eval.evaluation.missingness import evaluate_missingness
    from stdg_eval.evaluation.scoring import (
        compute_fidelity_score,
        compute_missingness_score,
        compute_composite_score,
    )
    from stdg_eval.utils.data_utils import load_config

    if args.config:
        cfg = load_config(args.config)
        real = pd.read_csv(cfg["real_data"])
        synth_entries = [(e["name"], e["path"]) for e in cfg.get("synthetic_datasets", [])]
        col_types = cfg.get("column_types") or None
    else:
        if not args.real or not args.synth:
            print("Error: provide either --config or both --real and --synth.", file=sys.stderr)
            sys.exit(1)
        real = pd.read_csv(args.real)
        synth_entries = [(Path(p).stem, p) for p in args.synth]
        col_types = None

    results = {}

    for name, synth_path in synth_entries:
        synth = pd.read_csv(synth_path)

        print(f"\n[{name}]")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fid = evaluate_fidelity(real, synth, col_types=col_types, verbose=True)
            miss = evaluate_missingness(real, synth, col_types=col_types, verbose=True)

        f_scores = compute_fidelity_score(fid)
        m_scores = compute_missingness_score(miss)
        comp = compute_composite_score(f_scores, m_scores)
        elapsed = time.time() - t0

        results[name] = {
            "fidelity_score": f_scores["overall"],
            "missingness_score": m_scores["overall"],
            "composite_score": comp["composite"],
            "group_scores": {
                "univariate": f_scores.get("univariate"),
                "bivariate": f_scores.get("bivariate"),
                "multivariate": f_scores.get("multivariate"),
            },
        }
        print(f"[{name}]  fidelity={f_scores['overall']:.4f}  "
              f"missingness={m_scores['overall']:.4f}  "
              f"composite={comp['composite']:.4f}  "
              f"| time elapsed: {elapsed:.1f}s")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out}")


def _cmd_precompute(args):
    import pandas as pd
    from stdg_eval.evaluation.fidelity import evaluate_fidelity
    from stdg_eval.evaluation.missingness import evaluate_missingness
    from stdg_eval.utils.data_utils import load_config, eval_config_from_dict
    from stdg_eval.utils.precomputed_io import save_precomputed

    cfg = load_config(args.config)
    real = pd.read_csv(cfg["real_data"])
    synth_entries = [(e["name"], e["path"]) for e in cfg.get("synthetic_datasets", [])]
    col_types = cfg.get("column_types") or None

    groups = tuple(args.groups)
    run_uni = "univariate" in groups
    run_bi = "bivariate" in groups
    run_multi = "multivariate" in groups
    run_miss = "missingness" in groups

    eval_cfg = eval_config_from_dict(cfg) if "metrics" in cfg else None

    all_results = {}
    n = len(synth_entries)
    for i, (name, synth_path) in enumerate(synth_entries):
        print(f"\n[{i + 1}/{n}] {name}", flush=True)
        synth = pd.read_csv(synth_path)

        combined: Dict = {}
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if run_uni or run_bi or run_multi:
                fid = evaluate_fidelity(
                    real, synth,
                    col_types=col_types,
                    config=eval_cfg,
                    run_univariate=run_uni,
                    run_bivariate=run_bi,
                    run_multivariate=run_multi,
                    verbose=True,
                )
                combined.update(fid)
            if run_miss:
                miss = evaluate_missingness(
                    real, synth,
                    col_types=col_types,
                    config=eval_cfg,
                    verbose=True,
                )
                combined["missingness"] = miss

        elapsed = time.time() - t0
        all_results[name] = combined
        score_parts = []
        for group in groups:
            if group not in combined:
                continue
            scores = [mr.score for mr in combined[group].values()]
            if scores:
                score_parts.append(f"{group}={sum(scores)/len(scores):.4f}")
        print(f"  → {', '.join(score_parts)}  | time elapsed: {elapsed:.1f}s")

    out = Path(args.output)
    save_precomputed(all_results, out, groups=groups)
    print(f"\nPrecomputed results saved to {out}")
    print("Reference in your config with:")
    print(f"  precomputed_results: {out}")


def main():
    parser = argparse.ArgumentParser(
        prog="stdg-eval",
        description="Evaluate tabular synthetic data fidelity and missingness.",
    )
    sub = parser.add_subparsers(dest="command")

    # dashboard sub-command
    dash_p = sub.add_parser("dashboard", help="Launch the interactive dashboard.")
    dash_p.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML or .txt config file.")
    dash_p.add_argument("--port", type=int, default=8501,
                        help="Port for the Streamlit server (default: 8501).")

    # evaluate sub-command
    eval_p = sub.add_parser("evaluate", help="Headless evaluation — outputs JSON scores.")
    eval_p.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file (real_data, synthetic_datasets, column_types).")
    eval_p.add_argument("--real", type=Path, default=None, help="Path to real dataset CSV.")
    eval_p.add_argument("--synth", type=Path, nargs="+", default=None,
                        help="Path(s) to synthetic dataset CSV(s).")
    eval_p.add_argument("--output", type=Path, default=None,
                        help="Where to write JSON results (optional).")

    # precompute sub-command
    pre_p = sub.add_parser(
        "precompute",
        help="Run expensive bivariate/multivariate metrics and save to JSON for dashboard reuse.",
    )
    pre_p.add_argument("--config", type=Path, required=True,
                       help="Path to a YAML config file.")
    pre_p.add_argument("--output", type=Path, required=True,
                       help="Where to write the precomputed JSON file.")
    pre_p.add_argument(
        "--groups", nargs="+",
        default=["univariate", "bivariate", "multivariate", "missingness"],
        help="Which metric groups to precompute (default: all). "
             "Currently supported: univariate, bivariate, multivariate, missingness.",
    )

    args = parser.parse_args()

    if args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "precompute":
        _cmd_precompute(args)
    else:
        parser.print_help()
