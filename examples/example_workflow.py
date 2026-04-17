"""
Example workflow for stdg-eval.

This script demonstrates the full programmatic API without the dashboard.
It creates a suite of toy datasets covering a range of fidelity and missingness
scenarios, runs all metrics, and prints a ranked comparison.

Dataset legend
--------------
  real          — ground truth (missingness in bmi, sbp, smoker)
  synth_ideal   — exact copy of real (upper bound; expected score ≈ 1.0)
  synth_fid1    — slight distribution shifts, similar missingness to real
  synth_miss1   — real distributions, 15 % missingness in different columns
  synth_fid1_miss1 — same slight shifts as fid1 (different draw) + 15 % missingness
  synth_fid2    — larger distribution shifts, similar missingness to real
  synth_miss2   — real distributions, 30 % missingness across more columns
  synth_fid2_miss2 — same larger shifts as fid2 (different draw) + 30 % missingness

Run:
    python examples/example_workflow.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

from stdg_eval.evaluation.fidelity import evaluate_fidelity
from stdg_eval.evaluation.missingness import evaluate_missingness
from stdg_eval.evaluation.scoring import (
    compute_composite_score,
    compute_fidelity_score,
    compute_missingness_score,
)
from stdg_eval.utils.data_utils import detect_column_types


# ---------------------------------------------------------------------------
# 1. Real dataset
#    Missingness: bmi ~12 %, sbp ~8 %, smoker ~20 %
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
N = 500

real_data = pd.DataFrame({
    "age":       rng.integers(18, 85, N).astype(float),
    "bmi":       rng.normal(26, 4, N),
    "sbp":       rng.normal(120, 15, N),
    "sex":       rng.choice(["M", "F"], N),
    "diagnosis": rng.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2]),
    "smoker":    rng.choice([0, 1], N, p=[0.7, 0.3]).astype(float),
})
for col, rate in [("bmi", 0.12), ("sbp", 0.08), ("smoker", 0.20)]:
    mask = rng.random(N) < rate
    real_data.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 2. synth_ideal — exact copy of real (perfect score baseline)
# ---------------------------------------------------------------------------

synth_ideal = real_data.copy()

# ---------------------------------------------------------------------------
# 3. synth_fid1 — slight distribution shifts, similar missingness to real
#    Expected: high fidelity score, high missingness score
# ---------------------------------------------------------------------------

rng_f1 = np.random.default_rng(100)
synth_fid1 = pd.DataFrame({
    "age":       rng_f1.integers(19, 84, N).astype(float),
    "bmi":       rng_f1.normal(26.3, 4.1, N),
    "sbp":       rng_f1.normal(120.5, 14.8, N),
    "sex":       rng_f1.choice(["M", "F"], N, p=[0.495, 0.505]),
    "diagnosis": rng_f1.choice(["A", "B", "C"], N, p=[0.49, 0.30, 0.21]),
    "smoker":    rng_f1.choice([0, 1], N, p=[0.71, 0.29]).astype(float),
})
for col, rate in [("bmi", 0.11), ("sbp", 0.09), ("smoker", 0.21)]:
    mask = rng_f1.random(N) < rate
    synth_fid1.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 4. synth_miss1 — real distributions, 15 % missingness in different columns
#    (age and diagnosis instead of bmi / sbp / smoker)
#    Expected: high fidelity score, lower missingness score
# ---------------------------------------------------------------------------

rng_m1 = np.random.default_rng(200)
synth_miss1 = pd.DataFrame({
    "age":       rng_m1.integers(18, 85, N).astype(float),
    "bmi":       rng_m1.normal(26, 4, N),
    "sbp":       rng_m1.normal(120, 15, N),
    "sex":       rng_m1.choice(["M", "F"], N),
    "diagnosis": rng_m1.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2]),
    "smoker":    rng_m1.choice([0, 1], N, p=[0.7, 0.3]).astype(float),
})
for col, rate in [("age", 0.15), ("diagnosis", 0.15)]:
    mask = rng_m1.random(N) < rate
    synth_miss1.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 5. synth_fid1_miss1 — same slight shifts as fid1 (different draw)
#    + 15 % missingness in sbp and smoker (different columns/rows from miss1)
#    Expected: moderate fidelity score, lower missingness score
# ---------------------------------------------------------------------------

rng_f1m1 = np.random.default_rng(300)
synth_fid1_miss1 = pd.DataFrame({
    "age":       rng_f1m1.integers(19, 84, N).astype(float),
    "bmi":       rng_f1m1.normal(26.3, 4.1, N),
    "sbp":       rng_f1m1.normal(120.5, 14.8, N),
    "sex":       rng_f1m1.choice(["M", "F"], N, p=[0.495, 0.505]),
    "diagnosis": rng_f1m1.choice(["A", "B", "C"], N, p=[0.49, 0.30, 0.21]),
    "smoker":    rng_f1m1.choice([0, 1], N, p=[0.71, 0.29]).astype(float),
})
for col, rate in [("sbp", 0.15), ("smoker", 0.15)]:
    mask = rng_f1m1.random(N) < rate
    synth_fid1_miss1.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 6. synth_fid2 — larger distribution shifts, similar missingness to real
#    Expected: lower fidelity score, high missingness score
# ---------------------------------------------------------------------------

rng_f2 = np.random.default_rng(400)
synth_fid2 = pd.DataFrame({
    "age":       rng_f2.integers(25, 70, N).astype(float),        # narrower range
    "bmi":       rng_f2.normal(29, 6, N),                         # higher mean, more spread
    "sbp":       rng_f2.normal(132, 20, N),                       # shifted up
    "sex":       rng_f2.choice(["M", "F"], N, p=[0.65, 0.35]),    # imbalanced
    "diagnosis": rng_f2.choice(["A", "B", "C"], N, p=[0.33, 0.33, 0.34]),  # near-uniform
    "smoker":    rng_f2.choice([0, 1], N, p=[0.5, 0.5]).astype(float),
})
for col, rate in [("bmi", 0.13), ("sbp", 0.07), ("smoker", 0.19)]:
    mask = rng_f2.random(N) < rate
    synth_fid2.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 7. synth_miss2 — real distributions, 30 % missingness across more columns
#    (bmi, sbp, smoker — same columns as real but much higher rates)
#    Expected: high fidelity score, lower missingness score
# ---------------------------------------------------------------------------

rng_m2 = np.random.default_rng(500)
synth_miss2 = pd.DataFrame({
    "age":       rng_m2.integers(18, 85, N).astype(float),
    "bmi":       rng_m2.normal(26, 4, N),
    "sbp":       rng_m2.normal(120, 15, N),
    "sex":       rng_m2.choice(["M", "F"], N),
    "diagnosis": rng_m2.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2]),
    "smoker":    rng_m2.choice([0, 1], N, p=[0.7, 0.3]).astype(float),
})
for col, rate in [("bmi", 0.30), ("sbp", 0.30), ("smoker", 0.30)]:
    mask = rng_m2.random(N) < rate
    synth_miss2.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 8. synth_fid2_miss2 — same larger shifts as fid2 (different draw)
#    + 30 % missingness in different variables (age, sbp, diagnosis)
#    Expected: lower fidelity score, lower missingness score
# ---------------------------------------------------------------------------

rng_f2m2 = np.random.default_rng(600)
synth_fid2_miss2 = pd.DataFrame({
    "age":       rng_f2m2.integers(25, 70, N).astype(float),
    "bmi":       rng_f2m2.normal(29, 6, N),
    "sbp":       rng_f2m2.normal(132, 20, N),
    "sex":       rng_f2m2.choice(["M", "F"], N, p=[0.65, 0.35]),
    "diagnosis": rng_f2m2.choice(["A", "B", "C"], N, p=[0.33, 0.33, 0.34]),
    "smoker":    rng_f2m2.choice([0, 1], N, p=[0.5, 0.5]).astype(float),
})
for col, rate in [("age", 0.30), ("sbp", 0.30), ("diagnosis", 0.30)]:
    mask = rng_f2m2.random(N) < rate
    synth_fid2_miss2.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 9. Save datasets and write example config
# ---------------------------------------------------------------------------

datasets = {
    "synth_ideal":      synth_ideal,
    "synth_fid1":       synth_fid1,
    "synth_miss1":      synth_miss1,
    "synth_fid1_miss1": synth_fid1_miss1,
    "synth_fid2":       synth_fid2,
    "synth_miss2":      synth_miss2,
    "synth_fid2_miss2": synth_fid2_miss2,
}

data_dir = pathlib.Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

real_data.to_csv(data_dir / "real.csv", index=False)
for name, df in datasets.items():
    df.to_csv(data_dir / f"{name}.csv", index=False)

precomputed_path = pathlib.Path(__file__).parent / "precomputed.json"
meta_eval_results_path = pathlib.Path(__file__).parent / "meta_eval" / "results.json"

config = {
    "real_data": str(data_dir / "real.csv"),
    "synthetic_datasets": [
        {"name": name, "path": str(data_dir / f"{name}.csv")}
        for name in datasets
    ],
    "column_types": {
        "age": "numerical",
        "bmi": "numerical",
        "sbp": "numerical",
        "sex": "categorical",
        "diagnosis": "categorical",
        "smoker": "categorical",
    },
    "precomputed_results": str(precomputed_path),
    "meta_eval_results": str(meta_eval_results_path),
}

config_path = pathlib.Path(__file__).parent / "example_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Saved datasets to  {data_dir}/")
print(f"Saved config to    {config_path}")
print(f"Launch dashboard:  streamlit run run_dashboard.py -- --config {config_path}")
print()

# ---------------------------------------------------------------------------
# 10. Detect column types
# ---------------------------------------------------------------------------

col_types = detect_column_types(real_data)
print("Detected column types:")
for col, ctype in col_types.items():
    print(f"  {col}: {ctype}")
print()

# ---------------------------------------------------------------------------
# 11. Evaluate all datasets
# ---------------------------------------------------------------------------

fidelity_weights  = [0.34, 0.33, 0.33]
miss_weights      = [0.25, 0.25, 0.25, 0.25]
composite_weights = [0.5, 0.5]

results = {}
for name, df in datasets.items():
    fid  = evaluate_fidelity(real_data, df, col_types=col_types)
    miss = evaluate_missingness(real_data, df, col_types=col_types)
    f_scores = compute_fidelity_score(fid, weights=fidelity_weights)
    m_scores = compute_missingness_score(miss, weights=miss_weights)
    comp     = compute_composite_score(f_scores, m_scores, weights=composite_weights)
    results[name] = {
        "fidelity":    f_scores["overall"],
        "missingness": m_scores["overall"],
        "composite":   comp["composite"],
    }

# ---------------------------------------------------------------------------
# 12. Summary table
# ---------------------------------------------------------------------------

print("=" * 65)
print(f"{'Dataset':<20}  {'Fidelity':>9}  {'Missingness':>11}  {'Composite':>9}")
print("-" * 65)
for name, s in results.items():
    print(f"  {name:<18}  {s['fidelity']:>9.4f}  {s['missingness']:>11.4f}  {s['composite']:>9.4f}")
print("=" * 65)

# ---------------------------------------------------------------------------
# 13. Ranking by composite score
# ---------------------------------------------------------------------------

print()
ranked = sorted(results.items(), key=lambda x: -x[1]["composite"])
print("Ranking (composite score):")
for rank, (name, s) in enumerate(ranked, 1):
    print(f"  #{rank}  {name:<20}  composite={s['composite']:.4f}")

print()
print("Done. Launch the dashboard for interactive exploration:")
print("  streamlit run run_dashboard.py")

# ---------------------------------------------------------------------------
# 14. Write meta-evaluation config
# ---------------------------------------------------------------------------

meta_eval_dir = pathlib.Path(__file__).parent / "meta_eval"
meta_eval_noisy_dir = meta_eval_dir / "noisy"
meta_eval_results_path = meta_eval_dir / "results.json"

meta_eval_config = {
    "input_data": str(data_dir / "real.csv"),
    "output_dir": str(meta_eval_noisy_dir),
    "results_path": str(meta_eval_results_path),
    "scenarios": [
        {"name": "fidelity_1", "n_datasets": 10},
        {"name": "fidelity_2", "n_datasets": 10},
        {"name": "fidelity_3", "n_datasets": 10},
        {"name": "fidelity_4", "n_datasets": 10},
        {"name": "fidelity_5", "n_datasets": 10},
        {"name": "missingness_1", "n_datasets": 10},
        {"name": "missingness_2", "n_datasets": 10},
        {"name": "missingness_3", "n_datasets": 10},
        {"name": "missingness_4", "n_datasets": 10},
        {"name": "missingness_5", "n_datasets": 10},
    ],
    "column_types": {
        "age": "numerical",
        "bmi": "numerical",
        "sbp": "numerical",
        "sex": "categorical",
        "diagnosis": "categorical",
        "smoker": "categorical",
    },
    "axes": ["fidelity", "missingness"],
    "random_seed": 42,
}

meta_eval_config_path = pathlib.Path(__file__).parent / "meta_eval_config.yaml"
with open(meta_eval_config_path, "w") as f:
    yaml.dump(meta_eval_config, f, default_flow_style=False, sort_keys=False)

print(f"Saved meta-eval config to  {meta_eval_config_path}")
print(f"Run meta-evaluation:       stdg-eval meta-eval --config {meta_eval_config_path}")
