"""
Example workflow for stdg-eval.

This script demonstrates the full programmatic API without the dashboard.
It creates small synthetic datasets using numpy and runs all metrics.

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
# 1. Create toy datasets
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

# Introduce ~15% missingness in a few columns
for col, rate in [("bmi", 0.12), ("sbp", 0.08), ("smoker", 0.20)]:
    mask = rng.random(N) < rate
    real_data.loc[mask, col] = np.nan

# --- High-fidelity synthetic dataset (close to real) ---
synth1 = pd.DataFrame({
    "age":       rng.integers(20, 83, N).astype(float),
    "bmi":       rng.normal(26.5, 4.2, N),
    "sbp":       rng.normal(119, 14, N),
    "sex":       rng.choice(["M", "F"], N, p=[0.49, 0.51]),
    "diagnosis": rng.choice(["A", "B", "C"], N, p=[0.48, 0.31, 0.21]),
    "smoker":    rng.choice([0, 1], N, p=[0.72, 0.28]).astype(float),
})
for col, rate in [("bmi", 0.11), ("sbp", 0.09), ("smoker", 0.19)]:
    mask = rng.random(N) < rate
    synth1.loc[mask, col] = np.nan

# --- Lower-fidelity synthetic dataset (shifted distributions) ---
synth2 = pd.DataFrame({
    "age":       rng.integers(25, 70, N).astype(float),   # narrower range
    "bmi":       rng.normal(29, 6, N),                    # higher mean, more spread
    "sbp":       rng.normal(130, 20, N),                  # shifted
    "sex":       rng.choice(["M", "F"], N, p=[0.65, 0.35]),  # imbalanced
    "diagnosis": rng.choice(["A", "B", "C"], N, p=[0.33, 0.33, 0.34]),  # uniform
    "smoker":    rng.choice([0, 1], N, p=[0.5, 0.5]).astype(float),
})
for col, rate in [("bmi", 0.25), ("sbp", 0.02), ("smoker", 0.05)]:
    mask = rng.random(N) < rate
    synth2.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# 2. Save toy datasets to examples/data/ and write example config
# ---------------------------------------------------------------------------

data_dir = pathlib.Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

real_data.to_csv(data_dir / "real.csv", index=False)
synth1.to_csv(data_dir / "synth1.csv", index=False)
synth2.to_csv(data_dir / "synth2.csv", index=False)

config = {
    "real_data": str(data_dir / "real.csv"),
    "synthetic_datasets": [
        {"name": "synth1", "path": str(data_dir / "synth1.csv")},
        {"name": "synth2", "path": str(data_dir / "synth2.csv")},
    ],
    "column_types": {
        "age": "numerical",
        "bmi": "numerical",
        "sbp": "numerical",
        "sex": "categorical",
        "diagnosis": "categorical",
        "smoker": "categorical",
    },
}

config_path = pathlib.Path(__file__).parent / "example_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Saved datasets to  {data_dir}/")
print(f"Saved config to    {config_path}")
print(f"Launch dashboard:  streamlit run run_dashboard.py -- --config {config_path}")
print()

# ---------------------------------------------------------------------------
# 3. Detect column types
# ---------------------------------------------------------------------------

col_types = detect_column_types(real_data)
print("Detected column types:")
for col, ctype in col_types.items():
    print(f"  {col}: {ctype}")
print()

# ---------------------------------------------------------------------------
# 3. Run fidelity evaluation
# ---------------------------------------------------------------------------

print("=" * 60)
print("Fidelity evaluation — synth1 (high fidelity)")
print("=" * 60)
fid1 = evaluate_fidelity(real_data, synth1, col_types=col_types)

for group, metrics in fid1.items():
    for metric_name, result in metrics.items():
        print(f"  [{group}] {result.metric_name}: score = {result.score:.4f}")

print()
print("=" * 60)
print("Fidelity evaluation — synth2 (lower fidelity)")
print("=" * 60)
fid2 = evaluate_fidelity(real_data, synth2, col_types=col_types)

for group, metrics in fid2.items():
    for metric_name, result in metrics.items():
        print(f"  [{group}] {result.metric_name}: score = {result.score:.4f}")

# ---------------------------------------------------------------------------
# 4. Run missingness evaluation
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Missingness evaluation — synth1")
print("=" * 60)
miss1 = evaluate_missingness(real_data, synth1, col_types=col_types)
for name, result in miss1.items():
    print(f"  {result.metric_name}: score = {result.score:.4f}")

print()
print("=" * 60)
print("Missingness evaluation — synth2")
print("=" * 60)
miss2 = evaluate_missingness(real_data, synth2, col_types=col_types)
for name, result in miss2.items():
    print(f"  {result.metric_name}: score = {result.score:.4f}")

# ---------------------------------------------------------------------------
# 5. Compute scores with custom weighting
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Scoring (custom weights: univariate=0.2, bivariate=0.2, multivariate=0.6)")
print("=" * 60)

fidelity_weights = [0.2, 0.2, 0.6]
missingness_weights = [0.3, 0.3, 0.2, 0.2]

for name, fid_res, miss_res in [("synth1", fid1, miss1), ("synth2", fid2, miss2)]:
    f_scores = compute_fidelity_score(fid_res, weights=fidelity_weights)
    m_scores = compute_missingness_score(miss_res, weights=missingness_weights)
    comp = compute_composite_score(f_scores, m_scores, weights=[0.5, 0.5])

    print(f"\n  {name}:")
    print(f"    Fidelity score   : {f_scores['overall']:.4f}")
    print(f"    Missingness score: {m_scores['overall']:.4f}")
    print(f"    Composite score  : {comp['composite']:.4f}")

# ---------------------------------------------------------------------------
# 6. Ranking
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Ranking")
print("=" * 60)
scores = {
    "synth1": compute_fidelity_score(fid1, weights=fidelity_weights)["overall"],
    "synth2": compute_fidelity_score(fid2, weights=fidelity_weights)["overall"],
}
ranked = sorted(scores.items(), key=lambda x: -x[1])
for rank, (name, score) in enumerate(ranked, 1):
    print(f"  #{rank}  {name}  fidelity={score:.4f}")

print()
print("Done. Launch the dashboard for interactive exploration:")
print("  streamlit run run_dashboard.py")
