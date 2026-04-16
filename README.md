# stdg-eval

A modular Python library for evaluating tabular synthetic data, with a focus on medical / clinical datasets. Covers fidelity and missingness axes, with an interactive Streamlit dashboard and a headless CLI.

---

## Evaluation axes

| Axis | Status |
|------|--------|
| **Fidelity** | ✅ Available |
| **Missingness** | ✅ Available |
| **Utility** | 🔜 Planned |
| **Privacy** | 🔜 Planned |

---

## Installation

```bash
git clone https://github.com/your-org/stdg-eval.git
cd stdg-eval
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**Requirements:** Python ≥ 3.9. Key dependencies: numpy, pandas, scipy, scikit-learn, plotly, streamlit, pyyaml, phik.

---

## Input format

All datasets must be **tabular CSV files** where:
- Each row is one sample / patient record.
- Each column is one variable.
- Missing values are represented as empty cells (standard CSV `NaN`).
- The real dataset and all synthetic datasets must share the **same column names**.

### Column type inference

`stdg-eval` automatically infers whether each column is **numerical** or **categorical**:
- Object / string / boolean columns → categorical.
- Numeric columns with ≤ 20 unique values **and** a cardinality fraction ≤ 5 % → categorical.
- All other numeric columns → numerical.

You can override this in a config file or via the dashboard sidebar.

---

## Quick start

### Programmatic API

```python
import pandas as pd
from stdg_eval import (
    evaluate_fidelity, evaluate_missingness,
    compute_fidelity_score, compute_missingness_score, compute_composite_score,
)

real  = pd.read_csv("data/real.csv")
synth = pd.read_csv("data/synth.csv")

fid  = evaluate_fidelity(real, synth)
miss = evaluate_missingness(real, synth)   # pass raw data — do not impute first

f_scores = compute_fidelity_score(fid)
m_scores = compute_missingness_score(miss)
comp     = compute_composite_score(f_scores, m_scores)

print(f_scores["overall"])    # e.g. 0.847
print(comp["composite"])      # e.g. 0.831
```

Each `evaluate_*` function returns a dict of `MetricResult` objects with:
- `score` — float in [0, 1], **1 = best**
- `details` — raw values, per-column breakdowns, matrices
- `column_scores` — per-column scores where applicable

Custom weights can be passed to the scoring functions:

```python
# weights: [univariate, bivariate, multivariate] — auto-normalised
f_scores = compute_fidelity_score(fid, weights=[0.2, 0.2, 0.6])

# weights: [rate, set_distribution, missing_auroc, dependency_structure]
m_scores = compute_missingness_score(miss, weights=[0.3, 0.3, 0.2, 0.2])

# weights: [fidelity, missingness]
comp = compute_composite_score(f_scores, m_scores, weights=[0.5, 0.5])
```

### CLI

```bash
# Headless evaluation → JSON
stdg-eval evaluate --config configs/my_config.yaml --output results.json

# Precompute expensive metrics once, reuse in dashboard
stdg-eval precompute --config configs/my_config.yaml --output precomputed.json

# Compute only specific groups
stdg-eval precompute --config configs/my_config.yaml --output precomputed.json \
  --groups multivariate missingness

# Launch dashboard
stdg-eval dashboard --config configs/my_config.yaml
```

### Dashboard

```bash
# Upload files interactively
streamlit run run_dashboard.py

# Load from a config file
streamlit run run_dashboard.py -- --config configs/my_config.yaml
```

---

## Config file

```yaml
real_data: data/real.csv
synthetic_datasets:
  - name: synth1
    path: data/synth1.csv
  - name: synth2
    path: data/synth2.csv

column_types:           # optional — auto-inferred if omitted
  age: numerical
  sex: categorical

metrics:                # optional — all true by default
  # Univariate
  wasserstein: true
  tvd: true
  hellinger: true
  # Bivariate
  spearman: true
  contingency: true
  pairwise_correlation_difference: true
  # Multivariate
  auc_roc: true
  propensity_mse: true
  crcl_rs: false        # computationally expensive — disable if not needed
  crcl_sr: false
  # Missingness
  rate: true
  set_distribution: true
  missing_auroc: true
  dependency_structure: true

precomputed_results: precomputed.json   # optional — skip recomputation in dashboard
```

### Precomputing expensive metrics

Bivariate, multivariate, and missingness metrics can be slow on large datasets. Precompute them once and reload in the dashboard without re-running evaluation:

```bash
stdg-eval precompute --config configs/my_config.yaml --output precomputed.json
```

Reference the output in your config with `precomputed_results: precomputed.json`.

---

## Implemented metrics

### Fidelity — univariate

| Metric | Applies to | Score |
|--------|-----------|-------|
| Wasserstein Distance | Numerical | `exp(−WD / (IQR_real + ε))` per column, mean across columns |
| Total Variation Distance | Categorical | `1 − TVD` per column, mean across columns |
| Hellinger Distance | Numerical + Categorical | `1 − HD` per column, mean across columns |

Hellinger Distance uses Scott's-rule histograms on the combined real + synthetic range for numerical columns, and frequency distributions for categorical columns.

### Fidelity — bivariate

| Metric | Applies to | Score |
|--------|-----------|-------|
| Spearman Correlation | Num × Num | `1 − mean(abs(ρ_real − ρ_synth))` across pairs |
| Contingency Matrix TVD | Cat × Cat, Num × Cat | `1 − mean(TVD)` across pairs |
| Pairwise Correlation Difference (PCD) | All pairs (φk) | `1 − mean(abs(φk_real − φk_synth))` across pairs |

PCD uses the φk (phi-k) correlation coefficient — a mixed-type association measure in [0, 1]. Binning uses Scott's rule on pooled real + synthetic values. A Student's t-test flags whether the mean absolute difference is statistically significant (α = 0.05).

### Fidelity — multivariate

| Metric | Score |
|--------|-------|
| AUC-ROC | `1 − 2 × abs(AUROC − 0.5)` — random forest discriminator trained via k-fold CV (default 5); AUROC = 0.5 → score = 1 |
| Propensity MSE | `1 − 4 × pMSE` — propensity score MSE normalised by worst case (0.25 for balanced labels); pMSE = 0 → score = 1 |
| CrCl-RS | `max(0, 1 − abs(mean_ratio − 1))` — train on real, test on synth; ratio = perf_synth / perf_real_held |
| CrCl-SR | same formula — train on synth, test on real; ratio = perf_real / perf_synth_held |

CrCl iterates over each variable as a prediction target using a decision tree (accuracy for categorical, R² for numerical). Complete case analysis is used by default (no imputation). A ratio of 1.0 per variable indicates perfect transfer. Reference: Goncalves et al. (2020) *BMC Med Res Methodol*.

### Missingness

| Metric | Score |
|--------|-------|
| Missingness Rate | `1 − mean(abs(rate_real − rate_synth))` across columns |
| Pattern Distribution | `1 − TVD` between distributions over joint missingness patterns (which combinations of columns are missing together) |
| Missing AUROC | `1 − mean(abs(AUROC_real − AUROC_synth))` — per-column logistic regression classifier predicting whether a cell is missing from all other columns; only columns with missingness rate in [1%, 99%] are included |
| Dependency Structure | `1 − mean(abs(corr_real − corr_synth))` — Pearson correlations between binary missingness indicator vectors across all column pairs with some missingness |

---

## Example workflow

[`examples/example_workflow.py`](examples/example_workflow.py) generates a suite of 7 toy clinical datasets and runs a full evaluation:

```bash
python examples/example_workflow.py
```

The real dataset (`real.csv`) has 500 records, 6 variables (`age`, `bmi`, `sbp`, `sex`, `diagnosis`, `smoker`), and realistic missingness in `bmi` (~12 %), `sbp` (~8 %), `smoker` (~20 %).

| Dataset | Fidelity | Missingness |
|---------|----------|-------------|
| `synth_ideal` | ≈ 1.0 | ≈ 1.0 | Exact copy — upper-bound baseline |
| `synth_fid1` | High | High | Slight distribution shifts |
| `synth_miss1` | High | Lower | Missingness in wrong columns |
| `synth_fid1_miss1` | Moderate | Lower | Shifts + missingness mismatch |
| `synth_fid2` | Lower | High | Larger distribution shifts |
| `synth_miss2` | High | Lower | 30 % missingness in key columns |
| `synth_fid2_miss2` | Lower | Lower | Large shifts + missingness mismatch |

The script writes a ready-to-use config to `examples/example_config.yaml`. Launch the dashboard directly with:

```bash
streamlit run run_dashboard.py -- --config examples/example_config.yaml
```

---

## Interactive dashboard

The dashboard has five tabs:

| Tab | Contents |
|-----|----------|
| **Dataset Description** | Overview of real and synthetic datasets: observation counts, column names and types, per-column missingness rates and unique value counts, flags for any column mismatches between real and synthetic |
| **Individual Report** | Per-dataset deep-dive: univariate CDFs / bar charts, bivariate heatmaps (real / synth / diff), multivariate results with per-variable plots, missingness rate bars + pattern heatmaps + dependency heatmaps |
| **Benchmarking Report** | Cross-dataset comparison: radar chart, score table, rankings by axis with configurable weight sliders |
| **Score Summary** | Three tables — individual metric scores, metric group scores, axis / composite scores |
| **Metric Correlations** | Pearson agreement heatmap between metrics across runs; per-variable score correlation |

**Precomputed results** are loaded automatically via the `precomputed_results` key in the config file. The dashboard injects them directly and only computes what is missing.

---

## Project structure

```
stdg-eval/
├── run_dashboard.py
├── configs/example_config.yaml
├── examples/
│   ├── example_config.yaml
│   └── example_workflow.py
├── tests/
│   ├── conftest.py
│   ├── metrics/
│   │   ├── fidelity/
│   │   │   ├── test_univariate.py
│   │   │   ├── test_bivariate.py
│   │   │   └── test_multivariate.py
│   │   └── missingness/
│   │       └── test_measures.py
│   ├── evaluation/
│   │   └── test_scoring.py
│   └── utils/
│       └── test_data_utils.py
└── stdg_eval/
    ├── cli.py                        # CLI entry point
    ├── config.py                     # EvalConfig, FidelityConfig, MissingnessConfig
    ├── metrics/
    │   ├── base.py                   # BaseMetric, MetricResult
    │   ├── fidelity/
    │   │   ├── univariate.py         # WassersteinDistance, TVD, HellingerDistance
    │   │   ├── bivariate.py          # SpearmanCorrelation, ContingencyMatrix, PCD
    │   │   └── multivariate.py       # AucRoc, PropensityMSE, CrossClassification (CrCl-RS/SR)
    │   └── missingness/
    │       └── measures.py           # MissingnessRate, SetDistribution, MissingAUROC, DependencyStructure
    ├── evaluation/
    │   ├── fidelity.py               # evaluate_fidelity()
    │   ├── missingness.py            # evaluate_missingness()
    │   └── scoring.py                # compute_*_score()
    ├── utils/
    │   ├── data_utils.py             # Loading, column type inference, config parsing
    │   └── precomputed_io.py         # save_precomputed(), load_precomputed()
    └── visualization/
        ├── metric_registry.py        # FIDELITY_GROUPS, MISSINGNESS_METRICS
        ├── plots.py                  # Plotly figure factory
        └── dashboard.py              # Streamlit app
```

---

## Adding a new metric

1. Subclass `BaseMetric` and implement `evaluate()` returning a `MetricResult` with `score` in [0, 1].
2. Register it in `evaluation/fidelity.py` or `evaluation/missingness.py`.
3. Add it to `FIDELITY_GROUPS` or `MISSINGNESS_METRICS` in `visualization/metric_registry.py` — the sidebar checkbox, weight slider, score tables, and precompute pipeline pick it up automatically.
4. Add a plot function in `visualization/plots.py` and wire it into the relevant expander in `dashboard.py`.

```python
from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes
import pandas as pd

class MyMetric(BaseMetric):
    name = "My Metric"
    description = "One-sentence description."
    axis = "fidelity"

    def evaluate(self, real: pd.DataFrame, synthetic: pd.DataFrame,
                 col_types: ColumnTypes) -> MetricResult:
        score = ...  # float in [0, 1]
        return MetricResult(metric_name=self.name, score=score, details={...})
```

---

## TODO

- [ ] Expand test coverage to include new metrics (CrCl-RS/SR, PCD, Hellinger, missingness metrics)
- [ ] Utility metrics (train-on-synthetic / test-on-real task performance)
- [ ] Privacy metrics (membership inference, nearest-neighbour distance ratios)
- [ ] Composite score weighting scheme once utility and privacy axes are implemented
- [ ] Per-dataset PDF / HTML report export
- [ ] Bootstrap confidence intervals for metric scores
