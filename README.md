# stdg-eval

A modular Python library for evaluating tabular synthetic data generation, with a focus on medical / clinical datasets. Designed for reproducibility, extensibility, and interactive exploration.

---

## Evaluation axes

| Axis | Status | Description |
|------|--------|-------------|
| **Fidelity** | ✅ Available | How closely the synthetic data matches the statistical properties of real data |
| **Missingness** | ✅ Available | How faithfully missing-data patterns are reproduced |
| **Utility** | 🔜 TODO | Downstream task performance on synthetic vs real data |
| **Privacy** | 🔜 TODO | Disclosure risk and membership inference assessments |

---

## Installation

It is strongly recommended to install inside a **virtual environment** to keep dependencies isolated and ensure reproducibility.

```bash
git clone https://github.com/your-org/stdg-eval.git
cd stdg-eval

# Create and activate a virtual environment (Python ≥ 3.9 required)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install the package in editable mode (includes all dependencies)
pip install -e .

# Or install dependencies without editable install:
# pip install -r requirements.txt
```

To deactivate the virtual environment when you're done:

```bash
deactivate
```

> **Note:** The `.venv/` directory is listed in `.gitignore` and will not be committed.

**Requirements:** Python ≥ 3.9, and the packages listed in `requirements.txt` (numpy, pandas, scipy, scikit-learn, plotly, streamlit, pyyaml, pyarrow).

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
- Numeric columns with ≤ 20 unique values **and** a cardinality fraction ≤ 5% → categorical.
- All other numeric columns → numerical.

You can override this in a config file or via the dashboard sidebar.

---

## Programmatic API

### 1. Evaluate fidelity

```python
import pandas as pd
from stdg_eval import evaluate_fidelity

real  = pd.read_csv("data/real.csv")
synth = pd.read_csv("data/synth1.csv")

results = evaluate_fidelity(real, synth)
# returns: {"univariate": {...}, "bivariate": {...}, "multivariate": {...}}
```

Each value in the nested dict is a `MetricResult` with:
- `score` — a normalised float in [0, 1] where **1 = perfect fidelity**
- `details` — raw metric values, per-column breakdowns, matrices, etc.
- `column_scores` — per-column scores (where applicable)

### 2. Evaluate missingness

```python
from stdg_eval import evaluate_missingness

miss_results = evaluate_missingness(real, synth)
# returns: {"rate": MetricResult, "set_distribution": ..., "classifier_auroc": ..., "dependency_structure": ...}
```

> **Important:** pass the datasets with their original missing values intact — do **not** impute before calling this function.

### 3. Compute scores

```python
from stdg_eval import compute_fidelity_score, compute_missingness_score, compute_composite_score

# Custom weights: [univariate, bivariate, multivariate] — auto-normalised
fidelity_score = compute_fidelity_score(results, weights=[0.2, 0.2, 0.6])
print(fidelity_score["overall"])   # e.g. 0.847

# Custom weights: [rate, set_distribution, classifier_auroc, dependency_structure]
miss_score = compute_missingness_score(miss_results, weights=[0.3, 0.3, 0.2, 0.2])

# Composite score (fidelity + missingness axes)
composite = compute_composite_score(fidelity_score, miss_score, weights=[0.5, 0.5])
print(composite["composite"])      # e.g. 0.831
```

### 4. Column type overrides

```python
from stdg_eval.utils import detect_column_types

col_types = detect_column_types(real, override={"sex": "categorical", "age": "numerical"})
results = evaluate_fidelity(real, synth, col_types=col_types)
```

### Full example

See [`examples/example_workflow.py`](examples/example_workflow.py) for a self-contained script that creates toy datasets, runs all metrics, and prints ranked scores.

```bash
python examples/example_workflow.py
```

---

## Interactive dashboard

### Launch with file upload (recommended for exploration)

```bash
streamlit run run_dashboard.py
```

Then open `http://localhost:8501` in your browser. Use the sidebar to upload your CSV files directly.

### Launch with a config file

Create a YAML config (see [`configs/example_config.yaml`](configs/example_config.yaml)):

```yaml
real_data: data/real.csv
synthetic_datasets:
  - name: synth1
    path: data/synth1.csv
  - name: synth2
    path: data/synth2.csv
  - name: synth3
    path: data/synth3.csv
column_types:           # optional overrides
  sex: categorical
  age: numerical
```

Or a plain-text config:

```text
# lines starting with # are ignored
data/real.csv
data/synth1.csv
data/synth2.csv
data/synth3.csv
```

Then launch:

```bash
stdg-eval dashboard --config configs/my_config.yaml
# or
streamlit run run_dashboard.py
```

### Dashboard workflow (example with 3 synthetic datasets)

1. Upload `real.csv` and `synth1.csv`, `synth2.csv`, `synth3.csv` in the sidebar (or point to a config file).
2. Optionally review and override column type assignments.
3. Click **▶ Run evaluation** — all metrics are computed for each synthetic dataset.
4. **Individual Report tab**: select a synthetic dataset from the dropdown to see:
   - CDF plots for each numerical column (with Wasserstein distance annotated)
   - Frequency bar charts for each categorical column (with TVD annotated)
   - Side-by-side Spearman correlation heatmaps + difference heatmap
   - Contingency table TVD per pair
   - Cross-classification AUROC and propensity MSE results
   - Per-variable missingness rate bar chart
   - Missingness pattern heatmaps (real vs synthetic)
   - Missingness dependency structure heatmaps
5. **Benchmarking Report tab**:
   - Adjust weight sliders for each metric group and evaluation axis.
   - Scores and rankings update interactively.
   - See the best-performing dataset per axis (Fidelity, Missingness, Composite).
   - Radar chart and score table for cross-dataset comparison.

### Headless evaluation (no dashboard)

```bash
stdg-eval evaluate --real data/real.csv --synth data/synth1.csv data/synth2.csv --output results.json
```

---

## Implemented metrics

### Fidelity — univariate

| Metric | Columns | Score normalisation |
|--------|---------|---------------------|
| **Wasserstein Distance** | Numerical | `exp(−WD / IQR_real)` per column; mean across columns |
| **Total Variation Distance (TVD)** | Categorical | `1 − TVD` per column; mean across columns |

### Fidelity — bivariate

| Metric | Columns | Score normalisation |
|--------|---------|---------------------|
| **Spearman Correlation** | Numerical × Numerical | `1 − mean(|ρ_real − ρ_synth|)` across all pairs |
| **Contingency Matrix** | Categorical × Categorical, Numerical × Categorical | `1 − mean(TVD)` across all pairs |
| **Pairwise Correlation Difference** | All | 🔜 TODO |

### Fidelity — multivariate

| Metric | Score normalisation |
|--------|---------------------|
| **Cross-Classification** | `1 − 2 × |AUROC − 0.5|` (0.5 AUROC → score = 1) |
| **Propensity MSE** | `1 − pMSE / pMSE_null` (null = shuffled labels baseline) |

### Missingness

| Metric | Score normalisation |
|--------|---------------------|
| **Missingness Rate** | `1 − mean(|rate_real − rate_synth|)` across columns |
| **Pattern Distribution** | `1 − TVD(pattern_dist_real, pattern_dist_synth)` |
| **Classifier AUROC** | `1 − mean(|AUROC_real − AUROC_synth|)` across columns |
| **Dependency Structure** | `1 − mean(|corr_real − corr_synth|)` across missingness indicator pairs |

---

## Project structure

```
stdg-eval/
├── run_dashboard.py           # Streamlit launcher
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── example_config.yaml
├── examples/
│   └── example_workflow.py
└── stdg_eval/
    ├── __init__.py            # Public API re-exports
    ├── cli.py                 # stdg-eval CLI
    ├── config.py              # Defaults and EvalConfig dataclass
    ├── utils/
    │   └── data_utils.py      # Data loading, column type inference
    ├── metrics/
    │   ├── base.py            # BaseMetric ABC + MetricResult dataclass
    │   ├── fidelity/
    │   │   ├── univariate.py  # WassersteinDistance, TotalVariationDistance
    │   │   ├── bivariate.py   # SpearmanCorrelation, ContingencyMatrix
    │   │   └── multivariate.py # CrossClassification, PropensityMSE
    │   └── missingness/
    │       └── measures.py    # MissingnessRate, MissingnessSetDistribution,
    │                          #   MissingnessClassifierAUROC, MissingnessDependencyStructure
    ├── evaluation/
    │   ├── fidelity.py        # evaluate_fidelity()
    │   ├── missingness.py     # evaluate_missingness()
    │   └── scoring.py         # compute_fidelity_score(), compute_missingness_score(),
    │                          #   compute_composite_score()
    └── visualization/
        ├── plots.py           # Plotly figure factory
        └── dashboard.py       # Streamlit dashboard
```

---

## Adding new metrics

1. Create a new class in the appropriate module (e.g., `stdg_eval/metrics/fidelity/univariate.py`) that subclasses `BaseMetric` and implements `evaluate()`.
2. `evaluate()` must return a `MetricResult` with a normalised `score` in [0, 1].
3. Register the metric in the relevant evaluation function (`evaluation/fidelity.py` or `evaluation/missingness.py`).
4. Add its weight slot to `compute_fidelity_score()` or `compute_missingness_score()` in `evaluation/scoring.py`.
5. Add a corresponding plot function in `visualization/plots.py` and wire it into the dashboard.

```python
# Example: adding a new univariate metric
from stdg_eval.metrics.base import BaseMetric, MetricResult
from stdg_eval.utils.data_utils import ColumnTypes
import pandas as pd

class MyNewMetric(BaseMetric):
    name = "My New Metric"
    description = "One-sentence description."
    axis = "fidelity"

    def evaluate(self, real: pd.DataFrame, synthetic: pd.DataFrame,
                 col_types: ColumnTypes) -> MetricResult:
        # ... compute raw_score in [0, 1] ...
        return MetricResult(
            metric_name=self.name,
            score=raw_score,
            details={"raw_values": ...},
        )
```

---

## TODO

- [ ] **Unit tests** — `pytest` test suite covering all metrics and scoring functions (need to double-check the implementation of all metrics).
- [ ] **Pairwise Correlation Difference** — unified mixed-type correlation measure (Cramér's V for categorical, point-biserial for mixed, Spearman for numerical).
- [ ] **Utility metrics** — downstream task performance (e.g., train-on-synthetic/test-on-real AUROC for a specified target variable).
- [ ] **Privacy metrics** — membership inference attack success rate, nearest-neighbour distance ratios, attribute disclosure risk.
- [ ] **Composite score design** — finalise the weighting scheme for the 4-axis composite score once utility and privacy are implemented.
- [ ] **Per-dataset PDF report export** — generate a self-contained HTML/PDF report for a single synthetic dataset.
- [ ] **Confidence intervals** — bootstrap CIs for metric scores.
