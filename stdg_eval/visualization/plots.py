"""
Plotly figure factory for stdg-eval.

Every function returns a ``plotly.graph_objects.Figure`` that can be rendered
in Streamlit (``st.plotly_chart``), Jupyter, or saved as HTML/PNG.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
REAL_COLOR = "#2196F3"     # blue
SYNTH_COLORS = [
    "#E91E63",  # pink
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
]

def _synth_color(idx: int) -> str:
    return SYNTH_COLORS[idx % len(SYNTH_COLORS)]


# ===========================================================================
# 1. Univariate – CDF / distribution comparison (Wasserstein)
# ===========================================================================

def plot_numerical_cdf(
    real: pd.Series,
    synthetic: pd.Series,
    column_name: str,
    wasserstein_distance: Optional[float] = None,
    hellinger_distance: Optional[float] = None,
    synth_label: str = "Synthetic",
    synth_color: str = SYNTH_COLORS[0],
) -> go.Figure:
    """
    Empirical CDF plot comparing real and synthetic distributions for one
    numerical column, with the Wasserstein distance annotated.
    """
    r = real.dropna().sort_values().values
    s = synthetic.dropna().sort_values().values

    r_cdf = np.arange(1, len(r) + 1) / len(r)
    s_cdf = np.arange(1, len(s) + 1) / len(s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=r_cdf, mode="lines", name="Real",
                             line=dict(color=REAL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=s, y=s_cdf, mode="lines", name=synth_label,
                             line=dict(color=synth_color, width=2, dash="dash")))

    title = f"CDF — {column_name}"
    annotations = []
    if wasserstein_distance is not None:
        annotations.append(f"WD = {wasserstein_distance:.4f}")
    if hellinger_distance is not None:
        annotations.append(f"HD = {hellinger_distance:.4f}")
    if annotations:
        title += "  (" + "  |  ".join(annotations) + ")"

    fig.update_layout(
        title=title,
        xaxis_title=column_name,
        yaxis_title="Cumulative probability",
        legend=dict(x=0.01, y=0.99),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_categorical_bars(
    real_freq: Dict,
    synth_freq: Dict,
    column_name: str,
    tvd: Optional[float] = None,
    hellinger_distance: Optional[float] = None,
    synth_label: str = "Synthetic",
    synth_color: str = SYNTH_COLORS[0],
) -> go.Figure:
    """
    Grouped bar chart comparing real vs synthetic category frequencies.
    """
    all_cats = sorted(set(real_freq) | set(synth_freq), key=str)
    r_vals = [real_freq.get(c, 0.0) for c in all_cats]
    s_vals = [synth_freq.get(c, 0.0) for c in all_cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(c) for c in all_cats], y=r_vals,
                         name="Real", marker_color=REAL_COLOR))
    fig.add_trace(go.Bar(x=[str(c) for c in all_cats], y=s_vals,
                         name=synth_label, marker_color=synth_color))

    title = f"Category frequencies — {column_name}"
    annotations = []
    if tvd is not None:
        annotations.append(f"TVD = {tvd:.4f}")
    if hellinger_distance is not None:
        annotations.append(f"HD = {hellinger_distance:.4f}")
    if annotations:
        title += "  (" + "  |  ".join(annotations) + ")"

    fig.update_layout(
        title=title,
        xaxis_title=column_name,
        yaxis_title="Relative frequency",
        barmode="group",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ===========================================================================
# 2. Bivariate – Correlation heatmaps (Spearman)
# ===========================================================================

def plot_correlation_heatmaps(
    real_corr: pd.DataFrame,
    synth_corr: pd.DataFrame,
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Side-by-side heatmaps of real vs synthetic Spearman correlation matrices,
    plus a difference heatmap.
    """
    cols = real_corr.columns.tolist()
    diff = (real_corr - synth_corr).abs()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Real", synth_label, "|Δ| Difference"],
        horizontal_spacing=0.08,
    )

    kw = dict(zmin=-1, zmax=1, colorscale="RdBu", showscale=False)

    fig.add_trace(go.Heatmap(z=real_corr.values, x=cols, y=cols,
                             colorscale="RdBu", zmin=-1, zmax=1, showscale=True,
                             colorbar=dict(x=0.28, len=0.9)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=synth_corr.values, x=cols, y=cols, **kw), row=1, col=2)
    fig.add_trace(go.Heatmap(z=diff.values, x=cols, y=cols,
                             zmin=0, zmax=1, colorscale="Reds", showscale=True,
                             colorbar=dict(x=1.0, len=0.9)), row=1, col=3)

    fig.update_layout(
        title="Spearman correlation matrices",
        height=max(350, 40 * len(cols) + 100),
        margin=dict(l=60, r=60, t=70, b=40),
    )
    return fig


def _build_symmetric_matrix(
    pair_values: Dict[str, float],
    all_cols: List[str],
    diagonal: Optional[float] = None,
) -> np.ndarray:
    """
    Build a symmetric N×N matrix from a dict keyed by ``"colA|colB"`` pairs.

    Cells with no entry remain ``nan``. The diagonal is set to ``diagonal``
    if provided, otherwise left as ``nan``.
    """
    n = len(all_cols)
    col_idx = {c: i for i, c in enumerate(all_cols)}
    mat = np.full((n, n), np.nan)
    if diagonal is not None:
        np.fill_diagonal(mat, diagonal)
    for key, val in pair_values.items():
        parts = key.split("|", 1)
        if len(parts) != 2:
            continue
        c1, c2 = parts
        if c1 in col_idx and c2 in col_idx:
            i, j = col_idx[c1], col_idx[c2]
            mat[i, j] = val
            mat[j, i] = val
    return mat


def plot_pcd_heatmaps(
    pair_real: Dict[str, float],
    pair_synth: Dict[str, float],
    pair_diffs: Dict[str, float],
    all_cols: List[str],
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Three-panel heatmap for PCD results: real phi-k, synthetic phi-k, and |Δ|.

    Produces a symmetric N×N matrix (all variables on both axes) with the
    diagonal set to 1.0 for real/synth panels and 0.0 for the difference panel.
    """
    real_mat = _build_symmetric_matrix(pair_real, all_cols, diagonal=1.0)
    synth_mat = _build_symmetric_matrix(pair_synth, all_cols, diagonal=1.0)
    diff_mat = _build_symmetric_matrix(pair_diffs, all_cols, diagonal=0.0)

    height = max(350, 40 * len(all_cols) + 100)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Real", synth_label, "|Δ| Difference"],
        horizontal_spacing=0.08,
    )

    def _fmt(mat):
        return [[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in mat]

    kw_assoc = dict(zmin=0, zmax=1, colorscale="Blues", showscale=False)
    fig.add_trace(go.Heatmap(
        z=real_mat, x=all_cols, y=all_cols, text=_fmt(real_mat),
        texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
        showscale=True, colorbar=dict(x=0.28, len=0.9, title="phi-k"),
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=synth_mat, x=all_cols, y=all_cols, text=_fmt(synth_mat),
        texttemplate="%{text}", **kw_assoc,
    ), row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=diff_mat, x=all_cols, y=all_cols, text=_fmt(diff_mat),
        texttemplate="%{text}", zmin=0, zmax=1, colorscale="Reds",
        showscale=True, colorbar=dict(x=1.0, len=0.9, title="|Δ|"),
    ), row=1, col=3)

    fig.update_layout(
        title="Pairwise Correlation Difference — phi-k association matrices",
        height=height,
        margin=dict(l=60, r=60, t=70, b=40),
    )
    return fig


def plot_contingency_tvd_heatmap(
    pair_tvds: Dict[str, float],
    all_cols: List[str],
    col_types: Dict[str, str],
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Single N×N heatmap of contingency TVD values.

    Cells covering numerical×numerical pairs are left blank (NaN) since the
    contingency metric only measures categorical and mixed pairs. The diagonal
    is set to 0 (no divergence between a column and itself).
    """
    mat = _build_symmetric_matrix(pair_tvds, all_cols, diagonal=0.0)
    fmt = [[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in mat]

    fig = go.Figure(go.Heatmap(
        z=mat, x=all_cols, y=all_cols,
        text=fmt, texttemplate="%{text}",
        zmin=0, zmax=1, colorscale="Reds",
        colorbar=dict(title="TVD"),
    ))
    fig.update_layout(
        title=f"Contingency TVD (Real vs {synth_label})",
        height=max(350, 40 * len(all_cols) + 100),
        margin=dict(l=60, r=60, t=70, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_contingency_pair(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    col1: str,
    col2: str,
    synth_label: str = "Synthetic",
) -> go.Figure:
    """Side-by-side normalised contingency heatmaps for a pair of categorical columns."""
    r_ct = pd.crosstab(real[col1], real[col2], normalize=True)
    s_ct = pd.crosstab(synthetic[col1], synthetic[col2], normalize=True)

    # Align indices
    idx = sorted(set(r_ct.index) | set(s_ct.index), key=str)
    cols = sorted(set(r_ct.columns) | set(s_ct.columns), key=str)
    r_ct = r_ct.reindex(index=idx, columns=cols, fill_value=0)
    s_ct = s_ct.reindex(index=idx, columns=cols, fill_value=0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Real", synth_label],
                        horizontal_spacing=0.12)

    kw = dict(colorscale="Blues", zmin=0, zmax=max(r_ct.values.max(), s_ct.values.max()))

    fig.add_trace(go.Heatmap(z=r_ct.values,
                             x=[str(c) for c in cols],
                             y=[str(i) for i in idx],
                             showscale=True,
                             colorbar=dict(x=0.44, len=0.9),
                             **kw), row=1, col=1)
    fig.add_trace(go.Heatmap(z=s_ct.values,
                             x=[str(c) for c in cols],
                             y=[str(i) for i in idx],
                             showscale=True,
                             colorbar=dict(x=1.0, len=0.9),
                             **kw), row=1, col=2)

    fig.update_layout(
        title=f"Contingency table: {col1} × {col2}",
        height=max(300, 30 * len(idx) + 100),
        margin=dict(l=60, r=60, t=70, b=40),
    )
    return fig


# ===========================================================================
# 3. Missingness – rate bar chart and pattern heatmap
# ===========================================================================

def plot_missingness_rates(
    real_rates: Dict[str, float],
    synth_rates: Dict[str, float],
    synth_label: str = "Synthetic",
    synth_color: str = SYNTH_COLORS[0],
) -> go.Figure:
    """
    Grouped bar chart of per-column missingness rates for real vs synthetic.
    Only shows columns that have any missingness in at least one dataset.
    """
    cols = sorted(
        [c for c in real_rates if real_rates.get(c, 0) > 0 or synth_rates.get(c, 0) > 0],
        key=lambda c: -real_rates.get(c, 0),
    )
    if not cols:
        cols = list(real_rates.keys())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cols, y=[real_rates.get(c, 0) for c in cols],
                         name="Real", marker_color=REAL_COLOR))
    fig.add_trace(go.Bar(x=cols, y=[synth_rates.get(c, 0) for c in cols],
                         name=synth_label, marker_color=synth_color))

    fig.update_layout(
        title="Missingness rates per variable",
        xaxis_title="Column",
        yaxis_title="Missing fraction",
        barmode="group",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=40, r=20, t=50, b=100),
    )
    return fig


def plot_missingness_pattern_heatmap(df: pd.DataFrame, title: str = "Missingness pattern") -> go.Figure:
    """
    Binary heatmap: rows = samples (sorted by missingness pattern), columns = variables.
    White = observed, dark = missing.
    """
    ind = df.isnull().astype(int)
    # Sort rows by pattern similarity (sort by missingness indicator as a string)
    ind_sorted = ind.loc[ind.apply(lambda r: "".join(r.astype(str)), axis=1).sort_values().index]

    # Show at most 500 rows for performance
    if len(ind_sorted) > 500:
        step = len(ind_sorted) // 500
        ind_sorted = ind_sorted.iloc[::step]

    fig = go.Figure(go.Heatmap(
        z=ind_sorted.values,
        x=ind_sorted.columns.tolist(),
        colorscale=[[0, "white"], [1, "#1565C0"]],
        showscale=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Samples (sorted)",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=60, r=20, t=50, b=100),
    )
    return fig


def plot_missingness_dependency(
    corr_matrix: pd.DataFrame,
    title: str = "Missingness dependency structure",
) -> go.Figure:
    """Heatmap of pairwise correlations between missingness indicators."""
    cols = corr_matrix.columns.tolist()
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=cols, y=cols,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=max(300, 30 * len(cols) + 80),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def plot_missingness_dependency_diff(
    real_corr: pd.DataFrame,
    synth_corr: pd.DataFrame,
    title: str = "Absolute difference |real − synth|",
) -> go.Figure:
    """Heatmap of elementwise absolute difference between real and synthetic missingness correlation matrices."""
    diff = abs(real_corr.values - synth_corr.values)
    cols = real_corr.columns.tolist()
    fig = go.Figure(go.Heatmap(
        z=diff,
        x=cols, y=cols,
        colorscale="Reds",
        zmin=0, zmax=float(max(diff.max(), 0.01)),
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=max(300, 30 * len(cols) + 80),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def plot_missing_auroc(
    auroc_real: Dict[str, float],
    auroc_synth: Dict[str, float],
    synth_label: str = "Synthetic",
    synth_color: str = SYNTH_COLORS[0],
) -> go.Figure:
    """
    Grouped bar chart of per-column missingness AUROC for real vs synthetic.

    Each bar shows how well a logistic classifier predicts whether that column
    is missing (using all other columns as features). A reference line at
    AUROC = 0.5 marks chance level (missingness is unpredictable). Columns are
    sorted by real AUROC descending.
    """
    cols = sorted(auroc_real.keys(), key=lambda c: -auroc_real.get(c, 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols,
        y=[auroc_real.get(c, float("nan")) for c in cols],
        name="Real",
        marker_color=REAL_COLOR,
    ))
    fig.add_trace(go.Bar(
        x=cols,
        y=[auroc_synth.get(c, float("nan")) for c in cols],
        name=synth_label,
        marker_color=synth_color,
    ))
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="grey",
        annotation_text="chance (0.5)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Missingness AUROC per variable",
        xaxis_title="Column",
        yaxis_title="AUROC",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(l=40, r=20, t=50, b=100),
    )
    return fig


# ===========================================================================
# 3b. UpSet plot — missingness pattern frequencies
# ===========================================================================

def plot_missingness_upset(
    df: pd.DataFrame,
    title: str = "Missingness patterns",
    top_n: int = 15,
    bar_color: str = REAL_COLOR,
) -> go.Figure:
    """
    UpSet-style plot for missingness patterns.

    Shows the *top_n* most common missingness patterns sorted by frequency
    (most common on the left).  Complete-case rows (no missing values) are
    excluded so the chart focuses on the patterns that matter.

    Layout
    ------
    Row 1 (top)   — vertical bar chart: proportion of rows with each pattern.
    Row 2 (bottom) — dot matrix: filled circle = variable is missing in that
                     pattern; empty circle = observed.  Vertical lines connect
                     the filled dots within each pattern column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse (real or synthetic).
    title : str
        Plot title.
    top_n : int
        Maximum number of patterns to display.
    bar_color : str
        Fill colour for the bars and filled dots.
    """
    ind = df.isnull().astype(int)
    cols = list(ind.columns)
    n_cols = len(cols)

    # Count patterns, drop all-observed rows, keep top_n
    pattern_counts = ind.apply(tuple, axis=1).value_counts()
    all_observed = tuple([0] * n_cols)
    pattern_counts = pattern_counts.drop(all_observed, errors="ignore")
    pattern_counts = pattern_counts.head(top_n)

    if pattern_counts.empty:
        # No missingness at all — return a simple message figure
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="No missing values", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)],
            height=300,
        )
        return fig

    patterns = list(pattern_counts.index)        # list of tuples
    proportions = (pattern_counts / len(df)).tolist()
    n_patterns = len(patterns)

    # x-axis positions for patterns (0-indexed)
    x_pos = list(range(n_patterns))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.45, 0.55],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # --- Row 1: bar chart ---
    fig.add_trace(
        go.Bar(
            x=x_pos,
            y=proportions,
            marker_color=bar_color,
            showlegend=False,
            hovertemplate="Pattern %{x}<br>Proportion: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Row 2: dot matrix ---
    # Determine y positions for each variable (top variable at highest y)
    y_labels = list(reversed(cols))  # variable 0 at bottom, last at top
    y_map = {col: i for i, col in enumerate(y_labels)}

    # Empty (background) dots for all cells
    all_x = [xi for xi in x_pos for _ in range(n_cols)]
    all_y = [yi for _ in x_pos for yi in range(n_cols)]
    fig.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers",
            marker=dict(color="lightgrey", size=10, symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # Filled dots + vertical connecting lines for missing variables
    for xi, pattern in enumerate(patterns):
        missing_ys = [y_map[col] for col, is_miss in zip(cols, pattern) if is_miss]
        if not missing_ys:
            continue
        # Vertical line connecting the filled dots
        fig.add_trace(
            go.Scatter(
                x=[xi] * len(missing_ys),
                y=missing_ys,
                mode="lines",
                line=dict(color=bar_color, width=3),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )
        # Filled dots
        fig.add_trace(
            go.Scatter(
                x=[xi] * len(missing_ys),
                y=missing_ys,
                mode="markers",
                marker=dict(color=bar_color, size=10, symbol="circle"),
                showlegend=False,
                hovertemplate=(
                    "Pattern %{x}<br>"
                    + "<br>".join(
                        cols[j] for j, is_miss in enumerate(pattern) if is_miss
                    )
                    + "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    bar_height = max(200, 30 * min(top_n, n_patterns) + 60)
    dot_height = max(160, 28 * n_cols + 40)

    fig.update_layout(
        title=title,
        height=bar_height + dot_height,
        margin=dict(l=100, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        tickvals=x_pos,
        ticktext=[f"P{i+1}" for i in x_pos],
        row=2, col=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Proportion", row=1, col=1)
    fig.update_yaxes(
        tickvals=list(range(n_cols)),
        ticktext=y_labels,
        row=2, col=1,
    )

    return fig


# ===========================================================================
# 4. Benchmarking / scoring summary
# ===========================================================================

def plot_score_radar(
    scores: Dict[str, Dict[str, float]],
    axes: Optional[List[str]] = None,
) -> go.Figure:
    """
    Radar (spider) chart comparing scores across evaluation axes for multiple
    synthetic datasets.

    Parameters
    ----------
    scores:
        ``{dataset_name: {"fidelity": 0.85, "missingness": 0.9, ...}}``
    axes:
        List of axis names to plot. Defaults to all keys in the first entry.
    """
    if not scores:
        return go.Figure()

    if axes is None:
        axes = list(next(iter(scores.values())).keys())

    fig = go.Figure()
    for i, (name, vals) in enumerate(scores.items()):
        r = [vals.get(ax, 0.0) for ax in axes]
        r.append(r[0])  # close the polygon
        theta = axes + [axes[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta, fill="toself",
            name=name,
            line=dict(color=_synth_color(i)),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Score comparison (radar)",
        height=450,
        showlegend=True,
    )
    return fig


def plot_metric_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Metric correlation") -> go.Figure:
    """
    Annotated heatmap of a metric-vs-metric Pearson correlation matrix.
    Values range from -1 to 1 on a RdBu scale.
    """
    labels = corr_df.columns.tolist()
    z = corr_df.values
    text = [[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title=title,
        height=max(300, 60 * len(labels) + 100),
        margin=dict(l=120, r=40, t=60, b=120),
        xaxis_tickangle=-40,
    )
    return fig


def plot_propensity_histogram(
    propensity_scores: List[float],
    labels: List[int],
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Overlapping histogram of propensity scores (P(synthetic)) split by true label.

    Real records (label=0) are shown in blue, synthetic records (label=1) in the
    synth colour. Under perfect fidelity both distributions should centre on 0.5.
    A vertical dashed line marks 0.5 as the ideal reference.
    """
    import numpy as np
    scores = np.array(propensity_scores)
    labs = np.array(labels)

    real_scores = scores[labs == 0]
    synth_scores = scores[labs == 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=real_scores,
        name="Real",
        marker_color=REAL_COLOR,
        opacity=0.6,
        nbinsx=30,
        histnorm="probability",
    ))
    fig.add_trace(go.Histogram(
        x=synth_scores,
        name=synth_label,
        marker_color=SYNTH_COLORS[0],
        opacity=0.6,
        nbinsx=30,
        histnorm="probability",
    ))
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="grey",
        annotation_text="0.5 (ideal)",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title="Propensity score distribution (P(synthetic))",
        xaxis=dict(title="Propensity score", range=[0, 1]),
        yaxis_title="Proportion",
        legend=dict(x=0.01, y=0.99),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_crcl_ratios(
    per_variable: Dict[str, dict],
    mode: str,
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Bar chart of per-variable cross-classification ratios.

    Each bar is perf_other / perf_held_out, with a reference line at 1.0
    (perfect score). Bars > 1 are coloured blue (better than expected),
    bars < 1 are coloured red (worse than expected).
    """
    cols = list(per_variable.keys())
    ratios = [per_variable[c]["ratio"] for c in cols]
    colors = ["steelblue" if r >= 1.0 else "tomato" for r in ratios]

    if mode == "RS":
        train_label, test_label = "Real (train)", f"{synth_label} (test)"
        hover = [
            f"perf_real_test: {per_variable[c].get('perf_real_test', float('nan')):.4f}<br>"
            f"perf_synth: {per_variable[c].get('perf_synth', float('nan')):.4f}<br>"
            f"type: {per_variable[c].get('target_type', '')}"
            for c in cols
        ]
    else:
        train_label, test_label = f"{synth_label} (train)", "Real (test)"
        hover = [
            f"perf_synth_test: {per_variable[c].get('perf_synth_test', float('nan')):.4f}<br>"
            f"perf_real: {per_variable[c].get('perf_real', float('nan')):.4f}<br>"
            f"type: {per_variable[c].get('target_type', '')}"
            for c in cols
        ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols,
        y=ratios,
        marker_color=colors,
        text=[f"{r:.3f}" for r in ratios],
        textposition="outside",
        hovertext=hover,
        hoverinfo="text+x",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=f"CrCl-{mode}: per-variable ratio ({train_label} → {test_label})",
        yaxis_title="perf_other / perf_held_out",
        xaxis_title="Target variable",
        height=max(300, 40 * len(cols) + 120),
        margin=dict(l=60, r=60, t=60, b=120),
        xaxis_tickangle=-45,
    )
    return fig


def plot_crcl_scores(
    per_variable: Dict[str, dict],
    mode: str,
    synth_label: str = "Synthetic",
) -> go.Figure:
    """
    Grouped bar chart showing the two raw performance scores per target variable
    for a CrCl-RS or CrCl-SR result.

    For RS: bars are perf_real_test (held-out) and perf_synth (other).
    For SR: bars are perf_synth_test (held-out) and perf_real (other).
    Score metric is accuracy for categorical targets, R² for numerical.
    """
    cols = list(per_variable.keys())

    if mode == "RS":
        held_key, other_key = "perf_real_test", "perf_synth"
        held_label = "Real (held-out test)"
        other_label = f"{synth_label}"
    else:
        held_key, other_key = "perf_synth_test", "perf_real"
        held_label = f"{synth_label} (held-out test)"
        other_label = "Real"

    held_vals = [per_variable[c].get(held_key, float("nan")) for c in cols]
    other_vals = [per_variable[c].get(other_key, float("nan")) for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=held_label,
        x=cols,
        y=held_vals,
        marker_color="steelblue",
        text=[f"{v:.3f}" if not (v != v) else "" for v in held_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=other_label,
        x=cols,
        y=other_vals,
        marker_color="darkorange",
        text=[f"{v:.3f}" if not (v != v) else "" for v in other_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title=f"CrCl-{mode}: per-variable performance scores (acc / R²)",
        yaxis_title="Accuracy (categorical) / R² (numerical)",
        xaxis_title="Target variable",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(350, 40 * len(cols) + 150),
        margin=dict(l=60, r=60, t=80, b=120),
        xaxis_tickangle=-45,
    )
    return fig


def plot_score_bar(
    scores: Dict[str, float],
    title: str = "Scores",
    color: Optional[str] = None,
) -> go.Figure:
    """Horizontal bar chart of scores per synthetic dataset."""
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    names = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = [color or _synth_color(i) for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=vals, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1.05], title="Score"),
        height=max(250, 40 * len(names) + 80),
        margin=dict(l=120, r=60, t=50, b=40),
    )
    return fig


def plot_score_table(summary_df: pd.DataFrame) -> go.Figure:
    """Render a summary DataFrame as a Plotly table."""
    header_vals = list(summary_df.columns)
    cell_vals = [summary_df[c].tolist() for c in summary_df.columns]

    # Format floats
    formatted = []
    for col in summary_df.columns:
        try:
            formatted.append([f"{v:.4f}" if isinstance(v, float) else str(v)
                               for v in summary_df[col]])
        except Exception:
            formatted.append([str(v) for v in summary_df[col]])

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#1565C0",
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=formatted,
            fill_color=[["#EFF3FF" if i % 2 == 0 else "white"
                          for i in range(len(summary_df))]],
            align=["left"] + ["center"] * (len(header_vals) - 1),
            font=dict(size=12),
        ),
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=50 + 35 * len(summary_df))
    return fig


# ===========================================================================
# 7. Meta-evaluation plots
# ===========================================================================

def plot_meta_eval_scenario(
    scenario_name: str,
    per_dataset: List[Dict],
    score_keys: List[str],
    score_labels: Dict[str, str],
) -> go.Figure:
    """
    Per-scenario scatter plot with error bars.

    Each *score_key* is shown as one group of points (mean ± std across the
    *n_datasets* replicates).  All groups are overlaid on the same axis so the
    relative performance across metrics for this scenario is immediately visible.

    Parameters
    ----------
    scenario_name : str
        Used in the chart title.
    per_dataset : list of dicts
        Each dict is one replicate row from ``results[scenario]["per_dataset"]``.
    score_keys : list of str
        Keys present in the per-dataset dicts to plot (e.g.
        ``["fidelity_overall", "missingness_overall", "composite_score"]``).
    score_labels : dict
        Human-readable label for each key.
    """
    fig = go.Figure()
    for i, key in enumerate(score_keys):
        vals = [row[key] for row in per_dataset if key in row]
        if not vals:
            continue
        arr = np.array(vals)
        mean, std = float(np.mean(arr)), float(np.std(arr))
        # Individual points
        fig.add_trace(go.Scatter(
            x=[score_labels.get(key, key)] * len(vals),
            y=vals,
            mode="markers",
            marker=dict(color=_synth_color(i), size=7, opacity=0.5),
            showlegend=False,
        ))
        # Mean ± std error bar
        fig.add_trace(go.Scatter(
            x=[score_labels.get(key, key)],
            y=[mean],
            mode="markers",
            marker=dict(color=_synth_color(i), size=12, symbol="diamond"),
            error_y=dict(type="data", array=[std], visible=True, color=_synth_color(i)),
            name=score_labels.get(key, key),
        ))

    fig.update_layout(
        title=f"{scenario_name}",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Metric",
        height=380,
        margin=dict(l=40, r=20, t=50, b=60),
        showlegend=False,
    )
    return fig


def plot_meta_eval_summary(
    results: Dict,
    score_keys: List[str],
    score_labels: Dict[str, str],
) -> go.Figure:
    """
    Summary plot: mean ± std of multiple scores across all scenarios.

    Shows one group of error-bar markers per scenario (one marker per score
    key), so fidelity, missingness and composite can be compared side-by-side.
    Scenarios are displayed in the order they appear in *results*.

    Parameters
    ----------
    results : dict
        Full meta-evaluation results dict (keyed by scenario name).
    score_keys : list of str
        Score keys to plot, e.g. ``["fidelity_overall", "missingness_overall",
        "composite_score"]``.
    score_labels : dict
        Mapping from score key to display label used in the legend.
    """
    # Collect per-scenario stats for each key, preserving results dict order
    scenario_stats: Dict[str, Dict[str, dict]] = {}  # scenario → key → {mean, std}
    for scenario, data in results.items():
        per_ds = data.get("per_dataset", [])
        for key in score_keys:
            vals = [row[key] for row in per_ds if key in row]
            if not vals:
                continue
            arr = np.array(vals)
            scenario_stats.setdefault(scenario, {})[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

    scenarios = list(scenario_stats.keys())

    trace_colors = [REAL_COLOR, SYNTH_COLORS[0], SYNTH_COLORS[2]]

    fig = go.Figure()
    for i, key in enumerate(score_keys):
        means = [scenario_stats[s].get(key, {}).get("mean", None) for s in scenarios]
        stds = [scenario_stats[s].get(key, {}).get("std", None) for s in scenarios]
        color = trace_colors[i % len(trace_colors)]
        fig.add_trace(go.Scatter(
            x=scenarios,
            y=means,
            name=score_labels.get(key, key),
            mode="markers",
            marker=dict(size=11, color=color),
            error_y=dict(type="data", array=stds, visible=True, color=color),
            text=[
                f"mean={m:.4f}<br>std={s:.4f}" if m is not None else ""
                for m, s in zip(means, stds)
            ],
            hovertemplate="%{x}<br>%{text}<extra>" + score_labels.get(key, key) + "</extra>",
        ))

    fig.update_layout(
        title="Summary across scenarios",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_tickangle=-35,
        height=420,
        margin=dict(l=40, r=20, t=50, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_meta_eval_scenario_grouped(
    base_scenario: str,
    size_results: Dict,
    score_keys: List[str],
    score_labels: Dict[str, str],
) -> go.Figure:
    """
    Per-scenario plot that overlays all sample sizes on a single figure.

    X-axis  — score labels (fidelity overall, missingness overall, composite).
    Color   — evaluation axis (score key); one color per key, dynamic palette.
    Shape   — sample size; one shape per size, dynamic palette.

    Individual replicate dots are shown small and semi-transparent.  The mean
    ± std is shown as a larger marker with an error bar.  The legend is placed
    to the right and split into **Axis** (colors) and **Sample size** (shapes)
    sections via invisible dummy traces.

    Parameters
    ----------
    base_scenario : str
        Base scenario name used in the chart title (e.g. ``"fidelity_1"``).
    size_results : dict
        Mapping ``{size_label: per_dataset_list}`` where *size_label* is an int
        or ``None`` (full dataset) and *per_dataset_list* is the ``per_dataset``
        list from the results dict.
    score_keys : list of str
        Score keys to plot.
    score_labels : dict
        Mapping from score key to display label.
    """
    _all_colors = [
        REAL_COLOR, SYNTH_COLORS[0], SYNTH_COLORS[2], SYNTH_COLORS[1],
        SYNTH_COLORS[3], SYNTH_COLORS[4], "#795548", "#607D8B",
    ]
    _all_symbols = [
        "circle", "square", "diamond", "triangle-up", "star",
        "cross", "hexagon", "pentagon", "triangle-down", "bowtie",
    ]

    all_sizes = sorted(size_results.keys(), key=lambda s: (s is None, s or 0))
    size_label_map = {s: ("full" if s is None else f"n={s:,}") for s in all_sizes}
    score_color = {sk: _all_colors[i % len(_all_colors)] for i, sk in enumerate(score_keys)}
    size_symbol  = {sz: _all_symbols[i % len(_all_symbols)] for i, sz in enumerate(all_sizes)}

    n_sizes = len(all_sizes)
    jitters = np.linspace(-0.25, 0.25, n_sizes) if n_sizes > 1 else [0.0]
    size_jitter = {sz: jitters[i] for i, sz in enumerate(all_sizes)}

    # x positions: one per score key
    x_labels = [score_labels.get(sk, sk) for sk in score_keys]
    x_pos = {sk: i for i, sk in enumerate(score_keys)}

    fig = go.Figure()

    # --- Legend section headers and dummy color/shape entries ----------------
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=0),
        name="<b>Axis</b>", showlegend=True,
        legendgroup="hdr_axis", hoverinfo="skip",
    ))
    for sk in score_keys:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=score_color[sk], size=10, symbol="circle"),
            name=score_labels.get(sk, sk), showlegend=True,
            legendgroup=f"axis_{sk}", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=0),
        name="<b>Sample size</b>", showlegend=True,
        legendgroup="hdr_size", hoverinfo="skip",
    ))
    for sz in all_sizes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="grey", size=10, symbol=size_symbol[sz]),
            name=size_label_map[sz], showlegend=True,
            legendgroup=f"size_{sz}", hoverinfo="skip",
        ))

    # --- Data traces ---------------------------------------------------------
    for sz in all_sizes:
        per_ds = size_results[sz]
        symbol = size_symbol[sz]
        for sk in score_keys:
            color = score_color[sk]
            vals = [row[sk] for row in per_ds if sk in row]
            if not vals:
                continue
            arr = np.array(vals)
            mean, std = float(np.mean(arr)), float(np.std(arr))
            xi = x_pos[sk] + size_jitter[sz]

            # Individual replicate dots
            fig.add_trace(go.Scatter(
                x=[xi] * len(vals), y=list(arr),
                mode="markers",
                marker=dict(color=color, size=6, symbol=symbol, opacity=0.35),
                showlegend=False, hoverinfo="skip",
            ))
            # Mean ± std
            fig.add_trace(go.Scatter(
                x=[xi], y=[mean],
                mode="markers",
                marker=dict(color=color, size=12, symbol=symbol,
                            line=dict(width=1, color="white")),
                error_y=dict(type="data", array=[std], visible=True, color=color),
                showlegend=False,
                text=[f"{score_labels.get(sk, sk)} | {size_label_map[sz]}<br>"
                      f"mean={mean:.4f}  std={std:.4f}"],
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        title=base_scenario,
        xaxis=dict(
            tickvals=list(x_pos.values()),
            ticktext=x_labels,
            range=[-0.6, len(score_keys) - 0.4],
        ),
        yaxis=dict(range=[0, 1], title="Score"),
        height=380,
        margin=dict(l=40, r=180, t=50, b=60),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    return fig


def plot_meta_eval_summary_grouped(
    results: Dict,
    score_keys: List[str],
    score_labels: Dict[str, str],
) -> go.Figure:
    """
    Summary plot for results that include sample-size suffixes.

    Groups result keys by their base scenario name and plots all sample sizes
    at the same x position.  **Color** encodes evaluation axis (score type);
    **marker shape** encodes sample size.  Both dimensions are dynamic —
    new colors are assigned for additional score keys, new shapes for additional
    sample sizes.  The legend is placed to the right and split into two sections
    (Axis / Sample size) using invisible dummy traces as section headers.

    Keys without a sample-size suffix (plain ``{scenario}`` keys) are treated
    as "full" and plotted alongside the suffixed ones.
    """
    import re

    # Expanded color palette — one per score key, dynamically extended
    _all_colors = [
        REAL_COLOR,       # blue
        SYNTH_COLORS[0],  # pink
        SYNTH_COLORS[2],  # orange
        SYNTH_COLORS[1],  # green
        SYNTH_COLORS[3],  # purple
        SYNTH_COLORS[4],  # cyan
        "#795548",        # brown
        "#607D8B",        # blue-grey
    ]
    # Expanded shape palette — one per sample size, dynamically extended
    _all_symbols = [
        "circle", "square", "diamond", "triangle-up",
        "star", "cross", "hexagon", "pentagon",
        "triangle-down", "triangle-left", "triangle-right", "bowtie",
    ]

    # --- Parse result keys → (base_scenario, size) --------------------------
    parsed: Dict[str, Dict] = {}  # base → {size → {score_key → {mean, std}}}
    for key, data in results.items():
        m = re.match(r"^(.+?)_n(\d+)$", key)
        if m:
            base, size = m.group(1), int(m.group(2))
        elif key.endswith("_full"):
            base, size = key[:-5], None
        else:
            base, size = key, None

        per_ds = data.get("per_dataset", [])
        for sk in score_keys:
            vals = [row[sk] for row in per_ds if sk in row]
            if not vals:
                continue
            arr = np.array(vals)
            parsed.setdefault(base, {}).setdefault(size, {})[sk] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

    # Ordered bases (preserve result dict order)
    bases = list(dict.fromkeys(
        re.match(r"^(.+?)_n\d+$", k).group(1) if re.match(r"^(.+?)_n\d+$", k)
        else (k[:-5] if k.endswith("_full") else k)
        for k in results
    ))

    # Ordered sample sizes: integers ascending, None (full) last
    all_sizes = sorted(
        {sz for b in parsed.values() for sz in b},
        key=lambda s: (s is None, s or 0),
    )
    size_label_map = {s: ("full" if s is None else f"n={s:,}") for s in all_sizes}

    # Assign colors (per score key) and symbols (per sample size)
    score_color = {sk: _all_colors[i % len(_all_colors)] for i, sk in enumerate(score_keys)}
    size_symbol  = {sz: _all_symbols[i % len(_all_symbols)] for i, sz in enumerate(all_sizes)}

    # Jitter so markers at same x don't overlap
    n_sizes = len(all_sizes)
    jitters = np.linspace(-0.3, 0.3, n_sizes) if n_sizes > 1 else [0.0]
    size_jitter = {sz: jitters[i] for i, sz in enumerate(all_sizes)}

    x_positions = {b: i for i, b in enumerate(bases)}

    fig = go.Figure()

    # --- Section header: "Axis" (invisible dummy, bold name in legend) -------
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=0),
        name="<b>Axis</b>",
        showlegend=True,
        legendgroup="header_axis",
        hoverinfo="skip",
    ))

    # --- Color legend entries (one per score key) ----------------------------
    for sk in score_keys:
        color = score_color[sk]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            name=score_labels.get(sk, sk),
            showlegend=True,
            legendgroup=f"axis_{sk}",
            hoverinfo="skip",
        ))

    # --- Section header: "Sample size" --------------------------------------
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=0),
        name="<b>Sample size</b>",
        showlegend=True,
        legendgroup="header_size",
        hoverinfo="skip",
    ))

    # --- Shape legend entries (one per sample size) --------------------------
    for sz in all_sizes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="grey", size=10, symbol=size_symbol[sz]),
            name=size_label_map[sz],
            showlegend=True,
            legendgroup=f"size_{sz}",
            hoverinfo="skip",
        ))

    # --- Data traces (no legend — represented by dummy entries above) --------
    for si, size in enumerate(all_sizes):
        symbol = size_symbol[size]
        for sk in score_keys:
            color = score_color[sk]
            xs, ys, errs, texts = [], [], [], []
            for base in bases:
                stats = parsed.get(base, {}).get(size, {}).get(sk)
                xs.append(x_positions[base] + size_jitter[size])
                if stats:
                    ys.append(stats["mean"])
                    errs.append(stats["std"])
                    texts.append(
                        f"{base} | {size_label_map[size]}<br>"
                        f"{score_labels.get(sk, sk)}<br>"
                        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                    )
                else:
                    ys.append(None)
                    errs.append(None)
                    texts.append("")

            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=color, size=11, symbol=symbol,
                    line=dict(width=1, color="white"),
                ),
                error_y=dict(type="data", array=errs, visible=True, color=color),
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            ))

    n_bases = len(bases)
    fig.update_layout(
        title="Summary across scenarios (by sample size)",
        xaxis=dict(
            tickvals=list(x_positions.values()),
            ticktext=list(x_positions.keys()),
            tickangle=-35,
            range=[-0.6, n_bases - 0.4],
        ),
        yaxis=dict(range=[0, 1], title="Score"),
        height=420,
        margin=dict(l=40, r=200, t=50, b=120),
        legend=dict(
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
        ),
    )
    return fig


def plot_meta_eval_sample_size_comparison(
    results: Dict,
    score_key: str,
    score_label: str,
) -> go.Figure:
    """
    Line plot showing mean score vs sample size for each base scenario.

    Expects result keys in the format ``{scenario}_n{size}`` or
    ``{scenario}_full`` (as produced when ``sample_sizes`` is set in the
    meta-eval config).  One line per base scenario, x-axis is sample size
    (ordered ascending, with full dataset last), y-axis is mean score with
    ± 1 std error band.

    Parameters
    ----------
    results : dict
        Full meta-evaluation results dict (keyed by result key).
    score_key : str
        Score key to plot, e.g. ``"fidelity_overall"``.
    score_label : str
        Display label for the y-axis / title.
    """
    import re

    # Parse each key into (base_scenario, sample_size) where sample_size is
    # an int or None for "full".
    parsed: Dict[str, Dict] = {}  # base_scenario → {size_label: {mean, std}}
    for key, data in results.items():
        m = re.match(r"^(.+?)_n(\d+)$", key)
        if m:
            base, size_label = m.group(1), int(m.group(2))
        elif key.endswith("_full"):
            base, size_label = key[:-5], None
        else:
            continue  # no sample-size suffix — skip

        per_ds = data.get("per_dataset", [])
        vals = [row[score_key] for row in per_ds if score_key in row]
        if not vals:
            continue
        arr = np.array(vals)
        parsed.setdefault(base, {})[size_label] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    if not parsed:
        fig = go.Figure()
        fig.update_layout(
            title=f"Sample size comparison — {score_label}",
            annotations=[dict(text="No sample-size results found", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)],
            height=300,
        )
        return fig

    # Collect all unique size labels and sort: integers ascending, None (full) last
    all_sizes = sorted(
        {sz for scenario_sizes in parsed.values() for sz in scenario_sizes},
        key=lambda s: (s is None, s or 0),
    )
    x_labels = [str(s) if s is not None else "full" for s in all_sizes]

    fig = go.Figure()
    for i, (base, size_data) in enumerate(sorted(parsed.items())):
        color = _synth_color(i)
        means = [size_data.get(s, {}).get("mean") for s in all_sizes]
        stds  = [size_data.get(s, {}).get("std", 0) for s in all_sizes]

        # Shaded std band
        y_upper = [m + s if m is not None else None for m, s in zip(means, stds)]
        y_lower = [m - s if m is not None else None for m, s in zip(means, stds)]
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=y_upper + y_lower[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.12,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=means,
            mode="lines+markers",
            name=base,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
            error_y=dict(type="data", array=stds, visible=True, color=color),
            hovertemplate=f"{base}<br>n=%{{x}}<br>mean=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Score vs sample size — {score_label}",
        xaxis=dict(title="Sample size", categoryorder="array", categoryarray=x_labels),
        yaxis=dict(range=[0, 1], title="Score"),
        height=420,
        margin=dict(l=40, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
