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
