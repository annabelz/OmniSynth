"""
Metric correlation figures for the stdg-eval meta-evaluation (Diabetes dataset).

Figures produced
----------------
fig_corr_all         — Full correlation matrix (all metrics, all scenarios, all sample sizes)
fig_corr_fidelity    — Fidelity metrics only
fig_corr_missingness — Missingness metrics only
fig_corr_by_size     — One full matrix per sample size (n=100, n=500, n=768/full)

Run:
    python figures/correlation_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# NeurIPS-compatible style  (matches publication_figures.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex":           False,
    "mathtext.fontset":      "stix",
    "font.family":           "STIXGeneral",
    "font.size":             9,
    "axes.titlesize":        9,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "lines.linewidth":       1.0,
    "patch.linewidth":       0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
})

COL_W  = 3.25
FULL_W = 6.75

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DIABETES_RESULTS = Path(__file__).parents[1] / "datasets/diabetes/diabetes_pub_results.json"
MIMIC_RESULTS    = Path(__file__).parents[1] / "datasets/mimic-iv_ed/mimiciv_pub_results.json"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Metric columns and display labels
# ---------------------------------------------------------------------------
# Full ordered list — includes TVD and Contingency (MIMIC only; absent from Diabetes)
FIDELITY_COLS = [
    "fidelity_wasserstein",
    "fidelity_tvd",
    "fidelity_hellinger",
    "fidelity_spearman",
    "fidelity_contingency",
    "fidelity_pcd",
    "fidelity_auc_roc",
    "fidelity_propensity_mse",
    "fidelity_crcl_rs",
    "fidelity_crcl_sr",
]

MISSINGNESS_COLS = [
    "missingness_rate",
    "missingness_set_distribution",
    "missingness_missing_auroc",
    "missingness_dependency_structure",
]

ALL_METRIC_COLS = FIDELITY_COLS + MISSINGNESS_COLS

# Short axis labels for the matrix
METRIC_LABELS = {
    "fidelity_wasserstein":              "Wasserstein",
    "fidelity_tvd":                      "TVD",
    "fidelity_hellinger":                "Hellinger",
    "fidelity_spearman":                 "Spearman",
    "fidelity_contingency":              "Contingency",
    "fidelity_pcd":                      "PCD",
    "fidelity_auc_roc":                  "AUC-ROC",
    "fidelity_propensity_mse":           "Prop. MSE",
    "fidelity_crcl_rs":                  "CrCl-RS",
    "fidelity_crcl_sr":                  "CrCl-SR",
    "missingness_rate":                  "Miss. Rate",
    "missingness_set_distribution":      "Miss. Pattern",
    "missingness_missing_auroc":         "Miss. AUROC",
    "missingness_dependency_structure":  "Miss. Dep.",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_flat_df(path: Path) -> pd.DataFrame:
    """Load results JSON → flat DataFrame with one row per replicate."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for key, entry in data.items():
        for row in entry["per_dataset"]:
            row = dict(row)
            row["scenario_key"] = key
            row["sample_size"] = entry["sample_size"]  # authoritative from top level
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop constant columns (zero variance) — they produce NaN correlations."""
    sub = df[cols].dropna()
    active = [c for c in cols if c in sub.columns and sub[c].std() > 1e-9]
    dropped = [c for c in cols if c not in active]
    if dropped:
        print(f"  [info] Excluded constant metric(s): {', '.join(METRIC_LABELS.get(c, c) for c in dropped)}")
    return active


def _filter_size(df: pd.DataFrame, size) -> pd.DataFrame:
    """Filter by sample_size, handling None → NaN correctly."""
    if size is None:
        return df[df["sample_size"].isna()]
    return df[df["sample_size"] == size]


def _kappa_matrix(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Pairwise Cohen's kappa between metric columns.

    Each column is binarised at its median (≥ median → 1, else → 0) so that
    every metric labels roughly half the replicates as 'high quality'.  Kappa
    then measures how much two metrics agree on those labels beyond chance.
    Diagonal entries are set to 1.0 (perfect self-agreement).
    """
    n = len(cols)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            med_i = sub[cols[i]].median()
            med_j = sub[cols[j]].median()
            a = (sub[cols[i]] >= med_i).astype(int).values
            b = (sub[cols[j]] >= med_j).astype(int).values
            try:
                k = cohen_kappa_score(a, b)
            except Exception:
                k = 0.0
            mat[i, j] = mat[j, i] = k
    return pd.DataFrame(mat, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Core correlation matrix drawing
# ---------------------------------------------------------------------------

def _draw_corr_matrix(
    ax: plt.Axes,
    df: pd.DataFrame,
    cols: list[str],
    title: str = "",
    show_colorbar: bool = True,
    fig: plt.Figure | None = None,
    annot_fontsize: float = 6.5,
    method: str = "pearson",
) -> None:
    """
    Draw a metric agreement matrix heatmap on *ax*.

    method="pearson" — Pearson correlation (default).
    method="kappa"   — Cohen's kappa with median binarisation.

    Cells are coloured on a diverging blue→white→red scale centred at 0.
    """
    labels = [METRIC_LABELS.get(c, c) for c in cols]
    sub = df[cols].dropna()
    if method == "kappa":
        corr = _kappa_matrix(sub, cols)
    else:
        corr = sub.corr(method=method)

    n = len(cols)
    cmap = mpl.cm.RdBu_r

    # Draw filled squares manually so we control size exactly
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = cmap((val + 1) / 2)   # map [-1,1] → [0,1]
            rect = mpl.patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.4,
            )
            ax.add_patch(rect)
            # Text colour: white on dark, black on light
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = "white" if lum < 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=annot_fontsize,
                color=txt_color,
            )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)   # y increases downward
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    if title:
        ax.set_title(title, fontsize=9, pad=4)

    if show_colorbar and fig is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(-1, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label("Cohen's κ" if method == "kappa" else "Pearson r", fontsize=8)
        cbar.ax.tick_params(labelsize=7)


def _draw_corr_matrix_with_groups(
    ax: plt.Axes,
    df: pd.DataFrame,
    cols: list[str],
    title: str = "",
    show_colorbar: bool = True,
    fig: plt.Figure | None = None,
    annot_fontsize: float = 6.5,
    group_boundary: int | None = None,
    method: str = "pearson",
) -> None:
    """
    Like _draw_corr_matrix but draws a visible boundary line between fidelity
    and missingness blocks when *group_boundary* is set (index of first
    missingness column).
    """
    _draw_corr_matrix(ax, df, cols, title, show_colorbar, fig, annot_fontsize, method)

    if group_boundary is not None:
        b = group_boundary - 0.5
        n = len(cols)
        lw = 1.2
        color = "#333333"
        ax.plot([b, b], [-0.5, n - 0.5], color=color, lw=lw, clip_on=False)
        ax.plot([-0.5, n - 0.5], [b, b], color=color, lw=lw, clip_on=False)


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_corr_all(df: pd.DataFrame, method: str = "pearson") -> None:
    """Full agreement matrix — all metrics, all scenarios, all sample sizes."""
    prefix = "kappa" if method == "kappa" else "corr"
    cols = _active_cols(df, [c for c in ALL_METRIC_COLS if c in df.columns])
    n = len(cols)
    size = n * 0.55 + 1.2
    fig, ax = plt.subplots(figsize=(size, size))

    n_fid = sum(1 for c in cols if c.startswith("fidelity_"))
    label = "Cohen's κ" if method == "kappa" else "Pearson r"
    _draw_corr_matrix_with_groups(
        ax, df, cols,
        title=f"Metric agreement ({label}) — all scenarios",
        show_colorbar=True, fig=fig,
        group_boundary=n_fid,
        method=method,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{prefix}_all.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


def fig_corr_fidelity(df: pd.DataFrame, method: str = "pearson") -> None:
    """Fidelity metrics only."""
    prefix = "kappa" if method == "kappa" else "corr"
    cols = _active_cols(df, [c for c in FIDELITY_COLS if c in df.columns])
    n = len(cols)
    size = n * 0.58 + 1.0
    fig, ax = plt.subplots(figsize=(size, size))

    label = "Cohen's κ" if method == "kappa" else "Pearson r"
    _draw_corr_matrix(
        ax, df, cols,
        title=f"Fidelity metric agreement ({label})",
        show_colorbar=True, fig=fig,
        method=method,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{prefix}_fidelity.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


def fig_corr_missingness(df: pd.DataFrame, method: str = "pearson") -> None:
    """Missingness metrics only."""
    prefix = "kappa" if method == "kappa" else "corr"
    cols = _active_cols(df, [c for c in MISSINGNESS_COLS if c in df.columns])
    n = len(cols)
    size = n * 0.7 + 1.0
    fig, ax = plt.subplots(figsize=(size, size))

    label = "Cohen's κ" if method == "kappa" else "Pearson r"
    _draw_corr_matrix(
        ax, df, cols,
        title=f"Missingness metric agreement ({label})",
        show_colorbar=True, fig=fig,
        method=method,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{prefix}_missingness.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


def fig_corr_by_size(df_d: pd.DataFrame, df_m: pd.DataFrame, method: str = "pearson") -> None:
    """
    Two rows of three matrices each.
    Row 0 (Diabetes):    n=100, n=500, n=768 (full).
    Row 1 (MIMIC-IV-ED): n=500, n=1,000, n=10,000.
    """
    prefix = "kappa" if method == "kappa" else "corr"
    rows = [
        ("Diabetes",         df_d, [(100, "n = 100"), (500, "n = 500"), (None, "n = 768 (full)")]),
        ("MIMIC-IV-ED (CC)", df_m, [(500, "n = 500"), (1000, "n = 1,000"), (10000, "n = 10,000")]),
    ]

    row_cols = [
        _active_cols(df, [c for c in ALL_METRIC_COLS if c in df.columns])
        for _, df, _ in rows
    ]

    max_n = max(len(c) for c in row_cols)
    cell = max_n * 0.52 + 1.0

    fig, axes = plt.subplots(
        2, 3,
        figsize=(cell * 3 + 0.6, cell * 2 + 0.3),
    )

    for row_idx, ((dataset_label, df, sizes), cols) in enumerate(zip(rows, row_cols)):
        for col_idx, (size, title) in enumerate(sizes):
            ax = axes[row_idx, col_idx]
            sub = _filter_size(df, size)
            panel_cols = _active_cols(sub, cols)
            panel_n_fid = sum(1 for c in panel_cols if c.startswith("fidelity_"))
            is_last_col = (col_idx == len(sizes) - 1)
            _draw_corr_matrix_with_groups(
                ax, sub, panel_cols,
                title=title,
                show_colorbar=is_last_col,
                fig=fig if is_last_col else None,
                annot_fontsize=6.0,
                group_boundary=panel_n_fid,
                method=method,
            )
            if col_idx != 0:
                ax.set_yticklabels([])

        axes[row_idx, 0].set_ylabel(dataset_label, fontsize=9, labelpad=8)

    fig.tight_layout(w_pad=0.5, h_pad=1.2)
    out = OUT_DIR / f"{prefix}_by_size.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


def fig_corr_side_by_side(df_d: pd.DataFrame, df_m: pd.DataFrame, method: str = "pearson") -> None:
    """
    Two full agreement matrices side-by-side: Diabetes (left), MIMIC (right).

    Each panel uses only the metrics active in that dataset, so columns differ
    between panels (e.g. TVD and Contingency appear only in MIMIC).
    """
    prefix = "kappa" if method == "kappa" else "corr"
    panels = [
        ("Diabetes",         df_d),
        ("MIMIC-IV-ED (CC)", df_m),
    ]

    panel_info = []
    for title, df in panels:
        cols = _active_cols(df, [c for c in ALL_METRIC_COLS if c in df.columns])
        n_fid = sum(1 for c in cols if c.startswith("fidelity_"))
        panel_info.append((title, df, cols, n_fid))

    max_n = max(len(cols) for _, _, cols, _ in panel_info)
    cell  = max_n * 0.54 + 0.8

    fig, axes = plt.subplots(1, 2, figsize=(cell * 2 + 1.2, cell))

    for ax, (title, df, cols, n_fid) in zip(axes, panel_info):
        is_right = (ax is axes[1])
        _draw_corr_matrix_with_groups(
            ax, df, cols,
            title=title,
            show_colorbar=is_right,
            fig=fig if is_right else None,
            annot_fontsize=6.0,
            group_boundary=n_fid,
            method=method,
        )

    fig.tight_layout(w_pad=1.0)
    out = OUT_DIR / f"{prefix}_side_by_side.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


def fig_corr_pooled(df_d: pd.DataFrame, df_m: pd.DataFrame, method: str = "pearson") -> None:
    """
    Single agreement matrix with replicates from both datasets combined.

    Metrics absent from one dataset (TVD, Contingency in Diabetes) have NaN
    for those rows; pearson corr() uses pairwise complete observations.
    For kappa, only rows where both columns are non-null are used.
    """
    prefix = "kappa" if method == "kappa" else "corr"
    label  = "Cohen's κ" if method == "kappa" else "Pearson r"
    df_d = df_d.copy()
    df_m = df_m.copy()
    df_d["dataset"] = "Diabetes"
    df_m["dataset"] = "MIMIC-IV-ED (CC)"
    df_combined = pd.concat([df_d, df_m], ignore_index=True)

    cols = _active_cols(df_combined, [c for c in ALL_METRIC_COLS if c in df_combined.columns])
    n_fid = sum(1 for c in cols if c.startswith("fidelity_"))
    n = len(cols)
    size = n * 0.55 + 1.2

    fig, ax = plt.subplots(figsize=(size, size))
    _draw_corr_matrix_with_groups(
        ax, df_combined, cols,
        title=f"Metric agreement ({label}) — pooled (Diabetes + MIMIC-IV-ED CC)",
        show_colorbar=True, fig=fig,
        group_boundary=n_fid,
        method=method,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{prefix}_pooled.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading Diabetes results...")
    df_d = _load_flat_df(DIABETES_RESULTS)
    print(f"  {len(df_d):,} replicate rows  ({df_d['scenario_key'].nunique()} scenario×size combinations)")

    print("Loading MIMIC-IV-ED results...")
    df_m = _load_flat_df(MIMIC_RESULTS)
    print(f"  {len(df_m):,} replicate rows  ({df_m['scenario_key'].nunique()} scenario×size combinations)")

    for method in ("pearson", "kappa"):
        label = "Kappa" if method == "kappa" else "Correlation"
        print(f"\nGenerating {label} matrices:")
        print("  Diabetes-only:")
        fig_corr_all(df_d, method=method)
        fig_corr_fidelity(df_d, method=method)
        fig_corr_missingness(df_d, method=method)
        fig_corr_by_size(df_d, df_m, method=method)

        print("  Cross-dataset:")
        fig_corr_side_by_side(df_d, df_m, method=method)
        fig_corr_pooled(df_d, df_m, method=method)

    print("\nDone.")


if __name__ == "__main__":
    main()
