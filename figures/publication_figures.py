"""
Publication figures for the stdg-eval meta-evaluation.

NeurIPS formatting
------------------
- Text width:  3.25 in (single column) / 6.75 in (full width)
- Font:        STIXGeneral (matches Computer Modern / LaTeX output)
- Figure font: 9 pt (axis labels), 8 pt (tick labels / legend)
- DPI:         300 (raster export), vector PDF as primary
- Palette:     Wong (2011) colorblind-safe 8-colour palette
- Spines:      top/right removed on all axes

Output is written to figures/output/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# NeurIPS-compatible style
# ---------------------------------------------------------------------------
# Use STIX fonts (designed to match Computer Modern / LaTeX output) without
# requiring a full LaTeX installation.  Math is rendered via mathtext.
mpl.rcParams.update({
    "text.usetex":          False,
    "mathtext.fontset":     "stix",
    "font.family":          "STIXGeneral",
    # Sizes — NeurIPS body is 10pt; figures should match or be slightly smaller
    "font.size":            9,
    "axes.titlesize":       9,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.title_fontsize": 8,
    # Lines / markers
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "lines.linewidth":      1.0,
    "patch.linewidth":      0.5,
    # Spines
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Grid — off for publication cleanliness
    "axes.grid":            False,
    # Figure / saving
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
})

# NeurIPS column widths (inches)
COL_W  = 3.25   # single column
FULL_W = 6.75   # full text width

# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
# ---------------------------------------------------------------------------
WONG = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermil":   "#D55E00",
    "pink":     "#CC79A7",
}

PALETTE = {
    "fidelity":    WONG["blue"],
    "missingness": WONG["vermil"],
    "composite":   WONG["green"],
}

# Two-shade orange used for MIMIC-IV-ED bars
# (fidelity = darker, missingness = lighter, both clearly orange)
ORANGE_DARK  = "#C45500"   # deep burnt orange  — fidelity
ORANGE_LIGHT = "#F4A623"   # warm golden orange — missingness

# Sample-size colour ramp (blues, colorblind-safe within blue channel)
def _size_colors(n: int):
    return [mpl.cm.Blues(v) for v in np.linspace(0.35, 0.85, max(n, 1))]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Main results: Diabetes + MIMIC-IV-ED complete-case (CC)
RESULTS = {
    "Diabetes":         Path("/home/annabelzhu/stdg-eval/datasets/Diabetes/diabetes_meta_eval_results.json"),
    "MIMIC-IV-ED (CC)": Path("/home/annabelzhu/stdg-eval/datasets/mimic-iv-ed-2.2/meta_eval/results_trimmed.json"),
}

# Sensitivity / supplementary: MIMIC-IV-ED non-CC (preserves inherent missingness)
RESULTS_NONCC = {
    "MIMIC-IV-ED": Path("/home/annabelzhu/stdg-eval/datasets/mimic-iv-ed-2.2/meta_eval/results_nonCC_trimmed.json"),
}

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Display labels  (LaTeX-safe)
# ---------------------------------------------------------------------------
FIDELITY_LABELS = {
    "fidelity_1": r"F1" + "\n" + r"Low noise" + "\n" + r"(all)",
    "fidelity_2": r"F2" + "\n" + r"Low noise" + "\n" + r"(num.)",
    "fidelity_3": r"F3" + "\n" + r"High noise" + "\n" + r"(all)",
    "fidelity_4": r"F4" + "\n" + r"High noise" + "\n" + r"(num.)",
    "fidelity_5": r"F5" + "\n" + r"Bivariate",
}
MISSINGNESS_LABELS = {
    "missingness_1": r"M1" + "\n" + r"10\% MCAR",
    "missingness_2": r"M2" + "\n" + r"20\% MCAR",
    "missingness_3": r"M3" + "\n" + r"30\% MCAR",
    "missingness_4": r"M4" + "\n" + r"MAR",
    "missingness_5": r"M5" + "\n" + r"MNAR",
}

METRIC_DISPLAY = {
    "fidelity_univariate":              "Fid. Uni.",
    "fidelity_bivariate":               "Fid. Bi.",
    "fidelity_multivariate":            "Fid. Multi.",
    "missingness_rate":                 "Miss. Rate",
    "missingness_set_distribution":     "Miss. Pattern",
    "missingness_missing_auroc":        "Miss. AUROC",
    "missingness_dependency_structure": "Miss. Dep.",
    "composite_score":                  "Composite",
}
METRIC_COLS = list(METRIC_DISPLAY.keys())

# Drop constant columns (zero-variance) from correlation analysis
def _active_metric_cols(df: pd.DataFrame) -> list[str]:
    sub = df[METRIC_COLS].dropna()
    return [c for c in METRIC_COLS if sub[c].std() > 1e-9]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def build_per_dataset_df(results: dict) -> pd.DataFrame:
    rows = []
    for key, entry in results.items():
        m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
        scenario = m.group(1) if m else key
        size = int(m.group(2)) if m and m.group(2) else None
        axis = (
            "fidelity"    if scenario.startswith("fidelity")    else
            "missingness" if scenario.startswith("missingness") else
            "composite"
        )
        for row in entry.get("per_dataset", []):
            rows.append({
                "key": key, "scenario": scenario,
                "sample_size": size, "axis": axis,
                **{c: row.get(c) for c in METRIC_COLS},
            })
    return pd.DataFrame(rows)


def build_summary_df(results: dict) -> pd.DataFrame:
    rows = []
    for key, entry in results.items():
        m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
        scenario = m.group(1) if m else key
        size = int(m.group(2)) if m and m.group(2) else None
        axis = (
            "fidelity"    if scenario.startswith("fidelity")    else
            "missingness" if scenario.startswith("missingness") else
            "composite"
        )

        def _get(d, *keys, stat="mean"):
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return np.nan
            return d.get(stat, np.nan) if isinstance(d, dict) else d

        rows.append({
            "key": key, "scenario": scenario,
            "sample_size": size, "axis": axis,
            "fidelity_mean":    _get(entry, "fidelity",    "overall", stat="mean"),
            "fidelity_std":     _get(entry, "fidelity",    "overall", stat="std"),
            "missingness_mean": _get(entry, "missingness", "overall", stat="mean"),
            "missingness_std":  _get(entry, "missingness", "overall", stat="std"),
            "composite_mean":   _get(entry, "composite",              stat="mean"),
            "composite_std":    _get(entry, "composite",              stat="std"),
        })
    return pd.DataFrame(rows)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, name: str):
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _significance_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _compute_significance(
    all_results: dict[str, dict],
    score_col: str,
    all_scenarios: list[str],
) -> dict[str, str]:
    """
    For each scenario in all_scenarios, collect per-replicate scores from all
    datasets, run a two-sided Mann-Whitney U test between the first and second
    dataset, apply Bonferroni correction, and return a dict of
    scenario → significance star string (empty string if not significant or
    fewer than 2 datasets have data).
    """
    # Map score_col (summary col name) to per_dataset row key
    col_map = {
        "fidelity_mean":    "fidelity_overall",
        "missingness_mean": "missingness_overall",
        "composite_mean":   "composite_score",
    }
    per_rep_col = col_map.get(score_col, score_col.replace("_mean", "_overall"))

    # Collect per-replicate arrays: {scenario: {dataset_name: [values]}}
    dataset_names = list(all_results.keys())
    rep_data: dict[str, dict[str, list]] = {s: {} for s in all_scenarios}

    for dname, results in all_results.items():
        for key, entry in results.items():
            m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
            scenario = m.group(1) if m else key
            if scenario not in rep_data:
                continue
            vals = [row.get(per_rep_col) for row in entry.get("per_dataset", [])
                    if row.get(per_rep_col) is not None]
            rep_data[scenario].setdefault(dname, []).extend(vals)

    # Run tests only where both datasets have data
    testable = [s for s in all_scenarios
                if all(len(rep_data[s].get(d, [])) > 1 for d in dataset_names[:2])]

    n_tests = len(testable)
    raw_p: dict[str, float] = {}
    for s in testable:
        a = rep_data[s][dataset_names[0]]
        b = rep_data[s][dataset_names[1]]
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        raw_p[s] = p

    # Bonferroni correction
    stars: dict[str, str] = {}
    for s in all_scenarios:
        if s in raw_p:
            corrected_p = min(raw_p[s] * n_tests, 1.0)
            stars[s] = _significance_stars(corrected_p)
        else:
            stars[s] = ""
    return stars


def _add_significance_annotations(
    ax,
    x: np.ndarray,
    all_scenarios: list[str],
    bar_tops: dict[str, float],
    stars: dict[str, str],
):
    """Annotate significant scenarios with a star above the taller bar."""
    for xi, s in zip(x, all_scenarios):
        star = stars.get(s, "")
        if not star:
            continue
        top = bar_tops.get(s, 0) + 0.02
        ax.text(xi, top, star, ha="center", va="bottom", fontsize=8,
                color="#222222", fontweight="bold")


def _scenario_sort_key(s: str):
    return (
        0 if s == "baseline"             else
        1 if s.startswith("fidelity")    else
        2 if s.startswith("missingness") else 3,
        int(re.search(r"(\d+)$", s).group(1))
        if re.search(r"(\d+)$", s) and not s.startswith("composite") else 0,
        int(re.search(r"f(\d+)", s).group(1)) if re.search(r"f(\d+)", s) else 0,
        int(re.search(r"m(\d+)", s).group(1)) if re.search(r"m(\d+)", s) else 0,
    )


def _tick_label(s: str) -> str:
    if s == "baseline":
        return "Base"
    if s.startswith("fidelity_"):
        return f"F{s.split('_')[1]}"
    if s.startswith("missingness_"):
        return f"M{s.split('_')[1]}"
    mf = re.search(r"f(\d+)", s)
    mm = re.search(r"m(\d+)", s)
    return f"F{mf.group(1)}\nM{mm.group(1)}" if mf and mm else s


def _draw_group_dividers(ax, all_scenarios: list[str]):
    """Draw vertical dotted dividers and group labels between scenario type groups."""
    # Define ordered groups; only include those present in all_scenarios
    group_defs = [
        ("Baseline",    lambda s: s == "baseline"),
        ("Fidelity",    lambda s: s.startswith("fidelity")),
        ("Missingness", lambda s: s.startswith("missingness")),
        ("Composite",   lambda s: s.startswith("composite")),
    ]
    groups: list[tuple[int, int, str]] = []  # (start_idx, end_idx, label)
    pos = 0
    for label, pred in group_defs:
        n = sum(1 for s in all_scenarios if pred(s))
        if n > 0:
            groups.append((pos, pos + n, label))
            pos += n

    if len(groups) <= 1:
        return  # nothing to divide

    for i, (_, end, _) in enumerate(groups[:-1]):
        ax.axvline(end - 0.5, color="#888888", lw=0.6, ls=":", zorder=0)

    for start, end, label in groups:
        ax.annotate(label, xy=((start + end - 1) / 2, -0.18),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, fontstyle="italic")


def _draw_bars_on_ax(
    ax,
    all_results: dict[str, dict],
    score_col: str,
    std_col: str,
    scenario_filter,
    ylabel: str,
    panel_label: str,
    dataset_colors: list | None = None,
    show_legend: bool = True,
) -> list:
    """
    Grouped bar chart (one bar per dataset) for filtered scenarios on a given ax.
    Adds significance stars between the two datasets where both have data.
    """
    def _aggregate(results: dict) -> pd.DataFrame:
        df = build_summary_df(results)
        return (
            df.groupby("scenario")
            .agg(
                score_mean=(score_col, "mean"),
                score_std=(std_col, lambda x: float(np.sqrt((x ** 2).mean()))),
            )
            .reset_index()
        )

    dataset_aggs  = {name: _aggregate(res) for name, res in all_results.items()}
    dataset_names = list(dataset_aggs.keys())

    all_scenarios = sorted(
        {sc for agg in dataset_aggs.values() for sc in agg["scenario"]
         if scenario_filter(sc)},
        key=_scenario_sort_key,
    )

    n_scenarios = len(all_scenarios)
    n_datasets  = len(dataset_aggs)
    group_width = 0.7
    bar_w       = group_width / n_datasets
    offsets     = np.linspace(
        -(group_width - bar_w) / 2,
         (group_width - bar_w) / 2,
        n_datasets,
    )
    _colors         = dataset_colors if dataset_colors is not None else [WONG["blue"], WONG["vermil"]]
    dataset_hatches = ["", ""]

    x = np.arange(n_scenarios)
    legend_handles = []

    for di, (dname, agg) in enumerate(dataset_aggs.items()):
        lookup = agg.set_index("scenario")
        color  = _colors[di % len(_colors)]
        hatch  = dataset_hatches[di % len(dataset_hatches)]
        xpos   = x + offsets[di]

        means = np.array([lookup.loc[s, "score_mean"] if s in lookup.index else np.nan
                          for s in all_scenarios])
        stds  = np.array([lookup.loc[s, "score_std"]  if s in lookup.index else np.nan
                          for s in all_scenarios])

        ax.bar(xpos, means, bar_w,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.4)
        ax.errorbar(xpos, means, yerr=stds,
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2, capthick=0.8, zorder=5)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, hatch=hatch, ec="white", lw=0.4,
                           label=dname)
        )

    # Significance annotations
    if len(dataset_names) >= 2:
        stars = _compute_significance(all_results, score_col, all_scenarios)
        bar_tops: dict[str, float] = {}
        for dname, agg in dataset_aggs.items():
            lookup = agg.set_index("scenario")
            for s in all_scenarios:
                if s in lookup.index:
                    top = (float(np.nan_to_num(lookup.loc[s, "score_mean"], nan=0.0))
                           + float(np.nan_to_num(lookup.loc[s, "score_std"], nan=0.0)))
                    bar_tops[s] = max(bar_tops.get(s, 0.0), top)
        _add_significance_annotations(ax, x, all_scenarios, bar_tops, stars)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel(ylabel)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _despine(ax)

    # Baseline reference lines — one per dataset, coloured to match that dataset
    for di, (dname, agg) in enumerate(dataset_aggs.items()):
        lookup = agg.set_index("scenario")
        if "baseline" in lookup.index:
            y = float(lookup.loc["baseline", "score_mean"])
            if not np.isnan(y):
                color = _colors[di % len(_colors)]
                ax.axhline(y, color=color, lw=0.8, ls=":", zorder=3)
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=0.8, ls=":",
                           label=f"{dname} — Baseline")
                )

    _draw_group_dividers(ax, all_scenarios)

    if show_legend:
        ax.legend(handles=legend_handles, ncol=1, frameon=False,
                  fontsize=7.5, handlelength=1.2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    return legend_handles


def _draw_composite_on_ax(
    ax,
    all_results: dict[str, dict],
    scenario_filter,
    panel_label: str,
    show_legend: bool = True,
) -> list:
    """
    Stacked bar chart on a given ax: fidelity×0.5 (bottom) + missingness×0.5 (top)
    = composite height.  Error bars show composite std.
    Adds significance stars between the two datasets where both have data.
    """
    def _aggregate(results: dict) -> pd.DataFrame:
        df = build_summary_df(results)
        return (
            df.groupby("scenario")
            .agg(
                fidelity_mean=("fidelity_mean", "mean"),
                missingness_mean=("missingness_mean", "mean"),
                composite_std=("composite_std", lambda x: float(np.sqrt((x ** 2).mean()))),
            )
            .reset_index()
        )

    dataset_aggs  = {name: _aggregate(res) for name, res in all_results.items()}
    dataset_names = list(dataset_aggs.keys())

    all_scenarios = sorted(
        {sc for agg in dataset_aggs.values() for sc in agg["scenario"]
         if scenario_filter(sc)},
        key=_scenario_sort_key,
    )

    n_scenarios = len(all_scenarios)
    n_datasets  = len(dataset_aggs)
    group_width = 0.7
    bar_w       = group_width / n_datasets
    offsets     = np.linspace(
        -(group_width - bar_w) / 2,
         (group_width - bar_w) / 2,
        n_datasets,
    )
    dataset_hatches = ["", ""]

    segment_colors: dict[str, dict] = {}
    color_pairs = [
        {"fidelity": WONG["blue"],  "missingness": WONG["sky"]},
        {"fidelity": ORANGE_DARK,   "missingness": ORANGE_LIGHT},
    ]
    for i, dname in enumerate(dataset_names):
        segment_colors[dname] = color_pairs[i % len(color_pairs)]

    x = np.arange(n_scenarios)
    legend_handles = []

    for di, (dname, agg) in enumerate(dataset_aggs.items()):
        lookup = agg.set_index("scenario")
        colors = segment_colors[dname]
        hatch  = dataset_hatches[di % len(dataset_hatches)]
        xpos   = x + offsets[di]

        fid_vals  = np.array([lookup.loc[s, "fidelity_mean"]    * 0.5
                               if s in lookup.index else np.nan
                               for s in all_scenarios])
        miss_vals = np.array([lookup.loc[s, "missingness_mean"] * 0.5
                               if s in lookup.index else np.nan
                               for s in all_scenarios])
        comp_std  = np.array([lookup.loc[s, "composite_std"]
                               if s in lookup.index else np.nan
                               for s in all_scenarios])

        ax.bar(xpos, fid_vals, bar_w,
               color=colors["fidelity"],
               hatch=hatch, edgecolor="white", linewidth=0.4)
        ax.bar(xpos, miss_vals, bar_w, bottom=fid_vals,
               color=colors["missingness"],
               hatch=hatch, edgecolor="white", linewidth=0.4)
        ax.errorbar(xpos, fid_vals + miss_vals, yerr=comp_std,
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2, capthick=0.8, zorder=5)

        legend_handles += [
            plt.Rectangle((0, 0), 1, 1,
                           fc=colors["fidelity"],
                           hatch=hatch, ec="white", lw=0.4,
                           label=f"{dname} — Fidelity"),
            plt.Rectangle((0, 0), 1, 1,
                           fc=colors["missingness"],
                           hatch=hatch, ec="white", lw=0.4,
                           label=f"{dname} — Missingness"),
        ]

    # Significance annotations
    if len(dataset_names) >= 2:
        stars = _compute_significance(all_results, "composite_mean", all_scenarios)
        bar_tops: dict[str, float] = {}
        for dname, agg in dataset_aggs.items():
            lookup = agg.set_index("scenario")
            for s in all_scenarios:
                if s in lookup.index:
                    fid  = float(np.nan_to_num(lookup.loc[s, "fidelity_mean"],  nan=0.0)) * 0.5
                    miss = float(np.nan_to_num(lookup.loc[s, "missingness_mean"], nan=0.0)) * 0.5
                    std  = float(np.nan_to_num(lookup.loc[s, "composite_std"],   nan=0.0))
                    bar_tops[s] = max(bar_tops.get(s, 0.0), fid + miss + std)
        _add_significance_annotations(ax, x, all_scenarios, bar_tops, stars)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Composite score (0–1)")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _despine(ax)

    # Baseline reference lines — one per dataset, coloured to match that dataset
    for dname, agg in dataset_aggs.items():
        lookup = agg.set_index("scenario")
        if "baseline" in lookup.index:
            fid  = float(np.nan_to_num(lookup.loc["baseline", "fidelity_mean"],    nan=np.nan))
            miss = float(np.nan_to_num(lookup.loc["baseline", "missingness_mean"], nan=np.nan))
            if not (np.isnan(fid) and np.isnan(miss)):
                y = np.nansum([fid * 0.5, miss * 0.5])
                color = segment_colors[dname]["fidelity"]
                ax.axhline(y, color=color, lw=0.8, ls=":", zorder=3)
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=0.8, ls=":",
                           label=f"{dname} — Baseline")
                )

    _draw_group_dividers(ax, all_scenarios)

    if show_legend:
        ax.legend(handles=legend_handles, ncol=1, frameon=False,
                  fontsize=7.5, handlelength=1.2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    return legend_handles


def fig_main(all_results: dict[str, dict]):
    """
    Main Figure 1 — three panels in a single row:

    (a) Fidelity score    for baseline + fidelity + missingness scenarios
    (b) Missingness score for baseline + fidelity + missingness scenarios
    (c) Composite score (stacked) for the same scenarios

    A single shared legend is drawn to the right of panel (c).
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W * 1.5, 3.2), layout="constrained")

    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")

    _draw_bars_on_ax(
        axes[0], all_results,
        score_col="fidelity_mean", std_col="fidelity_std",
        scenario_filter=single,
        ylabel="Fidelity score (0–1)",
        panel_label="(a)",
        dataset_colors=[WONG["blue"], ORANGE_DARK],
        show_legend=False,
    )
    _draw_bars_on_ax(
        axes[1], all_results,
        score_col="missingness_mean", std_col="missingness_std",
        scenario_filter=single,
        ylabel="Missingness score (0–1)",
        panel_label="(b)",
        dataset_colors=[WONG["sky"], ORANGE_LIGHT],
        show_legend=False,
    )
    handles = _draw_composite_on_ax(
        axes[2], all_results,
        scenario_filter=single,
        panel_label="(c)",
        show_legend=False,
    )

    fig.legend(
        handles=handles, ncol=len(handles), frameon=False,
        fontsize=7.5, handlelength=1.2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )

    _save(fig, "fig1_main")


def fig_appendix_composite(all_results: dict[str, dict]):
    """
    Appendix figure — stacked composite bars for all composite (F×M) scenarios.
    """
    composite_scenarios: set[str] = set()
    for res in all_results.values():
        for key in res:
            m = re.match(r"^(.+?)(?:_n\d+)?$", key)
            sc = m.group(1) if m else key
            if sc == "baseline" or sc.startswith("composite"):
                composite_scenarios.add(sc)

    fig_w = max(FULL_W, len(composite_scenarios) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2), layout="constrained")

    _draw_composite_on_ax(
        ax, all_results,
        scenario_filter=lambda s: s == "baseline" or s.startswith("composite"),
        panel_label="",
    )
    ax.set_title(
        r"Composite scenario scores  (composite $=$ fidelity $\times$ 0.5 $+$ missingness $\times$ 0.5)",
        fontsize=9,
    )

    _save(fig, "fig_appendix_composite")


def _print_summary(all_results: dict[str, dict]) -> None:
    """Print a per-dataset summary of scenarios and replicate counts."""
    print("\n" + "=" * 60)
    print("Loaded results summary")
    print("=" * 60)
    for dname, results in all_results.items():
        print(f"\n{dname}:")
        # Group scenario keys by base scenario name and sample size
        from collections import defaultdict
        groups: dict[tuple, list[str]] = defaultdict(list)
        for key, entry in results.items():
            m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
            scenario = m.group(1) if m else key
            sample_size = entry.get("sample_size")
            n_rep = entry.get("n_datasets", len(entry.get("per_dataset", [])))
            groups[(sample_size, n_rep)].append(scenario)

        for (sample_size, n_rep), scenarios in sorted(
            groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])
        ):
            size_str = f"n={sample_size:,}" if sample_size is not None else "n=full"
            scenario_list = ", ".join(sorted(scenarios))
            print(f"  {n_rep} replicates  {size_str}  —  {scenario_list}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    all_results = {}
    for name, path in RESULTS.items():
        res = load_results(path)
        if res is not None:
            all_results[name] = res

    _print_summary(all_results)

    print("Generating fig1_main (panels a, b, c)...")
    fig_main(all_results)

    print("Generating fig_appendix_composite...")
    fig_appendix_composite(all_results)

    print("Done.")

