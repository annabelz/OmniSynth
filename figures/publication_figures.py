"""
Publication figures for the OmniSynth meta-evaluation.

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
    "PID":         Path(__file__).parents[1] / "datasets/diabetes/diabetes_pub_results.json",
    "MIMIC-IV-ED (CC)": Path(__file__).parents[1] / "datasets/mimic-iv_ed/mimiciv_pub_results.json",
}

# Sensitivity / supplementary: MIMIC-IV-ED non-CC (preserves inherent missingness)
RESULTS_NONCC = {
    "MIMIC-IV-ED": Path("/home/annabelzhu/OmniSynth/datasets/mimic-iv-ed-2.2/meta_eval/results_nonCC_trimmed.json"),
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
    path = OUT_DIR / f"{name}.png"
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


def _draw_group_dividers(ax, all_scenarios: list[str], label_offset: float = -0.16):
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
        ax.annotate(label, xy=((start + end - 1) / 2, label_offset),
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

    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

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

    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

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
    Appendix figure — 3 rows × 1 column for composite (F×M) + baseline scenarios.
    Both datasets appear as side-by-side bars within each scenario group, using
    the same colours and legend as fig1_main.

    Rows : (a) fidelity score, (b) missingness score, (c) composite score
    """
    composite = lambda s: s == "baseline" or s.startswith("composite")

    fig, axes = plt.subplots(3, 1, figsize=(FULL_W * 1.5, 9.0), layout="constrained")

    _draw_bars_on_ax(
        axes[0], all_results,
        score_col="fidelity_mean", std_col="fidelity_std",
        scenario_filter=composite,
        ylabel="Fidelity score (0–1)",
        panel_label="(a)",
        dataset_colors=[WONG["blue"], ORANGE_DARK],
        show_legend=False,
    )
    _draw_bars_on_ax(
        axes[1], all_results,
        score_col="missingness_mean", std_col="missingness_std",
        scenario_filter=composite,
        ylabel="Missingness score (0–1)",
        panel_label="(b)",
        dataset_colors=[WONG["sky"], ORANGE_LIGHT],
        show_legend=False,
    )
    handles = _draw_composite_on_ax(
        axes[2], all_results,
        scenario_filter=composite,
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

    _save(fig, "fig_appendix_composite")


def _compute_pairwise_significance(
    results: dict,
    per_rep_col: str,
    all_scenarios: list[str],
    sample_sizes: list,
) -> dict[str, dict[tuple, str]]:
    """
    For each scenario, run pairwise Mann-Whitney U tests between every pair of
    sample sizes.  Apply Bonferroni correction across all tests (all scenarios ×
    all pairs).  Return {scenario: {(sz_a, sz_b): star_str}}.
    """
    # Collect per-replicate values: {scenario: {sample_size: [values]}}
    rep_data: dict[str, dict] = {s: {} for s in all_scenarios}
    for key, entry in results.items():
        m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
        scenario    = m.group(1) if m else key
        sample_size = entry.get("sample_size")
        if scenario not in rep_data:
            continue
        vals = [row.get(per_rep_col) for row in entry.get("per_dataset", [])
                if row.get(per_rep_col) is not None]
        rep_data[scenario].setdefault(sample_size, []).extend(vals)

    import itertools
    pairs = list(itertools.combinations(sample_sizes, 2))

    # Count testable pairs across all scenarios for Bonferroni denominator
    testable: list[tuple[str, tuple]] = []
    for s in all_scenarios:
        for pa, pb in pairs:
            if (len(rep_data[s].get(pa, [])) > 1 and
                    len(rep_data[s].get(pb, [])) > 1):
                testable.append((s, (pa, pb)))

    n_tests = len(testable)
    raw_p: dict[tuple, float] = {}
    for s, (pa, pb) in testable:
        a = rep_data[s][pa]
        b = rep_data[s][pb]
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        raw_p[(s, pa, pb)] = p

    result: dict[str, dict[tuple, str]] = {s: {} for s in all_scenarios}
    for s, (pa, pb) in testable:
        corrected = min(raw_p[(s, pa, pb)] * max(n_tests, 1), 1.0)
        star = _significance_stars(corrected)
        if star:
            result[s][(pa, pb)] = star
    return result


def _add_bracket_annotations(
    ax,
    all_scenarios: list[str],
    sample_sizes: list,
    offsets: np.ndarray,
    bar_tops: dict[str, np.ndarray],
    sig: dict[str, dict[tuple, str]],
    bar_w: float,
):
    """
    Draw significance brackets between pairs of bars within each scenario group.
    Brackets are stacked upward so they don't overlap.
    bar_tops: {scenario: array of top heights, one per sample_size index}
    """
    x = np.arange(len(all_scenarios))
    bracket_h  = 0.025   # vertical height of bracket tick
    bracket_gap = 0.018  # gap between stacked brackets

    for xi, s in enumerate(all_scenarios):
        pairs_for_s = sig.get(s, {})
        if not pairs_for_s:
            continue

        # Sort pairs by distance (adjacent first), then by left index
        sorted_pairs = sorted(
            pairs_for_s.items(),
            key=lambda kv: (
                abs(sample_sizes.index(kv[0][1]) - sample_sizes.index(kv[0][0])),
                sample_sizes.index(kv[0][0]),
            ),
        )

        # Track the highest bracket top per bar position so we can stack
        tops = bar_tops[s].copy()
        level: dict[int, float] = {i: tops[i] for i in range(len(sample_sizes))}

        for (pa, pb), star in sorted_pairs:
            ia = sample_sizes.index(pa)
            ib = sample_sizes.index(pb)
            xa = xi + offsets[ia]
            xb = xi + offsets[ib]
            # Bracket base sits above the taller of the two bars (+ any prior brackets)
            base = max(level[ia], level[ib]) + bracket_gap
            tip  = base + bracket_h

            tick = bracket_h * 0.6   # constant vertical tick length
            ax.plot([xa, xa, xb, xb], [tip - tick, tip, tip, tip - tick],
                    color="#444444", lw=0.7, clip_on=False)
            ax.text((xa + xb) / 2, tip, star,
                    ha="center", va="bottom", fontsize=7,
                    color="#222222", fontweight="bold")

            # Raise level for both bars so next bracket clears this one
            new_top = tip + bracket_h * 0.5
            for i in range(min(ia, ib), max(ia, ib) + 1):
                level[i] = max(level[i], new_top)


def _draw_samplesize_bars_on_ax(
    ax,
    results: dict,
    score_col: str,
    std_col: str,
    scenario_filter,
    ylabel: str,
    panel_label: str,
    show_legend: bool = True,
    cmap=None,
) -> list:
    """
    Grouped bar chart for a single dataset's results, one bar per sample size.
    Baseline gets a dotted horizontal reference line per sample size.
    """
    df = build_summary_df(results)
    df = df[df["scenario"].apply(scenario_filter)].copy()

    all_scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)
    sample_sizes  = sorted(df["sample_size"].unique(), key=lambda v: v if v is not None else -1)
    n_scenarios   = len(all_scenarios)
    n_sizes       = len(sample_sizes)

    _cmap = cmap if cmap is not None else mpl.cm.Blues
    colors = [_cmap(v) for v in np.linspace(0.35, 0.85, max(n_sizes, 1))]
    group_width = 0.7
    bar_w       = group_width / n_sizes
    offsets     = np.linspace(
        -(group_width - bar_w) / 2,
         (group_width - bar_w) / 2,
        n_sizes,
    )

    x = np.arange(n_scenarios)
    legend_handles = []

    # Track bar tops per scenario for bracket placement: shape (n_scenarios, n_sizes)
    bar_top_matrix = np.full((n_scenarios, n_sizes), np.nan)

    for si, (sz, color) in enumerate(zip(sample_sizes, colors)):
        sub = df[df["sample_size"] == sz].set_index("scenario")
        xpos  = x + offsets[si]
        means = np.array([sub.loc[s, score_col] if s in sub.index else np.nan
                          for s in all_scenarios])
        stds  = np.array([sub.loc[s, std_col]   if s in sub.index else np.nan
                          for s in all_scenarios])
        size_label = f"n={sz:,}" if sz is not None else "n=full"

        ax.bar(xpos, means, bar_w, color=color, edgecolor="white", linewidth=0.4)
        ax.errorbar(xpos, means, yerr=stds,
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2, capthick=0.8, zorder=5)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", lw=0.4, label=size_label)
        )

        bar_top_matrix[:, si] = np.where(
            np.isnan(means) | np.isnan(stds), means, means + stds
        )

        # Baseline reference line for this sample size
        if "baseline" in sub.index:
            y = float(sub.loc["baseline", score_col])
            if not np.isnan(y):
                ax.axhline(y, color=color, lw=0.8, ls=":", zorder=3)

    # Significance brackets between sample-size pairs within each scenario
    col_map = {
        "fidelity_mean":    "fidelity_overall",
        "missingness_mean": "missingness_overall",
    }
    per_rep_col = col_map.get(score_col, score_col.replace("_mean", "_overall"))
    sig = _compute_pairwise_significance(results, per_rep_col, all_scenarios, sample_sizes)
    bar_tops_dict = {s: bar_top_matrix[i] for i, s in enumerate(all_scenarios)}
    _add_bracket_annotations(ax, all_scenarios, sample_sizes, offsets,
                             bar_tops_dict, sig, bar_w)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _despine(ax)
    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

    if show_legend:
        ax.legend(handles=legend_handles, ncol=1, frameon=False,
                  fontsize=7.5, handlelength=1.2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    return legend_handles


def _draw_samplesize_composite_on_ax(
    ax,
    results: dict,
    scenario_filter,
    panel_label: str,
    show_legend: bool = True,
    cmap=None,
) -> list:
    """
    Stacked composite bars for a single dataset, one bar per sample size.
    Fidelity×0.5 (darker shade of size colour) stacked under missingness×0.5 (lighter).
    Baseline reference line per sample size.
    """
    df = build_summary_df(results)
    df = df[df["scenario"].apply(scenario_filter)].copy()

    all_scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)
    sample_sizes  = sorted(df["sample_size"].unique(), key=lambda v: v if v is not None else -1)
    n_scenarios   = len(all_scenarios)
    n_sizes       = len(sample_sizes)

    # Two-shade ramp: darker for fidelity, lighter for missingness
    _cmap = cmap if cmap is not None else mpl.cm.Blues
    colors_dark  = [_cmap(v) for v in np.linspace(0.35, 0.85, max(n_sizes, 1))]
    colors_light = [mpl.colors.to_rgba(c, alpha=0.55) for c in colors_dark]

    group_width = 0.7
    bar_w       = group_width / n_sizes
    offsets     = np.linspace(
        -(group_width - bar_w) / 2,
         (group_width - bar_w) / 2,
        n_sizes,
    )

    x = np.arange(n_scenarios)
    legend_handles = []
    bar_top_matrix = np.full((n_scenarios, n_sizes), np.nan)

    for si, sz in enumerate(sample_sizes):
        sub = df[df["sample_size"] == sz].set_index("scenario")
        xpos      = x + offsets[si]
        size_label = f"n={sz:,}" if sz is not None else "n=full"
        c_dark    = colors_dark[si]
        c_light   = colors_light[si]

        fid_vals  = np.array([sub.loc[s, "fidelity_mean"]    * 0.5
                               if s in sub.index else np.nan for s in all_scenarios])
        miss_vals = np.array([sub.loc[s, "missingness_mean"] * 0.5
                               if s in sub.index else np.nan for s in all_scenarios])
        comp_std  = np.array([sub.loc[s, "composite_std"]
                               if s in sub.index else np.nan for s in all_scenarios])

        ax.bar(xpos, fid_vals, bar_w, color=c_dark,  edgecolor="white", linewidth=0.4)
        ax.bar(xpos, miss_vals, bar_w, bottom=fid_vals,
               color=c_light, edgecolor="white", linewidth=0.4)
        ax.errorbar(xpos, fid_vals + miss_vals, yerr=comp_std,
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2, capthick=0.8, zorder=5)

        legend_handles += [
            plt.Rectangle((0, 0), 1, 1, fc=c_dark,  ec="white", lw=0.4,
                           label=f"{size_label} — Fidelity"),
            plt.Rectangle((0, 0), 1, 1, fc=c_light, ec="white", lw=0.4,
                           label=f"{size_label} — Missingness"),
        ]

        # Baseline reference line for this sample size
        if "baseline" in sub.index:
            fid  = float(np.nan_to_num(sub.loc["baseline", "fidelity_mean"],    nan=np.nan))
            miss = float(np.nan_to_num(sub.loc["baseline", "missingness_mean"], nan=np.nan))
            if not (np.isnan(fid) and np.isnan(miss)):
                y = np.nansum([fid * 0.5, miss * 0.5])
                ax.axhline(y, color=c_dark, lw=0.8, ls=":", zorder=3)

        bar_top_matrix[:, si] = np.where(
            np.isnan(fid_vals + miss_vals) | np.isnan(comp_std),
            fid_vals + miss_vals,
            fid_vals + miss_vals + comp_std,
        )

    # Significance brackets between sample-size pairs within each scenario
    sig = _compute_pairwise_significance(results, "composite_score",
                                         all_scenarios, sample_sizes)
    bar_tops_dict = {s: bar_top_matrix[i] for i, s in enumerate(all_scenarios)}
    _add_bracket_annotations(ax, all_scenarios, sample_sizes, offsets,
                             bar_tops_dict, sig, bar_w)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Composite score (0–1)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _despine(ax)
    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

    if show_legend:
        ax.legend(handles=legend_handles, ncol=1, frameon=False,
                  fontsize=7.5, handlelength=1.2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    return legend_handles


def fig_sample_sizes(all_results: dict[str, dict]):
    """
    Figure 2 — 2×3 grid with a dedicated legend row beneath each data row.

    Rows    : one per dataset (Diabetes, MIMIC-IV-ED CC)
    Columns : (a/d) fidelity score, (b/e) missingness score, (c/f) composite score
    Bars    : one group per scenario, one bar per sample size within each group
    Baseline reference lines match the colour of each sample size's bars.
    """
    dataset_names = list(all_results.keys())
    n_rows = len(dataset_names)

    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    row_cmaps = [mpl.cm.Blues, mpl.cm.Oranges]

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(FULL_W * 1.5, 3.5 * n_rows),
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.15)
    if n_rows == 1:
        axes = [axes]

    for ri, dname in enumerate(dataset_names):
        results  = all_results[dname]
        cmap     = row_cmaps[ri % len(row_cmaps)]
        row_axes = axes[ri]

        row_axes[1].set_title(dname, fontsize=9, loc="center", fontweight="bold", pad=10)

        _draw_samplesize_bars_on_ax(
            row_axes[0], results,
            score_col="fidelity_mean", std_col="fidelity_std",
            scenario_filter=single,
            ylabel="Fidelity score (0–1)",
            panel_label=panel_labels[ri][0],
            show_legend=False,
            cmap=cmap,
        )
        _draw_samplesize_bars_on_ax(
            row_axes[1], results,
            score_col="missingness_mean", std_col="missingness_std",
            scenario_filter=single,
            ylabel="Missingness score (0–1)",
            panel_label=panel_labels[ri][1],
            show_legend=False,
            cmap=cmap,
        )
        handles = _draw_samplesize_composite_on_ax(
            row_axes[2], results,
            scenario_filter=single,
            panel_label=panel_labels[ri][2],
            show_legend=False,
            cmap=cmap,
        )

        # Legend in a column to the right of the row
        row_axes[2].legend(
            handles=handles, ncol=1, frameon=False,
            fontsize=7.5, handlelength=1.2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )

    _save(fig, "fig2_sample_sizes")


def _draw_samplesize_violin_on_ax(
    ax,
    results: dict,
    score_col: str,
    scenario_filter,
    ylabel: str,
    panel_label: str,
    show_legend: bool = True,
    cmap=None,
) -> list:
    """
    Grouped violin plot for one dataset: one violin per (scenario, sample_size).
    Violin colour follows the dataset's colour ramp (same as the bar version).
    Median line inside each violin; baseline reference dotted line per sample size.
    """
    rows = []
    for key, entry in results.items():
        m = re.match(r"^(.+?)(?:_n(\d+))?$", key)
        scenario = m.group(1) if m else key
        size = int(m.group(2)) if m and m.group(2) else None
        for row in entry.get("per_dataset", []):
            val = row.get(score_col)
            if val is not None:
                rows.append({"scenario": scenario, "sample_size": size, "value": float(val)})
    df = pd.DataFrame(rows)
    df = df[df["scenario"].apply(scenario_filter)].copy()

    all_scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)
    sample_sizes  = sorted(df["sample_size"].unique(), key=lambda v: v if v is not None else -1)
    n_scenarios   = len(all_scenarios)
    n_sizes       = len(sample_sizes)

    _cmap = cmap if cmap is not None else mpl.cm.Blues
    colors = [_cmap(v) for v in np.linspace(0.35, 0.85, max(n_sizes, 1))]

    group_width = 0.7
    vln_w   = group_width / n_sizes
    offsets = np.linspace(
        -(group_width - vln_w) / 2,
         (group_width - vln_w) / 2,
        n_sizes,
    )
    x = np.arange(n_scenarios)
    legend_handles = []

    for si, (sz, color) in enumerate(zip(sample_sizes, colors)):
        sub = df[df["sample_size"] == sz]
        size_label = f"n={sz:,}" if sz is not None else "n=full"

        for xi, sc in enumerate(all_scenarios):
            data = sub[sub["scenario"] == sc]["value"].dropna().values
            if len(data) < 2:
                continue
            pos = x[xi] + offsets[si]
            parts = ax.violinplot(data, positions=[pos], widths=vln_w * 0.88,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(0.9)

        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, ec="none", alpha=0.75, label=size_label)
        )

        # Baseline reference line at mean of baseline replicates
        baseline_vals = sub[sub["scenario"] == "baseline"]["value"].dropna()
        if len(baseline_vals) > 0:
            ax.axhline(float(baseline_vals.mean()), color=color, lw=0.8, ls=":", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _despine(ax)
    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

    if show_legend:
        ax.legend(handles=legend_handles, ncol=1, frameon=False,
                  fontsize=7.5, handlelength=1.2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    return legend_handles


def fig_violin_sample_sizes(all_results: dict[str, dict]) -> None:
    """
    Violin version of fig_sample_sizes.

    Rows    : one per dataset (PID, MIMIC-IV-ED CC)
    Columns : fidelity score, missingness score, composite score
    Violins : one per (scenario, sample size); colour ramp matches fig2
    Baseline reference lines dotted per sample-size colour.
    """
    dataset_names = list(all_results.keys())
    n_rows = len(dataset_names)
    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    row_cmaps = [mpl.cm.Blues, mpl.cm.Oranges]

    col_specs = [
        ("fidelity_overall",    "Fidelity score (0–1)"),
        ("missingness_overall", "Missingness score (0–1)"),
        ("composite_score",     "Composite score (0–1)"),
    ]

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(FULL_W * 1.5, 3.5 * n_rows),
                             layout="constrained")
    if n_rows == 1:
        axes = [axes]

    for ri, dname in enumerate(dataset_names):
        results  = all_results[dname]
        cmap     = row_cmaps[ri % len(row_cmaps)]
        row_axes = axes[ri]

        row_axes[1].set_title(dname, fontsize=9, loc="center", fontweight="bold", pad=10)

        handles = []
        for ci, (score_col, ylabel) in enumerate(col_specs):
            h = _draw_samplesize_violin_on_ax(
                row_axes[ci], results,
                score_col=score_col,
                scenario_filter=single,
                ylabel=ylabel,
                panel_label=panel_labels[ri][ci],
                show_legend=False,
                cmap=cmap,
            )
            if ci == 0:
                handles = h

        row_axes[2].legend(
            handles=handles, ncol=1, frameon=False,
            fontsize=7.5, handlelength=1.2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )

    _save(fig, "fig_violin_sample_sizes")


def fig_appendix_sample_sizes(all_results: dict[str, dict]):
    """
    Appendix figure — 3 plot rows × 2 columns, composite (F×M) + baseline scenarios,
    one bar per sample size.  A short legend row sits beneath each column.

    Rows    : fidelity score, missingness score, composite score
    Columns : one per dataset
    """
    from matplotlib.gridspec import GridSpec

    dataset_names = list(all_results.keys())
    n_cols = len(dataset_names)
    composite = lambda s: s == "baseline" or s.startswith("composite")
    row_cmaps = [mpl.cm.Blues, mpl.cm.Oranges]

    row_specs = [
        ("fidelity_mean",    "fidelity_std",   "Fidelity score (0–1)",    "bars"),
        ("missingness_mean", "missingness_std", "Missingness score (0–1)", "bars"),
        (None,               None,              "Composite score (0–1)",   "composite"),
    ]
    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"]]

    # 3 plot rows + 1 legend row; legend row is short
    height_ratios = [1, 1, 1, 0.18]
    fig = plt.figure(figsize=(FULL_W * n_cols / 2 * 1.3, 10.0))
    gs  = GridSpec(4, n_cols, figure=fig,
                   height_ratios=height_ratios,
                   hspace=0.45, wspace=0.35)

    col_handles: dict[int, list] = {}

    for ci, dname in enumerate(dataset_names):
        cmap = row_cmaps[ci % len(row_cmaps)]

        for ri, (sc, st, ylabel, kind) in enumerate(row_specs):
            ax = fig.add_subplot(gs[ri, ci])
            if ri == 0:
                ax.set_title(dname, fontsize=9, loc="center",
                             fontweight="bold", pad=10)
            results = all_results[dname]

            if kind == "bars":
                _draw_samplesize_bars_on_ax(
                    ax, results,
                    score_col=sc, std_col=st,
                    scenario_filter=composite,
                    ylabel=ylabel,
                    panel_label=panel_labels[ri][ci],
                    show_legend=False,
                    cmap=cmap,
                )
            else:
                handles = _draw_samplesize_composite_on_ax(
                    ax, results,
                    scenario_filter=composite,
                    panel_label=panel_labels[ri][ci],
                    show_legend=False,
                    cmap=cmap,
                )
                col_handles[ci] = handles

        # Legend in the dedicated row beneath this column
        ax_leg = fig.add_subplot(gs[3, ci])
        ax_leg.set_axis_off()
        handles = col_handles.get(ci, [])
        if handles:
            # Fit ncol to column width: estimate ~1.3 in per legend entry
            ax_leg.legend(
                handles=handles, ncol=3, frameon=False,
                fontsize=7.5, handlelength=1.0,
                handletextpad=0.4, columnspacing=1.0,
                loc="center",
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=ax_leg.transAxes,
            )

    _save(fig, "fig_appendix_sample_sizes")


def _std_table_dataset_info(results: dict) -> tuple:
    """Return (df, all_scenarios, sample_sizes) for one dataset's std table."""
    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")
    df = build_summary_df(results)
    df = df[df["scenario"].apply(single)].copy()
    scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)
    sizes = sorted(df["sample_size"].unique(),
                   key=lambda v: v if v is not None else float("inf"))
    return df, scenarios, sizes


def _std_table_lines(
    datasets: dict[str, tuple],   # dname → (df, scenarios, sizes)
    score_specs: list[tuple],
    merged: bool,
) -> list[str]:
    """
    Build the LaTeX tabular lines for the std table.

    merged=False → one dataset only (datasets has a single entry)
    merged=True  → two datasets side by side with a top-level dataset header
    """
    n_scores = len(score_specs)
    sub_labels = ["{" + lbl + "}" for _, lbl in score_specs]

    # Compute per-dataset column counts and cumulative column offsets
    ds_info = []          # list of (dname, df, scenarios, sizes, n_cols, col_start)
    col_cursor = 2        # 1-based; column 1 = Scenario label
    for dname, (df, scenarios, sizes) in datasets.items():
        n_cols = len(sizes) * n_scores
        ds_info.append((dname, df, scenarios, sizes, n_cols, col_cursor))
        col_cursor += n_cols

    total_data_cols = col_cursor - 2
    col_spec = "l" + " S[table-format=1.3]" * total_data_cols

    # Union of scenarios in sorted order
    all_scenarios = sorted(
        {s for _, (_, scens, _) in datasets.items() for s in scens},
        key=_scenario_sort_key,
    )

    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # ---- Row 1: dataset headers (merged only) --------------------------------
    if merged:
        ds_headers = []
        for dname, _, _, _, n_cols, _ in ds_info:
            ds_headers.append(r"\multicolumn{" + str(n_cols) + r"}{c}{" + dname + "}")
        lines.append("& " + " & ".join(ds_headers) + r" \\")
        # cmidrules under each dataset block
        cmidrules = []
        for _, _, _, _, n_cols, col_start in ds_info:
            cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_start + n_cols - 1}}}")
        lines.append(" ".join(cmidrules))

    # ---- Row 2: sample-size headers ------------------------------------------
    sz_cells = []
    for _, _, _, sizes, _, _ in ds_info:
        for sz in sizes:
            label = f"$n={sz:,}$" if sz is not None else r"$n=\text{full}$"
            sz_cells.append(r"\multicolumn{" + str(n_scores) + r"}{c}{" + label + "}")
    lines.append("& " + " & ".join(sz_cells) + r" \\")

    # cmidrules under each size group
    cmidrules = []
    cur = 2
    for _, _, _, sizes, _, _ in ds_info:
        for _ in sizes:
            cmidrules.append(rf"\cmidrule(lr){{{cur}-{cur + n_scores - 1}}}")
            cur += n_scores
    lines.append(" ".join(cmidrules))

    # ---- Row 3: score-type sub-headers ---------------------------------------
    n_size_groups = sum(len(sizes) for _, (_, _, sizes) in datasets.items())
    lines.append("& " + " & ".join(sub_labels * n_size_groups) + r" \\")
    lines.append(r"\midrule")

    # ---- Data rows -----------------------------------------------------------
    prev_group = None
    for s in all_scenarios:
        group = (
            "baseline"    if s == "baseline"           else
            "fidelity"    if s.startswith("fidelity")  else
            "missingness"
        )
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        row_label = _tick_label(s).replace("\n", " ")
        cells = []
        for _, df, _, sizes, _, _ in ds_info:
            for sz in sizes:
                sub = df[df["sample_size"] == sz].set_index("scenario")
                for col, _ in score_specs:
                    val = sub.loc[s, col] if s in sub.index else float("nan")
                    cells.append(f"{val:.3f}" if not np.isnan(val) else r"\text{--}")
        lines.append(row_label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return lines


def export_std_table(all_results: dict[str, dict]) -> None:
    """
    Export the std-vs-sample-size data as LaTeX booktabs tables.

    Writes:
      - One per-dataset file: std_table_<dataset>.tex
      - One merged file with both datasets side by side: std_table_merged.tex

    Include in LaTeX with \\input{figures/output/std_table_merged.tex}.
    Requires in preamble: \\usepackage{booktabs}, \\usepackage{siunitx}

    The merged table is wide — wrap it in a table* environment or use
    \\resizebox{\\linewidth}{!}{\\input{...}} to fit the text width.
    """
    score_specs = [
        ("fidelity_std",    "Fid."),
        ("missingness_std", "Miss."),
        ("composite_std",   "Comp."),
    ]

    # Build per-dataset info once
    ds_info = {
        dname: _std_table_dataset_info(results)
        for dname, results in all_results.items()
    }

    # Individual per-dataset files
    for dname, info in ds_info.items():
        lines = _std_table_lines({dname: info}, score_specs, merged=False)
        slug = dname.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        out = OUT_DIR / f"std_table_{slug}.tex"
        out.write_text("\n".join(lines) + "\n")
        print(f"  Saved {out}")

    # Merged file (only meaningful when there are multiple datasets)
    if len(ds_info) > 1:
        lines = _std_table_lines(ds_info, score_specs, merged=True)
        out = OUT_DIR / "std_table_merged.tex"
        out.write_text("\n".join(lines) + "\n")
        print(f"  Saved {out}")


def fig_std_vs_sample_size(all_results: dict[str, dict]):
    """
    Figure — 2 dataset rows × 3 score columns.

    Rows    : one per dataset (Diabetes, MIMIC-IV-ED CC)
    Columns : fidelity std, missingness std, composite std
    X-axis  : scenarios (baseline + fidelity + missingness)
    Lines   : one per sample size, coloured by dataset ramp
    Legend  : to the right of each row (matching fig2 layout)
    """
    dataset_names = list(all_results.keys())
    n_rows = len(dataset_names)
    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")
    row_cmaps = [mpl.cm.Blues, mpl.cm.Oranges]

    col_specs = [
        ("fidelity_std",    "Fidelity std"),
        ("missingness_std", "Missingness std"),
        ("composite_std",   "Composite std"),
    ]
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(FULL_W * 1.5, 3.5 * n_rows),
                             layout="constrained")
    if n_rows == 1:
        axes = [axes]

    fig.get_layout_engine().set(hspace=0.15)

    for ri, dname in enumerate(dataset_names):
        results  = all_results[dname]
        cmap     = row_cmaps[ri % len(row_cmaps)]
        row_axes = axes[ri]

        row_axes[1].set_title(dname, fontsize=9, loc="center",
                               fontweight="bold", pad=10)

        df = build_summary_df(results)
        df = df[df["scenario"].apply(single)].copy()

        all_scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)
        sample_sizes  = sorted(df["sample_size"].unique(),
                               key=lambda v: v if v is not None else -1)
        n_sizes = len(sample_sizes)
        colors  = [cmap(v) for v in np.linspace(0.35, 0.85, max(n_sizes, 1))]
        x = np.arange(len(all_scenarios))

        legend_handles = []

        for ci, (std_col, ylabel) in enumerate(col_specs):
            ax = row_axes[ci]

            for si, (sz, color) in enumerate(zip(sample_sizes, colors)):
                sub = df[df["sample_size"] == sz].set_index("scenario")
                stds = np.array([sub.loc[s, std_col] if s in sub.index else np.nan
                                 for s in all_scenarios])
                size_label = f"n={sz:,}" if sz is not None else "n=full"
                ax.plot(x, stds, color=color, lw=1.2, marker="o",
                        markersize=3.5)
                if ci == 0:
                    legend_handles.append(
                        Line2D([0], [0], color=color, lw=1.2, marker="o",
                               markersize=3.5, label=size_label)
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            _despine(ax)
            _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)
            ax.text(-0.07, 1.02, panel_labels[ri][ci], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="left")

        # Legend to the right of the last column, matching fig2 style
        row_axes[2].legend(
            handles=legend_handles, ncol=1, frameon=False,
            fontsize=7.5, handlelength=1.2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )

    _save(fig, "fig3_std_vs_sample_size")


def _filter_results_by_sample_size(results: dict, sample_size) -> dict:
    """
    Return a results dict containing only entries whose ``sample_size`` matches
    *sample_size* (use ``None`` for full dataset).  Result keys are rewritten to
    the bare scenario name (dropping the ``_n{size}`` suffix) so downstream
    functions treat them identically to un-sampled results.
    """
    out = {}
    for key, entry in results.items():
        if entry.get("sample_size") == sample_size:
            m = re.match(r"^(.+?)(?:_n\d+)?$", key)
            scenario = m.group(1) if m else key
            out[scenario] = entry
    return out


def fig1_selected(all_results: dict[str, dict], size_per_dataset: dict):
    """
    Same 3-panel layout as fig_main but each dataset is filtered to a single
    sample size before plotting.

    Parameters
    ----------
    all_results : dict
        Full results dict keyed by dataset name.
    size_per_dataset : dict
        Maps dataset name → sample size (int or None for full).
        e.g. {"PID": None, "MIMIC-IV-ED (CC)": 10000}
    """
    filtered = {
        dname: _filter_results_by_sample_size(res, size_per_dataset.get(dname))
        for dname, res in all_results.items()
        if dname in size_per_dataset
    }

    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W * 1.5, 3.2), layout="constrained")

    _draw_bars_on_ax(
        axes[0], filtered,
        score_col="fidelity_mean", std_col="fidelity_std",
        scenario_filter=single,
        ylabel="Fidelity score (0–1)",
        panel_label="(a)",
        dataset_colors=[WONG["blue"], ORANGE_DARK],
        show_legend=False,
    )
    _draw_bars_on_ax(
        axes[1], filtered,
        score_col="missingness_mean", std_col="missingness_std",
        scenario_filter=single,
        ylabel="Missingness score (0–1)",
        panel_label="(b)",
        dataset_colors=[WONG["sky"], ORANGE_LIGHT],
        show_legend=False,
    )
    handles = _draw_composite_on_ax(
        axes[2], filtered,
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

    _save(fig, "fig1_selected")


def fig_baseline_metric_scores(
    all_results: dict[str, dict],
    size_per_dataset: Optional[dict] = None,
) -> None:
    """
    Grouped bar chart of per-metric fidelity scores in the baseline scenario.

    One group of bars per fidelity metric; within each group, one bar per
    dataset.  Bar height = mean across baseline replicates; error bars = ±1 std.
    Vertical dotted lines separate the univariate / bivariate / multivariate groups.

    Parameters
    ----------
    size_per_dataset : dict, optional
        Maps dataset name → sample size (int or None for full dataset).
        When provided, only replicates from that sample size are included.
        When omitted, all replicates are pooled across sample sizes.
    """
    METRIC_ORDER = [
        # univariate
        "fidelity_wasserstein", "fidelity_tvd", "fidelity_hellinger",
        # bivariate
        "fidelity_spearman", "fidelity_contingency", "fidelity_pcd",
        # multivariate
        "fidelity_auc_roc", "fidelity_propensity_mse",
        "fidelity_crcl_rs", "fidelity_crcl_sr",
    ]
    METRIC_LABELS_SHORT = {
        "fidelity_wasserstein":    "Wasserstein",
        "fidelity_tvd":            "TVD",
        "fidelity_hellinger":      "Hellinger",
        "fidelity_spearman":       "Spearman",
        "fidelity_contingency":    "Contingency",
        "fidelity_pcd":            "PCD",
        "fidelity_auc_roc":        "AUC-ROC",
        "fidelity_propensity_mse": "Prop. MSE",
        "fidelity_crcl_rs":        "CrCl-RS",
        "fidelity_crcl_sr":        "CrCl-SR",
    }
    METRIC_TO_GROUP = {
        "fidelity_wasserstein": "Univariate", "fidelity_tvd": "Univariate",
        "fidelity_hellinger":   "Univariate", "fidelity_spearman": "Bivariate",
        "fidelity_contingency": "Bivariate",  "fidelity_pcd": "Bivariate",
        "fidelity_auc_roc":        "Multivariate",
        "fidelity_propensity_mse": "Multivariate",
        "fidelity_crcl_rs":        "Multivariate",
        "fidelity_crcl_sr":        "Multivariate",
    }
    GROUP_ORDER = ["Univariate", "Bivariate", "Multivariate"]

    dataset_colors = {
        "PID":         WONG["blue"],
        "MIMIC-IV-ED (CC)": ORANGE_DARK,
    }
    dataset_names = list(all_results.keys())

    # Collect mean ± std per dataset per metric from baseline replicates
    stats: dict[str, dict[str, tuple]] = {}   # dataset → metric → (mean, std)
    for dname, results in all_results.items():
        target_size = size_per_dataset.get(dname) if size_per_dataset else None
        rows = []
        for key, entry in results.items():
            if not re.match(r"^baseline", key):
                continue
            if size_per_dataset is not None and entry.get("sample_size") != target_size:
                continue
            rows.extend(entry["per_dataset"])
        if not rows:
            continue
        ds_stats = {}
        for col in METRIC_ORDER:
            vals = [r[col] for r in rows if col in r and r[col] is not None]
            if vals:
                ds_stats[col] = (float(np.mean(vals)), float(np.std(vals)))
        stats[dname] = ds_stats

    # Only show metrics present in at least one dataset
    visible_metrics = [m for m in METRIC_ORDER
                       if any(m in stats.get(d, {}) for d in dataset_names)]

    n_metrics  = len(visible_metrics)
    n_datasets = len(dataset_names)
    bar_w      = 0.35
    group_gap  = 0.2                          # extra gap between metric groups
    offsets    = np.linspace(-(n_datasets - 1) / 2 * bar_w,
                              (n_datasets - 1) / 2 * bar_w, n_datasets)

    # Find indices where the group changes among visible metrics
    boundary_indices = {
        i for i in range(1, n_metrics)
        if METRIC_TO_GROUP.get(visible_metrics[i]) != METRIC_TO_GROUP.get(visible_metrics[i - 1])
    }

    # Compute x positions with extra gap at group boundaries
    x_positions = []
    extra = 0.0
    for i in range(n_metrics):
        if i in boundary_indices:
            extra += group_gap
        x_positions.append(i + extra)
    x_positions = np.array(x_positions)

    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))

    legend_handles = []
    for di, dname in enumerate(dataset_names):
        if dname not in stats:
            continue
        color = dataset_colors.get(dname, WONG["black"])
        means, errs = [], []
        for col in visible_metrics:
            if col in stats[dname]:
                m, s = stats[dname][col]
                means.append(m)
                errs.append(s)
            else:
                means.append(np.nan)
                errs.append(np.nan)

        bars = ax.bar(
            x_positions + offsets[di], means,
            width=bar_w, color=color, alpha=0.85,
            yerr=errs, error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
            label=dname,
        )
        legend_handles.append(bars)

    # Group divider lines
    for bi in sorted(boundary_indices):
        gap_x = (x_positions[bi - 1] + x_positions[bi]) / 2
        ax.axvline(gap_x, color="grey", lw=0.8, linestyle=":", alpha=0.7)

    # Group labels centred over their visible metrics
    from collections import defaultdict
    group_xs: dict[str, list] = defaultdict(list)
    for i, m in enumerate(visible_metrics):
        group_xs[METRIC_TO_GROUP[m]].append(x_positions[i])
    for group in GROUP_ORDER:
        if group not in group_xs:
            continue
        mid_x = np.mean(group_xs[group])
        ax.text(mid_x, -0.24, group, ha="center", va="top",
                fontsize=7.5, fontstyle="italic",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [METRIC_LABELS_SHORT[m] for m in visible_metrics],
        fontsize=7.5, rotation=30, ha="right",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.axhline(1.0, color="grey", lw=0.6, linestyle="--", alpha=0.5)
    _despine(ax)

    ax.legend(
        ncol=len(dataset_names), frameon=False,
        fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, -0.46),
    )

    fig.tight_layout()
    suffix = "_selected" if size_per_dataset else ""
    _save(fig, f"fig_baseline_metric_scores{suffix}")


def fig_scenario_metric_grid(
    all_results: dict[str, dict],
    size_per_dataset: Optional[dict] = None,
) -> None:
    """
    Grid of bar charts: one row per scenario, two columns (fidelity | missingness).

    Row 0 is the baseline.  Fidelity scenarios follow, then missingness scenarios.
    Each cell shows per-metric mean ± std for the specified sample size.

    Parameters
    ----------
    size_per_dataset : dict, optional
        Maps dataset name → sample size (int or None for full dataset).
        Defaults to the largest available size per dataset.
    """
    SCENARIO_ORDER = [
        ("baseline",      "Baseline"),
        ("fidelity_1",    "F1"),
        ("fidelity_3",    "F3"),
        ("fidelity_5",    "F5"),
        ("missingness_1", "M1"),
        ("missingness_2", "M2"),
        ("missingness_3", "M3"),
        ("missingness_4", "M4"),
        ("missingness_5", "M5"),
    ]
    # Section separators: row indices where a new section begins
    SECTION_STARTS = {0: "Baseline", 1: "Fidelity scenarios", 4: "Missingness scenarios"}

    FID_METRICS = [
        "fidelity_wasserstein", "fidelity_tvd", "fidelity_hellinger",
        "fidelity_spearman", "fidelity_contingency", "fidelity_pcd",
        "fidelity_auc_roc", "fidelity_propensity_mse",
        "fidelity_crcl_rs", "fidelity_crcl_sr",
    ]
    MISS_METRICS = [
        "missingness_rate", "missingness_set_distribution",
        "missingness_missing_auroc", "missingness_dependency_structure",
    ]
    METRIC_LABELS_SHORT = {
        "fidelity_wasserstein":              "Wass.",
        "fidelity_tvd":                      "TVD",
        "fidelity_hellinger":                "Hell.",
        "fidelity_spearman":                 "Spear.",
        "fidelity_contingency":              "Cont.",
        "fidelity_pcd":                      "PCD",
        "fidelity_auc_roc":                  "AUC-ROC",
        "fidelity_propensity_mse":           "Prop. MSE",
        "fidelity_crcl_rs":                  "CrCl-RS",
        "fidelity_crcl_sr":                  "CrCl-SR",
        "missingness_rate":                  "Rate",
        "missingness_set_distribution":      "Pattern",
        "missingness_missing_auroc":         "AUROC",
        "missingness_dependency_structure":  "Dep.",
    }
    METRIC_TO_GROUP = {
        "fidelity_wasserstein": "Univariate", "fidelity_tvd": "Univariate",
        "fidelity_hellinger":   "Univariate", "fidelity_spearman": "Bivariate",
        "fidelity_contingency": "Bivariate",  "fidelity_pcd": "Bivariate",
        "fidelity_auc_roc": "Multivariate", "fidelity_propensity_mse": "Multivariate",
        "fidelity_crcl_rs": "Multivariate", "fidelity_crcl_sr": "Multivariate",
    }

    dataset_names = list(all_results.keys())
    dataset_colors = {
        "PID":         WONG["blue"],
        "MIMIC-IV-ED (CC)": ORANGE_DARK,
    }

    # -----------------------------------------------------------------------
    # Collect stats: scenario_key → dataset → metric → (mean, std)
    # -----------------------------------------------------------------------
    def _collect(scenario_prefix: str) -> dict[str, dict[str, tuple]]:
        out: dict[str, dict[str, tuple]] = {}
        for dname, results in all_results.items():
            target_size = size_per_dataset.get(dname) if size_per_dataset else None
            rows = []
            for key, entry in results.items():
                if not re.match(rf"^{scenario_prefix}(\b|_n)", key):
                    continue
                if size_per_dataset is not None and entry.get("sample_size") != target_size:
                    continue
                rows.extend(entry["per_dataset"])
            if not rows:
                continue
            ds: dict[str, tuple] = {}
            for col in FID_METRICS + MISS_METRICS:
                vals = [r[col] for r in rows if col in r and r[col] is not None]
                if vals:
                    ds[col] = (float(np.mean(vals)), float(np.std(vals)))
            out[dname] = ds
        return out

    all_stats = {
        sc_key: _collect(sc_key) for sc_key, _ in SCENARIO_ORDER
    }

    # Determine which metrics are visible (present in ≥1 dataset across any scenario)
    all_rows_combined = {
        m: any(m in ds for sc_stats in all_stats.values() for ds in sc_stats.values())
        for m in FID_METRICS + MISS_METRICS
    }
    vis_fid  = [m for m in FID_METRICS  if all_rows_combined[m]]
    vis_miss = [m for m in MISS_METRICS if all_rows_combined[m]]

    # x positions for fidelity panel (with group gaps)
    fid_boundaries = {
        i for i in range(1, len(vis_fid))
        if METRIC_TO_GROUP.get(vis_fid[i]) != METRIC_TO_GROUP.get(vis_fid[i - 1])
    }
    def _xpos(metrics, boundaries):
        pos, extra = [], 0.0
        for i in range(len(metrics)):
            if i in boundaries:
                extra += 0.15
            pos.append(i + extra)
        return np.array(pos)

    xpos_fid  = _xpos(vis_fid,  fid_boundaries)
    xpos_miss = _xpos(vis_miss, set())

    n_scenarios = len(SCENARIO_ORDER)
    n_ds = len(dataset_names)
    bar_w = 0.3
    offsets = np.linspace(-(n_ds - 1) / 2 * bar_w, (n_ds - 1) / 2 * bar_w, n_ds)

    row_h = 1.15
    fig, axes = plt.subplots(
        n_scenarios, 2,
        figsize=(FULL_W, n_scenarios * row_h + 0.6),
        sharey=False,
    )

    col_headers = ["Fidelity metrics", "Missingness metrics"]
    for ci, hdr in enumerate(col_headers):
        axes[0, ci].set_title(hdr, fontsize=8, fontweight="bold", pad=3)

    for ri, (sc_key, sc_label) in enumerate(SCENARIO_ORDER):
        sc_stats = all_stats[sc_key]

        for ci, (metrics, xpos, boundaries) in enumerate([
            (vis_fid,  xpos_fid,  fid_boundaries),
            (vis_miss, xpos_miss, set()),
        ]):
            ax = axes[ri, ci]
            is_bottom = (ri == n_scenarios - 1)

            for di, dname in enumerate(dataset_names):
                if dname not in sc_stats:
                    continue
                color = dataset_colors.get(dname, WONG["black"])
                means = [sc_stats[dname].get(m, (np.nan, np.nan))[0] for m in metrics]
                errs  = [sc_stats[dname].get(m, (np.nan, np.nan))[1] for m in metrics]
                ax.bar(
                    xpos + offsets[di], means,
                    width=bar_w, color=color, alpha=0.85,
                    yerr=errs,
                    error_kw={"elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6},
                    label=dname if ri == 0 else None,
                )

            # Group dividers (fidelity column only)
            for bi in sorted(boundaries):
                gap_x = (xpos[bi - 1] + xpos[bi]) / 2
                ax.axvline(gap_x, color="grey", lw=0.6, linestyle=":", alpha=0.6)

            ax.set_xlim(xpos[0] - 0.5, xpos[-1] + 0.5)
            ax.set_ylim(0, 1.09)
            ax.axhline(1.0, color="grey", lw=0.5, linestyle="--", alpha=0.4)
            ax.tick_params(axis="y", labelsize=6)
            _despine(ax)

            # x-tick labels only on bottom row
            if is_bottom:
                ax.set_xticks(xpos)
                ax.set_xticklabels(
                    [METRIC_LABELS_SHORT[m] for m in metrics],
                    fontsize=6.5, rotation=30, ha="right",
                )
            else:
                ax.set_xticks([])

        # Scenario label on left column
        axes[ri, 0].set_ylabel(sc_label, fontsize=8, labelpad=4)

        # Horizontal line above section starts
        if ri in SECTION_STARTS and ri > 0:
            for ci in range(2):
                axes[ri, ci].spines["top"].set_visible(True)
                axes[ri, ci].spines["top"].set_linewidth(0.8)
                axes[ri, ci].spines["top"].set_color("#555555")

    # Shared y-label
    fig.text(0.0, 0.5, "Score", va="center", rotation="vertical", fontsize=8)

    # Legend from first-row handles
    handles = [
        mpl.patches.Patch(facecolor=dataset_colors.get(d, WONG["black"]), alpha=0.85, label=d)
        for d in dataset_names
    ]
    fig.legend(
        handles=handles, ncol=len(dataset_names), frameon=False,
        fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0.03, 0.03, 1.0, 1.0], h_pad=0.4, w_pad=0.6)
    suffix = "_selected" if size_per_dataset else ""
    _save(fig, f"fig_scenario_metric_grid{suffix}")


def export_scenario_table(
    all_results: dict[str, dict],
    size_per_dataset: Optional[dict] = None,
) -> None:
    """
    Export the scenario × metric data as a LaTeX booktabs table.

    Rows : scenarios (Baseline, F1, F3, F5, M1–M5), two sub-rows each
           (one per dataset).
    Cols : enabled fidelity metrics then missingness metrics.
    Cells: mean ± std across replicates at the specified sample size.

    Writes: figures/output/scenario_table.tex
    Requires in LaTeX preamble:
        \\usepackage{booktabs}
    """
    SCENARIO_ORDER = [
        ("baseline",      "Baseline"),
        ("fidelity_1",    "F1"),
        ("fidelity_2",    "F2"),
        ("fidelity_3",    "F3"),
        ("fidelity_4",    "F4"),
        ("fidelity_5",    "F5"),
        ("missingness_1", "M1"),
        ("missingness_2", "M2"),
        ("missingness_3", "M3"),
        ("missingness_4", "M4"),
        ("missingness_5", "M5"),
    ]
    SECTION_AFTER = {"baseline", "fidelity_5"}   # insert \midrule after these

    FID_METRICS = [
        "fidelity_wasserstein", "fidelity_tvd", "fidelity_hellinger",
        "fidelity_spearman", "fidelity_contingency", "fidelity_pcd",
        "fidelity_auc_roc", "fidelity_propensity_mse",
        "fidelity_crcl_rs", "fidelity_crcl_sr",
    ]
    MISS_METRICS = [
        "missingness_rate", "missingness_set_distribution",
        "missingness_missing_auroc", "missingness_dependency_structure",
    ]
    COL_LABELS = {
        "fidelity_wasserstein":             "Wass.",
        "fidelity_tvd":                     "TVD",
        "fidelity_hellinger":               "Hell.",
        "fidelity_spearman":                "Spear.",
        "fidelity_contingency":             "Cont.",
        "fidelity_pcd":                     "PCD",
        "fidelity_auc_roc":                 "AuRoc",
        "fidelity_propensity_mse":          "pMSE",
        "fidelity_crcl_rs":                 "CrCl-RS",
        "fidelity_crcl_sr":                 "CrCl-SR",
        "missingness_rate":                 "MissRateDiff",
        "missingness_set_distribution":     "MissSet",
        "missingness_missing_auroc":        "MissAuRoc",
        "missingness_dependency_structure": "MissDep.",
    }
    DS_LABELS = {
        "PID":         "PID",
        "MIMIC-IV-ED (CC)": "MIMIC",
    }

    dataset_names = list(all_results.keys())

    # -----------------------------------------------------------------------
    # Collect stats (same logic as fig_scenario_metric_grid)
    # -----------------------------------------------------------------------
    def _collect(scenario_prefix: str) -> dict:
        out: dict = {}
        for dname, results in all_results.items():
            target_size = size_per_dataset.get(dname) if size_per_dataset else None
            rows = []
            for key, entry in results.items():
                if not re.match(rf"^{scenario_prefix}(\b|_n)", key):
                    continue
                if size_per_dataset is not None and entry.get("sample_size") != target_size:
                    continue
                rows.extend(entry["per_dataset"])
            if not rows:
                continue
            ds: dict = {}
            for col in FID_METRICS + MISS_METRICS:
                vals = [r[col] for r in rows if col in r and r[col] is not None]
                if vals:
                    ds[col] = (float(np.mean(vals)), float(np.std(vals)))
            out[dname] = ds
        return out

    all_stats = {sc_key: _collect(sc_key) for sc_key, _ in SCENARIO_ORDER}

    # Determine visible metrics (present in ≥1 dataset across any scenario)
    def _visible(metrics):
        return [
            m for m in metrics
            if any(m in ds for sc in all_stats.values() for ds in sc.values())
        ]

    vis_fid  = _visible(FID_METRICS)
    vis_miss = _visible(MISS_METRICS)
    all_metrics = vis_fid + vis_miss
    n_fid  = len(vis_fid)
    n_miss = len(vis_miss)
    n_cols = n_fid + n_miss

    # -----------------------------------------------------------------------
    # Build LaTeX
    # -----------------------------------------------------------------------
    col_fmt = "ll" + "c" * n_cols
    lines = [f"\\begin{{tabular}}{{{col_fmt}}}"]
    lines.append("\\toprule")

    # Header row 1: group spans
    group_header = " & ".join([
        "",
        "",
        f"\\multicolumn{{{n_fid}}}{{c}}{{Fidelity}}",
        f"\\multicolumn{{{n_miss}}}{{c}}{{Missingness}}",
    ])
    lines.append(group_header + " \\\\")
    lines.append(
        f"\\cmidrule(lr){{3-{2 + n_fid}}} "
        f"\\cmidrule(lr){{{3 + n_fid}-{2 + n_cols}}}"
    )

    # Header row 2: metric names
    metric_header = "Scenario & Dataset & " + " & ".join(
        COL_LABELS[m] for m in all_metrics
    )
    lines.append(metric_header + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for sc_key, sc_label in SCENARIO_ORDER:
        sc_stats = all_stats[sc_key]
        for di, dname in enumerate(dataset_names):
            ds_stats = sc_stats.get(dname, {})
            ds_label = DS_LABELS.get(dname, dname)

            cells = []
            for m in all_metrics:
                if m in ds_stats:
                    mean, std = ds_stats[m]
                    cells.append(f"${mean:.3f} \\pm {std:.3f}$")
                else:
                    cells.append("--")

            # Scenario label only on first sub-row; blank on subsequent rows
            sc_cell = sc_label if di == 0 else ""

            row = " & ".join([sc_cell, ds_label] + cells)
            lines.append(row + " \\\\")

        if sc_key in SECTION_AFTER:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out = OUT_DIR / "scenario_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Saved {out}")


def fig_metric_heatmap(
    all_results: dict[str, dict],
    size_per_dataset: Optional[dict] = None,
) -> None:
    """
    Heatmap of per-metric mean scores across all scenarios, one panel per dataset.

    Rows    : scenarios (Baseline, F1–F5, M1–M5)
    Columns : individual metrics (fidelity group | missingness group)
    Colour  : mean score 0–1 (RdYlGn: red = low, green = high)
    Cells   : annotated with mean score; NaN shown in light grey

    Immediately reveals which metric is dragging a score down in each scenario.
    """
    SCENARIO_ORDER = [
        ("baseline",      "Baseline"),
        ("fidelity_1",    "F1"),
        ("fidelity_2",    "F2"),
        ("fidelity_3",    "F3"),
        ("fidelity_4",    "F4"),
        ("fidelity_5",    "F5"),
        ("missingness_1", "M1"),
        ("missingness_2", "M2"),
        ("missingness_3", "M3"),
        ("missingness_4", "M4"),
        ("missingness_5", "M5"),
    ]
    # Horizontal separators: draw line above these row indices
    H_SEP = {1, 6}   # between Baseline/Fidelity and Fidelity/Missingness

    FID_METRICS = [
        "fidelity_wasserstein", "fidelity_tvd", "fidelity_hellinger",
        "fidelity_spearman", "fidelity_contingency", "fidelity_pcd",
        "fidelity_auc_roc", "fidelity_propensity_mse",
        "fidelity_crcl_rs", "fidelity_crcl_sr",
    ]
    MISS_METRICS = [
        "missingness_rate", "missingness_set_distribution",
        "missingness_missing_auroc", "missingness_dependency_structure",
    ]
    METRIC_LABELS = {
        "fidelity_wasserstein":             "Wass.",
        "fidelity_tvd":                     "TVD",
        "fidelity_hellinger":               "Hell.",
        "fidelity_spearman":                "Spear.",
        "fidelity_contingency":             "Cont.",
        "fidelity_pcd":                     "PCD",
        "fidelity_auc_roc":                 "AuRoc",
        "fidelity_propensity_mse":          "pMSE",
        "fidelity_crcl_rs":                 "CrCl-RS",
        "fidelity_crcl_sr":                 "CrCl-SR",
        "missingness_rate":                 "MissRateDiff",
        "missingness_set_distribution":     "MissSet",
        "missingness_missing_auroc":        "MissAuRoc",
        "missingness_dependency_structure": "MissDep.",
    }

    all_metrics_ordered = FID_METRICS + MISS_METRICS
    dataset_names = list(all_results.keys())
    n_scenarios   = len(SCENARIO_ORDER)

    # -----------------------------------------------------------------------
    # Collect mean score per (dataset, scenario, metric)
    # -----------------------------------------------------------------------
    def _collect(scenario_prefix: str, dname: str, results: dict) -> dict:
        target_size = size_per_dataset.get(dname) if size_per_dataset else None
        rows = []
        for key, entry in results.items():
            if not re.match(rf"^{scenario_prefix}(\b|_n)", key):
                continue
            if size_per_dataset is not None and entry.get("sample_size") != target_size:
                continue
            rows.extend(entry["per_dataset"])
        if not rows:
            return {}
        out = {}
        for col in all_metrics_ordered:
            vals = [r[col] for r in rows if col in r and r[col] is not None]
            if vals:
                out[col] = float(np.mean(vals))
        return out

    # Determine visible metrics (present in ≥1 dataset × scenario)
    all_sc_stats: dict[str, dict[str, dict]] = {
        sc_key: {dname: _collect(sc_key, dname, results)
                 for dname, results in all_results.items()}
        for sc_key, _ in SCENARIO_ORDER
    }
    visible = [
        m for m in all_metrics_ordered
        if any(m in ds for sc in all_sc_stats.values() for ds in sc.values())
    ]
    n_metrics = len(visible)
    n_fid_vis = sum(1 for m in visible if m in FID_METRICS)

    # -----------------------------------------------------------------------
    # Build figure
    # -----------------------------------------------------------------------
    n_ds    = len(dataset_names)
    cell_w  = 0.48
    cell_h  = 0.42
    panel_w = n_metrics  * cell_w + 1.2
    panel_h = n_scenarios * cell_h + 1.0

    fig, axes = plt.subplots(
        1, n_ds,
        figsize=(panel_w * n_ds + 0.6, panel_h),
        layout="constrained",
    )
    if n_ds == 1:
        axes = [axes]

    cmap_base = mpl.cm.RdYlGn
    nan_color = "#d0d0d0"

    for di, (ax, dname) in enumerate(zip(axes, dataset_names)):
        results = all_results[dname]

        # Build score matrix [n_scenarios × n_metrics]
        mat = np.full((n_scenarios, n_metrics), np.nan)
        for si, (sc_key, _) in enumerate(SCENARIO_ORDER):
            ds_stats = all_sc_stats[sc_key].get(dname, {})
            for mi, m in enumerate(visible):
                if m in ds_stats:
                    mat[si, mi] = ds_stats[m]

        # Draw heatmap via imshow
        masked = np.ma.masked_invalid(mat)
        cmap_copy = cmap_base.copy()
        cmap_copy.set_bad(nan_color)
        im = ax.imshow(masked, cmap=cmap_copy, vmin=0, vmax=1,
                       aspect="auto", interpolation="none")

        # Cell annotations
        for si in range(n_scenarios):
            for mi in range(n_metrics):
                val = mat[si, mi]
                if np.isnan(val):
                    continue
                text_color = "black" if 0.25 < val < 0.85 else "white"
                ax.text(mi, si, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.0, color=text_color)

        # Vertical separator between fidelity and missingness metric groups
        if 0 < n_fid_vis < n_metrics:
            ax.axvline(n_fid_vis - 0.5, color="white", lw=2.0)

        # Horizontal separators between scenario groups
        for row_idx in H_SEP:
            if 0 < row_idx < n_scenarios:
                ax.axhline(row_idx - 0.5, color="white", lw=2.0)

        # Axis ticks and labels
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels([METRIC_LABELS[m] for m in visible],
                           rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(n_scenarios))
        ax.set_yticklabels([lbl for _, lbl in SCENARIO_ORDER], fontsize=7.5)
        ax.tick_params(length=0)

        # Title: dataset name + sample size on second line
        sz = size_per_dataset.get(dname) if size_per_dataset else None
        size_str = f"\nn={sz:,}" if sz is not None else ("\nn=full" if size_per_dataset else "")
        ax.set_title(f"{dname}{size_str}", fontsize=9, fontweight="bold", pad=6)

        # Metric group labels below x-tick labels
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        fid_center  = (n_fid_vis - 1) / 2
        miss_center = n_fid_vis + (n_metrics - n_fid_vis - 1) / 2
        ax.text(fid_center,  -0.11, "Fidelity",    ha="center", va="top",
                fontsize=7.5, fontstyle="italic", transform=trans)
        if n_metrics > n_fid_vis:
            ax.text(miss_center, -0.11, "Missingness", ha="center", va="top",
                    fontsize=7.5, fontstyle="italic", transform=trans)

    # Shared colourbar
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                 label="Mean score (0–1)")

    suffix = "_selected" if size_per_dataset else ""
    _save(fig, f"fig_metric_heatmap{suffix}")


def fig_metric_heatmap_by_size(all_results: dict[str, dict]) -> None:
    """
    Heatmap grid stratified by sample size.

    Rows    : one per dataset
    Columns : one per sample size (sorted ascending; datasets may differ)
    Each cell : scenarios × metrics heatmap (same colour scale as fig_metric_heatmap)
    """
    SCENARIO_ORDER = [
        ("baseline",      "Baseline"),
        ("fidelity_1",    "F1"),
        ("fidelity_2",    "F2"),
        ("fidelity_3",    "F3"),
        ("fidelity_4",    "F4"),
        ("fidelity_5",    "F5"),
        ("missingness_1", "M1"),
        ("missingness_2", "M2"),
        ("missingness_3", "M3"),
        ("missingness_4", "M4"),
        ("missingness_5", "M5"),
    ]
    H_SEP = {1, 6}

    FID_METRICS = [
        "fidelity_wasserstein", "fidelity_tvd", "fidelity_hellinger",
        "fidelity_spearman", "fidelity_contingency", "fidelity_pcd",
        "fidelity_auc_roc", "fidelity_propensity_mse",
        "fidelity_crcl_rs", "fidelity_crcl_sr",
    ]
    MISS_METRICS = [
        "missingness_rate", "missingness_set_distribution",
        "missingness_missing_auroc", "missingness_dependency_structure",
    ]
    METRIC_LABELS = {
        "fidelity_wasserstein":             "Wass.",
        "fidelity_tvd":                     "TVD",
        "fidelity_hellinger":               "Hell.",
        "fidelity_spearman":                "Spear.",
        "fidelity_contingency":             "Cont.",
        "fidelity_pcd":                     "PCD",
        "fidelity_auc_roc":                 "AuRoc",
        "fidelity_propensity_mse":          "pMSE",
        "fidelity_crcl_rs":                 "CrCl-RS",
        "fidelity_crcl_sr":                 "CrCl-SR",
        "missingness_rate":                 "MissRateDiff",
        "missingness_set_distribution":     "MissSet",
        "missingness_missing_auroc":        "MissAuRoc",
        "missingness_dependency_structure": "MissDep.",
    }

    all_metrics_ordered = FID_METRICS + MISS_METRICS
    dataset_names = list(all_results.keys())
    n_scenarios   = len(SCENARIO_ORDER)

    # -----------------------------------------------------------------------
    # Collect mean score per (dataset, sample_size, scenario, metric)
    # -----------------------------------------------------------------------
    def _collect_for_size(scenario_prefix: str, dname: str,
                          results: dict, target_size) -> dict:
        rows = []
        for key, entry in results.items():
            if not re.match(rf"^{scenario_prefix}(\b|_n)", key):
                continue
            if entry.get("sample_size") != target_size:
                continue
            rows.extend(entry["per_dataset"])
        if not rows:
            return {}
        out = {}
        for col in all_metrics_ordered:
            vals = [r[col] for r in rows if col in r and r[col] is not None]
            if vals:
                out[col] = float(np.mean(vals))
        return out

    # Discover sample sizes per dataset
    ds_sizes: dict[str, list] = {}
    for dname, results in all_results.items():
        sizes = sorted(
            {entry.get("sample_size") for entry in results.values()},
            key=lambda v: v if v is not None else float("inf"),
        )
        ds_sizes[dname] = sizes

    max_n_sizes = max(len(s) for s in ds_sizes.values())
    n_ds = len(dataset_names)

    # Determine globally visible metrics
    visible = [
        m for m in all_metrics_ordered
        if any(
            _collect_for_size(sc_key, dname, all_results[dname], sz).get(m) is not None
            for sc_key, _ in SCENARIO_ORDER
            for dname in dataset_names
            for sz in ds_sizes[dname]
        )
    ]
    n_metrics  = len(visible)
    n_fid_vis  = sum(1 for m in visible if m in FID_METRICS)

    # -----------------------------------------------------------------------
    # Build figure: rows=datasets, cols=sample sizes
    # -----------------------------------------------------------------------
    cell_w  = 0.46
    cell_h  = 0.40
    panel_w = n_metrics   * cell_w + 1.0
    panel_h = n_scenarios * cell_h + 0.8

    fig, axes = plt.subplots(
        n_ds, max_n_sizes,
        figsize=(panel_w * max_n_sizes + 0.6, panel_h * n_ds + 0.4),
        layout="constrained",
        squeeze=False,
    )

    cmap_base = mpl.cm.RdYlGn
    nan_color = "#d0d0d0"
    im_last   = None

    for ri, dname in enumerate(dataset_names):
        results = all_results[dname]
        sizes   = ds_sizes[dname]

        for ci in range(max_n_sizes):
            ax = axes[ri, ci]

            if ci >= len(sizes):
                ax.set_visible(False)
                continue

            sz = sizes[ci]
            size_label = f"n={sz:,}" if sz is not None else "n=full"

            # Build score matrix [n_scenarios × n_metrics]
            mat = np.full((n_scenarios, n_metrics), np.nan)
            for si, (sc_key, _) in enumerate(SCENARIO_ORDER):
                sc_stats = _collect_for_size(sc_key, dname, results, sz)
                for mi, m in enumerate(visible):
                    if m in sc_stats:
                        mat[si, mi] = sc_stats[m]

            masked   = np.ma.masked_invalid(mat)
            cmap_cp  = cmap_base.copy()
            cmap_cp.set_bad(nan_color)
            im = ax.imshow(masked, cmap=cmap_cp, vmin=0, vmax=1,
                           aspect="auto", interpolation="none")
            im_last = im

            # Cell annotations
            for si in range(n_scenarios):
                for mi in range(n_metrics):
                    val = mat[si, mi]
                    if np.isnan(val):
                        continue
                    text_color = "black" if 0.25 < val < 0.85 else "white"
                    ax.text(mi, si, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.5, color=text_color)

            # Separators
            if 0 < n_fid_vis < n_metrics:
                ax.axvline(n_fid_vis - 0.5, color="white", lw=2.0)
            for row_idx in H_SEP:
                if 0 < row_idx < n_scenarios:
                    ax.axhline(row_idx - 0.5, color="white", lw=2.0)

            # Column title: dataset name + sample size
            ax.set_title(f"{dname}\n{size_label}", fontsize=8, pad=4)

            # X ticks only on bottom row; metric group labels below them
            if ri == n_ds - 1 or axes[ri + 1, ci].get_visible() is False:
                ax.set_xticks(range(n_metrics))
                ax.set_xticklabels([METRIC_LABELS[m] for m in visible],
                                   rotation=40, ha="right", fontsize=6.5)
                # Group labels below x-tick labels (bottom row only)
                from matplotlib.transforms import blended_transform_factory
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                fid_center  = (n_fid_vis - 1) / 2
                miss_center = n_fid_vis + (n_metrics - n_fid_vis - 1) / 2
                ax.text(fid_center,  -0.38, "Fidelity",    ha="center", va="top",
                        fontsize=7, fontstyle="italic", transform=trans)
                if n_metrics > n_fid_vis:
                    ax.text(miss_center, -0.38, "Missingness", ha="center", va="top",
                            fontsize=7, fontstyle="italic", transform=trans)
            else:
                ax.set_xticks([])

            # Y ticks only on leftmost column (no dataset ylabel — name is in title)
            if ci == 0:
                ax.set_yticks(range(n_scenarios))
                ax.set_yticklabels([lbl for _, lbl in SCENARIO_ORDER], fontsize=7)
            else:
                ax.set_yticks([])

            ax.tick_params(length=0)

    if im_last is not None:
        fig.colorbar(im_last, ax=axes, orientation="vertical",
                     fraction=0.015, pad=0.02, label="Mean score (0–1)")

    _save(fig, "fig_metric_heatmap_by_size")


def fig_multivariate_across_scenarios(all_results: dict[str, dict]) -> None:
    """
    Line plot of multivariate fidelity metric scores across single scenarios.

    One line per (metric × dataset) combination: colour encodes the metric,
    line style encodes the dataset (solid = Diabetes, dashed = MIMIC).
    Each point is the mean across all replicates (pooled over sample sizes);
    shaded band = ±1 std.  Only baseline + fidelity + missingness scenarios shown.
    """
    MV_METRICS = {
        "fidelity_auc_roc":        ("AUC-ROC",    WONG["blue"]),
        "fidelity_propensity_mse": ("Prop. MSE",  WONG["vermil"]),
        "fidelity_crcl_rs":        ("CrCl-RS",    WONG["green"]),
        "fidelity_crcl_sr":        ("CrCl-SR",    WONG["orange"]),
    }
    DS_STYLES = {
        "PID":         {"ls": "-",  "marker": "o"},
        "MIMIC-IV-ED (CC)": {"ls": "--", "marker": "s"},
    }
    single = lambda s: s == "baseline" or s.startswith("fidelity") or s.startswith("missingness")

    # Aggregate: for each dataset, pool all replicates across sample sizes per scenario
    # Returns {scenario: {metric: (mean, std)}}
    def _aggregate(results):
        from collections import defaultdict
        buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for key, entry in results.items():
            m = re.match(r"^(.+?)(?:_n\d+)?$", key)
            scenario = m.group(1) if m else key
            if not single(scenario):
                continue
            for row in entry["per_dataset"]:
                for col in MV_METRICS:
                    if col in row and row[col] is not None:
                        buckets[scenario][col].append(float(row[col]))
        out = {}
        for s, metrics in buckets.items():
            out[s] = {col: (float(np.mean(v)), float(np.std(v)))
                      for col, v in metrics.items()}
        return out

    dataset_aggs = {dname: _aggregate(res) for dname, res in all_results.items()}

    # Unified scenario order across both datasets
    all_scenarios = sorted(
        {s for agg in dataset_aggs.values() for s in agg},
        key=_scenario_sort_key,
    )
    x = np.arange(len(all_scenarios))

    fig, ax = plt.subplots(figsize=(FULL_W * 1.2, 3.2))

    for col, (metric_label, color) in MV_METRICS.items():
        for dname, agg in dataset_aggs.items():
            style = DS_STYLES.get(dname, {"ls": "-", "marker": "o"})
            means = np.array([agg.get(s, {}).get(col, (np.nan, np.nan))[0]
                              for s in all_scenarios])
            stds  = np.array([agg.get(s, {}).get(col, (np.nan, np.nan))[1]
                              for s in all_scenarios])
            mask = ~np.isnan(means)
            ax.plot(x[mask], means[mask],
                    color=color, ls=style["ls"], marker=style["marker"],
                    markersize=3.5, lw=1.2,
                    label=f"{metric_label} ({dname})")
            ax.fill_between(x[mask],
                            means[mask] - stds[mask],
                            means[mask] + stds[mask],
                            color=color, alpha=0.10)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick_label(s) for s in all_scenarios], fontsize=7.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(all_scenarios) - 0.5)
    _despine(ax)
    _draw_group_dividers(ax, all_scenarios, label_offset=-0.10)

    # Legend: two-level — metric colour rows, then dataset linestyle note
    legend_handles = []
    for col, (metric_label, color) in MV_METRICS.items():
        legend_handles.append(
            Line2D([0], [0], color=color, lw=1.5, marker="o",
                   markersize=4, label=metric_label)
        )
    for dname, style in DS_STYLES.items():
        if dname in all_results:
            legend_handles.append(
                Line2D([0], [0], color="grey", lw=1.5,
                       ls=style["ls"], marker=style["marker"],
                       markersize=4, label=dname)
            )

    ax.legend(handles=legend_handles, ncol=2, frameon=False,
              fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.5, -0.38))

    fig.tight_layout()
    _save(fig, "fig_multivariate_across_scenarios")


def fig_incremental_degradation(all_results: dict[str, dict]) -> None:
    """
    Three-panel line plot showing how fidelity, missingness, and composite scores
    degrade across scenarios, with one line per (dataset × sample size).

    X-axis  : Baseline, F1–F4, M1–M3 (in order)
    Lines   : one per (dataset × sample size); Blues ramp for Diabetes,
              Oranges ramp for MIMIC; marker shape distinguishes dataset
    Bands   : ±1 std shading
    Panels  : (a) fidelity score, (b) missingness score, (c) composite score
    """
    SCENARIO_ORDER = [
        ("baseline",      "Baseline"),
        ("fidelity_1",    "F1"),
        ("fidelity_2",    "F2"),
        ("fidelity_3",    "F3"),
        ("fidelity_4",    "F4"),
        ("missingness_1", "M1"),
        ("missingness_2", "M2"),
        ("missingness_3", "M3"),
    ]
    DIVIDER_AFTER = 4   # last fidelity scenario index

    SCORE_SPECS = [
        ("fidelity_mean",    "fidelity_std",   "Fidelity score (0–1)",    "(a)"),
        ("missingness_mean", "missingness_std", "Missingness score (0–1)", "(b)"),
        ("composite_mean",   "composite_std",   "Composite score (0–1)",   "(c)"),
    ]

    DATASET_CMAPS = {
        "PID":         mpl.cm.Blues,
        "MIMIC-IV-ED (CC)": mpl.cm.Oranges,
    }
    DATASET_MARKERS = {
        "PID":         "o",
        "MIMIC-IV-ED (CC)": "s",
    }
    DS_SHORT = {
        "PID":         "PID",
        "MIMIC-IV-ED (CC)": "MIMIC",
    }

    scenario_keys   = [sc for sc, _ in SCENARIO_ORDER]
    scenario_labels = [lbl for _, lbl in SCENARIO_ORDER]
    x = np.arange(len(SCENARIO_ORDER))

    # Build per-dataset summary dfs (all sample sizes)
    dataset_dfs: dict[str, pd.DataFrame] = {}
    for dname, results in all_results.items():
        df = build_summary_df(results)
        df = df[df["scenario"].isin(scenario_keys)].copy()
        dataset_dfs[dname] = df

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W * 1.5, 3.2), layout="constrained")
    legend_handles: list = []

    for pi, (score_col, std_col, ylabel, panel_label) in enumerate(SCORE_SPECS):
        ax = axes[pi]

        for dname, df in dataset_dfs.items():
            cmap   = DATASET_CMAPS.get(dname, mpl.cm.Blues)
            marker = DATASET_MARKERS.get(dname, "o")
            short  = DS_SHORT.get(dname, dname)

            sample_sizes = sorted(
                df["sample_size"].unique(),
                key=lambda v: v if v is not None else float("inf"),
            )
            n_sizes = len(sample_sizes)
            colors  = [cmap(v) for v in np.linspace(0.35, 0.85, max(n_sizes, 1))]

            for si, sz in enumerate(sample_sizes):
                sub    = df[df["sample_size"] == sz].set_index("scenario")
                color  = colors[si]
                size_label = f"n={sz:,}" if sz is not None else "n=full"

                means = np.array([
                    float(sub.loc[sc, score_col]) if sc in sub.index else np.nan
                    for sc in scenario_keys
                ])
                stds = np.array([
                    float(sub.loc[sc, std_col]) if sc in sub.index else np.nan
                    for sc in scenario_keys
                ])
                mask = ~np.isnan(means)

                ax.plot(x[mask], means[mask],
                        color=color, ls="-", marker=marker,
                        markersize=3.5, lw=1.2)
                ax.fill_between(x[mask],
                                means[mask] - stds[mask],
                                means[mask] + stds[mask],
                                color=color, alpha=0.12)

                # Baseline reference line
                if "baseline" in sub.index:
                    baseline_val = float(sub.loc["baseline", score_col])
                    if not np.isnan(baseline_val):
                        ax.axhline(baseline_val, color=color, lw=0.7, ls=":",
                                   alpha=0.7, zorder=0)

                if pi == 0:
                    legend_handles.append(
                        Line2D([0], [0], color=color, lw=1.2, marker=marker,
                               markersize=3.5, label=f"{short} {size_label}")
                    )

        # Vertical dividers: baseline | fidelity | missingness
        ax.axvline(0.5, color="#888888", lw=0.7, ls=":", zorder=0)
        ax.axvline(DIVIDER_AFTER + 0.5, color="#888888", lw=0.7, ls=":", zorder=0)

        ax.annotate("Baseline",
                    xy=(0, -0.13),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, fontstyle="italic")
        ax.annotate("Fidelity",
                    xy=((1 + DIVIDER_AFTER) / 2, -0.13),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, fontstyle="italic")
        ax.annotate("Missingness",
                    xy=((DIVIDER_AFTER + 1 + len(SCENARIO_ORDER) - 1) / 2, -0.13),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, fontstyle="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel(ylabel)
        _despine(ax)
        ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    # Legend: Diabetes entries first, then MIMIC — naturally ordered by dataset loop
    n_per_row = max(len(legend_handles) // 2, 1)
    fig.legend(
        handles=legend_handles, ncol=n_per_row, frameon=False,
        fontsize=7.5, handlelength=1.4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )

    _save(fig, "fig_incremental_degradation")


def fig_fidelity_vs_missingness(
    all_results: dict[str, dict],
    size_per_dataset: Optional[dict] = None,
) -> None:
    """
    Scatter plot: fidelity score (x) vs. missingness score (y).

    Each point is one replicate.  Colour encodes condition type:
      - Baseline (grey)
      - Fidelity-degraded (F1–F5, blue)
      - Missingness-degraded (M1–M5, vermillion)

    A diagonal y=x reference line shows where both scores are equal.
    Pearson r is annotated per panel.  Faceted by dataset.
    """
    CONDITION_ORDER  = ["baseline", "fidelity", "missingness"]
    CONDITION_COLORS = {
        "baseline":    "#888888",
        "fidelity":    WONG["blue"],
        "missingness": WONG["vermil"],
    }
    CONDITION_LABELS = {
        "baseline":    "Baseline",
        "fidelity":    "Fidelity-degraded (F1–F5)",
        "missingness": "Missingness-degraded (M1–M5)",
    }

    dataset_names = list(all_results.keys())
    n_ds = len(dataset_names)

    fig, axes = plt.subplots(1, n_ds,
                             figsize=(3.6 * n_ds + 0.4, 3.6),
                             layout="constrained")
    if n_ds == 1:
        axes = [axes]

    for ax, dname in zip(axes, dataset_names):
        results     = all_results[dname]
        target_size = size_per_dataset.get(dname) if size_per_dataset else None

        rows = []
        for key, entry in results.items():
            m        = re.match(r"^(.+?)(?:_n(\d+))?$", key)
            scenario = m.group(1) if m else key

            if size_per_dataset is not None and entry.get("sample_size") != target_size:
                continue

            if scenario == "baseline":
                cond = "baseline"
            elif scenario.startswith("fidelity"):
                cond = "fidelity"
            elif scenario.startswith("missingness"):
                cond = "missingness"
            else:
                continue

            for row in entry.get("per_dataset", []):
                fid  = row.get("fidelity_overall")
                miss = row.get("missingness_overall")
                if fid is not None and miss is not None:
                    rows.append({"fidelity": float(fid), "missingness": float(miss),
                                 "condition": cond})

        df = pd.DataFrame(rows)

        # Plot condition groups: fidelity and missingness first, baseline on top
        for cond in ["fidelity", "missingness", "baseline"]:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            ax.scatter(sub["fidelity"], sub["missingness"],
                       c=CONDITION_COLORS[cond], alpha=0.35, s=7,
                       linewidths=0, rasterized=True,
                       label=CONDITION_LABELS[cond])

        # Diagonal y = x reference line
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.9, ls="--", zorder=0)

        # Pearson r annotation (all points pooled)
        if len(df) > 1:
            r, _ = stats.pearsonr(df["fidelity"], df["missingness"])
            ax.text(0.04, 0.96, f"r = {r:.2f}",
                    transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left", color="#333333")

        sz_str = f"  (n={target_size:,})" if target_size is not None else ""
        ax.set_title(f"{dname}{sz_str}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Fidelity score")
        ax.set_ylabel("Missingness score")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal")
        _despine(ax)

    legend_handles = [
        mpl.patches.Patch(facecolor=CONDITION_COLORS[c], alpha=0.7,
                          label=CONDITION_LABELS[c])
        for c in CONDITION_ORDER
    ]
    fig.legend(handles=legend_handles, ncol=len(legend_handles), frameon=False,
               fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)

    suffix = "_selected" if size_per_dataset else ""
    _save(fig, f"fig_fidelity_vs_missingness{suffix}")


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

    print("Generating fig1_selected (Diabetes full, MIMIC n=10,000)...")
    fig1_selected(all_results, {
        "PID":         None,
        "MIMIC-IV-ED (CC)": 10000,
    })

    print("Generating fig2_sample_sizes...")
    fig_sample_sizes(all_results)

    print("Generating fig_violin_sample_sizes...")
    fig_violin_sample_sizes(all_results)

    print("Generating fig3_std_vs_sample_size...")
    fig_std_vs_sample_size(all_results)

    print("Exporting std table (.tex)...")
    export_std_table(all_results)

    print("Generating fig_appendix_composite...")
    fig_appendix_composite(all_results)

    print("Generating fig_appendix_sample_sizes...")
    fig_appendix_sample_sizes(all_results)

    print("Generating fig_baseline_metric_scores...")
    fig_baseline_metric_scores(all_results)

    print("Generating fig_baseline_metric_scores_selected (PID full, MIMIC n=10,000)...")
    fig_baseline_metric_scores(all_results, size_per_dataset={
        "PID":              None,
        "MIMIC-IV-ED (CC)": 10000,
    })

    print("Generating fig_scenario_metric_grid_selected (PID full, MIMIC n=10,000)...")
    fig_scenario_metric_grid(all_results, size_per_dataset={
        "PID":              None,
        "MIMIC-IV-ED (CC)": 10000,
    })

    print("Generating fig_metric_heatmap_by_size...")
    fig_metric_heatmap_by_size(all_results)

    print("Generating fig_metric_heatmap_selected (PID full, MIMIC n=10,000)...")
    fig_metric_heatmap(all_results, size_per_dataset={
        "PID":              None,
        "MIMIC-IV-ED (CC)": 10000,
    })

    print("Generating fig_multivariate_across_scenarios...")
    fig_multivariate_across_scenarios(all_results)

    print("Generating fig_fidelity_vs_missingness (PID full, MIMIC n=10,000)...")
    fig_fidelity_vs_missingness(all_results, size_per_dataset={
        "PID":              None,
        "MIMIC-IV-ED (CC)": 10000,
    })

    print("Generating fig_incremental_degradation...")
    fig_incremental_degradation(all_results)

    print("Done.")

