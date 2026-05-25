"""Main-evaluation human-facing readout renderer.

Reads statistics/diagnostic CSVs
from disk and writes the human-facing readout artefacts:

- eight static PNG figures under
  ``<readout_dir>/figures/``;
- an optional GIF animation of the degradation curves;
- a concise labelling-only ``readout_summary.md`` under the same
  readout directory.

The module is read-only with respect to experiment records,
artefacts, protocol constants, fitting code, metric code, and the
decision log. No model is fit, no metric is recomputed, no
selection is made post hoc.

All prose is labelling/reference only; no method ranking, no
hypothesis verdict, no interpretive language.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EXPECTED_MAIN_EVALUATION_RUN_HASH12: str = "864fe6722256"

# Predeclared baseline conditions.
BASELINE_LABEL_PRIOR_FREE: str = "prior_free"
BASELINE_LABEL_MATCHED_L1: str = "matched_l1"
BASELINE_LABEL_SOFT_CLEAN_CONF1: str = "soft_frobenius_clean_conf1"
BASELINE_LABEL_HARD_EXCLUSION_CLEAN: str = "hard_exclusion_clean"

BASELINE_CONDITION_LABELS: tuple[str, ...] = (
    BASELINE_LABEL_PRIOR_FREE,
    BASELINE_LABEL_MATCHED_L1,
    BASELINE_LABEL_SOFT_CLEAN_CONF1,
    BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
)

# Lower-is-better metrics (SID, SHD, MMD). edge_count is a sparsity
# diagnostic; lower means sparser, not better.
METRIC_LOWER_IS_BETTER: tuple[str, ...] = ("sid", "shd", "mmd")
METRIC_EDGE_COUNT: str = "edge_count_from_thresholded_adjacency"

# Frozen protocol constants for caveats text only.
MATCHED_L1_LAMBDA1: float = 0.0625
LAMBDA_PRIOR: float = 2e-4
N_EVALUATION_SEEDS: int = 7

# Output filenames.
FIG_BASELINE: str = "fig01_baseline_comparison_sid_shd_mmd.png"
FIG_REFERENCE_FORBIDDEN: str = "fig02_reference_forbidden_edge_suppression.png"
FIG_DEGRADATION_SID: str = "fig03_degradation_curves_sid.png"
FIG_DEGRADATION_MMD: str = "fig04_degradation_curves_mmd.png"
FIG_SOFT_SID_HEATMAP: str = "fig05_soft_frobenius_sid_heatmap.png"
FIG_SOFT_MMD_HEATMAP: str = "fig06_soft_frobenius_mmd_heatmap.png"
FIG_SID_MMD_SCATTER: str = "fig07_sid_vs_mmd_correlation.png"
FIG_EDGE_COUNT_DIAG: str = "fig08_edge_count_and_engagement_diagnostic.png"

READOUT_SUMMARY_FILENAME: str = "readout_summary.md"
DEGRADATION_GIF_FILENAME: str = "degradation.gif"

# Input filenames.
FLAT_RECORDS_CSV: str = "main_evaluation_flat_records.csv"
BASELINE_COMPARISON_CSV: str = "baseline_comparison.csv"
PAIRED_SEED_COMPARISONS_CSV: str = "paired_seed_comparisons.csv"
METRIC_CORRELATIONS_CSV: str = "metric_correlations.csv"
DEGRADATION_SUMMARY_CSV: str = "degradation_summary.csv"
FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV: str = (
    "forbidden_edge_engagement_summary.csv"
)
REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV: str = (
    "reference_forbidden_edge_comparison.csv"
)
PER_INTERVENTION_MMD_SUMMARY_CSV: str = "per_intervention_mmd_summary.csv"
STATISTICS_SUMMARY_JSON: str = "statistics_summary.json"

# Method-family palette. Deterministic and colour-blind aware.
# Vivid four-colour set (blue / green / red / orange) drawn from the
# same family as the soft_frobenius confidence palette used by the
# degradation curves, so colours stay visually consistent across all
# eight figures.
_FAMILY_COLOURS: dict[str, str] = {
    "prior_free": "#0072B2",      # dark blue
    "matched_l1": "#009E73",      # bluish green
    "soft_frobenius": "#B53737",  # dark red
    "hard_exclusion": "#E07B39",  # warm orange
}

_BASELINE_COLOURS: dict[str, str] = {
    BASELINE_LABEL_PRIOR_FREE: _FAMILY_COLOURS["prior_free"],
    BASELINE_LABEL_MATCHED_L1: _FAMILY_COLOURS["matched_l1"],
    BASELINE_LABEL_SOFT_CLEAN_CONF1: _FAMILY_COLOURS["soft_frobenius"],
    BASELINE_LABEL_HARD_EXCLUSION_CLEAN: _FAMILY_COLOURS["hard_exclusion"],
}

# Ordered family list for legend ordering in figures that mix families.
_FAMILY_ORDER: tuple[str, ...] = (
    "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
)

# Shared short labels for the four baseline conditions.
_BASELINE_SHORT_LABELS: tuple[str, ...] = (
    "prior-free", "matched-L1", "soft (clean, conf=1)", "hard-excl (clean)",
)

# Global style settings.
_STYLE_DPI: int = 180
_STYLE_FONT_SIZE_BASE: float = 10.0
_STYLE_FONT_SIZE_TITLE: float = 11.0
_STYLE_FONT_SIZE_TICK: float = 9.0


def _apply_global_style() -> None:
    """Apply a single coherent matplotlib style to every figure.

    Idempotent; safe to call at the top of every plotting function.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "font.size": _STYLE_FONT_SIZE_BASE,
        "axes.titlesize": _STYLE_FONT_SIZE_TITLE,
        "axes.titleweight": "regular",
        "axes.labelsize": _STYLE_FONT_SIZE_BASE,
        "xtick.labelsize": _STYLE_FONT_SIZE_TICK,
        "ytick.labelsize": _STYLE_FONT_SIZE_TICK,
        "legend.fontsize": _STYLE_FONT_SIZE_TICK,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def main_evaluation_readout_dir(
    output_root: Path, main_evaluation_run_hash12: str
) -> Path:
    """Return the readout directory for a given main-evaluation run."""
    return (
        output_root
        / "results"
        / "main_study"
        / "main_evaluation"
        / main_evaluation_run_hash12
        / "readout"
    )


def read_csv_table(path: Path) -> pd.DataFrame:
    """Read a CSV. Empty cells become NaN."""
    if not isinstance(path, Path):
        raise TypeError(
            f"read_csv_table requires a pathlib.Path; got "
            f"{type(path).__name__}."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"read_csv_table: {path!r} does not exist."
        )
    return pd.read_csv(path)


def read_statistics_summary(path: Path) -> dict[str, Any]:
    """Read the statistics_summary.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"read_statistics_summary: {path!r} does not exist."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dirs(readout_dir: Path) -> dict[str, Path]:
    """Create ``figures/`` and ``gif_frames/`` under ``readout_dir``."""
    figures_dir = readout_dir / "figures"
    gif_frames_dir = readout_dir / "gif_frames"
    figures_dir.mkdir(parents=True, exist_ok=True)
    gif_frames_dir.mkdir(parents=True, exist_ok=True)
    return {
        "readout_dir": readout_dir,
        "figures_dir": figures_dir,
        "gif_frames_dir": gif_frames_dir,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_STYLE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


_PRESENTATION_DASH: str = "—"  # em dash


def _present_missing(value: Any) -> str:
    """Render a presentation-only cell. NaN/None become an em dash."""
    if value is None:
        return _PRESENTATION_DASH
    if isinstance(value, float) and math.isnan(value):
        return _PRESENTATION_DASH
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def presentation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation copy of ``df`` with NaN cells masked.

    Display-only transform; the source DataFrame is not mutated.
    Numeric cells are rendered with four significant figures and
    NaN/None cells become an em dash. Used by the notebook and by
    the readout-summary markdown writer to avoid showing raw
    ``Missing value`` text for method-inapplicable cells.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_present_missing)
    return out


def _baseline_row_filter(
    flat: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Return rows for one predeclared baseline condition."""
    if label == BASELINE_LABEL_PRIOR_FREE:
        return flat[flat["method_family"] == "prior_free"]
    if label == BASELINE_LABEL_MATCHED_L1:
        return flat[flat["method_family"] == "matched_l1"]
    if label == BASELINE_LABEL_SOFT_CLEAN_CONF1:
        return flat[
            (flat["method_family"] == "soft_frobenius")
            & (np.isclose(flat["corruption_fraction"].fillna(-1), 0.0))
            & (np.isclose(flat["confidence"].fillna(-1), 1.0))
        ]
    if label == BASELINE_LABEL_HARD_EXCLUSION_CLEAN:
        return flat[
            (flat["method_family"] == "hard_exclusion")
            & (np.isclose(flat["corruption_fraction"].fillna(-1), 0.0))
        ]
    raise ValueError(
        f"_baseline_row_filter: unknown baseline label {label!r}."
    )


def _draw_bar_with_dots(
    ax,
    *,
    x_positions: np.ndarray,
    values_by_label: dict[str, np.ndarray],
    seed_offset: int,
) -> None:
    """Bar (mean) + per-seed jittered dots, one column per baseline label.

    Bars use the family colour with light fill; per-seed dots use the
    same family colour with a white edge so individual points remain
    distinguishable against the bar.
    """
    for idx, label in enumerate(BASELINE_CONDITION_LABELS):
        values = values_by_label.get(label, np.array([]))
        n = len(values)
        colour = _BASELINE_COLOURS[label]
        if n == 0:
            continue
        mean_v = float(np.mean(values))
        ax.bar(
            x_positions[idx], mean_v,
            color=colour, alpha=0.42,
            edgecolor=colour, linewidth=0.9, width=0.7,
            zorder=1,
        )
        rng = np.random.default_rng(idx * 17 + seed_offset)
        jitter = rng.uniform(-0.18, 0.18, size=n)
        ax.scatter(
            np.full(n, x_positions[idx]) + jitter, values,
            s=34, color=colour,
            edgecolor="white", linewidth=0.9,
            alpha=0.95, zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        _BASELINE_SHORT_LABELS, rotation=18, ha="right",
    )
    ax.set_xlim(-0.5, len(x_positions) - 0.5)
    ax.grid(axis="y", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_baseline_comparison(
    flat: pd.DataFrame, output_path: Path
) -> Path:
    """Three panels (SID / SHD / MMD), bar with mean + per-seed dots.

    Lower-is-better is annotated on each panel. Per-seed dots use a
    white edge so individual seeds remain readable on top of the bar.
    The figure is descriptive only; no comparative ranking language
    is included.
    """
    _apply_global_style()
    metrics = ("sid", "shd", "mmd")
    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 3.8), constrained_layout=True
    )
    x_positions = np.arange(len(BASELINE_CONDITION_LABELS))
    for ax, metric in zip(axes, metrics):
        values_by_label = {
            label: _baseline_row_filter(flat, label)[metric]
            .dropna().values
            for label in BASELINE_CONDITION_LABELS
        }
        _draw_bar_with_dots(
            ax,
            x_positions=x_positions,
            values_by_label=values_by_label,
            seed_offset=3,
        )
        ax.set_title(f"{metric.upper()} (lower is better)")
        ax.set_ylabel(metric.upper())
    fig.suptitle(
        "Baseline comparison (bar = mean; dots = per-seed; n = 7 seeds)",
        fontsize=_STYLE_FONT_SIZE_TITLE,
    )
    return _save_figure(fig, output_path)


def plot_reference_forbidden_edge_suppression(
    reference_df: pd.DataFrame, output_path: Path
) -> Path:
    """Mean |W| on the per-seed clean-soft prior forbidden edges.

    Bar (mean) + per-seed dots per condition.
    """
    _apply_global_style()
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    x_positions = np.arange(len(BASELINE_CONDITION_LABELS))
    values_by_label = {
        label: reference_df[
            reference_df["condition_label"] == label
        ]["mean_abs_w_reference_forbidden_edges"].dropna().values
        for label in BASELINE_CONDITION_LABELS
    }
    _draw_bar_with_dots(
        ax,
        x_positions=x_positions,
        values_by_label=values_by_label,
        seed_offset=5,
    )
    ax.set_ylabel("mean |W| on reference forbidden edges")
    ax.set_title(
        "Reference forbidden-edge engagement (mechanism diagnostic)"
    )
    return _save_figure(fig, output_path)


def _aggregate_metric_by_corruption(
    rows: pd.DataFrame, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(corruption_levels, means, stds)`` for ``metric``."""
    cf_values = sorted(
        float(c) for c in rows["corruption_fraction"].dropna().unique()
    )
    means: list[float] = []
    stds: list[float] = []
    for cf in cf_values:
        sub = rows[
            np.isclose(rows["corruption_fraction"].fillna(-1), cf)
        ]
        values = sub[metric].dropna().values
        if len(values) == 0:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        means.append(float(np.mean(values)))
        if len(values) >= 2:
            stds.append(float(np.std(values, ddof=1)))
        else:
            stds.append(0.0)
    return (
        np.array(cf_values, dtype=float),
        np.array(means, dtype=float),
        np.array(stds, dtype=float),
    )


# Distinct, colour-blind-aware palette for the five soft_frobenius
# confidence levels. Chosen so every level is easily distinguishable
# from the warm-orange hard_exclusion line and from each other on a
# white background.
_SOFT_CONFIDENCE_PALETTE: tuple[str, ...] = (
    "#56B4E9",  # sky blue       (conf = 0.0)
    "#009E73",  # bluish green   (conf = 0.25)
    "#CC79A7",  # reddish purple (conf = 0.5)
    "#0072B2",  # dark blue      (conf = 0.75)
    "#B53737",  # dark red       (conf = 1.0)
)


def _soft_confidence_colour(k: int, n_total: int) -> str:
    """Return a distinct hue for the k-th soft_frobenius confidence level.

    Cycles through :data:`_SOFT_CONFIDENCE_PALETTE`; if more than five
    levels are present, later levels reuse the palette from the start
    (deterministic).
    """
    if n_total <= 0:
        return _SOFT_CONFIDENCE_PALETTE[0]
    return _SOFT_CONFIDENCE_PALETTE[k % len(_SOFT_CONFIDENCE_PALETTE)]


def plot_degradation_curve(
    flat: pd.DataFrame, *, metric: str, output_path: Path
) -> Path:
    """Mean ``metric`` vs corruption_fraction. One line per condition.

    hard_exclusion is one bold warm-orange line. Each soft_frobenius
    confidence level uses its own distinct, colour-blind-aware hue
    so the five levels are easy to tell apart. No uncertainty
    shading; the mean line is the only visual.
    """
    _apply_global_style()
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    hard_rows = flat[flat["method_family"] == "hard_exclusion"]
    cf_h, mean_h, _ = _aggregate_metric_by_corruption(hard_rows, metric)
    if len(cf_h) > 0:
        colour_h = _FAMILY_COLOURS["hard_exclusion"]
        ax.plot(
            cf_h, mean_h,
            label="hard_exclusion",
            color=colour_h, marker="s", markersize=6,
            linewidth=2.4, zorder=5,
        )
    soft_rows = flat[flat["method_family"] == "soft_frobenius"]
    if len(soft_rows) > 0:
        confidences = sorted(
            float(c) for c in soft_rows["confidence"].dropna().unique()
        )
        for k, cn in enumerate(confidences):
            sub = soft_rows[
                np.isclose(soft_rows["confidence"].fillna(-1), cn)
            ]
            cf_s, mean_s, _ = _aggregate_metric_by_corruption(sub, metric)
            colour = _soft_confidence_colour(k, len(confidences))
            ax.plot(
                cf_s, mean_s,
                label=f"soft conf={cn}",
                color=colour, marker="o", markersize=5,
                linewidth=1.8, alpha=1.0, zorder=3,
            )
    ax.set_xlabel("corruption_fraction")
    ax.set_ylabel(f"{metric.upper()} (lower is better)")
    ax.set_title(
        f"{metric.upper()} vs corruption (mean across 7 seeds)"
    )
    ax.grid(alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = ax.legend(
        loc="best", framealpha=0.92, fontsize=_STYLE_FONT_SIZE_TICK,
        ncol=1, handlelength=2.0,
    )
    leg.get_frame().set_edgecolor("#CCCCCC")
    return _save_figure(fig, output_path)


def plot_soft_frobenius_heatmap(
    flat: pd.DataFrame, *, metric: str, output_path: Path
) -> Path:
    """Mean ``metric`` over soft_frobenius (corruption x confidence)."""
    _apply_global_style()
    soft = flat[flat["method_family"] == "soft_frobenius"]
    if soft.empty:
        raise ValueError(
            "plot_soft_frobenius_heatmap: no soft_frobenius rows in "
            "the flat table."
        )
    cf_values = sorted(
        float(c) for c in soft["corruption_fraction"].dropna().unique()
    )
    cn_values = sorted(
        float(c) for c in soft["confidence"].dropna().unique()
    )
    matrix = np.full(
        (len(cn_values), len(cf_values)), float("nan"), dtype=float
    )
    for i, cn in enumerate(cn_values):
        for j, cf in enumerate(cf_values):
            sub = soft[
                np.isclose(soft["confidence"].fillna(-1), cn)
                & np.isclose(
                    soft["corruption_fraction"].fillna(-1), cf
                )
            ]
            values = sub[metric].dropna().values
            if len(values):
                matrix[i, j] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    im = ax.imshow(
        matrix, aspect="auto", origin="lower", cmap="viridis",
    )
    ax.set_xticks(range(len(cf_values)))
    ax.set_xticklabels([f"{cf:g}" for cf in cf_values])
    ax.set_yticks(range(len(cn_values)))
    ax.set_yticklabels([f"{cn:g}" for cn in cn_values])
    ax.set_xlabel("corruption_fraction")
    ax.set_ylabel("confidence")
    ax.set_title(
        f"soft_frobenius mean {metric.upper()} (n = 7 seeds)"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"mean {metric.upper()}")
    cbar.outline.set_edgecolor("#CCCCCC")
    cbar.outline.set_linewidth(0.6)
    # Annotate cells with the numeric mean.
    mid = float(np.nanmean(matrix))
    for i in range(len(cn_values)):
        for j in range(len(cf_values)):
            val = matrix[i, j]
            if not math.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val < mid else "black",
                    fontsize=8,
                )
    # Heatmaps don't need overlaid gridlines.
    ax.grid(False)
    return _save_figure(fig, output_path)


def plot_sid_mmd_scatter(
    flat: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> Path:
    """SID vs MMD across all records, coloured by method_family.

    Overall Pearson / Spearman / Kendall tau-b are annotated from
    metric_correlations.csv (the ``group_label == 'all'`` row for
    ``x_metric=='sid'`` and ``y_metric=='mmd'``).
    """
    _apply_global_style()
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    families_present = set(flat["method_family"].dropna().unique())
    markers = {"prior_free": "o", "matched_l1": "s",
               "soft_frobenius": "^", "hard_exclusion": "D"}
    for fam in _FAMILY_ORDER:
        if fam not in families_present:
            continue
        sub = flat[flat["method_family"] == fam]
        ax.scatter(
            sub["sid"], sub["mmd"],
            color=_FAMILY_COLOURS.get(fam, "gray"),
            marker=markers.get(fam, "o"),
            s=22, alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=fam,
        )
    ax.set_xlabel("SID (lower is better)")
    ax.set_ylabel("MMD (lower is better)")
    overall = correlations[
        (correlations["group_label"] == "all")
        & (correlations["x_metric"] == "sid")
        & (correlations["y_metric"] == "mmd")
    ]
    if not overall.empty:
        row = overall.iloc[0]
        ann = (
            f"n = {int(row['n'])}    "
            f"r = {float(row['pearson']):.2f}    "
            f"rho = {float(row['spearman']):.2f}    "
            f"tau_b = {float(row['kendall_tau_b']):.2f}"
        )
        ax.text(
            0.02, 0.98, ann,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=_STYLE_FONT_SIZE_TICK,
            bbox=dict(
                facecolor="white", alpha=0.92,
                edgecolor="#CCCCCC", boxstyle="round,pad=0.25",
                linewidth=0.6,
            ),
        )
    ax.set_title("SID vs MMD (all 224 records)")
    ax.grid(alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = ax.legend(
        loc="lower right", framealpha=0.92,
        fontsize=_STYLE_FONT_SIZE_TICK, handlelength=1.6,
        markerscale=1.1,
    )
    leg.get_frame().set_edgecolor("#CCCCCC")
    return _save_figure(fig, output_path)


def plot_edge_count_engagement_diagnostic(
    flat: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Two-panel diagnostic: edge_count and reference fraction-above-threshold.

    Both panels use bar (mean) + per-seed dots. Left panel is the
    thresholded-W edge count (sparser when lower), right panel is
    the fraction of reference forbidden edges with |W| >= 0.3.
    """
    _apply_global_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), constrained_layout=True
    )
    x_positions = np.arange(len(BASELINE_CONDITION_LABELS))
    # Left: edge_count per baseline.
    left_values = {
        label: _baseline_row_filter(flat, label)[METRIC_EDGE_COUNT]
        .dropna().values
        for label in BASELINE_CONDITION_LABELS
    }
    _draw_bar_with_dots(
        ax_left, x_positions=x_positions,
        values_by_label=left_values, seed_offset=7,
    )
    ax_left.set_ylabel("edge_count (sparser when lower; diagnostic)")
    ax_left.set_title("Thresholded-W edge count by baseline")
    # Right: reference forbidden-edge fraction above 0.3 threshold.
    right_values = {
        label: reference_df[
            reference_df["condition_label"] == label
        ]["fraction_reference_forbidden_above_threshold"]
        .dropna().values
        for label in BASELINE_CONDITION_LABELS
    }
    _draw_bar_with_dots(
        ax_right, x_positions=x_positions,
        values_by_label=right_values, seed_offset=11,
    )
    ax_right.set_ylabel(
        "fraction of reference forbidden edges with |W| >= 0.3"
    )
    ax_right.set_title("Reference forbidden fraction above threshold")
    fig.suptitle(
        "Edge-count and prior-edge engagement diagnostic (n = 7 seeds)",
        fontsize=_STYLE_FONT_SIZE_TITLE,
    )
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Optional GIF rendering
# ---------------------------------------------------------------------------


def _try_render_degradation_gif(
    flat: pd.DataFrame,
    *,
    metric: str,
    gif_frames_dir: Path,
    output_path: Path,
) -> Optional[Path]:
    """Best-effort degradation GIF via matplotlib + PIL/imageio.

    Returns the GIF path on success, ``None`` on any failure. Each
    frame is one corruption level; bars per condition show mean
    ``metric``.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        try:
            import imageio.v3 as iio  # type: ignore[import-not-found]
        except Exception:
            return None
        Image = None  # type: ignore[assignment]
    else:
        iio = None  # type: ignore[assignment]
    cf_values = sorted(
        float(c) for c in
        flat["corruption_fraction"].dropna().unique()
    )
    if not cf_values:
        return None
    soft = flat[flat["method_family"] == "soft_frobenius"]
    hard = flat[flat["method_family"] == "hard_exclusion"]
    if soft.empty and hard.empty:
        return None
    confidences = sorted(
        float(c) for c in soft["confidence"].dropna().unique()
    )
    cmap = plt.get_cmap("viridis")
    frame_paths: list[Path] = []
    overall_max = float(
        np.nanmax(
            np.concatenate([
                soft[metric].dropna().values,
                hard[metric].dropna().values,
            ])
        ) if (soft.empty is False or hard.empty is False) else 1.0
    )
    for i, cf in enumerate(cf_values):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        bars: list[tuple[str, float, str]] = []
        sub_hard = hard[
            np.isclose(hard["corruption_fraction"].fillna(-1), cf)
        ][metric].dropna().values
        if len(sub_hard):
            bars.append((
                "hard_exclusion",
                float(np.mean(sub_hard)),
                _FAMILY_COLOURS["hard_exclusion"],
            ))
        for k, cn in enumerate(confidences):
            sub_s = soft[
                np.isclose(soft["confidence"].fillna(-1), cn)
                & np.isclose(
                    soft["corruption_fraction"].fillna(-1), cf
                )
            ][metric].dropna().values
            if len(sub_s):
                colour = cmap(
                    0.15 + 0.7 * k / max(1, len(confidences) - 1)
                )
                bars.append((
                    f"soft conf={cn}", float(np.mean(sub_s)), colour,
                ))
        if not bars:
            plt.close(fig)
            continue
        names = [b[0] for b in bars]
        means = [b[1] for b in bars]
        colours = [b[2] for b in bars]
        ax.bar(range(len(bars)), means, color=colours, edgecolor="black")
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"mean {metric.upper()}")
        ax.set_ylim(0, overall_max * 1.05)
        ax.set_title(
            f"Degradation frame: corruption_fraction={cf:g} "
            f"(mean {metric.upper()} across 7 seeds)"
        )
        ax.grid(axis="y", alpha=0.25)
        frame_path = gif_frames_dir / (
            f"degradation_{metric}_cf{i:02d}.png"
        )
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)
    if not frame_paths:
        return None
    try:
        if Image is not None:
            frames = [
                Image.open(p).convert("RGB") for p in frame_paths
            ]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=900,
                loop=0,
            )
        else:
            images = [iio.imread(p) for p in frame_paths]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(output_path, images, duration=0.9, loop=0)
    except Exception:
        return None
    return output_path


# ---------------------------------------------------------------------------
# Readout summary (labelling/reference only)
# ---------------------------------------------------------------------------


def _by_condition_metric_means(
    flat: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label in BASELINE_CONDITION_LABELS:
        sub = _baseline_row_filter(flat, label)
        rows.append({
            "condition_label": label,
            "n": int(len(sub)),
            "mean_sid": (
                float(np.mean(sub["sid"].dropna().values))
                if not sub["sid"].dropna().empty else float("nan")
            ),
            "mean_shd": (
                float(np.mean(sub["shd"].dropna().values))
                if not sub["shd"].dropna().empty else float("nan")
            ),
            "mean_mmd": (
                float(np.mean(sub["mmd"].dropna().values))
                if not sub["mmd"].dropna().empty else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _reference_forbidden_means_by_condition(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label in BASELINE_CONDITION_LABELS:
        sub = reference_df[reference_df["condition_label"] == label]
        means = sub[
            "mean_abs_w_reference_forbidden_edges"
        ].dropna().values
        fracs = sub[
            "fraction_reference_forbidden_above_threshold"
        ].dropna().values
        rows.append({
            "condition_label": label,
            "n_seeds": int(len(sub)),
            "mean_abs_w_reference_forbidden_edges": (
                float(np.mean(means)) if len(means) else float("nan")
            ),
            "mean_fraction_above_threshold": (
                float(np.mean(fracs)) if len(fracs) else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _selected_correlation_rows(
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    keep = correlations[
        (correlations["group_label"] == "all")
        & (
            (correlations["x_metric"].isin(
                ("sid", "shd", "edge_count_from_thresholded_adjacency")
            ))
            & (correlations["y_metric"] == "mmd")
        )
    ].copy()
    return keep[[
        "group_label", "x_metric", "y_metric", "n",
        "pearson", "spearman", "kendall_tau_b",
    ]]


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub-style markdown table.

    NaN / None cells become an em dash to avoid showing raw
    ``Missing value`` text in the readout summary. Float cells are
    rendered with four significant figures.
    """
    if df.empty:
        return "(no rows)"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if v is None:
                cells.append(_PRESENTATION_DASH)
            elif isinstance(v, float):
                if math.isnan(v):
                    cells.append(_PRESENTATION_DASH)
                else:
                    cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines])


def write_readout_summary(
    *,
    main_evaluation_run_hash12: str,
    readout_dir: Path,
    flat: pd.DataFrame,
    reference_df: pd.DataFrame,
    correlations: pd.DataFrame,
    statistics_summary: dict[str, Any],
    figure_paths: dict[str, Path],
    extra_outputs: dict[str, Path],
    output_path: Path,
) -> Path:
    """Write a concise labelling-only ``readout_summary.md``."""
    means_df = _by_condition_metric_means(flat)
    ref_df = _reference_forbidden_means_by_condition(reference_df)
    corr_df = _selected_correlation_rows(correlations)

    figure_list = "\n".join(
        f"- `{name}`: `{p.relative_to(readout_dir)}`"
        for name, p in sorted(figure_paths.items())
    )
    extras_list = "\n".join(
        f"- `{name}`: `{p.relative_to(readout_dir)}`"
        for name, p in sorted(extra_outputs.items())
    )

    lines: list[str] = []
    lines.append("# Main-evaluation readout summary")
    lines.append("")
    lines.append("## 1. Run identity")
    lines.append("")
    lines.append(
        f"- `main_evaluation_run_hash12`: `{main_evaluation_run_hash12}`"
    )
    if "input_flat_csv" in statistics_summary:
        lines.append(
            f"- statistics input: `{statistics_summary['input_flat_csv']}`"
        )
    lines.append("")
    lines.append("## 2. Evidence files used")
    lines.append("")
    lines.append(
        "Inputs are upstream readout tables under "
        f"`results/main_study/main_evaluation/{main_evaluation_run_hash12}/readout/`."
    )
    counts_table = pd.DataFrame([
        {"input": FLAT_RECORDS_CSV,
         "rows": int(statistics_summary.get("n_flat_rows", len(flat)))},
        {"input": BASELINE_COMPARISON_CSV,
         "rows": int(statistics_summary.get("n_baseline_rows", 0))},
        {"input": PAIRED_SEED_COMPARISONS_CSV,
         "rows": int(statistics_summary.get(
             "n_paired_comparison_rows", 0))},
        {"input": METRIC_CORRELATIONS_CSV,
         "rows": int(statistics_summary.get("n_correlation_rows", 0))},
        {"input": DEGRADATION_SUMMARY_CSV,
         "rows": int(statistics_summary.get("n_degradation_rows", 0))},
        {"input": FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV,
         "rows": int(statistics_summary.get(
             "n_forbidden_engagement_rows", 0))},
        {"input": REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV,
         "rows": int(statistics_summary.get(
             "n_reference_forbidden_rows", 0))},
        {"input": PER_INTERVENTION_MMD_SUMMARY_CSV,
         "rows": int(statistics_summary.get(
             "n_per_intervention_mmd_summary_rows", 0))},
    ])
    lines.append("")
    lines.append(_df_to_markdown_table(counts_table))
    lines.append("")
    lines.append("## 3. Output files generated")
    lines.append("")
    lines.append("### Figures")
    lines.append("")
    lines.append(figure_list if figure_list else "(none)")
    lines.append("")
    if extras_list:
        lines.append("### Other artefacts")
        lines.append("")
        lines.append(extras_list)
        lines.append("")
    lines.append("## 4. Key numerical descriptors")
    lines.append("")
    lines.append(
        "Tables only; thesis interpretation is separate."
    )
    lines.append("")
    lines.append("### 4.1 Mean SID / SHD / MMD by baseline condition")
    lines.append("")
    lines.append(_df_to_markdown_table(means_df))
    lines.append("")
    lines.append(
        "### 4.2 Reference forbidden-edge engagement means by "
        "baseline condition"
    )
    lines.append("")
    lines.append(_df_to_markdown_table(ref_df))
    lines.append("")
    lines.append(
        "### 4.3 Selected overall correlation values "
        "(group_label = 'all')"
    )
    lines.append("")
    lines.append(_df_to_markdown_table(corr_df))
    lines.append("")
    lines.append("## 5. Caveats")
    lines.append("")
    lines.append(
        f"- `matched_l1_lambda1 = {MATCHED_L1_LAMBDA1}` is frozen "
        "via the matched-L1 calibration step."
    )
    lines.append(
        f"- `lambda_prior = {LAMBDA_PRIOR}` is frozen from earlier "
        "calibration."
    )
    lines.append(
        f"- n = {N_EVALUATION_SEEDS} evaluation seeds; the headline "
        "plan is paired by seed."
    )
    lines.append(
        "- Effect sizes and interval estimates are the primary "
        "evidence; p-values are secondary."
    )
    lines.append(
        "- No exploratory lambda_prior sensitivity is included in "
        "this readout."
    )
    lines.append(
        "- Any later M-10 sensitivity analysis is separate from the "
        "frozen primary result."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def render_all_readout_outputs(
    output_root: Path,
    main_evaluation_run_hash12: str,
    *,
    make_gif: bool = False,
) -> dict[str, Path]:
    """Render every readout artefact for one main-evaluation run."""
    if not isinstance(output_root, Path):
        raise TypeError(
            f"output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    if (
        not isinstance(main_evaluation_run_hash12, str)
        or len(main_evaluation_run_hash12) != 12
    ):
        raise ValueError(
            "main_evaluation_run_hash12 must be a 12-character "
            f"string; got {main_evaluation_run_hash12!r}."
        )
    readout_dir = main_evaluation_readout_dir(
        output_root, main_evaluation_run_hash12
    )
    dirs = ensure_output_dirs(readout_dir)
    figures_dir = dirs["figures_dir"]
    gif_frames_dir = dirs["gif_frames_dir"]

    flat = read_csv_table(readout_dir / FLAT_RECORDS_CSV)
    reference_df = read_csv_table(
        readout_dir / REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV
    )
    correlations = read_csv_table(
        readout_dir / METRIC_CORRELATIONS_CSV
    )
    statistics_summary = read_statistics_summary(
        readout_dir / STATISTICS_SUMMARY_JSON
    )

    figure_paths: dict[str, Path] = {}
    figure_paths["fig01_baseline_comparison_sid_shd_mmd"] = (
        plot_baseline_comparison(
            flat, figures_dir / FIG_BASELINE,
        )
    )
    figure_paths["fig02_reference_forbidden_edge_suppression"] = (
        plot_reference_forbidden_edge_suppression(
            reference_df, figures_dir / FIG_REFERENCE_FORBIDDEN,
        )
    )
    figure_paths["fig03_degradation_curves_sid"] = (
        plot_degradation_curve(
            flat, metric="sid",
            output_path=figures_dir / FIG_DEGRADATION_SID,
        )
    )
    figure_paths["fig04_degradation_curves_mmd"] = (
        plot_degradation_curve(
            flat, metric="mmd",
            output_path=figures_dir / FIG_DEGRADATION_MMD,
        )
    )
    figure_paths["fig05_soft_frobenius_sid_heatmap"] = (
        plot_soft_frobenius_heatmap(
            flat, metric="sid",
            output_path=figures_dir / FIG_SOFT_SID_HEATMAP,
        )
    )
    figure_paths["fig06_soft_frobenius_mmd_heatmap"] = (
        plot_soft_frobenius_heatmap(
            flat, metric="mmd",
            output_path=figures_dir / FIG_SOFT_MMD_HEATMAP,
        )
    )
    figure_paths["fig07_sid_vs_mmd_correlation"] = (
        plot_sid_mmd_scatter(
            flat, correlations,
            figures_dir / FIG_SID_MMD_SCATTER,
        )
    )
    figure_paths["fig08_edge_count_and_engagement_diagnostic"] = (
        plot_edge_count_engagement_diagnostic(
            flat, reference_df,
            figures_dir / FIG_EDGE_COUNT_DIAG,
        )
    )

    extra_outputs: dict[str, Path] = {}
    if make_gif:
        gif_path = _try_render_degradation_gif(
            flat, metric="sid",
            gif_frames_dir=gif_frames_dir,
            output_path=readout_dir / DEGRADATION_GIF_FILENAME,
        )
        if gif_path is not None:
            extra_outputs["degradation_gif"] = gif_path

    summary_path = write_readout_summary(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        readout_dir=readout_dir,
        flat=flat,
        reference_df=reference_df,
        correlations=correlations,
        statistics_summary=statistics_summary,
        figure_paths=figure_paths,
        extra_outputs=extra_outputs,
        output_path=readout_dir / READOUT_SUMMARY_FILENAME,
    )
    return {
        **figure_paths,
        **extra_outputs,
        "readout_summary": summary_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_readout_figures",
        description=(
            "Render the human-facing readout artefacts (eight figures "
            "plus a labelling-only summary). Read-only over the "
            "upstream readout tables; no fitting, no metric "
            "recomputation, no method ranking, no hypothesis verdict."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help=(
            "Root directory under which results/main_study/... is "
            "located."
        ),
    )
    parser.add_argument(
        "--main-evaluation-run-hash12", type=str, required=True,
        help="12-character main-evaluation run hash.",
    )
    parser.add_argument(
        "--make-gif", action="store_true",
        help=(
            "If set, also attempt a degradation-curve GIF under "
            "readout/. Best-effort; requires PIL or imageio."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = render_all_readout_outputs(
            args.output_root,
            args.main_evaluation_run_hash12,
            make_gif=bool(args.make_gif),
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            f"render_readout_figures: error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return 1
    for name, path in sorted(manifest.items()):
        sys.stdout.write(f"{name}: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASELINE_CONDITION_LABELS",
    "BASELINE_LABEL_HARD_EXCLUSION_CLEAN",
    "BASELINE_LABEL_MATCHED_L1",
    "BASELINE_LABEL_PRIOR_FREE",
    "BASELINE_LABEL_SOFT_CLEAN_CONF1",
    "EXPECTED_MAIN_EVALUATION_RUN_HASH12",
    "FIG_BASELINE",
    "FIG_DEGRADATION_MMD",
    "FIG_DEGRADATION_SID",
    "FIG_EDGE_COUNT_DIAG",
    "FIG_REFERENCE_FORBIDDEN",
    "FIG_SID_MMD_SCATTER",
    "FIG_SOFT_MMD_HEATMAP",
    "FIG_SOFT_SID_HEATMAP",
    "READOUT_SUMMARY_FILENAME",
    "ensure_output_dirs",
    "main",
    "main_evaluation_readout_dir",
    "presentation_table",
    "plot_baseline_comparison",
    "plot_degradation_curve",
    "plot_edge_count_engagement_diagnostic",
    "plot_reference_forbidden_edge_suppression",
    "plot_sid_mmd_scatter",
    "plot_soft_frobenius_heatmap",
    "read_csv_table",
    "read_statistics_summary",
    "render_all_readout_outputs",
    "write_readout_summary",
]
