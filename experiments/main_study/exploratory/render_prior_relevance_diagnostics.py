"""Visual readout renderer for the exploratory prior-relevance diagnostics.

Reads persisted artefacts from the main evaluation readout and from the
two exploratory diagnostic analyses (prior structural relevance, oracle
prior relevance) and produces a labelling-only visual readout: up to
eleven figures, a summary markdown, a figure manifest JSON, and a
claim-support matrix CSV; plus a labelling-only notebook.

This module performs no fitting, no metric recomputation, no protocol
change, and no new sampling. Every figure is built from a single
persisted file.

Output directory:
    ``<output_root>/results/main_study/exploratory/
    prior_relevance_diagnostics/``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


MAIN_EVALUATION_RUN_HASH_PREFIX: str = "864fe6722256"
PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX: str = "1b46785b59a4"
ORACLE_ANALYSIS_HASH_PREFIX: str = "1b95c563db88"

METHOD_FAMILY_COLOURS: dict[str, str] = {
    "prior_free": "#7A7A7A",
    "matched_l1": "#E08E45",
    "soft_frobenius": "#1F6FB5",
    "hard_exclusion": "#B53737",
}

SCENARIO_COLOURS: dict[str, str] = {
    "actual_reference_forbidden_removal": "#7A7A7A",
    "fp_remove_budget10_exact": "#1F6FB5",
    "fp_remove_all_false_positives": "#7FB3DA",
    "fn_add_budget10_greedy_acyclic": "#E08E45",
    "fn_add_full_greedy_acyclic": "#B53737",
}

SCENARIO_DISPLAY_LABELS: dict[str, str] = {
    "actual_reference_forbidden_removal": "actual reference\nforbidden removal",
    "fp_remove_budget10_exact": "exact FP\nbudget=10",
    "fp_remove_all_false_positives": "full FP\nremoval",
    "fn_add_budget10_greedy_acyclic": "greedy FN\nbudget=10",
    "fn_add_full_greedy_acyclic": "greedy FN\nfull",
}

METHOD_FAMILY_ORDER: tuple[str, ...] = (
    "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
)

ORACLE_SCENARIO_ORDER: tuple[str, ...] = (
    "actual_reference_forbidden_removal",
    "fp_remove_budget10_exact",
    "fp_remove_all_false_positives",
    "fn_add_budget10_greedy_acyclic",
    "fn_add_full_greedy_acyclic",
)

DIAGNOSTICS_DIR_NAME: str = "prior_relevance_diagnostics"
FIG_DIR_NAME: str = "figures"

# Figure file names. Each figure is independently skippable.
FIG01_NAME: str = "fig01_main_result_clean_metrics.png"
FIG02_NAME: str = "fig02_mechanism_engagement.png"
FIG03_NAME: str = "fig03_corruption_degradation.png"
FIG04_NAME: str = "fig04_error_decomposition.png"
FIG05_NAME: str = "fig05_prior_target_overlap.png"
FIG06_NAME: str = "fig06_offline_removal_effect.png"
FIG07_NAME: str = "fig07_aggregated_error_heatmap.png"
FIG08_NAME: str = "fig08_oracle_summary.png"
FIG09_NAME: str = "fig09_oracle_per_seed_sid_delta.png"
FIG10_NAME: str = "fig10_required_edge_acyclicity.png"
FIG11_NAME: str = "fig11_fp_vs_fn_reconciliation.png"

SUMMARY_MD_NAME: str = "prior_relevance_diagnostics_summary.md"
MANIFEST_JSON_NAME: str = "prior_relevance_diagnostics_manifest.json"
NOTEBOOK_NAME: str = "prior_relevance_diagnostics.ipynb"

_STYLE_DPI: int = 180
_FONT_BASE: float = 10.0
_FONT_TITLE: float = 11.0
_FONT_TICK: float = 9.0


def _apply_style() -> None:
    """Project-consistent matplotlib style. Safe to call repeatedly."""
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
        "font.size": _FONT_BASE,
        "axes.titlesize": _FONT_TITLE,
        "axes.titleweight": "regular",
        "axes.labelsize": _FONT_BASE,
        "xtick.labelsize": _FONT_TICK,
        "ytick.labelsize": _FONT_TICK,
        "legend.fontsize": _FONT_TICK,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    })


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def main_evaluation_readout_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results" / "main_study" / "main_evaluation"
        / MAIN_EVALUATION_RUN_HASH_PREFIX / "readout"
    )


def prior_relevance_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results" / "main_study" / "exploratory"
        / "prior_structural_relevance"
        / PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX
    )


def oracle_relevance_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results" / "main_study" / "exploratory"
        / "oracle_prior_relevance"
        / ORACLE_ANALYSIS_HASH_PREFIX
    )


def diagnostics_output_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results" / "main_study" / "exploratory"
        / DIAGNOSTICS_DIR_NAME
    )


def figures_output_dir(output_root: Path) -> Path:
    return diagnostics_output_dir(output_root) / FIG_DIR_NAME


def notebook_output_path(output_root: Path) -> Path:
    return output_root / "notebooks" / NOTEBOOK_NAME


# ---------------------------------------------------------------------------
# Input audit and loading
# ---------------------------------------------------------------------------


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Return the CSV as a DataFrame if it exists; ``None`` otherwise."""
    if not isinstance(path, Path):
        raise TypeError(
            f"read_csv_if_exists requires a Path; got {type(path).__name__}."
        )
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    """Return the JSON payload if the file exists; ``None`` otherwise."""
    if not isinstance(path, Path):
        raise TypeError(
            f"read_json_if_exists requires a Path; got {type(path).__name__}."
        )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_path_map(output_root: Path) -> dict[str, Path]:
    """Logical name -> absolute path map of every input the renderer reads."""
    me = main_evaluation_readout_dir(output_root)
    pr = prior_relevance_dir(output_root)
    orc = oracle_relevance_dir(output_root)
    return {
        "baseline_comparison_csv": me / "baseline_comparison.csv",
        "degradation_summary_csv": me / "degradation_summary.csv",
        "forbidden_engagement_summary_csv": (
            me / "forbidden_edge_engagement_summary.csv"
        ),
        "reference_forbidden_comparison_csv": (
            me / "reference_forbidden_edge_comparison.csv"
        ),
        "prior_target_overlap_csv": pr / "prior_target_overlap.csv",
        "prior_free_error_decomposition_csv": (
            pr / "prior_free_error_decomposition.csv"
        ),
        "offline_removal_effect_csv": (
            pr / "offline_forbidden_edge_removal_effect.csv"
        ),
        "aggregated_error_heatmap_png": (
            pr / "aggregated_error_heatmap.png"
        ),
        "prior_relevance_manifest_json": (
            pr / "investigation_manifest.json"
        ),
        "oracle_summary_csv": orc / "oracle_diagnostics_summary.csv",
        "oracle_per_seed_csv": orc / "oracle_diagnostics_per_seed.csv",
        "oracle_manifest_json": (
            orc / "oracle_prior_relevance_manifest.json"
        ),
    }


def audit_available_inputs(output_root: Path) -> dict[str, Any]:
    """Inspect candidate inputs and report present-or-not plus schema info."""
    inputs = _input_path_map(output_root)
    audit: dict[str, Any] = {}
    for name, path in inputs.items():
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            audit[name] = entry
            continue
        if path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                entry["parse_error"] = str(exc)
                audit[name] = entry
                continue
            entry["columns"] = list(df.columns)
            entry["n_rows"] = int(len(df))
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                entry["parse_error"] = str(exc)
                audit[name] = entry
                continue
            if isinstance(payload, dict):
                entry["top_level_keys"] = sorted(payload.keys())
            else:
                entry["top_level_type"] = type(payload).__name__
        elif path.suffix == ".png":
            entry["size_bytes"] = int(path.stat().st_size)
        else:
            entry["size_bytes"] = int(path.stat().st_size)
        audit[name] = entry
    return audit


@dataclass(frozen=True, kw_only=True)
class DiagnosticInputs:
    """Loaded diagnostic inputs (None for any that are missing)."""

    baseline_comparison: Optional[pd.DataFrame]
    degradation_summary: Optional[pd.DataFrame]
    forbidden_engagement_summary: Optional[pd.DataFrame]
    reference_forbidden_comparison: Optional[pd.DataFrame]
    prior_target_overlap: Optional[pd.DataFrame]
    prior_free_error_decomposition: Optional[pd.DataFrame]
    offline_removal_effect: Optional[pd.DataFrame]
    aggregated_error_heatmap_path: Optional[Path]
    prior_relevance_manifest: Optional[dict[str, Any]]
    oracle_summary: Optional[pd.DataFrame]
    oracle_per_seed: Optional[pd.DataFrame]
    oracle_manifest: Optional[dict[str, Any]]


def load_diagnostic_inputs(output_root: Path) -> DiagnosticInputs:
    paths = _input_path_map(output_root)
    heatmap_path = paths["aggregated_error_heatmap_png"]
    return DiagnosticInputs(
        baseline_comparison=read_csv_if_exists(
            paths["baseline_comparison_csv"]
        ),
        degradation_summary=read_csv_if_exists(
            paths["degradation_summary_csv"]
        ),
        forbidden_engagement_summary=read_csv_if_exists(
            paths["forbidden_engagement_summary_csv"]
        ),
        reference_forbidden_comparison=read_csv_if_exists(
            paths["reference_forbidden_comparison_csv"]
        ),
        prior_target_overlap=read_csv_if_exists(
            paths["prior_target_overlap_csv"]
        ),
        prior_free_error_decomposition=read_csv_if_exists(
            paths["prior_free_error_decomposition_csv"]
        ),
        offline_removal_effect=read_csv_if_exists(
            paths["offline_removal_effect_csv"]
        ),
        aggregated_error_heatmap_path=(
            heatmap_path if heatmap_path.exists() else None
        ),
        prior_relevance_manifest=read_json_if_exists(
            paths["prior_relevance_manifest_json"]
        ),
        oracle_summary=read_csv_if_exists(paths["oracle_summary_csv"]),
        oracle_per_seed=read_csv_if_exists(
            paths["oracle_per_seed_csv"]
        ),
        oracle_manifest=read_json_if_exists(
            paths["oracle_manifest_json"]
        ),
    )


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_STYLE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _method_colour(method_family: str) -> str:
    return METHOD_FAMILY_COLOURS.get(method_family, "#999999")


def _scenario_colour(scenario_label: str) -> str:
    return SCENARIO_COLOURS.get(scenario_label, "#999999")


# ---------------------------------------------------------------------------
# Figure plotters
# ---------------------------------------------------------------------------


def plot_main_result_clean_metrics(
    baseline_comparison: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Three-panel mean SID / MMD / SHD per method family on the clean grid."""
    _apply_style()
    df = baseline_comparison.copy()
    metric_specs = (("sid", "SID"), ("mmd", "MMD"), ("shd", "SHD"))
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.8), constrained_layout=True,
    )
    method_order = list(METHOD_FAMILY_ORDER)
    x_positions = np.arange(len(method_order))
    for ax, (metric_key, label) in zip(axes, metric_specs):
        for idx, method in enumerate(method_order):
            sub = df[
                (df["method_family"] == method)
                & (df["metric"] == metric_key)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            mean_value = float(row["mean"])
            std_value = (
                float(row["std"]) if pd.notna(row.get("std", np.nan))
                else 0.0
            )
            colour = _method_colour(method)
            ax.bar(
                x_positions[idx], mean_value,
                color=colour, alpha=0.55,
                edgecolor=colour, linewidth=0.9, width=0.7,
                yerr=std_value, ecolor="#333333", capsize=3,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_order, fontsize=8, rotation=15)
        ax.set_ylabel(label)
        ax.set_title(f"{label} (lower is better)")
        ax.set_xlim(-0.5, len(method_order) - 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Clean-grid mean per method family with standard deviation",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_mechanism_engagement(
    reference_forbidden_comparison: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-method mean targeted abs(W) and fraction-above-threshold on clean.

    Shows that the soft prior engages mechanically: it suppresses the
    targeted edges in continuous W and their threshold-crossing fraction
    compared with the prior-free baseline.
    """
    _apply_style()
    df = reference_forbidden_comparison.copy()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 3.8), constrained_layout=True,
    )
    method_order = list(METHOD_FAMILY_ORDER)
    x_positions = np.arange(len(method_order))
    metrics = (
        (
            "mean_abs_w_reference_forbidden_edges",
            "mean |W| on reference forbidden edges",
        ),
        (
            "fraction_reference_forbidden_above_threshold",
            "fraction of reference forbidden edges\nabove threshold",
        ),
    )
    for ax, (col, label) in zip(axes, metrics):
        for idx, method in enumerate(method_order):
            sub = df[df["method_family"] == method]
            if sub.empty or col not in sub.columns:
                continue
            values = sub[col].dropna().values
            if values.size == 0:
                continue
            mean_v = float(np.mean(values))
            colour = _method_colour(method)
            ax.bar(
                x_positions[idx], mean_v,
                color=colour, alpha=0.55,
                edgecolor=colour, linewidth=0.9, width=0.7,
            )
            rng = np.random.default_rng(idx * 11 + 3)
            jitter = rng.uniform(-0.18, 0.18, size=values.size)
            ax.scatter(
                np.full(values.size, x_positions[idx]) + jitter, values,
                s=24, color=colour,
                edgecolor="white", linewidth=0.6,
                alpha=0.95, zorder=3,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_order, fontsize=8, rotation=15)
        ax.set_ylabel(label)
        ax.set_xlim(-0.5, len(method_order) - 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Clean prior-engagement diagnostic (per-seed dots = 7 evaluation seeds)",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_corruption_degradation(
    degradation_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-confidence mean SID / MMD slope per method on the corruption axis."""
    _apply_style()
    df = degradation_summary.copy()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.0), constrained_layout=True,
    )
    for ax, (metric_key, label) in zip(
        axes, (("sid", "mean dSID per unit corruption"),
               ("mmd", "mean dMMD per unit corruption")),
    ):
        sub_all = df[df["metric"] == metric_key]
        for method in METHOD_FAMILY_ORDER:
            sub = sub_all[sub_all["method_family"] == method].copy()
            if sub.empty:
                continue
            colour = _method_colour(method)
            if method == "soft_frobenius":
                sub = sub.dropna(subset=["confidence"]).copy()
                sub["confidence"] = sub["confidence"].astype(float)
                sub = sub.sort_values("confidence")
                ax.plot(
                    sub["confidence"], sub["mean_slope"],
                    marker="o", linewidth=1.5, color=colour,
                    markersize=5, label=method,
                )
            else:
                # Other methods do not depend on confidence; show a
                # single horizontal reference line across the axis.
                if pd.notna(sub.iloc[0]["mean_slope"]):
                    value = float(sub.iloc[0]["mean_slope"])
                    ax.axhline(
                        value, color=colour, linewidth=1.2,
                        linestyle="--", alpha=0.85, label=method,
                    )
        ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle=":")
        ax.set_xlabel("confidence (soft-prior); horizontal lines: other methods")
        ax.set_ylabel(label)
        ax.set_title(f"{label}")
        ax.legend(loc="best", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Corruption-axis degradation slope by method family and confidence",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_error_decomposition(
    prior_free_error_decomposition: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-seed TP / FP / FN counts for prior-free, plus targeted-FP count."""
    _apply_style()
    df = prior_free_error_decomposition.copy().sort_values("seed_value")
    seeds = df["seed_value"].astype(int).tolist()
    x = np.arange(len(seeds))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    bars = (
        ("true_positive_count", "TP", "#1F6FB5"),
        ("false_positive_count", "FP", "#B53737"),
        ("false_negative_count", "FN", "#E08E45"),
        ("targeted_false_positive_count", "targeted FP", "#444444"),
    )
    for offset, (col, label, colour) in zip(
        (-1.5, -0.5, 0.5, 1.5), bars,
    ):
        ax.bar(
            x + offset * width, df[col].astype(float).values,
            width=width, label=label,
            color=colour, alpha=0.85,
            edgecolor="white", linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("evaluation seed")
    ax.set_ylabel("edge count (off-diagonal)")
    ax.set_title(
        "Prior-free per-seed error decomposition; "
        "targeted FP = overlap with original prior set"
    )
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_prior_target_overlap(
    prior_target_overlap: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Fraction of reference forbidden edges predicted per method family."""
    _apply_style()
    df = prior_target_overlap.copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    method_order = list(METHOD_FAMILY_ORDER)
    x_positions = np.arange(len(method_order))
    for idx, method in enumerate(method_order):
        sub = df[df["method_family"] == method]
        if sub.empty:
            continue
        values = sub["fraction_reference_edges_predicted"].astype(
            float
        ).values
        mean_v = float(np.mean(values))
        colour = _method_colour(method)
        ax.bar(
            x_positions[idx], mean_v,
            color=colour, alpha=0.55,
            edgecolor=colour, linewidth=0.9, width=0.7,
        )
        rng = np.random.default_rng(idx * 13 + 7)
        jitter = rng.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.size, x_positions[idx]) + jitter, values,
            s=28, color=colour,
            edgecolor="white", linewidth=0.6,
            alpha=0.95, zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_order, fontsize=9, rotation=15)
    ax.set_ylabel("fraction of reference forbidden edges predicted")
    ax.set_title(
        "Prior-target overlap per method (per-seed dots = 7 evaluation seeds)"
    )
    ax.set_xlim(-0.5, len(method_order) - 0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_offline_removal_effect(
    offline_removal_effect: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-seed dSID and dSHD when reference forbidden edges are zeroed."""
    _apply_style()
    df = offline_removal_effect.copy().sort_values("seed_value")
    seeds = df["seed_value"].astype(int).tolist()
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.0), constrained_layout=True,
    )
    for ax, (col, label) in zip(
        axes, (("sid_delta", "dSID (after - before)"),
               ("shd_delta", "dSHD (after - before)")),
    ):
        deltas = df[col].astype(float).values
        colours = ["#1F6FB5" if v < 0 else ("#B53737" if v > 0 else "#7A7A7A")
                   for v in deltas]
        ax.bar(x, deltas, color=colours, alpha=0.85,
               edgecolor="white", linewidth=0.6)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("evaluation seed")
        ax.set_ylabel(label)
        ax.set_title(label)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Offline reference-forbidden removal effect on prior-free predictions",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def copy_aggregated_error_heatmap(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """Copy the upstream aggregated error heatmap PNG into the figures dir."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return destination_path


def plot_oracle_diagnostic_summary(
    oracle_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Mean dSID and mean dSHD per oracle scenario, as a paired bar chart."""
    _apply_style()
    df = oracle_summary.copy()
    scenarios = [s for s in ORACLE_SCENARIO_ORDER
                 if s in set(df["scenario_label"].tolist())]
    x = np.arange(len(scenarios))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    sid_means: list[float] = []
    shd_means: list[float] = []
    for s in scenarios:
        row = df[df["scenario_label"] == s].iloc[0]
        sid_means.append(float(row["mean_sid_delta"]))
        shd_means.append(float(row["mean_shd_delta"]))
    ax.bar(
        x - width / 2, sid_means, width=width, label="mean dSID",
        color="#1F6FB5", alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )
    ax.bar(
        x + width / 2, shd_means, width=width, label="mean dSHD",
        color="#E08E45", alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_DISPLAY_LABELS.get(s, s) for s in scenarios],
        fontsize=8,
    )
    ax.set_ylabel("metric change (negative = improvement)")
    ax.set_title(
        "Oracle scenario summary: mean dSID and dSHD across 7 evaluation seeds"
    )
    ax.legend(loc="best", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_oracle_per_seed_sid_delta(
    oracle_per_seed: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-seed dSID strip plot per oracle scenario."""
    _apply_style()
    df = oracle_per_seed.copy()
    scenarios = [s for s in ORACLE_SCENARIO_ORDER
                 if s in set(df["scenario_label"].tolist())]
    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    x_positions = np.arange(len(scenarios))
    for idx, s in enumerate(scenarios):
        sub = df[df["scenario_label"] == s]
        values = sub["sid_delta"].astype(float).values
        if values.size == 0:
            continue
        colour = _scenario_colour(s)
        rng = np.random.default_rng(idx * 19 + 11)
        jitter = rng.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.size, x_positions[idx]) + jitter, values,
            s=44, color=colour,
            edgecolor="white", linewidth=0.8,
            alpha=0.95, zorder=3,
        )
        ax.scatter(
            [x_positions[idx]], [float(np.mean(values))],
            marker="D", s=70, color=colour,
            edgecolor="black", linewidth=0.8, zorder=4,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [SCENARIO_DISPLAY_LABELS.get(s, s) for s in scenarios],
        fontsize=8,
    )
    ax.set_ylabel("dSID (after - before)")
    ax.set_title(
        "Per-seed oracle dSID by scenario; diamond = scenario mean"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_required_edge_acyclicity(
    oracle_per_seed: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-seed selected-vs-skipped counts for the greedy FN scenarios."""
    _apply_style()
    df = oracle_per_seed.copy()
    fn_scenarios = [
        s for s in ORACLE_SCENARIO_ORDER
        if s.startswith("fn_") and s in set(df["scenario_label"].tolist())
    ]
    if not fn_scenarios:
        raise ValueError("no fn_* scenarios present in oracle_per_seed")
    sub = df[df["scenario_label"] == fn_scenarios[0]].copy()
    sub = sub.sort_values("seed_value")
    seeds = sub["seed_value"].astype(int).tolist()
    x = np.arange(len(seeds))
    selected = sub["n_selected_edges"].astype(float).values
    skipped = sub["n_skipped_cycle_edges"].astype(float).values
    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    ax.bar(
        x, selected, color="#1F6FB5", alpha=0.85,
        edgecolor="white", linewidth=0.6,
        label="n_selected_edges (acyclic-valid)",
    )
    ax.bar(
        x, skipped, bottom=selected,
        color="#B53737", alpha=0.85,
        edgecolor="white", linewidth=0.6,
        label="n_skipped_cycle_edges (would create a cycle)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("evaluation seed")
    ax.set_ylabel("candidate FN edges")
    ax.set_title(
        f"Acyclicity-constrained FN repair (scenario: {fn_scenarios[0]})"
    )
    ax.legend(loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_fp_vs_fn_reconciliation(
    prior_free_error_decomposition: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Mean FN > mean FP, but the SID leverage comes from FP removal."""
    _apply_style()
    fp_mean = float(
        prior_free_error_decomposition["false_positive_count"].mean()
    )
    fn_mean = float(
        prior_free_error_decomposition["false_negative_count"].mean()
    )
    summary = oracle_summary.set_index("scenario_label")
    fp_row_label = (
        "fp_remove_budget10_exact"
        if "fp_remove_budget10_exact" in summary.index
        else None
    )
    fn_row_label = (
        "fn_add_budget10_greedy_acyclic"
        if "fn_add_budget10_greedy_acyclic" in summary.index
        else None
    )
    if fp_row_label is None or fn_row_label is None:
        raise ValueError(
            "oracle_summary missing required scenario rows for "
            "fp_vs_fn_reconciliation."
        )
    fp_sid_delta = float(summary.loc[fp_row_label, "mean_sid_delta"])
    fn_sid_delta = float(summary.loc[fn_row_label, "mean_sid_delta"])

    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.0), constrained_layout=True,
    )
    ax_counts, ax_sid = axes
    ax_counts.bar(
        [0, 1], [fp_mean, fn_mean],
        color=["#B53737", "#E08E45"], alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )
    ax_counts.set_xticks([0, 1])
    ax_counts.set_xticklabels(["mean FP count", "mean FN count"])
    ax_counts.set_ylabel("edges per seed")
    ax_counts.set_title("Prior-free error counts (mean over 7 seeds)")
    for idx, v in enumerate((fp_mean, fn_mean)):
        ax_counts.text(idx, v + 0.2, f"{v:.2f}",
                       ha="center", va="bottom", fontsize=9)

    ax_sid.bar(
        [0, 1], [fp_sid_delta, fn_sid_delta],
        color=[_scenario_colour("fp_remove_budget10_exact"),
               _scenario_colour("fn_add_budget10_greedy_acyclic")],
        alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    ax_sid.axhline(0.0, color="#333333", linewidth=0.8)
    ax_sid.set_xticks([0, 1])
    ax_sid.set_xticklabels(
        ["FP budget=10\noracle dSID", "FN budget=10\noracle dSID"]
    )
    ax_sid.set_ylabel("mean dSID (negative = improvement)")
    ax_sid.set_title("Oracle SID leverage (mean over 7 seeds)")
    for idx, v in enumerate((fp_sid_delta, fn_sid_delta)):
        ax_sid.text(idx, v - 1.0, f"{v:+.2f}",
                    ha="center", va="top", fontsize=9)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "FN edges outnumber FPs, but FP removal carries the SID leverage",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Summary markdown and manifest
# ---------------------------------------------------------------------------


def write_summary_markdown(
    *,
    output_root: Path,
    audit: dict[str, Any],
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
    output_path: Path,
) -> Path:
    """Labelling-only markdown summary."""
    diagnostics_dir = diagnostics_output_dir(output_root)
    lines: list[str] = []
    lines.append("# Prior-relevance diagnostics: visual readout")
    lines.append("")
    lines.append("## Run identity")
    lines.append("")
    lines.append(
        f"- main_evaluation_run_hash12: `{MAIN_EVALUATION_RUN_HASH_PREFIX}`"
    )
    lines.append(
        f"- prior_relevance_analysis_hash12: "
        f"`{PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX}`"
    )
    lines.append(
        f"- oracle_analysis_hash12: `{ORACLE_ANALYSIS_HASH_PREFIX}`"
    )
    lines.append("")
    lines.append(
        "No model fits, no metric recomputation, no protocol change. "
        "Every figure is built from a single persisted artefact."
    )
    lines.append("")
    lines.append("## Inputs read")
    lines.append("")
    lines.append("| input | exists | path |")
    lines.append("| --- | --- | --- |")
    for name, entry in sorted(audit.items()):
        lines.append(
            f"| `{name}` | {entry.get('exists', False)} | "
            f"`{entry.get('path', '')}` |"
        )
    lines.append("")
    lines.append("## Figures generated")
    lines.append("")
    if generated_figures:
        for name, p in sorted(generated_figures.items()):
            try:
                rel = Path(str(p)).resolve().relative_to(
                    diagnostics_dir.resolve()
                )
                lines.append(f"- `{name}` -> `{rel.as_posix()}`")
            except ValueError:
                lines.append(f"- `{name}` -> `{p}`")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Figures skipped")
    lines.append("")
    if skipped_figures:
        for name, reason in sorted(skipped_figures.items()):
            lines.append(f"- `{name}`: {reason}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Investigative chain (labelling only)")
    lines.append("")
    lines.append(
        "1. Main result: the soft prior engaged mechanically on the "
        "targeted edges but the clean-grid SID / MMD did not show a "
        "clear improvement over the prior-free baseline."
    )
    lines.append(
        "2. Original forbidden-edge targets covered a small subset of "
        "the prior-free false positives."
    )
    lines.append(
        "3. Offline removal of those targets produced small SHD gains "
        "and mixed SID changes."
    )
    lines.append(
        "4. Exact budget-matched FP-targeted removal showed much larger "
        "available SID and SHD leverage."
    )
    lines.append(
        "5. Required-edge post-hoc repair was constrained by "
        "acyclicity: many beneficial candidates would have created "
        "cycles and were skipped."
    )
    lines.append(
        "6. Implication: future work points at improving target "
        "relevance / elicitation rather than at a stronger penalty "
        "weight."
    )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_figure_manifest(
    *,
    output_root: Path,
    audit: dict[str, Any],
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
    notebook_path: Path,
    summary_md_path: Path,
    output_path: Path,
) -> Path:
    inputs_used = sorted(
        k for k, v in audit.items() if v.get("exists", False)
    )
    manifest: dict[str, Any] = {
        "main_evaluation_run_hash12": MAIN_EVALUATION_RUN_HASH_PREFIX,
        "prior_relevance_analysis_hash12": (
            PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX
        ),
        "oracle_analysis_hash12": ORACLE_ANALYSIS_HASH_PREFIX,
        "inputs_used": inputs_used,
        "audit": audit,
        "generated_figures": {
            name: str(p) for name, p in sorted(generated_figures.items())
        },
        "skipped_figures": dict(sorted(skipped_figures.items())),
        "notebook_path": str(notebook_path),
        "summary_markdown_path": str(summary_md_path),
        "no_new_fits": True,
        "no_metric_recomputation": True,
        "no_new_sampling": True,
        "no_protocol_changes": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------


def _make_notebook_payload() -> dict[str, Any]:
    """Build the .ipynb JSON payload (labelling-only, 10 sections)."""
    cells: list[dict[str, Any]] = []

    def md(source_lines: list[str]) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [s + ("\n" if i < len(source_lines) - 1 else "")
                       for i, s in enumerate(source_lines)],
        })

    def code(source_lines: list[str]) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [s + ("\n" if i < len(source_lines) - 1 else "")
                       for i, s in enumerate(source_lines)],
        })

    md([
        "# Prior-relevance diagnostics: visual readout",
        "",
        "Labelling-only display of the frozen main result, the prior "
        "structural relevance analysis, and the oracle prior relevance "
        "analysis. Interpretation belongs in the thesis text; this "
        "notebook only shows the saved artefacts.",
    ])

    md(["## 1. Setup and run identity"])
    code([
        "from pathlib import Path",
        "import sys",
        "import json",
        "import pandas as pd",
        "from IPython.display import Image, Markdown, display",
        "",
        "OUTPUT_ROOT = Path.cwd().parent if (Path.cwd().name == "
        "\"notebooks\") else Path.cwd()",
        "if str(OUTPUT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(OUTPUT_ROOT))",
        "",
        f"MAIN_EVAL_HASH = \"{MAIN_EVALUATION_RUN_HASH_PREFIX}\"",
        f"PRIOR_RELEVANCE_HASH = "
        f"\"{PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX}\"",
        f"ORACLE_HASH = \"{ORACLE_ANALYSIS_HASH_PREFIX}\"",
        "",
        "DIAGNOSTICS_DIR = (",
        "    OUTPUT_ROOT / \"results\" / \"main_study\" / "
        "\"exploratory\"",
        f"    / \"{DIAGNOSTICS_DIR_NAME}\"",
        ")",
        f"FIG_DIR = DIAGNOSTICS_DIR / \"{FIG_DIR_NAME}\"",
        f"MANIFEST_PATH = DIAGNOSTICS_DIR / \"{MANIFEST_JSON_NAME}\"",
        "print(\"main_evaluation_run_hash12:\", MAIN_EVAL_HASH)",
        "print(\"prior_relevance_analysis_hash12:\", "
        "PRIOR_RELEVANCE_HASH)",
        "print(\"oracle_analysis_hash12:\", ORACLE_HASH)",
        "print(\"figures dir:\", FIG_DIR)",
    ])

    md(["## 2. Data availability audit"])
    code([
        "if MANIFEST_PATH.exists():",
        "    manifest = json.loads("
        "MANIFEST_PATH.read_text(encoding=\"utf-8\"))",
        "    display(Markdown(\"Manifest top-level keys:\"))",
        "    display(Markdown(\", \".join(sorted(manifest.keys()))))",
        "    display(Markdown(\"Inputs read:\"))",
        "    for name in manifest.get(\"inputs_used\", []):",
        "        print(\" -\", name)",
        "else:",
        "    display(Markdown(\"manifest not found; run the renderer "
        "first.\"))",
    ])

    md([
        "## 3. Main result: mechanism without clear clean-prior "
        "metric improvement",
    ])
    code([
        f"path = FIG_DIR / \"{FIG01_NAME}\"",
        "display(Markdown(\"Clean-grid mean SID / MMD / SHD per "
        "method family.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG02_NAME}\"",
        "display(Markdown(\"Clean prior-engagement diagnostic: "
        "targeted |W| and fraction-above-threshold per method.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG03_NAME}\"",
        "display(Markdown(\"Corruption-axis degradation slope per "
        "method family and confidence.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    md([
        "## 4. Prior structural relevance motivation: were the prior "
        "targets relevant?",
    ])
    code([
        f"path = FIG_DIR / \"{FIG04_NAME}\"",
        "display(Markdown(\"Prior-free per-seed error decomposition "
        "with targeted-FP count.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    md([
        "## 5. Prior structural relevance: target overlap and "
        "offline removal",
    ])
    code([
        f"path = FIG_DIR / \"{FIG05_NAME}\"",
        "display(Markdown(\"Fraction of reference forbidden edges "
        "predicted, per method family.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG06_NAME}\"",
        "display(Markdown(\"Per-seed dSID and dSHD when reference "
        "forbidden edges are zeroed offline.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG07_NAME}\"",
        "display(Markdown(\"Aggregated structural-error heatmap "
        "(copied from upstream analysis).\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    md([
        "## 6. Oracle motivation: what if targets were better aligned?",
    ])
    md([
        "Oracle scenarios use ground-truth information to select "
        "edges. They are diagnostic only; they are not deployable as "
        "priors.",
    ])

    md([
        "## 7. Oracle evidence: exact FP diagnostic and greedy FN "
        "diagnostic",
    ])
    code([
        f"path = FIG_DIR / \"{FIG08_NAME}\"",
        "display(Markdown(\"Mean dSID and dSHD per oracle scenario.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG09_NAME}\"",
        "display(Markdown(\"Per-seed dSID strip plot by oracle "
        "scenario; diamond = mean.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    code([
        f"path = FIG_DIR / \"{FIG10_NAME}\"",
        "display(Markdown(\"Per-seed selected vs skipped FN edge "
        "counts under the acyclicity constraint.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    md([
        "## 8. Reconciling false-negative counts with false-positive "
        "leverage",
    ])
    code([
        f"path = FIG_DIR / \"{FIG11_NAME}\"",
        "display(Markdown(\"Prior-free FN edges outnumber FPs, but FP "
        "removal carries the SID leverage.\"))",
        "if path.exists(): display(Image(filename=str(path)))",
        "else: display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    md(["## 9. Output manifest"])
    code([
        "if MANIFEST_PATH.exists():",
        "    print(\"Generated figures:\")",
        "    for name in sorted(manifest.get(\"generated_figures\", {})):",
        "        print(\" -\", name)",
        "    skipped = manifest.get(\"skipped_figures\", {})",
        "    if skipped:",
        "        print(\"Skipped figures:\")",
        "        for name, reason in sorted(skipped.items()):",
        "            print(\" -\", name, \"(\" + str(reason) + \")\")",
        "    print(\"All artefacts loaded. Thesis interpretation is "
        "separate.\")",
    ])

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def create_prior_relevance_diagnostics_notebook(
    notebook_path: Path,
) -> Path:
    payload = _make_notebook_payload()
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(payload, indent=1),
        encoding="utf-8",
    )
    return notebook_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_REQUIRED_BASELINE_COLS = {"method_family", "metric", "mean"}
_REQUIRED_REF_COMPARISON_COLS = {
    "method_family",
    "mean_abs_w_reference_forbidden_edges",
    "fraction_reference_forbidden_above_threshold",
}
_REQUIRED_DEGRADATION_COLS = {
    "method_family", "metric", "mean_slope",
}
_REQUIRED_ERROR_DECOMP_COLS = {
    "seed_value",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "targeted_false_positive_count",
}
_REQUIRED_OVERLAP_COLS = {
    "method_family", "fraction_reference_edges_predicted",
}
_REQUIRED_OFFLINE_REMOVAL_COLS = {
    "seed_value", "sid_delta", "shd_delta",
}
_REQUIRED_ORACLE_SUMMARY_COLS = {
    "scenario_label", "mean_sid_delta", "mean_shd_delta",
}
_REQUIRED_ORACLE_PER_SEED_COLS = {
    "seed_value", "scenario_label", "sid_delta",
    "n_selected_edges", "n_skipped_cycle_edges",
}


def _has_columns(df: Optional[pd.DataFrame], required: set[str]) -> bool:
    if df is None:
        return False
    return required.issubset(set(df.columns))


def render_prior_relevance_diagnostics(
    output_root: Path,
) -> dict[str, Any]:
    """Render the full visual readout. Returns a small result dict."""
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    audit = audit_available_inputs(output_root)
    inputs = load_diagnostic_inputs(output_root)
    diagnostics_dir = ensure_output_dir(diagnostics_output_dir(output_root))
    figs_dir = ensure_output_dir(figures_output_dir(output_root))

    generated_figures: dict[str, Path] = {}
    skipped_figures: dict[str, str] = {}

    if _has_columns(inputs.baseline_comparison, _REQUIRED_BASELINE_COLS):
        generated_figures[FIG01_NAME] = plot_main_result_clean_metrics(
            inputs.baseline_comparison, figs_dir / FIG01_NAME,
        )
    else:
        skipped_figures[FIG01_NAME] = (
            "baseline_comparison.csv missing or lacks required columns"
        )

    if _has_columns(
        inputs.reference_forbidden_comparison,
        _REQUIRED_REF_COMPARISON_COLS,
    ):
        generated_figures[FIG02_NAME] = plot_mechanism_engagement(
            inputs.reference_forbidden_comparison,
            figs_dir / FIG02_NAME,
        )
    else:
        skipped_figures[FIG02_NAME] = (
            "reference_forbidden_edge_comparison.csv missing or lacks "
            "required columns"
        )

    if _has_columns(
        inputs.degradation_summary, _REQUIRED_DEGRADATION_COLS,
    ):
        generated_figures[FIG03_NAME] = plot_corruption_degradation(
            inputs.degradation_summary, figs_dir / FIG03_NAME,
        )
    else:
        skipped_figures[FIG03_NAME] = (
            "degradation_summary.csv missing or lacks required columns"
        )

    if _has_columns(
        inputs.prior_free_error_decomposition,
        _REQUIRED_ERROR_DECOMP_COLS,
    ):
        generated_figures[FIG04_NAME] = plot_error_decomposition(
            inputs.prior_free_error_decomposition,
            figs_dir / FIG04_NAME,
        )
    else:
        skipped_figures[FIG04_NAME] = (
            "prior_free_error_decomposition.csv missing or lacks "
            "required columns"
        )

    if _has_columns(
        inputs.prior_target_overlap, _REQUIRED_OVERLAP_COLS,
    ):
        generated_figures[FIG05_NAME] = plot_prior_target_overlap(
            inputs.prior_target_overlap, figs_dir / FIG05_NAME,
        )
    else:
        skipped_figures[FIG05_NAME] = (
            "prior_target_overlap.csv missing or lacks required columns"
        )

    if _has_columns(
        inputs.offline_removal_effect, _REQUIRED_OFFLINE_REMOVAL_COLS,
    ):
        generated_figures[FIG06_NAME] = plot_offline_removal_effect(
            inputs.offline_removal_effect, figs_dir / FIG06_NAME,
        )
    else:
        skipped_figures[FIG06_NAME] = (
            "offline_forbidden_edge_removal_effect.csv missing or "
            "lacks required columns"
        )

    if inputs.aggregated_error_heatmap_path is not None:
        generated_figures[FIG07_NAME] = copy_aggregated_error_heatmap(
            inputs.aggregated_error_heatmap_path,
            figs_dir / FIG07_NAME,
        )
    else:
        skipped_figures[FIG07_NAME] = (
            "aggregated_error_heatmap.png missing"
        )

    if _has_columns(
        inputs.oracle_summary, _REQUIRED_ORACLE_SUMMARY_COLS,
    ):
        generated_figures[FIG08_NAME] = plot_oracle_diagnostic_summary(
            inputs.oracle_summary, figs_dir / FIG08_NAME,
        )
    else:
        skipped_figures[FIG08_NAME] = (
            "oracle_diagnostics_summary.csv missing or lacks "
            "required columns"
        )

    if _has_columns(
        inputs.oracle_per_seed, _REQUIRED_ORACLE_PER_SEED_COLS,
    ):
        generated_figures[FIG09_NAME] = plot_oracle_per_seed_sid_delta(
            inputs.oracle_per_seed, figs_dir / FIG09_NAME,
        )
    else:
        skipped_figures[FIG09_NAME] = (
            "oracle_diagnostics_per_seed.csv missing or lacks "
            "required columns for per-seed SID delta plot"
        )

    if _has_columns(
        inputs.oracle_per_seed, _REQUIRED_ORACLE_PER_SEED_COLS,
    ) and any(
        s.startswith("fn_") for s in
        set(inputs.oracle_per_seed["scenario_label"].tolist())
    ):
        generated_figures[FIG10_NAME] = plot_required_edge_acyclicity(
            inputs.oracle_per_seed, figs_dir / FIG10_NAME,
        )
    else:
        skipped_figures[FIG10_NAME] = (
            "oracle_diagnostics_per_seed.csv missing fn_* scenario "
            "rows or required columns"
        )

    if (
        _has_columns(
            inputs.prior_free_error_decomposition,
            _REQUIRED_ERROR_DECOMP_COLS,
        )
        and _has_columns(
            inputs.oracle_summary, _REQUIRED_ORACLE_SUMMARY_COLS,
        )
        and "fp_remove_budget10_exact" in set(
            inputs.oracle_summary["scenario_label"].tolist()
        )
        and "fn_add_budget10_greedy_acyclic" in set(
            inputs.oracle_summary["scenario_label"].tolist()
        )
    ):
        generated_figures[FIG11_NAME] = plot_fp_vs_fn_reconciliation(
            inputs.prior_free_error_decomposition,
            inputs.oracle_summary,
            figs_dir / FIG11_NAME,
        )
    else:
        skipped_figures[FIG11_NAME] = (
            "inputs for fp_vs_fn reconciliation incomplete"
        )

    summary_md_path = diagnostics_dir / SUMMARY_MD_NAME
    write_summary_markdown(
        output_root=output_root, audit=audit,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        output_path=summary_md_path,
    )

    nb_path = notebook_output_path(output_root)
    create_prior_relevance_diagnostics_notebook(nb_path)

    manifest_path = diagnostics_dir / MANIFEST_JSON_NAME
    write_figure_manifest(
        output_root=output_root,
        audit=audit,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        notebook_path=nb_path,
        summary_md_path=summary_md_path,
        output_path=manifest_path,
    )
    return {
        "manifest_path": str(manifest_path),
        "summary_markdown_path": str(summary_md_path),
        "notebook_path": str(nb_path),
        "generated_figures": {
            n: str(p) for n, p in generated_figures.items()
        },
        "skipped_figures": dict(skipped_figures),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_prior_relevance_diagnostics",
        description=(
            "Visual readout for the exploratory prior-relevance "
            "diagnostics. Read-only over persisted artefacts; no "
            "fitting, no metric recomputation, no protocol change."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root directory under which results/... is located.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = render_prior_relevance_diagnostics(args.output_root)
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            "render_prior_relevance_diagnostics: error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return 1
    sys.stdout.write(
        f"manifest: {result['manifest_path']}\n"
    )
    for name, p in sorted(result["generated_figures"].items()):
        sys.stdout.write(f"- {name}: {p}\n")
    for name, reason in sorted(result["skipped_figures"].items()):
        sys.stdout.write(f"- skipped {name}: {reason}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DIAGNOSTICS_DIR_NAME",
    "DiagnosticInputs",
    "FIG01_NAME",
    "FIG02_NAME",
    "FIG03_NAME",
    "FIG04_NAME",
    "FIG05_NAME",
    "FIG06_NAME",
    "FIG07_NAME",
    "FIG08_NAME",
    "FIG09_NAME",
    "FIG10_NAME",
    "FIG11_NAME",
    "FIG_DIR_NAME",
    "MAIN_EVALUATION_RUN_HASH_PREFIX",
    "MANIFEST_JSON_NAME",
    "METHOD_FAMILY_COLOURS",
    "METHOD_FAMILY_ORDER",
    "NOTEBOOK_NAME",
    "ORACLE_ANALYSIS_HASH_PREFIX",
    "ORACLE_SCENARIO_ORDER",
    "PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX",
    "SCENARIO_COLOURS",
    "SUMMARY_MD_NAME",
    "audit_available_inputs",
    "copy_aggregated_error_heatmap",
    "create_prior_relevance_diagnostics_notebook",
    "diagnostics_output_dir",
    "ensure_output_dir",
    "figures_output_dir",
    "load_diagnostic_inputs",
    "main",
    "main_evaluation_readout_dir",
    "notebook_output_path",
    "oracle_relevance_dir",
    "plot_corruption_degradation",
    "plot_error_decomposition",
    "plot_fp_vs_fn_reconciliation",
    "plot_main_result_clean_metrics",
    "plot_mechanism_engagement",
    "plot_offline_removal_effect",
    "plot_oracle_diagnostic_summary",
    "plot_oracle_per_seed_sid_delta",
    "plot_prior_target_overlap",
    "plot_required_edge_acyclicity",
    "prior_relevance_dir",
    "read_csv_if_exists",
    "read_json_if_exists",
    "render_prior_relevance_diagnostics",
    "write_figure_manifest",
    "write_summary_markdown",
]
