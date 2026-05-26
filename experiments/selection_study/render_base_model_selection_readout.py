"""Visual readout renderer for the completed base-model selection study.

Reads persisted calibration / held-out / adjudication artefacts and
produces a clean visual readout (eight target figures, a summary
markdown, and a figure manifest JSON) plus a labelling-only notebook
showing how DAGMA was carried forward as the base model. No new
fitting, no metric recomputation, no protocol change.

Output directory:
    ``<output_root>/results/model_selection/held_out/88da382e8672/
    readout/base_model_selection_figures/``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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


CALIBRATION_RUN_HASH_PREFIX: str = "4a67117a10b1"
HELDOUT_RUN_HASH_PREFIX: str = "88da382e8672"
BASE_MODEL_DECISION_LABEL: str = "DAGMA"

# Model palette: clean red/blue scheme.
MODEL_COLOURS: dict[str, str] = {
    "dagma": "#0072B2",   # dark blue
    "dcdi": "#B53737",    # dark red
}

CONDITION_LABELS: tuple[str, ...] = ("centred_only", "standardised")
MODEL_LABELS: tuple[str, ...] = ("dagma", "dcdi")

# Output filenames. Calibration handoff, selection rule, and final
# summary are now notebook DataFrame tables; their previous PNGs are
# intentionally removed from the renderer.
FIG_DIR_NAME: str = "base_model_selection_figures"
FIG02_NAME: str = "fig02_heldout_metric_means.png"
FIG02B_NAME: str = "fig02b_paired_model_differences.png"
FIG03_NAME: str = "fig03_heldout_sid_per_seed.png"
FIG05_NAME: str = "fig05_runtime_log_scale.png"
FIG06_NAME: str = "fig06_dcdi_fit_rng_sensitivity.png"
FIG07_NAME: str = "fig07_dagma_ceiling_and_headroom.png"
FIG_STATUS_NAME: str = "fig_status_reliability.png"

# Side artefacts read directly by the notebook as DataFrames.
SELECTED_CONFIG_TABLE_CSV: str = "selected_configurations_table.csv"
SELECTION_SUMMARY_TABLE_CSV: str = "selection_summary_table.csv"

# Figures removed in the current patch. Recorded in the manifest so
# downstream consumers can detect the change.
REMOVED_FIGURE_NAMES: tuple[str, ...] = (
    "fig01_calibration_selected_configurations.png",
    "fig04_selection_rule_visual.png",
    "fig08_selection_summary_matrix.png",
)

SUMMARY_MD_NAME: str = "base_model_selection_readout_summary.md"
MANIFEST_JSON_NAME: str = "base_model_selection_figure_manifest.json"
NOTEBOOK_NAME: str = "base_model_selection.ipynb"


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


def selection_readout_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results"
        / "model_selection"
        / "held_out"
        / HELDOUT_RUN_HASH_PREFIX
        / "readout"
    )


def calibration_dir(output_root: Path) -> Path:
    return (
        output_root
        / "results"
        / "model_selection"
        / "calibration"
        / CALIBRATION_RUN_HASH_PREFIX
    )


def figures_output_dir(output_root: Path) -> Path:
    return selection_readout_dir(output_root) / FIG_DIR_NAME


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
    """Return the JSON-decoded payload if the file exists; ``None`` otherwise."""
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
    """Map of logical input names to absolute paths under ``output_root``."""
    cal_dir = calibration_dir(output_root)
    rd = selection_readout_dir(output_root)
    return {
        "selected_configurations_json": (
            cal_dir / "selected_configurations.json"
        ),
        "heldout_evaluation_json": (
            output_root
            / "results" / "model_selection" / "held_out"
            / HELDOUT_RUN_HASH_PREFIX / "heldout_evaluation.json"
        ),
        "main_summary_csv": rd / "main_summary.csv",
        "per_seed_main_csv": rd / "per_seed_main.csv",
        "sensitivity_summary_csv": rd / "sensitivity_summary.csv",
        "status_summary_csv": rd / "status_summary.csv",
        "adjudication_md": (
            output_root / "docs" / "08h_selection_study_adjudication.md"
        ),
        "selection_doc_md": (
            output_root / "docs" / "02_base_model_selection.md"
        ),
    }


def audit_available_inputs(output_root: Path) -> dict[str, Any]:
    """Inspect candidate inputs and report present-or-not plus schema info.

    The audit reports, for every known input path:

    - whether it exists;
    - its absolute path;
    - for CSVs: column names plus row count;
    - for JSONs: top-level keys.

    Returns a dict mapping logical input name to a small status
    record. This audit is read-only; nothing on disk is altered.
    """
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
        else:
            # Markdown: just record size.
            entry["size_bytes"] = int(path.stat().st_size)
        audit[name] = entry
    return audit


@dataclass(frozen=True, kw_only=True)
class SelectionInputs:
    """Loaded readout inputs (None for any that are not on disk)."""

    selected_configurations: Optional[dict[str, Any]]
    heldout_evaluation: Optional[dict[str, Any]]
    main_summary: Optional[pd.DataFrame]
    per_seed_main: Optional[pd.DataFrame]
    sensitivity_summary: Optional[pd.DataFrame]
    status_summary: Optional[pd.DataFrame]


def load_selection_inputs(output_root: Path) -> SelectionInputs:
    paths = _input_path_map(output_root)
    return SelectionInputs(
        selected_configurations=read_json_if_exists(
            paths["selected_configurations_json"]
        ),
        heldout_evaluation=read_json_if_exists(
            paths["heldout_evaluation_json"]
        ),
        main_summary=read_csv_if_exists(paths["main_summary_csv"]),
        per_seed_main=read_csv_if_exists(paths["per_seed_main_csv"]),
        sensitivity_summary=read_csv_if_exists(
            paths["sensitivity_summary_csv"]
        ),
        status_summary=read_csv_if_exists(paths["status_summary_csv"]),
    )


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_STYLE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _short_cell_label(condition: str, model: str) -> str:
    cond_short = {
        "centred_only": "centred",
        "standardised": "stdised",
    }.get(condition, condition)
    return f"{cond_short}\n{model}"


def _cell_order() -> list[tuple[str, str]]:
    return [
        (c, m) for c in CONDITION_LABELS for m in MODEL_LABELS
    ]


# ---------------------------------------------------------------------------
# Figure plotters
# ---------------------------------------------------------------------------


def build_selected_configurations_table(
    selected_configurations: dict[str, Any],
) -> pd.DataFrame:
    """DataFrame of the calibration handoff: rank-1 per (condition, model).

    Columns: ``condition``, ``model``, ``configuration_hash_prefix``,
    ``hyperparameters``, ``calibration_mean_sid``.
    """
    ranking = selected_configurations.get("candidate_ranking", {})
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_LABELS:
        cond_block = ranking.get(condition, {})
        for model in MODEL_LABELS:
            cands = cond_block.get(model, [])
            if not cands:
                rows.append({
                    "condition": condition,
                    "model": model,
                    "configuration_hash_prefix": "",
                    "hyperparameters": "",
                    "calibration_mean_sid": None,
                })
                continue
            rank1 = cands[0]
            hp = rank1.get("hyperparameters", {}) or {}
            hp_text = ", ".join(
                f"{k}={v}" for k, v in sorted(hp.items())
            )
            agg = rank1.get("aggregate_metrics", {}) or {}
            mean_sid = agg.get("mean_sid")
            rows.append({
                "condition": condition,
                "model": model,
                "configuration_hash_prefix": str(
                    rank1.get("configuration_hash_prefix", "")
                ),
                "hyperparameters": hp_text,
                "calibration_mean_sid": (
                    None if mean_sid is None else float(mean_sid)
                ),
            })
    return pd.DataFrame(rows)


def plot_heldout_metric_means(
    main_summary: pd.DataFrame,
    per_seed_main: Optional[pd.DataFrame],
    output_path: Path,
) -> Path:
    """Three-panel comparison of mean SID / MMD / SHD on held-out."""
    _apply_style()
    metrics = (
        ("mean_sid", "SID"),
        ("mean_mmd_primary", "MMD"),
        ("mean_shd", "SHD"),
    )
    seed_cols = (
        ("sid", "SID"),
        ("mmd_primary", "MMD"),
        ("shd", "SHD"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.8), constrained_layout=True,
    )
    cell_order = _cell_order()
    x_positions = np.arange(len(cell_order))
    short_labels = [
        _short_cell_label(c, m) for (c, m) in cell_order
    ]
    for ax, (col, label), (seed_col, _) in zip(
        axes, metrics, seed_cols
    ):
        for idx, (condition, model) in enumerate(cell_order):
            sub = main_summary[
                (main_summary["condition"] == condition)
                & (main_summary["model"] == model)
            ]
            if sub.empty:
                continue
            value = float(sub.iloc[0][col])
            colour = MODEL_COLOURS.get(model, "#999999")
            ax.bar(
                x_positions[idx], value,
                color=colour, alpha=0.42,
                edgecolor=colour, linewidth=0.9, width=0.7,
                zorder=1,
            )
            if per_seed_main is not None:
                sub_seed = per_seed_main[
                    (per_seed_main["condition"] == condition)
                    & (per_seed_main["model"] == model)
                ]
                values = sub_seed[seed_col].dropna().values
                n = len(values)
                if n > 0:
                    rng = np.random.default_rng(idx * 17 + 3)
                    jitter = rng.uniform(-0.18, 0.18, size=n)
                    ax.scatter(
                        np.full(n, x_positions[idx]) + jitter, values,
                        s=30, color=colour,
                        edgecolor="white", linewidth=0.8,
                        alpha=0.95, zorder=3,
                    )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_title(f"{label} (lower is better)")
        ax.set_ylabel(label)
        ax.set_xlim(-0.5, len(cell_order) - 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Held-out mean SID / MMD / SHD (bar = mean, dots = per-seed)",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_heldout_sid_per_seed(
    per_seed_main: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Per-seed SID dots for every (condition, model) cell."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    cell_order = _cell_order()
    x_positions = np.arange(len(cell_order))
    short_labels = [
        _short_cell_label(c, m) for (c, m) in cell_order
    ]
    for idx, (condition, model) in enumerate(cell_order):
        sub = per_seed_main[
            (per_seed_main["condition"] == condition)
            & (per_seed_main["model"] == model)
        ]
        sid_values = sub["sid"].dropna().values
        colour = MODEL_COLOURS.get(model, "#999999")
        n = len(sid_values)
        if n == 0:
            continue
        rng = np.random.default_rng(idx * 17 + 5)
        jitter = rng.uniform(-0.16, 0.16, size=n)
        ax.scatter(
            np.full(n, x_positions[idx]) + jitter, sid_values,
            s=42, color=colour,
            edgecolor="white", linewidth=0.9,
            alpha=0.95, zorder=3,
        )
        # Mean marker.
        ax.scatter(
            [x_positions[idx]], [float(np.mean(sid_values))],
            marker="D", s=64, color=colour,
            edgecolor="black", linewidth=0.8, zorder=4,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("SID (lower is better)")
    ax.set_xlim(-0.5, len(cell_order) - 0.5)
    ax.set_title(
        "Per-seed held-out SID; diamond = mean (n = 5 held-out seeds)"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_runtime_log_scale(
    main_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Mean runtime per (condition, model) on a log y-axis."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    cell_order = _cell_order()
    x_positions = np.arange(len(cell_order))
    short_labels = [
        _short_cell_label(c, m) for (c, m) in cell_order
    ]
    for idx, (condition, model) in enumerate(cell_order):
        sub = main_summary[
            (main_summary["condition"] == condition)
            & (main_summary["model"] == model)
        ]
        if sub.empty:
            continue
        runtime = float(sub.iloc[0]["mean_runtime_seconds"])
        colour = MODEL_COLOURS.get(model, "#999999")
        ax.bar(
            x_positions[idx], runtime,
            color=colour, alpha=0.55,
            edgecolor=colour, linewidth=0.9, width=0.7,
        )
        # Annotate the bar with the runtime value.
        ax.text(
            x_positions[idx], runtime * 1.06,
            f"{runtime:.1f} s",
            ha="center", va="bottom", fontsize=8.5,
            color="#333333",
        )
    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("mean runtime per fit (seconds, log scale)")
    ax.set_xlim(-0.5, len(cell_order) - 0.5)
    ax.set_title(
        "Held-out mean runtime (log scale)"
    )
    ax.text(
        0.5, -0.30,
        "Runtime is feasibility evidence only; metric rule remains "
        "SID/MMD-led.",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=9, color="#555555", style="italic",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def plot_paired_model_differences(
    per_seed_main: pd.DataFrame, output_path: Path,
) -> Path:
    """Paired DAGMA-vs-DCDI per-seed differences on matched seeds.

    For each metric and condition, computes ``delta = DCDI - DAGMA``
    per matched ``seed_value``. Lower-is-better metrics make positive
    deltas favour DAGMA; the figure is labelled accordingly.
    """
    _apply_style()
    metric_specs = (
        ("sid", "SID"),
        ("mmd_primary", "MMD"),
        ("shd", "SHD"),
    )
    fig, axes = plt.subplots(
        1, len(metric_specs), figsize=(12.0, 4.0),
        constrained_layout=True,
    )
    cond_positions = np.arange(len(CONDITION_LABELS))
    for ax, (col, label) in zip(axes, metric_specs):
        if col not in per_seed_main.columns:
            ax.text(
                0.5, 0.5, f"{label} not measured",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#666666",
            )
            ax.axis("off")
            continue
        ax.axhline(
            0.0, color="#999999", linewidth=0.8,
            linestyle="--", alpha=0.7, zorder=1,
        )
        for idx, condition in enumerate(CONDITION_LABELS):
            dagma = per_seed_main[
                (per_seed_main["condition"] == condition)
                & (per_seed_main["model"] == "dagma")
            ][["seed_value", col]].copy()
            dcdi = per_seed_main[
                (per_seed_main["condition"] == condition)
                & (per_seed_main["model"] == "dcdi")
            ][["seed_value", col]].copy()
            merged = dagma.merge(
                dcdi, on="seed_value", suffixes=("_dagma", "_dcdi"),
            )
            if merged.empty:
                continue
            deltas = (
                merged[f"{col}_dcdi"].astype(float).values
                - merged[f"{col}_dagma"].astype(float).values
            )
            n = len(deltas)
            rng = np.random.default_rng(idx * 17 + 11)
            jitter = rng.uniform(-0.16, 0.16, size=n)
            # Colour the dots by which side they favour.
            colours = [
                MODEL_COLOURS["dagma"] if d > 0
                else MODEL_COLOURS["dcdi"] if d < 0
                else "#999999"
                for d in deltas
            ]
            ax.scatter(
                np.full(n, cond_positions[idx]) + jitter, deltas,
                s=44, c=colours,
                edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3,
            )
            mean_d = float(np.mean(deltas))
            ax.scatter(
                [cond_positions[idx]], [mean_d],
                marker="D", s=70,
                color="black", edgecolor="white", linewidth=0.8,
                zorder=4,
            )
        ax.set_xticks(cond_positions)
        ax.set_xticklabels(list(CONDITION_LABELS), fontsize=9)
        ax.set_ylabel(f"DCDI {label} - DAGMA {label}")
        ax.set_title(f"{label}: paired difference")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Paired per-seed differences on matched held-out seeds "
        "(positive values favour DAGMA)",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_status_reliability(
    status_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Horizontal bar chart of per-status counts from ``status_summary.csv``.

    One bar per ``(kind, status_field, status_value)`` triple,
    grouped by ``kind`` ("main" vs "sensitivity") via colour.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    df = status_summary.copy()
    df = df.sort_values(["kind", "status_field", "status_value"]).reset_index(
        drop=True
    )
    labels = [
        f"{r['kind']} | {r['status_field']} = {r['status_value']}"
        for _, r in df.iterrows()
    ]
    counts = df["count"].astype(int).values
    kinds = df["kind"].astype(str).values
    colours = [
        "#0072B2" if k == "main" else "#B53737" for k in kinds
    ]
    y_positions = np.arange(len(labels))
    ax.barh(
        y_positions, counts, color=colours, alpha=0.7,
        edgecolor=colours, linewidth=0.8,
    )
    for y, c in zip(y_positions, counts):
        ax.text(
            float(c) + 0.2, y, str(int(c)),
            va="center", ha="left", fontsize=8.5, color="#333333",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("count")
    ax.invert_yaxis()
    ax.set_title(
        "Held-out status counts (status_summary.csv)"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Legend chips.
    ax.scatter(
        [], [], color="#0072B2", label="main", s=30,
    )
    ax.scatter(
        [], [], color="#B53737", label="sensitivity", s=30,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    return _save(fig, output_path)


def plot_dcdi_fit_rng_sensitivity(
    sensitivity_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """DCDI fit-RNG sensitivity sweep.

    Shows the five alternative-fit-seed values alongside the main
    fit-seed value for SID, MMD, and SHD, on the DCDI/centred_only
    cell that the addendum probes.
    """
    _apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.8), constrained_layout=True,
    )
    metrics = (
        ("sid", "main_evaluation_sid_at_seed_301", "SID"),
        ("mmd_primary", "main_evaluation_mmd_primary_at_seed_301", "MMD"),
        ("shd", None, "SHD"),
    )
    sub = sensitivity_summary[
        sensitivity_summary["model"] == "dcdi"
    ].sort_values("fit_rng")
    if sub.empty:
        # Nothing measured for DCDI; produce an empty annotated figure.
        for ax in axes:
            ax.text(
                0.5, 0.5, "no DCDI sensitivity rows",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#666666",
            )
            ax.axis("off")
        fig.suptitle(
            "DCDI fit-RNG sensitivity addendum (no rows available)",
            fontsize=_FONT_TITLE,
        )
        return _save(fig, output_path)
    fit_rngs = sub["fit_rng"].astype(int).values
    for ax, (col, main_col, label) in zip(axes, metrics):
        if col not in sub.columns:
            ax.text(
                0.5, 0.5, f"{label} not measured",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#666666",
            )
            ax.axis("off")
            continue
        values = sub[col].astype(float).values
        ax.scatter(
            fit_rngs, values,
            s=46, color=MODEL_COLOURS["dcdi"],
            edgecolor="white", linewidth=0.9, zorder=3,
            label="alternative fit_rng",
        )
        if main_col is not None and main_col in sub.columns:
            main_value = float(sub.iloc[0][main_col])
            ax.axhline(
                main_value, color="#333333", linewidth=0.9,
                linestyle="--", alpha=0.8, zorder=2,
                label="main fit_rng=42",
            )
            ax.text(
                fit_rngs.min() - 0.3, main_value, f"{main_value:.3g}",
                fontsize=8, color="#333333", va="bottom",
            )
        ax.set_xlabel("alternative fit_rng")
        ax.set_ylabel(f"{label} (lower is better)")
        ax.set_title(f"{label} across DCDI fit_rngs (seed 301)")
        if main_col is not None and main_col in sub.columns:
            ax.legend(loc="best", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "DCDI fit-RNG sensitivity addendum (centred_only, scm_seed = 301)",
        fontsize=_FONT_TITLE,
    )
    return _save(fig, output_path)


def plot_dagma_ceiling_and_headroom(
    per_seed_main: pd.DataFrame,
    output_path: Path,
) -> Path:
    """DAGMA centred_only vs DAGMA standardised per-seed SID.

    Ceiling/headroom labels are only used when the per-seed SID
    values visibly support them: ceiling = at least one zero-SID
    seed in centred_only/dagma; headroom = positive SID across seeds
    in standardised/dagma.
    """
    _apply_style()
    centred = per_seed_main[
        (per_seed_main["condition"] == "centred_only")
        & (per_seed_main["model"] == "dagma")
    ]["sid"].dropna().astype(float).values
    stdised = per_seed_main[
        (per_seed_main["condition"] == "standardised")
        & (per_seed_main["model"] == "dagma")
    ]["sid"].dropna().astype(float).values

    has_zero_in_centred = bool(np.any(centred == 0.0))
    has_positive_in_stdised = bool(
        len(stdised) > 0 and np.all(stdised > 0.0)
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    x_positions = np.array([0.0, 1.0])
    cells = (
        ("centred_only / DAGMA", centred,
         MODEL_COLOURS["dagma"], "ceiling evidence"
         if has_zero_in_centred else "per-seed SID"),
        ("standardised / DAGMA", stdised,
         "#5BAEDE", "headroom evidence"
         if has_positive_in_stdised else "per-seed SID"),
    )
    for idx, (label, values, colour, sublabel) in enumerate(cells):
        if len(values) == 0:
            continue
        rng = np.random.default_rng(idx * 17 + 9)
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(
            np.full(len(values), x_positions[idx]) + jitter, values,
            s=48, color=colour,
            edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3,
        )
        ax.scatter(
            [x_positions[idx]], [float(np.mean(values))],
            marker="D", s=70, color=colour,
            edgecolor="black", linewidth=0.8, zorder=4,
        )
        ax.text(
            x_positions[idx], -3.0, sublabel,
            ha="center", va="top", fontsize=9, color="#333333",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.set_ylabel("SID (lower is better)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(
        "DAGMA per-seed SID: centred_only versus standardised"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, output_path)


def build_selection_summary_table(
    main_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Long-form summary table for the notebook.

    Rows: one per (condition, metric); columns: ``condition``,
    ``metric``, ``role``, ``dagma_value``, ``dcdi_value``,
    ``lower_model``. ``role`` is one of ``primary`` (SID),
    ``tie-breaker`` (MMD), ``diagnostic`` (SHD), or
    ``feasibility`` (runtime). ``lower_model`` is the model with the
    lower value at that cell (``"tie"`` if equal, ``""`` if a column
    is unavailable).
    """
    metrics: tuple[tuple[str, str, str], ...] = (
        ("mean_sid", "SID", "primary"),
        ("mean_mmd_primary", "MMD", "tie-breaker"),
        ("mean_shd", "SHD", "diagnostic"),
        ("mean_runtime_seconds", "runtime", "feasibility"),
    )
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_LABELS:
        sub = main_summary[main_summary["condition"] == condition]
        for col, label, role in metrics:
            if sub.empty or col not in sub.columns:
                rows.append({
                    "condition": condition,
                    "metric": label,
                    "role": role,
                    "dagma_value": None,
                    "dcdi_value": None,
                    "lower_model": "",
                })
                continue
            try:
                d_value = float(
                    sub[sub["model"] == "dagma"][col].iloc[0]
                )
                c_value = float(
                    sub[sub["model"] == "dcdi"][col].iloc[0]
                )
            except (IndexError, KeyError):
                rows.append({
                    "condition": condition,
                    "metric": label,
                    "role": role,
                    "dagma_value": None,
                    "dcdi_value": None,
                    "lower_model": "",
                })
                continue
            if d_value < c_value:
                lower = "DAGMA"
            elif c_value < d_value:
                lower = "DCDI"
            else:
                lower = "tie"
            rows.append({
                "condition": condition,
                "metric": label,
                "role": role,
                "dagma_value": d_value,
                "dcdi_value": c_value,
                "lower_model": lower,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manifest, summary markdown, notebook
# ---------------------------------------------------------------------------


def write_figure_manifest(
    *,
    output_root: Path,
    inputs_used: dict[str, Any],
    audit: dict[str, Any],
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
    notebook_path: Path,
    summary_md_path: Path,
    side_tables: dict[str, Path],
    output_path: Path,
) -> Path:
    manifest: dict[str, Any] = {
        "calibration_run_hash_prefix": CALIBRATION_RUN_HASH_PREFIX,
        "heldout_run_hash_prefix": HELDOUT_RUN_HASH_PREFIX,
        "selected_base_model": BASE_MODEL_DECISION_LABEL,
        "inputs_used": sorted(inputs_used),
        "audit": audit,
        "generated_figures": {
            name: str(p) for name, p in sorted(
                generated_figures.items()
            )
        },
        "skipped_figures": dict(sorted(skipped_figures.items())),
        "removed_figures": list(REMOVED_FIGURE_NAMES),
        "side_tables": {
            name: str(p) for name, p in sorted(side_tables.items())
        },
        "notebook_path": str(notebook_path),
        "summary_markdown_path": str(summary_md_path),
        "no_new_fits": True,
        "no_metric_recomputation": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def write_summary_markdown(
    *,
    output_root: Path,
    audit: dict[str, Any],
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
    output_path: Path,
) -> Path:
    """Concise labelling-only summary markdown."""
    rd = selection_readout_dir(output_root)
    lines: list[str] = []
    lines.append("# Base-model selection visual readout")
    lines.append("")
    lines.append("## Run identity")
    lines.append("")
    lines.append(
        f"- calibration_run_hash_prefix: `{CALIBRATION_RUN_HASH_PREFIX}`"
    )
    lines.append(
        f"- heldout_run_hash_prefix: `{HELDOUT_RUN_HASH_PREFIX}`"
    )
    lines.append(
        f"- base model carried forward: **{BASE_MODEL_DECISION_LABEL}**"
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
                rel = Path(str(p)).resolve().relative_to(rd.resolve())
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
    lines.append("## Decision context")
    lines.append("")
    lines.append(
        "This readout summarises evidence already recorded by the "
        "frozen base-model selection adjudication. No new claim is "
        "introduced. The selection rule is SID primary; MMD is the "
        "tie-breaker under the documented SID-margin condition; SHD "
        "and runtime are advisory."
    )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _make_notebook_payload(
    output_root: Path,
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
) -> dict[str, Any]:
    """Build the .ipynb JSON payload (labelling-only)."""
    fig_dir_rel = (
        Path("results") / "model_selection" / "held_out"
        / HELDOUT_RUN_HASH_PREFIX / "readout" / FIG_DIR_NAME
    )
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

    # 1. Setup
    md([
        "# Base-model selection visual readout",
        "",
        "Labelling-only display of the calibration handoff and the "
        "held-out comparison that the frozen selection rule was "
        "applied to. Interpretation belongs in the thesis text and "
        "in the existing adjudication; not here.",
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
        f"CALIBRATION_RUN_HASH_PREFIX = \"{CALIBRATION_RUN_HASH_PREFIX}\"",
        f"HELDOUT_RUN_HASH_PREFIX = \"{HELDOUT_RUN_HASH_PREFIX}\"",
        "",
        "READOUT_DIR = (",
        "    OUTPUT_ROOT / \"results\" / \"model_selection\" / "
        "\"held_out\"",
        f"    / HELDOUT_RUN_HASH_PREFIX / \"readout\"",
        ")",
        f"FIG_DIR = READOUT_DIR / \"{FIG_DIR_NAME}\"",
        "print(\"calibration_run_hash_prefix:\", "
        "CALIBRATION_RUN_HASH_PREFIX)",
        "print(\"heldout_run_hash_prefix:\", HELDOUT_RUN_HASH_PREFIX)",
        "print(\"figures dir:\", FIG_DIR)",
    ])

    md(["## 2. Data availability audit"])
    code([
        f"manifest_path = READOUT_DIR / \"{MANIFEST_JSON_NAME}\"",
        "if manifest_path.exists():",
        "    manifest = json.loads(manifest_path.read_text(encoding=\"utf-8\"))",
        "    display(Markdown(\"Figure manifest top-level keys:\"))",
        "    display(Markdown(\", \".join(sorted(manifest.keys()))))",
        "else:",
        "    display(Markdown(\"manifest not found; run the renderer first.\"))",
    ])

    # Section 3: Calibration handoff (DataFrame, no PNG).
    md([
        "## 3. Calibration handoff",
        "",
        "Rank-1 calibration configuration per (condition, model) "
        "cell. This is the input to the held-out comparison; it "
        "does not by itself decide the base model.",
    ])
    code([
        f"cal_path = READOUT_DIR / \"{SELECTED_CONFIG_TABLE_CSV}\"",
        "if cal_path.exists():",
        "    cal_df = pd.read_csv(cal_path)",
        "    display(Markdown(\"Calibration handoff: rank-1 "
        "configuration per cell.\"))",
        "    display(cal_df)",
        "else:",
        "    display(Markdown(\"(calibration table not generated; "
        "see manifest)\"))",
    ])

    # Section 4: Held-out evaluation status (data table + status fig).
    md(["## 4. Held-out evaluation status"])
    code([
        "status_path = READOUT_DIR / \"status_summary.csv\"",
        "if status_path.exists():",
        "    status_df = pd.read_csv(status_path)",
        "    display(Markdown(\"Held-out status counts.\"))",
        "    display(status_df)",
    ])
    code([
        f"path = FIG_DIR / \"{FIG_STATUS_NAME}\"",
        "display(Markdown(\"Figure: status counts by kind / "
        "status_field / status_value.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 5: Held-out metric means + paired differences (central).
    md([
        "## 5. Held-out metric means",
        "",
        "Per-condition / per-model means of SID, MMD, and SHD on "
        "the five held-out seeds, with per-seed dots.",
    ])
    code([
        f"path = FIG_DIR / \"{FIG02_NAME}\"",
        "display(Markdown(\"Figure 2: held-out mean SID / MMD / SHD "
        "with per-seed dots.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])
    md([
        "### Paired DAGMA vs DCDI differences on matched seeds",
        "",
        "Central evidential figure for the base-model choice. "
        "Positive values favour DAGMA because lower-is-better.",
    ])
    code([
        f"path = FIG_DIR / \"{FIG02B_NAME}\"",
        "display(Markdown(\"Figure 2b: paired per-seed differences "
        "(DCDI - DAGMA); positive values favour DAGMA.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 6: Per-seed SID evidence.
    md(["## 6. Per-seed SID evidence"])
    code([
        f"path = FIG_DIR / \"{FIG03_NAME}\"",
        "display(Markdown(\"Figure 3: per-seed held-out SID with "
        "mean marker.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 7: Frozen selection rule (text only).
    md([
        "## 7. Frozen selection rule",
        "",
        "SID is the primary criterion. MMD is the tie-breaker "
        "under the documented SID-margin condition. SHD is a "
        "diagnostic / advisory criterion. Runtime is feasibility "
        "evidence only.",
    ])

    # Section 8: Runtime.
    md(["## 8. Runtime and feasibility"])
    code([
        f"path = FIG_DIR / \"{FIG05_NAME}\"",
        "display(Markdown(\"Figure 5: held-out mean runtime on a "
        "log scale. Runtime is feasibility evidence only; metric "
        "rule remains SID/MMD-led.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 9: DCDI fit-RNG sensitivity.
    md(["## 9. DCDI fit-RNG sensitivity addendum"])
    code([
        f"path = FIG_DIR / \"{FIG06_NAME}\"",
        "display(Markdown(\"Figure 6: DCDI SID / MMD / SHD across "
        "alternative fit_rngs at seed 301.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 10: DAGMA ceiling/headroom.
    md(["## 10. DAGMA ceiling/headroom"])
    code([
        f"path = FIG_DIR / \"{FIG07_NAME}\"",
        "display(Markdown(\"Figure 7: DAGMA per-seed SID, "
        "centred_only versus standardised.\"))",
        "if path.exists():",
        "    display(Image(filename=str(path)))",
        "else:",
        "    display(Markdown(\"(figure not generated; see "
        "manifest)\"))",
    ])

    # Section 11: Selection summary as a DataFrame (no PNG).
    md([
        "## 11. Selection summary",
        "",
        "Per-(condition, metric) summary of which model is lower, "
        "with the selection-rule role of each metric. SID is the "
        "primary criterion, MMD the tie-breaker, SHD the "
        "diagnostic, runtime the feasibility / advisory criterion.",
    ])
    code([
        f"sel_path = READOUT_DIR / \"{SELECTION_SUMMARY_TABLE_CSV}\"",
        "if sel_path.exists():",
        "    sel_df = pd.read_csv(sel_path)",
        "    display(Markdown(\"Selection summary table.\"))",
        "    display(sel_df)",
        "    display(Markdown(\"**Base model carried forward: "
        f"{BASE_MODEL_DECISION_LABEL}**.\"))",
        "else:",
        "    display(Markdown(\"(selection summary table not "
        "generated; see manifest)\"))",
    ])

    md(["## 12. Output manifest"])
    code([
        "if manifest_path.exists():",
        "    print(\"Generated figures:\")",
        "    for name in sorted(manifest.get(\"generated_figures\", {})):",
        "        print(\" -\", name)",
        "    skipped = manifest.get(\"skipped_figures\", {})",
        "    if skipped:",
        "        print(\"Skipped figures:\")",
        "        for name, reason in sorted(skipped.items()):",
        "            print(\" -\", name, \"(\" + str(reason) + \")\")",
        "print(\"Base model carried forward: "
        f"{BASE_MODEL_DECISION_LABEL}\")",
        "print(\"All artefacts loaded. Thesis interpretation is "
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


def create_base_model_selection_notebook(
    *,
    output_root: Path,
    generated_figures: dict[str, Path],
    skipped_figures: dict[str, str],
    notebook_path: Path,
) -> Path:
    payload = _make_notebook_payload(
        output_root, generated_figures, skipped_figures,
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(payload, indent=1),
        encoding="utf-8",
    )
    return notebook_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def render_base_model_selection_readout(
    output_root: Path,
) -> dict[str, Any]:
    """Render the full visual readout. Returns the manifest dict."""
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    audit = audit_available_inputs(output_root)
    inputs = load_selection_inputs(output_root)
    rd = selection_readout_dir(output_root)
    figs_dir = ensure_output_dir(figures_output_dir(output_root))

    generated_figures: dict[str, Path] = {}
    skipped_figures: dict[str, str] = {}
    side_tables: dict[str, Path] = {}

    # Calibration handoff: write as DataFrame-backed CSV (no PNG).
    if inputs.selected_configurations is not None:
        cal_table = build_selected_configurations_table(
            inputs.selected_configurations
        )
        cal_path = rd / SELECTED_CONFIG_TABLE_CSV
        cal_table.to_csv(cal_path, index=False)
        side_tables["selected_configurations_table"] = cal_path
    else:
        side_tables["selected_configurations_table"] = (
            rd / SELECTED_CONFIG_TABLE_CSV
        )

    # Fig 02: held-out metric means.
    if (
        inputs.main_summary is not None
        and {"condition", "model", "mean_sid",
             "mean_mmd_primary", "mean_shd"}.issubset(
            set(inputs.main_summary.columns)
        )
    ):
        generated_figures[FIG02_NAME] = plot_heldout_metric_means(
            inputs.main_summary, inputs.per_seed_main,
            figs_dir / FIG02_NAME,
        )
    else:
        skipped_figures[FIG02_NAME] = (
            "main_summary.csv missing or lacks required columns"
        )

    # Fig 02b: paired DAGMA vs DCDI differences (central evidence).
    if (
        inputs.per_seed_main is not None
        and {"condition", "model", "seed_value"}.issubset(
            set(inputs.per_seed_main.columns)
        )
    ):
        generated_figures[FIG02B_NAME] = plot_paired_model_differences(
            inputs.per_seed_main, figs_dir / FIG02B_NAME,
        )
    else:
        skipped_figures[FIG02B_NAME] = (
            "per_seed_main.csv missing or lacks required columns"
        )

    # Fig 03: per-seed SID dots.
    if (
        inputs.per_seed_main is not None
        and {"condition", "model", "seed_value", "sid"}.issubset(
            set(inputs.per_seed_main.columns)
        )
    ):
        generated_figures[FIG03_NAME] = plot_heldout_sid_per_seed(
            inputs.per_seed_main, figs_dir / FIG03_NAME,
        )
    else:
        skipped_figures[FIG03_NAME] = (
            "per_seed_main.csv missing or lacks required columns"
        )

    # Fig 05: runtime log scale.
    if (
        inputs.main_summary is not None
        and "mean_runtime_seconds" in inputs.main_summary.columns
    ):
        generated_figures[FIG05_NAME] = plot_runtime_log_scale(
            inputs.main_summary, figs_dir / FIG05_NAME,
        )
    else:
        skipped_figures[FIG05_NAME] = (
            "main_summary.csv missing mean_runtime_seconds"
        )

    # Fig 06: DCDI fit-RNG sensitivity.
    required_sens_cols = {"condition", "model", "fit_rng", "sid"}
    if (
        inputs.sensitivity_summary is not None
        and required_sens_cols.issubset(
            set(inputs.sensitivity_summary.columns)
        )
        and not inputs.sensitivity_summary[
            inputs.sensitivity_summary["model"] == "dcdi"
        ].empty
    ):
        generated_figures[FIG06_NAME] = plot_dcdi_fit_rng_sensitivity(
            inputs.sensitivity_summary, figs_dir / FIG06_NAME,
        )
    else:
        skipped_figures[FIG06_NAME] = (
            "fit-RNG sensitivity data not available in persisted "
            "readout artefacts"
        )

    # Fig 07: DAGMA centred vs standardised per-seed SID.
    if (
        inputs.per_seed_main is not None
        and {"condition", "model", "sid"}.issubset(
            set(inputs.per_seed_main.columns)
        )
    ):
        generated_figures[FIG07_NAME] = plot_dagma_ceiling_and_headroom(
            inputs.per_seed_main, figs_dir / FIG07_NAME,
        )
    else:
        skipped_figures[FIG07_NAME] = (
            "per_seed_main.csv missing or lacks required columns"
        )

    # Status / reliability figure (from status_summary.csv).
    if (
        inputs.status_summary is not None
        and {"kind", "status_field", "status_value", "count"}.issubset(
            set(inputs.status_summary.columns)
        )
        and not inputs.status_summary.empty
    ):
        generated_figures[FIG_STATUS_NAME] = plot_status_reliability(
            inputs.status_summary, figs_dir / FIG_STATUS_NAME,
        )
    else:
        skipped_figures[FIG_STATUS_NAME] = (
            "status_summary.csv missing or lacks required columns"
        )

    # Selection summary table (no PNG).
    if (
        inputs.main_summary is not None
        and {"condition", "model", "mean_sid", "mean_mmd_primary",
             "mean_shd"}.issubset(set(inputs.main_summary.columns))
    ):
        summary_table = build_selection_summary_table(
            inputs.main_summary
        )
        sel_path = rd / SELECTION_SUMMARY_TABLE_CSV
        summary_table.to_csv(sel_path, index=False)
        side_tables["selection_summary_table"] = sel_path
    else:
        side_tables["selection_summary_table"] = (
            rd / SELECTION_SUMMARY_TABLE_CSV
        )

    summary_md_path = rd / SUMMARY_MD_NAME
    write_summary_markdown(
        output_root=output_root, audit=audit,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        output_path=summary_md_path,
    )

    nb_path = notebook_output_path(output_root)
    create_base_model_selection_notebook(
        output_root=output_root,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        notebook_path=nb_path,
    )

    manifest_path = rd / MANIFEST_JSON_NAME
    inputs_used = {
        k for k, v in audit.items() if v.get("exists", False)
    }
    write_figure_manifest(
        output_root=output_root,
        inputs_used=inputs_used,
        audit=audit,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        notebook_path=nb_path,
        summary_md_path=summary_md_path,
        side_tables=side_tables,
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
        "selected_base_model": BASE_MODEL_DECISION_LABEL,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_base_model_selection_readout",
        description=(
            "Visual readout of the completed base-model selection "
            "study. Read-only over persisted calibration / held-out "
            "/ adjudication artefacts; no fitting, no metric "
            "recomputation, no protocol change."
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
        result = render_base_model_selection_readout(args.output_root)
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            "render_base_model_selection_readout: error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return 1
    sys.stdout.write(
        f"selected_base_model: {result['selected_base_model']}\n"
    )
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
    "BASE_MODEL_DECISION_LABEL",
    "CALIBRATION_RUN_HASH_PREFIX",
    "FIG02_NAME",
    "FIG02B_NAME",
    "FIG03_NAME",
    "FIG05_NAME",
    "FIG06_NAME",
    "FIG07_NAME",
    "FIG_DIR_NAME",
    "FIG_STATUS_NAME",
    "HELDOUT_RUN_HASH_PREFIX",
    "MANIFEST_JSON_NAME",
    "NOTEBOOK_NAME",
    "REMOVED_FIGURE_NAMES",
    "SELECTED_CONFIG_TABLE_CSV",
    "SELECTION_SUMMARY_TABLE_CSV",
    "SUMMARY_MD_NAME",
    "SelectionInputs",
    "audit_available_inputs",
    "build_selected_configurations_table",
    "build_selection_summary_table",
    "calibration_dir",
    "create_base_model_selection_notebook",
    "ensure_output_dir",
    "figures_output_dir",
    "load_selection_inputs",
    "main",
    "notebook_output_path",
    "plot_dagma_ceiling_and_headroom",
    "plot_dcdi_fit_rng_sensitivity",
    "plot_heldout_metric_means",
    "plot_heldout_sid_per_seed",
    "plot_paired_model_differences",
    "plot_runtime_log_scale",
    "plot_status_reliability",
    "read_csv_if_exists",
    "read_json_if_exists",
    "render_base_model_selection_readout",
    "selection_readout_dir",
    "write_figure_manifest",
    "write_summary_markdown",
]
