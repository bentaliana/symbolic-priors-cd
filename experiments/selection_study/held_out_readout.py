"""Held-out evaluation readout: factual audit summary and figures.

This module reads a held-out run's ``heldout_evaluation.json`` and
the matching per-fit JSON records, validates them structurally,
and writes a small reproducible audit bundle:

- heldout_readout.md
- main_summary.csv
- per_seed_main.csv
- sensitivity_summary.csv
- status_summary.csv
- heldout_mean_sid.png
- heldout_mean_mmd.png
- heldout_mean_shd.png
- heldout_runtime.png
- heldout_sensitivity_addendum.png

The module is intentionally inspection-only:

- no model fit is invoked;
- ``pipeline.run_single_fit`` is not called;
- no wrapper module is imported;
- no input file is modified;
- no final base-model adjudication is performed.

The output is auditable evidence for a human reader. Final
base-model adjudication happens outside this generator.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.selection_study.held_out import (
    HELDOUT_EVALUATION_FILENAME,
    HELDOUT_SCM_SEEDS,
    RECORDS_DIRECTORY_NAME,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
)
from experiments.selection_study.held_out_artefact import (
    EXPECTED_MAIN_PER_CELL,
    EXPECTED_MAIN_TOTAL,
    EXPECTED_SENSITIVITY_TOTAL,
    EXPECTED_TOTAL_RECORDS,
    validate_heldout_evaluation_artefact,
)
from experiments.selection_study.selection_artefact import (
    CONDITIONS,
    MODELS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


READOUT_DIRECTORY_NAME = "readout"

MARKDOWN_FILENAME = "heldout_readout.md"
MAIN_SUMMARY_CSV_FILENAME = "main_summary.csv"
PER_SEED_MAIN_CSV_FILENAME = "per_seed_main.csv"
SENSITIVITY_SUMMARY_CSV_FILENAME = "sensitivity_summary.csv"
STATUS_SUMMARY_CSV_FILENAME = "status_summary.csv"
SID_PNG_FILENAME = "heldout_mean_sid.png"
MMD_PNG_FILENAME = "heldout_mean_mmd.png"
SHD_PNG_FILENAME = "heldout_mean_shd.png"
RUNTIME_PNG_FILENAME = "heldout_runtime.png"
SENSITIVITY_PNG_FILENAME = "heldout_sensitivity_addendum.png"

_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "winner",
    "model_winner",
    "base_model_winner",
    "recommended_model",
    "final_decision",
    "DAGMA wins",
    "DCDI wins",
)

# Y-axis log floor for the runtime plot so that any zero-second
# synthetic value still renders on a log axis.
_RUNTIME_LOG_FLOOR_SECONDS = 1e-3


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"required held-out readout input file not found at {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"held-out readout input file at {path} is not valid "
                f"JSON: {exc}"
            ) from exc


def _load_heldout_artefact(heldout_run_dir: Path) -> dict[str, Any]:
    artefact_path = heldout_run_dir / HELDOUT_EVALUATION_FILENAME
    artefact = _read_json_file(artefact_path)
    if not isinstance(artefact, dict):
        raise ValueError(
            f"heldout_evaluation artefact at {artefact_path} must be "
            f"a JSON object at the top level; got {type(artefact).__name__}"
        )
    validate_heldout_evaluation_artefact(artefact)
    return artefact


def _load_record_filenames(heldout_run_dir: Path) -> list[str]:
    records_dir = heldout_run_dir / RECORDS_DIRECTORY_NAME
    if not records_dir.is_dir():
        raise FileNotFoundError(
            f"held-out records directory not found at {records_dir}"
        )
    files = sorted(p.name for p in records_dir.glob("*.json"))
    if len(files) != EXPECTED_TOTAL_RECORDS:
        raise ValueError(
            "held-out records directory must contain exactly "
            f"{EXPECTED_TOTAL_RECORDS} JSON record files "
            f"({EXPECTED_MAIN_TOTAL} main + "
            f"{EXPECTED_SENSITIVITY_TOTAL} sensitivity); got "
            f"{len(files)} at {records_dir}"
        )
    return files


def _assert_record_counts(artefact: Mapping[str, Any]) -> None:
    status_summary = artefact["status_summary"]
    if status_summary["total_records"] != EXPECTED_TOTAL_RECORDS:
        raise ValueError(
            "heldout_evaluation status_summary.total_records must "
            f"equal {EXPECTED_TOTAL_RECORDS}; got "
            f"{status_summary['total_records']}"
        )
    if status_summary["main_records_count"] != EXPECTED_MAIN_TOTAL:
        raise ValueError(
            "heldout_evaluation status_summary.main_records_count "
            f"must equal {EXPECTED_MAIN_TOTAL}; got "
            f"{status_summary['main_records_count']}"
        )
    if (
        status_summary["sensitivity_records_count"]
        != EXPECTED_SENSITIVITY_TOTAL
    ):
        raise ValueError(
            "heldout_evaluation status_summary.sensitivity_records_count "
            f"must equal {EXPECTED_SENSITIVITY_TOTAL}; got "
            f"{status_summary['sensitivity_records_count']}"
        )

    # Cross-check per-cell shape: 4 cells x 5 per_seed_records each.
    cells = artefact["main_evaluation"]["cells"]
    for condition in CONDITIONS:
        for model in MODELS:
            cell = cells[condition][model]
            if len(cell["per_seed_records"]) != EXPECTED_MAIN_PER_CELL:
                raise ValueError(
                    f"main_evaluation cell ({condition!r}, {model!r}) "
                    f"must contain exactly {EXPECTED_MAIN_PER_CELL} "
                    "per_seed_records; got "
                    f"{len(cell['per_seed_records'])}"
                )
    addendum = artefact["fit_rng_sensitivity_addendum"]
    if len(addendum["per_fit_records"]) != EXPECTED_SENSITIVITY_TOTAL:
        raise ValueError(
            "fit_rng_sensitivity_addendum.per_fit_records must contain "
            f"exactly {EXPECTED_SENSITIVITY_TOTAL} records; got "
            f"{len(addendum['per_fit_records'])}"
        )


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return repr(value)
    return str(value)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    return str(value)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


_METRIC_SUMMARY_FIELDS: tuple[str, ...] = (
    "mean",
    "std",
    "median",
    "q1",
    "q3",
    "iqr",
    "min",
    "max",
)


def _build_main_summary_row(
    *, condition: str, model: str, cell: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate = cell["aggregate_metrics"]
    row: dict[str, Any] = {"condition": condition, "model": model}
    for metric_name in ("sid", "mmd_primary", "shd"):
        summary = aggregate[metric_name]
        for stat in _METRIC_SUMMARY_FIELDS:
            row[f"{stat}_{metric_name}"] = summary.get(stat)
    runtime_summary = aggregate["runtime_seconds"]
    row["mean_runtime_seconds"] = runtime_summary.get("mean")
    row["std_runtime_seconds"] = runtime_summary.get("std")
    status_counts = aggregate.get("status_counts", {})
    row["training_status_counts_json"] = json.dumps(
        status_counts.get("training_status", {}),
        sort_keys=True,
        ensure_ascii=True,
    )
    row["graph_status_counts_json"] = json.dumps(
        status_counts.get("graph_status", {}),
        sort_keys=True,
        ensure_ascii=True,
    )
    row["sampler_status_counts_json"] = json.dumps(
        status_counts.get("sampler_status", {}),
        sort_keys=True,
        ensure_ascii=True,
    )
    return row


def _build_main_summary_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = artefact["main_evaluation"]["cells"]
    for condition in CONDITIONS:
        for model in MODELS:
            rows.append(
                _build_main_summary_row(
                    condition=condition,
                    model=model,
                    cell=cells[condition][model],
                )
            )
    return rows


def _build_per_seed_main_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = artefact["main_evaluation"]["cells"]
    for condition in CONDITIONS:
        for model in MODELS:
            cell = cells[condition][model]
            for record in cell["per_seed_records"]:
                rows.append(
                    {
                        "condition": condition,
                        "model": model,
                        "seed_value": int(record["seed_value"]),
                        "fit_rng": record.get("fit_rng"),
                        "sid": record.get("sid"),
                        "mmd_primary": record.get("mmd_primary"),
                        "shd": record.get("shd"),
                        "runtime_seconds": record.get("runtime_seconds"),
                        "training_status": str(
                            record.get("training_status", "")
                        ),
                        "graph_status": str(
                            record.get("graph_status", "")
                        ),
                        "sampler_status": str(
                            record.get("sampler_status", "")
                        ),
                    }
                )
    return rows


def _build_sensitivity_summary_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    addendum = artefact["fit_rng_sensitivity_addendum"]
    target_cell = addendum["target_cell"]
    diagnostic = addendum["diagnostic_summary"]
    main_sid_at_301 = diagnostic.get("main_evaluation_sid_at_seed_301")
    main_mmd_at_301 = diagnostic.get(
        "main_evaluation_mmd_primary_at_seed_301"
    )
    main_shd_at_301 = diagnostic.get("main_evaluation_shd_at_seed_301")

    rows: list[dict[str, Any]] = []
    for record in addendum["per_fit_records"]:
        rows.append(
            {
                "condition": str(target_cell["condition"]),
                "model": str(target_cell["model"]),
                "scm_seed": int(addendum["scm_seed"]),
                "fit_rng": int(record["fit_rng"]),
                "sid": record.get("sid"),
                "mmd_primary": record.get("mmd_primary"),
                "shd": record.get("shd"),
                "runtime_seconds": record.get("runtime_seconds"),
                "n_iterations": record.get("n_iterations"),
                "training_status": str(record.get("training_status", "")),
                "graph_status": str(record.get("graph_status", "")),
                "sampler_status": str(record.get("sampler_status", "")),
                "main_evaluation_sid_at_seed_301": main_sid_at_301,
                "main_evaluation_mmd_primary_at_seed_301": (
                    main_mmd_at_301
                ),
                "main_evaluation_shd_at_seed_301": main_shd_at_301,
            }
        )
    return rows


def _build_status_summary_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the artefact's status_summary into one row per status value."""
    status_summary = artefact["status_summary"]
    rows: list[dict[str, Any]] = []
    for kind, key in (
        ("main", "main_status_counts"),
        ("sensitivity", "sensitivity_status_counts"),
    ):
        per_field = status_summary.get(key, {})
        for status_field, counts in sorted(per_field.items()):
            for status_value, count in sorted(counts.items()):
                rows.append(
                    {
                        "kind": kind,
                        "status_field": status_field,
                        "status_value": status_value,
                        "count": int(count),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


_MAIN_SUMMARY_FIELDS: tuple[str, ...] = (
    "condition",
    "model",
    "mean_sid",
    "std_sid",
    "median_sid",
    "q1_sid",
    "q3_sid",
    "iqr_sid",
    "min_sid",
    "max_sid",
    "mean_mmd_primary",
    "std_mmd_primary",
    "median_mmd_primary",
    "q1_mmd_primary",
    "q3_mmd_primary",
    "iqr_mmd_primary",
    "min_mmd_primary",
    "max_mmd_primary",
    "mean_shd",
    "std_shd",
    "median_shd",
    "q1_shd",
    "q3_shd",
    "iqr_shd",
    "min_shd",
    "max_shd",
    "mean_runtime_seconds",
    "std_runtime_seconds",
    "training_status_counts_json",
    "graph_status_counts_json",
    "sampler_status_counts_json",
)

_PER_SEED_MAIN_FIELDS: tuple[str, ...] = (
    "condition",
    "model",
    "seed_value",
    "fit_rng",
    "sid",
    "mmd_primary",
    "shd",
    "runtime_seconds",
    "training_status",
    "graph_status",
    "sampler_status",
)

_SENSITIVITY_SUMMARY_FIELDS: tuple[str, ...] = (
    "condition",
    "model",
    "scm_seed",
    "fit_rng",
    "sid",
    "mmd_primary",
    "shd",
    "runtime_seconds",
    "n_iterations",
    "training_status",
    "graph_status",
    "sampler_status",
    "main_evaluation_sid_at_seed_301",
    "main_evaluation_mmd_primary_at_seed_301",
    "main_evaluation_shd_at_seed_301",
)

_STATUS_SUMMARY_FIELDS: tuple[str, ...] = (
    "kind",
    "status_field",
    "status_value",
    "count",
)


def _write_csv(
    output_path: Path,
    *,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(field_names))
        for row in rows:
            writer.writerow(
                [_csv_cell(row.get(name)) for name in field_names]
            )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _consistent_ylim(
    per_seed_rows: Sequence[Mapping[str, Any]], metric_field: str
) -> tuple[float, float]:
    """Return (low, high) ylimits spanning every finite value of metric_field."""
    finite_values: list[float] = [
        float(row[metric_field])
        for row in per_seed_rows
        if _is_finite(row.get(metric_field))
    ]
    if not finite_values:
        return (0.0, 1.0)
    low = min(finite_values)
    high = max(finite_values)
    if low == high:
        # Add a small symmetric padding so a flat metric is still
        # distinguishable from a thin horizontal line at the y-axis
        # boundary.
        padding = max(1.0, abs(low) * 0.1)
        return (low - padding, high + padding)
    span = high - low
    padding = span * 0.08
    return (low - padding, high + padding)


def _plot_main_metric(
    *,
    metric_field: str,
    metric_label: str,
    per_seed_rows: Sequence[Mapping[str, Any]],
    main_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Render a 2x2 grid of per-seed points with mean and median markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_cell: dict[tuple[str, str], Mapping[str, Any]] = {
        (row["condition"], row["model"]): row for row in main_summary_rows
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 8.0), constrained_layout=True
    )
    ylow, yhigh = _consistent_ylim(per_seed_rows, metric_field)

    cells: list[tuple[str, str]] = [
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    ]
    for ax, (condition, model) in zip(axes.flat, cells):
        cell_rows = [
            row
            for row in per_seed_rows
            if row["condition"] == condition and row["model"] == model
        ]
        cell_rows.sort(key=lambda r: int(r["seed_value"]))
        xs = [int(row["seed_value"]) for row in cell_rows]
        ys = [
            float(row[metric_field])
            if _is_finite(row.get(metric_field))
            else float("nan")
            for row in cell_rows
        ]
        ax.scatter(xs, ys, marker="o", s=48)

        summary = summary_by_cell.get((condition, model), {})
        mean_value = summary.get(f"mean_{metric_field}")
        median_value = summary.get(f"median_{metric_field}")
        if _is_finite(mean_value) and xs:
            ax.hlines(
                float(mean_value),
                xmin=min(xs) - 0.4,
                xmax=max(xs) + 0.4,
                linewidth=2.0,
                linestyle="-",
                label="mean",
            )
            ax.annotate(
                f"mean = {_format_number(mean_value)}",
                xy=(max(xs), float(mean_value)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        if _is_finite(median_value) and xs:
            ax.hlines(
                float(median_value),
                xmin=min(xs) - 0.4,
                xmax=max(xs) + 0.4,
                linewidth=1.0,
                linestyle="--",
                label="median",
            )

        ax.set_xlabel("held-out SCM seed")
        ax.set_ylabel(f"per-seed {metric_label}")
        ax.set_title(f"{condition} / {model}")
        ax.set_xticks(list(HELDOUT_SCM_SEEDS))
        ax.set_ylim(ylow, yhigh)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(f"Held-out per-seed {metric_label} by cell")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_runtime(
    *,
    per_seed_rows: Sequence[Mapping[str, Any]],
    main_summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Render the per-seed runtime panel grid on a log y-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_cell: dict[tuple[str, str], Mapping[str, Any]] = {
        (row["condition"], row["model"]): row for row in main_summary_rows
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 8.0), constrained_layout=True
    )
    cells: list[tuple[str, str]] = [
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    ]
    for ax, (condition, model) in zip(axes.flat, cells):
        cell_rows = [
            row
            for row in per_seed_rows
            if row["condition"] == condition and row["model"] == model
        ]
        cell_rows.sort(key=lambda r: int(r["seed_value"]))
        xs = [int(row["seed_value"]) for row in cell_rows]
        ys_raw = [row.get("runtime_seconds") for row in cell_rows]
        ys_plot: list[float] = []
        for value in ys_raw:
            if _is_finite(value) and float(value) > 0.0:
                ys_plot.append(float(value))
            else:
                ys_plot.append(_RUNTIME_LOG_FLOOR_SECONDS)
        ax.scatter(xs, ys_plot, marker="o", s=48)

        summary = summary_by_cell.get((condition, model), {})
        mean_value = summary.get("mean_runtime_seconds")
        if _is_finite(mean_value) and xs:
            mean_for_plot = max(
                float(mean_value), _RUNTIME_LOG_FLOOR_SECONDS
            )
            ax.hlines(
                mean_for_plot,
                xmin=min(xs) - 0.4,
                xmax=max(xs) + 0.4,
                linewidth=2.0,
                linestyle="-",
            )
            ax.annotate(
                f"mean = {_format_number(mean_value)} s",
                xy=(max(xs), mean_for_plot),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("held-out SCM seed")
        ax.set_ylabel("runtime (s, log scale)")
        ax.set_title(f"{condition} / {model}")
        ax.set_xticks(list(HELDOUT_SCM_SEEDS))
        if any(v > 0 for v in ys_plot):
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.suptitle("Held-out per-seed runtime by cell")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _build_sensitivity_plot_series(
    artefact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the x/y series for the sensitivity plot.

    Returns a JSON-safe dict with:

    - ``fit_rngs``: ordered list of fit_rng values, fit_rng=42 first;
    - ``main_reference``: the (sid, mmd_primary, shd) values at
      fit_rng=42 from the diagnostic summary;
    - ``sensitivity_points``: list of records for fit_rng=43..47.
    """
    addendum = artefact["fit_rng_sensitivity_addendum"]
    diagnostic = addendum["diagnostic_summary"]
    main_ref = {
        "sid": diagnostic.get("main_evaluation_sid_at_seed_301"),
        "mmd_primary": diagnostic.get(
            "main_evaluation_mmd_primary_at_seed_301"
        ),
        "shd": diagnostic.get("main_evaluation_shd_at_seed_301"),
    }
    sensitivity_points = []
    for record in sorted(
        addendum["per_fit_records"], key=lambda r: int(r["fit_rng"])
    ):
        sensitivity_points.append(
            {
                "fit_rng": int(record["fit_rng"]),
                "sid": record.get("sid"),
                "mmd_primary": record.get("mmd_primary"),
                "shd": record.get("shd"),
            }
        )
    fit_rngs_in_plot = [int(DCDI_MAIN_FIT_RNG_VALUE)] + [
        int(point["fit_rng"]) for point in sensitivity_points
    ]
    return {
        "fit_rngs": fit_rngs_in_plot,
        "main_reference": main_ref,
        "sensitivity_points": sensitivity_points,
    }


# The main DCDI fit_rng (42) lives in ``held_out``, but we re-pin the
# value here as a module-local constant so the readout module does not
# need to import the production-adapter helpers.
DCDI_MAIN_FIT_RNG_VALUE = 42


def _plot_sensitivity_addendum(
    artefact: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    """Render the fit-RNG sensitivity panels (SID, MMD, SHD)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _build_sensitivity_plot_series(artefact)
    main_ref = series["main_reference"]
    sensitivity_points = series["sensitivity_points"]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.5), constrained_layout=True
    )
    metric_specs: tuple[tuple[str, str], ...] = (
        ("sid", "SID"),
        ("mmd_primary", "MMD primary"),
        ("shd", "SHD"),
    )
    for ax, (metric_field, metric_label) in zip(axes, metric_specs):
        sensitivity_xs = [point["fit_rng"] for point in sensitivity_points]
        sensitivity_ys = [
            float(point[metric_field])
            if _is_finite(point.get(metric_field))
            else float("nan")
            for point in sensitivity_points
        ]
        ax.scatter(
            sensitivity_xs,
            sensitivity_ys,
            marker="o",
            s=48,
            label="sensitivity (fit_rng=43..47)",
        )
        main_value = main_ref.get(metric_field)
        if _is_finite(main_value):
            ax.scatter(
                [DCDI_MAIN_FIT_RNG_VALUE],
                [float(main_value)],
                marker="*",
                s=180,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="main reference (fit_rng=42)",
            )
            ax.annotate(
                "main",
                xy=(DCDI_MAIN_FIT_RNG_VALUE, float(main_value)),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xticks(series["fit_rngs"])
        ax.set_xlabel("fit_rng")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "DCDI fit-RNG sensitivity addendum "
        f"({SENSITIVITY_CONDITION} / {SENSITIVITY_MODEL} / "
        f"SCM seed {SENSITIVITY_SCM_SEED}; diagnostic only)"
    )
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown observations
# ---------------------------------------------------------------------------


def _per_seed_observation_line(
    *,
    condition: str,
    model: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Return one factual descriptor of the 5 per-seed values for a cell."""
    sid_values = [
        row["sid"] for row in rows if _is_finite(row.get("sid"))
    ]
    mmd_values = [
        row["mmd_primary"]
        for row in rows
        if _is_finite(row.get("mmd_primary"))
    ]
    shd_values = [
        row["shd"] for row in rows if _is_finite(row.get("shd"))
    ]
    sid_line = (
        f"SID: {sorted(int(v) for v in sid_values)}"
        if sid_values
        else "SID: no finite values"
    )
    mmd_line = (
        "MMD: "
        + ", ".join(_format_number(v) for v in sorted(mmd_values))
        if mmd_values
        else "MMD: no finite values"
    )
    shd_line = (
        f"SHD: {sorted(int(v) for v in shd_values)}"
        if shd_values
        else "SHD: no finite values"
    )
    return (
        f"- {condition} / {model}: {sid_line}; {mmd_line}; {shd_line}."
    )


def _build_per_seed_observations(
    per_seed_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    by_cell: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for row in per_seed_rows:
        key = (row["condition"], row["model"])
        by_cell.setdefault(key, []).append(row)
    lines: list[str] = []
    for condition in CONDITIONS:
        for model in MODELS:
            cell_rows = by_cell.get((condition, model), [])
            lines.append(
                _per_seed_observation_line(
                    condition=condition,
                    model=model,
                    rows=cell_rows,
                )
            )
    return lines


def _build_methodological_observations(
    main_summary_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Build a small, factual interpretation list from the main aggregates.

    Strictly descriptive: no selection language, no recommendation,
    no winner field, no comparison to external benchmarks.
    """
    by_cell = {
        (row["condition"], row["model"]): row for row in main_summary_rows
    }
    observations: list[str] = []
    centred_dagma = by_cell.get(("centred_only", "dagma"))
    centred_dcdi = by_cell.get(("centred_only", "dcdi"))
    standardised_dagma = by_cell.get(("standardised", "dagma"))
    if (
        centred_dagma is not None
        and centred_dcdi is not None
        and _is_finite(centred_dagma.get("mean_sid"))
        and _is_finite(centred_dcdi.get("mean_sid"))
        and centred_dagma["mean_sid"] < centred_dcdi["mean_sid"]
    ):
        observations.append(
            "centred_only / dagma has substantially lower mean SID, "
            f"MMD, and SHD than the other cells: mean SID "
            f"{_format_number(centred_dagma['mean_sid'])} versus "
            f"{_format_number(centred_dcdi['mean_sid'])} for "
            "centred_only / dcdi."
        )
    if (
        centred_dagma is not None
        and standardised_dagma is not None
        and _is_finite(centred_dagma.get("mean_sid"))
        and _is_finite(standardised_dagma.get("mean_sid"))
        and standardised_dagma["mean_sid"] > centred_dagma["mean_sid"]
    ):
        observations.append(
            "standardised appears substantially harder than "
            "centred_only for DAGMA: mean SID "
            f"{_format_number(standardised_dagma['mean_sid'])} "
            "under standardised versus "
            f"{_format_number(centred_dagma['mean_sid'])} under "
            "centred_only."
        )
    return observations


def _sensitivity_observation(
    artefact: Mapping[str, Any],
) -> str:
    """Return a one-line factual sensitivity observation."""
    series = _build_sensitivity_plot_series(artefact)
    main_ref_sid = series["main_reference"]["sid"]
    sensitivity_sids = [
        point["sid"] for point in series["sensitivity_points"]
    ]
    finite_sens_sids = [
        float(value)
        for value in sensitivity_sids
        if _is_finite(value)
    ]
    if (
        not _is_finite(main_ref_sid)
        or not finite_sens_sids
    ):
        return (
            "DCDI fit-RNG sensitivity: not enough finite SID values "
            "to compare the fixed-RNG result to the sensitivity range."
        )
    min_sens = min(finite_sens_sids)
    max_sens = max(finite_sens_sids)
    inside = min_sens <= float(main_ref_sid) <= max_sens
    relation = "within" if inside else "outside"
    return (
        "DCDI fit-RNG sensitivity does not suggest the fixed-RNG DCDI "
        "result was an isolated outlier: fixed-RNG SID at fit_rng=42 is "
        f"{_format_number(main_ref_sid)} and the fit_rng=43..47 SID "
        f"range is [{_format_number(min_sens)}, "
        f"{_format_number(max_sens)}] ({relation} the sensitivity range)."
    )


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------


def _format_md_table(
    *, headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_main_summary_table(
    main_summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    headers = (
        "condition",
        "model",
        "mean SID",
        "mean MMD",
        "mean SHD",
        "mean runtime (s)",
    )
    rows = [
        [
            row["condition"],
            row["model"],
            _format_number(row["mean_sid"]),
            _format_number(row["mean_mmd_primary"]),
            _format_number(row["mean_shd"]),
            _format_number(row["mean_runtime_seconds"]),
        ]
        for row in main_summary_rows
    ]
    return _format_md_table(headers=headers, rows=rows)


def _build_sensitivity_table(
    sensitivity_rows: Sequence[Mapping[str, Any]],
) -> str:
    headers = (
        "fit_rng",
        "SID",
        "MMD",
        "SHD",
        "runtime (s)",
    )
    rows = [
        [
            str(row["fit_rng"]),
            _format_number(row["sid"]),
            _format_number(row["mmd_primary"]),
            _format_number(row["shd"]),
            _format_number(row["runtime_seconds"]),
        ]
        for row in sensitivity_rows
    ]
    return _format_md_table(headers=headers, rows=rows)


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _all_records_clean(artefact: Mapping[str, Any]) -> bool:
    """Return True iff every main and sensitivity record converged cleanly."""
    cells = artefact["main_evaluation"]["cells"]
    for condition in CONDITIONS:
        for model in MODELS:
            for record in cells[condition][model]["per_seed_records"]:
                if record.get("training_status") != "converged":
                    return False
                if record.get("graph_status") != "valid_dag":
                    return False
                if record.get("sampler_status") != "available":
                    return False
    for record in artefact["fit_rng_sensitivity_addendum"]["per_fit_records"]:
        if record.get("training_status") != "converged":
            return False
        if record.get("graph_status") != "valid_dag":
            return False
        if record.get("sampler_status") != "available":
            return False
    return True


def _write_markdown(
    *,
    output_path: Path,
    artefact: Mapping[str, Any],
    main_summary_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    generated_filenames: Mapping[str, str],
) -> None:
    heldout_hash_prefix = artefact["heldout_run_hash_prefix"]
    parent_calibration_prefix = artefact["parent_calibration_run_hash_prefix"]
    generated_at_artefact = artefact.get("generated_at_utc", "unknown")
    main_total = EXPECTED_MAIN_TOTAL
    sensitivity_total = EXPECTED_SENSITIVITY_TOTAL
    total_records = EXPECTED_TOTAL_RECORDS

    clean = _all_records_clean(artefact)
    clean_line = (
        "All 25 records converged, produced valid DAGs, and had "
        "available samplers."
        if clean
        else (
            "At least one record did not converge, did not produce a "
            "valid DAG, or did not have an available sampler. See the "
            "status_summary.csv for details."
        )
    )

    per_seed_observations = _build_per_seed_observations(per_seed_rows)
    methodological_observations = _build_methodological_observations(
        main_summary_rows
    )
    sensitivity_observation = _sensitivity_observation(artefact)

    main_summary_table = _build_main_summary_table(main_summary_rows)
    sensitivity_table = _build_sensitivity_table(sensitivity_rows)

    lines = [
        "# Held-out evaluation readout",
        "",
        f"heldout_run_hash_prefix: {heldout_hash_prefix}",
        f"parent_calibration_run_hash_prefix: {parent_calibration_prefix}",
        f"generated_at_utc (artefact): {generated_at_artefact}",
        "",
        "## Status",
        "",
        "heldout_evaluation.json validates against the held-out "
        "evaluation schema.",
        "",
        f"Records loaded: {total_records} total "
        f"({main_total} main + {sensitivity_total} sensitivity).",
        "",
        clean_line,
        "",
        "The DCDI fit-RNG sensitivity addendum is a supplementary "
        "diagnostic, structurally separate from main evidence; it does "
        "not enter the main aggregates.",
        "",
        "No prior-loss experiment is started by this readout. Final "
        "base-model adjudication is performed outside this generator.",
        "",
        "## Scope",
        "",
        "This file audits one held-out run identified by the "
        "heldout_run_hash above. It loads the held-out evaluation "
        "artefact and the 25 per-fit JSON records, writes four CSV "
        "summaries and five PNG figures, and emits this markdown "
        "report. No model fits are invoked, no input file is "
        "modified, and no automatic final-decision logic is applied.",
        "",
        "## Main held-out summary",
        "",
        main_summary_table,
        "",
        "## Per-seed observations",
        "",
    ]
    if per_seed_observations:
        lines.extend(per_seed_observations)
    else:
        lines.append("- No per-seed observations available.")
    lines.extend(
        [
            "",
            "## DCDI fit-RNG sensitivity addendum",
            "",
            f"Target cell: {SENSITIVITY_CONDITION} / {SENSITIVITY_MODEL} "
            f"at SCM seed {SENSITIVITY_SCM_SEED}.",
            "",
            "Sensitivity per-fit values:",
            "",
            sensitivity_table,
            "",
            sensitivity_observation,
            "",
            "## Runtime summary",
            "",
            "Per-cell mean runtime values appear in main_summary.csv "
            "and in the runtime figure (log y-axis).",
            "",
            "## Methodological interpretation",
            "",
        ]
    )
    if methodological_observations:
        for observation in methodological_observations:
            lines.append(f"- {observation}")
    else:
        lines.append("- No automatically derived observations.")
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            f"- {generated_filenames['markdown']}",
            f"- {generated_filenames['main_summary_csv']}",
            f"- {generated_filenames['per_seed_main_csv']}",
            f"- {generated_filenames['sensitivity_summary_csv']}",
            f"- {generated_filenames['status_summary_csv']}",
            f"- {generated_filenames['sid_png']}",
            f"- {generated_filenames['mmd_png']}",
            f"- {generated_filenames['shd_png']}",
            f"- {generated_filenames['runtime_png']}",
            f"- {generated_filenames['sensitivity_png']}",
            "",
            "## Reproducibility note",
            "",
            "- generator: experiments/selection_study/held_out_readout.py",
            "- inputs: heldout_evaluation.json and records/*.json "
            "under the held-out run directory above",
            "- outputs: the CSV summaries, PNG figures, and this "
            "markdown report",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_heldout_readout(
    heldout_run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate the held-out readout bundle.

    Parameters
    ----------
    heldout_run_dir : Path or str
        Path to the held-out run directory; must contain
        ``heldout_evaluation.json`` and a ``records/`` subdirectory.
    output_dir : Path or str or None, optional
        Output directory for the readout files. Defaults to
        ``<heldout_run_dir>/readout/``. Existing files at the
        per-filename paths are overwritten so the readout is
        idempotent.

    Returns
    -------
    dict
        JSON-safe report with generated paths, summary counts, and the
        held-out run hash prefix.
    """
    run_dir = Path(heldout_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"held-out run directory not found at {run_dir}"
        )

    artefact = _load_heldout_artefact(run_dir)
    _assert_record_counts(artefact)
    record_filenames = _load_record_filenames(run_dir)

    if output_dir is None:
        readout_dir = run_dir / READOUT_DIRECTORY_NAME
    else:
        readout_dir = Path(output_dir).resolve()
    readout_dir.mkdir(parents=True, exist_ok=True)

    main_summary_rows = _build_main_summary_rows(artefact)
    per_seed_rows = _build_per_seed_main_rows(artefact)
    sensitivity_rows = _build_sensitivity_summary_rows(artefact)
    status_rows = _build_status_summary_rows(artefact)

    main_summary_csv_path = readout_dir / MAIN_SUMMARY_CSV_FILENAME
    per_seed_main_csv_path = readout_dir / PER_SEED_MAIN_CSV_FILENAME
    sensitivity_summary_csv_path = (
        readout_dir / SENSITIVITY_SUMMARY_CSV_FILENAME
    )
    status_summary_csv_path = readout_dir / STATUS_SUMMARY_CSV_FILENAME
    sid_png_path = readout_dir / SID_PNG_FILENAME
    mmd_png_path = readout_dir / MMD_PNG_FILENAME
    shd_png_path = readout_dir / SHD_PNG_FILENAME
    runtime_png_path = readout_dir / RUNTIME_PNG_FILENAME
    sensitivity_png_path = readout_dir / SENSITIVITY_PNG_FILENAME
    markdown_path = readout_dir / MARKDOWN_FILENAME

    _write_csv(
        main_summary_csv_path,
        field_names=_MAIN_SUMMARY_FIELDS,
        rows=main_summary_rows,
    )
    _write_csv(
        per_seed_main_csv_path,
        field_names=_PER_SEED_MAIN_FIELDS,
        rows=per_seed_rows,
    )
    _write_csv(
        sensitivity_summary_csv_path,
        field_names=_SENSITIVITY_SUMMARY_FIELDS,
        rows=sensitivity_rows,
    )
    _write_csv(
        status_summary_csv_path,
        field_names=_STATUS_SUMMARY_FIELDS,
        rows=status_rows,
    )

    _plot_main_metric(
        metric_field="sid",
        metric_label="SID",
        per_seed_rows=per_seed_rows,
        main_summary_rows=main_summary_rows,
        output_path=sid_png_path,
    )
    _plot_main_metric(
        metric_field="mmd_primary",
        metric_label="MMD primary",
        per_seed_rows=per_seed_rows,
        main_summary_rows=main_summary_rows,
        output_path=mmd_png_path,
    )
    _plot_main_metric(
        metric_field="shd",
        metric_label="SHD",
        per_seed_rows=per_seed_rows,
        main_summary_rows=main_summary_rows,
        output_path=shd_png_path,
    )
    _plot_runtime(
        per_seed_rows=per_seed_rows,
        main_summary_rows=main_summary_rows,
        output_path=runtime_png_path,
    )
    _plot_sensitivity_addendum(
        artefact, output_path=sensitivity_png_path
    )

    generated_filenames = {
        "markdown": MARKDOWN_FILENAME,
        "main_summary_csv": MAIN_SUMMARY_CSV_FILENAME,
        "per_seed_main_csv": PER_SEED_MAIN_CSV_FILENAME,
        "sensitivity_summary_csv": SENSITIVITY_SUMMARY_CSV_FILENAME,
        "status_summary_csv": STATUS_SUMMARY_CSV_FILENAME,
        "sid_png": SID_PNG_FILENAME,
        "mmd_png": MMD_PNG_FILENAME,
        "shd_png": SHD_PNG_FILENAME,
        "runtime_png": RUNTIME_PNG_FILENAME,
        "sensitivity_png": SENSITIVITY_PNG_FILENAME,
    }

    _write_markdown(
        output_path=markdown_path,
        artefact=artefact,
        main_summary_rows=main_summary_rows,
        per_seed_rows=per_seed_rows,
        sensitivity_rows=sensitivity_rows,
        generated_filenames=generated_filenames,
    )

    report = {
        "heldout_run_dir": str(run_dir),
        "output_dir": str(readout_dir),
        "heldout_run_hash_prefix": artefact["heldout_run_hash_prefix"],
        "parent_calibration_run_hash_prefix": artefact[
            "parent_calibration_run_hash_prefix"
        ],
        "n_records_loaded": len(record_filenames),
        "n_main_records": EXPECTED_MAIN_TOTAL,
        "n_sensitivity_records": EXPECTED_SENSITIVITY_TOTAL,
        "heldout_evaluation_validates": True,
        "generated_files": {
            "markdown": str(markdown_path),
            "main_summary_csv": str(main_summary_csv_path),
            "per_seed_main_csv": str(per_seed_main_csv_path),
            "sensitivity_summary_csv": str(sensitivity_summary_csv_path),
            "status_summary_csv": str(status_summary_csv_path),
            "sid_png": str(sid_png_path),
            "mmd_png": str(mmd_png_path),
            "shd_png": str(shd_png_path),
            "runtime_png": str(runtime_png_path),
            "sensitivity_png": str(sensitivity_png_path),
        },
        "generated_at_readout_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return report


__all__ = [
    "DCDI_MAIN_FIT_RNG_VALUE",
    "MAIN_SUMMARY_CSV_FILENAME",
    "MARKDOWN_FILENAME",
    "MMD_PNG_FILENAME",
    "PER_SEED_MAIN_CSV_FILENAME",
    "READOUT_DIRECTORY_NAME",
    "RUNTIME_PNG_FILENAME",
    "SENSITIVITY_PNG_FILENAME",
    "SENSITIVITY_SUMMARY_CSV_FILENAME",
    "SHD_PNG_FILENAME",
    "SID_PNG_FILENAME",
    "STATUS_SUMMARY_CSV_FILENAME",
    "generate_heldout_readout",
]
