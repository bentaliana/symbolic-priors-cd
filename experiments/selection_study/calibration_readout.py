"""Calibration readout: audit summary and visualisation from artefacts.

This module reads a calibration run's selected_configurations.json and
the matching per-fit records, validates them structurally and against
the expected calibration identity (40 records, both models, both
conditions, only calibration seeds 201 and 202), and writes a small
reproducible audit bundle:

- calibration_readout.md
- selected_configurations_summary.csv
- candidate_ranking_summary.csv
- status_summary.csv
- calibration_mean_sid.png
- calibration_mean_mmd.png
- calibration_mean_shd.png

The module is read-only with respect to the calibration artefact and
the per-fit records. It does not invoke any model fit, does not modify
the input files, and does not record any final base-model decision:
the calibration handoff selects one configuration per model per
condition, and this readout audits and visualises only those
within-model, within-condition selections.

Public entry point
------------------
``generate_calibration_readout(calibration_run_dir, *, output_dir=None)``
returns a JSON-safe report dictionary describing the audit. Every
generated file is written under ``output_dir`` (default
``<calibration_run_dir>/readout/``); existing files at those paths are
overwritten so the readout is idempotent.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CANDIDATES_PER_CONDITION_PER_MODEL,
    CONDITIONS,
    HASH_PREFIX_LENGTH,
    MODELS,
    SELECTED_CONFIGURATIONS_FILENAME,
    validate_selected_configurations_artefact,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


READOUT_DIRECTORY_NAME = "readout"
RECORDS_DIRECTORY_NAME = "records"

MARKDOWN_FILENAME = "calibration_readout.md"
SELECTED_CSV_FILENAME = "selected_configurations_summary.csv"
RANKING_CSV_FILENAME = "candidate_ranking_summary.csv"
STATUS_CSV_FILENAME = "status_summary.csv"
SID_PNG_FILENAME = "calibration_mean_sid.png"
MMD_PNG_FILENAME = "calibration_mean_mmd.png"
SHD_PNG_FILENAME = "calibration_mean_shd.png"

EXPECTED_RECORD_COUNT = (
    len(CONDITIONS)
    * len(MODELS)
    * CANDIDATES_PER_CONDITION_PER_MODEL
    * len(CALIBRATION_SEEDS)
)
EXPECTED_CANDIDATES_PER_CELL = CANDIDATES_PER_CONDITION_PER_MODEL
EXPECTED_RECORDS_PER_CELL = (
    CANDIDATES_PER_CONDITION_PER_MODEL * len(CALIBRATION_SEEDS)
)

HELD_OUT_SEEDS: tuple[int, ...] = (301, 302, 303, 304, 305)
_HELD_OUT_SEED_SET: frozenset[int] = frozenset(HELD_OUT_SEEDS)
_CALIBRATION_SEED_SET: frozenset[int] = frozenset(CALIBRATION_SEEDS)
_MODEL_SET: frozenset[str] = frozenset(MODELS)
_CONDITION_SET: frozenset[str] = frozenset(CONDITIONS)

INCIDENT_REPORT_RELATIVE_PATH = (
    "docs/08g_file_exists_error_incident.md"
)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Any:
    """Read a UTF-8 JSON file and return the parsed object.

    Raises ``FileNotFoundError`` if the path is missing and
    ``ValueError`` if the file is not valid JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"required calibration input file not found at {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"calibration input file at {path} is not valid JSON: "
                f"{exc}"
            ) from exc


def _load_selected_configurations(
    calibration_run_dir: Path,
) -> dict[str, Any]:
    """Load and validate the selected_configurations.json artefact."""
    artefact_path = calibration_run_dir / SELECTED_CONFIGURATIONS_FILENAME
    artefact = _read_json_file(artefact_path)
    if not isinstance(artefact, dict):
        raise ValueError(
            f"selected_configurations artefact at {artefact_path} "
            "must be a JSON object at the top level; got "
            f"{type(artefact).__name__}"
        )
    validate_selected_configurations_artefact(artefact)
    return artefact


def _load_record_files(
    calibration_run_dir: Path,
) -> list[dict[str, Any]]:
    """Load every per-fit JSON record under ``records/`` and validate identities.

    The returned list is sorted by (condition, model,
    configuration_hash_prefix, seed_value) for deterministic
    downstream processing.
    """
    records_dir = calibration_run_dir / RECORDS_DIRECTORY_NAME
    if not records_dir.is_dir():
        raise FileNotFoundError(
            f"calibration records directory not found at {records_dir}"
        )

    record_paths = sorted(records_dir.glob("*.json"))
    records: list[dict[str, Any]] = []
    for path in record_paths:
        record = _read_json_file(path)
        if not isinstance(record, dict):
            raise ValueError(
                f"calibration record at {path} must be a JSON object "
                f"at the top level; got {type(record).__name__}"
            )
        _validate_record_identity(record, path)
        records.append(record)

    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "calibration records directory must contain exactly "
            f"{EXPECTED_RECORD_COUNT} JSON records "
            f"({len(MODELS)} models x {len(CONDITIONS)} conditions x "
            f"{CANDIDATES_PER_CONDITION_PER_MODEL} candidates x "
            f"{len(CALIBRATION_SEEDS)} seeds); got {len(records)} at "
            f"{records_dir}"
        )

    _assert_no_held_out_seeds(records)
    _assert_seed_coverage(records)
    _assert_model_and_condition_coverage(records)

    records.sort(
        key=lambda r: (
            r["condition"],
            r["model"],
            r["configuration_hash_prefix"],
            r["seed_value"],
        )
    )
    return records


def _validate_record_identity(
    record: Mapping[str, Any], path: Path
) -> None:
    """Validate the minimum identity fields on a per-fit record."""
    for field_name in (
        "model",
        "condition",
        "configuration_hash_full",
        "configuration_hash_prefix",
        "hyperparameters",
        "seed_value",
        "sid",
        "shd",
        "mmd_primary",
        "graph_status",
        "sampler_status",
        "training_status",
    ):
        if field_name not in record:
            raise ValueError(
                f"calibration record at {path} is missing required "
                f"field {field_name!r}"
            )
    model = record["model"]
    if model not in _MODEL_SET:
        raise ValueError(
            f"calibration record at {path} has unknown model "
            f"{model!r}; allowed values are {sorted(_MODEL_SET)}"
        )
    condition = record["condition"]
    if condition not in _CONDITION_SET:
        raise ValueError(
            f"calibration record at {path} has unknown condition "
            f"{condition!r}; allowed values are "
            f"{sorted(_CONDITION_SET)}"
        )
    seed_value = record["seed_value"]
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ValueError(
            f"calibration record at {path} has a non-int seed_value: "
            f"got {seed_value!r}"
        )


def _assert_no_held_out_seeds(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Raise if any record carries a held-out evaluation seed."""
    offenders: list[int] = []
    for record in records:
        if record["seed_value"] in _HELD_OUT_SEED_SET:
            offenders.append(int(record["seed_value"]))
    if offenders:
        raise ValueError(
            "calibration records must not contain held-out evaluation "
            f"seed values; got held-out seed values {sorted(set(offenders))} "
            "in records. Calibration accepts only "
            f"{sorted(_CALIBRATION_SEED_SET)}"
        )


def _assert_seed_coverage(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Raise if calibration records do not cover exactly the expected seeds."""
    seen = sorted({int(record["seed_value"]) for record in records})
    if tuple(seen) != tuple(sorted(_CALIBRATION_SEED_SET)):
        raise ValueError(
            "calibration records must cover exactly the calibration "
            f"seeds {sorted(_CALIBRATION_SEED_SET)}; got {seen}"
        )


def _assert_model_and_condition_coverage(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Raise if calibration records do not cover both models and both conditions."""
    seen_models = {record["model"] for record in records}
    if seen_models != set(MODELS):
        raise ValueError(
            "calibration records must cover both models "
            f"{sorted(_MODEL_SET)}; got {sorted(seen_models)}"
        )
    seen_conditions = {record["condition"] for record in records}
    if seen_conditions != set(CONDITIONS):
        raise ValueError(
            "calibration records must cover both conditions "
            f"{sorted(_CONDITION_SET)}; got {sorted(seen_conditions)}"
        )


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _csv_value(value: Any) -> str:
    """Render a CSV cell value in a stable, JSON-safe form.

    ``None`` becomes the empty string. Booleans become ``"true"`` /
    ``"false"`` so they remain human-readable when the CSV is opened
    in a spreadsheet. Floats are emitted via ``repr`` so the rounded
    form matches Python's reproducible repr.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _hyperparameters_json(hyperparameters: Mapping[str, Any]) -> str:
    """Return a compact, sort-keyed JSON encoding of a hyperparameters dict."""
    return json.dumps(
        dict(hyperparameters),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _hyperparameter_axis_value(
    hyperparameters: Mapping[str, Any]
) -> tuple[str, float]:
    """Return the single hyperparameter name and float value for plotting.

    DAGMA candidates carry ``lambda1``; DCDI candidates carry
    ``reg_coeff``. The function raises ``ValueError`` if the
    hyperparameters dict does not contain exactly one numeric key.
    """
    if len(hyperparameters) != 1:
        raise ValueError(
            "calibration plotting expects exactly one hyperparameter "
            f"per candidate; got {dict(hyperparameters)!r}"
        )
    name, raw_value = next(iter(hyperparameters.items()))
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"calibration plotting expects a numeric hyperparameter "
            f"value for {name!r}; got {raw_value!r}"
        )
    return str(name), float(raw_value)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _build_selected_summary_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build one row per (condition, model) for the selected-configurations CSV."""
    rows: list[dict[str, Any]] = []
    selections = artefact["selections"]
    for condition in CONDITIONS:
        for model in MODELS:
            selection = selections[condition][model]
            metrics = selection["selection_metrics"]
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "selected_hash": selection[
                        "selected_configuration_hash_prefix"
                    ],
                    "hyperparameters_json": _hyperparameters_json(
                        selection["selected_hyperparameters"]
                    ),
                    "mean_sid": metrics.get("mean_sid"),
                    "std_sid": metrics.get("std_sid"),
                    "mean_mmd_primary": metrics.get("mean_mmd_primary"),
                    "std_mmd_primary": metrics.get("std_mmd_primary"),
                    "mean_shd": metrics.get("mean_shd"),
                    "std_shd": metrics.get("std_shd"),
                    "degeneracy_flag": bool(
                        selection.get("degeneracy_flag", False)
                    ),
                    "has_non_finite_seed_metric": bool(
                        metrics.get("has_non_finite_seed_metric", False)
                    ),
                    "ranking_warning": metrics.get("ranking_warning", ""),
                }
            )
    return rows


def _build_candidate_ranking_rows(
    artefact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build one row per ranked candidate for the candidate-ranking CSV."""
    rows: list[dict[str, Any]] = []
    candidate_ranking = artefact["candidate_ranking"]
    selections = artefact["selections"]
    for condition in CONDITIONS:
        for model in MODELS:
            selected_hash_full = selections[condition][model][
                "selected_configuration_hash_full"
            ]
            for candidate in candidate_ranking[condition][model]:
                aggregate = candidate["aggregate_metrics"]
                is_selected = (
                    candidate["configuration_hash_full"]
                    == selected_hash_full
                )
                rows.append(
                    {
                        "condition": condition,
                        "model": model,
                        "rank": int(candidate["rank"]),
                        "configuration_hash_prefix": candidate[
                            "configuration_hash_prefix"
                        ],
                        "hyperparameters_json": _hyperparameters_json(
                            candidate["hyperparameters"]
                        ),
                        "mean_sid": aggregate.get("mean_sid"),
                        "std_sid": aggregate.get("std_sid"),
                        "mean_mmd_primary": aggregate.get("mean_mmd_primary"),
                        "std_mmd_primary": aggregate.get("std_mmd_primary"),
                        "mean_shd": aggregate.get("mean_shd"),
                        "std_shd": aggregate.get("std_shd"),
                        "sid_band_eligible": bool(
                            aggregate.get("sid_band_eligible", False)
                        ),
                        "has_non_finite_seed_metric": bool(
                            aggregate.get(
                                "has_non_finite_seed_metric", False
                            )
                        ),
                        "ranking_warning": aggregate.get(
                            "ranking_warning", ""
                        ),
                        "selected": is_selected,
                    }
                )
    return rows


def _build_status_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-fit records into status counts.

    Each row carries (model, condition, training_status, graph_status,
    sampler_status, count). Rows are sorted deterministically.
    """
    counts: dict[tuple[str, str, str, str, str], int] = {}
    for record in records:
        key = (
            str(record["model"]),
            str(record["condition"]),
            str(record["training_status"]),
            str(record["graph_status"]),
            str(record["sampler_status"]),
        )
        counts[key] = counts.get(key, 0) + 1
    rows: list[dict[str, Any]] = []
    for key in sorted(counts):
        model, condition, training_status, graph_status, sampler_status = key
        rows.append(
            {
                "model": model,
                "condition": condition,
                "training_status": training_status,
                "graph_status": graph_status,
                "sampler_status": sampler_status,
                "count": counts[key],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_csv(
    output_path: Path,
    *,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a list of rows to a CSV file with a stable header."""
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(field_names))
        for row in rows:
            writer.writerow([_csv_value(row.get(name)) for name in field_names])


_SELECTED_CSV_FIELDS: tuple[str, ...] = (
    "condition",
    "model",
    "selected_hash",
    "hyperparameters_json",
    "mean_sid",
    "std_sid",
    "mean_mmd_primary",
    "std_mmd_primary",
    "mean_shd",
    "std_shd",
    "degeneracy_flag",
    "has_non_finite_seed_metric",
    "ranking_warning",
)

_RANKING_CSV_FIELDS: tuple[str, ...] = (
    "condition",
    "model",
    "rank",
    "configuration_hash_prefix",
    "hyperparameters_json",
    "mean_sid",
    "std_sid",
    "mean_mmd_primary",
    "std_mmd_primary",
    "mean_shd",
    "std_shd",
    "sid_band_eligible",
    "has_non_finite_seed_metric",
    "ranking_warning",
    "selected",
)

_STATUS_CSV_FIELDS: tuple[str, ...] = (
    "model",
    "condition",
    "training_status",
    "graph_status",
    "sampler_status",
    "count",
)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


# Metrics whose values are nonnegative by construction. For these the
# error-bar rendering clips the lower whisker so it does not extend
# below zero, since a "mean - std" point would be physically
# meaningless on a nonnegative metric.
_NONNEGATIVE_METRIC_FIELDS: frozenset[str] = frozenset({"sid", "shd"})


def _nonnegative_lower_error(mean: float, std: float) -> float:
    """Return a nonnegative lower-error magnitude for a nonnegative metric.

    For a nonnegative-by-construction metric, the lower whisker of an
    error bar should not cross zero. The returned magnitude is the
    smaller of ``std`` and ``mean``, never less than ``0.0``. When
    ``mean`` is non-finite the function returns ``0.0`` (the data
    point itself will not render and the magnitude is unused). The
    stored ``std`` itself is not changed: only the plotted lower
    whisker is clipped.
    """
    if not math.isfinite(mean):
        return 0.0
    return max(0.0, min(float(std), float(mean)))


def _metric_yerr(
    metric_field: str,
    means: Sequence[float],
    stds: Sequence[float],
) -> list[float] | list[list[float]]:
    """Return the ``yerr`` argument for ``matplotlib.errorbar``.

    For nonnegative-by-construction metrics (SID, SHD) the result is
    a 2-by-N list ``[lower_magnitudes, upper_magnitudes]`` so the
    lower whisker is clipped at zero. For other metrics (MMD) the
    result is a flat list of magnitudes equal to ``stds``, preserving
    the symmetric error-bar behaviour.
    """
    if len(means) != len(stds):
        raise ValueError(
            "means and stds must have the same length; got "
            f"len(means)={len(means)} and len(stds)={len(stds)}"
        )
    if metric_field in _NONNEGATIVE_METRIC_FIELDS:
        lower = [
            _nonnegative_lower_error(float(m), float(s))
            for m, s in zip(means, stds)
        ]
        upper = [float(s) for s in stds]
        return [lower, upper]
    return [float(s) for s in stds]


def _plot_metric_panels(
    *,
    metric_field: str,
    metric_label: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Render a 2x2 grid of small panels for ``metric_field`` across cells.

    The four panels are (condition, model) cells in row-major order:
    ``(centred_only, dagma)``, ``(centred_only, dcdi)``,
    ``(standardised, dagma)``, ``(standardised, dcdi)``. Each panel
    plots the 5 candidates of that cell at their hyperparameter
    values (log-scaled x axis) with vertical error bars representing
    the sample standard deviation. The selected candidate (the rank-1
    row) is marked with a star and an explicit "selected" text
    annotation; no colour dependence is used to indicate selection.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells: list[tuple[str, str]] = [
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 8.0), constrained_layout=True
    )
    axes_flat = list(axes.flat)
    mean_key = f"mean_{metric_field}"
    std_key = f"std_{metric_field}"

    for ax, (condition, model) in zip(axes_flat, cells):
        cell_rows = [
            row
            for row in candidate_rows
            if row["condition"] == condition and row["model"] == model
        ]
        hp_axis_values: list[tuple[str, float]] = []
        for row in cell_rows:
            hp_decoded = json.loads(row["hyperparameters_json"])
            hp_axis_values.append(_hyperparameter_axis_value(hp_decoded))
        if not hp_axis_values:
            hp_name = ""
        else:
            hp_name = hp_axis_values[0][0]
            for name, _ in hp_axis_values:
                if name != hp_name:
                    raise ValueError(
                        "calibration plotting expects a single "
                        "hyperparameter name per (condition, model) "
                        "cell; got names "
                        f"{sorted({n for n, _ in hp_axis_values})} "
                        f"for cell ({condition!r}, {model!r})"
                    )

        paired = sorted(
            zip(hp_axis_values, cell_rows),
            key=lambda item: item[0][1],
        )
        xs = [value for (_, value), _ in paired]
        means: list[float] = []
        stds: list[float] = []
        for _, row in paired:
            mean_value = row.get(mean_key)
            std_value = row.get(std_key)
            means.append(
                float("nan") if mean_value is None else float(mean_value)
            )
            stds.append(0.0 if std_value is None else float(std_value))

        ax.errorbar(
            xs,
            means,
            yerr=_metric_yerr(metric_field, means, stds),
            fmt="o",
            capsize=4,
            linestyle="none",
        )

        for (_, hp_value), row in paired:
            if not row["selected"]:
                continue
            mean_value = row.get(mean_key)
            y = (
                float("nan")
                if mean_value is None
                else float(mean_value)
            )
            ax.plot(
                [hp_value],
                [y],
                marker="*",
                markersize=14,
                markerfacecolor="none",
                markeredgewidth=1.5,
            )
            ax.annotate(
                "selected",
                xy=(hp_value, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

        if xs and all(x > 0 for x in xs):
            ax.set_xscale("log")
        ax.set_xlabel(hp_name)
        ax.set_ylabel(f"mean {metric_label}")
        ax.set_title(f"{condition} / {model}")
        if xs:
            ax.set_xticks(xs)
            ax.set_xticklabels([repr(x) for x in xs], fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.suptitle(f"Calibration mean {metric_label} by candidate")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _format_metric_for_md(value: Any) -> str:
    """Render an aggregate metric value for the markdown table."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _build_observations(
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Build a short factual list of calibration observations."""
    by_cell: dict[tuple[str, str], Mapping[str, Any]] = {
        (row["condition"], row["model"]): row for row in selected_rows
    }

    observations: list[str] = []

    dagma_centred = by_cell.get(("centred_only", "dagma"))
    if dagma_centred is not None:
        hp_decoded = json.loads(dagma_centred["hyperparameters_json"])
        hp_pairs = ", ".join(
            f"{name}={value}" for name, value in sorted(hp_decoded.items())
        )
        observations.append(
            "DAGMA centred_only selected "
            f"{hp_pairs} with mean SID "
            f"{_format_metric_for_md(dagma_centred['mean_sid'])} and "
            f"mean SHD "
            f"{_format_metric_for_md(dagma_centred['mean_shd'])}."
        )

    dagma_std = by_cell.get(("standardised", "dagma"))
    if (
        dagma_centred is not None
        and dagma_std is not None
        and isinstance(dagma_centred.get("mean_sid"), (int, float))
        and isinstance(dagma_std.get("mean_sid"), (int, float))
        and dagma_std["mean_sid"] > dagma_centred["mean_sid"]
    ):
        observations.append(
            "Standardised calibration appears harder than centred_only "
            "for DAGMA: mean SID is "
            f"{_format_metric_for_md(dagma_std['mean_sid'])} "
            "under standardised versus "
            f"{_format_metric_for_md(dagma_centred['mean_sid'])} "
            "under centred_only."
        )

    dcdi_centred = by_cell.get(("centred_only", "dcdi"))
    if (
        dagma_centred is not None
        and dcdi_centred is not None
        and isinstance(dagma_centred.get("mean_sid"), (int, float))
        and isinstance(dcdi_centred.get("mean_sid"), (int, float))
        and dcdi_centred["mean_sid"] > dagma_centred["mean_sid"]
    ):
        observations.append(
            "DCDI calibration SID is materially higher than DAGMA in "
            "centred_only: "
            f"{_format_metric_for_md(dcdi_centred['mean_sid'])} versus "
            f"{_format_metric_for_md(dagma_centred['mean_sid'])}."
        )

    return observations


def _format_md_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    """Build a small GitHub-flavoured markdown table."""
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_markdown(
    *,
    output_path: Path,
    artefact: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    status_rows: Sequence[Mapping[str, Any]],
    n_records_loaded: int,
    any_selected_degenerate: bool,
    generated_filenames: Mapping[str, str],
) -> None:
    """Write the calibration_readout.md file."""
    cal_hash_prefix = artefact["calibration_run_hash_prefix"]
    generated_at = artefact.get("generated_at_utc", "unknown")

    selected_table = _format_md_table(
        headers=(
            "condition",
            "model",
            "selected hash",
            "hyperparameters",
            "mean SID",
            "mean MMD",
            "mean SHD",
            "degeneracy",
        ),
        rows=[
            [
                row["condition"],
                row["model"],
                row["selected_hash"],
                row["hyperparameters_json"],
                _format_metric_for_md(row["mean_sid"]),
                _format_metric_for_md(row["mean_mmd_primary"]),
                _format_metric_for_md(row["mean_shd"]),
                "true" if row["degeneracy_flag"] else "false",
            ]
            for row in selected_rows
        ],
    )

    ranking_table = _format_md_table(
        headers=(
            "condition",
            "model",
            "rank",
            "hash",
            "hyperparameters",
            "mean SID",
            "mean MMD",
            "mean SHD",
            "selected",
        ),
        rows=[
            [
                row["condition"],
                row["model"],
                str(row["rank"]),
                row["configuration_hash_prefix"],
                row["hyperparameters_json"],
                _format_metric_for_md(row["mean_sid"]),
                _format_metric_for_md(row["mean_mmd_primary"]),
                _format_metric_for_md(row["mean_shd"]),
                "true" if row["selected"] else "false",
            ]
            for row in ranking_rows
        ],
    )

    status_table = _format_md_table(
        headers=(
            "model",
            "condition",
            "training_status",
            "graph_status",
            "sampler_status",
            "count",
        ),
        rows=[
            [
                row["model"],
                row["condition"],
                row["training_status"],
                row["graph_status"],
                row["sampler_status"],
                str(row["count"]),
            ]
            for row in status_rows
        ],
    )

    observations = _build_observations(selected_rows)
    observations_section = (
        "\n".join(f"- {line}" for line in observations)
        if observations
        else "- No automatically derived observations."
    )

    if any_selected_degenerate:
        status_text = (
            "At least one selected configuration carries "
            "degeneracy_flag=true. The selection is recorded as-is "
            "and is flagged here for human inspection."
        )
    else:
        status_text = (
            "All four selected configurations carry "
            "degeneracy_flag=false. The artefact validates "
            "structurally and the 40 per-fit records load without "
            "identity errors."
        )

    lines = [
        "# Calibration readout",
        "",
        f"calibration_run_hash_prefix: {cal_hash_prefix}",
        f"generated_at_utc (artefact): {generated_at}",
        f"records loaded: {n_records_loaded}",
        "",
        "## Status",
        "",
        status_text,
        "",
        "This readout uses the selected configurations strictly as "
        "the within-model, within-condition calibration selections "
        "they represent. No base-model choice is made by this "
        "readout; that step belongs to held-out evaluation.",
        "",
        "## Scope",
        "",
        "This file audits one calibration run identified by the "
        "calibration_run_hash above. It loads the 40 per-fit records "
        "from the records directory, the rank-1 configuration per "
        "(condition, model), and the full 5-candidate ranking per "
        "(condition, model). It writes a markdown summary, three CSV "
        "tables, and three PNG figures into the readout directory. "
        "No model fits are run, no input file is modified, and no "
        "final base-model selection is made.",
        "",
        "## Selected configurations",
        "",
        selected_table,
        "",
        "## Candidate ranking summary",
        "",
        ranking_table,
        "",
        "## Status/failure summary",
        "",
        status_table,
        "",
        "## Calibration observations",
        "",
        observations_section,
        "",
        "## Incident note",
        "",
        "A FileExistsError incident affected the dagma / centred_only "
        "/ seed 201 fit. The stale per-run directory was a residue "
        "from an earlier interrupted attempt and produced one "
        "degenerate per-fit record. The incident was repaired before "
        "this readout was generated; this readout inspects the "
        "post-repair selected_configurations.json. The audit trail "
        f"for the incident lives at {INCIDENT_REPORT_RELATIVE_PATH}.",
        "",
        "## Standard-deviation note",
        "",
        "The std fields in the tables and the error bars in the "
        "figures are sample standard deviations computed from n=2 "
        "calibration seeds with ddof=1. They are range/variation "
        "indicators on a two-element sample, not strong uncertainty "
        "estimates. Held-out evaluation will use 5 seeds, which will "
        "provide more informative variability estimates.",
        "",
        "## Generated files",
        "",
        f"- {generated_filenames['markdown']}",
        f"- {generated_filenames['selected_csv']}",
        f"- {generated_filenames['ranking_csv']}",
        f"- {generated_filenames['status_csv']}",
        f"- {generated_filenames['sid_png']}",
        f"- {generated_filenames['mmd_png']}",
        f"- {generated_filenames['shd_png']}",
        "",
        "## Reproducibility note",
        "",
        "- generator: experiments/selection_study/calibration_readout.py",
        "- inputs: selected_configurations.json and records/*.json "
        "under the calibration run directory above",
        "- outputs: the CSV summaries and PNG figures listed above, "
        "plus this markdown report",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_calibration_readout(
    calibration_run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate the calibration readout bundle from a calibration run.

    Parameters
    ----------
    calibration_run_dir : Path or str
        Path to the calibration run directory. The directory must
        contain ``selected_configurations.json`` at its root and a
        ``records/`` subdirectory with the 40 per-fit JSON records.
    output_dir : Path or str, optional
        Directory to write the readout files into. When ``None`` (the
        default), files are written under
        ``<calibration_run_dir>/readout/``. The directory is created
        if it does not exist. Existing readout files at the
        per-filename paths are overwritten so the readout is
        idempotent.

    Returns
    -------
    dict
        JSON-safe report with the following keys:

        - ``calibration_run_dir``: absolute string path to the input
          run directory;
        - ``output_dir``: absolute string path to the readout
          directory that was written into;
        - ``calibration_run_hash_prefix``: the 12-character prefix
          carried by the artefact;
        - ``n_records_loaded``: integer record count (always 40 on
          success);
        - ``selected_configurations_validates``: ``True`` once the
          artefact has passed structural validation;
        - ``selected_degeneracy_flags``: nested dict
          ``[condition][model] -> bool`` of the per-cell selection
          degeneracy flags;
        - ``any_selected_degenerate``: ``True`` iff any
          ``selected_degeneracy_flags`` entry is ``True``;
        - ``generated_files``: dict mapping logical file role to the
          absolute string path that was written.

    Raises
    ------
    FileNotFoundError
        If the calibration run directory, the artefact JSON, or the
        records directory is missing.
    ValueError
        If the artefact fails schema validation, if the records
        directory does not contain exactly 40 JSON records, if the
        records do not cover both models, both conditions, and only
        the calibration seeds 201 and 202, or if any held-out
        evaluation seed value is found in the records.
    """
    run_dir = Path(calibration_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"calibration run directory not found at {run_dir}"
        )

    artefact = _load_selected_configurations(run_dir)
    records = _load_record_files(run_dir)

    if output_dir is None:
        readout_dir = run_dir / READOUT_DIRECTORY_NAME
    else:
        readout_dir = Path(output_dir).resolve()
    readout_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = _build_selected_summary_rows(artefact)
    ranking_rows = _build_candidate_ranking_rows(artefact)
    status_rows = _build_status_rows(records)

    selected_csv_path = readout_dir / SELECTED_CSV_FILENAME
    ranking_csv_path = readout_dir / RANKING_CSV_FILENAME
    status_csv_path = readout_dir / STATUS_CSV_FILENAME
    sid_png_path = readout_dir / SID_PNG_FILENAME
    mmd_png_path = readout_dir / MMD_PNG_FILENAME
    shd_png_path = readout_dir / SHD_PNG_FILENAME
    markdown_path = readout_dir / MARKDOWN_FILENAME

    _write_csv(
        selected_csv_path,
        field_names=_SELECTED_CSV_FIELDS,
        rows=selected_rows,
    )
    _write_csv(
        ranking_csv_path,
        field_names=_RANKING_CSV_FIELDS,
        rows=ranking_rows,
    )
    _write_csv(
        status_csv_path,
        field_names=_STATUS_CSV_FIELDS,
        rows=status_rows,
    )

    _plot_metric_panels(
        metric_field="sid",
        metric_label="SID",
        candidate_rows=ranking_rows,
        output_path=sid_png_path,
    )
    _plot_metric_panels(
        metric_field="mmd_primary",
        metric_label="MMD primary",
        candidate_rows=ranking_rows,
        output_path=mmd_png_path,
    )
    _plot_metric_panels(
        metric_field="shd",
        metric_label="SHD",
        candidate_rows=ranking_rows,
        output_path=shd_png_path,
    )

    selected_degeneracy_flags: dict[str, dict[str, bool]] = {
        condition: {} for condition in CONDITIONS
    }
    for row in selected_rows:
        selected_degeneracy_flags[row["condition"]][row["model"]] = bool(
            row["degeneracy_flag"]
        )
    any_selected_degenerate = any(
        flag
        for per_condition in selected_degeneracy_flags.values()
        for flag in per_condition.values()
    )

    generated_filenames = {
        "markdown": MARKDOWN_FILENAME,
        "selected_csv": SELECTED_CSV_FILENAME,
        "ranking_csv": RANKING_CSV_FILENAME,
        "status_csv": STATUS_CSV_FILENAME,
        "sid_png": SID_PNG_FILENAME,
        "mmd_png": MMD_PNG_FILENAME,
        "shd_png": SHD_PNG_FILENAME,
    }

    _write_markdown(
        output_path=markdown_path,
        artefact=artefact,
        selected_rows=selected_rows,
        ranking_rows=ranking_rows,
        status_rows=status_rows,
        n_records_loaded=len(records),
        any_selected_degenerate=any_selected_degenerate,
        generated_filenames=generated_filenames,
    )

    report = {
        "calibration_run_dir": str(run_dir),
        "output_dir": str(readout_dir),
        "calibration_run_hash_prefix": artefact["calibration_run_hash_prefix"],
        "n_records_loaded": len(records),
        "selected_configurations_validates": True,
        "selected_degeneracy_flags": selected_degeneracy_flags,
        "any_selected_degenerate": any_selected_degenerate,
        "generated_files": {
            "markdown": str(markdown_path),
            "selected_csv": str(selected_csv_path),
            "ranking_csv": str(ranking_csv_path),
            "status_csv": str(status_csv_path),
            "sid_png": str(sid_png_path),
            "mmd_png": str(mmd_png_path),
            "shd_png": str(shd_png_path),
        },
        "generated_at_readout_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return report


__all__ = [
    "EXPECTED_RECORD_COUNT",
    "HELD_OUT_SEEDS",
    "MARKDOWN_FILENAME",
    "MMD_PNG_FILENAME",
    "RANKING_CSV_FILENAME",
    "READOUT_DIRECTORY_NAME",
    "RECORDS_DIRECTORY_NAME",
    "SELECTED_CSV_FILENAME",
    "SHD_PNG_FILENAME",
    "SID_PNG_FILENAME",
    "STATUS_CSV_FILENAME",
    "generate_calibration_readout",
]
