"""Held-out evaluation artefact: schema, aggregation, validator, writer.

This module is the calibration-handoff counterpart at the held-out
stage: it accepts the enumerated workload (from
``experiments.selection_study.held_out``) and a sequence of per-fit
records, produces a JSON-safe artefact with cell-level aggregates,
keeps the DCDI fit-RNG sensitivity diagnostic structurally separate
from the main evaluation, validates the artefact against a strict
schema, and writes it atomically.

The module is intentionally side-effect free with respect to model
execution:

- no model fit is invoked;
- ``pipeline.run_single_fit`` is not called;
- no wrapper module is imported.

A separate later commit will wire a held-out orchestrator that
produces the per-fit records this module consumes.

Aggregation semantics
---------------------
Per-cell aggregates are computed from finite values only. For each
metric the artefact records ``mean``, ``std`` (ddof=1 sample standard
deviation when at least two finite values exist, otherwise ``None``),
``median``, ``q1``, ``q3``, ``iqr``, ``min``, ``max``, ``finite_count``,
and ``non_finite_count``. When every value for a metric is non-finite
the aggregate is ``None`` and the cell's ``degenerate_metric_names``
list names the metric.

Sensitivity diagnostic
----------------------
The fit-RNG sensitivity probe is recorded under
``fit_rng_sensitivity_addendum`` and never folded into
``main_evaluation``. Its ``diagnostic_summary`` includes the seed-301
DCDI/centred_only main-evaluation metric values so the operator can
compare ``fit_rng=42`` (main) to ``fit_rng in {43..47}`` (sensitivity)
at the same SCM seed. The addendum carries an explicit
``interpretation_note`` flagging the diagnostic as supplementary
rather than primary evidence.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.selection_study.held_out import (
    HELDOUT_FIT_RNG_SENSITIVITY_REF,
    HELDOUT_SCM_SEEDS,
    HeldoutWorkload,
    MAIN_JOB_KIND,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    compute_heldout_run_hash_full,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    FIT_RNG_POLICY_REF,
    FULL_HASH_LENGTH,
    HASH_PREFIX_LENGTH,
    INTERVENTION_POLICY_REF,
    MODELS,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


HELDOUT_EVALUATION_ARTEFACT_TYPE = "heldout_evaluation"
HELDOUT_EVALUATION_SCHEMA_VERSION = 1

INTERPRETATION_NOTE = (
    "Diagnostic only. The fit-RNG sensitivity probe is a "
    "supplementary local sensitivity estimate at one selected DCDI "
    "configuration and one held-out SCM seed; it is not part of "
    "the main base-model decision evidence."
)

EXPECTED_MAIN_PER_CELL = 5
EXPECTED_MAIN_TOTAL = (
    len(MODELS) * len(CONDITIONS) * EXPECTED_MAIN_PER_CELL
)
EXPECTED_SENSITIVITY_TOTAL = len(SENSITIVITY_FIT_RNGS)
EXPECTED_TOTAL_RECORDS = (
    EXPECTED_MAIN_TOTAL + EXPECTED_SENSITIVITY_TOTAL
)

_GENERATED_AT_UTC_FORMAT_LENGTH = len("YYYY-MM-DDTHH:MM:SSZ")

_HEX_DIGITS = frozenset("0123456789abcdef")
_CALIBRATION_SEED_SET: frozenset[int] = frozenset(CALIBRATION_SEEDS)
_HELDOUT_SEED_SET: frozenset[int] = frozenset(HELDOUT_SCM_SEEDS)
_SENSITIVITY_FIT_RNG_SET: frozenset[int] = frozenset(SENSITIVITY_FIT_RNGS)
_VALID_MODEL_SET: frozenset[str] = frozenset(MODELS)
_VALID_CONDITION_SET: frozenset[str] = frozenset(CONDITIONS)
_VALID_JOB_KIND_SET: frozenset[str] = frozenset(
    {MAIN_JOB_KIND, SENSITIVITY_JOB_KIND}
)

_METRIC_FIELDS: tuple[str, ...] = (
    "sid",
    "mmd_primary",
    "shd",
    "runtime_seconds",
)
_STATUS_FIELDS: tuple[str, ...] = (
    "training_status",
    "graph_status",
    "sampler_status",
)
_OPTIONAL_RECORD_FIELDS: tuple[str, ...] = (
    "n_iterations",
    "run_id",
)
_REQUIRED_FIT_RECORD_FIELDS: tuple[str, ...] = (
    "job_kind",
    "model",
    "condition",
    "configuration_hash_full",
    "configuration_hash_prefix",
    "hyperparameters",
    "scm_seed",
    "fit_rng",
    "sid",
    "shd",
    "mmd_primary",
    "runtime_seconds",
    "graph_status",
    "sampler_status",
    "training_status",
)

# Per-seed record fields preserved in main_evaluation cells.
_PER_SEED_RECORD_FIELDS: tuple[str, ...] = (
    "seed_value",
    "fit_rng",
    "sid",
    "shd",
    "mmd_primary",
    "runtime_seconds",
    "graph_status",
    "sampler_status",
    "training_status",
    "n_iterations",
)

# Per-fit record fields preserved in the sensitivity addendum.
_PER_FIT_RECORD_FIELDS: tuple[str, ...] = (
    "fit_rng",
    "sid",
    "shd",
    "mmd_primary",
    "runtime_seconds",
    "graph_status",
    "sampler_status",
    "training_status",
    "n_iterations",
)

_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "artefact_type",
    "schema_version",
    "parent_calibration_run_hash_full",
    "parent_calibration_run_hash_prefix",
    "heldout_run_hash_full",
    "heldout_run_hash_prefix",
    "selected_configurations_used",
    "main_heldout_seeds",
    "sensitivity_spec",
    "policy_refs",
    "main_evaluation",
    "fit_rng_sensitivity_addendum",
    "status_summary",
    "generated_at_utc",
)
_ALLOWED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    _REQUIRED_TOP_LEVEL_FIELDS
)

# Field names that would record a final base-model decision; rejected
# at every nesting depth by the validator.
_FORBIDDEN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    }
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    """Return True iff ``value`` is a finite int or float (booleans excluded)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _finite_or_none(value: Any) -> float | None:
    """Return ``float(value)`` when finite, else ``None``."""
    if _is_finite_number(value):
        return float(value)
    return None


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return the linear-interpolation quantile of an already-sorted sequence.

    The function matches the NumPy ``linear`` method and the pandas
    default. ``sorted_values`` must be non-empty and pre-sorted in
    ascending order.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError(
            "linear_quantile requires a non-empty sorted sequence"
        )
    if n == 1:
        return float(sorted_values[0])
    position = (n - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + fraction * (upper_value - lower_value)


def _summarise_metric(values: Sequence[Any]) -> dict[str, Any]:
    """Compute the full aggregate metric summary over a value sequence.

    Returns a JSON-safe dict with ``mean``, ``std`` (ddof=1 sample
    standard deviation when at least two finite values are present;
    ``None`` otherwise), ``median``, ``q1``, ``q3``, ``iqr``, ``min``,
    ``max``, ``finite_count``, and ``non_finite_count``. All aggregate
    fields are ``None`` when no finite values are present.
    """
    finite_values: list[float] = [
        float(v) for v in values if _is_finite_number(v)
    ]
    non_finite_count = len(values) - len(finite_values)
    if not finite_values:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
            "finite_count": 0,
            "non_finite_count": non_finite_count,
        }

    sorted_values = sorted(finite_values)
    n = len(sorted_values)
    mean = sum(sorted_values) / n
    if n >= 2:
        variance = sum(
            (value - mean) * (value - mean) for value in sorted_values
        ) / (n - 1)
        std: float | None = math.sqrt(variance)
    else:
        std = None
    median = _linear_quantile(sorted_values, 0.5)
    q1 = _linear_quantile(sorted_values, 0.25)
    q3 = _linear_quantile(sorted_values, 0.75)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "finite_count": n,
        "non_finite_count": non_finite_count,
    }


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def _validate_fit_record(
    record: Mapping[str, Any], *, record_index: int
) -> dict[str, Any]:
    """Validate one input per-fit record and return a plain ``dict`` copy."""
    if not isinstance(record, Mapping):
        raise ValueError(
            f"held-out record at index {record_index} must be a "
            f"Mapping; got {type(record).__name__}"
        )
    missing = [
        name for name in _REQUIRED_FIT_RECORD_FIELDS if name not in record
    ]
    if missing:
        raise ValueError(
            f"held-out record at index {record_index} is missing "
            f"required field(s): {missing}"
        )

    job_kind = record["job_kind"]
    if job_kind not in _VALID_JOB_KIND_SET:
        raise ValueError(
            f"held-out record at index {record_index} has unknown "
            f"job_kind {job_kind!r}; allowed values are "
            f"{sorted(_VALID_JOB_KIND_SET)}"
        )

    model = record["model"]
    if model not in _VALID_MODEL_SET:
        raise ValueError(
            f"held-out record at index {record_index} has unknown "
            f"model {model!r}; allowed values are "
            f"{sorted(_VALID_MODEL_SET)}"
        )

    condition = record["condition"]
    if condition not in _VALID_CONDITION_SET:
        raise ValueError(
            f"held-out record at index {record_index} has unknown "
            f"condition {condition!r}; allowed values are "
            f"{sorted(_VALID_CONDITION_SET)}"
        )

    scm_seed = record["scm_seed"]
    if isinstance(scm_seed, bool) or not isinstance(scm_seed, int):
        raise ValueError(
            f"held-out record at index {record_index} has a non-int "
            f"scm_seed: got {scm_seed!r}"
        )
    if scm_seed in _CALIBRATION_SEED_SET:
        raise ValueError(
            f"held-out record at index {record_index} has calibration "
            f"SCM seed {scm_seed}; calibration seeds "
            f"{sorted(_CALIBRATION_SEED_SET)} must not appear in any "
            "held-out record"
        )

    fit_rng = record["fit_rng"]
    if fit_rng is not None and (
        isinstance(fit_rng, bool) or not isinstance(fit_rng, int)
    ):
        raise ValueError(
            f"held-out record at index {record_index} has a non-int "
            f"fit_rng: got {fit_rng!r}"
        )

    _validate_hex_string(
        record["configuration_hash_full"],
        length=FULL_HASH_LENGTH,
        where=(
            f"held-out record at index {record_index} "
            "configuration_hash_full"
        ),
    )
    prefix = record["configuration_hash_prefix"]
    expected_prefix = record["configuration_hash_full"][:HASH_PREFIX_LENGTH]
    if prefix != expected_prefix:
        raise ValueError(
            f"held-out record at index {record_index} has a "
            "configuration_hash_prefix that does not match the first "
            f"{HASH_PREFIX_LENGTH} characters of "
            f"configuration_hash_full; got prefix={prefix!r}, "
            f"expected={expected_prefix!r}"
        )
    if not isinstance(record["hyperparameters"], Mapping):
        raise ValueError(
            f"held-out record at index {record_index} has "
            "hyperparameters that is not a mapping"
        )
    return dict(record)


def _validate_hex_string(value: object, *, length: int, where: str) -> None:
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a string; got {type(value).__name__}"
        )
    if len(value) != length:
        raise ValueError(
            f"{where} must be a {length}-character lowercase hex "
            f"string; got length {len(value)}"
        )
    for ch in value:
        if ch not in _HEX_DIGITS:
            raise ValueError(
                f"{where} must contain only lowercase hex digits "
                f"0-9 and a-f; got character {ch!r}"
            )


# ---------------------------------------------------------------------------
# Cell aggregation
# ---------------------------------------------------------------------------


def _per_seed_record_for_main(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project an input record into the artefact's per_seed_records shape."""
    return {
        "seed_value": int(record["scm_seed"]),
        "fit_rng": record["fit_rng"],
        "sid": _finite_or_none(record["sid"]),
        "shd": _finite_or_none(record["shd"]),
        "mmd_primary": _finite_or_none(record["mmd_primary"]),
        "runtime_seconds": _finite_or_none(record["runtime_seconds"]),
        "graph_status": str(record["graph_status"]),
        "sampler_status": str(record["sampler_status"]),
        "training_status": str(record["training_status"]),
        "n_iterations": record.get("n_iterations"),
    }


def _per_fit_record_for_sensitivity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an input record into the sensitivity per_fit_records shape."""
    return {
        "fit_rng": int(record["fit_rng"]),
        "sid": _finite_or_none(record["sid"]),
        "shd": _finite_or_none(record["shd"]),
        "mmd_primary": _finite_or_none(record["mmd_primary"]),
        "runtime_seconds": _finite_or_none(record["runtime_seconds"]),
        "graph_status": str(record["graph_status"]),
        "sampler_status": str(record["sampler_status"]),
        "training_status": str(record["training_status"]),
        "n_iterations": record.get("n_iterations"),
    }


def _status_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return per-status counts for a record sequence."""
    counts: dict[str, dict[str, int]] = {}
    for status_field in _STATUS_FIELDS:
        counts[status_field] = dict(
            Counter(str(record[status_field]) for record in records)
        )
    return counts


def _aggregate_cell(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the aggregate_metrics block for a single main-evaluation cell."""
    metrics: dict[str, Any] = {}
    degenerate_metric_names: list[str] = []
    has_non_finite_seed_metric = False
    for metric_name in _METRIC_FIELDS:
        values = [record[metric_name] for record in records]
        summary = _summarise_metric(values)
        metrics[metric_name] = summary
        if summary["non_finite_count"] > 0:
            has_non_finite_seed_metric = True
            degenerate_metric_names.append(metric_name)
    metrics["has_non_finite_seed_metric"] = has_non_finite_seed_metric
    metrics["degenerate_metric_names"] = degenerate_metric_names
    metrics["status_counts"] = _status_counts(records)
    return metrics


def aggregate_main_heldout_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the ``main_evaluation`` block from main per-fit records.

    Parameters
    ----------
    records : sequence of Mapping
        The 20 main per-fit records produced by the held-out runner.
        Records carrying ``job_kind != "main"`` are rejected.

    Returns
    -------
    dict
        A JSON-safe ``main_evaluation`` block of the form
        ``{"cells": {condition: {model: {per_seed_records, aggregate_metrics}}}}``.
    """
    validated = [
        _validate_fit_record(record, record_index=index)
        for index, record in enumerate(records)
    ]
    main_records = [r for r in validated if r["job_kind"] == MAIN_JOB_KIND]
    if len(main_records) != EXPECTED_MAIN_TOTAL:
        raise ValueError(
            "main_evaluation requires exactly "
            f"{EXPECTED_MAIN_TOTAL} main records; got "
            f"{len(main_records)}"
        )

    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in main_records:
        key = (str(record["condition"]), str(record["model"]))
        by_cell.setdefault(key, []).append(record)

    expected_cells = {
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    }
    missing_cells = expected_cells - set(by_cell.keys())
    if missing_cells:
        raise ValueError(
            "main_evaluation is missing (condition, model) cell(s): "
            f"{sorted(missing_cells)}"
        )
    unexpected_cells = set(by_cell.keys()) - expected_cells
    if unexpected_cells:
        raise ValueError(
            "main_evaluation received unexpected (condition, model) "
            f"cell(s): {sorted(unexpected_cells)}"
        )

    cells_output: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        cells_output[condition] = {}
        for model in MODELS:
            cell_records = by_cell[(condition, model)]
            if len(cell_records) != EXPECTED_MAIN_PER_CELL:
                raise ValueError(
                    f"main_evaluation cell ({condition!r}, {model!r}) "
                    f"must contain exactly {EXPECTED_MAIN_PER_CELL} "
                    f"records; got {len(cell_records)}"
                )
            seeds_seen = sorted(int(r["scm_seed"]) for r in cell_records)
            if seeds_seen != sorted(HELDOUT_SCM_SEEDS):
                raise ValueError(
                    f"main_evaluation cell ({condition!r}, {model!r}) "
                    f"must cover SCM seeds {sorted(HELDOUT_SCM_SEEDS)}; "
                    f"got {seeds_seen}"
                )
            sorted_records = sorted(
                cell_records, key=lambda r: int(r["scm_seed"])
            )
            per_seed = [
                _per_seed_record_for_main(record)
                for record in sorted_records
            ]
            aggregate = _aggregate_cell(sorted_records)
            cells_output[condition][model] = {
                "per_seed_records": per_seed,
                "aggregate_metrics": aggregate,
            }
    return {"cells": cells_output}


def aggregate_fit_rng_sensitivity_records(
    records: Sequence[Mapping[str, Any]],
    main_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the ``fit_rng_sensitivity_addendum`` block.

    Parameters
    ----------
    records : sequence of Mapping
        The 5 sensitivity per-fit records. Records with
        ``job_kind != "fit_rng_sensitivity"`` are rejected.
    main_records : sequence of Mapping
        The full main record set. Used to attach the
        ``main_evaluation_*_at_seed_301`` reference values to the
        diagnostic summary.

    Returns
    -------
    dict
        A JSON-safe diagnostic addendum, separate from
        ``main_evaluation``.
    """
    validated = [
        _validate_fit_record(record, record_index=index)
        for index, record in enumerate(records)
    ]
    sensitivity_records = [
        r for r in validated if r["job_kind"] == SENSITIVITY_JOB_KIND
    ]
    if len(sensitivity_records) != EXPECTED_SENSITIVITY_TOTAL:
        raise ValueError(
            "fit_rng_sensitivity_addendum requires exactly "
            f"{EXPECTED_SENSITIVITY_TOTAL} sensitivity records; got "
            f"{len(sensitivity_records)}"
        )

    fit_rngs_seen = sorted(
        int(record["fit_rng"]) for record in sensitivity_records
    )
    if fit_rngs_seen != sorted(SENSITIVITY_FIT_RNGS):
        raise ValueError(
            "fit_rng_sensitivity_addendum fit_rngs must equal "
            f"{sorted(SENSITIVITY_FIT_RNGS)}; got {fit_rngs_seen}"
        )

    for record in sensitivity_records:
        if record["model"] != SENSITIVITY_MODEL:
            raise ValueError(
                "fit_rng_sensitivity_addendum requires model="
                f"{SENSITIVITY_MODEL!r}; got {record['model']!r}"
            )
        if record["condition"] != SENSITIVITY_CONDITION:
            raise ValueError(
                "fit_rng_sensitivity_addendum requires condition="
                f"{SENSITIVITY_CONDITION!r}; got {record['condition']!r}"
            )
        if int(record["scm_seed"]) != SENSITIVITY_SCM_SEED:
            raise ValueError(
                "fit_rng_sensitivity_addendum requires scm_seed="
                f"{SENSITIVITY_SCM_SEED}; got {record['scm_seed']!r}"
            )

    sorted_records = sorted(
        sensitivity_records, key=lambda r: int(r["fit_rng"])
    )
    per_fit = [
        _per_fit_record_for_sensitivity(record)
        for record in sorted_records
    ]

    # The diagnostic summary uses the same per-metric statistical
    # surface as the main cells, but does not enter main_evaluation.
    diagnostic_summary: dict[str, Any] = {}
    degenerate_metric_names: list[str] = []
    has_non_finite_seed_metric = False
    for metric_name in _METRIC_FIELDS:
        values = [record[metric_name] for record in sorted_records]
        summary = _summarise_metric(values)
        diagnostic_summary[metric_name] = summary
        if summary["non_finite_count"] > 0:
            has_non_finite_seed_metric = True
            degenerate_metric_names.append(metric_name)
    diagnostic_summary["has_non_finite_seed_metric"] = (
        has_non_finite_seed_metric
    )
    diagnostic_summary["degenerate_metric_names"] = degenerate_metric_names

    main_reference = _extract_main_seed_301_values(main_records)
    diagnostic_summary["main_evaluation_sid_at_seed_301"] = (
        main_reference["sid"]
    )
    diagnostic_summary["main_evaluation_mmd_primary_at_seed_301"] = (
        main_reference["mmd_primary"]
    )
    diagnostic_summary["main_evaluation_shd_at_seed_301"] = (
        main_reference["shd"]
    )

    target_record = sorted_records[0]
    target_cell = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "configuration_hash_full": str(
            target_record["configuration_hash_full"]
        ),
        "configuration_hash_prefix": str(
            target_record["configuration_hash_prefix"]
        ),
        "hyperparameters": dict(target_record["hyperparameters"]),
    }
    return {
        "target_cell": target_cell,
        "scm_seed": SENSITIVITY_SCM_SEED,
        "fit_rng_values": list(SENSITIVITY_FIT_RNGS),
        "per_fit_records": per_fit,
        "diagnostic_summary": diagnostic_summary,
        "interpretation_note": INTERPRETATION_NOTE,
    }


def _extract_main_seed_301_values(
    main_records: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Return the seed-301 main metrics for dcdi/centred_only.

    The returned dict is always JSON-safe (non-finite values become
    ``None``). When no matching main record exists the function
    raises ``ValueError`` so the caller cannot silently emit an
    addendum without a comparison reference.
    """
    matching: list[Mapping[str, Any]] = []
    for record in main_records:
        if record.get("job_kind") != MAIN_JOB_KIND:
            continue
        if str(record.get("model")) != SENSITIVITY_MODEL:
            continue
        if str(record.get("condition")) != SENSITIVITY_CONDITION:
            continue
        if int(record.get("scm_seed", -1)) != SENSITIVITY_SCM_SEED:
            continue
        matching.append(record)
    if len(matching) != 1:
        raise ValueError(
            "fit_rng_sensitivity_addendum requires exactly one main "
            f"record at model={SENSITIVITY_MODEL!r}, condition="
            f"{SENSITIVITY_CONDITION!r}, scm_seed={SENSITIVITY_SCM_SEED}; "
            f"found {len(matching)} in the supplied main_records "
            "sequence"
        )
    reference = matching[0]
    return {
        "sid": _finite_or_none(reference["sid"]),
        "mmd_primary": _finite_or_none(reference["mmd_primary"]),
        "shd": _finite_or_none(reference["shd"]),
    }


# ---------------------------------------------------------------------------
# Top-level artefact builder
# ---------------------------------------------------------------------------


def _compute_heldout_run_hash_from_workload(
    workload: HeldoutWorkload,
) -> str:
    """Re-compute the held-out run hash from a workload."""
    selected_hashes = sorted(
        record["configuration_hash_full"]
        for record in workload.selected_configurations_used
    )
    sensitivity_spec = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "scm_seed": SENSITIVITY_SCM_SEED,
        "fit_rngs": list(SENSITIVITY_FIT_RNGS),
    }
    return compute_heldout_run_hash_full(
        parent_calibration_run_hash_full=workload.calibration_run_hash_full,
        selected_configuration_hashes_full=selected_hashes,
        main_heldout_seeds=list(HELDOUT_SCM_SEEDS),
        sensitivity_spec=sensitivity_spec,
        selection_rule_id=SELECTION_RULE_ID,
        selection_rule_ref=SELECTION_RULE_REF,
        intervention_policy_ref=INTERVENTION_POLICY_REF,
        fit_rng_policy_ref=FIT_RNG_POLICY_REF,
        heldout_fit_rng_sensitivity_ref=HELDOUT_FIT_RNG_SENSITIVITY_REF,
    )


def _build_status_summary(
    main_records: Sequence[Mapping[str, Any]],
    sensitivity_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the top-level status_summary block."""
    return {
        "total_records": len(main_records) + len(sensitivity_records),
        "main_records_count": len(main_records),
        "sensitivity_records_count": len(sensitivity_records),
        "main_status_counts": _status_counts(main_records),
        "sensitivity_status_counts": _status_counts(sensitivity_records),
    }


def _cross_check_records_against_workload(
    validated_records: Sequence[Mapping[str, Any]],
    workload: HeldoutWorkload,
) -> None:
    """Ensure the workload jobs and supplied records cover the same identities."""
    def _job_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(item["job_kind"]),
            str(item["model"]),
            str(item["condition"]),
            str(item["configuration_hash_full"]),
            int(item["scm_seed"]),
            item["fit_rng"],
        )

    expected_keys = {
        (
            job.job_kind,
            job.model,
            job.condition,
            job.configuration_hash_full,
            int(job.scm_seed),
            job.fit_rng,
        )
        for job in (
            *workload.main_jobs,
            *workload.sensitivity_jobs,
        )
    }
    actual_keys = {_job_key(record) for record in validated_records}

    missing = expected_keys - actual_keys
    if missing:
        raise ValueError(
            "held-out records do not cover every workload job; "
            f"missing {len(missing)} job identity tuple(s): "
            f"{sorted(missing)}"
        )
    extra = actual_keys - expected_keys
    if extra:
        raise ValueError(
            "held-out records contain unexpected job identity "
            f"tuple(s) not present in the workload: {sorted(extra)}"
        )


def build_heldout_evaluation_artefact(
    *,
    workload: HeldoutWorkload,
    records: Sequence[Mapping[str, Any]],
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build the JSON-safe held-out evaluation artefact.

    Parameters
    ----------
    workload : HeldoutWorkload
        Enumerated workload from ``enumerate_heldout_workload``.
    records : sequence of Mapping
        The full 25-record set (20 main + 5 sensitivity).
    generated_at_utc : str
        ``YYYY-MM-DDTHH:MM:SSZ`` formatted timestamp.

    Returns
    -------
    dict
        The artefact ready for validation and atomic write.
    """
    if not isinstance(generated_at_utc, str):
        raise ValueError(
            "generated_at_utc must be a string; got "
            f"{type(generated_at_utc).__name__}"
        )

    validated_records = [
        _validate_fit_record(record, record_index=index)
        for index, record in enumerate(records)
    ]
    if len(validated_records) != EXPECTED_TOTAL_RECORDS:
        raise ValueError(
            "build_heldout_evaluation_artefact requires exactly "
            f"{EXPECTED_TOTAL_RECORDS} records "
            f"({EXPECTED_MAIN_TOTAL} main + "
            f"{EXPECTED_SENSITIVITY_TOTAL} sensitivity); got "
            f"{len(validated_records)}"
        )
    _cross_check_records_against_workload(validated_records, workload)

    main_records = [
        r for r in validated_records if r["job_kind"] == MAIN_JOB_KIND
    ]
    sensitivity_records = [
        r
        for r in validated_records
        if r["job_kind"] == SENSITIVITY_JOB_KIND
    ]

    main_evaluation = aggregate_main_heldout_records(main_records)
    fit_rng_sensitivity_addendum = aggregate_fit_rng_sensitivity_records(
        sensitivity_records, main_records
    )
    status_summary = _build_status_summary(
        main_records, sensitivity_records
    )

    heldout_run_hash_full = _compute_heldout_run_hash_from_workload(
        workload
    )
    heldout_run_hash_prefix = heldout_run_hash_full[:HASH_PREFIX_LENGTH]

    selected_configurations_used = [
        dict(record) for record in workload.selected_configurations_used
    ]

    sensitivity_spec = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "scm_seed": SENSITIVITY_SCM_SEED,
        "fit_rngs": list(SENSITIVITY_FIT_RNGS),
    }

    policy_refs = {
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "heldout_fit_rng_sensitivity_ref": HELDOUT_FIT_RNG_SENSITIVITY_REF,
    }

    artefact = {
        "artefact_type": HELDOUT_EVALUATION_ARTEFACT_TYPE,
        "schema_version": HELDOUT_EVALUATION_SCHEMA_VERSION,
        "parent_calibration_run_hash_full": (
            workload.calibration_run_hash_full
        ),
        "parent_calibration_run_hash_prefix": (
            workload.calibration_run_hash_prefix
        ),
        "heldout_run_hash_full": heldout_run_hash_full,
        "heldout_run_hash_prefix": heldout_run_hash_prefix,
        "selected_configurations_used": selected_configurations_used,
        "main_heldout_seeds": list(HELDOUT_SCM_SEEDS),
        "sensitivity_spec": sensitivity_spec,
        "policy_refs": policy_refs,
        "main_evaluation": main_evaluation,
        "fit_rng_sensitivity_addendum": fit_rng_sensitivity_addendum,
        "status_summary": status_summary,
        "generated_at_utc": generated_at_utc,
    }
    return artefact


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _scan_for_forbidden_field_names(obj: Any, path: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in _FORBIDDEN_FIELD_NAMES:
                raise ValueError(
                    f"forbidden field name {key!r} found at {path}; "
                    "the held-out evaluation artefact must not record "
                    "a final DAGMA-vs-DCDI base-model decision"
                )
            _scan_for_forbidden_field_names(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _scan_for_forbidden_field_names(item, f"{path}[{index}]")


def _assert_json_safe(obj: Any, path: str) -> None:
    """Walk ``obj`` and raise if any non-JSON-safe value is present."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"non-string mapping key at {path}: {key!r}"
                )
            _assert_json_safe(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _assert_json_safe(item, f"{path}[{index}]")
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(
                f"non-finite float {obj!r} at {path}; the artefact "
                "must encode non-finite values as JSON null"
            )
        return
    if obj is None:
        return
    if isinstance(obj, str):
        return
    raise ValueError(
        f"value at {path} has non-JSON-safe type "
        f"{type(obj).__name__}: {obj!r}"
    )


def _validate_generated_at_utc(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(
            "$.generated_at_utc must be a string; got "
            f"{type(value).__name__}"
        )
    if len(value) != _GENERATED_AT_UTC_FORMAT_LENGTH:
        raise ValueError(
            "$.generated_at_utc must be formatted as "
            "YYYY-MM-DDTHH:MM:SSZ; got "
            f"{value!r} (length {len(value)})"
        )
    if value[-1] != "Z":
        raise ValueError(
            "$.generated_at_utc must end with 'Z'; got "
            f"{value!r}"
        )
    expected_separators = {(4, "-"), (7, "-"), (10, "T"), (13, ":"), (16, ":")}
    for index, expected in expected_separators:
        if value[index] != expected:
            raise ValueError(
                "$.generated_at_utc must be formatted as "
                "YYYY-MM-DDTHH:MM:SSZ; offending character at "
                f"position {index} of {value!r}"
            )
    digit_positions = [
        0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18,
    ]
    for index in digit_positions:
        if not value[index].isdigit():
            raise ValueError(
                "$.generated_at_utc must contain digits at the "
                f"expected positions; got non-digit at position "
                f"{index} of {value!r}"
            )


def _validate_main_evaluation(main_evaluation: Any) -> None:
    if not isinstance(main_evaluation, Mapping):
        raise ValueError(
            "$.main_evaluation must be a mapping; got "
            f"{type(main_evaluation).__name__}"
        )
    if "cells" not in main_evaluation:
        raise ValueError(
            "$.main_evaluation is missing required field 'cells'"
        )
    cells = main_evaluation["cells"]
    if not isinstance(cells, Mapping):
        raise ValueError(
            "$.main_evaluation.cells must be a mapping; got "
            f"{type(cells).__name__}"
        )
    for condition in CONDITIONS:
        if condition not in cells:
            raise ValueError(
                "$.main_evaluation.cells is missing condition "
                f"{condition!r}; required conditions are "
                f"{list(CONDITIONS)}"
            )
        per_condition = cells[condition]
        if not isinstance(per_condition, Mapping):
            raise ValueError(
                f"$.main_evaluation.cells[{condition!r}] must be a "
                f"mapping; got {type(per_condition).__name__}"
            )
        for model in MODELS:
            if model not in per_condition:
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}] is "
                    f"missing model {model!r}; required models are "
                    f"{list(MODELS)}"
                )
            cell = per_condition[model]
            if not isinstance(cell, Mapping):
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}]"
                    f"[{model!r}] must be a mapping; got "
                    f"{type(cell).__name__}"
                )
            per_seed = cell.get("per_seed_records")
            if not isinstance(per_seed, list):
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}]"
                    f"[{model!r}].per_seed_records must be a list; "
                    f"got {type(per_seed).__name__}"
                )
            if len(per_seed) != EXPECTED_MAIN_PER_CELL:
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}]"
                    f"[{model!r}].per_seed_records must contain "
                    f"exactly {EXPECTED_MAIN_PER_CELL} records; got "
                    f"{len(per_seed)}"
                )
            # Per-record checks first so a calibration seed or a
            # sensitivity fit_rng leak surfaces with its specific
            # diagnostic rather than the generic cell-seed-set check.
            for record in per_seed:
                seed = int(record["seed_value"])
                if seed in _CALIBRATION_SEED_SET:
                    raise ValueError(
                        "calibration SCM seed "
                        f"{seed} appears in "
                        f"$.main_evaluation.cells[{condition!r}]"
                        f"[{model!r}].per_seed_records"
                    )
                if (
                    condition == SENSITIVITY_CONDITION
                    and model == SENSITIVITY_MODEL
                    and seed == SENSITIVITY_SCM_SEED
                ):
                    fit_rng = record.get("fit_rng")
                    if fit_rng in _SENSITIVITY_FIT_RNG_SET:
                        raise ValueError(
                            "sensitivity fit_rng "
                            f"{fit_rng} found in "
                            "$.main_evaluation.cells["
                            f"{SENSITIVITY_CONDITION!r}]["
                            f"{SENSITIVITY_MODEL!r}].per_seed_records "
                            "at seed 301; sensitivity records must "
                            "live only in the addendum"
                        )
            seeds_seen = sorted(
                int(record["seed_value"]) for record in per_seed
            )
            if seeds_seen != sorted(HELDOUT_SCM_SEEDS):
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}]"
                    f"[{model!r}].per_seed_records must cover SCM "
                    f"seeds {sorted(HELDOUT_SCM_SEEDS)}; got "
                    f"{seeds_seen}"
                )
            aggregate = cell.get("aggregate_metrics")
            if not isinstance(aggregate, Mapping):
                raise ValueError(
                    f"$.main_evaluation.cells[{condition!r}]"
                    f"[{model!r}].aggregate_metrics must be a "
                    f"mapping; got {type(aggregate).__name__}"
                )
            for metric_name in _METRIC_FIELDS:
                if metric_name not in aggregate:
                    raise ValueError(
                        f"$.main_evaluation.cells[{condition!r}]"
                        f"[{model!r}].aggregate_metrics is missing "
                        f"metric {metric_name!r}"
                    )


def _validate_sensitivity_addendum(addendum: Any) -> None:
    if not isinstance(addendum, Mapping):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum must be a mapping; got "
            f"{type(addendum).__name__}"
        )
    for required in (
        "target_cell",
        "scm_seed",
        "fit_rng_values",
        "per_fit_records",
        "diagnostic_summary",
        "interpretation_note",
    ):
        if required not in addendum:
            raise ValueError(
                "$.fit_rng_sensitivity_addendum is missing required "
                f"field {required!r}"
            )

    target_cell = addendum["target_cell"]
    if not isinstance(target_cell, Mapping):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.target_cell must be a "
            f"mapping; got {type(target_cell).__name__}"
        )
    if target_cell.get("model") != SENSITIVITY_MODEL:
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.target_cell.model must "
            f"equal {SENSITIVITY_MODEL!r}; got "
            f"{target_cell.get('model')!r}"
        )
    if target_cell.get("condition") != SENSITIVITY_CONDITION:
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.target_cell.condition "
            f"must equal {SENSITIVITY_CONDITION!r}; got "
            f"{target_cell.get('condition')!r}"
        )

    if addendum["scm_seed"] != SENSITIVITY_SCM_SEED:
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.scm_seed must equal "
            f"{SENSITIVITY_SCM_SEED}; got {addendum['scm_seed']!r}"
        )

    fit_rng_values = addendum["fit_rng_values"]
    if not isinstance(fit_rng_values, list):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.fit_rng_values must be a "
            f"list; got {type(fit_rng_values).__name__}"
        )
    if sorted(fit_rng_values) != sorted(SENSITIVITY_FIT_RNGS):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.fit_rng_values must "
            f"equal {sorted(SENSITIVITY_FIT_RNGS)}; got "
            f"{fit_rng_values!r}"
        )

    per_fit = addendum["per_fit_records"]
    if not isinstance(per_fit, list):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.per_fit_records must be "
            f"a list; got {type(per_fit).__name__}"
        )
    if len(per_fit) != EXPECTED_SENSITIVITY_TOTAL:
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.per_fit_records must "
            f"contain exactly {EXPECTED_SENSITIVITY_TOTAL} records; "
            f"got {len(per_fit)}"
        )
    fit_rngs_in_records = sorted(int(r["fit_rng"]) for r in per_fit)
    if fit_rngs_in_records != sorted(SENSITIVITY_FIT_RNGS):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.per_fit_records fit_rng "
            f"values must equal {sorted(SENSITIVITY_FIT_RNGS)}; got "
            f"{fit_rngs_in_records}"
        )

    diagnostic_summary = addendum["diagnostic_summary"]
    if not isinstance(diagnostic_summary, Mapping):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.diagnostic_summary must "
            "be a mapping"
        )
    for required in (
        "main_evaluation_sid_at_seed_301",
        "main_evaluation_mmd_primary_at_seed_301",
        "main_evaluation_shd_at_seed_301",
    ):
        if required not in diagnostic_summary:
            raise ValueError(
                "$.fit_rng_sensitivity_addendum.diagnostic_summary "
                f"is missing field {required!r}"
            )
    if not isinstance(addendum["interpretation_note"], str):
        raise ValueError(
            "$.fit_rng_sensitivity_addendum.interpretation_note must "
            "be a string"
        )


def validate_heldout_evaluation_artefact(
    artefact: Mapping[str, Any],
) -> None:
    """Validate a held-out evaluation artefact against the strict schema."""
    if not isinstance(artefact, Mapping):
        raise ValueError(
            "held-out evaluation artefact must be a mapping; got "
            f"{type(artefact).__name__}"
        )
    top = dict(artefact)

    # Forbidden-decision scan first so a forbidden key is reported
    # even when other fields are also malformed.
    _scan_for_forbidden_field_names(top, "$")

    unknown = [
        key for key in top if key not in _ALLOWED_TOP_LEVEL_FIELDS
    ]
    if unknown:
        raise ValueError(
            "held-out evaluation artefact contains unknown top-level "
            f"field(s): {sorted(unknown)}; allowed fields are "
            f"{sorted(_ALLOWED_TOP_LEVEL_FIELDS)}"
        )
    missing = [
        name for name in _REQUIRED_TOP_LEVEL_FIELDS if name not in top
    ]
    if missing:
        raise ValueError(
            "held-out evaluation artefact is missing required "
            f"top-level field(s): {missing}"
        )

    if top["artefact_type"] != HELDOUT_EVALUATION_ARTEFACT_TYPE:
        raise ValueError(
            "$.artefact_type must equal "
            f"{HELDOUT_EVALUATION_ARTEFACT_TYPE!r}; got "
            f"{top['artefact_type']!r}"
        )
    if top["schema_version"] != HELDOUT_EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "$.schema_version must equal "
            f"{HELDOUT_EVALUATION_SCHEMA_VERSION}; got "
            f"{top['schema_version']!r}"
        )

    _validate_hex_string(
        top["parent_calibration_run_hash_full"],
        length=FULL_HASH_LENGTH,
        where="$.parent_calibration_run_hash_full",
    )
    expected_parent_prefix = top["parent_calibration_run_hash_full"][
        :HASH_PREFIX_LENGTH
    ]
    if top["parent_calibration_run_hash_prefix"] != expected_parent_prefix:
        raise ValueError(
            "$.parent_calibration_run_hash_prefix must equal the "
            f"first {HASH_PREFIX_LENGTH} characters of "
            "$.parent_calibration_run_hash_full"
        )
    _validate_hex_string(
        top["heldout_run_hash_full"],
        length=FULL_HASH_LENGTH,
        where="$.heldout_run_hash_full",
    )
    expected_heldout_prefix = top["heldout_run_hash_full"][
        :HASH_PREFIX_LENGTH
    ]
    if top["heldout_run_hash_prefix"] != expected_heldout_prefix:
        raise ValueError(
            "$.heldout_run_hash_prefix must equal the first "
            f"{HASH_PREFIX_LENGTH} characters of "
            "$.heldout_run_hash_full"
        )

    main_heldout_seeds = top["main_heldout_seeds"]
    if not isinstance(main_heldout_seeds, list):
        raise ValueError(
            "$.main_heldout_seeds must be a list; got "
            f"{type(main_heldout_seeds).__name__}"
        )
    if sorted(main_heldout_seeds) != sorted(HELDOUT_SCM_SEEDS):
        raise ValueError(
            "$.main_heldout_seeds must equal "
            f"{list(HELDOUT_SCM_SEEDS)}; got {main_heldout_seeds!r}"
        )
    for seed in main_heldout_seeds:
        if seed in _CALIBRATION_SEED_SET:
            raise ValueError(
                f"calibration SCM seed {seed} appears in "
                "$.main_heldout_seeds; calibration seeds must not "
                "appear in the held-out artefact"
            )

    sensitivity_spec = top["sensitivity_spec"]
    if not isinstance(sensitivity_spec, Mapping):
        raise ValueError(
            "$.sensitivity_spec must be a mapping; got "
            f"{type(sensitivity_spec).__name__}"
        )
    for required in ("model", "condition", "scm_seed", "fit_rngs"):
        if required not in sensitivity_spec:
            raise ValueError(
                "$.sensitivity_spec is missing required field "
                f"{required!r}"
            )

    policy_refs = top["policy_refs"]
    if not isinstance(policy_refs, Mapping):
        raise ValueError(
            "$.policy_refs must be a mapping; got "
            f"{type(policy_refs).__name__}"
        )
    for required in (
        "selection_rule_id",
        "selection_rule_ref",
        "intervention_policy_ref",
        "fit_rng_policy_ref",
        "heldout_fit_rng_sensitivity_ref",
    ):
        if required not in policy_refs:
            raise ValueError(
                "$.policy_refs is missing required field "
                f"{required!r}"
            )

    selected = top["selected_configurations_used"]
    if not isinstance(selected, list):
        raise ValueError(
            "$.selected_configurations_used must be a list; got "
            f"{type(selected).__name__}"
        )

    _validate_main_evaluation(top["main_evaluation"])
    _validate_sensitivity_addendum(top["fit_rng_sensitivity_addendum"])

    status_summary = top["status_summary"]
    if not isinstance(status_summary, Mapping):
        raise ValueError(
            "$.status_summary must be a mapping; got "
            f"{type(status_summary).__name__}"
        )

    _validate_generated_at_utc(top["generated_at_utc"])

    # Final whole-artefact JSON-safety walk.
    _assert_json_safe(top, "$")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _atomic_write_json(
    artefact: Mapping[str, Any], output_path: Path
) -> None:
    """Write ``artefact`` to ``output_path`` atomically with read-back validation."""
    parent = output_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    moved = False
    try:
        with os.fdopen(
            fd, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                artefact,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        with tmp_path.open("r", encoding="utf-8") as handle:
            read_back = json.load(handle)
        validate_heldout_evaluation_artefact(read_back)
        os.replace(tmp_path, output_path)
        moved = True
    finally:
        if not moved:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_heldout_evaluation_artefact(
    artefact: Mapping[str, Any],
    path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Validate and atomically write the held-out evaluation artefact.

    Parameters
    ----------
    artefact : Mapping
        The artefact to validate and write.
    path : Path or str
        Destination path. Parent directories are created if missing.
    force : bool, optional
        When ``False`` (the default), an existing file at ``path``
        causes ``FileExistsError`` to be raised before any temporary
        file is written.

    Returns
    -------
    Path
        The ``path`` as a regular file on disk.
    """
    validate_heldout_evaluation_artefact(artefact)
    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(
            "refusing to overwrite existing held-out evaluation "
            f"artefact at {output_path}; pass force=True to allow "
            "overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(artefact, output_path)
    return output_path


__all__ = [
    "EXPECTED_MAIN_PER_CELL",
    "EXPECTED_MAIN_TOTAL",
    "EXPECTED_SENSITIVITY_TOTAL",
    "EXPECTED_TOTAL_RECORDS",
    "HELDOUT_EVALUATION_ARTEFACT_TYPE",
    "HELDOUT_EVALUATION_SCHEMA_VERSION",
    "INTERPRETATION_NOTE",
    "aggregate_fit_rng_sensitivity_records",
    "aggregate_main_heldout_records",
    "build_heldout_evaluation_artefact",
    "validate_heldout_evaluation_artefact",
    "write_heldout_evaluation_artefact",
]
