"""Pure within-model calibration ranking.

This module ranks calibration records within each (model, condition)
cell using a deterministic lexicographic rule. The output is the
nested ``candidate_ranking[condition][model]`` and
``selections[condition][model]`` structure that the selected-
configurations artefact writer consumes.

The module is intentionally pure: it does not invoke any model fit,
does not touch the filesystem, does not import any wrapper, and does
not orchestrate calibration runs. It produces no field naming a final
DAGMA-vs-DCDI winner; the ranking is strictly within each
(model, condition) cell.

Ranking algorithm
-----------------
For each (model, condition) cell of 5 candidates x 2 calibration
seeds, the ranker:

1. Aggregates per-seed metrics over finite values only. Mean is the
   arithmetic mean of finite values; standard deviation is the
   ddof=1 sample standard deviation when at least two finite values
   are present, otherwise ``None``. If no finite values are present
   the mean and std are ``None``. ``NaN``, ``None``, ``+inf``, and
   ``-inf`` are non-finite; ``-inf`` is never treated as a winning
   low value.

2. Computes the 10 percent SID band. The reference is
   ``best_finite_mean_sid`` over the cell (the minimum finite
   ``mean_sid``). A candidate is inside the band iff its
   ``mean_sid`` is finite and
   ``candidate_mean_sid <= best_finite_mean_sid * 1.10``. The
   boundary is inclusive. When ``best_finite_mean_sid == 0`` only
   candidates with ``mean_sid == 0`` are inside the band.

3. Partitions the 5 candidates into three layers:

   - Layer 0: finite SID and inside the band. Sorted ascending by
     ``(mean_mmd_primary, mean_shd, configuration_hash_full)`` with
     finite values ranking strictly better than non-finite at each
     metric step.
   - Layer 1: finite SID and outside the band. Sorted ascending by
     ``(mean_sid, mean_shd, configuration_hash_full)``.
   - Layer 2: non-finite ``mean_sid``. Sorted ascending by
     ``(mean_mmd_primary, mean_shd, configuration_hash_full)``.

4. Concatenates the layers in order (0 then 1 then 2). The output
   is always a complete 5-candidate ranking. If every candidate is
   non-finite on ``mean_sid`` (all in Layer 2), the chain falls
   through to MMD; if every candidate is also non-finite on
   ``mean_mmd_primary``, the chain falls through to SHD; if every
   candidate is also non-finite on ``mean_shd``, ranking is decided
   by full ``configuration_hash`` alone.

Public functions
----------------
- ``rank_condition_model_cell(records)``: rank a single cell from
  exactly 10 records.
- ``rank_calibration_records(records)``: rank the full 40-record
  calibration set across all four (model, condition) cells; returns
  the nested ``{"candidate_ranking": ..., "selections": ...}``
  output dict.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_VALID_MODELS: frozenset[str] = frozenset({"dagma", "dcdi"})
_VALID_CONDITIONS: frozenset[str] = frozenset({"centred_only", "standardised"})
_VALID_CALIBRATION_SEEDS: frozenset[int] = frozenset({201, 202})
_EXPECTED_CALIBRATION_SEEDS: tuple[int, ...] = (201, 202)

_EXPECTED_CANDIDATES_PER_CELL = 5
_EXPECTED_SEEDS_PER_CANDIDATE = 2
_EXPECTED_RECORDS_PER_CELL = (
    _EXPECTED_CANDIDATES_PER_CELL * _EXPECTED_SEEDS_PER_CANDIDATE
)
_EXPECTED_FULL_SET_RECORDS = 4 * _EXPECTED_RECORDS_PER_CELL

_HASH_PREFIX_LENGTH = 12
_FULL_HASH_LENGTH = 64
_HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdef")

_SID_BAND_THRESHOLD_FACTOR: float = 1.10

# Required fields on every input calibration record. The ordering
# here is documentation-only; the validator reports missing names as
# a list.
_REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "model",
    "condition",
    "configuration_hash_full",
    "configuration_hash_prefix",
    "hyperparameters",
    "seed_value",
    "shd",
    "sid",
    "mmd_primary",
    "graph_status",
    "sampler_status",
    "training_status",
    "runtime_seconds",
    "n_iterations",
    "threshold_metrics",
    "mmd_by_intervention",
    "bandwidth_summaries",
    "run_id",
)

# Per-seed fields preserved in the artefact's per_seed_metrics list,
# in the order the writer's schema expects.
_PER_SEED_FIELDS: tuple[str, ...] = (
    "seed_value",
    "shd",
    "sid",
    "mmd_primary",
    "graph_status",
    "sampler_status",
    "training_status",
    "runtime_seconds",
    "n_iterations",
    "threshold_metrics",
    "mmd_by_intervention",
    "bandwidth_summaries",
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _is_finite(value: object) -> bool:
    """Return True iff ``value`` is a finite int or float.

    ``None``, ``NaN``, ``+inf``, and ``-inf`` are non-finite. Booleans
    are not treated as numbers (Python's ``bool`` is technically a
    subclass of ``int`` but is rejected here so a bool seed-level
    metric does not silently rank as ``0`` or ``1``).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _finite_or_none(value: object) -> float | None:
    """Return ``float(value)`` if ``value`` is finite, else ``None``.

    Used to produce JSON-safe aggregate metrics: the artefact must
    not contain raw ``NaN``, ``+inf``, or ``-inf`` values.
    """
    if _is_finite(value):
        return float(value)
    return None


def _compute_mean_std(values: Sequence[object]) -> tuple[float | None, float | None]:
    """Return ``(mean, std)`` over the finite entries of ``values``.

    Mean is the arithmetic mean of finite values. Standard deviation
    uses ddof=1 (sample standard deviation). The function returns
    ``(None, None)`` when no finite values are present, and
    ``(mean, None)`` when only one finite value is present.
    """
    finite_values: list[float] = [
        float(v) for v in values if _is_finite(v)
    ]
    if not finite_values:
        return (None, None)
    mean = sum(finite_values) / len(finite_values)
    if len(finite_values) < 2:
        return (mean, None)
    variance = sum(
        (v - mean) * (v - mean) for v in finite_values
    ) / (len(finite_values) - 1)
    return (mean, math.sqrt(variance))


def _sanitise_for_json(obj: Any) -> Any:
    """Return a copy of ``obj`` with every non-finite float replaced by None.

    Walks ``dict`` and ``list`` structures recursively. Non-finite
    floats (``NaN``, ``+inf``, ``-inf``) are replaced by ``None`` in
    the returned structure so the artefact-facing output never
    carries a JSON-hostile float. Booleans are preserved unchanged
    (``bool`` is a subclass of ``int`` but is not a float in this
    context). Integers, strings, and ``None`` pass through unchanged.

    The sanitisation is applied after the degeneracy flags have
    already been computed from the original per-seed inputs, so
    flagging information about non-finite inputs is preserved.
    """
    if isinstance(obj, dict):
        return {key: _sanitise_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitise_for_json(item) for item in obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _metric_sort_key(value: object) -> tuple[int, float]:
    """Return a sort key where finite values rank strictly below non-finite.

    The first slot is 0 for finite values and 1 for non-finite. The
    second slot carries the numerical value for finite entries
    (allowing ascending comparison among finite values) and is
    constant ``0.0`` for non-finite entries (so non-finite values
    tie at the second slot and fall through to the next metric).
    """
    if _is_finite(value):
        return (0, float(value))
    return (1, 0.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_full_hash(value: object, *, where: str) -> str:
    """Return ``value`` if it is a 64-char lowercase hex string, else raise."""
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a string; got {type(value).__name__}"
        )
    if len(value) != _FULL_HASH_LENGTH:
        raise ValueError(
            f"{where} must be a {_FULL_HASH_LENGTH}-character "
            f"lowercase hex string; got length {len(value)}"
        )
    for ch in value:
        if ch not in _HEX_DIGITS:
            raise ValueError(
                f"{where} must contain only lowercase hex digits "
                f"0-9 and a-f; got character {ch!r}"
            )
    return value


def _validate_input_record(
    record: Mapping[str, Any], *, record_index: int
) -> dict[str, Any]:
    """Validate one calibration input record and return a normalised dict.

    Raises ``ValueError`` with a record-index-qualified message if
    any required field is missing or malformed. The function does
    not mutate the input mapping.
    """
    if not isinstance(record, Mapping):
        raise ValueError(
            f"calibration record at index {record_index} must be a "
            f"Mapping; got {type(record).__name__}"
        )
    missing = [
        name for name in _REQUIRED_RECORD_FIELDS if name not in record
    ]
    if missing:
        raise ValueError(
            f"calibration record at index {record_index} is missing "
            f"required field(s): {missing}"
        )

    model = record["model"]
    if model not in _VALID_MODELS:
        raise ValueError(
            f"calibration record at index {record_index} has unknown "
            f"model {model!r}; allowed values are "
            f"{sorted(_VALID_MODELS)}"
        )

    condition = record["condition"]
    if condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"calibration record at index {record_index} has unknown "
            f"condition {condition!r}; allowed values are "
            f"{sorted(_VALID_CONDITIONS)}"
        )

    seed_value = record["seed_value"]
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ValueError(
            f"calibration record at index {record_index} has a non-int "
            f"seed_value: got {seed_value!r}"
        )
    if seed_value not in _VALID_CALIBRATION_SEEDS:
        raise ValueError(
            f"calibration record at index {record_index} has "
            f"seed_value={seed_value}; calibration ranker accepts only "
            f"{sorted(_VALID_CALIBRATION_SEEDS)} (held-out evaluation "
            "seeds must not appear in calibration ranking input)"
        )

    config_hash_full = _validate_full_hash(
        record["configuration_hash_full"],
        where=(
            f"calibration record at index {record_index} "
            "configuration_hash_full"
        ),
    )

    config_hash_prefix = record["configuration_hash_prefix"]
    expected_prefix = config_hash_full[:_HASH_PREFIX_LENGTH]
    if config_hash_prefix != expected_prefix:
        raise ValueError(
            f"calibration record at index {record_index} has a "
            "configuration_hash_prefix that does not match the first "
            f"{_HASH_PREFIX_LENGTH} characters of "
            f"configuration_hash_full; prefix={config_hash_prefix!r}, "
            f"expected={expected_prefix!r}"
        )

    hyperparameters = record["hyperparameters"]
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(
            f"calibration record at index {record_index} has "
            "hyperparameters that is not a mapping; got "
            f"{type(hyperparameters).__name__}"
        )

    threshold_metrics = record["threshold_metrics"]
    if not isinstance(threshold_metrics, list):
        raise ValueError(
            f"calibration record at index {record_index} has "
            "threshold_metrics that is not a list; got "
            f"{type(threshold_metrics).__name__}"
        )

    mmd_by_intervention = record["mmd_by_intervention"]
    if not isinstance(mmd_by_intervention, list):
        raise ValueError(
            f"calibration record at index {record_index} has "
            "mmd_by_intervention that is not a list; got "
            f"{type(mmd_by_intervention).__name__}"
        )

    run_id = record["run_id"]
    if not isinstance(run_id, str):
        raise ValueError(
            f"calibration record at index {record_index} has a "
            f"non-string run_id: got {type(run_id).__name__}"
        )

    return dict(record)


# ---------------------------------------------------------------------------
# Aggregation, band-eligibility, ranking
# ---------------------------------------------------------------------------


@dataclass
class _RankedCandidate:
    """Internal record for one candidate during ranking.

    Holds the candidate identity, the preserved per-seed records,
    the aggregate metrics, the diagnostic flags, and (after sorting)
    the rank position. Defined as a non-frozen dataclass so the
    ``rank`` field can be assigned after the sort.
    """

    model: str
    condition: str
    configuration_hash_full: str
    configuration_hash_prefix: str
    hyperparameters: dict[str, Any]
    per_seed_records: list[dict[str, Any]]
    source_run_ids: list[str]
    aggregate_metrics: dict[str, Any]
    has_non_finite_seed_metric: bool
    degenerate_metric_names: list[str]
    ranking_warning: str
    sid_band_eligible: bool
    sid_band_reference: float | None
    sid_band_threshold: float | None
    rank: int = 0


def _extract_per_seed_record(input_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied per-seed record containing only the preserved fields.

    The per_seed_metrics list in the artefact carries the 12 fields
    declared by the artefact schema, in the same shape as the input
    records. Candidate-level identity fields (``model``, ``condition``,
    ``configuration_hash_full``, etc.) are not preserved at the
    per-seed level.
    """
    extracted: dict[str, Any] = {}
    for field_name in _PER_SEED_FIELDS:
        extracted[field_name] = copy.deepcopy(input_record[field_name])
    return extracted


def _build_aggregate_metrics(
    per_seed_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool, list[str], str]:
    """Compute aggregate metrics and diagnostic flags from per-seed records.

    Returns a 4-tuple:

    - ``aggregate_metrics``: dict of statistical and diagnostic fields,
      JSON-safe (no raw ``NaN`` / ``inf`` / ``-inf``);
    - ``has_non_finite_seed_metric``: bool, ``True`` if any per-seed
      value for ``sid``, ``mmd_primary``, or ``shd`` was non-finite;
    - ``degenerate_metric_names``: list of aggregate-metric names whose
      per-seed inputs were not all finite, drawn from
      ``{"mean_sid", "mean_mmd_primary", "mean_shd"}``;
    - ``ranking_warning``: short human-readable description of the
      degeneracy, or an empty string if none.
    """
    sid_values = [record["sid"] for record in per_seed_records]
    mmd_values = [record["mmd_primary"] for record in per_seed_records]
    shd_values = [record["shd"] for record in per_seed_records]

    mean_sid, std_sid = _compute_mean_std(sid_values)
    mean_mmd, std_mmd = _compute_mean_std(mmd_values)
    mean_shd, std_shd = _compute_mean_std(shd_values)

    degenerate_metric_names: list[str] = []
    for seed_values, aggregate_name in (
        (sid_values, "mean_sid"),
        (mmd_values, "mean_mmd_primary"),
        (shd_values, "mean_shd"),
    ):
        if any(not _is_finite(v) for v in seed_values):
            degenerate_metric_names.append(aggregate_name)

    has_non_finite_seed_metric = bool(degenerate_metric_names)

    if has_non_finite_seed_metric:
        ranking_warning = (
            "one or more per-seed ranking metrics were non-finite; "
            "affected aggregates: "
            f"{', '.join(degenerate_metric_names)}"
        )
    else:
        ranking_warning = ""

    aggregate_metrics: dict[str, Any] = {
        "mean_sid": _finite_or_none(mean_sid),
        "mean_mmd_primary": _finite_or_none(mean_mmd),
        "mean_shd": _finite_or_none(mean_shd),
        "std_sid": _finite_or_none(std_sid),
        "std_mmd_primary": _finite_or_none(std_mmd),
        "std_shd": _finite_or_none(std_shd),
        "has_non_finite_seed_metric": has_non_finite_seed_metric,
        "degenerate_metric_names": list(degenerate_metric_names),
        "ranking_warning": ranking_warning,
    }
    return (
        aggregate_metrics,
        has_non_finite_seed_metric,
        degenerate_metric_names,
        ranking_warning,
    )


def _candidate_in_sid_band(
    candidate_mean_sid: object,
    *,
    best_finite_mean_sid: float | None,
) -> bool:
    """Return True iff the candidate is inside the 10 percent SID band.

    The band exists only when the cell has at least one finite
    ``mean_sid`` (otherwise ``best_finite_mean_sid`` is ``None`` and
    no candidate is inside the band). Inside-band membership requires
    a finite ``candidate_mean_sid`` and the inclusive predicate
    ``candidate_mean_sid <= best_finite_mean_sid * 1.10``. When
    ``best_finite_mean_sid == 0`` only candidates with
    ``candidate_mean_sid == 0`` are inside the band.
    """
    if best_finite_mean_sid is None:
        return False
    if not _is_finite(candidate_mean_sid):
        return False
    candidate_value = float(candidate_mean_sid)
    if best_finite_mean_sid == 0.0:
        return candidate_value == 0.0
    return candidate_value <= best_finite_mean_sid * _SID_BAND_THRESHOLD_FACTOR


def _candidate_layer(candidate: _RankedCandidate) -> int:
    """Return the layer index used to order the candidate during sorting.

    Layer 0 (in-band finite SID) ranks before Layer 1 (out-of-band
    finite SID), which ranks before Layer 2 (non-finite mean SID).
    The layer indices guarantee that non-finite SID candidates sort
    below finite-SID candidates unless every candidate in the cell
    is non-finite on SID.
    """
    mean_sid = candidate.aggregate_metrics.get("mean_sid")
    if not _is_finite(mean_sid):
        return 2
    if candidate.sid_band_eligible:
        return 0
    return 1


def _candidate_sort_key(candidate: _RankedCandidate) -> tuple[Any, ...]:
    """Return the lexicographic sort key for a candidate.

    The sort key encodes the layer index, the layer-appropriate
    primary metric (MMD inside the band and for non-finite-SID
    layers; SID outside the band), the SHD tiebreaker, and the
    deterministic ``configuration_hash_full`` final tiebreaker.
    Finite values rank strictly below non-finite within each metric
    step.
    """
    layer = _candidate_layer(candidate)
    aggregate = candidate.aggregate_metrics
    if layer == 0:
        return (
            0,
            _metric_sort_key(aggregate.get("mean_mmd_primary")),
            _metric_sort_key(aggregate.get("mean_shd")),
            candidate.configuration_hash_full,
        )
    if layer == 1:
        return (
            1,
            _metric_sort_key(aggregate.get("mean_sid")),
            _metric_sort_key(aggregate.get("mean_shd")),
            candidate.configuration_hash_full,
        )
    return (
        2,
        _metric_sort_key(aggregate.get("mean_mmd_primary")),
        _metric_sort_key(aggregate.get("mean_shd")),
        candidate.configuration_hash_full,
    )


# ---------------------------------------------------------------------------
# Cell ranking
# ---------------------------------------------------------------------------


def _group_records_by_candidate_hash(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_condition: str,
    expected_model: str,
) -> dict[str, list[dict[str, Any]]]:
    """Group cell records by ``configuration_hash_full`` and validate shape.

    Raises ``ValueError`` if the cell does not contain exactly five
    distinct candidate hashes, exactly two records per hash, or if
    the two records per hash do not cover seed values ``(201, 202)``
    exactly once each.
    """
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["condition"] != expected_condition:
            raise ValueError(
                f"calibration cell records mix conditions: cell "
                f"({expected_condition!r}, {expected_model!r}) "
                f"received a record with condition="
                f"{record['condition']!r}"
            )
        if record["model"] != expected_model:
            raise ValueError(
                f"calibration cell records mix models: cell "
                f"({expected_condition!r}, {expected_model!r}) "
                f"received a record with model={record['model']!r}"
            )
        by_hash[record["configuration_hash_full"]].append(record)

    if len(by_hash) != _EXPECTED_CANDIDATES_PER_CELL:
        raise ValueError(
            f"calibration cell ({expected_condition!r}, "
            f"{expected_model!r}) contains "
            f"{len(by_hash)} candidate hashes; expected exactly "
            f"{_EXPECTED_CANDIDATES_PER_CELL}"
        )

    for config_hash, hash_records in by_hash.items():
        if len(hash_records) != _EXPECTED_SEEDS_PER_CANDIDATE:
            raise ValueError(
                f"calibration cell ({expected_condition!r}, "
                f"{expected_model!r}) candidate "
                f"{config_hash[:_HASH_PREFIX_LENGTH]} has "
                f"{len(hash_records)} records; expected exactly "
                f"{_EXPECTED_SEEDS_PER_CANDIDATE}"
            )
        seeds_seen = sorted(record["seed_value"] for record in hash_records)
        if tuple(seeds_seen) != _EXPECTED_CALIBRATION_SEEDS:
            raise ValueError(
                f"calibration cell ({expected_condition!r}, "
                f"{expected_model!r}) candidate "
                f"{config_hash[:_HASH_PREFIX_LENGTH]} carries "
                f"seed_values={seeds_seen}; expected "
                f"{list(_EXPECTED_CALIBRATION_SEEDS)}"
            )

    return dict(by_hash)


def _build_candidate_from_cell_records(
    config_hash_full: str,
    hash_records: Sequence[dict[str, Any]],
    *,
    best_finite_mean_sid: float | None,
) -> _RankedCandidate:
    """Build a ``_RankedCandidate`` from the records sharing one hash.

    The per-seed records are deep-copied and sorted by ``seed_value``.
    The aggregate metrics, diagnostic flags, and band-eligibility
    fields are computed from the per-seed records and the cell's
    ``best_finite_mean_sid`` reference.
    """
    sorted_records = sorted(hash_records, key=lambda r: r["seed_value"])
    per_seed_metrics = [
        _extract_per_seed_record(record) for record in sorted_records
    ]
    (
        aggregate_metrics,
        has_non_finite_seed_metric,
        degenerate_metric_names,
        ranking_warning,
    ) = _build_aggregate_metrics(per_seed_metrics)

    in_band = _candidate_in_sid_band(
        aggregate_metrics["mean_sid"],
        best_finite_mean_sid=best_finite_mean_sid,
    )
    if best_finite_mean_sid is None:
        sid_band_threshold: float | None = None
    else:
        sid_band_threshold = (
            best_finite_mean_sid * _SID_BAND_THRESHOLD_FACTOR
        )

    aggregate_metrics["sid_band_eligible"] = in_band
    aggregate_metrics["sid_band_reference"] = best_finite_mean_sid
    aggregate_metrics["sid_band_threshold"] = sid_band_threshold

    source_run_ids = [record["run_id"] for record in sorted_records]
    sample_record = sorted_records[0]
    return _RankedCandidate(
        model=sample_record["model"],
        condition=sample_record["condition"],
        configuration_hash_full=config_hash_full,
        configuration_hash_prefix=config_hash_full[:_HASH_PREFIX_LENGTH],
        hyperparameters=copy.deepcopy(sample_record["hyperparameters"]),
        per_seed_records=per_seed_metrics,
        source_run_ids=source_run_ids,
        aggregate_metrics=aggregate_metrics,
        has_non_finite_seed_metric=has_non_finite_seed_metric,
        degenerate_metric_names=list(degenerate_metric_names),
        ranking_warning=ranking_warning,
        sid_band_eligible=in_band,
        sid_band_reference=best_finite_mean_sid,
        sid_band_threshold=sid_band_threshold,
    )


def _candidate_output_dict(candidate: _RankedCandidate) -> dict[str, Any]:
    """Build the artefact-shaped candidate_ranking entry for one candidate.

    The returned dict is recursively sanitised so no raw ``NaN``,
    ``+inf``, or ``-inf`` float can appear anywhere inside per-seed
    metrics, threshold-robustness rows, or per-intervention rows;
    such values are converted to ``None`` while the candidate-level
    degeneracy flags continue to record that non-finite inputs were
    seen.
    """
    raw = {
        "rank": candidate.rank,
        "configuration_hash_prefix": candidate.configuration_hash_prefix,
        "configuration_hash_full": candidate.configuration_hash_full,
        "hyperparameters": copy.deepcopy(candidate.hyperparameters),
        "aggregate_metrics": copy.deepcopy(candidate.aggregate_metrics),
        "per_seed_metrics": [
            copy.deepcopy(record) for record in candidate.per_seed_records
        ],
        "source_run_ids": list(candidate.source_run_ids),
        "n_calibration_records": len(candidate.per_seed_records),
    }
    return _sanitise_for_json(raw)


def _selection_output_dict(rank_one: _RankedCandidate) -> dict[str, Any]:
    """Build the artefact-shaped selections[condition][model] entry.

    The returned dict is recursively sanitised by the same rule used
    for candidate entries: every non-finite float is replaced by
    ``None``, booleans pass through, and the rank-1 degeneracy flag
    is preserved.
    """
    raw = {
        "selected_configuration_hash_prefix": (
            rank_one.configuration_hash_prefix
        ),
        "selected_configuration_hash_full": (
            rank_one.configuration_hash_full
        ),
        "selected_hyperparameters": copy.deepcopy(rank_one.hyperparameters),
        "selected_rank": 1,
        "selection_metrics": copy.deepcopy(rank_one.aggregate_metrics),
        "source_run_ids": list(rank_one.source_run_ids),
        "degeneracy_flag": rank_one.has_non_finite_seed_metric,
    }
    return _sanitise_for_json(raw)


def rank_condition_model_cell(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank one (model, condition) cell from exactly 10 calibration records.

    The input must contain exactly 10 records, all from the same
    ``(model, condition)`` pair, distributed as 5 distinct
    ``configuration_hash_full`` values with 2 records each (one per
    calibration seed).

    Returns
    -------
    tuple
        ``(candidate_ranking, selection)``.

        - ``candidate_ranking`` is a list of 5 artefact-shaped
          candidate dictionaries in rank order (ranks 1..5).
        - ``selection`` is the artefact-shaped selection dictionary
          for the rank-1 candidate.

    Raises
    ------
    ValueError
        If the input is not a sequence, is empty, has the wrong
        record count, mixes models or conditions, has the wrong
        candidate / seed cardinality, or contains a malformed
        individual record.
    """
    if records is None:
        raise ValueError("calibration cell records must not be None")
    records_list = list(records)
    if len(records_list) != _EXPECTED_RECORDS_PER_CELL:
        raise ValueError(
            "calibration cell ranker requires exactly "
            f"{_EXPECTED_RECORDS_PER_CELL} records "
            f"(5 candidates x 2 seeds); got {len(records_list)}"
        )
    validated_records = [
        _validate_input_record(record, record_index=idx)
        for idx, record in enumerate(records_list)
    ]
    cell_condition = validated_records[0]["condition"]
    cell_model = validated_records[0]["model"]
    by_hash = _group_records_by_candidate_hash(
        validated_records,
        expected_condition=cell_condition,
        expected_model=cell_model,
    )
    return _rank_grouped_cell(
        by_hash,
        cell_condition=cell_condition,
        cell_model=cell_model,
    )


def _rank_grouped_cell(
    by_hash: Mapping[str, Sequence[dict[str, Any]]],
    *,
    cell_condition: str,
    cell_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build, sort, and serialise candidates for one already-grouped cell."""
    # First pass: compute every candidate's mean_sid so we can derive
    # the cell's best_finite_mean_sid before constructing band-aware
    # candidate records.
    candidate_mean_sids: dict[str, float | None] = {}
    for config_hash, hash_records in by_hash.items():
        sid_values = [record["sid"] for record in hash_records]
        mean_sid, _ = _compute_mean_std(sid_values)
        candidate_mean_sids[config_hash] = _finite_or_none(mean_sid)

    finite_mean_sids = [
        v for v in candidate_mean_sids.values() if v is not None
    ]
    best_finite_mean_sid = min(finite_mean_sids) if finite_mean_sids else None

    candidates: list[_RankedCandidate] = []
    for config_hash, hash_records in by_hash.items():
        candidates.append(
            _build_candidate_from_cell_records(
                config_hash,
                hash_records,
                best_finite_mean_sid=best_finite_mean_sid,
            )
        )

    candidates.sort(key=_candidate_sort_key)
    for index, candidate in enumerate(candidates, start=1):
        candidate.rank = index

    candidate_ranking_output = [
        _candidate_output_dict(candidate) for candidate in candidates
    ]
    selection_output = _selection_output_dict(candidates[0])
    return candidate_ranking_output, selection_output


# ---------------------------------------------------------------------------
# Full calibration set
# ---------------------------------------------------------------------------


def rank_calibration_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank the full 40-record calibration set across all four cells.

    The input must contain exactly 40 records distributed as 4
    cells (centred_only/dagma, centred_only/dcdi, standardised/dagma,
    standardised/dcdi) with 5 candidate hashes per cell and 2
    calibration seeds per hash.

    Returns
    -------
    dict
        Nested output dict with two keys:

        - ``candidate_ranking[condition][model]`` is a list of 5
          ranked candidate dictionaries.
        - ``selections[condition][model]`` is the rank-1 selection
          dictionary.

        The result is artefact-shape compatible: passing it as the
        ``candidate_ranking`` and ``selections`` fields of a valid
        top-level artefact envelope produces an artefact that the
        selected-configurations validator accepts.

    Raises
    ------
    ValueError
        If the input does not contain exactly 40 records, if the
        cells are not exactly the four expected ``(condition, model)``
        pairs, or if any individual record fails validation.
    """
    if records is None:
        raise ValueError("calibration records must not be None")
    records_list = list(records)
    if len(records_list) != _EXPECTED_FULL_SET_RECORDS:
        raise ValueError(
            "full-set calibration ranker requires exactly "
            f"{_EXPECTED_FULL_SET_RECORDS} records "
            "(2 models x 2 conditions x 5 candidates x 2 seeds); got "
            f"{len(records_list)}"
        )
    validated_records = [
        _validate_input_record(record, record_index=idx)
        for idx, record in enumerate(records_list)
    ]

    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in validated_records:
        by_cell[(record["condition"], record["model"])].append(record)

    expected_cells = frozenset(
        (condition, model)
        for condition in sorted(_VALID_CONDITIONS)
        for model in sorted(_VALID_MODELS)
    )
    actual_cells = frozenset(by_cell.keys())
    missing_cells = expected_cells - actual_cells
    if missing_cells:
        raise ValueError(
            "full-set calibration ranker is missing one or more "
            f"(condition, model) cells: {sorted(missing_cells)}"
        )
    unexpected_cells = actual_cells - expected_cells
    if unexpected_cells:
        raise ValueError(
            "full-set calibration ranker received unexpected "
            f"(condition, model) cells: {sorted(unexpected_cells)}"
        )

    candidate_ranking: dict[str, dict[str, list[dict[str, Any]]]] = {}
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in sorted(_VALID_CONDITIONS):
        candidate_ranking[condition] = {}
        selections[condition] = {}
        for model in sorted(_VALID_MODELS):
            cell_records = by_cell[(condition, model)]
            by_hash = _group_records_by_candidate_hash(
                cell_records,
                expected_condition=condition,
                expected_model=model,
            )
            cell_ranking, cell_selection = _rank_grouped_cell(
                by_hash,
                cell_condition=condition,
                cell_model=model,
            )
            candidate_ranking[condition][model] = cell_ranking
            selections[condition][model] = cell_selection

    return {
        "candidate_ranking": candidate_ranking,
        "selections": selections,
    }


__all__ = [
    "rank_calibration_records",
    "rank_condition_model_cell",
]
