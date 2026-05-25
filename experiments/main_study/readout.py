"""Main-evaluation readout foundation.

Build the canonical flat dataset from the 224 persisted main-
evaluation records. This module is strictly read-only with respect
to experiment records: it loads JSON records, reads npz artefacts,
flattens each record into one CSV row, computes direct
prior-edge engagement columns from the persisted continuous-W
matrix, and writes the foundation files all later readout stages
depend on. No model is fit, no metric is recomputed, no
hypothesis-level statistic is calculated, and no plot is produced.

Output layout
-------------
``<output_root>/results/main_study/main_evaluation/<main_evaluation_run_hash12>/readout/`` ::

    main_evaluation_flat_records.csv
    cell_summary.csv
    status_summary.csv
    validation_summary.json
    forbidden_edge_engagement.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from experiments.main_study.records import (
    MainStudyRunRecord,
    record_from_json,
)
from experiments.main_study.run_io import resolve_relative_path
from experiments.main_study.schema import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EXPECTED_MAIN_EVALUATION_RUN_HASH12: str = "864fe6722256"

EXPECTED_RECORD_COUNT: int = 224

EXPECTED_COUNTS_BY_METHOD: dict[str, int] = {
    "prior_free": 7,
    "matched_l1": 7,
    "soft_frobenius": 175,
    "hard_exclusion": 35,
}

EVALUATION_SEED_VALUES: tuple[int, ...] = tuple(EVALUATION_SEEDS)
FORBIDDEN_CALIBRATION_SEEDS: tuple[int, ...] = tuple(CALIBRATION_SEEDS)

EXPECTED_MATCHED_L1_LAMBDA1: float = 0.0625
EXPECTED_LAMBDA_PRIOR: float = float(FROZEN_LAMBDA_PRIOR)

PROJECT_THRESHOLD: float = 0.3

CORRUPTION_GRID_SIZE: int = 5
CONFIDENCE_GRID_SIZE: int = 5

THRESHOLDED_ADJACENCY_KEY: str = "thresholded_adjacency"
CONTINUOUS_W_KEY: str = "continuous_w"

# Output filenames.
FLAT_RECORDS_CSV: str = "main_evaluation_flat_records.csv"
CELL_SUMMARY_CSV: str = "cell_summary.csv"
STATUS_SUMMARY_CSV: str = "status_summary.csv"
VALIDATION_SUMMARY_JSON: str = "validation_summary.json"
FORBIDDEN_EDGE_ENGAGEMENT_CSV: str = "forbidden_edge_engagement.csv"


# Canonical flat-CSV column order. The writer iterates this tuple
# verbatim so column order is reproducible and pandas reads back the
# exact schema downstream.
FLAT_CSV_COLUMNS: tuple[str, ...] = (
    # identity
    "run_id",
    "configuration_hash_full",
    "configuration_hash_prefix",
    "record_path",
    # config
    "method_family",
    "seed_value",
    "seed_population",
    "confidence",
    "corruption_fraction",
    "corruption_index",
    "lambda_prior",
    "matched_l1_lambda1",
    "dagma_lambda1",
    "parent_heldout_run_hash_full",
    # status
    "fit_status",
    "metric_status",
    "graph_status",
    "sampler_status",
    # metrics
    "sid",
    "shd",
    "mmd",
    "edge_count_from_thresholded_adjacency",
    # artefact paths
    "continuous_w_path",
    "thresholded_adjacency_path",
    "true_adjacency_path",
    # engagement
    "n_targeted_forbidden_edges",
    "mean_abs_w_targeted_forbidden_edges",
    "fraction_targeted_forbidden_above_threshold",
    "mean_abs_w_non_targeted_edges",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FlatRecordRow:
    """One flattened row of the canonical readout table.

    The field set is exactly the columns listed in
    :data:`FLAT_CSV_COLUMNS`. Nullable fields use ``None``; the
    canonical CSV writer renders ``None`` as an empty cell so a
    downstream ``pandas.read_csv`` parses it as missing.
    """

    run_id: str
    configuration_hash_full: str
    configuration_hash_prefix: str
    record_path: str

    method_family: str
    seed_value: int
    seed_population: str
    confidence: Optional[float]
    corruption_fraction: Optional[float]
    corruption_index: Optional[int]
    lambda_prior: Optional[float]
    matched_l1_lambda1: Optional[float]
    dagma_lambda1: float
    parent_heldout_run_hash_full: str

    fit_status: str
    metric_status: str
    graph_status: Optional[str]
    sampler_status: Optional[str]

    sid: Optional[float]
    shd: Optional[float]
    mmd: Optional[float]
    edge_count_from_thresholded_adjacency: Optional[int]

    continuous_w_path: Optional[str]
    thresholded_adjacency_path: Optional[str]
    true_adjacency_path: Optional[str]

    n_targeted_forbidden_edges: Optional[int]
    mean_abs_w_targeted_forbidden_edges: Optional[float]
    fraction_targeted_forbidden_above_threshold: Optional[float]
    mean_abs_w_non_targeted_edges: Optional[float]


@dataclass(frozen=True, kw_only=True)
class ValidationSummary:
    """Aggregate result of :func:`validate_flat_rows`."""

    main_evaluation_run_hash12: str
    n_records: int
    method_family_counts: dict[str, int]
    seed_values: tuple[int, ...]
    all_statuses_computed: bool
    all_metrics_finite: bool
    all_required_artifacts_resolved: bool
    soft_frobenius_cell_count: int
    hard_exclusion_cell_count: int
    prior_free_count: int
    matched_l1_count: int
    validation_errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def main_evaluation_records_dir(
    output_root: Path, main_evaluation_run_hash12: str
) -> Path:
    """Return the canonical per-run-records directory under output_root."""
    return (
        output_root
        / "results"
        / "main_study"
        / main_evaluation_run_hash12
        / "records"
    )


def main_evaluation_readout_dir(
    output_root: Path, main_evaluation_run_hash12: str
) -> Path:
    """Return the readout-output directory under the main_evaluation summary tree."""
    return (
        output_root
        / "results"
        / "main_study"
        / "main_evaluation"
        / main_evaluation_run_hash12
        / "readout"
    )


def resolve_artifact_path(
    relative_path: Optional[str], *, base_dir: Path
) -> Optional[Path]:
    """Resolve a non-``None`` relative artefact path under ``base_dir``.

    ``None`` passes through unchanged. A non-``None`` path that does
    not exist on disk raises ``FileNotFoundError`` naming the
    relative path and the base directory.
    """
    if relative_path is None:
        return None
    full = resolve_relative_path(relative_path, base_dir=base_dir)
    if not full.exists():
        raise FileNotFoundError(
            f"artefact at {relative_path!r} is missing under base_dir "
            f"{base_dir!r}; resolved path {full!r} does not exist."
        )
    return full


def load_npz_array(
    relative_path: Optional[str],
    *,
    base_dir: Path,
    expected_key: str,
) -> Optional[np.ndarray]:
    """Load a single named array from an npz artefact.

    Returns ``None`` if ``relative_path`` is ``None``. Otherwise
    resolves the path under ``base_dir`` (raising if absent), opens
    the npz container, and returns ``data[expected_key]`` as a copy.
    A missing key raises ``ValueError`` naming the path and the
    expected key.
    """
    full = resolve_artifact_path(relative_path, base_dir=base_dir)
    if full is None:
        return None
    with np.load(full) as data:
        if expected_key not in data.files:
            raise ValueError(
                f"npz at {relative_path!r} is missing expected key "
                f"{expected_key!r}; available keys: {list(data.files)}."
            )
        return np.asarray(data[expected_key]).copy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_main_evaluation_records(
    records_dir: Path,
) -> tuple[MainStudyRunRecord, ...]:
    """Load every ``*.json`` record under ``records_dir`` deterministically.

    Records are sorted lexicographically by run_id. The function
    rejects an empty directory, duplicate run-ids, and duplicate
    configuration_hash_full values.
    """
    if not isinstance(records_dir, Path):
        raise TypeError(
            "records_dir must be a pathlib.Path; got "
            f"{type(records_dir).__name__}."
        )
    if not records_dir.exists() or not records_dir.is_dir():
        raise FileNotFoundError(
            f"records_dir {records_dir!r} does not exist or is not a "
            "directory."
        )
    paths = sorted(records_dir.glob("*.json"))
    if not paths:
        raise ValueError(
            f"records_dir {records_dir!r} contains no *.json records."
        )
    records: list[MainStudyRunRecord] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        record = record_from_json(text)
        if record.run_id in seen_ids:
            raise ValueError(
                f"duplicate run_id {record.run_id!r} in records_dir "
                f"{records_dir!r}."
            )
        if record.configuration_hash_full in seen_hashes:
            raise ValueError(
                f"duplicate configuration_hash_full "
                f"{record.configuration_hash_full!r} in records_dir "
                f"{records_dir!r}."
            )
        seen_ids.add(record.run_id)
        seen_hashes.add(record.configuration_hash_full)
        records.append(record)
    records.sort(key=lambda r: r.run_id)
    return tuple(records)


def record_config_value(
    record: MainStudyRunRecord, dotted_path: str, default: Any = None
) -> Any:
    """Read a nested config field by dotted path. Returns ``default`` if missing.

    Example: ``record_config_value(rec,
    "config.corrupted_prior_spec.corruption_fraction")``.
    Intermediate ``None`` values short-circuit to ``default``.
    """
    if not dotted_path:
        raise ValueError("dotted_path must be a non-empty string.")
    parts = dotted_path.split(".")
    if parts[0] != "config":
        raise ValueError(
            "dotted_path must begin with 'config'; got "
            f"{dotted_path!r}."
        )
    current: Any = record.config
    for part in parts[1:]:
        if current is None:
            return default
        if dataclasses.is_dataclass(current):
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
            continue
        if isinstance(current, dict):
            current = current.get(part, default)
            continue
        return default
    return current


# ---------------------------------------------------------------------------
# Edge-count and prior-edge engagement
# ---------------------------------------------------------------------------


def off_diagonal_edge_count(thresholded_adjacency: np.ndarray) -> int:
    """Count off-diagonal ``True`` entries; diagonal is excluded.

    The input must be a 2D square ndarray with bool or integer
    dtype; floats are rejected because this measure is defined on the
    project's thresholded boolean adjacency, not on the continuous
    weight matrix.
    """
    if not isinstance(thresholded_adjacency, np.ndarray):
        raise TypeError(
            "off_diagonal_edge_count requires a numpy ndarray; got "
            f"{type(thresholded_adjacency).__name__}."
        )
    if (
        thresholded_adjacency.ndim != 2
        or thresholded_adjacency.shape[0] != thresholded_adjacency.shape[1]
    ):
        raise ValueError(
            "off_diagonal_edge_count requires a 2D square array; got "
            f"shape {thresholded_adjacency.shape}."
        )
    if thresholded_adjacency.dtype.kind not in "bi":
        raise ValueError(
            "off_diagonal_edge_count requires a bool-like or integer "
            f"array; got dtype {thresholded_adjacency.dtype}."
        )
    a = np.asarray(thresholded_adjacency, dtype=bool).copy()
    np.fill_diagonal(a, False)
    return int(a.sum())


def forbidden_edges_from_record(
    record: MainStudyRunRecord,
) -> Optional[tuple[tuple[int, int], ...]]:
    """Return the run's prior forbidden-edge set, or ``None`` if absent.

    The set lives at
    ``record.config.corrupted_prior_spec.forbidden_edges``. The
    function validates each edge is a pair of ints in
    ``[0, n_nodes)`` and preserves the order in which the edges were
    stored (no resort).
    """
    cps = record_config_value(record, "config.corrupted_prior_spec")
    if cps is None:
        return None
    raw = getattr(cps, "forbidden_edges", None)
    if raw is None:
        return None
    n_nodes = int(getattr(cps, "n_nodes", record.n_nodes))
    out: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                "every forbidden edge must be a length-2 (i, j) pair; "
                f"got {entry!r}."
            )
        i_val, j_val = entry
        if isinstance(i_val, bool) or isinstance(j_val, bool):
            raise ValueError(
                "forbidden-edge indices must be ints, not bools; got "
                f"{entry!r}."
            )
        if not isinstance(i_val, int) or not isinstance(j_val, int):
            raise ValueError(
                "forbidden-edge indices must be ints; got "
                f"{entry!r}."
            )
        if not (0 <= int(i_val) < n_nodes) or not (
            0 <= int(j_val) < n_nodes
        ):
            raise ValueError(
                f"forbidden-edge index out of range [0, {n_nodes}); "
                f"got {entry!r}."
            )
        out.append((int(i_val), int(j_val)))
    return tuple(out)


_ENGAGEMENT_NULL: dict[str, Optional[Any]] = {
    "n_targeted_forbidden_edges": None,
    "mean_abs_w_targeted_forbidden_edges": None,
    "fraction_targeted_forbidden_above_threshold": None,
    "mean_abs_w_non_targeted_edges": None,
}


def compute_prior_edge_engagement(
    continuous_w: Optional[np.ndarray],
    forbidden_edges: Optional[tuple[tuple[int, int], ...]],
    *,
    threshold: float = PROJECT_THRESHOLD,
) -> dict[str, Any]:
    """Compute direct prior-edge engagement on the continuous-W matrix.

    ``targeted`` entries are ``|W[i, j]|`` for the run's own prior
    forbidden edges; ``non_targeted`` entries are ``|W[i, j]|`` over
    all off-diagonal positions that are not in the forbidden set.
    The result is computed from the run's own continuous weights;
    the true graph is intentionally not used here.

    When ``continuous_w`` or ``forbidden_edges`` is ``None`` (e.g. a
    prior-free or matched-L1 run with no prior set, or a failed run
    with no continuous-W artefact), all four output fields are
    ``None``.
    """
    if continuous_w is None or forbidden_edges is None:
        return dict(_ENGAGEMENT_NULL)
    if not isinstance(continuous_w, np.ndarray):
        raise TypeError(
            "continuous_w must be a numpy ndarray; got "
            f"{type(continuous_w).__name__}."
        )
    if (
        continuous_w.ndim != 2
        or continuous_w.shape[0] != continuous_w.shape[1]
    ):
        raise ValueError(
            "continuous_w must be a 2D square array; got shape "
            f"{continuous_w.shape}."
        )
    n_nodes = int(continuous_w.shape[0])
    abs_w = np.abs(continuous_w.astype(float))
    targeted_set = {tuple(e) for e in forbidden_edges}
    targeted_values: list[float] = []
    for (i, j) in targeted_set:
        if not (0 <= i < n_nodes) or not (0 <= j < n_nodes):
            raise ValueError(
                f"forbidden edge {(i, j)!r} out of range for "
                f"W of shape {continuous_w.shape}."
            )
        if i == j:
            raise ValueError(
                f"forbidden edge {(i, j)!r} is on the diagonal; "
                "self-loops are not valid prior edges."
            )
        targeted_values.append(float(abs_w[i, j]))
    non_targeted_values: list[float] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if (i, j) in targeted_set:
                continue
            non_targeted_values.append(float(abs_w[i, j]))
    n_targeted = len(targeted_values)
    mean_targeted: Optional[float]
    fraction_above: Optional[float]
    if n_targeted == 0:
        mean_targeted = None
        fraction_above = None
    else:
        mean_targeted = float(np.mean(targeted_values))
        above = sum(
            1 for v in targeted_values if v >= float(threshold)
        )
        fraction_above = float(above) / float(n_targeted)
    mean_non_targeted = (
        float(np.mean(non_targeted_values))
        if non_targeted_values else None
    )
    return {
        "n_targeted_forbidden_edges": int(n_targeted),
        "mean_abs_w_targeted_forbidden_edges": mean_targeted,
        "fraction_targeted_forbidden_above_threshold": fraction_above,
        "mean_abs_w_non_targeted_edges": mean_non_targeted,
    }


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_record_metrics(record: MainStudyRunRecord) -> None:
    """For computed metric_status, enforce SID/SHD/MMD validity.

    Negative finite MMD is accepted because the raw unbiased RBF
    MMD-squared estimator can be negative in finite samples.
    """
    if record.metric_status != "computed":
        return
    for label, value in (("sid", record.sid), ("shd", record.shd)):
        if value is None or not _is_finite_real(value) or float(value) < 0.0:
            raise ValueError(
                "metric_status=='computed' requires "
                f"{label} to be a finite non-negative number; got "
                f"{value!r} on run {record.run_id!r}."
            )
    if (
        record.mmd is None
        or isinstance(record.mmd, bool)
        or not isinstance(record.mmd, (int, float))
        or not math.isfinite(float(record.mmd))
    ):
        raise ValueError(
            "metric_status=='computed' requires mmd to be a finite "
            f"real number (negative finite values are accepted); got "
            f"{record.mmd!r} on run {record.run_id!r}."
        )


def flatten_record(
    record: MainStudyRunRecord, *, base_dir: Path
) -> FlatRecordRow:
    """Flatten one record + its persisted artefacts into a :class:`FlatRecordRow`.

    The function reads ``thresholded_adjacency.npz`` (key
    ``"thresholded_adjacency"``) for the edge-count column and
    ``continuous_w.npz`` (key ``"continuous_w"``) for the
    prior-edge engagement columns. SID/SHD/MMD are taken from the
    record as-is, after the metric-validity gate; they are never
    recomputed here. Failed-fit records produce a row with ``None``
    metric, ``None`` engagement, and ``None`` edge count.
    """
    _validate_record_metrics(record)
    cfg = record.config
    confidence: Optional[float] = (
        None if cfg.confidence is None else float(cfg.confidence)
    )
    cps = cfg.corrupted_prior_spec
    corruption_fraction: Optional[float] = (
        None if cps is None else float(cps.corruption_fraction)
    )
    corruption_index: Optional[int] = (
        None if cps is None else int(cps.corruption_index)
    )
    lambda_prior: Optional[float] = (
        None if cfg.lambda_prior is None else float(cfg.lambda_prior)
    )
    matched_l1_lambda1: Optional[float] = (
        None if cfg.matched_l1_lambda1 is None
        else float(cfg.matched_l1_lambda1)
    )
    dagma_lambda1 = float(cfg.dagma_config.lambda1)

    # Edge count from thresholded adjacency; None if no artefact.
    thr_arr = load_npz_array(
        record.thresholded_adjacency_path,
        base_dir=base_dir,
        expected_key=THRESHOLDED_ADJACENCY_KEY,
    )
    edge_count: Optional[int] = (
        None if thr_arr is None else off_diagonal_edge_count(thr_arr)
    )

    # Engagement from continuous W.
    cont_w = load_npz_array(
        record.continuous_w_path,
        base_dir=base_dir,
        expected_key=CONTINUOUS_W_KEY,
    )
    forbidden = forbidden_edges_from_record(record)
    engagement = compute_prior_edge_engagement(
        cont_w, forbidden, threshold=PROJECT_THRESHOLD
    )

    return FlatRecordRow(
        run_id=record.run_id,
        configuration_hash_full=record.configuration_hash_full,
        configuration_hash_prefix=record.configuration_hash_prefix,
        record_path=_derive_record_relative_path(record, base_dir),
        method_family=cfg.method_family,
        seed_value=int(cfg.seed_value),
        seed_population=cfg.seed_population,
        confidence=confidence,
        corruption_fraction=corruption_fraction,
        corruption_index=corruption_index,
        lambda_prior=lambda_prior,
        matched_l1_lambda1=matched_l1_lambda1,
        dagma_lambda1=dagma_lambda1,
        parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
        fit_status=record.fit_status,
        metric_status=record.metric_status,
        graph_status=record.graph_status,
        sampler_status=record.sampler_status,
        sid=(None if record.sid is None else float(record.sid)),
        shd=(None if record.shd is None else float(record.shd)),
        mmd=(None if record.mmd is None else float(record.mmd)),
        edge_count_from_thresholded_adjacency=edge_count,
        continuous_w_path=record.continuous_w_path,
        thresholded_adjacency_path=record.thresholded_adjacency_path,
        true_adjacency_path=record.true_adjacency_path,
        n_targeted_forbidden_edges=engagement[
            "n_targeted_forbidden_edges"
        ],
        mean_abs_w_targeted_forbidden_edges=engagement[
            "mean_abs_w_targeted_forbidden_edges"
        ],
        fraction_targeted_forbidden_above_threshold=engagement[
            "fraction_targeted_forbidden_above_threshold"
        ],
        mean_abs_w_non_targeted_edges=engagement[
            "mean_abs_w_non_targeted_edges"
        ],
    )


def _derive_record_relative_path(
    record: MainStudyRunRecord, base_dir: Path
) -> str:
    """Best-effort relative path of the persisted record under base_dir.

    The record schema does not store its own path; we reconstruct
    the canonical location from the run_id and the parent layout.
    """
    # The record JSON's own path is not stored on the record. Derive
    # the canonical relative path using the project layout.
    # results/main_study/<hash>/records/<run_id>.json - but <hash>
    # is not visible from record fields, so we fall back to a path
    # constructed from the configuration_hash_prefix-keyed records
    # directory when present in the artefact tree.
    if record.continuous_w_path is not None:
        # The continuous_w_path embeds the parent hash:
        # results/main_study/<hash>/artefacts/<run_id>/continuous_w.npz
        parts = record.continuous_w_path.split("/")
        if (
            len(parts) >= 4
            and parts[0] == "results"
            and parts[1] == "main_study"
            and parts[3] == "artefacts"
        ):
            return (
                f"{parts[0]}/{parts[1]}/{parts[2]}/records/"
                f"{record.run_id}.json"
            )
    # Fall back to an empty string if we cannot derive it; the
    # validation step will not enforce this field's value.
    return ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0.0)


def _expected_soft_frobenius_cells() -> set[tuple[int, float, float]]:
    from experiments.main_study.priors import CORRUPTION_GRID
    from experiments.main_study.schema import CONFIDENCE_GRID
    out: set[tuple[int, float, float]] = set()
    for seed in EVALUATION_SEED_VALUES:
        for cf in CORRUPTION_GRID:
            for cn in CONFIDENCE_GRID:
                out.add((int(seed), float(cf), float(cn)))
    return out


def _expected_hard_exclusion_cells() -> set[tuple[int, float]]:
    from experiments.main_study.priors import CORRUPTION_GRID
    out: set[tuple[int, float]] = set()
    for seed in EVALUATION_SEED_VALUES:
        for cf in CORRUPTION_GRID:
            out.add((int(seed), float(cf)))
    return out


def validate_flat_rows(
    rows: tuple[FlatRecordRow, ...], *, strict: bool = True
) -> ValidationSummary:
    """Validate the 224-row grid; return a :class:`ValidationSummary`.

    With ``strict=True`` (the default) any validation error raises
    ``ValueError`` listing all collected errors. With ``strict=False``
    the errors are returned in
    :attr:`ValidationSummary.validation_errors`.
    """
    errors: list[str] = []
    method_counts: dict[str, int] = {}
    seed_set: set[int] = set()
    statuses_ok = True
    metrics_ok = True
    artifacts_ok = True
    soft_cells_seen: set[tuple[int, float, float]] = set()
    hard_cells_seen: set[tuple[int, float]] = set()
    main_evaluation_run_hash12 = ""

    for r in rows:
        method_counts[r.method_family] = method_counts.get(
            r.method_family, 0
        ) + 1
        seed_set.add(int(r.seed_value))
        if r.seed_value in FORBIDDEN_CALIBRATION_SEEDS:
            errors.append(
                f"calibration seed {r.seed_value!r} appears in run "
                f"{r.run_id!r}; only evaluation seeds 501-507 are "
                "allowed."
            )
        if r.fit_status != "success":
            statuses_ok = False
            errors.append(
                f"run {r.run_id!r} fit_status={r.fit_status!r} "
                "(expected 'success')."
            )
        if r.metric_status != "computed":
            statuses_ok = False
            errors.append(
                f"run {r.run_id!r} metric_status={r.metric_status!r} "
                "(expected 'computed')."
            )
        if r.graph_status != "valid_dag":
            statuses_ok = False
            errors.append(
                f"run {r.run_id!r} graph_status={r.graph_status!r} "
                "(expected 'valid_dag')."
            )
        if r.sampler_status != "available":
            statuses_ok = False
            errors.append(
                f"run {r.run_id!r} sampler_status={r.sampler_status!r} "
                "(expected 'available')."
            )
        # Metric validity (SID/SHD non-negative finite; MMD finite).
        for label, value, allow_negative in (
            ("sid", r.sid, False),
            ("shd", r.shd, False),
            ("mmd", r.mmd, True),
        ):
            if value is None or not _is_finite_real(value):
                metrics_ok = False
                errors.append(
                    f"run {r.run_id!r} {label} is not a finite number; "
                    f"got {value!r}."
                )
                continue
            if not allow_negative and float(value) < 0.0:
                metrics_ok = False
                errors.append(
                    f"run {r.run_id!r} {label} must be non-negative; "
                    f"got {value!r}."
                )
        # Required artefact paths must be present.
        for required_field in (
            "continuous_w_path",
            "thresholded_adjacency_path",
            "true_adjacency_path",
        ):
            if getattr(r, required_field) is None:
                artifacts_ok = False
                errors.append(
                    f"run {r.run_id!r} is missing required artefact "
                    f"path {required_field!r}."
                )
        if r.parent_heldout_run_hash_full and not main_evaluation_run_hash12:
            # We cannot derive the main_evaluation_run_hash12 from
            # the row directly; populate from the record_path when
            # possible (parts[2] in the canonical layout).
            parts = r.record_path.split("/")
            if len(parts) >= 3 and parts[1] == "main_study":
                main_evaluation_run_hash12 = parts[2]
        # Method-specific lambda checks.
        if r.method_family == "matched_l1":
            if r.matched_l1_lambda1 is None or not _is_close(
                r.matched_l1_lambda1, EXPECTED_MATCHED_L1_LAMBDA1
            ):
                errors.append(
                    f"matched_l1 run {r.run_id!r} matched_l1_lambda1="
                    f"{r.matched_l1_lambda1!r}; expected "
                    f"{EXPECTED_MATCHED_L1_LAMBDA1!r}."
                )
            if not _is_close(r.dagma_lambda1, EXPECTED_MATCHED_L1_LAMBDA1):
                errors.append(
                    f"matched_l1 run {r.run_id!r} dagma_lambda1="
                    f"{r.dagma_lambda1!r}; expected "
                    f"{EXPECTED_MATCHED_L1_LAMBDA1!r}."
                )
        if r.method_family == "soft_frobenius":
            if r.lambda_prior is None or not _is_close(
                r.lambda_prior, EXPECTED_LAMBDA_PRIOR
            ):
                errors.append(
                    f"soft_frobenius run {r.run_id!r} lambda_prior="
                    f"{r.lambda_prior!r}; expected "
                    f"{EXPECTED_LAMBDA_PRIOR!r}."
                )
            if r.confidence is None or r.corruption_fraction is None:
                errors.append(
                    f"soft_frobenius run {r.run_id!r} must carry "
                    "confidence and corruption_fraction; got "
                    f"confidence={r.confidence!r}, "
                    f"corruption_fraction={r.corruption_fraction!r}."
                )
            else:
                soft_cells_seen.add((
                    int(r.seed_value),
                    float(r.corruption_fraction),
                    float(r.confidence),
                ))
        if r.method_family == "hard_exclusion":
            if r.confidence is not None:
                errors.append(
                    f"hard_exclusion run {r.run_id!r} must not carry a "
                    f"confidence axis; got confidence={r.confidence!r}."
                )
            if r.corruption_fraction is None:
                errors.append(
                    f"hard_exclusion run {r.run_id!r} must carry a "
                    "corruption_fraction; got None."
                )
            else:
                hard_cells_seen.add((
                    int(r.seed_value), float(r.corruption_fraction)
                ))

    # Aggregate-grid checks.
    if len(rows) != EXPECTED_RECORD_COUNT:
        errors.append(
            f"expected exactly {EXPECTED_RECORD_COUNT} rows; got "
            f"{len(rows)}."
        )
    for family, expected in EXPECTED_COUNTS_BY_METHOD.items():
        if method_counts.get(family, 0) != expected:
            errors.append(
                f"method_family={family!r} count "
                f"{method_counts.get(family, 0)} != expected {expected}."
            )
    if seed_set != set(EVALUATION_SEED_VALUES):
        errors.append(
            f"seed set {sorted(seed_set)} does not equal expected "
            f"{sorted(EVALUATION_SEED_VALUES)}."
        )
    expected_soft = _expected_soft_frobenius_cells()
    missing_soft = expected_soft - soft_cells_seen
    if missing_soft:
        errors.append(
            "soft_frobenius grid is missing cells "
            f"{sorted(missing_soft)}."
        )
    expected_hard = _expected_hard_exclusion_cells()
    missing_hard = expected_hard - hard_cells_seen
    if missing_hard:
        errors.append(
            "hard_exclusion grid is missing cells "
            f"{sorted(missing_hard)}."
        )

    summary = ValidationSummary(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        n_records=len(rows),
        method_family_counts=dict(sorted(method_counts.items())),
        seed_values=tuple(sorted(seed_set)),
        all_statuses_computed=statuses_ok,
        all_metrics_finite=metrics_ok,
        all_required_artifacts_resolved=artifacts_ok,
        soft_frobenius_cell_count=len(soft_cells_seen),
        hard_exclusion_cell_count=len(hard_cells_seen),
        prior_free_count=method_counts.get("prior_free", 0),
        matched_l1_count=method_counts.get("matched_l1", 0),
        validation_errors=tuple(errors),
    )
    if strict and errors:
        raise ValueError(
            "validate_flat_rows: validation failed with "
            f"{len(errors)} error(s):\n- "
            + "\n- ".join(errors)
        )
    return summary


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _csv_cell(value: Any) -> str:
    """Render ``value`` for a CSV cell. ``None`` becomes empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def write_flat_records_csv(
    rows: Iterable[FlatRecordRow], path: Path
) -> None:
    """Write the canonical flat-records CSV at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(FLAT_CSV_COLUMNS))
        for r in rows:
            row_cells = [
                _csv_cell(getattr(r, col)) for col in FLAT_CSV_COLUMNS
            ]
            writer.writerow(row_cells)


def write_status_summary_csv(
    rows: Iterable[FlatRecordRow], path: Path
) -> None:
    """Write per-(family, fit, metric, graph, sampler) status counts."""
    counts: dict[tuple[str, str, str, Optional[str], Optional[str]], int] = {}
    for r in rows:
        key = (
            r.method_family,
            r.fit_status,
            r.metric_status,
            r.graph_status,
            r.sampler_status,
        )
        counts[key] = counts.get(key, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_family",
        "fit_status",
        "metric_status",
        "graph_status",
        "sampler_status",
        "count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (fam, fit, met, gph, smp), n in sorted(counts.items()):
            writer.writerow({
                "method_family": fam,
                "fit_status": fit,
                "metric_status": met,
                "graph_status": "" if gph is None else gph,
                "sampler_status": "" if smp is None else smp,
                "count": n,
            })


def _descriptive(
    values: list[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    mean = float(statistics.fmean(values))
    median = float(statistics.median(values))
    std = (
        float(statistics.stdev(values))
        if len(values) >= 2 else 0.0
    )
    return mean, std, median


def write_cell_summary_csv(
    rows: Iterable[FlatRecordRow], path: Path
) -> None:
    """Write per-cell descriptive statistics. No claims, no ranking."""
    grouped: dict[
        tuple[str, Optional[float], Optional[float]],
        list[FlatRecordRow],
    ] = {}
    for r in rows:
        key = (
            r.method_family,
            r.corruption_fraction,
            r.confidence,
        )
        grouped.setdefault(key, []).append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_family",
        "corruption_fraction",
        "confidence",
        "n",
        "sid_mean", "sid_std", "sid_median",
        "shd_mean", "shd_std", "shd_median",
        "mmd_mean", "mmd_std", "mmd_median",
        "edge_count_mean", "edge_count_std", "edge_count_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Stable sort: method_family ascending, then numeric fields
        # (treat None as -inf so prior_free / matched_l1 sort first).
        def sort_key(k: tuple[str, Optional[float], Optional[float]]):
            cf = float("-inf") if k[1] is None else k[1]
            cn = float("-inf") if k[2] is None else k[2]
            return (k[0], cf, cn)
        for key in sorted(grouped.keys(), key=sort_key):
            members = grouped[key]
            sids = [
                float(m.sid) for m in members if m.sid is not None
            ]
            shds = [
                float(m.shd) for m in members if m.shd is not None
            ]
            mmds = [
                float(m.mmd) for m in members if m.mmd is not None
            ]
            ecs = [
                float(m.edge_count_from_thresholded_adjacency)
                for m in members
                if m.edge_count_from_thresholded_adjacency is not None
            ]
            sid_m, sid_s, sid_med = _descriptive(sids)
            shd_m, shd_s, shd_med = _descriptive(shds)
            mmd_m, mmd_s, mmd_med = _descriptive(mmds)
            ec_m, ec_s, ec_med = _descriptive(ecs)
            writer.writerow({
                "method_family": key[0],
                "corruption_fraction": _csv_cell(key[1]),
                "confidence": _csv_cell(key[2]),
                "n": len(members),
                "sid_mean": _csv_cell(sid_m),
                "sid_std": _csv_cell(sid_s),
                "sid_median": _csv_cell(sid_med),
                "shd_mean": _csv_cell(shd_m),
                "shd_std": _csv_cell(shd_s),
                "shd_median": _csv_cell(shd_med),
                "mmd_mean": _csv_cell(mmd_m),
                "mmd_std": _csv_cell(mmd_s),
                "mmd_median": _csv_cell(mmd_med),
                "edge_count_mean": _csv_cell(ec_m),
                "edge_count_std": _csv_cell(ec_s),
                "edge_count_median": _csv_cell(ec_med),
            })


def write_forbidden_edge_engagement_csv(
    rows: Iterable[FlatRecordRow], path: Path
) -> None:
    """Write one engagement row per :class:`FlatRecordRow`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "method_family",
        "seed_value",
        "corruption_fraction",
        "confidence",
        "n_targeted_forbidden_edges",
        "mean_abs_w_targeted_forbidden_edges",
        "fraction_targeted_forbidden_above_threshold",
        "mean_abs_w_non_targeted_edges",
        "edge_count_from_thresholded_adjacency",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "run_id": r.run_id,
                "method_family": r.method_family,
                "seed_value": int(r.seed_value),
                "corruption_fraction": _csv_cell(r.corruption_fraction),
                "confidence": _csv_cell(r.confidence),
                "n_targeted_forbidden_edges": _csv_cell(
                    r.n_targeted_forbidden_edges
                ),
                "mean_abs_w_targeted_forbidden_edges": _csv_cell(
                    r.mean_abs_w_targeted_forbidden_edges
                ),
                "fraction_targeted_forbidden_above_threshold": _csv_cell(
                    r.fraction_targeted_forbidden_above_threshold
                ),
                "mean_abs_w_non_targeted_edges": _csv_cell(
                    r.mean_abs_w_non_targeted_edges
                ),
                "edge_count_from_thresholded_adjacency": _csv_cell(
                    r.edge_count_from_thresholded_adjacency
                ),
            })


def write_validation_summary_json(
    summary: ValidationSummary, path: Path
) -> None:
    """Write the validation summary as canonical JSON."""
    payload = {
        "main_evaluation_run_hash12": summary.main_evaluation_run_hash12,
        "n_records": int(summary.n_records),
        "method_family_counts": {
            k: int(v) for k, v in sorted(
                summary.method_family_counts.items()
            )
        },
        "seed_values": [int(s) for s in summary.seed_values],
        "all_statuses_computed": bool(summary.all_statuses_computed),
        "all_metrics_finite": bool(summary.all_metrics_finite),
        "all_required_artifacts_resolved": bool(
            summary.all_required_artifacts_resolved
        ),
        "soft_frobenius_cell_count": int(
            summary.soft_frobenius_cell_count
        ),
        "hard_exclusion_cell_count": int(
            summary.hard_exclusion_cell_count
        ),
        "prior_free_count": int(summary.prior_free_count),
        "matched_l1_count": int(summary.matched_l1_count),
        "validation_errors": list(summary.validation_errors),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_readout_foundation(
    *,
    output_root: Path,
    main_evaluation_run_hash12: str,
    strict: bool = True,
) -> ValidationSummary:
    """Load records, flatten, validate, and write all five M-9a outputs.

    The function is strictly read-only over experiment records: it
    never re-fits a model, never recomputes SID/SHD/MMD, never
    modifies any persisted record or artefact. Output files are
    written under
    ``<output_root>/results/main_study/main_evaluation/<main_evaluation_run_hash12>/readout/``.
    """
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
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
    records_dir = main_evaluation_records_dir(
        output_root, main_evaluation_run_hash12
    )
    records = load_main_evaluation_records(records_dir)
    rows = tuple(
        flatten_record(rec, base_dir=output_root) for rec in records
    )
    summary = validate_flat_rows(rows, strict=strict)
    readout_dir = main_evaluation_readout_dir(
        output_root, main_evaluation_run_hash12
    )
    readout_dir.mkdir(parents=True, exist_ok=True)
    write_flat_records_csv(rows, readout_dir / FLAT_RECORDS_CSV)
    write_cell_summary_csv(rows, readout_dir / CELL_SUMMARY_CSV)
    write_status_summary_csv(rows, readout_dir / STATUS_SUMMARY_CSV)
    write_forbidden_edge_engagement_csv(
        rows, readout_dir / FORBIDDEN_EDGE_ENGAGEMENT_CSV
    )
    write_validation_summary_json(
        summary, readout_dir / VALIDATION_SUMMARY_JSON
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK: int = 0
_EXIT_ERROR: int = 1


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readout",
        description=(
            "Build the main-evaluation readout foundation (flat "
            "records, cell summaries, status summary, validation "
            "summary, forbidden-edge engagement). Read-only over "
            "experiment records; no fitting, no metric recomputation, "
            "no plots, no notebooks."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Root directory under which results/main_study/... is "
            "located."
        ),
    )
    parser.add_argument(
        "--main-evaluation-run-hash12",
        type=str,
        required=True,
        help=(
            "12-character main-evaluation run hash naming the per-run "
            "records directory."
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help=(
            "If set, validation errors are written to the validation "
            "summary instead of raising."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_readout_foundation(
            output_root=args.output_root,
            main_evaluation_run_hash12=args.main_evaluation_run_hash12,
            strict=not args.non_strict,
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            f"readout: error: {type(exc).__name__}: {exc}\n"
        )
        return _EXIT_ERROR
    if summary.validation_errors:
        sys.stderr.write(
            "readout: validation errors recorded; see "
            f"{VALIDATION_SUMMARY_JSON}.\n"
        )
        return _EXIT_ERROR
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CELL_SUMMARY_CSV",
    "CONTINUOUS_W_KEY",
    "EVALUATION_SEED_VALUES",
    "EXPECTED_COUNTS_BY_METHOD",
    "EXPECTED_LAMBDA_PRIOR",
    "EXPECTED_MAIN_EVALUATION_RUN_HASH12",
    "EXPECTED_MATCHED_L1_LAMBDA1",
    "EXPECTED_RECORD_COUNT",
    "FLAT_CSV_COLUMNS",
    "FLAT_RECORDS_CSV",
    "FORBIDDEN_CALIBRATION_SEEDS",
    "FORBIDDEN_EDGE_ENGAGEMENT_CSV",
    "FlatRecordRow",
    "PROJECT_THRESHOLD",
    "STATUS_SUMMARY_CSV",
    "THRESHOLDED_ADJACENCY_KEY",
    "VALIDATION_SUMMARY_JSON",
    "ValidationSummary",
    "compute_prior_edge_engagement",
    "flatten_record",
    "forbidden_edges_from_record",
    "generate_readout_foundation",
    "load_main_evaluation_records",
    "load_npz_array",
    "main",
    "main_evaluation_readout_dir",
    "main_evaluation_records_dir",
    "off_diagonal_edge_count",
    "record_config_value",
    "resolve_artifact_path",
    "validate_flat_rows",
    "write_cell_summary_csv",
    "write_flat_records_csv",
    "write_forbidden_edge_engagement_csv",
    "write_status_summary_csv",
    "write_validation_summary_json",
]
