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

# M-9b output filenames (statistics and diagnostics).
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
PER_INTERVENTION_MMD_LONG_CSV: str = "per_intervention_mmd_long.csv"
PER_INTERVENTION_MMD_SUMMARY_CSV: str = "per_intervention_mmd_summary.csv"
STATISTICS_SUMMARY_JSON: str = "statistics_summary.json"

# Core metrics. For SID, SHD, and MMD, lower means lower error. For
# edge_count_from_thresholded_adjacency, lower means sparser learned
# graph, NOT necessarily better. The wins_a_lower column in the
# paired-comparison CSV must therefore be interpreted as "a is
# sparser", not "a is better", whenever metric == edge_count.
METRIC_COLUMNS: tuple[str, ...] = (
    "sid",
    "shd",
    "mmd",
    "edge_count_from_thresholded_adjacency",
)

# Predeclared baseline conditions (no post-hoc confidence selection).
_BASELINE_LABEL_PRIOR_FREE: str = "prior_free"
_BASELINE_LABEL_MATCHED_L1: str = "matched_l1"
_BASELINE_LABEL_SOFT_CLEAN_CONF1: str = "soft_frobenius_clean_conf1"
_BASELINE_LABEL_HARD_EXCLUSION_CLEAN: str = "hard_exclusion_clean"

BASELINE_CONDITION_LABELS: tuple[str, ...] = (
    _BASELINE_LABEL_PRIOR_FREE,
    _BASELINE_LABEL_MATCHED_L1,
    _BASELINE_LABEL_SOFT_CLEAN_CONF1,
    _BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
)

# Predeclared paired comparisons (A-E). diff convention is a - b.
PAIRED_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    (_BASELINE_LABEL_SOFT_CLEAN_CONF1, _BASELINE_LABEL_PRIOR_FREE),    # A
    (_BASELINE_LABEL_SOFT_CLEAN_CONF1, _BASELINE_LABEL_MATCHED_L1),    # B
    (
        _BASELINE_LABEL_SOFT_CLEAN_CONF1,
        _BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
    ),                                                                  # C
    (_BASELINE_LABEL_MATCHED_L1, _BASELINE_LABEL_PRIOR_FREE),          # D
    (_BASELINE_LABEL_HARD_EXCLUSION_CLEAN, _BASELINE_LABEL_PRIOR_FREE),# E
)

DIFF_CONVENTION: str = "a - b"


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
# Numeric coercion / CSV reading
# ---------------------------------------------------------------------------


# Columns parsed as float (None for empty cells).
_FLAT_FLOAT_COLUMNS: frozenset[str] = frozenset({
    "confidence",
    "corruption_fraction",
    "lambda_prior",
    "matched_l1_lambda1",
    "dagma_lambda1",
    "sid",
    "shd",
    "mmd",
    "mean_abs_w_targeted_forbidden_edges",
    "fraction_targeted_forbidden_above_threshold",
    "mean_abs_w_non_targeted_edges",
})

# Columns parsed as int (None for empty cells).
_FLAT_INT_COLUMNS: frozenset[str] = frozenset({
    "seed_value",
    "corruption_index",
    "edge_count_from_thresholded_adjacency",
    "n_targeted_forbidden_edges",
})


def _parse_cell(value: str, *, column: str) -> Any:
    """Parse one CSV cell into the right Python type for ``column``.

    Empty cells become ``None``. Float and int columns are coerced;
    everything else is returned as the original string.
    """
    if value == "":
        return None
    if column in _FLAT_INT_COLUMNS:
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    if column in _FLAT_FLOAT_COLUMNS:
        return float(value)
    return value


def load_flat_records_csv(path: Path) -> tuple[dict[str, Any], ...]:
    """Load the M-9a flat CSV and parse numeric cells.

    Validates that the header matches :data:`FLAT_CSV_COLUMNS` in
    order. Returns a tuple of dicts (one per row); empty cells
    become ``None``.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"load_flat_records_csv requires a Path; got "
            f"{type(path).__name__}."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"load_flat_records_csv: {path!r} does not exist."
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(
                f"load_flat_records_csv: {path!r} is empty."
            )
        if tuple(header) != FLAT_CSV_COLUMNS:
            raise ValueError(
                "load_flat_records_csv: header does not match the "
                f"canonical column order. expected {FLAT_CSV_COLUMNS}, "
                f"got {tuple(header)}."
            )
        out: list[dict[str, Any]] = []
        for raw in reader:
            if len(raw) != len(FLAT_CSV_COLUMNS):
                raise ValueError(
                    "load_flat_records_csv: row has "
                    f"{len(raw)} cells; expected "
                    f"{len(FLAT_CSV_COLUMNS)}."
                )
            row = {
                col: _parse_cell(raw[i], column=col)
                for i, col in enumerate(FLAT_CSV_COLUMNS)
            }
            out.append(row)
    return tuple(out)


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


def condition_key(
    row: dict[str, Any],
) -> tuple[str, Optional[float], Optional[float]]:
    """Return ``(method_family, corruption_fraction, confidence)``."""
    return (
        str(row["method_family"]),
        None if row["corruption_fraction"] is None
        else float(row["corruption_fraction"]),
        None if row["confidence"] is None
        else float(row["confidence"]),
    )


def select_condition(
    rows: Iterable[dict[str, Any]],
    *,
    method_family: str,
    corruption_fraction: Optional[float] = None,
    confidence: Optional[float] = None,
) -> tuple[dict[str, Any], ...]:
    """Filter ``rows`` by condition and return seed-sorted matches.

    ``None`` for ``corruption_fraction`` or ``confidence`` matches
    rows whose own value is ``None`` (i.e. prior_free / matched_l1
    / hard_exclusion's confidence axis), not a wildcard.
    """
    matches: list[dict[str, Any]] = []
    for r in rows:
        if r["method_family"] != method_family:
            continue
        cf = r["corruption_fraction"]
        cn = r["confidence"]
        if corruption_fraction is None:
            if cf is not None:
                continue
        else:
            if cf is None or not _is_close(
                float(cf), float(corruption_fraction)
            ):
                continue
        if confidence is None:
            if cn is not None:
                continue
        else:
            if cn is None or not _is_close(
                float(cn), float(confidence)
            ):
                continue
        matches.append(r)
    matches.sort(key=lambda r: int(r["seed_value"]))
    return tuple(matches)


def _select_baseline_rows(
    rows: Iterable[dict[str, Any]], label: str
) -> tuple[dict[str, Any], ...]:
    """Return seed-sorted rows for one predeclared baseline label."""
    if label == _BASELINE_LABEL_PRIOR_FREE:
        return select_condition(rows, method_family="prior_free")
    if label == _BASELINE_LABEL_MATCHED_L1:
        return select_condition(rows, method_family="matched_l1")
    if label == _BASELINE_LABEL_SOFT_CLEAN_CONF1:
        return select_condition(
            rows,
            method_family="soft_frobenius",
            corruption_fraction=0.0,
            confidence=1.0,
        )
    if label == _BASELINE_LABEL_HARD_EXCLUSION_CLEAN:
        return select_condition(
            rows,
            method_family="hard_exclusion",
            corruption_fraction=0.0,
        )
    raise ValueError(
        f"_select_baseline_rows: unknown baseline label {label!r}; "
        f"expected one of {BASELINE_CONDITION_LABELS}."
    )


def _baseline_condition_key(
    label: str,
) -> tuple[str, Optional[float], Optional[float]]:
    if label == _BASELINE_LABEL_PRIOR_FREE:
        return ("prior_free", None, None)
    if label == _BASELINE_LABEL_MATCHED_L1:
        return ("matched_l1", None, None)
    if label == _BASELINE_LABEL_SOFT_CLEAN_CONF1:
        return ("soft_frobenius", 0.0, 1.0)
    if label == _BASELINE_LABEL_HARD_EXCLUSION_CLEAN:
        return ("hard_exclusion", 0.0, None)
    raise ValueError(
        f"_baseline_condition_key: unknown label {label!r}."
    )


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def summary_stats(
    values: Iterable[float],
) -> dict[str, Any]:
    """Return ``n``, ``mean``, sample-``std``, ``median``, ``min``, ``max``.

    Negative finite values (e.g. the unbiased RBF MMD-squared
    estimator) are accepted. Inputs are filtered to finite floats
    before summarising; ``n`` is the post-filter count.
    """
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        if not math.isfinite(f):
            continue
        out.append(f)
    n = len(out)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    mean = float(statistics.fmean(out))
    median = float(statistics.median(out))
    std = float(statistics.stdev(out)) if n >= 2 else 0.0
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "median": median,
        "min": float(min(out)),
        "max": float(max(out)),
    }


def compute_baseline_comparison(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Emit one descriptive row per (baseline condition, metric).

    No method ranking. Output column order is the writer's
    canonical order.
    """
    rows_t = tuple(rows)
    out: list[dict[str, Any]] = []
    for label in BASELINE_CONDITION_LABELS:
        family, cf, cn = _baseline_condition_key(label)
        selected = _select_baseline_rows(rows_t, label)
        for metric in METRIC_COLUMNS:
            values = [r[metric] for r in selected]
            stats = summary_stats(values)
            out.append({
                "condition_label": label,
                "method_family": family,
                "corruption_fraction": cf,
                "confidence": cn,
                "metric": metric,
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "median": stats["median"],
                "min": stats["min"],
                "max": stats["max"],
            })
    return tuple(out)


# ---------------------------------------------------------------------------
# Paired-seed comparisons
# ---------------------------------------------------------------------------


def _two_sided_sign_test_p(
    n_pos: int, n_neg: int
) -> Optional[float]:
    """Two-sided sign-test p-value for paired data (excluding ties).

    Under H0, P(positive) == 0.5. p = 2 * min(P(X <= k_min),
    P(X >= k_max)) clipped to 1. Returns ``None`` if both counts are
    zero (no informative pairs).
    """
    n = int(n_pos) + int(n_neg)
    if n == 0:
        return None
    k = min(int(n_pos), int(n_neg))
    cumulative = 0.0
    for i in range(0, k + 1):
        cumulative += math.comb(n, i) * (0.5 ** n)
    p = 2.0 * cumulative
    return min(1.0, float(p))


def _benjamini_hochberg_q_values(
    p_values: list[Optional[float]],
) -> list[Optional[float]]:
    """Benjamini-Hochberg adjusted q-values, preserving input order.

    ``None`` entries stay ``None`` and are excluded from m.
    """
    indexed = [
        (i, p) for i, p in enumerate(p_values) if p is not None
    ]
    m = len(indexed)
    if m == 0:
        return [None] * len(p_values)
    indexed.sort(key=lambda x: float(x[1]))
    raw_q: list[float] = []
    for rank_idx, (_, p) in enumerate(indexed, start=1):
        raw_q.append(float(p) * float(m) / float(rank_idx))
    # Enforce monotone non-decreasing from largest to smallest sorted p.
    for k in range(len(raw_q) - 2, -1, -1):
        if raw_q[k] > raw_q[k + 1]:
            raw_q[k] = raw_q[k + 1]
    out: list[Optional[float]] = [None] * len(p_values)
    for rank_idx, (orig_i, _) in enumerate(indexed):
        out[orig_i] = min(1.0, float(raw_q[rank_idx]))
    return out


def paired_seed_difference(
    rows_a: Iterable[dict[str, Any]],
    rows_b: Iterable[dict[str, Any]],
    *,
    metric: str,
    label_a: str,
    label_b: str,
    n_bootstrap: int = 10000,
    random_seed: int = 90210,
) -> dict[str, Any]:
    """Paired-by-seed difference + percentile bootstrap CI.

    The difference convention is fixed: ``diff = value_a - value_b``.
    The function requires identical seed sets and emits one summary
    dictionary covering effect sizes, win/tie counts, and a
    two-sided sign-test p-value. For ``metric ==
    edge_count_from_thresholded_adjacency``, ``wins_a_lower`` means
    "a is sparser", not "a is better".
    """
    if metric not in METRIC_COLUMNS:
        raise ValueError(
            f"paired_seed_difference: metric must be one of "
            f"{METRIC_COLUMNS}; got {metric!r}."
        )
    a_by_seed = {
        int(r["seed_value"]): r for r in rows_a
    }
    b_by_seed = {
        int(r["seed_value"]): r for r in rows_b
    }
    if set(a_by_seed.keys()) != set(b_by_seed.keys()):
        raise ValueError(
            "paired_seed_difference: seed sets differ between "
            f"label_a={label_a!r} (seeds={sorted(a_by_seed)}) and "
            f"label_b={label_b!r} (seeds={sorted(b_by_seed)})."
        )
    seeds_sorted = sorted(a_by_seed.keys())
    a_values: list[float] = []
    b_values: list[float] = []
    for s in seeds_sorted:
        a_v = a_by_seed[s].get(metric)
        b_v = b_by_seed[s].get(metric)
        if a_v is None or b_v is None:
            raise ValueError(
                f"paired_seed_difference: missing {metric!r} value "
                f"at seed {s} (a={a_v!r}, b={b_v!r})."
            )
        a_values.append(float(a_v))
        b_values.append(float(b_v))
    a_arr = np.array(a_values, dtype=float)
    b_arr = np.array(b_values, dtype=float)
    diffs = a_arr - b_arr
    n_pairs = int(diffs.shape[0])
    mean_a = float(np.mean(a_arr)) if n_pairs else None
    mean_b = float(np.mean(b_arr)) if n_pairs else None
    mean_diff = float(np.mean(diffs)) if n_pairs else None
    median_diff = float(np.median(diffs)) if n_pairs else None

    wins_a_lower = int(np.sum(a_arr < b_arr))
    wins_b_lower = int(np.sum(a_arr > b_arr))
    ties = int(np.sum(a_arr == b_arr))

    # Sign test on the directionality of the paired difference.
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    sign_p = _two_sided_sign_test_p(n_pos, n_neg)

    # Percentile paired bootstrap CI of the mean difference.
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    if n_pairs > 0 and int(n_bootstrap) > 0:
        rng = np.random.default_rng(int(random_seed))
        indices = rng.integers(
            0, n_pairs, size=(int(n_bootstrap), n_pairs)
        )
        boot_means = diffs[indices].mean(axis=1)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "metric": metric,
        "diff_convention": DIFF_CONVENTION,
        "n_pairs": n_pairs,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wins_a_lower": wins_a_lower,
        "wins_b_lower": wins_b_lower,
        "ties": ties,
        "sign_test_p_two_sided": sign_p,
        "bh_q_value": None,
    }


def compute_paired_seed_comparisons(
    rows: Iterable[dict[str, Any]],
    *,
    n_bootstrap: int = 10000,
    random_seed: int = 90210,
) -> tuple[dict[str, Any], ...]:
    """Compute the five predeclared paired comparisons across all metrics.

    No best-confidence or post-hoc selected comparison is produced.
    BH q-values are computed across the entire table.
    """
    rows_t = tuple(rows)
    out: list[dict[str, Any]] = []
    for (label_a, label_b) in PAIRED_COMPARISON_PAIRS:
        rows_a = _select_baseline_rows(rows_t, label_a)
        rows_b = _select_baseline_rows(rows_t, label_b)
        for metric in METRIC_COLUMNS:
            out.append(paired_seed_difference(
                rows_a,
                rows_b,
                metric=metric,
                label_a=label_a,
                label_b=label_b,
                n_bootstrap=int(n_bootstrap),
                random_seed=int(random_seed),
            ))
    p_values = [row["sign_test_p_two_sided"] for row in out]
    q_values = _benjamini_hochberg_q_values(p_values)
    for row, q in zip(out, q_values):
        row["bh_q_value"] = q
    return tuple(out)


# ---------------------------------------------------------------------------
# Correlations (Pearson / Spearman / Kendall tau-b)
# ---------------------------------------------------------------------------


def rank_values_average_ties(
    values: Iterable[float],
) -> list[float]:
    """Return average-tie ranks (1-indexed) for ``values``."""
    arr = [float(v) for v in values]
    n = len(arr)
    order = sorted(range(n), key=lambda i: arr[i])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        # ranks i..j are tied; assign average rank
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = float(avg)
        i = j + 1
    return ranks


def _pairwise_finite(
    x: Iterable[float], y: Iterable[float]
) -> tuple[list[float], list[float]]:
    xs = list(x)
    ys = list(y)
    if len(xs) != len(ys):
        raise ValueError(
            "pairwise correlation requires equal-length sequences; "
            f"got len(x)={len(xs)}, len(y)={len(ys)}."
        )
    xf: list[float] = []
    yf: list[float] = []
    for xv, yv in zip(xs, ys):
        if xv is None or yv is None:
            continue
        if isinstance(xv, bool) or isinstance(yv, bool):
            continue
        if not isinstance(xv, (int, float)) or not isinstance(
            yv, (int, float)
        ):
            continue
        xf_v = float(xv)
        yf_v = float(yv)
        if not math.isfinite(xf_v) or not math.isfinite(yf_v):
            continue
        xf.append(xf_v)
        yf.append(yf_v)
    return xf, yf


def pearson_corr(
    x: Iterable[float], y: Iterable[float]
) -> Optional[float]:
    """Pearson correlation, or ``None`` if undefined."""
    xs, ys = _pairwise_finite(x, y)
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((xv - mx) ** 2 for xv in xs)
    syy = sum((yv - my) ** 2 for yv in ys)
    if sxx == 0.0 or syy == 0.0:
        return None
    sxy = sum((xv - mx) * (yv - my) for xv, yv in zip(xs, ys))
    return float(sxy / math.sqrt(sxx * syy))


def spearman_corr(
    x: Iterable[float], y: Iterable[float]
) -> Optional[float]:
    """Spearman rank correlation with average-tie ranks; ``None`` if undefined."""
    xs, ys = _pairwise_finite(x, y)
    if len(xs) < 2:
        return None
    rx = rank_values_average_ties(xs)
    ry = rank_values_average_ties(ys)
    return pearson_corr(rx, ry)


def kendall_tau_b(
    x: Iterable[float], y: Iterable[float]
) -> Optional[float]:
    """Kendall tau-b (tie-corrected); ``None`` if undefined."""
    xs, ys = _pairwise_finite(x, y)
    n = len(xs)
    if n < 2:
        return None
    n_concord = 0
    n_discord = 0
    ties_x_only = 0
    ties_y_only = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_x_only += 1
                continue
            if dy == 0.0:
                ties_y_only += 1
                continue
            if (dx > 0) == (dy > 0):
                n_concord += 1
            else:
                n_discord += 1
    total_pairs = n * (n - 1) // 2
    denom_x = total_pairs - ties_x_only
    denom_y = total_pairs - ties_y_only
    if denom_x <= 0 or denom_y <= 0:
        return None
    denom = math.sqrt(float(denom_x) * float(denom_y))
    if denom == 0.0:
        return None
    return float((n_concord - n_discord) / denom)


# Metric pairs used by compute_metric_correlations. Descriptive,
# never causal.
_CORRELATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("sid", "mmd"),
    ("shd", "mmd"),
    ("edge_count_from_thresholded_adjacency", "mmd"),
    ("sid", "shd"),
)


def compute_metric_correlations(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Compute metric correlations overall and per method_family."""
    rows_t = tuple(rows)
    out: list[dict[str, Any]] = []
    for x_metric, y_metric in _CORRELATION_PAIRS:
        xs = [r[x_metric] for r in rows_t]
        ys = [r[y_metric] for r in rows_t]
        out.append({
            "group_label": "all",
            "method_family": "",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "n": len(_pairwise_finite(xs, ys)[0]),
            "pearson": pearson_corr(xs, ys),
            "spearman": spearman_corr(xs, ys),
            "kendall_tau_b": kendall_tau_b(xs, ys),
        })
    families = sorted({r["method_family"] for r in rows_t})
    for family in families:
        family_rows = [
            r for r in rows_t if r["method_family"] == family
        ]
        for x_metric, y_metric in _CORRELATION_PAIRS:
            xs = [r[x_metric] for r in family_rows]
            ys = [r[y_metric] for r in family_rows]
            xf, _ = _pairwise_finite(xs, ys)
            out.append({
                "group_label": f"method_family:{family}",
                "method_family": family,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n": len(xf),
                "pearson": pearson_corr(xs, ys),
                "spearman": spearman_corr(xs, ys),
                "kendall_tau_b": kendall_tau_b(xs, ys),
            })
    return tuple(out)


# ---------------------------------------------------------------------------
# Degradation slopes (descriptive only)
# ---------------------------------------------------------------------------


def linear_slope(
    x: Iterable[float], y: Iterable[float]
) -> Optional[float]:
    """Least-squares slope of y on x; ``None`` if undefined."""
    xs, ys = _pairwise_finite(x, y)
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((xv - mx) ** 2 for xv in xs)
    if sxx == 0.0:
        return None
    sxy = sum((xv - mx) * (yv - my) for xv, yv in zip(xs, ys))
    return float(sxy / sxx)


def _summarise_slopes(
    slopes: list[Optional[float]],
) -> dict[str, Any]:
    finite = [
        float(s) for s in slopes
        if s is not None and math.isfinite(float(s))
    ]
    n = len(finite)
    if n == 0:
        return {
            "n_seed_slopes": 0,
            "mean_slope": None,
            "std_slope": None,
            "median_slope": None,
            "min_slope": None,
            "max_slope": None,
        }
    return {
        "n_seed_slopes": int(n),
        "mean_slope": float(statistics.fmean(finite)),
        "std_slope": (
            float(statistics.stdev(finite)) if n >= 2 else 0.0
        ),
        "median_slope": float(statistics.median(finite)),
        "min_slope": float(min(finite)),
        "max_slope": float(max(finite)),
    }


def compute_degradation_summary(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Per-seed metric-vs-corruption slopes, then descriptive summaries.

    Descriptive only; no graceful-degradation verdict.
    """
    rows_t = tuple(rows)
    out: list[dict[str, Any]] = []
    # hard_exclusion: one slope per seed per metric.
    hard_by_seed: dict[int, list[dict[str, Any]]] = {}
    for r in rows_t:
        if r["method_family"] == "hard_exclusion":
            hard_by_seed.setdefault(
                int(r["seed_value"]), []
            ).append(r)
    for metric in METRIC_COLUMNS:
        slopes: list[Optional[float]] = []
        for seed in sorted(hard_by_seed):
            members = sorted(
                hard_by_seed[seed],
                key=lambda r: float(r["corruption_fraction"] or 0.0),
            )
            xs = [
                float(m["corruption_fraction"]) for m in members
                if m["corruption_fraction"] is not None
            ]
            ys = [
                m[metric] for m in members
                if m[metric] is not None
                and m["corruption_fraction"] is not None
            ]
            slopes.append(linear_slope(xs, ys))
        stats = _summarise_slopes(slopes)
        out.append({
            "method_family": "hard_exclusion",
            "confidence": None,
            "metric": metric,
            **stats,
        })
    # soft_frobenius: one slope per (confidence, seed) per metric.
    soft_by_conf_seed: dict[
        float, dict[int, list[dict[str, Any]]]
    ] = {}
    for r in rows_t:
        if r["method_family"] == "soft_frobenius":
            cn = (
                None if r["confidence"] is None
                else float(r["confidence"])
            )
            if cn is None:
                continue
            soft_by_conf_seed.setdefault(cn, {}).setdefault(
                int(r["seed_value"]), []
            ).append(r)
    for cn in sorted(soft_by_conf_seed):
        for metric in METRIC_COLUMNS:
            slopes = []
            for seed in sorted(soft_by_conf_seed[cn]):
                members = sorted(
                    soft_by_conf_seed[cn][seed],
                    key=lambda r: float(
                        r["corruption_fraction"] or 0.0
                    ),
                )
                xs = [
                    float(m["corruption_fraction"]) for m in members
                    if m["corruption_fraction"] is not None
                ]
                ys = [
                    m[metric] for m in members
                    if m[metric] is not None
                    and m["corruption_fraction"] is not None
                ]
                slopes.append(linear_slope(xs, ys))
            stats = _summarise_slopes(slopes)
            out.append({
                "method_family": "soft_frobenius",
                "confidence": cn,
                "metric": metric,
                **stats,
            })
    return tuple(out)


# ---------------------------------------------------------------------------
# Forbidden-edge engagement summary
# ---------------------------------------------------------------------------


def compute_forbidden_edge_engagement_summary(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Group M-9a direct engagement columns and summarise per cell."""
    rows_t = tuple(rows)
    grouped: dict[
        tuple[str, Optional[float], Optional[float]],
        list[dict[str, Any]],
    ] = {}
    for r in rows_t:
        key = (
            r["method_family"],
            None if r["corruption_fraction"] is None
            else float(r["corruption_fraction"]),
            None if r["confidence"] is None
            else float(r["confidence"]),
        )
        grouped.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    def sort_key(k: tuple[str, Optional[float], Optional[float]]):
        cf = float("-inf") if k[1] is None else k[1]
        cn = float("-inf") if k[2] is None else k[2]
        return (k[0], cf, cn)
    for key in sorted(grouped.keys(), key=sort_key):
        members = grouped[key]
        targeted_means = [
            m["mean_abs_w_targeted_forbidden_edges"] for m in members
        ]
        targeted_fracs = [
            m["fraction_targeted_forbidden_above_threshold"]
            for m in members
        ]
        non_targeted_means = [
            m["mean_abs_w_non_targeted_edges"] for m in members
        ]
        ecs = [
            m["edge_count_from_thresholded_adjacency"] for m in members
        ]
        ts = summary_stats(targeted_means)
        fs = summary_stats(targeted_fracs)
        ns = summary_stats(non_targeted_means)
        es = summary_stats(ecs)
        out.append({
            "method_family": key[0],
            "corruption_fraction": key[1],
            "confidence": key[2],
            "n": len(members),
            "mean_targeted_abs_w_mean": ts["mean"],
            "std_targeted_abs_w_mean": ts["std"],
            "median_targeted_abs_w_mean": ts["median"],
            "mean_fraction_targeted_above_threshold": fs["mean"],
            "std_fraction_targeted_above_threshold": fs["std"],
            "median_fraction_targeted_above_threshold": fs["median"],
            "mean_non_targeted_abs_w_mean": ns["mean"],
            "std_non_targeted_abs_w_mean": ns["std"],
            "median_non_targeted_abs_w_mean": ns["median"],
            "mean_edge_count": es["mean"],
            "std_edge_count": es["std"],
            "median_edge_count": es["median"],
        })
    return tuple(out)


# ---------------------------------------------------------------------------
# Reference (clean soft) forbidden-edge comparison
# ---------------------------------------------------------------------------


def reference_forbidden_edges_by_seed(
    records: Iterable[MainStudyRunRecord],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Return the seed -> clean-soft forbidden_edges mapping.

    The reference is the soft_frobenius record at corruption=0.0 and
    confidence=1.0; exactly one such record must exist per seed.
    Missing or duplicate references raise ``ValueError``.
    """
    found: dict[int, list[MainStudyRunRecord]] = {}
    for rec in records:
        cfg = rec.config
        if cfg.method_family != "soft_frobenius":
            continue
        if cfg.corrupted_prior_spec is None:
            continue
        if not _is_close(
            float(cfg.corrupted_prior_spec.corruption_fraction), 0.0
        ):
            continue
        if cfg.confidence is None or not _is_close(
            float(cfg.confidence), 1.0
        ):
            continue
        found.setdefault(int(cfg.seed_value), []).append(rec)
    out: dict[int, tuple[tuple[int, int], ...]] = {}
    for seed, recs in found.items():
        if len(recs) != 1:
            raise ValueError(
                "reference_forbidden_edges_by_seed: expected exactly "
                f"one soft_frobenius (corruption=0.0, confidence=1.0) "
                f"reference at seed {seed!r}; got {len(recs)}."
            )
        fe = forbidden_edges_from_record(recs[0])
        if fe is None:
            raise ValueError(
                "reference_forbidden_edges_by_seed: clean-soft "
                f"reference at seed {seed!r} has no forbidden_edges."
            )
        out[int(seed)] = tuple(fe)
    if not out:
        raise ValueError(
            "reference_forbidden_edges_by_seed: no clean-soft "
            "reference found in the supplied records."
        )
    return out


def _flat_lookup_by_run_id(
    flat_rows: Iterable[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {str(r["run_id"]): r for r in flat_rows}


def _reference_comparison_target(
    records: Iterable[MainStudyRunRecord], label: str, seed: int
) -> Optional[MainStudyRunRecord]:
    key = _baseline_condition_key(label)
    family, cf, cn = key
    for rec in records:
        cfg = rec.config
        if cfg.method_family != family:
            continue
        if int(cfg.seed_value) != int(seed):
            continue
        # corruption_fraction check
        rec_cf: Optional[float] = (
            None if cfg.corrupted_prior_spec is None
            else float(cfg.corrupted_prior_spec.corruption_fraction)
        )
        if cf is None:
            if rec_cf is not None:
                continue
        else:
            if rec_cf is None or not _is_close(float(rec_cf), float(cf)):
                continue
        # confidence check
        rec_cn = (
            None if cfg.confidence is None else float(cfg.confidence)
        )
        if cn is None:
            if rec_cn is not None:
                continue
        else:
            if rec_cn is None or not _is_close(float(rec_cn), float(cn)):
                continue
        return rec
    return None


def compute_reference_forbidden_edge_comparison(
    records: Iterable[MainStudyRunRecord],
    flat_rows: Iterable[dict[str, Any]],
    *,
    base_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """Compare four baseline conditions on the clean-soft forbidden-edge set.

    For each seed and each of the four baseline labels:

    - prior_free
    - matched_l1
    - soft_frobenius_clean_conf1
    - hard_exclusion_clean

    load the condition's ``continuous_w.npz`` and compute engagement
    on the seed-specific clean-soft forbidden-edge set. The true
    graph is intentionally not used.
    """
    records_t = tuple(records)
    flat_lookup = _flat_lookup_by_run_id(flat_rows)
    ref_edges_by_seed = reference_forbidden_edges_by_seed(records_t)
    out: list[dict[str, Any]] = []
    for seed in sorted(ref_edges_by_seed):
        ref_edges = ref_edges_by_seed[int(seed)]
        for label in BASELINE_CONDITION_LABELS:
            family, cf, cn = _baseline_condition_key(label)
            rec = _reference_comparison_target(records_t, label, seed)
            if rec is None:
                raise ValueError(
                    "compute_reference_forbidden_edge_comparison: "
                    f"missing record for seed {seed!r}, condition "
                    f"{label!r}."
                )
            cont_w = load_npz_array(
                rec.continuous_w_path,
                base_dir=base_dir,
                expected_key=CONTINUOUS_W_KEY,
            )
            engagement = compute_prior_edge_engagement(
                cont_w, ref_edges, threshold=PROJECT_THRESHOLD,
            )
            thr_arr = load_npz_array(
                rec.thresholded_adjacency_path,
                base_dir=base_dir,
                expected_key=THRESHOLDED_ADJACENCY_KEY,
            )
            edge_count: Optional[int] = (
                None if thr_arr is None
                else off_diagonal_edge_count(thr_arr)
            )
            flat_row = flat_lookup.get(rec.run_id, {})
            out.append({
                "seed_value": int(seed),
                "condition_label": label,
                "method_family": family,
                "corruption_fraction": cf,
                "confidence": cn,
                "n_reference_forbidden_edges": int(len(ref_edges)),
                "mean_abs_w_reference_forbidden_edges": engagement[
                    "mean_abs_w_targeted_forbidden_edges"
                ],
                "fraction_reference_forbidden_above_threshold": engagement[
                    "fraction_targeted_forbidden_above_threshold"
                ],
                "mean_abs_w_reference_non_targeted_edges": engagement[
                    "mean_abs_w_non_targeted_edges"
                ],
                "edge_count_from_thresholded_adjacency": edge_count,
                "sid": flat_row.get("sid"),
                "shd": flat_row.get("shd"),
                "mmd": flat_row.get("mmd"),
            })
    return tuple(out)


# ---------------------------------------------------------------------------
# Per-intervention MMD extraction and summary
# ---------------------------------------------------------------------------


_REQUIRED_INTERVENTION_FIELDS: tuple[str, ...] = (
    "intervention_id",
    "target_node",
    "value_raw",
    "value_model_frame",
    "ground_truth_sampling_seed",
    "model_sampling_seed",
    "n_ground_truth_samples",
    "n_model_samples",
    "mmd_value",
    "mmd_status",
    "bandwidth_used",
    "bandwidth_sweep",
    "sampler_status_for_intervention",
    "sampler_reason",
)

_REQUIRED_BANDWIDTH_KEYS: tuple[str, ...] = ("0.5x", "1.0x", "2.0x")


def extract_per_intervention_mmd(
    record: MainStudyRunRecord, *, base_dir: Path
) -> tuple[dict[str, Any], ...]:
    """Read and flatten per-intervention MMD records from disk.

    The record's ``interventions_mmd_path`` must be non-``None``
    whenever ``metric_status == "computed"``; missing path raises
    ``ValueError``. The artefact JSON must contain every key in
    :data:`_REQUIRED_INTERVENTION_FIELDS` for every record entry,
    and the bandwidth sweep must contain ``"0.5x"``, ``"1.0x"``,
    ``"2.0x"`` keys. Mismatches raise ``ValueError`` naming the
    missing key. MMD values are returned exactly as persisted; no
    recomputation occurs here.
    """
    if (
        record.metric_status == "computed"
        and record.interventions_mmd_path is None
    ):
        raise ValueError(
            "extract_per_intervention_mmd: record "
            f"{record.run_id!r} has metric_status='computed' but "
            "interventions_mmd_path is None."
        )
    if record.interventions_mmd_path is None:
        return ()
    full = resolve_artifact_path(
        record.interventions_mmd_path, base_dir=base_dir
    )
    if full is None:
        return ()
    payload = json.loads(full.read_text(encoding="utf-8"))
    if "records" not in payload or not isinstance(
        payload["records"], list
    ):
        raise ValueError(
            "extract_per_intervention_mmd: payload at "
            f"{record.interventions_mmd_path!r} is missing the "
            "'records' list."
        )
    cfg = record.config
    method_family = cfg.method_family
    seed_value = int(cfg.seed_value)
    cf = (
        None if cfg.corrupted_prior_spec is None
        else float(cfg.corrupted_prior_spec.corruption_fraction)
    )
    cn = (
        None if cfg.confidence is None else float(cfg.confidence)
    )
    out: list[dict[str, Any]] = []
    for entry in payload["records"]:
        for f in _REQUIRED_INTERVENTION_FIELDS:
            if f not in entry:
                raise ValueError(
                    "extract_per_intervention_mmd: intervention "
                    f"record at {record.interventions_mmd_path!r} is "
                    f"missing required field {f!r}."
                )
        sweep = entry["bandwidth_sweep"]
        if not isinstance(sweep, dict):
            raise ValueError(
                "extract_per_intervention_mmd: bandwidth_sweep at "
                f"{record.interventions_mmd_path!r} must be a dict; "
                f"got {type(sweep).__name__}."
            )
        for k in _REQUIRED_BANDWIDTH_KEYS:
            if k not in sweep:
                raise ValueError(
                    "extract_per_intervention_mmd: bandwidth_sweep "
                    f"at {record.interventions_mmd_path!r} is missing "
                    f"required key {k!r}."
                )
        out.append({
            "run_id": record.run_id,
            "method_family": method_family,
            "seed_value": seed_value,
            "corruption_fraction": cf,
            "confidence": cn,
            "intervention_id": str(entry["intervention_id"]),
            "target_node": int(entry["target_node"]),
            "value_raw": (
                None if entry["value_raw"] is None
                else float(entry["value_raw"])
            ),
            "mmd_value": (
                None if entry["mmd_value"] is None
                else float(entry["mmd_value"])
            ),
            "mmd_status": str(entry["mmd_status"]),
            "bandwidth_used": (
                None if entry["bandwidth_used"] is None
                else float(entry["bandwidth_used"])
            ),
            "bandwidth_sweep_0_5x": (
                None if sweep["0.5x"] is None else float(sweep["0.5x"])
            ),
            "bandwidth_sweep_1_0x": (
                None if sweep["1.0x"] is None else float(sweep["1.0x"])
            ),
            "bandwidth_sweep_2_0x": (
                None if sweep["2.0x"] is None else float(sweep["2.0x"])
            ),
            "sampler_status_for_intervention": str(
                entry["sampler_status_for_intervention"]
            ),
            "sampler_reason": (
                None if entry["sampler_reason"] is None
                else str(entry["sampler_reason"])
            ),
        })
    return tuple(out)


def compute_per_intervention_mmd_summary(
    per_intervention_rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Group per-intervention MMD long rows and summarise mmd_value.

    Only rows whose ``mmd_status`` indicates an available value and
    whose ``mmd_value`` is finite are summarised.
    """
    grouped: dict[
        tuple[
            str, Optional[float], Optional[float],
            str, int, Optional[float],
        ],
        list[float],
    ] = {}
    for r in per_intervention_rows:
        status = str(r.get("mmd_status", ""))
        if status not in ("available", "computed"):
            continue
        v = r.get("mmd_value")
        if v is None or not _is_finite_real(v):
            continue
        key = (
            str(r["method_family"]),
            None if r["corruption_fraction"] is None
            else float(r["corruption_fraction"]),
            None if r["confidence"] is None
            else float(r["confidence"]),
            str(r["intervention_id"]),
            int(r["target_node"]),
            None if r["value_raw"] is None
            else float(r["value_raw"]),
        )
        grouped.setdefault(key, []).append(float(v))
    out: list[dict[str, Any]] = []
    def sort_key(k):
        cf = float("-inf") if k[1] is None else k[1]
        cn = float("-inf") if k[2] is None else k[2]
        vr = float("-inf") if k[5] is None else k[5]
        return (k[0], cf, cn, k[3], k[4], vr)
    for key in sorted(grouped.keys(), key=sort_key):
        values = grouped[key]
        stats = summary_stats(values)
        out.append({
            "method_family": key[0],
            "corruption_fraction": key[1],
            "confidence": key[2],
            "intervention_id": key[3],
            "target_node": key[4],
            "value_raw": key[5],
            "n": stats["n"],
            "mean_mmd": stats["mean"],
            "std_mmd": stats["std"],
            "median_mmd": stats["median"],
            "min_mmd": stats["min"],
            "max_mmd": stats["max"],
        })
    return tuple(out)


# ---------------------------------------------------------------------------
# Generic dict-rows CSV writer
# ---------------------------------------------------------------------------


def write_dict_rows_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    """Write ``rows`` to ``path`` with the given column order.

    None becomes an empty cell, matching the M-9a CSV null convention.
    Unknown fields in any row raise ``ValueError`` naming the row.
    Missing expected fields are written as empty cells.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fn_set = set(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(fieldnames))
        for r in rows:
            extras = set(r.keys()) - fn_set
            if extras:
                raise ValueError(
                    "write_dict_rows_csv: row contains unexpected "
                    f"keys {sorted(extras)} (allowed: "
                    f"{sorted(fn_set)})."
                )
            writer.writerow([
                _csv_cell(r.get(col, None)) for col in fieldnames
            ])


# Canonical M-9b column orders.
_BASELINE_COMPARISON_COLUMNS: tuple[str, ...] = (
    "condition_label",
    "method_family",
    "corruption_fraction",
    "confidence",
    "metric",
    "n", "mean", "std", "median", "min", "max",
)

_PAIRED_COMPARISON_COLUMNS: tuple[str, ...] = (
    "label_a", "label_b", "metric", "diff_convention", "n_pairs",
    "mean_a", "mean_b", "mean_diff", "median_diff",
    "bootstrap_ci_low", "bootstrap_ci_high",
    "wins_a_lower", "wins_b_lower", "ties",
    "sign_test_p_two_sided", "bh_q_value",
)

_METRIC_CORRELATIONS_COLUMNS: tuple[str, ...] = (
    "group_label", "method_family", "x_metric", "y_metric", "n",
    "pearson", "spearman", "kendall_tau_b",
)

_DEGRADATION_COLUMNS: tuple[str, ...] = (
    "method_family", "confidence", "metric", "n_seed_slopes",
    "mean_slope", "std_slope", "median_slope",
    "min_slope", "max_slope",
)

_FORBIDDEN_ENGAGEMENT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "method_family", "corruption_fraction", "confidence", "n",
    "mean_targeted_abs_w_mean", "std_targeted_abs_w_mean",
    "median_targeted_abs_w_mean",
    "mean_fraction_targeted_above_threshold",
    "std_fraction_targeted_above_threshold",
    "median_fraction_targeted_above_threshold",
    "mean_non_targeted_abs_w_mean", "std_non_targeted_abs_w_mean",
    "median_non_targeted_abs_w_mean",
    "mean_edge_count", "std_edge_count", "median_edge_count",
)

_REFERENCE_FORBIDDEN_COMPARISON_COLUMNS: tuple[str, ...] = (
    "seed_value", "condition_label", "method_family",
    "corruption_fraction", "confidence",
    "n_reference_forbidden_edges",
    "mean_abs_w_reference_forbidden_edges",
    "fraction_reference_forbidden_above_threshold",
    "mean_abs_w_reference_non_targeted_edges",
    "edge_count_from_thresholded_adjacency",
    "sid", "shd", "mmd",
)

_PER_INTERVENTION_MMD_LONG_COLUMNS: tuple[str, ...] = (
    "run_id", "method_family", "seed_value",
    "corruption_fraction", "confidence",
    "intervention_id", "target_node", "value_raw",
    "mmd_value", "mmd_status", "bandwidth_used",
    "bandwidth_sweep_0_5x", "bandwidth_sweep_1_0x",
    "bandwidth_sweep_2_0x",
    "sampler_status_for_intervention", "sampler_reason",
)

_PER_INTERVENTION_MMD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "method_family", "corruption_fraction", "confidence",
    "intervention_id", "target_node", "value_raw",
    "n", "mean_mmd", "std_mmd", "median_mmd",
    "min_mmd", "max_mmd",
)


def write_statistics_summary_json(
    summary: dict[str, Any], path: Path
) -> None:
    """Write the M-9b statistics summary as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_hypothesis_statistics(
    *,
    output_root: Path,
    main_evaluation_run_hash12: str,
    n_bootstrap: int = 10000,
    random_seed: int = 90210,
) -> dict[str, Any]:
    """Compute and write all M-9b outputs into the existing readout dir.

    Reads the M-9a flat CSV plus the original record JSONs (and the
    referenced continuous-W / interventions-MMD artefacts). Writes
    eight new CSV files plus the statistics summary JSON. Does not
    plot, rank methods, or write hypothesis verdicts.
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
    readout_dir = main_evaluation_readout_dir(
        output_root, main_evaluation_run_hash12
    )
    flat_csv_path = readout_dir / FLAT_RECORDS_CSV
    if not flat_csv_path.exists():
        raise FileNotFoundError(
            "generate_hypothesis_statistics: the M-9a flat CSV "
            f"{flat_csv_path!r} does not exist; run "
            "generate_readout_foundation first."
        )
    flat_rows = load_flat_records_csv(flat_csv_path)

    records_dir = main_evaluation_records_dir(
        output_root, main_evaluation_run_hash12
    )
    records = load_main_evaluation_records(records_dir)

    baseline_rows = compute_baseline_comparison(flat_rows)
    paired_rows = compute_paired_seed_comparisons(
        flat_rows,
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
    )
    correlation_rows = compute_metric_correlations(flat_rows)
    degradation_rows = compute_degradation_summary(flat_rows)
    engagement_summary_rows = (
        compute_forbidden_edge_engagement_summary(flat_rows)
    )
    reference_comparison_rows = (
        compute_reference_forbidden_edge_comparison(
            records, flat_rows, base_dir=output_root,
        )
    )
    per_intervention_long_rows: list[dict[str, Any]] = []
    for rec in records:
        per_intervention_long_rows.extend(
            extract_per_intervention_mmd(rec, base_dir=output_root)
        )
    per_intervention_long_rows_t = tuple(per_intervention_long_rows)
    per_intervention_summary_rows = (
        compute_per_intervention_mmd_summary(
            per_intervention_long_rows_t
        )
    )

    readout_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        BASELINE_COMPARISON_CSV: readout_dir / BASELINE_COMPARISON_CSV,
        PAIRED_SEED_COMPARISONS_CSV: (
            readout_dir / PAIRED_SEED_COMPARISONS_CSV
        ),
        METRIC_CORRELATIONS_CSV: readout_dir / METRIC_CORRELATIONS_CSV,
        DEGRADATION_SUMMARY_CSV: readout_dir / DEGRADATION_SUMMARY_CSV,
        FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV: (
            readout_dir / FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV
        ),
        REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV: (
            readout_dir / REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV
        ),
        PER_INTERVENTION_MMD_LONG_CSV: (
            readout_dir / PER_INTERVENTION_MMD_LONG_CSV
        ),
        PER_INTERVENTION_MMD_SUMMARY_CSV: (
            readout_dir / PER_INTERVENTION_MMD_SUMMARY_CSV
        ),
        STATISTICS_SUMMARY_JSON: readout_dir / STATISTICS_SUMMARY_JSON,
    }
    write_dict_rows_csv(
        baseline_rows,
        output_paths[BASELINE_COMPARISON_CSV],
        _BASELINE_COMPARISON_COLUMNS,
    )
    write_dict_rows_csv(
        paired_rows,
        output_paths[PAIRED_SEED_COMPARISONS_CSV],
        _PAIRED_COMPARISON_COLUMNS,
    )
    write_dict_rows_csv(
        correlation_rows,
        output_paths[METRIC_CORRELATIONS_CSV],
        _METRIC_CORRELATIONS_COLUMNS,
    )
    write_dict_rows_csv(
        degradation_rows,
        output_paths[DEGRADATION_SUMMARY_CSV],
        _DEGRADATION_COLUMNS,
    )
    write_dict_rows_csv(
        engagement_summary_rows,
        output_paths[FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV],
        _FORBIDDEN_ENGAGEMENT_SUMMARY_COLUMNS,
    )
    write_dict_rows_csv(
        reference_comparison_rows,
        output_paths[REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV],
        _REFERENCE_FORBIDDEN_COMPARISON_COLUMNS,
    )
    write_dict_rows_csv(
        per_intervention_long_rows_t,
        output_paths[PER_INTERVENTION_MMD_LONG_CSV],
        _PER_INTERVENTION_MMD_LONG_COLUMNS,
    )
    write_dict_rows_csv(
        per_intervention_summary_rows,
        output_paths[PER_INTERVENTION_MMD_SUMMARY_CSV],
        _PER_INTERVENTION_MMD_SUMMARY_COLUMNS,
    )
    summary: dict[str, Any] = {
        "main_evaluation_run_hash12": main_evaluation_run_hash12,
        "input_flat_csv": str(
            flat_csv_path.relative_to(output_root)
        ).replace("\\", "/"),
        "output_files": sorted(output_paths.keys()),
        "n_flat_rows": int(len(flat_rows)),
        "n_baseline_rows": int(len(baseline_rows)),
        "n_paired_comparison_rows": int(len(paired_rows)),
        "n_correlation_rows": int(len(correlation_rows)),
        "n_degradation_rows": int(len(degradation_rows)),
        "n_forbidden_engagement_rows": int(
            len(engagement_summary_rows)
        ),
        "n_reference_forbidden_rows": int(
            len(reference_comparison_rows)
        ),
        "n_per_intervention_mmd_rows": int(
            len(per_intervention_long_rows_t)
        ),
        "n_per_intervention_mmd_summary_rows": int(
            len(per_intervention_summary_rows)
        ),
        "no_plots_created": True,
        "no_notebook_created": True,
        "no_hypothesis_verdicts": True,
    }
    write_statistics_summary_json(
        summary, output_paths[STATISTICS_SUMMARY_JSON]
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
            "summary, forbidden-edge engagement) and the M-9b "
            "hypothesis-statistics tables. Read-only over experiment "
            "records; no fitting, no metric recomputation, no plots, "
            "no notebooks, no hypothesis verdicts."
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
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help=(
            "Number of bootstrap replicates used for paired-difference "
            "confidence intervals. Default 10000."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=90210,
        help=(
            "Deterministic seed for the paired bootstrap. Default "
            "90210."
        ),
    )
    parser.add_argument(
        "--skip-hypothesis-statistics",
        action="store_true",
        help=(
            "If set, skip the M-9b hypothesis-statistics stage and "
            "produce only the M-9a foundation outputs."
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
    if not args.skip_hypothesis_statistics:
        try:
            generate_hypothesis_statistics(
                output_root=args.output_root,
                main_evaluation_run_hash12=(
                    args.main_evaluation_run_hash12
                ),
                n_bootstrap=int(args.n_bootstrap),
                random_seed=int(args.random_seed),
            )
        except SystemExit:
            raise
        except BaseException as exc:
            sys.stderr.write(
                "readout: hypothesis-statistics error: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return _EXIT_ERROR
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BASELINE_COMPARISON_CSV",
    "BASELINE_CONDITION_LABELS",
    "CELL_SUMMARY_CSV",
    "CONTINUOUS_W_KEY",
    "DEGRADATION_SUMMARY_CSV",
    "DIFF_CONVENTION",
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
    "FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV",
    "FlatRecordRow",
    "METRIC_COLUMNS",
    "METRIC_CORRELATIONS_CSV",
    "PAIRED_COMPARISON_PAIRS",
    "PAIRED_SEED_COMPARISONS_CSV",
    "PER_INTERVENTION_MMD_LONG_CSV",
    "PER_INTERVENTION_MMD_SUMMARY_CSV",
    "PROJECT_THRESHOLD",
    "REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV",
    "STATISTICS_SUMMARY_JSON",
    "STATUS_SUMMARY_CSV",
    "THRESHOLDED_ADJACENCY_KEY",
    "VALIDATION_SUMMARY_JSON",
    "ValidationSummary",
    "compute_baseline_comparison",
    "compute_degradation_summary",
    "compute_forbidden_edge_engagement_summary",
    "compute_metric_correlations",
    "compute_paired_seed_comparisons",
    "compute_per_intervention_mmd_summary",
    "compute_prior_edge_engagement",
    "compute_reference_forbidden_edge_comparison",
    "condition_key",
    "extract_per_intervention_mmd",
    "flatten_record",
    "forbidden_edges_from_record",
    "generate_hypothesis_statistics",
    "generate_readout_foundation",
    "kendall_tau_b",
    "linear_slope",
    "load_flat_records_csv",
    "load_main_evaluation_records",
    "load_npz_array",
    "main",
    "main_evaluation_readout_dir",
    "main_evaluation_records_dir",
    "off_diagonal_edge_count",
    "paired_seed_difference",
    "pearson_corr",
    "rank_values_average_ties",
    "record_config_value",
    "reference_forbidden_edges_by_seed",
    "resolve_artifact_path",
    "select_condition",
    "spearman_corr",
    "summary_stats",
    "validate_flat_rows",
    "write_cell_summary_csv",
    "write_dict_rows_csv",
    "write_flat_records_csv",
    "write_forbidden_edge_engagement_csv",
    "write_statistics_summary_json",
    "write_status_summary_csv",
    "write_validation_summary_json",
]
