"""Matched-L1 calibration script for the main-study pipeline.

Selects a single global ``matched_l1_lambda1`` by sparsity matching
against the soft-prior target condition on calibration seeds only.
Evaluation seeds are never used. SID, SHD, MMD, runtime, and
diagnostic-grid anomalies must not influence the selected value.

The script:

- enumerates the soft-prior diagnostic grid plus the matched-L1
  Stage-1 candidates as one combined workload, executes them through
  the injected runner in ``"skip"`` mode, and persists the resulting
  per-run records and artefacts via the standard run-I/O layer;
- loads the persisted records, derives one row per calibration run
  with the off-diagonal edge count read from the persisted
  thresholded adjacency, and computes per-candidate sparsity
  summaries on the calibration-seed pair;
- ranks Stage-1 candidates by valid-DAG count then absolute edge-count
  gap, generates exactly five evenly spaced Stage-2 candidates within
  the rule-determined refinement interval, skips Stage-1 duplicates,
  runs the remaining Stage-2 candidates, and selects a final value
  over the Stage-1 plus new Stage-2 union;
- writes a Stage-1 intermediate JSON, a final summary JSON, a flat
  per-run CSV, and a human-readable readout under
  ``results/main_study/calibration/matched_l1/<calibration_run_hash12>/``.

The script never modifies the decision log. It exits with status
``0`` on completed selection, ``2`` on no-valid-DAG halt,
``3`` on outward-boundary poor-match halt, and ``1`` on argparse or
unexpected errors.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from experiments.main_study.backends import DAGMAConfig
from experiments.main_study.records import (
    MainStudyRunRecord,
    record_from_json,
)
from experiments.main_study.run_io import (
    load_existing_record,
    resolve_parent_hash_from_prefix,
    resolve_relative_path,
    validate_parent_hash_full,
)
from experiments.main_study.runner import (
    RunSummary,
    run_main_study,
)
from experiments.main_study.schema import (
    CONFIDENCE_GRID,
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    PROTOCOL_DAGMA_LAMBDA1,
    PROTOCOL_DAGMA_MAX_ITER,
    PROTOCOL_DAGMA_WARM_ITER,
    MainStudyConfig,
    build_protocol_dagma_config,
    canonicalize_for_json,
    make_main_study_config,
)
from experiments.main_study.priors import CORRUPTION_GRID
from experiments.main_study.workloads import (
    PlannedRun,
    enumerate_planned_runs,
)


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


CALIBRATION_PROTOCOL_VERSION: str = "matched_l1_v1"

CALIBRATION_SEED_VALUES: tuple[int, ...] = (401, 402)

STAGE_1_CANDIDATES: tuple[float, ...] = (
    0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
)

TARGET_CORRUPTION_FRACTION: float = 0.0
TARGET_CONFIDENCE: float = 1.0
CLOSE_MATCH_EDGE_TOLERANCE: float = 1.0

HALT_COMPLETED: str = "completed"
HALT_NO_VALID_DAG: str = "halt_no_valid_dag"
HALT_BOUNDARY_POOR_MATCH: str = "halt_boundary_poor_match"

STAGE_1_LOWER_BOUNDARY_INTERVAL: tuple[float, float] = (0.0125, 0.05)
STAGE_1_UPPER_BOUNDARY_INTERVAL: tuple[float, float] = (0.2, 0.3)

STAGE_2_NUM_VALUES: int = 5

STAGE_TAG_DIAGNOSTIC: str = "diagnostic_soft"
STAGE_TAG_STAGE_1: str = "stage_1"
STAGE_TAG_STAGE_2: str = "stage_2"

DEFAULT_N_NODES: int = 10
DEFAULT_EXPECTED_EDGES: int = 20

SELECTION_RULE: str = (
    "Exclude candidates with zero valid-DAG fits. Among remaining, "
    "prefer maximum valid-DAG count, then minimum absolute edge-count "
    "gap, then smaller lambda1."
)

CALIBRATION_OUTPUT_SUBDIR: str = "results/main_study/calibration/matched_l1"

STAGE_1_INTERMEDIATE_FILENAME: str = "matched_l1_stage1_intermediate_summary.json"
SUMMARY_FILENAME: str = "matched_l1_calibration_summary.json"
TABLE_FILENAME: str = "matched_l1_calibration_table.csv"
READOUT_FILENAME: str = "matched_l1_calibration_readout.md"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CalibrationRunSpec:
    """Identity of one matched-L1 calibration run."""

    parent_heldout_run_hash_full: str
    calibration_run_hash12: str
    output_dir_relative: str
    code_version: Optional[str]
    protocol_version: str = CALIBRATION_PROTOCOL_VERSION


@dataclass(frozen=True, kw_only=True)
class CandidateRunRow:
    """One persisted-record row used for matched-L1 sparsity matching."""

    candidate_lambda1: Optional[float]
    seed: int
    stage: str
    method_family: str
    corruption_fraction: Optional[float]
    confidence: Optional[float]
    edge_count: Optional[int]
    fit_status: str
    graph_status: Optional[str]
    sampler_status: Optional[str]
    metric_status: str
    record_path: str
    configuration_hash_full: str
    configuration_hash_prefix: str


@dataclass(frozen=True, kw_only=True)
class CandidateSummary:
    """Per-candidate sparsity summary across calibration seeds."""

    candidate_lambda1: float
    stage: str
    per_seed_edge_counts: tuple[Optional[int], ...]
    per_seed_valid_dag: tuple[bool, ...]
    valid_dag_count: int
    mean_edge_count: Optional[float]
    absolute_gap: Optional[float]
    fragile_valid_dag: bool


@dataclass(frozen=True, kw_only=True)
class MatchedL1CalibrationSummary:
    """Final matched-L1 calibration result for review."""

    halt_status: str
    parent_heldout_run_hash_full: str
    calibration_run_hash12: str
    output_dir_relative: str
    code_version: Optional[str]
    target_mean_edge_count: Optional[float]
    target_per_seed_edge_counts: tuple[Optional[int], ...]
    stage_1_candidates: tuple[float, ...]
    stage_2_interval: Optional[tuple[float, float]]
    stage_2_generated_candidates: tuple[float, ...]
    stage_2_candidates: tuple[float, ...]
    stage_2_skipped_duplicates: tuple[float, ...]
    all_evaluated_candidates: tuple[float, ...]
    selected_lambda1: Optional[float]
    selected_candidate_mean_edge_count: Optional[float]
    selected_absolute_gap: Optional[float]
    selected_valid_dag_count: Optional[int]
    within_one_edge_tolerance: Optional[bool]
    diagnostic_grid_anomalies: tuple[str, ...]
    selection_rule: str = SELECTION_RULE


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def default_utc_factory() -> str:
    """Return the current UTC instant as an ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_code_version() -> Optional[str]:
    """Capture the current git commit hash; return ``None`` on failure.

    Calls ``git rev-parse HEAD`` once with a short timeout. Any
    subprocess error, missing-executable error, non-zero exit, or
    empty stdout yields ``None``. The result is provenance only; it
    must never participate in scientific selection.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out if out else None


def compute_calibration_run_hash12(
    *,
    parent_full_hash: str,
    protocol_version: str = CALIBRATION_PROTOCOL_VERSION,
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEED_VALUES,
    stage_1_candidates: tuple[float, ...] = STAGE_1_CANDIDATES,
    target_corruption_fraction: float = TARGET_CORRUPTION_FRACTION,
    target_confidence: float = TARGET_CONFIDENCE,
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID,
    corruption_grid: tuple[float, ...] = CORRUPTION_GRID,
    protocol_dagma_lambda1: float = PROTOCOL_DAGMA_LAMBDA1,
    protocol_dagma_warm_iter: int = PROTOCOL_DAGMA_WARM_ITER,
    protocol_dagma_max_iter: int = PROTOCOL_DAGMA_MAX_ITER,
) -> str:
    """Deterministic 12-char hex hash from scientific protocol identity.

    ``code_version`` is intentionally not included. The hash covers
    only inputs that change the experimental contract: protocol
    version, parent provenance, calibration seeds, Stage-1 grid,
    target soft-prior cell, the diagnostic confidence/corruption
    grid, and the DAGMA backbone hyperparameters that define the
    soft-prior reference baseline (``lambda1``, ``warm_iter``,
    ``max_iter``).
    """
    payload = {
        "protocol_version": protocol_version,
        "parent_full_hash": parent_full_hash,
        "calibration_seeds": [int(s) for s in calibration_seeds],
        "stage_1_candidates": [float(x) for x in stage_1_candidates],
        "target_corruption_fraction": float(target_corruption_fraction),
        "target_confidence": float(target_confidence),
        "diagnostic_confidence_grid": [float(x) for x in confidence_grid],
        "diagnostic_corruption_grid": [float(x) for x in corruption_grid],
        "protocol_dagma_lambda1": float(protocol_dagma_lambda1),
        "protocol_dagma_warm_iter": int(protocol_dagma_warm_iter),
        "protocol_dagma_max_iter": int(protocol_dagma_max_iter),
    }
    serialised = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Edge-count helpers
# ---------------------------------------------------------------------------


def off_diagonal_edge_count(adjacency: np.ndarray) -> int:
    """Count off-diagonal ``True`` entries in a bool-like square ndarray.

    Self-loops on the diagonal are excluded. The input must be a
    2D square ndarray with bool or integer dtype; floats are
    rejected to keep this operation strictly defined on a thresholded
    boolean matrix.
    """
    if not isinstance(adjacency, np.ndarray):
        raise TypeError(
            "off_diagonal_edge_count requires a numpy ndarray; got "
            f"{type(adjacency).__name__}."
        )
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "off_diagonal_edge_count requires a 2D square array; got "
            f"shape {adjacency.shape}."
        )
    if adjacency.dtype.kind not in "bi":
        raise ValueError(
            "off_diagonal_edge_count requires a bool-like or integer "
            f"array (no floats); got dtype {adjacency.dtype}."
        )
    a = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(a, False)
    return int(a.sum())


def row_from_record(
    *,
    record: MainStudyRunRecord,
    record_path: str,
    stage: str,
    candidate_lambda1: Optional[float],
    base_dir: Path,
) -> CandidateRunRow:
    """Build one :class:`CandidateRunRow` from a persisted record.

    ``edge_count`` is read from the persisted ``thresholded_adjacency.npz``
    only; the continuous-W path is never inspected and the threshold
    is never re-applied. When the record has no thresholded-adjacency
    artefact (e.g. a fit failure), ``edge_count`` is ``None``.
    """
    edge_count: Optional[int] = None
    if record.thresholded_adjacency_path is not None:
        full = resolve_relative_path(
            record.thresholded_adjacency_path, base_dir=base_dir
        )
        with np.load(full) as data:
            if "thresholded_adjacency" not in data.files:
                raise ValueError(
                    "thresholded_adjacency.npz must contain key "
                    "'thresholded_adjacency'; record path "
                    f"{record.thresholded_adjacency_path!r}."
                )
            adjacency = data["thresholded_adjacency"]
        edge_count = off_diagonal_edge_count(adjacency)
    corruption_fraction: Optional[float] = None
    if record.config.corrupted_prior_spec is not None:
        corruption_fraction = float(
            record.config.corrupted_prior_spec.corruption_fraction
        )
    confidence: Optional[float] = None
    if record.config.confidence is not None:
        confidence = float(record.config.confidence)
    return CandidateRunRow(
        candidate_lambda1=(
            None
            if candidate_lambda1 is None
            else float(candidate_lambda1)
        ),
        seed=int(record.config.seed_value),
        stage=stage,
        method_family=record.config.method_family,
        corruption_fraction=corruption_fraction,
        confidence=confidence,
        edge_count=edge_count,
        fit_status=record.fit_status,
        graph_status=record.graph_status,
        sampler_status=record.sampler_status,
        metric_status=record.metric_status,
        record_path=record_path,
        configuration_hash_full=record.configuration_hash_full,
        configuration_hash_prefix=record.configuration_hash_prefix,
    )


# ---------------------------------------------------------------------------
# Target selection and per-candidate summaries
# ---------------------------------------------------------------------------


def _is_close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9, rel_tol=0.0)


def select_target_rows(
    rows: tuple[CandidateRunRow, ...]
) -> tuple[CandidateRunRow, ...]:
    """Return the two soft-prior target rows (corruption=0.0, confidence=1.0).

    Raises ``ValueError`` if exactly two such rows are not present.
    """
    selected = tuple(
        r for r in rows
        if r.method_family == "soft_frobenius"
        and r.corruption_fraction is not None
        and r.confidence is not None
        and _is_close(r.corruption_fraction, TARGET_CORRUPTION_FRACTION)
        and _is_close(r.confidence, TARGET_CONFIDENCE)
    )
    if len(selected) != 2:
        raise ValueError(
            "exactly two soft-prior target rows are required "
            f"(corruption={TARGET_CORRUPTION_FRACTION}, "
            f"confidence={TARGET_CONFIDENCE}); got {len(selected)}."
        )
    return selected


def _seed_sorted_pair(
    rows: tuple[CandidateRunRow, ...]
) -> tuple[CandidateRunRow, ...]:
    """Return ``rows`` sorted by seed ascending."""
    return tuple(sorted(rows, key=lambda r: r.seed))


def target_mean_and_per_seed(
    target_rows: tuple[CandidateRunRow, ...]
) -> tuple[float, tuple[Optional[int], ...]]:
    """Mean and per-seed edge counts for the soft-prior target rows.

    The mean is computed over rows whose ``edge_count`` is not
    ``None``. If both target rows have no edge_count, the mean is
    raised as a ``ValueError`` because the target is undefined.
    """
    sorted_rows = _seed_sorted_pair(target_rows)
    per_seed: tuple[Optional[int], ...] = tuple(
        r.edge_count for r in sorted_rows
    )
    defined = [v for v in per_seed if v is not None]
    if not defined:
        raise ValueError(
            "soft-prior target rows have no defined edge_count; cannot "
            "compute a target mean."
        )
    mean = float(sum(defined)) / float(len(defined))
    return mean, per_seed


def summarise_candidate(
    *,
    candidate_lambda1: float,
    rows: tuple[CandidateRunRow, ...],
    stage: str,
    target_mean: float,
) -> CandidateSummary:
    """Reduce a candidate's per-seed rows into a :class:`CandidateSummary`.

    A row is counted as valid-DAG iff ``graph_status == "valid_dag"``.
    Mean edge count is computed over rows with a defined ``edge_count``;
    the result is ``None`` if no row has one.
    """
    sorted_rows = _seed_sorted_pair(rows)
    per_seed_edge: tuple[Optional[int], ...] = tuple(
        r.edge_count for r in sorted_rows
    )
    per_seed_valid: tuple[bool, ...] = tuple(
        r.graph_status == "valid_dag" for r in sorted_rows
    )
    defined = [v for v in per_seed_edge if v is not None]
    mean: Optional[float] = None
    gap: Optional[float] = None
    if defined:
        mean = float(sum(defined)) / float(len(defined))
        gap = abs(mean - float(target_mean))
    valid_count = sum(1 for v in per_seed_valid if v)
    fragile = (
        valid_count == 1 and len(per_seed_valid) == 2
    )
    return CandidateSummary(
        candidate_lambda1=float(candidate_lambda1),
        stage=stage,
        per_seed_edge_counts=per_seed_edge,
        per_seed_valid_dag=per_seed_valid,
        valid_dag_count=int(valid_count),
        mean_edge_count=mean,
        absolute_gap=gap,
        fragile_valid_dag=fragile,
    )


# ---------------------------------------------------------------------------
# Ranking / selection
# ---------------------------------------------------------------------------


def rank_candidates(
    summaries: tuple[CandidateSummary, ...]
) -> tuple[CandidateSummary, ...]:
    """Return ``summaries`` ordered by the selection hierarchy.

    Exclude candidates with zero valid-DAG fits, prefer the maximum
    valid-DAG count, break ties by the smallest absolute edge-count
    gap, and finally by the smaller ``candidate_lambda1``. Candidates
    excluded for having no valid fit do not appear in the returned
    tuple.
    """
    eligible = tuple(
        s for s in summaries if s.valid_dag_count > 0
    )
    return tuple(
        sorted(
            eligible,
            key=lambda s: (
                -int(s.valid_dag_count),
                float("inf") if s.absolute_gap is None else float(s.absolute_gap),
                float(s.candidate_lambda1),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Stage-2 interval and candidate generation
# ---------------------------------------------------------------------------


def stage_2_interval_for_winner(
    winner_lambda1: float,
    *,
    stage_1: tuple[float, ...] = STAGE_1_CANDIDATES,
) -> tuple[float, float]:
    """Return the Stage-2 refinement interval for a Stage-1 winner.

    Internal winners refine between their immediate Stage-1
    neighbours. Lower-boundary winners refine over
    :data:`STAGE_1_LOWER_BOUNDARY_INTERVAL`. Upper-boundary winners
    refine over :data:`STAGE_1_UPPER_BOUNDARY_INTERVAL`. The winner
    must equal one of the Stage-1 candidates to within numeric
    tolerance.
    """
    sorted_stage_1 = tuple(sorted(float(x) for x in stage_1))
    match_idx: Optional[int] = None
    for i, v in enumerate(sorted_stage_1):
        if _is_close(float(winner_lambda1), v):
            match_idx = i
            break
    if match_idx is None:
        raise ValueError(
            f"winner_lambda1={winner_lambda1!r} is not a Stage-1 "
            f"candidate in {sorted_stage_1}."
        )
    if match_idx == 0:
        return STAGE_1_LOWER_BOUNDARY_INTERVAL
    if match_idx == len(sorted_stage_1) - 1:
        return STAGE_1_UPPER_BOUNDARY_INTERVAL
    return (
        sorted_stage_1[match_idx - 1],
        sorted_stage_1[match_idx + 1],
    )


def generate_stage_2_candidates(
    interval: tuple[float, float]
) -> tuple[float, ...]:
    """Generate exactly five evenly spaced values inclusive of endpoints."""
    lo, hi = float(interval[0]), float(interval[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(
            f"interval endpoints must be finite; got ({lo}, {hi})."
        )
    if hi <= lo:
        raise ValueError(
            "interval must satisfy hi > lo; got "
            f"({lo}, {hi})."
        )
    step = (hi - lo) / float(STAGE_2_NUM_VALUES - 1)
    return tuple(
        lo + step * i for i in range(STAGE_2_NUM_VALUES)
    )


def split_stage_2_duplicates(
    generated: tuple[float, ...],
    *,
    stage_1: tuple[float, ...] = STAGE_1_CANDIDATES,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Partition ``generated`` into ``(new, skipped_duplicates)``.

    The split preserves the input order. A generated value is a
    duplicate if it is numerically close to any Stage-1 value.
    """
    new: list[float] = []
    skipped: list[float] = []
    for v in generated:
        if any(_is_close(float(v), float(s)) for s in stage_1):
            skipped.append(float(v))
        else:
            new.append(float(v))
    return tuple(new), tuple(skipped)


# ---------------------------------------------------------------------------
# Diagnostic anomaly detection
# ---------------------------------------------------------------------------


def compute_diagnostic_anomalies(
    rows: tuple[CandidateRunRow, ...]
) -> tuple[str, ...]:
    """Surface advisory soft-prior anomalies. Never alters selection.

    Returns a tuple of human-readable anomaly strings. Two checks
    are applied:

    - mean edge_count per cell less than 1 (sparsity collapse);
    - confidence axis non-monotonic at fixed corruption (neither
      non-increasing nor non-decreasing across the configured
      confidence grid).
    """
    soft_rows = tuple(
        r for r in rows
        if r.method_family == "soft_frobenius"
        and r.corruption_fraction is not None
        and r.confidence is not None
        and r.edge_count is not None
    )
    if not soft_rows:
        return ()
    cells: dict[tuple[float, float], list[int]] = {}
    for r in soft_rows:
        key = (float(r.corruption_fraction), float(r.confidence))
        cells.setdefault(key, []).append(int(r.edge_count))
    anomalies: list[str] = []
    for (cf, cn), counts in sorted(cells.items()):
        mean = float(sum(counts)) / float(len(counts))
        if mean < 1.0:
            anomalies.append(
                f"sparsity_collapse: corruption={cf}, confidence={cn}, "
                f"mean_edge_count={mean:.3f}"
            )
    by_corruption: dict[float, list[tuple[float, float]]] = {}
    for (cf, cn), counts in cells.items():
        mean = float(sum(counts)) / float(len(counts))
        by_corruption.setdefault(float(cf), []).append((float(cn), mean))
    for cf, entries in sorted(by_corruption.items()):
        entries_sorted = sorted(entries, key=lambda x: x[0])
        if len(entries_sorted) < 3:
            continue
        means = [e[1] for e in entries_sorted]
        increasing = all(
            means[i] <= means[i + 1] for i in range(len(means) - 1)
        )
        decreasing = all(
            means[i] >= means[i + 1] for i in range(len(means) - 1)
        )
        if not (increasing or decreasing):
            anomalies.append(
                f"non_monotonic_confidence: corruption={cf}, "
                f"mean_edge_counts_by_confidence={means}"
            )
    return tuple(anomalies)


# ---------------------------------------------------------------------------
# Workload enumeration
# ---------------------------------------------------------------------------


def _enumerate_diagnostic_planned(
    *,
    calibration_run_hash12: str,
    parent_full_hash: str,
    base_dagma_config: DAGMAConfig,
    n_nodes: int,
    expected_edges: int,
) -> tuple[PlannedRun, ...]:
    return enumerate_planned_runs(
        main_study_run_hash12=calibration_run_hash12,
        seed_population="main_calibration",
        seed_values=CALIBRATION_SEED_VALUES,
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=parent_full_hash,
        method_families=("soft_frobenius",),
        n_nodes=n_nodes,
        expected_edges=expected_edges,
    )


def _enumerate_matched_l1_for_candidate(
    *,
    calibration_run_hash12: str,
    parent_full_hash: str,
    base_dagma_config: DAGMAConfig,
    n_nodes: int,
    expected_edges: int,
    candidate_lambda1: float,
) -> tuple[PlannedRun, ...]:
    candidate_dagma = dataclasses.replace(
        base_dagma_config, lambda1=float(candidate_lambda1)
    )
    return enumerate_planned_runs(
        main_study_run_hash12=calibration_run_hash12,
        seed_population="main_calibration",
        seed_values=CALIBRATION_SEED_VALUES,
        base_dagma_config=candidate_dagma,
        parent_heldout_run_hash_full=parent_full_hash,
        method_families=("matched_l1",),
        n_nodes=n_nodes,
        expected_edges=expected_edges,
        matched_l1_lambda1=float(candidate_lambda1),
    )


def _build_stage_a_planned(
    *,
    calibration_run_hash12: str,
    parent_full_hash: str,
    base_dagma_config: DAGMAConfig,
    n_nodes: int,
    expected_edges: int,
) -> tuple[tuple[PlannedRun, ...], dict[str, list[PlannedRun]]]:
    """Return (combined Stage-A planned runs, index by stage tag)."""
    diagnostic = _enumerate_diagnostic_planned(
        calibration_run_hash12=calibration_run_hash12,
        parent_full_hash=parent_full_hash,
        base_dagma_config=base_dagma_config,
        n_nodes=n_nodes,
        expected_edges=expected_edges,
    )
    stage_1_by_lambda: dict[float, tuple[PlannedRun, ...]] = {}
    stage_1_all: list[PlannedRun] = []
    for lam in STAGE_1_CANDIDATES:
        planned = _enumerate_matched_l1_for_candidate(
            calibration_run_hash12=calibration_run_hash12,
            parent_full_hash=parent_full_hash,
            base_dagma_config=base_dagma_config,
            n_nodes=n_nodes,
            expected_edges=expected_edges,
            candidate_lambda1=lam,
        )
        stage_1_by_lambda[float(lam)] = planned
        stage_1_all.extend(planned)
    combined: tuple[PlannedRun, ...] = tuple(diagnostic) + tuple(stage_1_all)
    by_stage = {
        STAGE_TAG_DIAGNOSTIC: list(diagnostic),
        STAGE_TAG_STAGE_1: list(stage_1_all),
    }
    return combined, by_stage


# ---------------------------------------------------------------------------
# Record loading and row construction
# ---------------------------------------------------------------------------


def _load_record_for_planned(
    planned: PlannedRun, *, base_dir: Path
) -> MainStudyRunRecord:
    record = load_existing_record(planned.record_path, base_dir=base_dir)
    if record is None:
        raise FileNotFoundError(
            "expected persisted record after runner_fn call but none "
            f"was found at {planned.record_path!r}."
        )
    return record


def _rows_for_planned_group(
    planned_runs: tuple[PlannedRun, ...],
    *,
    base_dir: Path,
    stage: str,
    candidate_lambda1_per_planned: Optional[
        dict[str, Optional[float]]
    ] = None,
) -> tuple[CandidateRunRow, ...]:
    rows: list[CandidateRunRow] = []
    for planned in planned_runs:
        record = _load_record_for_planned(planned, base_dir=base_dir)
        cand: Optional[float]
        if candidate_lambda1_per_planned is not None:
            cand = candidate_lambda1_per_planned.get(planned.run_id)
        else:
            cand = planned.config.matched_l1_lambda1
        rows.append(
            row_from_record(
                record=record,
                record_path=planned.record_path,
                stage=stage,
                candidate_lambda1=cand,
                base_dir=base_dir,
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Stage-1 intermediate / final summary writers
# ---------------------------------------------------------------------------


def _ensure_output_dir(output_root: Path, calibration_run_hash12: str) -> Path:
    """Create the calibration output directory under ``output_root``."""
    rel = f"{CALIBRATION_OUTPUT_SUBDIR}/{calibration_run_hash12}"
    full = (output_root / rel)
    full.mkdir(parents=True, exist_ok=True)
    return full


def _write_json_atomic(payload: dict, path: Path) -> None:
    """Write ``payload`` as canonical JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    path.write_text(text, encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _candidate_summary_to_dict(s: CandidateSummary) -> dict[str, Any]:
    return {
        "candidate_lambda1": float(s.candidate_lambda1),
        "stage": s.stage,
        "per_seed_edge_counts": list(s.per_seed_edge_counts),
        "per_seed_valid_dag": list(s.per_seed_valid_dag),
        "valid_dag_count": int(s.valid_dag_count),
        "mean_edge_count": (
            None if s.mean_edge_count is None else float(s.mean_edge_count)
        ),
        "absolute_gap": (
            None if s.absolute_gap is None else float(s.absolute_gap)
        ),
        "fragile_valid_dag": bool(s.fragile_valid_dag),
    }


def write_stage_1_intermediate_summary(
    *,
    output_dir: Path,
    spec: CalibrationRunSpec,
    target_mean: float,
    target_per_seed: tuple[Optional[int], ...],
    stage_1_summaries: tuple[CandidateSummary, ...],
    stage_1_winner_lambda1: Optional[float],
    stage_2_interval: Optional[tuple[float, float]],
    stage_2_generated: tuple[float, ...],
    stage_2_skipped_duplicates: tuple[float, ...],
    timestamp_utc: str,
) -> Path:
    payload = {
        "calibration_run_hash12": spec.calibration_run_hash12,
        "parent_heldout_run_hash_full": spec.parent_heldout_run_hash_full,
        "target_mean_edge_count": float(target_mean),
        "target_per_seed_edge_counts": list(target_per_seed),
        "stage_1_candidates": [float(x) for x in STAGE_1_CANDIDATES],
        "stage_1_candidate_summaries": [
            _candidate_summary_to_dict(s) for s in stage_1_summaries
        ],
        "stage_1_winner_lambda1": (
            None
            if stage_1_winner_lambda1 is None
            else float(stage_1_winner_lambda1)
        ),
        "stage_2_interval": (
            None
            if stage_2_interval is None
            else [float(stage_2_interval[0]), float(stage_2_interval[1])]
        ),
        "stage_2_generated_candidates": [float(x) for x in stage_2_generated],
        "stage_2_skipped_duplicates": [
            float(x) for x in stage_2_skipped_duplicates
        ],
        "timestamp_utc": timestamp_utc,
    }
    path = output_dir / STAGE_1_INTERMEDIATE_FILENAME
    _write_json_atomic(payload, path)
    return path


def write_calibration_table_csv(
    *,
    output_dir: Path,
    rows: tuple[CandidateRunRow, ...],
) -> Path:
    """Write the flat per-run CSV table.

    One CSV row per :class:`CandidateRunRow`. Ordering follows the
    input ``rows`` order.
    """
    fieldnames = [
        "candidate_lambda1",
        "seed",
        "stage",
        "method_family",
        "corruption_fraction",
        "confidence",
        "edge_count",
        "fit_status",
        "graph_status",
        "sampler_status",
        "metric_status",
        "record_path",
        "configuration_hash_full",
        "configuration_hash_prefix",
    ]
    path = output_dir / TABLE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "candidate_lambda1": (
                    "" if r.candidate_lambda1 is None
                    else repr(float(r.candidate_lambda1))
                ),
                "seed": int(r.seed),
                "stage": r.stage,
                "method_family": r.method_family,
                "corruption_fraction": (
                    "" if r.corruption_fraction is None
                    else repr(float(r.corruption_fraction))
                ),
                "confidence": (
                    "" if r.confidence is None
                    else repr(float(r.confidence))
                ),
                "edge_count": (
                    "" if r.edge_count is None else int(r.edge_count)
                ),
                "fit_status": r.fit_status,
                "graph_status": (
                    "" if r.graph_status is None else r.graph_status
                ),
                "sampler_status": (
                    "" if r.sampler_status is None else r.sampler_status
                ),
                "metric_status": r.metric_status,
                "record_path": r.record_path,
                "configuration_hash_full": r.configuration_hash_full,
                "configuration_hash_prefix": r.configuration_hash_prefix,
            })
    return path


def _summary_to_dict(s: MatchedL1CalibrationSummary) -> dict[str, Any]:
    return {
        "halt_status": s.halt_status,
        "parent_heldout_run_hash_full": s.parent_heldout_run_hash_full,
        "calibration_run_hash12": s.calibration_run_hash12,
        "output_dir": s.output_dir_relative,
        "code_version": s.code_version,
        "target_mean_edge_count": (
            None if s.target_mean_edge_count is None
            else float(s.target_mean_edge_count)
        ),
        "target_per_seed_edge_counts": list(s.target_per_seed_edge_counts),
        "stage_1_candidates": [float(x) for x in s.stage_1_candidates],
        "stage_2_interval": (
            None
            if s.stage_2_interval is None
            else [float(s.stage_2_interval[0]), float(s.stage_2_interval[1])]
        ),
        "stage_2_generated_candidates": [
            float(x) for x in s.stage_2_generated_candidates
        ],
        "stage_2_candidates": [float(x) for x in s.stage_2_candidates],
        "stage_2_skipped_duplicates": [
            float(x) for x in s.stage_2_skipped_duplicates
        ],
        "all_evaluated_candidates": [
            float(x) for x in s.all_evaluated_candidates
        ],
        "selected_lambda1": (
            None if s.selected_lambda1 is None else float(s.selected_lambda1)
        ),
        "selected_candidate_mean_edge_count": (
            None if s.selected_candidate_mean_edge_count is None
            else float(s.selected_candidate_mean_edge_count)
        ),
        "selected_absolute_gap": (
            None if s.selected_absolute_gap is None
            else float(s.selected_absolute_gap)
        ),
        "selected_valid_dag_count": (
            None if s.selected_valid_dag_count is None
            else int(s.selected_valid_dag_count)
        ),
        "within_one_edge_tolerance": (
            None if s.within_one_edge_tolerance is None
            else bool(s.within_one_edge_tolerance)
        ),
        "diagnostic_metric_fields_used_for_selection": False,
        "evaluation_seeds_used": False,
        "diagnostic_grid_anomalies": list(s.diagnostic_grid_anomalies),
        "selection_rule": s.selection_rule,
    }


def write_final_summary_json(
    *, output_dir: Path, summary: MatchedL1CalibrationSummary
) -> Path:
    path = output_dir / SUMMARY_FILENAME
    _write_json_atomic(_summary_to_dict(summary), path)
    return path


def write_readout_markdown(
    *,
    output_dir: Path,
    summary: MatchedL1CalibrationSummary,
    stage_1_summaries: tuple[CandidateSummary, ...],
    stage_2_summaries: tuple[CandidateSummary, ...],
    invalid_fit_count: int,
    fragile_warning_lambda1s: tuple[float, ...],
) -> Path:
    lines: list[str] = []
    lines.append("# Matched-L1 Calibration Readout")
    lines.append("")
    lines.append(f"- halt_status: {summary.halt_status}")
    lines.append(
        f"- parent_heldout_run_hash_full: {summary.parent_heldout_run_hash_full}"
    )
    lines.append(
        f"- calibration_run_hash12: {summary.calibration_run_hash12}"
    )
    lines.append(f"- output_dir: {summary.output_dir_relative}")
    lines.append(f"- code_version: {summary.code_version}")
    lines.append("")
    lines.append("## Target")
    lines.append("")
    lines.append(
        f"- target_mean_edge_count: {summary.target_mean_edge_count}"
    )
    lines.append(
        f"- target_per_seed_edge_counts: "
        f"{list(summary.target_per_seed_edge_counts)}"
    )
    lines.append("")
    lines.append("## Stage 1")
    lines.append("")
    for s in stage_1_summaries:
        lines.append(
            f"- lambda1={s.candidate_lambda1}: "
            f"mean_edge_count={s.mean_edge_count}, "
            f"valid_dag_count={s.valid_dag_count}, "
            f"absolute_gap={s.absolute_gap}, "
            f"fragile={s.fragile_valid_dag}"
        )
    lines.append("")
    lines.append("## Stage 2")
    lines.append("")
    n_generated = len(summary.stage_2_generated_candidates)
    n_skipped = len(summary.stage_2_skipped_duplicates)
    n_new = len(summary.stage_2_candidates)
    lines.append(
        f"Stage 2 generated {n_generated} candidate values over the "
        f"selected interval. {n_skipped} value(s) coincided with "
        f"Stage 1 candidates and were skipped as duplicates, leaving "
        f"{n_new} new candidate value(s) to evaluate."
    )
    lines.append("")
    lines.append(f"- stage_2_interval: {summary.stage_2_interval}")
    lines.append(
        f"- stage_2_generated_candidates: "
        f"{list(summary.stage_2_generated_candidates)}"
    )
    lines.append(
        f"- stage_2_skipped_duplicates: "
        f"{list(summary.stage_2_skipped_duplicates)}"
    )
    lines.append(
        f"- stage_2_candidates (new): {list(summary.stage_2_candidates)}"
    )
    for s in stage_2_summaries:
        lines.append(
            f"- lambda1={s.candidate_lambda1}: "
            f"mean_edge_count={s.mean_edge_count}, "
            f"valid_dag_count={s.valid_dag_count}, "
            f"absolute_gap={s.absolute_gap}, "
            f"fragile={s.fragile_valid_dag}"
        )
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    lines.append(f"- selected_lambda1: {summary.selected_lambda1}")
    lines.append(
        f"- selected_candidate_mean_edge_count: "
        f"{summary.selected_candidate_mean_edge_count}"
    )
    lines.append(
        f"- selected_absolute_gap: {summary.selected_absolute_gap}"
    )
    lines.append(
        f"- selected_valid_dag_count: "
        f"{summary.selected_valid_dag_count}"
    )
    lines.append(
        f"- within_one_edge_tolerance: "
        f"{summary.within_one_edge_tolerance}"
    )
    lines.append("")
    lines.append("## Failure and fragility")
    lines.append("")
    lines.append(f"- invalid_or_failed_fit_count: {invalid_fit_count}")
    lines.append(
        f"- candidates_with_only_one_valid_dag_fit: "
        f"{list(fragile_warning_lambda1s)}"
    )
    lines.append("")
    lines.append("## Diagnostic anomalies (advisory only)")
    lines.append("")
    for a in summary.diagnostic_grid_anomalies:
        lines.append(f"- {a}")
    if not summary.diagnostic_grid_anomalies:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Confirmations")
    lines.append("")
    lines.append(
        "- SID, SHD, and MMD were NOT used for selection of "
        "matched_l1_lambda1."
    )
    lines.append(
        "- Evaluation seeds were NOT used; only calibration seeds "
        f"{list(CALIBRATION_SEED_VALUES)} were used."
    )
    lines.append(
        "- Diagnostic-grid anomalies are advisory; they did not "
        "alter selected_lambda1."
    )
    lines.append("")
    lines.append(f"halt_status: {summary.halt_status}")
    path = output_dir / READOUT_FILENAME
    _write_text("\n".join(lines), path)
    return path


# ---------------------------------------------------------------------------
# Halt logic
# ---------------------------------------------------------------------------


def _halt_status_for(
    *,
    ranked: tuple[CandidateSummary, ...],
    stage_2_interval: Optional[tuple[float, float]],
    final_winner: Optional[CandidateSummary],
) -> str:
    if not ranked or final_winner is None:
        return HALT_NO_VALID_DAG
    if stage_2_interval is None:
        return HALT_COMPLETED
    lo, hi = stage_2_interval
    is_outward = _is_close(lo, STAGE_1_LOWER_BOUNDARY_INTERVAL[0]) or _is_close(
        hi, STAGE_1_UPPER_BOUNDARY_INTERVAL[1]
    )
    if (
        is_outward
        and final_winner.absolute_gap is not None
        and float(final_winner.absolute_gap) > CLOSE_MATCH_EDGE_TOLERANCE
    ):
        return HALT_BOUNDARY_POOR_MATCH
    return HALT_COMPLETED


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_matched_l1_calibration(
    *,
    output_root: Path,
    parent_hash: str,
    parent_hash_search_root: Optional[Path] = None,
    code_version: Optional[str] = None,
    generated_at_utc_factory: Callable[[], str] = default_utc_factory,
    runner_fn: Optional[Callable[..., RunSummary]] = None,
    mode: str = "skip",
    n_nodes: int = DEFAULT_N_NODES,
    expected_edges: int = DEFAULT_EXPECTED_EDGES,
    base_dagma_config: Optional[DAGMAConfig] = None,
    data_loader: Optional[Callable[..., Any]] = None,
    fit_backend: Optional[Callable[..., Any]] = None,
    metric_backend: Optional[Callable[..., Any]] = None,
    n_nodes_for_failure_record: Optional[int] = None,
    logger: Optional[Any] = None,
) -> MatchedL1CalibrationSummary:
    """Run matched-L1 calibration end-to-end. Never modifies the decision log.

    Required behaviour is documented in
    ``docs/10_matched_l1_calibration_plan.md``. Selection uses only
    sparsity matching on calibration seeds; SID, SHD, MMD, runtime,
    and diagnostic-grid anomalies must not affect the selected value.
    """
    if mode != "skip":
        raise ValueError(
            "run_matched_l1_calibration requires mode='skip'; got "
            f"{mode!r}."
        )
    # Look up the default runner_fn from the module namespace at call
    # time. This lets tests monkeypatch the module-level reference and
    # have the CLI pick up the patched function.
    if runner_fn is None:
        runner_fn = run_main_study
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    if not isinstance(parent_hash, str) or parent_hash == "":
        raise ValueError(
            "parent_hash must be a non-empty string; got "
            f"{parent_hash!r}."
        )

    if len(parent_hash) == 12:
        if parent_hash_search_root is None:
            raise ValueError(
                "parent_hash_search_root is required when parent_hash "
                "is a 12-character prefix."
            )
        parent_full = resolve_parent_hash_from_prefix(
            parent_hash, search_root=parent_hash_search_root
        )
    elif len(parent_hash) == 64:
        parent_full = validate_parent_hash_full(parent_hash)
    else:
        raise ValueError(
            "parent_hash must be exactly 12 or 64 lowercase hex "
            f"characters; got length {len(parent_hash)}."
        )

    if code_version is None:
        code_version = capture_code_version()

    if base_dagma_config is None:
        base_dagma_config = build_protocol_dagma_config()
    if (
        float(base_dagma_config.lambda1) != float(PROTOCOL_DAGMA_LAMBDA1)
        or int(base_dagma_config.warm_iter)
        != int(PROTOCOL_DAGMA_WARM_ITER)
        or int(base_dagma_config.max_iter)
        != int(PROTOCOL_DAGMA_MAX_ITER)
    ):
        raise ValueError(
            "matched_l1 calibration requires base_dagma_config to "
            "carry the main-study protocol values "
            f"(lambda1={PROTOCOL_DAGMA_LAMBDA1!r}, "
            f"warm_iter={PROTOCOL_DAGMA_WARM_ITER!r}, "
            f"max_iter={PROTOCOL_DAGMA_MAX_ITER!r}); got "
            f"(lambda1={base_dagma_config.lambda1!r}, "
            f"warm_iter={base_dagma_config.warm_iter!r}, "
            f"max_iter={base_dagma_config.max_iter!r}). "
            "Build the config via build_protocol_dagma_config()."
        )

    if data_loader is None or fit_backend is None or metric_backend is None:
        data_loader, fit_backend, metric_backend = _build_default_backends(
            n_nodes=n_nodes,
            expected_edges=expected_edges,
            existing_data_loader=data_loader,
            existing_fit_backend=fit_backend,
            existing_metric_backend=metric_backend,
        )

    if n_nodes_for_failure_record is None:
        n_nodes_for_failure_record = int(n_nodes)

    calibration_run_hash12 = compute_calibration_run_hash12(
        parent_full_hash=parent_full
    )
    output_dir = _ensure_output_dir(output_root, calibration_run_hash12)
    output_dir_relative = (
        f"{CALIBRATION_OUTPUT_SUBDIR}/{calibration_run_hash12}"
    )
    spec = CalibrationRunSpec(
        parent_heldout_run_hash_full=parent_full,
        calibration_run_hash12=calibration_run_hash12,
        output_dir_relative=output_dir_relative,
        code_version=code_version,
    )

    stage_a_planned, by_stage = _build_stage_a_planned(
        calibration_run_hash12=calibration_run_hash12,
        parent_full_hash=parent_full,
        base_dagma_config=base_dagma_config,
        n_nodes=n_nodes,
        expected_edges=expected_edges,
    )

    # Run Stage A: diagnostic soft + Stage 1 matched_l1.
    runner_fn(
        stage_a_planned,
        base_dir=output_root,
        data_loader=data_loader,
        fit_backend=fit_backend,
        metric_backend=metric_backend,
        mode="skip",
        code_version=code_version,
        generated_at_utc_factory=generated_at_utc_factory,
        n_nodes_for_failure_record=int(n_nodes_for_failure_record),
        logger=logger,
    )

    diagnostic_planned = tuple(by_stage[STAGE_TAG_DIAGNOSTIC])
    stage_1_planned = tuple(by_stage[STAGE_TAG_STAGE_1])

    diagnostic_rows = _rows_for_planned_group(
        diagnostic_planned,
        base_dir=output_root,
        stage=STAGE_TAG_DIAGNOSTIC,
    )
    stage_1_rows = _rows_for_planned_group(
        stage_1_planned,
        base_dir=output_root,
        stage=STAGE_TAG_STAGE_1,
    )

    target_rows = select_target_rows(diagnostic_rows)
    target_mean, target_per_seed = target_mean_and_per_seed(target_rows)

    stage_1_summaries: list[CandidateSummary] = []
    for lam in STAGE_1_CANDIDATES:
        per_lam_rows = tuple(
            r for r in stage_1_rows
            if r.candidate_lambda1 is not None
            and _is_close(float(r.candidate_lambda1), float(lam))
        )
        stage_1_summaries.append(
            summarise_candidate(
                candidate_lambda1=float(lam),
                rows=per_lam_rows,
                stage=STAGE_TAG_STAGE_1,
                target_mean=target_mean,
            )
        )
    stage_1_summaries_t = tuple(stage_1_summaries)

    stage_1_ranked = rank_candidates(stage_1_summaries_t)
    stage_1_winner: Optional[CandidateSummary] = (
        stage_1_ranked[0] if stage_1_ranked else None
    )

    stage_2_interval: Optional[tuple[float, float]] = None
    stage_2_generated: tuple[float, ...] = ()
    stage_2_new: tuple[float, ...] = ()
    stage_2_skipped: tuple[float, ...] = ()
    if stage_1_winner is not None:
        stage_2_interval = stage_2_interval_for_winner(
            stage_1_winner.candidate_lambda1
        )
        stage_2_generated = generate_stage_2_candidates(stage_2_interval)
        stage_2_new, stage_2_skipped = split_stage_2_duplicates(
            stage_2_generated
        )

    write_stage_1_intermediate_summary(
        output_dir=output_dir,
        spec=spec,
        target_mean=target_mean,
        target_per_seed=target_per_seed,
        stage_1_summaries=stage_1_summaries_t,
        stage_1_winner_lambda1=(
            None if stage_1_winner is None
            else stage_1_winner.candidate_lambda1
        ),
        stage_2_interval=stage_2_interval,
        stage_2_generated=stage_2_generated,
        stage_2_skipped_duplicates=stage_2_skipped,
        timestamp_utc=generated_at_utc_factory(),
    )

    stage_2_rows: tuple[CandidateRunRow, ...] = ()
    stage_2_summaries: tuple[CandidateSummary, ...] = ()

    if stage_2_new:
        stage_2_planned_list: list[PlannedRun] = []
        for lam in stage_2_new:
            stage_2_planned_list.extend(
                _enumerate_matched_l1_for_candidate(
                    calibration_run_hash12=calibration_run_hash12,
                    parent_full_hash=parent_full,
                    base_dagma_config=base_dagma_config,
                    n_nodes=n_nodes,
                    expected_edges=expected_edges,
                    candidate_lambda1=lam,
                )
            )
        stage_2_planned = tuple(stage_2_planned_list)

        runner_fn(
            stage_2_planned,
            base_dir=output_root,
            data_loader=data_loader,
            fit_backend=fit_backend,
            metric_backend=metric_backend,
            mode="skip",
            code_version=code_version,
            generated_at_utc_factory=generated_at_utc_factory,
            n_nodes_for_failure_record=int(n_nodes_for_failure_record),
            logger=logger,
        )

        stage_2_rows = _rows_for_planned_group(
            stage_2_planned,
            base_dir=output_root,
            stage=STAGE_TAG_STAGE_2,
        )
        stage_2_summary_list: list[CandidateSummary] = []
        for lam in stage_2_new:
            per_lam_rows = tuple(
                r for r in stage_2_rows
                if r.candidate_lambda1 is not None
                and _is_close(float(r.candidate_lambda1), float(lam))
            )
            stage_2_summary_list.append(
                summarise_candidate(
                    candidate_lambda1=float(lam),
                    rows=per_lam_rows,
                    stage=STAGE_TAG_STAGE_2,
                    target_mean=target_mean,
                )
            )
        stage_2_summaries = tuple(stage_2_summary_list)

    final_pool = stage_1_summaries_t + stage_2_summaries
    final_ranked = rank_candidates(final_pool)
    final_winner: Optional[CandidateSummary] = (
        final_ranked[0] if final_ranked else None
    )

    diagnostic_anomalies = compute_diagnostic_anomalies(diagnostic_rows)

    all_evaluated = tuple(
        sorted({
            float(s.candidate_lambda1) for s in final_pool
        })
    )
    halt_status = _halt_status_for(
        ranked=final_ranked,
        stage_2_interval=stage_2_interval,
        final_winner=final_winner,
    )

    selected_lambda1 = (
        None if final_winner is None
        else float(final_winner.candidate_lambda1)
    )
    selected_mean = (
        None if final_winner is None
        else final_winner.mean_edge_count
    )
    selected_gap = (
        None if final_winner is None
        else final_winner.absolute_gap
    )
    selected_valid = (
        None if final_winner is None
        else int(final_winner.valid_dag_count)
    )
    within_one = (
        None if (final_winner is None or final_winner.absolute_gap is None)
        else bool(
            float(final_winner.absolute_gap) <= CLOSE_MATCH_EDGE_TOLERANCE
        )
    )

    summary = MatchedL1CalibrationSummary(
        halt_status=halt_status,
        parent_heldout_run_hash_full=parent_full,
        calibration_run_hash12=calibration_run_hash12,
        output_dir_relative=output_dir_relative,
        code_version=code_version,
        target_mean_edge_count=target_mean,
        target_per_seed_edge_counts=target_per_seed,
        stage_1_candidates=STAGE_1_CANDIDATES,
        stage_2_interval=stage_2_interval,
        stage_2_generated_candidates=stage_2_generated,
        stage_2_candidates=stage_2_new,
        stage_2_skipped_duplicates=stage_2_skipped,
        all_evaluated_candidates=all_evaluated,
        selected_lambda1=selected_lambda1,
        selected_candidate_mean_edge_count=selected_mean,
        selected_absolute_gap=selected_gap,
        selected_valid_dag_count=selected_valid,
        within_one_edge_tolerance=within_one,
        diagnostic_grid_anomalies=diagnostic_anomalies,
    )

    all_rows = (
        tuple(diagnostic_rows) + tuple(stage_1_rows) + tuple(stage_2_rows)
    )
    write_calibration_table_csv(output_dir=output_dir, rows=all_rows)
    write_final_summary_json(output_dir=output_dir, summary=summary)
    fragile_lambda1s = tuple(
        s.candidate_lambda1 for s in final_pool if s.fragile_valid_dag
    )
    invalid_fit_count = sum(
        1 for r in (stage_1_rows + stage_2_rows)
        if r.fit_status != "success" or r.graph_status != "valid_dag"
    )
    write_readout_markdown(
        output_dir=output_dir,
        summary=summary,
        stage_1_summaries=stage_1_summaries_t,
        stage_2_summaries=stage_2_summaries,
        invalid_fit_count=invalid_fit_count,
        fragile_warning_lambda1s=fragile_lambda1s,
    )
    return summary


# ---------------------------------------------------------------------------
# Default-backends helper (lazy import for production CLI use)
# ---------------------------------------------------------------------------


def _build_default_backends(
    *,
    n_nodes: int,
    expected_edges: int,
    existing_data_loader: Optional[Callable[..., Any]],
    existing_fit_backend: Optional[Callable[..., Any]],
    existing_metric_backend: Optional[Callable[..., Any]],
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Construct any missing real backends from the main-study layer.

    The three backends are imported and instantiated lazily so a
    caller that injects all three never causes the production
    wrapper code to load. Any combination of injected and default
    backends is supported.
    """
    from experiments.main_study.backends import (
        DataBundleLoader,
        MainStudyFitBackend,
        RealMetricBackend,
    )
    data_loader = (
        existing_data_loader
        if existing_data_loader is not None
        else DataBundleLoader(n_nodes=int(n_nodes), expected_edges=int(expected_edges))
    )
    fit_backend = (
        existing_fit_backend
        if existing_fit_backend is not None
        else MainStudyFitBackend()
    )
    metric_backend = (
        existing_metric_backend
        if existing_metric_backend is not None
        else RealMetricBackend()
    )
    return data_loader, fit_backend, metric_backend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK: int = 0
_EXIT_ARGS_OR_UNEXPECTED: int = 1
_EXIT_NO_VALID_DAG: int = 2
_EXIT_BOUNDARY_POOR_MATCH: int = 3


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_matched_l1",
        description=(
            "Run the matched-L1 calibration. Writes summary outputs "
            "for human review; never modifies the decision log."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Root directory under which results/main_study/... is "
            "created."
        ),
    )
    parser.add_argument(
        "--parent-hash",
        type=str,
        required=True,
        help=(
            "12-character prefix or 64-character full held-out parent "
            "run hash."
        ),
    )
    parser.add_argument(
        "--parent-hash-search-root",
        type=Path,
        default=None,
        help=(
            "Directory under which a 12-character prefix is resolved "
            "to a unique full 64-character hash. Required if "
            "--parent-hash is a prefix."
        ),
    )
    return parser


def _exit_code_for_halt(halt_status: str) -> int:
    if halt_status == HALT_COMPLETED:
        return _EXIT_OK
    if halt_status == HALT_NO_VALID_DAG:
        return _EXIT_NO_VALID_DAG
    if halt_status == HALT_BOUNDARY_POOR_MATCH:
        return _EXIT_BOUNDARY_POOR_MATCH
    return _EXIT_ARGS_OR_UNEXPECTED


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_matched_l1_calibration(
            output_root=args.output_root,
            parent_hash=args.parent_hash,
            parent_hash_search_root=args.parent_hash_search_root,
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            f"calibrate_matched_l1: unexpected error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return _EXIT_ARGS_OR_UNEXPECTED
    return _exit_code_for_halt(summary.halt_status)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CALIBRATION_PROTOCOL_VERSION",
    "CALIBRATION_SEED_VALUES",
    "STAGE_1_CANDIDATES",
    "STAGE_1_LOWER_BOUNDARY_INTERVAL",
    "STAGE_1_UPPER_BOUNDARY_INTERVAL",
    "STAGE_2_NUM_VALUES",
    "TARGET_CORRUPTION_FRACTION",
    "TARGET_CONFIDENCE",
    "CLOSE_MATCH_EDGE_TOLERANCE",
    "HALT_COMPLETED",
    "HALT_NO_VALID_DAG",
    "HALT_BOUNDARY_POOR_MATCH",
    "SELECTION_RULE",
    "CalibrationRunSpec",
    "CandidateRunRow",
    "CandidateSummary",
    "MatchedL1CalibrationSummary",
    "capture_code_version",
    "compute_calibration_run_hash12",
    "compute_diagnostic_anomalies",
    "default_utc_factory",
    "generate_stage_2_candidates",
    "main",
    "off_diagonal_edge_count",
    "rank_candidates",
    "row_from_record",
    "run_matched_l1_calibration",
    "select_target_rows",
    "split_stage_2_duplicates",
    "stage_2_interval_for_winner",
    "summarise_candidate",
    "target_mean_and_per_seed",
    "write_calibration_table_csv",
    "write_final_summary_json",
    "write_readout_markdown",
    "write_stage_1_intermediate_summary",
]
