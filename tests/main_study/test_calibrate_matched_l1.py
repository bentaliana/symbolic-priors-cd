"""Tests for the matched-L1 calibration script.

Every test writes only under pytest's ``tmp_path``. The real DAGMA
wrappers are never invoked; the real metric backend is never invoked;
no main-evaluation seed is used. ``runner_fn`` is replaced by a
deterministic in-process fake that constructs and persists fake
records and artefacts at the locations declared by each
:class:`PlannedRun`.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from experiments.main_study import calibrate_matched_l1 as cal_mod
from experiments.main_study.schema import build_protocol_dagma_config
from experiments.main_study.calibrate_matched_l1 import (
    CALIBRATION_PROTOCOL_VERSION,
    CALIBRATION_SEED_VALUES,
    CLOSE_MATCH_EDGE_TOLERANCE,
    HALT_BOUNDARY_POOR_MATCH,
    HALT_COMPLETED,
    HALT_NO_VALID_DAG,
    READOUT_FILENAME,
    SELECTION_RULE,
    STAGE_1_CANDIDATES,
    STAGE_1_LOWER_BOUNDARY_INTERVAL,
    STAGE_1_UPPER_BOUNDARY_INTERVAL,
    STAGE_2_NUM_VALUES,
    STAGE_1_INTERMEDIATE_FILENAME,
    STAGE_TAG_DIAGNOSTIC,
    STAGE_TAG_STAGE_1,
    STAGE_TAG_STAGE_2,
    SUMMARY_FILENAME,
    TABLE_FILENAME,
    TARGET_CONFIDENCE,
    TARGET_CORRUPTION_FRACTION,
    CalibrationRunSpec,
    CandidateRunRow,
    CandidateSummary,
    MatchedL1CalibrationSummary,
    capture_code_version,
    compute_calibration_run_hash12,
    compute_diagnostic_anomalies,
    default_utc_factory,
    generate_stage_2_candidates,
    main as cli_main,
    off_diagonal_edge_count,
    rank_candidates,
    row_from_record,
    run_matched_l1_calibration,
    select_target_rows,
    split_stage_2_duplicates,
    stage_2_interval_for_winner,
    summarise_candidate,
    target_mean_and_per_seed,
)
from experiments.main_study.records import (
    MainStudyRunRecord,
    SCHEMA_VERSION,
    record_to_json,
)
from experiments.main_study.run_io import (
    persist_record_atomic,
)
from experiments.main_study.runner import RunSummary, WorkloadStatus
from experiments.main_study.schema import (
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import (
    PlannedRun,
    make_planned_run,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_PARENT_HASH_OTHER = "b" * 64
_RUN_HASH12 = "0123456789ab"
_GENERATED_AT = "2026-05-24T12:00:00Z"
# Use the protocol-standard SCM dimension so the corrupted-prior
# helper has enough true positives to substitute at every corruption
# level (PRIOR_K=10, max corruption 0.8 needs >=8 true positives).
_N_NODES = 10
_EXPECTED_EDGES = 20


def _planned_record_kwargs(
    planned: PlannedRun,
    *,
    edge_count: int,
    n_nodes: int = _N_NODES,
    fit_status: str = "success",
    graph_status: Optional[str] = "valid_dag",
    sampler_status: Optional[str] = "available",
    metric_status: str = "computed",
    failure_kind: Optional[str] = None,
    failure_message: str = "",
) -> dict[str, Any]:
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=n_nodes,
        fit_status=fit_status,
        graph_status=graph_status,
        sampler_status=sampler_status,
        metric_status=metric_status,
        failure_kind=failure_kind,
        failure_message=failure_message,
        runtime_seconds=1.0,
        fit_runtime_seconds=0.8,
        wrapper_diagnostics={"training_status": "converged"},
        parent_heldout_run_hash_full=planned.config.parent_heldout_run_hash_full,
        generated_at_utc=_GENERATED_AT,
    )
    if fit_status == "success":
        kwargs["sid"] = 1.0 if metric_status == "computed" else None
        kwargs["shd"] = 2.0 if metric_status == "computed" else None
        kwargs["mmd"] = -1e-4 if metric_status == "computed" else None
        kwargs["metric_runtime_seconds"] = (
            0.2 if metric_status == "computed" else None
        )
        kwargs["continuous_w_path"] = planned.artefact_paths["continuous_w.npz"]
        kwargs["thresholded_adjacency_path"] = planned.artefact_paths[
            "thresholded_adjacency.npz"
        ]
        kwargs["true_adjacency_path"] = planned.artefact_paths[
            "true_adjacency.npz"
        ]
        if metric_status == "computed":
            kwargs["interventions_mmd_path"] = planned.artefact_paths[
                "interventions_mmd.json"
            ]
        if family == "soft_frobenius":
            kwargs["confidence_mask_path"] = planned.artefact_paths[
                "confidence_mask.npz"
            ]
            kwargs["prior_edge_set_clean_path"] = planned.artefact_paths[
                "prior_edge_set_clean.json"
            ]
            kwargs["prior_edge_set_corrupted_path"] = planned.artefact_paths[
                "prior_edge_set_corrupted.json"
            ]
            kwargs["per_edge_labels_path"] = planned.artefact_paths[
                "per_edge_labels.json"
            ]
    return kwargs


def _adjacency_with_edges(n_nodes: int, edge_count: int) -> np.ndarray:
    """Build a deterministic off-diagonal bool adjacency with given count."""
    mat = np.zeros((n_nodes, n_nodes), dtype=bool)
    placed = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if placed >= edge_count:
                break
            mat[i, j] = True
            placed += 1
        if placed >= edge_count:
            break
    if placed < edge_count:
        raise ValueError(
            f"cannot place {edge_count} off-diagonal edges in an "
            f"{n_nodes}x{n_nodes} matrix."
        )
    return mat


def _write_fake_artefacts_for_planned(
    planned: PlannedRun,
    *,
    base_dir: Path,
    edge_count: int,
    n_nodes: int = _N_NODES,
) -> None:
    """Write the npz/json artefacts required by the planned's method family."""
    adjacency = _adjacency_with_edges(n_nodes, edge_count)
    for name, rel in planned.artefact_paths.items():
        full = base_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        if name == "thresholded_adjacency.npz":
            np.savez(full, thresholded_adjacency=adjacency)
        elif name == "continuous_w.npz":
            w = adjacency.astype(float)
            np.savez(full, continuous_w=w)
        elif name == "true_adjacency.npz":
            np.savez(
                full,
                true_adjacency=np.zeros((n_nodes, n_nodes), dtype=bool),
            )
        elif name == "confidence_mask.npz":
            np.savez(
                full,
                confidence_mask=np.zeros((n_nodes, n_nodes), dtype=float),
            )
        elif name == "interventions_mmd.json":
            full.write_text(
                json.dumps({"records": [], "mmd_primary": 0.0}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_clean.json":
            full.write_text(
                json.dumps({"n_nodes": n_nodes, "forbidden_edges": []}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_corrupted.json":
            full.write_text(
                json.dumps({
                    "n_nodes": n_nodes,
                    "corruption_fraction": 0.0,
                    "forbidden_edges": [],
                }),
                encoding="utf-8",
            )
        elif name == "per_edge_labels.json":
            full.write_text(json.dumps({}), encoding="utf-8")


def _edge_count_for_planned(planned: PlannedRun) -> int:
    """Default edge-count rule used by the fake runner.

    For soft_frobenius diagnostic runs, return a constant
    target-friendly count of 5. For matched_l1 candidates, return
    abs(7 - lambda1*70) clamped to >= 1 so the optimal Stage-1
    candidate has minimum gap from the target mean of 5.
    """
    cfg = planned.config
    if cfg.method_family == "soft_frobenius":
        return 5
    if cfg.method_family == "matched_l1":
        lam = float(cfg.matched_l1_lambda1 or 0.0)
        # Plain linear schedule, monotonic in lambda1. The minimum
        # of |edge_count - 5| over the Stage-1 grid is at lambda1=0.1.
        # edge_count(lambda1) = 12 - lambda1 * 70
        # lambda1=0.025 -> 10.25 (rounded to 10), gap 5
        # lambda1=0.05  -> 8.5 (8),  gap 3
        # lambda1=0.075 -> 6.75 (7), gap 2
        # lambda1=0.1   -> 5.0 (5),  gap 0
        # lambda1=0.15  -> 1.5 (2),  gap 3
        # lambda1=0.2   -> -2.0 -> 1 (clamped), gap 4
        # lambda1=0.25  -> -5.5 -> 1, gap 4
        ec = int(round(12 - lam * 70))
        return max(1, ec)
    return 5


def make_fake_runner(
    *,
    edge_count_for_planned: Callable[[PlannedRun], int] = _edge_count_for_planned,
    record_kwargs_override: Optional[
        Callable[[PlannedRun], dict[str, Any]]
    ] = None,
    n_nodes_for_artefact: int = _N_NODES,
    call_log: Optional[list[dict[str, Any]]] = None,
    on_call: Optional[Callable[[tuple[PlannedRun, ...], Path], None]] = None,
) -> Callable[..., RunSummary]:
    """Return a fake ``run_main_study`` that writes deterministic records.

    Each invocation persists, for every planned run, a thresholded-
    adjacency artefact (and any other artefacts required by the
    method-family contract) plus a JSON record whose
    ``thresholded_adjacency_path`` points at the artefact.
    """
    def fake(
        planned_runs,
        *,
        base_dir: Path,
        data_loader,
        fit_backend,
        metric_backend,
        mode: str,
        code_version,
        generated_at_utc_factory,
        n_nodes_for_failure_record,
        logger=None,
        **kwargs,
    ) -> RunSummary:
        planned_tuple = tuple(planned_runs)
        if call_log is not None:
            call_log.append({
                "base_dir": base_dir,
                "mode": mode,
                "n_planned": len(planned_tuple),
                "code_version": code_version,
                "planned_runs": planned_tuple,
            })
        if on_call is not None:
            on_call(planned_tuple, base_dir)
        statuses: list[WorkloadStatus] = []
        for planned in planned_tuple:
            ec = int(edge_count_for_planned(planned))
            _write_fake_artefacts_for_planned(
                planned,
                base_dir=base_dir,
                edge_count=ec,
                n_nodes=n_nodes_for_artefact,
            )
            override = (
                record_kwargs_override(planned)
                if record_kwargs_override is not None else {}
            )
            kw = _planned_record_kwargs(
                planned,
                edge_count=ec,
                n_nodes=n_nodes_for_artefact,
            )
            kw.update(override)
            record = MainStudyRunRecord(**kw)
            persist_record_atomic(
                record, planned.record_path, base_dir=base_dir
            )
            statuses.append(WorkloadStatus(
                run_id=planned.run_id,
                configuration_hash_prefix=planned.configuration_hash_prefix,
                method_family=planned.config.method_family,
                final_status=(
                    "success_computed"
                    if record.metric_status == "computed"
                    else "success_metric_unavailable"
                ),
                record_path=planned.record_path,
                runtime_seconds=0.0,
                message="",
            ))
        return RunSummary(
            n_planned=len(planned_tuple),
            n_executed=len(planned_tuple),
            n_success_computed=sum(
                1 for s in statuses if s.final_status == "success_computed"
            ),
            n_success_metric_unavailable=sum(
                1 for s in statuses
                if s.final_status == "success_metric_unavailable"
            ),
            n_model_fit_failure=0,
            n_skipped=0,
            n_overwritten=0,
            n_infrastructure_failure=0,
            total_runtime_seconds=0.0,
            per_workload_status=tuple(statuses),
        )
    return fake


def _make_diagnostic_planned(seed: int = 401) -> PlannedRun:
    from experiments.main_study.priors import CorruptedPriorSpec
    spec = CorruptedPriorSpec(
        n_nodes=_N_NODES,
        scm_seed=seed,
        corruption_fraction=0.0,
        corruption_index=0,
        corruption_seed=9100 + seed + 0,
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        n_correct=3,
        n_corrupted=0,
        removed_clean_edges=(),
        added_true_positive_edges=(),
        edge_labels={
            "0,2": "true_negative_retained",
            "1,3": "true_negative_retained",
            "2,4": "true_negative_retained",
        },
    )
    cfg = make_main_study_config(
        method_family="soft_frobenius",
        seed_value=seed,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        confidence=TARGET_CONFIDENCE,
        corrupted_prior_spec=spec,
    )
    return make_planned_run(cfg, _RUN_HASH12)


def _make_matched_l1_planned(
    seed: int = 401, lam: float = 0.1
) -> PlannedRun:
    cfg = make_main_study_config(
        method_family="matched_l1",
        seed_value=seed,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(lambda1=lam),
        parent_heldout_run_hash_full=_PARENT_HASH,
        matched_l1_lambda1=lam,
    )
    return make_planned_run(cfg, _RUN_HASH12)


def _row(
    *,
    method_family: str = "matched_l1",
    seed: int = 401,
    lam: Optional[float] = 0.1,
    edge_count: Optional[int] = 5,
    graph_status: Optional[str] = "valid_dag",
    confidence: Optional[float] = None,
    corruption_fraction: Optional[float] = None,
    stage: str = STAGE_TAG_STAGE_1,
) -> CandidateRunRow:
    return CandidateRunRow(
        candidate_lambda1=lam,
        seed=seed,
        stage=stage,
        method_family=method_family,
        corruption_fraction=corruption_fraction,
        confidence=confidence,
        edge_count=edge_count,
        fit_status="success",
        graph_status=graph_status,
        sampler_status="available",
        metric_status="computed",
        record_path=f"results/main_study/{_RUN_HASH12}/records/r_{seed}.json",
        configuration_hash_full="a" * 64,
        configuration_hash_prefix="a" * 12,
    )


# ===========================================================================
# A. Constants and protocol
# ===========================================================================


def test_calibration_protocol_version():
    assert CALIBRATION_PROTOCOL_VERSION == "matched_l1_v1"


def test_calibration_seed_values_are_calibration_only():
    assert CALIBRATION_SEED_VALUES == (401, 402)
    for s in CALIBRATION_SEED_VALUES:
        assert s not in EVALUATION_SEEDS


def test_stage_1_candidates_exact():
    assert STAGE_1_CANDIDATES == (
        0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
    )


def test_target_condition_constants():
    assert TARGET_CORRUPTION_FRACTION == 0.0
    assert TARGET_CONFIDENCE == 1.0
    assert CLOSE_MATCH_EDGE_TOLERANCE == 1.0


def test_halt_status_string_values():
    assert HALT_COMPLETED == "completed"
    assert HALT_NO_VALID_DAG == "halt_no_valid_dag"
    assert HALT_BOUNDARY_POOR_MATCH == "halt_boundary_poor_match"


def test_selection_rule_string_present():
    assert "Exclude candidates with zero valid-DAG fits" in SELECTION_RULE
    assert "smaller lambda1" in SELECTION_RULE


def test_stage_2_num_values_is_five():
    assert STAGE_2_NUM_VALUES == 5


# ===========================================================================
# B. Parent hash resolution
# ===========================================================================


def test_compute_calibration_run_hash12_is_12_hex():
    h = compute_calibration_run_hash12(parent_full_hash=_PARENT_HASH)
    assert isinstance(h, str)
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_compute_calibration_run_hash12_changes_with_parent():
    a = compute_calibration_run_hash12(parent_full_hash=_PARENT_HASH)
    b = compute_calibration_run_hash12(parent_full_hash=_PARENT_HASH_OTHER)
    assert a != b


def test_compute_calibration_run_hash12_deterministic():
    a = compute_calibration_run_hash12(parent_full_hash=_PARENT_HASH)
    b = compute_calibration_run_hash12(parent_full_hash=_PARENT_HASH)
    assert a == b


def test_calibration_run_hash12_does_not_depend_on_code_version(tmp_path):
    """The hash is computed without code_version anywhere in the input.

    Calling run_matched_l1_calibration with two different
    code_version strings must yield the same calibration_run_hash12.
    """
    fake_a = make_fake_runner()
    fake_b = make_fake_runner()
    summary_a = run_matched_l1_calibration(
        output_root=tmp_path / "a",
        parent_hash=_PARENT_HASH,
        code_version="aaaaaaa",
        runner_fn=fake_a,
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        base_dagma_config=build_protocol_dagma_config(),
        data_loader=object(),
        fit_backend=object(),
        metric_backend=object(),
    )
    summary_b = run_matched_l1_calibration(
        output_root=tmp_path / "b",
        parent_hash=_PARENT_HASH,
        code_version="bbbbbbb",
        runner_fn=fake_b,
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        base_dagma_config=build_protocol_dagma_config(),
        data_loader=object(),
        fit_backend=object(),
        metric_backend=object(),
    )
    assert summary_a.calibration_run_hash12 == summary_b.calibration_run_hash12
    assert summary_a.code_version == "aaaaaaa"
    assert summary_b.code_version == "bbbbbbb"


def test_parent_hash_full_accepted(tmp_path):
    fake = make_fake_runner()
    summary = run_matched_l1_calibration(
        output_root=tmp_path,
        parent_hash=_PARENT_HASH,
        runner_fn=fake,
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        base_dagma_config=build_protocol_dagma_config(),
        data_loader=object(),
        fit_backend=object(),
        metric_backend=object(),
    )
    assert summary.parent_heldout_run_hash_full == _PARENT_HASH


def test_parent_hash_prefix_resolves(tmp_path):
    # Pre-stage a directory whose name is the 64-char full hash so the
    # prefix can be resolved.
    full = "c" * 64
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / full).mkdir()
    fake = make_fake_runner()
    summary = run_matched_l1_calibration(
        output_root=tmp_path / "out",
        parent_hash=full[:12],
        parent_hash_search_root=search_root,
        runner_fn=fake,
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        base_dagma_config=build_protocol_dagma_config(),
        data_loader=object(),
        fit_backend=object(),
        metric_backend=object(),
    )
    assert summary.parent_heldout_run_hash_full == full


def test_parent_hash_invalid_length_raises(tmp_path):
    with pytest.raises(ValueError, match="12 or 64"):
        run_matched_l1_calibration(
            output_root=tmp_path,
            parent_hash="abc",
            runner_fn=make_fake_runner(),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
            base_dagma_config=build_protocol_dagma_config(),
            data_loader=object(),
            fit_backend=object(),
            metric_backend=object(),
        )


# ===========================================================================
# C. Edge-count and row construction
# ===========================================================================


def test_off_diagonal_edge_count_excludes_diagonal():
    mat = np.array([
        [True, True, False],
        [False, True, True],
        [True, False, True],
    ])
    # off-diagonal True entries: (0,1), (1,2), (2,0) = 3
    assert off_diagonal_edge_count(mat) == 3


def test_off_diagonal_edge_count_zero():
    mat = np.zeros((4, 4), dtype=bool)
    np.fill_diagonal(mat, True)
    assert off_diagonal_edge_count(mat) == 0


def test_off_diagonal_edge_count_rejects_float():
    with pytest.raises(ValueError, match="bool-like or integer"):
        off_diagonal_edge_count(np.zeros((3, 3), dtype=float))


def test_off_diagonal_edge_count_rejects_non_square():
    with pytest.raises(ValueError, match="2D square"):
        off_diagonal_edge_count(np.zeros((3, 4), dtype=bool))


def test_row_from_record_reads_thresholded_not_continuous(tmp_path):
    planned = _make_matched_l1_planned(seed=401, lam=0.1)
    # Write a thresholded artefact with 7 edges and a DIFFERENT
    # continuous_w with many more. row_from_record must read 7.
    _write_fake_artefacts_for_planned(
        planned, base_dir=tmp_path, edge_count=7
    )
    # Overwrite continuous_w with a dense ndarray (would give very
    # different count if mistakenly used).
    cw_full = tmp_path / planned.artefact_paths["continuous_w.npz"]
    dense = np.ones((_N_NODES, _N_NODES), dtype=float)
    np.savez(cw_full, continuous_w=dense)
    kw = _planned_record_kwargs(planned, edge_count=7)
    record = MainStudyRunRecord(**kw)
    row = row_from_record(
        record=record,
        record_path=planned.record_path,
        stage=STAGE_TAG_STAGE_1,
        candidate_lambda1=0.1,
        base_dir=tmp_path,
    )
    assert row.edge_count == 7


def test_row_from_record_missing_thresholded_returns_none(tmp_path):
    planned = _make_matched_l1_planned(seed=401, lam=0.1)
    kw = _planned_record_kwargs(
        planned,
        edge_count=0,
        fit_status="model_fit_failure",
        graph_status=None,
        sampler_status=None,
        metric_status="not_computed_due_to_fit_failure",
        failure_kind=None,
        failure_message="forced",
    )
    # Drop all artefact paths for the failure record.
    for key in (
        "continuous_w_path",
        "thresholded_adjacency_path",
        "true_adjacency_path",
        "interventions_mmd_path",
    ):
        kw[key] = None
    kw["sid"] = None
    kw["shd"] = None
    kw["mmd"] = None
    kw["metric_runtime_seconds"] = None
    record = MainStudyRunRecord(**kw)
    row = row_from_record(
        record=record,
        record_path=planned.record_path,
        stage=STAGE_TAG_STAGE_1,
        candidate_lambda1=0.1,
        base_dir=tmp_path,
    )
    assert row.edge_count is None


# ===========================================================================
# D. Target selection and per-candidate summaries
# ===========================================================================


def test_select_target_rows_requires_exactly_two():
    only_one = (
        _row(method_family="soft_frobenius", seed=401,
             confidence=1.0, corruption_fraction=0.0, lam=None),
    )
    with pytest.raises(ValueError, match="exactly two"):
        select_target_rows(only_one)


def test_select_target_rows_filters_wrong_cell():
    rows = (
        _row(method_family="soft_frobenius", seed=401,
             confidence=1.0, corruption_fraction=0.0, lam=None),
        _row(method_family="soft_frobenius", seed=402,
             confidence=0.5, corruption_fraction=0.0, lam=None),
    )
    with pytest.raises(ValueError, match="exactly two"):
        select_target_rows(rows)


def test_select_target_rows_accepts_exactly_two():
    rows = (
        _row(method_family="soft_frobenius", seed=401,
             confidence=1.0, corruption_fraction=0.0, lam=None,
             edge_count=4),
        _row(method_family="soft_frobenius", seed=402,
             confidence=1.0, corruption_fraction=0.0, lam=None,
             edge_count=6),
    )
    selected = select_target_rows(rows)
    assert len(selected) == 2


def test_target_mean_and_per_seed():
    rows = (
        _row(method_family="soft_frobenius", seed=402,
             confidence=1.0, corruption_fraction=0.0, lam=None,
             edge_count=6),
        _row(method_family="soft_frobenius", seed=401,
             confidence=1.0, corruption_fraction=0.0, lam=None,
             edge_count=4),
    )
    mean, per_seed = target_mean_and_per_seed(rows)
    assert mean == pytest.approx(5.0)
    # per_seed sorted by seed: (401, 402) -> (4, 6)
    assert per_seed == (4, 6)


def test_summarise_candidate_mean_valid_fragile():
    rows = (
        _row(seed=401, lam=0.05, edge_count=4, graph_status="valid_dag"),
        _row(seed=402, lam=0.05, edge_count=None, graph_status="cyclic"),
    )
    s = summarise_candidate(
        candidate_lambda1=0.05,
        rows=rows,
        stage=STAGE_TAG_STAGE_1,
        target_mean=5.0,
    )
    assert s.mean_edge_count == pytest.approx(4.0)
    assert s.valid_dag_count == 1
    assert s.fragile_valid_dag is True
    assert s.absolute_gap == pytest.approx(1.0)


def test_summarise_candidate_zero_valid_dag():
    rows = (
        _row(seed=401, lam=0.05, edge_count=8, graph_status="cyclic"),
        _row(seed=402, lam=0.05, edge_count=10, graph_status="cyclic"),
    )
    s = summarise_candidate(
        candidate_lambda1=0.05,
        rows=rows,
        stage=STAGE_TAG_STAGE_1,
        target_mean=5.0,
    )
    assert s.valid_dag_count == 0
    assert s.fragile_valid_dag is False
    assert s.mean_edge_count == pytest.approx(9.0)


# ===========================================================================
# E. Ranking
# ===========================================================================


def _summary(
    *,
    lam: float,
    valid: int,
    gap: Optional[float],
    mean: Optional[float] = 5.0,
) -> CandidateSummary:
    return CandidateSummary(
        candidate_lambda1=lam,
        stage=STAGE_TAG_STAGE_1,
        per_seed_edge_counts=(int(mean) if mean is not None else None,) * 2,
        per_seed_valid_dag=(True,) * valid + (False,) * (2 - valid),
        valid_dag_count=valid,
        mean_edge_count=mean,
        absolute_gap=gap,
        fragile_valid_dag=(valid == 1),
    )


def test_rank_excludes_zero_valid_dag():
    summaries = (
        _summary(lam=0.05, valid=0, gap=0.0),
        _summary(lam=0.1, valid=2, gap=1.0),
    )
    ranked = rank_candidates(summaries)
    assert len(ranked) == 1
    assert ranked[0].candidate_lambda1 == 0.1


def test_rank_prefers_more_valid_dags_over_smaller_gap():
    summaries = (
        _summary(lam=0.05, valid=1, gap=0.0),   # smaller gap but fragile
        _summary(lam=0.1, valid=2, gap=5.0),    # bigger gap, more valid
    )
    ranked = rank_candidates(summaries)
    assert ranked[0].candidate_lambda1 == 0.1


def test_rank_minimises_gap_then_tie_breaks_smaller_lambda1():
    summaries = (
        _summary(lam=0.15, valid=2, gap=1.0),
        _summary(lam=0.075, valid=2, gap=1.0),  # tie-break: smaller wins
        _summary(lam=0.2, valid=2, gap=0.5),    # smallest gap wins overall
    )
    ranked = rank_candidates(summaries)
    assert ranked[0].candidate_lambda1 == 0.2
    assert ranked[1].candidate_lambda1 == 0.075
    assert ranked[2].candidate_lambda1 == 0.15


# ===========================================================================
# F. Stage-2 generation and duplicates
# ===========================================================================


def test_stage_2_interval_lower_boundary():
    interval = stage_2_interval_for_winner(STAGE_1_CANDIDATES[0])
    assert interval == STAGE_1_LOWER_BOUNDARY_INTERVAL


def test_stage_2_interval_upper_boundary():
    interval = stage_2_interval_for_winner(STAGE_1_CANDIDATES[-1])
    assert interval == STAGE_1_UPPER_BOUNDARY_INTERVAL


def test_stage_2_interval_internal_winner():
    # winner 0.1 has neighbours 0.075 and 0.15.
    interval = stage_2_interval_for_winner(0.1)
    assert interval == (0.075, 0.15)


def test_stage_2_interval_rejects_non_stage_1_winner():
    with pytest.raises(ValueError, match="Stage-1 candidate"):
        stage_2_interval_for_winner(0.123)


def test_generate_stage_2_candidates_count_and_endpoints():
    cands = generate_stage_2_candidates((0.1, 0.2))
    assert len(cands) == STAGE_2_NUM_VALUES == 5
    assert cands[0] == pytest.approx(0.1)
    assert cands[-1] == pytest.approx(0.2)
    # evenly spaced
    diffs = [cands[i + 1] - cands[i] for i in range(4)]
    assert all(d == pytest.approx(diffs[0]) for d in diffs)


def test_generate_stage_2_candidates_rejects_bad_interval():
    with pytest.raises(ValueError):
        generate_stage_2_candidates((0.2, 0.1))


def test_split_stage_2_duplicates_internal_case():
    # interval [0.075, 0.15] -> linspace = [0.075, 0.09375, 0.1125, 0.13125, 0.15]
    interval = (0.075, 0.15)
    gen = generate_stage_2_candidates(interval)
    new, skipped = split_stage_2_duplicates(gen)
    # 0.075 and 0.15 are Stage-1 candidates.
    assert pytest.approx(0.075) in skipped
    assert pytest.approx(0.15) in skipped
    assert len(skipped) == 2
    assert len(new) == 3


def test_split_stage_2_duplicates_upper_boundary_case():
    # interval [0.2, 0.3] -> [0.2, 0.225, 0.25, 0.275, 0.3]
    # 0.2 and 0.25 are Stage-1; 0.3 is not.
    interval = STAGE_1_UPPER_BOUNDARY_INTERVAL
    gen = generate_stage_2_candidates(interval)
    new, skipped = split_stage_2_duplicates(gen)
    assert len(skipped) == 2
    assert len(new) == 3
    assert pytest.approx(0.3) in new


# ===========================================================================
# G. Diagnostic anomalies (advisory only)
# ===========================================================================


def test_diagnostic_anomalies_flags_sparsity_collapse():
    rows = []
    # All soft cells have edge_count 0 -> sparsity collapse on every cell.
    for seed in (401, 402):
        for cf in (0.0, 0.5, 1.0):
            for cn in (0.0, 0.5, 1.0):
                rows.append(_row(
                    method_family="soft_frobenius",
                    seed=seed,
                    confidence=cn,
                    corruption_fraction=cf,
                    edge_count=0,
                    lam=None,
                    stage=STAGE_TAG_DIAGNOSTIC,
                ))
    anomalies = compute_diagnostic_anomalies(tuple(rows))
    assert any("sparsity_collapse" in a for a in anomalies)


def test_diagnostic_anomalies_flags_non_monotonic():
    rows = []
    # corruption = 0.0; confidence -> mean edge count goes 5, 8, 5.
    pattern = [(0.0, 5), (0.5, 8), (1.0, 5)]
    for seed in (401, 402):
        for cn, ec in pattern:
            rows.append(_row(
                method_family="soft_frobenius",
                seed=seed,
                confidence=cn,
                corruption_fraction=0.0,
                edge_count=ec,
                lam=None,
                stage=STAGE_TAG_DIAGNOSTIC,
            ))
    anomalies = compute_diagnostic_anomalies(tuple(rows))
    assert any("non_monotonic_confidence" in a for a in anomalies)


def test_diagnostic_anomalies_does_not_alter_selection(tmp_path):
    """An injected anomaly is recorded but selected_lambda1 is unchanged."""
    # First run: baseline rewards 0.1.
    baseline_summary = _run_calibration_with_fake_runner(
        tmp_path / "baseline",
        edge_for_planned=_edge_count_for_planned,
    )
    # Second run: same Stage-1 ranking, but inject sparsity collapse
    # on diagnostic soft cells.
    def edge_for_planned_with_anomaly(planned: PlannedRun) -> int:
        if planned.config.method_family == "soft_frobenius":
            cp = planned.config.corrupted_prior_spec
            if (
                cp is not None
                and float(cp.corruption_fraction) == 0.0
                and planned.config.confidence is not None
                and float(planned.config.confidence) == 0.0
            ):
                return 0  # cell collapses
            return 5
        return _edge_count_for_planned(planned)

    anomaly_summary = _run_calibration_with_fake_runner(
        tmp_path / "anomaly",
        edge_for_planned=edge_for_planned_with_anomaly,
    )
    assert anomaly_summary.selected_lambda1 == baseline_summary.selected_lambda1
    assert anomaly_summary.diagnostic_grid_anomalies
    assert not baseline_summary.diagnostic_grid_anomalies


def _run_calibration_with_fake_runner(
    tmp_path: Path,
    *,
    edge_for_planned: Callable[[PlannedRun], int] = _edge_count_for_planned,
    record_kwargs_override: Optional[
        Callable[[PlannedRun], dict[str, Any]]
    ] = None,
    call_log: Optional[list[dict[str, Any]]] = None,
    on_call: Optional[Callable[[tuple[PlannedRun, ...], Path], None]] = None,
) -> MatchedL1CalibrationSummary:
    fake = make_fake_runner(
        edge_count_for_planned=edge_for_planned,
        record_kwargs_override=record_kwargs_override,
        call_log=call_log,
        on_call=on_call,
    )
    return run_matched_l1_calibration(
        output_root=tmp_path,
        parent_hash=_PARENT_HASH,
        runner_fn=fake,
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        base_dagma_config=build_protocol_dagma_config(),
        data_loader=object(),
        fit_backend=object(),
        metric_backend=object(),
    )


# ===========================================================================
# H. End-to-end orchestration
# ===========================================================================


def test_run_main_calibration_writes_all_expected_files(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    out_dir = tmp_path / summary.output_dir_relative
    assert (out_dir / STAGE_1_INTERMEDIATE_FILENAME).exists()
    assert (out_dir / SUMMARY_FILENAME).exists()
    assert (out_dir / TABLE_FILENAME).exists()
    assert (out_dir / READOUT_FILENAME).exists()


def test_runner_called_first_for_stage_a_then_stage_b(tmp_path):
    calls: list[dict[str, Any]] = []
    summary = _run_calibration_with_fake_runner(tmp_path, call_log=calls)
    # First call is Stage A (diagnostic + Stage 1): 50 + 14 = 64 runs.
    assert calls[0]["n_planned"] == 64
    # Second call is Stage B: at most 5 unique Stage-2 lambdas, each
    # evaluated on 2 calibration seeds, so at most 10 planned runs.
    if len(calls) == 2:
        assert 0 < calls[1]["n_planned"] <= 10


def test_mode_skip_is_enforced_on_every_runner_call(tmp_path):
    calls: list[dict[str, Any]] = []
    _run_calibration_with_fake_runner(tmp_path, call_log=calls)
    for c in calls:
        assert c["mode"] == "skip"


def test_run_main_calibration_rejects_non_skip_mode(tmp_path):
    fake = make_fake_runner()
    with pytest.raises(ValueError, match="skip"):
        run_matched_l1_calibration(
            output_root=tmp_path,
            parent_hash=_PARENT_HASH,
            runner_fn=fake,
            mode="raise",
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
            base_dagma_config=build_protocol_dagma_config(),
            data_loader=object(),
            fit_backend=object(),
            metric_backend=object(),
        )


def test_stage_1_intermediate_written_before_stage_2_runner_call(tmp_path):
    """The Stage-1 intermediate JSON must be on disk before Stage 2 runs.

    Stage A contains a mix of ``soft_frobenius`` (diagnostic) and
    ``matched_l1`` (Stage-1) workloads. Stage B contains only
    ``matched_l1`` workloads. The detection uses the method-family
    composition of the planned-run batch rather than its length, so
    the test does not break if the diagnostic grid size or the
    Stage-1 grid size changes.
    """
    out_root = tmp_path
    intermediate_state: dict[str, bool] = {"present_at_stage_b": False}

    def on_call(planned_tuple: tuple[PlannedRun, ...], base_dir: Path) -> None:
        families = {p.config.method_family for p in planned_tuple}
        # Stage B is detected as the matched_l1-only batch; Stage A
        # always contains both soft_frobenius and matched_l1.
        if families == {"matched_l1"}:
            cal_dir = out_root / cal_mod.CALIBRATION_OUTPUT_SUBDIR
            for sub in cal_dir.iterdir():
                p = sub / STAGE_1_INTERMEDIATE_FILENAME
                if p.exists():
                    intermediate_state["present_at_stage_b"] = True
                    return

    _run_calibration_with_fake_runner(out_root, on_call=on_call)
    assert intermediate_state["present_at_stage_b"] is True


def test_no_evaluation_seeds_in_any_planned_run(tmp_path):
    captured_planned: list[PlannedRun] = []

    def on_call(planned_tuple: tuple[PlannedRun, ...], base_dir: Path) -> None:
        captured_planned.extend(planned_tuple)

    _run_calibration_with_fake_runner(tmp_path, on_call=on_call)
    seen_seeds = {p.config.seed_value for p in captured_planned}
    assert seen_seeds == set(CALIBRATION_SEED_VALUES)
    for s in EVALUATION_SEEDS:
        assert s not in seen_seeds


def test_only_make_main_study_config_constructs_configs(tmp_path):
    """Captured PlannedRun configs are exactly what make_main_study_config
    would have produced (round-trip equality)."""
    captured: list[PlannedRun] = []

    def on_call(planned_tuple: tuple[PlannedRun, ...], base_dir: Path) -> None:
        captured.extend(planned_tuple)

    _run_calibration_with_fake_runner(tmp_path, on_call=on_call)
    for p in captured:
        cfg = p.config
        if cfg.method_family == "soft_frobenius":
            rebuilt = make_main_study_config(
                method_family="soft_frobenius",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=cfg.dagma_config,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
                confidence=cfg.confidence,
                corrupted_prior_spec=cfg.corrupted_prior_spec,
            )
        else:
            rebuilt = make_main_study_config(
                method_family="matched_l1",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=cfg.dagma_config,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
                matched_l1_lambda1=cfg.matched_l1_lambda1,
            )
        assert rebuilt == cfg


# ===========================================================================
# I. Output schemas
# ===========================================================================


def test_stage_1_intermediate_json_schema(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    p = tmp_path / summary.output_dir_relative / STAGE_1_INTERMEDIATE_FILENAME
    payload = json.loads(p.read_text(encoding="utf-8"))
    required = {
        "calibration_run_hash12",
        "parent_heldout_run_hash_full",
        "target_mean_edge_count",
        "target_per_seed_edge_counts",
        "stage_1_candidates",
        "stage_1_candidate_summaries",
        "stage_1_winner_lambda1",
        "stage_2_interval",
        "stage_2_generated_candidates",
        "stage_2_skipped_duplicates",
        "timestamp_utc",
    }
    assert required.issubset(set(payload.keys()))


def test_final_summary_json_schema(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    p = tmp_path / summary.output_dir_relative / SUMMARY_FILENAME
    payload = json.loads(p.read_text(encoding="utf-8"))
    required = {
        "halt_status",
        "parent_heldout_run_hash_full",
        "calibration_run_hash12",
        "output_dir",
        "code_version",
        "target_mean_edge_count",
        "target_per_seed_edge_counts",
        "stage_1_candidates",
        "stage_2_interval",
        "stage_2_generated_candidates",
        "stage_2_candidates",
        "stage_2_skipped_duplicates",
        "all_evaluated_candidates",
        "selected_lambda1",
        "selected_candidate_mean_edge_count",
        "selected_absolute_gap",
        "selected_valid_dag_count",
        "within_one_edge_tolerance",
        "diagnostic_metric_fields_used_for_selection",
        "evaluation_seeds_used",
        "diagnostic_grid_anomalies",
        "selection_rule",
    }
    assert required.issubset(set(payload.keys()))
    assert payload["diagnostic_metric_fields_used_for_selection"] is False
    assert payload["evaluation_seeds_used"] is False
    assert payload["selection_rule"] == SELECTION_RULE


def test_csv_row_fields(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    p = tmp_path / summary.output_dir_relative / TABLE_FILENAME
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    required = {
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
    }
    assert required.issubset(set(rows[0].keys()))
    assert len(rows) > 0


def test_readout_states_metrics_and_evaluation_seeds_not_used(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    text = (
        tmp_path / summary.output_dir_relative / READOUT_FILENAME
    ).read_text(encoding="utf-8")
    assert "SID" in text and "SHD" in text and "MMD" in text
    assert "NOT used for selection" in text
    assert "Evaluation seeds were NOT used" in text


def test_readout_explains_stage_2_duplicate_skipping(tmp_path):
    """The readout must describe Stage-2 duplicate skipping in prose,
    naming the generated count, the skipped count, and the new count.
    """
    summary = _run_calibration_with_fake_runner(tmp_path)
    text = (
        tmp_path / summary.output_dir_relative / READOUT_FILENAME
    ).read_text(encoding="utf-8")
    n_generated = len(summary.stage_2_generated_candidates)
    n_skipped = len(summary.stage_2_skipped_duplicates)
    n_new = len(summary.stage_2_candidates)
    assert f"Stage 2 generated {n_generated} candidate values" in text
    assert "skipped as duplicates" in text
    assert f"{n_skipped} value(s) coincided with Stage 1" in text
    assert f"leaving {n_new} new candidate value(s)" in text


# ===========================================================================
# J. Halt logic
# ===========================================================================


def test_halt_no_valid_dag(tmp_path):
    def all_cyclic(planned: PlannedRun) -> dict[str, Any]:
        return {
            "graph_status": "cyclic",
            "sampler_status": "unavailable_invalid_graph",
            "metric_status": "unavailable_graph_invalid",
            "sid": None,
            "shd": None,
            "mmd": None,
            "metric_runtime_seconds": None,
            "interventions_mmd_path": None,
        }

    summary = _run_calibration_with_fake_runner(
        tmp_path, record_kwargs_override=all_cyclic
    )
    assert summary.halt_status == HALT_NO_VALID_DAG
    assert summary.selected_lambda1 is None


def test_halt_boundary_poor_match(tmp_path):
    """Force the Stage-1 winner to a boundary AND large gap.

    The fake gives lambda1=0.025 (lowest) the smallest gap, then the
    Stage-2 refinement uses interval [0.0125, 0.05]. All Stage-2
    candidates inherit the same scheme; the final winner stays at the
    lower boundary with gap > 1.
    """
    target_mean = 5.0  # diagnostic returns 5

    def edge_low_winner(planned: PlannedRun) -> int:
        cfg = planned.config
        if cfg.method_family == "soft_frobenius":
            return int(target_mean)  # target stays 5
        # matched_l1: edge_count = 12 (target 5, gap 7) but decreases
        # quickly with lambda1. We want lambda1=0.025 to be the closest
        # candidate (smallest gap among Stage 1) yet still have gap > 1.
        # Linear: edge_count = 12 - 50 * lambda1
        lam = float(cfg.matched_l1_lambda1 or 0.0)
        # At lam=0.025: 10.75 -> gap 5.75. At lam=0.05: 9.5 gap 4.5.
        # Larger lam gives smaller gap, contradicting the "low winner"
        # plan; flip the sign.
        ec = int(round(8 + lam * 50))
        return max(1, ec)
    # With edge_low_winner: lam=0.025 -> 9.25(9) gap 4; lam=0.25 -> 20.5
    # gap 15. So lam=0.025 wins. Stage-2 -> [0.0125, 0.05]. Stage-2 candidates
    # also have edge counts close to 8+, all > target+1.

    summary = _run_calibration_with_fake_runner(
        tmp_path, edge_for_planned=edge_low_winner
    )
    assert summary.halt_status == HALT_BOUNDARY_POOR_MATCH
    assert summary.selected_absolute_gap is not None
    assert summary.selected_absolute_gap > CLOSE_MATCH_EDGE_TOLERANCE


def test_completed_halt(tmp_path):
    summary = _run_calibration_with_fake_runner(tmp_path)
    assert summary.halt_status == HALT_COMPLETED
    # With _edge_count_for_planned, Stage-1 winner is lam=0.1 (gap 0).
    # Stage-2 refinement interval (0.075, 0.15) generates new values
    # 0.09375, 0.1125, 0.13125. 0.09375 also achieves gap 0 (5.4375
    # rounds to 5). The selection hierarchy tie-breaks by the smaller
    # lambda1, so the final selected value is the smaller of the
    # gap-0 candidates: 0.09375. Both achieve within-one-edge.
    assert summary.selected_lambda1 == pytest.approx(0.09375)
    assert summary.within_one_edge_tolerance is True


# ===========================================================================
# K. CLI and exit codes
# ===========================================================================


def test_cli_exit_code_completed(tmp_path, monkeypatch):
    fake = make_fake_runner()
    monkeypatch.setattr(cal_mod, "run_main_study", fake)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-hash", _PARENT_HASH,
    ])
    assert rc == 0


def test_cli_exit_code_no_valid_dag(tmp_path, monkeypatch):
    def all_cyclic_override(planned):
        return {
            "graph_status": "cyclic",
            "sampler_status": "unavailable_invalid_graph",
            "metric_status": "unavailable_graph_invalid",
            "sid": None,
            "shd": None,
            "mmd": None,
            "metric_runtime_seconds": None,
            "interventions_mmd_path": None,
        }

    fake = make_fake_runner(record_kwargs_override=all_cyclic_override)
    monkeypatch.setattr(cal_mod, "run_main_study", fake)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-hash", _PARENT_HASH,
    ])
    assert rc == 2


def test_cli_exit_code_boundary_poor_match(tmp_path, monkeypatch):
    """CLI returns 3 when halt_status == 'halt_boundary_poor_match'.

    The fake gives matched_l1 candidates an edge-count schedule that
    makes lam=0.025 (the Stage-1 lower boundary) the closest fit while
    keeping the absolute gap to the target above 1.0 mean edge,
    triggering the outward-boundary halt rule.
    """
    def edge_low_winner(planned: PlannedRun) -> int:
        cfg = planned.config
        if cfg.method_family == "soft_frobenius":
            return 5  # target mean edge count
        lam = float(cfg.matched_l1_lambda1 or 0.0)
        # Linear schedule that monotonically increases with lambda1,
        # so smaller lambda1 wins. Even the smallest Stage-1 value
        # leaves a gap > 1, forcing the boundary-poor-match halt.
        ec = int(round(8 + lam * 50))
        return max(1, ec)

    fake = make_fake_runner(edge_count_for_planned=edge_low_winner)
    monkeypatch.setattr(cal_mod, "run_main_study", fake)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-hash", _PARENT_HASH,
    ])
    assert rc == 3


def test_cli_exit_code_argparse_error_via_systemexit(tmp_path):
    """argparse failure -> SystemExit with non-zero code (CLI returns
    via the argparse error path, not via our _EXIT_ARGS_OR_UNEXPECTED;
    this test documents the argparse contract)."""
    with pytest.raises(SystemExit):
        cli_main([])  # required args missing


# ===========================================================================
# L. Decision-log non-modification
# ===========================================================================


def test_docs_03_decision_log_not_modified_by_calibration(tmp_path):
    # The calibration writes only under tmp_path/results/... There is
    # no docs/03 file under tmp_path. We assert no docs/ directory was
    # created.
    _run_calibration_with_fake_runner(tmp_path)
    assert not (tmp_path / "docs").exists()


# ===========================================================================
# M. Import allowlist / forbidden prefixes
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "hashlib",
    "json",
    "logging",
    "math",
    "subprocess",
    "sys",
    "time",
    "datetime",
    "pathlib",
    "typing",
    "numpy",
    "experiments.main_study.backends",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "experiments.main_study.runner",
    "experiments.main_study.schema",
    "experiments.main_study.priors",
    "experiments.main_study.workloads",
})


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "matplotlib",
    "seaborn",
    "PIL",
    "dagma",
    "dcdi",
    "tests",
)


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_module_imports_are_allowlisted():
    src = Path(cal_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"calibrate_matched_l1.py import {mod!r} is not in the "
            f"allowlist {sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(cal_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"calibrate_matched_l1.py must not import {mod!r}; "
                f"forbidden prefix {forbidden!r}."
            )


# ===========================================================================
# N. Provenance helpers (advisory, smoke only)
# ===========================================================================


def test_capture_code_version_returns_str_or_none():
    val = capture_code_version()
    assert val is None or (isinstance(val, str) and val)


def test_default_utc_factory_returns_zulu_string():
    s = default_utc_factory()
    assert s.endswith("Z")
    assert len(s) == len("YYYY-MM-DDTHH:MM:SSZ")


# ===========================================================================
# O. Factory usage confirmation (CalibrationRunSpec / dataclasses)
# ===========================================================================


def test_calibration_run_spec_carries_protocol_version():
    spec = CalibrationRunSpec(
        parent_heldout_run_hash_full=_PARENT_HASH,
        calibration_run_hash12="0123456789ab",
        output_dir_relative="results/main_study/calibration/matched_l1/0123456789ab",
        code_version=None,
    )
    assert spec.protocol_version == CALIBRATION_PROTOCOL_VERSION


def test_matched_l1_calibration_summary_defaults():
    summary = MatchedL1CalibrationSummary(
        halt_status=HALT_COMPLETED,
        parent_heldout_run_hash_full=_PARENT_HASH,
        calibration_run_hash12="0123456789ab",
        output_dir_relative="results/x/y",
        code_version=None,
        target_mean_edge_count=5.0,
        target_per_seed_edge_counts=(4, 6),
        stage_1_candidates=STAGE_1_CANDIDATES,
        stage_2_interval=None,
        stage_2_generated_candidates=(),
        stage_2_candidates=(),
        stage_2_skipped_duplicates=(),
        all_evaluated_candidates=STAGE_1_CANDIDATES,
        selected_lambda1=0.1,
        selected_candidate_mean_edge_count=5.0,
        selected_absolute_gap=0.0,
        selected_valid_dag_count=2,
        within_one_edge_tolerance=True,
        diagnostic_grid_anomalies=(),
    )
    assert summary.selection_rule == SELECTION_RULE
