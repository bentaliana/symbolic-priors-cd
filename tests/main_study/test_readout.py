"""Tests for the M-9a main-evaluation readout foundation.

All tests run under ``tmp_path``. The real DAGMA wrappers, the real
metric backend, and the real record set are never used. Each
test builds minimal synthetic records and writes only npz/json
artefacts the readout step needs.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from experiments.main_study import readout as readout_mod
from experiments.main_study.priors import (
    CORRUPTION_GRID,
    CorruptedPriorSpec,
)
from experiments.main_study.readout import (
    BASELINE_COMPARISON_CSV,
    BASELINE_CONDITION_LABELS,
    CELL_SUMMARY_CSV,
    CONTINUOUS_W_KEY,
    DEGRADATION_SUMMARY_CSV,
    DIFF_CONVENTION,
    EVALUATION_SEED_VALUES,
    EXPECTED_COUNTS_BY_METHOD,
    EXPECTED_LAMBDA_PRIOR,
    EXPECTED_MAIN_EVALUATION_RUN_HASH12,
    EXPECTED_MATCHED_L1_LAMBDA1,
    EXPECTED_RECORD_COUNT,
    FLAT_CSV_COLUMNS,
    FLAT_RECORDS_CSV,
    FORBIDDEN_CALIBRATION_SEEDS,
    FORBIDDEN_EDGE_ENGAGEMENT_CSV,
    FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV,
    FlatRecordRow,
    METRIC_COLUMNS,
    METRIC_CORRELATIONS_CSV,
    PAIRED_COMPARISON_PAIRS,
    PAIRED_SEED_COMPARISONS_CSV,
    PER_INTERVENTION_MMD_LONG_CSV,
    PER_INTERVENTION_MMD_SUMMARY_CSV,
    PROJECT_THRESHOLD,
    REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV,
    STATISTICS_SUMMARY_JSON,
    STATUS_SUMMARY_CSV,
    THRESHOLDED_ADJACENCY_KEY,
    VALIDATION_SUMMARY_JSON,
    ValidationSummary,
    compute_baseline_comparison,
    compute_degradation_summary,
    compute_forbidden_edge_engagement_summary,
    compute_metric_correlations,
    compute_paired_seed_comparisons,
    compute_per_intervention_mmd_summary,
    compute_prior_edge_engagement,
    compute_reference_forbidden_edge_comparison,
    condition_key,
    extract_per_intervention_mmd,
    flatten_record,
    forbidden_edges_from_record,
    generate_hypothesis_statistics,
    generate_readout_foundation,
    kendall_tau_b,
    linear_slope,
    load_flat_records_csv,
    load_main_evaluation_records,
    load_npz_array,
    main as cli_main,
    main_evaluation_readout_dir,
    main_evaluation_records_dir,
    off_diagonal_edge_count,
    paired_seed_difference,
    pearson_corr,
    rank_values_average_ties,
    record_config_value,
    reference_forbidden_edges_by_seed,
    resolve_artifact_path,
    select_condition,
    spearman_corr,
    summary_stats,
    validate_flat_rows,
    write_cell_summary_csv,
    write_dict_rows_csv,
    write_flat_records_csv,
    write_forbidden_edge_engagement_csv,
    write_statistics_summary_json,
    write_status_summary_csv,
    write_validation_summary_json,
)
from experiments.main_study.records import (
    SCHEMA_VERSION,
    MainStudyRunRecord,
)
from experiments.main_study.run_io import persist_record_atomic
from experiments.main_study.schema import (
    CONFIDENCE_GRID,
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import make_planned_run
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_RUN_HASH12 = EXPECTED_MAIN_EVALUATION_RUN_HASH12  # "864fe6722256"
_N_NODES = 10
_EXPECTED_EDGES = 20
_GENERATED_AT = "2026-05-25T12:00:00Z"


def _corrupted_spec(
    *,
    seed: int,
    corruption_fraction: float,
    corruption_index: int,
    forbidden: tuple[tuple[int, int], ...],
) -> CorruptedPriorSpec:
    labels = {f"{i},{j}": "true_negative_retained" for (i, j) in forbidden}
    return CorruptedPriorSpec(
        n_nodes=_N_NODES,
        scm_seed=seed,
        corruption_fraction=float(corruption_fraction),
        corruption_index=int(corruption_index),
        corruption_seed=9100 + seed + corruption_index,
        forbidden_edges=forbidden,
        n_correct=len(forbidden),
        n_corrupted=0,
        removed_clean_edges=(),
        added_true_positive_edges=(),
        edge_labels=labels,
    )


_DEFAULT_FORBIDDEN: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (0, 6), (0, 7), (0, 8), (0, 9), (1, 2),
)


def _make_config(
    *,
    method_family: str,
    seed: int,
    confidence: Optional[float] = None,
    corruption_fraction: Optional[float] = None,
    corruption_index: Optional[int] = None,
    matched_l1: Optional[float] = None,
) -> MainStudyConfig:
    dagma_lambda1 = 0.05
    if method_family == "matched_l1":
        dagma_lambda1 = EXPECTED_MATCHED_L1_LAMBDA1
    base_dagma = DAGMAConfig(lambda1=dagma_lambda1)
    kwargs: dict[str, Any] = dict(
        method_family=method_family,
        seed_value=seed,
        seed_population="main_evaluation",
        dagma_config=base_dagma,
        parent_heldout_run_hash_full=_PARENT_HASH,
    )
    if method_family == "soft_frobenius":
        spec = _corrupted_spec(
            seed=seed,
            corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
            forbidden=_DEFAULT_FORBIDDEN,
        )
        kwargs["confidence"] = confidence
        kwargs["corrupted_prior_spec"] = spec
    elif method_family == "hard_exclusion":
        spec = _corrupted_spec(
            seed=seed,
            corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
            forbidden=_DEFAULT_FORBIDDEN,
        )
        kwargs["corrupted_prior_spec"] = spec
    elif method_family == "matched_l1":
        kwargs["matched_l1_lambda1"] = (
            matched_l1
            if matched_l1 is not None
            else EXPECTED_MATCHED_L1_LAMBDA1
        )
    return make_main_study_config(**kwargs)


def _write_artefacts(
    planned, *, base_dir: Path,
    thresholded: np.ndarray, continuous_w: np.ndarray,
) -> None:
    for name, rel in planned.artefact_paths.items():
        full = base_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        if name == "thresholded_adjacency.npz":
            np.savez(full, thresholded_adjacency=thresholded)
        elif name == "continuous_w.npz":
            np.savez(full, continuous_w=continuous_w)
        elif name == "true_adjacency.npz":
            np.savez(
                full,
                true_adjacency=np.zeros(
                    (_N_NODES, _N_NODES), dtype=bool
                ),
            )
        elif name == "confidence_mask.npz":
            np.savez(
                full,
                confidence_mask=np.zeros(
                    (_N_NODES, _N_NODES), dtype=float
                ),
            )
        elif name == "interventions_mmd.json":
            full.write_text(
                json.dumps({"records": [], "mmd_primary": 0.001}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_clean.json":
            full.write_text(
                json.dumps({"n_nodes": _N_NODES, "forbidden_edges": []}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_corrupted.json":
            full.write_text(
                json.dumps({
                    "n_nodes": _N_NODES,
                    "corruption_fraction": 0.0,
                    "forbidden_edges": [],
                }),
                encoding="utf-8",
            )
        elif name == "per_edge_labels.json":
            full.write_text(json.dumps({}), encoding="utf-8")


def _make_record(
    planned,
    *,
    base_dir: Path,
    sid: Optional[float] = 5.0,
    shd: Optional[float] = 3.0,
    mmd: Optional[float] = 0.01,
    fit_status: str = "success",
    metric_status: str = "computed",
    graph_status: Optional[str] = "valid_dag",
    sampler_status: Optional[str] = "available",
    failure_kind: Optional[str] = None,
    failure_message: str = "",
    thresholded: Optional[np.ndarray] = None,
    continuous_w: Optional[np.ndarray] = None,
) -> MainStudyRunRecord:
    if thresholded is None:
        thresholded = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    if continuous_w is None:
        continuous_w = np.zeros((_N_NODES, _N_NODES), dtype=float)
    _write_artefacts(
        planned,
        base_dir=base_dir,
        thresholded=thresholded,
        continuous_w=continuous_w,
    )
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=_N_NODES,
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
    kwargs["continuous_w_path"] = planned.artefact_paths["continuous_w.npz"]
    kwargs["thresholded_adjacency_path"] = planned.artefact_paths[
        "thresholded_adjacency.npz"
    ]
    kwargs["true_adjacency_path"] = planned.artefact_paths[
        "true_adjacency.npz"
    ]
    if metric_status == "computed":
        kwargs["sid"] = sid
        kwargs["shd"] = shd
        kwargs["mmd"] = mmd
        kwargs["metric_runtime_seconds"] = 0.2
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
    elif family == "hard_exclusion":
        kwargs["prior_edge_set_clean_path"] = planned.artefact_paths[
            "prior_edge_set_clean.json"
        ]
        kwargs["prior_edge_set_corrupted_path"] = planned.artefact_paths[
            "prior_edge_set_corrupted.json"
        ]
        kwargs["per_edge_labels_path"] = planned.artefact_paths[
            "per_edge_labels.json"
        ]
    return MainStudyRunRecord(**kwargs)


def _persist(record, planned, base_dir: Path) -> None:
    persist_record_atomic(record, planned.record_path, base_dir=base_dir)


def _build_full_grid(base_dir: Path) -> None:
    """Persist the canonical 224-record main-evaluation grid under base_dir.

    Records and artefacts go under
    ``base_dir/results/main_study/<_RUN_HASH12>/...``.
    """
    for seed in EVALUATION_SEED_VALUES:
        # prior_free
        cfg = _make_config(method_family="prior_free", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        rec = _make_record(planned, base_dir=base_dir)
        _persist(rec, planned, base_dir)
        # matched_l1
        cfg = _make_config(method_family="matched_l1", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        rec = _make_record(planned, base_dir=base_dir)
        _persist(rec, planned, base_dir)
        # hard_exclusion: 5 corruption levels
        for idx, cf in enumerate(CORRUPTION_GRID):
            cfg = _make_config(
                method_family="hard_exclusion",
                seed=seed,
                corruption_fraction=cf,
                corruption_index=idx,
            )
            planned = make_planned_run(cfg, _RUN_HASH12)
            rec = _make_record(planned, base_dir=base_dir)
            _persist(rec, planned, base_dir)
        # soft_frobenius: 5 corruption x 5 confidence
        for idx, cf in enumerate(CORRUPTION_GRID):
            for cn in CONFIDENCE_GRID:
                cfg = _make_config(
                    method_family="soft_frobenius",
                    seed=seed,
                    confidence=cn,
                    corruption_fraction=cf,
                    corruption_index=idx,
                )
                planned = make_planned_run(cfg, _RUN_HASH12)
                rec = _make_record(planned, base_dir=base_dir)
                _persist(rec, planned, base_dir)


# ===========================================================================
# A. Loading and path resolution
# ===========================================================================


def test_load_main_evaluation_records_rejects_empty_directory(tmp_path):
    rd = main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    rd.mkdir(parents=True)
    with pytest.raises(ValueError, match="no .*records"):
        load_main_evaluation_records(rd)


def test_load_main_evaluation_records_sorts_deterministically(tmp_path):
    # Build two records out of natural lexical order.
    cfg_a = _make_config(method_family="prior_free", seed=502)
    cfg_b = _make_config(method_family="prior_free", seed=501)
    p_a = make_planned_run(cfg_a, _RUN_HASH12)
    p_b = make_planned_run(cfg_b, _RUN_HASH12)
    _persist(_make_record(p_a, base_dir=tmp_path), p_a, tmp_path)
    _persist(_make_record(p_b, base_dir=tmp_path), p_b, tmp_path)
    rd = main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    a = load_main_evaluation_records(rd)
    b = load_main_evaluation_records(rd)
    assert tuple(r.run_id for r in a) == tuple(r.run_id for r in b)
    assert list(r.run_id for r in a) == sorted(r.run_id for r in a)


def test_load_main_evaluation_records_rejects_duplicate_run_id(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    _persist(rec, planned, tmp_path)
    # Manually write a second file with the same run_id under a
    # different filename.
    rd = main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    duplicate = rd / "duplicate.json"
    duplicate.write_text(
        (rd / f"{planned.run_id}.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate run_id"):
        load_main_evaluation_records(rd)


def test_load_main_evaluation_records_rejects_duplicate_configuration_hash(
    tmp_path,
):
    """Two distinct planned-run JSON files cannot share a config hash."""
    cfg_a = _make_config(method_family="prior_free", seed=501)
    planned_a = make_planned_run(cfg_a, _RUN_HASH12)
    rec_a = _make_record(planned_a, base_dir=tmp_path)
    _persist(rec_a, planned_a, tmp_path)
    rd = main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    # Duplicate: copy the record but change its filename and rewrite
    # the inner run_id (so the run_id check doesn't fire first).
    text = (rd / f"{planned_a.run_id}.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    payload["run_id"] = "different_run_id"
    (rd / "other.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_main_evaluation_records(rd)


def test_resolve_artifact_path_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_artifact_path(
            "results/main_study/x/artefacts/r/thresholded_adjacency.npz",
            base_dir=tmp_path,
        )


def test_load_npz_array_rejects_missing_key(tmp_path):
    rel = "results/main_study/x/artefacts/r/data.npz"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    np.savez(full, wrong_key=np.zeros(3))
    with pytest.raises(ValueError, match="missing expected key"):
        load_npz_array(
            rel, base_dir=tmp_path, expected_key=THRESHOLDED_ADJACENCY_KEY
        )


# ===========================================================================
# B. Edge count and engagement
# ===========================================================================


def test_off_diagonal_edge_count_excludes_diagonal():
    mat = np.array([
        [True, True, False],
        [False, True, True],
        [True, False, True],
    ])
    assert off_diagonal_edge_count(mat) == 3


def test_off_diagonal_edge_count_rejects_non_square():
    with pytest.raises(ValueError, match="2D square"):
        off_diagonal_edge_count(np.zeros((3, 4), dtype=bool))


def test_off_diagonal_edge_count_rejects_float():
    with pytest.raises(ValueError, match="bool-like or integer"):
        off_diagonal_edge_count(np.zeros((3, 3), dtype=float))


def test_forbidden_edges_from_record_for_hard_exclusion(tmp_path):
    cfg = _make_config(
        method_family="hard_exclusion",
        seed=501,
        corruption_fraction=0.0,
        corruption_index=0,
    )
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    fe = forbidden_edges_from_record(rec)
    assert fe == _DEFAULT_FORBIDDEN


def test_forbidden_edges_from_record_returns_none_for_prior_free(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    assert forbidden_edges_from_record(rec) is None


def test_forbidden_edges_from_record_returns_none_for_matched_l1(tmp_path):
    cfg = _make_config(method_family="matched_l1", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    assert forbidden_edges_from_record(rec) is None


def test_compute_prior_edge_engagement_basic():
    n = 4
    w = np.zeros((n, n), dtype=float)
    # Targeted (forbidden) edges with various |W| values.
    forbidden = ((0, 1), (0, 2), (0, 3))
    w[0, 1] = 0.5   # above threshold (0.3)
    w[0, 2] = 0.1   # below
    w[0, 3] = -0.4  # |W|=0.4 above
    # Non-targeted off-diagonals.
    w[1, 0] = 0.2
    w[2, 0] = 0.0
    w[3, 0] = 0.0
    # diagonal ignored
    out = compute_prior_edge_engagement(
        w, forbidden, threshold=PROJECT_THRESHOLD
    )
    assert out["n_targeted_forbidden_edges"] == 3
    assert out["mean_abs_w_targeted_forbidden_edges"] == pytest.approx(
        (0.5 + 0.1 + 0.4) / 3
    )
    # Two of three are >= 0.3 -> fraction 2/3.
    assert out["fraction_targeted_forbidden_above_threshold"] == pytest.approx(
        2 / 3
    )
    # Non-targeted off-diagonals: (1,0)=0.2, (1,2)=0, (1,3)=0,
    # (2,0)=0, (2,1)=0, (2,3)=0, (3,0)=0, (3,1)=0, (3,2)=0
    # 9 entries, mean = 0.2/9
    assert out["mean_abs_w_non_targeted_edges"] == pytest.approx(0.2 / 9)


def test_compute_prior_edge_engagement_none_inputs_return_none_fields():
    out = compute_prior_edge_engagement(None, None)
    assert out["n_targeted_forbidden_edges"] is None
    assert out["mean_abs_w_targeted_forbidden_edges"] is None
    assert out["fraction_targeted_forbidden_above_threshold"] is None
    assert out["mean_abs_w_non_targeted_edges"] is None


# ===========================================================================
# C. Flattening
# ===========================================================================


def test_flatten_record_extracts_corruption_fraction(tmp_path):
    cfg = _make_config(
        method_family="hard_exclusion",
        seed=501,
        corruption_fraction=0.4,
        corruption_index=2,
    )
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.corruption_fraction == pytest.approx(0.4)
    assert row.corruption_index == 2


def test_flatten_record_extracts_confidence(tmp_path):
    cfg = _make_config(
        method_family="soft_frobenius",
        seed=501,
        confidence=0.75,
        corruption_fraction=0.2,
        corruption_index=1,
    )
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.confidence == pytest.approx(0.75)


def test_flatten_record_extracts_dagma_lambda1(tmp_path):
    cfg = _make_config(method_family="matched_l1", seed=502)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.dagma_lambda1 == pytest.approx(EXPECTED_MATCHED_L1_LAMBDA1)


def test_flatten_record_edge_count_from_thresholded_not_continuous(
    tmp_path,
):
    """edge_count must come from the persisted thresholded adjacency,
    not from a count of nonzero entries in continuous_w."""
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    # Build a thresholded matrix with 3 off-diagonal edges; build a
    # dense continuous_w whose nonzero-count would suggest many more.
    thr = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    thr[0, 1] = True
    thr[1, 2] = True
    thr[2, 3] = True
    cont = np.ones((_N_NODES, _N_NODES), dtype=float) * 0.5
    rec = _make_record(
        planned,
        base_dir=tmp_path,
        thresholded=thr,
        continuous_w=cont,
    )
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.edge_count_from_thresholded_adjacency == 3


def test_flatten_record_engagement_from_continuous_w(tmp_path):
    cfg = _make_config(
        method_family="hard_exclusion",
        seed=501,
        corruption_fraction=0.0,
        corruption_index=0,
    )
    planned = make_planned_run(cfg, _RUN_HASH12)
    cont = np.zeros((_N_NODES, _N_NODES), dtype=float)
    # forbidden = _DEFAULT_FORBIDDEN; set the first one above threshold
    cont[_DEFAULT_FORBIDDEN[0]] = 0.9
    rec = _make_record(
        planned, base_dir=tmp_path, continuous_w=cont,
    )
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.n_targeted_forbidden_edges == len(_DEFAULT_FORBIDDEN)
    # Fraction above threshold = 1/10 because only one forbidden entry
    # exceeds 0.3 in absolute value.
    assert row.fraction_targeted_forbidden_above_threshold == pytest.approx(
        1 / len(_DEFAULT_FORBIDDEN)
    )


def test_flatten_record_accepts_finite_negative_mmd(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path, mmd=-1e-3)
    row = flatten_record(rec, base_dir=tmp_path)
    assert row.mmd == pytest.approx(-1e-3)


def test_flatten_record_rejects_nan_mmd(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    # Build a record where mmd will be NaN; MainStudyRunRecord
    # itself validates this for computed metric_status, so we
    # bypass via object.__setattr__ to test the readout-side guard.
    rec = _make_record(planned, base_dir=tmp_path)
    object.__setattr__(rec, "mmd", float("nan"))
    with pytest.raises(ValueError, match="mmd"):
        flatten_record(rec, base_dir=tmp_path)


def test_flatten_record_rejects_negative_sid(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    object.__setattr__(rec, "sid", -1.0)
    with pytest.raises(ValueError, match="sid"):
        flatten_record(rec, base_dir=tmp_path)


def test_flatten_record_rejects_negative_shd(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    object.__setattr__(rec, "shd", -1.0)
    with pytest.raises(ValueError, match="shd"):
        flatten_record(rec, base_dir=tmp_path)


# ===========================================================================
# D. Validation
# ===========================================================================


def _build_full_rows(base_dir: Path) -> tuple[FlatRecordRow, ...]:
    _build_full_grid(base_dir)
    rd = main_evaluation_records_dir(base_dir, _RUN_HASH12)
    records = load_main_evaluation_records(rd)
    return tuple(
        flatten_record(rec, base_dir=base_dir) for rec in records
    )


def test_validate_flat_rows_accepts_full_grid(tmp_path):
    rows = _build_full_rows(tmp_path)
    summary = validate_flat_rows(rows, strict=True)
    assert summary.n_records == EXPECTED_RECORD_COUNT
    assert summary.method_family_counts == EXPECTED_COUNTS_BY_METHOD
    assert summary.seed_values == tuple(sorted(EVALUATION_SEED_VALUES))
    assert summary.soft_frobenius_cell_count == 175
    assert summary.hard_exclusion_cell_count == 35
    assert summary.prior_free_count == 7
    assert summary.matched_l1_count == 7
    assert summary.validation_errors == ()


def test_validate_flat_rows_rejects_wrong_total_count(tmp_path):
    rows = _build_full_rows(tmp_path)
    truncated = rows[:-1]
    with pytest.raises(ValueError, match="exactly 224"):
        validate_flat_rows(truncated, strict=True)


def test_validate_flat_rows_rejects_wrong_method_family_count(tmp_path):
    rows = _build_full_rows(tmp_path)
    # Drop one prior_free row -> total 223 AND prior_free count off.
    dropped_one = False
    kept = []
    for r in rows:
        if not dropped_one and r.method_family == "prior_free":
            dropped_one = True
            continue
        kept.append(r)
    with pytest.raises(ValueError, match="prior_free|exactly 224"):
        validate_flat_rows(tuple(kept), strict=True)


def test_validate_flat_rows_rejects_calibration_seed(tmp_path):
    rows = list(_build_full_rows(tmp_path))
    # Mutate one row to use a calibration seed.
    bad = dataclasses_replace(rows[0], seed_value=401)
    rows[0] = bad
    with pytest.raises(ValueError, match="calibration seed"):
        validate_flat_rows(tuple(rows), strict=True)


def test_validate_flat_rows_rejects_missing_soft_frobenius_cell(tmp_path):
    rows = _build_full_rows(tmp_path)
    # Drop the first soft_frobenius row only -> total 223 AND
    # soft_frobenius missing one cell.
    dropped_one = False
    kept = []
    for r in rows:
        if not dropped_one and r.method_family == "soft_frobenius":
            dropped_one = True
            continue
        kept.append(r)
    with pytest.raises(ValueError, match="missing cells|exactly 224"):
        validate_flat_rows(tuple(kept), strict=True)


def test_validate_flat_rows_rejects_hard_exclusion_confidence_axis(tmp_path):
    rows = list(_build_full_rows(tmp_path))
    # Mutate first hard_exclusion row to carry a confidence value.
    for i, r in enumerate(rows):
        if r.method_family == "hard_exclusion":
            rows[i] = dataclasses_replace(r, confidence=0.5)
            break
    with pytest.raises(ValueError, match="must not carry a confidence"):
        validate_flat_rows(tuple(rows), strict=True)


def test_validate_flat_rows_rejects_matched_l1_lambda_mismatch(tmp_path):
    rows = list(_build_full_rows(tmp_path))
    for i, r in enumerate(rows):
        if r.method_family == "matched_l1":
            rows[i] = dataclasses_replace(
                r, matched_l1_lambda1=0.1, dagma_lambda1=0.1
            )
            break
    with pytest.raises(ValueError, match="matched_l1"):
        validate_flat_rows(tuple(rows), strict=True)


def test_validate_flat_rows_rejects_soft_frobenius_lambda_prior_mismatch(
    tmp_path,
):
    rows = list(_build_full_rows(tmp_path))
    for i, r in enumerate(rows):
        if r.method_family == "soft_frobenius":
            rows[i] = dataclasses_replace(r, lambda_prior=0.001)
            break
    with pytest.raises(ValueError, match="lambda_prior"):
        validate_flat_rows(tuple(rows), strict=True)


def test_validate_flat_rows_rejects_non_success_status(tmp_path):
    rows = list(_build_full_rows(tmp_path))
    rows[0] = dataclasses_replace(rows[0], fit_status="model_fit_failure")
    with pytest.raises(ValueError, match="fit_status"):
        validate_flat_rows(tuple(rows), strict=True)


def test_validate_flat_rows_rejects_non_finite_metric(tmp_path):
    rows = list(_build_full_rows(tmp_path))
    rows[0] = dataclasses_replace(rows[0], sid=float("inf"))
    with pytest.raises(ValueError, match="sid"):
        validate_flat_rows(tuple(rows), strict=True)


# Small helper since dataclasses.replace works on frozen kw_only ones.
def dataclasses_replace(obj: FlatRecordRow, **changes) -> FlatRecordRow:
    import dataclasses as _dc
    return _dc.replace(obj, **changes)


# ===========================================================================
# E. Output writing
# ===========================================================================


def test_write_flat_records_csv_canonical_column_order(tmp_path):
    rows = _build_full_rows(tmp_path)
    out = tmp_path / "out.csv"
    write_flat_records_csv(rows, out)
    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert tuple(header) == FLAT_CSV_COLUMNS
    assert len(body) == EXPECTED_RECORD_COUNT


def test_write_flat_records_csv_none_as_empty(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    row = flatten_record(rec, base_dir=tmp_path)
    out = tmp_path / "single.csv"
    write_flat_records_csv((row,), out)
    text = out.read_text(encoding="utf-8")
    # Confidence and corruption_fraction are None for prior_free.
    forbidden_tokens = ["None", "null", "NaN", "nan"]
    for token in forbidden_tokens:
        assert (
            f",{token}," not in text
        ), f"None should be empty; found {token!r} in CSV"
    # Header followed by one row.
    lines = [line for line in text.split("\n") if line]
    assert len(lines) == 2


def test_write_status_summary_csv(tmp_path):
    rows = _build_full_rows(tmp_path)
    out = tmp_path / "status.csv"
    write_status_summary_csv(rows, out)
    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        body = list(reader)
    assert {"method_family", "fit_status", "count"}.issubset(
        set(body[0].keys())
    )
    total = sum(int(r["count"]) for r in body)
    assert total == EXPECTED_RECORD_COUNT


def test_write_cell_summary_csv(tmp_path):
    rows = _build_full_rows(tmp_path)
    out = tmp_path / "cells.csv"
    write_cell_summary_csv(rows, out)
    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        body = list(reader)
    cols = set(body[0].keys())
    for required in (
        "method_family", "corruption_fraction", "confidence",
        "n", "sid_mean", "sid_std", "sid_median",
        "shd_mean", "shd_std", "shd_median",
        "mmd_mean", "mmd_std", "mmd_median",
        "edge_count_mean", "edge_count_std", "edge_count_median",
    ):
        assert required in cols
    # Sum of cell counts equals total record count.
    total = sum(int(r["n"]) for r in body)
    assert total == EXPECTED_RECORD_COUNT


def test_write_forbidden_edge_engagement_csv(tmp_path):
    rows = _build_full_rows(tmp_path)
    out = tmp_path / "engagement.csv"
    write_forbidden_edge_engagement_csv(rows, out)
    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        body = list(reader)
    assert len(body) == EXPECTED_RECORD_COUNT
    for required in (
        "run_id", "method_family", "seed_value",
        "corruption_fraction", "confidence",
        "n_targeted_forbidden_edges",
        "mean_abs_w_targeted_forbidden_edges",
        "fraction_targeted_forbidden_above_threshold",
        "mean_abs_w_non_targeted_edges",
        "edge_count_from_thresholded_adjacency",
    ):
        assert required in body[0]


def test_write_validation_summary_json_fields(tmp_path):
    rows = _build_full_rows(tmp_path)
    summary = validate_flat_rows(rows, strict=True)
    out = tmp_path / "vs.json"
    write_validation_summary_json(summary, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    required = {
        "main_evaluation_run_hash12",
        "n_records",
        "method_family_counts",
        "seed_values",
        "all_statuses_computed",
        "all_metrics_finite",
        "all_required_artifacts_resolved",
        "soft_frobenius_cell_count",
        "hard_exclusion_cell_count",
        "prior_free_count",
        "matched_l1_count",
        "validation_errors",
    }
    assert required.issubset(set(payload.keys()))


def test_generate_readout_foundation_writes_all_outputs(tmp_path):
    _build_full_grid(tmp_path)
    summary = generate_readout_foundation(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        strict=True,
    )
    rd = main_evaluation_readout_dir(tmp_path, _RUN_HASH12)
    assert (rd / FLAT_RECORDS_CSV).exists()
    assert (rd / CELL_SUMMARY_CSV).exists()
    assert (rd / STATUS_SUMMARY_CSV).exists()
    assert (rd / FORBIDDEN_EDGE_ENGAGEMENT_CSV).exists()
    assert (rd / VALIDATION_SUMMARY_JSON).exists()
    assert summary.n_records == EXPECTED_RECORD_COUNT
    assert summary.validation_errors == ()


def test_csv_round_trip_determinism(tmp_path):
    rows = _build_full_rows(tmp_path)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    write_flat_records_csv(rows, out_a)
    write_flat_records_csv(rows, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()
    with out_a.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        body = list(reader)
    # Verify identity columns round-tripped.
    by_id = {r.run_id: r for r in rows}
    for csv_row in body:
        orig = by_id[csv_row["run_id"]]
        assert csv_row["configuration_hash_full"] == (
            orig.configuration_hash_full
        )
        assert csv_row["method_family"] == orig.method_family


# ===========================================================================
# F. CLI
# ===========================================================================


def test_cli_returns_zero_on_success(tmp_path):
    _build_full_grid(tmp_path)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 0


def test_cli_returns_one_on_validation_failure(tmp_path):
    _build_full_grid(tmp_path)
    # Sabotage one record: rewrite a prior_free record with a
    # different run_id so subsequent runs duplicate the
    # configuration_hash.
    rd = main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    files = sorted(rd.glob("prior_free*.json"))
    assert len(files) >= 2
    target = files[0]
    other = files[1]
    target.write_text(
        other.read_text(encoding="utf-8"), encoding="utf-8"
    )
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 1


# ===========================================================================
# G. Scope / imports
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "json",
    "math",
    "pathlib",
    "statistics",
    "sys",
    "typing",
    "numpy",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "experiments.main_study.schema",
    "experiments.main_study.priors",
})


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.backends",
    "experiments.main_study.executor",
    "experiments.main_study.runner",
    "experiments.main_study.workloads",
    "experiments.main_study.run_main_evaluation",
    "experiments.main_study.calibrate_matched_l1",
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
    src = Path(readout_mod.__file__).read_text(encoding="utf-8")
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
            f"readout.py import {mod!r} is not in the allowlist "
            f"{sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(readout_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"readout.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


# ===========================================================================
# H. Side-effect discipline
# ===========================================================================


def test_no_writes_outside_tmp_path(tmp_path):
    _build_full_grid(tmp_path)
    generate_readout_foundation(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        strict=True,
    )
    # Everything we wrote is under tmp_path.
    assert (tmp_path / "results").is_dir()


def test_no_notebook_or_plot_files_are_created(tmp_path):
    _build_full_grid(tmp_path)
    generate_readout_foundation(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        strict=True,
    )
    for ext in ("*.ipynb", "*.png", "*.jpg", "*.pdf", "*.svg"):
        assert not list((tmp_path).rglob(ext)), (
            f"no {ext} files should be produced by M-9a"
        )


def test_docs_03_not_modified(tmp_path):
    _build_full_grid(tmp_path)
    generate_readout_foundation(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        strict=True,
    )
    assert not (tmp_path / "docs").exists()



# ===========================================================================
# ESTS
# ===========================================================================


def _build_grid_and_flat(tmp_path: Path) -> tuple[tuple[dict, ...], Path]:
    """Persist the canonical 224-record grid plus the M-9a flat CSV.

    Each record gets sid/shd/mmd that vary by confidence and
    corruption so downstream slopes and correlations have
    non-trivial values.
    """
    for seed in EVALUATION_SEED_VALUES:
        # prior_free
        cfg = _make_config(method_family="prior_free", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        rec = _make_record(
            planned, base_dir=tmp_path,
            sid=10.0 + seed - 501,
            shd=6.0 + (seed - 501) * 0.5,
            mmd=0.05 + (seed - 501) * 0.005,
        )
        _persist(rec, planned, tmp_path)
        # matched_l1
        cfg = _make_config(method_family="matched_l1", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        rec = _make_record(
            planned, base_dir=tmp_path,
            sid=8.0 + (seed - 501) * 0.5,
            shd=5.0 + (seed - 501) * 0.5,
            mmd=0.04 + (seed - 501) * 0.005,
        )
        _persist(rec, planned, tmp_path)
        # hard_exclusion: corruption-axis only
        for idx, cf in enumerate(CORRUPTION_GRID):
            cfg = _make_config(
                method_family="hard_exclusion",
                seed=seed,
                corruption_fraction=cf,
                corruption_index=idx,
            )
            planned = make_planned_run(cfg, _RUN_HASH12)
            rec = _make_record(
                planned, base_dir=tmp_path,
                sid=5.0 + cf * 10,
                shd=3.0 + cf * 5,
                mmd=0.03 + cf * 0.04,
            )
            _persist(rec, planned, tmp_path)
        # soft_frobenius: 5x5 grid
        for idx, cf in enumerate(CORRUPTION_GRID):
            for cn in CONFIDENCE_GRID:
                cfg = _make_config(
                    method_family="soft_frobenius",
                    seed=seed,
                    confidence=cn,
                    corruption_fraction=cf,
                    corruption_index=idx,
                )
                planned = make_planned_run(cfg, _RUN_HASH12)
                rec = _make_record(
                    planned, base_dir=tmp_path,
                    sid=4.0 + cf * 8 + (1 - cn) * 2,
                    shd=2.0 + cf * 4 + (1 - cn) * 1,
                    mmd=0.02 + cf * 0.05 + (1 - cn) * 0.01,
                )
                _persist(rec, planned, tmp_path)
    generate_readout_foundation(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        strict=True,
    )
    flat_csv = main_evaluation_readout_dir(
        tmp_path, _RUN_HASH12
    ) / FLAT_RECORDS_CSV
    flat = load_flat_records_csv(flat_csv)
    return flat, flat_csv


# ---------------------------------------------------------------------------
# I. load_flat_records_csv
# ---------------------------------------------------------------------------


def test_load_flat_records_csv_exact_column_order(tmp_path):
    flat, csv_path = _build_grid_and_flat(tmp_path)
    assert len(flat) == EXPECTED_RECORD_COUNT


def test_load_flat_records_csv_rejects_wrong_header(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("not,the,canonical,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical column order"):
        load_flat_records_csv(bad)


def test_load_flat_records_csv_empty_cells_become_none(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    pf = [r for r in flat if r["method_family"] == "prior_free"][0]
    assert pf["confidence"] is None
    assert pf["corruption_fraction"] is None


def test_load_flat_records_csv_numeric_coercion(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    soft = [r for r in flat if r["method_family"] == "soft_frobenius"][0]
    assert isinstance(soft["confidence"], float)
    assert isinstance(soft["seed_value"], int)
    assert isinstance(soft["edge_count_from_thresholded_adjacency"], int)
    assert isinstance(soft["sid"], float)


# ---------------------------------------------------------------------------
# J. Condition helpers
# ---------------------------------------------------------------------------


def test_condition_key_all_families(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    keys = {condition_key(r) for r in flat}
    assert ("prior_free", None, None) in keys
    assert ("matched_l1", None, None) in keys
    assert ("soft_frobenius", 0.0, 1.0) in keys
    assert ("hard_exclusion", 0.0, None) in keys


def test_select_condition_seed_sorted(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = select_condition(flat, method_family="prior_free")
    assert tuple(r["seed_value"] for r in rows) == tuple(
        sorted(EVALUATION_SEED_VALUES)
    )


def test_select_condition_clean_soft_conf1(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = select_condition(
        flat,
        method_family="soft_frobenius",
        corruption_fraction=0.0,
        confidence=1.0,
    )
    assert len(rows) == 7


# ---------------------------------------------------------------------------
# K. Descriptive summaries
# ---------------------------------------------------------------------------


def test_summary_stats_sample_std_and_median():
    s = summary_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(3.0)
    assert s["median"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(math.sqrt(2.5))
    assert s["min"] == 1.0
    assert s["max"] == 5.0


def test_summary_stats_accepts_negative_mmd():
    s = summary_stats([-0.01, 0.0, 0.01])
    assert s["n"] == 3
    assert s["min"] < 0.0


def test_summary_stats_empty():
    s = summary_stats([None, None])
    assert s["n"] == 0
    assert s["mean"] is None


def test_compute_baseline_comparison_four_conditions_x_all_metrics(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_baseline_comparison(flat)
    assert len(rows) == 16
    labels = {r["condition_label"] for r in rows}
    assert labels == set(BASELINE_CONDITION_LABELS)
    metrics = {r["metric"] for r in rows}
    assert metrics == set(METRIC_COLUMNS)


# ---------------------------------------------------------------------------
# L. Paired-seed comparisons
# ---------------------------------------------------------------------------


def test_paired_seed_difference_pairs_by_seed(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    a = select_condition(flat, method_family="prior_free")
    b = select_condition(flat, method_family="matched_l1")
    out = paired_seed_difference(
        a, b, metric="sid",
        label_a="prior_free", label_b="matched_l1",
        n_bootstrap=200, random_seed=42,
    )
    assert out["n_pairs"] == 7
    assert out["diff_convention"] == "a - b"


def test_paired_seed_difference_rejects_mismatched_seeds():
    rows_a = [{"seed_value": 501, "sid": 5.0}]
    rows_b = [{"seed_value": 502, "sid": 4.0}]
    with pytest.raises(ValueError, match="seed sets differ"):
        paired_seed_difference(
            rows_a, rows_b, metric="sid",
            label_a="A", label_b="B",
            n_bootstrap=10, random_seed=1,
        )


def test_paired_seed_difference_diff_convention_is_a_minus_b():
    rows_a = [
        {"seed_value": 501, "sid": 10.0},
        {"seed_value": 502, "sid": 12.0},
    ]
    rows_b = [
        {"seed_value": 501, "sid": 4.0},
        {"seed_value": 502, "sid": 5.0},
    ]
    out = paired_seed_difference(
        rows_a, rows_b, metric="sid",
        label_a="A", label_b="B",
        n_bootstrap=100, random_seed=1,
    )
    assert out["mean_diff"] == pytest.approx(6.5)
    assert out["wins_a_lower"] == 0
    assert out["wins_b_lower"] == 2


def test_paired_seed_bootstrap_ci_is_deterministic():
    rows_a = [
        {"seed_value": s, "sid": float(s)}
        for s in EVALUATION_SEED_VALUES
    ]
    rows_b = [
        {"seed_value": s, "sid": float(s) - 1.0}
        for s in EVALUATION_SEED_VALUES
    ]
    a = paired_seed_difference(
        rows_a, rows_b, metric="sid",
        label_a="A", label_b="B",
        n_bootstrap=500, random_seed=12345,
    )
    b = paired_seed_difference(
        rows_a, rows_b, metric="sid",
        label_a="A", label_b="B",
        n_bootstrap=500, random_seed=12345,
    )
    assert a["bootstrap_ci_low"] == b["bootstrap_ci_low"]
    assert a["bootstrap_ci_high"] == b["bootstrap_ci_high"]


def test_paired_seed_wins_counts_lower_is_better():
    rows_a = [{"seed_value": 501, "sid": 2.0}, {"seed_value": 502, "sid": 9.0}]
    rows_b = [{"seed_value": 501, "sid": 4.0}, {"seed_value": 502, "sid": 9.0}]
    out = paired_seed_difference(
        rows_a, rows_b, metric="sid",
        label_a="A", label_b="B",
        n_bootstrap=10, random_seed=1,
    )
    assert out["wins_a_lower"] == 1
    assert out["wins_b_lower"] == 0
    assert out["ties"] == 1


def test_paired_seed_edge_count_wins_phrased_as_sparser():
    doc = (paired_seed_difference.__doc__ or "")
    assert "sparser" in doc.lower()
    assert "not " in doc.lower() and "better" in doc.lower()


def test_compute_paired_seed_comparisons_only_predeclared(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_paired_seed_comparisons(
        flat, n_bootstrap=200, random_seed=42,
    )
    assert len(rows) == 20
    pairs_seen = {(r["label_a"], r["label_b"]) for r in rows}
    assert pairs_seen == set(PAIRED_COMPARISON_PAIRS)


def test_compute_paired_seed_no_best_confidence_comparison(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_paired_seed_comparisons(
        flat, n_bootstrap=200, random_seed=42,
    )
    for r in rows:
        for fld in ("label_a", "label_b"):
            label = str(r[fld]).lower()
            assert "best" not in label
            assert "argmax" not in label


# ---------------------------------------------------------------------------
# M. Correlations
# ---------------------------------------------------------------------------


def test_rank_values_average_ties_basic():
    assert rank_values_average_ties([1.0, 2.0, 2.0, 3.0]) == [
        1.0, 2.5, 2.5, 4.0,
    ]


def test_pearson_perfect_linear():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert pearson_corr(x, y) == pytest.approx(1.0)


def test_pearson_negative_linear():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    assert pearson_corr(x, y) == pytest.approx(-1.0)


def test_pearson_returns_none_on_constant():
    assert pearson_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_spearman_monotone_with_ties():
    x = [1.0, 2.0, 2.0, 3.0]
    y = [10.0, 20.0, 30.0, 40.0]
    s = spearman_corr(x, y)
    assert s is not None
    assert 0.9 <= s <= 1.0


def test_kendall_tau_b_simple_cases():
    assert kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_compute_metric_correlations_overall_and_per_method(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_metric_correlations(flat)
    groups = {r["group_label"] for r in rows}
    assert "all" in groups
    assert any(g.startswith("method_family:") for g in groups)
    assert len(rows) == 20


# ---------------------------------------------------------------------------
# N. Degradation
# ---------------------------------------------------------------------------


def test_linear_slope_simple():
    assert linear_slope([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]) == pytest.approx(2.0)


def test_linear_slope_constant_x_returns_none():
    assert linear_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_compute_degradation_summary_hard_exclusion_and_soft(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_degradation_summary(flat)
    families = {r["method_family"] for r in rows}
    assert "hard_exclusion" in families
    assert "soft_frobenius" in families
    soft_rows = [r for r in rows if r["method_family"] == "soft_frobenius"]
    assert len(soft_rows) == 20
    hard_rows = [r for r in rows if r["method_family"] == "hard_exclusion"]
    assert len(hard_rows) == 4
    for r in rows:
        text = " ".join(str(v) for v in r.values()).lower()
        for forbidden in (
            "graceful", "robust", "winner", "best", "supported",
        ):
            assert forbidden not in text


# ---------------------------------------------------------------------------
# O. Forbidden-edge engagement summary
# ---------------------------------------------------------------------------


def test_forbidden_engagement_summary_groups_correctly(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    rows = compute_forbidden_edge_engagement_summary(flat)
    assert len(rows) == 32


def test_reference_forbidden_edges_by_seed_exactly_one_per_seed(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    records = load_main_evaluation_records(
        main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    )
    ref = reference_forbidden_edges_by_seed(records)
    assert set(ref.keys()) == set(EVALUATION_SEED_VALUES)


def test_reference_forbidden_edges_rejects_missing(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    _persist(_make_record(planned, base_dir=tmp_path), planned, tmp_path)
    records = load_main_evaluation_records(
        main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    )
    with pytest.raises(ValueError, match="no clean-soft"):
        reference_forbidden_edges_by_seed(records)


def test_reference_comparison_uses_clean_soft_edge_set(tmp_path):
    flat, _ = _build_grid_and_flat(tmp_path)
    records = load_main_evaluation_records(
        main_evaluation_records_dir(tmp_path, _RUN_HASH12)
    )
    rows = compute_reference_forbidden_edge_comparison(
        records, flat, base_dir=tmp_path,
    )
    assert len(rows) == 28
    for seed in EVALUATION_SEED_VALUES:
        per_seed = [r for r in rows if r["seed_value"] == seed]
        ref_n = {r["n_reference_forbidden_edges"] for r in per_seed}
        assert len(ref_n) == 1


# ---------------------------------------------------------------------------
# P. Per-intervention MMD
# ---------------------------------------------------------------------------


def _write_interventions_mmd(
    planned, *, base_dir: Path, n_records: int = 2,
) -> None:
    rel = planned.artefact_paths["interventions_mmd.json"]
    full = base_dir / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for k in range(n_records):
        records.append({
            "intervention_id": f"do_X{k}_pos",
            "target_node": k,
            "value_raw": 1.0,
            "value_model_frame": 1.1,
            "ground_truth_sampling_seed": 10501 + k,
            "model_sampling_seed": 20501 + k,
            "n_ground_truth_samples": 1000,
            "n_model_samples": 1000,
            "mmd_value": 0.1 + 0.01 * k,
            "mmd_status": "available",
            "bandwidth_used": 1.5,
            "bandwidth_sweep": {
                "0.5x": 0.05 + 0.001 * k,
                "1.0x": 0.1 + 0.01 * k,
                "2.0x": 0.2 + 0.02 * k,
            },
            "sampler_status_for_intervention": "available",
            "sampler_reason": None,
        })
    payload = {
        "records": records,
        "mmd_primary": 0.1,
        "mmd_bandwidth_sweep": {"0.5x": 0.05, "1.0x": 0.1, "2.0x": 0.2},
    }
    full.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_per_intervention_mmd_parses_expected_schema(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    _write_interventions_mmd(planned, base_dir=tmp_path, n_records=3)
    rows = extract_per_intervention_mmd(rec, base_dir=tmp_path)
    assert len(rows) == 3
    for r in rows:
        assert r["intervention_id"]
        assert isinstance(r["target_node"], int)
        assert r["mmd_status"] == "available"
        assert r["bandwidth_sweep_0_5x"] is not None
        assert r["bandwidth_sweep_1_0x"] is not None
        assert r["bandwidth_sweep_2_0x"] is not None


def test_extract_per_intervention_mmd_missing_path_on_computed_raises(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    object.__setattr__(rec, "interventions_mmd_path", None)
    with pytest.raises(ValueError, match="interventions_mmd_path is None"):
        extract_per_intervention_mmd(rec, base_dir=tmp_path)


def test_extract_per_intervention_mmd_missing_bandwidth_key_raises(tmp_path):
    cfg = _make_config(method_family="prior_free", seed=501)
    planned = make_planned_run(cfg, _RUN_HASH12)
    rec = _make_record(planned, base_dir=tmp_path)
    rel = planned.artefact_paths["interventions_mmd.json"]
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [{
            "intervention_id": "x",
            "target_node": 0,
            "value_raw": 0.0,
            "value_model_frame": 0.0,
            "ground_truth_sampling_seed": 10501,
            "model_sampling_seed": 20501,
            "n_ground_truth_samples": 1000,
            "n_model_samples": 1000,
            "mmd_value": 0.1,
            "mmd_status": "available",
            "bandwidth_used": 1.0,
            "bandwidth_sweep": {"0.5x": 0.05, "1.0x": 0.1},
            "sampler_status_for_intervention": "available",
            "sampler_reason": None,
        }],
        "mmd_primary": 0.1,
    }
    full.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="2.0x"):
        extract_per_intervention_mmd(rec, base_dir=tmp_path)


def test_per_intervention_mmd_summary_finite_only():
    rows = [
        {
            "method_family": "prior_free",
            "seed_value": 501,
            "corruption_fraction": None,
            "confidence": None,
            "intervention_id": "do_X0_pos",
            "target_node": 0,
            "value_raw": 1.0,
            "mmd_value": 0.1,
            "mmd_status": "available",
        },
        {
            "method_family": "prior_free",
            "seed_value": 502,
            "corruption_fraction": None,
            "confidence": None,
            "intervention_id": "do_X0_pos",
            "target_node": 0,
            "value_raw": 1.0,
            "mmd_value": float("nan"),
            "mmd_status": "available",
        },
        {
            "method_family": "prior_free",
            "seed_value": 503,
            "corruption_fraction": None,
            "confidence": None,
            "intervention_id": "do_X0_pos",
            "target_node": 0,
            "value_raw": 1.0,
            "mmd_value": 0.3,
            "mmd_status": "available",
        },
    ]
    out = compute_per_intervention_mmd_summary(rows)
    assert len(out) == 1
    assert out[0]["n"] == 2
    assert out[0]["mean_mmd"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Q. Output and orchestrator
# ---------------------------------------------------------------------------


def test_write_dict_rows_csv_deterministic_field_order(tmp_path):
    rows = [{"b": 2, "a": 1, "c": None}, {"a": 10, "b": 20, "c": 30}]
    out = tmp_path / "dr.csv"
    write_dict_rows_csv(rows, out, ("a", "b", "c"))
    text = out.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0] == "a,b,c"
    assert lines[1] == "1,2,"
    assert lines[2] == "10,20,30"


def test_write_dict_rows_csv_rejects_unknown_keys(tmp_path):
    out = tmp_path / "x.csv"
    with pytest.raises(ValueError, match="unexpected keys"):
        write_dict_rows_csv(
            [{"a": 1, "extra": 2}], out, ("a",)
        )


def test_statistics_summary_json_required_keys(tmp_path):
    _build_grid_and_flat(tmp_path)
    summary = generate_hypothesis_statistics(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        n_bootstrap=100,
        random_seed=90210,
    )
    required = {
        "main_evaluation_run_hash12",
        "input_flat_csv",
        "output_files",
        "n_flat_rows",
        "n_baseline_rows",
        "n_paired_comparison_rows",
        "n_correlation_rows",
        "n_degradation_rows",
        "n_forbidden_engagement_rows",
        "n_reference_forbidden_rows",
        "n_per_intervention_mmd_rows",
        "n_per_intervention_mmd_summary_rows",
        "no_plots_created",
        "no_notebook_created",
        "no_hypothesis_verdicts",
    }
    assert required.issubset(set(summary.keys()))
    assert summary["no_plots_created"] is True
    assert summary["no_notebook_created"] is True
    assert summary["no_hypothesis_verdicts"] is True


def test_generate_hypothesis_statistics_writes_all_files(tmp_path):
    _build_grid_and_flat(tmp_path)
    generate_hypothesis_statistics(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        n_bootstrap=100,
        random_seed=90210,
    )
    rd = main_evaluation_readout_dir(tmp_path, _RUN_HASH12)
    assert (rd / BASELINE_COMPARISON_CSV).exists()
    assert (rd / PAIRED_SEED_COMPARISONS_CSV).exists()
    assert (rd / METRIC_CORRELATIONS_CSV).exists()
    assert (rd / DEGRADATION_SUMMARY_CSV).exists()
    assert (rd / FORBIDDEN_EDGE_ENGAGEMENT_SUMMARY_CSV).exists()
    assert (rd / REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV).exists()
    assert (rd / PER_INTERVENTION_MMD_LONG_CSV).exists()
    assert (rd / PER_INTERVENTION_MMD_SUMMARY_CSV).exists()
    assert (rd / STATISTICS_SUMMARY_JSON).exists()


def test_generate_hypothesis_statistics_no_plots_or_verdicts(tmp_path):
    _build_grid_and_flat(tmp_path)
    generate_hypothesis_statistics(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
        n_bootstrap=100,
        random_seed=90210,
    )
    for ext in ("*.png", "*.jpg", "*.pdf", "*.svg", "*.ipynb"):
        assert not list(tmp_path.rglob(ext))
    rd = main_evaluation_readout_dir(tmp_path, _RUN_HASH12)
    forbidden = (
        "h1 is supported", "h2 is supported", "h3 is supported",
        "refuted", "proven", "winner", "best method",
    )
    for path in rd.glob("*.csv"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, (
                f"forbidden token in {path.name}"
            )


def test_cli_full_pipeline_returns_zero(tmp_path):
    _build_full_grid(tmp_path)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
        "--n-bootstrap", "100",
        "--random-seed", "42",
    ])
    assert rc == 0


def test_cli_skip_hypothesis_statistics_returns_zero(tmp_path):
    _build_full_grid(tmp_path)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
        "--skip-hypothesis-statistics",
    ])
    assert rc == 0
    rd = main_evaluation_readout_dir(tmp_path, _RUN_HASH12)
    assert not (rd / BASELINE_COMPARISON_CSV).exists()


def test_cli_returns_one_on_failure(tmp_path):
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# R. Static scope checks (M-9b)
# ---------------------------------------------------------------------------


_M9B_FORBIDDEN_EXTRA_PREFIXES: tuple[str, ...] = (
    "scipy",
    "pandas",
    "statsmodels",
    "sklearn",
)


def test_readout_does_not_import_scipy_pandas_statsmodels_sklearn():
    src = Path(readout_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _M9B_FORBIDDEN_EXTRA_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"readout.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_readout_source_has_no_final_verdict_phrases():
    src = Path(readout_mod.__file__).read_text(encoding="utf-8")
    src_lower = src.lower()
    forbidden_phrases = (
        "h1 is supported",
        "h2 is supported",
        "h3 is supported",
        "refuted",
        "proven",
        "winner",
        "best method",
    )
    for phrase in forbidden_phrases:
        assert phrase not in src_lower, (
            f"readout.py must not contain final-verdict phrase "
            f"{phrase!r}."
        )


def test_readout_source_has_no_post_hoc_confidence_selection():
    src = Path(readout_mod.__file__).read_text(encoding="utf-8")
    src_lower = src.lower()
    for phrase in ("best confidence", "max confidence", "argmax"):
        assert phrase not in src_lower, (
            f"readout.py must not contain post-hoc selection phrase "
            f"{phrase!r}."
        )
