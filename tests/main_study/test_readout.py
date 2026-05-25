"""Tests for the M-9a main-evaluation readout foundation.

All tests run under ``tmp_path``. The real DAGMA wrappers, the real
metric backend, and the real M-8 record set are never used. Each
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
    CELL_SUMMARY_CSV,
    CONTINUOUS_W_KEY,
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
    FlatRecordRow,
    PROJECT_THRESHOLD,
    STATUS_SUMMARY_CSV,
    THRESHOLDED_ADJACENCY_KEY,
    VALIDATION_SUMMARY_JSON,
    ValidationSummary,
    compute_prior_edge_engagement,
    flatten_record,
    forbidden_edges_from_record,
    generate_readout_foundation,
    load_main_evaluation_records,
    load_npz_array,
    main as cli_main,
    main_evaluation_readout_dir,
    main_evaluation_records_dir,
    off_diagonal_edge_count,
    record_config_value,
    resolve_artifact_path,
    validate_flat_rows,
    write_cell_summary_csv,
    write_flat_records_csv,
    write_forbidden_edge_engagement_csv,
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
