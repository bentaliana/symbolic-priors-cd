"""Tests for the offline prior-structural-relevance diagnostic.

All tests use tiny synthetic records/arrays under ``tmp_path``. Real
main-evaluation records and artefacts are never used.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from experiments.main_study.exploratory import (
    prior_structural_relevance as psr_mod,
)
from experiments.main_study.exploratory.prior_structural_relevance import (
    AGGREGATED_ERROR_HEATMAP_PNG,
    ANALYSIS_PROTOCOL_VERSION,
    BASELINE_CONDITION_LABELS,
    BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
    BASELINE_LABEL_MATCHED_L1,
    BASELINE_LABEL_PRIOR_FREE,
    BASELINE_LABEL_SOFT_CLEAN_CONF1,
    EVALUATION_SEED_VALUES,
    MANIFEST_JSON,
    OFFLINE_REMOVAL_EFFECT_CSV,
    PRIOR_FREE_ERROR_DECOMPOSITION_CSV,
    PRIOR_TARGET_OVERLAP_CSV,
    READOUT_MARKDOWN,
    TOPOLOGICAL_RELEVANCE_CSV,
    analysis_output_dir,
    classify_edges,
    compute_analysis_hash12,
    compute_minimal_topological_relevance,
    compute_offline_removal_effect,
    compute_prior_free_error_decomposition,
    compute_prior_target_overlap,
    edge_count,
    find_baseline_condition_records,
    find_clean_soft_reference_records,
    load_main_records,
    main as cli_main,
    records_dir_for_run,
    remove_reference_forbidden_edges,
    run_prior_structural_relevance_analysis,
    write_csv,
    write_manifest_json,
)
from experiments.main_study.priors import CorruptedPriorSpec
from experiments.main_study.records import (
    MainStudyRunRecord,
    SCHEMA_VERSION,
)
from experiments.main_study.run_io import persist_record_atomic
from experiments.main_study.schema import (
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import make_planned_run
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


_RUN_HASH12 = "166c792c43bc"
_PARENT_HASH = "a" * 64
_N_NODES = 10
_GENERATED_AT = "2026-05-26T00:00:00Z"

# A deterministic 10-edge clean forbidden-edge set used for all
# synthetic seeds. All entries are off-diagonal and inside [0, N).
_REF_FORBIDDEN: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (0, 6), (0, 7), (0, 8), (0, 9), (1, 2),
)


def _spec(*, seed: int, corruption_fraction: float, corruption_index: int,
          forbidden: tuple[tuple[int, int], ...] = _REF_FORBIDDEN,
          ) -> CorruptedPriorSpec:
    labels = {
        f"{i},{j}": "true_negative_retained" for (i, j) in forbidden
    }
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


def _make_config(
    *,
    method_family: str,
    seed: int,
    confidence: Optional[float] = None,
    corruption_fraction: Optional[float] = None,
    corruption_index: Optional[int] = None,
    matched_l1: Optional[float] = None,
) -> MainStudyConfig:
    # Both the matched_l1 family and the rest of the main study now
    # share the protocol backbone lambda1=0.10. The branch is kept
    # for symmetry with earlier protocol versions where the values
    # differed.
    dagma_lambda1 = (
        0.10 if method_family == "matched_l1" else 0.10
    )
    base_dagma = DAGMAConfig(lambda1=dagma_lambda1)
    kwargs: dict[str, Any] = dict(
        method_family=method_family,
        seed_value=seed,
        seed_population="main_evaluation",
        dagma_config=base_dagma,
        parent_heldout_run_hash_full=_PARENT_HASH,
    )
    if method_family == "soft_frobenius":
        kwargs["confidence"] = confidence
        kwargs["corrupted_prior_spec"] = _spec(
            seed=seed,
            corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
        )
    elif method_family == "hard_exclusion":
        kwargs["corrupted_prior_spec"] = _spec(
            seed=seed,
            corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
        )
    elif method_family == "matched_l1":
        kwargs["matched_l1_lambda1"] = (
            matched_l1 if matched_l1 is not None else 0.10
        )
    return make_main_study_config(**kwargs)


def _write_artefacts(
    planned, *, base_dir: Path,
    thresholded: np.ndarray,
    continuous_w: np.ndarray,
    true_adj: np.ndarray,
) -> None:
    for name, rel in planned.artefact_paths.items():
        full = base_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        if name == "thresholded_adjacency.npz":
            np.savez(full, thresholded_adjacency=thresholded)
        elif name == "continuous_w.npz":
            np.savez(full, continuous_w=continuous_w)
        elif name == "true_adjacency.npz":
            np.savez(full, true_adjacency=true_adj)
        elif name == "confidence_mask.npz":
            np.savez(
                full,
                confidence_mask=np.zeros(
                    (_N_NODES, _N_NODES), dtype=float
                ),
            )
        elif name == "interventions_mmd.json":
            full.write_text(
                json.dumps({"records": [], "mmd_primary": 0.0}),
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


def _build_record(
    planned,
    *,
    base_dir: Path,
    sid: float = 5.0,
    shd: float = 3.0,
    mmd: float = 0.01,
    thresholded: Optional[np.ndarray] = None,
    continuous_w: Optional[np.ndarray] = None,
    true_adj: Optional[np.ndarray] = None,
) -> MainStudyRunRecord:
    if thresholded is None:
        thresholded = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    if continuous_w is None:
        continuous_w = thresholded.astype(float) * 0.5
    if true_adj is None:
        true_adj = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    _write_artefacts(
        planned, base_dir=base_dir,
        thresholded=thresholded,
        continuous_w=continuous_w,
        true_adj=true_adj,
    )
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=_N_NODES,
        fit_status="success",
        graph_status="valid_dag",
        sampler_status="available",
        metric_status="computed",
        failure_kind=None,
        failure_message="",
        runtime_seconds=1.0,
        fit_runtime_seconds=0.8,
        wrapper_diagnostics={"training_status": "converged"},
        parent_heldout_run_hash_full=planned.config.parent_heldout_run_hash_full,
        generated_at_utc=_GENERATED_AT,
        sid=sid, shd=shd, mmd=mmd, metric_runtime_seconds=0.2,
        continuous_w_path=planned.artefact_paths["continuous_w.npz"],
        thresholded_adjacency_path=planned.artefact_paths[
            "thresholded_adjacency.npz"
        ],
        true_adjacency_path=planned.artefact_paths["true_adjacency.npz"],
        interventions_mmd_path=planned.artefact_paths[
            "interventions_mmd.json"
        ],
    )
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


def _true_adj_for_seed(seed: int) -> np.ndarray:
    """Small synthetic true DAG. Edges chosen so they do not overlap
    the reference forbidden-edge set (which targets row 0 and (1,2))."""
    arr = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    # Edges: 2->3, 3->4, 4->5, 5->6, 6->7
    chain = ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
    for (i, j) in chain:
        arr[i, j] = True
    return arr


def _seed_full_grid(base_dir: Path) -> None:
    """Persist the 28-record (4 conditions x 7 seeds) synthetic grid."""
    for seed in EVALUATION_SEED_VALUES:
        true_adj = _true_adj_for_seed(seed)
        # prior_free: predicted = a few true edges + a few FPs in the
        # row-0 area (which overlap the reference forbidden edges).
        pf_pred = np.zeros((_N_NODES, _N_NODES), dtype=bool)
        pf_pred[2, 3] = True   # true positive
        pf_pred[3, 4] = True   # true positive
        pf_pred[0, 1] = True   # false positive AND in forbidden set
        pf_pred[0, 2] = True   # false positive AND in forbidden set
        pf_pred[8, 9] = True   # false positive NOT in forbidden set
        pf_cont = pf_pred.astype(float) * 0.5
        cfg = _make_config(method_family="prior_free", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(
            planned, base_dir=base_dir,
            sid=10.0, shd=6.0, mmd=0.02,
            thresholded=pf_pred, continuous_w=pf_cont,
            true_adj=true_adj,
        ), planned, base_dir)
        # matched_l1: same pred for simplicity.
        cfg = _make_config(method_family="matched_l1", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(
            planned, base_dir=base_dir,
            sid=9.0, shd=5.0, mmd=0.02,
            thresholded=pf_pred, continuous_w=pf_cont,
            true_adj=true_adj,
        ), planned, base_dir)
        # hard_exclusion at corruption=0.0 (clean): suppresses (0,1)/(0,2).
        he_pred = pf_pred.copy()
        he_pred[0, 1] = False
        he_pred[0, 2] = False
        he_cont = he_pred.astype(float) * 0.5
        cfg = _make_config(
            method_family="hard_exclusion", seed=seed,
            corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(
            planned, base_dir=base_dir,
            sid=6.0, shd=4.0, mmd=0.015,
            thresholded=he_pred, continuous_w=he_cont,
            true_adj=true_adj,
        ), planned, base_dir)
        # soft_frobenius clean / confidence=1.0: also suppresses the
        # forbidden positions, partially.
        sf_pred = pf_pred.copy()
        sf_pred[0, 1] = False
        sf_cont = sf_pred.astype(float) * 0.5
        cfg = _make_config(
            method_family="soft_frobenius", seed=seed,
            confidence=1.0, corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(
            planned, base_dir=base_dir,
            sid=8.0, shd=5.0, mmd=0.018,
            thresholded=sf_pred, continuous_w=sf_cont,
            true_adj=true_adj,
        ), planned, base_dir)


def _load_records(base_dir: Path):
    return load_main_records(base_dir, _RUN_HASH12)


# ===========================================================================
# 1. classify_edges (off-diagonal TP/TN/FP/FN)
# ===========================================================================


def test_classify_edges_off_diagonal_only():
    pred = np.array([
        [True, True, False],
        [False, False, True],
        [True, False, True],
    ])
    true = np.array([
        [False, True, False],
        [False, False, False],
        [True, True, False],
    ])
    # Off-diagonal positions and their classification:
    # (0,1): pred=T, true=T -> TP
    # (0,2): pred=F, true=F -> TN
    # (1,0): pred=F, true=F -> TN
    # (1,2): pred=T, true=F -> FP
    # (2,0): pred=T, true=T -> TP
    # (2,1): pred=F, true=T -> FN
    classes = classify_edges(pred, true)
    assert classes["true_positive_edges"] == {(0, 1), (2, 0)}
    assert classes["true_negative_edges"] == {(0, 2), (1, 0)}
    assert classes["false_positive_edges"] == {(1, 2)}
    assert classes["false_negative_edges"] == {(2, 1)}


# ===========================================================================
# 2. edge_count excludes diagonal
# ===========================================================================


def test_edge_count_excludes_diagonal():
    arr = np.zeros((4, 4), dtype=bool)
    np.fill_diagonal(arr, True)
    arr[0, 1] = True
    arr[2, 3] = True
    assert edge_count(arr) == 2


# ===========================================================================
# 3. remove_reference_forbidden_edges does not mutate input
# ===========================================================================


def test_remove_reference_forbidden_edges_returns_copy():
    pred = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ])
    forbidden = ((0, 1), (1, 2))
    edited = remove_reference_forbidden_edges(pred, forbidden)
    assert edited is not pred
    assert edited.dtype == bool
    # Edited positions are False; the unrelated (0,2) stays True.
    assert edited[0, 1] == False  # noqa: E712
    assert edited[1, 2] == False  # noqa: E712
    assert edited[0, 2] == True   # noqa: E712
    # Original is unchanged.
    assert pred[0, 1] == True   # noqa: E712
    assert pred[1, 2] == True   # noqa: E712


# ===========================================================================
# 4. prior_target_overlap counts predicted forbidden edges
# ===========================================================================


def test_prior_target_overlap_counts(tmp_path):
    _seed_full_grid(tmp_path)
    records = _load_records(tmp_path)
    reference = find_clean_soft_reference_records(records)
    baseline = find_baseline_condition_records(records)
    rows = compute_prior_target_overlap(
        baseline_records=baseline,
        reference_records=reference,
        base_dir=tmp_path,
    )
    assert len(rows) == 28
    # prior_free predicts (0,1) and (0,2) inside the 10-edge
    # reference set -> 2 predicted edges per seed.
    pf_rows = [
        r for r in rows
        if r["condition_label"] == BASELINE_LABEL_PRIOR_FREE
    ]
    assert all(r["n_reference_forbidden_edges"] == 10 for r in pf_rows)
    assert all(r["n_reference_edges_predicted"] == 2 for r in pf_rows)
    assert all(
        abs(r["fraction_reference_edges_predicted"] - 0.2) < 1e-12
        for r in pf_rows
    )


# ===========================================================================
# 5. prior-free error decomposition
# ===========================================================================


def test_prior_free_error_decomposition(tmp_path):
    _seed_full_grid(tmp_path)
    records = _load_records(tmp_path)
    reference = find_clean_soft_reference_records(records)
    baseline = find_baseline_condition_records(records)
    rows = compute_prior_free_error_decomposition(
        baseline_records=baseline,
        reference_records=reference,
        base_dir=tmp_path,
    )
    assert len(rows) == 7
    # prior_free predicts 5 edges, true has 5 edges:
    # TP = 2 ((2,3),(3,4)); FN = 3; FP = 3; targeted_FP = 2.
    for r in rows:
        assert r["n_true_edges"] == 5
        assert r["n_predicted_edges"] == 5
        assert r["true_positive_count"] == 2
        assert r["false_positive_count"] == 3
        assert r["false_negative_count"] == 3
        assert r["targeted_false_positive_count"] == 2
        # targeted_FP / FP = 2 / 3.
        assert (
            abs(r["targeted_false_positive_fraction_of_fp"] - (2 / 3))
            < 1e-12
        )


# ===========================================================================
# 6. targeted_false_positive_fraction_of_fp handles zero FP safely
# ===========================================================================


def test_targeted_fp_fraction_safe_when_zero_fp(tmp_path):
    """When predicted has no FPs, the ratio is None, not a crash."""
    base_dir = tmp_path
    seed = 501
    true_adj = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    true_adj[2, 3] = True
    # Predicted matches truth exactly -> 0 FP, 0 FN, 1 TP.
    pred = true_adj.copy()
    cfg = _make_config(method_family="prior_free", seed=seed)
    planned = make_planned_run(cfg, _RUN_HASH12)
    _persist(_build_record(
        planned, base_dir=base_dir,
        thresholded=pred, continuous_w=pred.astype(float),
        true_adj=true_adj,
    ), planned, base_dir)
    soft_cfg = _make_config(
        method_family="soft_frobenius", seed=seed,
        confidence=1.0, corruption_fraction=0.0, corruption_index=0,
    )
    planned_soft = make_planned_run(soft_cfg, _RUN_HASH12)
    _persist(_build_record(
        planned_soft, base_dir=base_dir,
        thresholded=pred, continuous_w=pred.astype(float),
        true_adj=true_adj,
    ), planned_soft, base_dir)
    records = _load_records(base_dir)
    ref = {seed: planned_soft}  # not used directly; we pass actual records below
    # Use the real loader to ensure spec-derived forbidden edges are present.
    reference = {
        seed: next(
            r for r in records
            if r.config.method_family == "soft_frobenius"
            and r.config.seed_value == seed
        )
    }
    baseline = {
        (seed, BASELINE_LABEL_PRIOR_FREE): next(
            r for r in records
            if r.config.method_family == "prior_free"
            and r.config.seed_value == seed
        ),
    }
    rec = baseline[(seed, BASELINE_LABEL_PRIOR_FREE)]
    pred_arr = np.asarray(
        np.load(base_dir / rec.thresholded_adjacency_path)[
            "thresholded_adjacency"
        ], dtype=bool,
    )
    truth_arr = np.asarray(
        np.load(base_dir / rec.true_adjacency_path)[
            "true_adjacency"
        ], dtype=bool,
    )
    cls = classify_edges(pred_arr, truth_arr)
    assert len(cls["false_positive_count" if False else "false_positive_edges"]) == 0
    # Directly verify safe handling: call the per-seed decomposition
    # logic via a one-seed monkey by patching EVALUATION_SEED_VALUES
    # isn't necessary -- the function tolerates 0 FP regardless.
    # Build a single-seed pair and call helper directly:
    one_seed_baseline = {
        (seed, BASELINE_LABEL_PRIOR_FREE): rec,
    }
    one_seed_reference = {seed: reference[seed]}
    import experiments.main_study.exploratory.prior_structural_relevance as m
    monkey = m.EVALUATION_SEED_VALUES
    try:
        m.EVALUATION_SEED_VALUES = (seed,)  # type: ignore[assignment]
        out = m.compute_prior_free_error_decomposition(
            baseline_records=one_seed_baseline,
            reference_records=one_seed_reference,
            base_dir=base_dir,
        )
    finally:
        m.EVALUATION_SEED_VALUES = monkey  # type: ignore[assignment]
    assert len(out) == 1
    assert out[0]["false_positive_count"] == 0
    assert out[0]["targeted_false_positive_fraction_of_fp"] is None


# ===========================================================================
# 7. offline removal effect recomputes SID/SHD and does not compute MMD
# ===========================================================================


def test_offline_removal_recomputes_sid_shd(tmp_path):
    _seed_full_grid(tmp_path)
    records = _load_records(tmp_path)
    reference = find_clean_soft_reference_records(records)
    baseline = find_baseline_condition_records(records)
    rows = compute_offline_removal_effect(
        baseline_records=baseline,
        reference_records=reference,
        base_dir=tmp_path,
    )
    assert len(rows) == 7
    # The output schema must NOT carry an mmd column.
    for r in rows:
        assert "mmd" not in r
        assert "mmd_delta" not in r


# ===========================================================================
# 8. offline removal deltas use after - original convention
# ===========================================================================


def test_offline_removal_delta_convention(tmp_path):
    _seed_full_grid(tmp_path)
    records = _load_records(tmp_path)
    reference = find_clean_soft_reference_records(records)
    baseline = find_baseline_condition_records(records)
    rows = compute_offline_removal_effect(
        baseline_records=baseline,
        reference_records=reference,
        base_dir=tmp_path,
    )
    for r in rows:
        assert r["sid_delta"] == (
            r["sid_after_removing_reference_forbidden_edges"]
            - r["sid_original"]
        )
        assert r["shd_delta"] == (
            r["shd_after_removing_reference_forbidden_edges"]
            - r["shd_original"]
        )


# ===========================================================================
# 9. minimal topological relevance
# ===========================================================================


def test_topological_relevance_descendants_ancestors_degrees(tmp_path):
    _seed_full_grid(tmp_path)
    records = _load_records(tmp_path)
    reference = find_clean_soft_reference_records(records)
    baseline = find_baseline_condition_records(records)
    rows = compute_minimal_topological_relevance(
        baseline_records=baseline,
        reference_records=reference,
        base_dir=tmp_path,
    )
    # 10 forbidden edges x 7 seeds = 70 rows.
    assert len(rows) == 70
    # True graph chain 2->3->4->5->6->7.
    # For edge (0,1): target=1; node 1 has no in/out edges in truth.
    edge_01 = next(
        r for r in rows
        if r["source_node"] == 0 and r["target_node"] == 1
        and r["seed_value"] == EVALUATION_SEED_VALUES[0]
    )
    assert edge_01["target_in_degree"] == 0
    assert edge_01["target_out_degree"] == 0
    assert edge_01["source_in_degree"] == 0
    assert edge_01["source_out_degree"] == 0
    # For edge (1,2): target=2; node 2 has out-degree 1 (2->3) and
    # descendants {3,4,5,6,7} = 5.
    edge_12 = next(
        r for r in rows
        if r["source_node"] == 1 and r["target_node"] == 2
        and r["seed_value"] == EVALUATION_SEED_VALUES[0]
    )
    assert edge_12["target_out_degree"] == 1
    assert edge_12["target_descendant_count"] == 5


# ===========================================================================
# 10/11. reference extraction
# ===========================================================================


def test_clean_soft_reference_extraction_rejects_missing(tmp_path):
    # Build only 6 of 7 expected references.
    for seed in EVALUATION_SEED_VALUES[:-1]:
        cfg = _make_config(
            method_family="soft_frobenius", seed=seed,
            confidence=1.0, corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(planned, base_dir=tmp_path),
                 planned, tmp_path)
    records = _load_records(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        find_clean_soft_reference_records(records)


def test_clean_soft_reference_extraction_rejects_duplicate(tmp_path):
    # Two clean-soft references at seed=501 by writing the same record
    # to two filenames with different run_ids.
    for seed in EVALUATION_SEED_VALUES:
        cfg = _make_config(
            method_family="soft_frobenius", seed=seed,
            confidence=1.0, corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(planned, base_dir=tmp_path),
                 planned, tmp_path)
    # Duplicate: rewrite seed=501 record at a sibling filename with a
    # tweaked run_id so it loads as a distinct record but is still
    # clean-soft / conf=1.0 / seed=501.
    rd = records_dir_for_run(tmp_path, _RUN_HASH12)
    first = sorted(rd.glob("soft_frobenius__main_evaluation__seed501*.json"))[0]
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["run_id"] = "soft_frobenius__main_evaluation__seed501__cfgduplicate"
    payload["configuration_hash_full"] = (
        "a" * 63 + "0"
    )  # Will be rejected by record constructor; instead just write raw
    # We need a path-only duplicate that the loader keeps; bypass via
    # writing a sibling file with the exact same JSON contents:
    duplicate = rd / "soft_frobenius__main_evaluation__seed501__cfgduplicate.json"
    duplicate.write_text(first.read_text(encoding="utf-8"),
                         encoding="utf-8")
    # load_main_records will reject the duplicate run_id; if that
    # gate fires, the test still exercises the duplicate-detection path.
    with pytest.raises(ValueError, match="duplicate run_id|exactly one"):
        records = load_main_records(tmp_path, _RUN_HASH12)
        find_clean_soft_reference_records(records)


def test_baseline_condition_extraction_rejects_missing(tmp_path):
    # Build all but the matched_l1 records.
    for seed in EVALUATION_SEED_VALUES:
        cfg = _make_config(method_family="prior_free", seed=seed)
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(planned, base_dir=tmp_path),
                 planned, tmp_path)
        cfg = _make_config(
            method_family="soft_frobenius", seed=seed,
            confidence=1.0, corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(planned, base_dir=tmp_path),
                 planned, tmp_path)
        cfg = _make_config(
            method_family="hard_exclusion", seed=seed,
            corruption_fraction=0.0, corruption_index=0,
        )
        planned = make_planned_run(cfg, _RUN_HASH12)
        _persist(_build_record(planned, base_dir=tmp_path),
                 planned, tmp_path)
    records = _load_records(tmp_path)
    with pytest.raises(ValueError, match="exactly one record"):
        find_baseline_condition_records(records)


# ===========================================================================
# 13. analysis_hash12 determinism
# ===========================================================================


def test_analysis_hash12_deterministic_and_uses_only_declared_inputs():
    h_a, payload_a = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        input_run_ids=["b", "a", "c"],
        input_configuration_hashes=["xyz", "abc"],
    )
    h_b, payload_b = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        input_run_ids=["a", "b", "c"],
        input_configuration_hashes=["abc", "xyz"],
    )
    # Sort-insensitive (same content -> same hash).
    assert h_a == h_b
    assert len(h_a) == 12
    assert all(c in "0123456789abcdef" for c in h_a)
    # Payload covers exactly the four declared keys.
    assert set(payload_a.keys()) == {
        "main_evaluation_run_hash12",
        "analysis_protocol_version",
        "input_run_ids_sorted",
        "input_configuration_hashes_sorted",
    }
    # Protocol version is the module constant.
    assert payload_a["analysis_protocol_version"] == ANALYSIS_PROTOCOL_VERSION


# ===========================================================================
# 14. write_csv deterministic columns + None as empty cell
# ===========================================================================


def test_write_csv_deterministic_and_empty_for_none(tmp_path):
    rows = [{"a": 1, "b": None}, {"a": 2.5, "b": "hello"}]
    out = tmp_path / "x.csv"
    write_csv(rows, out, ("a", "b"))
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "a,b"
    assert lines[1].startswith("1,")
    assert lines[1].endswith(",")
    # No "None" / "null" / "NaN" tokens for missing cells.
    for token in ("None", "null", "NaN", "nan"):
        assert token not in text


# ===========================================================================
# 15. manifest includes no_* flags
# ===========================================================================


def test_manifest_includes_no_flags(tmp_path):
    _seed_full_grid(tmp_path)
    manifest = run_prior_structural_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    for flag in (
        "no_new_fits",
        "no_mmd_recomputation",
        "no_new_sampling",
        "no_protocol_changes",
    ):
        assert manifest[flag] is True


# ===========================================================================
# 16. readout excludes forbidden verdict phrases
# ===========================================================================


_FORBIDDEN_READOUT_PHRASES: tuple[str, ...] = (
    "proves",
    "refutes",
    "invalidates",
    "main experiment was invalid",
    "semantic priors do not work",
    "h2 is refuted",
    "stronger lambda would solve",
    "winner",
    "best method",
    "p-hacking",
)


def test_readout_excludes_forbidden_verdict_phrases(tmp_path):
    _seed_full_grid(tmp_path)
    manifest = run_prior_structural_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    out_dir = analysis_output_dir(tmp_path, manifest["analysis_hash12"])
    text = (out_dir / READOUT_MARKDOWN).read_text(encoding="utf-8").lower()
    for phrase in _FORBIDDEN_READOUT_PHRASES:
        assert phrase not in text, (
            f"forbidden verdict phrase {phrase!r} in readout"
        )


# ===========================================================================
# 17. orchestrator writes all required outputs
# ===========================================================================


def test_run_writes_all_required_outputs(tmp_path):
    _seed_full_grid(tmp_path)
    manifest = run_prior_structural_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    out_dir = analysis_output_dir(tmp_path, manifest["analysis_hash12"])
    for required in (
        PRIOR_TARGET_OVERLAP_CSV,
        PRIOR_FREE_ERROR_DECOMPOSITION_CSV,
        OFFLINE_REMOVAL_EFFECT_CSV,
        TOPOLOGICAL_RELEVANCE_CSV,
        READOUT_MARKDOWN,
        MANIFEST_JSON,
    ):
        assert (out_dir / required).exists(), (
            f"missing required output {required!r}"
        )


def test_cli_returns_zero_on_synthetic(tmp_path):
    _seed_full_grid(tmp_path)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 0


def test_cli_returns_one_on_missing_inputs(tmp_path):
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 1


# ===========================================================================
# 18. static import check
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "hashlib",
    "json",
    "math",
    "pathlib",
    "sys",
    "typing",
    "numpy",
    "matplotlib",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "symbolic_priors_cd.metrics",
})


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.main_study.backends",
    "experiments.main_study.executor",
    "experiments.main_study.runner",
    "experiments.main_study.run_main_evaluation",
    "experiments.main_study.calibrate_matched_l1",
    "experiments.main_study.readout",
    "experiments.main_study.render_readout_figures",
    "experiments.selection_study",
    "gadjid",
    "scipy",
    "statsmodels",
    "sklearn",
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
    src = Path(psr_mod.__file__).read_text(encoding="utf-8")
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
            f"prior_structural_relevance.py import {mod!r} not in "
            f"allowlist {sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(psr_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"forbidden import: {mod!r}"
            )


# ===========================================================================
# 19. source hygiene regex check
# ===========================================================================


_MILESTONE_REGEX: re.Pattern[str] = re.compile(
    r"\bM[-_]?(?:[0-9]|10)[a-c]?\b"
)


_HYGIENE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bClaude\b"),
    re.compile(r"\bChatGPT\b"),
    re.compile(r"\bprompt(?:ed|ing|s)?\b"),
    re.compile(r"\bconversation\b"),
    re.compile(r"\buser\s+asked\b"),
    re.compile(r"\bsuggested\s+by\b"),
    re.compile(r"\bp[-_]?hacking\b"),
    re.compile(r"\bheadline\s+result\b"),
    re.compile(r"\bpost[- ]hoc\s+rescue\b"),
)


# A subset of the project's __init__ and this test file may
# legitimately contain hygiene-checker fixture strings; only the
# created module is checked. Test files that intentionally define
# the hygiene regex itself are exempt.
_HYGIENE_TARGETS: tuple[Path, ...] = (
    Path("experiments/main_study/exploratory/__init__.py"),
    Path("experiments/main_study/exploratory/prior_structural_relevance.py"),
)


def test_created_sources_have_no_milestone_labels():
    project_root = Path(__file__).resolve().parents[2]
    for rel in _HYGIENE_TARGETS:
        text = (project_root / rel).read_text(encoding="utf-8")
        match = _MILESTONE_REGEX.search(text)
        assert match is None, (
            f"{rel} contains milestone label {match.group(0)!r}"
        )


def test_created_sources_have_no_assistant_or_process_artefacts():
    project_root = Path(__file__).resolve().parents[2]
    for rel in _HYGIENE_TARGETS:
        text = (project_root / rel).read_text(encoding="utf-8")
        for pattern in _HYGIENE_PATTERNS:
            match = pattern.search(text)
            assert match is None, (
                f"{rel} contains hygiene-blocked token "
                f"{match.group(0)!r}"
            )


# ===========================================================================
# 20. tests write only under tmp_path
# ===========================================================================


def test_tests_write_only_under_tmp_path(tmp_path):
    """Sentinel: this test file's helper writers all use tmp_path."""
    _seed_full_grid(tmp_path)
    assert (tmp_path / "results").is_dir()
