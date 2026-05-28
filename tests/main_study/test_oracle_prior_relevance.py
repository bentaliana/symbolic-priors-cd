"""Tests for the offline alternative-prior relevance diagnostic.

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
    oracle_prior_relevance as opr_mod,
)
from experiments.main_study.exploratory.oracle_prior_relevance import (
    ANALYSIS_PROTOCOL_VERSION,
    EVALUATION_SEED_VALUES,
    ORACLE_BUDGET_K,
    ORACLE_MANIFEST_JSON,
    ORACLE_PER_SEED_CSV,
    ORACLE_READOUT_MD,
    ORACLE_SUMMARY_CSV,
    ORACLE_SUMMARY_PLOT_PNG,
    SCENARIO_ACTUAL_REFERENCE,
    SCENARIO_FN_BUDGET_GREEDY,
    SCENARIO_FN_FULL_GREEDY,
    SCENARIO_FP_BUDGET_EXACT,
    SCENARIO_FP_REMOVE_ALL,
    SCENARIO_LABELS,
    actual_reference_forbidden_removal,
    add_edges_with_acyclicity_guard,
    analysis_output_dir,
    classify_edges,
    compute_all_oracle_diagnostics,
    compute_analysis_hash12,
    compute_oracle_diagnostics_for_seed,
    compute_sid_shd,
    evaluate_single_edge_addition,
    exact_budget_false_positive_removal,
    full_false_positive_removal,
    greedy_acyclic_false_negative_addition,
    is_dag,
    load_clean_soft_reference_records,
    load_prior_free_records,
    main as cli_main,
    remove_edges,
    run_oracle_prior_relevance_analysis,
    summarise_oracle_diagnostics,
    write_csv,
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
# Fixtures
# ---------------------------------------------------------------------------


_RUN_HASH12 = "166c792c43bc"
_PARENT_HASH = "a" * 64
_N_NODES = 10
_GENERATED_AT = "2026-05-27T00:00:00Z"

_REF_FORBIDDEN: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (0, 6), (0, 7), (0, 8), (0, 9), (1, 2),
)


def _spec(*, seed: int, corruption_fraction: float,
          corruption_index: int) -> CorruptedPriorSpec:
    return CorruptedPriorSpec(
        n_nodes=_N_NODES, scm_seed=seed,
        corruption_fraction=float(corruption_fraction),
        corruption_index=int(corruption_index),
        corruption_seed=9100 + seed + corruption_index,
        forbidden_edges=_REF_FORBIDDEN,
        n_correct=len(_REF_FORBIDDEN), n_corrupted=0,
        removed_clean_edges=(), added_true_positive_edges=(),
        edge_labels={
            f"{i},{j}": "true_negative_retained"
            for (i, j) in _REF_FORBIDDEN
        },
    )


def _make_config(*, method_family: str, seed: int,
                 confidence: Optional[float] = None,
                 corruption_fraction: Optional[float] = None,
                 corruption_index: Optional[int] = None) -> MainStudyConfig:
    base_dagma = DAGMAConfig(lambda1=0.05)
    kwargs: dict[str, Any] = dict(
        method_family=method_family, seed_value=seed,
        seed_population="main_evaluation",
        dagma_config=base_dagma,
        parent_heldout_run_hash_full=_PARENT_HASH,
    )
    if method_family == "soft_frobenius":
        kwargs["confidence"] = confidence
        kwargs["corrupted_prior_spec"] = _spec(
            seed=seed, corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
        )
    elif method_family == "hard_exclusion":
        kwargs["corrupted_prior_spec"] = _spec(
            seed=seed, corruption_fraction=corruption_fraction,
            corruption_index=corruption_index,
        )
    elif method_family == "matched_l1":
        kwargs["matched_l1_lambda1"] = 0.10
    return make_main_study_config(**kwargs)


def _write_artefacts(planned, *, base_dir: Path,
                     thresholded: np.ndarray,
                     continuous_w: np.ndarray,
                     true_adj: np.ndarray) -> None:
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
                }), encoding="utf-8",
            )
        elif name == "per_edge_labels.json":
            full.write_text(json.dumps({}), encoding="utf-8")


def _build_record(planned, *, base_dir: Path,
                  sid: float = 5.0, shd: float = 3.0, mmd: float = 0.01,
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
        thresholded=thresholded, continuous_w=continuous_w,
        true_adj=true_adj,
    )
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION, config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id, n_nodes=_N_NODES,
        fit_status="success", graph_status="valid_dag",
        sampler_status="available", metric_status="computed",
        failure_kind=None, failure_message="",
        runtime_seconds=1.0, fit_runtime_seconds=0.8,
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


def _true_adj_chain() -> np.ndarray:
    """A small 10-node chain 2->3->4->5->6->7 with no other edges."""
    arr = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    for (i, j) in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
        arr[i, j] = True
    return arr


def _build_synthetic_28_record_grid(base_dir: Path) -> None:
    """Persist a 28-record grid (4 conditions x 7 seeds)."""
    for seed in EVALUATION_SEED_VALUES:
        true_adj = _true_adj_chain()
        # prior_free: predicts 2 TPs, 2 targeted FPs, 1 non-targeted FP.
        pred = np.zeros((_N_NODES, _N_NODES), dtype=bool)
        pred[2, 3] = True
        pred[3, 4] = True
        pred[0, 1] = True   # FP (in forbidden set)
        pred[0, 2] = True   # FP (in forbidden set)
        pred[8, 9] = True   # FP (NOT in forbidden set)
        cont = pred.astype(float) * 0.5
        for family, conf, cf, ci in (
            ("prior_free", None, None, None),
            ("matched_l1", None, None, None),
            ("soft_frobenius", 1.0, 0.0, 0),
            ("hard_exclusion", None, 0.0, 0),
        ):
            cfg = _make_config(
                method_family=family, seed=seed,
                confidence=conf, corruption_fraction=cf,
                corruption_index=ci,
            )
            planned = make_planned_run(cfg, _RUN_HASH12)
            _persist(_build_record(
                planned, base_dir=base_dir,
                thresholded=pred, continuous_w=cont,
                true_adj=true_adj,
                sid=10.0, shd=6.0, mmd=0.02,
            ), planned, base_dir)


# ===========================================================================
# 1. is_dag detects DAGs and cycles
# ===========================================================================


def test_is_dag_detects_simple_dag():
    arr = np.zeros((3, 3), dtype=bool)
    arr[0, 1] = True
    arr[1, 2] = True
    assert is_dag(arr) is True


def test_is_dag_detects_two_cycle():
    arr = np.zeros((3, 3), dtype=bool)
    arr[0, 1] = True
    arr[1, 0] = True
    assert is_dag(arr) is False


def test_is_dag_detects_three_cycle():
    arr = np.zeros((4, 4), dtype=bool)
    arr[0, 1] = True
    arr[1, 2] = True
    arr[2, 0] = True
    assert is_dag(arr) is False


def test_is_dag_detects_self_loop():
    arr = np.zeros((3, 3), dtype=bool)
    arr[0, 0] = True
    assert is_dag(arr) is False


# ===========================================================================
# 2. classify_edges (delegates to M-10 helper)
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
    classes = classify_edges(pred, true)
    assert classes["true_positive_edges"] == {(0, 1), (2, 0)}
    assert classes["true_negative_edges"] == {(0, 2), (1, 0)}
    assert classes["false_positive_edges"] == {(1, 2)}
    assert classes["false_negative_edges"] == {(2, 1)}


# ===========================================================================
# 3. remove_edges returns copy and does not mutate input
# ===========================================================================


def test_remove_edges_returns_copy():
    pred = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ])
    edited = remove_edges(pred, ((0, 1), (1, 2)))
    assert edited is not pred
    assert pred[0, 1] == True   # noqa: E712
    assert pred[1, 2] == True   # noqa: E712
    assert edited[0, 1] == False   # noqa: E712
    assert edited[1, 2] == False   # noqa: E712
    # Unrelated entry preserved.
    assert edited[0, 2] == True   # noqa: E712


# ===========================================================================
# 4. add_edges_with_acyclicity_guard
# ===========================================================================


def test_add_edges_with_acyclicity_guard_skips_cycles():
    base = np.zeros((3, 3), dtype=bool)
    base[0, 1] = True
    base[1, 2] = True
    edited, added, skipped = add_edges_with_acyclicity_guard(
        base, [(0, 2), (2, 0), (2, 1)],
    )
    # (0,2) is safe: still a DAG.
    assert (0, 2) in added
    # (2,0) creates a cycle with the new (0,2) edge.
    assert (2, 0) in skipped
    # (2,1) creates a cycle with (1,2).
    assert (2, 1) in skipped
    # Input untouched.
    assert base[0, 2] == False   # noqa: E712


def test_add_edges_with_acyclicity_guard_rejects_self_loops():
    base = np.zeros((3, 3), dtype=bool)
    edited, added, skipped = add_edges_with_acyclicity_guard(
        base, [(1, 1)],
    )
    assert (1, 1) in skipped
    assert added == []


# ===========================================================================
# 5. evaluate_single_edge_addition returns invalid for cycles
# ===========================================================================


def test_evaluate_single_edge_addition_cycle_is_invalid():
    current = np.zeros((3, 3), dtype=bool)
    current[0, 1] = True
    current[1, 2] = True
    true_adj = np.zeros((3, 3), dtype=bool)
    res = evaluate_single_edge_addition(
        current=current, true=true_adj, edge=(2, 0),
    )
    assert res["valid"] is False
    assert res["invalid_reason"] == "cycle"
    assert res["sid_after"] is None
    assert res["shd_after"] is None
    assert res["sid_delta"] is None
    assert res["shd_delta"] is None


def test_evaluate_single_edge_addition_valid_returns_deltas():
    # A 3-node graph; current is empty, true has edge (0,1).
    current = np.zeros((3, 3), dtype=bool)
    true_adj = np.zeros((3, 3), dtype=bool)
    true_adj[0, 1] = True
    res = evaluate_single_edge_addition(
        current=current, true=true_adj, edge=(0, 1),
    )
    assert res["valid"] is True
    assert res["invalid_reason"] is None
    assert res["sid_after"] is not None
    assert res["shd_after"] is not None


# ===========================================================================
# 6/7/8. exact_budget_false_positive_removal
# ===========================================================================


def _simple_pred_true() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic predicted/true pair with a known FP set."""
    n = 5
    true_adj = np.zeros((n, n), dtype=bool)
    true_adj[2, 3] = True
    pred = np.zeros((n, n), dtype=bool)
    pred[2, 3] = True   # TP
    pred[0, 1] = True   # FP
    pred[0, 2] = True   # FP
    pred[1, 4] = True   # FP
    return pred, true_adj


def test_exact_budget_fp_removal_enumerates_subsets_up_to_k():
    pred, true_adj = _simple_pred_true()
    candidates = [(0, 1), (0, 2), (1, 4)]
    out = exact_budget_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=2,
    )
    # n_selected_edges must respect the budget.
    assert out["n_selected_edges"] <= 2
    assert out["search_strategy"] == "exact_subset_sid_primary"


def test_exact_budget_fp_removal_includes_empty_subset():
    """The empty subset is always considered, so the selected result
    cannot be worse than the original under the selection rule."""
    pred, true_adj = _simple_pred_true()
    candidates = [(0, 1), (0, 2), (1, 4)]
    out = exact_budget_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=3,
    )
    assert out["sid_after"] <= out["sid_original"] or (
        out["sid_after"] == out["sid_original"]
        and out["shd_after"] <= out["shd_original"]
    )


def test_exact_budget_fp_removal_is_deterministic():
    pred, true_adj = _simple_pred_true()
    candidates = [(0, 1), (0, 2), (1, 4)]
    a = exact_budget_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=3,
    )
    b = exact_budget_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=3,
    )
    assert a == b


# ===========================================================================
# 9/10. full_false_positive_removal
# ===========================================================================


def test_full_fp_removal_removes_all_candidates():
    pred, true_adj = _simple_pred_true()
    candidates = [(0, 1), (0, 2), (1, 4)]
    out = full_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates,
    )
    assert out["n_selected_edges"] == 3
    assert out["search_strategy"] == "remove_all_false_positives"


def test_full_fp_removal_reduces_shd_by_count_in_simple_case():
    """Removing 3 spurious FPs reduces SHD by exactly 3 in this case."""
    pred, true_adj = _simple_pred_true()
    candidates = [(0, 1), (0, 2), (1, 4)]
    out = full_false_positive_removal(
        predicted=pred, true=true_adj,
        candidate_edges=candidates,
    )
    assert out["shd_delta"] == -3


# ===========================================================================
# 11/12/13/14. greedy_acyclic_false_negative_addition
# ===========================================================================


def test_greedy_fn_skips_cycle_inducing_candidates():
    n = 3
    true_adj = np.zeros((n, n), dtype=bool)
    true_adj[0, 1] = True
    true_adj[1, 2] = True
    # Predicted has the reverse direction (1,0), so adding the
    # true edge (0,1) would create a 2-cycle.
    pred = np.zeros((n, n), dtype=bool)
    pred[1, 0] = True
    candidates = [(0, 1), (1, 2)]
    out = greedy_acyclic_false_negative_addition(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=10,
    )
    assert (0, 1) in out["skipped_cycle_edges"]
    assert out["n_skipped_cycle_edges"] >= 1


def test_greedy_fn_reports_skipped_edges():
    n = 3
    true_adj = np.zeros((n, n), dtype=bool)
    pred = np.zeros((n, n), dtype=bool)
    pred[0, 1] = True
    pred[1, 2] = True
    # All candidates would form cycles or be self-loops.
    candidates = [(1, 0), (2, 1), (2, 2)]
    out = greedy_acyclic_false_negative_addition(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=10,
    )
    assert out["n_selected_edges"] == 0
    assert out["n_skipped_cycle_edges"] >= 1


def test_greedy_fn_stops_when_no_beneficial_edit_remains():
    n = 4
    # Predicted already equals truth -> no beneficial FN candidate.
    true_adj = np.zeros((n, n), dtype=bool)
    true_adj[0, 1] = True
    pred = true_adj.copy()
    candidates = [(2, 3)]  # Adding (2,3) doesn't reduce SID since it's FP.
    out = greedy_acyclic_false_negative_addition(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=10,
    )
    # The only candidate is not a FN of pred; adding it would increase SHD.
    assert out["n_selected_edges"] == 0
    assert out["sid_delta"] == 0
    assert out["shd_delta"] == 0


def test_greedy_fn_respects_budget_k():
    n = 6
    true_adj = np.zeros((n, n), dtype=bool)
    for (i, j) in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        true_adj[i, j] = True
    pred = np.zeros((n, n), dtype=bool)
    # All 5 true edges are FNs of pred.
    candidates = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    out = greedy_acyclic_false_negative_addition(
        predicted=pred, true=true_adj,
        candidate_edges=candidates, budget_k=2,
    )
    assert out["n_selected_edges"] <= 2


# ===========================================================================
# 15. actual_reference_forbidden_removal
# ===========================================================================


def test_actual_reference_forbidden_removal_matches_edited_graph():
    pred = np.zeros((5, 5), dtype=bool)
    pred[0, 1] = True
    pred[0, 2] = True
    pred[2, 3] = True
    true_adj = np.zeros((5, 5), dtype=bool)
    true_adj[2, 3] = True
    reference = ((0, 1), (0, 2))
    out = actual_reference_forbidden_removal(
        predicted=pred, true=true_adj,
        reference_forbidden_edges=reference,
    )
    # 2 FPs removed -> shd_delta == -2.
    assert out["shd_delta"] == -2
    assert out["n_candidate_edges"] == 2
    assert out["n_selected_edges"] == 2
    assert out["search_strategy"] == "remove_reference_forbidden"


# ===========================================================================
# 16. summarise_oracle_diagnostics
# ===========================================================================


def test_summarise_oracle_diagnostics_grouped_stats():
    rows = [
        {
            "seed_value": 1, "scenario_label": SCENARIO_FP_REMOVE_ALL,
            "search_strategy": "remove_all_false_positives",
            "n_candidate_edges": 3, "n_selected_edges": 3,
            "n_skipped_cycle_edges": 0,
            "sid_original": 10, "sid_after": 5, "sid_delta": -5,
            "shd_original": 6, "shd_after": 3, "shd_delta": -3,
            "selected_edges_json": "[]",
            "skipped_cycle_edges_json": "[]",
        },
        {
            "seed_value": 2, "scenario_label": SCENARIO_FP_REMOVE_ALL,
            "search_strategy": "remove_all_false_positives",
            "n_candidate_edges": 5, "n_selected_edges": 5,
            "n_skipped_cycle_edges": 0,
            "sid_original": 12, "sid_after": 8, "sid_delta": -4,
            "shd_original": 7, "shd_after": 2, "shd_delta": -5,
            "selected_edges_json": "[]",
            "skipped_cycle_edges_json": "[]",
        },
    ]
    summary = summarise_oracle_diagnostics(rows)
    fp_row = next(
        s for s in summary if s["scenario_label"] == SCENARIO_FP_REMOVE_ALL
    )
    assert fp_row["n_seeds"] == 2
    assert fp_row["mean_sid_delta"] == pytest.approx(-4.5)
    assert fp_row["median_sid_delta"] == pytest.approx(-4.5)
    assert fp_row["min_sid_delta"] == -5
    assert fp_row["max_sid_delta"] == -4
    assert fp_row["mean_shd_delta"] == pytest.approx(-4.0)
    assert fp_row["mean_n_candidate_edges"] == pytest.approx(4.0)


# ===========================================================================
# 17. analysis_hash12 determinism
# ===========================================================================


def test_analysis_hash12_uses_only_declared_payload_fields():
    h_a, payload_a = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        prior_relevance_analysis_hash12="6f660aaeef3d",
        budget_k=10,
        sorted_input_run_ids=["b", "a", "c"],
        sorted_input_configuration_hashes=["xyz", "abc"],
    )
    h_b, payload_b = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        prior_relevance_analysis_hash12="6f660aaeef3d",
        budget_k=10,
        sorted_input_run_ids=["a", "b", "c"],
        sorted_input_configuration_hashes=["abc", "xyz"],
    )
    assert h_a == h_b
    assert len(h_a) == 12
    assert all(c in "0123456789abcdef" for c in h_a)
    assert set(payload_a.keys()) == {
        "main_evaluation_run_hash12",
        "analysis_protocol_version",
        "input_run_ids_sorted",
        "input_configuration_hashes_sorted",
        "prior_relevance_analysis_hash12",
        "budget_k",
    }
    assert payload_a["analysis_protocol_version"] == ANALYSIS_PROTOCOL_VERSION
    assert payload_a["budget_k"] == 10


def test_analysis_hash12_changes_with_budget_k():
    h_10, _ = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        prior_relevance_analysis_hash12="6f660aaeef3d",
        budget_k=10,
        sorted_input_run_ids=["a"],
        sorted_input_configuration_hashes=["x"],
    )
    h_5, _ = compute_analysis_hash12(
        main_evaluation_run_hash12="166c792c43bc",
        prior_relevance_analysis_hash12="6f660aaeef3d",
        budget_k=5,
        sorted_input_run_ids=["a"],
        sorted_input_configuration_hashes=["x"],
    )
    assert h_10 != h_5


# ===========================================================================
# 18. manifest includes required no_* flags
# ===========================================================================


def test_manifest_includes_required_no_flags_and_final_marker(tmp_path):
    _build_synthetic_28_record_grid(tmp_path)
    manifest = run_oracle_prior_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    for flag in (
        "no_new_fits",
        "no_mmd_recomputation",
        "no_new_sampling",
        "no_protocol_changes",
        "final_scheduled_exploratory_diagnostic",
    ):
        assert manifest[flag] is True


# ===========================================================================
# 19. readout excludes forbidden verdict phrases
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
    _build_synthetic_28_record_grid(tmp_path)
    manifest = run_oracle_prior_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    out_dir = analysis_output_dir(tmp_path, manifest["analysis_hash12"])
    text = (out_dir / ORACLE_READOUT_MD).read_text(encoding="utf-8").lower()
    for phrase in _FORBIDDEN_READOUT_PHRASES:
        assert phrase not in text, (
            f"forbidden verdict phrase {phrase!r} in readout"
        )


# ===========================================================================
# 20/21/22. orchestrator and CLI
# ===========================================================================


def test_run_oracle_writes_all_required_outputs(tmp_path):
    _build_synthetic_28_record_grid(tmp_path)
    manifest = run_oracle_prior_relevance_analysis(
        output_root=tmp_path,
        main_evaluation_run_hash12=_RUN_HASH12,
    )
    out_dir = analysis_output_dir(tmp_path, manifest["analysis_hash12"])
    for required in (
        ORACLE_PER_SEED_CSV,
        ORACLE_SUMMARY_CSV,
        ORACLE_READOUT_MD,
        ORACLE_MANIFEST_JSON,
    ):
        assert (out_dir / required).exists(), (
            f"missing required output {required!r}"
        )


def test_cli_returns_zero_on_synthetic(tmp_path):
    _build_synthetic_28_record_grid(tmp_path)
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
# 23. static import check
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "hashlib",
    "itertools",
    "json",
    "math",
    "pathlib",
    "sys",
    "typing",
    "numpy",
    "matplotlib",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "experiments.main_study.exploratory",
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
    src = Path(opr_mod.__file__).read_text(encoding="utf-8")
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
            f"oracle_prior_relevance.py import {mod!r} not in "
            f"allowlist {sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(opr_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"forbidden import: {mod!r}"
            )


# ===========================================================================
# 24. source hygiene regex check
# ===========================================================================


_MILESTONE_REGEX: re.Pattern[str] = re.compile(
    r"\bM[-_]?(?:[0-9]|10|11)[a-c]?\b"
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


_HYGIENE_TARGETS: tuple[Path, ...] = (
    Path("experiments/main_study/exploratory/oracle_prior_relevance.py"),
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
# 25. tests write only under tmp_path
# ===========================================================================


def test_tests_write_only_under_tmp_path(tmp_path):
    _build_synthetic_28_record_grid(tmp_path)
    assert (tmp_path / "results").is_dir()
