"""Tests for the dependency-injected main-study single-run executor.

All tests use fake data/fit/metric backends; no real DAGMA fit
runs, no metric module is imported, and no filesystem I/O occurs.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from experiments.main_study import executor as executor_mod
from experiments.main_study.executor import (
    DataBundle,
    ExecutionResult,
    FitOutcome,
    MetricOutcome,
    ModelFitFailure,
    execute_planned_run,
)
from experiments.main_study.priors import (
    PRIOR_SEED_BASE,
    CorruptedPriorSpec,
    build_confidence_mask,
    edge_tuple_to_key,
)
from experiments.main_study.records import (
    GRAPH_STATUS_VALUES,
    SAMPLER_STATUS_VALUES,
    MainStudyRunRecord,
)
from experiments.main_study.schema import (
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import (
    PlannedRun,
    expected_artefact_names_for_method,
    make_planned_run,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_RUN_HASH12 = "0123456789ab"
_GENERATED_AT = "2026-05-24T12:00:00Z"
_N_NODES = 5


def _simple_true_adj() -> np.ndarray:
    """Bool 5x5 with edges 0->1, 1->2, 2->3, 3->4 (a chain)."""
    a = np.zeros((_N_NODES, _N_NODES), dtype=bool)
    for i in range(_N_NODES - 1):
        a[i, i + 1] = True
    return a


def _x_train() -> np.ndarray:
    return np.zeros((20, _N_NODES), dtype=float)


def _make_corrupted_spec() -> CorruptedPriorSpec:
    """A consistent CorruptedPriorSpec for d=5, prior_k=3, fraction=0.4.

    Clean prior had forbidden edges {(0,2), (2,3), (3,0)}; corruption
    removed (2,3) and added (0,1) (a true positive on the chain).
    Final forbidden = (clean - removed) | added = {(0,1), (0,2), (3,0)}.
    """
    forbidden = ((0, 1), (0, 2), (3, 0))
    removed = ((2, 3),)
    added = ((0, 1),)
    labels = {
        "0,1": "true_positive_corrupted_replacement",
        "0,2": "true_negative_retained",
        "3,0": "true_negative_retained",
    }
    return CorruptedPriorSpec(
        n_nodes=_N_NODES,
        scm_seed=42,
        corruption_fraction=0.4,
        corruption_index=2,
        corruption_seed=9100 + 42 + 2,
        forbidden_edges=forbidden,
        n_correct=2,
        n_corrupted=1,
        removed_clean_edges=removed,
        added_true_positive_edges=added,
        edge_labels=labels,
    )


def _prior_free_config() -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
    )


def _matched_l1_config(matched: float = 0.07) -> MainStudyConfig:
    return make_main_study_config(
        method_family="matched_l1",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(lambda1=matched),
        parent_heldout_run_hash_full=_PARENT_HASH,
        matched_l1_lambda1=matched,
    )


def _soft_frobenius_config(confidence: float = 0.5) -> MainStudyConfig:
    return make_main_study_config(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        confidence=confidence,
        corrupted_prior_spec=_make_corrupted_spec(),
    )


def _hard_exclusion_config() -> MainStudyConfig:
    return make_main_study_config(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        corrupted_prior_spec=_make_corrupted_spec(),
    )


_UNSET = object()


def _default_sampler() -> Any:
    """A no-op callable that satisfies the model_sampler contract."""

    def _sampler(*args: Any, **kwargs: Any) -> None:
        return None

    return _sampler


def _make_fit_outcome(
    *,
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
    training_status: str = "converged",
    wrapper_diagnostics: dict[str, Any] | None = None,
    model_sampler: Any = _UNSET,
) -> FitOutcome:
    if model_sampler is _UNSET:
        model_sampler = (
            _default_sampler() if sampler_status == "available" else None
        )
    return FitOutcome(
        continuous_w=np.zeros((_N_NODES, _N_NODES), dtype=float),
        thresholded_adjacency=np.zeros(
            (_N_NODES, _N_NODES), dtype=bool
        ),
        graph_status=graph_status,
        sampler_status=sampler_status,
        training_status=training_status,
        wrapper_diagnostics=(
            {"key": "value"}
            if wrapper_diagnostics is None
            else wrapper_diagnostics
        ),
        model_sampler=model_sampler,
    )


def _make_metric_outcome(
    *,
    sid: float = 1.0,
    shd: float = 2.0,
    mmd: float = -1e-4,
    metric_runtime_seconds: float = 0.1,
) -> MetricOutcome:
    return MetricOutcome(
        sid=sid,
        shd=shd,
        mmd=mmd,
        interventions_mmd={"records": [], "mmd_primary": mmd},
        metric_runtime_seconds=metric_runtime_seconds,
    )


def _make_data_bundle(metadata: dict | None = None) -> DataBundle:
    return DataBundle(
        x_train=_x_train(),
        true_adjacency=_simple_true_adj(),
        scm_seed=42,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# T-1: DataBundle validation
# ---------------------------------------------------------------------------


def test_databundle_accepts_valid_inputs():
    db = _make_data_bundle()
    assert db.x_train.shape == (20, _N_NODES)
    assert db.true_adjacency.shape == (_N_NODES, _N_NODES)
    assert db.scm_seed == 42


def test_databundle_rejects_non_2d_x_train():
    with pytest.raises(ValueError, match="2D"):
        DataBundle(
            x_train=np.zeros(10),
            true_adjacency=_simple_true_adj(),
            scm_seed=42,
        )


def test_databundle_rejects_non_square_true_adjacency():
    with pytest.raises(ValueError, match="square"):
        DataBundle(
            x_train=_x_train(),
            true_adjacency=np.zeros((3, 4), dtype=bool),
            scm_seed=42,
        )


def test_databundle_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="match"):
        DataBundle(
            x_train=_x_train(),
            true_adjacency=np.zeros((3, 3), dtype=bool),
            scm_seed=42,
        )


def test_databundle_rejects_bool_scm_seed():
    with pytest.raises(ValueError, match="scm_seed"):
        DataBundle(
            x_train=_x_train(),
            true_adjacency=_simple_true_adj(),
            scm_seed=True,
        )


def test_databundle_rejects_non_finite_x_train():
    bad = _x_train()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DataBundle(
            x_train=bad,
            true_adjacency=_simple_true_adj(),
            scm_seed=42,
        )


# ---------------------------------------------------------------------------
# T-2: FitOutcome validation
# ---------------------------------------------------------------------------


def test_fitoutcome_accepts_valid_inputs():
    fo = _make_fit_outcome()
    assert fo.graph_status == "valid_dag"
    assert fo.sampler_status == "available"


def test_fitoutcome_rejects_shape_mismatch_between_w_and_thresholded():
    with pytest.raises(ValueError, match="shape"):
        FitOutcome(
            continuous_w=np.zeros((_N_NODES, _N_NODES)),
            thresholded_adjacency=np.zeros((3, 3), dtype=bool),
            graph_status="valid_dag",
            sampler_status="available",
            training_status="converged",
            wrapper_diagnostics={},
        )


def test_fitoutcome_rejects_invalid_graph_status():
    with pytest.raises(ValueError, match="graph_status"):
        _make_fit_outcome(graph_status="not_a_status")


def test_fitoutcome_rejects_invalid_sampler_status():
    with pytest.raises(ValueError, match="sampler_status"):
        _make_fit_outcome(sampler_status="not_a_status")


def test_fitoutcome_rejects_non_square_continuous_w():
    with pytest.raises(ValueError, match="square"):
        FitOutcome(
            continuous_w=np.zeros((3, 4)),
            thresholded_adjacency=np.zeros((3, 4), dtype=bool),
            graph_status="valid_dag",
            sampler_status="available",
            training_status="converged",
            wrapper_diagnostics={},
        )


# ---------------------------------------------------------------------------
# T-3: MetricOutcome validation
# ---------------------------------------------------------------------------


def test_metricoutcome_accepts_finite_negative_mmd():
    mo = _make_metric_outcome(mmd=-0.5)
    assert mo.mmd == -0.5


def test_metricoutcome_accepts_zero_mmd():
    mo = _make_metric_outcome(mmd=0.0)
    assert mo.mmd == 0.0


def test_metricoutcome_rejects_negative_sid():
    with pytest.raises(ValueError, match="sid"):
        _make_metric_outcome(sid=-1.0)


def test_metricoutcome_rejects_negative_shd():
    with pytest.raises(ValueError, match="shd"):
        _make_metric_outcome(shd=-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_metricoutcome_rejects_non_finite_mmd(bad):
    with pytest.raises(ValueError, match="mmd"):
        _make_metric_outcome(mmd=bad)


def test_metricoutcome_rejects_negative_metric_runtime_seconds():
    with pytest.raises(ValueError, match="metric_runtime_seconds"):
        _make_metric_outcome(metric_runtime_seconds=-0.001)


# ---------------------------------------------------------------------------
# T-4: prior_free computed path
# ---------------------------------------------------------------------------


def _run_prior_free_with(backends_calls: dict, *, fit_outcome=None, metric_outcome=None):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    data_bundle = _make_data_bundle()

    def fake_loader(c):
        backends_calls.setdefault("loader", []).append(c)
        return data_bundle

    def fake_fit(p, d, mask):
        backends_calls.setdefault("fit", []).append((p, d, mask))
        return fit_outcome if fit_outcome is not None else _make_fit_outcome()

    def fake_metric(p, d, fo):
        backends_calls.setdefault("metric", []).append((p, d, fo))
        return metric_outcome if metric_outcome is not None else _make_metric_outcome()

    return execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )


def test_prior_free_computed_calls_each_backend_once():
    calls: dict = {}
    result = _run_prior_free_with(calls)
    assert len(calls["loader"]) == 1
    assert len(calls["fit"]) == 1
    assert len(calls["metric"]) == 1


def test_prior_free_passes_none_confidence_mask_to_fit_backend():
    calls: dict = {}
    _run_prior_free_with(calls)
    _planned, _data, mask = calls["fit"][0]
    assert mask is None


def test_prior_free_computed_record_carries_metrics_including_negative_mmd():
    calls: dict = {}
    result = _run_prior_free_with(
        calls,
        metric_outcome=_make_metric_outcome(
            sid=3.0, shd=4.0, mmd=-1e-4, metric_runtime_seconds=0.2,
        ),
    )
    rec = result.record
    assert rec.metric_status == "computed"
    assert rec.sid == 3.0
    assert rec.shd == 4.0
    assert rec.mmd == -1e-4
    assert rec.metric_runtime_seconds == 0.2


def test_prior_free_computed_artefacts_match_expected_set():
    calls: dict = {}
    result = _run_prior_free_with(calls)
    assert set(result.artefacts.keys()) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "interventions_mmd.json",
    }


def test_prior_free_does_not_include_prior_artefacts_or_confidence_mask():
    calls: dict = {}
    result = _run_prior_free_with(calls)
    assert "confidence_mask.npz" not in result.artefacts
    assert "prior_edge_set_clean.json" not in result.artefacts
    assert "prior_edge_set_corrupted.json" not in result.artefacts
    assert "per_edge_labels.json" not in result.artefacts


# ---------------------------------------------------------------------------
# T-5: matched_l1 computed path
# ---------------------------------------------------------------------------


def test_matched_l1_computed_path():
    cfg = _matched_l1_config(matched=0.07)
    planned = make_planned_run(cfg, _RUN_HASH12)
    db = _make_data_bundle()

    calls: dict = {}

    def fake_loader(c):
        calls.setdefault("loader", []).append(c)
        return db

    def fake_fit(p, d, mask):
        calls.setdefault("fit", []).append((p, d, mask))
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        calls.setdefault("metric", []).append((p, d, fo))
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    rec = result.record
    assert rec.config.method_family == "matched_l1"
    assert rec.config.matched_l1_lambda1 == pytest.approx(0.07)
    assert rec.config.dagma_config.lambda1 == pytest.approx(0.07)
    # Same artefact set as prior_free.
    assert set(result.artefacts.keys()) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "interventions_mmd.json",
    }
    # Confidence mask was not passed.
    assert calls["fit"][0][2] is None


# ---------------------------------------------------------------------------
# T-6: soft_frobenius computed path
# ---------------------------------------------------------------------------


def test_soft_frobenius_passes_built_confidence_mask_to_fit_backend():
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)
    db = _make_data_bundle()

    captured_mask: dict[str, Any] = {}

    def fake_loader(c):
        return db

    def fake_fit(p, d, mask):
        captured_mask["mask"] = mask
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    mask = captured_mask["mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (_N_NODES, _N_NODES)
    # confidence=0.5 sits at every forbidden edge.
    expected_mask = build_confidence_mask(
        cfg.corrupted_prior_spec, 0.5
    )
    np.testing.assert_array_equal(mask, expected_mask)
    # Off-edge positions are zero.
    forbidden = set(cfg.corrupted_prior_spec.forbidden_edges)
    for i in range(_N_NODES):
        for j in range(_N_NODES):
            if (i, j) in forbidden:
                assert mask[i, j] == pytest.approx(0.5)
            else:
                assert mask[i, j] == 0.0


def test_soft_frobenius_record_lambda_prior_is_frozen():
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert result.record.config.lambda_prior == pytest.approx(
        FROZEN_LAMBDA_PRIOR
    )


def test_soft_frobenius_artefacts_include_confidence_mask_and_prior_files():
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert set(result.artefacts.keys()) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "confidence_mask.npz",
        "prior_edge_set_clean.json",
        "prior_edge_set_corrupted.json",
        "per_edge_labels.json",
        "interventions_mmd.json",
    }


def test_soft_frobenius_clean_prior_payload_reconstructed_from_corrupted_spec():
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    payload = result.artefacts["prior_edge_set_clean.json"]
    cp = cfg.corrupted_prior_spec
    # clean = (forbidden - added_TP) | removed_clean — pure set arithmetic
    # over CorruptedPriorSpec fields; no prior-regeneration call.
    forbidden = {tuple(e) for e in cp.forbidden_edges}
    added = {tuple(e) for e in cp.added_true_positive_edges}
    removed = {tuple(e) for e in cp.removed_clean_edges}
    expected_clean = sorted((forbidden - added) | removed)
    assert payload["forbidden_edges"] == [
        [int(i), int(j)] for (i, j) in expected_clean
    ]
    assert payload["n_nodes"] == cp.n_nodes
    assert payload["scm_seed"] == cp.scm_seed
    # prior_selection_seed is derived from the frozen project rule
    # (PRIOR_SEED_BASE + scm_seed); this is protocol provenance, not an
    # invented value.
    assert payload["prior_selection_seed"] == PRIOR_SEED_BASE + cp.scm_seed


def test_clean_prior_payload_omits_selection_seed_when_scm_seed_is_none():
    """A CorruptedPriorSpec with scm_seed=None cannot derive the
    canonical prior_selection_seed, so the payload omits it rather
    than inventing a value."""
    from experiments.main_study.executor import (
        _clean_prior_payload_from_corrupted_spec,
    )

    cp_no_seed = CorruptedPriorSpec(
        n_nodes=_N_NODES,
        scm_seed=None,
        corruption_fraction=0.0,
        corruption_index=0,
        corruption_seed=9100,
        forbidden_edges=((0, 1),),
        n_correct=1,
        n_corrupted=0,
        removed_clean_edges=(),
        added_true_positive_edges=(),
        edge_labels={"0,1": "true_negative_retained"},
    )
    payload = _clean_prior_payload_from_corrupted_spec(cp_no_seed)
    assert payload["scm_seed"] is None
    assert "prior_selection_seed" not in payload


def test_clean_prior_payload_clean_set_recovered_by_set_arithmetic():
    """Independent regression on the (forbidden - added) | removed
    identity using a hand-crafted CorruptedPriorSpec."""
    from experiments.main_study.executor import (
        _clean_prior_payload_from_corrupted_spec,
    )

    cp = CorruptedPriorSpec(
        n_nodes=6,
        scm_seed=99,
        corruption_fraction=0.4,
        corruption_index=2,
        corruption_seed=9100 + 99 + 2,
        # final forbidden after corruption (sorted)
        forbidden_edges=((0, 2), (1, 4), (3, 5)),
        n_correct=2,
        n_corrupted=1,
        # one clean edge was removed (replaced by a TP)
        removed_clean_edges=((2, 5),),
        # one TP was injected as the replacement
        added_true_positive_edges=((1, 4),),
        edge_labels={
            "0,2": "true_negative_retained",
            "1,4": "true_positive_corrupted_replacement",
            "3,5": "true_negative_retained",
        },
    )
    payload = _clean_prior_payload_from_corrupted_spec(cp)
    # Expected clean set: ({(0,2),(1,4),(3,5)} - {(1,4)}) | {(2,5)}
    #                  = {(0,2),(3,5)} | {(2,5)}
    #                  = {(0,2),(2,5),(3,5)}
    assert payload["forbidden_edges"] == [[0, 2], [2, 5], [3, 5]]
    # And the protocol-derived prior_selection_seed is present.
    assert payload["prior_selection_seed"] == PRIOR_SEED_BASE + 99


def test_executor_does_not_import_or_call_prior_regeneration_helpers():
    """Regression: the clean-prior payload must come from set arithmetic
    over the CorruptedPriorSpec, never from a prior-generation call."""
    src = Path(executor_mod.__file__).read_text(encoding="utf-8")
    assert "generate_prior_for_scm_seed" not in src
    assert "sample_clean_forbidden_edges" not in src


# ---------------------------------------------------------------------------
# T-7: hard_exclusion computed path
# ---------------------------------------------------------------------------


def test_hard_exclusion_confidence_mask_is_none_in_fit_call():
    cfg = _hard_exclusion_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    captured: dict[str, Any] = {}

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        captured["mask"] = mask
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert captured["mask"] is None


def test_hard_exclusion_dagma_exclude_edges_match_forbidden_edges():
    cfg = _hard_exclusion_config()
    excl = tuple(sorted(cfg.dagma_config.exclude_edges))
    forb = tuple(sorted(cfg.corrupted_prior_spec.forbidden_edges))
    assert excl == forb


def test_hard_exclusion_artefacts_include_prior_files_not_confidence_mask():
    cfg = _hard_exclusion_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert "confidence_mask.npz" not in result.artefacts
    assert set(result.artefacts.keys()) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "prior_edge_set_clean.json",
        "prior_edge_set_corrupted.json",
        "per_edge_labels.json",
        "interventions_mmd.json",
    }


# ---------------------------------------------------------------------------
# T-8: graph-invalid path
# ---------------------------------------------------------------------------


def test_graph_invalid_does_not_call_metric_backend():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    db = _make_data_bundle()

    metric_calls: list = []

    def fake_loader(c):
        return db

    def fake_fit(p, d, mask):
        return _make_fit_outcome(
            graph_status="cyclic",
            sampler_status="unavailable_invalid_graph",
        )

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert metric_calls == []
    rec = result.record
    assert rec.fit_status == "success"
    assert rec.metric_status == "unavailable_graph_invalid"
    assert rec.sid is None
    assert rec.shd is None
    assert rec.mmd is None
    # Success-required artefacts remain present.
    assert "continuous_w.npz" in result.artefacts
    assert "thresholded_adjacency.npz" in result.artefacts
    assert "true_adjacency.npz" in result.artefacts
    # interventions_mmd absent because metrics were not computed.
    assert "interventions_mmd.json" not in result.artefacts


# ---------------------------------------------------------------------------
# T-9: sampler-unavailable path
# ---------------------------------------------------------------------------


def test_sampler_unavailable_does_not_call_metric_backend():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    metric_calls: list = []

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome(
            graph_status="valid_dag",
            sampler_status="unavailable_unresolved_noise_policy",
        )

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert metric_calls == []
    assert result.record.metric_status == "unavailable_sampler_failure"
    assert result.record.sid is None
    assert "interventions_mmd.json" not in result.artefacts


# ---------------------------------------------------------------------------
# T-10: model-fit failure path
# ---------------------------------------------------------------------------


def test_model_fit_failure_returns_failure_record_without_propagating():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    metric_calls: list = []

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        raise ModelFitFailure("DAGMA diverged at stage 1")

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    rec = result.record
    assert rec.fit_status == "model_fit_failure"
    assert rec.metric_status == "not_computed_due_to_fit_failure"
    assert rec.sid is None
    assert rec.shd is None
    assert rec.mmd is None
    assert "DAGMA diverged" in rec.failure_message
    assert result.artefacts == {}
    # Metric backend never called.
    assert metric_calls == []


def test_model_fit_failure_with_empty_message_still_validates():
    """Empty exception messages should be replaced by the exception
    type name so the record validator (which requires a non-empty
    failure_message when failure_kind is None) does not reject."""
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        raise ModelFitFailure("")

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert result.record.failure_message != ""


# ---------------------------------------------------------------------------
# T-11: infrastructure failure path
# ---------------------------------------------------------------------------


def test_infrastructure_failure_in_data_loader_propagates():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    fit_calls: list = []
    metric_calls: list = []

    def fake_loader(c):
        raise RuntimeError("disk unavailable")

    def fake_fit(p, d, mask):
        fit_calls.append(1)
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    with pytest.raises(RuntimeError, match="disk unavailable"):
        execute_planned_run(
            planned,
            data_loader=fake_loader,
            fit_backend=fake_fit,
            metric_backend=fake_metric,
            generated_at_utc=_GENERATED_AT,
        )
    assert fit_calls == []
    assert metric_calls == []


def test_unexpected_fit_exception_propagates():
    """Exception types other than ModelFitFailure propagate."""
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        raise RuntimeError("unexpected internal error")

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    with pytest.raises(RuntimeError, match="unexpected internal error"):
        execute_planned_run(
            planned,
            data_loader=fake_loader,
            fit_backend=fake_fit,
            metric_backend=fake_metric,
            generated_at_utc=_GENERATED_AT,
        )


# ---------------------------------------------------------------------------
# T-12: diagnostics canonicalisation
# ---------------------------------------------------------------------------


def test_wrapper_diagnostics_canonicalised_and_isolated_from_caller():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    payload = {
        "loss_history": np.array([1.0, 2.0, 3.0]),
        "n_iter": np.int32(2500),
        "training_status": "converged",
    }

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome(wrapper_diagnostics=payload)

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    stored = result.record.wrapper_diagnostics
    # Canonicalised: ndarray -> list, np.int32 -> int.
    assert stored["loss_history"] == [1.0, 2.0, 3.0]
    assert isinstance(stored["loss_history"], list)
    assert stored["n_iter"] == 2500
    assert type(stored["n_iter"]) is int

    # Mutate the caller's dict; record stays unchanged.
    payload["new_key"] = "intruder"
    payload["loss_history"][0] = 999.0
    assert "new_key" not in stored
    assert stored["loss_history"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# T-13: artefact keys match record paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_builder",
    [
        _prior_free_config,
        _matched_l1_config,
        _soft_frobenius_config,
        _hard_exclusion_config,
    ],
)
def test_artefact_keys_match_record_non_none_paths(config_builder):
    cfg = config_builder()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    record = result.record
    name_to_field = {
        "continuous_w.npz": "continuous_w_path",
        "thresholded_adjacency.npz": "thresholded_adjacency_path",
        "true_adjacency.npz": "true_adjacency_path",
        "confidence_mask.npz": "confidence_mask_path",
        "interventions_mmd.json": "interventions_mmd_path",
        "prior_edge_set_clean.json": "prior_edge_set_clean_path",
        "prior_edge_set_corrupted.json": "prior_edge_set_corrupted_path",
        "per_edge_labels.json": "per_edge_labels_path",
    }
    # Every artefact key has a non-None record path.
    for name in result.artefacts:
        assert getattr(record, name_to_field[name]) is not None, (
            f"artefact {name!r} has no record path set"
        )
    # Every non-None record path has a corresponding artefact key.
    for name, field_name in name_to_field.items():
        if getattr(record, field_name) is not None:
            assert name in result.artefacts, (
                f"record path {field_name!r} set but artefact "
                f"{name!r} missing"
            )


# ---------------------------------------------------------------------------
# T-14: no disk writes
# ---------------------------------------------------------------------------


def test_execute_does_not_open_files_or_create_directories(monkeypatch):
    """Sentinel monkeypatches verify the executor performs no filesystem I/O."""
    call_log: list[str] = []

    def trap_open(*args, **kwargs):
        call_log.append("open")
        raise AssertionError("execute_planned_run must not call open()")

    monkeypatch.setattr("builtins.open", trap_open)

    def trap_mkdir(*args, **kwargs):
        call_log.append("mkdir")
        raise AssertionError("execute_planned_run must not create dirs")

    monkeypatch.setattr(Path, "mkdir", trap_mkdir)

    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome()

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert isinstance(result, ExecutionResult)
    assert call_log == []


# ---------------------------------------------------------------------------
# T-15: invalid planned type
# ---------------------------------------------------------------------------


def test_execute_planned_run_rejects_non_plannedrun_input():
    def loader(c):
        return _make_data_bundle()

    def fit(p, d, m):
        return _make_fit_outcome()

    def metric(p, d, f):
        return _make_metric_outcome()

    with pytest.raises(TypeError, match="PlannedRun"):
        execute_planned_run(
            "not a planned run",  # type: ignore[arg-type]
            data_loader=loader,
            fit_backend=fit,
            metric_backend=metric,
            generated_at_utc=_GENERATED_AT,
        )


# ---------------------------------------------------------------------------
# T-16: import allowlist
# ---------------------------------------------------------------------------


_EXECUTOR_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "dataclasses",
    "math",
    "time",
    "typing",
    "datetime",
    "numpy",
    "experiments.main_study.priors",
    "experiments.main_study.records",
    "experiments.main_study.schema",
    "experiments.main_study.workloads",
    "experiments.main_study.paths",
})


_EXECUTOR_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.calibration_lambda_prior",
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


def test_executor_module_does_not_import_forbidden_packages():
    src = Path(executor_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _EXECUTOR_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"executor.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_executor_module_imports_are_allowlisted():
    src = Path(executor_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _EXECUTOR_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _EXECUTOR_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"executor.py import {mod!r} is not in the allowlist "
            f"{sorted(_EXECUTOR_ALLOWED_PREFIXES)}."
        )


# ===========================================================================
# FitOutcome.model_sampler
# ===========================================================================


# ---------------------------------------------------------------------------
# FitOutcome validation for model_sampler
# ---------------------------------------------------------------------------


def test_fitoutcome_accepts_callable_model_sampler_when_sampler_available():
    callable_sampler = _default_sampler()
    fo = FitOutcome(
        continuous_w=np.zeros((_N_NODES, _N_NODES)),
        thresholded_adjacency=np.zeros((_N_NODES, _N_NODES), dtype=bool),
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=callable_sampler,
    )
    assert fo.model_sampler is callable_sampler
    assert callable(fo.model_sampler)


def test_fitoutcome_rejects_none_model_sampler_when_sampler_available():
    with pytest.raises(ValueError, match="model_sampler"):
        FitOutcome(
            continuous_w=np.zeros((_N_NODES, _N_NODES)),
            thresholded_adjacency=np.zeros(
                (_N_NODES, _N_NODES), dtype=bool
            ),
            graph_status="valid_dag",
            sampler_status="available",
            training_status="converged",
            wrapper_diagnostics={},
            model_sampler=None,
        )


def test_fitoutcome_rejects_non_callable_model_sampler_when_sampler_available():
    with pytest.raises(ValueError, match="model_sampler"):
        FitOutcome(
            continuous_w=np.zeros((_N_NODES, _N_NODES)),
            thresholded_adjacency=np.zeros(
                (_N_NODES, _N_NODES), dtype=bool
            ),
            graph_status="valid_dag",
            sampler_status="available",
            training_status="converged",
            wrapper_diagnostics={},
            model_sampler="not a callable",
        )


def test_fitoutcome_accepts_none_model_sampler_when_sampler_unavailable():
    fo = FitOutcome(
        continuous_w=np.zeros((_N_NODES, _N_NODES)),
        thresholded_adjacency=np.zeros((_N_NODES, _N_NODES), dtype=bool),
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=None,
    )
    assert fo.model_sampler is None


def test_fitoutcome_default_model_sampler_is_none():
    """Omitting the kwarg yields None — the default that supports the
    unavailable-sampler path."""
    fo = FitOutcome(
        continuous_w=np.zeros((_N_NODES, _N_NODES)),
        thresholded_adjacency=np.zeros((_N_NODES, _N_NODES), dtype=bool),
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
        training_status="converged",
        wrapper_diagnostics={},
    )
    assert fo.model_sampler is None


# ---------------------------------------------------------------------------
# Metric backend receives the FitOutcome (with callable model_sampler) intact
# ---------------------------------------------------------------------------


def test_metric_backend_receives_fitoutcome_with_callable_model_sampler():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    sampler_marker = _default_sampler()

    captured: dict[str, Any] = {}

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome(model_sampler=sampler_marker)

    def fake_metric(p, d, fo):
        captured["fit_outcome"] = fo
        # The metric backend can inspect or call the sampler.
        assert callable(fo.model_sampler)
        captured["sampler_call_result"] = fo.model_sampler(1, 2, kwarg="x")
        return _make_metric_outcome()

    execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    forwarded = captured["fit_outcome"]
    assert isinstance(forwarded, FitOutcome)
    # Identity preservation: the executor forwards the exact callable
    # the fit backend produced, no wrapping.
    assert forwarded.model_sampler is sampler_marker


# ---------------------------------------------------------------------------
# model_sampler isolation from record, diagnostics, and artefacts
# ---------------------------------------------------------------------------


def test_model_sampler_is_not_in_record_wrapper_diagnostics():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    sampler = _default_sampler()

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        # wrapper_diagnostics is a separate field; the sampler must
        # never appear inside it.
        return _make_fit_outcome(
            wrapper_diagnostics={"training_status": "converged"},
            model_sampler=sampler,
        )

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    diag = result.record.wrapper_diagnostics
    assert "model_sampler" not in diag
    # And no nested value is the sampler callable.
    for value in diag.values():
        assert value is not sampler
        assert not callable(value)


def test_model_sampler_is_not_in_any_artefact_payload():
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)
    sampler = _default_sampler()

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome(model_sampler=sampler)

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )

    def _walk(value):
        """Yield every nested value in a JSON-like / npz-dict payload."""
        yield value
        if isinstance(value, dict):
            for v in value.values():
                yield from _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from _walk(v)

    for artefact_name, payload in result.artefacts.items():
        for value in _walk(payload):
            assert value is not sampler, (
                f"sampler leaked into artefact {artefact_name!r}"
            )
            # Per-value callable check: arrays and dicts are not
            # callable; if any value in the payload is callable,
            # that would be a leak.
            if callable(value):
                # Numpy ndarrays are not callable; allow only non-
                # callable payload values.
                raise AssertionError(
                    f"callable value found in artefact {artefact_name!r}: "
                    f"{value!r}"
                )


def test_main_study_run_record_has_no_model_sampler_field():
    """Sanity: the record schema does not expose a model_sampler field,
    so the sampler cannot accidentally be persisted there."""
    import dataclasses as _dc

    field_names = {f.name for f in _dc.fields(MainStudyRunRecord)}
    assert "model_sampler" not in field_names


# ---------------------------------------------------------------------------
# Gating: invalid graph / unavailable sampler still skip metric_backend
# ---------------------------------------------------------------------------


def test_graph_invalid_path_does_not_require_model_sampler():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    metric_calls: list = []

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        # No model_sampler needed; sampler_status is not "available".
        return _make_fit_outcome(
            graph_status="cyclic",
            sampler_status="unavailable_invalid_graph",
            model_sampler=None,
        )

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert metric_calls == []
    assert result.record.metric_status == "unavailable_graph_invalid"


def test_sampler_unavailable_path_does_not_require_model_sampler():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    metric_calls: list = []

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        return _make_fit_outcome(
            graph_status="valid_dag",
            sampler_status="unavailable_unresolved_noise_policy",
            model_sampler=None,
        )

    def fake_metric(p, d, fo):
        metric_calls.append(1)
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert metric_calls == []
    assert result.record.metric_status == "unavailable_sampler_failure"


def test_model_fit_failure_still_returns_failure_record_with_empty_artefacts():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)

    def fake_loader(c):
        return _make_data_bundle()

    def fake_fit(p, d, mask):
        raise ModelFitFailure("diverged")

    def fake_metric(p, d, fo):
        return _make_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=fake_loader,
        fit_backend=fake_fit,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    assert result.record.fit_status == "model_fit_failure"
    assert result.artefacts == {}
