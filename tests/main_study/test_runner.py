"""Tests for the main-study orchestration loop and resumability policy.

Every test runs against pytest's ``tmp_path``; no file is created
or modified outside ``tmp_path``. Test fixtures build small
:class:`MainStudyConfig`, :class:`PlannedRun`,
:class:`MainStudyRunRecord`, and :class:`ExecutionResult` instances
locally so the orchestrator can be exercised without invoking the
real model, the real metric backend, or the real data loader.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.main_study import runner as runner_mod
from experiments.main_study.executor import (
    ExecutionResult,
    ModelFitFailure,
)
from experiments.main_study.priors import CorruptedPriorSpec
from experiments.main_study.records import (
    SCHEMA_VERSION,
    MainStudyRunRecord,
    make_failure_record,
)
from experiments.main_study.run_io import (
    load_existing_record,
    persist_record_atomic,
)
from experiments.main_study.runner import (
    FINAL_STATUSES,
    RUN_MODES,
    RunSummary,
    WorkloadStatus,
    run_main_study,
)
from experiments.main_study.schema import (
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import (
    PlannedRun,
    make_planned_run,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_PARENT_HASH_OTHER = "b" * 64
_RUN_HASH12 = "0123456789ab"
_GENERATED_AT = "2026-05-24T12:00:00Z"
_N_NODES = 5


def _corrupted_spec() -> CorruptedPriorSpec:
    return CorruptedPriorSpec(
        n_nodes=_N_NODES,
        scm_seed=401,
        corruption_fraction=0.0,
        corruption_index=0,
        corruption_seed=9100 + 401 + 0,
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


def _prior_free_config(
    seed_value: int = 401,
    *,
    parent_hash: str = _PARENT_HASH,
) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=parent_hash,
    )


def _make_planned(cfg: MainStudyConfig) -> PlannedRun:
    return make_planned_run(cfg, _RUN_HASH12)


def _success_record(planned: PlannedRun) -> MainStudyRunRecord:
    return MainStudyRunRecord(
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
        sid=1.0,
        shd=2.0,
        mmd=-1e-4,
        runtime_seconds=1.0,
        fit_runtime_seconds=0.8,
        metric_runtime_seconds=0.2,
        wrapper_diagnostics={"training_status": "converged"},
        continuous_w_path=planned.artefact_paths["continuous_w.npz"],
        thresholded_adjacency_path=planned.artefact_paths[
            "thresholded_adjacency.npz"
        ],
        true_adjacency_path=planned.artefact_paths["true_adjacency.npz"],
        interventions_mmd_path=planned.artefact_paths[
            "interventions_mmd.json"
        ],
        parent_heldout_run_hash_full=planned.config.parent_heldout_run_hash_full,
        generated_at_utc=_GENERATED_AT,
    )


def _success_unavailable_record(
    planned: PlannedRun,
) -> MainStudyRunRecord:
    """A success fit whose sampler was not available."""
    return MainStudyRunRecord(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=_N_NODES,
        fit_status="success",
        graph_status="valid_dag",
        sampler_status="unavailable_no_api",
        metric_status="unavailable_sampler_failure",
        failure_kind=None,
        failure_message="",
        sid=None,
        shd=None,
        mmd=None,
        runtime_seconds=1.0,
        fit_runtime_seconds=0.8,
        metric_runtime_seconds=None,
        wrapper_diagnostics={"training_status": "converged"},
        continuous_w_path=planned.artefact_paths["continuous_w.npz"],
        thresholded_adjacency_path=planned.artefact_paths[
            "thresholded_adjacency.npz"
        ],
        true_adjacency_path=planned.artefact_paths["true_adjacency.npz"],
        parent_heldout_run_hash_full=planned.config.parent_heldout_run_hash_full,
        generated_at_utc=_GENERATED_AT,
    )


def _model_fit_failure_record(planned: PlannedRun) -> MainStudyRunRecord:
    return make_failure_record(
        config=planned.config,
        n_nodes=_N_NODES,
        fit_status="model_fit_failure",
        failure_kind=None,
        failure_message="simulated model fit failure",
        runtime_seconds=0.5,
        fit_runtime_seconds=0.5,
        wrapper_diagnostics={},
        generated_at_utc=_GENERATED_AT,
    )


def _build_artefacts(record: MainStudyRunRecord) -> dict[str, object]:
    payloads: dict[str, object] = {}
    if record.continuous_w_path is not None:
        payloads["continuous_w.npz"] = {
            "continuous_w": np.zeros((_N_NODES, _N_NODES), dtype=float),
        }
    if record.thresholded_adjacency_path is not None:
        payloads["thresholded_adjacency.npz"] = {
            "thresholded_adjacency": np.zeros(
                (_N_NODES, _N_NODES), dtype=bool
            ),
        }
    if record.true_adjacency_path is not None:
        payloads["true_adjacency.npz"] = {
            "true_adjacency": np.zeros(
                (_N_NODES, _N_NODES), dtype=bool
            ),
        }
    if record.interventions_mmd_path is not None:
        payloads["interventions_mmd.json"] = {
            "records": [],
            "mmd_primary": 0.0,
        }
    return payloads


def _success_execution_result(planned: PlannedRun) -> ExecutionResult:
    record = _success_record(planned)
    return ExecutionResult(
        record=record, artefacts=_build_artefacts(record)
    )


def _success_unavailable_execution_result(
    planned: PlannedRun,
) -> ExecutionResult:
    record = _success_unavailable_record(planned)
    return ExecutionResult(
        record=record, artefacts=_build_artefacts(record)
    )


def _model_fit_failure_execution_result(
    planned: PlannedRun,
) -> ExecutionResult:
    record = _model_fit_failure_record(planned)
    return ExecutionResult(record=record, artefacts={})


def _constant_factory(value: str = _GENERATED_AT):
    def _f() -> str:
        return value

    return _f


def _sentinel():
    return object()


def _make_fake_execute(
    *,
    result_factory=None,
    side_effect=None,
):
    """Build a fake ``execute_fn`` that records calls.

    Either ``result_factory(planned)`` returns an ``ExecutionResult``
    or ``side_effect(planned)`` raises. Exactly one must be set.
    """
    if (result_factory is None) == (side_effect is None):
        raise AssertionError(
            "_make_fake_execute requires exactly one of "
            "result_factory or side_effect."
        )
    calls: list[dict[str, Any]] = []

    def fake(planned, **kwargs):
        calls.append({"planned": planned, **kwargs})
        if side_effect is not None:
            return side_effect(planned)
        return result_factory(planned)

    return fake, calls


# ===========================================================================
# A. Dataclass validation
# ===========================================================================


def test_workload_status_accepts_valid():
    s = WorkloadStatus(
        run_id="r",
        configuration_hash_prefix="abcdef012345",
        method_family="prior_free",
        final_status="success_computed",
        record_path="results/main_study/abc/records/r.json",
        runtime_seconds=0.0,
        message="",
    )
    assert s.final_status == "success_computed"


def test_workload_status_rejects_unknown_final_status():
    with pytest.raises(ValueError, match="final_status"):
        WorkloadStatus(
            run_id="r",
            configuration_hash_prefix="abcdef012345",
            method_family="prior_free",
            final_status="not_a_status",
            record_path="a/b.json",
            runtime_seconds=0.0,
            message="",
        )


@pytest.mark.parametrize(
    "bad_runtime",
    [-1.0, float("inf"), float("-inf"), float("nan"), True],
)
def test_workload_status_rejects_bad_runtime(bad_runtime):
    with pytest.raises(ValueError, match="runtime_seconds"):
        WorkloadStatus(
            run_id="r",
            configuration_hash_prefix="abcdef012345",
            method_family="prior_free",
            final_status="success_computed",
            record_path="a/b.json",
            runtime_seconds=bad_runtime,
            message="",
        )


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("run_id", ""),
        ("configuration_hash_prefix", ""),
        ("method_family", ""),
        ("record_path", ""),
    ],
)
def test_workload_status_rejects_empty_required_strings(field_name, value):
    kwargs = dict(
        run_id="r",
        configuration_hash_prefix="abcdef012345",
        method_family="prior_free",
        final_status="success_computed",
        record_path="a/b.json",
        runtime_seconds=0.0,
        message="",
    )
    kwargs[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        WorkloadStatus(**kwargs)


@pytest.mark.parametrize("flag", [True, False])
def test_workload_status_accepts_bool_was_overwritten(flag):
    s = WorkloadStatus(
        run_id="r",
        configuration_hash_prefix="abcdef012345",
        method_family="prior_free",
        final_status="success_computed",
        record_path="a/b.json",
        runtime_seconds=0.0,
        message="",
        was_overwritten=flag,
    )
    assert s.was_overwritten is flag


@pytest.mark.parametrize(
    "bad",
    [0, 1, "true", None, 0.0, 1.0, "False"],
)
def test_workload_status_rejects_non_bool_was_overwritten(bad):
    with pytest.raises(ValueError, match="was_overwritten"):
        WorkloadStatus(
            run_id="r",
            configuration_hash_prefix="abcdef012345",
            method_family="prior_free",
            final_status="success_computed",
            record_path="a/b.json",
            runtime_seconds=0.0,
            message="",
            was_overwritten=bad,
        )


def _empty_summary_kwargs():
    return dict(
        n_planned=0,
        n_executed=0,
        n_success_computed=0,
        n_success_metric_unavailable=0,
        n_model_fit_failure=0,
        n_skipped=0,
        n_overwritten=0,
        n_infrastructure_failure=0,
        total_runtime_seconds=0.0,
        per_workload_status=(),
    )


def test_run_summary_accepts_empty_zero_counts():
    s = RunSummary(**_empty_summary_kwargs())
    assert s.n_planned == 0
    assert s.per_workload_status == ()


def test_run_summary_rejects_n_planned_len_mismatch():
    status = WorkloadStatus(
        run_id="r",
        configuration_hash_prefix="abcdef012345",
        method_family="prior_free",
        final_status="success_computed",
        record_path="a/b.json",
        runtime_seconds=0.0,
        message="",
    )
    kwargs = _empty_summary_kwargs()
    kwargs["per_workload_status"] = (status,)
    kwargs["n_planned"] = 0  # mismatch
    with pytest.raises(ValueError, match="n_planned"):
        RunSummary(**kwargs)


def test_run_summary_rejects_n_executed_sum_mismatch():
    kwargs = _empty_summary_kwargs()
    kwargs["n_executed"] = 5  # but sum of three buckets == 0
    with pytest.raises(ValueError, match="n_executed"):
        RunSummary(**kwargs)


def test_run_summary_rejects_negative_count():
    kwargs = _empty_summary_kwargs()
    kwargs["n_skipped"] = -1
    with pytest.raises(ValueError, match="n_skipped"):
        RunSummary(**kwargs)


def test_run_summary_rejects_bool_count():
    kwargs = _empty_summary_kwargs()
    kwargs["n_skipped"] = True
    with pytest.raises(ValueError, match="n_skipped"):
        RunSummary(**kwargs)


def _status(
    *,
    final_status: str = "success_computed",
    was_overwritten: bool = False,
    run_id: str = "r",
) -> WorkloadStatus:
    return WorkloadStatus(
        run_id=run_id,
        configuration_hash_prefix="abcdef012345",
        method_family="prior_free",
        final_status=final_status,
        record_path=f"a/{run_id}.json",
        runtime_seconds=0.0,
        message="",
        was_overwritten=was_overwritten,
    )


def test_run_summary_accepts_n_overwritten_matching_count():
    statuses = (
        _status(was_overwritten=True, run_id="r1"),
        _status(was_overwritten=False, run_id="r2"),
        _status(was_overwritten=True, run_id="r3"),
    )
    kwargs = _empty_summary_kwargs()
    kwargs.update(
        n_planned=3,
        n_executed=3,
        n_success_computed=3,
        n_overwritten=2,
        per_workload_status=statuses,
    )
    s = RunSummary(**kwargs)
    assert s.n_overwritten == 2


def test_run_summary_rejects_n_overwritten_inconsistent_with_statuses():
    statuses = (
        _status(was_overwritten=True, run_id="r1"),
        _status(was_overwritten=False, run_id="r2"),
    )
    kwargs = _empty_summary_kwargs()
    kwargs.update(
        n_planned=2,
        n_executed=2,
        n_success_computed=2,
        n_overwritten=2,  # observed is 1
        per_workload_status=statuses,
    )
    with pytest.raises(ValueError, match="n_overwritten"):
        RunSummary(**kwargs)


# ===========================================================================
# B. Mode validation and preflight
# ===========================================================================


def test_run_main_study_rejects_unknown_mode(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    with pytest.raises(ValueError, match="mode"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="not_a_mode",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


def test_run_main_study_rejects_empty_planned_runs(tmp_path):
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    with pytest.raises(ValueError, match="non-empty"):
        run_main_study(
            [],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


def test_run_main_study_rejects_mixed_parent_hashes(tmp_path):
    p_a = _make_planned(
        _prior_free_config(seed_value=401, parent_hash=_PARENT_HASH)
    )
    p_b = _make_planned(
        _prior_free_config(
            seed_value=402, parent_hash=_PARENT_HASH_OTHER
        )
    )
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    with pytest.raises(ValueError, match="parent_heldout_run_hash_full"):
        run_main_study(
            [p_a, p_b],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


def test_run_main_study_raise_mode_rejects_existing_record(tmp_path):
    planned = _make_planned(_prior_free_config())
    # Pre-stage an existing record file at the expected path.
    full = tmp_path / planned.record_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("{}", encoding="utf-8")
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    with pytest.raises(FileExistsError, match="already exist"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


# ===========================================================================
# C. Skip mode
# ===========================================================================


def test_skip_mode_executes_when_record_absent(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="skip",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert len(calls) == 1
    assert summary.n_skipped == 0
    assert summary.n_success_computed == 1
    assert (tmp_path / planned.record_path).exists()


def test_skip_mode_skips_when_record_present_and_compatible(tmp_path):
    planned = _make_planned(_prior_free_config())
    # Persist a compatible record first.
    record = _success_record(planned)
    persist_record_atomic(record, planned.record_path, base_dir=tmp_path)
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="skip",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert calls == []
    assert summary.n_skipped == 1
    assert summary.n_executed == 0
    assert summary.n_overwritten == 0
    assert summary.per_workload_status[0].final_status == "skipped"
    assert summary.per_workload_status[0].was_overwritten is False


def test_skip_mode_raises_on_incompatible_existing_record(tmp_path):
    planned_a = _make_planned(_prior_free_config(seed_value=401))
    planned_b = _make_planned(_prior_free_config(seed_value=402))
    # Persist planned_a's record at planned_b's record_path to force a
    # configuration-hash mismatch when planned_b's skip-check loads it.
    record_a = _success_record(planned_a)
    # Use planned_b's record path with planned_a's record content. The
    # most direct way is to write the JSON for record_a under planned_b's
    # path manually.
    from experiments.main_study.records import record_to_json

    full = tmp_path / planned_b.record_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(record_to_json(record_a), encoding="utf-8")
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    with pytest.raises(ValueError):
        run_main_study(
            [planned_b],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="skip",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


def test_skip_mode_mixed_present_and_absent(tmp_path):
    p1 = _make_planned(_prior_free_config(seed_value=401))
    p2 = _make_planned(_prior_free_config(seed_value=402))
    # Pre-persist p1's record only.
    persist_record_atomic(
        _success_record(p1), p1.record_path, base_dir=tmp_path
    )
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [p1, p2],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="skip",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_skipped == 1
    assert summary.n_success_computed == 1
    assert summary.n_executed == 1
    assert [s.final_status for s in summary.per_workload_status] == [
        "skipped",
        "success_computed",
    ]
    assert len(calls) == 1


# ===========================================================================
# D. Overwrite mode
# ===========================================================================


def test_overwrite_mode_executes_when_record_absent(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="overwrite",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert len(calls) == 1
    assert summary.n_overwritten == 0
    assert summary.n_executed == 1
    assert summary.n_success_computed == 1
    assert summary.per_workload_status[0].final_status == "success_computed"
    assert summary.per_workload_status[0].was_overwritten is False


def test_overwrite_mode_success_computed_with_existing_record(tmp_path):
    planned = _make_planned(_prior_free_config())
    # Pre-persist a compatible record (re-running the same config).
    existing_record = _success_record(planned)
    persist_record_atomic(
        existing_record, planned.record_path, base_dir=tmp_path
    )
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )

    persist_calls: list[str] = []
    real_persist = runner_mod.persist_execution_result_atomic

    def tracked_persist(result, record_path, *, base_dir):
        persist_calls.append("persist")
        return real_persist(result, record_path, base_dir=base_dir)

    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="overwrite",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
        persist_execution_result_fn=tracked_persist,
    )
    assert len(calls) == 1
    assert persist_calls == ["persist"]
    assert summary.n_executed == 1
    assert summary.n_success_computed == 1
    assert summary.n_success_metric_unavailable == 0
    assert summary.n_model_fit_failure == 0
    assert summary.n_overwritten == 1
    status = summary.per_workload_status[0]
    assert status.final_status == "success_computed"
    assert status.was_overwritten is True
    assert "atomically replaced" in status.message
    # Record is loadable and reflects the new fit outcome.
    loaded = load_existing_record(
        planned.record_path, base_dir=tmp_path
    )
    assert loaded is not None
    assert loaded.fit_status == "success"


def test_overwrite_mode_metric_unavailable_with_existing_record(tmp_path):
    planned = _make_planned(_prior_free_config())
    existing_record = _success_record(planned)
    persist_record_atomic(
        existing_record, planned.record_path, base_dir=tmp_path
    )
    fake, _ = _make_fake_execute(
        result_factory=_success_unavailable_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="overwrite",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_executed == 1
    assert summary.n_success_metric_unavailable == 1
    assert summary.n_overwritten == 1
    status = summary.per_workload_status[0]
    assert status.final_status == "success_metric_unavailable"
    assert status.was_overwritten is True


def test_overwrite_mode_model_fit_failure_with_existing_record(tmp_path):
    planned = _make_planned(_prior_free_config())
    existing_record = _success_record(planned)
    persist_record_atomic(
        existing_record, planned.record_path, base_dir=tmp_path
    )
    fake, _ = _make_fake_execute(
        result_factory=_model_fit_failure_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="overwrite",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_executed == 1
    assert summary.n_model_fit_failure == 1
    assert summary.n_overwritten == 1
    status = summary.per_workload_status[0]
    assert status.final_status == "model_fit_failure"
    assert status.was_overwritten is True
    assert "simulated model fit failure" in status.message


def test_overwrite_mode_corrupt_existing_record_raises_before_execute(
    tmp_path,
):
    planned = _make_planned(_prior_free_config())
    # Pre-stage corrupt JSON at the expected record path.
    full = tmp_path / planned.record_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("{ this is not a record }", encoding="utf-8")

    execute_calls: list[str] = []

    def fake_execute(planned, **kwargs):
        execute_calls.append("called")
        return _success_execution_result(planned)

    persist_calls: list[str] = []

    def fake_persist(result, record_path, *, base_dir):
        persist_calls.append("called")

    with pytest.raises(RuntimeError, match="corrupt"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="overwrite",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake_execute,
            persist_execution_result_fn=fake_persist,
        )
    assert execute_calls == []
    assert persist_calls == []


def test_overwrite_mode_incompatible_existing_record_raises_before_execute(
    tmp_path,
):
    planned_a = _make_planned(_prior_free_config(seed_value=401))
    planned_b = _make_planned(_prior_free_config(seed_value=402))
    # Persist planned_a's record at planned_b's path so the
    # compatibility check on planned_b detects a mismatch.
    from experiments.main_study.records import record_to_json

    full = tmp_path / planned_b.record_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        record_to_json(_success_record(planned_a)), encoding="utf-8"
    )

    execute_calls: list[str] = []

    def fake_execute(planned, **kwargs):
        execute_calls.append("called")
        return _success_execution_result(planned)

    persist_calls: list[str] = []

    def fake_persist(result, record_path, *, base_dir):
        persist_calls.append("called")

    with pytest.raises(ValueError):
        run_main_study(
            [planned_b],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="overwrite",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake_execute,
            persist_execution_result_fn=fake_persist,
        )
    assert execute_calls == []
    assert persist_calls == []


# ===========================================================================
# E. Execution-result classification
# ===========================================================================


def test_classify_success_computed(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_success_computed == 1
    assert summary.n_executed == 1


def test_classify_success_metric_unavailable(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_success_unavailable_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_success_metric_unavailable == 1
    assert summary.n_executed == 1
    assert (
        summary.per_workload_status[0].final_status
        == "success_metric_unavailable"
    )


def test_classify_model_fit_failure(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_model_fit_failure_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert summary.n_model_fit_failure == 1
    assert summary.n_executed == 1
    status = summary.per_workload_status[0]
    assert status.final_status == "model_fit_failure"
    assert "simulated model fit failure" in status.message


# ===========================================================================
# F. Infrastructure failure policy
# ===========================================================================


class _SyntheticInfraError(RuntimeError):
    pass


def test_infrastructure_failure_writes_record_and_reraises(tmp_path):
    planned = _make_planned(_prior_free_config())

    def boom(_planned):
        raise _SyntheticInfraError("disk on fire")

    fake, _ = _make_fake_execute(side_effect=boom)
    with pytest.raises(_SyntheticInfraError, match="disk on fire"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )
    # Best-effort: an infrastructure-failure record was written.
    loaded = load_existing_record(
        planned.record_path, base_dir=tmp_path
    )
    assert loaded is not None
    assert loaded.fit_status == "infrastructure_failure_during_fit"
    assert loaded.failure_kind == "infrastructure"
    assert "_SyntheticInfraError" in loaded.failure_message


def test_infrastructure_failure_secondary_persist_does_not_mask(tmp_path):
    planned = _make_planned(_prior_free_config())

    def boom(_planned):
        raise _SyntheticInfraError("primary failure")

    fake_execute, _ = _make_fake_execute(side_effect=boom)

    def failing_persist_record(*args, **kwargs):
        raise OSError("secondary persistence failure")

    with pytest.raises(_SyntheticInfraError, match="primary failure"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake_execute,
            persist_record_fn=failing_persist_record,
        )
    # Record was not written because both attempts failed.
    assert not (tmp_path / planned.record_path).exists()


def test_model_fit_failure_leak_is_contract_violation(tmp_path):
    planned = _make_planned(_prior_free_config())

    def leak(_planned):
        raise ModelFitFailure("executor must catch this")

    fake, _ = _make_fake_execute(side_effect=leak)
    with pytest.raises(RuntimeError, match="leaked a ModelFitFailure"):
        run_main_study(
            [planned],
            base_dir=tmp_path,
            data_loader=_sentinel(),
            fit_backend=_sentinel(),
            metric_backend=_sentinel(),
            mode="raise",
            code_version=None,
            generated_at_utc_factory=_constant_factory(),
            n_nodes_for_failure_record=_N_NODES,
            execute_fn=fake,
        )


# ===========================================================================
# G. Timestamp and code_version threading
# ===========================================================================


def test_generated_at_factory_called_per_run(tmp_path):
    p1 = _make_planned(_prior_free_config(seed_value=401))
    p2 = _make_planned(_prior_free_config(seed_value=402))
    timestamps = iter(
        ["2026-05-24T12:00:00Z", "2026-05-24T12:01:00Z"]
    )
    factory_calls: list[int] = []

    def factory() -> str:
        factory_calls.append(1)
        return next(timestamps)

    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    run_main_study(
        [p1, p2],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=factory,
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert len(factory_calls) == 2
    assert [c["generated_at_utc"] for c in calls] == [
        "2026-05-24T12:00:00Z",
        "2026-05-24T12:01:00Z",
    ]


def test_code_version_threaded_into_executor_call(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, calls = _make_fake_execute(
        result_factory=_success_execution_result
    )
    run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version="deadbeef",
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert calls[0]["code_version"] == "deadbeef"


# ===========================================================================
# H. Logger behaviour
# ===========================================================================


def test_logger_none_does_not_crash(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    summary = run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        logger=None,
        execute_fn=fake,
    )
    assert summary.n_planned == 1


def test_logger_receives_messages(tmp_path):
    planned = _make_planned(_prior_free_config())
    captured: list[tuple[int, str]] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append((record.levelno, record.getMessage()))

    handler = _Capture()
    test_logger = logging.getLogger("test_runner.main_study")
    test_logger.handlers = []
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        logger=test_logger,
        execute_fn=fake,
    )
    assert captured, "logger should have received at least one record"
    assert any("executed" in msg for _, msg in captured)


# ===========================================================================
# I. Persistence integration
# ===========================================================================


def test_normal_flow_writes_record_and_artefacts(tmp_path):
    planned = _make_planned(_prior_free_config())
    fake, _ = _make_fake_execute(
        result_factory=_success_execution_result
    )
    run_main_study(
        [planned],
        base_dir=tmp_path,
        data_loader=_sentinel(),
        fit_backend=_sentinel(),
        metric_backend=_sentinel(),
        mode="raise",
        code_version=None,
        generated_at_utc_factory=_constant_factory(),
        n_nodes_for_failure_record=_N_NODES,
        execute_fn=fake,
    )
    assert (tmp_path / planned.record_path).exists()
    for art_path in planned.artefact_paths.values():
        assert (tmp_path / art_path).exists()


# ===========================================================================
# J. Scope / imports
# ===========================================================================


_RUNNER_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "dataclasses",
    "logging",
    "math",
    "time",
    "pathlib",
    "typing",
    "experiments.main_study.executor",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "experiments.main_study.workloads",
})


_RUNNER_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.backends",
    "experiments.main_study.calibration_lambda_prior",
    "experiments.main_study.priors",
    "experiments.main_study.paths",
    "experiments.main_study.schema",
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


def test_runner_module_imports_are_allowlisted():
    src = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _RUNNER_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _RUNNER_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"runner.py import {mod!r} is not in the allowlist "
            f"{sorted(_RUNNER_ALLOWED_PREFIXES)}."
        )


def test_runner_module_does_not_import_forbidden_packages():
    src = Path(runner_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _RUNNER_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"runner.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_run_modes_and_final_statuses_are_tuples_of_strings():
    assert isinstance(RUN_MODES, tuple)
    assert isinstance(FINAL_STATUSES, tuple)
    assert all(isinstance(v, str) for v in RUN_MODES)
    assert all(isinstance(v, str) for v in FINAL_STATUSES)
    assert set(RUN_MODES) == {"raise", "skip", "overwrite"}
    assert set(FINAL_STATUSES) == {
        "success_computed",
        "success_metric_unavailable",
        "model_fit_failure",
        "skipped",
        "infrastructure_failure",
    }


def test_final_statuses_does_not_contain_overwritten():
    assert "overwritten" not in FINAL_STATUSES
