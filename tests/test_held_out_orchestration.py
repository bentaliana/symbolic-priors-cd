"""Tests for the held-out evaluation orchestration entry point.

These tests exercise ``run_held_out_evaluation`` end to end with an
injected fit_runner and a synthetic calibration handoff artefact
under ``tmp_path``. No test runs a real model fit, calls
``pipeline.run_single_fit``, imports a wrapper module, or writes to
the live results tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.held_out import (
    DCDI_MAIN_FIT_RNG,
    EXPECTED_MAIN_JOB_COUNT,
    EXPECTED_SENSITIVITY_JOB_COUNT,
    EXPECTED_TOTAL_JOB_COUNT,
    HELDOUT_EVALUATION_FILENAME,
    HELDOUT_RUN_DIRECTORY,
    HELDOUT_SCM_SEEDS,
    HeldoutJob,
    MAIN_JOB_KIND,
    RECORDS_DIRECTORY_NAME,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    _HeldoutInfrastructureError,
    _record_filename_for_job,
    enumerate_heldout_workload,
    heldout_evaluation_path,
    heldout_records_dir_path,
    heldout_run_dir_path,
    run_held_out_evaluation,
)
from experiments.selection_study.held_out_artefact import (
    HELDOUT_EVALUATION_ARTEFACT_TYPE,
    validate_heldout_evaluation_artefact,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    FIT_RNG_POLICY_REF,
    HASH_PREFIX_LENGTH,
    INTERVENTION_POLICY_REF,
    MODELS,
    MODEL_SELECTION_DIRECTORY,
    SELECTED_CONFIGURATIONS_FILENAME,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
    write_selected_configurations,
)


# ---------------------------------------------------------------------------
# Synthetic calibration artefact
# ---------------------------------------------------------------------------


_DAGMA_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25)
_DCDI_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)


def _grid_for(model: str) -> tuple[float, ...]:
    return _DAGMA_GRID if model == "dagma" else _DCDI_GRID


def _hyperparameter_name_for(model: str) -> str:
    return "lambda1" if model == "dagma" else "reg_coeff"


def _candidate_hash_for(
    model: str, condition: str, hyper_value: float
) -> str:
    seed_str = (
        f"heldout-orchestration-test|{model}|{condition}|{hyper_value!r}"
    )
    return hashlib.sha256(seed_str.encode("utf-8")).hexdigest()


def _synthetic_threshold_metrics(model: str) -> list[dict[str, Any]]:
    thresholds = (0.2, 0.3, 0.4) if model == "dagma" else (0.4, 0.5, 0.6)
    return [
        {
            "threshold": float(value),
            "shd": 0,
            "sid": 0,
            "mmd_primary": None,
        }
        for value in thresholds
    ]


def _synthetic_mmd_by_intervention() -> list[dict[str, Any]]:
    return [
        {
            "intervention_target": target,
            "intervention_value": value,
            "mmd_primary": 0.001,
        }
        for target in range(10)
        for value in (-2.0, 2.0)
    ]


def _synthetic_bandwidth_summaries() -> dict[str, float]:
    return {
        f"do_X{target}_{sign}": 50.0 + target
        for target in range(10)
        for sign in ("neg2", "pos2")
    }


def _make_calibration_record(
    *,
    model: str,
    condition: str,
    hyper_value: float,
    seed_value: int,
) -> dict[str, Any]:
    config_hash_full = _candidate_hash_for(model, condition, hyper_value)
    return {
        "model": model,
        "condition": condition,
        "configuration_hash_full": config_hash_full,
        "configuration_hash_prefix": config_hash_full[:HASH_PREFIX_LENGTH],
        "hyperparameters": {
            _hyperparameter_name_for(model): hyper_value
        },
        "seed_value": seed_value,
        "shd": 0,
        "sid": 0,
        "mmd_primary": 0.001 + 0.0001 * hyper_value,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "runtime_seconds": 0.5,
        "n_iterations": None,
        "threshold_metrics": _synthetic_threshold_metrics(model),
        "mmd_by_intervention": _synthetic_mmd_by_intervention(),
        "bandwidth_summaries": _synthetic_bandwidth_summaries(),
        "run_id": (
            f"{model}__{condition}__calibration__"
            f"seed{seed_value - 201}__cfg{config_hash_full}"
        ),
    }


def _write_calibration_artefact(tmp_path: Path) -> Path:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model in MODELS:
            for hyper_value in _grid_for(model):
                for seed_value in CALIBRATION_SEEDS:
                    records.append(
                        _make_calibration_record(
                            model=model,
                            condition=condition,
                            hyper_value=hyper_value,
                            seed_value=seed_value,
                        )
                    )
    rank_output = rank_calibration_records(records)
    full_hash = hashlib.sha256(
        b"synthetic-heldout-orchestration-parent"
    ).hexdigest()
    artefact = {
        "schema_version": 1,
        "artefact_type": "calibration_selected_configurations",
        "decision_scope": "within_model_configuration_selection",
        "base_model_decision_made": False,
        "selected_configuration_semantics": "rank_1_within_model_and_condition",
        "calibration_run_hash_prefix": full_hash[:HASH_PREFIX_LENGTH],
        "calibration_run_hash_full": full_hash,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "seed_population": "calibration",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "selections": rank_output["selections"],
        "candidate_ranking": rank_output["candidate_ranking"],
        "generated_at_utc": "2026-05-22T20:00:00Z",
    }
    path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    write_selected_configurations(artefact, path)
    return path


# ---------------------------------------------------------------------------
# Fit-runner builders
# ---------------------------------------------------------------------------


def _job_record(
    job: HeldoutJob,
    *,
    sid: Any | None = None,
    shd: Any | None = None,
    mmd_primary: Any | None = None,
    runtime_seconds: float = 1.0,
    training_status: str = "converged",
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
) -> dict[str, Any]:
    """Build a fit_runner return value coherent with the job identity."""
    if sid is None:
        sid = float(int(job.scm_seed) - 301)
    if shd is None:
        shd = float((int(job.scm_seed) - 301) % 3)
    if mmd_primary is None:
        mmd_primary = 0.005 + 0.001 * (int(job.scm_seed) - 301)
    return {
        "job_kind": job.job_kind,
        "model": job.model,
        "condition": job.condition,
        "configuration_hash_full": job.configuration_hash_full,
        "configuration_hash_prefix": job.configuration_hash_prefix,
        "hyperparameters": dict(job.hyperparameters),
        "scm_seed": int(job.scm_seed),
        "fit_rng": job.fit_rng,
        "sid": sid,
        "shd": shd,
        "mmd_primary": mmd_primary,
        "runtime_seconds": runtime_seconds,
        "graph_status": graph_status,
        "sampler_status": sampler_status,
        "training_status": training_status,
        "n_iterations": None,
        "run_id": (
            f"{job.model}__{job.condition}__held_out__"
            f"scm{job.scm_seed}__fitrng{job.fit_rng}__"
            f"cfg{job.configuration_hash_full}"
        ),
        "calibration_run_hash_prefix": job.calibration_run_hash_prefix,
    }


def _recording_runner() -> tuple[
    Callable[[HeldoutJob], dict[str, Any]], list[HeldoutJob]
]:
    """Return a fit_runner that records every job it processes."""
    seen: list[HeldoutJob] = []

    def runner(job: HeldoutJob) -> dict[str, Any]:
        seen.append(job)
        return _job_record(job)

    return runner, seen


def _fit_runner_with_exception_on(
    target_filter: Callable[[HeldoutJob], bool],
    *,
    exc: Exception,
) -> tuple[Callable[[HeldoutJob], dict[str, Any]], list[HeldoutJob]]:
    """Return a fit_runner that raises ``exc`` on jobs matching the filter."""
    seen: list[HeldoutJob] = []

    def runner(job: HeldoutJob) -> dict[str, Any]:
        seen.append(job)
        if target_filter(job):
            raise exc
        return _job_record(job)

    return runner, seen


_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "winner",
    "model_winner",
    "base_model_winner",
    "recommended_model",
    "final_decision",
    "DAGMA wins",
    "DCDI wins",
)


def _assert_no_forbidden_language(text: str) -> None:
    lower = text.lower()
    for phrase in _FORBIDDEN_PHRASES:
        assert phrase.lower() not in lower, (
            f"forbidden phrase {phrase!r} appeared in orchestration "
            "output text"
        )


def _planned_run_directories(
    artefact_path: Path, results_root: Path
) -> tuple[Path, Path, Path]:
    """Return (run_dir, records_dir, artefact_path) for a given input."""
    workload = enumerate_heldout_workload(artefact_path)
    # Re-run the hash computation via the public helper; the
    # orchestrator uses the same path under the hood.
    from experiments.selection_study.held_out import (
        _heldout_run_hash_inputs_from_workload,
        compute_heldout_run_hash_full,
    )

    full_hash = compute_heldout_run_hash_full(
        **_heldout_run_hash_inputs_from_workload(workload)
    )
    hash12 = full_hash[:HASH_PREFIX_LENGTH]
    run_dir = heldout_run_dir_path(
        heldout_run_hash12=hash12, results_root=results_root
    )
    records_dir = heldout_records_dir_path(
        heldout_run_hash12=hash12, results_root=results_root
    )
    artefact_out = heldout_evaluation_path(
        heldout_run_hash12=hash12, results_root=results_root
    )
    return run_dir, records_dir, artefact_out


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_orchestration_calls_fit_runner_exactly_25_times(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    runner, seen = _recording_runner()
    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )

    assert len(seen) == EXPECTED_TOTAL_JOB_COUNT == 25
    main_count = sum(1 for job in seen if job.job_kind == MAIN_JOB_KIND)
    sensitivity_count = sum(
        1 for job in seen if job.job_kind == SENSITIVITY_JOB_KIND
    )
    assert main_count == EXPECTED_MAIN_JOB_COUNT == 20
    assert sensitivity_count == EXPECTED_SENSITIVITY_JOB_COUNT == 5
    assert artefact_out.is_file()


def test_record_filenames_follow_main_and_sensitivity_naming(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    runner, _seen = _recording_runner()
    run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )

    _, records_dir, _ = _planned_run_directories(
        artefact_path, results_root
    )
    record_files = sorted(records_dir.glob("*.json"))
    assert len(record_files) == EXPECTED_TOTAL_JOB_COUNT

    main_files = [p for p in record_files if "_fitrng" not in p.name]
    sensitivity_files = [p for p in record_files if "_fitrng" in p.name]
    assert len(main_files) == EXPECTED_MAIN_JOB_COUNT
    assert len(sensitivity_files) == EXPECTED_SENSITIVITY_JOB_COUNT

    for path in sensitivity_files:
        # Sensitivity filenames always carry a fit_rng suffix.
        stem = path.stem
        # Every sensitivity file must mention one of the sensitivity
        # fit_rng values 43..47.
        assert any(
            stem.endswith(f"_fitrng{value}") for value in SENSITIVITY_FIT_RNGS
        ), stem


def test_main_and_sensitivity_records_do_not_collide_at_dcdi_centred_seed_301(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    runner, _seen = _recording_runner()
    run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )

    _, records_dir, _ = _planned_run_directories(
        artefact_path, results_root
    )
    # The DCDI/centred_only/seed=301 main record and the 5
    # sensitivity records all carry seed=301 and the same prefix.
    # Filenames must remain unique because sensitivity files append
    # the fit_rng suffix.
    workload = enumerate_heldout_workload(artefact_path)
    main_dcdi_301 = next(
        job
        for job in workload.main_jobs
        if job.job_kind == MAIN_JOB_KIND
        and job.model == SENSITIVITY_MODEL
        and job.condition == SENSITIVITY_CONDITION
        and job.scm_seed == SENSITIVITY_SCM_SEED
    )
    main_path = records_dir / _record_filename_for_job(main_dcdi_301)
    assert main_path.is_file()
    assert "_fitrng" not in main_path.name

    sensitivity_paths = [
        records_dir / _record_filename_for_job(job)
        for job in workload.sensitivity_jobs
    ]
    for path in sensitivity_paths:
        assert path.is_file()
        assert path != main_path


def test_heldout_evaluation_artefact_is_written_and_validates(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()

    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )

    assert artefact_out.name == HELDOUT_EVALUATION_FILENAME
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(payload)
    assert payload["artefact_type"] == HELDOUT_EVALUATION_ARTEFACT_TYPE


def test_main_evaluation_uses_only_main_records(tmp_path: Path) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()

    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    cells = payload["main_evaluation"]["cells"]
    for condition in CONDITIONS:
        for model in MODELS:
            cell = cells[condition][model]
            assert len(cell["per_seed_records"]) == 5
            for record in cell["per_seed_records"]:
                # Sensitivity fit_rngs must not enter main cells.
                assert record["fit_rng"] not in set(SENSITIVITY_FIT_RNGS)


def test_sensitivity_addendum_uses_only_sensitivity_records(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()

    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    addendum = payload["fit_rng_sensitivity_addendum"]
    assert len(addendum["per_fit_records"]) == EXPECTED_SENSITIVITY_JOB_COUNT
    seen_fit_rngs = sorted(
        int(record["fit_rng"]) for record in addendum["per_fit_records"]
    )
    assert seen_fit_rngs == list(SENSITIVITY_FIT_RNGS)


def test_sensitivity_addendum_includes_seed_301_main_comparison(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()

    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    diagnostic = payload["fit_rng_sensitivity_addendum"][
        "diagnostic_summary"
    ]
    # The synthetic fit_runner returns sid=0, shd=0, mmd_primary=0.005
    # for SCM seed 301 in the DCDI/centred_only main cell.
    assert diagnostic["main_evaluation_sid_at_seed_301"] == 0.0
    assert diagnostic["main_evaluation_shd_at_seed_301"] == 0.0
    assert diagnostic["main_evaluation_mmd_primary_at_seed_301"] == 0.005


# ---------------------------------------------------------------------------
# Force semantics
# ---------------------------------------------------------------------------


def test_force_false_refuses_existing_artefact(tmp_path: Path) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()
    run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    # Second call without force must refuse to overwrite the existing
    # heldout_evaluation.json that the first call wrote.
    second_runner, _seen_b = _recording_runner()
    with pytest.raises(FileExistsError, match="held-out evaluation"):
        run_held_out_evaluation(
            artefact_path, results_root, fit_runner=second_runner
        )


def test_force_true_overwrites_existing_artefact(tmp_path: Path) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    runner_a, _ = _recording_runner()
    out_a = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner_a
    )
    payload_a = json.loads(out_a.read_text(encoding="utf-8"))

    runner_b, _ = _recording_runner()
    out_b = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner_b, force=True
    )
    payload_b = json.loads(out_b.read_text(encoding="utf-8"))

    assert out_a == out_b
    # The hashes are identical because the workload identity is the
    # same; the generated_at timestamp may differ slightly. We assert
    # only that the second write produced a fresh validating artefact.
    validate_heldout_evaluation_artefact(payload_b)
    assert (
        payload_b["heldout_run_hash_full"]
        == payload_a["heldout_run_hash_full"]
    )


# ---------------------------------------------------------------------------
# fit_runner=None deferral
# ---------------------------------------------------------------------------


def test_fit_runner_none_raises_not_implemented_error(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    with pytest.raises(NotImplementedError, match="fit_runner"):
        run_held_out_evaluation(artefact_path, tmp_path / "results")


# ---------------------------------------------------------------------------
# Identity mismatch
# ---------------------------------------------------------------------------


def test_identity_mismatch_raises_infrastructure_error(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    def runner(job: HeldoutJob) -> dict[str, Any]:
        record = _job_record(job)
        record["scm_seed"] = 999  # intentional identity mismatch
        return record

    with pytest.raises(_HeldoutInfrastructureError, match="scm_seed"):
        run_held_out_evaluation(
            artefact_path, results_root, fit_runner=runner
        )

    _, _, planned_artefact_path = _planned_run_directories(
        artefact_path, results_root
    )
    assert not planned_artefact_path.exists()


# ---------------------------------------------------------------------------
# Failure-handling
# ---------------------------------------------------------------------------


def test_ordinary_exception_becomes_degenerate_record_and_continues(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    # Make the DAGMA centred_only seed=302 main job raise an
    # ordinary exception. The orchestrator must persist a degenerate
    # record and continue.
    def _target(job: HeldoutJob) -> bool:
        return (
            job.job_kind == MAIN_JOB_KIND
            and job.model == "dagma"
            and job.condition == "centred_only"
            and job.scm_seed == 302
        )

    runner, _seen = _fit_runner_with_exception_on(
        _target, exc=RuntimeError("simulated model-fit blow-up")
    )

    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    assert artefact_out.is_file()
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(payload)

    cell = payload["main_evaluation"]["cells"]["centred_only"]["dagma"]
    seed_302_record = next(
        record
        for record in cell["per_seed_records"]
        if record["seed_value"] == 302
    )
    # Degenerate-record markers project through to the artefact's
    # per_seed_records via the artefact builder.
    assert seed_302_record["training_status"] == "failed"
    assert seed_302_record["graph_status"] == "failed"
    assert seed_302_record["sampler_status"] == "failed"
    assert seed_302_record["sid"] is None
    assert seed_302_record["shd"] is None
    assert seed_302_record["mmd_primary"] is None

    # The cell's aggregate SID summary reports 4 finite + 1 non-finite.
    sid_summary = cell["aggregate_metrics"]["sid"]
    assert sid_summary["finite_count"] == 4
    assert sid_summary["non_finite_count"] == 1
    assert (
        cell["aggregate_metrics"]["has_non_finite_seed_metric"] is True
    )

    # On-disk record file carries the failure_type / failure_message
    # diagnostic that the orchestrator captured.
    _, records_dir, _ = _planned_run_directories(
        artefact_path, results_root
    )
    workload = enumerate_heldout_workload(artefact_path)
    target_job = next(
        job for job in workload.main_jobs if _target(job)
    )
    target_path = records_dir / _record_filename_for_job(target_job)
    on_disk = json.loads(target_path.read_text(encoding="utf-8"))
    assert on_disk["failure_type"] == "RuntimeError"
    assert "simulated model-fit blow-up" in on_disk["failure_message"]


def test_file_exists_error_propagates_as_infrastructure_failure(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    def _target(job: HeldoutJob) -> bool:
        return (
            job.job_kind == MAIN_JOB_KIND
            and job.model == "dagma"
            and job.condition == "centred_only"
            and job.scm_seed == 303
        )

    runner, _seen = _fit_runner_with_exception_on(
        _target,
        exc=FileExistsError(
            "stale per-run directory left by an earlier attempt"
        ),
    )

    with pytest.raises(
        _HeldoutInfrastructureError, match="FileExistsError"
    ):
        run_held_out_evaluation(
            artefact_path, results_root, fit_runner=runner
        )

    _, _, planned_artefact_path = _planned_run_directories(
        artefact_path, results_root
    )
    assert not planned_artefact_path.exists()


def test_records_before_infrastructure_failure_remain_on_disk(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    # Targets DAGMA centred_only seed 303 (the third main job in
    # canonical iteration order: condition=centred_only,
    # model=dagma, scm_seed in 301..305). The two earlier records
    # should be persisted before the FileExistsError fires.
    def _target(job: HeldoutJob) -> bool:
        return (
            job.job_kind == MAIN_JOB_KIND
            and job.model == "dagma"
            and job.condition == "centred_only"
            and job.scm_seed == 303
        )

    runner, seen = _fit_runner_with_exception_on(
        _target, exc=FileExistsError("stale residue")
    )

    with pytest.raises(_HeldoutInfrastructureError):
        run_held_out_evaluation(
            artefact_path, results_root, fit_runner=runner
        )

    _, records_dir, planned_artefact_path = _planned_run_directories(
        artefact_path, results_root
    )
    # The orchestrator iterates condition x model x seed in order.
    # The first DAGMA centred_only jobs (seeds 301 and 302) must have
    # written their records before the seed 303 failure.
    workload = enumerate_heldout_workload(artefact_path)
    persisted_jobs = [
        job
        for job in workload.main_jobs
        if job.model == "dagma"
        and job.condition == "centred_only"
        and job.scm_seed in {301, 302}
    ]
    assert len(persisted_jobs) == 2
    for job in persisted_jobs:
        path = records_dir / _record_filename_for_job(job)
        assert path.is_file()
    assert not planned_artefact_path.exists()


# ---------------------------------------------------------------------------
# Side-effect invariants
# ---------------------------------------------------------------------------


def test_pipeline_run_single_fit_is_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.selection_study.pipeline as pipeline

    sentinel = {"called": False}

    def _poison(*args: object, **kwargs: object) -> None:
        sentinel["called"] = True
        raise AssertionError(
            "held-out orchestration must not call "
            "pipeline.run_single_fit"
        )

    monkeypatch.setattr(pipeline, "run_single_fit", _poison)

    artefact_path = _write_calibration_artefact(tmp_path)
    runner, _seen = _recording_runner()
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        fit_runner=runner,
    )
    assert sentinel["called"] is False


def test_held_out_module_does_not_import_wrapper_modules(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    runner, _seen = _recording_runner()
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        fit_runner=runner,
    )
    import experiments.selection_study.held_out as held_out_module

    forbidden_globals = {
        name
        for name in vars(held_out_module).keys()
        if name.startswith(("dagma", "dcdi"))
    }
    assert forbidden_globals == set(), forbidden_globals


def test_output_text_contains_no_forbidden_language(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()
    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )
    text = artefact_out.read_text(encoding="utf-8")
    _assert_no_forbidden_language(text)


def test_run_dir_is_created_under_results_root(tmp_path: Path) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"
    runner, _seen = _recording_runner()
    artefact_out = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=runner
    )

    run_dir, records_dir, _ = _planned_run_directories(
        artefact_path, results_root
    )
    expected_prefix = (
        results_root
        / MODEL_SELECTION_DIRECTORY
        / HELDOUT_RUN_DIRECTORY
    )
    assert run_dir.is_dir()
    assert records_dir.is_dir()
    assert str(run_dir).startswith(str(expected_prefix))
    assert artefact_out.parent == run_dir
