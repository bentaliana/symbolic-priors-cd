"""Tests for the held-out evaluation preflight.

These tests exercise ``enumerate_heldout_workload`` and
``preflight_heldout_evaluation`` against synthetic
``selected_configurations.json`` artefacts built under ``tmp_path``.
No test runs a fit, writes a record, writes a held-out artefact, or
calls ``pipeline.run_single_fit``; the preflight is required to be
side-effect free.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.held_out import (
    DAGMA_MAIN_FIT_RNG,
    DCDI_MAIN_FIT_RNG,
    EXPECTED_MAIN_JOB_COUNT,
    EXPECTED_SENSITIVITY_JOB_COUNT,
    EXPECTED_TOTAL_JOB_COUNT,
    HELDOUT_EVALUATION_FILENAME,
    HELDOUT_FIT_RNG_SENSITIVITY_REF,
    HELDOUT_PREFLIGHT_REPORT_ARTEFACT_TYPE,
    HELDOUT_PREFLIGHT_SCHEMA_VERSION,
    HELDOUT_RUN_DIRECTORY,
    HELDOUT_SCM_SEEDS,
    MAIN_JOB_KIND,
    RECORDS_DIRECTORY_NAME,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    compute_heldout_run_hash12,
    compute_heldout_run_hash_full,
    enumerate_heldout_workload,
    preflight_heldout_evaluation,
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
# Synthetic artefact builder
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
        f"heldout-preflight-test|{model}|{condition}|{hyper_value!r}"
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
    rows: list[dict[str, Any]] = []
    for target in range(10):
        for value in (-2.0, 2.0):
            rows.append(
                {
                    "intervention_target": target,
                    "intervention_value": value,
                    "mmd_primary": 0.001,
                }
            )
    return rows


def _synthetic_bandwidth_summaries() -> dict[str, float]:
    summaries: dict[str, float] = {}
    for target in range(10):
        for sign in ("neg2", "pos2"):
            summaries[f"do_X{target}_{sign}"] = 50.0 + target
    return summaries


def _make_record(
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


def _make_all_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model in MODELS:
            for hyper_value in _grid_for(model):
                for seed_value in CALIBRATION_SEEDS:
                    records.append(
                        _make_record(
                            model=model,
                            condition=condition,
                            hyper_value=hyper_value,
                            seed_value=seed_value,
                        )
                    )
    return records


def _build_artefact(records: list[dict[str, Any]]) -> dict[str, Any]:
    rank_output = rank_calibration_records(records)
    full_hash = hashlib.sha256(
        b"synthetic-heldout-preflight-test-artefact"
    ).hexdigest()
    return {
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


def _write_synthetic_artefact(tmp_path: Path) -> Path:
    """Write a valid synthetic artefact to tmp_path/selected_configurations.json."""
    records = _make_all_records()
    artefact = _build_artefact(records)
    path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    write_selected_configurations(artefact, path)
    return path


def _write_degenerate_synthetic_artefact(tmp_path: Path) -> Path:
    """Write an artefact whose centred_only/dagma selection is degenerate."""
    records = _make_all_records()
    artefact = _build_artefact(records)
    artefact["selections"]["centred_only"]["dagma"]["degeneracy_flag"] = True
    path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    write_selected_configurations(artefact, path)
    return path


def _write_nonfinite_metric_synthetic_artefact(tmp_path: Path) -> Path:
    """Write an artefact whose centred_only/dagma metric is None."""
    records = _make_all_records()
    artefact = _build_artefact(records)
    artefact["selections"]["centred_only"]["dagma"]["selection_metrics"][
        "mean_sid"
    ] = None
    path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    write_selected_configurations(artefact, path)
    return path


def _snapshot_directory(directory: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    if not directory.exists():
        return snapshot
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(directory))] = path.read_bytes()
    return snapshot


class _FixedClock:
    def __init__(
        self,
        value: datetime = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    ) -> None:
        self._value = value

    def __call__(self) -> datetime:
        return self._value


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
            f"forbidden language {phrase!r} appeared in held-out "
            "preflight text output"
        )


# ---------------------------------------------------------------------------
# Workload enumeration tests
# ---------------------------------------------------------------------------


def test_enumerate_returns_25_jobs(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    assert len(workload.main_jobs) == 20
    assert len(workload.sensitivity_jobs) == 5
    assert workload.total_job_count == 25
    assert EXPECTED_MAIN_JOB_COUNT == 20
    assert EXPECTED_SENSITIVITY_JOB_COUNT == 5
    assert EXPECTED_TOTAL_JOB_COUNT == 25


def test_main_seeds_are_held_out_only(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    main_seeds = sorted({job.scm_seed for job in workload.main_jobs})
    assert main_seeds == [301, 302, 303, 304, 305]
    for job in workload.main_jobs:
        assert job.scm_seed not in set(CALIBRATION_SEEDS)


def test_no_calibration_seed_in_any_job(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    all_seeds = {job.scm_seed for job in workload.main_jobs}
    all_seeds |= {job.scm_seed for job in workload.sensitivity_jobs}
    for cal_seed in CALIBRATION_SEEDS:
        assert cal_seed not in all_seeds


def test_sensitivity_jobs_target_dcdi_centred_only_seed_301(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    for job in workload.sensitivity_jobs:
        assert job.model == SENSITIVITY_MODEL == "dcdi"
        assert job.condition == SENSITIVITY_CONDITION == "centred_only"
        assert job.scm_seed == SENSITIVITY_SCM_SEED == 301
        assert job.fit_rng in SENSITIVITY_FIT_RNGS
        assert job.job_kind == SENSITIVITY_JOB_KIND


def test_sensitivity_fit_rngs_are_43_to_47_in_order(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    fit_rngs = [job.fit_rng for job in workload.sensitivity_jobs]
    assert fit_rngs == [43, 44, 45, 46, 47]


def test_main_dcdi_fit_rng_is_42(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    dcdi_main = [job for job in workload.main_jobs if job.model == "dcdi"]
    assert dcdi_main
    for job in dcdi_main:
        assert job.fit_rng == DCDI_MAIN_FIT_RNG == 42


def test_main_dagma_fit_rng_is_none(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    dagma_main = [job for job in workload.main_jobs if job.model == "dagma"]
    assert dagma_main
    for job in dagma_main:
        assert job.fit_rng is DAGMA_MAIN_FIT_RNG
        assert job.fit_rng is None


def test_main_and_sensitivity_are_structurally_separate(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    for job in workload.main_jobs:
        assert job.job_kind == MAIN_JOB_KIND
    for job in workload.sensitivity_jobs:
        assert job.job_kind == SENSITIVITY_JOB_KIND

    main_identities = {
        (
            job.job_kind,
            job.model,
            job.condition,
            job.configuration_hash_full,
            job.scm_seed,
            job.fit_rng,
        )
        for job in workload.main_jobs
    }
    sensitivity_identities = {
        (
            job.job_kind,
            job.model,
            job.condition,
            job.configuration_hash_full,
            job.scm_seed,
            job.fit_rng,
        )
        for job in workload.sensitivity_jobs
    }
    assert main_identities.isdisjoint(sensitivity_identities)


def test_enumeration_is_deterministic(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    first = enumerate_heldout_workload(artefact_path)
    second = enumerate_heldout_workload(artefact_path)
    assert first.main_jobs == second.main_jobs
    assert first.sensitivity_jobs == second.sensitivity_jobs
    assert (
        first.calibration_run_hash_full == second.calibration_run_hash_full
    )


def test_main_job_count_per_cell_is_five(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)
    counts: dict[tuple[str, str], int] = {}
    for job in workload.main_jobs:
        key = (job.condition, job.model)
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 4
    for cell, count in counts.items():
        assert count == 5, cell


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_degeneracy_flag_true_is_rejected(tmp_path: Path) -> None:
    artefact_path = _write_degenerate_synthetic_artefact(tmp_path)
    with pytest.raises(ValueError, match="degeneracy_flag"):
        enumerate_heldout_workload(artefact_path)


def test_non_finite_selection_metric_is_rejected(tmp_path: Path) -> None:
    artefact_path = _write_nonfinite_metric_synthetic_artefact(tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        enumerate_heldout_workload(artefact_path)


def test_missing_artefact_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="selected_configurations"):
        enumerate_heldout_workload(missing_path)


def test_invalid_json_is_rejected_with_self_contained_error(
    tmp_path: Path,
) -> None:
    bad_path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    bad_path.write_text("not-json-content", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        enumerate_heldout_workload(bad_path)
    message = str(excinfo.value)
    assert "selected_configurations artefact" in message
    assert "not valid JSON" in message


def test_top_level_non_object_is_rejected(tmp_path: Path) -> None:
    bad_path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    bad_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        enumerate_heldout_workload(bad_path)


# ---------------------------------------------------------------------------
# Hash tests
# ---------------------------------------------------------------------------


def _hash_inputs_default(
    parent_full_hash: str,
) -> dict[str, Any]:
    return {
        "parent_calibration_run_hash_full": parent_full_hash,
        "selected_configuration_hashes_full": [
            "0" * 64,
            "1" * 64,
            "2" * 64,
            "3" * 64,
        ],
        "main_heldout_seeds": list(HELDOUT_SCM_SEEDS),
        "sensitivity_spec": {
            "model": SENSITIVITY_MODEL,
            "condition": SENSITIVITY_CONDITION,
            "scm_seed": SENSITIVITY_SCM_SEED,
            "fit_rngs": list(SENSITIVITY_FIT_RNGS),
        },
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "heldout_fit_rng_sensitivity_ref": HELDOUT_FIT_RNG_SENSITIVITY_REF,
    }


def test_heldout_hash_is_deterministic() -> None:
    parent = "a" * 64
    inputs = _hash_inputs_default(parent)
    first_full = compute_heldout_run_hash_full(**inputs)
    second_full = compute_heldout_run_hash_full(**inputs)
    assert first_full == second_full
    assert len(first_full) == 64
    assert compute_heldout_run_hash12(**inputs) == first_full[:12]


def test_heldout_hash_is_caller_order_independent() -> None:
    parent = "a" * 64
    inputs_a = _hash_inputs_default(parent)
    hashes = list(inputs_a["selected_configuration_hashes_full"])
    inputs_b = deepcopy(inputs_a)
    inputs_b["selected_configuration_hashes_full"] = list(reversed(hashes))
    assert compute_heldout_run_hash_full(
        **inputs_a
    ) == compute_heldout_run_hash_full(**inputs_b)


def test_changing_sensitivity_spec_changes_hash() -> None:
    parent = "a" * 64
    inputs_a = _hash_inputs_default(parent)
    inputs_b = deepcopy(inputs_a)
    inputs_b["sensitivity_spec"] = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "scm_seed": SENSITIVITY_SCM_SEED,
        # Different fit_rng set -> different hash.
        "fit_rngs": [48, 49, 50, 51, 52],
    }
    assert compute_heldout_run_hash_full(
        **inputs_a
    ) != compute_heldout_run_hash_full(**inputs_b)


def test_changing_parent_calibration_hash_changes_hash() -> None:
    inputs_a = _hash_inputs_default("a" * 64)
    inputs_b = _hash_inputs_default("b" * 64)
    assert compute_heldout_run_hash_full(
        **inputs_a
    ) != compute_heldout_run_hash_full(**inputs_b)


def test_changing_held_out_seeds_changes_hash() -> None:
    parent = "a" * 64
    inputs_a = _hash_inputs_default(parent)
    inputs_b = deepcopy(inputs_a)
    inputs_b["main_heldout_seeds"] = [401, 402, 403, 404, 405]
    assert compute_heldout_run_hash_full(
        **inputs_a
    ) != compute_heldout_run_hash_full(**inputs_b)


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------


_EXPECTED_REPORT_KEYS: frozenset[str] = frozenset({
    "artefact_type",
    "schema_version",
    "calibration_run_hash_prefix",
    "heldout_run_hash_full",
    "heldout_run_hash_prefix",
    "main_job_count",
    "sensitivity_job_count",
    "total_job_count",
    "main_heldout_seeds",
    "sensitivity_spec",
    "planned_run_dir",
    "planned_records_dir",
    "planned_artefact_path",
    "selected_configurations_used",
    "existing_output_status",
    "policy_refs",
    "generated_at_utc",
})


def test_preflight_returns_documented_top_level_fields(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    assert set(report.keys()) == _EXPECTED_REPORT_KEYS


def test_preflight_report_top_level_constants(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    assert report["artefact_type"] == HELDOUT_PREFLIGHT_REPORT_ARTEFACT_TYPE
    assert report["schema_version"] == HELDOUT_PREFLIGHT_SCHEMA_VERSION
    assert report["main_job_count"] == 20
    assert report["sensitivity_job_count"] == 5
    assert report["total_job_count"] == 25
    assert report["main_heldout_seeds"] == [301, 302, 303, 304, 305]
    assert report["sensitivity_spec"] == {
        "model": "dcdi",
        "condition": "centred_only",
        "scm_seed": 301,
        "fit_rngs": [43, 44, 45, 46, 47],
    }


def test_preflight_planned_paths_are_under_results_root(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_heldout_evaluation(artefact_path, results_root)

    planned_run_dir = Path(report["planned_run_dir"])
    planned_records_dir = Path(report["planned_records_dir"])
    planned_artefact_path = Path(report["planned_artefact_path"])

    hash12 = report["heldout_run_hash_prefix"]
    expected_run_dir = (
        results_root
        / MODEL_SELECTION_DIRECTORY
        / HELDOUT_RUN_DIRECTORY
        / hash12
    )
    expected_records_dir = expected_run_dir / RECORDS_DIRECTORY_NAME
    expected_artefact_path = expected_run_dir / HELDOUT_EVALUATION_FILENAME

    assert planned_run_dir == expected_run_dir
    assert planned_records_dir == expected_records_dir
    assert planned_artefact_path == expected_artefact_path


def test_preflight_existing_output_status_is_would_be_created(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    assert report["existing_output_status"] == {
        "run_dir": "would_be_created",
        "records_dir": "would_be_created",
        "artefact_path": "would_be_created",
    }


def test_preflight_existing_output_status_when_dir_exists(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    results_root = tmp_path / "results"
    # First, do a side-effect-free read to discover the planned path.
    initial_report = preflight_heldout_evaluation(
        artefact_path, results_root
    )
    Path(initial_report["planned_run_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(initial_report["planned_records_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    follow_up = preflight_heldout_evaluation(artefact_path, results_root)
    assert (
        follow_up["existing_output_status"]["run_dir"]
        == "already_exists"
    )
    assert (
        follow_up["existing_output_status"]["records_dir"]
        == "already_exists"
    )
    assert (
        follow_up["existing_output_status"]["artefact_path"]
        == "would_be_created"
    )


def test_preflight_generated_at_uses_now_fn(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    clock = _FixedClock(
        datetime(2026, 5, 23, 9, 30, 0, tzinfo=timezone.utc)
    )
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results", now_fn=clock
    )
    assert report["generated_at_utc"] == "2026-05-23T09:30:00Z"


def test_preflight_returns_json_safe_dict(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    serialised = json.dumps(report, sort_keys=True, ensure_ascii=True)
    round_trip = json.loads(serialised)
    assert round_trip["total_job_count"] == 25


def test_preflight_writes_nothing(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    results_root = tmp_path / "results"

    snapshot_before_results = _snapshot_directory(results_root)
    artefact_bytes_before = artefact_path.read_bytes()

    preflight_heldout_evaluation(artefact_path, results_root)

    # results/ must not have been created.
    assert not results_root.exists()
    # Snapshot is unchanged trivially.
    assert _snapshot_directory(results_root) == snapshot_before_results
    # The input artefact bytes are unchanged.
    assert artefact_path.read_bytes() == artefact_bytes_before


def test_preflight_does_not_call_pipeline_run_single_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)

    sentinel = {"called": False}

    def _raise_if_called(*args: object, **kwargs: object) -> None:
        sentinel["called"] = True
        raise AssertionError(
            "preflight must not invoke pipeline.run_single_fit"
        )

    import experiments.selection_study.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_single_fit", _raise_if_called)

    preflight_heldout_evaluation(artefact_path, tmp_path / "results")
    assert sentinel["called"] is False


def test_preflight_reports_selected_configurations(tmp_path: Path) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    records = report["selected_configurations_used"]
    assert len(records) == 4
    seen = {(record["condition"], record["model"]) for record in records}
    expected = {
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    }
    assert seen == expected
    for record in records:
        assert isinstance(record["configuration_hash_full"], str)
        assert len(record["configuration_hash_full"]) == 64
        assert record["configuration_hash_prefix"] == (
            record["configuration_hash_full"][:HASH_PREFIX_LENGTH]
        )
        assert (
            record["selection_metrics_summary"]["degeneracy_flag"]
            is False
        )


def test_preflight_text_contains_no_forbidden_language(
    tmp_path: Path,
) -> None:
    artefact_path = _write_synthetic_artefact(tmp_path)
    report = preflight_heldout_evaluation(
        artefact_path, tmp_path / "results"
    )
    # Strip filesystem paths whose pytest tmpdir name could embed the
    # test function name and contaminate the substring scan.
    report_for_check = {
        key: value
        for key, value in report.items()
        if key not in {
            "planned_run_dir",
            "planned_records_dir",
            "planned_artefact_path",
        }
    }
    serialised = json.dumps(
        report_for_check, sort_keys=True, ensure_ascii=True
    )
    _assert_no_forbidden_language(serialised)


def test_held_out_orchestration_remains_unimplemented() -> None:
    from experiments.selection_study.held_out import (
        run_held_out_evaluation,
    )

    with pytest.raises(NotImplementedError):
        run_held_out_evaluation(None)
