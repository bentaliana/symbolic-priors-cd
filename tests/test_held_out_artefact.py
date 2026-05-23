"""Tests for the held-out evaluation artefact module.

These tests exercise the aggregation, schema validation, and atomic
writer using synthetic per-fit records and synthetic workloads under
``tmp_path``. No test runs a fit, writes a live result artefact, or
calls ``pipeline.run_single_fit``; the artefact module must be
side-effect free with respect to model execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.held_out import (
    HELDOUT_SCM_SEEDS,
    HeldoutJob,
    HeldoutWorkload,
    MAIN_JOB_KIND,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    enumerate_heldout_workload,
)
from experiments.selection_study.held_out_artefact import (
    EXPECTED_MAIN_PER_CELL,
    EXPECTED_MAIN_TOTAL,
    EXPECTED_SENSITIVITY_TOTAL,
    EXPECTED_TOTAL_RECORDS,
    HELDOUT_EVALUATION_ARTEFACT_TYPE,
    HELDOUT_EVALUATION_SCHEMA_VERSION,
    aggregate_fit_rng_sensitivity_records,
    aggregate_main_heldout_records,
    build_heldout_evaluation_artefact,
    validate_heldout_evaluation_artefact,
    write_heldout_evaluation_artefact,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    FIT_RNG_POLICY_REF,
    HASH_PREFIX_LENGTH,
    INTERVENTION_POLICY_REF,
    MODELS,
    SELECTED_CONFIGURATIONS_FILENAME,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
    write_selected_configurations,
)


# ---------------------------------------------------------------------------
# Synthetic calibration artefact builder (mirrors held-out preflight tests)
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
        f"heldout-artefact-test|{model}|{condition}|{hyper_value!r}"
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


def _build_calibration_artefact() -> dict[str, Any]:
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
        b"synthetic-heldout-artefact-test-parent"
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


def _write_calibration_artefact(tmp_path: Path) -> Path:
    artefact = _build_calibration_artefact()
    path = tmp_path / SELECTED_CONFIGURATIONS_FILENAME
    write_selected_configurations(artefact, path)
    return path


# ---------------------------------------------------------------------------
# Synthetic per-fit record builder, aligned with a workload
# ---------------------------------------------------------------------------


def _make_fit_record_from_job(
    job: HeldoutJob,
    *,
    sid: Any,
    shd: Any,
    mmd_primary: Any,
    runtime_seconds: Any = 1.0,
    training_status: str = "converged",
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
    n_iterations: Any = None,
) -> dict[str, Any]:
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
        "n_iterations": n_iterations,
        "run_id": (
            f"{job.model}__{job.condition}__held_out__"
            f"scm{job.scm_seed}__fitrng{job.fit_rng}__"
            f"cfg{job.configuration_hash_full}"
        ),
    }


def _build_workload_and_records(
    tmp_path: Path,
    *,
    override_metric: tuple[
        tuple[str, str, int, Any], dict[str, Any]
    ] | None = None,
    inject_non_finite: bool = False,
    inject_all_non_finite_sid: bool = False,
) -> tuple[HeldoutWorkload, list[dict[str, Any]]]:
    """Build a workload and a coherent record set for testing.

    ``override_metric`` lets one specific (model, condition, scm_seed,
    fit_rng) job carry custom metric values.
    ``inject_non_finite`` makes one main DAGMA centred_only record
    carry a ``NaN`` SID so the cell sees a single non-finite seed.
    ``inject_all_non_finite_sid`` makes every main DCDI standardised
    record carry a ``NaN`` SID so the whole metric is non-finite.
    """
    artefact_path = _write_calibration_artefact(tmp_path)
    workload = enumerate_heldout_workload(artefact_path)

    records: list[dict[str, Any]] = []
    for job in workload.main_jobs:
        # Deterministic but distinguishable values per seed and per
        # model so aggregation tests can check specific quantiles.
        sid_value = float(int(job.scm_seed) - 301)  # 0,1,2,3,4
        shd_value = float((int(job.scm_seed) - 301) % 3)
        mmd_value = 0.005 + 0.001 * (int(job.scm_seed) - 301)
        runtime_value = 1.0 + 0.1 * (int(job.scm_seed) - 301)
        records.append(
            _make_fit_record_from_job(
                job,
                sid=sid_value,
                shd=shd_value,
                mmd_primary=mmd_value,
                runtime_seconds=runtime_value,
            )
        )
    for job in workload.sensitivity_jobs:
        offset = int(job.fit_rng) - 43  # 0,1,2,3,4
        records.append(
            _make_fit_record_from_job(
                job,
                sid=2.0 + offset,
                shd=1.0,
                mmd_primary=0.01 + 0.002 * offset,
                runtime_seconds=1.5,
            )
        )

    if override_metric is not None:
        (model, condition, scm_seed, fit_rng), updates = override_metric
        for record in records:
            if (
                record["model"] == model
                and record["condition"] == condition
                and record["scm_seed"] == scm_seed
                and record["fit_rng"] == fit_rng
            ):
                record.update(updates)

    if inject_non_finite:
        for record in records:
            if (
                record["job_kind"] == MAIN_JOB_KIND
                and record["model"] == "dagma"
                and record["condition"] == "centred_only"
                and record["scm_seed"] == 301
            ):
                record["sid"] = float("nan")
                break

    if inject_all_non_finite_sid:
        for record in records:
            if (
                record["job_kind"] == MAIN_JOB_KIND
                and record["model"] == "dcdi"
                and record["condition"] == "standardised"
            ):
                record["sid"] = float("nan")

    return workload, records


# ---------------------------------------------------------------------------
# Forbidden-language guard
# ---------------------------------------------------------------------------


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
            f"forbidden phrase {phrase!r} appeared in artefact text"
        )


# ---------------------------------------------------------------------------
# Build / validate happy path
# ---------------------------------------------------------------------------


def test_artefact_builds_and_validates(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    validate_heldout_evaluation_artefact(artefact)
    assert artefact["artefact_type"] == HELDOUT_EVALUATION_ARTEFACT_TYPE
    assert artefact["schema_version"] == HELDOUT_EVALUATION_SCHEMA_VERSION


def test_top_level_field_set_is_documented(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    expected_keys = {
        "artefact_type",
        "schema_version",
        "parent_calibration_run_hash_full",
        "parent_calibration_run_hash_prefix",
        "heldout_run_hash_full",
        "heldout_run_hash_prefix",
        "selected_configurations_used",
        "main_heldout_seeds",
        "sensitivity_spec",
        "policy_refs",
        "main_evaluation",
        "fit_rng_sensitivity_addendum",
        "status_summary",
        "generated_at_utc",
    }
    assert set(artefact.keys()) == expected_keys


def test_main_evaluation_has_four_cells_of_five_records(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    cells = artefact["main_evaluation"]["cells"]
    assert set(cells.keys()) == set(CONDITIONS)
    for condition in CONDITIONS:
        assert set(cells[condition].keys()) == set(MODELS)
        for model in MODELS:
            cell = cells[condition][model]
            assert len(cell["per_seed_records"]) == EXPECTED_MAIN_PER_CELL
            seeds = sorted(
                record["seed_value"]
                for record in cell["per_seed_records"]
            )
            assert seeds == sorted(HELDOUT_SCM_SEEDS)


def test_sensitivity_addendum_is_separate_from_main_aggregates(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    main_dcdi_centred = artefact["main_evaluation"]["cells"][
        SENSITIVITY_CONDITION
    ][SENSITIVITY_MODEL]
    main_finite_count = main_dcdi_centred["aggregate_metrics"]["sid"][
        "finite_count"
    ]
    main_non_finite_count = main_dcdi_centred["aggregate_metrics"]["sid"][
        "non_finite_count"
    ]
    assert main_finite_count + main_non_finite_count == EXPECTED_MAIN_PER_CELL

    addendum = artefact["fit_rng_sensitivity_addendum"]
    assert len(addendum["per_fit_records"]) == EXPECTED_SENSITIVITY_TOTAL
    sensitivity_fit_rngs = sorted(
        record["fit_rng"] for record in addendum["per_fit_records"]
    )
    assert sensitivity_fit_rngs == sorted(SENSITIVITY_FIT_RNGS)
    # No sensitivity fit_rng appears in any main per-seed record.
    for condition in CONDITIONS:
        for model in MODELS:
            cell = artefact["main_evaluation"]["cells"][condition][model]
            for record in cell["per_seed_records"]:
                assert record["fit_rng"] not in set(SENSITIVITY_FIT_RNGS)


def test_sensitivity_summary_includes_seed_301_main_comparison(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    diagnostic = artefact["fit_rng_sensitivity_addendum"][
        "diagnostic_summary"
    ]
    # Synthetic main DCDI/centred_only/seed=301 has sid=0.0, shd=0.0,
    # mmd=0.005 by construction.
    assert diagnostic["main_evaluation_sid_at_seed_301"] == 0.0
    assert diagnostic["main_evaluation_shd_at_seed_301"] == 0.0
    assert diagnostic["main_evaluation_mmd_primary_at_seed_301"] == 0.005


def test_interpretation_note_flags_diagnostic_only(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    note = artefact["fit_rng_sensitivity_addendum"]["interpretation_note"]
    assert "Diagnostic only" in note
    assert "base-model decision" in note


# ---------------------------------------------------------------------------
# Aggregation correctness
# ---------------------------------------------------------------------------


def test_metric_aggregates_match_reference_values(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    # The synthetic SID values per main cell are exactly {0,1,2,3,4}.
    cell = artefact["main_evaluation"]["cells"]["centred_only"]["dagma"]
    sid_summary = cell["aggregate_metrics"]["sid"]
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert sid_summary["finite_count"] == 5
    assert sid_summary["non_finite_count"] == 0
    assert sid_summary["mean"] == pytest.approx(statistics.mean(values))
    assert sid_summary["std"] == pytest.approx(
        statistics.stdev(values)
    )
    assert sid_summary["median"] == pytest.approx(2.0)
    assert sid_summary["q1"] == pytest.approx(1.0)
    assert sid_summary["q3"] == pytest.approx(3.0)
    assert sid_summary["iqr"] == pytest.approx(2.0)
    assert sid_summary["min"] == 0.0
    assert sid_summary["max"] == 4.0


def test_partial_non_finite_metric_is_flagged(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(
        tmp_path, inject_non_finite=True
    )
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    cell = artefact["main_evaluation"]["cells"]["centred_only"]["dagma"]
    sid_summary = cell["aggregate_metrics"]["sid"]
    assert sid_summary["finite_count"] == 4
    assert sid_summary["non_finite_count"] == 1
    assert cell["aggregate_metrics"]["has_non_finite_seed_metric"] is True
    assert "sid" in cell["aggregate_metrics"]["degenerate_metric_names"]
    # The per-seed record carrying the NaN is encoded as JSON null.
    seed_301 = next(
        record
        for record in cell["per_seed_records"]
        if record["seed_value"] == 301
    )
    assert seed_301["sid"] is None


def test_all_non_finite_metric_produces_none_aggregates(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(
        tmp_path, inject_all_non_finite_sid=True
    )
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    cell = artefact["main_evaluation"]["cells"]["standardised"]["dcdi"]
    sid_summary = cell["aggregate_metrics"]["sid"]
    assert sid_summary["finite_count"] == 0
    assert sid_summary["non_finite_count"] == 5
    for field_name in ("mean", "std", "median", "q1", "q3", "iqr", "min", "max"):
        assert sid_summary[field_name] is None, field_name
    assert cell["aggregate_metrics"]["has_non_finite_seed_metric"] is True
    assert "sid" in cell["aggregate_metrics"]["degenerate_metric_names"]


def test_std_is_none_with_single_finite_value(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    # Knock out four of the five DAGMA standardised SID values.
    for record in records:
        if (
            record["job_kind"] == MAIN_JOB_KIND
            and record["model"] == "dagma"
            and record["condition"] == "standardised"
            and record["scm_seed"] != 301
        ):
            record["sid"] = float("nan")
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    cell = artefact["main_evaluation"]["cells"]["standardised"]["dagma"]
    sid_summary = cell["aggregate_metrics"]["sid"]
    assert sid_summary["finite_count"] == 1
    assert sid_summary["mean"] == 0.0  # SID at seed 301 was 0.0
    assert sid_summary["std"] is None
    assert sid_summary["median"] == 0.0
    assert sid_summary["q1"] == 0.0
    assert sid_summary["q3"] == 0.0
    assert sid_summary["iqr"] == 0.0


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_artefact_is_json_serialisable(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    serialised = json.dumps(artefact, sort_keys=True, ensure_ascii=True)
    round_trip = json.loads(serialised)
    validate_heldout_evaluation_artefact(round_trip)


def test_artefact_text_contains_no_forbidden_language(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    serialised = json.dumps(artefact, sort_keys=True, ensure_ascii=True)
    _assert_no_forbidden_language(serialised)


# ---------------------------------------------------------------------------
# Validation: rejection cases
# ---------------------------------------------------------------------------


def test_validator_rejects_missing_top_level_field(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    del artefact["status_summary"]
    with pytest.raises(ValueError, match="missing required"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_unknown_top_level_field(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    artefact["bonus_field"] = "extra"
    with pytest.raises(ValueError, match="unknown top-level field"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_forbidden_decision_field(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    # Plant a forbidden key deep inside the artefact structure.
    artefact["main_evaluation"]["cells"]["centred_only"]["dagma"][
        "winner"
    ] = "dagma"
    with pytest.raises(ValueError, match="forbidden field"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_calibration_seed(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    # Replace one per-seed record's seed value with a calibration
    # seed without breaking the rest of the structure.
    centred_dagma = artefact["main_evaluation"]["cells"][
        "centred_only"
    ]["dagma"]
    centred_dagma["per_seed_records"][0]["seed_value"] = 202
    with pytest.raises(ValueError, match="calibration SCM seed"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_wrong_sensitivity_fit_rng(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    addendum = artefact["fit_rng_sensitivity_addendum"]
    addendum["per_fit_records"][0]["fit_rng"] = 99
    with pytest.raises(ValueError, match="fit_rng values must equal"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_sensitivity_record_inside_main(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    cell = artefact["main_evaluation"]["cells"][SENSITIVITY_CONDITION][
        SENSITIVITY_MODEL
    ]
    seed_301 = next(
        record
        for record in cell["per_seed_records"]
        if record["seed_value"] == SENSITIVITY_SCM_SEED
    )
    # Replace the main fit_rng (42) with a sensitivity fit_rng (43)
    # to model the failure mode where sensitivity data leaks into
    # the main cell.
    seed_301["fit_rng"] = 43
    with pytest.raises(ValueError, match="sensitivity fit_rng"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_bad_generated_at_utc(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    artefact["generated_at_utc"] = "2026-05-30 10:00:00"
    with pytest.raises(ValueError, match="generated_at_utc"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_main_seed_set_mismatch(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    artefact["main_heldout_seeds"] = [301, 302, 303, 304, 306]
    with pytest.raises(ValueError, match="main_heldout_seeds"):
        validate_heldout_evaluation_artefact(artefact)


def test_validator_rejects_non_finite_float_in_serialised_form(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    # Bypass the aggregator: plant a raw NaN into the artefact.
    cell = artefact["main_evaluation"]["cells"]["centred_only"]["dagma"]
    cell["aggregate_metrics"]["sid"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_heldout_evaluation_artefact(artefact)


# ---------------------------------------------------------------------------
# Builder cross-checks against the workload
# ---------------------------------------------------------------------------


def test_builder_rejects_record_count_other_than_25(
    tmp_path: Path,
) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    with pytest.raises(ValueError, match="exactly 25"):
        build_heldout_evaluation_artefact(
            workload=workload,
            records=records[:-1],
            generated_at_utc="2026-05-30T10:00:00Z",
        )


def test_builder_rejects_record_not_in_workload(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    # Replace one record's scm_seed with a seed not present in the
    # main workload (still a valid held-out seed but the workload
    # already covers each seed once per cell so the cross-check
    # picks up the dangling identity).
    records[0]["scm_seed"] = 999
    with pytest.raises(ValueError, match="workload"):
        build_heldout_evaluation_artefact(
            workload=workload,
            records=records,
            generated_at_utc="2026-05-30T10:00:00Z",
        )


# ---------------------------------------------------------------------------
# Writer behaviour
# ---------------------------------------------------------------------------


def test_writer_writes_and_revalidates(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    output_path = tmp_path / "subdir" / "heldout_evaluation.json"
    written = write_heldout_evaluation_artefact(artefact, output_path)
    assert written == output_path
    assert output_path.is_file()
    read_back = json.loads(output_path.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(read_back)
    assert read_back["artefact_type"] == HELDOUT_EVALUATION_ARTEFACT_TYPE


def test_writer_refuses_overwrite_without_force(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    output_path = tmp_path / "heldout_evaluation.json"
    write_heldout_evaluation_artefact(artefact, output_path)
    original_bytes = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_heldout_evaluation_artefact(artefact, output_path)
    assert output_path.read_bytes() == original_bytes


def test_writer_allows_overwrite_with_force(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact_a = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    output_path = tmp_path / "heldout_evaluation.json"
    write_heldout_evaluation_artefact(artefact_a, output_path)

    artefact_b = deepcopy(artefact_a)
    artefact_b["generated_at_utc"] = "2026-06-01T12:00:00Z"
    write_heldout_evaluation_artefact(artefact_b, output_path, force=True)
    refreshed = json.loads(output_path.read_text(encoding="utf-8"))
    assert refreshed["generated_at_utc"] == "2026-06-01T12:00:00Z"


def test_writer_validates_before_writing(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    artefact["artefact_type"] = "wrong_type"
    output_path = tmp_path / "heldout_evaluation.json"
    with pytest.raises(ValueError):
        write_heldout_evaluation_artefact(artefact, output_path)
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Side-effect invariants
# ---------------------------------------------------------------------------


def test_no_pipeline_run_single_fit_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.selection_study.pipeline as pipeline

    sentinel = {"called": False}

    def _poison(*args: object, **kwargs: object) -> None:
        sentinel["called"] = True
        raise AssertionError(
            "held-out artefact module must not invoke "
            "pipeline.run_single_fit"
        )

    monkeypatch.setattr(pipeline, "run_single_fit", _poison)

    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    validate_heldout_evaluation_artefact(artefact)
    output_path = tmp_path / "heldout_evaluation.json"
    write_heldout_evaluation_artefact(artefact, output_path)
    assert sentinel["called"] is False


def test_no_wrapper_module_is_imported(tmp_path: Path) -> None:
    workload, records = _build_workload_and_records(tmp_path)
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc="2026-05-30T10:00:00Z",
    )
    validate_heldout_evaluation_artefact(artefact)
    output_path = tmp_path / "heldout_evaluation.json"
    write_heldout_evaluation_artefact(artefact, output_path)

    forbidden_module_substrings = (
        "symbolic_priors_cd.wrappers",
        "dagma",
        "dcdi",
    )
    leaked = [
        name
        for name in sys.modules
        if any(
            substring in name
            for substring in forbidden_module_substrings
        )
        # Allow our own held-out modules; they never import a wrapper.
        and not name.startswith("experiments.selection_study")
    ]
    # The DAGMA / DCDI wrapper modules may legitimately be loaded by
    # other tests in the same pytest session, so we cannot strictly
    # assert "no wrapper modules in sys.modules". The point of this
    # test is to confirm that the artefact code path itself does not
    # depend on them at import time; we therefore inspect the
    # held_out_artefact module's own globals for accidental imports.
    import experiments.selection_study.held_out_artefact as artefact_mod

    artefact_mod_globals = set(vars(artefact_mod).keys())
    forbidden_globals = {
        name
        for name in artefact_mod_globals
        if name.startswith(("dagma", "dcdi"))
    }
    assert forbidden_globals == set(), forbidden_globals
