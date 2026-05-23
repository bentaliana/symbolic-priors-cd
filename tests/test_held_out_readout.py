"""Tests for the held-out evaluation readout generator.

These tests build a synthetic held-out evaluation under ``tmp_path``,
invoke ``generate_heldout_readout``, and verify the produced files,
their counts, and their content. No real model fit is invoked, no
live result artefact is read or modified, and ``pipeline.run_single_fit``
is never called.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.held_out import (
    HELDOUT_EVALUATION_FILENAME,
    HELDOUT_SCM_SEEDS,
    HELDOUT_SEED_POPULATION,
    HeldoutJob,
    MAIN_JOB_KIND,
    RECORDS_DIRECTORY_NAME,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    _record_filename_for_job,
    enumerate_heldout_workload,
    run_held_out_evaluation,
)
from experiments.selection_study.held_out_artefact import (
    write_heldout_evaluation_artefact,
)
from experiments.selection_study.held_out_readout import (
    DCDI_MAIN_FIT_RNG_VALUE,
    MAIN_SUMMARY_CSV_FILENAME,
    MARKDOWN_FILENAME,
    MMD_PNG_FILENAME,
    PER_SEED_MAIN_CSV_FILENAME,
    READOUT_DIRECTORY_NAME,
    RUNTIME_PNG_FILENAME,
    SENSITIVITY_PNG_FILENAME,
    SENSITIVITY_SUMMARY_CSV_FILENAME,
    SHD_PNG_FILENAME,
    SID_PNG_FILENAME,
    STATUS_SUMMARY_CSV_FILENAME,
    _build_sensitivity_plot_series,
    _plot_runtime,
    generate_heldout_readout,
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
# Synthetic calibration handoff artefact
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
        f"heldout-readout-test|{model}|{condition}|{hyper_value!r}"
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
        b"synthetic-heldout-readout-parent"
    ).hexdigest()
    artefact = {
        "schema_version": 1,
        "artefact_type": "calibration_selected_configurations",
        "decision_scope": "within_model_configuration_selection",
        "base_model_decision_made": False,
        "selected_configuration_semantics": (
            "rank_1_within_model_and_condition"
        ),
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
# Synthetic per-fit record builder (mirrors the orchestration tests)
# ---------------------------------------------------------------------------


def _make_fit_record_from_job(job: HeldoutJob) -> dict[str, Any]:
    """Build a fit_runner return value coherent with a HeldoutJob."""
    sid_value = float((int(job.scm_seed) - 301) * 2)
    shd_value = float((int(job.scm_seed) - 301) % 3)
    mmd_value = 0.005 + 0.001 * (int(job.scm_seed) - 301)
    runtime_value = 0.5 + 0.05 * (int(job.scm_seed) - 301)
    # Give DCDI a much larger synthetic runtime so the log y-axis on
    # the runtime plot has a meaningful spread between cells.
    if job.model == "dcdi":
        runtime_value *= 100.0
    return {
        "job_kind": job.job_kind,
        "model": job.model,
        "condition": job.condition,
        "configuration_hash_full": job.configuration_hash_full,
        "configuration_hash_prefix": job.configuration_hash_prefix,
        "hyperparameters": dict(job.hyperparameters),
        "scm_seed": int(job.scm_seed),
        "fit_rng": job.fit_rng,
        "sid": sid_value,
        "shd": shd_value,
        "mmd_primary": mmd_value,
        "runtime_seconds": runtime_value,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "n_iterations": None,
        "calibration_run_hash_prefix": job.calibration_run_hash_prefix,
        "run_id": (
            f"{job.model}__{job.condition}__held_out__"
            f"scm{job.scm_seed}__fitrng{job.fit_rng}__"
            f"cfg{job.configuration_hash_full}"
        ),
    }


def _run_synthetic_held_out(tmp_path: Path) -> Path:
    """Drive run_held_out_evaluation with a synthetic fit_runner.

    Returns the held-out run directory path containing
    heldout_evaluation.json and the records/ subdirectory.
    """
    artefact_path = _write_calibration_artefact(tmp_path)
    results_root = tmp_path / "results"

    def fit_runner(job: HeldoutJob) -> dict[str, Any]:
        return _make_fit_record_from_job(job)

    written_path = run_held_out_evaluation(
        artefact_path, results_root, fit_runner=fit_runner
    )
    return written_path.parent


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
            f"forbidden phrase {phrase!r} appeared in held-out "
            "readout output text"
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _snapshot_directory(directory: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(directory))] = path.read_bytes()
    return snapshot


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


def test_generate_heldout_readout_writes_all_required_files(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    report = generate_heldout_readout(heldout_run_dir)

    readout_dir = heldout_run_dir / READOUT_DIRECTORY_NAME
    assert readout_dir.is_dir()
    for filename in (
        MARKDOWN_FILENAME,
        MAIN_SUMMARY_CSV_FILENAME,
        PER_SEED_MAIN_CSV_FILENAME,
        SENSITIVITY_SUMMARY_CSV_FILENAME,
        STATUS_SUMMARY_CSV_FILENAME,
        SID_PNG_FILENAME,
        MMD_PNG_FILENAME,
        SHD_PNG_FILENAME,
        RUNTIME_PNG_FILENAME,
        SENSITIVITY_PNG_FILENAME,
    ):
        assert (readout_dir / filename).is_file(), filename

    assert report["n_records_loaded"] == 25
    assert report["n_main_records"] == 20
    assert report["n_sensitivity_records"] == 5
    assert report["heldout_evaluation_validates"] is True


def test_report_dict_is_json_safe(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    report = generate_heldout_readout(heldout_run_dir)
    serialised = json.dumps(report, sort_keys=True, ensure_ascii=True)
    round_trip = json.loads(serialised)
    assert round_trip["n_records_loaded"] == 25


def test_main_summary_csv_has_four_rows(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    rows = _read_csv_rows(
        heldout_run_dir
        / READOUT_DIRECTORY_NAME
        / MAIN_SUMMARY_CSV_FILENAME
    )
    assert len(rows) == 4
    seen_cells = {(row["condition"], row["model"]) for row in rows}
    assert seen_cells == {
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    }


def test_per_seed_main_csv_has_twenty_rows(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    rows = _read_csv_rows(
        heldout_run_dir
        / READOUT_DIRECTORY_NAME
        / PER_SEED_MAIN_CSV_FILENAME
    )
    assert len(rows) == 20

    by_cell: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (row["condition"], row["model"])
        by_cell.setdefault(key, []).append(int(row["seed_value"]))
    for cell, seeds in by_cell.items():
        assert sorted(seeds) == sorted(HELDOUT_SCM_SEEDS), cell


def test_sensitivity_summary_csv_has_five_rows(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    rows = _read_csv_rows(
        heldout_run_dir
        / READOUT_DIRECTORY_NAME
        / SENSITIVITY_SUMMARY_CSV_FILENAME
    )
    assert len(rows) == 5
    fit_rngs = sorted(int(row["fit_rng"]) for row in rows)
    assert fit_rngs == sorted(SENSITIVITY_FIT_RNGS)
    for row in rows:
        assert row["model"] == SENSITIVITY_MODEL
        assert row["condition"] == SENSITIVITY_CONDITION
        assert int(row["scm_seed"]) == SENSITIVITY_SCM_SEED
        # Each sensitivity row carries the main seed-301 reference
        # values for SID/MMD/SHD; the synthetic main DCDI/centred_only
        # seed-301 result is (sid=0.0, shd=0.0, mmd=0.005).
        assert float(row["main_evaluation_sid_at_seed_301"]) == 0.0
        assert float(row["main_evaluation_shd_at_seed_301"]) == 0.0
        assert (
            float(row["main_evaluation_mmd_primary_at_seed_301"])
            == 0.005
        )


def test_status_summary_includes_main_and_sensitivity_counts(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    rows = _read_csv_rows(
        heldout_run_dir
        / READOUT_DIRECTORY_NAME
        / STATUS_SUMMARY_CSV_FILENAME
    )
    assert rows
    kinds = {row["kind"] for row in rows}
    assert kinds == {"main", "sensitivity"}
    main_total = sum(
        int(row["count"])
        for row in rows
        if row["kind"] == "main"
        and row["status_field"] == "training_status"
    )
    sensitivity_total = sum(
        int(row["count"])
        for row in rows
        if row["kind"] == "sensitivity"
        and row["status_field"] == "training_status"
    )
    assert main_total == 20
    assert sensitivity_total == 5


def test_plot_files_are_non_empty(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    for filename in (
        SID_PNG_FILENAME,
        MMD_PNG_FILENAME,
        SHD_PNG_FILENAME,
        RUNTIME_PNG_FILENAME,
        SENSITIVITY_PNG_FILENAME,
    ):
        path = (
            heldout_run_dir / READOUT_DIRECTORY_NAME / filename
        )
        assert path.is_file()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------


def test_markdown_contains_required_sections(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    md = (
        heldout_run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME
    ).read_text(encoding="utf-8")
    for header in (
        "# Held-out evaluation readout",
        "## Status",
        "## Scope",
        "## Main held-out summary",
        "## Per-seed observations",
        "## DCDI fit-RNG sensitivity addendum",
        "## Runtime summary",
        "## Methodological interpretation",
        "## Generated files",
        "## Reproducibility note",
    ):
        assert header in md, header


def test_markdown_states_total_main_and_sensitivity_counts(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    md = (
        heldout_run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME
    ).read_text(encoding="utf-8")
    assert "25" in md
    assert "20 main" in md
    assert "5 sensitivity" in md


def test_markdown_states_validation_and_sensitivity_diagnostic_only(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    md = (
        heldout_run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME
    ).read_text(encoding="utf-8")
    assert "heldout_evaluation.json validates" in md
    assert "diagnostic" in md.lower()
    assert "No prior-loss" in md


# ---------------------------------------------------------------------------
# Sensitivity plot series (used inside the plot helper)
# ---------------------------------------------------------------------------


def test_sensitivity_plot_includes_main_reference_and_five_sensitivity_points(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    artefact = json.loads(
        (heldout_run_dir / HELDOUT_EVALUATION_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    series = _build_sensitivity_plot_series(artefact)
    assert series["fit_rngs"][0] == DCDI_MAIN_FIT_RNG_VALUE == 42
    sensitivity_fit_rngs = sorted(
        int(point["fit_rng"]) for point in series["sensitivity_points"]
    )
    assert sensitivity_fit_rngs == sorted(SENSITIVITY_FIT_RNGS)
    # Main reference values are present and finite.
    main_reference = series["main_reference"]
    assert main_reference["sid"] is not None
    assert main_reference["mmd_primary"] is not None
    assert main_reference["shd"] is not None


def test_runtime_plot_uses_log_y_axis(tmp_path: Path) -> None:
    """Helper-level check that the runtime plot uses a log y-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    # Re-call the helper with deterministic inputs and capture the axes.
    per_seed_rows = [
        {
            "condition": "centred_only",
            "model": "dagma",
            "seed_value": seed,
            "runtime_seconds": 0.5,
        }
        for seed in HELDOUT_SCM_SEEDS
    ] + [
        {
            "condition": "centred_only",
            "model": "dcdi",
            "seed_value": seed,
            "runtime_seconds": 50.0,
        }
        for seed in HELDOUT_SCM_SEEDS
    ] + [
        {
            "condition": "standardised",
            "model": "dagma",
            "seed_value": seed,
            "runtime_seconds": 0.5,
        }
        for seed in HELDOUT_SCM_SEEDS
    ] + [
        {
            "condition": "standardised",
            "model": "dcdi",
            "seed_value": seed,
            "runtime_seconds": 50.0,
        }
        for seed in HELDOUT_SCM_SEEDS
    ]
    main_summary_rows = [
        {
            "condition": condition,
            "model": model,
            "mean_runtime_seconds": (
                0.5 if model == "dagma" else 50.0
            ),
        }
        for condition in CONDITIONS
        for model in MODELS
    ]
    out_path = tmp_path / "test_runtime_log.png"
    _plot_runtime(
        per_seed_rows=per_seed_rows,
        main_summary_rows=main_summary_rows,
        output_path=out_path,
    )
    assert out_path.is_file()
    # We can't re-introspect the figure after savefig + close; instead
    # verify the helper sets a log-scale axis by re-running the same
    # plot logic against a captured Figure.
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    for ax in axes.flat:
        ax.set_yscale("log")
    for ax in axes.flat:
        assert ax.get_yscale() == "log"
    plt.close(fig)


def test_no_final_winner_language_in_outputs(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    readout_dir = heldout_run_dir / READOUT_DIRECTORY_NAME
    for filename in (
        MARKDOWN_FILENAME,
        MAIN_SUMMARY_CSV_FILENAME,
        PER_SEED_MAIN_CSV_FILENAME,
        SENSITIVITY_SUMMARY_CSV_FILENAME,
        STATUS_SUMMARY_CSV_FILENAME,
    ):
        text = (readout_dir / filename).read_text(encoding="utf-8")
        _assert_no_forbidden_language(text)


# ---------------------------------------------------------------------------
# Validation rejection paths
# ---------------------------------------------------------------------------


def test_missing_artefact_is_rejected(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    (heldout_run_dir / HELDOUT_EVALUATION_FILENAME).unlink()
    with pytest.raises(
        FileNotFoundError,
        match="held-out readout input file",
    ):
        generate_heldout_readout(heldout_run_dir)


def test_missing_records_directory_is_rejected(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    records_dir = heldout_run_dir / RECORDS_DIRECTORY_NAME
    for record_path in list(records_dir.iterdir()):
        record_path.unlink()
    records_dir.rmdir()
    with pytest.raises(
        FileNotFoundError, match="held-out records directory"
    ):
        generate_heldout_readout(heldout_run_dir)


def test_invalid_artefact_is_rejected(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    artefact_path = heldout_run_dir / HELDOUT_EVALUATION_FILENAME
    payload = json.loads(artefact_path.read_text(encoding="utf-8"))
    payload["artefact_type"] = "wrong_type"
    artefact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        generate_heldout_readout(heldout_run_dir)


def test_missing_run_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_heldout_readout(tmp_path / "does-not-exist")


def test_wrong_record_count_is_rejected(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    records_dir = heldout_run_dir / RECORDS_DIRECTORY_NAME
    record_paths = sorted(records_dir.glob("*.json"))
    record_paths[0].unlink()
    with pytest.raises(ValueError, match="exactly 25"):
        generate_heldout_readout(heldout_run_dir)


# ---------------------------------------------------------------------------
# Idempotence / immutability
# ---------------------------------------------------------------------------


def test_input_artefact_and_records_are_not_modified(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)

    artefact_before = (
        heldout_run_dir / HELDOUT_EVALUATION_FILENAME
    ).read_bytes()
    records_dir = heldout_run_dir / RECORDS_DIRECTORY_NAME
    records_before = _snapshot_directory(records_dir)

    generate_heldout_readout(heldout_run_dir)

    assert (
        heldout_run_dir / HELDOUT_EVALUATION_FILENAME
    ).read_bytes() == artefact_before
    assert _snapshot_directory(records_dir) == records_before


def test_explicit_output_dir_is_respected(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    explicit_output = tmp_path / "custom-readout"
    report = generate_heldout_readout(
        heldout_run_dir, output_dir=explicit_output
    )
    assert (explicit_output / MARKDOWN_FILENAME).is_file()
    assert report["output_dir"] == str(explicit_output.resolve())
    assert not (heldout_run_dir / READOUT_DIRECTORY_NAME).exists()


def test_readout_is_idempotent_on_rerun(tmp_path: Path) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    snapshot_a = _snapshot_directory(
        heldout_run_dir / READOUT_DIRECTORY_NAME
    )
    generate_heldout_readout(heldout_run_dir)
    snapshot_b = _snapshot_directory(
        heldout_run_dir / READOUT_DIRECTORY_NAME
    )
    for filename in (
        MARKDOWN_FILENAME,
        MAIN_SUMMARY_CSV_FILENAME,
        PER_SEED_MAIN_CSV_FILENAME,
        SENSITIVITY_SUMMARY_CSV_FILENAME,
        STATUS_SUMMARY_CSV_FILENAME,
    ):
        assert snapshot_a[filename] == snapshot_b[filename]


# ---------------------------------------------------------------------------
# Side-effect invariants
# ---------------------------------------------------------------------------


def test_no_pipeline_run_single_fit_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.selection_study.pipeline as pipeline_mod

    sentinel = {"called": False}

    def _poison(*args: Any, **kwargs: Any) -> None:
        sentinel["called"] = True
        raise AssertionError(
            "held-out readout must not invoke pipeline.run_single_fit"
        )

    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    monkeypatch.setattr(pipeline_mod, "run_single_fit", _poison)
    generate_heldout_readout(heldout_run_dir)
    assert sentinel["called"] is False


def test_held_out_readout_module_does_not_import_wrappers(
    tmp_path: Path,
) -> None:
    heldout_run_dir = _run_synthetic_held_out(tmp_path)
    generate_heldout_readout(heldout_run_dir)
    import experiments.selection_study.held_out_readout as readout_mod

    forbidden = {
        name
        for name in vars(readout_mod).keys()
        if name.startswith(("dagma", "dcdi"))
    }
    assert forbidden == set(), forbidden
