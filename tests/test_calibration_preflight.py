"""Tests for the calibration dry-run / preflight code path.

These tests exercise ``preflight_calibration`` and the CLI dry-run
plumbing without invoking any real DAGMA or DCDI fit. Every test
either uses the pure ``preflight_calibration`` Python entry point or
the ``--phase calibration --dry-run`` CLI via a subprocess; no test
runs a fit, writes a result artefact, or modifies the on-disk
config directory.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.calibration import (
    preflight_calibration,
    run_calibration,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    FIT_RNG_POLICY_REF,
    INTERVENTION_POLICY_REF,
    SELECTED_CONFIGURATIONS_FILENAME,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CALIBRATION_CONFIG_DIR = (
    _PROJECT_ROOT
    / "experiments"
    / "selection_study"
    / "configs"
    / "calibration"
)
_PARENT_FILENAMES: tuple[str, ...] = (
    "dagma_calibration_centred_only.json",
    "dagma_calibration_standardised.json",
    "dcdi_calibration_centred_only.json",
    "dcdi_calibration_standardised.json",
)
_EXPECTED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset({
    "artefact_type",
    "schema_version",
    "calibration_run_hash_full",
    "calibration_run_hash_prefix",
    "config_dir",
    "results_root",
    "expected_artefact_path",
    "expected_records_dir",
    "workload_summary",
    "policy_refs",
    "artefact_status",
    "records_status",
    "generated_at_utc",
})


def _copy_calibration_configs_to_tmp(tmp_path: Path) -> Path:
    """Copy the four parent calibration JSONs into a tmp directory."""
    target = tmp_path / "configs"
    target.mkdir()
    for filename in _PARENT_FILENAMES:
        shutil.copy(_CALIBRATION_CONFIG_DIR / filename, target / filename)
    return target


class _FixedClock:
    """Always returns the same datetime; used for deterministic timestamps."""

    def __init__(
        self,
        value: datetime = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    ) -> None:
        self._value = value

    def __call__(self) -> datetime:
        return self._value


def _snapshot_directory_state(directory: Path) -> dict[str, bytes]:
    """Return a hash-friendly snapshot of every file under ``directory``."""
    snapshot: dict[str, bytes] = {}
    if not directory.exists():
        return snapshot
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = path.relative_to(directory).as_posix()
            snapshot[rel] = path.read_bytes()
    return snapshot


# ---------------------------------------------------------------------------
# Preflight report structure
# ---------------------------------------------------------------------------


def test_preflight_returns_exact_top_level_field_set(
    tmp_path: Path,
) -> None:
    """The report carries exactly the 13 documented top-level fields."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    assert set(report.keys()) == _EXPECTED_TOP_LEVEL_FIELDS


def test_preflight_report_workload_summary_has_20_candidates_40_jobs(
    tmp_path: Path,
) -> None:
    """workload_summary pins the frozen 20-candidate / 40-job arithmetic."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    workload_summary = report["workload_summary"]
    assert workload_summary["total_candidates"] == 20
    assert workload_summary["total_fit_jobs"] == 40


def test_preflight_report_has_four_condition_model_cells(
    tmp_path: Path,
) -> None:
    """candidates_per_cell carries the four (condition, model) pairs."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    per_cell = report["workload_summary"]["candidates_per_cell"]
    assert set(per_cell.keys()) == {"centred_only", "standardised"}
    for condition_block in per_cell.values():
        assert set(condition_block.keys()) == {"dagma", "dcdi"}


def test_preflight_report_has_five_candidates_per_cell(
    tmp_path: Path,
) -> None:
    """Each (condition, model) cell has exactly five candidates."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    per_cell = report["workload_summary"]["candidates_per_cell"]
    for condition in per_cell:
        for model in per_cell[condition]:
            assert per_cell[condition][model] == 5


def test_preflight_report_seeds_match_calibration_pool(
    tmp_path: Path,
) -> None:
    """workload_summary['seeds'] equals the frozen calibration pool."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    assert report["workload_summary"]["seeds"] == list(CALIBRATION_SEEDS)


def test_preflight_report_carries_no_held_out_seeds(
    tmp_path: Path,
) -> None:
    """No held-out seed integer (301..305) appears anywhere in the report.

    Walks the report tree and inspects every int leaf. A substring
    check over the JSON dump is too coarse because the calibration
    run hash may legitimately contain digit sequences like ``302``
    inside its 64-character hex string; the structural walk
    distinguishes integer-valued seed leaves from hash characters.
    """
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    forbidden_seed_values = {301, 302, 303, 304, 305}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, int):
            assert obj not in forbidden_seed_values, (
                f"forbidden held-out seed {obj} appeared as an int "
                "leaf in the preflight report"
            )

    walk(report)


def test_preflight_report_calibration_run_hash_is_64_lowercase_hex(
    tmp_path: Path,
) -> None:
    """calibration_run_hash_full is a 64-char lowercase hex string."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    full = report["calibration_run_hash_full"]
    assert isinstance(full, str)
    assert len(full) == 64
    assert full == full.lower()
    assert all(ch in "0123456789abcdef" for ch in full)
    assert report["calibration_run_hash_prefix"] == full[:12]


def test_preflight_report_paths_use_expected_layout(
    tmp_path: Path,
) -> None:
    """expected_artefact_path and expected_records_dir follow the convention."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    artefact_path = Path(report["expected_artefact_path"])
    records_dir = Path(report["expected_records_dir"])
    assert artefact_path.name == SELECTED_CONFIGURATIONS_FILENAME
    assert artefact_path.parent.parent.name == "calibration"
    assert records_dir.parent == artefact_path.parent
    assert records_dir.name == "records"
    # The artefact path lives under <results_root>/model_selection/calibration/<hash12>/
    assert "model_selection" in artefact_path.parts
    assert report["calibration_run_hash_prefix"] in artefact_path.parts


def test_preflight_report_policy_refs_match_selection_artefact_constants(
    tmp_path: Path,
) -> None:
    """policy_refs carries the four stable identifiers verbatim."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    refs = report["policy_refs"]
    assert refs["selection_rule_id"] == SELECTION_RULE_ID
    assert refs["selection_rule_ref"] == SELECTION_RULE_REF
    assert refs["intervention_policy_ref"] == INTERVENTION_POLICY_REF
    assert refs["fit_rng_policy_ref"] == FIT_RNG_POLICY_REF


def test_preflight_report_generated_at_utc_is_deterministic_with_fixed_now_fn(
    tmp_path: Path,
) -> None:
    """A fixed now_fn produces a deterministic generated_at_utc timestamp."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    now_fn = _FixedClock(
        value=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    )
    report = preflight_calibration(config_dir, results_root, now_fn=now_fn)
    assert report["generated_at_utc"] == "2026-05-22T12:00:00Z"


def test_preflight_report_is_deterministic_across_two_calls(
    tmp_path: Path,
) -> None:
    """Two calls with the same input and now_fn produce byte-identical JSON."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    now_fn = _FixedClock()
    report_a = preflight_calibration(
        config_dir, results_root, now_fn=now_fn
    )
    report_b = preflight_calibration(
        config_dir, results_root, now_fn=now_fn
    )
    assert json.dumps(report_a, sort_keys=True) == json.dumps(
        report_b, sort_keys=True
    )


def test_preflight_report_status_fields_when_outputs_do_not_exist(
    tmp_path: Path,
) -> None:
    """A fresh results_root yields would_be_created statuses."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    assert report["artefact_status"] == "would_be_created"
    assert report["records_status"] == "would_be_created"


def test_preflight_report_status_fields_when_outputs_already_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a prior real run the statuses flip to already_exists.

    Drives the orchestrator with a recording fit_runner to produce a
    real selected_configurations.json plus a records/ directory,
    then re-runs preflight against the same config_dir + results_root
    and checks the status fields.
    """
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def _happy_fit_runner(job: Any) -> dict[str, Any]:
        candidate = job.candidate
        hyperparameters = dict(candidate.grid_point_hyperparameter)
        return {
            "model": candidate.model,
            "condition": candidate.condition,
            "configuration_hash_full": candidate.configuration_hash_full,
            "configuration_hash_prefix": (
                candidate.configuration_hash_prefix
            ),
            "hyperparameters": hyperparameters,
            "seed_value": int(job.seed_value),
            "shd": 5,
            "sid": 10,
            "mmd_primary": 0.05,
            "graph_status": "valid_dag",
            "sampler_status": "available",
            "training_status": "converged",
            "runtime_seconds": 12.5,
            "n_iterations": None,
            "threshold_metrics": [
                {
                    "threshold": float(t),
                    "shd": 5,
                    "sid": 10,
                    "mmd_primary": 0.05,
                }
                for t in (0.2, 0.3, 0.4)
            ],
            "mmd_by_intervention": [
                {
                    "intervention_target": int(node),
                    "intervention_value": float(sign),
                    "mmd_primary": 0.05,
                }
                for node in range(10)
                for sign in (-2, 2)
            ],
            "bandwidth_summaries": {"median_heuristic": 1.0},
            "run_id": (
                f"{candidate.model}__{candidate.condition}__"
                f"calibration__seed{int(job.seed_replicate_index)}__"
                f"cfg{candidate.configuration_hash_full}"
            ),
        }

    # Pre-populate the calibration tree with a real run.
    run_calibration(
        config_dir,
        results_root,
        fit_runner=_happy_fit_runner,
        now_fn=_FixedClock(),
    )
    report = preflight_calibration(config_dir, results_root)
    assert report["artefact_status"] == "already_exists"
    assert report["records_status"] == "already_exists"


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_preflight_missing_config_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Removing one parent config makes preflight raise FileNotFoundError."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    (config_dir / "dagma_calibration_centred_only.json").unlink()
    results_root = tmp_path / "results"
    with pytest.raises(FileNotFoundError) as excinfo:
        preflight_calibration(config_dir, results_root)
    assert "dagma_calibration_centred_only.json" in str(excinfo.value)


def test_preflight_missing_config_dir_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """A non-existent config_dir raises FileNotFoundError."""
    config_dir = tmp_path / "absent"
    results_root = tmp_path / "results"
    with pytest.raises(FileNotFoundError):
        preflight_calibration(config_dir, results_root)


def test_preflight_invalid_calibration_parent_raises_value_error(
    tmp_path: Path,
) -> None:
    """A parent whose seed_population is wrong fails the guard."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    # Corrupt one parent config so it fails the calibration-stage guard.
    target = config_dir / "dagma_calibration_centred_only.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["seed_populations"] = {"calibration": [201, 999]}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_root = tmp_path / "results"
    with pytest.raises(ValueError) as excinfo:
        preflight_calibration(config_dir, results_root)
    assert "calibration" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Side-effect-free contract
# ---------------------------------------------------------------------------


def test_preflight_does_not_modify_results_root(tmp_path: Path) -> None:
    """preflight leaves the results_root unchanged."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    assert not results_root.exists()


def test_preflight_does_not_create_calibration_subtree(
    tmp_path: Path,
) -> None:
    """No calibration-tree directory is created by a fresh dry-run."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    calibration_tree = results_root / "model_selection" / "calibration"
    assert not calibration_tree.exists()


def test_preflight_does_not_modify_config_dir(tmp_path: Path) -> None:
    """The four parent config files are unchanged after preflight."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    snapshot_before = _snapshot_directory_state(config_dir)
    preflight_calibration(config_dir, results_root)
    snapshot_after = _snapshot_directory_state(config_dir)
    assert snapshot_before == snapshot_after


def test_preflight_does_not_invoke_pipeline_run_single_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight does not call pipeline.run_single_fit even once."""
    from experiments.selection_study import pipeline

    invocations = {"count": 0}

    def fake_run_single_fit(*args: Any, **kwargs: Any) -> Path:
        invocations["count"] += 1
        raise AssertionError(
            "pipeline.run_single_fit must not be invoked from a "
            "calibration preflight code path"
        )

    monkeypatch.setattr(pipeline, "run_single_fit", fake_run_single_fit)
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    assert invocations["count"] == 0


def test_preflight_does_not_invoke_rank_calibration_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight does not call rank_calibration_records."""
    from experiments.selection_study import calibration_ranking

    invocations = {"count": 0}

    def fake_rank(records: Any) -> Any:
        invocations["count"] += 1
        raise AssertionError(
            "rank_calibration_records must not be invoked from a "
            "calibration preflight code path"
        )

    monkeypatch.setattr(
        calibration_ranking, "rank_calibration_records", fake_rank
    )
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    assert invocations["count"] == 0


def test_preflight_does_not_invoke_write_selected_configurations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight does not call write_selected_configurations."""
    from experiments.selection_study import selection_artefact

    invocations = {"count": 0}

    def fake_write(*args: Any, **kwargs: Any) -> Any:
        invocations["count"] += 1
        raise AssertionError(
            "write_selected_configurations must not be invoked from "
            "a calibration preflight code path"
        )

    monkeypatch.setattr(
        selection_artefact, "write_selected_configurations", fake_write
    )
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    assert invocations["count"] == 0


def test_preflight_does_not_modify_existing_outputs(
    tmp_path: Path,
) -> None:
    """If an artefact and records/ already exist, preflight does not touch them."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def _happy_fit_runner(job: Any) -> dict[str, Any]:
        candidate = job.candidate
        return {
            "model": candidate.model,
            "condition": candidate.condition,
            "configuration_hash_full": candidate.configuration_hash_full,
            "configuration_hash_prefix": (
                candidate.configuration_hash_prefix
            ),
            "hyperparameters": dict(
                candidate.grid_point_hyperparameter
            ),
            "seed_value": int(job.seed_value),
            "shd": 5,
            "sid": 10,
            "mmd_primary": 0.05,
            "graph_status": "valid_dag",
            "sampler_status": "available",
            "training_status": "converged",
            "runtime_seconds": 12.5,
            "n_iterations": None,
            "threshold_metrics": [
                {
                    "threshold": float(t),
                    "shd": 5,
                    "sid": 10,
                    "mmd_primary": 0.05,
                }
                for t in (0.2, 0.3, 0.4)
            ],
            "mmd_by_intervention": [
                {
                    "intervention_target": int(node),
                    "intervention_value": float(sign),
                    "mmd_primary": 0.05,
                }
                for node in range(10)
                for sign in (-2, 2)
            ],
            "bandwidth_summaries": {"median_heuristic": 1.0},
            "run_id": (
                f"{candidate.model}__{candidate.condition}__"
                f"calibration__seed{int(job.seed_replicate_index)}__"
                f"cfg{candidate.configuration_hash_full}"
            ),
        }

    run_calibration(
        config_dir,
        results_root,
        fit_runner=_happy_fit_runner,
        now_fn=_FixedClock(),
    )
    snapshot_before = _snapshot_directory_state(results_root)
    preflight_calibration(config_dir, results_root)
    snapshot_after = _snapshot_directory_state(results_root)
    assert snapshot_before == snapshot_after


def test_preflight_does_not_create_log_file(tmp_path: Path) -> None:
    """No calibration_run.log appears under results_root after preflight."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    preflight_calibration(config_dir, results_root)
    log_candidates = list(results_root.rglob("calibration_run.log"))
    assert log_candidates == []


# ---------------------------------------------------------------------------
# Forbidden winner fields in report
# ---------------------------------------------------------------------------


def test_preflight_report_contains_no_forbidden_winner_fields(
    tmp_path: Path,
) -> None:
    """No winner-shaped field name appears anywhere in the report tree."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    report = preflight_calibration(config_dir, results_root)
    forbidden = {
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    }

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert key not in forbidden, (
                    f"forbidden field {key!r} in preflight report"
                )
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(report)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_dry_run_calibration_prints_valid_json(
    tmp_path: Path,
) -> None:
    """``--phase calibration --dry-run`` prints a JSON report to stdout."""
    from experiments.selection_study import run as run_module

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    captured = io.StringIO()
    with redirect_stdout(captured):
        run_module.main(
            [
                "--phase",
                "calibration",
                "--dry-run",
                "--config",
                str(config_dir),
                "--output-root",
                str(results_root),
            ]
        )
    stdout = captured.getvalue()
    report = json.loads(stdout)
    assert report["artefact_type"] == "calibration_dry_run_report"
    assert report["schema_version"] == 1


def test_cli_dry_run_calibration_report_contains_workload_and_paths(
    tmp_path: Path,
) -> None:
    """The CLI-emitted report carries workload arithmetic and planned paths."""
    from experiments.selection_study import run as run_module

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    captured = io.StringIO()
    with redirect_stdout(captured):
        run_module.main(
            [
                "--phase",
                "calibration",
                "--dry-run",
                "--config",
                str(config_dir),
                "--output-root",
                str(results_root),
            ]
        )
    report = json.loads(captured.getvalue())
    assert report["workload_summary"]["total_candidates"] == 20
    assert report["workload_summary"]["total_fit_jobs"] == 40
    assert "calibration_run_hash_full" in report
    assert "expected_artefact_path" in report
    assert "expected_records_dir" in report
    assert report["artefact_status"] in {
        "would_be_created",
        "already_exists",
    }


def test_cli_dry_run_calibration_does_not_create_results_subtree(
    tmp_path: Path,
) -> None:
    """CLI dry-run does not create the calibration results subtree."""
    from experiments.selection_study import run as run_module

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    captured = io.StringIO()
    with redirect_stdout(captured):
        run_module.main(
            [
                "--phase",
                "calibration",
                "--dry-run",
                "--config",
                str(config_dir),
                "--output-root",
                str(results_root),
            ]
        )
    assert not results_root.exists()


def test_cli_dry_run_calibration_exits_zero_via_subprocess(
    tmp_path: Path,
) -> None:
    """A subprocess invocation of the CLI dry-run exits with status 0."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.selection_study.run",
            "--phase",
            "calibration",
            "--dry-run",
            "--config",
            str(config_dir),
            "--output-root",
            str(results_root),
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess exited non-zero: stderr={result.stderr!r}"
    )
    report = json.loads(result.stdout)
    assert report["artefact_type"] == "calibration_dry_run_report"


def test_cli_dry_run_reproduction_pass_raises(tmp_path: Path) -> None:
    """``--phase reproduction_pass --dry-run`` is rejected with a clear error."""
    from experiments.selection_study import run as run_module

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        run_module.main(
            [
                "--phase",
                "reproduction_pass",
                "--dry-run",
                "--config",
                str(config_dir),
            ]
        )
    message = str(excinfo.value)
    assert "--dry-run" in message
    assert "reproduction_pass" in message


def test_cli_dry_run_calibration_failure_exits_nonzero_via_subprocess(
    tmp_path: Path,
) -> None:
    """A malformed config dir makes the subprocess CLI exit non-zero."""
    # Empty config dir: missing all four parent files.
    config_dir = tmp_path / "empty_configs"
    config_dir.mkdir()
    results_root = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.selection_study.run",
            "--phase",
            "calibration",
            "--dry-run",
            "--config",
            str(config_dir),
            "--output-root",
            str(results_root),
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        f"subprocess exited zero on missing configs: stdout="
        f"{result.stdout!r}, stderr={result.stderr!r}"
    )


def test_cli_calibration_without_dry_run_keeps_existing_behaviour(
    tmp_path: Path,
) -> None:
    """``--phase calibration`` without --dry-run still runs the enumeration helper.

    The non-dry-run code path under --phase calibration is the
    enumeration-only path inherited from Commit 9.1; it logs the
    workload via the module logger and does not invoke any model
    fit. The test verifies the CLI accepts the same flags as before
    and does not raise.
    """
    from experiments.selection_study import run as run_module

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    # ``--phase calibration`` without ``--dry-run`` is wired to the
    # enumeration helper (no fit invocation). It must accept the
    # same flags as before and return without raising.
    run_module.main(
        [
            "--phase",
            "calibration",
            "--config",
            str(config_dir),
        ]
    )
