"""Tests for the minimal selection-study run-record loader.

Exercises load_run on a synthetic fixture: valid record loads from
both a file path and a directory path; bad schema_version is
rejected; a missing mandatory field is rejected and named in the
error; a wrong-typed mandatory field is rejected and named in the
error; load_runs remains a NotImplementedError stub.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.selection_study.loader import RunRecord, load_run, load_runs


# ---------------------------------------------------------------------------
# Fixture: a minimal valid run.json payload covering every mandatory field
# ---------------------------------------------------------------------------


def _make_valid_record() -> dict:
    return {
        "run_id": (
            "dagma__centred_only__calibration__seed0__cfg"
            + ("a" * 64)
        ),
        "schema_version": 1,
        "model": "dagma",
        "condition": "centred_only",
        "seed_population": "calibration",
        "seed_replicate_index": 0,
        "configuration_hash": "a" * 64,
        "graph_seed": 11,
        "git_hash": "unknown",
        "env_snapshot": "python=3.12.0",
        "config_resolved": {"placeholder": True},
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "model_sampling_seed_base": 12345,
        "model_sampling_seed_derivation_rule": (
            "sha256_first8_bytes_mod_2pow31_purpose_label_v1"
        ),
        "train_data_seed": 7,
        "validation_data_seed": None,
        "intervention_ground_truth_seed_base": 999,
        "training_status": "converged",
        "n_iterations": None,
        "runtime_seconds": 0.1,
        "loss_history": None,
        "loss_history_status": "unavailable_no_api",
        "graph_status": "valid_dag",
        "graph_status_reason": None,
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": 1,
        "sid": 2,
        "mmd_primary": None,
        "mmd_sensitivity_unit_variance": None,
        "mmd_bandwidth_sweep": {"0.5x": None, "1.0x": None, "2.0x": None},
        "validation_nll": None,
        "sampler_status": "available",
        "sampler_status_reason": None,
        "sampler_policy_used": "residual_fitted",
        "mmd_available_count": 0,
        "mmd_missing_count": 2,
        "invalid_graph_for_this_run": False,
        "shd_reversal_cost": 2,
        "mmd_bandwidth_used_value": {},
        "mmd_clip_policy": "no_clip",
        "sid_backend": "gadjid",
        "sid_backend_version": "0.1.0",
        "sid_argument_order": "predicted_then_true",
        "sid_return_value": "raw_mistake_count",
        "configuration_hash_algorithm": "sha256_canonical_json_sorted_keys",
        "wrapper_diagnostics": {"training_status": "converged"},
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": [],
    }


def _write_record(tmp_path: Path, record: dict) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    json_path = run_dir / "run.json"
    json_path.write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return json_path


# ---------------------------------------------------------------------------
# Valid load
# ---------------------------------------------------------------------------


def test_load_run_from_file_path(tmp_path: Path) -> None:
    record = _make_valid_record()
    json_path = _write_record(tmp_path, record)
    out = load_run(json_path)
    assert isinstance(out, RunRecord)
    assert out.data["run_id"] == record["run_id"]
    assert out.path == json_path.resolve()


def test_load_run_from_directory_path(tmp_path: Path) -> None:
    record = _make_valid_record()
    json_path = _write_record(tmp_path, record)
    out = load_run(json_path.parent)
    assert isinstance(out, RunRecord)
    assert out.data["run_id"] == record["run_id"]
    assert out.path == json_path.resolve()


def test_load_run_record_is_frozen(tmp_path: Path) -> None:
    record = _make_valid_record()
    json_path = _write_record(tmp_path, record)
    out = load_run(json_path)
    with pytest.raises(Exception):
        out.path = Path("/elsewhere")  # type: ignore[misc]


def test_load_run_accepts_string_path(tmp_path: Path) -> None:
    record = _make_valid_record()
    json_path = _write_record(tmp_path, record)
    out = load_run(str(json_path))
    assert out.data["schema_version"] == 1


# ---------------------------------------------------------------------------
# Negative cases: bad schema version
# ---------------------------------------------------------------------------


def test_load_run_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    record = _make_valid_record()
    record["schema_version"] = 2
    json_path = _write_record(tmp_path, record)
    with pytest.raises(ValueError, match="schema_version"):
        load_run(json_path)


def test_load_run_rejects_string_schema_version(tmp_path: Path) -> None:
    record = _make_valid_record()
    record["schema_version"] = "1"
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError, match="schema_version"):
        load_run(json_path)


def test_load_run_rejects_bool_schema_version(tmp_path: Path) -> None:
    record = _make_valid_record()
    record["schema_version"] = True
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError, match="schema_version"):
        load_run(json_path)


def test_load_run_rejects_missing_schema_version(tmp_path: Path) -> None:
    record = _make_valid_record()
    del record["schema_version"]
    json_path = _write_record(tmp_path, record)
    with pytest.raises(ValueError, match="schema_version"):
        load_run(json_path)


# ---------------------------------------------------------------------------
# Negative cases: missing mandatory field
# ---------------------------------------------------------------------------


def test_load_run_rejects_missing_run_id_field(tmp_path: Path) -> None:
    record = _make_valid_record()
    del record["run_id"]
    json_path = _write_record(tmp_path, record)
    with pytest.raises(ValueError) as excinfo:
        load_run(json_path)
    assert "run_id" in str(excinfo.value)


def test_load_run_rejects_missing_sid_field(tmp_path: Path) -> None:
    record = _make_valid_record()
    del record["sid"]
    json_path = _write_record(tmp_path, record)
    with pytest.raises(ValueError) as excinfo:
        load_run(json_path)
    assert "sid" in str(excinfo.value)


def test_load_run_rejects_missing_interventions_field(tmp_path: Path) -> None:
    record = _make_valid_record()
    del record["interventions"]
    json_path = _write_record(tmp_path, record)
    with pytest.raises(ValueError) as excinfo:
        load_run(json_path)
    assert "interventions" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Negative cases: wrong-typed mandatory field
# ---------------------------------------------------------------------------


def test_load_run_rejects_string_shd(tmp_path: Path) -> None:
    record = _make_valid_record()
    record["shd"] = "five"
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError) as excinfo:
        load_run(json_path)
    assert "shd" in str(excinfo.value)


def test_load_run_rejects_bool_for_int_field(tmp_path: Path) -> None:
    """Bool must not satisfy an int-typed field even though bool is int subclass."""
    record = _make_valid_record()
    record["seed_replicate_index"] = True
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError) as excinfo:
        load_run(json_path)
    assert "seed_replicate_index" in str(excinfo.value)


def test_load_run_rejects_string_invalid_graph_for_this_run(
    tmp_path: Path,
) -> None:
    """invalid_graph_for_this_run is bool; a string must be rejected by name."""
    record = _make_valid_record()
    record["invalid_graph_for_this_run"] = "false"
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError) as excinfo:
        load_run(json_path)
    assert "invalid_graph_for_this_run" in str(excinfo.value)


def test_load_run_rejects_int_for_list_field(tmp_path: Path) -> None:
    record = _make_valid_record()
    record["interventions"] = 0
    json_path = _write_record(tmp_path, record)
    with pytest.raises(TypeError) as excinfo:
        load_run(json_path)
    assert "interventions" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Negative cases: I/O and JSON
# ---------------------------------------------------------------------------


def test_load_run_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "no_such_run.json")


def test_load_run_rejects_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{ not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_run(run_dir)


def test_load_run_rejects_non_object_top_level(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_run(run_dir)


# ---------------------------------------------------------------------------
# load_runs stub
# ---------------------------------------------------------------------------


def test_load_runs_remains_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_runs(None)
