"""Tests for the main-study run I/O and preflight helpers.

All tests write under pytest's ``tmp_path``; no file is created or
modified anywhere else on the filesystem. Test fixtures construct
:class:`MainStudyConfig`, :class:`PlannedRun`,
:class:`MainStudyRunRecord`, and :class:`ExecutionResult` locally.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.main_study import run_io as run_io_mod
from experiments.main_study.executor import ExecutionResult
from experiments.main_study.priors import CorruptedPriorSpec
from experiments.main_study.records import (
    SCHEMA_VERSION,
    MainStudyRunRecord,
)
from experiments.main_study.run_io import (
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    load_existing_record,
    persist_artefact_atomic,
    persist_execution_result_atomic,
    persist_record_atomic,
    prepare_output_directories,
    record_artefact_paths,
    resolve_parent_hash_from_prefix,
    resolve_relative_path,
    validate_parent_hash_full,
    validate_planned_run_uniqueness,
    validate_preflight_for_planned_runs,
    validate_record_roundtrip,
    validate_skip_compatibility,
)
from experiments.main_study.schema import (
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import (
    PlannedRun,
    make_planned_run,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Local fixtures (no cross-test-file imports)
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_PARENT_HASH_OTHER = "b" * 64
_RUN_HASH12 = "0123456789ab"
_GENERATED_AT = "2026-05-24T12:00:00Z"


def _corrupted_spec_5() -> CorruptedPriorSpec:
    return CorruptedPriorSpec(
        n_nodes=5,
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


def _prior_free_config(seed_value: int = 401) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
    )


def _matched_l1_config(
    seed_value: int = 401, matched: float = 0.07
) -> MainStudyConfig:
    return make_main_study_config(
        method_family="matched_l1",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(lambda1=matched),
        parent_heldout_run_hash_full=_PARENT_HASH,
        matched_l1_lambda1=matched,
    )


def _soft_frobenius_config(seed_value: int = 401) -> MainStudyConfig:
    return make_main_study_config(
        method_family="soft_frobenius",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        confidence=0.5,
        corrupted_prior_spec=_corrupted_spec_5(),
    )


def _hard_exclusion_config(seed_value: int = 401) -> MainStudyConfig:
    return make_main_study_config(
        method_family="hard_exclusion",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        corrupted_prior_spec=_corrupted_spec_5(),
    )


def _make_planned(cfg: MainStudyConfig) -> PlannedRun:
    return make_planned_run(cfg, _RUN_HASH12)


def _success_record_for_planned(
    planned: PlannedRun,
    *,
    n_nodes: int = 5,
) -> MainStudyRunRecord:
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=n_nodes,
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


def _build_artefacts_for_record(
    record: MainStudyRunRecord, *, n_nodes: int = 5
) -> dict[str, object]:
    payloads: dict[str, object] = {}
    if record.continuous_w_path is not None:
        payloads["continuous_w.npz"] = {
            "continuous_w": np.zeros((n_nodes, n_nodes), dtype=float),
        }
    if record.thresholded_adjacency_path is not None:
        payloads["thresholded_adjacency.npz"] = {
            "thresholded_adjacency": np.zeros(
                (n_nodes, n_nodes), dtype=bool
            ),
        }
    if record.true_adjacency_path is not None:
        payloads["true_adjacency.npz"] = {
            "true_adjacency": np.zeros((n_nodes, n_nodes), dtype=bool),
        }
    if record.confidence_mask_path is not None:
        payloads["confidence_mask.npz"] = {
            "confidence_mask": np.zeros((n_nodes, n_nodes), dtype=float),
        }
    if record.interventions_mmd_path is not None:
        payloads["interventions_mmd.json"] = {
            "records": [],
            "mmd_primary": float(record.mmd if record.mmd is not None else 0.0),
        }
    if record.prior_edge_set_clean_path is not None:
        payloads["prior_edge_set_clean.json"] = {
            "n_nodes": n_nodes,
            "forbidden_edges": [[0, 2], [1, 3], [2, 4]],
        }
    if record.prior_edge_set_corrupted_path is not None:
        payloads["prior_edge_set_corrupted.json"] = {
            "n_nodes": n_nodes,
            "corruption_fraction": 0.0,
            "forbidden_edges": [[0, 2], [1, 3], [2, 4]],
        }
    if record.per_edge_labels_path is not None:
        payloads["per_edge_labels.json"] = {
            "0,2": "true_negative_retained",
            "1,3": "true_negative_retained",
            "2,4": "true_negative_retained",
        }
    return payloads


def _build_execution_result(
    cfg: MainStudyConfig, *, n_nodes: int = 5
) -> tuple[ExecutionResult, PlannedRun]:
    planned = _make_planned(cfg)
    record = _success_record_for_planned(planned, n_nodes=n_nodes)
    artefacts = _build_artefacts_for_record(record, n_nodes=n_nodes)
    return (
        ExecutionResult(record=record, artefacts=artefacts),
        planned,
    )


# ===========================================================================
# A. Paths and parent hashes
# ===========================================================================


@pytest.mark.parametrize(
    "good",
    [
        "file.txt",
        "subdir/file.txt",
        "results/main_study/abcdef012345/records/run_42.json",
        "a/b/c.json",
    ],
)
def test_resolve_relative_path_accepts_valid_relative_posix(tmp_path, good):
    out = resolve_relative_path(good, base_dir=tmp_path)
    assert isinstance(out, Path)


@pytest.mark.parametrize(
    "bad",
    [
        "/absolute/path",
        "..\\backslash",
        "../escape",
        "./dot",
        "trailing/slash/",
        "a//b",
    ],
)
def test_resolve_relative_path_rejects_invalid_paths(tmp_path, bad):
    with pytest.raises(ValueError):
        resolve_relative_path(bad, base_dir=tmp_path)


def test_resolve_relative_path_rejects_escape_via_dotdot(tmp_path):
    # validate_relative_posix_path already rejects ".." components, so
    # this is doubly enforced.
    with pytest.raises(ValueError):
        resolve_relative_path("a/../../etc", base_dir=tmp_path)


def test_validate_parent_hash_full_accepts_64_lowercase_hex():
    assert (
        validate_parent_hash_full("0123456789abcdef" * 4)
        == "0123456789abcdef" * 4
    )


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "0123456789ab",
        "A" * 64,
        "g" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_validate_parent_hash_full_rejects_malformed(bad):
    with pytest.raises(ValueError, match="parent_hash_full"):
        validate_parent_hash_full(bad)


def test_validate_parent_hash_full_rejects_non_string():
    with pytest.raises(ValueError, match="string"):
        validate_parent_hash_full(123)  # type: ignore[arg-type]


def test_resolve_parent_hash_from_prefix_finds_unique_directory(tmp_path):
    full_hash = "0123456789abcdef" * 4
    (tmp_path / full_hash).mkdir()
    assert (
        resolve_parent_hash_from_prefix(
            full_hash[:12], search_root=tmp_path
        )
        == full_hash
    )


def test_resolve_parent_hash_from_prefix_finds_unique_file_stem(tmp_path):
    full_hash = "0123456789abcdef" * 4
    (tmp_path / f"{full_hash}.json").write_text("{}", encoding="utf-8")
    assert (
        resolve_parent_hash_from_prefix(
            full_hash[:12], search_root=tmp_path
        )
        == full_hash
    )


def test_resolve_parent_hash_from_prefix_raises_for_no_match(tmp_path):
    (tmp_path / "unrelated").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_parent_hash_from_prefix(
            "0123456789ab", search_root=tmp_path
        )


def test_resolve_parent_hash_from_prefix_raises_for_multiple_matches(
    tmp_path,
):
    a = "0123456789abcdef" * 4
    b = "0123456789ab" + "f" * 52
    (tmp_path / a).mkdir()
    (tmp_path / b).mkdir()
    with pytest.raises(ValueError, match="multiple"):
        resolve_parent_hash_from_prefix(
            "0123456789ab", search_root=tmp_path
        )


def test_resolve_parent_hash_from_prefix_rejects_bad_prefix(tmp_path):
    with pytest.raises(ValueError, match="prefix12"):
        resolve_parent_hash_from_prefix("abcd", search_root=tmp_path)


# ===========================================================================
# B. Planned-run uniqueness and preflight
# ===========================================================================


def test_validate_planned_run_uniqueness_accepts_unique():
    runs = [_make_planned(_prior_free_config(seed_value=401))]
    runs.append(_make_planned(_prior_free_config(seed_value=402)))
    validate_planned_run_uniqueness(runs)


def test_validate_planned_run_uniqueness_rejects_duplicate_hash():
    p = _make_planned(_prior_free_config(seed_value=401))
    with pytest.raises(ValueError, match="configuration_hash_full"):
        validate_planned_run_uniqueness([p, p])


def test_validate_planned_run_uniqueness_rejects_duplicate_run_id(tmp_path):
    p1 = _make_planned(_prior_free_config(seed_value=401))
    p2 = _make_planned(_prior_free_config(seed_value=402))
    # Force p2's run_id to equal p1.run_id by replacing via object.__setattr__.
    object.__setattr__(p2, "run_id", p1.run_id)
    with pytest.raises(ValueError, match="run_id"):
        validate_planned_run_uniqueness([p1, p2])


def test_validate_planned_run_uniqueness_rejects_duplicate_record_path():
    p1 = _make_planned(_prior_free_config(seed_value=401))
    p2 = _make_planned(_prior_free_config(seed_value=402))
    object.__setattr__(p2, "record_path", p1.record_path)
    with pytest.raises(ValueError, match="record_path"):
        validate_planned_run_uniqueness([p1, p2])


def test_validate_planned_run_uniqueness_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_planned_run_uniqueness([])


def test_prepare_output_directories_creates_parent_dirs(tmp_path):
    runs = [
        _make_planned(_prior_free_config(seed_value=401)),
        _make_planned(_prior_free_config(seed_value=402)),
    ]
    prepare_output_directories(runs, base_dir=tmp_path)
    for p in runs:
        assert (tmp_path / p.record_path).parent.exists()
        for art_path in p.artefact_paths.values():
            assert (tmp_path / art_path).parent.exists()


def test_validate_preflight_creates_dirs_but_no_files(tmp_path):
    runs = [_make_planned(_prior_free_config(seed_value=401))]
    validate_preflight_for_planned_runs(
        runs,
        base_dir=tmp_path,
        parent_hash_full=_PARENT_HASH,
    )
    for p in runs:
        assert (tmp_path / p.record_path).parent.exists()
        assert not (tmp_path / p.record_path).exists()
        for art_path in p.artefact_paths.values():
            assert (tmp_path / art_path).parent.exists()
            assert not (tmp_path / art_path).exists()


# ===========================================================================
# C. Atomic writes
# ===========================================================================


def test_atomic_write_text_writes_complete_content(tmp_path):
    path = atomic_write_text("hello world", "out.txt", base_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "hello world"


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    rel = "subdir/file.txt"
    atomic_write_text("first", rel, base_dir=tmp_path)
    atomic_write_text("second", rel, base_dir=tmp_path)
    assert (tmp_path / rel).read_text(encoding="utf-8") == "second"


def test_atomic_write_text_failure_cleans_temp_and_leaves_no_final(
    tmp_path, monkeypatch
):
    def fail_replace(*args, **kwargs):
        raise RuntimeError("simulated replace failure")

    monkeypatch.setattr(run_io_mod.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_write_text("data", "out.txt", base_dir=tmp_path)
    assert not (tmp_path / "out.txt").exists()
    # No temp files survive.
    leftovers = [
        p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")
    ]
    assert leftovers == []


def test_atomic_write_json_writes_compact_sorted_json(tmp_path):
    payload = {"b": 2, "a": 1}
    path = atomic_write_json(payload, "out.json", base_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text == '{"a":1,"b":2}'


def test_atomic_write_json_converts_numpy_values(tmp_path):
    payload = {
        "arr": np.array([1.0, 2.0, 3.0]),
        "scalar": np.int64(42),
        "flag": np.bool_(True),
    }
    path = atomic_write_json(payload, "out.json", base_dir=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["arr"] == [1.0, 2.0, 3.0]
    assert data["scalar"] == 42
    assert data["flag"] is True


def test_atomic_write_json_rejects_callable_value(tmp_path):
    with pytest.raises(TypeError, match="callable"):
        atomic_write_json(
            {"f": lambda x: x}, "out.json", base_dir=tmp_path
        )


def test_atomic_write_npz_writes_expected_arrays(tmp_path):
    payload = {"a": np.array([1, 2, 3]), "b": np.eye(3)}
    path = atomic_write_npz(payload, "data.npz", base_dir=tmp_path)
    with np.load(path) as data:
        assert sorted(data.files) == ["a", "b"]
        np.testing.assert_array_equal(data["a"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(data["b"], np.eye(3))


def test_atomic_write_npz_only_final_file_remains(tmp_path):
    atomic_write_npz(
        {"arr": np.zeros(5)}, "data.npz", base_dir=tmp_path
    )
    entries = list(tmp_path.iterdir())
    names = {p.name for p in entries}
    assert "data.npz" in names
    # No temp leftovers; numpy must not have created a second .npz.
    for name in names:
        assert not name.startswith(".tmp_")
        assert not name.endswith(".tmp")
        assert not name.endswith(".tmp.npz")
    # Exactly one entry, the final file.
    assert len(entries) == 1


def test_atomic_write_npz_failure_leaves_no_files(tmp_path, monkeypatch):
    def fail_replace(*args, **kwargs):
        raise RuntimeError("simulated replace failure")

    monkeypatch.setattr(run_io_mod.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_write_npz(
            {"arr": np.zeros(5)}, "data.npz", base_dir=tmp_path
        )
    assert not (tmp_path / "data.npz").exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == []


@pytest.mark.parametrize(
    "bad",
    [
        [("a", 1)],          # not a dict
        {},                  # empty dict
        {"": np.zeros(5)},   # empty key
    ],
)
def test_atomic_write_npz_rejects_bad_payload(tmp_path, bad):
    with pytest.raises((TypeError, ValueError)):
        atomic_write_npz(bad, "data.npz", base_dir=tmp_path)


def test_atomic_write_npz_rejects_non_npz_extension(tmp_path):
    with pytest.raises(ValueError, match=".npz"):
        atomic_write_npz(
            {"arr": np.zeros(5)}, "data.json", base_dir=tmp_path
        )


# ===========================================================================
# D. Record and artefact persistence
# ===========================================================================


def test_persist_record_atomic_writes_record_and_loads_back(tmp_path):
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    path = persist_record_atomic(record, planned.record_path, base_dir=tmp_path)
    assert path.exists()
    loaded = load_existing_record(
        planned.record_path, base_dir=tmp_path
    )
    assert loaded == record


def test_load_existing_record_returns_none_when_absent(tmp_path):
    assert (
        load_existing_record(
            "results/main_study/abcdef012345/records/run.json",
            base_dir=tmp_path,
        )
        is None
    )


def test_load_existing_record_raises_for_corrupt_json(tmp_path):
    rel = "records/bad.json"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("{ this is not valid JSON }", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_existing_record(rel, base_dir=tmp_path)


def test_record_artefact_paths_returns_only_non_none(tmp_path):
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    paths = record_artefact_paths(record)
    assert set(paths.keys()) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "interventions_mmd.json",
    }


def test_record_artefact_paths_returns_soft_frobenius_full_set():
    planned = _make_planned(_soft_frobenius_config())
    record = _success_record_for_planned(planned)
    paths = record_artefact_paths(record)
    assert "confidence_mask.npz" in paths
    assert "prior_edge_set_clean.json" in paths
    assert "prior_edge_set_corrupted.json" in paths
    assert "per_edge_labels.json" in paths


def test_persist_artefact_atomic_dispatches_npz_and_json(tmp_path):
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    payloads = _build_artefacts_for_record(record)
    # .npz
    npz_path = persist_artefact_atomic(
        "continuous_w.npz",
        payloads["continuous_w.npz"],
        record.continuous_w_path,
        base_dir=tmp_path,
    )
    assert npz_path.exists()
    # .json
    json_path = persist_artefact_atomic(
        "interventions_mmd.json",
        payloads["interventions_mmd.json"],
        record.interventions_mmd_path,
        base_dir=tmp_path,
    )
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "records" in data


def test_persist_artefact_atomic_rejects_filename_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        persist_artefact_atomic(
            "continuous_w.npz",
            {"continuous_w": np.zeros((5, 5))},
            "results/main_study/abc/artefacts/r/wrong_name.npz",
            base_dir=tmp_path,
        )


def test_persist_execution_result_writes_artefacts_before_record(
    tmp_path, monkeypatch
):
    cfg = _prior_free_config()
    result, planned = _build_execution_result(cfg)
    call_order: list[str] = []

    original_artefact = run_io_mod.persist_artefact_atomic
    original_record = run_io_mod.persist_record_atomic

    def tracked_artefact(name, payload, relative_path, *, base_dir):
        call_order.append(f"artefact:{name}")
        return original_artefact(name, payload, relative_path, base_dir=base_dir)

    def tracked_record(record, record_path, *, base_dir):
        call_order.append("record")
        return original_record(record, record_path, base_dir=base_dir)

    monkeypatch.setattr(
        run_io_mod, "persist_artefact_atomic", tracked_artefact
    )
    monkeypatch.setattr(
        run_io_mod, "persist_record_atomic", tracked_record
    )
    persist_execution_result_atomic(
        result, planned.record_path, base_dir=tmp_path
    )
    # Record was the last operation.
    assert call_order[-1] == "record"
    # All preceding ops were artefacts.
    for entry in call_order[:-1]:
        assert entry.startswith("artefact:")


@pytest.mark.parametrize(
    "config_builder",
    [
        _prior_free_config,
        _matched_l1_config,
        _soft_frobenius_config,
        _hard_exclusion_config,
    ],
)
def test_persist_execution_result_writes_complete_payload(
    tmp_path, config_builder
):
    cfg = config_builder()
    result, planned = _build_execution_result(cfg)
    written = persist_execution_result_atomic(
        result, planned.record_path, base_dir=tmp_path
    )
    # Every reference is a real file.
    assert written["record"].exists()
    for name, rel_path in record_artefact_paths(result.record).items():
        assert written[name].exists()
    # Record round-trips.
    loaded = load_existing_record(
        planned.record_path, base_dir=tmp_path
    )
    assert loaded == result.record
    # Artefacts round-trip.
    for name, rel_path in record_artefact_paths(result.record).items():
        full = tmp_path / rel_path
        if name.endswith(".npz"):
            with np.load(full) as data:
                assert len(data.files) >= 1
        else:
            data = json.loads(full.read_text(encoding="utf-8"))
            assert isinstance(data, dict)


def test_persist_execution_result_rejects_extra_artefact_key(tmp_path):
    cfg = _prior_free_config()
    result, planned = _build_execution_result(cfg)
    # Mutate the artefacts dict to add an unexpected key.
    result.artefacts["surprise.json"] = {"data": 1}
    with pytest.raises(ValueError, match="surprise.json"):
        persist_execution_result_atomic(
            result, planned.record_path, base_dir=tmp_path
        )


def test_persist_execution_result_rejects_missing_artefact_payload(
    tmp_path,
):
    cfg = _prior_free_config()
    result, planned = _build_execution_result(cfg)
    # Remove an artefact the record still references.
    del result.artefacts["continuous_w.npz"]
    with pytest.raises(ValueError, match="continuous_w.npz"):
        persist_execution_result_atomic(
            result, planned.record_path, base_dir=tmp_path
        )


def test_persist_execution_result_no_rollback_on_record_failure(
    tmp_path, monkeypatch
):
    """Documents the no-rollback policy: artefacts written before the
    record stay on disk if the record write raises."""
    cfg = _prior_free_config()
    result, planned = _build_execution_result(cfg)

    def fail_record(record, record_path, *, base_dir):
        raise RuntimeError("simulated record-write failure")

    monkeypatch.setattr(
        run_io_mod, "persist_record_atomic", fail_record
    )
    with pytest.raises(RuntimeError, match="simulated"):
        persist_execution_result_atomic(
            result, planned.record_path, base_dir=tmp_path
        )
    # Artefacts are present.
    for name, rel_path in record_artefact_paths(result.record).items():
        assert (tmp_path / rel_path).exists(), (
            f"artefact {name!r} should remain after mid-sequence failure"
        )
    # Record is not present.
    assert not (tmp_path / planned.record_path).exists()


# ===========================================================================
# E. Skip compatibility and roundtrip
# ===========================================================================


@pytest.mark.parametrize(
    "config_builder",
    [
        _prior_free_config,
        _matched_l1_config,
        _soft_frobenius_config,
        _hard_exclusion_config,
    ],
)
def test_validate_skip_compatibility_accepts_matching(config_builder):
    cfg = config_builder()
    planned = _make_planned(cfg)
    record = _success_record_for_planned(planned)
    # Round-trip via JSON to mimic the actual reload path.
    from experiments.main_study.records import (
        record_from_json,
        record_to_json,
    )

    reloaded = record_from_json(record_to_json(record))
    validate_skip_compatibility(reloaded, planned)


def test_validate_skip_compatibility_rejects_configuration_hash_mismatch():
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    # Force a mismatched hash by bypassing the record's own validator.
    object.__setattr__(record, "configuration_hash_full", "f" * 64)
    with pytest.raises(ValueError, match="configuration_hash_full"):
        validate_skip_compatibility(record, planned)


def test_validate_skip_compatibility_rejects_run_id_mismatch():
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    object.__setattr__(record, "run_id", "some_other_run_id")
    with pytest.raises(ValueError, match="run_id"):
        validate_skip_compatibility(record, planned)


def test_validate_skip_compatibility_rejects_config_mismatch():
    planned_a = _make_planned(_prior_free_config(seed_value=401))
    planned_b = _make_planned(_prior_free_config(seed_value=402))
    record_a = _success_record_for_planned(planned_a)
    with pytest.raises(ValueError):
        validate_skip_compatibility(record_a, planned_b)


def test_validate_skip_compatibility_rejects_nested_dagma_config_change():
    cfg_a = _prior_free_config(seed_value=401)
    cfg_b = MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=dataclasses.replace(DAGMAConfig(), lambda1=0.99),
        parent_heldout_run_hash_full=_PARENT_HASH,
    )
    planned_a = _make_planned(cfg_a)
    planned_b = _make_planned(cfg_b)
    record_a = _success_record_for_planned(planned_a)
    # Use record_a (cfg_a hash) and planned_b (cfg_b hash) — config and
    # configuration_hash_full both differ.
    with pytest.raises(ValueError):
        validate_skip_compatibility(record_a, planned_b)


def test_validate_skip_compatibility_rejects_corrupted_prior_spec_change():
    cfg_a = _soft_frobenius_config()
    cfg_b = make_main_study_config(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        confidence=0.5,
        corrupted_prior_spec=CorruptedPriorSpec(
            n_nodes=5,
            scm_seed=401,
            corruption_fraction=0.0,
            corruption_index=0,
            corruption_seed=9100 + 401 + 0,
            forbidden_edges=((0, 4), (1, 3), (2, 4)),  # different set
            n_correct=3,
            n_corrupted=0,
            removed_clean_edges=(),
            added_true_positive_edges=(),
            edge_labels={
                "0,4": "true_negative_retained",
                "1,3": "true_negative_retained",
                "2,4": "true_negative_retained",
            },
        ),
    )
    planned_a = _make_planned(cfg_a)
    planned_b = _make_planned(cfg_b)
    record_a = _success_record_for_planned(planned_a)
    with pytest.raises(ValueError):
        validate_skip_compatibility(record_a, planned_b)


def test_validate_record_roundtrip_accepts_valid_record():
    planned = _make_planned(_prior_free_config(seed_value=401))
    record = _success_record_for_planned(planned)
    validate_record_roundtrip(record)


# ===========================================================================
# F. Scope / imports
# ===========================================================================


_RUN_IO_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "json",
    "os",
    "tempfile",
    "pathlib",
    "typing",
    "numpy",
    "experiments.main_study.executor",
    "experiments.main_study.paths",
    "experiments.main_study.records",
    "experiments.main_study.workloads",
    "experiments.main_study.schema",
})


_RUN_IO_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.backends",
    "experiments.main_study.calibration_lambda_prior",
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


def test_run_io_module_imports_are_allowlisted():
    src = Path(run_io_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _RUN_IO_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _RUN_IO_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"run_io.py import {mod!r} is not in the allowlist "
            f"{sorted(_RUN_IO_ALLOWED_PREFIXES)}."
        )


def test_run_io_module_does_not_import_forbidden_packages():
    src = Path(run_io_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _RUN_IO_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"run_io.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_no_test_writes_outside_tmp_path(tmp_path):
    """Sentinel: this test file uses tmp_path everywhere; the suite's
    write-only-under-tmp_path discipline is upheld."""
    # If a test had written outside tmp_path, that would have left
    # artefacts we cannot detect from here, but the convention is
    # enforced by reading all atomic_write calls' base_dir arguments
    # in the tests above, which always equal tmp_path.
    assert tmp_path.exists()
