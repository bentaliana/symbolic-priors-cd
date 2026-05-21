"""Tests for the reproduction-pass runner.

The runner orchestrates ``load_config``, the real-study protocol
guard, manifest enumeration, manifest validation, the
schema-conformance pipeline, threshold-robustness recomputation,
and the reproduction-pass summary writer. These tests mock the
pipeline and threshold-robustness functions so they do not call
wrapper code, DAGMA, or DCDI; the runner's orchestration logic is
the unit under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study import (
    reproduction_pass as reproduction_pass_module,
)
from experiments.selection_study.config import load_config
from experiments.selection_study.identity import derive_run_directory
from experiments.selection_study.reproduction_pass import (
    ReproductionPassRunRecord,
    ReproductionPassSummary,
    run_reproduction_pass,
)
from experiments.selection_study.pipeline import (
    InvalidGraphForSchemaGateError,
    SchemaGateError,
)
from experiments.selection_study.preflight import Manifest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPRODUCTION_CONFIG_DIR = (
    _PROJECT_ROOT
    / "experiments"
    / "selection_study"
    / "configs"
    / "reproduction"
)
_DAGMA_PATH = _REPRODUCTION_CONFIG_DIR / "dagma_reproduction.json"
_DCDI_PATH = _REPRODUCTION_CONFIG_DIR / "dcdi_reproduction.json"


# ---------------------------------------------------------------------------
# Stub run-record builder
# ---------------------------------------------------------------------------


def _stub_run_record(
    *,
    manifest: Manifest,
    entry_index: int,
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
    shd_value: int = 0,
    sid_value: int = 0,
    mmd_primary: float | None = 0.5,
) -> dict[str, Any]:
    """Build a minimal-valid run.json record matching the loader schema."""
    entry = manifest.entries[entry_index]
    resolved = dict(manifest.resolved_config)
    return {
        "run_id": entry.expected_run_id,
        "schema_version": 1,
        "model": entry.model,
        "condition": entry.condition,
        "seed_population": entry.seed_population,
        "seed_replicate_index": int(entry.seed_replicate_index),
        "configuration_hash": entry.configuration_hash,
        "graph_seed": int(entry.graph_seed),
        "git_hash": "stub",
        "env_snapshot": "stub",
        "config_resolved": resolved,
        "seed_torch": (
            None if resolved.get("seed_torch") is None else int(resolved["seed_torch"])
        ),
        "seed_numpy": (
            None if resolved.get("seed_numpy") is None else int(resolved["seed_numpy"])
        ),
        "seed_dagma": (
            None if resolved.get("seed_dagma") is None else int(resolved["seed_dagma"])
        ),
        "model_sampling_seed_base": int(entry.model_sampling_seed_base),
        "model_sampling_seed_derivation_rule": (
            manifest.seed_derivation_rule
        ),
        "train_data_seed": int(entry.train_data_seed),
        "validation_data_seed": (
            None
            if entry.validation_data_seed is None
            else int(entry.validation_data_seed)
        ),
        "intervention_ground_truth_seed_base": int(
            entry.intervention_ground_truth_seed_base
        ),
        "training_status": "max_iter",
        "n_iterations": None,
        "runtime_seconds": 0.0,
        "loss_history": None,
        "loss_history_status": "unavailable_no_api",
        "graph_status": graph_status,
        "graph_status_reason": (
            None if graph_status == "valid_dag" else "stub-invalid"
        ),
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": int(shd_value),
        "sid": int(sid_value),
        "mmd_primary": mmd_primary,
        "mmd_sensitivity_unit_variance": None,
        "mmd_bandwidth_sweep": {},
        "validation_nll": None,
        "sampler_status": sampler_status,
        "sampler_status_reason": (
            None if sampler_status == "available" else "stub-unavailable"
        ),
        "sampler_policy_used": (
            "residual_fitted" if entry.model == "dagma" else "dcdi_native"
        ),
        "mmd_available_count": 0,
        "mmd_missing_count": 0,
        "invalid_graph_for_this_run": graph_status != "valid_dag",
        "shd_reversal_cost": 2,
        "mmd_bandwidth_used_value": {},
        "mmd_clip_policy": "no_clip",
        "sid_backend": "gadjid",
        "sid_backend_version": "0.1.0",
        "sid_argument_order": "predicted_then_true",
        "sid_return_value": "raw_mistake_count",
        "configuration_hash_algorithm": (
            "sha256_canonical_json_sorted_keys"
        ),
        "wrapper_diagnostics": {},
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": [],
    }


def _make_fake_pipeline(
    *,
    failures_by_index: dict[int, Exception] | None = None,
    shd_by_index: dict[int, int] | None = None,
    sid_by_index: dict[int, int] | None = None,
    graph_status_by_index: dict[int, str] | None = None,
    sampler_status_by_index: dict[int, str] | None = None,
    skip_threshold_sibling_indices: tuple[int, ...] = (),
):
    """Build a fake ``run_single_fit`` / ``recompute_at_thresholds`` pair.

    The pair writes a minimal-valid ``run.json`` to the canonical
    run directory and (optionally) a sibling
    ``threshold_robustness.json``. Specific entry indices can be
    asked to raise a chosen exception, to write a record with
    non-default ``graph_status`` or ``sampler_status``, or to skip
    the sibling write to simulate a recompute that did not persist.
    """
    failures = dict(failures_by_index or {})
    shd_map = dict(shd_by_index or {})
    sid_map = dict(sid_by_index or {})
    graph_status_map = dict(graph_status_by_index or {})
    sampler_status_map = dict(sampler_status_by_index or {})
    skip_sibling = set(int(i) for i in skip_threshold_sibling_indices)
    fit_order: list[int] = []
    fit_dirs: dict[int, Path] = {}

    def fake_run_single_fit(
        manifest: Manifest,
        entry_index: int,
        *,
        run_root: Path,
    ) -> Path:
        if entry_index in failures:
            raise failures[entry_index]
        entry = manifest.entries[entry_index]
        run_dir = derive_run_directory(
            model=entry.model,
            condition=entry.condition,
            seed_population=entry.seed_population,
            seed_replicate_index=entry.seed_replicate_index,
            configuration_hash=entry.configuration_hash,
            base_dir=run_root,
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        record = _stub_run_record(
            manifest=manifest,
            entry_index=entry_index,
            graph_status=graph_status_map.get(entry_index, "valid_dag"),
            sampler_status=sampler_status_map.get(
                entry_index, "available"
            ),
            shd_value=shd_map.get(entry_index, 0),
            sid_value=sid_map.get(entry_index, 0),
        )
        run_json_path = run_dir / "run.json"
        run_json_path.write_text(
            json.dumps(record, sort_keys=True, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        fit_order.append(entry_index)
        fit_dirs[entry_index] = run_dir
        return run_json_path

    def fake_recompute_at_thresholds(
        run_dir: Path | str,
        *,
        write_sibling: bool = True,
    ) -> dict[str, Any]:
        path = Path(run_dir)
        directory = path.parent if path.is_file() else path
        # Look up which entry_index this directory belongs to so the
        # per-index skip rule can fire.
        index_for_dir: int | None = None
        for idx, recorded_dir in fit_dirs.items():
            if recorded_dir == directory:
                index_for_dir = idx
                break
        out: dict[str, Any] = {
            "run_id": "stub",
            "model": "stub",
            "condition": "stub",
            "configuration_hash": "stub",
            "continuous_edge_object_artefact": "continuous_edge_object.npz",
            "threshold_triple": [0.0, 0.0, 0.0],
            "primary_threshold": 0.0,
            "primary_threshold_index": 1,
            "shd_reversal_cost": 2,
            "records": [],
        }
        skip_for_this = (
            index_for_dir is not None and index_for_dir in skip_sibling
        )
        if write_sibling and not skip_for_this:
            sibling_path = directory / "threshold_robustness.json"
            sibling_path.write_text(
                json.dumps(
                    out, sort_keys=True, ensure_ascii=True, indent=2
                ),
                encoding="utf-8",
            )
        return out

    return fake_run_single_fit, fake_recompute_at_thresholds


# ---------------------------------------------------------------------------
# Happy path: full end-to-end with mocks
# ---------------------------------------------------------------------------


def _patch_pipeline(monkeypatch, fake_fit, fake_recompute) -> None:
    """Install the fake pipeline and threshold-recompute in reproduction_pass."""
    monkeypatch.setattr(
        reproduction_pass_module, "run_single_fit", fake_fit
    )
    monkeypatch.setattr(
        reproduction_pass_module, "recompute_at_thresholds", fake_recompute
    )


def test_run_reproduction_pass_dagma_completes_with_passed_status(
    tmp_path, monkeypatch
) -> None:
    """Three reproduction entries complete; status is 'passed'."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert isinstance(summary, ReproductionPassSummary)
    assert summary.model == "dagma"
    assert summary.condition == "centred_only"
    assert summary.seed_population == "reproduction"
    assert summary.seed_values == (101, 102, 103)
    assert summary.completed_run_count == 3
    assert summary.failed_run_count == 0
    assert summary.reproduction_pass_status == "passed"
    assert summary.threshold_robustness_available_count == 3
    assert len(summary.records) == 3
    assert all(r.status == "completed" for r in summary.records)


def test_run_reproduction_pass_dcdi_completes_with_passed_status(
    tmp_path, monkeypatch
) -> None:
    """DCDI reproduction pass also runs end-to-end through the mocked pipeline."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DCDI_PATH, output_root=tmp_path)

    assert summary.model == "dcdi"
    assert summary.seed_values == (101, 102, 103)
    assert summary.completed_run_count == 3
    assert summary.reproduction_pass_status == "passed"


def test_run_reproduction_pass_writes_summary_at_canonical_path(
    tmp_path, monkeypatch
) -> None:
    """Summary lives at <output_root>/reproduction_pass_summary/<prefix>/file.json.

    The directory leaf is the first 12 characters of the
    ``configuration_hash``, matching the per-run directory leaf
    convention.
    """
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    expected_dir = (
        tmp_path
        / "reproduction_pass_summary"
        / summary.configuration_hash[:12]
    )
    expected_file = expected_dir / "reproduction_pass_summary.json"
    assert Path(summary.summary_path) == expected_file
    assert expected_file.is_file()
    payload = json.loads(expected_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model"] == "dagma"
    assert payload["reproduction_pass_status"] == "passed"
    assert payload["completed_run_count"] == 3


def test_run_reproduction_pass_summary_field_carries_full_hash(
    tmp_path, monkeypatch
) -> None:
    """The summary's configuration_hash field is the full 64-char digest.

    The directory leaf uses the 12-character prefix, but every
    on-disk record of the configuration identity keeps the full
    SHA-256 hex digest. This regression pins both the in-memory
    field and the on-disk JSON value to the full 64-character
    lowercase hex form.
    """
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert len(summary.configuration_hash) == 64
    assert all(
        ch in "0123456789abcdef" for ch in summary.configuration_hash
    )
    payload = json.loads(
        Path(summary.summary_path).read_text(encoding="utf-8")
    )
    assert payload["configuration_hash"] == summary.configuration_hash
    assert len(payload["configuration_hash"]) == 64


def test_run_reproduction_pass_summary_records_carry_run_metrics(
    tmp_path, monkeypatch
) -> None:
    """Per-entry records carry graph/sampler/training status and metrics."""
    fake_fit, fake_recompute = _make_fake_pipeline(
        shd_by_index={0: 3, 1: 5, 2: 7},
        sid_by_index={0: 1, 1: 2, 2: 3},
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert summary.shd_values == (3, 5, 7)
    assert summary.sid_values == (1, 2, 3)
    assert summary.mmd_primary_values == (0.5, 0.5, 0.5)
    assert summary.graph_status_counts == {"valid_dag": 3}
    assert summary.training_status_counts == {"max_iter": 3}
    assert summary.sampler_status_counts == {"available": 3}


def test_run_reproduction_pass_run_ids_match_reproduction_entries(
    tmp_path, monkeypatch
) -> None:
    """The summary's run_ids equal the manifest's reproduction run_ids."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    config = load_config(_DAGMA_PATH)
    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert all(
        rid.startswith("dagma__centred_only__reproduction")
        for rid in summary.run_ids
    )
    assert len(summary.run_ids) == 3
    assert len(set(summary.run_ids)) == 3
    # The configuration_hash recorded by the summary matches a
    # second derivation from the loaded Configuration.
    from experiments.selection_study.config import (
        configuration_hash as compute_configuration_hash,
    )
    assert summary.configuration_hash == compute_configuration_hash(config)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_run_reproduction_pass_rejects_config_failing_real_study_guard(
    tmp_path, monkeypatch
) -> None:
    """A configuration off the protocol anchors fails the guard."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    # Build a toy config that loads but fails the guard. Easiest:
    # write a JSON variant of the DAGMA config with n_nodes flipped
    # to the toy value 3.
    raw = json.loads(_DAGMA_PATH.read_text(encoding="utf-8"))
    raw["n_nodes"] = 3
    raw["expected_edges"] = 3
    raw["n_train"] = 64
    raw["mmd_n_samples"] = 64
    toy_path = tmp_path / "toy_dagma.json"
    toy_path.write_text(
        json.dumps(raw, ensure_ascii=True), encoding="utf-8"
    )

    with pytest.raises(ValueError) as excinfo:
        run_reproduction_pass(toy_path, output_root=tmp_path / "runs")
    # The real-study guard prefixes its violation messages with the
    # canonical "real-study protocol violation" phrase; asserting on
    # that substring keeps this test independent of the guard's
    # internal stage label.
    assert "real-study protocol violation" in str(excinfo.value)


def test_run_reproduction_pass_records_schema_gate_failure_as_warning(
    tmp_path, monkeypatch
) -> None:
    """When the first entry hits a schema-gate stop, status flips."""
    failure = InvalidGraphForSchemaGateError(
        "stub: thresholded graph is bidirected"
    )
    fake_fit, fake_recompute = _make_fake_pipeline(
        failures_by_index={0: failure}
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert summary.completed_run_count == 2
    assert summary.failed_run_count == 1
    assert summary.reproduction_pass_status == "completed_with_warnings"

    failed_records = [r for r in summary.records if r.status == "failed"]
    assert len(failed_records) == 1
    assert failed_records[0].failure_type == (
        "InvalidGraphForSchemaGateError"
    )
    assert "bidirected" in (failed_records[0].failure_message or "")


def test_run_reproduction_pass_propagates_non_schema_gate_exceptions(
    tmp_path, monkeypatch
) -> None:
    """Generic exceptions are not swallowed; they propagate."""
    failure = RuntimeError("stub: unexpected error")
    fake_fit, fake_recompute = _make_fake_pipeline(
        failures_by_index={1: failure}
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    with pytest.raises(RuntimeError) as excinfo:
        run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)
    assert "stub: unexpected error" in str(excinfo.value)


def test_run_reproduction_pass_raises_on_missing_config_file(tmp_path) -> None:
    """A non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_reproduction_pass(
            tmp_path / "does_not_exist.json", output_root=tmp_path
        )


# ---------------------------------------------------------------------------
# Filesystem effects
# ---------------------------------------------------------------------------


def test_run_reproduction_pass_creates_run_directories_under_output_root(
    tmp_path, monkeypatch
) -> None:
    """Each completed run produces a directory under output_root."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    for record in summary.records:
        assert record.status == "completed"
        run_json = Path(record.run_json_path or "")
        assert run_json.is_file()
        assert run_json.name == "run.json"
        sibling = run_json.parent / "threshold_robustness.json"
        assert sibling.is_file()


def test_run_reproduction_pass_writes_summary_with_expected_top_level_fields(
    tmp_path, monkeypatch
) -> None:
    """Summary JSON carries every declared top-level field."""
    fake_fit, fake_recompute = _make_fake_pipeline()
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)
    payload = json.loads(
        Path(summary.summary_path).read_text(encoding="utf-8")
    )

    required = {
        "schema_version",
        "config_path",
        "model",
        "condition",
        "configuration_hash",
        "seed_population",
        "seed_values",
        "run_ids",
        "completed_run_count",
        "failed_run_count",
        "graph_status_counts",
        "sampler_status_counts",
        "training_status_counts",
        "shd_values",
        "sid_values",
        "mmd_primary_values",
        "threshold_robustness_available_count",
        "records",
        "reproduction_pass_status",
        "output_root",
        "summary_path",
    }
    missing = required - set(payload)
    assert missing == set(), f"summary JSON missing fields: {missing!r}"


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_reproduction_pass_invokes_runner(tmp_path, monkeypatch) -> None:
    """--phase reproduction_pass --config PATH calls run_reproduction_pass."""
    from experiments.selection_study import run as run_module

    captured: dict[str, Any] = {}

    def fake_runner(
        config_path: Path | str,
        *,
        output_root: Path | None = None,
    ) -> ReproductionPassSummary:
        captured["config_path"] = Path(config_path)
        captured["output_root"] = (
            None if output_root is None else Path(output_root)
        )
        return ReproductionPassSummary(
            schema_version=1,
            config_path=str(config_path),
            model="dagma",
            condition="centred_only",
            configuration_hash="0" * 64,
            seed_population="reproduction",
            seed_values=(101,),
            run_ids=("dagma__centred_only__reproduction__seed0__cfg" + "0" * 64,),
            completed_run_count=1,
            failed_run_count=0,
            graph_status_counts={"valid_dag": 1},
            sampler_status_counts={"available": 1},
            training_status_counts={"max_iter": 1},
            shd_values=(0,),
            sid_values=(0,),
            mmd_primary_values=(),
            threshold_robustness_available_count=1,
            records=(),
            reproduction_pass_status="passed",
            output_root=str(tmp_path),
            summary_path=str(tmp_path / "summary.json"),
        )

    monkeypatch.setattr(
        "experiments.selection_study.reproduction_pass.run_reproduction_pass",
        fake_runner,
    )

    run_module.main(
        [
            "--phase",
            "reproduction_pass",
            "--config",
            str(_DAGMA_PATH),
            "--output-root",
            str(tmp_path),
        ]
    )
    assert captured["config_path"] == _DAGMA_PATH
    assert captured["output_root"] == tmp_path


def test_cli_reproduction_pass_requires_config(tmp_path) -> None:
    """--phase reproduction_pass without --config raises ValueError."""
    from experiments.selection_study import run as run_module

    with pytest.raises(ValueError) as excinfo:
        run_module.main(["--phase", "reproduction_pass"])
    assert "--config" in str(excinfo.value)


def test_cli_rejects_unknown_phase() -> None:
    """argparse rejects unknown --phase values with SystemExit."""
    from experiments.selection_study import run as run_module

    with pytest.raises(SystemExit):
        run_module.main(["--phase", "unknown_stage", "--config", "/dev/null"])


def test_cli_rejects_unsupported_phase_value() -> None:
    """An unsupported --phase value is rejected by argparse."""
    from experiments.selection_study import run as run_module

    with pytest.raises(SystemExit):
        run_module.main(
            ["--phase", "not_a_stage", "--config", "/dev/null"]
        )


# ---------------------------------------------------------------------------
# Stricter reproduction_pass_status semantics
# ---------------------------------------------------------------------------


def test_run_reproduction_pass_status_warns_when_threshold_sibling_missing(
    tmp_path, monkeypatch
) -> None:
    """All runs complete but a missing sibling demotes status to warnings."""
    fake_fit, fake_recompute = _make_fake_pipeline(
        skip_threshold_sibling_indices=(1,),
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert summary.completed_run_count == 3
    assert summary.failed_run_count == 0
    assert summary.threshold_robustness_available_count == 2
    assert summary.reproduction_pass_status == "completed_with_warnings"
    # The specific entry that did not get a sibling is flagged as
    # threshold_robustness_available=False at the record level.
    missing_records = [
        r for r in summary.records
        if not r.threshold_robustness_available
    ]
    assert len(missing_records) == 1


def test_run_reproduction_pass_status_warns_when_graph_status_not_valid_dag(
    tmp_path, monkeypatch
) -> None:
    """A completed run with a non-valid_dag graph demotes status."""
    fake_fit, fake_recompute = _make_fake_pipeline(
        graph_status_by_index={2: "bidirected"},
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert summary.completed_run_count == 3
    assert summary.failed_run_count == 0
    assert summary.graph_status_counts.get("bidirected", 0) == 1
    assert summary.graph_status_counts.get("valid_dag", 0) == 2
    assert summary.reproduction_pass_status == "completed_with_warnings"


def test_run_reproduction_pass_status_warns_when_sampler_unavailable(
    tmp_path, monkeypatch
) -> None:
    """A completed run with an unavailable sampler demotes status."""
    fake_fit, fake_recompute = _make_fake_pipeline(
        sampler_status_by_index={0: "unavailable_invalid_graph"},
    )
    _patch_pipeline(monkeypatch, fake_fit, fake_recompute)

    summary = run_reproduction_pass(_DAGMA_PATH, output_root=tmp_path)

    assert summary.completed_run_count == 3
    assert summary.failed_run_count == 0
    assert (
        summary.sampler_status_counts.get("unavailable_invalid_graph", 0)
        == 1
    )
    assert summary.sampler_status_counts.get("available", 0) == 2
    assert summary.reproduction_pass_status == "completed_with_warnings"
