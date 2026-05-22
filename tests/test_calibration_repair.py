"""Tests for the targeted calibration repair entry point.

These tests exercise ``repair_single_calibration_job`` end-to-end
without invoking any real DAGMA or DCDI fit. Each test first uses
``run_calibration`` with an injected synthetic fit_runner to lay
down a healthy 40-record calibration tree under ``tmp_path``; then
the repair entry point is invoked against that tree to verify the
one-record overwrite behaviour, the rerank-and-rewrite contract,
and the identity / precondition validation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from experiments.selection_study.calibration import (
    CalibrationFitJob,
    _CalibrationInfrastructureError,
    repair_single_calibration_job,
    run_calibration,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    HASH_PREFIX_LENGTH,
    SELECTED_CONFIGURATIONS_FILENAME,
    validate_selected_configurations_artefact,
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


# ---------------------------------------------------------------------------
# Synthetic fit-result helpers
# ---------------------------------------------------------------------------


def _copy_calibration_configs_to_tmp(tmp_path: Path) -> Path:
    """Copy the four parent calibration JSONs into a tmp directory."""
    target = tmp_path / "configs"
    target.mkdir()
    for filename in _PARENT_FILENAMES:
        shutil.copy(_CALIBRATION_CONFIG_DIR / filename, target / filename)
    return target


def _build_threshold_metrics(*, sid: int, shd: int, mmd: float) -> list:
    return [
        {
            "threshold": float(threshold),
            "shd": int(shd),
            "sid": int(sid),
            "mmd_primary": float(mmd),
        }
        for threshold in (0.2, 0.3, 0.4)
    ]


def _build_mmd_by_intervention(*, mmd: float) -> list:
    return [
        {
            "intervention_target": int(node),
            "intervention_value": float(sign),
            "mmd_primary": float(mmd),
        }
        for node in range(10)
        for sign in (-2, 2)
    ]


def _expected_run_id(job: CalibrationFitJob) -> str:
    from experiments.selection_study.identity import derive_run_id

    candidate = job.candidate
    return derive_run_id(
        model=candidate.model,
        condition=candidate.condition,
        seed_population="calibration",
        seed_replicate_index=int(job.seed_replicate_index),
        configuration_hash=candidate.configuration_hash_full,
    )


def _baseline_fit_result(
    job: CalibrationFitJob,
    *,
    shd: int | None = None,
    sid: int | None = None,
    mmd_primary: float | None = None,
) -> dict[str, Any]:
    """Build a happy synthetic fit result for ``job``.

    The default metric values are chosen so each candidate has a
    distinct mean SID (driven off the candidate's hash prefix) so
    the ranking is unambiguous and reproducible across test runs.
    """
    candidate = job.candidate
    hyperparameters = dict(candidate.grid_point_hyperparameter)
    if sid is None:
        # Derive a deterministic SID from the hash prefix so different
        # candidates produce different SIDs.
        sid = (
            int(candidate.configuration_hash_full[:6], 16) % 50
        )
    if shd is None:
        shd = 5
    if mmd_primary is None:
        mmd_primary = 0.05 + 0.001 * (
            int(candidate.configuration_hash_full[6:10], 16) % 20
        )
    return {
        "model": candidate.model,
        "condition": candidate.condition,
        "configuration_hash_full": candidate.configuration_hash_full,
        "configuration_hash_prefix": (
            candidate.configuration_hash_prefix
        ),
        "hyperparameters": hyperparameters,
        "seed_value": int(job.seed_value),
        "shd": shd,
        "sid": sid,
        "mmd_primary": mmd_primary,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "runtime_seconds": 12.5,
        "n_iterations": (
            None if candidate.model == "dagma" else 100000
        ),
        "threshold_metrics": _build_threshold_metrics(
            sid=sid, shd=shd, mmd=mmd_primary
        ),
        "mmd_by_intervention": _build_mmd_by_intervention(mmd=mmd_primary),
        "bandwidth_summaries": {"median_heuristic": 1.0},
        "run_id": _expected_run_id(job),
    }


class _FakeClock:
    def __init__(
        self,
        start: datetime = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
        step_seconds: float = 1.0,
    ) -> None:
        self._current = start
        self._step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        result = self._current
        self._current += self._step
        return result


def _lay_down_calibration_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a healthy 40-record calibration tree under tmp_path.

    Returns (config_dir, results_root, artefact_path).
    """
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_baseline_fit_result,
        now_fn=_FakeClock(),
    )
    return config_dir, results_root, artefact_path


def _records_dir_of(artefact_path: Path) -> Path:
    return artefact_path.parent / "records"


def _snapshot_record_bytes(records_dir: Path) -> dict[str, bytes]:
    """Return a name -> bytes snapshot of every record file in the directory."""
    return {
        path.name: path.read_bytes()
        for path in sorted(records_dir.glob("*.json"))
    }


def _pick_repair_target_dagma_centred_only(
    artefact_path: Path,
) -> tuple[str, int]:
    """Return (configuration_hash_prefix, seed_value) for a healthy DAGMA repair target."""
    records_dir = _records_dir_of(artefact_path)
    for path in sorted(records_dir.glob("dagma_centred_only_*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record["graph_status"] == "valid_dag":
            return (
                str(record["configuration_hash_prefix"]),
                int(record["seed_value"]),
            )
    raise AssertionError(
        "no healthy dagma centred_only record found in the test "
        "calibration tree"
    )


# ---------------------------------------------------------------------------
# Happy-path repair behaviour
# ---------------------------------------------------------------------------


def test_repair_calls_fit_runner_exactly_once(tmp_path: Path) -> None:
    """repair_single_calibration_job runs only the matching job."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )

    invocations: list[CalibrationFitJob] = []

    def counting_fit_runner(job: CalibrationFitJob) -> dict[str, Any]:
        invocations.append(job)
        return _baseline_fit_result(job)

    repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash,
        seed_value=target_seed,
        fit_runner=counting_fit_runner,
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )

    assert len(invocations) == 1
    job = invocations[0]
    assert job.candidate.model == "dagma"
    assert job.candidate.condition == "centred_only"
    assert job.candidate.configuration_hash_prefix == target_hash
    assert int(job.seed_value) == target_seed


def test_repair_overwrites_only_the_target_record(tmp_path: Path) -> None:
    """The 39 untouched records are byte-identical before and after repair."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    records_dir = _records_dir_of(artefact_path)

    snapshot_before = _snapshot_record_bytes(records_dir)
    target_record_name = (
        f"dagma_centred_only_{target_hash}_seed{target_seed}.json"
    )
    assert target_record_name in snapshot_before

    repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash,
        seed_value=target_seed,
        fit_runner=lambda job: _baseline_fit_result(
            job, shd=2, sid=1, mmd_primary=0.02
        ),
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )

    snapshot_after = _snapshot_record_bytes(records_dir)
    assert set(snapshot_after.keys()) == set(snapshot_before.keys())
    changed_names = {
        name
        for name in snapshot_after
        if snapshot_after[name] != snapshot_before[name]
    }
    assert changed_names == {target_record_name}


def test_repair_rewrites_validating_selected_configurations(
    tmp_path: Path,
) -> None:
    """After repair the artefact still validates and matches the rank-1 hash."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )

    repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash,
        seed_value=target_seed,
        fit_runner=lambda job: _baseline_fit_result(
            job, shd=2, sid=1, mmd_primary=0.02
        ),
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )

    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    validate_selected_configurations_artefact(artefact)
    # Cross-check: the rank-1 candidate's full hash must equal the
    # selections entry's full hash for every (condition, model).
    for condition in ("centred_only", "standardised"):
        for model in ("dagma", "dcdi"):
            rank_one = artefact["candidate_ranking"][condition][model][0]
            selection = artefact["selections"][condition][model]
            assert (
                rank_one["configuration_hash_full"]
                == selection["selected_configuration_hash_full"]
            )


def test_repair_can_change_selected_configuration_when_metrics_change(
    tmp_path: Path,
) -> None:
    """A drastically worse repaired metric promotes a different rank-1 candidate.

    The baseline run uses hash-derived SIDs that are non-degenerate.
    The repair fit_runner inflates the repaired job's SID by a large
    amount; if that job's candidate previously held the (centred_only,
    dagma) rank-1 slot, the ranker must promote a different candidate.
    Conversely, if the original candidate is already non-rank-1, the
    rank-1 hash remains unchanged. In either case the artefact must
    remain valid and the rank-1 candidate must be consistent with
    the selections entry.
    """
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    with artefact_path.open(encoding="utf-8") as handle:
        baseline_artefact = json.load(handle)
    original_rank_one_hashes = {
        (condition, model): (
            baseline_artefact["selections"][condition][model][
                "selected_configuration_hash_full"
            ]
        )
        for condition in ("centred_only", "standardised")
        for model in ("dagma", "dcdi")
    }

    # Target the centred_only DAGMA rank-1 candidate on seed 201 so
    # that an inflated SID makes the candidate lose its rank-1 slot.
    target_hash_full = original_rank_one_hashes[("centred_only", "dagma")]
    target_hash_prefix = target_hash_full[:HASH_PREFIX_LENGTH]

    # Inflate the repaired job's SID so it ranks last in its cell.
    repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash_prefix,
        seed_value=201,
        fit_runner=lambda job: _baseline_fit_result(
            job, shd=99, sid=999, mmd_primary=0.9
        ),
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )

    with artefact_path.open(encoding="utf-8") as handle:
        repaired_artefact = json.load(handle)
    validate_selected_configurations_artefact(repaired_artefact)
    new_rank_one_dagma_centred = repaired_artefact["selections"][
        "centred_only"
    ]["dagma"]["selected_configuration_hash_full"]
    # The previously rank-1 candidate must no longer be rank-1 in
    # (centred_only, dagma) because its inflated SID pushes it down
    # the lexicographic chain.
    assert new_rank_one_dagma_centred != target_hash_full
    # Other cells were not touched; their rank-1 hashes must be
    # unchanged.
    for cell, prior_hash in original_rank_one_hashes.items():
        if cell == ("centred_only", "dagma"):
            continue
        condition, model = cell
        observed = repaired_artefact["selections"][condition][model][
            "selected_configuration_hash_full"
        ]
        assert observed == prior_hash


def test_repair_report_carries_documented_fields(tmp_path: Path) -> None:
    """The returned dict carries every documented repair-report field."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )

    report = repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash,
        seed_value=target_seed,
        fit_runner=lambda job: _baseline_fit_result(
            job, shd=2, sid=1, mmd_primary=0.02
        ),
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )

    required_fields = {
        "calibration_run_hash_prefix",
        "repaired_record_path",
        "selected_configurations_path",
        "model",
        "condition",
        "configuration_hash_prefix",
        "seed_value",
        "previous_record_status",
        "new_record_status",
        "selected_hashes_after_repair",
        "selected_degeneracy_flags_after_repair",
    }
    assert required_fields.issubset(report.keys())
    assert report["model"] == "dagma"
    assert report["condition"] == "centred_only"
    assert report["configuration_hash_prefix"] == target_hash
    assert report["seed_value"] == target_seed
    assert report["new_record_status"] == "converged"
    assert isinstance(report["selected_hashes_after_repair"], dict)
    assert set(
        report["selected_hashes_after_repair"].keys()
    ) == {"centred_only", "standardised"}
    for condition_block in report[
        "selected_hashes_after_repair"
    ].values():
        assert set(condition_block.keys()) == {"dagma", "dcdi"}


def test_repair_report_contains_no_forbidden_winner_fields(
    tmp_path: Path,
) -> None:
    """No winner-shaped field name appears anywhere in the repair report."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    report = repair_single_calibration_job(
        config_dir,
        results_root,
        model="dagma",
        condition="centred_only",
        configuration_hash_prefix=target_hash,
        seed_value=target_seed,
        fit_runner=lambda job: _baseline_fit_result(job),
        now_fn=_FakeClock(
            start=datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
        ),
    )
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
                    f"forbidden field {key!r} in repair report"
                )
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(report)


# ---------------------------------------------------------------------------
# Identity / precondition validation
# ---------------------------------------------------------------------------


def test_repair_refuses_unknown_model(tmp_path: Path) -> None:
    """An unknown model value is rejected before any filesystem access."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="notamodel",
            condition="centred_only",
            configuration_hash_prefix=target_hash,
            seed_value=target_seed,
        )
    assert "model" in str(excinfo.value).lower()


def test_repair_refuses_unknown_condition(tmp_path: Path) -> None:
    """An unknown condition value is rejected before any filesystem access."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="raw",
            configuration_hash_prefix=target_hash,
            seed_value=target_seed,
        )
    assert "condition" in str(excinfo.value).lower()


def test_repair_refuses_unknown_seed_value(tmp_path: Path) -> None:
    """A held-out seed value is rejected before any filesystem access."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, _ = _pick_repair_target_dagma_centred_only(artefact_path)
    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix=target_hash,
            seed_value=301,
        )
    assert "seed_value" in str(excinfo.value)


def test_repair_refuses_malformed_hash_prefix(tmp_path: Path) -> None:
    """A non-hex configuration_hash_prefix is rejected."""
    config_dir, results_root, _ = _lay_down_calibration_tree(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix="not-hex-1234",
            seed_value=201,
        )
    assert "configuration_hash_prefix" in str(excinfo.value)


def test_repair_refuses_unknown_hash(tmp_path: Path) -> None:
    """A correctly-shaped but non-matching hash prefix is rejected with no match."""
    config_dir, results_root, _ = _lay_down_calibration_tree(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix="0123456789ab",
            seed_value=201,
        )
    assert "no calibration job matched" in str(excinfo.value)


def test_repair_refuses_missing_records_directory(tmp_path: Path) -> None:
    """Removing the records directory makes repair raise FileNotFoundError."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    records_dir = _records_dir_of(artefact_path)
    shutil.rmtree(records_dir)
    with pytest.raises(FileNotFoundError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix=target_hash,
            seed_value=target_seed,
        )
    assert "records directory" in str(excinfo.value)


def test_repair_refuses_missing_selected_configurations(
    tmp_path: Path,
) -> None:
    """Removing the artefact file makes repair raise FileNotFoundError."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    artefact_path.unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix=target_hash,
            seed_value=target_seed,
        )
    assert "selected_configurations.json" in str(excinfo.value)


def test_repair_refuses_when_record_count_is_not_40(tmp_path: Path) -> None:
    """Removing one record makes repair raise before invoking the fit_runner."""
    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    records_dir = _records_dir_of(artefact_path)
    # Remove an unrelated record so the count drops to 39.
    other_record = next(
        path
        for path in records_dir.glob("*.json")
        if not path.name.startswith(
            f"dagma_centred_only_{target_hash}"
        )
    )
    other_record.unlink()
    invocations: list[Any] = []

    def counting_fit_runner(job: Any) -> Any:
        invocations.append(job)
        return _baseline_fit_result(job)

    with pytest.raises(ValueError) as excinfo:
        repair_single_calibration_job(
            config_dir,
            results_root,
            model="dagma",
            condition="centred_only",
            configuration_hash_prefix=target_hash,
            seed_value=target_seed,
            fit_runner=counting_fit_runner,
        )
    assert "40 per-fit records" in str(excinfo.value)
    assert invocations == []


# ---------------------------------------------------------------------------
# Infrastructure-failure propagation
# ---------------------------------------------------------------------------


def test_repair_propagates_file_exists_error_as_infrastructure(
    tmp_path: Path,
) -> None:
    """A FileExistsError from the default fit path aborts the repair.

    Drives the default fit-runner factory with a monkeypatched
    ``pipeline.run_single_fit`` that raises ``FileExistsError``; the
    runner re-raises as ``_CalibrationInfrastructureError`` and the
    repair entry point must let it propagate without writing the
    artefact or overwriting the per-fit record.
    """
    from experiments.selection_study import pipeline
    from experiments.selection_study import threshold_robustness

    config_dir, results_root, artefact_path = (
        _lay_down_calibration_tree(tmp_path)
    )
    target_hash, target_seed = (
        _pick_repair_target_dagma_centred_only(artefact_path)
    )
    records_dir = _records_dir_of(artefact_path)
    target_record_path = (
        records_dir
        / f"dagma_centred_only_{target_hash}_seed{target_seed}.json"
    )
    record_bytes_before = target_record_path.read_bytes()
    artefact_bytes_before = artefact_path.read_bytes()

    def fake_run_single_fit_raising(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        entry = manifest.entries[entry_index]
        raise FileExistsError(
            "run directory is already populated; refusing to "
            "overwrite: "
            f"{run_root / entry.model / entry.condition / 'calibration'}"
            f"/seed{int(entry.seed_replicate_index)}/"
            f"{entry.configuration_hash[:HASH_PREFIX_LENGTH]}"
        )

    import experiments.selection_study.pipeline as _pipeline_module

    monkeypatch_pipeline = pytest.MonkeyPatch()
    try:
        monkeypatch_pipeline.setattr(
            _pipeline_module,
            "run_single_fit",
            fake_run_single_fit_raising,
        )
        monkeypatch_pipeline.setattr(
            threshold_robustness,
            "recompute_at_thresholds",
            lambda run_dir, *, write_sibling=True: {"records": []},
        )
        with pytest.raises(_CalibrationInfrastructureError) as excinfo:
            repair_single_calibration_job(
                config_dir,
                results_root,
                model="dagma",
                condition="centred_only",
                configuration_hash_prefix=target_hash,
                seed_value=target_seed,
                fit_runner=None,
            )
    finally:
        monkeypatch_pipeline.undo()

    assert "pre-existing raw run directory" in str(excinfo.value)
    # The target record and the artefact must be untouched.
    assert target_record_path.read_bytes() == record_bytes_before
    assert artefact_path.read_bytes() == artefact_bytes_before
