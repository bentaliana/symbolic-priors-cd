"""Tests for the calibration orchestration entry point.

These tests exercise ``run_calibration`` end-to-end without
invoking any real DAGMA or DCDI fit. All fits are produced by a
fake ``fit_runner`` callable passed directly into
``run_calibration``; the project's ``pipeline.run_single_fit`` is
monkeypatched in every test as a backstop and the patched sentinel
must remain at zero invocations.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pytest

from experiments.selection_study.calibration import (
    CalibrationFitJob,
    run_calibration,
)
from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.identity import derive_run_id
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    FIT_RNG_POLICY_REF,
    INTERVENTION_POLICY_REF,
    MODELS,
    SCHEMA_VERSION,
    SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
    SELECTED_CONFIGURATIONS_FILENAME,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
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
# Test helpers
# ---------------------------------------------------------------------------


def _copy_calibration_configs_to_tmp(tmp_path: Path) -> Path:
    """Copy the four parent calibration JSONs into a tmp directory."""
    target = tmp_path / "configs"
    target.mkdir()
    for filename in _PARENT_FILENAMES:
        shutil.copy(_CALIBRATION_CONFIG_DIR / filename, target / filename)
    return target


def _build_threshold_metrics(
    *, sid: int, shd: int, mmd: float
) -> list[dict[str, Any]]:
    """Return three threshold-metric rows for one fit result."""
    return [
        {
            "threshold": float(threshold),
            "shd": int(shd),
            "sid": int(sid),
            "mmd_primary": float(mmd),
        }
        for threshold in (0.2, 0.3, 0.4)
    ]


def _build_mmd_by_intervention(*, mmd: float) -> list[dict[str, Any]]:
    """Return 20 intervention rows (10 nodes x both signs)."""
    return [
        {
            "intervention_target": int(node),
            "intervention_value": float(sign),
            "mmd_primary": float(mmd),
        }
        for node in range(10)
        for sign in (-2, 2)
    ]


def _build_bandwidth_summaries() -> dict[str, Any]:
    return {
        "median_heuristic": 1.0,
        "scaled_0p5x": 0.5,
        "scaled_1p0x": 1.0,
        "scaled_2p0x": 2.0,
    }


def _expected_run_id(job: CalibrationFitJob) -> str:
    """Compute the run_id the orchestrator expects for a job."""
    candidate = job.candidate
    return derive_run_id(
        model=candidate.model,
        condition=candidate.condition,
        seed_population="calibration",
        seed_replicate_index=int(job.seed_replicate_index),
        configuration_hash=candidate.configuration_hash_full,
    )


def _happy_fit_result(job: CalibrationFitJob) -> dict[str, Any]:
    """Build a fully-formed happy-path fit result for ``job``.

    The SID value is offset by the grid-point index so the five
    candidates in each cell have distinct mean SIDs, which produces
    a unique deterministic ranking.
    """
    candidate = job.candidate
    hyperparameters = dict(candidate.grid_point_hyperparameter)
    hyperparameter_value = next(iter(hyperparameters.values()))
    # Make SID a function of the hyperparameter value so the five
    # candidates per cell have distinct mean SIDs.
    sid_value = int(round(100.0 + 10.0 * float(hyperparameter_value)))
    shd_value = 5
    mmd_value = 0.05
    return {
        "model": candidate.model,
        "condition": candidate.condition,
        "configuration_hash_full": candidate.configuration_hash_full,
        "configuration_hash_prefix": candidate.configuration_hash_prefix,
        "hyperparameters": hyperparameters,
        "seed_value": int(job.seed_value),
        "shd": shd_value,
        "sid": sid_value,
        "mmd_primary": mmd_value,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "runtime_seconds": 12.5,
        "n_iterations": None if candidate.model == "dagma" else 100000,
        "threshold_metrics": _build_threshold_metrics(
            sid=sid_value, shd=shd_value, mmd=mmd_value
        ),
        "mmd_by_intervention": _build_mmd_by_intervention(mmd=mmd_value),
        "bandwidth_summaries": _build_bandwidth_summaries(),
        "run_id": _expected_run_id(job),
    }


class _RecordingFitRunner:
    """Fake fit_runner that records each call and returns happy results.

    The recorder lets tests assert exact invocation count and inspect
    the jobs passed to the runner.
    """

    def __init__(
        self,
        result_factory: Callable[[CalibrationFitJob], dict[str, Any]] = _happy_fit_result,
    ) -> None:
        self.calls: list[CalibrationFitJob] = []
        self._result_factory = result_factory

    def __call__(self, job: CalibrationFitJob) -> dict[str, Any]:
        self.calls.append(job)
        return self._result_factory(job)


class _FakeClock:
    """Deterministic clock that increments by a fixed step on each call."""

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


class _FixedClock:
    """Clock that always returns the same datetime."""

    def __init__(
        self,
        value: datetime = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    ) -> None:
        self._value = value

    def __call__(self) -> datetime:
        return self._value


class _RunSingleFitSentinel:
    """Monkeypatch replacement that fails if pipeline.run_single_fit is called."""

    def __init__(self) -> None:
        self.invocations: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Path:
        self.invocations += 1
        raise AssertionError(
            "pipeline.run_single_fit was invoked during a calibration "
            "orchestration test; tests must use the injected "
            "fit_runner and must never trigger the real fit path"
        )


@pytest.fixture
def block_real_fits(monkeypatch: pytest.MonkeyPatch) -> _RunSingleFitSentinel:
    """Monkeypatch ``pipeline.run_single_fit`` and assert never-called."""
    from experiments.selection_study import pipeline

    sentinel = _RunSingleFitSentinel()
    monkeypatch.setattr(pipeline, "run_single_fit", sentinel)
    yield sentinel
    assert sentinel.invocations == 0, (
        "pipeline.run_single_fit was invoked "
        f"{sentinel.invocations} time(s) during the test; the "
        "orchestrator must drive the injected fit_runner only"
    )


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def test_run_calibration_returns_selected_configurations_path(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """run_calibration returns the on-disk artefact path."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fit_runner = _RecordingFitRunner()
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    assert artefact_path.is_file()
    assert artefact_path.name == SELECTED_CONFIGURATIONS_FILENAME
    assert artefact_path.parent.parent.name == "calibration"


def test_missing_config_raises_file_not_found_error(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Removing one parent config makes run_calibration raise FileNotFoundError."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    (config_dir / "dagma_calibration_centred_only.json").unlink()
    results_root = tmp_path / "results"
    with pytest.raises(FileNotFoundError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=_RecordingFitRunner(),
            now_fn=_FakeClock(),
        )
    assert "dagma_calibration_centred_only.json" in str(excinfo.value)


def test_missing_config_dir_raises_file_not_found_error(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A non-existent config_dir raises FileNotFoundError."""
    config_dir = tmp_path / "absent"
    results_root = tmp_path / "results"
    with pytest.raises(FileNotFoundError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=_RecordingFitRunner(),
            now_fn=_FakeClock(),
        )
    assert str(config_dir) in str(excinfo.value)


def test_fit_runner_called_exactly_40_times(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The fake fit_runner is invoked exactly 40 times for one full run."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fit_runner = _RecordingFitRunner()
    run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    assert len(fit_runner.calls) == 40


def test_fit_runner_sees_calibration_seeds_only(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Every job passed to the fit_runner has a calibration seed value."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fit_runner = _RecordingFitRunner()
    run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    expected_seeds = set(CALIBRATION_SEEDS)
    forbidden_seeds = {301, 302, 303, 304, 305}
    for job in fit_runner.calls:
        assert job.seed_value in expected_seeds
        assert job.seed_value not in forbidden_seeds


def test_writes_exactly_40_per_fit_records(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The records directory contains exactly 40 record files after one run."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    record_files = sorted(records_dir.glob("*.json"))
    assert len(record_files) == 40


def test_record_filenames_are_deterministic_across_two_runs(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Re-running the orchestrator with force=True yields identical filenames."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path_1 = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    filenames_1 = sorted(
        p.name for p in (artefact_path_1.parent / "records").iterdir()
    )
    artefact_path_2 = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
        force=True,
    )
    filenames_2 = sorted(
        p.name for p in (artefact_path_2.parent / "records").iterdir()
    )
    assert filenames_1 == filenames_2


def test_record_filename_format_is_model_condition_hash_seed(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Each record filename matches <model>_<condition>_<hash12>_seed<NN>.json."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    for record_file in records_dir.iterdir():
        stem = record_file.stem
        parts = stem.split("_")
        # model has one underscore-free token. condition has either
        # one token (centred_only -> centred + only -> 2 tokens, but
        # joined by underscore) so the split is more nuanced. Just
        # verify the file ends with "_seed<int>".
        assert "seed" in stem
        seed_part = stem.rsplit("_seed", 1)[1]
        assert seed_part in {"201", "202"}


def test_persisted_records_match_ranker_input_contract(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Loading the 40 records and ranking them succeeds without error."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    records = []
    for record_file in sorted(records_dir.iterdir()):
        with record_file.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    # rank_calibration_records is strict about cardinality and shape.
    output = rank_calibration_records(records)
    assert "candidate_ranking" in output
    assert "selections" in output


def test_written_artefact_validates(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The written selected_configurations.json passes the validator."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    validate_selected_configurations_artefact(artefact)


def test_artefact_carries_stable_policy_identifiers(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The artefact metadata carries the four stable policy refs."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    assert artefact["selection_rule_id"] == SELECTION_RULE_ID
    assert artefact["selection_rule_ref"] == SELECTION_RULE_REF
    assert artefact["intervention_policy_ref"] == INTERVENTION_POLICY_REF
    assert artefact["fit_rng_policy_ref"] == FIT_RNG_POLICY_REF
    assert artefact["seed_population"] == "calibration"
    assert artefact["calibration_seeds"] == list(CALIBRATION_SEEDS)
    assert artefact["base_model_decision_made"] is False
    assert (
        artefact["selected_configuration_semantics"]
        == "rank_1_within_model_and_condition"
    )


def test_generated_at_utc_uses_fixed_now_fn_format(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """generated_at_utc is formatted as YYYY-MM-DDTHH:MM:SSZ from now_fn."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fake_now = _FakeClock(
        start=datetime(2026, 7, 1, 9, 30, 45, tzinfo=timezone.utc),
        step_seconds=1.0,
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=fake_now,
    )
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    timestamp = artefact["generated_at_utc"]
    # Match YYYY-MM-DDTHH:MM:SSZ exactly.
    parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    assert timestamp.endswith("Z")
    assert parsed.year == 2026 and parsed.month == 7 and parsed.day == 1


# ---------------------------------------------------------------------------
# Fit-runner injection
# ---------------------------------------------------------------------------


def test_explicit_fit_runner_is_used_not_default(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The explicit fit_runner sees every job; the default is not invoked."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fit_runner = _RecordingFitRunner()
    run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    assert len(fit_runner.calls) == 40


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_fit_runner_exception_creates_degenerate_record_and_continues(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A fit_runner exception produces a degenerate record without aborting."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def failing_factory(job: CalibrationFitJob) -> dict[str, Any]:
        candidate = job.candidate
        # Fail on the first DCDI standardised candidate's first seed.
        if (
            candidate.model == "dcdi"
            and candidate.condition == "standardised"
            and int(job.seed_value) == 201
            and candidate.grid_point_name == "reg_coeff_0p01"
        ):
            raise RuntimeError("synthetic fit failure for testing")
        return _happy_fit_result(job)

    fit_runner = _RecordingFitRunner(result_factory=failing_factory)
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    assert artefact_path.is_file()
    records_dir = artefact_path.parent / "records"
    assert len(list(records_dir.glob("*.json"))) == 40
    # Inspect the degenerate record on disk.
    degenerate_files = list(
        records_dir.glob("dcdi_standardised_*_seed201.json")
    )
    assert len(degenerate_files) == 5
    # Identify the failing candidate's record by reading its
    # hyperparameters.
    failing_records = []
    for path in degenerate_files:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("training_status") == "failed":
            failing_records.append(record)
    assert len(failing_records) == 1
    failing = failing_records[0]
    assert failing["graph_status"] == "failed"
    assert failing["sampler_status"] == "failed"
    assert failing["training_status"] == "failed"
    assert failing["shd"] is None
    assert failing["sid"] is None
    assert failing["mmd_primary"] is None
    assert failing["threshold_metrics"] == []
    assert failing["mmd_by_intervention"] == []
    assert failing["bandwidth_summaries"] == {}
    assert "failure_message" in failing


def test_degenerate_record_uses_none_not_nan(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Degenerate numeric metrics are persisted as JSON null, not NaN."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def failing_for_all(job: CalibrationFitJob) -> dict[str, Any]:
        raise RuntimeError("synthetic failure on every fit")

    fit_runner = _RecordingFitRunner(result_factory=failing_for_all)
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    sample_file = next(iter(records_dir.glob("*.json")))
    raw_text = sample_file.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text


def test_ranker_consumes_records_including_degenerate(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The ranker runs successfully even when some records are degenerate."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def fail_one_per_cell(job: CalibrationFitJob) -> dict[str, Any]:
        candidate = job.candidate
        if (
            candidate.grid_point_name in ("lambda1_0p25", "reg_coeff_1p0")
            and int(job.seed_value) == 201
        ):
            raise RuntimeError("synthetic failure on one fit per cell")
        return _happy_fit_result(job)

    fit_runner = _RecordingFitRunner(result_factory=fail_one_per_cell)
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    validate_selected_configurations_artefact(artefact)


def test_malformed_returned_record_fails_fast(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A fit_runner result missing a required field raises ValueError fast."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def malformed_factory(job: CalibrationFitJob) -> dict[str, Any]:
        result = _happy_fit_result(job)
        del result["mmd_primary"]
        return result

    fit_runner = _RecordingFitRunner(result_factory=malformed_factory)
    with pytest.raises(ValueError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    assert "mmd_primary" in str(excinfo.value)


def test_wrong_seed_in_fit_result_fails_fast(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A fit_runner result with the wrong seed_value raises ValueError fast."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def wrong_seed_factory(job: CalibrationFitJob) -> dict[str, Any]:
        result = _happy_fit_result(job)
        result["seed_value"] = 999
        return result

    fit_runner = _RecordingFitRunner(result_factory=wrong_seed_factory)
    with pytest.raises(ValueError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    assert "seed_value" in str(excinfo.value)


def test_wrong_hash_in_fit_result_fails_fast(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A fit_runner result with a wrong configuration_hash raises ValueError."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    wrong_hash = "0" * 64

    def wrong_hash_factory(job: CalibrationFitJob) -> dict[str, Any]:
        result = _happy_fit_result(job)
        result["configuration_hash_full"] = wrong_hash
        result["configuration_hash_prefix"] = wrong_hash[:12]
        return result

    fit_runner = _RecordingFitRunner(result_factory=wrong_hash_factory)
    with pytest.raises(ValueError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    assert "configuration_hash" in str(excinfo.value)


def test_wrong_model_in_fit_result_fails_fast(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """A fit_runner result with the wrong model raises ValueError fast."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def wrong_model_factory(job: CalibrationFitJob) -> dict[str, Any]:
        result = _happy_fit_result(job)
        result["model"] = "dcdi" if job.candidate.model == "dagma" else "dagma"
        return result

    fit_runner = _RecordingFitRunner(result_factory=wrong_model_factory)
    with pytest.raises(ValueError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    assert "model" in str(excinfo.value).lower()


def test_infrastructure_failure_does_not_write_artefact(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """An infrastructure failure leaves no selected_configurations.json on disk."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def malformed_factory(job: CalibrationFitJob) -> dict[str, Any]:
        result = _happy_fit_result(job)
        del result["sid"]
        return result

    fit_runner = _RecordingFitRunner(result_factory=malformed_factory)
    with pytest.raises(ValueError):
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    # No selected_configurations.json should exist anywhere under
    # the calibration tree.
    candidate_paths = list(
        (results_root / "model_selection" / "calibration").rglob(
            SELECTED_CONFIGURATIONS_FILENAME
        )
    )
    assert candidate_paths == []


def test_records_persisted_before_infrastructure_failure_remain(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Records persisted before an infrastructure failure stay on disk."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    call_index = {"value": 0}

    def fail_at_third(job: CalibrationFitJob) -> dict[str, Any]:
        call_index["value"] += 1
        if call_index["value"] == 3:
            # Return a malformed result on the third invocation.
            result = _happy_fit_result(job)
            del result["graph_status"]
            return result
        return _happy_fit_result(job)

    fit_runner = _RecordingFitRunner(result_factory=fail_at_third)
    with pytest.raises(ValueError):
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    # The first two successful records must remain on disk.
    records_dirs = list(
        (results_root / "model_selection" / "calibration").glob(
            "*/records"
        )
    )
    assert len(records_dirs) == 1
    record_files = list(records_dirs[0].iterdir())
    assert len(record_files) == 2


# ---------------------------------------------------------------------------
# Overwrite safety
# ---------------------------------------------------------------------------


def test_existing_artefact_with_force_false_raises_before_fits(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """An existing selected_configurations.json blocks the run before any fit."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fit_runner = _RecordingFitRunner()
    # First run to produce the artefact.
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=fit_runner,
        now_fn=_FakeClock(),
    )
    assert artefact_path.is_file()
    # Second run with force=False must raise before invoking fits.
    blocking_fit_runner = _RecordingFitRunner()
    with pytest.raises(FileExistsError):
        run_calibration(
            config_dir,
            results_root,
            fit_runner=blocking_fit_runner,
            now_fn=_FakeClock(),
        )
    assert len(blocking_fit_runner.calls) == 0


def test_force_true_permits_artefact_replacement(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """force=True allows the artefact to be replaced after a prior run."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path_a = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FixedClock(
            value=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
        ),
    )
    artefact_path_b = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FixedClock(
            value=datetime(2026, 5, 22, 13, 0, 0, tzinfo=timezone.utc)
        ),
        force=True,
    )
    assert artefact_path_a == artefact_path_b
    with artefact_path_b.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    assert artefact["generated_at_utc"] == "2026-05-22T13:00:00Z"


def test_existing_per_fit_record_with_force_false_raises(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """An existing per-fit record blocks the run when force is False."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    # First run to populate records.
    run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    # Delete the artefact so the artefact-level check does not pre-empt
    # the per-fit record check; the per-fit guard is what we want to
    # exercise here.
    artefact_path = next(
        (results_root / "model_selection" / "calibration").rglob(
            SELECTED_CONFIGURATIONS_FILENAME
        )
    )
    artefact_path.unlink()
    fit_runner = _RecordingFitRunner()
    with pytest.raises(FileExistsError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=fit_runner,
            now_fn=_FakeClock(),
        )
    assert "per-fit record" in str(excinfo.value)


def test_force_true_permits_per_fit_record_replacement(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """force=True permits per-fit record overwrite as well as artefact."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
        force=True,
    )
    records_dir = artefact_path.parent / "records"
    assert len(list(records_dir.glob("*.json"))) == 40


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_calibration_run_log_is_created(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The calibration_run.log file is created under the run directory."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    log_path = artefact_path.parent / "calibration_run.log"
    assert log_path.is_file()


def test_log_contains_progress_lines_for_each_fit(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The log contains at least one START and one END line per fit."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    log_path = artefact_path.parent / "calibration_run.log"
    text = log_path.read_text(encoding="utf-8")
    start_lines = [line for line in text.splitlines() if " START " in line]
    end_lines = [line for line in text.splitlines() if " END " in line]
    assert len(start_lines) == 40
    assert len(end_lines) == 40


def test_log_lines_carry_utc_timestamps(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """Every START/END log line carries a UTC YYYY-MM-DDTHH:MM:SSZ timestamp."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    fake_clock = _FakeClock(
        start=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
        step_seconds=1.0,
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=fake_clock,
    )
    log_path = artefact_path.parent / "calibration_run.log"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "START" in line or "END" in line:
            # Each line begins with "[YYYY-MM-DDTHH:MM:SSZ]"
            timestamp = line.split("]", 1)[0].lstrip("[")
            parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            assert parsed.year == 2026
            assert parsed.month == 5
            assert parsed.day == 22


def test_log_contains_no_forbidden_winner_language(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """The log carries no winner / recommendation / final-decision phrases."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    log_path = artefact_path.parent / "calibration_run.log"
    text = log_path.read_text(encoding="utf-8").lower()
    forbidden_phrases = (
        "winner",
        "best model",
        "selected model",
        "recommended",
        "dagma wins",
        "dcdi wins",
        "final decision",
    )
    for phrase in forbidden_phrases:
        assert phrase not in text, (
            f"forbidden phrase {phrase!r} appeared in calibration_run.log"
        )


# ---------------------------------------------------------------------------
# Forbidden field names anywhere in the artefact tree
# ---------------------------------------------------------------------------


def test_artefact_contains_no_forbidden_winner_fields(
    tmp_path: Path, block_real_fits: _RunSingleFitSentinel
) -> None:
    """No forbidden field name appears at any depth of the written artefact."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=_RecordingFitRunner(),
        now_fn=_FakeClock(),
    )
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    forbidden = {
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    }

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert key not in forbidden, (
                    f"forbidden field name {key!r} at {path}"
                )
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")

    walk(artefact, "$")


# ---------------------------------------------------------------------------
# Scope confirmations
# ---------------------------------------------------------------------------


def test_run_calibration_module_does_not_expose_winner_names() -> None:
    """The calibration module exposes no public name resembling a winner field."""
    from experiments.selection_study import calibration

    public_names = [
        name for name in dir(calibration) if not name.startswith("_")
    ]
    forbidden_substrings = (
        "winner",
        "WINNER",
        "model_winner",
        "MODEL_WINNER",
        "base_model_winner",
        "BASE_MODEL_WINNER",
        "recommended_model",
        "RECOMMENDED_MODEL",
        "final_decision",
        "FINAL_DECISION",
    )
    for name in public_names:
        for substring in forbidden_substrings:
            assert substring not in name, (
                f"public name {name!r} contains forbidden substring "
                f"{substring!r}"
            )
