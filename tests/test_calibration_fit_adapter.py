"""Tests for the production fit-result adapter.

These tests exercise ``_adapt_fit_result_to_calibration_record`` and
the orchestration's default fit_runner end-to-end without invoking
any real DAGMA or DCDI fit. The fake production fit result is built
to match the project's run.json schema; the production pipeline is
monkeypatched with a function that returns a path to a synthetic
run.json on disk so the adapter exercises the full lazy-import +
load_run + recompute-skip flow.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from experiments.selection_study.calibration import (
    CalibrationFitJob,
    _CalibrationInfrastructureError,
    _adapt_fit_result_to_calibration_record,
    enumerate_calibration_workload,
    run_calibration,
)
from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.config import load_config
from experiments.selection_study.identity import derive_run_id
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
# Synthetic run.json factory
# ---------------------------------------------------------------------------


def _synthetic_intervention_records() -> list[dict[str, Any]]:
    """Return 20 intervention records using the production field names."""
    return [
        {
            "intervention_id": f"do_X{node}_{'pos' if sign > 0 else 'neg'}2",
            "target_node": int(node),
            "value_raw": float(sign),
            "value_model_frame": float(sign),
            "ground_truth_sampling_seed": 1000 + node,
            "model_sampling_seed": 2000 + node,
            "n_ground_truth_samples": 1000,
            "n_model_samples": 1000,
            "mmd_value": 0.05,
            "mmd_status": "available",
            "bandwidth_used": 1.0,
            "bandwidth_sweep": {
                "scaled_0p5x": 0.04,
                "scaled_1p0x": 0.05,
                "scaled_2p0x": 0.06,
            },
            "sampler_status_for_intervention": "available",
            "sampler_reason": None,
        }
        for node in range(10)
        for sign in (-2, 2)
    ]


def _synthetic_threshold_records() -> list[dict[str, Any]]:
    """Return three threshold-robustness records in the project's shape."""
    return [
        {
            "threshold": float(threshold),
            "threshold_role": role,
            "edge_count": 20,
            "graph_status": "valid_dag",
            "graph_status_reason": None,
            "shd": 5,
            "shd_unavailable_reason": None,
            "sid": 10,
            "sid_unavailable_reason": None,
        }
        for threshold, role in (
            (0.2, "minus"),
            (0.3, "primary"),
            (0.4, "plus"),
        )
    ]


def _expected_run_id(job: CalibrationFitJob) -> str:
    candidate = job.candidate
    return derive_run_id(
        model=candidate.model,
        condition=candidate.condition,
        seed_population="calibration",
        seed_replicate_index=int(job.seed_replicate_index),
        configuration_hash=candidate.configuration_hash_full,
    )


def _synthetic_fit_result(
    job: CalibrationFitJob,
    *,
    shd: int = 5,
    sid: int = 10,
    mmd_primary: float | None = 0.05,
    training_status: str = "converged",
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
    runtime_seconds: float = 12.5,
    n_iterations: int | None = None,
    interventions: list[dict[str, Any]] | None = None,
    bandwidth_summaries: dict[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a synthetic run.json-shaped dict for ``job``.

    The result carries every required production field plus a
    selection of optional ones; tests can override specific values
    via keyword arguments to probe the adapter's reactions.
    """
    candidate = job.candidate
    if interventions is None:
        interventions = _synthetic_intervention_records()
    if bandwidth_summaries is None:
        bandwidth_summaries = {
            "do_X0_pos2": 1.0,
            "do_X0_neg2": 1.0,
        }
    payload: dict[str, Any] = {
        "run_id": _expected_run_id(job),
        "schema_version": 1,
        "model": candidate.model,
        "condition": candidate.condition,
        "seed_population": "calibration",
        "seed_replicate_index": int(job.seed_replicate_index),
        "configuration_hash": candidate.configuration_hash_full,
        "graph_seed": 42,
        "git_hash": "deadbeef",
        "env_snapshot": "synthetic-env",
        "config_resolved": {},
        "seed_torch": (42 if candidate.model == "dcdi" else None),
        "seed_numpy": (42 if candidate.model == "dcdi" else None),
        "seed_dagma": None,
        "model_sampling_seed_base": 99,
        "model_sampling_seed_derivation_rule": (
            "sha256_first8_bytes_mod_2pow31_purpose_label_v1"
        ),
        "train_data_seed": 100,
        "validation_data_seed": (
            None if candidate.model == "dagma" else 200
        ),
        "intervention_ground_truth_seed_base": 300,
        "training_status": training_status,
        "n_iterations": n_iterations,
        "runtime_seconds": runtime_seconds,
        "loss_history": None,
        "loss_history_status": "unavailable_no_api",
        "graph_status": graph_status,
        "graph_status_reason": None,
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": shd,
        "sid": sid,
        "mmd_primary": mmd_primary,
        "mmd_sensitivity_unit_variance": None,
        "mmd_bandwidth_sweep": {},
        "validation_nll": None,
        "sampler_status": sampler_status,
        "sampler_status_reason": None,
        "sampler_policy_used": "residual_fitted",
        "mmd_available_count": 20,
        "mmd_missing_count": 0,
        "invalid_graph_for_this_run": False,
        "shd_reversal_cost": 2,
        "mmd_bandwidth_used_value": bandwidth_summaries,
        "mmd_clip_policy": "none",
        "sid_backend": "gadjid",
        "sid_backend_version": "0.1.0",
        "sid_argument_order": "predicted_then_true",
        "sid_return_value": "scalar_int",
        "configuration_hash_algorithm": (
            "sha256_canonical_json_sorted_keys"
        ),
        "wrapper_diagnostics": {},
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": interventions,
    }
    if extra is not None:
        payload.update(extra)
    return payload


def _make_jobs() -> list[CalibrationFitJob]:
    """Enumerate the 40 calibration jobs from the on-disk parent configs."""
    parents = tuple(
        load_config(_CALIBRATION_CONFIG_DIR / name)
        for name in _PARENT_FILENAMES
    )
    workload = enumerate_calibration_workload(parents)
    return list(workload.fit_jobs)


def _pick_job(model: str, condition: str) -> CalibrationFitJob:
    """Return the first calibration job matching ``(model, condition)``."""
    for job in _make_jobs():
        if (
            job.candidate.model == model
            and job.candidate.condition == condition
        ):
            return job
    raise AssertionError(
        f"no calibration job found for model={model!r} condition={condition!r}"
    )


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


def test_adapter_maps_synthetic_dagma_fit_result_to_ranker_schema() -> None:
    """A synthetic DAGMA fit result produces a ranker-shaped record."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    record = _adapt_fit_result_to_calibration_record(
        job,
        fit_result,
        threshold_robustness_records=_synthetic_threshold_records(),
    )
    assert record["model"] == "dagma"
    assert record["condition"] == "centred_only"
    assert record["configuration_hash_full"] == (
        job.candidate.configuration_hash_full
    )
    assert record["configuration_hash_prefix"] == (
        job.candidate.configuration_hash_prefix
    )
    assert record["seed_value"] in set(CALIBRATION_SEEDS)
    assert record["hyperparameters"] == dict(
        job.candidate.grid_point_hyperparameter
    )
    assert record["graph_status"] == "valid_dag"
    assert record["sampler_status"] == "available"
    assert record["training_status"] == "converged"
    assert record["shd"] == 5
    assert record["sid"] == 10
    assert record["mmd_primary"] == 0.05
    assert record["runtime_seconds"] == 12.5
    assert record["n_iterations"] is None
    assert record["run_id"] == _expected_run_id(job)
    assert isinstance(record["threshold_metrics"], list)
    assert len(record["threshold_metrics"]) == 3
    assert isinstance(record["mmd_by_intervention"], list)
    assert len(record["mmd_by_intervention"]) == 20
    assert isinstance(record["bandwidth_summaries"], dict)


def test_adapter_maps_synthetic_dcdi_fit_result_to_ranker_schema() -> None:
    """A synthetic DCDI fit result with non-null n_iterations adapts cleanly."""
    job = _pick_job("dcdi", "standardised")
    fit_result = _synthetic_fit_result(
        job, n_iterations=80000
    )
    record = _adapt_fit_result_to_calibration_record(
        job,
        fit_result,
        threshold_robustness_records=_synthetic_threshold_records(),
    )
    assert record["model"] == "dcdi"
    assert record["condition"] == "standardised"
    assert record["n_iterations"] == 80000
    assert record["hyperparameters"] == dict(
        job.candidate.grid_point_hyperparameter
    )


def test_adapter_output_passes_ranker_consumption() -> None:
    """A full 40-record set built by the adapter is acceptable to the ranker.

    Builds one adapted record per calibration job and feeds the
    collection into rank_calibration_records to confirm structural
    compatibility end-to-end.
    """
    jobs = _make_jobs()
    records = []
    for job in jobs:
        fit_result = _synthetic_fit_result(job)
        records.append(
            _adapt_fit_result_to_calibration_record(
                job,
                fit_result,
                threshold_robustness_records=_synthetic_threshold_records(),
            )
        )
    output = rank_calibration_records(records)
    assert "candidate_ranking" in output
    assert "selections" in output


def test_adapter_rejects_wrong_model() -> None:
    """An identity mismatch on model raises ValueError."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["model"] = "dcdi"
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "model" in str(excinfo.value)


def test_adapter_rejects_wrong_condition() -> None:
    """An identity mismatch on condition raises ValueError."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["condition"] = "standardised"
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "condition" in str(excinfo.value)


def test_adapter_rejects_wrong_configuration_hash() -> None:
    """An identity mismatch on configuration_hash raises ValueError."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["configuration_hash"] = "0" * 64
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "configuration_hash" in str(excinfo.value)


def test_adapter_rejects_wrong_seed_replicate_index() -> None:
    """A mismatch on seed_replicate_index raises ValueError."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["seed_replicate_index"] = (
        0 if int(job.seed_replicate_index) == 1 else 1
    )
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "seed_replicate_index" in str(excinfo.value)


def test_adapter_rejects_wrong_seed_population() -> None:
    """A non-calibration seed_population raises ValueError."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["seed_population"] = "held_out_evaluation"
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "seed_population" in str(excinfo.value)


def test_adapter_rejects_wrong_run_id() -> None:
    """A run_id that does not match the canonical derivation raises."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["run_id"] = "fabricated__id"
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "run_id" in str(excinfo.value)


@pytest.mark.parametrize(
    "missing_field",
    [
        "shd",
        "sid",
        "mmd_primary",
        "graph_status",
        "sampler_status",
        "training_status",
        "runtime_seconds",
        "n_iterations",
        "interventions",
        "configuration_hash",
        "seed_replicate_index",
        "seed_population",
        "model",
        "condition",
        "run_id",
    ],
)
def test_adapter_rejects_missing_production_field(
    missing_field: str,
) -> None:
    """Removing one required production field raises ValueError naming it."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result.pop(missing_field, None)
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert missing_field in str(excinfo.value)


def test_adapter_threshold_records_none_yields_empty_list() -> None:
    """Passing ``None`` for threshold records yields an empty list."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    record = _adapt_fit_result_to_calibration_record(
        job, fit_result, threshold_robustness_records=None
    )
    assert record["threshold_metrics"] == []


def test_adapter_missing_bandwidth_summaries_yields_empty_dict() -> None:
    """A missing mmd_bandwidth_used_value yields an empty bandwidth_summaries."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(job)
    fit_result["mmd_bandwidth_used_value"] = None
    record = _adapt_fit_result_to_calibration_record(
        job,
        fit_result,
        threshold_robustness_records=_synthetic_threshold_records(),
    )
    assert record["bandwidth_summaries"] == {}


def test_adapter_non_finite_metric_becomes_none() -> None:
    """A NaN mmd_primary input is converted to None in the adapted record."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(
        job, mmd_primary=float("nan")
    )
    record = _adapt_fit_result_to_calibration_record(
        job,
        fit_result,
        threshold_robustness_records=_synthetic_threshold_records(),
    )
    assert record["mmd_primary"] is None


def test_adapter_no_non_finite_floats_in_output() -> None:
    """The adapted record contains no raw NaN/+inf/-inf float at any depth."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(
        job,
        mmd_primary=float("inf"),
        runtime_seconds=12.5,
    )
    # Force a non-finite MMD value inside the intervention list.
    fit_result["interventions"][0]["mmd_value"] = float("-inf")
    with pytest.raises(ValueError):
        # runtime_seconds is fine, but mmd_primary=inf is rejected
        # at the runtime-seconds validator step only if it shows up
        # there; mmd_primary inf is sanitised by the adapter and
        # not rejected. Validate first that sanitisation works.
        _adapt_fit_result_to_calibration_record(
            job,
            {**fit_result, "runtime_seconds": float("nan")},
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    # Now exercise the sanitisation path: finite runtime, non-finite
    # mmd inputs that the adapter must convert to None.
    record = _adapt_fit_result_to_calibration_record(
        job,
        fit_result,
        threshold_robustness_records=_synthetic_threshold_records(),
    )

    def _walk(obj: Any, path: str = "$") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                _walk(item, f"{path}[{index}]")
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, float):
            assert math.isfinite(obj), (
                f"non-finite float at {path}: {obj!r}"
            )

    _walk(record)


def test_adapter_rejects_non_finite_runtime_seconds() -> None:
    """A non-finite runtime_seconds is rejected as infrastructure failure."""
    job = _pick_job("dagma", "centred_only")
    fit_result = _synthetic_fit_result(
        job, runtime_seconds=float("nan")
    )
    with pytest.raises(ValueError) as excinfo:
        _adapt_fit_result_to_calibration_record(
            job,
            fit_result,
            threshold_robustness_records=_synthetic_threshold_records(),
        )
    assert "runtime_seconds" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Default fit-runner / lazy-import end-to-end
# ---------------------------------------------------------------------------


def _copy_calibration_configs_to_tmp(tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    target.mkdir()
    for filename in _PARENT_FILENAMES:
        shutil.copy(_CALIBRATION_CONFIG_DIR / filename, target / filename)
    return target


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


def _write_synthetic_run_json(
    *,
    fit_result: Mapping[str, Any],
    threshold_records: list[dict[str, Any]],
    base_dir: Path,
) -> Path:
    """Write a synthetic run.json under the per-run directory.

    The path mirrors the layout used by the production pipeline
    (``base_dir/model/condition/seed_population/seed<idx>/<hash12>/``)
    and the sibling threshold_robustness.json is written too so the
    threshold-robustness recomputation in the default fit_runner
    does not need to access numeric arrays. (The recompute call is
    monkeypatched separately to skip the heavy work.)
    """
    model = fit_result["model"]
    condition = fit_result["condition"]
    seed_population = fit_result["seed_population"]
    seed_replicate_index = int(fit_result["seed_replicate_index"])
    configuration_hash = fit_result["configuration_hash"]
    run_dir = (
        base_dir
        / model
        / condition
        / seed_population
        / f"seed{seed_replicate_index}"
        / configuration_hash[:HASH_PREFIX_LENGTH]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = run_dir / "run.json"
    run_json_path.write_text(
        json.dumps(
            fit_result, sort_keys=True, ensure_ascii=True, indent=2
        ),
        encoding="utf-8",
    )
    # Sibling artefact placeholder so loader does not check it. The
    # threshold-robustness records come back from the patched
    # recompute_at_thresholds below.
    (run_dir / "threshold_robustness.json").write_text(
        json.dumps(
            {"records": threshold_records},
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_json_path


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fit_result_factory: Callable[[CalibrationFitJob], Mapping[str, Any]],
    on_fit: Callable[[CalibrationFitJob], None] | None = None,
) -> dict[str, int]:
    """Monkeypatch pipeline.run_single_fit and threshold_robustness.recompute.

    Returns a counter dict whose ``count`` key tracks the number of
    fake run_single_fit invocations.
    """
    from experiments.selection_study import pipeline
    from experiments.selection_study import threshold_robustness

    counter = {"count": 0}

    def fake_run_single_fit(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        counter["count"] += 1
        entry = manifest.entries[entry_index]
        # Locate the orchestration job for this entry by matching on
        # identity. We need a CalibrationFitJob to call the fit
        # factory, so reconstruct one by walking the full workload.
        from experiments.selection_study.calibration import (
            enumerate_calibration_workload,
        )

        parents = tuple(
            load_config(_CALIBRATION_CONFIG_DIR / name)
            for name in _PARENT_FILENAMES
        )
        workload = enumerate_calibration_workload(parents)
        matching_job: CalibrationFitJob | None = None
        for job in workload.fit_jobs:
            if (
                job.candidate.model == entry.model
                and job.candidate.condition == entry.condition
                and job.candidate.configuration_hash_full
                == entry.configuration_hash
                and int(job.seed_replicate_index)
                == int(entry.seed_replicate_index)
            ):
                matching_job = job
                break
        if matching_job is None:
            raise AssertionError(
                "fake pipeline could not match manifest entry to a "
                "CalibrationFitJob"
            )
        if on_fit is not None:
            on_fit(matching_job)
        fit_result = fit_result_factory(matching_job)
        return _write_synthetic_run_json(
            fit_result=fit_result,
            threshold_records=_synthetic_threshold_records(),
            base_dir=run_root,
        )

    monkeypatch.setattr(pipeline, "run_single_fit", fake_run_single_fit)

    def fake_recompute(
        run_dir: Any, *, write_sibling: bool = True
    ) -> dict[str, Any]:
        return {"records": _synthetic_threshold_records()}

    monkeypatch.setattr(
        threshold_robustness, "recompute_at_thresholds", fake_recompute
    )
    return counter


def test_default_fit_runner_calls_pipeline_run_single_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_calibration(..., fit_runner=None) drives pipeline.run_single_fit."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    counter = _install_fake_pipeline(
        monkeypatch,
        fit_result_factory=lambda job: _synthetic_fit_result(job),
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=None,
        now_fn=_FakeClock(),
    )
    assert artefact_path.is_file()
    assert counter["count"] == 40


def test_pipeline_import_is_lazy() -> None:
    """Importing calibration.py at module load does not pull in pipeline.

    Runs the import probe in a clean subprocess so the check does not
    perturb sys.modules in the parent pytest process. The subprocess
    imports ``experiments.selection_study.calibration`` from a fresh
    interpreter and reports whether
    ``experiments.selection_study.pipeline`` was loaded as a side
    effect of the import.
    """
    import subprocess
    import sys as _sys

    probe = (
        "import sys\n"
        "import experiments.selection_study.calibration\n"
        "loaded = ("
        "'experiments.selection_study.pipeline' in sys.modules"
        ")\n"
        "wrappers_loaded = any(\n"
        "    name == 'dagma' or name.startswith('dagma.') or\n"
        "    name == 'dcdi' or name.startswith('dcdi.') or\n"
        "    name.startswith('symbolic_priors_cd.wrappers')\n"
        "    for name in sys.modules\n"
        ")\n"
        "sys.stdout.write(\n"
        "    f'PIPELINE_LOADED={loaded}; "
        "WRAPPERS_LOADED={wrappers_loaded}\\n'\n"
        ")\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", probe],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess probe failed: stderr={result.stderr!r}"
    )
    assert "PIPELINE_LOADED=False" in result.stdout, (
        "importing experiments.selection_study.calibration pulled in "
        "experiments.selection_study.pipeline at module import time; "
        "the production fit path must be imported lazily inside the "
        "default fit-runner factory. stdout="
        f"{result.stdout!r}"
    )
    assert "WRAPPERS_LOADED=False" in result.stdout, (
        "importing experiments.selection_study.calibration pulled in "
        "a wrapper namespace at module import time; the calibration "
        "module must remain wrapper-free at import. stdout="
        f"{result.stdout!r}"
    )


def test_default_fit_path_writes_artefact_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full orchestration via the default fit_runner completes cleanly."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    _install_fake_pipeline(
        monkeypatch,
        fit_result_factory=lambda job: _synthetic_fit_result(job),
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=None,
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    assert len(list(records_dir.glob("*.json"))) == 40
    with artefact_path.open(encoding="utf-8") as handle:
        artefact = json.load(handle)
    validate_selected_configurations_artefact(artefact)


def test_default_fit_path_writes_records_through_synthetic_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path drives the fake pipeline, no wrapper code path needed.

    Confirms the contract by checking that the orchestration completes
    using only the monkeypatched ``pipeline.run_single_fit`` (the
    sentinel counter increments once per fit) and that the resulting
    selected-configurations artefact validates. The wrapper namespace
    is not directly inspected here because pytest's full-suite order
    may have loaded wrapper modules in unrelated tests; the lazy-
    import subprocess probe (``test_pipeline_import_is_lazy``) is
    where the wrapper-namespace invariant is checked.
    """
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    counter = _install_fake_pipeline(
        monkeypatch,
        fit_result_factory=lambda job: _synthetic_fit_result(job),
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=None,
        now_fn=_FakeClock(),
    )
    assert counter["count"] == 40
    with artefact_path.open(encoding="utf-8") as handle:
        validate_selected_configurations_artefact(json.load(handle))


# ---------------------------------------------------------------------------
# Failure handling via the default fit path
# ---------------------------------------------------------------------------


def test_default_fit_path_failure_creates_degenerate_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pipeline.run_single_fit exception produces a degenerate record."""
    from experiments.selection_study import pipeline

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    # Build a fake pipeline that raises on a deterministic subset of
    # calibration jobs (DCDI standardised first grid point seed 201).
    counter = {"count": 0}

    def fake_run_single_fit(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        counter["count"] += 1
        entry = manifest.entries[entry_index]
        if (
            entry.model == "dcdi"
            and entry.condition == "standardised"
            and int(entry.seed_replicate_index) == 0
        ):
            raise RuntimeError(
                "synthetic pipeline failure for testing"
            )
        # For other jobs, build a synthetic run.json matching the
        # manifest entry's identity.
        from experiments.selection_study.calibration import (
            enumerate_calibration_workload,
        )

        parents = tuple(
            load_config(_CALIBRATION_CONFIG_DIR / name)
            for name in _PARENT_FILENAMES
        )
        workload = enumerate_calibration_workload(parents)
        matching_job: CalibrationFitJob | None = None
        for job in workload.fit_jobs:
            if (
                job.candidate.model == entry.model
                and job.candidate.condition == entry.condition
                and job.candidate.configuration_hash_full
                == entry.configuration_hash
                and int(job.seed_replicate_index)
                == int(entry.seed_replicate_index)
            ):
                matching_job = job
                break
        assert matching_job is not None
        fit_result = _synthetic_fit_result(matching_job)
        return _write_synthetic_run_json(
            fit_result=fit_result,
            threshold_records=_synthetic_threshold_records(),
            base_dir=run_root,
        )

    monkeypatch.setattr(pipeline, "run_single_fit", fake_run_single_fit)
    monkeypatch.setattr(
        "experiments.selection_study.threshold_robustness."
        "recompute_at_thresholds",
        lambda run_dir, *, write_sibling=True: {
            "records": _synthetic_threshold_records()
        },
    )

    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=None,
        now_fn=_FakeClock(),
    )
    records_dir = artefact_path.parent / "records"
    files = sorted(records_dir.glob("dcdi_standardised_*_seed201.json"))
    assert len(files) == 5
    degenerate_records = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record["graph_status"] == "failed":
            degenerate_records.append(record)
    assert len(degenerate_records) == 5  # all 5 grid points share the trigger
    for record in degenerate_records:
        assert record["sampler_status"] == "failed"
        assert record["training_status"] == "failed"
        assert record["shd"] is None
        assert record["sid"] is None
        assert record["mmd_primary"] is None
        assert record["threshold_metrics"] == []
        assert record["mmd_by_intervention"] == []
        assert record["bandwidth_summaries"] == {}


def test_default_fit_path_malformed_pipeline_output_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If pipeline.run_single_fit writes a run.json missing a required field,
    the orchestrator fails fast and does not write selected_configurations.json.
    """
    from experiments.selection_study import pipeline

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    def malformed_pipeline(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        entry = manifest.entries[entry_index]
        from experiments.selection_study.calibration import (
            enumerate_calibration_workload,
        )

        parents = tuple(
            load_config(_CALIBRATION_CONFIG_DIR / name)
            for name in _PARENT_FILENAMES
        )
        workload = enumerate_calibration_workload(parents)
        matching_job = next(
            job
            for job in workload.fit_jobs
            if (
                job.candidate.model == entry.model
                and job.candidate.condition == entry.condition
                and job.candidate.configuration_hash_full
                == entry.configuration_hash
                and int(job.seed_replicate_index)
                == int(entry.seed_replicate_index)
            )
        )
        fit_result = _synthetic_fit_result(matching_job)
        del fit_result["sid"]
        return _write_synthetic_run_json(
            fit_result=fit_result,
            threshold_records=_synthetic_threshold_records(),
            base_dir=run_root,
        )

    monkeypatch.setattr(pipeline, "run_single_fit", malformed_pipeline)
    monkeypatch.setattr(
        "experiments.selection_study.threshold_robustness."
        "recompute_at_thresholds",
        lambda run_dir, *, write_sibling=True: {
            "records": _synthetic_threshold_records()
        },
    )
    # The malformed pipeline output is caught by the loader (which
    # requires all mandatory run.json fields); the default fit
    # runner wraps the resulting error as
    # _CalibrationInfrastructureError so the orchestrator aborts
    # rather than converting the broken assumption into a degenerate
    # record.
    with pytest.raises(_CalibrationInfrastructureError):
        run_calibration(
            config_dir,
            results_root,
            fit_runner=None,
            now_fn=_FakeClock(),
        )
    candidate_paths = list(
        (results_root / "model_selection" / "calibration").rglob(
            SELECTED_CONFIGURATIONS_FILENAME
        )
    )
    assert candidate_paths == []


def test_default_fit_path_keeps_persisted_records_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records persisted before infrastructure failure remain on disk."""
    from experiments.selection_study import pipeline

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    call_index = {"value": 0}

    def maybe_malformed(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        call_index["value"] += 1
        entry = manifest.entries[entry_index]
        from experiments.selection_study.calibration import (
            enumerate_calibration_workload,
        )

        parents = tuple(
            load_config(_CALIBRATION_CONFIG_DIR / name)
            for name in _PARENT_FILENAMES
        )
        workload = enumerate_calibration_workload(parents)
        matching_job = next(
            job
            for job in workload.fit_jobs
            if (
                job.candidate.model == entry.model
                and job.candidate.condition == entry.condition
                and job.candidate.configuration_hash_full
                == entry.configuration_hash
                and int(job.seed_replicate_index)
                == int(entry.seed_replicate_index)
            )
        )
        fit_result = _synthetic_fit_result(matching_job)
        if call_index["value"] == 3:
            # Use a mismatched model on the third invocation so the
            # adapter raises before the malformed run.json reaches
            # disk via the loader.
            other_model = (
                "dcdi" if fit_result["model"] == "dagma" else "dagma"
            )
            fit_result["model"] = other_model
        return _write_synthetic_run_json(
            fit_result=fit_result,
            threshold_records=_synthetic_threshold_records(),
            base_dir=run_root,
        )

    monkeypatch.setattr(pipeline, "run_single_fit", maybe_malformed)
    monkeypatch.setattr(
        "experiments.selection_study.threshold_robustness."
        "recompute_at_thresholds",
        lambda run_dir, *, write_sibling=True: {
            "records": _synthetic_threshold_records()
        },
    )
    with pytest.raises(_CalibrationInfrastructureError):
        run_calibration(
            config_dir,
            results_root,
            fit_runner=None,
            now_fn=_FakeClock(),
        )
    record_dirs = list(
        (results_root / "model_selection" / "calibration").glob(
            "*/records"
        )
    )
    assert len(record_dirs) == 1
    record_files = list(record_dirs[0].iterdir())
    # The third invocation raises before persistence, so the first
    # two records remain on disk.
    assert len(record_files) == 2


# ---------------------------------------------------------------------------
# Scope confirmations
# ---------------------------------------------------------------------------


def test_artefact_carries_no_forbidden_winner_fields_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artefact written by the default path carries no winner field."""
    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    _install_fake_pipeline(
        monkeypatch,
        fit_result_factory=lambda job: _synthetic_fit_result(job),
    )
    artefact_path = run_calibration(
        config_dir,
        results_root,
        fit_runner=None,
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

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert key not in forbidden
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(artefact)


def test_missing_manifest_entry_is_classified_as_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest with no matching calibration entry aborts the run.

    Monkeypatches ``preflight.enumerate_manifest`` so the returned
    manifest carries no entry matching the job's seed_replicate_index.
    The default fit runner detects the broken assumption and raises
    ``_CalibrationInfrastructureError``; the orchestrator must let
    the exception propagate so the run aborts rather than masking
    the broken assumption with a degenerate record. No
    ``selected_configurations.json`` is written.
    """
    from experiments.selection_study import preflight
    from experiments.selection_study.preflight import Manifest

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"

    real_enumerate = preflight.enumerate_manifest

    def empty_manifest_enumerate(
        config: Any, *, base_dir: Path = preflight._DEFAULT_BASE_DIR
    ) -> Manifest:
        full_manifest = real_enumerate(config, base_dir=base_dir)
        # Drop every calibration entry so the default fit runner's
        # lookup cannot locate a matching entry for any job.
        filtered_entries = tuple(
            entry
            for entry in full_manifest.entries
            if entry.seed_population != "calibration"
        )
        return Manifest(
            configuration_hash=full_manifest.configuration_hash,
            schema_version=full_manifest.schema_version,
            seed_derivation_rule=full_manifest.seed_derivation_rule,
            configuration_hash_algorithm=(
                full_manifest.configuration_hash_algorithm
            ),
            resolved_config=full_manifest.resolved_config,
            entries=filtered_entries,
        )

    monkeypatch.setattr(
        preflight, "enumerate_manifest", empty_manifest_enumerate
    )
    # The default fit runner imports preflight.enumerate_manifest
    # lazily via a from-import; patching the attribute on the module
    # object is sufficient because the from-import resolves on call.
    # Also patch the symbol used by run_calibration's own workload
    # enumeration so it does not also fail; the orchestrator builds
    # the workload before invoking the default fit runner.
    # ``enumerate_calibration_workload`` does not rely on
    # ``enumerate_manifest`` so the orchestrator's workload step is
    # unaffected.

    with pytest.raises(_CalibrationInfrastructureError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=None,
            now_fn=_FakeClock(),
        )
    message = str(excinfo.value)
    assert "manifest entry" in message
    # The error message must name the offending job's identity so
    # the operator can locate the failure quickly.
    assert "seed_value=" in message
    assert "seed_replicate_index=" in message
    # No artefact is written; previously persisted records (if any
    # were possible) would remain on disk. In this scenario the
    # very first job fails to locate its manifest entry so no
    # records are persisted at all.
    candidate_paths = list(
        (results_root / "model_selection" / "calibration").rglob(
            SELECTED_CONFIGURATIONS_FILENAME
        )
    )
    assert candidate_paths == []


def test_file_exists_error_from_pipeline_is_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing raw run directory aborts the run as infrastructure failure.

    pipeline.run_single_fit calls identity.create_run_directory, which
    raises FileExistsError when the target per-run directory is already
    populated. That is on-disk state left by a prior interrupted attempt
    (infrastructure) and not a model-fit fault. The default fit runner
    must re-raise the exception as _CalibrationInfrastructureError so
    the orchestrator aborts the run, leaves earlier records on disk,
    does not persist a degenerate placeholder for the affected job, and
    does not write selected_configurations.json.
    """
    from experiments.selection_study import pipeline
    from experiments.selection_study import threshold_robustness

    config_dir = _copy_calibration_configs_to_tmp(tmp_path)
    results_root = tmp_path / "results"
    call_index = {"value": 0}

    def fake_run_single_fit(
        manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        call_index["value"] += 1
        entry = manifest.entries[entry_index]
        # Raise FileExistsError on the third invocation so that the
        # first two jobs persist records before the abort, and we can
        # verify "earlier records remain on disk".
        if call_index["value"] == 3:
            stale_path = (
                run_root
                / entry.model
                / entry.condition
                / entry.seed_population
                / f"seed{int(entry.seed_replicate_index)}"
                / entry.configuration_hash[:HASH_PREFIX_LENGTH]
            )
            raise FileExistsError(
                "run directory is already populated; refusing to "
                f"overwrite: {stale_path}"
            )
        from experiments.selection_study.calibration import (
            enumerate_calibration_workload,
        )

        parents = tuple(
            load_config(_CALIBRATION_CONFIG_DIR / name)
            for name in _PARENT_FILENAMES
        )
        workload = enumerate_calibration_workload(parents)
        matching_job = next(
            job
            for job in workload.fit_jobs
            if (
                job.candidate.model == entry.model
                and job.candidate.condition == entry.condition
                and job.candidate.configuration_hash_full
                == entry.configuration_hash
                and int(job.seed_replicate_index)
                == int(entry.seed_replicate_index)
            )
        )
        fit_result = _synthetic_fit_result(matching_job)
        return _write_synthetic_run_json(
            fit_result=fit_result,
            threshold_records=_synthetic_threshold_records(),
            base_dir=run_root,
        )

    monkeypatch.setattr(pipeline, "run_single_fit", fake_run_single_fit)
    monkeypatch.setattr(
        threshold_robustness,
        "recompute_at_thresholds",
        lambda run_dir, *, write_sibling=True: {
            "records": _synthetic_threshold_records()
        },
    )

    with pytest.raises(_CalibrationInfrastructureError) as excinfo:
        run_calibration(
            config_dir,
            results_root,
            fit_runner=None,
            now_fn=_FakeClock(),
        )
    message = str(excinfo.value)
    assert "pre-existing raw run directory" in message
    assert "seed_value=" in message
    assert "model=" in message and "condition=" in message

    # The run aborted with infrastructure failure: no
    # selected_configurations.json was written under the calibration
    # tree.
    calibration_run_root = results_root / "model_selection" / "calibration"
    artefact_paths = list(
        calibration_run_root.rglob(SELECTED_CONFIGURATIONS_FILENAME)
    )
    assert artefact_paths == []

    # Two records (the successful first two fits) remain on disk; no
    # degenerate record was written for the FileExistsError job.
    record_dirs = list(calibration_run_root.glob("*/records"))
    assert len(record_dirs) == 1
    record_files = sorted(record_dirs[0].glob("*.json"))
    assert len(record_files) == 2
    for record_path in record_files:
        with record_path.open(encoding="utf-8") as handle:
            persisted = json.load(handle)
        # Each remaining record must be a healthy converged fit, not
        # a degenerate FileExistsError placeholder.
        assert persisted["graph_status"] == "valid_dag"
        assert persisted["sampler_status"] == "available"
        assert persisted["training_status"] == "converged"
        assert persisted.get("failure_type") is None
