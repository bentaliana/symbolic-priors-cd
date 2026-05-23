"""Tests for the held-out production-style fit-runner adapter.

These tests exercise ``_build_default_heldout_fit_runner`` end-to-end
with a synthetic Configuration factory, a monkeypatched
``pipeline.run_single_fit`` that writes a complete-enough synthetic
run.json, and a real ``loader.load_run`` reading it back. No real
wrapper fit is invoked, no live results are written, and the real
25-job held-out study is never executed.

Inspection tests also pin the underlying identity/path semantics:
seed_torch and seed_numpy enter the canonical Configuration payload,
so changing them changes both the configuration_hash and the run
directory path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    CalibrationConfiguration,
    Configuration,
    InterventionSpec,
    SEED_DERIVATION_RULE_NAME,
    canonical_json,
    configuration_hash as compute_configuration_hash,
)
from experiments.selection_study.held_out import (
    DCDI_MAIN_FIT_RNG,
    HELDOUT_SCM_SEEDS,
    HELDOUT_SEED_POPULATION,
    HeldoutJob,
    MAIN_JOB_KIND,
    SENSITIVITY_CONDITION,
    SENSITIVITY_FIT_RNGS,
    SENSITIVITY_JOB_KIND,
    SENSITIVITY_MODEL,
    SENSITIVITY_SCM_SEED,
    _HeldoutInfrastructureError,
    _adapt_pipeline_run_to_heldout_record,
    _apply_fit_rng_to_configuration,
    _build_default_heldout_fit_runner,
    enumerate_heldout_workload,
    run_held_out_evaluation,
)
from experiments.selection_study.held_out_artefact import (
    HELDOUT_EVALUATION_ARTEFACT_TYPE,
    validate_heldout_evaluation_artefact,
)
from experiments.selection_study.identity import derive_run_directory
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
        f"heldout-production-adapter-test|{model}|{condition}|"
        f"{hyper_value!r}"
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
        b"synthetic-heldout-production-adapter-parent"
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
# Minimal-valid Configuration builder
# ---------------------------------------------------------------------------


_INTERVENTION_A = InterventionSpec(
    intervention_id="intv_a", target_node=0, value_raw=2.0
)
_INTERVENTION_B = InterventionSpec(
    intervention_id="intv_b", target_node=1, value_raw=-2.0
)

_DAGMA_SCHEMA_GATE_FIELDS = dict(
    dagma_warm_iter=20000,
    dagma_max_iter=70000,
    dagma_lr=3e-4,
    dagma_beta_1=0.99,
    dagma_beta_2=0.999,
)
_DCDI_SCHEMA_GATE_FIELDS = dict(
    n_val_dcdi=32,
    dcdi_num_train_iter=30,
    dcdi_stop_crit_win=10,
    dcdi_train_patience=5,
    dcdi_train_batch_size=8,
    dcdi_lr=1e-3,
    dcdi_h_threshold=1e-8,
    dcdi_hidden_units=16,
    dcdi_hidden_layers=2,
)


def _make_configuration_for_job(job: HeldoutJob) -> Configuration:
    calibration_cfg = CalibrationConfiguration(
        name="selected",
        hyperparameters=tuple(
            (name, float(value))
            for name, value in sorted(job.hyperparameters.items())
        ),
    )
    seed_populations = (
        (HELDOUT_SEED_POPULATION, tuple(int(s) for s in HELDOUT_SCM_SEEDS)),
    )
    if job.model == "dagma":
        return Configuration(
            model="dagma",
            condition=job.condition,
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=seed_populations,
            intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
            calibration_configurations=(calibration_cfg,),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
            seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
            configuration_hash_algorithm=(
                CONFIGURATION_HASH_ALGORITHM_NAME
            ),
            **_DAGMA_SCHEMA_GATE_FIELDS,
        )
    return Configuration(
        model="dcdi",
        condition=job.condition,
        seed_torch=int(DCDI_MAIN_FIT_RNG),
        seed_numpy=int(DCDI_MAIN_FIT_RNG),
        seed_dagma=None,
        seed_populations=seed_populations,
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        calibration_configurations=(calibration_cfg,),
        threshold_robustness_triple=(0.4, 0.5, 0.6),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **_DCDI_SCHEMA_GATE_FIELDS,
    )


# ---------------------------------------------------------------------------
# Fake pipeline.run_single_fit
# ---------------------------------------------------------------------------


def _make_full_run_json(
    *,
    entry: Any,
    resolved_config: dict[str, Any],
    sid: int,
    shd: int,
    mmd_primary: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build a complete-enough run.json payload accepted by loader.load_run."""
    return {
        "run_id": entry.expected_run_id,
        "schema_version": 1,
        "model": entry.model,
        "condition": entry.condition,
        "seed_population": entry.seed_population,
        "seed_replicate_index": int(entry.seed_replicate_index),
        "configuration_hash": entry.configuration_hash,
        "graph_seed": int(entry.graph_seed),
        "git_hash": "synthetic",
        "env_snapshot": "synthetic-test",
        "config_resolved": resolved_config,
        "seed_torch": resolved_config.get("seed_torch"),
        "seed_numpy": resolved_config.get("seed_numpy"),
        "seed_dagma": resolved_config.get("seed_dagma"),
        "model_sampling_seed_base": int(entry.model_sampling_seed_base),
        "model_sampling_seed_derivation_rule": SEED_DERIVATION_RULE_NAME,
        "train_data_seed": int(entry.train_data_seed),
        "validation_data_seed": (
            None
            if entry.validation_data_seed is None
            else int(entry.validation_data_seed)
        ),
        "intervention_ground_truth_seed_base": int(
            entry.intervention_ground_truth_seed_base
        ),
        "training_status": "converged",
        "n_iterations": None,
        "runtime_seconds": float(runtime_seconds),
        "loss_history": None,
        "loss_history_status": "unavailable_no_api",
        "graph_status": "valid_dag",
        "graph_status_reason": None,
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": int(shd),
        "sid": int(sid),
        "mmd_primary": float(mmd_primary),
        "mmd_sensitivity_unit_variance": None,
        "mmd_bandwidth_sweep": {},
        "validation_nll": None,
        "sampler_status": "available",
        "sampler_status_reason": None,
        "sampler_policy_used": (
            "dcdi_native" if entry.model == "dcdi" else "residual_fitted"
        ),
        "mmd_available_count": 0,
        "mmd_missing_count": 0,
        "invalid_graph_for_this_run": False,
        "shd_reversal_cost": 2,
        "mmd_bandwidth_used_value": {},
        "mmd_clip_policy": "no_clip",
        "sid_backend": "gadjid",
        "sid_backend_version": "0.1.0",
        "sid_argument_order": "predicted_then_true",
        "sid_return_value": "raw_mistake_count",
        "configuration_hash_algorithm": CONFIGURATION_HASH_ALGORITHM_NAME,
        "wrapper_diagnostics": {},
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": [],
    }


class _FakePipeline:
    """Recording-and-writing fake for ``pipeline.run_single_fit``.

    Each invocation captures the manifest and entry, writes a
    complete-enough run.json into the derived run directory, and
    returns the path to that run.json.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, manifest: Any, entry_index: int, *, run_root: Path
    ) -> Path:
        entry = manifest.entries[entry_index]
        resolved_config = dict(manifest.resolved_config)
        run_dir = derive_run_directory(
            model=entry.model,
            condition=entry.condition,
            seed_population=entry.seed_population,
            seed_replicate_index=int(entry.seed_replicate_index),
            configuration_hash=entry.configuration_hash,
            base_dir=Path(run_root),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "manifest": manifest,
                "entry": entry,
                "entry_index": entry_index,
                "run_root": Path(run_root),
                "run_dir": run_dir,
                "seed_torch": resolved_config.get("seed_torch"),
                "seed_numpy": resolved_config.get("seed_numpy"),
                "seed_dagma": resolved_config.get("seed_dagma"),
                "configuration_hash": entry.configuration_hash,
            }
        )
        # Deterministic metrics so the artefact-builder is happy.
        sid = int((entry.seed_replicate_index * 7) % 11)
        shd = int((entry.seed_replicate_index * 3) % 5)
        mmd_primary = 0.001 * (entry.seed_replicate_index + 1)
        payload = _make_full_run_json(
            entry=entry,
            resolved_config=resolved_config,
            sid=sid,
            shd=shd,
            mmd_primary=mmd_primary,
            runtime_seconds=0.25,
        )
        run_json_path = run_dir / "run.json"
        run_json_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return run_json_path


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakePipeline:
    """Monkeypatch the pipeline import the adapter uses."""
    import experiments.selection_study.pipeline as pipeline_mod

    fake = _FakePipeline()
    monkeypatch.setattr(pipeline_mod, "run_single_fit", fake)
    return fake


def _install_raising_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, exception: Exception
) -> None:
    import experiments.selection_study.pipeline as pipeline_mod

    def _raiser(*args: Any, **kwargs: Any) -> None:
        raise exception

    monkeypatch.setattr(pipeline_mod, "run_single_fit", _raiser)


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
            f"forbidden phrase {phrase!r} appeared in production "
            "adapter output text"
        )


# ---------------------------------------------------------------------------
# Identity-semantics inspection
# ---------------------------------------------------------------------------


def test_seed_torch_and_seed_numpy_are_part_of_configuration_hash() -> None:
    """Changing seed_torch/seed_numpy changes the configuration_hash."""
    base_job = HeldoutJob(
        job_kind=MAIN_JOB_KIND,
        model="dcdi",
        condition="centred_only",
        configuration_hash_full="x" * 64,
        configuration_hash_prefix="x" * 12,
        hyperparameters={"reg_coeff": 0.1},
        scm_seed=301,
        fit_rng=42,
        calibration_run_hash_prefix="a" * 12,
    )
    base_config = _make_configuration_for_job(base_job)
    hashes: dict[int, str] = {}
    for fit_rng in (42, 43, 44, 45, 46, 47):
        modified = dataclasses.replace(
            base_config,
            seed_torch=int(fit_rng),
            seed_numpy=int(fit_rng),
        )
        hashes[fit_rng] = compute_configuration_hash(modified)
    assert len(set(hashes.values())) == len(hashes), hashes


def test_dcdi_seed_301_run_directories_are_distinct_across_fit_rngs(
    tmp_path: Path,
) -> None:
    """The on-disk run directory differs for each fit_rng in 42..47."""
    base_job = HeldoutJob(
        job_kind=MAIN_JOB_KIND,
        model="dcdi",
        condition="centred_only",
        configuration_hash_full="x" * 64,
        configuration_hash_prefix="x" * 12,
        hyperparameters={"reg_coeff": 0.1},
        scm_seed=SENSITIVITY_SCM_SEED,
        fit_rng=42,
        calibration_run_hash_prefix="a" * 12,
    )
    base_config = _make_configuration_for_job(base_job)
    base_dir = tmp_path / "results" / "model_selection"
    seen_dirs: set[Path] = set()
    for fit_rng in (42, 43, 44, 45, 46, 47):
        modified = dataclasses.replace(
            base_config,
            seed_torch=int(fit_rng),
            seed_numpy=int(fit_rng),
        )
        config_hash = compute_configuration_hash(modified)
        run_dir = derive_run_directory(
            model="dcdi",
            condition="centred_only",
            seed_population=HELDOUT_SEED_POPULATION,
            seed_replicate_index=0,
            configuration_hash=config_hash,
            base_dir=base_dir,
        )
        seen_dirs.add(run_dir)
    assert len(seen_dirs) == 6, seen_dirs


def test_apply_fit_rng_dcdi_main_sets_seed_42() -> None:
    job = HeldoutJob(
        job_kind=MAIN_JOB_KIND,
        model="dcdi",
        condition="centred_only",
        configuration_hash_full="x" * 64,
        configuration_hash_prefix="x" * 12,
        hyperparameters={"reg_coeff": 0.1},
        scm_seed=301,
        fit_rng=42,
        calibration_run_hash_prefix="a" * 12,
    )
    config = _make_configuration_for_job(job)
    modified = _apply_fit_rng_to_configuration(config, job)
    assert modified.seed_torch == DCDI_MAIN_FIT_RNG == 42
    assert modified.seed_numpy == DCDI_MAIN_FIT_RNG == 42
    assert modified.seed_dagma is None


def test_apply_fit_rng_dcdi_sensitivity_sets_fit_rng() -> None:
    for fit_rng in SENSITIVITY_FIT_RNGS:
        job = HeldoutJob(
            job_kind=SENSITIVITY_JOB_KIND,
            model=SENSITIVITY_MODEL,
            condition=SENSITIVITY_CONDITION,
            configuration_hash_full="x" * 64,
            configuration_hash_prefix="x" * 12,
            hyperparameters={"reg_coeff": 0.1},
            scm_seed=SENSITIVITY_SCM_SEED,
            fit_rng=int(fit_rng),
            calibration_run_hash_prefix="a" * 12,
        )
        config = _make_configuration_for_job(job)
        modified = _apply_fit_rng_to_configuration(config, job)
        assert modified.seed_torch == int(fit_rng), fit_rng
        assert modified.seed_numpy == int(fit_rng), fit_rng
        assert modified.seed_dagma is None


def test_apply_fit_rng_dagma_leaves_seeds_none() -> None:
    job = HeldoutJob(
        job_kind=MAIN_JOB_KIND,
        model="dagma",
        condition="centred_only",
        configuration_hash_full="x" * 64,
        configuration_hash_prefix="x" * 12,
        hyperparameters={"lambda1": 0.1},
        scm_seed=301,
        fit_rng=None,
        calibration_run_hash_prefix="a" * 12,
    )
    config = _make_configuration_for_job(job)
    modified = _apply_fit_rng_to_configuration(config, job)
    assert modified is config
    assert modified.seed_torch is None
    assert modified.seed_numpy is None
    assert modified.seed_dagma is None


# ---------------------------------------------------------------------------
# Default fit_runner availability
# ---------------------------------------------------------------------------


def test_default_fit_runner_is_now_available(tmp_path: Path) -> None:
    runner = _build_default_heldout_fit_runner(
        results_root=tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )
    assert callable(runner)


def test_default_fit_runner_requires_configuration_factory(
    tmp_path: Path,
) -> None:
    with pytest.raises(NotImplementedError, match="configuration_factory"):
        _build_default_heldout_fit_runner(
            results_root=tmp_path / "results",
            configuration_factory=None,  # type: ignore[arg-type]
        )


def test_run_held_out_evaluation_uses_default_when_factory_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)

    artefact_out = run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    assert len(fake_pipeline.calls) == 25
    assert artefact_out.is_file()
    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(payload)
    assert payload["artefact_type"] == HELDOUT_EVALUATION_ARTEFACT_TYPE


def test_no_explicit_args_still_raises_not_implemented(
    tmp_path: Path,
) -> None:
    artefact_path = _write_calibration_artefact(tmp_path)
    with pytest.raises(NotImplementedError, match="fit_runner"):
        run_held_out_evaluation(artefact_path, tmp_path / "results")


# ---------------------------------------------------------------------------
# Seed-RNG injection observed at the pipeline boundary
# ---------------------------------------------------------------------------


def _calls_matching(
    fake: _FakePipeline,
    *,
    model: str,
    condition: str,
    seed_replicate_index: int | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for call in fake.calls:
        entry = call["entry"]
        if entry.model != model:
            continue
        if entry.condition != condition:
            continue
        if (
            seed_replicate_index is not None
            and int(entry.seed_replicate_index) != seed_replicate_index
        ):
            continue
        matches.append(call)
    return matches


def test_dcdi_main_jobs_use_seed_torch_and_seed_numpy_42(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    # Inspect every DCDI main call (10 calls = 2 conditions x 5 seeds).
    dcdi_main_calls = [
        call
        for call in fake_pipeline.calls
        if call["entry"].model == "dcdi"
        and call["entry"].seed_population == HELDOUT_SEED_POPULATION
        # The DCDI main calls share the configuration_hash inside
        # the calibration-selected cell; sensitivity calls don't end
        # up under any main cell because they live at scm_seed=301
        # only with fit_rng overrides 43..47. Distinguish here by
        # the captured seed values.
        and call["seed_torch"] == DCDI_MAIN_FIT_RNG
    ]
    assert len(dcdi_main_calls) == 10
    for call in dcdi_main_calls:
        assert call["seed_torch"] == 42
        assert call["seed_numpy"] == 42
        assert call["seed_dagma"] is None


def test_dcdi_sensitivity_jobs_use_seed_torch_and_seed_numpy_43_to_47(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    # Sensitivity calls are the only ones with seed_torch in {43..47}.
    sensitivity_seeds = {
        int(call["seed_torch"])
        for call in fake_pipeline.calls
        if call["seed_torch"] in set(SENSITIVITY_FIT_RNGS)
    }
    assert sensitivity_seeds == set(SENSITIVITY_FIT_RNGS)

    sensitivity_calls = [
        call
        for call in fake_pipeline.calls
        if call["seed_torch"] in set(SENSITIVITY_FIT_RNGS)
    ]
    assert len(sensitivity_calls) == 5
    for call in sensitivity_calls:
        entry = call["entry"]
        assert entry.model == SENSITIVITY_MODEL == "dcdi"
        assert entry.condition == SENSITIVITY_CONDITION == "centred_only"
        assert int(entry.seed_replicate_index) == 0  # SCM seed 301
        assert call["seed_torch"] == call["seed_numpy"]
        assert call["seed_dagma"] is None


def test_dagma_jobs_do_not_receive_dcdi_fit_rng_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    dagma_calls = [
        call
        for call in fake_pipeline.calls
        if call["entry"].model == "dagma"
    ]
    assert len(dagma_calls) == 10  # 2 conditions x 5 seeds
    for call in dagma_calls:
        assert call["seed_torch"] is None
        assert call["seed_numpy"] is None
        assert call["seed_dagma"] is None


def test_dcdi_seed_301_fit_rng_42_to_47_yield_distinct_run_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    # The DCDI/centred_only/scm_seed=301 jobs span fit_rng=42 (one
    # main) plus fit_rng in {43..47} (five sensitivity). All six
    # share model+condition+seed_replicate_index=0 but the
    # configuration_hash differs because seed_torch/seed_numpy enter
    # the hash, so the run directories must all be distinct.
    target_calls = [
        call
        for call in fake_pipeline.calls
        if (
            call["entry"].model == "dcdi"
            and call["entry"].condition == "centred_only"
            and int(call["entry"].seed_replicate_index) == 0
        )
    ]
    assert len(target_calls) == 6
    distinct_dirs = {call["run_dir"] for call in target_calls}
    assert len(distinct_dirs) == 6
    distinct_hashes = {
        call["configuration_hash"] for call in target_calls
    }
    assert len(distinct_hashes) == 6


# ---------------------------------------------------------------------------
# Failure modes through the production adapter
# ---------------------------------------------------------------------------


def test_pipeline_file_exists_error_propagates_as_infrastructure_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_raising_pipeline(
        monkeypatch,
        exception=FileExistsError("stale per-run directory"),
    )
    artefact_path = _write_calibration_artefact(tmp_path)
    with pytest.raises(
        _HeldoutInfrastructureError, match="pre-existing raw run directory"
    ):
        run_held_out_evaluation(
            artefact_path,
            tmp_path / "results",
            configuration_factory=_make_configuration_for_job,
        )


def test_ordinary_pipeline_exception_becomes_degenerate_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.selection_study.pipeline as pipeline_mod

    fake_pipeline = _FakePipeline()
    counter = {"calls": 0}

    def runner(*args: Any, **kwargs: Any) -> Path:
        counter["calls"] += 1
        if counter["calls"] == 3:
            raise RuntimeError("simulated model-fit blow-up")
        return fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "run_single_fit", runner)

    artefact_path = _write_calibration_artefact(tmp_path)
    artefact_out = run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(payload)
    # Exactly one cell has a degenerate record (the third call's
    # job). The aggregator surfaces the non-finite count.
    degenerate_counts = 0
    for condition in CONDITIONS:
        for model in MODELS:
            cell = payload["main_evaluation"]["cells"][condition][model]
            sid_summary = cell["aggregate_metrics"]["sid"]
            degenerate_counts += sid_summary["non_finite_count"]
    assert degenerate_counts >= 1


def test_full_run_with_default_adapter_writes_25_records_and_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    artefact_out = run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    payload = json.loads(artefact_out.read_text(encoding="utf-8"))
    validate_heldout_evaluation_artefact(payload)
    assert payload["status_summary"]["total_records"] == 25
    # The 25 records also exist as JSON files inside the held-out
    # records directory.
    workload = enumerate_heldout_workload(artefact_path)
    from experiments.selection_study.held_out import (
        heldout_records_dir_path,
        _heldout_run_hash_inputs_from_workload,
        compute_heldout_run_hash_full,
    )

    heldout_hash = compute_heldout_run_hash_full(
        **_heldout_run_hash_inputs_from_workload(workload)
    )
    hash12 = heldout_hash[:HASH_PREFIX_LENGTH]
    records_dir = heldout_records_dir_path(
        heldout_run_hash12=hash12,
        results_root=tmp_path / "results",
    )
    record_files = list(records_dir.glob("*.json"))
    assert len(record_files) == 25


def test_provenance_uses_calibration_hash_in_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pipeline = _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )

    # The per-fit records carry the calibration-selected
    # configuration_hash_full (provenance) and the execution
    # configuration_hash_full (the actually-executed Configuration's
    # hash). The two MUST differ because the synthetic calibration
    # hash is unrelated to the executable Configuration's hash.
    workload = enumerate_heldout_workload(artefact_path)
    from experiments.selection_study.held_out import (
        _heldout_run_hash_inputs_from_workload,
        _record_filename_for_job,
        compute_heldout_run_hash_full,
        heldout_records_dir_path,
    )

    heldout_hash = compute_heldout_run_hash_full(
        **_heldout_run_hash_inputs_from_workload(workload)
    )
    records_dir = heldout_records_dir_path(
        heldout_run_hash12=heldout_hash[:HASH_PREFIX_LENGTH],
        results_root=tmp_path / "results",
    )
    sample_job = workload.main_jobs[0]
    record_path = records_dir / _record_filename_for_job(sample_job)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert (
        record["configuration_hash_full"]
        == sample_job.configuration_hash_full
    )
    assert "execution_configuration_hash_full" in record
    assert (
        record["execution_configuration_hash_full"]
        != sample_job.configuration_hash_full
    )


# ---------------------------------------------------------------------------
# Forbidden language
# ---------------------------------------------------------------------------


def test_no_final_winner_language_in_default_adapter_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pipeline(monkeypatch)
    artefact_path = _write_calibration_artefact(tmp_path)
    artefact_out = run_held_out_evaluation(
        artefact_path,
        tmp_path / "results",
        configuration_factory=_make_configuration_for_job,
    )
    text = artefact_out.read_text(encoding="utf-8")
    _assert_no_forbidden_language(text)


# ---------------------------------------------------------------------------
# Adapter unit pieces
# ---------------------------------------------------------------------------


def test_adapt_pipeline_run_preserves_provenance_fields() -> None:
    job = HeldoutJob(
        job_kind=MAIN_JOB_KIND,
        model="dcdi",
        condition="centred_only",
        configuration_hash_full="a" * 64,
        configuration_hash_prefix="a" * 12,
        hyperparameters={"reg_coeff": 0.1},
        scm_seed=301,
        fit_rng=42,
        calibration_run_hash_prefix="c" * 12,
    )
    run_payload = {
        "run_id": "test-run",
        "sid": 3,
        "shd": 1,
        "mmd_primary": 0.005,
        "runtime_seconds": 1.5,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "n_iterations": None,
        "configuration_hash": "b" * 64,
    }
    record = _adapt_pipeline_run_to_heldout_record(
        run_payload=run_payload, job=job
    )
    assert record["configuration_hash_full"] == "a" * 64
    assert record["configuration_hash_prefix"] == "a" * 12
    assert record["calibration_run_hash_prefix"] == "c" * 12
    assert record["execution_configuration_hash_full"] == "b" * 64
    assert record["job_kind"] == MAIN_JOB_KIND
    assert record["scm_seed"] == 301
    assert record["fit_rng"] == 42
    assert record["sid"] == 3
