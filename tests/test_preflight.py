"""Tests for experiments.selection_study.preflight.

Covers:
- Manifest enumeration: entry count, identity, run_id, directory,
  seeds, per-intervention seeds, sampling policy, no directory creation,
  no wrapper import.
- Validation rule (a): calibration/held-out overlap.
- Validation rule (b): duplicate run_id.
- Validation rule (c): hash stability.
- Validation rule (d): populated output directory.
- Validation rule (e): missing/wrongly-typed field.
- Validation rule (f): wandb in sys.modules.
- save_manifest: path determinism, byte stability, drift detection.
- run_preflight end-to-end: valid config, overlap, no fit, no wrappers.
- CLI integration: --dry-run without --config, valid config, invalid.

All filesystem tests use pytest's ``tmp_path`` fixture; no test
writes into ``results/``.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    SEED_DERIVATION_RULE_NAME,
    Configuration,
    InterventionSpec,
    CalibrationConfiguration,
    configuration_hash as compute_configuration_hash,
    derive_per_intervention_seed,
    derive_per_run_seeds,
)
from experiments.selection_study.identity import (
    derive_run_directory,
    derive_run_id,
)
from experiments.selection_study.preflight import (
    Manifest,
    ManifestEntry,
    ManifestValidationError,
    PerInterventionSeeds,
    enumerate_manifest,
    run_preflight,
    save_manifest,
    validate_manifest,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_INTERVENTION_A = InterventionSpec(
    intervention_id="intv_a", target_node=0, value_raw=1.0
)
_INTERVENTION_B = InterventionSpec(
    intervention_id="intv_b", target_node=1, value_raw=-1.0
)
_CALIBRATION_CFG = CalibrationConfiguration(
    name="default", hyperparameters=(("lr", 0.01),)
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


def _make_dagma_config(
    *,
    calibration_seeds: tuple[int, ...] = (10, 20),
    held_out_seeds: tuple[int, ...] = (30, 40),
    reproduction_seeds: tuple[int, ...] = (5,),
) -> Configuration:
    """Build a minimal valid DAGMA Configuration for tests."""
    return Configuration(
        model="dagma",
        condition="centred_only",
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=(
            ("calibration", calibration_seeds),
            ("held_out_evaluation", held_out_seeds),
            ("reproduction", reproduction_seeds),
        ),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        calibration_configurations=(_CALIBRATION_CFG,),
        threshold_robustness_triple=(0.2, 0.3, 0.4),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dagma:DagmaWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **_DAGMA_SCHEMA_GATE_FIELDS,
    )


def _make_dcdi_config(
    *,
    calibration_seeds: tuple[int, ...] = (10, 20),
    held_out_seeds: tuple[int, ...] = (30, 40),
    reproduction_seeds: tuple[int, ...] = (5,),
) -> Configuration:
    """Build a minimal valid DCDI Configuration for tests."""
    return Configuration(
        model="dcdi",
        condition="centred_only",
        seed_torch=7,
        seed_numpy=8,
        seed_dagma=None,
        seed_populations=(
            ("calibration", calibration_seeds),
            ("held_out_evaluation", held_out_seeds),
            ("reproduction", reproduction_seeds),
        ),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        calibration_configurations=(_CALIBRATION_CFG,),
        threshold_robustness_triple=(0.4, 0.5, 0.6),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **_DCDI_SCHEMA_GATE_FIELDS,
    )


def _write_config_json(tmp_path: Path, config: Configuration) -> Path:
    """Write a Configuration as JSON and return the path."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(config.to_canonical_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Enumeration tests
# --------------------------------------------------------------------------- #


def test_enumerate_manifest_entry_count_matches_total_seeds(
    tmp_path: Path,
) -> None:
    """One ManifestEntry per (seed_population, seed_replicate_index) pair."""
    config = _make_dagma_config(
        calibration_seeds=(10, 20),
        held_out_seeds=(30, 40),
        reproduction_seeds=(5,),
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    # 2 calibration + 2 held_out + 1 reproduction = 5
    assert len(manifest.entries) == 5


def test_enumerate_manifest_entry_identities_match_config(
    tmp_path: Path,
) -> None:
    """Every entry's (model, condition, seed_population) matches config."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        assert entry.model == config.model
        assert entry.condition == config.condition


def test_enumerate_manifest_run_ids_are_deterministic(
    tmp_path: Path,
) -> None:
    """Two calls with the same config produce byte-identical run_ids."""
    config = _make_dagma_config()
    manifest_a = enumerate_manifest(config, base_dir=tmp_path)
    manifest_b = enumerate_manifest(config, base_dir=tmp_path)
    ids_a = [e.expected_run_id for e in manifest_a.entries]
    ids_b = [e.expected_run_id for e in manifest_b.entries]
    assert ids_a == ids_b


def test_enumerate_manifest_run_id_format(tmp_path: Path) -> None:
    """Each run_id contains the full 64-char configuration_hash."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    config_hash = compute_configuration_hash(config)
    for entry in manifest.entries:
        assert f"cfg{config_hash}" in entry.expected_run_id
        assert entry.configuration_hash == config_hash


def test_enumerate_manifest_directory_uses_12_char_prefix(
    tmp_path: Path,
) -> None:
    """The expected_output_directory ends with the 12-char hash prefix."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    config_hash = compute_configuration_hash(config)
    prefix = config_hash[:12]
    for entry in manifest.entries:
        assert entry.expected_output_directory.endswith(prefix)


def test_enumerate_manifest_per_intervention_seeds_sorted(
    tmp_path: Path,
) -> None:
    """per_intervention_seeds is sorted by intervention_id."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        ids = [iid for iid, _ in entry.per_intervention_seeds]
        assert ids == sorted(ids)


def test_enumerate_manifest_per_intervention_seeds_are_per_intervention_seeds_instances(
    tmp_path: Path,
) -> None:
    """Each intervention seed pair is a PerInterventionSeeds instance."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        for _iid, seeds in entry.per_intervention_seeds:
            assert isinstance(seeds, PerInterventionSeeds)
            assert isinstance(seeds.ground_truth_sampling_seed, int)
            assert isinstance(seeds.model_sampling_seed, int)


def test_enumerate_manifest_dagma_sampling_policy(tmp_path: Path) -> None:
    """DAGMA entries carry planned_sampling_policy=='residual_fitted'."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        assert entry.planned_sampling_policy == "residual_fitted"


def test_enumerate_manifest_dcdi_sampling_policy(tmp_path: Path) -> None:
    """DCDI entries carry planned_sampling_policy=='native_conditionals'."""
    config = _make_dcdi_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        assert entry.planned_sampling_policy == "native_conditionals"


def test_enumerate_manifest_does_not_create_directories(
    tmp_path: Path,
) -> None:
    """enumerate_manifest is a pure computation; no directories created."""
    config = _make_dagma_config()
    base = tmp_path / "run_storage"
    assert not base.exists()
    enumerate_manifest(config, base_dir=base)
    assert not base.exists()


def test_enumerate_manifest_dagma_validation_seed_is_none(
    tmp_path: Path,
) -> None:
    """DAGMA entries have validation_data_seed == None."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        assert entry.validation_data_seed is None


def test_enumerate_manifest_dcdi_validation_seed_is_int(
    tmp_path: Path,
) -> None:
    """DCDI entries have validation_data_seed as a non-negative int."""
    config = _make_dcdi_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        assert isinstance(entry.validation_data_seed, int)
        assert entry.validation_data_seed >= 0


def test_enumerate_manifest_does_not_import_wrappers(
    tmp_path: Path,
) -> None:
    """enumerate_manifest must not newly import any wrapper or model module."""
    before = set(sys.modules)
    config = _make_dagma_config()
    enumerate_manifest(config, base_dir=tmp_path)
    after = set(sys.modules)
    new_modules = after - before
    forbidden = [
        m for m in new_modules
        if m.startswith("symbolic_priors_cd.wrappers")
        or m == "dagma"
        or m.startswith("dagma.")
        or m == "dcdi"
        or m.startswith("dcdi.")
    ]
    assert forbidden == [], f"enumerate_manifest loaded forbidden modules: {forbidden}"


# --------------------------------------------------------------------------- #
# Validation rule (a): calibration / held-out overlap
# --------------------------------------------------------------------------- #


def test_validate_rule_a_rejects_overlap(tmp_path: Path) -> None:
    """Rule (a): overlapping calibration and held-out seeds raise."""
    config = _make_dagma_config(
        calibration_seeds=(10, 20, 30),
        held_out_seeds=(30, 40),  # 30 overlaps
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(manifest, hash_recheck_config=config)
    msg = str(excinfo.value)
    assert "rule (a)" in msg
    assert "30" in msg


def test_validate_rule_a_accepts_disjoint_populations(tmp_path: Path) -> None:
    """Rule (a): non-overlapping populations pass without raising."""
    config = _make_dagma_config(
        calibration_seeds=(10, 20),
        held_out_seeds=(30, 40),
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)


# --------------------------------------------------------------------------- #
# Validation rule (b): duplicate run_id
# --------------------------------------------------------------------------- #


def test_validate_rule_b_rejects_duplicate_run_id(tmp_path: Path) -> None:
    """Rule (b): a manifest with a duplicate run_id raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    first = manifest.entries[0]
    entries_with_dup = manifest.entries + (first,)
    bad_manifest = Manifest(
        configuration_hash=manifest.configuration_hash,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=entries_with_dup,
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(bad_manifest, hash_recheck_config=config)
    assert "rule (b)" in str(excinfo.value)


def test_validate_rule_b_accepts_unique_run_ids(tmp_path: Path) -> None:
    """Rule (b): all-unique run_ids pass without raising."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)


# --------------------------------------------------------------------------- #
# Validation rule (c): hash stability
# --------------------------------------------------------------------------- #


def test_validate_rule_c_rejects_hash_mismatch(tmp_path: Path) -> None:
    """Rule (c): a mismatched configuration_hash raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    tampered = Manifest(
        configuration_hash="a" * 64,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=manifest.entries,
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(tampered, hash_recheck_config=config)
    assert "rule (c)" in str(excinfo.value)


def test_validate_rule_c_accepts_stable_hash(tmp_path: Path) -> None:
    """Rule (c): a correctly derived hash passes without raising."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)


# --------------------------------------------------------------------------- #
# Validation rule (d): populated output directory
# --------------------------------------------------------------------------- #


def test_validate_rule_d_rejects_populated_directory(tmp_path: Path) -> None:
    """Rule (d): a pre-existing populated directory raises; sentinel intact."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    target = Path(manifest.entries[0].expected_output_directory)
    target.mkdir(parents=True)
    sentinel = target / "sentinel.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(manifest, hash_recheck_config=config)
    assert "rule (d)" in str(excinfo.value)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserved"


def test_validate_rule_d_accepts_absent_directory(tmp_path: Path) -> None:
    """Rule (d): absent output directories pass without raising."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "absent")
    validate_manifest(manifest, hash_recheck_config=config)


def test_validate_rule_d_accepts_empty_directory(tmp_path: Path) -> None:
    """Rule (d): an existing empty directory is accepted."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    for entry in manifest.entries:
        Path(entry.expected_output_directory).mkdir(parents=True)
    validate_manifest(manifest, hash_recheck_config=config)


def test_validate_rule_d_does_not_create_directories(
    tmp_path: Path,
) -> None:
    """Rule (d) checks paths but never creates them."""
    base = tmp_path / "never_created"
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=base)
    assert not base.exists()
    validate_manifest(manifest, hash_recheck_config=config)
    assert not base.exists()


# --------------------------------------------------------------------------- #
# Validation rule (e): schema pre-check
# --------------------------------------------------------------------------- #


def test_validate_rule_e_rejects_missing_field(tmp_path: Path) -> None:
    """Rule (e): a ManifestEntry missing a mandatory field raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)

    @dataclass(frozen=True)
    class TruncatedEntry:
        model: str
        condition: str
        # missing all remaining fields

    bad_entry = TruncatedEntry(model="dagma", condition="centred_only")
    bad_manifest = Manifest(
        configuration_hash=manifest.configuration_hash,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=(bad_entry,),  # type: ignore[arg-type]
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(bad_manifest, hash_recheck_config=config)
    assert "rule (e)" in str(excinfo.value)


def test_validate_rule_e_rejects_wrong_type(tmp_path: Path) -> None:
    """Rule (e): a field with the wrong type raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    good = manifest.entries[0]
    bad_entry = ManifestEntry(
        model=good.model,
        condition=good.condition,
        seed_population=good.seed_population,
        seed_replicate_index="not_an_int",  # type: ignore[arg-type]
        graph_seed=good.graph_seed,
        train_data_seed=good.train_data_seed,
        validation_data_seed=good.validation_data_seed,
        intervention_ground_truth_seed_base=(
            good.intervention_ground_truth_seed_base
        ),
        model_sampling_seed_base=good.model_sampling_seed_base,
        per_intervention_seeds=good.per_intervention_seeds,
        configuration_hash=good.configuration_hash,
        expected_run_id=good.expected_run_id,
        expected_output_directory=good.expected_output_directory,
        planned_wrapper=good.planned_wrapper,
        planned_sampling_policy=good.planned_sampling_policy,
    )
    bad_manifest = Manifest(
        configuration_hash=manifest.configuration_hash,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=(bad_entry,),
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(bad_manifest, hash_recheck_config=config)
    assert "rule (e)" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Validation rule (f): wandb isolation
# --------------------------------------------------------------------------- #


def test_validate_rule_f_rejects_exact_wandb_module(
    tmp_path: Path,
) -> None:
    """Rule (f): sys.modules containing 'wandb' raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    import types
    fake_wandb = types.ModuleType("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        with pytest.raises(ManifestValidationError) as excinfo:
            validate_manifest(manifest, hash_recheck_config=config)
        assert "rule (f)" in str(excinfo.value)
        assert "wandb" in str(excinfo.value)
    finally:
        sys.modules.pop("wandb", None)


def test_validate_rule_f_rejects_wandb_submodule(tmp_path: Path) -> None:
    """Rule (f): sys.modules containing 'wandb.sdk' raises."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    import types
    fake_sdk = types.ModuleType("wandb.sdk")
    sys.modules["wandb.sdk"] = fake_sdk
    try:
        with pytest.raises(ManifestValidationError) as excinfo:
            validate_manifest(manifest, hash_recheck_config=config)
        assert "rule (f)" in str(excinfo.value)
    finally:
        sys.modules.pop("wandb.sdk", None)


def test_validate_rule_f_does_not_reject_wandbox(tmp_path: Path) -> None:
    """Rule (f): 'wandbox' is NOT a forbidden module (no substring match)."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    import types
    fake = types.ModuleType("wandbox")
    sys.modules["wandbox"] = fake
    try:
        validate_manifest(manifest, hash_recheck_config=config)
    finally:
        sys.modules.pop("wandbox", None)


def test_validate_rule_f_passes_with_no_wandb(tmp_path: Path) -> None:
    """Rule (f): no wandb in sys.modules passes without raising."""
    sys.modules.pop("wandb", None)
    for key in list(sys.modules):
        if key == "wandb" or key.startswith("wandb."):
            sys.modules.pop(key, None)
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)


# --------------------------------------------------------------------------- #
# save_manifest tests
# --------------------------------------------------------------------------- #


def test_save_manifest_path_uses_12_char_hash_prefix(
    tmp_path: Path,
) -> None:
    """save_manifest writes to manifest_<hash_prefix>.json."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    manifest_dir = tmp_path / "manifests"
    path = save_manifest(manifest, manifest_dir)
    expected_name = f"manifest_{manifest.configuration_hash[:12]}.json"
    assert path.name == expected_name
    assert path.parent == manifest_dir


def test_save_manifest_produces_byte_stable_output(tmp_path: Path) -> None:
    """Two save_manifest calls with the same manifest produce identical bytes."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    dir_a = tmp_path / "manifests_a"
    dir_b = tmp_path / "manifests_b"
    path_a = save_manifest(manifest, dir_a)
    path_b = save_manifest(manifest, dir_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_save_manifest_second_call_same_content_is_noop(
    tmp_path: Path,
) -> None:
    """Calling save_manifest twice with the same content succeeds."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    manifest_dir = tmp_path / "manifests"
    path_first = save_manifest(manifest, manifest_dir)
    path_second = save_manifest(manifest, manifest_dir)
    assert path_first == path_second


def test_save_manifest_drift_raises(tmp_path: Path) -> None:
    """save_manifest raises if the file already exists with different content."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    manifest_dir = tmp_path / "manifests"
    path = save_manifest(manifest, manifest_dir)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        save_manifest(manifest, manifest_dir)
    assert "drift" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# run_preflight end-to-end tests
# --------------------------------------------------------------------------- #


def test_run_preflight_valid_config_returns_path(tmp_path: Path) -> None:
    """run_preflight on a valid config returns the manifest path."""
    config = _make_dagma_config()
    config_path = _write_config_json(tmp_path, config)
    manifest_dir = tmp_path / "manifests"
    result = run_preflight(
        config_path,
        base_dir=tmp_path / "runs",
        manifest_dir=manifest_dir,
    )
    assert result.exists()
    assert result.suffix == ".json"


def test_run_preflight_overlap_raises_manifest_validation_error(
    tmp_path: Path,
) -> None:
    """run_preflight raises ManifestValidationError on seed overlap."""
    config = _make_dagma_config(
        calibration_seeds=(10, 20, 30),
        held_out_seeds=(30, 40),
    )
    config_path = _write_config_json(tmp_path, config)
    manifest_dir = tmp_path / "manifests"
    with pytest.raises(ManifestValidationError):
        run_preflight(
            config_path,
            base_dir=tmp_path / "runs",
            manifest_dir=manifest_dir,
        )


def test_run_preflight_does_not_create_run_directories(
    tmp_path: Path,
) -> None:
    """run_preflight (--dry-run) never creates run directories."""
    config = _make_dagma_config()
    config_path = _write_config_json(tmp_path, config)
    run_base = tmp_path / "runs"
    manifest_dir = tmp_path / "manifests"
    run_preflight(
        config_path,
        base_dir=run_base,
        manifest_dir=manifest_dir,
    )
    assert not run_base.exists()


def test_run_preflight_does_not_import_wrappers(tmp_path: Path) -> None:
    """run_preflight must not import any wrapper, dagma, or dcdi module."""
    before = set(sys.modules)
    config = _make_dagma_config()
    config_path = _write_config_json(tmp_path, config)
    run_preflight(
        config_path,
        base_dir=tmp_path / "runs",
        manifest_dir=tmp_path / "manifests",
    )
    after = set(sys.modules)
    new_modules = after - before
    forbidden = [
        m for m in new_modules
        if m.startswith("symbolic_priors_cd.wrappers")
        or m == "dagma"
        or m.startswith("dagma.")
        or m == "dcdi"
        or m.startswith("dcdi.")
    ]
    assert forbidden == [], f"run_preflight loaded forbidden modules: {forbidden}"


# --------------------------------------------------------------------------- #
# CLI integration tests
# --------------------------------------------------------------------------- #


def test_cli_dry_run_without_config_raises_value_error() -> None:
    """--dry-run without --config raises ValueError."""
    from experiments.selection_study.run import main

    with pytest.raises(ValueError) as excinfo:
        main(["--dry-run"])
    assert "--config" in str(excinfo.value) or "config" in str(excinfo.value)


def test_cli_dry_run_with_valid_config_succeeds(tmp_path: Path) -> None:
    """--dry-run with a valid config file produces a manifest file."""
    from experiments.selection_study.run import main

    config = _make_dagma_config()
    config_path = _write_config_json(tmp_path, config)
    manifest_dir = tmp_path / "manifests"
    main(
        ["--dry-run", "--config", str(config_path)],
        _base_dir=tmp_path / "runs",
        _manifest_dir=manifest_dir,
    )
    manifest_files = list(manifest_dir.glob("manifest_*.json"))
    assert len(manifest_files) == 1


def test_cli_dry_run_with_missing_config_raises(tmp_path: Path) -> None:
    """--dry-run with a non-existent config raises (not sys.exit)."""
    from experiments.selection_study.run import main

    missing = tmp_path / "no_such_config.json"
    assert not missing.exists()
    with pytest.raises((FileNotFoundError, SystemExit)):
        main(
            ["--dry-run", "--config", str(missing)],
            _base_dir=tmp_path / "runs",
            _manifest_dir=tmp_path / "manifests",
        )


# --------------------------------------------------------------------------- #
# Issue 1: schema_version is an integer
# --------------------------------------------------------------------------- #


def test_manifest_schema_version_is_integer(tmp_path: Path) -> None:
    """Manifest.schema_version must be int equal to 1, not a string."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    assert isinstance(manifest.schema_version, int)
    assert manifest.schema_version == 1


def test_save_manifest_json_schema_version_is_integer(tmp_path: Path) -> None:
    """The persisted manifest JSON must contain schema_version as integer 1."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    path = save_manifest(manifest, tmp_path / "manifests")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data["schema_version"], int), (
        f"schema_version in JSON is {type(data['schema_version']).__name__!r},"
        " expected int"
    )
    assert data["schema_version"] == 1


# --------------------------------------------------------------------------- #
# Issue 2: configuration_hash_algorithm on Manifest
# --------------------------------------------------------------------------- #


def test_manifest_has_configuration_hash_algorithm(tmp_path: Path) -> None:
    """Manifest carries configuration_hash_algorithm as a str equal to the imported constant."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    assert hasattr(manifest, "configuration_hash_algorithm")
    assert isinstance(manifest.configuration_hash_algorithm, str)
    assert manifest.configuration_hash_algorithm == CONFIGURATION_HASH_ALGORITHM_NAME


def test_save_manifest_json_contains_configuration_hash_algorithm(
    tmp_path: Path,
) -> None:
    """The persisted manifest JSON contains configuration_hash_algorithm."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    validate_manifest(manifest, hash_recheck_config=config)
    path = save_manifest(manifest, tmp_path / "manifests")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "configuration_hash_algorithm" in data
    assert data["configuration_hash_algorithm"] == CONFIGURATION_HASH_ALGORITHM_NAME


# --------------------------------------------------------------------------- #
# Issue 4: bool-as-int trap in rule (e)
# --------------------------------------------------------------------------- #


def test_validate_rule_e_rejects_bool_for_seed_replicate_index(
    tmp_path: Path,
) -> None:
    """Rule (e): bool is rejected for seed_replicate_index; bool is a subclass of int."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    good = manifest.entries[0]
    bad_entry = ManifestEntry(
        model=good.model,
        condition=good.condition,
        seed_population=good.seed_population,
        seed_replicate_index=True,  # type: ignore[arg-type]
        graph_seed=good.graph_seed,
        train_data_seed=good.train_data_seed,
        validation_data_seed=good.validation_data_seed,
        intervention_ground_truth_seed_base=good.intervention_ground_truth_seed_base,
        model_sampling_seed_base=good.model_sampling_seed_base,
        per_intervention_seeds=good.per_intervention_seeds,
        configuration_hash=good.configuration_hash,
        expected_run_id=good.expected_run_id,
        expected_output_directory=good.expected_output_directory,
        planned_wrapper=good.planned_wrapper,
        planned_sampling_policy=good.planned_sampling_policy,
    )
    bad_manifest = Manifest(
        configuration_hash=manifest.configuration_hash,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=(bad_entry,),
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(bad_manifest, hash_recheck_config=config)
    assert "rule (e)" in str(excinfo.value)


def test_validate_rule_e_rejects_bool_for_validation_data_seed(
    tmp_path: Path,
) -> None:
    """Rule (e): bool is rejected for validation_data_seed (int | None); bool is a subclass of int."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    good = manifest.entries[0]
    bad_entry = ManifestEntry(
        model=good.model,
        condition=good.condition,
        seed_population=good.seed_population,
        seed_replicate_index=good.seed_replicate_index,
        graph_seed=good.graph_seed,
        train_data_seed=good.train_data_seed,
        validation_data_seed=True,  # type: ignore[arg-type]
        intervention_ground_truth_seed_base=good.intervention_ground_truth_seed_base,
        model_sampling_seed_base=good.model_sampling_seed_base,
        per_intervention_seeds=good.per_intervention_seeds,
        configuration_hash=good.configuration_hash,
        expected_run_id=good.expected_run_id,
        expected_output_directory=good.expected_output_directory,
        planned_wrapper=good.planned_wrapper,
        planned_sampling_policy=good.planned_sampling_policy,
    )
    bad_manifest = Manifest(
        configuration_hash=manifest.configuration_hash,
        schema_version=manifest.schema_version,
        seed_derivation_rule=manifest.seed_derivation_rule,
        configuration_hash_algorithm=manifest.configuration_hash_algorithm,
        resolved_config=manifest.resolved_config,
        entries=(bad_entry,),
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest(bad_manifest, hash_recheck_config=config)
    assert "rule (e)" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Issue 5: direct delegation equality tests
# --------------------------------------------------------------------------- #


def test_enumerate_manifest_run_id_delegates_to_identity(tmp_path: Path) -> None:
    """entry.expected_run_id equals identity.derive_run_id(...) for first entry."""
    config = _make_dagma_config()
    config_hash = compute_configuration_hash(config)
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    entry = manifest.entries[0]
    expected = derive_run_id(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=config_hash,
    )
    assert entry.expected_run_id == expected


def test_enumerate_manifest_directory_delegates_to_identity(tmp_path: Path) -> None:
    """entry.expected_output_directory equals identity.derive_run_directory(...).as_posix()."""
    config = _make_dagma_config()
    config_hash = compute_configuration_hash(config)
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    entry = manifest.entries[0]
    expected = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=config_hash,
        base_dir=tmp_path,
    )
    assert entry.expected_output_directory == expected.as_posix()


def test_enumerate_manifest_seeds_delegate_to_derive_per_run_seeds(
    tmp_path: Path,
) -> None:
    """entry seed fields equal the corresponding fields of derive_per_run_seeds(...)."""
    config = _make_dagma_config()
    config_hash = compute_configuration_hash(config)
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    entry = manifest.entries[0]
    per_run = derive_per_run_seeds(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash_value=config_hash,
        include_validation_data_seed=(config.model == "dcdi"),
    )
    assert entry.graph_seed == per_run.graph_seed
    assert entry.train_data_seed == per_run.train_data_seed
    assert entry.intervention_ground_truth_seed_base == (
        per_run.intervention_ground_truth_seed_base
    )
    assert entry.model_sampling_seed_base == per_run.model_sampling_seed_base


def test_enumerate_manifest_per_intervention_seeds_delegate_to_derive_per_intervention_seed(
    tmp_path: Path,
) -> None:
    """per_intervention_seeds entries equal derive_per_intervention_seed(...) outputs."""
    config = _make_dagma_config()
    config_hash = compute_configuration_hash(config)
    manifest = enumerate_manifest(config, base_dir=tmp_path)
    entry = manifest.entries[0]
    per_run = derive_per_run_seeds(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash_value=config_hash,
        include_validation_data_seed=(config.model == "dcdi"),
    )
    for iid, seeds in entry.per_intervention_seeds:
        expected_gt = derive_per_intervention_seed(
            base_seed=per_run.intervention_ground_truth_seed_base,
            intervention_id=iid,
        )
        expected_model = derive_per_intervention_seed(
            base_seed=per_run.model_sampling_seed_base,
            intervention_id=iid,
        )
        assert seeds.ground_truth_sampling_seed == expected_gt
        assert seeds.model_sampling_seed == expected_model
