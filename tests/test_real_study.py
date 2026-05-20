"""Tests for the real-study protocol guard.

Verifies that ``assert_real_study_constants`` accepts valid Phase A
configurations and rejects toy/schema-gate values, wrong threshold
triples, and cross-model field leakage. The guard is policy-only;
it is never invoked from ``Configuration.__post_init__``, so toy
fixtures remain constructible even though they do not pass the
Phase A guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    Configuration,
    InterventionSpec,
    PhaseBConfiguration,
    SEED_DERIVATION_RULE_NAME,
)
from experiments.selection_study.real_study import (
    assert_real_study_constants,
)


# ---------------------------------------------------------------------------
# Shared Phase A construction helpers
# ---------------------------------------------------------------------------


_INTERVENTION_A = InterventionSpec(
    intervention_id="do_X0_neg2",
    target_node=0,
    value_raw=-2.0,
)
_INTERVENTION_B = InterventionSpec(
    intervention_id="do_X0_pos2",
    target_node=0,
    value_raw=2.0,
)

# The guard-construction tests use a placeholder reproduction
# seed because ``assert_real_study_constants`` only requires a
# non-empty reproduction population. The on-disk Phase A
# config-file tests below pin the frozen reproduction seeds
# (101, 102, 103).
_REPRODUCTION_SEEDS: tuple[int, ...] = (1,)


def _phase_a_dagma_kwargs() -> dict[str, Any]:
    return {
        "model": "dagma",
        "condition": "centred_only",
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "seed_populations": (
            ("reproduction", _REPRODUCTION_SEEDS),
        ),
        "intervention_set": (_INTERVENTION_A, _INTERVENTION_B),
        "phase_b_configurations": (
            PhaseBConfiguration(
                name="anchor",
                hyperparameters=(("lambda1", 0.05),),
            ),
        ),
        "threshold_robustness_triple": (0.2, 0.3, 0.4),
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
        "n_nodes": 10,
        "expected_edges": 20,
        "noise_scale": 1.0,
        "weight_magnitude_range": (0.5, 2.0),
        "n_train": 1000,
        "mmd_n_samples": 1000,
        "dagma_warm_iter": 20000,
        "dagma_max_iter": 70000,
        "dagma_lr": 3e-4,
        "dagma_beta_1": 0.99,
        "dagma_beta_2": 0.999,
        "seed_derivation_rule": SEED_DERIVATION_RULE_NAME,
        "configuration_hash_algorithm": (
            CONFIGURATION_HASH_ALGORITHM_NAME
        ),
    }


def _phase_a_dcdi_kwargs() -> dict[str, Any]:
    return {
        "model": "dcdi",
        "condition": "centred_only",
        "seed_torch": 42,
        "seed_numpy": 42,
        "seed_dagma": None,
        "seed_populations": (
            ("reproduction", _REPRODUCTION_SEEDS),
        ),
        "intervention_set": (_INTERVENTION_A, _INTERVENTION_B),
        "phase_b_configurations": (
            PhaseBConfiguration(
                name="anchor",
                hyperparameters=(("reg_coeff", 0.1),),
            ),
        ),
        "threshold_robustness_triple": (0.4, 0.5, 0.6),
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
        "n_nodes": 10,
        "expected_edges": 20,
        "noise_scale": 1.0,
        "weight_magnitude_range": (0.5, 2.0),
        "n_train": 1000,
        "mmd_n_samples": 1000,
        "n_val_dcdi": 200,
        "dcdi_num_train_iter": 300000,
        "dcdi_stop_crit_win": 100,
        "dcdi_train_patience": 5,
        "dcdi_train_batch_size": 64,
        "dcdi_lr": 1e-3,
        "dcdi_h_threshold": 1e-8,
        "dcdi_hidden_units": 16,
        "dcdi_hidden_layers": 2,
        "seed_derivation_rule": SEED_DERIVATION_RULE_NAME,
        "configuration_hash_algorithm": (
            CONFIGURATION_HASH_ALGORITHM_NAME
        ),
    }


# ---------------------------------------------------------------------------
# Happy-path acceptance
# ---------------------------------------------------------------------------


def test_valid_dagma_phase_a_config_passes_guard() -> None:
    """A DAGMA Configuration carrying the Phase A real-study values passes."""
    config = Configuration(**_phase_a_dagma_kwargs())
    assert_real_study_constants(config, stage="phase_a")


def test_valid_dcdi_phase_a_config_passes_guard() -> None:
    """A DCDI Configuration carrying the Phase A real-study values passes."""
    config = Configuration(**_phase_a_dcdi_kwargs())
    assert_real_study_constants(config, stage="phase_a")


# ---------------------------------------------------------------------------
# Shared real-run constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,toy_value",
    [
        ("n_train", 64),
        ("mmd_n_samples", 64),
        ("n_nodes", 3),
        ("expected_edges", 3),
        ("noise_scale", 0.5),
    ],
)
def test_shared_field_toy_value_is_rejected_for_phase_a(
    field_name: str, toy_value: Any,
) -> None:
    """Each shared field's toy value is rejected as a Phase A constant."""
    if field_name == "n_nodes":
        kwargs = {**_phase_a_dagma_kwargs(), "n_nodes": 3, "expected_edges": 3}
    elif field_name == "expected_edges":
        kwargs = {**_phase_a_dagma_kwargs(), "expected_edges": 2}
    else:
        kwargs = {**_phase_a_dagma_kwargs(), field_name: toy_value}
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert field_name in str(excinfo.value)


def test_weight_magnitude_range_off_anchor_is_rejected() -> None:
    """A weight magnitude range off the Phase A anchor is rejected."""
    kwargs = {
        **_phase_a_dagma_kwargs(),
        "weight_magnitude_range": (0.5, 1.5),
    }
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert "weight_magnitude_range" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DAGMA-only checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,toy_value",
    [
        ("dagma_warm_iter", 30000),
        ("dagma_max_iter", 60000),
        ("dagma_lr", 1e-3),
        ("dagma_beta_1", 0.9),
        ("dagma_beta_2", 0.99),
    ],
)
def test_dagma_field_off_anchor_is_rejected(
    field_name: str, toy_value: Any,
) -> None:
    """Each DAGMA-only field's off-anchor value is rejected."""
    kwargs = {**_phase_a_dagma_kwargs(), field_name: toy_value}
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert field_name in str(excinfo.value)


def test_dagma_phase_a_rejects_wrong_threshold_triple() -> None:
    """DAGMA Phase A configs must carry the DAGMA threshold triple."""
    kwargs = {
        **_phase_a_dagma_kwargs(),
        "threshold_robustness_triple": (0.4, 0.5, 0.6),
    }
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert "threshold_robustness_triple" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DCDI-only checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,toy_value",
    [
        ("dcdi_num_train_iter", 30),
        ("dcdi_stop_crit_win", 10),
        ("dcdi_train_batch_size", 8),
        ("n_val_dcdi", 32),
        ("dcdi_train_patience", 1),
        ("dcdi_hidden_units", 8),
        ("dcdi_hidden_layers", 1),
        ("dcdi_lr", 5e-4),
        ("dcdi_h_threshold", 1e-6),
    ],
)
def test_dcdi_field_off_anchor_is_rejected(
    field_name: str, toy_value: Any,
) -> None:
    """Each DCDI-only field's off-anchor value is rejected."""
    kwargs = {**_phase_a_dcdi_kwargs(), field_name: toy_value}
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert field_name in str(excinfo.value)


def test_dcdi_phase_a_rejects_wrong_threshold_triple() -> None:
    """DCDI Phase A configs must carry the DCDI threshold triple."""
    kwargs = {
        **_phase_a_dcdi_kwargs(),
        "threshold_robustness_triple": (0.2, 0.3, 0.4),
    }
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert "threshold_robustness_triple" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Cross-model field leakage (None / non-None policy)
# ---------------------------------------------------------------------------


def test_dagma_phase_a_with_a_dcdi_only_field_is_rejected() -> None:
    """A DAGMA Phase A config cannot have a DCDI-only field set.

    Configuration validation already enforces this at construction
    time; the guard catches the synthetic case where a hand-built
    Configuration attempts the leak. Because Configuration
    construction would reject the cross-model field outright, the
    test verifies the construction-time error mentions both models,
    which is the same end-state the guard guarantees.
    """
    kwargs = {**_phase_a_dagma_kwargs(), "dcdi_num_train_iter": 300000}
    with pytest.raises(ValueError) as excinfo:
        Configuration(**kwargs)
    assert "dcdi_num_train_iter" in str(excinfo.value)


def test_dcdi_phase_a_with_a_dagma_only_field_is_rejected() -> None:
    """A DCDI Phase A config cannot have a DAGMA-only field set."""
    kwargs = {**_phase_a_dcdi_kwargs(), "dagma_warm_iter": 20000}
    with pytest.raises(ValueError) as excinfo:
        Configuration(**kwargs)
    assert "dagma_warm_iter" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Seed-population check
# ---------------------------------------------------------------------------


def test_phase_a_requires_reproduction_seed_population() -> None:
    """Phase A configs must carry the 'reproduction' seed population."""
    kwargs = {
        **_phase_a_dagma_kwargs(),
        "seed_populations": (("calibration", (1,)),),
    }
    config = Configuration(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_a")
    assert "reproduction" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stage handling
# ---------------------------------------------------------------------------


def test_unknown_stage_is_rejected() -> None:
    """An unknown stage label raises ValueError."""
    config = Configuration(**_phase_a_dagma_kwargs())
    with pytest.raises(ValueError) as excinfo:
        assert_real_study_constants(config, stage="phase_b")
    assert "stage" in str(excinfo.value).lower()
    assert "phase_b" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Toy / schema-gate fixtures remain constructible
# ---------------------------------------------------------------------------


def test_toy_configuration_remains_constructible_outside_guard() -> None:
    """Schema-gate Configurations construct without the Phase A guard.

    The guard must be policy-only; ``Configuration.__post_init__``
    must not silently invoke it. A schema-gate-sized Configuration
    is constructed here and then explicitly fails the Phase A
    guard, demonstrating the separation.
    """
    schema_gate_kwargs = {
        **_phase_a_dagma_kwargs(),
        "n_nodes": 3,
        "expected_edges": 3,
        "n_train": 64,
        "mmd_n_samples": 64,
    }
    config = Configuration(**schema_gate_kwargs)
    with pytest.raises(ValueError):
        assert_real_study_constants(config, stage="phase_a")



# ---------------------------------------------------------------------------
# Phase A reproduction config files on disk
# ---------------------------------------------------------------------------


import subprocess
import sys
from pathlib import Path

from experiments.selection_study.config import load_config
from experiments.selection_study.preflight import (
    enumerate_manifest,
    validate_manifest,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHASE_A_DIR = (
    _PROJECT_ROOT
    / "experiments"
    / "selection_study"
    / "configs"
    / "phase_a"
)
_DAGMA_PATH = _PHASE_A_DIR / "dagma_reproduction.json"
_DCDI_PATH = _PHASE_A_DIR / "dcdi_reproduction.json"


def test_phase_a_dagma_reproduction_config_file_exists() -> None:
    """The DAGMA Phase A reproduction config file is present on disk."""
    assert _DAGMA_PATH.is_file(), (
        f"missing Phase A DAGMA config file: {_DAGMA_PATH}"
    )


def test_phase_a_dcdi_reproduction_config_file_exists() -> None:
    """The DCDI Phase A reproduction config file is present on disk."""
    assert _DCDI_PATH.is_file(), (
        f"missing Phase A DCDI config file: {_DCDI_PATH}"
    )


def test_phase_a_dagma_config_loads_through_load_config() -> None:
    """load_config accepts the DAGMA Phase A reproduction file."""
    config = load_config(_DAGMA_PATH)
    assert config.model == "dagma"
    assert config.condition == "centred_only"


def test_phase_a_dcdi_config_loads_through_load_config() -> None:
    """load_config accepts the DCDI Phase A reproduction file."""
    config = load_config(_DCDI_PATH)
    assert config.model == "dcdi"
    assert config.condition == "centred_only"


def test_phase_a_dagma_config_passes_real_study_guard() -> None:
    """The DAGMA Phase A config satisfies the real-study guard."""
    config = load_config(_DAGMA_PATH)
    assert_real_study_constants(config, stage="phase_a")


def test_phase_a_dcdi_config_passes_real_study_guard() -> None:
    """The DCDI Phase A config satisfies the real-study guard."""
    config = load_config(_DCDI_PATH)
    assert_real_study_constants(config, stage="phase_a")


def test_phase_a_dagma_config_carries_reproduction_seeds() -> None:
    """The DAGMA Phase A config carries the frozen reproduction seeds."""
    config = load_config(_DAGMA_PATH)
    pops = dict(config.seed_populations)
    assert "reproduction" in pops
    assert pops["reproduction"] == (101, 102, 103)
    assert "calibration" not in pops
    assert "held_out_evaluation" not in pops


def test_phase_a_dcdi_config_carries_reproduction_seeds() -> None:
    """The DCDI Phase A config carries the frozen reproduction seeds."""
    config = load_config(_DCDI_PATH)
    pops = dict(config.seed_populations)
    assert "reproduction" in pops
    assert pops["reproduction"] == (101, 102, 103)
    assert "calibration" not in pops
    assert "held_out_evaluation" not in pops


def test_phase_a_dagma_config_enumerates_through_manifest(tmp_path) -> None:
    """enumerate_manifest produces three reproduction entries for DAGMA."""
    config = load_config(_DAGMA_PATH)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    assert len(manifest.entries) == 3
    for entry in manifest.entries:
        assert entry.seed_population == "reproduction"


def test_phase_a_dcdi_config_enumerates_through_manifest(tmp_path) -> None:
    """enumerate_manifest produces three reproduction entries for DCDI."""
    config = load_config(_DCDI_PATH)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    assert len(manifest.entries) == 3
    for entry in manifest.entries:
        assert entry.seed_population == "reproduction"


def test_phase_a_dagma_manifest_validates(tmp_path) -> None:
    """validate_manifest succeeds on the DAGMA Phase A manifest."""
    config = load_config(_DAGMA_PATH)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    validate_manifest(manifest, hash_recheck_config=config)


def test_phase_a_dcdi_manifest_validates(tmp_path) -> None:
    """validate_manifest succeeds on the DCDI Phase A manifest."""
    config = load_config(_DCDI_PATH)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    validate_manifest(manifest, hash_recheck_config=config)


def _check_preflight_does_not_load_forbidden_modules(
    config_path: Path,
) -> None:
    """Run preflight in a fresh subprocess and inspect sys.modules.

    Forbidden module prefixes are 'symbolic_priors_cd.wrappers',
    'dagma', 'dcdi', and 'wandb'. The probe loads the config,
    enumerates and validates the manifest, then reports any
    module under the forbidden prefixes that ended up loaded.
    """
    probe = (
        "import sys, tempfile\n"
        "from pathlib import Path\n"
        "from experiments.selection_study.config import load_config\n"
        "from experiments.selection_study.preflight import "
        "enumerate_manifest, validate_manifest\n"
        f"config = load_config(Path({str(config_path)!r}))\n"
        "with tempfile.TemporaryDirectory() as tmp_dir:\n"
        "    base = Path(tmp_dir) / 'runs'\n"
        "    manifest = enumerate_manifest(config, base_dir=base)\n"
        "    validate_manifest(manifest, hash_recheck_config=config)\n"
        "prefixes = ('symbolic_priors_cd.wrappers', 'dagma', 'dcdi', 'wandb')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == p or name.startswith(p + '.') for p in prefixes)\n"
        ")\n"
        "sys.stdout.write('FORBIDDEN=' + ';'.join(loaded) + chr(10))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed: stderr={result.stderr!r}"
    )
    markers = [
        line[len("FORBIDDEN="):]
        for line in result.stdout.splitlines()
        if line.startswith("FORBIDDEN=")
    ]
    assert len(markers) == 1, (
        f"probe did not emit FORBIDDEN line; stdout={result.stdout!r}"
    )
    loaded = markers[0].strip()
    assert loaded == "", (
        f"preflight loaded forbidden modules: {loaded!r}"
    )


def test_dagma_phase_a_preflight_does_not_import_wrappers_or_wandb() -> None:
    """Preflight on the DAGMA config loads no wrapper/DAGMA/wandb module."""
    _check_preflight_does_not_load_forbidden_modules(_DAGMA_PATH)


def test_dcdi_phase_a_preflight_does_not_import_wrappers_or_wandb() -> None:
    """Preflight on the DCDI config loads no wrapper/DCDI/wandb module."""
    _check_preflight_does_not_load_forbidden_modules(_DCDI_PATH)


def test_phase_a_dagma_manifest_does_not_create_run_directories(
    tmp_path,
) -> None:
    """enumerate_manifest + validate_manifest create no run dirs."""
    config = load_config(_DAGMA_PATH)
    base = tmp_path / "runs"
    manifest = enumerate_manifest(config, base_dir=base)
    validate_manifest(manifest, hash_recheck_config=config)
    if not base.exists():
        return
    leftover = [p for p in base.rglob("*") if p.is_dir() or p.is_file()]
    assert leftover == [], (
        f"preflight created filesystem artefacts: {leftover!r}"
    )


def test_phase_a_dcdi_manifest_does_not_create_run_directories(
    tmp_path,
) -> None:
    """enumerate_manifest + validate_manifest create no run dirs."""
    config = load_config(_DCDI_PATH)
    base = tmp_path / "runs"
    manifest = enumerate_manifest(config, base_dir=base)
    validate_manifest(manifest, hash_recheck_config=config)
    if not base.exists():
        return
    leftover = [p for p in base.rglob("*") if p.is_dir() or p.is_file()]
    assert leftover == [], (
        f"preflight created filesystem artefacts: {leftover!r}"
    )



# ---------------------------------------------------------------------------
# Phase A-only choices pinned by the on-disk config files
# ---------------------------------------------------------------------------


def test_phase_a_dagma_config_uses_centred_only_condition() -> None:
    """The DAGMA Phase A config uses the centred_only preprocessing condition."""
    config = load_config(_DAGMA_PATH)
    assert config.condition == "centred_only"


def test_phase_a_dcdi_config_uses_centred_only_condition() -> None:
    """The DCDI Phase A config uses the centred_only preprocessing condition."""
    config = load_config(_DCDI_PATH)
    assert config.condition == "centred_only"


def _assert_minimal_phase_a_intervention_set(config) -> None:
    """Verify the Phase A intervention set is the minimal +/-2 pair on node 0.

    The two interventions must target node 0 and carry values
    -2.0 and +2.0. The exact ``intervention_id`` strings are
    fixture conventions and are not pinned here.
    """
    interventions = config.intervention_set
    assert len(interventions) == 2, (
        f"Phase A intervention set must have exactly 2 entries; "
        f"got {len(interventions)}"
    )
    for spec in interventions:
        assert spec.target_node == 0, (
            f"Phase A interventions must target node 0; got "
            f"{spec.target_node} on intervention "
            f"{spec.intervention_id!r}"
        )
    values = sorted(float(spec.value_raw) for spec in interventions)
    assert values == [-2.0, 2.0], (
        f"Phase A intervention values must be exactly -2.0 and +2.0; "
        f"got {values}"
    )


def test_phase_a_dagma_config_uses_minimal_intervention_set() -> None:
    """DAGMA Phase A carries do(X0 = +/-2) and nothing else."""
    config = load_config(_DAGMA_PATH)
    _assert_minimal_phase_a_intervention_set(config)


def test_phase_a_dcdi_config_uses_minimal_intervention_set() -> None:
    """DCDI Phase A carries do(X0 = +/-2) and nothing else."""
    config = load_config(_DCDI_PATH)
    _assert_minimal_phase_a_intervention_set(config)


def test_phase_a_dagma_config_has_empty_phase_b_configurations() -> None:
    """DAGMA Phase A carries an empty phase_b_configurations tuple."""
    config = load_config(_DAGMA_PATH)
    assert config.phase_b_configurations == ()


def test_phase_a_dcdi_config_has_empty_phase_b_configurations() -> None:
    """DCDI Phase A carries an empty phase_b_configurations tuple."""
    config = load_config(_DCDI_PATH)
    assert config.phase_b_configurations == ()


def test_phase_a_dagma_config_seed_fields_are_all_none() -> None:
    """DAGMA Phase A has null seed_torch / seed_numpy / seed_dagma.

    DAGMA does not call torch.manual_seed, np.random.seed, or
    dagma.utils.set_random_seed, so null seed fields are the
    correct representation of the fit's deterministic-by-
    construction behaviour.
    """
    config = load_config(_DAGMA_PATH)
    assert config.seed_torch is None
    assert config.seed_numpy is None
    assert config.seed_dagma is None


def test_phase_a_dcdi_config_uses_fixed_fit_seed_42() -> None:
    """DCDI Phase A carries seed_torch = seed_numpy = 42, seed_dagma = None.

    DCDI requires matched non-null seed_torch / seed_numpy at fit
    time; the scalar 42 is an explicit Phase A config value that
    enters configuration_hash.
    """
    config = load_config(_DCDI_PATH)
    assert config.seed_torch == 42
    assert config.seed_numpy == 42
    assert config.seed_dagma is None
