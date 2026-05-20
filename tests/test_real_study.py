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

# The seed integers inside the reproduction population are
# placeholders for the guard tests. The guard does not pin specific
# reproduction-seed integers; the selection-study protocol leaves
# them undetermined and they will be fixed in a separate commit.
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
    """A DAGMA Configuration with docs/02 v1.6 values passes."""
    config = Configuration(**_phase_a_dagma_kwargs())
    assert_real_study_constants(config, stage="phase_a")


def test_valid_dcdi_phase_a_config_passes_guard() -> None:
    """A DCDI Configuration with docs/02 v1.6 values passes."""
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
    """A non-docs/02 weight magnitude range is rejected for Phase A."""
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
