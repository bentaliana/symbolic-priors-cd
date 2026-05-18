"""Tests for the selection-study runner configuration schema.

The tests in this file verify the Configuration dataclass, the
canonical JSON serialisation, the SHA-256 configuration_hash, the
configuration_hash_prefix, the per-purpose seed-derivation rule, the
per-intervention seed derivation, the disk loader, and the
Option A seed-discipline policy that allows DAGMA runs to record
``seed_torch``, ``seed_numpy``, and ``seed_dagma`` as JSON null.
"""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study import config as config_module
from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    Configuration,
    InterventionSpec,
    PerRunSeeds,
    PhaseBConfiguration,
    SEED_DERIVATION_RULE_NAME,
    canonical_json,
    configuration_hash,
    configuration_hash_prefix,
    derive_per_intervention_seed,
    derive_per_run_seeds,
    load_config,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_dagma_configuration() -> Configuration:
    """Return a representative DAGMA Configuration."""
    return Configuration(
        model="dagma",
        condition="centred_only",
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=(
            ("calibration", (1, 2)),
            ("held_out_evaluation", (10, 11, 12, 13, 14)),
            ("reproduction", (100,)),
        ),
        intervention_set=(
            InterventionSpec(
                intervention_id="do_X0_neg2",
                target_node=0,
                value_raw=-2.0,
            ),
            InterventionSpec(
                intervention_id="do_X0_pos2",
                target_node=0,
                value_raw=2.0,
            ),
        ),
        phase_b_configurations=(
            PhaseBConfiguration(
                name="cfg_1",
                hyperparameters=(("T", 4), ("lambda1", 0.05)),
            ),
            PhaseBConfiguration(
                name="cfg_2",
                hyperparameters=(("T", 5), ("lambda1", 0.10)),
            ),
        ),
        threshold_robustness_triple=(0.2, 0.3, 0.4),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
    )


def _make_dcdi_configuration() -> Configuration:
    """Return a representative DCDI Configuration."""
    return Configuration(
        model="dcdi",
        condition="centred_only",
        seed_torch=42,
        seed_numpy=43,
        seed_dagma=None,
        seed_populations=(
            ("calibration", (1, 2)),
            ("held_out_evaluation", (10, 11, 12, 13, 14)),
            ("reproduction", (100,)),
        ),
        intervention_set=(
            InterventionSpec(
                intervention_id="do_X0_pos2",
                target_node=0,
                value_raw=2.0,
            ),
        ),
        phase_b_configurations=(
            PhaseBConfiguration(
                name="cfg_1",
                hyperparameters=(("hid_dim", 8), ("num_layers", 2)),
            ),
        ),
        threshold_robustness_triple=(0.4, 0.5, 0.6),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
    )


def _dump_config_to_json(config: Configuration, path: Path) -> None:
    """Write a Configuration to ``path`` using its canonical JSON form."""
    path.write_text(canonical_json(config), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Canonical JSON
# --------------------------------------------------------------------------- #


def test_canonical_json_byte_stable() -> None:
    """Serialising the same Configuration twice yields identical bytes."""
    config = _make_dagma_configuration()
    first = canonical_json(config).encode("utf-8")
    second = canonical_json(config).encode("utf-8")
    assert first == second


def test_canonical_json_sorts_keys_at_every_level() -> None:
    """Canonical JSON sorts keys at every nesting level."""
    config = _make_dcdi_configuration()
    text = canonical_json(config)
    parsed = json.loads(text)

    def assert_dict_keys_sorted(value: Any) -> None:
        if isinstance(value, dict):
            keys = list(value.keys())
            assert keys == sorted(keys), (
                f"dict keys not sorted at level: {keys}"
            )
            for nested in value.values():
                assert_dict_keys_sorted(nested)
        elif isinstance(value, list):
            for item in value:
                assert_dict_keys_sorted(item)

    assert_dict_keys_sorted(parsed)


# --------------------------------------------------------------------------- #
# configuration_hash
# --------------------------------------------------------------------------- #


_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


def test_configuration_hash_sha256_deterministic() -> None:
    """Hashing the same Configuration twice yields the same digest."""
    config = _make_dagma_configuration()
    first = configuration_hash(config)
    second = configuration_hash(config)
    assert first == second
    assert len(first) == 64
    assert _HEX_PATTERN.fullmatch(first), (
        f"digest is not lowercase hex: {first!r}"
    )


def test_configuration_hash_changes_when_resolved_config_changes() -> None:
    """Configurations differing in any field have different hashes."""
    base = _make_dagma_configuration()
    other = _make_dcdi_configuration()
    assert configuration_hash(base) != configuration_hash(other)

    same_model_different_condition = Configuration(
        model="dagma",
        condition="standardised",
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=base.seed_populations,
        intervention_set=base.intervention_set,
        phase_b_configurations=base.phase_b_configurations,
        threshold_robustness_triple=base.threshold_robustness_triple,
        wrapper_api_reference=base.wrapper_api_reference,
    )
    assert configuration_hash(base) != configuration_hash(
        same_model_different_condition
    )


def test_configuration_hash_prefix_is_first_12_chars() -> None:
    """The prefix is the first 12 characters of the full digest."""
    config = _make_dagma_configuration()
    full = configuration_hash(config)
    prefix = configuration_hash_prefix(config)
    assert prefix == full[:12]
    assert len(prefix) == 12


# --------------------------------------------------------------------------- #
# Seed derivation
# --------------------------------------------------------------------------- #


def _common_identity_kwargs() -> dict[str, Any]:
    return {
        "model": "dagma",
        "condition": "centred_only",
        "seed_population": "calibration",
        "seed_replicate_index": 0,
        "configuration_hash_value": "0" * 64,
        "include_validation_data_seed": False,
    }


def test_per_purpose_seeds_deterministic() -> None:
    """The seed set for a fixed identity is reproducible."""
    first = derive_per_run_seeds(**_common_identity_kwargs())
    second = derive_per_run_seeds(**_common_identity_kwargs())
    assert first == second
    assert isinstance(first, PerRunSeeds)


def test_per_purpose_seeds_distinct_across_purpose_labels() -> None:
    """For a fixed identity, distinct purposes produce distinct seeds.

    Collisions are theoretically possible but vanishingly unlikely on
    the cryptographic SHA-256 hash. The fixture is concrete and the
    test asserts pairwise inequality on it.
    """
    kwargs = _common_identity_kwargs()
    kwargs["include_validation_data_seed"] = True
    seeds = derive_per_run_seeds(**kwargs)

    values = [
        seeds.graph_seed,
        seeds.train_data_seed,
        seeds.validation_data_seed,
        seeds.intervention_ground_truth_seed_base,
        seeds.model_sampling_seed_base,
    ]
    assert seeds.validation_data_seed is not None
    pairs = [
        (a, b)
        for i, a in enumerate(values)
        for b in values[i + 1:]
    ]
    for a, b in pairs:
        assert a != b, (
            "two purpose labels produced the same seed on this "
            f"fixture: {a} == {b}"
        )


def test_per_purpose_seeds_fit_in_signed_32_bit_range() -> None:
    """Every derived seed is in ``[0, 2**31)``."""
    kwargs = _common_identity_kwargs()
    kwargs["include_validation_data_seed"] = True
    seeds = derive_per_run_seeds(**kwargs)
    bound = 2 ** 31
    for value in (
        seeds.graph_seed,
        seeds.train_data_seed,
        seeds.validation_data_seed,
        seeds.intervention_ground_truth_seed_base,
        seeds.model_sampling_seed_base,
    ):
        assert value is not None
        assert 0 <= value < bound, (
            f"seed {value} is not in [0, {bound})"
        )


def test_validation_data_seed_is_none_when_excluded() -> None:
    """``validation_data_seed`` is None when not included."""
    seeds = derive_per_run_seeds(**_common_identity_kwargs())
    assert seeds.validation_data_seed is None


def test_per_intervention_seeds_derived_from_bases_and_intervention_id() -> None:
    """Per-intervention seeds vary with ``intervention_id``."""
    base_gt = 12345
    base_model = 67890

    seed_a_gt = derive_per_intervention_seed(
        base_seed=base_gt, intervention_id="A"
    )
    seed_b_gt = derive_per_intervention_seed(
        base_seed=base_gt, intervention_id="B"
    )
    seed_a_gt_again = derive_per_intervention_seed(
        base_seed=base_gt, intervention_id="A"
    )
    assert seed_a_gt != seed_b_gt
    assert seed_a_gt == seed_a_gt_again

    seed_a_model = derive_per_intervention_seed(
        base_seed=base_model, intervention_id="A"
    )
    seed_b_model = derive_per_intervention_seed(
        base_seed=base_model, intervention_id="B"
    )
    seed_a_model_again = derive_per_intervention_seed(
        base_seed=base_model, intervention_id="A"
    )
    assert seed_a_model != seed_b_model
    assert seed_a_model == seed_a_model_again
    assert seed_a_gt != seed_a_model


def test_per_intervention_seed_in_signed_32_bit_range() -> None:
    """Per-intervention seeds are in ``[0, 2**31)``."""
    seed = derive_per_intervention_seed(
        base_seed=12345, intervention_id="do_X0_neg2"
    )
    assert 0 <= seed < 2 ** 31


def test_seed_derivation_does_not_use_python_builtin_hash() -> None:
    """Source scan: no bare ``hash(`` call in the seed-derivation module."""
    source_path = Path(config_module.__file__)
    text = source_path.read_text(encoding="utf-8")
    matches = re.findall(r"\bhash\s*\(", text)
    assert matches == [], (
        "experiments/selection_study/config.py must not call Python's "
        f"built-in hash(); offending matches: {matches}"
    )


# --------------------------------------------------------------------------- #
# Option A regression tests (Section 16.1)
# --------------------------------------------------------------------------- #


def test_dagma_run_permits_null_torch_numpy_dagma_seeds() -> None:
    """DAGMA Configuration accepts None for all three global seeds.

    Canonical JSON serialises these fields as JSON null.
    """
    config = _make_dagma_configuration()
    assert config.seed_torch is None
    assert config.seed_numpy is None
    assert config.seed_dagma is None

    payload = json.loads(canonical_json(config))
    assert payload["seed_torch"] is None
    assert payload["seed_numpy"] is None
    assert payload["seed_dagma"] is None


def test_dcdi_run_records_non_null_seeds_when_applicable() -> None:
    """DCDI Configuration requires non-null seed_torch and seed_numpy."""
    config = _make_dcdi_configuration()
    assert config.seed_torch is not None
    assert config.seed_numpy is not None
    assert config.seed_dagma is None

    payload = json.loads(canonical_json(config))
    assert isinstance(payload["seed_torch"], int)
    assert isinstance(payload["seed_numpy"], int)
    assert payload["seed_dagma"] is None


def test_dagma_with_non_null_seed_torch_is_rejected() -> None:
    """Validation rejects a DAGMA Configuration with a non-null seed."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=42,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(("calibration", (1,)),),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference="symbolic_priors_cd.wrappers.dagma:DAGMAWrapper",
        )
    assert "dagma" in str(excinfo.value).lower()
    assert "seed_torch" in str(excinfo.value)


def test_dcdi_with_null_seed_torch_is_rejected() -> None:
    """Validation rejects a DCDI Configuration with seed_torch=None."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dcdi",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=43,
            seed_dagma=None,
            seed_populations=(("calibration", (1,)),),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.4, 0.5, 0.6),
            wrapper_api_reference="symbolic_priors_cd.wrappers.dcdi:DCDIWrapper",
        )
    assert "dcdi" in str(excinfo.value).lower()
    assert "seed_torch" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


def test_configuration_is_frozen() -> None:
    """Mutating a Configuration raises ``FrozenInstanceError``."""
    config = _make_dagma_configuration()
    with pytest.raises(FrozenInstanceError):
        config.model = "dcdi"  # type: ignore[misc]


def test_intervention_spec_is_frozen() -> None:
    """Mutating an InterventionSpec raises ``FrozenInstanceError``."""
    spec = InterventionSpec(
        intervention_id="A", target_node=0, value_raw=1.0
    )
    with pytest.raises(FrozenInstanceError):
        spec.target_node = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# load_config
# --------------------------------------------------------------------------- #


def test_load_config_reads_valid_json_and_returns_frozen_config(
    tmp_path: Path,
) -> None:
    """A valid JSON file loads into the expected frozen Configuration."""
    original = _make_dagma_configuration()
    file_path = tmp_path / "config.json"
    _dump_config_to_json(original, file_path)

    loaded = load_config(file_path)

    assert loaded == original
    with pytest.raises(FrozenInstanceError):
        loaded.model = "dcdi"  # type: ignore[misc]


def test_load_config_reads_dcdi_configuration(tmp_path: Path) -> None:
    """A valid DCDI JSON file loads with the expected seed policy."""
    original = _make_dcdi_configuration()
    file_path = tmp_path / "config_dcdi.json"
    _dump_config_to_json(original, file_path)

    loaded = load_config(file_path)

    assert loaded == original
    assert loaded.seed_torch is not None
    assert loaded.seed_numpy is not None
    assert loaded.seed_dagma is None


def test_load_config_rejects_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON raises ValueError with an informative message."""
    file_path = tmp_path / "broken.json"
    file_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    assert "not valid JSON" in str(excinfo.value)


def test_load_config_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    """Missing required fields raise ValueError naming the fields."""
    payload = {
        "model": "dagma",
        "condition": "centred_only",
        # Missing: seed_torch, seed_numpy, seed_dagma,
        # seed_populations, intervention_set,
        # phase_b_configurations, threshold_robustness_triple,
        # wrapper_api_reference.
    }
    file_path = tmp_path / "incomplete.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    message = str(excinfo.value)
    assert "missing required field" in message
    for missing_name in (
        "seed_torch",
        "seed_numpy",
        "seed_dagma",
        "seed_populations",
        "intervention_set",
        "phase_b_configurations",
        "threshold_robustness_triple",
        "wrapper_api_reference",
    ):
        assert missing_name in message, (
            f"missing-field error did not name {missing_name!r}: "
            f"{message!r}"
        )


def test_load_config_rejects_non_object_top_level(tmp_path: Path) -> None:
    """A JSON top-level array raises ValueError."""
    file_path = tmp_path / "array.json"
    file_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    assert "top-level value must be an object" in str(excinfo.value)


def test_load_config_raises_file_not_found(tmp_path: Path) -> None:
    """Missing file raises FileNotFoundError."""
    file_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_config(file_path)


# --------------------------------------------------------------------------- #
# Algorithm-name constants
# --------------------------------------------------------------------------- #


def test_algorithm_name_constants_match_module_values() -> None:
    """The Configuration carries the module's algorithm-name constants."""
    config = _make_dagma_configuration()
    assert (
        config.seed_derivation_rule == SEED_DERIVATION_RULE_NAME
    )
    assert (
        config.configuration_hash_algorithm
        == CONFIGURATION_HASH_ALGORITHM_NAME
    )
    assert (
        SEED_DERIVATION_RULE_NAME
        == "sha256_first8_bytes_mod_2pow31_purpose_label_v1"
    )
    assert (
        CONFIGURATION_HASH_ALGORITHM_NAME
        == "sha256_canonical_json_sorted_keys"
    )
