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


# Schema-gate-honest values. DAGMA fixtures use the docs/02 v1.6
# paper optimisation values that the pipeline now passes through;
# DCDI fixtures use the toy values the schema-gate pipeline
# consumes.
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
        **_DAGMA_SCHEMA_GATE_FIELDS,
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
        **_DCDI_SCHEMA_GATE_FIELDS,
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
        **_DAGMA_SCHEMA_GATE_FIELDS,
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


def test_configuration_rejects_unknown_seed_population_key() -> None:
    """Configuration validation rejects ``seed_populations`` keys that
    are not members of ``VALID_SEED_POPULATIONS``.

    A single unknown key produces a ``ValueError`` whose message names
    that key. Multiple unknown keys present at construction time are
    all reported in a single error message.
    """
    from experiments.selection_study.config import VALID_SEED_POPULATIONS

    assert "banana" not in VALID_SEED_POPULATIONS
    assert "apple" not in VALID_SEED_POPULATIONS

    with pytest.raises(ValueError) as excinfo_single:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(
                ("banana", (1, 2)),
                ("calibration", (3, 4)),
            ),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message_single = str(excinfo_single.value)
    assert "seed_populations" in message_single
    assert "banana" in message_single

    with pytest.raises(ValueError) as excinfo_multi:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(
                ("banana", (1, 2)),
                ("calibration", (3, 4)),
                ("apple", (5,)),
            ),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message_multi = str(excinfo_multi.value)
    assert "seed_populations" in message_multi
    assert "banana" in message_multi, (
        f"first unknown key not named in error: {message_multi!r}"
    )
    assert "apple" in message_multi, (
        f"second unknown key not named in error: {message_multi!r}"
    )


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


def test_seed_populations_rejects_duplicate_population_labels() -> None:
    """Duplicate seed-population names are rejected by validation.

    The contract is that each valid population name in
    ``VALID_SEED_POPULATIONS`` may appear at most once in
    ``seed_populations``; a duplicate is treated as a configuration
    error rather than a silent merge.
    """
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(
                ("calibration", (1, 2)),
                ("calibration", (3, 4)),
            ),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message = str(excinfo.value)
    assert "seed_populations" in message
    assert "duplicate" in message.lower()
    assert "calibration" in message


def test_seed_populations_rejects_negative_seed_values() -> None:
    """Negative seed values are rejected by validation."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(("calibration", (1, -2, 3)),),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message = str(excinfo.value)
    assert "seed_populations" in message
    assert "calibration" in message
    assert "-2" in message
    assert ">=" in message


def test_seed_populations_rejects_bool_seed_values() -> None:
    """Bool seed values are rejected explicitly (bool subclasses int).

    Without the bool guard, Python would silently accept ``True`` and
    ``False`` as seeds 1 and 0 because ``bool`` is a subclass of
    ``int``. The guard distinguishes the two types at validation
    time.
    """
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(("calibration", (1, True, 3)),),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message = str(excinfo.value)
    assert "seed_populations" in message
    assert "calibration" in message
    assert "bool" in message.lower()


def test_seed_populations_rejects_non_int_seed_values() -> None:
    """Non-int seed values are rejected by validation.

    A float in the seed tuple is the canonical regression case: a
    silent ``int(seed)`` cast would have truncated ``1.5`` to ``1``;
    the explicit ``isinstance`` check rejects it instead.
    """
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            model="dagma",
            condition="centred_only",
            seed_torch=None,
            seed_numpy=None,
            seed_dagma=None,
            seed_populations=(("calibration", (1, 1.5, 3)),),
            intervention_set=(),
            phase_b_configurations=(),
            threshold_robustness_triple=(0.2, 0.3, 0.4),
            wrapper_api_reference=(
                "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
            ),
        )
    message = str(excinfo.value)
    assert "seed_populations" in message
    assert "calibration" in message
    assert "1.5" in message
    assert "int" in message


def test_load_config_rejects_seed_populations_with_bool_seed(
    tmp_path: Path,
) -> None:
    """A JSON config with a bool seed value is rejected by load_config.

    Proves the validation chain fires from disk through
    ``_configuration_from_dict`` into ``__post_init__`` without
    silent ``int(...)`` truncation of ``True`` to ``1``.
    """
    payload = {
        "model": "dagma",
        "condition": "centred_only",
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "seed_populations": {"calibration": [True]},
        "intervention_set": [],
        "phase_b_configurations": [],
        "threshold_robustness_triple": [0.2, 0.3, 0.4],
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
        "n_nodes": 3,
        "expected_edges": 3,
        "noise_scale": 1.0,
        "weight_magnitude_range": [0.5, 2.0],
        "n_train": 64,
        "mmd_n_samples": 64,
        "n_val_dcdi": None,
        "dcdi_num_train_iter": None,
        "dcdi_stop_crit_win": None,
        "dcdi_train_patience": None,
        "dcdi_train_batch_size": None,
        "dcdi_lr": None,
        "dcdi_h_threshold": None,
        "dcdi_hidden_units": None,
        "dcdi_hidden_layers": None,
        "dagma_warm_iter": 20000,
        "dagma_max_iter": 70000,
        "dagma_lr": 3e-4,
        "dagma_beta_1": 0.99,
        "dagma_beta_2": 0.999,
    }
    file_path = tmp_path / "bool_seed.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    message = str(excinfo.value)
    assert "seed_populations" in message
    assert "bool" in message.lower()


def test_valid_models_matches_configuration_model_field_type() -> None:
    """``VALID_MODELS`` matches the ``Literal`` on ``Configuration.model``.

    The contract locks the single-source-of-truth between the module
    constant and the dataclass field annotation. Any drift between the
    two (for example, adding a new model name to the ``Literal`` but
    forgetting to extend the constant) is caught here.
    """
    import typing

    hints = typing.get_type_hints(Configuration)
    args = typing.get_args(hints["model"])
    assert set(args) == set(config_module.VALID_MODELS), (
        f"VALID_MODELS={config_module.VALID_MODELS} does not match "
        f"Configuration.model Literal args={args}"
    )


def test_valid_conditions_matches_configuration_condition_field_type() -> None:
    """``VALID_CONDITIONS`` matches the ``Literal`` on ``condition``.

    Mirror of the model contract for the preprocessing-condition
    field.
    """
    import typing

    hints = typing.get_type_hints(Configuration)
    args = typing.get_args(hints["condition"])
    assert set(args) == set(config_module.VALID_CONDITIONS), (
        f"VALID_CONDITIONS={config_module.VALID_CONDITIONS} does not "
        f"match Configuration.condition Literal args={args}"
    )


# --------------------------------------------------------------------------- #
# SCM-generation fields on Configuration
# --------------------------------------------------------------------------- #


def _make_dagma_kwargs() -> dict[str, Any]:
    """Return a minimal-valid kwargs dict for a DAGMA Configuration."""
    return {
        "model": "dagma",
        "condition": "centred_only",
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "seed_populations": (("calibration", (1,)),),
        "intervention_set": (),
        "phase_b_configurations": (),
        "threshold_robustness_triple": (0.2, 0.3, 0.4),
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
        **_DAGMA_SCHEMA_GATE_FIELDS,
    }


def _make_dcdi_kwargs() -> dict[str, Any]:
    """Return a minimal-valid kwargs dict for a DCDI Configuration."""
    return {
        "model": "dcdi",
        "condition": "centred_only",
        "seed_torch": 7,
        "seed_numpy": 8,
        "seed_dagma": None,
        "seed_populations": (("calibration", (1,)),),
        "intervention_set": (),
        "phase_b_configurations": (),
        "threshold_robustness_triple": (0.4, 0.5, 0.6),
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
        **_DCDI_SCHEMA_GATE_FIELDS,
    }


def test_canonical_dict_includes_scm_generation_fields() -> None:
    """``to_canonical_dict`` carries the four SCM-generation fields."""
    config = _make_dagma_configuration()
    payload = config.to_canonical_dict()
    assert payload["n_nodes"] == config.n_nodes
    assert payload["expected_edges"] == config.expected_edges
    assert payload["noise_scale"] == float(config.noise_scale)
    assert payload["weight_magnitude_range"] == [
        float(config.weight_magnitude_range[0]),
        float(config.weight_magnitude_range[1]),
    ]


def test_load_config_preserves_scm_generation_fields(
    tmp_path: Path,
) -> None:
    """A round-trip through ``load_config`` preserves the four fields."""
    original = Configuration(
        **{
            **_make_dagma_kwargs(),
            "n_nodes": 4,
            "expected_edges": 3,
            "noise_scale": 0.5,
            "weight_magnitude_range": (0.7, 1.8),
        }
    )
    file_path = tmp_path / "with_scm.json"
    _dump_config_to_json(original, file_path)
    loaded = load_config(file_path)
    assert loaded.n_nodes == 4
    assert loaded.expected_edges == 3
    assert loaded.noise_scale == 0.5
    assert loaded.weight_magnitude_range == (0.7, 1.8)
    assert loaded == original


def test_configuration_hash_changes_when_n_nodes_changes() -> None:
    """Changing ``n_nodes`` produces a different ``configuration_hash``."""
    base = Configuration(**_make_dagma_kwargs())
    other = Configuration(
        **{
            **_make_dagma_kwargs(),
            "n_nodes": 10,
            "expected_edges": 20,
        }
    )
    assert configuration_hash(base) != configuration_hash(other)


def test_configuration_hash_changes_when_noise_scale_changes() -> None:
    """Changing ``noise_scale`` produces a different ``configuration_hash``."""
    base = Configuration(**_make_dagma_kwargs())
    other = Configuration(
        **{**_make_dagma_kwargs(), "noise_scale": 0.5}
    )
    assert configuration_hash(base) != configuration_hash(other)


def test_configuration_hash_changes_when_weight_range_changes() -> None:
    """Changing ``weight_magnitude_range`` changes ``configuration_hash``."""
    base = Configuration(**_make_dagma_kwargs())
    other = Configuration(
        **{
            **_make_dagma_kwargs(),
            "weight_magnitude_range": (0.7, 1.5),
        }
    )
    assert configuration_hash(base) != configuration_hash(other)


def test_configuration_rejects_n_nodes_bool() -> None:
    """``n_nodes`` must be a plain int, not ``bool``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "n_nodes": True})
    assert "n_nodes" in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


def test_configuration_rejects_n_nodes_below_two() -> None:
    """``n_nodes`` must be >= 2."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "n_nodes": 1})
    message = str(excinfo.value)
    assert "n_nodes" in message
    assert ">=" in message or "2" in message


def test_configuration_rejects_n_nodes_non_int() -> None:
    """``n_nodes`` must be an integer type."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "n_nodes": 3.0})
    assert "n_nodes" in str(excinfo.value)


def test_configuration_rejects_expected_edges_bool() -> None:
    """``expected_edges`` must be a plain int, not ``bool``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{**_make_dagma_kwargs(), "expected_edges": False}
        )
    assert "expected_edges" in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


def test_configuration_rejects_expected_edges_negative() -> None:
    """``expected_edges`` must be >= 0."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{**_make_dagma_kwargs(), "expected_edges": -1}
        )
    assert "expected_edges" in str(excinfo.value)


def test_configuration_rejects_expected_edges_above_dag_maximum() -> None:
    """``expected_edges`` must not exceed ``n_nodes*(n_nodes-1)//2``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "n_nodes": 3,
                "expected_edges": 10,
            }
        )
    message = str(excinfo.value)
    assert "expected_edges" in message
    assert "n_nodes" in message


def test_configuration_rejects_noise_scale_zero() -> None:
    """``noise_scale`` must be strictly positive."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "noise_scale": 0.0})
    assert "noise_scale" in str(excinfo.value)


def test_configuration_rejects_noise_scale_negative() -> None:
    """A negative ``noise_scale`` is rejected."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "noise_scale": -1.0})
    assert "noise_scale" in str(excinfo.value)


def test_configuration_rejects_noise_scale_bool() -> None:
    """``noise_scale`` must not be a ``bool``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), "noise_scale": True})
    assert "noise_scale" in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


def test_configuration_rejects_noise_scale_non_finite() -> None:
    """A non-finite ``noise_scale`` is rejected."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{**_make_dagma_kwargs(), "noise_scale": float("inf")}
        )
    assert "noise_scale" in str(excinfo.value)


def test_configuration_rejects_weight_range_wrong_length() -> None:
    """``weight_magnitude_range`` must contain exactly two values."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "weight_magnitude_range": (0.5,),
            }
        )
    assert "weight_magnitude_range" in str(excinfo.value)


def test_configuration_rejects_weight_range_non_positive() -> None:
    """``weight_magnitude_range`` low must be strictly positive."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "weight_magnitude_range": (0.0, 2.0),
            }
        )
    assert "weight_magnitude_range" in str(excinfo.value)


def test_configuration_rejects_weight_range_low_greater_than_high() -> None:
    """``weight_magnitude_range`` must satisfy ``low <= high``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "weight_magnitude_range": (2.0, 1.0),
            }
        )
    assert "weight_magnitude_range" in str(excinfo.value)


def test_configuration_rejects_weight_range_bool_member() -> None:
    """No ``bool`` value is accepted inside ``weight_magnitude_range``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "weight_magnitude_range": (True, 2.0),
            }
        )
    assert "weight_magnitude_range" in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


def test_weight_magnitude_range_normalised_to_tuple() -> None:
    """A list ``weight_magnitude_range`` is stored as a tuple of floats."""
    config = Configuration(
        **{
            **_make_dagma_kwargs(),
            "weight_magnitude_range": [0.5, 2.0],
        }
    )
    assert isinstance(config.weight_magnitude_range, tuple)
    assert config.weight_magnitude_range == (0.5, 2.0)
    assert all(
        isinstance(value, float)
        for value in config.weight_magnitude_range
    )


def test_weight_magnitude_range_isolated_from_input_list_mutation() -> None:
    """Mutating the input list after construction does not leak in."""
    original = [0.5, 2.0]
    config = Configuration(
        **{
            **_make_dagma_kwargs(),
            "weight_magnitude_range": original,
        }
    )
    original[0] = 99.0
    original[1] = -7.0
    assert config.weight_magnitude_range == (0.5, 2.0)


def test_noise_scale_int_input_stored_as_float() -> None:
    """An ``int`` ``noise_scale`` is stored as a ``float``."""
    config = Configuration(
        **{**_make_dagma_kwargs(), "noise_scale": 1}
    )
    assert isinstance(config.noise_scale, float)
    assert config.noise_scale == 1.0


def test_load_config_rejects_missing_n_nodes(tmp_path: Path) -> None:
    """A JSON config without the SCM fields is rejected by load_config."""
    payload = {
        "model": "dagma",
        "condition": "centred_only",
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "seed_populations": {"calibration": [1]},
        "intervention_set": [],
        "phase_b_configurations": [],
        "threshold_robustness_triple": [0.2, 0.3, 0.4],
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
    }
    file_path = tmp_path / "no_scm.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    message = str(excinfo.value)
    assert "missing required field" in message
    for field_name in (
        "n_nodes",
        "expected_edges",
        "noise_scale",
        "weight_magnitude_range",
    ):
        assert field_name in message, (
            f"missing-field error did not name {field_name!r}: "
            f"{message!r}"
        )


# --------------------------------------------------------------------------- #
# Real-run constants (n_train, mmd_n_samples, DCDI-only, DAGMA-only)
# --------------------------------------------------------------------------- #


_NEW_REAL_RUN_FIELDS = (
    "n_train",
    "mmd_n_samples",
    "n_val_dcdi",
    "dcdi_num_train_iter",
    "dcdi_stop_crit_win",
    "dcdi_train_patience",
    "dcdi_train_batch_size",
    "dcdi_lr",
    "dcdi_h_threshold",
    "dcdi_hidden_units",
    "dcdi_hidden_layers",
    "dagma_warm_iter",
    "dagma_max_iter",
    "dagma_lr",
    "dagma_beta_1",
    "dagma_beta_2",
)


def test_canonical_dict_includes_all_new_real_run_fields() -> None:
    """``to_canonical_dict`` carries every new real-run field."""
    dagma_payload = _make_dagma_configuration().to_canonical_dict()
    dcdi_payload = _make_dcdi_configuration().to_canonical_dict()
    for name in _NEW_REAL_RUN_FIELDS:
        assert name in dagma_payload, f"DAGMA payload missing {name!r}"
        assert name in dcdi_payload, f"DCDI payload missing {name!r}"


def test_lambda1_and_reg_coeff_are_not_top_level_configuration_fields() -> None:
    """Phase B sparsity knobs do not appear at the top of Configuration."""
    config = _make_dagma_configuration()
    payload = config.to_canonical_dict()
    assert "lambda1" not in payload
    assert "reg_coeff" not in payload
    # The Configuration dataclass also does not expose these as
    # attributes; Phase B sparsity lives inside PhaseBConfiguration.
    assert not hasattr(config, "lambda1")
    assert not hasattr(config, "reg_coeff")


@pytest.mark.parametrize("field_name", ["n_train", "mmd_n_samples"])
def test_shared_field_change_changes_hash(field_name: str) -> None:
    """Each shared real-run field participates in configuration_hash."""
    base = _make_dagma_configuration()
    base_hash = configuration_hash(base)
    bumped_value = int(getattr(base, field_name)) + 1
    other = Configuration(
        **{
            **_make_dagma_kwargs(),
            field_name: bumped_value,
        }
    )
    assert configuration_hash(other) != base_hash


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("n_val_dcdi", 33),
        ("dcdi_num_train_iter", 31),
        ("dcdi_stop_crit_win", 11),
        ("dcdi_train_patience", 6),
        ("dcdi_train_batch_size", 9),
        ("dcdi_lr", 2e-3),
        ("dcdi_h_threshold", 2e-8),
        ("dcdi_hidden_units", 17),
        ("dcdi_hidden_layers", 3),
    ],
)
def test_dcdi_only_field_change_changes_hash(
    field_name: str, new_value: Any
) -> None:
    """Each DCDI-only field participates in configuration_hash."""
    base_kwargs = _make_dcdi_kwargs()
    base_hash = configuration_hash(Configuration(**base_kwargs))
    other = Configuration(**{**base_kwargs, field_name: new_value})
    assert configuration_hash(other) != base_hash


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("dagma_warm_iter", 30001),
        ("dagma_max_iter", 60001),
        ("dagma_lr", 4e-4),
        ("dagma_beta_1", 0.991),
        ("dagma_beta_2", 0.9991),
    ],
)
def test_dagma_only_field_change_changes_hash(
    field_name: str, new_value: Any
) -> None:
    """Each DAGMA-only field participates in configuration_hash."""
    base_kwargs = _make_dagma_kwargs()
    base_hash = configuration_hash(Configuration(**base_kwargs))
    other = Configuration(**{**base_kwargs, field_name: new_value})
    assert configuration_hash(other) != base_hash


def test_load_config_rejects_missing_new_real_run_fields(
    tmp_path: Path,
) -> None:
    """A JSON config without the new real-run fields is rejected."""
    payload = {
        "model": "dagma",
        "condition": "centred_only",
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "seed_populations": {"calibration": [1]},
        "intervention_set": [],
        "phase_b_configurations": [],
        "threshold_robustness_triple": [0.2, 0.3, 0.4],
        "wrapper_api_reference": (
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
        "n_nodes": 3,
        "expected_edges": 3,
        "noise_scale": 1.0,
        "weight_magnitude_range": [0.5, 2.0],
    }
    file_path = tmp_path / "no_real_run_fields.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(file_path)
    message = str(excinfo.value)
    assert "missing required field" in message
    for field_name in _NEW_REAL_RUN_FIELDS:
        assert field_name in message, (
            f"missing-field error did not name {field_name!r}: "
            f"{message!r}"
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "n_train",
        "mmd_n_samples",
    ],
)
def test_shared_field_rejects_bool(field_name: str) -> None:
    """Bool values are rejected for each shared real-run int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), field_name: True})
    assert field_name in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "field_name",
    [
        "n_train",
        "mmd_n_samples",
    ],
)
def test_shared_field_rejects_zero(field_name: str) -> None:
    """Zero is rejected for each shared real-run int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), field_name: 0})
    assert field_name in str(excinfo.value)


@pytest.mark.parametrize(
    "field_name",
    [
        "n_train",
        "mmd_n_samples",
    ],
)
def test_shared_field_rejects_negative(field_name: str) -> None:
    """Negative integers are rejected for each shared real-run int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), field_name: -1})
    assert field_name in str(excinfo.value)


@pytest.mark.parametrize(
    "field_name",
    [
        "n_val_dcdi",
        "dcdi_num_train_iter",
        "dcdi_stop_crit_win",
        "dcdi_train_patience",
        "dcdi_train_batch_size",
        "dcdi_hidden_units",
        "dcdi_hidden_layers",
    ],
)
def test_dcdi_only_int_field_rejects_bool(field_name: str) -> None:
    """Bool values are rejected for each DCDI-only int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dcdi_kwargs(), field_name: True})
    assert field_name in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "field_name",
    [
        "n_val_dcdi",
        "dcdi_num_train_iter",
        "dcdi_stop_crit_win",
        "dcdi_train_patience",
        "dcdi_train_batch_size",
        "dcdi_hidden_units",
        "dcdi_hidden_layers",
    ],
)
def test_dcdi_only_int_field_rejects_zero(field_name: str) -> None:
    """Zero is rejected for each DCDI-only int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dcdi_kwargs(), field_name: 0})
    assert field_name in str(excinfo.value)


@pytest.mark.parametrize(
    "field_name",
    [
        "dcdi_lr",
        "dcdi_h_threshold",
    ],
)
def test_dcdi_only_float_field_rejects_bool(field_name: str) -> None:
    """Bool values are rejected for each DCDI-only float field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dcdi_kwargs(), field_name: True})
    assert field_name in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "field_name",
    [
        "dcdi_lr",
        "dcdi_h_threshold",
    ],
)
def test_dcdi_only_float_field_rejects_zero(field_name: str) -> None:
    """Zero is rejected for each DCDI-only float field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dcdi_kwargs(), field_name: 0.0})
    assert field_name in str(excinfo.value)


@pytest.mark.parametrize(
    "field_name",
    [
        "dagma_warm_iter",
        "dagma_max_iter",
    ],
)
def test_dagma_only_int_field_rejects_bool(field_name: str) -> None:
    """Bool values are rejected for each DAGMA-only int field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), field_name: True})
    assert field_name in str(excinfo.value)
    assert "bool" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "field_name",
    [
        "dagma_lr",
        "dagma_beta_1",
        "dagma_beta_2",
    ],
)
def test_dagma_only_float_field_rejects_zero(field_name: str) -> None:
    """Zero is rejected for each DAGMA-only float field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(**{**_make_dagma_kwargs(), field_name: 0.0})
    assert field_name in str(excinfo.value)


def test_n_val_dcdi_must_be_smaller_than_n_train() -> None:
    """``n_val_dcdi`` must be strictly smaller than ``n_train``."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dcdi_kwargs(),
                "n_train": 100,
                "n_val_dcdi": 100,
            }
        )
    message = str(excinfo.value)
    assert "n_val_dcdi" in message
    assert "n_train" in message


def test_dcdi_requires_dcdi_only_fields_non_none() -> None:
    """A DCDI Configuration must set every DCDI-only field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dcdi_kwargs(),
                "dcdi_num_train_iter": None,
            }
        )
    assert "dcdi_num_train_iter" in str(excinfo.value)
    assert "dcdi" in str(excinfo.value).lower()


def test_dcdi_rejects_non_none_dagma_only_fields() -> None:
    """A DCDI Configuration must leave every DAGMA-only field None."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dcdi_kwargs(),
                "dagma_warm_iter": 30000,
            }
        )
    assert "dagma_warm_iter" in str(excinfo.value)
    assert "dcdi" in str(excinfo.value).lower()


def test_dagma_requires_dagma_only_fields_non_none() -> None:
    """A DAGMA Configuration must set every DAGMA-only field."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "dagma_warm_iter": None,
            }
        )
    assert "dagma_warm_iter" in str(excinfo.value)
    assert "dagma" in str(excinfo.value).lower()


def test_dagma_rejects_non_none_dcdi_only_fields() -> None:
    """A DAGMA Configuration must leave every DCDI-only field None."""
    with pytest.raises(ValueError) as excinfo:
        Configuration(
            **{
                **_make_dagma_kwargs(),
                "dcdi_num_train_iter": 30,
            }
        )
    assert "dcdi_num_train_iter" in str(excinfo.value)
    assert "dagma" in str(excinfo.value).lower()


def test_load_config_round_trip_preserves_new_real_run_fields(
    tmp_path: Path,
) -> None:
    """A round trip through ``load_config`` preserves every new field."""
    original_dcdi = _make_dcdi_configuration()
    dcdi_path = tmp_path / "dcdi.json"
    _dump_config_to_json(original_dcdi, dcdi_path)
    loaded_dcdi = load_config(dcdi_path)
    for name in _NEW_REAL_RUN_FIELDS:
        assert getattr(loaded_dcdi, name) == getattr(
            original_dcdi, name
        ), f"DCDI {name!r} did not round-trip"

    original_dagma = _make_dagma_configuration()
    dagma_path = tmp_path / "dagma.json"
    _dump_config_to_json(original_dagma, dagma_path)
    loaded_dagma = load_config(dagma_path)
    for name in _NEW_REAL_RUN_FIELDS:
        assert getattr(loaded_dagma, name) == getattr(
            original_dagma, name
        ), f"DAGMA {name!r} did not round-trip"
