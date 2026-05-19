"""Configuration handling for the selection-study runner.

This module defines the frozen ``Configuration`` dataclass that
captures the experimental setup, the canonical JSON serialisation
used as the source-of-truth byte stream, the SHA-256-based
``configuration_hash`` derived from that byte stream, the 12-
character ``configuration_hash_prefix`` used as the directory-path
component, and the per-purpose seed-derivation rule that converts a
run identity tuple into deterministic seeds.

The seed-derivation rule encoded here uses ``hashlib.sha256`` over a
canonical byte encoding of the run identity plus a purpose label.
Python's built-in ``hash`` function is not used anywhere in this
module: ``hash`` is salted per process and would break cross-process
determinism.

Public surface
--------------
- ``Configuration``: frozen dataclass.
- ``InterventionSpec``: frozen dataclass for one intervention.
- ``PhaseBConfiguration``: frozen dataclass for one Phase B point.
- ``PerRunSeeds``: frozen dataclass returned by the seed-derivation
  function.
- ``canonical_json``: deterministic JSON serialisation.
- ``configuration_hash``: SHA-256 hex digest of the canonical JSON.
- ``configuration_hash_prefix``: first 12 characters of the digest.
- ``derive_per_run_seeds``: per-purpose seed derivation.
- ``derive_per_intervention_seed``: per-intervention seed derivation
  from a base seed and an ``intervention_id``.
- ``load_config``: JSON-file loader returning a ``Configuration``.
- ``CONFIGURATION_HASH_ALGORITHM_NAME``: stable algorithm name.
- ``SEED_DERIVATION_RULE_NAME``: stable derivation-rule name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


CONFIGURATION_HASH_ALGORITHM_NAME = "sha256_canonical_json_sorted_keys"
SEED_DERIVATION_RULE_NAME = (
    "sha256_first8_bytes_mod_2pow31_purpose_label_v1"
)

VALID_MODELS: tuple[str, ...] = ("dagma", "dcdi")
VALID_CONDITIONS: tuple[str, ...] = ("centred_only", "standardised")
VALID_SEED_POPULATIONS: tuple[str, ...] = (
    "reproduction",
    "calibration",
    "held_out_evaluation",
)

PURPOSE_GRAPH_SEED = "graph_seed"
PURPOSE_TRAIN_DATA_SEED = "train_data_seed"
PURPOSE_VALIDATION_DATA_SEED = "validation_data_seed"
PURPOSE_INTERVENTION_GROUND_TRUTH_SEED_BASE = (
    "intervention_ground_truth_seed_base"
)
PURPOSE_MODEL_SAMPLING_SEED_BASE = "model_sampling_seed_base"

_SEED_BOUND = 2 ** 31


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InterventionSpec:
    """Specification of a single intervention used in MMD evaluation.

    Each intervention carries a stable ``intervention_id`` that
    serves as input to the per-intervention seed derivation.

    Parameters
    ----------
    intervention_id : str
        Stable identifier for the intervention.
    target_node : int
        Index of the intervened node in the SCM.
    value_raw : float
        Intervention value in raw SCM units.
    """

    intervention_id: str
    target_node: int
    value_raw: float

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a primitive-typed dict for canonical serialisation."""
        return {
            "intervention_id": self.intervention_id,
            "target_node": int(self.target_node),
            "value_raw": float(self.value_raw),
        }


@dataclass(frozen=True)
class PhaseBConfiguration:
    """A single Phase B calibration configuration.

    Captures one point in the Phase B hyperparameter grid for the
    selection-study runner.

    Parameters
    ----------
    name : str
        Human-readable name of this configuration.
    hyperparameters : tuple of (str, primitive) pairs
        Hyperparameter overrides for this configuration, stored as an
        ordered tuple of ``(name, value)`` pairs. The values must be
        JSON-serialisable primitives.
    """

    name: str
    hyperparameters: tuple[tuple[str, Any], ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a primitive-typed dict for canonical serialisation."""
        return {
            "name": self.name,
            "hyperparameters": {
                str(key): value for key, value in self.hyperparameters
            },
        }


@dataclass(frozen=True)
class PerRunSeeds:
    """Per-purpose seeds derived for a single run identity tuple.

    Parameters
    ----------
    graph_seed : int
        Seed for SCM construction.
    train_data_seed : int
        Seed for observational training-data sampling.
    validation_data_seed : int or None
        Seed for validation-split sampling. ``None`` for candidates
        that do not use a validation split.
    intervention_ground_truth_seed_base : int
        Base seed for ground-truth interventional sampling.
    model_sampling_seed_base : int
        Base seed for model-generated interventional sampling.
    """

    graph_seed: int
    train_data_seed: int
    validation_data_seed: int | None
    intervention_ground_truth_seed_base: int
    model_sampling_seed_base: int


@dataclass(frozen=True)
class Configuration:
    """Frozen, serialisable selection-study configuration.

    A ``Configuration`` is the canonical record of an experimental
    setup. Two ``Configuration`` instances with identical resolved
    fields produce the same canonical JSON and the same
    ``configuration_hash``. The dataclass is frozen: attempting to
    mutate any field raises ``dataclasses.FrozenInstanceError``.

    Parameters
    ----------
    model : str
        Either ``"dagma"`` or ``"dcdi"``.
    condition : str
        Either ``"centred_only"`` or ``"standardised"``.
    seed_torch : int or None
        Seed passed to ``torch.manual_seed`` when the candidate's fit
        depends on PyTorch global RNG state. ``None`` when the
        candidate's fit is deterministic by construction and does not
        call the corresponding setter.
    seed_numpy : int or None
        Seed passed to ``numpy.random.seed`` when the candidate's fit
        depends on NumPy global RNG state. ``None`` when the
        candidate's fit is deterministic by construction.
    seed_dagma : int or None
        Seed passed to ``dagma.utils.set_random_seed`` when used.
        ``None`` when the candidate's fit does not call this setter.
    seed_populations : tuple of (str, tuple of int) pairs
        Ordered mapping from population name to its tuple of integer
        replicate seeds. Population names are drawn from
        ``VALID_SEED_POPULATIONS``.
    intervention_set : tuple of InterventionSpec
        Interventions evaluated for MMD on every run.
    phase_b_configurations : tuple of PhaseBConfiguration
        Calibration grid points evaluated during Phase B.
    threshold_robustness_triple : tuple of three float
        Threshold values used for offline threshold-robustness
        re-computation.
    wrapper_api_reference : str
        Stable, dotted ``module:attribute`` reference to the wrapper
        class or factory the runner invokes. The reference is stored
        as a string; the wrapper module is not imported by this
        module.
    seed_derivation_rule : str
        Stable identifier of the seed-derivation rule encoded by
        this module. Must equal ``SEED_DERIVATION_RULE_NAME``.
    configuration_hash_algorithm : str
        Stable identifier of the hashing algorithm encoded by this
        module. Must equal ``CONFIGURATION_HASH_ALGORITHM_NAME``.

    Raises
    ------
    ValueError
        On construction if any validation rule fails. The rules
        enforce: valid ``model`` and ``condition`` values; the
        seed-discipline policy mapping ``model`` to which of
        ``seed_torch``, ``seed_numpy``, ``seed_dagma`` are non-null;
        a length-three ``threshold_robustness_triple``; and the
        algorithm-name constants matching this module's values.
    """

    model: Literal["dagma", "dcdi"]
    condition: Literal["centred_only", "standardised"]
    seed_torch: int | None
    seed_numpy: int | None
    seed_dagma: int | None
    seed_populations: tuple[tuple[str, tuple[int, ...]], ...]
    intervention_set: tuple[InterventionSpec, ...]
    phase_b_configurations: tuple[PhaseBConfiguration, ...]
    threshold_robustness_triple: tuple[float, float, float]
    wrapper_api_reference: str
    n_nodes: int = 3
    expected_edges: int = 3
    noise_scale: float = 1.0
    weight_magnitude_range: tuple[float, float] = (0.5, 2.0)
    seed_derivation_rule: str = field(default=SEED_DERIVATION_RULE_NAME)
    configuration_hash_algorithm: str = field(
        default=CONFIGURATION_HASH_ALGORITHM_NAME
    )

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.model not in VALID_MODELS:
            raise ValueError(
                "model must be one of "
                f"{VALID_MODELS}; got {self.model!r}"
            )
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                "condition must be one of "
                f"{VALID_CONDITIONS}; got {self.condition!r}"
            )
        if self.model == "dagma":
            offenders = [
                name
                for name, value in (
                    ("seed_torch", self.seed_torch),
                    ("seed_numpy", self.seed_numpy),
                    ("seed_dagma", self.seed_dagma),
                )
                if value is not None
            ]
            if offenders:
                raise ValueError(
                    "model='dagma' requires "
                    f"{', '.join(offenders)} to be None: the fit is "
                    "deterministic by construction and does not call "
                    "the corresponding global RNG setters"
                )
        else:
            missing = [
                name
                for name, value in (
                    ("seed_torch", self.seed_torch),
                    ("seed_numpy", self.seed_numpy),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "model='dcdi' requires "
                    f"{', '.join(missing)} to be non-None: the DCDI "
                    "wrapper depends on global PyTorch and NumPy "
                    "RNG state during fit"
                )
            if self.seed_dagma is not None:
                raise ValueError(
                    "model='dcdi' requires seed_dagma to be None: "
                    "the DCDI wrapper does not call "
                    "dagma.utils.set_random_seed"
                )
        if len(self.threshold_robustness_triple) != 3:
            raise ValueError(
                "threshold_robustness_triple must have exactly three "
                f"values; got {len(self.threshold_robustness_triple)}"
            )
        if self.seed_derivation_rule != SEED_DERIVATION_RULE_NAME:
            raise ValueError(
                "seed_derivation_rule must equal "
                f"{SEED_DERIVATION_RULE_NAME!r}; "
                f"got {self.seed_derivation_rule!r}"
            )
        if (
            self.configuration_hash_algorithm
            != CONFIGURATION_HASH_ALGORITHM_NAME
        ):
            raise ValueError(
                "configuration_hash_algorithm must equal "
                f"{CONFIGURATION_HASH_ALGORITHM_NAME!r}; "
                f"got {self.configuration_hash_algorithm!r}"
            )
        unknown_population_keys = [
            name
            for name, _ in self.seed_populations
            if name not in VALID_SEED_POPULATIONS
        ]
        if unknown_population_keys:
            raise ValueError(
                "seed_populations contains key(s) not in "
                f"VALID_SEED_POPULATIONS={VALID_SEED_POPULATIONS}: "
                f"{', '.join(repr(k) for k in unknown_population_keys)}"
            )
        seen_population_names: set[str] = set()
        duplicate_population_names: list[str] = []
        for population_name, _ in self.seed_populations:
            if population_name in seen_population_names:
                if population_name not in duplicate_population_names:
                    duplicate_population_names.append(population_name)
            else:
                seen_population_names.add(population_name)
        if duplicate_population_names:
            raise ValueError(
                "seed_populations contains duplicate population "
                "name(s): "
                + ", ".join(
                    repr(n) for n in duplicate_population_names
                )
                + "; each population name may appear at most once"
            )
        for population_name, seeds in self.seed_populations:
            for seed_value in seeds:
                if isinstance(seed_value, bool):
                    raise ValueError(
                        "seed_populations["
                        f"{population_name!r}] contains a bool "
                        f"seed value {seed_value!r}; seeds must "
                        "be int (not bool)"
                    )
                if not isinstance(seed_value, int):
                    raise ValueError(
                        "seed_populations["
                        f"{population_name!r}] contains a non-int "
                        f"seed value {seed_value!r} (type "
                        f"{type(seed_value).__name__}); seeds "
                        "must be int"
                    )
                if seed_value < 0:
                    raise ValueError(
                        "seed_populations["
                        f"{population_name!r}] contains a negative "
                        f"seed value {seed_value}; seeds must be "
                        ">= 0"
                    )

        # SCM-generation field validation. These fields participate
        # in the configuration_hash and drive run-time SCM
        # construction; defaults preserve the schema-gate cell so
        # existing fixtures keep working until Phase A/B configs
        # override with the real selection-study values.
        if isinstance(self.n_nodes, bool) or not isinstance(
            self.n_nodes, int
        ):
            raise ValueError(
                "n_nodes must be a plain int (not bool); "
                f"got {self.n_nodes!r} of type "
                f"{type(self.n_nodes).__name__}"
            )
        if self.n_nodes < 2:
            raise ValueError(
                f"n_nodes must be >= 2; got {self.n_nodes}"
            )
        if isinstance(self.expected_edges, bool) or not isinstance(
            self.expected_edges, int
        ):
            raise ValueError(
                "expected_edges must be a plain int (not bool); "
                f"got {self.expected_edges!r} of type "
                f"{type(self.expected_edges).__name__}"
            )
        if self.expected_edges < 0:
            raise ValueError(
                "expected_edges must be >= 0; "
                f"got {self.expected_edges}"
            )
        max_edges = self.n_nodes * (self.n_nodes - 1) // 2
        if self.expected_edges > max_edges:
            raise ValueError(
                "expected_edges must be <= n_nodes*(n_nodes-1)//2"
                f"={max_edges} for n_nodes={self.n_nodes}; "
                f"got {self.expected_edges}"
            )
        if isinstance(self.noise_scale, bool) or not isinstance(
            self.noise_scale, (int, float)
        ):
            raise ValueError(
                "noise_scale must be a finite positive number "
                "(not bool); "
                f"got {self.noise_scale!r} of type "
                f"{type(self.noise_scale).__name__}"
            )
        noise_scale_float = float(self.noise_scale)
        if not (
            noise_scale_float == noise_scale_float
            and noise_scale_float not in (float("inf"), float("-inf"))
            and noise_scale_float > 0.0
        ):
            raise ValueError(
                "noise_scale must be a finite positive number; "
                f"got {self.noise_scale!r}"
            )
        if not isinstance(self.weight_magnitude_range, (tuple, list)):
            raise ValueError(
                "weight_magnitude_range must be a length-2 tuple "
                "or list of finite positive numbers; "
                f"got {self.weight_magnitude_range!r} of type "
                f"{type(self.weight_magnitude_range).__name__}"
            )
        if len(self.weight_magnitude_range) != 2:
            raise ValueError(
                "weight_magnitude_range must have exactly 2 values "
                "(low, high); "
                f"got {len(self.weight_magnitude_range)}"
            )
        low_raw, high_raw = self.weight_magnitude_range
        for label, value in (("low", low_raw), ("high", high_raw)):
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"weight_magnitude_range {label} must be a "
                    "finite positive number (not bool); "
                    f"got {value!r} of type {type(value).__name__}"
                )
            value_float = float(value)
            if not (
                value_float == value_float
                and value_float not in (
                    float("inf"), float("-inf")
                )
                and value_float > 0.0
            ):
                raise ValueError(
                    f"weight_magnitude_range {label} must be a "
                    f"finite positive number; got {value!r}"
                )
        if not (float(low_raw) <= float(high_raw)):
            raise ValueError(
                "weight_magnitude_range must satisfy "
                "0 < low <= high; "
                f"got (low={low_raw!r}, high={high_raw!r})"
            )

        # Frozen-dataclass storage normalisation. The fields are
        # validated above against the raw inputs; here they are
        # coerced to canonical immutable forms so a list passed at
        # construction time cannot be mutated through its outside
        # reference and leak through to the Configuration, and so
        # an int passed for noise_scale is stored as a float.
        object.__setattr__(self, "noise_scale", float(self.noise_scale))
        object.__setattr__(
            self,
            "weight_magnitude_range",
            (float(low_raw), float(high_raw)),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the Configuration as a primitive-typed dict.

        The returned dict contains only JSON-serialisable primitives:
        ``str``, ``int``, ``float``, ``bool``, ``None``, ``list``,
        and ``dict``. No dataclass repr, no NumPy scalar, no
        ``Path`` instance.
        """
        return {
            "model": self.model,
            "condition": self.condition,
            "seed_torch": self.seed_torch,
            "seed_numpy": self.seed_numpy,
            "seed_dagma": self.seed_dagma,
            "seed_populations": {
                str(population_name): [int(seed) for seed in seeds]
                for population_name, seeds in self.seed_populations
            },
            "intervention_set": [
                intervention.to_canonical_dict()
                for intervention in self.intervention_set
            ],
            "phase_b_configurations": [
                phase_b.to_canonical_dict()
                for phase_b in self.phase_b_configurations
            ],
            "threshold_robustness_triple": [
                float(value)
                for value in self.threshold_robustness_triple
            ],
            "wrapper_api_reference": self.wrapper_api_reference,
            "n_nodes": int(self.n_nodes),
            "expected_edges": int(self.expected_edges),
            "noise_scale": float(self.noise_scale),
            "weight_magnitude_range": [
                float(self.weight_magnitude_range[0]),
                float(self.weight_magnitude_range[1]),
            ],
            "seed_derivation_rule": self.seed_derivation_rule,
            "configuration_hash_algorithm": (
                self.configuration_hash_algorithm
            ),
        }


# --------------------------------------------------------------------------- #
# Canonical JSON, hash, prefix
# --------------------------------------------------------------------------- #


def canonical_json(config: Configuration) -> str:
    """Serialise a Configuration to canonical JSON.

    The serialisation uses ``json.dumps`` with ``sort_keys=True`` and
    tight separators ``(",", ":")``. Float values are serialised via
    Python's standard ``repr`` for floats, which produces the
    shortest decimal that round-trips and is deterministic across
    Python 3.1 and later. The Configuration is first converted to a
    primitive-typed dict via ``to_canonical_dict`` so no dataclass or
    NumPy scalar repr leaks into the output.

    Parameters
    ----------
    config : Configuration
        The Configuration to serialise.

    Returns
    -------
    str
        Canonical JSON string.
    """
    payload = config.to_canonical_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configuration_hash(config: Configuration) -> str:
    """Compute the SHA-256 hex digest of the canonical JSON encoding.

    Parameters
    ----------
    config : Configuration

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    encoded = canonical_json(config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configuration_hash_prefix(config: Configuration) -> str:
    """Return the first 12 characters of ``configuration_hash``.

    Parameters
    ----------
    config : Configuration

    Returns
    -------
    str
        First 12 characters of the SHA-256 hex digest. Used as the
        directory-path component for run storage.
    """
    return configuration_hash(config)[:12]


# --------------------------------------------------------------------------- #
# Seed derivation
# --------------------------------------------------------------------------- #


def _derive_seed_value(canonical_input: bytes) -> int:
    """Hash an encoded canonical input to a seed in ``[0, 2**31)``.

    The derivation rule is ``SEED_DERIVATION_RULE_NAME``:
    ``hashlib.sha256(canonical_input).digest()`` yields 32 bytes; the
    first 8 bytes are interpreted as an unsigned big-endian integer;
    the result is reduced modulo ``2**31`` to land inside the signed
    32-bit non-negative range. Python's built-in ``hash`` is not used
    anywhere in this derivation.
    """
    digest = hashlib.sha256(canonical_input).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % _SEED_BOUND


def _encode_run_identity_with_purpose(
    *,
    model: str,
    condition: str,
    seed_population: str,
    seed_replicate_index: int,
    configuration_hash_value: str,
    purpose_label: str,
) -> bytes:
    """Build the canonical byte encoding of an identity + purpose."""
    parts = (
        f"model={model}",
        f"condition={condition}",
        f"seed_population={seed_population}",
        f"seed_replicate_index={seed_replicate_index}",
        f"configuration_hash={configuration_hash_value}",
        f"purpose={purpose_label}",
    )
    return "|".join(parts).encode("utf-8")


def derive_per_run_seeds(
    *,
    model: str,
    condition: str,
    seed_population: str,
    seed_replicate_index: int,
    configuration_hash_value: str,
    include_validation_data_seed: bool,
) -> PerRunSeeds:
    """Derive the per-purpose seeds for a single run identity.

    Parameters
    ----------
    model : str
        Either ``"dagma"`` or ``"dcdi"``.
    condition : str
        Either ``"centred_only"`` or ``"standardised"``.
    seed_population : str
        The seed-population label, drawn from
        ``VALID_SEED_POPULATIONS``.
    seed_replicate_index : int
        Within-population replicate index.
    configuration_hash_value : str
        The full SHA-256 hex digest of the resolved Configuration.
    include_validation_data_seed : bool
        When ``True``, ``validation_data_seed`` is derived. When
        ``False``, ``validation_data_seed`` is ``None``.

    Returns
    -------
    PerRunSeeds
        Per-purpose seed record.

    Raises
    ------
    ValueError
        If ``model``, ``condition``, or ``seed_population`` is not
        from the declared set, or if ``seed_replicate_index`` is
        negative, or if ``configuration_hash_value`` is not a 64-
        character lowercase hex string.
    """
    if model not in VALID_MODELS:
        raise ValueError(
            f"model must be one of {VALID_MODELS}; got {model!r}"
        )
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"condition must be one of {VALID_CONDITIONS}; "
            f"got {condition!r}"
        )
    if seed_population not in VALID_SEED_POPULATIONS:
        raise ValueError(
            "seed_population must be one of "
            f"{VALID_SEED_POPULATIONS}; got {seed_population!r}"
        )
    if seed_replicate_index < 0:
        raise ValueError(
            "seed_replicate_index must be >= 0; "
            f"got {seed_replicate_index}"
        )
    if (
        len(configuration_hash_value) != 64
        or not all(
            ch in "0123456789abcdef"
            for ch in configuration_hash_value
        )
    ):
        raise ValueError(
            "configuration_hash_value must be a 64-character "
            "lowercase hex string"
        )

    def derive(purpose: str) -> int:
        return _derive_seed_value(
            _encode_run_identity_with_purpose(
                model=model,
                condition=condition,
                seed_population=seed_population,
                seed_replicate_index=seed_replicate_index,
                configuration_hash_value=configuration_hash_value,
                purpose_label=purpose,
            )
        )

    return PerRunSeeds(
        graph_seed=derive(PURPOSE_GRAPH_SEED),
        train_data_seed=derive(PURPOSE_TRAIN_DATA_SEED),
        validation_data_seed=(
            derive(PURPOSE_VALIDATION_DATA_SEED)
            if include_validation_data_seed
            else None
        ),
        intervention_ground_truth_seed_base=derive(
            PURPOSE_INTERVENTION_GROUND_TRUTH_SEED_BASE
        ),
        model_sampling_seed_base=derive(
            PURPOSE_MODEL_SAMPLING_SEED_BASE
        ),
    )


def derive_per_intervention_seed(
    *, base_seed: int, intervention_id: str
) -> int:
    """Derive a per-intervention seed from a base seed and id.

    Parameters
    ----------
    base_seed : int
        One of ``intervention_ground_truth_seed_base`` or
        ``model_sampling_seed_base`` for a particular run.
    intervention_id : str
        Stable identifier of the intervention.

    Returns
    -------
    int
        Deterministic seed in ``[0, 2**31)``.

    Raises
    ------
    TypeError
        If ``base_seed`` is not an integer.
    ValueError
        If ``base_seed`` is outside ``[0, 2**31)``.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError(
            f"base_seed must be int; got {type(base_seed).__name__}"
        )
    if base_seed < 0 or base_seed >= _SEED_BOUND:
        raise ValueError(
            f"base_seed must be in [0, {_SEED_BOUND}); got {base_seed}"
        )
    encoded = (
        f"base_seed={base_seed}"
        f"|intervention_id={intervention_id}"
    ).encode("utf-8")
    return _derive_seed_value(encoded)


# --------------------------------------------------------------------------- #
# Config loading from disk
# --------------------------------------------------------------------------- #


_REQUIRED_FIELDS: tuple[str, ...] = (
    "model",
    "condition",
    "seed_torch",
    "seed_numpy",
    "seed_dagma",
    "seed_populations",
    "intervention_set",
    "phase_b_configurations",
    "threshold_robustness_triple",
    "wrapper_api_reference",
    "n_nodes",
    "expected_edges",
    "noise_scale",
    "weight_magnitude_range",
)


def load_config(path: str | Path) -> Configuration:
    """Load a Configuration from a JSON file on disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the JSON configuration file.

    Returns
    -------
    Configuration
        Frozen Configuration constructed from the file contents.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file does not parse as JSON, if the top-level value is
        not a JSON object, if any required field is missing, or if
        Configuration validation fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"configuration file not found at {file_path}"
        )
    raw_text = file_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "configuration file at "
            f"{file_path} is not valid JSON: {exc}"
        ) from exc
    return _configuration_from_dict(data)


def _configuration_from_dict(data: Any) -> Configuration:
    """Construct a Configuration from a parsed JSON object."""
    if not isinstance(data, dict):
        raise ValueError(
            "configuration JSON top-level value must be an object; "
            f"got {type(data).__name__}"
        )

    missing = [name for name in _REQUIRED_FIELDS if name not in data]
    if missing:
        raise ValueError(
            "configuration JSON missing required field(s): "
            + ", ".join(missing)
        )

    seed_populations_raw = data["seed_populations"]
    if not isinstance(seed_populations_raw, dict):
        raise ValueError(
            "seed_populations must be a JSON object; "
            f"got {type(seed_populations_raw).__name__}"
        )
    seed_populations: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
        (
            str(population_name),
            tuple(seeds),
        )
        for population_name, seeds in sorted(
            seed_populations_raw.items()
        )
    )

    intervention_set_raw = data["intervention_set"]
    if not isinstance(intervention_set_raw, list):
        raise ValueError(
            "intervention_set must be a JSON array; "
            f"got {type(intervention_set_raw).__name__}"
        )
    intervention_set = tuple(
        InterventionSpec(
            intervention_id=str(item["intervention_id"]),
            target_node=int(item["target_node"]),
            value_raw=float(item["value_raw"]),
        )
        for item in intervention_set_raw
    )

    phase_b_raw = data["phase_b_configurations"]
    if not isinstance(phase_b_raw, list):
        raise ValueError(
            "phase_b_configurations must be a JSON array; "
            f"got {type(phase_b_raw).__name__}"
        )
    phase_b_configurations = tuple(
        PhaseBConfiguration(
            name=str(item["name"]),
            hyperparameters=tuple(
                sorted(
                    (str(key), value)
                    for key, value in item["hyperparameters"].items()
                )
            ),
        )
        for item in phase_b_raw
    )

    threshold_triple_raw = data["threshold_robustness_triple"]
    if (
        not isinstance(threshold_triple_raw, list)
        or len(threshold_triple_raw) != 3
    ):
        raise ValueError(
            "threshold_robustness_triple must be a JSON array of "
            "length 3"
        )
    threshold_robustness_triple = (
        float(threshold_triple_raw[0]),
        float(threshold_triple_raw[1]),
        float(threshold_triple_raw[2]),
    )

    seed_torch_raw = data["seed_torch"]
    seed_numpy_raw = data["seed_numpy"]
    seed_dagma_raw = data["seed_dagma"]

    n_nodes_raw = data["n_nodes"]
    if isinstance(n_nodes_raw, bool) or not isinstance(
        n_nodes_raw, int
    ):
        raise ValueError(
            "configuration JSON field 'n_nodes' must be an int "
            f"(not bool); got {n_nodes_raw!r}"
        )
    expected_edges_raw = data["expected_edges"]
    if isinstance(expected_edges_raw, bool) or not isinstance(
        expected_edges_raw, int
    ):
        raise ValueError(
            "configuration JSON field 'expected_edges' must be an "
            f"int (not bool); got {expected_edges_raw!r}"
        )
    noise_scale_raw = data["noise_scale"]
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError(
            "configuration JSON field 'noise_scale' must be a "
            f"number (not bool); got {noise_scale_raw!r}"
        )
    weight_magnitude_range_raw = data["weight_magnitude_range"]
    if (
        not isinstance(weight_magnitude_range_raw, list)
        or len(weight_magnitude_range_raw) != 2
    ):
        raise ValueError(
            "configuration JSON field 'weight_magnitude_range' "
            "must be a JSON array of length 2"
        )

    return Configuration(
        model=str(data["model"]),
        condition=str(data["condition"]),
        seed_torch=(
            None if seed_torch_raw is None else int(seed_torch_raw)
        ),
        seed_numpy=(
            None if seed_numpy_raw is None else int(seed_numpy_raw)
        ),
        seed_dagma=(
            None if seed_dagma_raw is None else int(seed_dagma_raw)
        ),
        seed_populations=seed_populations,
        intervention_set=intervention_set,
        phase_b_configurations=phase_b_configurations,
        threshold_robustness_triple=threshold_robustness_triple,
        wrapper_api_reference=str(data["wrapper_api_reference"]),
        n_nodes=int(n_nodes_raw),
        expected_edges=int(expected_edges_raw),
        noise_scale=float(noise_scale_raw),
        weight_magnitude_range=(
            float(weight_magnitude_range_raw[0]),
            float(weight_magnitude_range_raw[1]),
        ),
        seed_derivation_rule=str(
            data.get(
                "seed_derivation_rule", SEED_DERIVATION_RULE_NAME
            )
        ),
        configuration_hash_algorithm=str(
            data.get(
                "configuration_hash_algorithm",
                CONFIGURATION_HASH_ALGORITHM_NAME,
            )
        ),
    )
