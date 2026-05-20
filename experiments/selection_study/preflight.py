"""Preflight manifest enumeration and validation for the selection-study runner.

This module enumerates every planned run into a ``Manifest``, validates
the manifest against six rules before any fit can be invoked, and saves
the validated manifest as a JSON artefact.

The six validation rules are:

- (a) No calibration seed appears in the held-out evaluation population,
  and vice versa.
- (b) No duplicate ``run_id`` appears in the manifest.
- (c) The ``configuration_hash`` is stable: a second call inside the same
  preflight invocation returns the same digest.
- (d) Every ``expected_output_directory`` is either absent or an empty
  directory; no pre-existing populated run directory is accepted.
- (e) Every ``ManifestEntry`` carries all 15 mandatory fields with the
  correct types; a schema-level pre-check, no actual values are computed.
- (f) No ``wandb`` module is reachable from the preflight code path.

No wrapper fit is invoked from any code path in this module. No wrapper
or DAGMA/DCDI import is present.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    Configuration,
    canonical_json,
    configuration_hash as compute_configuration_hash,
    derive_per_intervention_seed,
    derive_per_run_seeds,
    load_config,
)
from experiments.selection_study.identity import (
    derive_run_directory,
    derive_run_id,
)


_DEFAULT_BASE_DIR = Path("results/model_selection")
_DEFAULT_MANIFEST_DIR = Path("results/model_selection/_preflight")
_SCHEMA_VERSION = 1
_MISSING = object()

_ENTRY_FIELD_TYPES: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("model", str),
    ("condition", str),
    ("seed_population", str),
    ("seed_replicate_index", int),
    ("graph_seed", int),
    ("train_data_seed", int),
    ("validation_data_seed", (int, type(None))),
    ("intervention_ground_truth_seed_base", int),
    ("model_sampling_seed_base", int),
    ("per_intervention_seeds", tuple),
    ("configuration_hash", str),
    ("expected_run_id", str),
    ("expected_output_directory", str),
    ("planned_wrapper", str),
    ("planned_sampling_policy", str),
)


class ManifestValidationError(ValueError):
    """Raised when any preflight validation rule is violated.

    Inherits from ``ValueError`` so callers can catch it with a broad
    ``except ValueError`` if needed. The message names the failing rule
    and the offending value(s).
    """


@dataclass(frozen=True)
class PerInterventionSeeds:
    """Deterministic seeds for a single intervention within a single run.

    Parameters
    ----------
    ground_truth_sampling_seed : int
        Seed used to draw ground-truth interventional samples from the SCM.
    model_sampling_seed : int
        Seed used to draw model-generated interventional samples.
    """

    ground_truth_sampling_seed: int
    model_sampling_seed: int


@dataclass(frozen=True)
class ManifestEntry:
    """A single planned-run record within a ``Manifest``.

    Contains the 15 mandatory fields that describe the run identity,
    derived seeds, expected filesystem artefact location, and the
    wrapper/sampling policy selected for the run.

    All 15 fields are present for every entry. The ``validation_data_seed``
    field is ``None`` for DAGMA runs (which do not use a validation split)
    and an ``int`` for DCDI runs.

    The ``per_intervention_seeds`` field is a sorted tuple of
    ``(intervention_id, PerInterventionSeeds)`` pairs, ordered by
    ``intervention_id``. This representation is hashable, which allows
    ``ManifestEntry`` to remain a frozen dataclass.

    The ``expected_output_directory`` is the POSIX string form of the
    run-directory path (``Path.as_posix()``), not a ``Path`` object, to
    keep the type JSON-round-trippable without a custom encoder.
    """

    model: str
    condition: str
    seed_population: str
    seed_replicate_index: int
    graph_seed: int
    train_data_seed: int
    validation_data_seed: int | None
    intervention_ground_truth_seed_base: int
    model_sampling_seed_base: int
    per_intervention_seeds: tuple[tuple[str, PerInterventionSeeds], ...]
    configuration_hash: str
    expected_run_id: str
    expected_output_directory: str
    planned_wrapper: str
    planned_sampling_policy: str


@dataclass(frozen=True)
class Manifest:
    """The full preflight manifest for a configuration.

    A ``Manifest`` is not hashable because its ``resolved_config`` field
    is a ``dict``. This is acceptable: ``Manifest`` objects are never
    used as dictionary keys or set members.

    Parameters
    ----------
    configuration_hash : str
        Full 64-character SHA-256 digest of the resolved configuration.
    schema_version : int
        Version integer for the manifest schema. The initial value is
        ``1``.
    seed_derivation_rule : str
        The stable derivation-rule name carried from the configuration.
    configuration_hash_algorithm : str
        The algorithm name used to compute ``configuration_hash``. Equal
        to ``CONFIGURATION_HASH_ALGORITHM_NAME`` from the ``config``
        module.
    resolved_config : dict
        The canonical dict of the configuration (primitive-typed).
    entries : tuple of ManifestEntry
        One entry per planned run, in enumeration order.
    """

    configuration_hash: str
    schema_version: int
    seed_derivation_rule: str
    configuration_hash_algorithm: str
    resolved_config: dict[str, Any]
    entries: tuple[ManifestEntry, ...]

    def __hash__(self) -> int:  # type: ignore[override]
        raise TypeError(
            "Manifest is not hashable (resolved_config is a dict)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Manifest):
            return NotImplemented
        return (
            self.configuration_hash == other.configuration_hash
            and self.schema_version == other.schema_version
            and self.seed_derivation_rule == other.seed_derivation_rule
            and self.configuration_hash_algorithm == other.configuration_hash_algorithm
            and self.resolved_config == other.resolved_config
            and self.entries == other.entries
        )


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #


def enumerate_manifest(
    config: Configuration,
    *,
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> Manifest:
    """Enumerate every planned run into a ``Manifest``.

    Iterates over every ``(seed_population, seed_replicate_index)`` pair
    in the configuration, derives the per-purpose seeds for each run via
    the ``config`` module's seed-derivation rule, and assembles a
    ``ManifestEntry`` for each planned run. The entries are in the order
    the populations appear in ``config.seed_populations``, with
    ``seed_replicate_index`` increasing within each population.

    This function is a pure computation: it does not create any
    directories, does not touch the filesystem beyond reading the
    configuration object already in memory, and does not import any
    wrapper or model code.

    Parameters
    ----------
    config : Configuration
        The resolved, frozen configuration for this selection study.
    base_dir : pathlib.Path, optional
        Root of the run-storage tree. Defaults to
        ``Path("results/model_selection")``. Pass a ``tmp_path``-relative
        path in tests to keep all filesystem operations hermetic.

    Returns
    -------
    Manifest
        The complete manifest for the given configuration. Same
        ``Configuration`` always produces a byte-identical manifest.
    """
    config_hash = compute_configuration_hash(config)
    resolved_config: dict[str, Any] = json.loads(canonical_json(config))
    include_validation_seed = config.model == "dcdi"
    planned_sampling_policy = (
        "residual_fitted" if config.model == "dagma" else "native_conditionals"
    )

    entries: list[ManifestEntry] = []
    for seed_population, seeds in config.seed_populations:
        for seed_replicate_index in range(len(seeds)):
            per_run = derive_per_run_seeds(
                model=config.model,
                condition=config.condition,
                seed_population=seed_population,
                seed_replicate_index=seed_replicate_index,
                configuration_hash_value=config_hash,
                include_validation_data_seed=include_validation_seed,
            )
            per_intervention: list[tuple[str, PerInterventionSeeds]] = []
            for intervention in config.intervention_set:
                iid = intervention.intervention_id
                per_intervention.append(
                    (
                        iid,
                        PerInterventionSeeds(
                            ground_truth_sampling_seed=derive_per_intervention_seed(
                                base_seed=per_run.intervention_ground_truth_seed_base,
                                intervention_id=iid,
                            ),
                            model_sampling_seed=derive_per_intervention_seed(
                                base_seed=per_run.model_sampling_seed_base,
                                intervention_id=iid,
                            ),
                        ),
                    )
                )
            per_intervention_sorted = tuple(
                sorted(per_intervention, key=lambda x: x[0])
            )
            run_id = derive_run_id(
                model=config.model,
                condition=config.condition,
                seed_population=seed_population,
                seed_replicate_index=seed_replicate_index,
                configuration_hash=config_hash,
            )
            run_dir = derive_run_directory(
                model=config.model,
                condition=config.condition,
                seed_population=seed_population,
                seed_replicate_index=seed_replicate_index,
                configuration_hash=config_hash,
                base_dir=base_dir,
            )
            entries.append(
                ManifestEntry(
                    model=config.model,
                    condition=config.condition,
                    seed_population=seed_population,
                    seed_replicate_index=seed_replicate_index,
                    graph_seed=per_run.graph_seed,
                    train_data_seed=per_run.train_data_seed,
                    validation_data_seed=per_run.validation_data_seed,
                    intervention_ground_truth_seed_base=(
                        per_run.intervention_ground_truth_seed_base
                    ),
                    model_sampling_seed_base=per_run.model_sampling_seed_base,
                    per_intervention_seeds=per_intervention_sorted,
                    configuration_hash=config_hash,
                    expected_run_id=run_id,
                    expected_output_directory=run_dir.as_posix(),
                    planned_wrapper=config.wrapper_api_reference,
                    planned_sampling_policy=planned_sampling_policy,
                )
            )

    return Manifest(
        configuration_hash=config_hash,
        schema_version=_SCHEMA_VERSION,
        seed_derivation_rule=config.seed_derivation_rule,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        resolved_config=resolved_config,
        entries=tuple(entries),
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_manifest(
    manifest: Manifest,
    *,
    hash_recheck_config: Configuration,
) -> None:
    """Validate the manifest against the six preflight rules.

    Rules are checked in order (e), (a), (b), (c), (d), (f). Rule (e)
    is a structural pre-check that verifies every entry field exists and
    has the correct type; it runs first so that subsequent rules can
    access entry fields without AttributeError. The first failing rule
    raises ``ManifestValidationError`` immediately; subsequent rules are
    not checked after a failure.

    Parameters
    ----------
    manifest : Manifest
        The manifest to validate, produced by ``enumerate_manifest``.
    hash_recheck_config : Configuration
        The original ``Configuration`` used to produce the manifest;
        used by rule (c) to recompute and compare the hash.

    Returns
    -------
    None
        On success.

    Raises
    ------
    ManifestValidationError
        On the first failing rule. The message names the rule letter and
        the offending value(s).
    """
    _validate_rule_e(manifest)
    _validate_rule_a(manifest)
    _validate_rule_b(manifest)
    _validate_rule_c(manifest, hash_recheck_config)
    _validate_rule_d(manifest)
    _validate_rule_f()


def _validate_rule_a(manifest: Manifest) -> None:
    """(a) Calibration and held-out-evaluation seed sets must be disjoint."""
    seed_pops = manifest.resolved_config.get("seed_populations", {})
    calibration_seeds = set(seed_pops.get("calibration", []))
    held_out_seeds = set(seed_pops.get("held_out_evaluation", []))
    overlap = calibration_seeds & held_out_seeds
    if overlap:
        raise ManifestValidationError(
            "rule (a): calibration and held_out_evaluation seed populations "
            "overlap; the overlapping seed value(s) would produce runs that "
            "simultaneously belong to both populations, violating the "
            "non-overlap invariant. Overlapping value(s): "
            + repr(sorted(overlap))
        )


def _validate_rule_b(manifest: Manifest) -> None:
    """(b) All run_ids in the manifest must be unique."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in manifest.entries:
        rid = entry.expected_run_id
        if rid in seen and rid not in duplicates:
            duplicates.append(rid)
        seen.add(rid)
    if duplicates:
        raise ManifestValidationError(
            "rule (b): duplicate run_id value(s) found in the manifest. "
            "Duplicate(s): " + repr(duplicates)
        )


def _validate_rule_c(
    manifest: Manifest, hash_recheck_config: Configuration
) -> None:
    """(c) configuration_hash must be stable across a second derivation."""
    recomputed = compute_configuration_hash(hash_recheck_config)
    if recomputed != manifest.configuration_hash:
        raise ManifestValidationError(
            "rule (c): configuration_hash is not stable. "
            f"Manifest carries {manifest.configuration_hash!r} but a "
            f"second derivation produced {recomputed!r}. The Configuration "
            "object passed to validate_manifest must be the same one "
            "used to produce the manifest."
        )


def _validate_rule_d(manifest: Manifest) -> None:
    """(d) Every expected_output_directory must be absent or empty."""
    for entry in manifest.entries:
        path = Path(entry.expected_output_directory)
        if not path.exists():
            continue
        if not path.is_dir():
            raise ManifestValidationError(
                "rule (d): expected_output_directory exists but is not a "
                f"directory: {path}. The path belongs to run_id "
                f"{entry.expected_run_id!r}."
            )
        contents = list(path.iterdir())
        if contents:
            raise ManifestValidationError(
                "rule (d): expected_output_directory is already populated; "
                "refusing to overwrite: "
                f"{path}. The path belongs to run_id "
                f"{entry.expected_run_id!r}."
            )


def _type_includes_int(expected_type: type | tuple[type, ...]) -> bool:
    """Return True when ``int`` is one of the accepted types.

    Used by ``_validate_rule_e`` to detect the bool-as-int trap: Python's
    ``bool`` is a subclass of ``int``, so ``isinstance(True, int)`` is
    ``True``. Fields declared as ``int`` or ``int | None`` must explicitly
    reject ``bool`` values.
    """
    if isinstance(expected_type, tuple):
        return int in expected_type
    return expected_type is int


def _validate_rule_e(manifest: Manifest) -> None:
    """(e) Every ManifestEntry must carry all 15 fields with correct types.

    Bool values are rejected for any field whose declared type includes
    ``int``. Python's ``bool`` is a subclass of ``int``, so a naive
    ``isinstance`` check would silently accept ``True`` or ``False`` where
    an integer seed is required.
    """
    for idx, entry in enumerate(manifest.entries):
        run_id_label = getattr(entry, "expected_run_id", "<unknown>")
        for field_name, expected_type in _ENTRY_FIELD_TYPES:
            value = getattr(entry, field_name, _MISSING)
            if value is _MISSING:
                raise ManifestValidationError(
                    f"rule (e): ManifestEntry at index {idx} "
                    f"(run_id={run_id_label!r}) is missing "
                    f"mandatory field {field_name!r}."
                )
            if _type_includes_int(expected_type) and isinstance(value, bool):
                raise ManifestValidationError(
                    f"rule (e): ManifestEntry at index {idx} "
                    f"(run_id={run_id_label!r}) field "
                    f"{field_name!r} must be int (not bool); bool is a "
                    f"subclass of int in Python but is not accepted for "
                    f"seed or index fields. Got {value!r}."
                )
            if not isinstance(value, expected_type):
                raise ManifestValidationError(
                    f"rule (e): ManifestEntry at index {idx} "
                    f"(run_id={run_id_label!r}) field "
                    f"{field_name!r} has wrong type: expected "
                    f"{expected_type}, got {type(value).__name__!r} "
                    f"with value {value!r}."
                )


def _validate_rule_f() -> None:
    """(f) No wandb module may be loaded in the current process."""
    forbidden = [
        name
        for name in sys.modules
        if name == "wandb" or name.startswith("wandb.")
    ]
    if forbidden:
        raise ManifestValidationError(
            "rule (f): wandb module(s) are loaded in sys.modules, which "
            "indicates the preflight code path is reachable via wandb. "
            "Forbidden module(s): " + repr(sorted(forbidden))
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _entry_to_dict(entry: ManifestEntry) -> dict[str, Any]:
    """Serialise a ``ManifestEntry`` to a JSON-ready primitive dict."""
    return {
        "model": entry.model,
        "condition": entry.condition,
        "seed_population": entry.seed_population,
        "seed_replicate_index": entry.seed_replicate_index,
        "graph_seed": entry.graph_seed,
        "train_data_seed": entry.train_data_seed,
        "validation_data_seed": entry.validation_data_seed,
        "intervention_ground_truth_seed_base": (
            entry.intervention_ground_truth_seed_base
        ),
        "model_sampling_seed_base": entry.model_sampling_seed_base,
        "per_intervention_seeds": {
            iid: {
                "ground_truth_sampling_seed": seeds.ground_truth_sampling_seed,
                "model_sampling_seed": seeds.model_sampling_seed,
            }
            for iid, seeds in entry.per_intervention_seeds
        },
        "configuration_hash": entry.configuration_hash,
        "expected_run_id": entry.expected_run_id,
        "expected_output_directory": entry.expected_output_directory,
        "planned_wrapper": entry.planned_wrapper,
        "planned_sampling_policy": entry.planned_sampling_policy,
    }


def _manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    """Serialise a ``Manifest`` to a JSON-ready primitive dict."""
    return {
        "configuration_hash": manifest.configuration_hash,
        "schema_version": manifest.schema_version,
        "seed_derivation_rule": manifest.seed_derivation_rule,
        "configuration_hash_algorithm": manifest.configuration_hash_algorithm,
        "resolved_config": manifest.resolved_config,
        "entries": [_entry_to_dict(e) for e in manifest.entries],
    }


def save_manifest(manifest: Manifest, manifest_dir: Path) -> Path:
    """Persist the manifest to a deterministic JSON file.

    The file name is ``manifest_<hash_prefix>.json`` where
    ``<hash_prefix>`` is the first 12 characters of
    ``manifest.configuration_hash``. The JSON is compact, sorted, and
    ASCII-safe. If the file already exists, the new payload is compared
    byte-for-byte with the existing content; a drift raises rather than
    overwriting.

    Parameters
    ----------
    manifest : Manifest
        The validated manifest to persist.
    manifest_dir : pathlib.Path
        Directory in which to write the manifest file. Created (with
        parents) if it does not exist.

    Returns
    -------
    pathlib.Path
        The path of the written (or already-existing identical) file.

    Raises
    ------
    ValueError
        If the file already exists and its content differs from the new
        payload (manifest drift).
    """
    path = manifest_dir / f"manifest_{manifest.configuration_hash[:12]}.json"
    payload = json.dumps(
        _manifest_to_dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise ValueError(
                f"manifest drift detected: {path} exists but its content "
                "differs from the newly generated manifest. The configuration "
                "or enumeration logic has changed since the manifest was last "
                "saved. Delete the existing manifest and re-run preflight."
            )
        return path
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_preflight(
    config_path: Path,
    *,
    base_dir: Path = _DEFAULT_BASE_DIR,
    manifest_dir: Path = _DEFAULT_MANIFEST_DIR,
) -> Path:
    """Load, enumerate, validate, and save the preflight manifest.

    This is the function wired to the ``--dry-run`` CLI flag. It performs
    no model fit: no wrapper is imported, no DAGMA or DCDI module is
    touched, and no SCM is constructed.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the JSON configuration file on disk.
    base_dir : pathlib.Path, optional
        Root of the run-storage tree. Defaults to
        ``Path("results/model_selection")``.
    manifest_dir : pathlib.Path, optional
        Directory in which the manifest JSON is written. Defaults to
        ``Path("results/model_selection/_preflight")``.

    Returns
    -------
    pathlib.Path
        The path of the saved manifest file.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist on disk.
    ManifestValidationError
        If any of the six preflight validation rules fails.
    ValueError
        If the manifest file already exists and its content drifts.
    """
    config = load_config(config_path)
    manifest = enumerate_manifest(config, base_dir=base_dir)
    validate_manifest(manifest, hash_recheck_config=config)
    return save_manifest(manifest, manifest_dir)
