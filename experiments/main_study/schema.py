"""Planning-side schema, factory, and canonical hashing for the main-study pipeline.

This module defines the immutable :class:`MainStudyConfig` describing
a single experimental condition (one ``(method_family, seed,
hyperparameters)`` point), a factory that enforces method-family
invariants while injecting the protocol's frozen scalars, a
deterministic canonical-JSON + SHA-256 configuration-hash function
that mirrors the project's existing run-identity hash pattern, and a
short run-id derived from that hash.

The module does not enumerate workloads, run any model, define any
post-run record schema, embed wrapper diagnostics, or persist
anything to disk.

Edge representation follows the project's row-source /
column-destination convention.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from typing import Any, Optional

from experiments.main_study.priors import (
    CORRUPTION_GRID,
    PRIOR_K,
    CorruptedPriorSpec,
    PriorSpec,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 2

METHOD_FAMILIES: tuple[str, ...] = (
    "prior_free",
    "soft_frobenius",
    "matched_l1",
    "hard_exclusion",
)

SEED_POPULATIONS: tuple[str, ...] = (
    "main_calibration",
    "main_evaluation",
)

CALIBRATION_SEEDS: tuple[int, ...] = (401, 402)
EVALUATION_SEEDS: tuple[int, ...] = (501, 502, 503, 504, 505, 506, 507)

FROZEN_LAMBDA_PRIOR: float = 2e-4

# Protocol-frozen DAGMA hyperparameters for the main study. These
# values come from the frozen tactical-constants block inherited
# from the closed selection study, and from the held-out
# adjudication that selected DAGMA-standardised at hash
# 7b345b1b2e85 with lambda1=0.10. The wrapper-level DAGMAConfig
# defaults (lambda1=0.05, warm_iter=30000, max_iter=60000) are the
# DAGMA-paper reproduction anchor for the wrapper's own contract
# and are intentionally NOT changed; the main-study entry points
# are responsible for overriding them with the protocol values
# below.
PROTOCOL_DAGMA_LAMBDA1: float = 0.1
PROTOCOL_DAGMA_WARM_ITER: int = 20000
PROTOCOL_DAGMA_MAX_ITER: int = 70000


def build_protocol_dagma_config() -> DAGMAConfig:
    """Return the DAGMAConfig at the main-study protocol point.

    The wrapper-level :class:`DAGMAConfig` defaults are the
    DAGMA-paper reproduction anchor (``lambda1=0.05``,
    ``warm_iter=30000``, ``max_iter=60000``) and must not be relied
    on by the main study. Every call site that needs a base
    configuration for main-study workloads must obtain it through
    this factory so the protocol values are always honoured.
    """
    return DAGMAConfig(
        lambda1=PROTOCOL_DAGMA_LAMBDA1,
        warm_iter=PROTOCOL_DAGMA_WARM_ITER,
        max_iter=PROTOCOL_DAGMA_MAX_ITER,
    )


CONFIDENCE_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# Field names whose values must be canonicalised as a sorted list of
# ``[i, j]`` edge pairs. This applies recursively whenever the
# canonicaliser sees a field with one of these names.
_EDGE_FIELD_NAMES: frozenset[str] = frozenset({
    "forbidden_edges",
    "removed_clean_edges",
    "added_true_positive_edges",
    "exclude_edges",
})

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Main-study configuration dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class MainStudyConfig:
    """Immutable description of a single main-study condition.

    The dataclass is frozen and keyword-only at construction. The
    ``__post_init__`` validators enforce: schema-version, method-
    family membership, seed-population / seed-value coherence, the
    64-character lowercase hex form of ``parent_heldout_run_hash_full``,
    method-family field invariants, and (for hard exclusion) the
    set equality between ``dagma_config.exclude_edges`` and
    ``corrupted_prior_spec.forbidden_edges``.

    The dimension of the condition is taken from
    ``corrupted_prior_spec.n_nodes`` when a corrupted-prior spec is
    present, because :class:`DAGMAConfig` does not expose ``n_nodes``;
    a follow-up record schema may carry the dimension explicitly.
    """

    method_family: str
    seed_value: int
    seed_population: str
    dagma_config: DAGMAConfig
    parent_heldout_run_hash_full: str
    lambda_prior: Optional[float] = None
    confidence: Optional[float] = None
    corrupted_prior_spec: Optional[CorruptedPriorSpec] = None
    matched_l1_lambda1: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Schema version.
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{SCHEMA_VERSION}; got {self.schema_version!r}."
            )

        # Method family.
        if self.method_family not in METHOD_FAMILIES:
            raise ValueError(
                f"method_family must be one of {METHOD_FAMILIES}; "
                f"got {self.method_family!r}."
            )

        # Seed population.
        if self.seed_population not in SEED_POPULATIONS:
            raise ValueError(
                f"seed_population must be one of {SEED_POPULATIONS}; "
                f"got {self.seed_population!r}."
            )

        # Seed-value vs seed-population boundary.
        if self.seed_population == "main_calibration":
            if self.seed_value not in CALIBRATION_SEEDS:
                raise ValueError(
                    f"seed_value {self.seed_value!r} not allowed for "
                    f"main_calibration; allowed: {CALIBRATION_SEEDS}."
                )
        elif self.seed_population == "main_evaluation":
            if self.seed_value not in EVALUATION_SEEDS:
                raise ValueError(
                    f"seed_value {self.seed_value!r} not allowed for "
                    f"main_evaluation; allowed: {EVALUATION_SEEDS}."
                )

        # parent_heldout_run_hash_full: 64-char lowercase hex.
        if not isinstance(self.parent_heldout_run_hash_full, str):
            raise ValueError(
                "parent_heldout_run_hash_full must be a string; got "
                f"{type(self.parent_heldout_run_hash_full).__name__}."
            )
        if not _HEX_64_RE.fullmatch(self.parent_heldout_run_hash_full):
            raise ValueError(
                "parent_heldout_run_hash_full must be exactly 64 "
                "lowercase hex characters; got "
                f"{self.parent_heldout_run_hash_full!r}."
            )

        # Method-family field invariants.
        if self.method_family == "prior_free":
            self._validate_prior_free()
        elif self.method_family == "soft_frobenius":
            self._validate_soft_frobenius()
        elif self.method_family == "matched_l1":
            self._validate_matched_l1()
        elif self.method_family == "hard_exclusion":
            self._validate_hard_exclusion()
        # No other branches reachable: METHOD_FAMILIES is exhaustive
        # and was checked above.

    # -- Per-family validators --------------------------------------

    def _validate_prior_free(self) -> None:
        for label, value in (
            ("confidence", self.confidence),
            ("corrupted_prior_spec", self.corrupted_prior_spec),
            ("lambda_prior", self.lambda_prior),
            ("matched_l1_lambda1", self.matched_l1_lambda1),
        ):
            if value is not None:
                raise ValueError(
                    "prior_free does not accept "
                    f"{label}={value!r}."
                )
        if self.dagma_config.exclude_edges is not None:
            raise ValueError(
                "prior_free requires dagma_config.exclude_edges=None; "
                f"got {self.dagma_config.exclude_edges!r}."
            )

    def _validate_soft_frobenius(self) -> None:
        if self.confidence is None:
            raise ValueError("soft_frobenius requires confidence.")
        if not _is_close_to_grid(self.confidence, CONFIDENCE_GRID):
            raise ValueError(
                "soft_frobenius confidence must be in "
                f"{CONFIDENCE_GRID}; got {self.confidence!r}."
            )
        if self.corrupted_prior_spec is None:
            raise ValueError(
                "soft_frobenius requires corrupted_prior_spec."
            )
        cf = self.corrupted_prior_spec.corruption_fraction
        if not _is_close_to_grid(cf, CORRUPTION_GRID):
            raise ValueError(
                "soft_frobenius corrupted_prior_spec.corruption_fraction "
                f"must be in {CORRUPTION_GRID}; got {cf!r}."
            )
        if self.lambda_prior is None:
            raise ValueError("soft_frobenius requires lambda_prior.")
        if not math.isclose(
            self.lambda_prior, FROZEN_LAMBDA_PRIOR, abs_tol=1e-12
        ):
            raise ValueError(
                "soft_frobenius requires "
                f"lambda_prior=={FROZEN_LAMBDA_PRIOR}; "
                f"got {self.lambda_prior!r}."
            )
        if self.matched_l1_lambda1 is not None:
            raise ValueError(
                "soft_frobenius does not accept "
                f"matched_l1_lambda1={self.matched_l1_lambda1!r}."
            )
        if self.dagma_config.exclude_edges is not None:
            raise ValueError(
                "soft_frobenius requires "
                "dagma_config.exclude_edges=None; got "
                f"{self.dagma_config.exclude_edges!r}."
            )

    def _validate_matched_l1(self) -> None:
        if self.confidence is not None:
            raise ValueError(
                "matched_l1 does not accept "
                f"confidence={self.confidence!r}."
            )
        if self.corrupted_prior_spec is not None:
            raise ValueError(
                "matched_l1 does not accept corrupted_prior_spec."
            )
        if self.lambda_prior is not None:
            raise ValueError(
                "matched_l1 does not accept "
                f"lambda_prior={self.lambda_prior!r}."
            )
        if self.matched_l1_lambda1 is None:
            raise ValueError("matched_l1 requires matched_l1_lambda1.")
        if not isinstance(self.matched_l1_lambda1, (int, float)) or isinstance(
            self.matched_l1_lambda1, bool
        ):
            raise ValueError(
                "matched_l1_lambda1 must be a finite positive number; "
                f"got {self.matched_l1_lambda1!r}."
            )
        value = float(self.matched_l1_lambda1)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "matched_l1_lambda1 must be finite and > 0; "
                f"got {value}."
            )
        if self.dagma_config.exclude_edges is not None:
            raise ValueError(
                "matched_l1 requires dagma_config.exclude_edges=None; "
                f"got {self.dagma_config.exclude_edges!r}."
            )

    def _validate_hard_exclusion(self) -> None:
        if self.confidence is not None:
            raise ValueError(
                "hard_exclusion does not accept "
                f"confidence={self.confidence!r}."
            )
        if self.corrupted_prior_spec is None:
            raise ValueError(
                "hard_exclusion requires corrupted_prior_spec."
            )
        cf = self.corrupted_prior_spec.corruption_fraction
        if not _is_close_to_grid(cf, CORRUPTION_GRID):
            raise ValueError(
                "hard_exclusion corrupted_prior_spec.corruption_fraction "
                f"must be in {CORRUPTION_GRID}; got {cf!r}."
            )
        if self.lambda_prior is not None:
            raise ValueError(
                "hard_exclusion does not accept "
                f"lambda_prior={self.lambda_prior!r}."
            )
        if self.matched_l1_lambda1 is not None:
            raise ValueError(
                "hard_exclusion does not accept "
                f"matched_l1_lambda1={self.matched_l1_lambda1!r}."
            )
        if self.dagma_config.exclude_edges is None:
            raise ValueError(
                "hard_exclusion requires dagma_config.exclude_edges to "
                "be set to the same edges as "
                "corrupted_prior_spec.forbidden_edges."
            )
        exclude_sorted = tuple(sorted(self.dagma_config.exclude_edges))
        forbidden_sorted = tuple(
            sorted(self.corrupted_prior_spec.forbidden_edges)
        )
        if exclude_sorted != forbidden_sorted:
            raise ValueError(
                "hard_exclusion requires "
                "tuple(sorted(dagma_config.exclude_edges)) == "
                "tuple(sorted(corrupted_prior_spec.forbidden_edges)); "
                "the two edge lists differ in length or contents "
                "(possible duplicate edges in exclude_edges). Got "
                f"exclude(len={len(exclude_sorted)})={exclude_sorted!r}, "
                f"forbidden(len={len(forbidden_sorted)})="
                f"{forbidden_sorted!r}."
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_main_study_config(
    *,
    method_family: str,
    seed_value: int,
    seed_population: str,
    dagma_config: DAGMAConfig,
    parent_heldout_run_hash_full: str,
    confidence: Optional[float] = None,
    corrupted_prior_spec: Optional[CorruptedPriorSpec] = None,
    matched_l1_lambda1: Optional[float] = None,
) -> MainStudyConfig:
    """Construct a :class:`MainStudyConfig` enforcing method-family invariants.

    The factory:

    - Injects ``lambda_prior = FROZEN_LAMBDA_PRIOR`` for
      ``soft_frobenius``. The signature does not expose
      ``lambda_prior``; the protocol value is the only acceptable
      input.
    - For ``hard_exclusion``, replaces the supplied ``dagma_config``
      with one whose ``exclude_edges`` is the lexicographically sorted
      tuple form of ``corrupted_prior_spec.forbidden_edges``. If the
      caller already set a non-``None`` ``dagma_config.exclude_edges``
      that disagrees with ``corrupted_prior_spec.forbidden_edges``,
      the factory raises ``ValueError`` rather than silently
      overwriting.
    - For ``prior_free``, rejects ``confidence``,
      ``corrupted_prior_spec``, and ``matched_l1_lambda1``.
    - For ``matched_l1``, requires ``matched_l1_lambda1`` and rejects
      ``confidence`` and ``corrupted_prior_spec``.
    - Returns the constructed ``MainStudyConfig``. Final invariants
      are enforced inside ``MainStudyConfig.__post_init__``.
    """
    if method_family not in METHOD_FAMILIES:
        raise ValueError(
            f"method_family must be one of {METHOD_FAMILIES}; "
            f"got {method_family!r}."
        )

    resolved_dagma_config = dagma_config
    resolved_lambda_prior: Optional[float] = None

    if method_family == "prior_free":
        if confidence is not None:
            raise ValueError(
                "prior_free does not accept confidence."
            )
        if corrupted_prior_spec is not None:
            raise ValueError(
                "prior_free does not accept corrupted_prior_spec."
            )
        if matched_l1_lambda1 is not None:
            raise ValueError(
                "prior_free does not accept matched_l1_lambda1."
            )

    elif method_family == "soft_frobenius":
        if corrupted_prior_spec is None:
            raise ValueError(
                "soft_frobenius requires corrupted_prior_spec."
            )
        if confidence is None:
            raise ValueError("soft_frobenius requires confidence.")
        if matched_l1_lambda1 is not None:
            raise ValueError(
                "soft_frobenius does not accept matched_l1_lambda1."
            )
        resolved_lambda_prior = FROZEN_LAMBDA_PRIOR

    elif method_family == "matched_l1":
        if confidence is not None:
            raise ValueError(
                "matched_l1 does not accept confidence."
            )
        if corrupted_prior_spec is not None:
            raise ValueError(
                "matched_l1 does not accept corrupted_prior_spec."
            )
        if matched_l1_lambda1 is None:
            raise ValueError(
                "matched_l1 requires matched_l1_lambda1."
            )

    elif method_family == "hard_exclusion":
        if corrupted_prior_spec is None:
            raise ValueError(
                "hard_exclusion requires corrupted_prior_spec."
            )
        if confidence is not None:
            raise ValueError(
                "hard_exclusion does not accept confidence."
            )
        if matched_l1_lambda1 is not None:
            raise ValueError(
                "hard_exclusion does not accept matched_l1_lambda1."
            )
        sorted_forbidden = tuple(
            sorted(corrupted_prior_spec.forbidden_edges)
        )
        if dagma_config.exclude_edges is not None:
            caller_sorted = tuple(sorted(dagma_config.exclude_edges))
            if caller_sorted != sorted_forbidden:
                raise ValueError(
                    "hard_exclusion: caller-supplied "
                    "dagma_config.exclude_edges does not match "
                    "corrupted_prior_spec.forbidden_edges under "
                    "tuple(sorted(...)) equality; the two edge lists "
                    "differ in length or contents (possible duplicate "
                    "edges in exclude_edges). Got "
                    f"caller(len={len(caller_sorted)})={caller_sorted!r}, "
                    f"forbidden(len={len(sorted_forbidden)})="
                    f"{sorted_forbidden!r}."
                )
            resolved_dagma_config = dataclasses.replace(
                dagma_config, exclude_edges=sorted_forbidden
            )
        else:
            resolved_dagma_config = dataclasses.replace(
                dagma_config, exclude_edges=sorted_forbidden
            )

    return MainStudyConfig(
        method_family=method_family,
        seed_value=seed_value,
        seed_population=seed_population,
        dagma_config=resolved_dagma_config,
        parent_heldout_run_hash_full=parent_heldout_run_hash_full,
        lambda_prior=resolved_lambda_prior,
        confidence=confidence,
        corrupted_prior_spec=corrupted_prior_spec,
        matched_l1_lambda1=matched_l1_lambda1,
    )


# ---------------------------------------------------------------------------
# Canonicalisation + configuration hash
# ---------------------------------------------------------------------------


def canonicalize_for_json(
    value: Any, field_name: Optional[str] = None
) -> Any:
    """Recursively convert ``value`` into a JSON-serialisable form.

    Conversions
    -----------
    - ``None``, ``bool``, ``int``, ``float``, ``str`` -> returned as is.
      ``True``/``False`` are not coerced to ``1``/``0``.
    - dataclass instances -> dict keyed by field name, recursing with
      the field name carried as ``field_name``.
    - ``dict`` -> dict with lexicographically sorted string keys. Non-
      string keys raise ``TypeError``.
    - ``set`` -> sorted list of canonicalised elements.
    - ``tuple``/``list`` -> list of canonicalised elements.
    - When ``field_name`` is one of ``forbidden_edges``,
      ``removed_clean_edges``, ``added_true_positive_edges``, or
      ``exclude_edges`` and the value is not ``None``, the value is
      treated as a collection of ``(i, j)`` edge pairs, sorted
      lexicographically, and converted to ``[[i, j], ...]``.

    Any other type raises ``TypeError``.
    """
    # Special-case edge collections by field name. Applies to
    # tuples/lists/sets when the recursive walk arrives here under a
    # known edge-field name. ``None`` falls through to the None branch.
    if value is not None and field_name in _EDGE_FIELD_NAMES:
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise TypeError(
                f"edge field {field_name!r} must be a list/tuple/set; "
                f"got {type(value).__name__}."
            )
        pairs: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError(
                    f"edge in {field_name!r} must be a length-2 "
                    f"list/tuple; got {item!r}."
                )
            i_val, j_val = item
            if isinstance(i_val, bool) or isinstance(j_val, bool):
                raise TypeError(
                    f"edge indices in {field_name!r} must be ints, "
                    f"not bools; got {item!r}."
                )
            if not isinstance(i_val, int) or not isinstance(j_val, int):
                raise TypeError(
                    f"edge indices in {field_name!r} must be ints; "
                    f"got {item!r}."
                )
            pairs.append((int(i_val), int(j_val)))
        pairs.sort()
        return [[i, j] for (i, j) in pairs]

    if value is None:
        return None
    if isinstance(value, bool):
        # Keep bool as bool so json.dumps writes "true"/"false".
        return value
    if isinstance(value, (int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out_dc: dict[str, Any] = {}
        for f in dataclasses.fields(value):
            out_dc[f.name] = canonicalize_for_json(
                getattr(value, f.name), field_name=f.name
            )
        return out_dc
    if isinstance(value, dict):
        out_dict: dict[str, Any] = {}
        sorted_keys = sorted(value.keys())
        for k in sorted_keys:
            if not isinstance(k, str):
                raise TypeError(
                    "dict keys must be strings; got "
                    f"{type(k).__name__}."
                )
            out_dict[k] = canonicalize_for_json(value[k])
        return out_dict
    if isinstance(value, (set, frozenset)):
        return sorted(canonicalize_for_json(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [canonicalize_for_json(v) for v in value]
    raise TypeError(
        "Cannot canonicalize value of type "
        f"{type(value).__name__}."
    )


def compute_configuration_hash(config: MainStudyConfig) -> str:
    """Compute the canonical SHA-256 configuration hash.

    The hash covers the experimental condition only:
    ``method_family``, ``seed_value``, ``seed_population``,
    ``dagma_config``, ``lambda_prior``, ``confidence``,
    ``corrupted_prior_spec``, and ``matched_l1_lambda1``.
    ``parent_heldout_run_hash_full`` and ``schema_version`` are
    intentionally excluded so that two configurations differing only
    in parent provenance or schema version share the same condition
    hash.

    Returns a 64-character lowercase hexadecimal digest.
    """
    if not isinstance(config, MainStudyConfig):
        raise TypeError(
            "config must be a MainStudyConfig; got "
            f"{type(config).__name__}."
        )
    condition: dict[str, Any] = {
        "method_family": canonicalize_for_json(config.method_family),
        "seed_value": canonicalize_for_json(config.seed_value),
        "seed_population": canonicalize_for_json(
            config.seed_population
        ),
        "dagma_config": canonicalize_for_json(config.dagma_config),
        "lambda_prior": canonicalize_for_json(config.lambda_prior),
        "confidence": canonicalize_for_json(config.confidence),
        "corrupted_prior_spec": canonicalize_for_json(
            config.corrupted_prior_spec
        ),
        "matched_l1_lambda1": canonicalize_for_json(
            config.matched_l1_lambda1
        ),
    }
    payload = json.dumps(
        condition, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configuration_hash_prefix(config: MainStudyConfig) -> str:
    """Return the first 12 hex characters of the configuration hash."""
    return compute_configuration_hash(config)[:12]


def make_run_id(config: MainStudyConfig) -> str:
    """Return the run-id string for ``config``.

    Format: ``"{method_family}__{seed_population}__seed{seed_value}__cfg{hash12}"``.
    """
    return (
        f"{config.method_family}__{config.seed_population}__seed"
        f"{int(config.seed_value)}__cfg"
        f"{configuration_hash_prefix(config)}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_close_to_grid(
    value: Any, grid: tuple[float, ...]
) -> bool:
    """True when ``value`` matches a ``grid`` entry within abs_tol=1e-12."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return any(
        math.isclose(v, g, abs_tol=1e-12, rel_tol=0.0) for g in grid
    )


__all__ = [
    "SCHEMA_VERSION",
    "METHOD_FAMILIES",
    "SEED_POPULATIONS",
    "CALIBRATION_SEEDS",
    "EVALUATION_SEEDS",
    "FROZEN_LAMBDA_PRIOR",
    "PROTOCOL_DAGMA_LAMBDA1",
    "PROTOCOL_DAGMA_WARM_ITER",
    "PROTOCOL_DAGMA_MAX_ITER",
    "build_protocol_dagma_config",
    "CONFIDENCE_GRID",
    "PRIOR_K",
    "CORRUPTION_GRID",
    "PriorSpec",
    "CorruptedPriorSpec",
    "MainStudyConfig",
    "make_main_study_config",
    "canonicalize_for_json",
    "compute_configuration_hash",
    "configuration_hash_prefix",
    "make_run_id",
]
