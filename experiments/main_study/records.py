"""Core post-run record schema for the main-study pipeline.

Defines :class:`MainStudyRunRecord` (the immutable per-run record
emitted after fitting and scoring a single experimental condition)
together with the status-value tuples used for validation. The
record is frozen, keyword-only, and validates every field at
construction time.

Serialisation helpers (``record_to_dict`` / ``record_from_dict`` /
``record_to_json`` / ``record_from_json``) and the convenience
``make_failure_record`` factory are deliberately not implemented
here; they will be added in a follow-up commit.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from datetime import datetime
from typing import Any, Optional, get_args

import numpy as np

from experiments.main_study.paths import validate_relative_posix_path
from experiments.main_study.priors import CorruptedPriorSpec
from experiments.main_study.schema import (
    SCHEMA_VERSION,
    MainStudyConfig,
    canonicalize_for_json,
    compute_configuration_hash,
    configuration_hash_prefix,
    make_run_id,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig
from symbolic_priors_cd.wrappers.status import GraphStatus, SamplerStatus


# ---------------------------------------------------------------------------
# Status taxonomy
# ---------------------------------------------------------------------------


FIT_STATUSES: tuple[str, ...] = (
    "success",
    "model_fit_failure",
    "infrastructure_failure_during_fit",
)


METRIC_STATUSES: tuple[str, ...] = (
    "computed",
    "unavailable_graph_invalid",
    "unavailable_sampler_failure",
    "unavailable_dependency_missing",
    "not_computed_due_to_fit_failure",
)


FAILURE_KINDS: tuple[Optional[str], ...] = (
    None,
    "non_convergence",
    "invalid_graph",
    "sampler_unavailable",
    "metric_unavailable",
    "infrastructure",
)


# Graph and sampler status sets are derived from the wrapper module's
# Literal type aliases rather than redefined here, so any future
# extension in the wrapper layer flows through automatically.
GRAPH_STATUS_VALUES: tuple[str, ...] = tuple(get_args(GraphStatus))
SAMPLER_STATUS_VALUES: tuple[str, ...] = tuple(get_args(SamplerStatus))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


_ARTEFACT_PATH_FIELDS: tuple[str, ...] = (
    "continuous_w_path",
    "thresholded_adjacency_path",
    "confidence_mask_path",
    "interventions_mmd_path",
    "prior_edge_set_clean_path",
    "prior_edge_set_corrupted_path",
    "per_edge_labels_path",
    "true_adjacency_path",
)


_SUCCESS_REQUIRED_PATHS: tuple[str, ...] = (
    "continuous_w_path",
    "thresholded_adjacency_path",
    "true_adjacency_path",
)


_PRIOR_BACKED_PATHS: tuple[str, ...] = (
    "prior_edge_set_clean_path",
    "prior_edge_set_corrupted_path",
    "per_edge_labels_path",
)


def _is_plain_int(value: Any) -> bool:
    """True when ``value`` is an ``int`` and not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_iso8601_with_timezone(value: str) -> None:
    """Validate ``value`` is a timezone-aware ISO-8601 string.

    Accepts the ``Z`` suffix (UTC) and explicit numeric offsets such
    as ``+00:00`` or ``-05:00``. Rejects naive timestamps and
    anything that does not parse via ``datetime.fromisoformat``.
    """
    if not isinstance(value, str) or value == "":
        raise ValueError(
            "generated_at_utc must be a non-empty string; "
            f"got {value!r}."
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "generated_at_utc must be a valid ISO-8601 string; "
            f"got {value!r}: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(
            "generated_at_utc must include a timezone "
            "(e.g. 'Z' or '+00:00'); got naive timestamp "
            f"{value!r}."
        )


# ---------------------------------------------------------------------------
# Post-run record dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class MainStudyRunRecord:
    """Immutable post-run record for a single main-study condition.

    The record carries the full :class:`MainStudyConfig`, the
    re-derivable configuration hash and run identifier, the
    structured status taxonomy, the three primary metrics (or
    ``None`` when not computed), the runtime breakdown, the
    JSON-canonicalisable wrapper diagnostics, and the relative POSIX
    paths to per-run artefacts.

    Validation runs in :meth:`__post_init__`; constructing a record
    is the canonical correctness gate.
    """

    schema_version: int
    config: MainStudyConfig
    configuration_hash_full: str
    configuration_hash_prefix: str
    run_id: str
    n_nodes: int

    fit_status: str
    graph_status: Optional[str]
    sampler_status: Optional[str]
    metric_status: str
    failure_kind: Optional[str]
    failure_message: str

    runtime_seconds: float
    fit_runtime_seconds: float

    wrapper_diagnostics: dict[str, Any]

    parent_heldout_run_hash_full: str
    generated_at_utc: str

    sid: Optional[float] = None
    shd: Optional[float] = None
    mmd: Optional[float] = None

    metric_runtime_seconds: Optional[float] = None

    continuous_w_path: Optional[str] = None
    thresholded_adjacency_path: Optional[str] = None
    confidence_mask_path: Optional[str] = None
    interventions_mmd_path: Optional[str] = None
    prior_edge_set_clean_path: Optional[str] = None
    prior_edge_set_corrupted_path: Optional[str] = None
    per_edge_labels_path: Optional[str] = None
    true_adjacency_path: Optional[str] = None

    code_version: Optional[str] = None

    def __post_init__(self) -> None:
        self._validate_schema_and_config_identity()
        self._validate_status_values()
        self._validate_n_nodes()
        self._validate_timings()
        _validate_iso8601_with_timezone(self.generated_at_utc)
        self._validate_code_version()
        self._validate_and_copy_wrapper_diagnostics()
        self._validate_failure_and_metric_semantics()
        self._validate_artefact_paths()

    # -- Identity ----------------------------------------------------

    def _validate_schema_and_config_identity(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{SCHEMA_VERSION}; got {self.schema_version!r}."
            )
        if not isinstance(self.config, MainStudyConfig):
            raise TypeError(
                "config must be a MainStudyConfig; got "
                f"{type(self.config).__name__}."
            )

        if not isinstance(self.configuration_hash_full, str):
            raise ValueError(
                "configuration_hash_full must be a string; got "
                f"{type(self.configuration_hash_full).__name__}."
            )
        if not _HEX_64_RE.fullmatch(self.configuration_hash_full):
            raise ValueError(
                "configuration_hash_full must be 64 lowercase hex "
                f"characters; got {self.configuration_hash_full!r}."
            )

        recomputed_full = compute_configuration_hash(self.config)
        if self.configuration_hash_full != recomputed_full:
            raise ValueError(
                "configuration_hash_full does not match "
                "compute_configuration_hash(config). "
                f"got {self.configuration_hash_full!r}, "
                f"expected {recomputed_full!r}."
            )

        recomputed_prefix = configuration_hash_prefix(self.config)
        if self.configuration_hash_prefix != recomputed_prefix:
            raise ValueError(
                "configuration_hash_prefix does not match "
                "configuration_hash_prefix(config). "
                f"got {self.configuration_hash_prefix!r}, "
                f"expected {recomputed_prefix!r}."
            )
        if self.configuration_hash_prefix != self.configuration_hash_full[:12]:
            raise ValueError(
                "configuration_hash_prefix must equal the first 12 "
                "characters of configuration_hash_full. "
                f"got prefix {self.configuration_hash_prefix!r}, "
                f"full {self.configuration_hash_full!r}."
            )

        expected_run_id = make_run_id(self.config)
        if self.run_id != expected_run_id:
            raise ValueError(
                "run_id does not match make_run_id(config). "
                f"got {self.run_id!r}, expected {expected_run_id!r}."
            )

        if (
            self.parent_heldout_run_hash_full
            != self.config.parent_heldout_run_hash_full
        ):
            raise ValueError(
                "parent_heldout_run_hash_full does not match "
                "config.parent_heldout_run_hash_full. "
                f"got {self.parent_heldout_run_hash_full!r}, "
                f"expected {self.config.parent_heldout_run_hash_full!r}."
            )

    # -- Status / enums ---------------------------------------------

    def _validate_status_values(self) -> None:
        if self.fit_status not in FIT_STATUSES:
            raise ValueError(
                f"fit_status must be one of {FIT_STATUSES}; "
                f"got {self.fit_status!r}."
            )
        if self.metric_status not in METRIC_STATUSES:
            raise ValueError(
                f"metric_status must be one of {METRIC_STATUSES}; "
                f"got {self.metric_status!r}."
            )
        if self.failure_kind not in FAILURE_KINDS:
            raise ValueError(
                f"failure_kind must be one of {FAILURE_KINDS}; "
                f"got {self.failure_kind!r}."
            )
        if self.graph_status is not None and (
            self.graph_status not in GRAPH_STATUS_VALUES
        ):
            raise ValueError(
                "graph_status must be None or in "
                f"{GRAPH_STATUS_VALUES}; got {self.graph_status!r}."
            )
        if self.sampler_status is not None and (
            self.sampler_status not in SAMPLER_STATUS_VALUES
        ):
            raise ValueError(
                "sampler_status must be None or in "
                f"{SAMPLER_STATUS_VALUES}; got {self.sampler_status!r}."
            )

    # -- Dimensions / timings --------------------------------------

    def _validate_n_nodes(self) -> None:
        if type(self.n_nodes) is not int:
            raise ValueError(
                "n_nodes must be a plain int (no bool, no float); "
                f"got {type(self.n_nodes).__name__}: {self.n_nodes!r}."
            )
        if self.n_nodes <= 0:
            raise ValueError(
                f"n_nodes must be positive; got {self.n_nodes}."
            )

    def _validate_timings(self) -> None:
        for label, value in (
            ("runtime_seconds", self.runtime_seconds),
            ("fit_runtime_seconds", self.fit_runtime_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{label} must be a finite non-negative number; "
                    f"got {value!r}."
                )
            v = float(value)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"{label} must be finite and >= 0; got {value!r}."
                )
        if self.metric_runtime_seconds is not None:
            value = self.metric_runtime_seconds
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    "metric_runtime_seconds must be None or a finite "
                    f"non-negative number; got {value!r}."
                )
            v = float(value)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    "metric_runtime_seconds must be None or finite and "
                    f">= 0; got {value!r}."
                )

    def _validate_code_version(self) -> None:
        if self.code_version is None:
            return
        if not isinstance(self.code_version, str) or self.code_version == "":
            raise ValueError(
                "code_version must be None or a non-empty string; "
                f"got {self.code_version!r}."
            )

    # -- wrapper_diagnostics ----------------------------------------

    def _validate_and_copy_wrapper_diagnostics(self) -> None:
        if type(self.wrapper_diagnostics) is not dict:
            raise TypeError(
                "wrapper_diagnostics must be exactly dict; got "
                f"{type(self.wrapper_diagnostics).__name__}."
            )
        # Verify canonicalisability before deep-copying; raises
        # TypeError on unsupported nested types via canonicalize_for_json.
        canonicalize_for_json(self.wrapper_diagnostics)
        object.__setattr__(
            self,
            "wrapper_diagnostics",
            copy.deepcopy(self.wrapper_diagnostics),
        )

    # -- Failure / metrics semantics --------------------------------

    def _validate_failure_and_metric_semantics(self) -> None:
        if not isinstance(self.failure_message, str):
            raise ValueError(
                "failure_message must be a string; got "
                f"{type(self.failure_message).__name__}."
            )

        if self.fit_status == "success":
            if self.failure_kind is not None:
                raise ValueError(
                    "success record requires failure_kind=None; "
                    f"got {self.failure_kind!r}."
                )
            if self.failure_message != "":
                raise ValueError(
                    "success record requires failure_message=''; "
                    f"got {self.failure_message!r}."
                )
            if self.graph_status is None:
                raise ValueError(
                    "success record requires graph_status; got None."
                )
            if self.sampler_status is None:
                raise ValueError(
                    "success record requires sampler_status; got None."
                )
        else:
            if self.metric_status == "computed":
                raise ValueError(
                    "non-success record must not have "
                    f"metric_status='computed'; got fit_status="
                    f"{self.fit_status!r}, metric_status="
                    f"{self.metric_status!r}."
                )
            if self.failure_kind is None and self.failure_message == "":
                raise ValueError(
                    "non-success record requires either a "
                    "failure_kind or a non-empty failure_message."
                )

        if self.metric_status == "computed":
            if self.fit_status != "success":
                raise ValueError(
                    "metric_status='computed' requires "
                    f"fit_status='success'; got {self.fit_status!r}."
                )
            for label, value in (
                ("sid", self.sid),
                ("shd", self.shd),
                ("mmd", self.mmd),
            ):
                if value is None or not isinstance(
                    value, (int, float)
                ) or isinstance(value, bool):
                    raise ValueError(
                        f"metric_status='computed' requires finite "
                        f"non-negative {label}; got {value!r}."
                    )
                v = float(value)
                if not math.isfinite(v) or v < 0.0:
                    raise ValueError(
                        f"metric_status='computed' requires finite "
                        f"non-negative {label}; got {value!r}."
                    )
            if self.metric_runtime_seconds is None:
                raise ValueError(
                    "metric_status='computed' requires "
                    "metric_runtime_seconds (finite, >= 0)."
                )
            if self.interventions_mmd_path is None:
                raise ValueError(
                    "metric_status='computed' requires "
                    "interventions_mmd_path to be set."
                )
        else:
            for label, value in (
                ("sid", self.sid),
                ("shd", self.shd),
                ("mmd", self.mmd),
            ):
                if value is not None:
                    raise ValueError(
                        f"metric_status={self.metric_status!r} "
                        f"requires {label}=None; got {value!r}."
                    )

    # -- Artefact paths ---------------------------------------------

    def _validate_artefact_paths(self) -> None:
        # Validate format of every non-None path.
        for field_name in _ARTEFACT_PATH_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                continue
            try:
                validate_relative_posix_path(value)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}={value!r} is not a valid relative "
                    f"POSIX path: {exc}"
                ) from exc

        family = self.config.method_family

        # Success-only required paths.
        if self.fit_status == "success":
            for required in _SUCCESS_REQUIRED_PATHS:
                if getattr(self, required) is None:
                    raise ValueError(
                        f"success record requires {required}."
                    )

        # Prior-backed paths required for soft_frobenius and
        # hard_exclusion success records.
        if (
            self.fit_status == "success"
            and family in ("soft_frobenius", "hard_exclusion")
        ):
            for required in _PRIOR_BACKED_PATHS:
                if getattr(self, required) is None:
                    raise ValueError(
                        f"{family} success record requires {required}."
                    )

        # confidence_mask_path required for soft_frobenius success.
        if (
            self.fit_status == "success"
            and family == "soft_frobenius"
        ):
            if self.confidence_mask_path is None:
                raise ValueError(
                    "soft_frobenius success record requires "
                    "confidence_mask_path."
                )

        # prior_free and matched_l1: confidence_mask + prior paths
        # must always be None.
        if family in ("prior_free", "matched_l1"):
            for not_allowed in (
                "confidence_mask_path",
                "prior_edge_set_clean_path",
                "prior_edge_set_corrupted_path",
                "per_edge_labels_path",
            ):
                if getattr(self, not_allowed) is not None:
                    raise ValueError(
                        f"{family} record must have "
                        f"{not_allowed}=None; got "
                        f"{getattr(self, not_allowed)!r}."
                    )

        # hard_exclusion: confidence_mask_path must always be None.
        if family == "hard_exclusion":
            if self.confidence_mask_path is not None:
                raise ValueError(
                    "hard_exclusion record must have "
                    f"confidence_mask_path=None; got "
                    f"{self.confidence_mask_path!r}."
                )


# ---------------------------------------------------------------------------
# Diagnostics canonicalisation
# ---------------------------------------------------------------------------


def _convert_diag(value: Any, path: str) -> Any:
    """Recursively convert one diagnostic value to JSON-friendly form.

    The ``path`` string is included in any error message so callers
    can locate the offending position inside the original
    diagnostics structure.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"diagnostics dict key at {path} must be a string; "
                    f"got {type(key).__name__}: {key!r}."
                )
            out[key] = _convert_diag(sub, f"{path}[{key!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [
            _convert_diag(item, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    raise TypeError(
        f"diagnostics value at {path} has unsupported type "
        f"{type(value).__name__}: {value!r}."
    )


def diagnostics_to_canonical(diag: dict) -> dict:
    """Return a JSON-friendly canonical copy of ``diag``.

    Converts NumPy arrays via ``.tolist()`` and NumPy scalars via
    ``.item()``; recurses into dicts, lists, and tuples; preserves
    primitives unchanged. Rejects non-string dict keys and
    unsupported value types with a path-aware ``TypeError`` message.
    The input dict is never mutated; a new dict (and new nested
    containers) are constructed. The returned value is passed
    through :func:`canonicalize_for_json` as a final safety check.
    """
    if type(diag) is not dict:
        raise TypeError(
            "diagnostics_to_canonical requires a dict at the top "
            f"level; got {type(diag).__name__}."
        )
    converted = _convert_diag(diag, "diag")
    # Final safety net: the canonicaliser raises TypeError on any
    # nested value that slipped past _convert_diag.
    canonicalize_for_json(converted)
    return converted


# ---------------------------------------------------------------------------
# Metric-status derivation for failure/noncomputed records
# ---------------------------------------------------------------------------


def derive_metric_status_for_failure(
    *,
    fit_status: str,
    graph_status: Optional[str],
    sampler_status: Optional[str],
) -> str:
    """Derive the metric_status for a fit failure or downstream noncomputed case.

    Precedence (first matching branch returns):

    1. ``fit_status != "success"`` -> ``"not_computed_due_to_fit_failure"``.
    2. ``graph_status not in (None, "valid_dag")`` ->
       ``"unavailable_graph_invalid"``.
    3. ``sampler_status not in (None, "available")`` ->
       ``"unavailable_sampler_failure"``.

    Any combination not matching the above is treated as a
    success-with-available-sampler condition and rejected with
    ``ValueError``; this helper is for failure/noncomputed paths only.

    Validates each input against the allowed status values.
    """
    if fit_status not in FIT_STATUSES:
        raise ValueError(
            f"derive_metric_status_for_failure: fit_status must be one "
            f"of {FIT_STATUSES}; got {fit_status!r}."
        )
    if graph_status is not None and graph_status not in GRAPH_STATUS_VALUES:
        raise ValueError(
            "derive_metric_status_for_failure: graph_status must be "
            f"None or in {GRAPH_STATUS_VALUES}; got {graph_status!r}."
        )
    if (
        sampler_status is not None
        and sampler_status not in SAMPLER_STATUS_VALUES
    ):
        raise ValueError(
            "derive_metric_status_for_failure: sampler_status must be "
            f"None or in {SAMPLER_STATUS_VALUES}; got "
            f"{sampler_status!r}."
        )

    if fit_status != "success":
        return "not_computed_due_to_fit_failure"
    if graph_status is not None and graph_status != "valid_dag":
        return "unavailable_graph_invalid"
    if sampler_status is not None and sampler_status != "available":
        return "unavailable_sampler_failure"
    raise ValueError(
        "derive_metric_status_for_failure requires a fit failure or a "
        "downstream noncomputed condition; got fit_status="
        f"{fit_status!r}, graph_status={graph_status!r}, "
        f"sampler_status={sampler_status!r} (these describe a "
        "success-with-available-sampler state)."
    )


# ---------------------------------------------------------------------------
# Failure-record factory
# ---------------------------------------------------------------------------


def make_failure_record(
    *,
    config: MainStudyConfig,
    n_nodes: int,
    fit_status: str,
    failure_kind: Optional[str],
    failure_message: str,
    runtime_seconds: float,
    fit_runtime_seconds: float,
    wrapper_diagnostics: dict,
    generated_at_utc: str,
    metric_status: Optional[str] = None,
    graph_status: Optional[str] = None,
    sampler_status: Optional[str] = None,
    code_version: Optional[str] = None,
    continuous_w_path: Optional[str] = None,
    thresholded_adjacency_path: Optional[str] = None,
    confidence_mask_path: Optional[str] = None,
    interventions_mmd_path: Optional[str] = None,
    prior_edge_set_clean_path: Optional[str] = None,
    prior_edge_set_corrupted_path: Optional[str] = None,
    per_edge_labels_path: Optional[str] = None,
    true_adjacency_path: Optional[str] = None,
) -> "MainStudyRunRecord":
    """Construct a :class:`MainStudyRunRecord` for a fit-failure or
    downstream noncomputed case.

    Sets ``sid``, ``shd``, ``mmd``, and ``metric_runtime_seconds`` to
    ``None`` unconditionally. Derives ``configuration_hash_full``,
    ``configuration_hash_prefix``, ``run_id``, and
    ``parent_heldout_run_hash_full`` from ``config``. When
    ``metric_status`` is ``None``, it is derived via
    :func:`derive_metric_status_for_failure`; if derivation rejects
    the inputs, the error is re-raised with an explanatory message.
    ``wrapper_diagnostics`` is canonicalised via
    :func:`diagnostics_to_canonical` before construction; the
    resulting record validation runs the usual rules.

    The factory never infers or fabricates ``graph_status`` or
    ``sampler_status``: callers supply whatever the wrapper reported,
    including ``None``.
    """
    if metric_status is None:
        try:
            metric_status = derive_metric_status_for_failure(
                fit_status=fit_status,
                graph_status=graph_status,
                sampler_status=sampler_status,
            )
        except ValueError as exc:
            raise ValueError(
                "make_failure_record requires a fit failure or "
                "downstream noncomputed condition when metric_status "
                "is not supplied. The derivation helper rejected the "
                f"inputs: {exc}"
            ) from exc

    canonical_diag = diagnostics_to_canonical(wrapper_diagnostics)

    return MainStudyRunRecord(
        schema_version=SCHEMA_VERSION,
        config=config,
        configuration_hash_full=compute_configuration_hash(config),
        configuration_hash_prefix=configuration_hash_prefix(config),
        run_id=make_run_id(config),
        n_nodes=n_nodes,
        fit_status=fit_status,
        graph_status=graph_status,
        sampler_status=sampler_status,
        metric_status=metric_status,
        failure_kind=failure_kind,
        failure_message=failure_message,
        sid=None,
        shd=None,
        mmd=None,
        runtime_seconds=runtime_seconds,
        fit_runtime_seconds=fit_runtime_seconds,
        metric_runtime_seconds=None,
        wrapper_diagnostics=canonical_diag,
        continuous_w_path=continuous_w_path,
        thresholded_adjacency_path=thresholded_adjacency_path,
        confidence_mask_path=confidence_mask_path,
        interventions_mmd_path=interventions_mmd_path,
        prior_edge_set_clean_path=prior_edge_set_clean_path,
        prior_edge_set_corrupted_path=prior_edge_set_corrupted_path,
        per_edge_labels_path=per_edge_labels_path,
        true_adjacency_path=true_adjacency_path,
        parent_heldout_run_hash_full=config.parent_heldout_run_hash_full,
        generated_at_utc=generated_at_utc,
        code_version=code_version,
    )


# ---------------------------------------------------------------------------
# Serialisation / deserialisation
# ---------------------------------------------------------------------------


# DAGMAConfig fields whose runtime type is a plain ``tuple`` of
# primitives. Reconstruction converts JSON list -> tuple.
_DAGMA_CONFIG_TUPLE_FIELDS: tuple[str, ...] = ("s",)


# DAGMAConfig fields whose runtime type is ``tuple[tuple[int, int], ...]``
# or ``Optional[...]`` thereof. Reconstruction converts JSON
# ``[[i, j], ...]`` back to ``((i, j), ...)``.
_DAGMA_CONFIG_EDGE_FIELDS: tuple[str, ...] = ("exclude_edges",)


# CorruptedPriorSpec fields whose runtime type is
# ``tuple[tuple[int, int], ...]``. Reconstruction converts JSON
# ``[[i, j], ...]`` back to ``((i, j), ...)``.
_CORRUPTED_PRIOR_EDGE_FIELDS: tuple[str, ...] = (
    "forbidden_edges",
    "removed_clean_edges",
    "added_true_positive_edges",
)


def _check_keys_strict(
    d: object, dataclass_type: type, label: str
) -> None:
    """Strict key check for a nested-dict reconstruction.

    Raises ``ValueError`` when ``d`` is not a dict, when it contains
    keys not declared on ``dataclass_type``, or when it omits any
    declared field. Persisted records use explicit-null storage for
    every optional field, so missing keys are rejected even when the
    target dataclass would otherwise have supplied a default.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"{label} must be a dict; got {type(d).__name__}."
        )
    expected = {f.name for f in dataclasses.fields(dataclass_type)}
    actual = set(d.keys())
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(
            f"unknown {label} fields: {sorted(unknown)} "
            f"(allowed: {sorted(expected)})."
        )
    if missing:
        raise ValueError(
            f"missing {label} fields: {sorted(missing)}."
        )


def _normalise_edge_list(
    value: object, field_label: str
) -> tuple[tuple[int, int], ...]:
    """Convert JSON ``[[i, j], ...]`` to ``((i, j), ...)``.

    Raises ``ValueError`` if ``value`` is not a list of length-2
    sublists. Does not validate index values or types beyond
    structure; downstream dataclass validators apply value-level
    rules.
    """
    if not isinstance(value, list):
        raise ValueError(
            f"{field_label} must be a list of [i, j] pairs; got "
            f"{type(value).__name__}: {value!r}."
        )
    normalised: list[tuple[int, int]] = []
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                f"{field_label} edge must be a length-2 list; got "
                f"{entry!r}."
            )
        normalised.append((entry[0], entry[1]))
    return tuple(normalised)


def _reconstruct_dagma_config(d: object) -> DAGMAConfig:
    """Reconstruct :class:`DAGMAConfig` from a JSON-decoded dict."""
    _check_keys_strict(d, DAGMAConfig, "dagma_config")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(DAGMAConfig):
        value = d[f.name]
        if f.name in _DAGMA_CONFIG_TUPLE_FIELDS:
            if value is not None and isinstance(value, list):
                value = tuple(value)
        elif f.name in _DAGMA_CONFIG_EDGE_FIELDS:
            if value is not None:
                value = _normalise_edge_list(
                    value, f"dagma_config.{f.name}"
                )
        kwargs[f.name] = value
    try:
        return DAGMAConfig(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"dagma_config reconstruction failed: {exc}"
        ) from exc


def _reconstruct_corrupted_prior_spec(d: object) -> CorruptedPriorSpec:
    """Reconstruct :class:`CorruptedPriorSpec` from a JSON-decoded dict."""
    _check_keys_strict(d, CorruptedPriorSpec, "corrupted_prior_spec")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(CorruptedPriorSpec):
        value = d[f.name]
        if f.name in _CORRUPTED_PRIOR_EDGE_FIELDS:
            value = _normalise_edge_list(
                value, f"corrupted_prior_spec.{f.name}"
            )
        elif f.name == "edge_labels":
            if not isinstance(value, dict):
                raise ValueError(
                    "corrupted_prior_spec.edge_labels must be a dict; "
                    f"got {type(value).__name__}."
                )
            value = dict(value)
        kwargs[f.name] = value
    try:
        return CorruptedPriorSpec(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"corrupted_prior_spec reconstruction failed: {exc}"
        ) from exc


def _reconstruct_main_study_config(d: object) -> MainStudyConfig:
    """Reconstruct :class:`MainStudyConfig` from a JSON-decoded dict."""
    _check_keys_strict(d, MainStudyConfig, "config")
    dagma = _reconstruct_dagma_config(d["dagma_config"])
    if d["corrupted_prior_spec"] is None:
        corrupted: Optional[CorruptedPriorSpec] = None
    else:
        corrupted = _reconstruct_corrupted_prior_spec(
            d["corrupted_prior_spec"]
        )
    return MainStudyConfig(
        method_family=d["method_family"],
        seed_value=d["seed_value"],
        seed_population=d["seed_population"],
        dagma_config=dagma,
        parent_heldout_run_hash_full=d["parent_heldout_run_hash_full"],
        lambda_prior=d["lambda_prior"],
        confidence=d["confidence"],
        corrupted_prior_spec=corrupted,
        matched_l1_lambda1=d["matched_l1_lambda1"],
        schema_version=d["schema_version"],
    )


def record_to_dict(record: "MainStudyRunRecord") -> dict:
    """Serialise a :class:`MainStudyRunRecord` to a JSON-safe dict.

    Iterates the dataclass fields in declaration order and routes
    every value through :func:`canonicalize_for_json`. The returned
    dict contains exactly the record's declared field set; optional
    fields are present with ``None`` when unset.
    """
    if not isinstance(record, MainStudyRunRecord):
        raise TypeError(
            "record_to_dict requires a MainStudyRunRecord instance; "
            f"got {type(record).__name__}."
        )
    out: dict[str, Any] = {}
    for f in dataclasses.fields(MainStudyRunRecord):
        out[f.name] = canonicalize_for_json(
            getattr(record, f.name), field_name=f.name
        )
    expected = {f.name for f in dataclasses.fields(MainStudyRunRecord)}
    if set(out.keys()) != expected:
        raise RuntimeError(
            "record_to_dict produced a key set that does not match "
            f"MainStudyRunRecord fields. expected {sorted(expected)}, "
            f"got {sorted(out.keys())}."
        )
    return out


def record_from_dict(d: object) -> "MainStudyRunRecord":
    """Reconstruct a :class:`MainStudyRunRecord` from a JSON-decoded dict.

    The loader is strict: it requires the exact top-level and nested
    field sets to match the declared dataclasses, does not infer
    defaults for missing fields, and propagates all dataclass
    validators by running ``__post_init__`` during construction.
    """
    if not isinstance(d, dict):
        raise TypeError(
            "record_from_dict requires a dict; got "
            f"{type(d).__name__}."
        )
    if "schema_version" not in d:
        raise ValueError(
            "record dict is missing required field 'schema_version'."
        )
    sv = d["schema_version"]
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {sv!r}, expected "
            f"{SCHEMA_VERSION}."
        )
    _check_keys_strict(d, MainStudyRunRecord, "record")
    config = _reconstruct_main_study_config(d["config"])
    kwargs = {k: v for k, v in d.items() if k != "config"}
    kwargs["config"] = config
    return MainStudyRunRecord(**kwargs)


def record_to_json(record: "MainStudyRunRecord") -> str:
    """Serialise a :class:`MainStudyRunRecord` to a canonical JSON string.

    Uses ``json.dumps`` with ``sort_keys=True`` and tight separators
    so two calls on the same record produce byte-identical strings.
    """
    return json.dumps(
        record_to_dict(record),
        sort_keys=True,
        separators=(",", ":"),
    )


def record_from_json(s: str) -> "MainStudyRunRecord":
    """Parse a canonical JSON string into a :class:`MainStudyRunRecord`.

    ``json.JSONDecodeError`` propagates unchanged on invalid JSON.
    Non-object JSON values (arrays, strings, numbers, booleans, null)
    are rejected with ``ValueError``.
    """
    if not isinstance(s, str):
        raise TypeError(
            "record_from_json requires a string; got "
            f"{type(s).__name__}."
        )
    parsed = json.loads(s)
    if not isinstance(parsed, dict):
        raise ValueError(
            "record_from_json requires a JSON object at the top "
            f"level; got {type(parsed).__name__}."
        )
    return record_from_dict(parsed)


__all__ = [
    "FIT_STATUSES",
    "METRIC_STATUSES",
    "FAILURE_KINDS",
    "GRAPH_STATUS_VALUES",
    "SAMPLER_STATUS_VALUES",
    "MainStudyRunRecord",
    "diagnostics_to_canonical",
    "derive_metric_status_for_failure",
    "make_failure_record",
    "record_to_dict",
    "record_from_dict",
    "record_to_json",
    "record_from_json",
]
