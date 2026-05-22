"""Selected-configurations artefact: identity hash, schema, and writer.

This module produces the calibration-stage handoff artefact that
records the rank-1 configuration per model per condition, together
with the full five-candidate ranking per model per condition for
audit and visualisation. The artefact is consumed by the held-out
evaluation runner via an explicit path argument; no implicit
discovery is supported.

The module is intentionally write-only with respect to the calibration
workload: it does not invoke any model fit, does not implement the
within-model ranking rule, and does not emit a final DAGMA-vs-DCDI
winner. Forbidden field names that would record such a winner are
rejected by the schema validator at every nesting depth.

Public functions
----------------
- ``build_calibration_run_identity_payload``: build the canonical
  identity payload over which the calibration_run_hash is computed.
- ``compute_calibration_run_hash_full``: SHA-256 over the canonical
  identity payload.
- ``compute_calibration_run_hash12``: the 12-character lowercase hex
  prefix of the full hash.
- ``selected_configurations_path``: build the canonical on-disk path
  ``<results_root>/model_selection/calibration/<hash12>/selected_configurations.json``.
- ``validate_selected_configurations_artefact``: raise ``ValueError``
  if the artefact does not conform to the schema.
- ``write_selected_configurations``: validate, atomically write, and
  read-back-validate the artefact under no-overwrite-by-default
  semantics.

The module accepts mapping inputs in the public API and converts
them to plain ``dict`` internally so subclasses and ``MappingProxyType``
inputs behave identically.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SELECTED_CONFIGURATIONS_FILENAME = "selected_configurations.json"
CALIBRATION_RUN_DIRECTORY = "calibration"
MODEL_SELECTION_DIRECTORY = "model_selection"

SCHEMA_VERSION = 1
CALIBRATION_RUN_IDENTITY_ARTEFACT_TYPE = "calibration_run_identity"
SELECTED_CONFIGURATIONS_ARTEFACT_TYPE = "calibration_selected_configurations"
STAGE_LABEL = "calibration"
SEED_POPULATION_LABEL = "calibration"
DECISION_SCOPE = "within_model_configuration_selection"
SELECTED_CONFIGURATION_SEMANTICS = "rank_1_within_model_and_condition"

MODELS: tuple[str, ...] = ("dagma", "dcdi")
CONDITIONS: tuple[str, ...] = ("centred_only", "standardised")
CALIBRATION_SEEDS: tuple[int, ...] = (201, 202)
CANDIDATES_PER_CONDITION_PER_MODEL = 5
HASH_PREFIX_LENGTH = 12
FULL_HASH_LENGTH = 64

_HEX_DIGITS = frozenset("0123456789abcdef")

_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "schema_version",
    "artefact_type",
    "decision_scope",
    "base_model_decision_made",
    "selected_configuration_semantics",
    "calibration_run_hash_prefix",
    "calibration_run_hash_full",
    "selection_rule_id",
    "selection_rule_ref",
    "seed_population",
    "calibration_seeds",
    "intervention_policy_ref",
    "fit_rng_policy_ref",
    "selections",
    "candidate_ranking",
    "generated_at_utc",
)
_OPTIONAL_TOP_LEVEL_FIELDS: tuple[str, ...] = ("code_version",)
_ALLOWED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    _REQUIRED_TOP_LEVEL_FIELDS + _OPTIONAL_TOP_LEVEL_FIELDS
)

_REQUIRED_SELECTION_FIELDS: tuple[str, ...] = (
    "selected_configuration_hash_prefix",
    "selected_configuration_hash_full",
    "selected_hyperparameters",
    "selected_rank",
    "selection_metrics",
    "source_run_ids",
)

_REQUIRED_CANDIDATE_FIELDS: tuple[str, ...] = (
    "rank",
    "configuration_hash_prefix",
    "configuration_hash_full",
    "hyperparameters",
    "aggregate_metrics",
    "per_seed_metrics",
    "source_run_ids",
    "n_calibration_records",
)

_REQUIRED_AGGREGATE_METRIC_FIELDS: tuple[str, ...] = (
    "mean_sid",
    "mean_mmd_primary",
    "mean_shd",
    "std_sid",
    "std_mmd_primary",
    "std_shd",
)

_REQUIRED_PER_SEED_FIELDS: tuple[str, ...] = (
    "seed_value",
    "shd",
    "sid",
    "mmd_primary",
    "graph_status",
    "sampler_status",
    "training_status",
    "runtime_seconds",
    "n_iterations",
    "threshold_metrics",
    "mmd_by_intervention",
    "bandwidth_summaries",
)

_REQUIRED_THRESHOLD_METRIC_FIELDS: tuple[str, ...] = (
    "threshold",
    "shd",
    "sid",
    "mmd_primary",
)

_REQUIRED_MMD_BY_INTERVENTION_FIELDS: tuple[str, ...] = (
    "intervention_target",
    "intervention_value",
    "mmd_primary",
)

# Field names that would record a final base-model winner. Such a
# decision is out of scope for the calibration handoff artefact and
# is rejected at every nesting depth by the validator.
_FORBIDDEN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    }
)


# ---------------------------------------------------------------------------
# Calibration-run identity payload and hash
# ---------------------------------------------------------------------------


def _validate_full_hash_string(value: object, where: str) -> None:
    """Raise ValueError if ``value`` is not a 64-char lowercase hex string."""
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a string; got "
            f"{type(value).__name__}"
        )
    if len(value) != FULL_HASH_LENGTH:
        raise ValueError(
            f"{where} must be a {FULL_HASH_LENGTH}-character "
            f"lowercase hex string; got length {len(value)}"
        )
    for ch in value:
        if ch not in _HEX_DIGITS:
            raise ValueError(
                f"{where} must contain only lowercase hex digits "
                f"0-9 and a-f; got character {ch!r}"
            )


def _validate_executable_candidate_identity(
    candidate: Mapping[str, Any], index: int
) -> tuple[str, str, int, str]:
    """Validate one executable candidate identity record.

    Returns the four sort-key components ``(model, condition,
    grid_point_order, configuration_hash_full)`` as a tuple. Raises
    ValueError on any missing or malformed field.
    """
    required_keys = (
        "model",
        "condition",
        "grid_point_order",
        "configuration_hash_full",
    )
    missing = [name for name in required_keys if name not in candidate]
    if missing:
        raise ValueError(
            f"executable candidate identity at index {index} is "
            f"missing required field(s): {missing}"
        )
    model = candidate["model"]
    condition = candidate["condition"]
    grid_point_order = candidate["grid_point_order"]
    config_hash = candidate["configuration_hash_full"]
    if model not in MODELS:
        raise ValueError(
            f"executable candidate identity at index {index} has "
            f"unknown model {model!r}; allowed values are {list(MODELS)}"
        )
    if condition not in CONDITIONS:
        raise ValueError(
            f"executable candidate identity at index {index} has "
            f"unknown condition {condition!r}; allowed values are "
            f"{list(CONDITIONS)}"
        )
    if isinstance(grid_point_order, bool) or not isinstance(
        grid_point_order, int
    ):
        raise ValueError(
            f"executable candidate identity at index {index} has a "
            f"non-int grid_point_order: got {grid_point_order!r}"
        )
    if grid_point_order < 0:
        raise ValueError(
            f"executable candidate identity at index {index} has a "
            f"negative grid_point_order: got {grid_point_order}"
        )
    _validate_full_hash_string(
        config_hash,
        f"executable candidate identity at index {index} "
        "configuration_hash_full",
    )
    return (str(model), str(condition), int(grid_point_order), str(config_hash))


def build_calibration_run_identity_payload(
    *,
    executable_candidate_identities: Sequence[Mapping[str, Any]],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
) -> dict[str, Any]:
    """Build the canonical identity payload for the calibration run.

    The payload is the input to ``compute_calibration_run_hash_full``.
    The 20 executable candidate hashes are sorted by ``(model,
    condition, grid_point_order, configuration_hash_full)`` so the
    output is independent of the caller's iteration order. Only the
    full 64-character hashes are stored in the payload; no prefix or
    candidate metadata leaks into the identity.

    Parameters
    ----------
    executable_candidate_identities : sequence of Mapping
        One record per executable candidate. Each record must carry
        ``model``, ``condition``, ``grid_point_order``, and
        ``configuration_hash_full``.
    selection_rule_id : str
        Stable identifier for the within-model ranking rule that the
        calibration runner will apply.
    selection_rule_ref : str
        Free-form text reference to the rule (typically a short
        sentence describing the rule for human readers).
    intervention_policy_ref : str
        Stable identifier for the eligible-nodes intervention-set
        policy used by calibration.
    fit_rng_policy_ref : str
        Stable identifier for the DCDI fit-RNG convention used by
        calibration.

    Returns
    -------
    dict
        The canonical identity payload, ready for canonical-JSON
        serialisation.

    Raises
    ------
    ValueError
        If any candidate record is malformed or if the candidate set
        is empty.
    """
    if not executable_candidate_identities:
        raise ValueError(
            "executable_candidate_identities must contain at least "
            "one record; got an empty sequence"
        )

    sort_keys: list[tuple[str, str, int, str]] = []
    for index, candidate in enumerate(executable_candidate_identities):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"executable candidate identity at index {index} "
                f"must be a Mapping; got {type(candidate).__name__}"
            )
        sort_keys.append(
            _validate_executable_candidate_identity(candidate, index)
        )

    sort_keys.sort()
    hashes_sorted = [hash_full for _model, _cond, _order, hash_full in sort_keys]

    if not isinstance(selection_rule_id, str):
        raise ValueError(
            "selection_rule_id must be a string; got "
            f"{type(selection_rule_id).__name__}"
        )
    if not isinstance(selection_rule_ref, str):
        raise ValueError(
            "selection_rule_ref must be a string; got "
            f"{type(selection_rule_ref).__name__}"
        )
    if not isinstance(intervention_policy_ref, str):
        raise ValueError(
            "intervention_policy_ref must be a string; got "
            f"{type(intervention_policy_ref).__name__}"
        )
    if not isinstance(fit_rng_policy_ref, str):
        raise ValueError(
            "fit_rng_policy_ref must be a string; got "
            f"{type(fit_rng_policy_ref).__name__}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artefact_type": CALIBRATION_RUN_IDENTITY_ARTEFACT_TYPE,
        "stage": STAGE_LABEL,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "executable_configuration_hashes_full": hashes_sorted,
        "selection_rule_id": selection_rule_id,
        "selection_rule_ref": selection_rule_ref,
        "intervention_policy_ref": intervention_policy_ref,
        "fit_rng_policy_ref": fit_rng_policy_ref,
    }


def compute_calibration_run_hash_full(
    *,
    executable_candidate_identities: Sequence[Mapping[str, Any]],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
) -> str:
    """Compute the SHA-256 hex digest of the canonical identity payload.

    The canonical JSON encoding uses ``sort_keys=True``,
    ``separators=(",", ":")``, and ``ensure_ascii=True``; the bytes
    are UTF-8. The result is a 64-character lowercase hex string.
    """
    payload = build_calibration_run_identity_payload(
        executable_candidate_identities=executable_candidate_identities,
        selection_rule_id=selection_rule_id,
        selection_rule_ref=selection_rule_ref,
        intervention_policy_ref=intervention_policy_ref,
        fit_rng_policy_ref=fit_rng_policy_ref,
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_calibration_run_hash12(
    *,
    executable_candidate_identities: Sequence[Mapping[str, Any]],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
) -> str:
    """Return the first 12 hex characters of the calibration_run_hash."""
    return compute_calibration_run_hash_full(
        executable_candidate_identities=executable_candidate_identities,
        selection_rule_id=selection_rule_id,
        selection_rule_ref=selection_rule_ref,
        intervention_policy_ref=intervention_policy_ref,
        fit_rng_policy_ref=fit_rng_policy_ref,
    )[:HASH_PREFIX_LENGTH]


# ---------------------------------------------------------------------------
# On-disk path
# ---------------------------------------------------------------------------


def selected_configurations_path(
    *,
    calibration_run_hash12: str,
    results_root: Path | str,
) -> Path:
    """Build the canonical on-disk path for selected_configurations.json.

    The path layout is

        <results_root>/model_selection/calibration/<calibration_run_hash12>/selected_configurations.json

    where ``calibration_run_hash12`` is the first 12 lowercase hex
    characters of the full calibration_run_hash.

    Parameters
    ----------
    calibration_run_hash12 : str
        12-character lowercase hex prefix.
    results_root : Path or str
        Root of the results tree (typically ``Path("results")``;
        tests pass ``tmp_path``).

    Returns
    -------
    Path
        The canonical artefact path. The path is not created on disk.

    Raises
    ------
    ValueError
        If ``calibration_run_hash12`` is not a 12-character lowercase
        hex string.
    """
    if (
        not isinstance(calibration_run_hash12, str)
        or len(calibration_run_hash12) != HASH_PREFIX_LENGTH
        or any(ch not in _HEX_DIGITS for ch in calibration_run_hash12)
    ):
        raise ValueError(
            "calibration_run_hash12 must be a "
            f"{HASH_PREFIX_LENGTH}-character lowercase hex string; "
            f"got {calibration_run_hash12!r}"
        )
    return (
        Path(results_root)
        / MODEL_SELECTION_DIRECTORY
        / CALIBRATION_RUN_DIRECTORY
        / calibration_run_hash12
        / SELECTED_CONFIGURATIONS_FILENAME
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _scan_for_forbidden_field_names(obj: Any, path: str) -> None:
    """Walk a JSON-like structure and raise on any forbidden key name.

    Forbidden names are exactly the strings that would record a
    final base-model winner. The scan inspects dict keys at every
    depth; list elements are recursed into but lists themselves do
    not carry keys.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in _FORBIDDEN_FIELD_NAMES:
                raise ValueError(
                    f"forbidden field name {key!r} found at {path}; "
                    "the artefact must not record a final "
                    "DAGMA-vs-DCDI winner. The calibration handoff "
                    "selects one configuration per model per "
                    "condition; the base-model decision is made by "
                    "a separate step."
                )
            _scan_for_forbidden_field_names(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _scan_for_forbidden_field_names(item, f"{path}[{index}]")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    """Raise if ``value`` is not a Mapping; return a plain dict copy."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{path} must be a mapping; got {type(value).__name__}"
        )
    return dict(value)


def _require_list(value: Any, path: str) -> list[Any]:
    """Raise if ``value`` is not a list."""
    if not isinstance(value, list):
        raise ValueError(
            f"{path} must be a JSON array (list); got "
            f"{type(value).__name__}"
        )
    return value


def _require_fields_exact(
    mapping: dict[str, Any],
    *,
    required: Sequence[str],
    path: str,
) -> None:
    """Require ``mapping`` to contain every field in ``required``."""
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError(
            f"{path} is missing required field(s): {missing}"
        )


def _validate_threshold_metrics(
    threshold_metrics: Any, *, path: str
) -> None:
    """Validate a per-seed ``threshold_metrics`` list."""
    items = _require_list(threshold_metrics, path)
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        record = _require_mapping(item, item_path)
        _require_fields_exact(
            record,
            required=_REQUIRED_THRESHOLD_METRIC_FIELDS,
            path=item_path,
        )


def _validate_mmd_by_intervention(
    mmd_by_intervention: Any, *, path: str
) -> None:
    """Validate a per-seed ``mmd_by_intervention`` list."""
    items = _require_list(mmd_by_intervention, path)
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        record = _require_mapping(item, item_path)
        _require_fields_exact(
            record,
            required=_REQUIRED_MMD_BY_INTERVENTION_FIELDS,
            path=item_path,
        )


def _validate_per_seed_record(
    per_seed: Any, *, path: str
) -> None:
    """Validate one per-seed metrics record."""
    record = _require_mapping(per_seed, path)
    _require_fields_exact(
        record, required=_REQUIRED_PER_SEED_FIELDS, path=path
    )
    if "threshold_metrics" not in record:
        # _require_fields_exact already raised when missing.
        return
    _validate_threshold_metrics(
        record["threshold_metrics"],
        path=f"{path}.threshold_metrics",
    )
    _validate_mmd_by_intervention(
        record["mmd_by_intervention"],
        path=f"{path}.mmd_by_intervention",
    )


def _validate_aggregate_metrics(
    aggregate_metrics: Any, *, path: str
) -> None:
    """Validate an ``aggregate_metrics`` mapping."""
    mapping = _require_mapping(aggregate_metrics, path)
    _require_fields_exact(
        mapping,
        required=_REQUIRED_AGGREGATE_METRIC_FIELDS,
        path=path,
    )


def _validate_candidate_record(
    candidate: Any,
    *,
    path: str,
    expected_rank: int | None = None,
) -> dict[str, Any]:
    """Validate one candidate record under candidate_ranking."""
    record = _require_mapping(candidate, path)
    _require_fields_exact(
        record, required=_REQUIRED_CANDIDATE_FIELDS, path=path
    )
    rank = record["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(
            f"{path}.rank must be a positive int; got {rank!r}"
        )
    if expected_rank is not None and rank != expected_rank:
        raise ValueError(
            f"{path}.rank must equal {expected_rank}; got {rank}"
        )
    _validate_full_hash_string(
        record["configuration_hash_full"],
        f"{path}.configuration_hash_full",
    )
    prefix = record["configuration_hash_prefix"]
    if (
        not isinstance(prefix, str)
        or prefix != record["configuration_hash_full"][:HASH_PREFIX_LENGTH]
    ):
        raise ValueError(
            f"{path}.configuration_hash_prefix must equal the first "
            f"{HASH_PREFIX_LENGTH} characters of "
            f"{path}.configuration_hash_full; got prefix={prefix!r}"
        )
    _validate_aggregate_metrics(
        record["aggregate_metrics"],
        path=f"{path}.aggregate_metrics",
    )
    per_seed_list = _require_list(
        record["per_seed_metrics"], f"{path}.per_seed_metrics"
    )
    for index, entry in enumerate(per_seed_list):
        _validate_per_seed_record(
            entry, path=f"{path}.per_seed_metrics[{index}]"
        )
    if not isinstance(record["source_run_ids"], list):
        raise ValueError(
            f"{path}.source_run_ids must be a JSON array (list); got "
            f"{type(record['source_run_ids']).__name__}"
        )
    n_records = record["n_calibration_records"]
    if isinstance(n_records, bool) or not isinstance(n_records, int):
        raise ValueError(
            f"{path}.n_calibration_records must be an int; got "
            f"{n_records!r}"
        )
    return record


def _validate_selection_record(
    selection: Any,
    *,
    path: str,
) -> dict[str, Any]:
    """Validate one selections[condition][model] mapping."""
    record = _require_mapping(selection, path)
    _require_fields_exact(
        record, required=_REQUIRED_SELECTION_FIELDS, path=path
    )
    rank = record["selected_rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(
            f"{path}.selected_rank must be a positive int; got "
            f"{rank!r}"
        )
    if rank != 1:
        raise ValueError(
            f"{path}.selected_rank must equal 1 because the artefact "
            "records the rank-1 configuration; got "
            f"selected_rank={rank}"
        )
    _validate_full_hash_string(
        record["selected_configuration_hash_full"],
        f"{path}.selected_configuration_hash_full",
    )
    prefix = record["selected_configuration_hash_prefix"]
    expected_prefix = record["selected_configuration_hash_full"][
        :HASH_PREFIX_LENGTH
    ]
    if not isinstance(prefix, str) or prefix != expected_prefix:
        raise ValueError(
            f"{path}.selected_configuration_hash_prefix must equal "
            f"the first {HASH_PREFIX_LENGTH} characters of "
            f"{path}.selected_configuration_hash_full; got "
            f"prefix={prefix!r}, expected={expected_prefix!r}"
        )
    if not isinstance(record["source_run_ids"], list):
        raise ValueError(
            f"{path}.source_run_ids must be a JSON array (list); got "
            f"{type(record['source_run_ids']).__name__}"
        )
    return record


def validate_selected_configurations_artefact(
    artefact: Mapping[str, Any],
) -> None:
    """Validate the selected-configurations artefact against the schema.

    Performs a strict top-level field check (no unknown fields apart
    from the optional ``code_version``), enforces the literal values
    of identity fields (``schema_version``, ``artefact_type``,
    ``decision_scope``, ``base_model_decision_made``,
    ``selected_configuration_semantics``, ``seed_population``,
    ``calibration_seeds``), recursively scans every nesting depth
    for forbidden field names that would record a base-model
    winner, validates ``selections[condition][model]`` and
    ``candidate_ranking[condition][model]`` for both conditions and
    both models, checks that each ``candidate_ranking`` group
    contains exactly 5 candidates with ranks 1..5, and checks that
    the rank-1 candidate's hash agrees with the ``selections``
    entry's selected hash.

    Parameters
    ----------
    artefact : Mapping[str, Any]
        The artefact to validate.

    Raises
    ------
    ValueError
        On any schema violation. The error message names the exact
        offending path and value.
    """
    if not isinstance(artefact, Mapping):
        raise ValueError(
            "artefact must be a mapping; got "
            f"{type(artefact).__name__}"
        )

    top = dict(artefact)

    # Forbidden-winner-field scan runs first so a forbidden key at
    # any depth is reported even if other top-level fields are also
    # malformed.
    _scan_for_forbidden_field_names(top, "$")

    # Top-level unknown fields are rejected.
    unknown = [
        key for key in top if key not in _ALLOWED_TOP_LEVEL_FIELDS
    ]
    if unknown:
        raise ValueError(
            "artefact contains unknown top-level field(s): "
            f"{sorted(unknown)}; allowed fields are "
            f"{sorted(_ALLOWED_TOP_LEVEL_FIELDS)}"
        )

    _require_fields_exact(
        top, required=_REQUIRED_TOP_LEVEL_FIELDS, path="$"
    )

    if top["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "$.schema_version must equal "
            f"{SCHEMA_VERSION}; got {top['schema_version']!r}"
        )
    if top["artefact_type"] != SELECTED_CONFIGURATIONS_ARTEFACT_TYPE:
        raise ValueError(
            f"$.artefact_type must equal "
            f"{SELECTED_CONFIGURATIONS_ARTEFACT_TYPE!r}; got "
            f"{top['artefact_type']!r}"
        )
    if top["decision_scope"] != DECISION_SCOPE:
        raise ValueError(
            f"$.decision_scope must equal {DECISION_SCOPE!r}; got "
            f"{top['decision_scope']!r}"
        )
    if top["selected_configuration_semantics"] != (
        SELECTED_CONFIGURATION_SEMANTICS
    ):
        raise ValueError(
            "$.selected_configuration_semantics must equal "
            f"{SELECTED_CONFIGURATION_SEMANTICS!r}; got "
            f"{top['selected_configuration_semantics']!r}"
        )
    if top["base_model_decision_made"] is not False:
        raise ValueError(
            "$.base_model_decision_made must be the literal value "
            "False because the calibration handoff does not make "
            "the DAGMA-vs-DCDI base-model decision; got "
            f"{top['base_model_decision_made']!r}"
        )
    if top["seed_population"] != SEED_POPULATION_LABEL:
        raise ValueError(
            "$.seed_population must equal "
            f"{SEED_POPULATION_LABEL!r}; got "
            f"{top['seed_population']!r}"
        )
    cal_seeds = top["calibration_seeds"]
    if not isinstance(cal_seeds, list):
        raise ValueError(
            "$.calibration_seeds must be a JSON array (list); got "
            f"{type(cal_seeds).__name__}"
        )
    if tuple(cal_seeds) != CALIBRATION_SEEDS:
        raise ValueError(
            "$.calibration_seeds must equal "
            f"{list(CALIBRATION_SEEDS)}; got {cal_seeds!r}"
        )

    _validate_full_hash_string(
        top["calibration_run_hash_full"],
        "$.calibration_run_hash_full",
    )
    expected_prefix = top["calibration_run_hash_full"][
        :HASH_PREFIX_LENGTH
    ]
    if top["calibration_run_hash_prefix"] != expected_prefix:
        raise ValueError(
            "$.calibration_run_hash_prefix must equal the first "
            f"{HASH_PREFIX_LENGTH} characters of "
            "$.calibration_run_hash_full; got prefix="
            f"{top['calibration_run_hash_prefix']!r}, expected="
            f"{expected_prefix!r}"
        )

    for ref_name in (
        "selection_rule_id",
        "selection_rule_ref",
        "intervention_policy_ref",
        "fit_rng_policy_ref",
        "generated_at_utc",
    ):
        if not isinstance(top[ref_name], str):
            raise ValueError(
                f"$.{ref_name} must be a string; got "
                f"{type(top[ref_name]).__name__}"
            )
    if "code_version" in top and not isinstance(
        top["code_version"], str
    ):
        raise ValueError(
            "$.code_version must be a string when present; got "
            f"{type(top['code_version']).__name__}"
        )

    selections = _require_mapping(top["selections"], "$.selections")
    candidate_ranking = _require_mapping(
        top["candidate_ranking"], "$.candidate_ranking"
    )

    selection_records: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in CONDITIONS:
        if condition not in selections:
            raise ValueError(
                f"$.selections is missing condition {condition!r}; "
                f"required conditions are {list(CONDITIONS)}"
            )
        per_condition = _require_mapping(
            selections[condition], f"$.selections[{condition!r}]"
        )
        for model in MODELS:
            if model not in per_condition:
                raise ValueError(
                    f"$.selections[{condition!r}] is missing model "
                    f"{model!r}; required models are {list(MODELS)}"
                )
            record = _validate_selection_record(
                per_condition[model],
                path=f"$.selections[{condition!r}][{model!r}]",
            )
            selection_records[(condition, model)] = record

    for condition in CONDITIONS:
        if condition not in candidate_ranking:
            raise ValueError(
                f"$.candidate_ranking is missing condition "
                f"{condition!r}; required conditions are "
                f"{list(CONDITIONS)}"
            )
        per_condition = _require_mapping(
            candidate_ranking[condition],
            f"$.candidate_ranking[{condition!r}]",
        )
        for model in MODELS:
            if model not in per_condition:
                raise ValueError(
                    f"$.candidate_ranking[{condition!r}] is missing "
                    f"model {model!r}; required models are "
                    f"{list(MODELS)}"
                )
            candidates_list = _require_list(
                per_condition[model],
                f"$.candidate_ranking[{condition!r}][{model!r}]",
            )
            if len(candidates_list) != CANDIDATES_PER_CONDITION_PER_MODEL:
                raise ValueError(
                    f"$.candidate_ranking[{condition!r}][{model!r}]"
                    " must contain exactly "
                    f"{CANDIDATES_PER_CONDITION_PER_MODEL} candidates; "
                    f"got {len(candidates_list)}"
                )
            ranks_seen: list[int] = []
            for index, candidate in enumerate(candidates_list):
                candidate_record = _validate_candidate_record(
                    candidate,
                    path=(
                        f"$.candidate_ranking[{condition!r}]"
                        f"[{model!r}][{index}]"
                    ),
                    expected_rank=index + 1,
                )
                ranks_seen.append(candidate_record["rank"])
            if ranks_seen != list(
                range(1, CANDIDATES_PER_CONDITION_PER_MODEL + 1)
            ):
                raise ValueError(
                    f"$.candidate_ranking[{condition!r}][{model!r}]"
                    " ranks must be exactly 1..5 in order; got "
                    f"{ranks_seen}"
                )

            rank_one = candidates_list[0]
            selection_record = selection_records[(condition, model)]
            sel_full = selection_record[
                "selected_configuration_hash_full"
            ]
            if rank_one["configuration_hash_full"] != sel_full:
                raise ValueError(
                    f"$.candidate_ranking[{condition!r}][{model!r}]"
                    "[0].configuration_hash_full must equal "
                    f"$.selections[{condition!r}][{model!r}]."
                    "selected_configuration_hash_full because the "
                    "rank-1 candidate is the selected one; got "
                    f"candidate={rank_one['configuration_hash_full']!r}, "
                    f"selection={sel_full!r}"
                )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _atomic_write_json(
    artefact: Mapping[str, Any], output_path: Path
) -> None:
    """Write ``artefact`` to ``output_path`` atomically.

    The function writes to a temporary file inside the parent
    directory, reads it back, re-validates the parsed JSON, and
    atomically replaces ``output_path`` via ``os.replace``. On any
    failure the temporary file is removed and ``output_path`` is
    left untouched.
    """
    parent = output_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{SELECTED_CONFIGURATIONS_FILENAME}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    moved = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                artefact,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        with tmp_path.open("r", encoding="utf-8") as handle:
            read_back = json.load(handle)
        validate_selected_configurations_artefact(read_back)
        os.replace(tmp_path, output_path)
        moved = True
    finally:
        if not moved:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_selected_configurations(
    artefact: Mapping[str, Any],
    output_path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Validate and atomically write a selected-configurations artefact.

    The flow is: (1) validate the in-memory artefact; (2) refuse to
    overwrite an existing file at ``output_path`` unless
    ``force=True``; (3) create the parent directories; (4) write a
    UTF-8 JSON file (``indent=2``, ``sort_keys=True``,
    ``ensure_ascii=True``) to a same-directory temporary file; (5)
    read it back and re-validate the parsed JSON; (6) atomically
    replace ``output_path`` via ``os.replace``. If any step fails the
    temporary file is removed and the canonical path is left
    untouched.

    Parameters
    ----------
    artefact : Mapping[str, Any]
        The artefact to validate and write.
    output_path : Path or str
        Destination path. Typically built via
        ``selected_configurations_path``.
    force : bool, optional
        When ``False`` (the default), an existing file at
        ``output_path`` causes ``FileExistsError`` to be raised
        before any temporary file is created. When ``True``, the
        existing file is replaced atomically.

    Returns
    -------
    Path
        The ``output_path`` (now a regular file on disk).

    Raises
    ------
    ValueError
        If the artefact fails validation either before writing or
        after the read-back step. No file is written in either case.
    FileExistsError
        If ``output_path`` already exists and ``force`` is ``False``.
    """
    validate_selected_configurations_artefact(artefact)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing selected-configurations "
            f"file at {output_path}; pass force=True to allow "
            "overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(artefact, output_path)
    return output_path


__all__ = [
    "CALIBRATION_RUN_DIRECTORY",
    "CALIBRATION_RUN_IDENTITY_ARTEFACT_TYPE",
    "CALIBRATION_SEEDS",
    "CANDIDATES_PER_CONDITION_PER_MODEL",
    "CONDITIONS",
    "DECISION_SCOPE",
    "FULL_HASH_LENGTH",
    "HASH_PREFIX_LENGTH",
    "MODELS",
    "MODEL_SELECTION_DIRECTORY",
    "SCHEMA_VERSION",
    "SEED_POPULATION_LABEL",
    "SELECTED_CONFIGURATIONS_ARTEFACT_TYPE",
    "SELECTED_CONFIGURATIONS_FILENAME",
    "SELECTED_CONFIGURATION_SEMANTICS",
    "STAGE_LABEL",
    "build_calibration_run_identity_payload",
    "compute_calibration_run_hash12",
    "compute_calibration_run_hash_full",
    "selected_configurations_path",
    "validate_selected_configurations_artefact",
    "write_selected_configurations",
]
