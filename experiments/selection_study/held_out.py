"""Held-out evaluation workload enumeration and preflight.

This module turns a repaired ``selected_configurations.json`` artefact
into the held-out evaluation workload: the 20 main fit jobs (one per
selected configuration per held-out SCM seed) and the 5 DCDI fit-RNG
sensitivity diagnostic jobs.

The module is intentionally side-effect free:

- no directory is created;
- no record is written;
- no artefact is written;
- no log is written;
- no wrapper module is imported;
- ``pipeline.run_single_fit`` is not called.

It computes path strings, returns dataclasses, and reports a JSON-safe
preflight dict so the operator can audit the planned held-out run
before any real fit is invoked.

Workload arithmetic
-------------------
- 4 selected configurations x 5 main held-out SCM seeds = 20 main jobs.
- 1 calibration-selected DCDI / centred_only configuration
  x 1 held-out SCM seed (301) x 5 fit-RNG values = 5 sensitivity jobs.
- Total: 25 jobs. Main and sensitivity jobs are structurally
  separated by ``job_kind`` and never share an entry in the workload.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    FIT_RNG_POLICY_REF,
    FULL_HASH_LENGTH,
    HASH_PREFIX_LENGTH,
    INTERVENTION_POLICY_REF,
    MODELS,
    MODEL_SELECTION_DIRECTORY,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
    validate_selected_configurations_artefact,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


HELDOUT_RUN_DIRECTORY = "held_out"
HELDOUT_EVALUATION_FILENAME = "heldout_evaluation.json"
RECORDS_DIRECTORY_NAME = "records"

HELDOUT_SCM_SEEDS: tuple[int, ...] = (301, 302, 303, 304, 305)
MAIN_JOB_KIND = "main"
SENSITIVITY_JOB_KIND = "fit_rng_sensitivity"

SENSITIVITY_MODEL = "dcdi"
SENSITIVITY_CONDITION = "centred_only"
SENSITIVITY_SCM_SEED = 301
SENSITIVITY_FIT_RNGS: tuple[int, ...] = (43, 44, 45, 46, 47)

# DAGMA wrappers are deterministic by construction in this project,
# so the held-out workload records fit_rng=None for DAGMA main jobs
# and the calibration-stage fixed fit_rng=42 for DCDI main jobs.
DAGMA_MAIN_FIT_RNG: None = None
DCDI_MAIN_FIT_RNG: int = 42

# Stable policy identifier for the held-out fit-RNG sensitivity probe;
# kept local to this module so the existing calibration-stage policy
# refs in ``selection_artefact`` remain untouched.
HELDOUT_FIT_RNG_SENSITIVITY_REF = (
    "dcdi_fit_rng_sensitivity_seeds_43_44_45_46_47_v1"
)

HELDOUT_RUN_IDENTITY_ARTEFACT_TYPE = "heldout_run_identity"
HELDOUT_PREFLIGHT_REPORT_ARTEFACT_TYPE = "heldout_preflight_report"
HELDOUT_STAGE_LABEL = "held_out_evaluation"
HELDOUT_PREFLIGHT_SCHEMA_VERSION = 1

_STATUS_WOULD_BE_CREATED = "would_be_created"
_STATUS_ALREADY_EXISTS = "already_exists"

_GENERATED_AT_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_HEX_DIGITS = frozenset("0123456789abcdef")
_HELDOUT_SEED_SET: frozenset[int] = frozenset(HELDOUT_SCM_SEEDS)
_CALIBRATION_SEED_SET: frozenset[int] = frozenset(CALIBRATION_SEEDS)

EXPECTED_MAIN_JOB_COUNT = len(MODELS) * len(CONDITIONS) * len(HELDOUT_SCM_SEEDS)
EXPECTED_SENSITIVITY_JOB_COUNT = len(SENSITIVITY_FIT_RNGS)
EXPECTED_TOTAL_JOB_COUNT = (
    EXPECTED_MAIN_JOB_COUNT + EXPECTED_SENSITIVITY_JOB_COUNT
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _format_utc(value: datetime) -> str:
    """Format a UTC ``datetime`` as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return value.strftime(_GENERATED_AT_UTC_FORMAT)


def _default_now_fn() -> datetime:
    """Return the current time as a UTC zone-aware ``datetime``."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldoutJob:
    """One held-out fit job, either a main job or a sensitivity job.

    Attributes
    ----------
    job_kind : str
        Either ``"main"`` or ``"fit_rng_sensitivity"``. Main jobs and
        sensitivity jobs never appear in the same enumerated list.
    model : str
        ``"dagma"`` or ``"dcdi"``.
    condition : str
        ``"centred_only"`` or ``"standardised"``.
    configuration_hash_full : str
        Full 64-character lowercase hex hash of the selected
        configuration this job evaluates.
    configuration_hash_prefix : str
        First 12 characters of ``configuration_hash_full``.
    hyperparameters : Mapping
        Hyperparameter mapping copied verbatim from the
        calibration-selected configuration.
    scm_seed : int
        Held-out SCM seed value. Never a calibration seed (201, 202).
    fit_rng : int or None
        Optimiser RNG value. ``None`` for DAGMA main jobs. ``42`` for
        DCDI main jobs. One of ``SENSITIVITY_FIT_RNGS`` for
        sensitivity jobs.
    calibration_run_hash_prefix : str
        Parent calibration-run identity carried on every job so the
        downstream held-out artefact can be traced back to the
        calibration selection it consumed.
    """

    job_kind: str
    model: str
    condition: str
    configuration_hash_full: str
    configuration_hash_prefix: str
    hyperparameters: Mapping[str, Any]
    scm_seed: int
    fit_rng: int | None
    calibration_run_hash_prefix: str


@dataclass(frozen=True)
class HeldoutWorkload:
    """Enumerated held-out evaluation workload.

    Attributes
    ----------
    calibration_run_hash_full : str
        Parent calibration_run_hash_full from the input artefact.
    calibration_run_hash_prefix : str
        First 12 characters of ``calibration_run_hash_full``.
    selected_configurations_used : tuple of dict
        Identity-and-metric summary of the four selected
        configurations, in canonical (condition, model) order.
    main_jobs : tuple of HeldoutJob
        Exactly ``EXPECTED_MAIN_JOB_COUNT`` jobs.
    sensitivity_jobs : tuple of HeldoutJob
        Exactly ``EXPECTED_SENSITIVITY_JOB_COUNT`` jobs.
    """

    calibration_run_hash_full: str
    calibration_run_hash_prefix: str
    selected_configurations_used: tuple[Mapping[str, Any], ...]
    main_jobs: tuple[HeldoutJob, ...]
    sensitivity_jobs: tuple[HeldoutJob, ...] = field(default_factory=tuple)

    @property
    def total_job_count(self) -> int:
        return len(self.main_jobs) + len(self.sensitivity_jobs)


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------


def _read_artefact_json(path: Path) -> dict[str, Any]:
    """Read and parse the selected_configurations JSON file."""
    if not path.is_file():
        raise FileNotFoundError(
            "selected_configurations artefact not found at "
            f"{path}; the held-out preflight requires an "
            "already-written calibration handoff artefact"
        )
    with path.open("r", encoding="utf-8") as handle:
        try:
            artefact = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "selected_configurations artefact at "
                f"{path} is not valid JSON: {exc}"
            ) from exc
    if not isinstance(artefact, dict):
        raise ValueError(
            "selected_configurations artefact at "
            f"{path} must be a JSON object at the top level; "
            f"got {type(artefact).__name__}"
        )
    return artefact


def _is_finite_number(value: Any) -> bool:
    """Return True iff ``value`` is a finite int or float (not bool)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    return False


def _check_selected_configurations(
    artefact: Mapping[str, Any], *, source_path: Path
) -> None:
    """Reject degenerate or non-finite selected configurations."""
    selections = artefact["selections"]
    metric_fields = ("mean_sid", "mean_mmd_primary", "mean_shd")
    for condition in CONDITIONS:
        for model in MODELS:
            selection = selections[condition][model]
            degeneracy_flag = selection.get("degeneracy_flag", False)
            if degeneracy_flag is True:
                raise ValueError(
                    "selected_configurations artefact at "
                    f"{source_path} carries degeneracy_flag=True for "
                    f"selections[{condition!r}][{model!r}]; the "
                    "held-out evaluation refuses to consume a "
                    "degenerate calibration selection. Repair the "
                    "calibration record set before retrying."
                )
            metrics = selection.get("selection_metrics", {})
            if not isinstance(metrics, Mapping):
                raise ValueError(
                    "selected_configurations artefact at "
                    f"{source_path} has a non-mapping selection_metrics "
                    f"at selections[{condition!r}][{model!r}]"
                )
            non_finite_fields: list[str] = []
            for metric_name in metric_fields:
                if metric_name not in metrics:
                    raise ValueError(
                        "selected_configurations artefact at "
                        f"{source_path} is missing aggregate metric "
                        f"{metric_name!r} at selections[{condition!r}]"
                        f"[{model!r}].selection_metrics"
                    )
                if not _is_finite_number(metrics[metric_name]):
                    non_finite_fields.append(metric_name)
            if non_finite_fields:
                raise ValueError(
                    "selected_configurations artefact at "
                    f"{source_path} has non-finite aggregate metric(s) "
                    f"{non_finite_fields} at selections[{condition!r}]"
                    f"[{model!r}].selection_metrics; the held-out "
                    "evaluation refuses to consume a calibration "
                    "selection whose mean metrics are not finite"
                )


# ---------------------------------------------------------------------------
# Workload enumeration
# ---------------------------------------------------------------------------


def _selected_summary_record(
    *,
    condition: str,
    model: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-safe summary record for the selected configuration."""
    metrics = selection["selection_metrics"]
    return {
        "condition": condition,
        "model": model,
        "configuration_hash_full": selection[
            "selected_configuration_hash_full"
        ],
        "configuration_hash_prefix": selection[
            "selected_configuration_hash_prefix"
        ],
        "hyperparameters": dict(selection["selected_hyperparameters"]),
        "selection_metrics_summary": {
            "mean_sid": metrics["mean_sid"],
            "mean_mmd_primary": metrics["mean_mmd_primary"],
            "mean_shd": metrics["mean_shd"],
            "degeneracy_flag": bool(
                selection.get("degeneracy_flag", False)
            ),
        },
    }


def _main_fit_rng_for(model: str) -> int | None:
    """Return the fit_rng value for a main held-out job."""
    if model == "dagma":
        return DAGMA_MAIN_FIT_RNG
    if model == "dcdi":
        return DCDI_MAIN_FIT_RNG
    raise ValueError(
        f"unknown model {model!r} in held-out enumeration; allowed "
        f"values are {list(MODELS)}"
    )


def _enumerate_main_jobs(
    *,
    artefact: Mapping[str, Any],
    calibration_run_hash_prefix: str,
) -> tuple[HeldoutJob, ...]:
    """Build the 20 main held-out jobs in deterministic order."""
    selections = artefact["selections"]
    jobs: list[HeldoutJob] = []
    for condition in CONDITIONS:
        for model in MODELS:
            selection = selections[condition][model]
            config_hash_full = str(
                selection["selected_configuration_hash_full"]
            )
            config_hash_prefix = str(
                selection["selected_configuration_hash_prefix"]
            )
            hyperparameters = dict(selection["selected_hyperparameters"])
            fit_rng = _main_fit_rng_for(model)
            for scm_seed in HELDOUT_SCM_SEEDS:
                jobs.append(
                    HeldoutJob(
                        job_kind=MAIN_JOB_KIND,
                        model=model,
                        condition=condition,
                        configuration_hash_full=config_hash_full,
                        configuration_hash_prefix=config_hash_prefix,
                        hyperparameters=hyperparameters,
                        scm_seed=int(scm_seed),
                        fit_rng=fit_rng,
                        calibration_run_hash_prefix=(
                            calibration_run_hash_prefix
                        ),
                    )
                )
    return tuple(jobs)


def _enumerate_sensitivity_jobs(
    *,
    artefact: Mapping[str, Any],
    calibration_run_hash_prefix: str,
) -> tuple[HeldoutJob, ...]:
    """Build the 5 DCDI fit-RNG sensitivity jobs in deterministic order."""
    selection = artefact["selections"][SENSITIVITY_CONDITION][
        SENSITIVITY_MODEL
    ]
    config_hash_full = str(selection["selected_configuration_hash_full"])
    config_hash_prefix = str(selection["selected_configuration_hash_prefix"])
    hyperparameters = dict(selection["selected_hyperparameters"])
    jobs: list[HeldoutJob] = []
    for fit_rng in SENSITIVITY_FIT_RNGS:
        jobs.append(
            HeldoutJob(
                job_kind=SENSITIVITY_JOB_KIND,
                model=SENSITIVITY_MODEL,
                condition=SENSITIVITY_CONDITION,
                configuration_hash_full=config_hash_full,
                configuration_hash_prefix=config_hash_prefix,
                hyperparameters=hyperparameters,
                scm_seed=int(SENSITIVITY_SCM_SEED),
                fit_rng=int(fit_rng),
                calibration_run_hash_prefix=calibration_run_hash_prefix,
            )
        )
    return tuple(jobs)


def _assert_workload_invariants(
    workload: HeldoutWorkload, *, source_path: Path
) -> None:
    """Pin the documented workload cardinalities and seed invariants."""
    if len(workload.main_jobs) != EXPECTED_MAIN_JOB_COUNT:
        raise ValueError(
            "held-out main workload must contain exactly "
            f"{EXPECTED_MAIN_JOB_COUNT} jobs; built "
            f"{len(workload.main_jobs)} from {source_path}"
        )
    if len(workload.sensitivity_jobs) != EXPECTED_SENSITIVITY_JOB_COUNT:
        raise ValueError(
            "held-out sensitivity workload must contain exactly "
            f"{EXPECTED_SENSITIVITY_JOB_COUNT} jobs; built "
            f"{len(workload.sensitivity_jobs)} from {source_path}"
        )
    main_seeds = {job.scm_seed for job in workload.main_jobs}
    if main_seeds != set(HELDOUT_SCM_SEEDS):
        raise ValueError(
            "held-out main jobs must cover SCM seeds "
            f"{sorted(HELDOUT_SCM_SEEDS)}; got "
            f"{sorted(main_seeds)} from {source_path}"
        )
    for job in workload.main_jobs:
        if job.scm_seed in _CALIBRATION_SEED_SET:
            raise ValueError(
                "held-out main job uses calibration SCM seed "
                f"{job.scm_seed}; calibration seeds "
                f"{sorted(CALIBRATION_SEEDS)} must not appear in the "
                "held-out workload"
            )
    for job in workload.sensitivity_jobs:
        if job.scm_seed != SENSITIVITY_SCM_SEED:
            raise ValueError(
                "held-out sensitivity job uses SCM seed "
                f"{job.scm_seed}; the sensitivity probe requires SCM "
                f"seed {SENSITIVITY_SCM_SEED}"
            )
        if job.scm_seed in _CALIBRATION_SEED_SET:
            raise ValueError(
                "held-out sensitivity job uses calibration SCM seed "
                f"{job.scm_seed}; calibration seeds must not appear "
                "in the held-out workload"
            )
        if job.fit_rng not in SENSITIVITY_FIT_RNGS:
            raise ValueError(
                "held-out sensitivity job uses fit_rng "
                f"{job.fit_rng}; allowed sensitivity fit_rng values "
                f"are {list(SENSITIVITY_FIT_RNGS)}"
            )
    sensitivity_fit_rngs = tuple(
        job.fit_rng for job in workload.sensitivity_jobs
    )
    if sensitivity_fit_rngs != SENSITIVITY_FIT_RNGS:
        raise ValueError(
            "held-out sensitivity fit_rngs must equal "
            f"{list(SENSITIVITY_FIT_RNGS)} in order; got "
            f"{list(sensitivity_fit_rngs)}"
        )
    main_keys = {
        (
            job.job_kind,
            job.model,
            job.condition,
            job.configuration_hash_full,
            job.scm_seed,
            job.fit_rng,
        )
        for job in workload.main_jobs
    }
    sensitivity_keys = {
        (
            job.job_kind,
            job.model,
            job.condition,
            job.configuration_hash_full,
            job.scm_seed,
            job.fit_rng,
        )
        for job in workload.sensitivity_jobs
    }
    overlap = main_keys & sensitivity_keys
    if overlap:
        raise ValueError(
            "held-out main jobs and sensitivity jobs must be "
            "structurally separate; got overlapping job identities: "
            f"{sorted(overlap)}"
        )
    for job in workload.sensitivity_jobs:
        if job.job_kind != SENSITIVITY_JOB_KIND:
            raise ValueError(
                "every sensitivity job must carry "
                f"job_kind={SENSITIVITY_JOB_KIND!r}; got "
                f"{job.job_kind!r}"
            )
    for job in workload.main_jobs:
        if job.job_kind != MAIN_JOB_KIND:
            raise ValueError(
                "every main job must carry "
                f"job_kind={MAIN_JOB_KIND!r}; got {job.job_kind!r}"
            )


def enumerate_heldout_workload(
    selected_configurations_path: Path | str,
) -> HeldoutWorkload:
    """Enumerate the held-out evaluation workload from a calibration artefact.

    Parameters
    ----------
    selected_configurations_path : Path or str
        Path to a written ``selected_configurations.json`` artefact.

    Returns
    -------
    HeldoutWorkload
        The enumerated workload: 20 main jobs and 5 sensitivity jobs
        in deterministic order, plus the parent calibration_run_hash
        and a summary of the four selected configurations.

    Raises
    ------
    FileNotFoundError
        If the artefact file does not exist.
    ValueError
        If the artefact fails structural validation, if any
        selected configuration carries ``degeneracy_flag=True``, or
        if any selected configuration has a non-finite aggregate
        metric (``mean_sid``, ``mean_mmd_primary``, ``mean_shd``).
    """
    source_path = Path(selected_configurations_path)
    artefact = _read_artefact_json(source_path)
    validate_selected_configurations_artefact(artefact)
    _check_selected_configurations(artefact, source_path=source_path)

    calibration_run_hash_full = str(artefact["calibration_run_hash_full"])
    calibration_run_hash_prefix = str(
        artefact["calibration_run_hash_prefix"]
    )

    selections = artefact["selections"]
    selected_summary: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model in MODELS:
            selected_summary.append(
                _selected_summary_record(
                    condition=condition,
                    model=model,
                    selection=selections[condition][model],
                )
            )

    main_jobs = _enumerate_main_jobs(
        artefact=artefact,
        calibration_run_hash_prefix=calibration_run_hash_prefix,
    )
    sensitivity_jobs = _enumerate_sensitivity_jobs(
        artefact=artefact,
        calibration_run_hash_prefix=calibration_run_hash_prefix,
    )

    workload = HeldoutWorkload(
        calibration_run_hash_full=calibration_run_hash_full,
        calibration_run_hash_prefix=calibration_run_hash_prefix,
        selected_configurations_used=tuple(selected_summary),
        main_jobs=main_jobs,
        sensitivity_jobs=sensitivity_jobs,
    )
    _assert_workload_invariants(workload, source_path=source_path)
    return workload


# ---------------------------------------------------------------------------
# Identity hash
# ---------------------------------------------------------------------------


def _validate_hex_string(value: object, *, length: int, where: str) -> None:
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a string; got {type(value).__name__}"
        )
    if len(value) != length:
        raise ValueError(
            f"{where} must be a {length}-character lowercase hex "
            f"string; got length {len(value)}"
        )
    for ch in value:
        if ch not in _HEX_DIGITS:
            raise ValueError(
                f"{where} must contain only lowercase hex digits "
                f"0-9 and a-f; got character {ch!r}"
            )


def build_heldout_run_identity_payload(
    *,
    parent_calibration_run_hash_full: str,
    selected_configuration_hashes_full: Sequence[str],
    main_heldout_seeds: Sequence[int],
    sensitivity_spec: Mapping[str, Any],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
    heldout_fit_rng_sensitivity_ref: str,
) -> dict[str, Any]:
    """Build the canonical identity payload for the held-out run.

    The payload is the input to ``compute_heldout_run_hash_full``.
    Selected-configuration hashes are sorted lexicographically so the
    output is independent of caller iteration order. Held-out SCM
    seeds and sensitivity fit_rngs are sorted as ints.
    """
    _validate_hex_string(
        parent_calibration_run_hash_full,
        length=FULL_HASH_LENGTH,
        where="parent_calibration_run_hash_full",
    )
    hashes_validated: list[str] = []
    for index, value in enumerate(selected_configuration_hashes_full):
        _validate_hex_string(
            value,
            length=FULL_HASH_LENGTH,
            where=f"selected_configuration_hashes_full[{index}]",
        )
        hashes_validated.append(str(value))

    seeds_sorted = sorted(int(seed) for seed in main_heldout_seeds)
    if not isinstance(sensitivity_spec, Mapping):
        raise ValueError(
            "sensitivity_spec must be a mapping; got "
            f"{type(sensitivity_spec).__name__}"
        )
    for key in ("model", "condition", "scm_seed", "fit_rngs"):
        if key not in sensitivity_spec:
            raise ValueError(
                "sensitivity_spec is missing required field "
                f"{key!r}"
            )
    canonical_sensitivity: dict[str, Any] = {
        "model": str(sensitivity_spec["model"]),
        "condition": str(sensitivity_spec["condition"]),
        "scm_seed": int(sensitivity_spec["scm_seed"]),
        "fit_rngs": sorted(
            int(value) for value in sensitivity_spec["fit_rngs"]
        ),
    }

    for ref_name, ref_value in (
        ("selection_rule_id", selection_rule_id),
        ("selection_rule_ref", selection_rule_ref),
        ("intervention_policy_ref", intervention_policy_ref),
        ("fit_rng_policy_ref", fit_rng_policy_ref),
        ("heldout_fit_rng_sensitivity_ref", heldout_fit_rng_sensitivity_ref),
    ):
        if not isinstance(ref_value, str):
            raise ValueError(
                f"{ref_name} must be a string; got "
                f"{type(ref_value).__name__}"
            )

    return {
        "schema_version": HELDOUT_PREFLIGHT_SCHEMA_VERSION,
        "artefact_type": HELDOUT_RUN_IDENTITY_ARTEFACT_TYPE,
        "stage": HELDOUT_STAGE_LABEL,
        "parent_calibration_run_hash_full": str(
            parent_calibration_run_hash_full
        ),
        "selected_configuration_hashes_full": sorted(hashes_validated),
        "main_heldout_seeds": seeds_sorted,
        "sensitivity_spec": canonical_sensitivity,
        "selection_rule_id": str(selection_rule_id),
        "selection_rule_ref": str(selection_rule_ref),
        "intervention_policy_ref": str(intervention_policy_ref),
        "fit_rng_policy_ref": str(fit_rng_policy_ref),
        "heldout_fit_rng_sensitivity_ref": str(
            heldout_fit_rng_sensitivity_ref
        ),
    }


def compute_heldout_run_hash_full(
    *,
    parent_calibration_run_hash_full: str,
    selected_configuration_hashes_full: Sequence[str],
    main_heldout_seeds: Sequence[int],
    sensitivity_spec: Mapping[str, Any],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
    heldout_fit_rng_sensitivity_ref: str,
) -> str:
    """Return the SHA-256 hex digest of the canonical identity payload."""
    payload = build_heldout_run_identity_payload(
        parent_calibration_run_hash_full=parent_calibration_run_hash_full,
        selected_configuration_hashes_full=selected_configuration_hashes_full,
        main_heldout_seeds=main_heldout_seeds,
        sensitivity_spec=sensitivity_spec,
        selection_rule_id=selection_rule_id,
        selection_rule_ref=selection_rule_ref,
        intervention_policy_ref=intervention_policy_ref,
        fit_rng_policy_ref=fit_rng_policy_ref,
        heldout_fit_rng_sensitivity_ref=heldout_fit_rng_sensitivity_ref,
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_heldout_run_hash12(
    *,
    parent_calibration_run_hash_full: str,
    selected_configuration_hashes_full: Sequence[str],
    main_heldout_seeds: Sequence[int],
    sensitivity_spec: Mapping[str, Any],
    selection_rule_id: str,
    selection_rule_ref: str,
    intervention_policy_ref: str,
    fit_rng_policy_ref: str,
    heldout_fit_rng_sensitivity_ref: str,
) -> str:
    """Return the 12-character hex prefix of the held-out run hash."""
    return compute_heldout_run_hash_full(
        parent_calibration_run_hash_full=parent_calibration_run_hash_full,
        selected_configuration_hashes_full=selected_configuration_hashes_full,
        main_heldout_seeds=main_heldout_seeds,
        sensitivity_spec=sensitivity_spec,
        selection_rule_id=selection_rule_id,
        selection_rule_ref=selection_rule_ref,
        intervention_policy_ref=intervention_policy_ref,
        fit_rng_policy_ref=fit_rng_policy_ref,
        heldout_fit_rng_sensitivity_ref=heldout_fit_rng_sensitivity_ref,
    )[:HASH_PREFIX_LENGTH]


def _heldout_run_hash_inputs_from_workload(
    workload: HeldoutWorkload,
) -> dict[str, Any]:
    """Collect the hash inputs from a workload in canonical form."""
    selected_hashes = sorted(
        record["configuration_hash_full"]
        for record in workload.selected_configurations_used
    )
    sensitivity_spec = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "scm_seed": SENSITIVITY_SCM_SEED,
        "fit_rngs": list(SENSITIVITY_FIT_RNGS),
    }
    return {
        "parent_calibration_run_hash_full": (
            workload.calibration_run_hash_full
        ),
        "selected_configuration_hashes_full": selected_hashes,
        "main_heldout_seeds": list(HELDOUT_SCM_SEEDS),
        "sensitivity_spec": sensitivity_spec,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "heldout_fit_rng_sensitivity_ref": HELDOUT_FIT_RNG_SENSITIVITY_REF,
    }


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


def heldout_run_dir_path(
    *, heldout_run_hash12: str, results_root: Path | str
) -> Path:
    """Return the canonical held-out run directory path."""
    _validate_hex_string(
        heldout_run_hash12,
        length=HASH_PREFIX_LENGTH,
        where="heldout_run_hash12",
    )
    return (
        Path(results_root)
        / MODEL_SELECTION_DIRECTORY
        / HELDOUT_RUN_DIRECTORY
        / heldout_run_hash12
    )


def heldout_evaluation_path(
    *, heldout_run_hash12: str, results_root: Path | str
) -> Path:
    """Return the canonical held-out evaluation artefact path."""
    return (
        heldout_run_dir_path(
            heldout_run_hash12=heldout_run_hash12,
            results_root=results_root,
        )
        / HELDOUT_EVALUATION_FILENAME
    )


def heldout_records_dir_path(
    *, heldout_run_hash12: str, results_root: Path | str
) -> Path:
    """Return the canonical held-out per-fit records directory path."""
    return (
        heldout_run_dir_path(
            heldout_run_hash12=heldout_run_hash12,
            results_root=results_root,
        )
        / RECORDS_DIRECTORY_NAME
    )


# ---------------------------------------------------------------------------
# Preflight entry point
# ---------------------------------------------------------------------------


def preflight_heldout_evaluation(
    selected_configurations_path: Path | str,
    results_root: Path | str,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe preflight report for the held-out workload.

    This function performs every load / validate / enumerate / hash /
    path-resolve step that the held-out runner will perform but stops
    before any fit, directory creation, record write, log write, or
    artefact write. It does not import any wrapper module and does
    not call ``pipeline.run_single_fit``.

    Parameters
    ----------
    selected_configurations_path : Path or str
        Path to the calibration handoff artefact.
    results_root : Path or str
        Root of the results tree. The reported run directory is
        ``<results_root>/model_selection/held_out/<heldout_run_hash12>/``.
    now_fn : callable or None, optional
        Optional injection point for the wall-clock time source.
        When omitted, the current UTC time is used.

    Returns
    -------
    dict
        JSON-safe preflight report.

    Raises
    ------
    FileNotFoundError
        If the artefact file does not exist.
    ValueError
        If the artefact fails validation or carries degenerate or
        non-finite selected configurations.
    """
    source_path = Path(selected_configurations_path)
    workload = enumerate_heldout_workload(source_path)
    resolved_now_fn: Callable[[], datetime] = (
        now_fn if now_fn is not None else _default_now_fn
    )

    hash_inputs = _heldout_run_hash_inputs_from_workload(workload)
    heldout_run_hash_full = compute_heldout_run_hash_full(**hash_inputs)
    heldout_run_hash_prefix = heldout_run_hash_full[:HASH_PREFIX_LENGTH]

    run_dir = heldout_run_dir_path(
        heldout_run_hash12=heldout_run_hash_prefix,
        results_root=results_root,
    )
    records_dir = heldout_records_dir_path(
        heldout_run_hash12=heldout_run_hash_prefix,
        results_root=results_root,
    )
    artefact_path = heldout_evaluation_path(
        heldout_run_hash12=heldout_run_hash_prefix,
        results_root=results_root,
    )

    existing_output_status = {
        "run_dir": (
            _STATUS_ALREADY_EXISTS
            if run_dir.exists()
            else _STATUS_WOULD_BE_CREATED
        ),
        "records_dir": (
            _STATUS_ALREADY_EXISTS
            if records_dir.exists()
            else _STATUS_WOULD_BE_CREATED
        ),
        "artefact_path": (
            _STATUS_ALREADY_EXISTS
            if artefact_path.exists()
            else _STATUS_WOULD_BE_CREATED
        ),
    }

    sensitivity_spec: dict[str, Any] = {
        "model": SENSITIVITY_MODEL,
        "condition": SENSITIVITY_CONDITION,
        "scm_seed": SENSITIVITY_SCM_SEED,
        "fit_rngs": list(SENSITIVITY_FIT_RNGS),
    }

    report: dict[str, Any] = {
        "artefact_type": HELDOUT_PREFLIGHT_REPORT_ARTEFACT_TYPE,
        "schema_version": HELDOUT_PREFLIGHT_SCHEMA_VERSION,
        "calibration_run_hash_prefix": workload.calibration_run_hash_prefix,
        "heldout_run_hash_full": heldout_run_hash_full,
        "heldout_run_hash_prefix": heldout_run_hash_prefix,
        "main_job_count": len(workload.main_jobs),
        "sensitivity_job_count": len(workload.sensitivity_jobs),
        "total_job_count": workload.total_job_count,
        "main_heldout_seeds": list(HELDOUT_SCM_SEEDS),
        "sensitivity_spec": sensitivity_spec,
        "planned_run_dir": run_dir.as_posix(),
        "planned_records_dir": records_dir.as_posix(),
        "planned_artefact_path": artefact_path.as_posix(),
        "selected_configurations_used": [
            dict(record) for record in workload.selected_configurations_used
        ],
        "existing_output_status": existing_output_status,
        "policy_refs": {
            "selection_rule_id": SELECTION_RULE_ID,
            "selection_rule_ref": SELECTION_RULE_REF,
            "intervention_policy_ref": INTERVENTION_POLICY_REF,
            "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
            "heldout_fit_rng_sensitivity_ref": (
                HELDOUT_FIT_RNG_SENSITIVITY_REF
            ),
        },
        "generated_at_utc": _format_utc(resolved_now_fn()),
    }
    return report


# ---------------------------------------------------------------------------
# Held-out orchestration entry point
# ---------------------------------------------------------------------------


class _HeldoutInfrastructureError(Exception):
    """Held-out infrastructure failure that aborts the run.

    Raised when the orchestrator cannot continue safely: a fit_runner
    raised a ``FileExistsError`` indicating a pre-existing per-fit
    output directory, or the fit_runner returned a structurally
    broken result that cannot be translated into a degenerate record.
    Records already persisted before the failure remain on disk for
    inspection.
    """


_FAILED_STATUS_VALUE = "failed"

# Fields that every fit_runner return value must carry; the
# orchestrator cross-checks the identity-bearing subset against the
# requesting ``HeldoutJob`` and translates a mismatch into a fatal
# infrastructure error. Metric fields may be ``None`` / non-finite on
# a failed fit; the artefact aggregator handles non-finite values.
_REQUIRED_FIT_RESULT_FIELDS: tuple[str, ...] = (
    "job_kind",
    "model",
    "condition",
    "configuration_hash_full",
    "configuration_hash_prefix",
    "hyperparameters",
    "scm_seed",
    "fit_rng",
    "sid",
    "shd",
    "mmd_primary",
    "runtime_seconds",
    "graph_status",
    "sampler_status",
    "training_status",
)


def _record_filename_for_job(job: HeldoutJob) -> str:
    """Build the deterministic on-disk filename for a held-out job record.

    Main jobs use ``<model>_<condition>_<hash_prefix>_seed<scm_seed>.json``.
    Sensitivity jobs append ``_fitrng<fit_rng>`` so the DCDI seed-301
    fit_rng=42 main record does not collide with the fit_rng=43..47
    sensitivity records that share the same model, condition,
    configuration_hash_prefix, and scm_seed.
    """
    base = (
        f"{job.model}_{job.condition}_"
        f"{job.configuration_hash_prefix}_seed{int(job.scm_seed)}"
    )
    if job.job_kind == SENSITIVITY_JOB_KIND:
        return f"{base}_fitrng{int(job.fit_rng)}.json"
    return f"{base}.json"


def _expected_run_id_for_job(job: HeldoutJob) -> str:
    """Build a deterministic ``run_id`` for a held-out job."""
    fit_rng_label = (
        "none" if job.fit_rng is None else str(int(job.fit_rng))
    )
    return (
        f"{job.model}__{job.condition}__held_out__"
        f"scm{int(job.scm_seed)}__"
        f"fitrng{fit_rng_label}__"
        f"cfg{job.configuration_hash_full}"
    )


def _elapsed_seconds(start: datetime, end: datetime) -> float:
    """Return the wall-clock interval ``(end - start)`` in seconds."""
    return float((end - start).total_seconds())


def _build_degenerate_heldout_record(
    *,
    job: HeldoutJob,
    runtime_seconds: float,
    failure_type: str,
    failure_message: str,
) -> dict[str, Any]:
    """Build a structurally valid degenerate record for a failed fit.

    The record carries the job's identity verbatim, marks every
    metric as ``None`` (the artefact aggregator treats ``None`` as
    non-finite), and records ``training_status="failed"`` /
    ``graph_status="failed"`` / ``sampler_status="failed"`` so the
    cell's status counts surface the failure honestly. The
    ``failure_type`` and ``failure_message`` fields preserve the
    underlying exception type and message for offline auditing.
    """
    return {
        "job_kind": job.job_kind,
        "model": job.model,
        "condition": job.condition,
        "configuration_hash_full": job.configuration_hash_full,
        "configuration_hash_prefix": job.configuration_hash_prefix,
        "hyperparameters": dict(job.hyperparameters),
        "scm_seed": int(job.scm_seed),
        "fit_rng": job.fit_rng,
        "calibration_run_hash_prefix": job.calibration_run_hash_prefix,
        "sid": None,
        "shd": None,
        "mmd_primary": None,
        "runtime_seconds": float(runtime_seconds),
        "graph_status": _FAILED_STATUS_VALUE,
        "sampler_status": _FAILED_STATUS_VALUE,
        "training_status": _FAILED_STATUS_VALUE,
        "n_iterations": None,
        "run_id": _expected_run_id_for_job(job),
        "failure_type": str(failure_type),
        "failure_message": str(failure_message),
    }


def _validate_heldout_fit_result(
    raw_result: Any, *, job: HeldoutJob
) -> dict[str, Any]:
    """Cross-check a fit_runner return value against the job identity.

    A structurally broken return value or an identity mismatch is an
    infrastructure failure: the orchestrator aborts the run rather
    than masking the broken assumption with a degenerate record.
    """
    if not isinstance(raw_result, Mapping):
        raise _HeldoutInfrastructureError(
            "fit_runner result for held-out job "
            f"(job_kind={job.job_kind!r}, model={job.model!r}, "
            f"condition={job.condition!r}, "
            f"scm_seed={int(job.scm_seed)}, "
            f"fit_rng={job.fit_rng!r}) is not a mapping; got "
            f"{type(raw_result).__name__}"
        )
    missing = [
        name
        for name in _REQUIRED_FIT_RESULT_FIELDS
        if name not in raw_result
    ]
    if missing:
        raise _HeldoutInfrastructureError(
            "fit_runner result for held-out job "
            f"(job_kind={job.job_kind!r}, model={job.model!r}, "
            f"condition={job.condition!r}, "
            f"scm_seed={int(job.scm_seed)}, "
            f"fit_rng={job.fit_rng!r}) is missing required "
            f"field(s): {missing}"
        )
    identity_checks: tuple[tuple[str, Any], ...] = (
        ("job_kind", job.job_kind),
        ("model", job.model),
        ("condition", job.condition),
        ("configuration_hash_full", job.configuration_hash_full),
        ("configuration_hash_prefix", job.configuration_hash_prefix),
        ("scm_seed", int(job.scm_seed)),
        ("fit_rng", job.fit_rng),
    )
    for field_name, expected in identity_checks:
        actual = raw_result.get(field_name)
        if actual != expected:
            raise _HeldoutInfrastructureError(
                "fit_runner result for held-out job "
                f"(job_kind={job.job_kind!r}, "
                f"model={job.model!r}, condition={job.condition!r}, "
                f"scm_seed={int(job.scm_seed)}, "
                f"fit_rng={job.fit_rng!r}) has mismatching "
                f"{field_name}: expected {expected!r}, got {actual!r}"
            )
    if (
        "calibration_run_hash_prefix" in raw_result
        and raw_result["calibration_run_hash_prefix"]
        != job.calibration_run_hash_prefix
    ):
        raise _HeldoutInfrastructureError(
            "fit_runner result for held-out job "
            f"(job_kind={job.job_kind!r}, model={job.model!r}, "
            f"condition={job.condition!r}, "
            f"scm_seed={int(job.scm_seed)}, "
            f"fit_rng={job.fit_rng!r}) has mismatching "
            "calibration_run_hash_prefix: expected "
            f"{job.calibration_run_hash_prefix!r}, got "
            f"{raw_result['calibration_run_hash_prefix']!r}"
        )
    return dict(raw_result)


def _atomic_write_json_record(
    record: Mapping[str, Any], output_path: Path
) -> None:
    """Write a per-fit record JSON file atomically into its parent directory."""
    parent = output_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    moved = False
    try:
        with os.fdopen(
            fd, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                record,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        os.replace(tmp_path, output_path)
        moved = True
    finally:
        if not moved:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def run_held_out_evaluation(
    selected_configurations_path: Path | str,
    results_root: Path | str,
    *,
    fit_runner: Callable[[HeldoutJob], Mapping[str, Any]] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    force: bool = False,
) -> Path:
    """Drive the held-out evaluation workload through an injected fit_runner.

    The orchestrator loads the calibration handoff artefact, enumerates
    the 25 held-out jobs (20 main + 5 fit-RNG sensitivity), creates
    the canonical held-out run directory and records directory, drives
    every job through ``fit_runner`` one at a time, persists each
    per-fit record immediately as JSON, and finally builds, validates,
    and writes ``heldout_evaluation.json``.

    The orchestrator never invokes any model fit directly: it does
    not import any wrapper module and does not call
    ``pipeline.run_single_fit``. ``fit_runner`` is the only execution
    path; supplying ``None`` raises ``NotImplementedError`` because a
    production execution adapter requires translating the held-out
    DCDI fit-RNG variation into the lower-level
    ``seed_torch`` / ``seed_numpy`` fields of an executable
    configuration, which is the responsibility of a later commit.

    Failure handling
    ----------------
    - ``fit_runner`` raising ``FileExistsError`` is treated as a
      pre-existing per-run directory observed by the runner: the
      orchestrator translates the exception into
      ``_HeldoutInfrastructureError`` and aborts the run without
      writing the final artefact. Records already persisted to disk
      are not removed.
    - ``fit_runner`` raising any other exception is treated as a
      model-fit failure: a structurally valid degenerate record is
      persisted in the failed job's slot and orchestration continues
      with the remaining jobs.
    - The final ``heldout_evaluation.json`` is written only after
      every one of the 25 jobs has a record on disk.

    Parameters
    ----------
    selected_configurations_path : Path or str
        Path to a written ``selected_configurations.json`` artefact
        from the calibration handoff.
    results_root : Path or str
        Root of the results tree. The held-out run directory is
        created at
        ``<results_root>/model_selection/held_out/<heldout_run_hash12>/``.
    fit_runner : callable or None, optional
        Required injection point for the per-job runner. Must accept
        a single ``HeldoutJob`` and return a JSON-safe mapping
        carrying the documented per-fit record fields plus identity
        fields matching the job.
    now_fn : callable or None, optional
        Optional injection point for the wall-clock time source.
        When omitted, ``datetime.now(tz=timezone.utc)`` is used.
    force : bool, optional
        When ``False`` (the default), the orchestrator refuses to
        overwrite an existing ``heldout_evaluation.json`` or any
        pre-existing per-fit record file before any fit is run. When
        ``True``, existing files are replaced atomically.

    Returns
    -------
    Path
        The on-disk path to the written ``heldout_evaluation.json``.

    Raises
    ------
    NotImplementedError
        When ``fit_runner`` is ``None``.
    FileNotFoundError
        When the input ``selected_configurations.json`` does not
        exist.
    FileExistsError
        When ``heldout_evaluation.json`` or any per-fit record
        already exists and ``force`` is ``False``.
    _HeldoutInfrastructureError
        When ``fit_runner`` raises ``FileExistsError`` or returns a
        structurally broken or identity-mismatching mapping. The
        final artefact is not written in either case.
    ValueError
        When the input artefact fails validation, when the workload
        invariants are violated, or when the final artefact fails
        schema validation.
    """
    if fit_runner is None:
        raise NotImplementedError(
            "run_held_out_evaluation requires an explicit fit_runner "
            "for now: production execution via pipeline.run_single_fit "
            "is deferred because the existing pipeline does not accept "
            "a fit_rng argument directly, and the DCDI fit-RNG "
            "sensitivity probe requires distinct fit_rng values per "
            "job. Supply a fit_runner callable that takes one "
            "HeldoutJob and returns a per-fit record mapping."
        )

    # Lazy import to avoid a circular import on package load between
    # held_out.py and held_out_artefact.py.
    from experiments.selection_study.held_out_artefact import (
        build_heldout_evaluation_artefact,
        write_heldout_evaluation_artefact,
    )

    workload = enumerate_heldout_workload(selected_configurations_path)
    resolved_now_fn: Callable[[], datetime] = (
        now_fn if now_fn is not None else _default_now_fn
    )

    hash_inputs = _heldout_run_hash_inputs_from_workload(workload)
    heldout_run_hash_full = compute_heldout_run_hash_full(**hash_inputs)
    heldout_run_hash12 = heldout_run_hash_full[:HASH_PREFIX_LENGTH]

    run_dir = heldout_run_dir_path(
        heldout_run_hash12=heldout_run_hash12,
        results_root=results_root,
    )
    records_dir = heldout_records_dir_path(
        heldout_run_hash12=heldout_run_hash12,
        results_root=results_root,
    )
    artefact_path = heldout_evaluation_path(
        heldout_run_hash12=heldout_run_hash12,
        results_root=results_root,
    )

    if artefact_path.exists() and not force:
        raise FileExistsError(
            "refusing to overwrite existing held-out evaluation "
            f"artefact at {artefact_path}; pass force=True to allow "
            "overwrite"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[HeldoutJob] = list(workload.main_jobs) + list(
        workload.sensitivity_jobs
    )
    records: list[dict[str, Any]] = []

    for job in jobs:
        record_path = records_dir / _record_filename_for_job(job)
        if record_path.exists() and not force:
            raise FileExistsError(
                "refusing to overwrite existing held-out per-fit "
                f"record at {record_path}; pass force=True to allow "
                "overwrite"
            )

        start_dt = resolved_now_fn()
        failure_type: str | None = None
        failure_message: str | None = None
        raw_result: Any = None
        try:
            raw_result = fit_runner(job)
        except FileExistsError as exc:
            raise _HeldoutInfrastructureError(
                "fit_runner raised FileExistsError on held-out job "
                f"(job_kind={job.job_kind!r}, model={job.model!r}, "
                f"condition={job.condition!r}, "
                f"scm_seed={int(job.scm_seed)}, "
                f"fit_rng={job.fit_rng!r}); aborting the held-out run "
                "without writing the final artefact: "
                f"{exc}"
            ) from exc
        except _HeldoutInfrastructureError:
            # Re-raise unchanged: a structurally broken fit_runner
            # return value already aborted the run via the
            # orchestrator's own validation path.
            raise
        except Exception as exc:
            failure_type = type(exc).__name__
            failure_message = str(exc)

        end_dt = resolved_now_fn()
        runtime_seconds = _elapsed_seconds(start_dt, end_dt)

        if failure_type is not None:
            record = _build_degenerate_heldout_record(
                job=job,
                runtime_seconds=runtime_seconds,
                failure_type=failure_type,
                failure_message=failure_message or "",
            )
        else:
            record = _validate_heldout_fit_result(raw_result, job=job)
            if "runtime_seconds" not in record:
                record["runtime_seconds"] = runtime_seconds

        _atomic_write_json_record(record, record_path)
        records.append(record)

    generated_at_utc = _format_utc(resolved_now_fn())
    artefact = build_heldout_evaluation_artefact(
        workload=workload,
        records=records,
        generated_at_utc=generated_at_utc,
    )
    write_heldout_evaluation_artefact(
        artefact, artefact_path, force=force
    )
    return artefact_path


__all__ = [
    "DAGMA_MAIN_FIT_RNG",
    "DCDI_MAIN_FIT_RNG",
    "EXPECTED_MAIN_JOB_COUNT",
    "EXPECTED_SENSITIVITY_JOB_COUNT",
    "EXPECTED_TOTAL_JOB_COUNT",
    "HELDOUT_EVALUATION_FILENAME",
    "HELDOUT_FIT_RNG_SENSITIVITY_REF",
    "HELDOUT_PREFLIGHT_REPORT_ARTEFACT_TYPE",
    "HELDOUT_PREFLIGHT_SCHEMA_VERSION",
    "HELDOUT_RUN_DIRECTORY",
    "HELDOUT_RUN_IDENTITY_ARTEFACT_TYPE",
    "HELDOUT_SCM_SEEDS",
    "HELDOUT_STAGE_LABEL",
    "HeldoutJob",
    "HeldoutWorkload",
    "MAIN_JOB_KIND",
    "RECORDS_DIRECTORY_NAME",
    "SENSITIVITY_CONDITION",
    "SENSITIVITY_FIT_RNGS",
    "SENSITIVITY_JOB_KIND",
    "SENSITIVITY_MODEL",
    "SENSITIVITY_SCM_SEED",
    "build_heldout_run_identity_payload",
    "compute_heldout_run_hash12",
    "compute_heldout_run_hash_full",
    "enumerate_heldout_workload",
    "heldout_evaluation_path",
    "heldout_records_dir_path",
    "heldout_run_dir_path",
    "preflight_heldout_evaluation",
    "run_held_out_evaluation",
    "_HeldoutInfrastructureError",
]
