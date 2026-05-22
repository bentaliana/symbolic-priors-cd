"""Calibration runner: workload enumeration, candidate expansion, orchestration.

This module enumerates the calibration workload from four parent
calibration Configurations, expands each parent into one executable
Configuration per sparsity grid point, and orchestrates the per-fit
calibration loop end-to-end. The orchestration glues together (a)
the per-fit fit-runner callable (injected for tests, defaulted to
the production pipeline path lazily for production), (b) per-fit
record persistence under a calibration-run directory, (c) the
within-model ranker, and (d) the selected-configurations artefact
writer.

The expanded workload is structured as 20 executable candidate
Configurations (2 models x 2 conditions x 5 grid points) combined
with the calibration seed pool (201, 202) to yield 40 fit jobs. Each
executable candidate has a distinct full configuration_hash because
its single-element calibration_configurations tuple differs from
every other candidate's; a SHA-256 collision across executable
candidates is treated as an error and surfaced by an explicit
exception rather than silently merged.

Failure-handling policy: a fit-runner exception during a model fit
becomes a degenerate calibration record with non-finite metric
fields recorded as ``None`` and the three wrapper status fields set
to ``"failed"``; the run continues so completed expensive fits are
preserved. Any condition that would make the artefact structurally
untrustworthy (malformed config, identity mismatch in a fit result,
filesystem error, writer refusal) fails fast and stops the run
without writing the selected-configurations artefact.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.config import (
    CalibrationConfiguration,
    Configuration,
    configuration_hash as compute_configuration_hash,
    load_config,
)
from experiments.selection_study.identity import derive_run_id
from experiments.selection_study.real_study import (
    assert_real_study_constants,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    DECISION_SCOPE,
    FIT_RNG_POLICY_REF,
    INTERVENTION_POLICY_REF,
    SCHEMA_VERSION,
    SEED_POPULATION_LABEL,
    SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
    SELECTED_CONFIGURATION_SEMANTICS,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
    compute_calibration_run_hash12,
    compute_calibration_run_hash_full,
    selected_configurations_path,
    write_selected_configurations,
)


_LOGGER = logging.getLogger(__name__)


_CALIBRATION_STAGE_LABEL = "calibration"
_CALIBRATION_SEED_POPULATION = "calibration"
_HASH_PREFIX_LENGTH = 12


# Orchestration constants. The four parent configuration filenames
# are pinned exactly; missing files are reported by name.
_PARENT_CONFIG_FILENAMES: tuple[str, ...] = (
    "dagma_calibration_centred_only.json",
    "dagma_calibration_standardised.json",
    "dcdi_calibration_centred_only.json",
    "dcdi_calibration_standardised.json",
)

_EXPECTED_CANDIDATE_COUNT = 20
_EXPECTED_FIT_JOB_COUNT = 40
_CALIBRATION_LOG_FILENAME = "calibration_run.log"
_PER_FIT_RECORDS_SUBDIR = "records"
_FAILED_STATUS_VALUE = "failed"
_GENERATED_AT_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The set of per-fit record fields the ranker requires. The
# orchestrator's pre-persist validator enforces presence and
# identity consistency against the job.
_REQUIRED_FIT_RESULT_FIELDS: tuple[str, ...] = (
    "model",
    "condition",
    "configuration_hash_full",
    "configuration_hash_prefix",
    "hyperparameters",
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
    "run_id",
)


@dataclass(frozen=True)
class CalibrationCandidate:
    """One executable calibration candidate.

    A candidate is the unit of work produced by expanding a parent
    calibration Configuration over its sparsity grid: it carries the
    parent's frozen real-study constants and a single sparsity grid
    point recorded both in its ``grid_point_name`` /
    ``grid_point_hyperparameter`` metadata and inside the executable
    Configuration's ``calibration_configurations`` tuple (which is a
    single-element tuple by construction).

    Attributes
    ----------
    model : str
        Either ``"dagma"`` or ``"dcdi"``. Mirrors
        ``configuration.model``.
    condition : str
        Either ``"centred_only"`` or ``"standardised"``. Mirrors
        ``configuration.condition``.
    grid_point_name : str
        Stable human-readable name of the grid point. Matches the
        ``name`` field of the underlying CalibrationConfiguration.
    grid_point_hyperparameter : tuple of (str, primitive) pairs
        The single hyperparameter override carried by this candidate,
        as an ordered tuple of ``(name, value)`` pairs. For DAGMA
        candidates this contains one ``("lambda1", value)`` entry;
        for DCDI it contains one ``("reg_coeff", value)`` entry.
    configuration : Configuration
        The executable Configuration whose
        ``calibration_configurations`` is a one-element tuple holding
        the grid point above. Its configuration_hash is distinct from
        every other candidate's by construction.
    """

    model: str
    condition: str
    grid_point_name: str
    grid_point_hyperparameter: tuple[tuple[str, Any], ...]
    configuration: Configuration

    @property
    def configuration_hash_full(self) -> str:
        """Return the full 64-character SHA-256 hex of the executable config."""
        return compute_configuration_hash(self.configuration)

    @property
    def configuration_hash_prefix(self) -> str:
        """Return the first 12 hex characters of the executable config hash."""
        return self.configuration_hash_full[:_HASH_PREFIX_LENGTH]


@dataclass(frozen=True)
class CalibrationFitJob:
    """One (candidate, calibration seed) fit job.

    A fit job is the leaf unit of work the calibration runner will
    eventually drive through the pipeline. It pairs an executable
    CalibrationCandidate with a single calibration seed value and the
    within-population replicate index that locates the seed inside
    the calibration seed pool.

    Attributes
    ----------
    candidate : CalibrationCandidate
        The executable candidate to fit.
    seed_replicate_index : int
        Within-population replicate index of the seed inside the
        candidate's calibration seed pool. Used by the existing
        identity / preflight machinery to derive per-purpose seeds.
    seed_value : int
        The integer calibration seed itself. Drawn from
        ``CalibrationWorkload.calibration_seeds``.
    """

    candidate: CalibrationCandidate
    seed_replicate_index: int
    seed_value: int


@dataclass(frozen=True)
class CalibrationWorkload:
    """The calibration workload after expansion and seed assignment.

    Attributes
    ----------
    schema_version : int
        Version integer for the workload object. Initial value 1.
    candidates : tuple of CalibrationCandidate
        The 20 executable candidates produced by expanding the
        parents (2 models x 2 conditions x 5 grid points).
    fit_jobs : tuple of CalibrationFitJob
        The 40 fit jobs produced by combining each candidate with the
        two calibration seeds.
    calibration_seeds : tuple of int
        The calibration seed values used to produce the fit jobs. For
        the frozen selection study this is exactly ``(201, 202)``.
    """

    schema_version: int
    candidates: tuple[CalibrationCandidate, ...]
    fit_jobs: tuple[CalibrationFitJob, ...]
    calibration_seeds: tuple[int, ...]


def _calibration_seed_pool(config: Configuration) -> tuple[int, ...]:
    """Return the calibration seed tuple from a Configuration.

    Raises
    ------
    ValueError
        If the Configuration does not carry a ``"calibration"``
        seed population.
    """
    for population_name, seeds in config.seed_populations:
        if population_name == _CALIBRATION_SEED_POPULATION:
            return tuple(int(s) for s in seeds)
    raise ValueError(
        "calibration Configuration must carry a 'calibration' "
        "seed population; found populations "
        f"{tuple(name for name, _ in config.seed_populations)!r}"
    )


def expand_calibration_candidates(
    parent: Configuration,
) -> tuple[CalibrationCandidate, ...]:
    """Expand a parent calibration Configuration over its sparsity grid.

    For each entry in ``parent.calibration_configurations``, produce
    one executable Configuration that is byte-identical to the parent
    in every field except ``calibration_configurations``, which is
    reduced to a single-element tuple holding the current grid point.
    Each executable Configuration is wrapped in a
    ``CalibrationCandidate`` carrying its metadata.

    The reduction is required because a Configuration whose
    ``calibration_configurations`` field holds all five grid points
    has one configuration_hash regardless of which grid point the
    runner would later fit. Producing one executable Configuration
    per grid point gives each candidate a distinct
    configuration_hash, which the downstream
    ``selected_configurations.json`` schema relies on.

    Parameters
    ----------
    parent : Configuration
        A parent calibration Configuration whose
        ``calibration_configurations`` tuple contains every sparsity
        grid point for this (model, condition) pair.

    Returns
    -------
    tuple of CalibrationCandidate
        Candidates in the order they appear in
        ``parent.calibration_configurations``.

    Raises
    ------
    ValueError
        If ``parent.calibration_configurations`` is empty.
    """
    if not parent.calibration_configurations:
        raise ValueError(
            "parent calibration Configuration must carry at least "
            "one calibration_configurations grid point; got an empty "
            "tuple"
        )

    candidates: list[CalibrationCandidate] = []
    for grid_point in parent.calibration_configurations:
        executable_config = replace(
            parent,
            calibration_configurations=(grid_point,),
        )
        candidates.append(
            CalibrationCandidate(
                model=parent.model,
                condition=parent.condition,
                grid_point_name=grid_point.name,
                grid_point_hyperparameter=tuple(grid_point.hyperparameters),
                configuration=executable_config,
            )
        )
    return tuple(candidates)


def _validate_globally_distinct_hashes(
    candidates: Sequence[CalibrationCandidate],
) -> None:
    """Raise if any two executable candidates share a configuration_hash.

    A genuine SHA-256 collision across the 20 executable candidates
    is treated as an error rather than silently merged: silently
    merging would collapse two distinct candidate rows into one
    selected_configurations entry. The check uses the full 64-character
    hash, not the 12-character prefix, to avoid false positives on
    prefix-only collisions.
    """
    seen: dict[str, CalibrationCandidate] = {}
    for candidate in candidates:
        digest = candidate.configuration_hash_full
        if digest in seen:
            existing = seen[digest]
            raise ValueError(
                "two executable calibration candidates share the "
                "same configuration_hash; this indicates either a "
                "SHA-256 collision or a logic error in candidate "
                "expansion. Offending candidates: "
                f"(model={existing.model!r}, condition="
                f"{existing.condition!r}, name="
                f"{existing.grid_point_name!r}) and "
                f"(model={candidate.model!r}, condition="
                f"{candidate.condition!r}, name="
                f"{candidate.grid_point_name!r}); shared hash="
                f"{digest!r}"
            )
        seen[digest] = candidate


def _build_fit_jobs(
    candidates: Sequence[CalibrationCandidate],
    calibration_seeds: Sequence[int],
) -> tuple[CalibrationFitJob, ...]:
    """Combine each candidate with each calibration seed once.

    The product is taken in candidate-major order: for each candidate
    in the supplied order, every calibration seed is paired with its
    within-population replicate index (the seed's position in
    ``calibration_seeds``). The replicate index is what the existing
    identity and preflight machinery use to derive per-purpose seeds.
    """
    jobs: list[CalibrationFitJob] = []
    for candidate in candidates:
        for replicate_index, seed_value in enumerate(calibration_seeds):
            jobs.append(
                CalibrationFitJob(
                    candidate=candidate,
                    seed_replicate_index=replicate_index,
                    seed_value=int(seed_value),
                )
            )
    return tuple(jobs)


def enumerate_calibration_workload(
    parents: Sequence[Configuration],
) -> CalibrationWorkload:
    """Validate the parent configs and enumerate the executable workload.

    Each parent Configuration is validated against the calibration-
    stage real-study protocol guard, expanded into per-grid-point
    executable candidates, and combined with the calibration seed
    pool to yield the full fit-job list. The function performs no
    model fits and writes no artefact.

    Parameters
    ----------
    parents : Sequence of Configuration
        The parent calibration Configurations, one per (model,
        condition) pair. For the frozen selection study this is a
        sequence of exactly four parents, but the function does not
        enforce that count here; the count is enforced at the
        workload level by the (model, condition) coverage check.

    Returns
    -------
    CalibrationWorkload
        Workload object carrying the executable candidates, the fit
        jobs, and the calibration seed pool.

    Raises
    ------
    ValueError
        If any parent fails the calibration-stage real-study guard,
        if any parent's calibration seed pool disagrees with another
        parent's, if any (model, condition) pair appears more than
        once across parents, or if two executable candidates share a
        configuration_hash.
    """
    if not parents:
        raise ValueError(
            "enumerate_calibration_workload requires at least one "
            "parent Configuration; got an empty sequence"
        )

    seen_groups: set[tuple[str, str]] = set()
    calibration_seeds: tuple[int, ...] | None = None
    all_candidates: list[CalibrationCandidate] = []
    for parent in parents:
        assert_real_study_constants(
            parent, stage=_CALIBRATION_STAGE_LABEL
        )
        group_key = (parent.model, parent.condition)
        if group_key in seen_groups:
            raise ValueError(
                "duplicate (model, condition) pair across parent "
                f"configurations: {group_key!r}"
            )
        seen_groups.add(group_key)

        parent_seeds = _calibration_seed_pool(parent)
        if calibration_seeds is None:
            calibration_seeds = parent_seeds
        elif parent_seeds != calibration_seeds:
            raise ValueError(
                "parent calibration Configurations disagree on the "
                "calibration seed pool: "
                f"{calibration_seeds!r} vs {parent_seeds!r}"
            )

        all_candidates.extend(expand_calibration_candidates(parent))

    candidates_tuple = tuple(all_candidates)
    _validate_globally_distinct_hashes(candidates_tuple)

    # mypy / static-analysis hint: at this point calibration_seeds
    # is non-None because the loop above ran at least one iteration.
    assert calibration_seeds is not None
    fit_jobs = _build_fit_jobs(candidates_tuple, calibration_seeds)

    return CalibrationWorkload(
        schema_version=1,
        candidates=candidates_tuple,
        fit_jobs=fit_jobs,
        calibration_seeds=calibration_seeds,
    )


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _load_parent_configs(config_dir: Path) -> tuple[Configuration, ...]:
    """Load the four parent calibration configs from ``config_dir``.

    Raises ``FileNotFoundError`` naming every missing file if one or
    more of the expected filenames is absent.
    """
    if not config_dir.exists():
        raise FileNotFoundError(
            "calibration config_dir does not exist: "
            f"{config_dir}"
        )
    if not config_dir.is_dir():
        raise NotADirectoryError(
            "calibration config_dir must be a directory containing "
            "the four parent calibration JSON files; got a path "
            f"that is not a directory: {config_dir}"
        )
    missing = [
        name
        for name in _PARENT_CONFIG_FILENAMES
        if not (config_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "calibration config_dir is missing required parent "
            f"config file(s): {missing}; directory={config_dir}"
        )
    return tuple(
        load_config(config_dir / name) for name in _PARENT_CONFIG_FILENAMES
    )


def _build_executable_candidate_identities(
    workload: "CalibrationWorkload",
) -> list[dict[str, Any]]:
    """Build the identity records the calibration_run_hash helper consumes.

    The helper sorts by ``(model, condition, grid_point_order,
    configuration_hash_full)`` and ignores incoming order; here the
    ``grid_point_order`` is derived from the candidate's position
    inside its ``(model, condition)`` group, with the group order
    matching the order produced by ``enumerate_calibration_workload``.
    """
    by_group: dict[tuple[str, str], list[CalibrationCandidate]] = {}
    for candidate in workload.candidates:
        by_group.setdefault(
            (candidate.model, candidate.condition), []
        ).append(candidate)
    identities: list[dict[str, Any]] = []
    for (model, condition), group_candidates in by_group.items():
        for grid_order, candidate in enumerate(group_candidates):
            identities.append(
                {
                    "model": model,
                    "condition": condition,
                    "grid_point_order": grid_order,
                    "configuration_hash_full": (
                        candidate.configuration_hash_full
                    ),
                }
            )
    return identities


def _sorted_fit_jobs(
    workload: "CalibrationWorkload",
) -> tuple["CalibrationFitJob", ...]:
    """Return the fit jobs sorted by ``(model, condition, hash, seed)``.

    Sorting on full identity yields a stable per-run iteration order
    that does not depend on Python dict iteration or workload
    construction order, so the calibration log and the on-disk record
    layout are reproducible across reruns of the same input.
    """
    return tuple(
        sorted(
            workload.fit_jobs,
            key=lambda job: (
                job.candidate.model,
                job.candidate.condition,
                job.candidate.configuration_hash_full,
                int(job.seed_value),
            ),
        )
    )


def _build_record_id(
    *,
    model: str,
    condition: str,
    configuration_hash_prefix: str,
    seed_value: int,
) -> str:
    """Return the deterministic per-fit record filename stem."""
    return (
        f"{model}_{condition}_{configuration_hash_prefix}"
        f"_seed{int(seed_value)}"
    )


def _expected_run_id_for_job(job: "CalibrationFitJob") -> str:
    """Return the canonical run_id the fit_runner output must carry."""
    candidate = job.candidate
    return derive_run_id(
        model=candidate.model,
        condition=candidate.condition,
        seed_population=_CALIBRATION_SEED_POPULATION,
        seed_replicate_index=int(job.seed_replicate_index),
        configuration_hash=candidate.configuration_hash_full,
    )


def _build_degenerate_record(
    *,
    job: "CalibrationFitJob",
    runtime_seconds: float,
    failure_type: str,
    failure_message: str,
) -> dict[str, Any]:
    """Build the structurally valid degenerate record for a failed fit."""
    candidate = job.candidate
    return {
        "model": candidate.model,
        "condition": candidate.condition,
        "configuration_hash_full": candidate.configuration_hash_full,
        "configuration_hash_prefix": candidate.configuration_hash_prefix,
        "hyperparameters": dict(candidate.grid_point_hyperparameter),
        "seed_value": int(job.seed_value),
        "shd": None,
        "sid": None,
        "mmd_primary": None,
        "graph_status": _FAILED_STATUS_VALUE,
        "sampler_status": _FAILED_STATUS_VALUE,
        "training_status": _FAILED_STATUS_VALUE,
        "runtime_seconds": float(runtime_seconds),
        "n_iterations": None,
        "threshold_metrics": [],
        "mmd_by_intervention": [],
        "bandwidth_summaries": {},
        "run_id": _expected_run_id_for_job(job),
        "failure_type": failure_type,
        "failure_message": failure_message,
    }


def _validate_fit_result(
    fit_result: Any, *, job: "CalibrationFitJob"
) -> dict[str, Any]:
    """Verify a fit_runner return value matches the job and the contract.

    Returns a normalised ``dict`` copy of the validated fit result.
    Raises ``ValueError`` if the return value is not a mapping, is
    missing required fields, or carries identity fields that do not
    match the job (wrong model, condition, hash, hash-prefix, seed,
    hyperparameters, or run_id). The orchestrator treats every
    mismatch as an infrastructure failure and fails fast.
    """
    if not isinstance(fit_result, Mapping):
        raise ValueError(
            "fit_runner result for job "
            f"(model={job.candidate.model!r}, "
            f"condition={job.candidate.condition!r}, "
            f"hash_prefix={job.candidate.configuration_hash_prefix!r}, "
            f"seed={int(job.seed_value)}) is not a mapping; got "
            f"{type(fit_result).__name__}"
        )

    missing = [
        name
        for name in _REQUIRED_FIT_RESULT_FIELDS
        if name not in fit_result
    ]
    if missing:
        raise ValueError(
            "fit_runner result for job "
            f"(model={job.candidate.model!r}, "
            f"condition={job.candidate.condition!r}, "
            f"hash_prefix={job.candidate.configuration_hash_prefix!r}, "
            f"seed={int(job.seed_value)}) is missing required "
            f"field(s): {missing}"
        )

    candidate = job.candidate
    expected_run_id = _expected_run_id_for_job(job)
    expected_hyperparameters = dict(candidate.grid_point_hyperparameter)

    identity_checks = (
        ("model", candidate.model),
        ("condition", candidate.condition),
        (
            "configuration_hash_full",
            candidate.configuration_hash_full,
        ),
        (
            "configuration_hash_prefix",
            candidate.configuration_hash_prefix,
        ),
        ("seed_value", int(job.seed_value)),
        ("run_id", expected_run_id),
    )
    for field_name, expected in identity_checks:
        observed = fit_result[field_name]
        if observed != expected:
            raise ValueError(
                "fit_runner result identity mismatch for job "
                f"(model={candidate.model!r}, "
                f"condition={candidate.condition!r}, "
                f"hash_prefix="
                f"{candidate.configuration_hash_prefix!r}, "
                f"seed={int(job.seed_value)}): field "
                f"{field_name!r} should be {expected!r}; got "
                f"{observed!r}"
            )

    observed_hyperparameters = fit_result["hyperparameters"]
    if not isinstance(observed_hyperparameters, Mapping):
        raise ValueError(
            "fit_runner result for job "
            f"hash_prefix={candidate.configuration_hash_prefix!r} "
            f"seed={int(job.seed_value)} has non-mapping "
            "hyperparameters: got "
            f"{type(observed_hyperparameters).__name__}"
        )
    if dict(observed_hyperparameters) != expected_hyperparameters:
        raise ValueError(
            "fit_runner result hyperparameters mismatch for job "
            f"(model={candidate.model!r}, "
            f"condition={candidate.condition!r}, "
            f"hash_prefix={candidate.configuration_hash_prefix!r}, "
            f"seed={int(job.seed_value)}): expected "
            f"{expected_hyperparameters!r}; got "
            f"{dict(observed_hyperparameters)!r}"
        )

    record = dict(fit_result)
    # Coerce identity fields to canonical forms in the persisted record.
    record["hyperparameters"] = expected_hyperparameters
    record["seed_value"] = int(job.seed_value)
    return record


def _atomic_write_record(
    record: Mapping[str, Any], output_path: Path
) -> None:
    """Atomically write one per-fit record to ``output_path``.

    Writes to a temporary file in the same directory, reads the
    parsed JSON back to confirm the file is valid, and atomically
    replaces ``output_path`` via ``os.replace``. On failure the
    temporary file is removed and ``output_path`` is left untouched.
    """
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    moved = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                record,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
            )
            handle.write("\n")
        with tmp_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        os.replace(tmp_path, output_path)
        moved = True
    finally:
        if not moved:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _format_utc(value: datetime) -> str:
    """Format a ``datetime`` as ``YYYY-MM-DDTHH:MM:SSZ``.

    The function does not convert timezones; callers are expected to
    pass a UTC ``datetime`` (zone-aware preferred, naive accepted).
    """
    return value.strftime(_GENERATED_AT_UTC_FORMAT)


def _default_now_fn() -> datetime:
    """Return the current time as a UTC zone-aware ``datetime``."""
    return datetime.now(tz=timezone.utc)


def _default_fit_runner(job: "CalibrationFitJob") -> dict[str, Any]:
    """Lazy-import production default fit runner.

    The real fit path is imported here, not at module top, so
    importing ``experiments.selection_study.calibration`` never
    triggers DAGMA, DCDI, or any wrapper code. The production
    adapter that drives ``pipeline.run_single_fit`` and reshapes
    the resulting ``run.json`` into the calibration-ranker input
    shape is not wired up by this module; callers that need real
    calibration execution must pass an explicit ``fit_runner`` to
    ``run_calibration`` for now.
    """
    # Lazy import to keep this module wrapper-free at import time.
    from experiments.selection_study import pipeline as _pipeline  # noqa: F401

    raise NotImplementedError(
        "the default production fit runner is not wired up; pass an "
        "explicit fit_runner to run_calibration to drive real or "
        "synthetic fits. The lazy import of "
        "experiments.selection_study.pipeline is performed inside "
        "this default so the calibration module does not pull in "
        "wrapper code at import time."
    )


class _CalibrationProgressLogger:
    """Human-readable progress logger backed by a file in the run directory.

    Writes one line per event to ``calibration_run.log`` inside the
    calibration run directory and also emits the message through the
    module logger so callers that configure standard logging see the
    same lines on stdout/stderr. The log format is documentation-
    facing only and must not be parsed by downstream consumers.
    """

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._handle = log_path.open("w", encoding="utf-8", newline="\n")

    @property
    def path(self) -> Path:
        return self._log_path

    def info(self, message: str) -> None:
        self._handle.write(message + "\n")
        self._handle.flush()
        _LOGGER.info(message)

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


def _format_fit_start_line(
    *,
    fit_index: int,
    total: int,
    job: "CalibrationFitJob",
    timestamp_utc: str,
) -> str:
    """Build the human-readable progress line emitted before a fit starts."""
    candidate = job.candidate
    hyperparameter_pairs = ", ".join(
        f"{name}={value}"
        for name, value in candidate.grid_point_hyperparameter
    )
    return (
        f"[{timestamp_utc}] fit {fit_index}/{total} START "
        f"model={candidate.model} condition={candidate.condition} "
        f"hyperparameters=({hyperparameter_pairs}) "
        f"seed={int(job.seed_value)} "
        f"hash_prefix={candidate.configuration_hash_prefix}"
    )


def _format_fit_end_line(
    *,
    fit_index: int,
    total: int,
    job: "CalibrationFitJob",
    timestamp_utc: str,
    runtime_seconds: float,
    status: str,
    past_runtimes_seconds: Sequence[float],
) -> str:
    """Build the human-readable progress line emitted after a fit ends."""
    candidate = job.candidate
    eta_segment = ""
    if past_runtimes_seconds:
        remaining = max(0, total - fit_index)
        mean_runtime = sum(past_runtimes_seconds) / len(past_runtimes_seconds)
        eta_seconds = mean_runtime * remaining
        eta_segment = f" eta_seconds={eta_seconds:.1f}"
    return (
        f"[{timestamp_utc}] fit {fit_index}/{total} END "
        f"model={candidate.model} condition={candidate.condition} "
        f"seed={int(job.seed_value)} "
        f"hash_prefix={candidate.configuration_hash_prefix} "
        f"status={status} "
        f"runtime_seconds={runtime_seconds:.3f}{eta_segment}"
    )


def _elapsed_seconds(start: datetime, end: datetime) -> float:
    """Return ``(end - start).total_seconds()`` with a non-negative floor.

    A ``now_fn`` test fake that goes backwards in time would
    otherwise produce a negative runtime; clamping to zero keeps
    the log and the persisted record values sensible.
    """
    delta = (end - start).total_seconds()
    return float(delta if delta >= 0.0 else 0.0)


# ---------------------------------------------------------------------------
# Public entry point: run_calibration
# ---------------------------------------------------------------------------


def run_calibration(
    config_dir: Path | str,
    results_root: Path | str,
    *,
    fit_runner: Callable[["CalibrationFitJob"], Mapping[str, Any]] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    force: bool = False,
) -> Path:
    """Drive end-to-end calibration and return the artefact path.

    Loads the four parent calibration configs from ``config_dir``,
    validates each against the calibration-stage real-study guard,
    enumerates the 20 executable candidates and 40 fit jobs, drives
    the per-fit loop through ``fit_runner`` (with each per-fit
    record persisted immediately under the run directory), ranks
    the collected records, and writes the selected-configurations
    handoff artefact via the artefact writer.

    Parameters
    ----------
    config_dir : Path or str
        Directory containing the four parent calibration JSON files.
    results_root : Path or str
        Root of the results tree (the artefact's calibration run
        directory will be created at
        ``<results_root>/model_selection/calibration/<hash12>/``).
    fit_runner : callable or None, optional
        Optional injection point for the per-fit runner. The
        callable receives one ``CalibrationFitJob`` and must return
        a mapping carrying the per-fit record fields the ranker
        consumes. When omitted, the production default lazily
        imports ``pipeline.run_single_fit``; callers that need
        real calibration execution should provide an explicit
        ``fit_runner``.
    now_fn : callable or None, optional
        Optional injection point for the wall-clock time source.
        When omitted, the current UTC time is used. Tests pass a
        fake ``now_fn`` so timestamps and elapsed-time fields are
        reproducible.
    force : bool, optional
        When ``True``, overwrite any pre-existing per-fit record
        file or the selected-configurations artefact at the
        canonical paths. When ``False`` (the default), the runner
        refuses to overwrite either and raises ``FileExistsError``
        before invoking any fit.

    Returns
    -------
    Path
        The on-disk path to the written
        ``selected_configurations.json``.

    Raises
    ------
    FileNotFoundError
        If ``config_dir`` does not exist or is missing one of the
        four expected parent config filenames.
    ValueError
        If any parent config fails the calibration-stage real-study
        guard, if the enumerated workload is not exactly 20
        candidates and 40 fit jobs, or if a fit_runner return value
        fails the orchestrator's structural validation.
    FileExistsError
        If a per-fit record or the artefact already exists at the
        canonical path and ``force`` is ``False``.
    """
    config_dir_path = Path(config_dir)
    results_root_path = Path(results_root)

    parent_configs = _load_parent_configs(config_dir_path)
    for index, parent in enumerate(parent_configs):
        try:
            assert_real_study_constants(
                parent, stage=_CALIBRATION_STAGE_LABEL
            )
        except ValueError as exc:
            raise ValueError(
                "calibration parent configuration at "
                f"{config_dir_path / _PARENT_CONFIG_FILENAMES[index]} "
                f"failed the calibration-stage guard: {exc}"
            ) from exc

    workload = enumerate_calibration_workload(parent_configs)
    if len(workload.candidates) != _EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            "calibration workload contains "
            f"{len(workload.candidates)} executable candidates; "
            f"expected exactly {_EXPECTED_CANDIDATE_COUNT}"
        )
    if len(workload.fit_jobs) != _EXPECTED_FIT_JOB_COUNT:
        raise ValueError(
            "calibration workload contains "
            f"{len(workload.fit_jobs)} fit jobs; expected exactly "
            f"{_EXPECTED_FIT_JOB_COUNT}"
        )

    executable_identities = _build_executable_candidate_identities(workload)
    hash_kwargs = {
        "executable_candidate_identities": executable_identities,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
    }
    calibration_run_hash_full = compute_calibration_run_hash_full(**hash_kwargs)
    calibration_run_hash12 = compute_calibration_run_hash12(**hash_kwargs)

    artefact_path = selected_configurations_path(
        calibration_run_hash12=calibration_run_hash12,
        results_root=results_root_path,
    )
    if artefact_path.exists() and not force:
        raise FileExistsError(
            "refusing to overwrite existing selected_configurations "
            f"file at {artefact_path}; pass force=True to allow "
            "overwrite. No fit_runner was invoked."
        )

    run_dir = artefact_path.parent
    records_dir = run_dir / _PER_FIT_RECORDS_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / _CALIBRATION_LOG_FILENAME

    resolved_now_fn: Callable[[], datetime] = (
        now_fn if now_fn is not None else _default_now_fn
    )
    resolved_fit_runner: Callable[
        ["CalibrationFitJob"], Mapping[str, Any]
    ] = (
        fit_runner if fit_runner is not None else _default_fit_runner
    )

    sorted_jobs = _sorted_fit_jobs(workload)
    total_jobs = len(sorted_jobs)
    records: list[dict[str, Any]] = []
    past_runtimes_seconds: list[float] = []
    progress_logger = _CalibrationProgressLogger(log_path)
    try:
        for fit_index, job in enumerate(sorted_jobs, start=1):
            candidate = job.candidate
            record_id = _build_record_id(
                model=candidate.model,
                condition=candidate.condition,
                configuration_hash_prefix=candidate.configuration_hash_prefix,
                seed_value=int(job.seed_value),
            )
            record_path = records_dir / f"{record_id}.json"
            if record_path.exists() and not force:
                raise FileExistsError(
                    "refusing to overwrite existing per-fit record "
                    f"at {record_path}; pass force=True to allow "
                    "overwrite."
                )

            start_dt = resolved_now_fn()
            progress_logger.info(
                _format_fit_start_line(
                    fit_index=fit_index,
                    total=total_jobs,
                    job=job,
                    timestamp_utc=_format_utc(start_dt),
                )
            )

            failed_with_exception = False
            failure_type: str | None = None
            failure_message: str | None = None
            raw_result: Any = None
            try:
                raw_result = resolved_fit_runner(job)
            except Exception as exc:
                failed_with_exception = True
                failure_type = type(exc).__name__
                failure_message = str(exc)

            end_dt = resolved_now_fn()
            runtime_seconds = _elapsed_seconds(start_dt, end_dt)

            if failed_with_exception:
                record = _build_degenerate_record(
                    job=job,
                    runtime_seconds=runtime_seconds,
                    failure_type=failure_type or "Exception",
                    failure_message=failure_message or "",
                )
            else:
                record = _validate_fit_result(raw_result, job=job)
                # Preserve the orchestrator-measured wall-clock time
                # when the fit_runner did not provide its own.
                if "runtime_seconds" not in record:
                    record["runtime_seconds"] = runtime_seconds

            _atomic_write_record(record, record_path)
            records.append(record)
            past_runtimes_seconds.append(runtime_seconds)

            progress_logger.info(
                _format_fit_end_line(
                    fit_index=fit_index,
                    total=total_jobs,
                    job=job,
                    timestamp_utc=_format_utc(end_dt),
                    runtime_seconds=runtime_seconds,
                    status=str(record.get("training_status", "unknown")),
                    past_runtimes_seconds=past_runtimes_seconds,
                )
            )
    finally:
        progress_logger.close()

    ranking_output = rank_calibration_records(records)

    artefact = {
        "schema_version": SCHEMA_VERSION,
        "artefact_type": SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
        "decision_scope": DECISION_SCOPE,
        "base_model_decision_made": False,
        "selected_configuration_semantics": SELECTED_CONFIGURATION_SEMANTICS,
        "calibration_run_hash_prefix": calibration_run_hash12,
        "calibration_run_hash_full": calibration_run_hash_full,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "seed_population": SEED_POPULATION_LABEL,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "selections": ranking_output["selections"],
        "candidate_ranking": ranking_output["candidate_ranking"],
        "generated_at_utc": _format_utc(resolved_now_fn()),
    }

    write_selected_configurations(artefact, artefact_path, force=force)
    return artefact_path


# ---------------------------------------------------------------------------
# Legacy placeholder for the ranking entry point that lives in
# calibration_ranking.py. The placeholder remains so existing
# scaffolding tests that exercise NotImplementedError stubs continue
# to pass; the real ranking implementation is exposed by
# ``calibration_ranking.rank_calibration_records``.
# ---------------------------------------------------------------------------


def calibration_ranking(records: Any) -> NoReturn:
    """Placeholder kept for backwards-compatibility with stub tests.

    The real within-model calibration ranking is implemented in
    ``experiments.selection_study.calibration_ranking`` and is the
    function called by ``run_calibration`` above. This placeholder
    exists only so existing scaffolding tests that exercise
    ``NotImplementedError`` stubs on ``calibration.calibration_ranking``
    keep their behaviour.

    Raises
    ------
    NotImplementedError
        Always. Callers should use
        ``calibration_ranking.rank_calibration_records`` instead.
    """
    raise NotImplementedError(
        "experiments.selection_study.calibration.calibration_ranking "
        "is not the ranking entry point; use "
        "experiments.selection_study.calibration_ranking."
        "rank_calibration_records instead."
    )


__all__ = [
    "CalibrationCandidate",
    "CalibrationFitJob",
    "CalibrationWorkload",
    "calibration_ranking",
    "enumerate_calibration_workload",
    "expand_calibration_candidates",
    "run_calibration",
]
