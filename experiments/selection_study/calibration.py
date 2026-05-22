"""Calibration runner: workload enumeration and candidate expansion.

This module enumerates the calibration workload from a set of parent
calibration Configurations and expands each parent into one
executable Configuration per sparsity grid point. The enumeration is
a pure computation: it does not invoke any model fit, does not touch
the filesystem beyond Configuration validation already performed by
the caller, and does not write any artefact.

The expanded workload is structured as 20 executable candidate
Configurations (2 models x 2 conditions x 5 grid points) combined
with the calibration seed pool (201, 202) to yield 40 fit jobs. Each
executable candidate has a distinct full configuration_hash because
its single-element calibration_configurations tuple differs from
every other candidate's; a SHA-256 collision across executable
candidates is treated as an error and surfaced by an explicit
exception rather than silently merged.

Within-model ranking, the selected-configurations artefact writer,
and any real fit execution are NOT implemented here. The
public ``run_calibration`` and ``calibration_ranking`` entry points
remain placeholders that raise ``NotImplementedError`` until future work
replaces them with the corresponding orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NoReturn, Sequence

from experiments.selection_study.config import (
    CalibrationConfiguration,
    Configuration,
    configuration_hash as compute_configuration_hash,
)
from experiments.selection_study.real_study import (
    assert_real_study_constants,
)


_CALIBRATION_STAGE_LABEL = "calibration"
_CALIBRATION_SEED_POPULATION = "calibration"
_HASH_PREFIX_LENGTH = 12


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
# Placeholders for future orchestration entry points
# ---------------------------------------------------------------------------


def run_calibration(config: Any) -> NoReturn:
    """Drive real calibration fits over the enumerated workload.

    The real-fit orchestration is not part of workload enumeration.
    A separate component is responsible for wiring
    ``enumerate_calibration_workload`` through the existing pipeline
    machinery to the per-run record persistence used by the
    reproduction-pass runner; that wiring is not present in this
    module.

    Parameters
    ----------
    config : Any
        Placeholder argument; the concrete type is set by the
        orchestration component when it is introduced.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "experiments.selection_study.calibration.run_calibration is "
        "not implemented yet."
    )


def calibration_ranking(records: Any) -> NoReturn:
    """Apply the within-model calibration ranking rule.

    The within-model ranking is handled by a separate component and
    is not part of workload enumeration. When that component is
    introduced it will implement the lexicographic rule (mean SID,
    then mean MMD inside the SID tie margin, then mean SHD, then
    deterministic fallback by full configuration_hash) over
    calibration records only.

    Parameters
    ----------
    records : Any
        Placeholder argument; the concrete type is set by the
        ranking component when it is introduced.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "experiments.selection_study.calibration.calibration_ranking "
        "is not implemented yet."
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
