"""Main-study orchestration loop with resumability policy.

Drives a sequence of :class:`PlannedRun` instances through the
single-run executor and the atomic persistence layer, applying one
of three resumability modes (``"raise"``, ``"skip"``, ``"overwrite"``)
per call. The orchestrator itself runs no model, computes no metric,
and performs no filesystem I/O directly: every disk operation is
routed through dependency-injected helpers, and every model/metric
call is routed through dependency-injected backends supplied by the
caller.

Failure policy:

* Recoverable model-side errors are :class:`ModelFitFailure` instances
  that the single-run executor catches and converts into a failure
  record before returning. The orchestrator persists that record like
  any other.
* Any other exception raised by the executor function is treated as
  infrastructure failure: the orchestrator makes a best-effort attempt
  to construct and persist a ``fit_status="infrastructure_failure_during_fit"``
  record for the affected planned run, then re-raises the original
  exception. Failure to construct or persist the infrastructure-
  failure record is logged but does not mask the original exception.
* Persistence failures (raised by the injected persistence helpers
  for an otherwise successful execution) are not caught: they
  propagate to the caller as infrastructure failure and the partial
  on-disk state follows the no-rollback policy documented in the I/O
  module.

The orchestrator returns a :class:`RunSummary` with one
:class:`WorkloadStatus` per input :class:`PlannedRun`, in the same
order as the input sequence.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from experiments.main_study.executor import (
    ExecutionResult,
    ModelFitFailure,
    execute_planned_run,
)
from experiments.main_study.records import (
    MainStudyRunRecord,
    make_failure_record,
)
from experiments.main_study.run_io import (
    load_existing_record,
    persist_execution_result_atomic,
    persist_record_atomic,
    validate_preflight_for_planned_runs,
    validate_skip_compatibility,
)
from experiments.main_study.workloads import PlannedRun


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


RUN_MODES: tuple[str, ...] = ("raise", "skip", "overwrite")


FINAL_STATUSES: tuple[str, ...] = (
    "success_computed",
    "success_metric_unavailable",
    "model_fit_failure",
    "skipped",
    "infrastructure_failure",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_non_negative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    v = float(value)
    return math.isfinite(v) and v >= 0.0


@dataclass(frozen=True, kw_only=True)
class WorkloadStatus:
    """Per-planned-run terminal state recorded by the orchestrator.

    ``final_status`` is one of :data:`FINAL_STATUSES` and reflects the
    actual scientific/run outcome (or ``"skipped"`` / ``"infrastructure_failure"``);
    it never encodes the fact that storage was overwritten. The
    boolean ``was_overwritten`` is the structured overwrite indicator
    and is orthogonal to ``final_status``. ``runtime_seconds`` is a
    finite non-negative real measuring the wall-clock time the
    orchestrator spent processing this workload (including skip-check,
    persistence, and infrastructure-record assembly). ``message`` is a
    free-form short string used for non-success outcomes and for an
    optional human-readable overwrite note; it must not be relied on
    as the machine-readable source of overwrite information.
    """

    run_id: str
    configuration_hash_prefix: str
    method_family: str
    final_status: str
    record_path: str
    runtime_seconds: float
    message: str
    was_overwritten: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("configuration_hash_prefix", self.configuration_hash_prefix),
            ("method_family", self.method_family),
            ("record_path", self.record_path),
        ):
            if not isinstance(value, str) or value == "":
                raise ValueError(
                    f"WorkloadStatus.{label} must be a non-empty "
                    f"string; got {value!r}."
                )
        if self.final_status not in FINAL_STATUSES:
            raise ValueError(
                "WorkloadStatus.final_status must be one of "
                f"{FINAL_STATUSES}; got {self.final_status!r}."
            )
        if not _is_finite_non_negative(self.runtime_seconds):
            raise ValueError(
                "WorkloadStatus.runtime_seconds must be a finite "
                f"non-negative number; got {self.runtime_seconds!r}."
            )
        if not isinstance(self.message, str):
            raise ValueError(
                "WorkloadStatus.message must be a string (empty allowed); "
                f"got {type(self.message).__name__}."
            )
        # Use type(...) is bool to reject bool subclasses and any
        # truthy/falsy non-bool value (0, 1, "true", None, ...).
        if type(self.was_overwritten) is not bool:
            raise ValueError(
                "WorkloadStatus.was_overwritten must be a bool; got "
                f"{type(self.was_overwritten).__name__}: "
                f"{self.was_overwritten!r}."
            )


@dataclass(frozen=True, kw_only=True)
class RunSummary:
    """Aggregate counts plus per-workload state for one orchestrator call.

    Invariants enforced in :meth:`__post_init__`:

    * Every count field is a non-bool ``int`` with value ``>= 0``.
    * ``n_planned == len(per_workload_status)``.
    * ``n_executed == n_success_computed + n_success_metric_unavailable
      + n_model_fit_failure``. The outcome buckets are mutually
      exclusive; ``n_overwritten`` is **not** one of them.
    * ``n_overwritten ==
      sum(1 for s in per_workload_status if s.was_overwritten)``.
      Overwrite is a storage policy outcome and overlaps freely with
      the scientific outcome buckets.
    * ``total_runtime_seconds`` is finite and ``>= 0``.
    """

    n_planned: int
    n_executed: int
    n_success_computed: int
    n_success_metric_unavailable: int
    n_model_fit_failure: int
    n_skipped: int
    n_overwritten: int
    n_infrastructure_failure: int
    total_runtime_seconds: float
    per_workload_status: tuple[WorkloadStatus, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("n_planned", self.n_planned),
            ("n_executed", self.n_executed),
            ("n_success_computed", self.n_success_computed),
            (
                "n_success_metric_unavailable",
                self.n_success_metric_unavailable,
            ),
            ("n_model_fit_failure", self.n_model_fit_failure),
            ("n_skipped", self.n_skipped),
            ("n_overwritten", self.n_overwritten),
            (
                "n_infrastructure_failure",
                self.n_infrastructure_failure,
            ),
        ):
            if not _is_plain_int(value) or value < 0:
                raise ValueError(
                    f"RunSummary.{label} must be a non-bool int >= 0; "
                    f"got {value!r}."
                )
        if not _is_finite_non_negative(self.total_runtime_seconds):
            raise ValueError(
                "RunSummary.total_runtime_seconds must be a finite "
                f"non-negative number; got {self.total_runtime_seconds!r}."
            )
        if not isinstance(self.per_workload_status, tuple):
            raise TypeError(
                "RunSummary.per_workload_status must be a tuple; got "
                f"{type(self.per_workload_status).__name__}."
            )
        for idx, entry in enumerate(self.per_workload_status):
            if not isinstance(entry, WorkloadStatus):
                raise TypeError(
                    f"RunSummary.per_workload_status[{idx}] must be a "
                    f"WorkloadStatus; got {type(entry).__name__}."
                )
        if self.n_planned != len(self.per_workload_status):
            raise ValueError(
                "RunSummary.n_planned must equal "
                f"len(per_workload_status); got {self.n_planned} vs "
                f"{len(self.per_workload_status)}."
            )
        executed_sum = (
            self.n_success_computed
            + self.n_success_metric_unavailable
            + self.n_model_fit_failure
        )
        if self.n_executed != executed_sum:
            raise ValueError(
                "RunSummary.n_executed must equal "
                "n_success_computed + n_success_metric_unavailable + "
                f"n_model_fit_failure; got n_executed={self.n_executed} "
                f"vs sum={executed_sum}."
            )
        overwritten_count = sum(
            1 for s in self.per_workload_status if s.was_overwritten
        )
        if self.n_overwritten != overwritten_count:
            raise ValueError(
                "RunSummary.n_overwritten must equal the number of "
                "per_workload_status entries with was_overwritten=True; "
                f"got n_overwritten={self.n_overwritten} vs "
                f"observed={overwritten_count}."
            )


# ---------------------------------------------------------------------------
# Type aliases for injected helpers
# ---------------------------------------------------------------------------


ExecuteFn = Callable[..., ExecutionResult]
PersistExecutionResultFn = Callable[..., dict[str, Path]]
PersistRecordFn = Callable[..., Path]
GeneratedAtUtcFactory = Callable[[], str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_mode(mode: object) -> str:
    if not isinstance(mode, str):
        raise ValueError(
            f"mode must be a string in {RUN_MODES}; got "
            f"{type(mode).__name__}."
        )
    if mode not in RUN_MODES:
        raise ValueError(
            f"mode must be one of {RUN_MODES}; got {mode!r}."
        )
    return mode


def _classify_record_final_status(record: MainStudyRunRecord) -> str:
    """Map a persisted record's status fields to a :data:`FINAL_STATUSES` value.

    Precedence:

    1. ``fit_status == "success"`` and ``metric_status == "computed"``
       -> ``"success_computed"``.
    2. ``fit_status == "success"`` and ``metric_status != "computed"``
       -> ``"success_metric_unavailable"``.
    3. ``fit_status == "model_fit_failure"`` -> ``"model_fit_failure"``.
    4. ``fit_status == "infrastructure_failure_during_fit"``
       -> ``"infrastructure_failure"``.

    Any other combination is treated as a contract violation and
    raises ``ValueError``.
    """
    if record.fit_status == "success":
        if record.metric_status == "computed":
            return "success_computed"
        return "success_metric_unavailable"
    if record.fit_status == "model_fit_failure":
        return "model_fit_failure"
    if record.fit_status == "infrastructure_failure_during_fit":
        return "infrastructure_failure"
    raise ValueError(
        "record.fit_status must be one of the documented values; got "
        f"{record.fit_status!r}."
    )


def _make_status_from_record(
    *,
    planned: PlannedRun,
    record: MainStudyRunRecord,
    final_status: str,
    runtime_seconds: float,
    message: str,
    was_overwritten: bool = False,
) -> WorkloadStatus:
    """Build a :class:`WorkloadStatus` from a persisted/loaded record.

    ``was_overwritten`` defaults to ``False``. Callers in the skip
    and infrastructure-failure branches always pass the default; the
    overwrite branch passes ``True`` only when a compatible existing
    record was atomically replaced.
    """
    return WorkloadStatus(
        run_id=planned.run_id,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        method_family=planned.config.method_family,
        final_status=final_status,
        record_path=planned.record_path,
        runtime_seconds=float(runtime_seconds),
        message=message,
        was_overwritten=was_overwritten,
    )


def _preflight_raise_mode_conflicts(
    planned_runs: tuple[PlannedRun, ...], *, base_dir: Path
) -> None:
    """Raise mode pre-check: error if any record file already exists.

    Called once before iteration begins when ``mode == "raise"``.
    Aggregates every conflicting ``record_path`` into a single
    ``FileExistsError`` so the caller sees the full list rather than
    one path at a time. Does not load or parse the records; only
    checks for filesystem presence.
    """
    conflicts: list[str] = []
    for planned in planned_runs:
        full = base_dir / planned.record_path
        if full.exists():
            conflicts.append(planned.record_path)
    if conflicts:
        raise FileExistsError(
            "mode='raise' but the following record path(s) already "
            f"exist under base_dir: {sorted(conflicts)}."
        )


def _make_infrastructure_failure_record(
    *,
    planned: PlannedRun,
    n_nodes_for_failure_record: int,
    error: BaseException,
    runtime_seconds: float,
    fit_runtime_seconds: float,
    generated_at_utc: str,
    code_version: Optional[str],
) -> MainStudyRunRecord:
    """Construct an infrastructure-failure record for one planned run.

    The record uses ``fit_status="infrastructure_failure_during_fit"``,
    ``failure_kind="infrastructure"``, and a failure message of the
    form ``"<ExceptionType>: <str(exc)>"`` (with the class name alone
    when the exception has no string form). ``wrapper_diagnostics`` is
    an empty dict, and graph/sampler/metric paths are all omitted.
    """
    type_name = type(error).__name__
    body = str(error)
    failure_message = (
        f"{type_name}: {body}" if body else type_name
    )
    return make_failure_record(
        config=planned.config,
        n_nodes=int(n_nodes_for_failure_record),
        fit_status="infrastructure_failure_during_fit",
        failure_kind="infrastructure",
        failure_message=failure_message,
        runtime_seconds=float(runtime_seconds),
        fit_runtime_seconds=float(fit_runtime_seconds),
        wrapper_diagnostics={},
        generated_at_utc=generated_at_utc,
        code_version=code_version,
    )


def _log(logger: Optional[logging.Logger], level: int, msg: str) -> None:
    """No-op when ``logger`` is None; otherwise emit at ``level``."""
    if logger is not None:
        logger.log(level, msg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_main_study(
    planned_runs: Union[list, tuple],
    *,
    base_dir: Path,
    data_loader: Callable[..., Any],
    fit_backend: Callable[..., Any],
    metric_backend: Callable[..., Any],
    mode: str,
    code_version: Optional[str],
    generated_at_utc_factory: GeneratedAtUtcFactory,
    n_nodes_for_failure_record: int,
    logger: Optional[logging.Logger] = None,
    execute_fn: ExecuteFn = execute_planned_run,
    persist_execution_result_fn: PersistExecutionResultFn = (
        persist_execution_result_atomic
    ),
    persist_record_fn: PersistRecordFn = persist_record_atomic,
) -> RunSummary:
    """Drive a sequence of planned runs end-to-end and return a summary.

    Behaviour (in order):

    1. Validate ``mode`` and basic input types.
    2. Coerce ``planned_runs`` to a tuple and verify every entry is a
       :class:`PlannedRun`.
    3. Validate ``n_nodes_for_failure_record`` is a non-bool positive int.
    4. Verify every planned run shares a single
       ``parent_heldout_run_hash_full`` (mixed-parent batches are
       refused).
    5. Run :func:`validate_preflight_for_planned_runs` once with that
       single parent hash.
    6. Iterate ``planned_runs`` in order. For each planned run:

       * ``mode == "raise"``: rely on the upfront preflight check;
         execute and persist.
       * ``mode == "skip"``: if the record file exists, load it,
         verify compatibility with the planned config, classify, and
         record as ``"skipped"``; otherwise execute and persist.
       * ``mode == "overwrite"``: if a record file already exists,
         load and validate-compatibility *before* invoking the
         executor (a corrupt or incompatible existing record raises
         and execution is not invoked); then execute and persist,
         which atomically replaces the on-disk record. The
         workload's ``final_status`` reflects the actual run outcome
         from the new execution (``success_computed``,
         ``success_metric_unavailable``, or ``model_fit_failure``);
         the structured overwrite signal lives on
         ``WorkloadStatus.was_overwritten`` and on the overlapping
         count ``RunSummary.n_overwritten``.

    7. ``execute_fn`` returns an :class:`ExecutionResult`; the
       orchestrator calls ``persist_execution_result_fn`` to write
       both artefacts and the record.
    8. If ``execute_fn`` raises anything other than the
       executor-internal :class:`ModelFitFailure` (which the executor
       has already converted to a failure record), the orchestrator
       makes a best-effort attempt to build and persist an
       infrastructure-failure record via
       ``persist_record_fn``. Best-effort here means: a secondary
       exception from record construction or persistence is logged
       but does not mask the original. The original exception is
       then re-raised.

    9. Every workload contributes one :class:`WorkloadStatus`
       collected in input order; the final :class:`RunSummary` is
       returned with the accumulated counts and
       ``total_runtime_seconds`` set to the sum of per-workload
       runtimes.
    """
    _validate_mode(mode)

    if not isinstance(base_dir, Path):
        raise TypeError(
            "base_dir must be a pathlib.Path; got "
            f"{type(base_dir).__name__}."
        )
    if not isinstance(planned_runs, (list, tuple)):
        raise TypeError(
            "planned_runs must be a list or tuple; got "
            f"{type(planned_runs).__name__}."
        )
    planned_tuple: tuple[PlannedRun, ...] = tuple(planned_runs)
    if len(planned_tuple) == 0:
        raise ValueError("planned_runs must be non-empty.")
    for idx, entry in enumerate(planned_tuple):
        if not isinstance(entry, PlannedRun):
            raise TypeError(
                f"planned_runs[{idx}] must be a PlannedRun; got "
                f"{type(entry).__name__}."
            )

    if not _is_plain_int(n_nodes_for_failure_record) or (
        n_nodes_for_failure_record <= 0
    ):
        raise ValueError(
            "n_nodes_for_failure_record must be a non-bool positive "
            f"int; got {n_nodes_for_failure_record!r}."
        )
    if not callable(generated_at_utc_factory):
        raise TypeError(
            "generated_at_utc_factory must be callable; got "
            f"{type(generated_at_utc_factory).__name__}."
        )
    # data_loader, fit_backend, and metric_backend are passed through
    # to execute_fn and never invoked directly by the orchestrator;
    # the executor is responsible for validating them. The three
    # callables the orchestrator itself invokes are validated here.
    for label, value in (
        ("execute_fn", execute_fn),
        ("persist_execution_result_fn", persist_execution_result_fn),
        ("persist_record_fn", persist_record_fn),
    ):
        if not callable(value):
            raise TypeError(
                f"{label} must be callable; got "
                f"{type(value).__name__}."
            )

    parent_hashes = {
        p.config.parent_heldout_run_hash_full for p in planned_tuple
    }
    if len(parent_hashes) != 1:
        raise ValueError(
            "all planned_runs must share a single "
            "parent_heldout_run_hash_full; got "
            f"{sorted(parent_hashes)}."
        )
    parent_hash_full = next(iter(parent_hashes))
    validate_preflight_for_planned_runs(
        planned_tuple,
        base_dir=base_dir,
        parent_hash_full=parent_hash_full,
    )

    if mode == "raise":
        _preflight_raise_mode_conflicts(
            planned_tuple, base_dir=base_dir
        )

    statuses: list[WorkloadStatus] = []
    n_executed = 0
    n_success_computed = 0
    n_success_metric_unavailable = 0
    n_model_fit_failure = 0
    n_skipped = 0
    n_overwritten = 0
    n_infrastructure_failure = 0

    for planned in planned_tuple:
        t_start = time.perf_counter()
        record_full = base_dir / planned.record_path
        existed_before = record_full.exists()
        was_overwritten = False

        if mode == "skip" and existed_before:
            try:
                existing = load_existing_record(
                    planned.record_path, base_dir=base_dir
                )
            except ValueError as exc:
                raise RuntimeError(
                    "run_main_study: mode='skip' could not load "
                    f"existing record for run_id={planned.run_id!r} "
                    f"at {planned.record_path!r}: {exc}"
                ) from exc
            if existing is None:
                # File vanished between exists() and load: treat as
                # absent and fall through to execution.
                existed_before = False
            else:
                validate_skip_compatibility(existing, planned)
                runtime = time.perf_counter() - t_start
                status = _make_status_from_record(
                    planned=planned,
                    record=existing,
                    final_status="skipped",
                    runtime_seconds=runtime,
                    message=(
                        "record already present; loaded and verified "
                        "compatible"
                    ),
                    was_overwritten=False,
                )
                statuses.append(status)
                n_skipped += 1
                _log(
                    logger,
                    logging.INFO,
                    f"run_main_study: skipped run_id={planned.run_id!r}",
                )
                continue

        if mode == "overwrite" and existed_before:
            # Pre-validate the existing record before invoking the
            # executor. Corrupt JSON or an incompatible config must
            # raise here so the caller cannot silently replace
            # mismatched on-disk state. validate_skip_compatibility
            # is the same compatibility gate the skip branch uses;
            # passing it means the existing record describes the
            # same planned configuration we are about to re-run.
            try:
                existing = load_existing_record(
                    planned.record_path, base_dir=base_dir
                )
            except ValueError as exc:
                raise RuntimeError(
                    "run_main_study: mode='overwrite' refuses to "
                    "replace a corrupt existing record for "
                    f"run_id={planned.run_id!r} at "
                    f"{planned.record_path!r}: {exc}"
                ) from exc
            if existing is None:
                existed_before = False
            else:
                validate_skip_compatibility(existing, planned)
                was_overwritten = True

        generated_at_utc = generated_at_utc_factory()
        try:
            result = execute_fn(
                planned,
                data_loader=data_loader,
                fit_backend=fit_backend,
                metric_backend=metric_backend,
                generated_at_utc=generated_at_utc,
                code_version=code_version,
            )
        except ModelFitFailure as exc:
            # By executor contract this branch is unreachable: the
            # single-run executor catches ModelFitFailure and returns
            # a failure record. We re-raise as a contract violation so
            # the misbehaviour cannot be silently absorbed.
            raise RuntimeError(
                "execute_fn leaked a ModelFitFailure to the orchestrator; "
                "the single-run executor must catch this and return a "
                f"failure record. underlying: {exc!r}"
            ) from exc
        except BaseException as exc:
            t_after_exec = time.perf_counter()
            exec_runtime = t_after_exec - t_start
            _log(
                logger,
                logging.ERROR,
                "run_main_study: infrastructure failure during "
                f"execute_fn for run_id={planned.run_id!r}: "
                f"{type(exc).__name__}: {exc}",
            )
            try:
                infra_record = _make_infrastructure_failure_record(
                    planned=planned,
                    n_nodes_for_failure_record=int(
                        n_nodes_for_failure_record
                    ),
                    error=exc,
                    runtime_seconds=float(exec_runtime),
                    fit_runtime_seconds=float(exec_runtime),
                    generated_at_utc=generated_at_utc,
                    code_version=code_version,
                )
                persist_record_fn(
                    infra_record,
                    planned.record_path,
                    base_dir=base_dir,
                )
            except BaseException as secondary:
                _log(
                    logger,
                    logging.ERROR,
                    "run_main_study: failed to persist infrastructure-"
                    f"failure record for run_id={planned.run_id!r}: "
                    f"{type(secondary).__name__}: {secondary}",
                )
            raise

        if not isinstance(result, ExecutionResult):
            raise TypeError(
                "execute_fn must return an ExecutionResult; got "
                f"{type(result).__name__}."
            )

        persist_execution_result_fn(
            result,
            planned.record_path,
            base_dir=base_dir,
        )
        runtime = time.perf_counter() - t_start

        # Classify by the actual run outcome the executor produced.
        # was_overwritten is orthogonal storage information set above.
        final_status = _classify_record_final_status(result.record)
        message = ""
        if final_status == "success_computed":
            n_success_computed += 1
        elif final_status == "success_metric_unavailable":
            n_success_metric_unavailable += 1
        elif final_status == "model_fit_failure":
            n_model_fit_failure += 1
            message = result.record.failure_message
        elif final_status == "infrastructure_failure":
            n_infrastructure_failure += 1
            message = result.record.failure_message
        n_executed += 1
        if was_overwritten:
            n_overwritten += 1
            overwrite_note = (
                "existing record atomically replaced "
                f"(new outcome: {final_status})"
            )
            if message:
                message = f"{overwrite_note}: {message}"
            else:
                message = overwrite_note
        if final_status == "model_fit_failure":
            _log(
                logger,
                logging.WARNING,
                "run_main_study: model_fit_failure for "
                f"run_id={planned.run_id!r}: {message}",
            )
        else:
            _log(
                logger,
                logging.INFO,
                "run_main_study: executed "
                f"run_id={planned.run_id!r} -> {final_status}"
                + (" (overwritten)" if was_overwritten else ""),
            )

        statuses.append(
            _make_status_from_record(
                planned=planned,
                record=result.record,
                final_status=final_status,
                runtime_seconds=runtime,
                message=message,
                was_overwritten=was_overwritten,
            )
        )

    total_runtime = sum(s.runtime_seconds for s in statuses)
    return RunSummary(
        n_planned=len(planned_tuple),
        n_executed=n_executed,
        n_success_computed=n_success_computed,
        n_success_metric_unavailable=n_success_metric_unavailable,
        n_model_fit_failure=n_model_fit_failure,
        n_skipped=n_skipped,
        n_overwritten=n_overwritten,
        n_infrastructure_failure=n_infrastructure_failure,
        total_runtime_seconds=float(total_runtime),
        per_workload_status=tuple(statuses),
    )


__all__ = [
    "FINAL_STATUSES",
    "RUN_MODES",
    "RunSummary",
    "WorkloadStatus",
    "run_main_study",
]
