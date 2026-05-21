"""Reproduction-pass runner.

Drives each model under paper-grounded defaults on its
paper-aligned reference cell. Loads a reproduction-pass
configuration file, validates it against the real-study protocol
guard, enumerates and validates the preflight manifest, runs every
``reproduction`` entry through :func:`run_single_fit`, invokes
offline threshold-robustness recomputation against each completed
run directory, and writes a reproduction-pass summary JSON.

The reproduction pass is reproduction-pass evidence only. It does
not implement calibration, held-out evaluation, prior-loss work, or
model selection; those phases live in their own modules and are
out of scope for this runner.

Per-entry failures are recorded only for the project's declared
schema-gate stop conditions
(:class:`experiments.selection_study.pipeline.SchemaGateError` and
its subclasses). Any other exception propagates unhandled, in line
with the project's no-broad-exception-swallowing policy.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from experiments.selection_study.config import (
    Configuration,
    load_config,
)
from experiments.selection_study.loader import load_run
from experiments.selection_study.pipeline import (
    SchemaGateError,
    run_single_fit,
)
from experiments.selection_study.preflight import (
    Manifest,
    ManifestEntry,
    enumerate_manifest,
    validate_manifest,
)
from experiments.selection_study.real_study import (
    assert_real_study_constants,
)
from experiments.selection_study.threshold_robustness import (
    recompute_at_thresholds,
)


_LOGGER = logging.getLogger(__name__)


_DEFAULT_OUTPUT_ROOT = Path("results/model_selection")
_SUMMARY_SUBDIR = "reproduction_pass_summary"
_SUMMARY_FILENAME = "reproduction_pass_summary.json"
_REPRODUCTION_PASS_SUMMARY_SCHEMA_VERSION = 1
_TARGET_SEED_POPULATION = "reproduction"
_NOTE_REPRODUCTION_ONLY = (
    "The reproduction pass is reproduction-pass evidence only. "
    "The summary documents that the runner completed end to end on "
    "the paper-aligned reference cell; it does not constitute base-"
    "model selection evidence and does not include calibration, "
    "held-out evaluation, or prior-loss work."
)
_REAL_STUDY_STAGE_LABEL = "reproduction_pass"


@dataclass(frozen=True)
class ReproductionPassRunRecord:
    """One reproduction-pass per-entry record.

    Attributes
    ----------
    run_id : str
        The canonical run identifier.
    seed_replicate_index : int
        The within-population replicate index.
    graph_seed : int
        The SCM construction seed for this entry.
    status : str
        Either ``"completed"`` or ``"failed"``.
    failure_type : str or None
        Exception class name for failed entries; ``None`` for
        completed entries.
    failure_message : str or None
        Exception message text for failed entries; ``None`` for
        completed entries.
    run_json_path : str or None
        POSIX path to the written ``run.json``; ``None`` for failed
        entries that did not reach a written record.
    threshold_robustness_available : bool
        ``True`` when a sibling ``threshold_robustness.json`` was
        written for this entry.
    graph_status : str or None
        Graph status read back from the run record; ``None`` for
        failed entries.
    sampler_status : str or None
        Sampler status read back from the run record; ``None`` for
        failed entries.
    training_status : str or None
        Training status read back from the run record; ``None`` for
        failed entries.
    shd : int or None
        SHD value read back from the run record; ``None`` for failed
        entries.
    sid : int or None
        SID value read back from the run record; ``None`` for failed
        entries.
    mmd_primary : float or None
        Primary MMD value read back from the run record; ``None``
        when unavailable or for failed entries.
    """

    run_id: str
    seed_replicate_index: int
    graph_seed: int
    status: str
    failure_type: Optional[str]
    failure_message: Optional[str]
    run_json_path: Optional[str]
    threshold_robustness_available: bool
    graph_status: Optional[str]
    sampler_status: Optional[str]
    training_status: Optional[str]
    shd: Optional[int]
    sid: Optional[int]
    mmd_primary: Optional[float]


@dataclass(frozen=True)
class ReproductionPassSummary:
    """Reproduction-pass summary.

    Attributes
    ----------
    schema_version : int
        Version integer for the summary schema (initial value 1).
    config_path : str
        POSIX path to the reproduction-pass configuration file
        consumed.
    model : str
        ``"dagma"`` or ``"dcdi"``.
    condition : str
        ``"centred_only"`` or ``"standardised"``.
    configuration_hash : str
        Full 64-character SHA-256 digest of the resolved config.
    seed_population : str
        Always ``"reproduction"`` for the reproduction pass.
    seed_values : tuple of int
        The reproduction seed values consumed (purely informational).
    run_ids : tuple of str
        Run identifiers for every reproduction entry, in manifest
        order.
    completed_run_count : int
        Number of entries that completed and produced a ``run.json``.
    failed_run_count : int
        Number of entries that hit a schema-gate stop condition.
    graph_status_counts : Mapping[str, int]
        Counts of ``graph_status`` values across completed runs.
    sampler_status_counts : Mapping[str, int]
        Counts of ``sampler_status`` values across completed runs.
    training_status_counts : Mapping[str, int]
        Counts of ``training_status`` values across completed runs.
    shd_values : tuple of int
        SHD values across completed runs, in run-id order.
    sid_values : tuple of int
        SID values across completed runs, in run-id order.
    mmd_primary_values : tuple of float
        Available primary MMD values across completed runs.
    threshold_robustness_available_count : int
        Number of completed runs with a sibling
        ``threshold_robustness.json``.
    records : tuple of ReproductionPassRunRecord
        Per-entry records, in manifest order.
    reproduction_pass_status : str
        One of ``"passed"``, ``"completed_with_warnings"``, or
        ``"failed_mechanical_gate"``.
    note : str
        Caveat noting that the reproduction pass is reproduction-
        pass evidence only.
    output_root : str
        POSIX path to the run-storage base directory used by the
        runner.
    summary_path : str
        POSIX path to the written summary JSON.
    """

    schema_version: int
    config_path: str
    model: str
    condition: str
    configuration_hash: str
    seed_population: str
    seed_values: tuple[int, ...]
    run_ids: tuple[str, ...]
    completed_run_count: int
    failed_run_count: int
    graph_status_counts: dict[str, int]
    sampler_status_counts: dict[str, int]
    training_status_counts: dict[str, int]
    shd_values: tuple[int, ...]
    sid_values: tuple[int, ...]
    mmd_primary_values: tuple[float, ...]
    threshold_robustness_available_count: int
    records: tuple[ReproductionPassRunRecord, ...] = field(
        default_factory=tuple
    )
    reproduction_pass_status: str = "passed"
    note: str = _NOTE_REPRODUCTION_ONLY
    output_root: str = ""
    summary_path: str = ""


def _record_to_dict(record: ReproductionPassRunRecord) -> dict[str, Any]:
    """Serialise a per-entry record to a JSON-ready dict."""
    return {
        "run_id": record.run_id,
        "seed_replicate_index": int(record.seed_replicate_index),
        "graph_seed": int(record.graph_seed),
        "status": record.status,
        "failure_type": record.failure_type,
        "failure_message": record.failure_message,
        "run_json_path": record.run_json_path,
        "threshold_robustness_available": bool(
            record.threshold_robustness_available
        ),
        "graph_status": record.graph_status,
        "sampler_status": record.sampler_status,
        "training_status": record.training_status,
        "shd": record.shd,
        "sid": record.sid,
        "mmd_primary": record.mmd_primary,
    }


def _summary_to_dict(summary: ReproductionPassSummary) -> dict[str, Any]:
    """Serialise a ReproductionPassSummary to a JSON-ready dict."""
    return {
        "schema_version": int(summary.schema_version),
        "config_path": summary.config_path,
        "model": summary.model,
        "condition": summary.condition,
        "configuration_hash": summary.configuration_hash,
        "seed_population": summary.seed_population,
        "seed_values": [int(s) for s in summary.seed_values],
        "run_ids": list(summary.run_ids),
        "completed_run_count": int(summary.completed_run_count),
        "failed_run_count": int(summary.failed_run_count),
        "graph_status_counts": dict(summary.graph_status_counts),
        "sampler_status_counts": dict(summary.sampler_status_counts),
        "training_status_counts": dict(summary.training_status_counts),
        "shd_values": [int(v) for v in summary.shd_values],
        "sid_values": [int(v) for v in summary.sid_values],
        "mmd_primary_values": [
            float(v) for v in summary.mmd_primary_values
        ],
        "threshold_robustness_available_count": int(
            summary.threshold_robustness_available_count
        ),
        "records": [_record_to_dict(r) for r in summary.records],
        "reproduction_pass_status": summary.reproduction_pass_status,
        "note": summary.note,
        "output_root": summary.output_root,
        "summary_path": summary.summary_path,
    }


def _filter_reproduction_entries(
    manifest: Manifest,
) -> tuple[ManifestEntry, ...]:
    """Return the reproduction-population entries in manifest order."""
    return tuple(
        entry
        for entry in manifest.entries
        if entry.seed_population == _TARGET_SEED_POPULATION
    )


def _reproduction_seed_values(config: Configuration) -> tuple[int, ...]:
    """Return the reproduction seed integers from the configuration."""
    for name, seeds in config.seed_populations:
        if name == _TARGET_SEED_POPULATION:
            return tuple(int(s) for s in seeds)
    return ()


def _read_completed_record_fields(
    run_json_path: Path,
) -> dict[str, Any]:
    """Read graph_status, sampler_status, training_status, SHD, SID, MMD."""
    record = load_run(run_json_path).data
    return {
        "graph_status": str(record["graph_status"]),
        "sampler_status": str(record["sampler_status"]),
        "training_status": str(record["training_status"]),
        "shd": int(record["shd"]),
        "sid": int(record["sid"]),
        "mmd_primary": (
            None
            if record["mmd_primary"] is None
            else float(record["mmd_primary"])
        ),
    }


def _execute_entry(
    entry: ManifestEntry,
    manifest: Manifest,
    entry_index: int,
    run_root: Path,
) -> ReproductionPassRunRecord:
    """Drive one manifest entry through the pipeline.

    The pipeline's declared schema-gate stop conditions are caught
    and recorded as per-entry failures. Any other exception
    propagates unhandled.
    """
    try:
        run_json_path = run_single_fit(
            manifest, entry_index, run_root=run_root
        )
    except SchemaGateError as exc:
        _LOGGER.error(
            "reproduction-pass entry %s failed with %s: %s",
            entry.expected_run_id,
            type(exc).__name__,
            exc,
        )
        return ReproductionPassRunRecord(
            run_id=entry.expected_run_id,
            seed_replicate_index=int(entry.seed_replicate_index),
            graph_seed=int(entry.graph_seed),
            status="failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            run_json_path=None,
            threshold_robustness_available=False,
            graph_status=None,
            sampler_status=None,
            training_status=None,
            shd=None,
            sid=None,
            mmd_primary=None,
        )

    fields = _read_completed_record_fields(run_json_path)
    recompute_at_thresholds(run_json_path.parent, write_sibling=True)
    threshold_robustness_path = (
        run_json_path.parent / "threshold_robustness.json"
    )

    return ReproductionPassRunRecord(
        run_id=entry.expected_run_id,
        seed_replicate_index=int(entry.seed_replicate_index),
        graph_seed=int(entry.graph_seed),
        status="completed",
        failure_type=None,
        failure_message=None,
        run_json_path=run_json_path.as_posix(),
        threshold_robustness_available=threshold_robustness_path.is_file(),
        graph_status=fields["graph_status"],
        sampler_status=fields["sampler_status"],
        training_status=fields["training_status"],
        shd=fields["shd"],
        sid=fields["sid"],
        mmd_primary=fields["mmd_primary"],
    )


def _assemble_summary(
    *,
    config: Configuration,
    config_path: Path,
    manifest: Manifest,
    records: tuple[ReproductionPassRunRecord, ...],
    output_root: Path,
) -> ReproductionPassSummary:
    """Build the ReproductionPassSummary from per-entry records."""
    completed = tuple(r for r in records if r.status == "completed")
    failed = tuple(r for r in records if r.status == "failed")

    graph_counts: Counter = Counter()
    sampler_counts: Counter = Counter()
    training_counts: Counter = Counter()
    shd_values: list[int] = []
    sid_values: list[int] = []
    mmd_primary_values: list[float] = []
    threshold_robust_count = 0
    for record in completed:
        if record.graph_status is not None:
            graph_counts[record.graph_status] += 1
        if record.sampler_status is not None:
            sampler_counts[record.sampler_status] += 1
        if record.training_status is not None:
            training_counts[record.training_status] += 1
        if record.shd is not None:
            shd_values.append(int(record.shd))
        if record.sid is not None:
            sid_values.append(int(record.sid))
        if record.mmd_primary is not None:
            mmd_primary_values.append(float(record.mmd_primary))
        if record.threshold_robustness_available:
            threshold_robust_count += 1

    all_completed_valid_dag = all(
        r.graph_status == "valid_dag" for r in completed
    )
    all_completed_sampler_available = all(
        r.sampler_status == "available" for r in completed
    )
    all_completed_threshold_robust = all(
        r.threshold_robustness_available for r in completed
    )
    if (
        len(failed) == 0
        and all_completed_valid_dag
        and all_completed_sampler_available
        and all_completed_threshold_robust
    ):
        status = "passed"
    else:
        status = "completed_with_warnings"

    # Run directories follow the existing derive_run_directory
    # convention (hash prefix folder). The reproduction-pass summary
    # uses the full configuration_hash for its directory component
    # so the run-set-level artefact remains unambiguous if two
    # configurations ever share the same 12-character prefix.
    summary_dir = (
        output_root / _SUMMARY_SUBDIR / manifest.configuration_hash
    )
    summary_path = summary_dir / _SUMMARY_FILENAME

    return ReproductionPassSummary(
        schema_version=_REPRODUCTION_PASS_SUMMARY_SCHEMA_VERSION,
        config_path=Path(config_path).as_posix(),
        model=config.model,
        condition=config.condition,
        configuration_hash=manifest.configuration_hash,
        seed_population=_TARGET_SEED_POPULATION,
        seed_values=_reproduction_seed_values(config),
        run_ids=tuple(r.run_id for r in records),
        completed_run_count=len(completed),
        failed_run_count=len(failed),
        graph_status_counts=dict(graph_counts),
        sampler_status_counts=dict(sampler_counts),
        training_status_counts=dict(training_counts),
        shd_values=tuple(shd_values),
        sid_values=tuple(sid_values),
        mmd_primary_values=tuple(mmd_primary_values),
        threshold_robustness_available_count=threshold_robust_count,
        records=records,
        reproduction_pass_status=status,
        note=_NOTE_REPRODUCTION_ONLY,
        output_root=output_root.as_posix(),
        summary_path=summary_path.as_posix(),
    )


def _write_summary(summary: ReproductionPassSummary) -> Path:
    """Write the reproduction-pass summary JSON to its canonical path."""
    summary_path = Path(summary.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _summary_to_dict(summary),
        sort_keys=True,
        ensure_ascii=True,
        indent=2,
    )
    summary_path.write_text(payload, encoding="utf-8")
    return summary_path


def run_reproduction_pass(
    config_path: Path | str,
    *,
    output_root: Path | None = None,
) -> ReproductionPassSummary:
    """Run the reproduction pass.

    Loads ``config_path``, validates the resolved configuration
    against the real-study protocol guard, enumerates and validates
    the preflight manifest, runs every ``reproduction``-population
    entry through the schema-conformance pipeline, invokes offline
    threshold-robustness recomputation against each completed run
    directory, and writes a reproduction-pass summary JSON.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Path to the reproduction-pass configuration JSON file.
    output_root : pathlib.Path or None, optional
        Run-storage base directory. When ``None`` the default
        ``Path("results/model_selection")`` is used. Tests pass a
        ``tmp_path``-relative path to keep filesystem operations
        hermetic.

    Returns
    -------
    ReproductionPassSummary
        Summary of the reproduction pass. The summary is also
        written to disk at
        ``<output_root>/reproduction_pass_summary/<configuration_hash>/
        reproduction_pass_summary.json``.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist on disk.
    ValueError
        If the configuration fails the real-study protocol guard,
        the preflight manifest fails validation, or no reproduction
        entries are present after enumeration.
    """
    config_path_obj = Path(config_path)
    config = load_config(config_path_obj)
    assert_real_study_constants(config, stage=_REAL_STUDY_STAGE_LABEL)

    base_dir = (
        Path(output_root) if output_root is not None else _DEFAULT_OUTPUT_ROOT
    )

    manifest = enumerate_manifest(config, base_dir=base_dir)
    validate_manifest(manifest, hash_recheck_config=config)

    reproduction_entries = _filter_reproduction_entries(manifest)
    if not reproduction_entries:
        raise ValueError(
            "reproduction-pass manifest contains no reproduction "
            "entries; the configuration must carry a non-empty "
            "'reproduction' seed population"
        )

    entry_indices = {
        entry.expected_run_id: idx
        for idx, entry in enumerate(manifest.entries)
    }

    records: list[ReproductionPassRunRecord] = []
    for entry in reproduction_entries:
        entry_index = entry_indices[entry.expected_run_id]
        record = _execute_entry(entry, manifest, entry_index, base_dir)
        records.append(record)

    summary = _assemble_summary(
        config=config,
        config_path=config_path_obj,
        manifest=manifest,
        records=tuple(records),
        output_root=base_dir,
    )
    _write_summary(summary)
    return summary


__all__ = [
    "ReproductionPassRunRecord",
    "ReproductionPassSummary",
    "run_reproduction_pass",
]
