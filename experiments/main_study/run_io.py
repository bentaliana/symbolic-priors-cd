"""Atomic record/artefact persistence and preflight helpers.

Provides the I/O surface used to write a planned-run's record and
artefacts to disk atomically, to load existing records back into the
in-memory :class:`MainStudyRunRecord` form, and to validate that a
planned-run set is preflight-safe (unique, well-formed paths, no
escapes outside the chosen base directory). This module performs
filesystem I/O only; it does not run models, compute metrics,
orchestrate workloads, or import any wrapper / data-generator /
metric / selection-study module.

The persistence policy is documented in
:func:`persist_execution_result_atomic`: artefacts are written before
the record, and there is no rollback on mid-sequence failure. The
absence of the record file is the canonical signal of an incomplete
persistence attempt.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from experiments.main_study.executor import ExecutionResult
from experiments.main_study.paths import (
    ARTEFACT_NAMES,
    validate_relative_posix_path,
)
from experiments.main_study.records import (
    MainStudyRunRecord,
    record_from_json,
    record_to_json,
)
from experiments.main_study.workloads import PlannedRun


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------


_HEX_CHARS: frozenset[str] = frozenset("0123456789abcdef")


# Mapping from artefact filename -> MainStudyRunRecord path field. Used
# by record_artefact_paths to collect non-None artefact paths.
_ARTEFACT_NAME_TO_RECORD_FIELD: dict[str, str] = {
    "continuous_w.npz": "continuous_w_path",
    "thresholded_adjacency.npz": "thresholded_adjacency_path",
    "true_adjacency.npz": "true_adjacency_path",
    "confidence_mask.npz": "confidence_mask_path",
    "interventions_mmd.json": "interventions_mmd_path",
    "prior_edge_set_clean.json": "prior_edge_set_clean_path",
    "prior_edge_set_corrupted.json": "prior_edge_set_corrupted_path",
    "per_edge_labels.json": "per_edge_labels_path",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory.

    POSIX systems support fsync on a directory file descriptor;
    Windows does not, and ``os.open`` on a directory path raises
    ``PermissionError``. Both branches are swallowed so callers can
    rely on the surrounding write being durable on POSIX without
    needing platform-specific code.
    """
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        try:
            os.fsync(dir_fd)
        except OSError:
            return
    finally:
        os.close(dir_fd)


def _unlink_if_exists(path: Path) -> None:
    """Remove ``path`` if present, swallowing OSError silently."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_CHARS for c in value)
    )


def _is_hex12(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 12
        and all(c in _HEX_CHARS for c in value)
    )


def _to_json_safe(value: Any, *, path: str = "value") -> Any:
    """Recursively convert ``value`` to a JSON-serialisable form.

    Numpy scalars become Python scalars; numpy arrays become nested
    lists; dicts are emitted with sorted string keys; tuples become
    lists. Callable or otherwise-unsupported values raise
    ``TypeError`` with a path-aware message.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise TypeError(
                    "JSON dict keys must be strings; got "
                    f"{type(key).__name__} at {path}."
                )
            out[key] = _to_json_safe(
                value[key], path=f"{path}[{key!r}]"
            )
        return out
    if isinstance(value, (list, tuple)):
        return [
            _to_json_safe(item, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    if callable(value):
        raise TypeError(
            f"value at {path} is a callable ({type(value).__name__}); "
            "JSON payloads must not contain callables."
        )
    raise TypeError(
        f"value at {path} has unsupported type "
        f"{type(value).__name__}; cannot serialise to JSON."
    )


# ---------------------------------------------------------------------------
# Path resolution and parent-hash helpers
# ---------------------------------------------------------------------------


def resolve_relative_path(
    relative_path: str, *, base_dir: Path
) -> Path:
    """Resolve ``relative_path`` under ``base_dir``.

    Validates the relative path via
    :func:`validate_relative_posix_path` (no absolute paths, no
    backslashes, no ``..``/``.``/empty components, no trailing slash)
    and asserts the resolved path lies inside ``base_dir`` after
    symlink resolution. No directories are created.
    """
    validate_relative_posix_path(relative_path)
    if not isinstance(base_dir, Path):
        raise TypeError(
            "resolve_relative_path requires base_dir of type Path; "
            f"got {type(base_dir).__name__}."
        )
    base_resolved = base_dir.resolve()
    full = (base_dir / relative_path).resolve()
    try:
        full.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(
            f"resolved path {full!r} escapes base_dir "
            f"{base_resolved!r}."
        ) from exc
    return full


def validate_parent_hash_full(parent_hash_full: object) -> str:
    """Validate a 64-character lowercase hex string. Return it unchanged."""
    if not isinstance(parent_hash_full, str):
        raise ValueError(
            "parent_hash_full must be a string; got "
            f"{type(parent_hash_full).__name__}."
        )
    if not _is_hex64(parent_hash_full):
        raise ValueError(
            "parent_hash_full must be exactly 64 lowercase hex "
            f"characters; got {parent_hash_full!r}."
        )
    return parent_hash_full


def resolve_parent_hash_from_prefix(
    prefix12: str, *, search_root: Path
) -> str:
    """Find the unique 64-char hex name beginning with ``prefix12``.

    Looks at every entry under ``search_root``. For directories the
    name is matched directly; for files the stem (basename without
    extension) is matched. Returns the unique 64-char lowercase hex
    match. Raises ``FileNotFoundError`` for zero matches and
    ``ValueError`` for more than one.
    """
    if not _is_hex12(prefix12):
        raise ValueError(
            "prefix12 must be exactly 12 lowercase hex characters; "
            f"got {prefix12!r}."
        )
    if not isinstance(search_root, Path):
        raise TypeError(
            "search_root must be a Path; got "
            f"{type(search_root).__name__}."
        )
    if not search_root.exists() or not search_root.is_dir():
        raise FileNotFoundError(
            f"search_root {search_root!r} does not exist or is not "
            "a directory."
        )
    matches: set[str] = set()
    for entry in search_root.iterdir():
        if entry.is_dir():
            candidate = entry.name
        elif entry.is_file():
            candidate = entry.stem
        else:
            continue
        if (
            _is_hex64(candidate)
            and candidate.startswith(prefix12)
        ):
            matches.add(candidate)
    if not matches:
        raise FileNotFoundError(
            f"no 64-char lowercase hex name under {search_root!r} "
            f"starts with prefix {prefix12!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple 64-char lowercase hex names under "
            f"{search_root!r} start with prefix {prefix12!r}: "
            f"{sorted(matches)}."
        )
    return next(iter(matches))


# ---------------------------------------------------------------------------
# Planned-run uniqueness and preflight
# ---------------------------------------------------------------------------


def validate_planned_run_uniqueness(
    planned_runs: Union[list, tuple],
) -> None:
    """Verify ``planned_runs`` is non-empty and contains no duplicates.

    Checks that every entry is a :class:`PlannedRun` and that the
    configuration_hash_full, run_id, and record_path values are
    unique across the collection. Error messages name the offending
    field and value.
    """
    if not isinstance(planned_runs, (list, tuple)):
        raise TypeError(
            "planned_runs must be a list or tuple; got "
            f"{type(planned_runs).__name__}."
        )
    if len(planned_runs) == 0:
        raise ValueError(
            "planned_runs must be non-empty."
        )
    for idx, entry in enumerate(planned_runs):
        if not isinstance(entry, PlannedRun):
            raise TypeError(
                f"planned_runs[{idx}] must be a PlannedRun; got "
                f"{type(entry).__name__}."
            )
    seen_hashes: dict[str, int] = {}
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    for idx, p in enumerate(planned_runs):
        if p.configuration_hash_full in seen_hashes:
            raise ValueError(
                "duplicate configuration_hash_full in planned_runs: "
                f"{p.configuration_hash_full!r} appears at indices "
                f"{seen_hashes[p.configuration_hash_full]} and {idx}."
            )
        seen_hashes[p.configuration_hash_full] = idx
        if p.run_id in seen_ids:
            raise ValueError(
                "duplicate run_id in planned_runs: "
                f"{p.run_id!r} appears at indices "
                f"{seen_ids[p.run_id]} and {idx}."
            )
        seen_ids[p.run_id] = idx
        if p.record_path in seen_paths:
            raise ValueError(
                "duplicate record_path in planned_runs: "
                f"{p.record_path!r} appears at indices "
                f"{seen_paths[p.record_path]} and {idx}."
            )
        seen_paths[p.record_path] = idx


def prepare_output_directories(
    planned_runs: Union[list, tuple], *, base_dir: Path
) -> None:
    """Create parent directories for every planned record / artefact path.

    Calls :func:`validate_planned_run_uniqueness` first. Every path
    is required to resolve under ``base_dir``. No record or artefact
    file is written; only directories are created.
    """
    validate_planned_run_uniqueness(planned_runs)
    if not isinstance(base_dir, Path):
        raise TypeError(
            "prepare_output_directories requires base_dir of type "
            f"Path; got {type(base_dir).__name__}."
        )
    for planned in planned_runs:
        record_full = resolve_relative_path(
            planned.record_path, base_dir=base_dir
        )
        record_full.parent.mkdir(parents=True, exist_ok=True)
        for _, art_path in planned.artefact_paths.items():
            art_full = resolve_relative_path(
                art_path, base_dir=base_dir
            )
            art_full.parent.mkdir(parents=True, exist_ok=True)


def validate_preflight_for_planned_runs(
    planned_runs: Union[list, tuple],
    *,
    base_dir: Path,
    parent_hash_full: Optional[str] = None,
) -> None:
    """End-to-end preflight check for a planned-run set.

    Validates uniqueness, optionally validates ``parent_hash_full``,
    asserts every path resolves under ``base_dir``, and prepares the
    required parent directories. No record or artefact file is
    written.
    """
    validate_planned_run_uniqueness(planned_runs)
    if parent_hash_full is not None:
        validate_parent_hash_full(parent_hash_full)
    if not isinstance(base_dir, Path):
        raise TypeError(
            "validate_preflight_for_planned_runs requires base_dir "
            f"of type Path; got {type(base_dir).__name__}."
        )
    # Path-resolution check (also catches escapes early).
    for planned in planned_runs:
        resolve_relative_path(
            planned.record_path, base_dir=base_dir
        )
        for art_path in planned.artefact_paths.values():
            resolve_relative_path(art_path, base_dir=base_dir)
    prepare_output_directories(planned_runs, base_dir=base_dir)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write_text(
    text: str, relative_path: str, *, base_dir: Path
) -> Path:
    """Write ``text`` to ``relative_path`` under ``base_dir`` atomically.

    Writes a temp file in the same directory, flushes and ``fsync`` s
    the file descriptor, then uses ``os.replace`` to swap the temp
    file into place atomically. Best-effort ``fsync`` is also issued
    on the parent directory. On any failure the temp file is removed.
    """
    if not isinstance(text, str):
        raise TypeError(
            "atomic_write_text requires text to be a string; got "
            f"{type(text).__name__}."
        )
    final_path = resolve_relative_path(relative_path, base_dir=base_dir)
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(parent),
        prefix=".tmp_",
        suffix=final_path.suffix or "",
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(final_path))
    except BaseException:
        _unlink_if_exists(tmp_path)
        raise
    _try_fsync_dir(parent)
    return final_path


def atomic_write_json(
    payload: object, relative_path: str, *, base_dir: Path
) -> Path:
    """Atomically write a JSON-encoded representation of ``payload``.

    Converts numpy scalars and arrays via :func:`_to_json_safe`,
    rejects callables and other unsupported types with ``TypeError``,
    and emits canonical compact JSON (``sort_keys=True``,
    ``separators=(",", ":")``).
    """
    safe = _to_json_safe(payload, path="payload")
    text = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return atomic_write_text(text, relative_path, base_dir=base_dir)


def atomic_write_npz(
    payload: dict, relative_path: str, *, base_dir: Path
) -> Path:
    """Atomically write a NumPy ``.npz`` archive of ``payload``.

    ``payload`` must be a non-empty ``dict`` whose keys are non-empty
    strings and whose values are convertible to NumPy arrays via
    ``np.asarray``. ``relative_path`` must end with ``.npz``. The
    temp file is opened directly (not by name) and passed to
    ``np.savez`` as a binary file handle so NumPy does not auto-append
    a second ``.npz`` suffix; the result is exactly one ``.npz`` file
    at ``relative_path`` and no leftover ``.tmp.npz`` entry.
    """
    if not isinstance(relative_path, str):
        raise TypeError(
            "atomic_write_npz requires relative_path to be a string; "
            f"got {type(relative_path).__name__}."
        )
    if not relative_path.endswith(".npz"):
        raise ValueError(
            "atomic_write_npz requires relative_path to end with "
            f"'.npz'; got {relative_path!r}."
        )
    if not isinstance(payload, dict):
        raise TypeError(
            "atomic_write_npz requires payload to be a dict; got "
            f"{type(payload).__name__}."
        )
    if len(payload) == 0:
        raise ValueError(
            "atomic_write_npz requires a non-empty payload dict."
        )
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or key == "":
            raise ValueError(
                "atomic_write_npz payload keys must be non-empty "
                f"strings; got {key!r}."
            )
        try:
            arrays[key] = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"atomic_write_npz payload[{key!r}] could not be "
                f"coerced to an ndarray: {exc}"
            ) from exc

    final_path = resolve_relative_path(relative_path, base_dir=base_dir)
    parent = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(parent), prefix=".tmp_", suffix=".npz"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(final_path))
    except BaseException:
        _unlink_if_exists(tmp_path)
        raise
    _try_fsync_dir(parent)
    return final_path


# ---------------------------------------------------------------------------
# Record / artefact persistence
# ---------------------------------------------------------------------------


def persist_record_atomic(
    record: MainStudyRunRecord, record_path: str, *, base_dir: Path
) -> Path:
    """Persist a :class:`MainStudyRunRecord` via canonical JSON.

    ``record_path`` is supplied explicitly (typically from
    :class:`PlannedRun`) and is used as-is; the function does not
    derive it from the record. Calls :func:`record_to_json` then
    :func:`atomic_write_text`.
    """
    if not isinstance(record, MainStudyRunRecord):
        raise TypeError(
            "persist_record_atomic requires a MainStudyRunRecord; "
            f"got {type(record).__name__}."
        )
    text = record_to_json(record)
    return atomic_write_text(text, record_path, base_dir=base_dir)


def record_artefact_paths(
    record: MainStudyRunRecord,
) -> dict[str, str]:
    """Return artefact-filename -> relative-path for non-None record paths.

    Iterates every known artefact slot on the record and includes
    only those with a non-None path. Each path is validated via
    :func:`validate_relative_posix_path`.
    """
    if not isinstance(record, MainStudyRunRecord):
        raise TypeError(
            "record_artefact_paths requires a MainStudyRunRecord; "
            f"got {type(record).__name__}."
        )
    out: dict[str, str] = {}
    for art_name, field_name in _ARTEFACT_NAME_TO_RECORD_FIELD.items():
        if art_name not in ARTEFACT_NAMES:
            raise RuntimeError(
                f"internal: artefact name {art_name!r} is not in "
                "ARTEFACT_NAMES."
            )
        path = getattr(record, field_name)
        if path is None:
            continue
        if not isinstance(path, str):
            raise TypeError(
                f"record.{field_name} must be a string when set; got "
                f"{type(path).__name__}."
            )
        validate_relative_posix_path(path)
        out[art_name] = path
    return out


def persist_artefact_atomic(
    artefact_name: str,
    payload: object,
    relative_path: str,
    *,
    base_dir: Path,
) -> Path:
    """Dispatch artefact persistence by file extension.

    ``artefact_name`` must be in :data:`ARTEFACT_NAMES`. The basename
    of ``relative_path`` must equal ``artefact_name``. Files ending
    in ``.npz`` are routed through :func:`atomic_write_npz`; files
    ending in ``.json`` through :func:`atomic_write_json`. Unknown
    extensions raise ``ValueError``.
    """
    if artefact_name not in ARTEFACT_NAMES:
        raise ValueError(
            "persist_artefact_atomic: artefact_name "
            f"{artefact_name!r} is not in ARTEFACT_NAMES."
        )
    full = resolve_relative_path(relative_path, base_dir=base_dir)
    if full.name != artefact_name:
        raise ValueError(
            "persist_artefact_atomic: filename of relative_path "
            f"({full.name!r}) does not match artefact_name "
            f"({artefact_name!r})."
        )
    if artefact_name.endswith(".npz"):
        if not isinstance(payload, dict):
            raise TypeError(
                "persist_artefact_atomic: .npz payload must be a "
                f"dict; got {type(payload).__name__}."
            )
        return atomic_write_npz(
            payload, relative_path, base_dir=base_dir
        )
    if artefact_name.endswith(".json"):
        return atomic_write_json(
            payload, relative_path, base_dir=base_dir
        )
    raise ValueError(
        "persist_artefact_atomic: unknown extension for artefact "
        f"{artefact_name!r}; expected '.npz' or '.json'."
    )


def persist_execution_result_atomic(
    result: ExecutionResult,
    record_path: str,
    *,
    base_dir: Path,
) -> dict[str, Path]:
    """Persist a complete :class:`ExecutionResult` to disk.

    Order of operations:

    1. Validate input types.
    2. Verify ``result.artefacts`` exactly matches the artefact-path
       set declared by ``result.record`` (no missing, no extra).
    3. Persist every artefact via :func:`persist_artefact_atomic`.
    4. Persist the record last via :func:`persist_record_atomic`.

    No rollback policy: on mid-sequence failure the partially-written
    artefacts are left on disk; the absence of the record file is the
    canonical signal of incomplete persistence. Resumability and
    incomplete-run policy are deferred to a separate module.

    Returns a mapping with one entry per artefact name plus the
    special key ``"record"`` mapping to the written record path.
    """
    if not isinstance(result, ExecutionResult):
        raise TypeError(
            "persist_execution_result_atomic requires an "
            f"ExecutionResult; got {type(result).__name__}."
        )
    expected = record_artefact_paths(result.record)
    actual_keys = set(result.artefacts.keys())
    expected_keys = set(expected.keys())
    extra = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    if extra:
        raise ValueError(
            "persist_execution_result_atomic: result.artefacts "
            f"contains key(s) {sorted(extra)} that the record does "
            "not reference as non-None artefact paths."
        )
    if missing:
        raise ValueError(
            "persist_execution_result_atomic: result.artefacts is "
            f"missing key(s) {sorted(missing)} that the record "
            "references as non-None artefact paths."
        )

    written: dict[str, Path] = {}
    for name in sorted(actual_keys):
        rel_path = expected[name]
        full = persist_artefact_atomic(
            name,
            result.artefacts[name],
            rel_path,
            base_dir=base_dir,
        )
        written[name] = full

    record_full = persist_record_atomic(
        result.record, record_path, base_dir=base_dir
    )
    written["record"] = record_full
    return written


# ---------------------------------------------------------------------------
# Loading and skip-compatibility
# ---------------------------------------------------------------------------


def load_existing_record(
    record_path: str, *, base_dir: Path
) -> Optional[MainStudyRunRecord]:
    """Load and return a previously-persisted record, or ``None`` if absent.

    ``record_path`` is the relative path under ``base_dir``. Corrupt
    JSON, missing fields, or any deserialisation error surfaces as a
    ``ValueError`` whose message includes the resolved file path.
    """
    final = resolve_relative_path(record_path, base_dir=base_dir)
    if not final.exists():
        return None
    try:
        text = final.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"load_existing_record: could not read {final!r}: {exc}"
        ) from exc
    try:
        return record_from_json(text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"load_existing_record: failed to parse record at "
            f"{final!r}: {exc}"
        ) from exc


def validate_skip_compatibility(
    existing_record: MainStudyRunRecord, planned: PlannedRun
) -> None:
    """Verify a loaded record is compatible with a planned run.

    Compares ``configuration_hash_full``, ``configuration_hash_prefix``,
    ``run_id``, ``config`` (deep dataclass equality), and
    ``parent_heldout_run_hash_full``. Any mismatch raises
    ``ValueError`` naming the offending field.
    """
    if not isinstance(existing_record, MainStudyRunRecord):
        raise TypeError(
            "validate_skip_compatibility requires a "
            f"MainStudyRunRecord; got {type(existing_record).__name__}."
        )
    if not isinstance(planned, PlannedRun):
        raise TypeError(
            "validate_skip_compatibility requires a PlannedRun; "
            f"got {type(planned).__name__}."
        )
    if existing_record.configuration_hash_full != planned.configuration_hash_full:
        raise ValueError(
            "validate_skip_compatibility: configuration_hash_full "
            f"mismatch: existing {existing_record.configuration_hash_full!r} "
            f"vs planned {planned.configuration_hash_full!r}."
        )
    if existing_record.configuration_hash_prefix != planned.configuration_hash_prefix:
        raise ValueError(
            "validate_skip_compatibility: configuration_hash_prefix "
            f"mismatch: existing {existing_record.configuration_hash_prefix!r} "
            f"vs planned {planned.configuration_hash_prefix!r}."
        )
    if existing_record.run_id != planned.run_id:
        raise ValueError(
            "validate_skip_compatibility: run_id mismatch: existing "
            f"{existing_record.run_id!r} vs planned {planned.run_id!r}."
        )
    if existing_record.config != planned.config:
        raise ValueError(
            "validate_skip_compatibility: config mismatch (deep "
            "dataclass inequality between existing record and "
            "planned config); a nested field such as dagma_config or "
            "corrupted_prior_spec has changed."
        )
    if (
        existing_record.parent_heldout_run_hash_full
        != planned.config.parent_heldout_run_hash_full
    ):
        raise ValueError(
            "validate_skip_compatibility: parent_heldout_run_hash_full "
            f"mismatch: existing {existing_record.parent_heldout_run_hash_full!r} "
            f"vs planned {planned.config.parent_heldout_run_hash_full!r}."
        )


def validate_record_roundtrip(record: MainStudyRunRecord) -> None:
    """Round-trip ``record`` through JSON and assert equality.

    Calls :func:`record_to_json` then :func:`record_from_json` and
    checks the reconstructed record compares equal to the original.
    Raises ``ValueError`` on any mismatch.
    """
    if not isinstance(record, MainStudyRunRecord):
        raise TypeError(
            "validate_record_roundtrip requires a MainStudyRunRecord; "
            f"got {type(record).__name__}."
        )
    serialised = record_to_json(record)
    try:
        reconstructed = record_from_json(serialised)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(
            "validate_record_roundtrip: record_from_json failed: "
            f"{exc}"
        ) from exc
    if reconstructed != record:
        raise ValueError(
            "validate_record_roundtrip: reconstructed record does "
            "not equal the original (deep dataclass inequality)."
        )


__all__ = [
    "ARTEFACT_NAMES",
    "atomic_write_json",
    "atomic_write_npz",
    "atomic_write_text",
    "load_existing_record",
    "persist_artefact_atomic",
    "persist_execution_result_atomic",
    "persist_record_atomic",
    "prepare_output_directories",
    "record_artefact_paths",
    "resolve_parent_hash_from_prefix",
    "resolve_relative_path",
    "validate_parent_hash_full",
    "validate_planned_run_uniqueness",
    "validate_preflight_for_planned_runs",
    "validate_record_roundtrip",
    "validate_skip_compatibility",
]
