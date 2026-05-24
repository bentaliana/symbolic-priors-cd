"""Pure string-and-path helpers for the main-study artefact layout.

Every helper returns a relative POSIX path with no trailing slash.
No filesystem I/O is performed; no directories are created. Callers
are responsible for translating these strings into concrete
``pathlib.Path`` objects when they need to read or write.

Layout
------
    results/main_study/<prefix>/
        records/                    <- one JSON record per run
        artefacts/<run_id>/<name>   <- per-run artefact files
        summary/                    <- per-cell summary CSV/JSON
        readout/                    <- figures + markdown
"""

from __future__ import annotations

import re


ARTEFACT_NAMES: tuple[str, ...] = (
    "continuous_w.npz",
    "thresholded_adjacency.npz",
    "confidence_mask.npz",
    "interventions_mmd.json",
    "prior_edge_set_clean.json",
    "prior_edge_set_corrupted.json",
    "per_edge_labels.json",
    "true_adjacency.npz",
)


_HEX12_RE = re.compile(r"^[0-9a-f]{12}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
_VALID_ARTEFACT_NAMES: frozenset[str] = frozenset(ARTEFACT_NAMES)


def validate_relative_posix_path(path_str: str) -> str:
    """Validate ``path_str`` is a relative POSIX path with no traversal.

    Rejects absolute paths (POSIX or Windows-drive), backslashes,
    ``..`` components, empty components (including ``"a//b"``), and
    trailing slashes. Returns ``path_str`` unchanged on success.
    """
    if not isinstance(path_str, str):
        raise ValueError(
            "path must be a string; got "
            f"{type(path_str).__name__}."
        )
    if path_str == "":
        raise ValueError("path must not be empty.")
    if path_str.startswith("/"):
        raise ValueError(
            "path must be relative (no leading '/'); got "
            f"{path_str!r}."
        )
    if "\\" in path_str:
        raise ValueError(
            "path must not contain backslashes; got "
            f"{path_str!r}."
        )
    # Windows-drive paths like "C:/foo" or "C:\\foo" (the backslash
    # case is already rejected above).
    if len(path_str) >= 2 and path_str[1] == ":":
        raise ValueError(
            "path must not be a Windows drive path; got "
            f"{path_str!r}."
        )
    if path_str.endswith("/"):
        raise ValueError(
            "path must not end with a trailing slash; got "
            f"{path_str!r}."
        )
    for part in path_str.split("/"):
        if part == "":
            raise ValueError(
                "path contains an empty component; got "
                f"{path_str!r}."
            )
        if part == "..":
            raise ValueError(
                "path contains a '..' component; got "
                f"{path_str!r}."
            )
        if part == ".":
            raise ValueError(
                "path contains a '.' component; got "
                f"{path_str!r}."
            )
    return path_str


def validate_run_hash_prefix(prefix: str) -> str:
    """Validate ``prefix`` is exactly 12 lowercase hex characters."""
    if not isinstance(prefix, str):
        raise ValueError(
            "prefix must be a string; got "
            f"{type(prefix).__name__}."
        )
    if not _HEX12_RE.fullmatch(prefix):
        raise ValueError(
            "prefix must be exactly 12 lowercase hex characters; "
            f"got {prefix!r}."
        )
    return prefix


def _validate_run_id(run_id: str) -> str:
    """Validate ``run_id`` is a non-empty alphanumeric-underscore token.

    Rejects slashes, spaces, dots, leading dots, and empty strings.
    The underlying regex ``^[A-Za-z0-9_]+$`` rejects every disallowed
    character. Returns ``run_id`` unchanged on success.
    """
    if not isinstance(run_id, str):
        raise ValueError(
            "run_id must be a string; got "
            f"{type(run_id).__name__}."
        )
    if run_id == "":
        raise ValueError("run_id must not be empty.")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only [A-Za-z0-9_]; got "
            f"{run_id!r}."
        )
    return run_id


def main_study_run_root(prefix: str) -> str:
    """Return the run root directory under ``results/main_study``."""
    validate_run_hash_prefix(prefix)
    return validate_relative_posix_path(
        f"results/main_study/{prefix}"
    )


def records_dir(prefix: str) -> str:
    """Return the per-run records directory."""
    return validate_relative_posix_path(
        f"{main_study_run_root(prefix)}/records"
    )


def record_filename(run_id: str) -> str:
    """Return ``f"{run_id}.json"`` after validating ``run_id``."""
    _validate_run_id(run_id)
    return f"{run_id}.json"


def artefacts_dir(prefix: str, run_id: str) -> str:
    """Return the per-run artefacts directory."""
    _validate_run_id(run_id)
    return validate_relative_posix_path(
        f"{main_study_run_root(prefix)}/artefacts/{run_id}"
    )


def artefact_path(
    prefix: str, run_id: str, artefact_name: str
) -> str:
    """Return the canonical path for one artefact in a run."""
    if not isinstance(artefact_name, str):
        raise ValueError(
            "artefact_name must be a string; got "
            f"{type(artefact_name).__name__}."
        )
    if artefact_name not in _VALID_ARTEFACT_NAMES:
        raise ValueError(
            f"artefact_name {artefact_name!r} is not in "
            f"ARTEFACT_NAMES {ARTEFACT_NAMES}."
        )
    return validate_relative_posix_path(
        f"{artefacts_dir(prefix, run_id)}/{artefact_name}"
    )


def summary_dir(prefix: str) -> str:
    """Return the per-run summary directory."""
    return validate_relative_posix_path(
        f"{main_study_run_root(prefix)}/summary"
    )


def readout_dir(prefix: str) -> str:
    """Return the per-run readout directory."""
    return validate_relative_posix_path(
        f"{main_study_run_root(prefix)}/readout"
    )


__all__ = [
    "ARTEFACT_NAMES",
    "validate_relative_posix_path",
    "validate_run_hash_prefix",
    "main_study_run_root",
    "records_dir",
    "record_filename",
    "artefacts_dir",
    "artefact_path",
    "summary_dir",
    "readout_dir",
]
