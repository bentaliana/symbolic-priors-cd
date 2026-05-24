"""Tests for the main-study path helpers.

All helpers are pure string operations; no filesystem I/O occurs.
Tests verify path validation, generator output, the artefact-name
allowlist, run-id validation, and the import allowlist for the
``paths`` module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments.main_study import paths as paths_mod
from experiments.main_study.paths import (
    ARTEFACT_NAMES,
    artefact_path,
    artefacts_dir,
    main_study_run_root,
    readout_dir,
    record_filename,
    records_dir,
    summary_dir,
    validate_relative_posix_path,
    validate_run_hash_prefix,
)


_VALID_PREFIX = "0123456789ab"
_VALID_RUN_ID = "prior_free__main_calibration__seed401__cfgabcdef012345"


# ---------------------------------------------------------------------------
# T-24: valid relative POSIX paths accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "results",
        "results/main_study",
        "results/main_study/abcdef012345",
        "a/b/c/d.json",
        "x.json",
    ],
)
def test_validate_relative_posix_path_accepts_valid_paths(good):
    assert validate_relative_posix_path(good) == good


# ---------------------------------------------------------------------------
# T-25: absolute paths rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/foo/bar",
        "/results/main_study",
        "C:\\foo\\bar",
        "C:/foo/bar",
        "D:/x/y",
    ],
)
def test_validate_relative_posix_path_rejects_absolute_paths(bad):
    with pytest.raises(ValueError):
        validate_relative_posix_path(bad)


# ---------------------------------------------------------------------------
# T-26: ".." components rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "..",
        "../foo",
        "foo/../bar",
        "results/../etc/passwd",
    ],
)
def test_validate_relative_posix_path_rejects_dotdot(bad):
    with pytest.raises(ValueError, match="'\\.\\.'"):
        validate_relative_posix_path(bad)


@pytest.mark.parametrize(
    "bad",
    [
        ".",
        "./foo",
        "foo/./bar",
        "results/./main_study",
    ],
)
def test_validate_relative_posix_path_rejects_single_dot(bad):
    with pytest.raises(ValueError, match="'\\.'"):
        validate_relative_posix_path(bad)


def test_path_generators_do_not_emit_dot_components():
    """No path generator emits '.' components in its output."""
    candidates = [
        main_study_run_root(_VALID_PREFIX),
        records_dir(_VALID_PREFIX),
        artefacts_dir(_VALID_PREFIX, _VALID_RUN_ID),
        summary_dir(_VALID_PREFIX),
        readout_dir(_VALID_PREFIX),
    ]
    for name in ARTEFACT_NAMES:
        candidates.append(artefact_path(_VALID_PREFIX, _VALID_RUN_ID, name))
    for p in candidates:
        parts = p.split("/")
        assert "." not in parts, (
            f"path {p!r} contains a '.' component"
        )
        assert ".." not in parts, (
            f"path {p!r} contains a '..' component"
        )


# ---------------------------------------------------------------------------
# T-27: backslashes rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "foo\\bar",
        "a\\b\\c",
        "results\\main_study",
    ],
)
def test_validate_relative_posix_path_rejects_backslashes(bad):
    with pytest.raises(ValueError, match="backslash"):
        validate_relative_posix_path(bad)


# ---------------------------------------------------------------------------
# T-28: empty components and trailing slashes rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "a//b",
        "a///b",
        "foo/",
        "results/main_study/",
    ],
)
def test_validate_relative_posix_path_rejects_empties_and_trailing(bad):
    with pytest.raises(ValueError):
        validate_relative_posix_path(bad)


def test_validate_relative_posix_path_rejects_empty_string():
    with pytest.raises(ValueError, match="empty"):
        validate_relative_posix_path("")


# ---------------------------------------------------------------------------
# T-29: all path generators produce valid relative POSIX paths
# ---------------------------------------------------------------------------


def test_path_generators_produce_valid_relative_posix_paths():
    candidates = [
        main_study_run_root(_VALID_PREFIX),
        records_dir(_VALID_PREFIX),
        artefacts_dir(_VALID_PREFIX, _VALID_RUN_ID),
        summary_dir(_VALID_PREFIX),
        readout_dir(_VALID_PREFIX),
    ]
    for name in ARTEFACT_NAMES:
        candidates.append(artefact_path(_VALID_PREFIX, _VALID_RUN_ID, name))

    for p in candidates:
        # validate_relative_posix_path raises if invalid; returns the
        # path unchanged on success.
        assert validate_relative_posix_path(p) == p


# ---------------------------------------------------------------------------
# T-30: no path generator returns a trailing slash
# ---------------------------------------------------------------------------


def test_no_generator_returns_trailing_slash():
    candidates = [
        main_study_run_root(_VALID_PREFIX),
        records_dir(_VALID_PREFIX),
        artefacts_dir(_VALID_PREFIX, _VALID_RUN_ID),
        summary_dir(_VALID_PREFIX),
        readout_dir(_VALID_PREFIX),
    ]
    for name in ARTEFACT_NAMES:
        candidates.append(artefact_path(_VALID_PREFIX, _VALID_RUN_ID, name))
    for p in candidates:
        assert not p.endswith("/"), f"trailing slash in {p!r}"


# ---------------------------------------------------------------------------
# T-31: artefact_path rejects unknown artefact names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "unknown.npz",
        "continuous_w.txt",
        "true_adjacency",
        "",
        "interventions_mmd.csv",
    ],
)
def test_artefact_path_rejects_unknown_names(bad):
    with pytest.raises(ValueError, match="ARTEFACT_NAMES"):
        artefact_path(_VALID_PREFIX, _VALID_RUN_ID, bad)


# ---------------------------------------------------------------------------
# T-32: ARTEFACT_NAMES contents
# ---------------------------------------------------------------------------


def test_artefact_names_contents():
    expected = (
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "confidence_mask.npz",
        "interventions_mmd.json",
        "prior_edge_set_clean.json",
        "prior_edge_set_corrupted.json",
        "per_edge_labels.json",
        "true_adjacency.npz",
    )
    assert ARTEFACT_NAMES == expected
    assert len(ARTEFACT_NAMES) == 8


# ---------------------------------------------------------------------------
# T-33: validate_run_hash_prefix
# ---------------------------------------------------------------------------


def test_validate_run_hash_prefix_accepts_12_lowercase_hex():
    assert validate_run_hash_prefix("0123456789ab") == "0123456789ab"
    assert validate_run_hash_prefix("abcdef012345") == "abcdef012345"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "0123456789abc",   # 13 chars
        "0123456789a",     # 11 chars
        "0123456789AB",    # uppercase
        "0123456789ZZ",    # non-hex
        "0123456789-0",    # punctuation
    ],
)
def test_validate_run_hash_prefix_rejects_invalid(bad):
    with pytest.raises(ValueError, match="prefix"):
        validate_run_hash_prefix(bad)


# ---------------------------------------------------------------------------
# T-34: record_filename validation
# ---------------------------------------------------------------------------


def test_record_filename_round_trip_for_valid_run_id():
    assert record_filename(_VALID_RUN_ID) == f"{_VALID_RUN_ID}.json"
    assert record_filename("abc_123") == "abc_123.json"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a/b",
        "a b",
        "a.b",
        ".hidden",
        "..",
        "../etc/passwd",
        "name.json",
        "with-dash",
    ],
)
def test_record_filename_rejects_invalid_run_id(bad):
    with pytest.raises(ValueError):
        record_filename(bad)


# ---------------------------------------------------------------------------
# T-35: import allowlist for paths.py
# ---------------------------------------------------------------------------


_PATHS_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "re",
    "typing",
})


def test_paths_module_imports_are_allowlisted():
    src = Path(paths_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for mod in names:
            first = mod.split(".")[0]
            assert (
                mod in _PATHS_ALLOWED_PREFIXES
                or first in _PATHS_ALLOWED_PREFIXES
            ), (
                f"paths.py import {mod!r} not in allowlist "
                f"{sorted(_PATHS_ALLOWED_PREFIXES)}."
            )
