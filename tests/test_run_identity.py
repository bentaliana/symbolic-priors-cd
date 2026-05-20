"""Tests for the run identity and directory utilities.

The tests cover deterministic ``run_id`` derivation, run-directory
derivation, ``parse_run_id`` inverse semantics, input validation,
no-overwrite filesystem creation, and the identity-consistency
check between a ``run_id`` and its directory path.

All filesystem tests use pytest's ``tmp_path`` fixture; no test
writes into ``results/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study import config as config_module
from experiments.selection_study import identity as identity_module
from experiments.selection_study.identity import (
    RunIdentity,
    assert_run_id_matches_directory,
    create_run_directory,
    derive_run_directory,
    derive_run_id,
    parse_run_id,
)


_VALID_HASH = "a" * 64
_OTHER_HASH = "b" * 64
_THIRD_HASH = "c" * 64
_VALID_PREFIX = _VALID_HASH[:12]


def _valid_identity_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a fresh kwargs dict for a valid identity tuple."""
    base: dict[str, Any] = {
        "model": "dagma",
        "condition": "centred_only",
        "seed_population": "calibration",
        "seed_replicate_index": 0,
        "configuration_hash": _VALID_HASH,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Identity derivation
# --------------------------------------------------------------------------- #


def test_run_id_format_matches_canonical_layout() -> None:
    """``derive_run_id`` joins identity components with ``__`` separators.

    The canonical layout is
    ``<model>__<condition>__<seed_population>__seed<idx>__cfg<hash>``.
    """
    run_id = derive_run_id(**_valid_identity_kwargs())
    expected = (
        f"dagma__centred_only__calibration__seed0__cfg{_VALID_HASH}"
    )
    assert run_id == expected


def test_run_id_contains_full_64_char_configuration_hash() -> None:
    """The ``cfg`` segment carries the full 64-character digest."""
    run_id = derive_run_id(**_valid_identity_kwargs())
    cfg_match = re.search(r"cfg([0-9a-f]+)$", run_id)
    assert cfg_match is not None
    assert len(cfg_match.group(1)) == 64
    assert cfg_match.group(1) == _VALID_HASH


def test_run_id_is_deterministic() -> None:
    """``derive_run_id`` is a pure function of its inputs."""
    kwargs = _valid_identity_kwargs()
    assert derive_run_id(**kwargs) == derive_run_id(**kwargs)


def test_parse_run_id_returns_full_configuration_hash() -> None:
    """``parse_run_id`` returns the full 64-character hash, not the prefix."""
    run_id = derive_run_id(**_valid_identity_kwargs())
    parsed = parse_run_id(run_id)
    assert isinstance(parsed, RunIdentity)
    assert parsed.configuration_hash == _VALID_HASH
    assert len(parsed.configuration_hash) == 64


def test_parse_run_id_inverts_derive_run_id() -> None:
    """``parse_run_id(derive_run_id(t)) == t`` for representative tuples."""
    cases: list[dict[str, Any]] = [
        {
            "model": "dagma",
            "condition": "centred_only",
            "seed_population": "calibration",
            "seed_replicate_index": 0,
            "configuration_hash": _VALID_HASH,
        },
        {
            "model": "dcdi",
            "condition": "standardised",
            "seed_population": "held_out_evaluation",
            "seed_replicate_index": 7,
            "configuration_hash": _OTHER_HASH,
        },
        {
            "model": "dagma",
            "condition": "standardised",
            "seed_population": "reproduction",
            "seed_replicate_index": 42,
            "configuration_hash": _THIRD_HASH,
        },
    ]
    for case in cases:
        run_id = derive_run_id(**case)
        parsed = parse_run_id(run_id)
        assert parsed.model == case["model"]
        assert parsed.condition == case["condition"]
        assert parsed.seed_population == case["seed_population"]
        assert parsed.seed_replicate_index == case["seed_replicate_index"]
        assert parsed.configuration_hash == case["configuration_hash"]


def test_parse_run_id_rejects_malformed_string() -> None:
    """Malformed ``run_id`` strings raise ``ValueError``."""
    with pytest.raises(ValueError):
        parse_run_id("not_a_valid_run_id")
    with pytest.raises(ValueError):
        parse_run_id("dagma__centred_only__calibration__seed0")
    with pytest.raises(ValueError):
        parse_run_id(
            "dagma__centred_only__calibration__seedX__cfg"
            + _VALID_HASH
        )
    with pytest.raises(ValueError):
        parse_run_id(
            "dagma__centred_only__calibration__seed0__cfg" + "a" * 12
        )


# --------------------------------------------------------------------------- #
# Directory derivation
# --------------------------------------------------------------------------- #


def test_directory_path_uses_first_12_chars_of_configuration_hash() -> None:
    """The derived directory's final component is the 12-char prefix."""
    path = derive_run_directory(**_valid_identity_kwargs())
    assert path.name == _VALID_PREFIX
    assert len(path.name) == 12


def test_directory_path_encodes_same_identity_as_run_id() -> None:
    """The derived directory and the derived run_id agree on identity."""
    kwargs = _valid_identity_kwargs(
        model="dcdi",
        seed_population="reproduction",
        seed_replicate_index=3,
    )
    run_id = derive_run_id(**kwargs)
    directory = derive_run_directory(**kwargs)
    assert_run_id_matches_directory(run_id, directory)


def test_derive_run_directory_uses_base_dir_parameter(
    tmp_path: Path,
) -> None:
    """A custom ``base_dir`` roots the derived path under ``tmp_path``."""
    custom_base = tmp_path / "alt_results"
    path = derive_run_directory(
        **_valid_identity_kwargs(), base_dir=custom_base
    )
    assert path.is_relative_to(custom_base), (
        f"derived path {path} is not relative to {custom_base}"
    )


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_derive_run_id_rejects_invalid_model() -> None:
    """An unknown model name is rejected by ``derive_run_id``."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(**_valid_identity_kwargs(model="banana"))
    assert "model" in str(excinfo.value)
    assert "banana" in str(excinfo.value)


def test_derive_run_id_rejects_invalid_condition() -> None:
    """An unknown condition is rejected by ``derive_run_id``."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(**_valid_identity_kwargs(condition="invalid"))
    assert "condition" in str(excinfo.value)
    assert "invalid" in str(excinfo.value)


def test_derive_run_id_rejects_invalid_seed_population() -> None:
    """An unknown ``seed_population`` is rejected by ``derive_run_id``."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(seed_population="banana")
        )
    assert "seed_population" in str(excinfo.value)
    assert "banana" in str(excinfo.value)


def test_derive_run_id_rejects_negative_seed_replicate_index() -> None:
    """A negative ``seed_replicate_index`` is rejected."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(seed_replicate_index=-1)
        )
    assert "seed_replicate_index" in str(excinfo.value)
    assert ">=" in str(excinfo.value)


def test_derive_run_id_rejects_boolean_seed_replicate_index() -> None:
    """``bool`` is explicitly rejected for ``seed_replicate_index``."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(seed_replicate_index=True),
        )
    message = str(excinfo.value)
    assert "seed_replicate_index" in message
    assert "bool" in message.lower()


def test_derive_run_id_rejects_short_configuration_hash() -> None:
    """The 12-char prefix is not accepted in place of the full hash."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(configuration_hash=_VALID_PREFIX)
        )
    message = str(excinfo.value)
    assert "configuration_hash" in message
    assert "64" in message


def test_derive_run_id_rejects_uppercase_configuration_hash() -> None:
    """Uppercase hex strings are rejected; the contract is lowercase."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(configuration_hash="A" * 64)
        )
    assert "configuration_hash" in str(excinfo.value)


def test_derive_run_id_rejects_non_hex_configuration_hash() -> None:
    """Non-hex characters are rejected even at the correct length."""
    with pytest.raises(ValueError) as excinfo:
        derive_run_id(
            **_valid_identity_kwargs(configuration_hash="g" * 64)
        )
    assert "configuration_hash" in str(excinfo.value)


def test_derive_run_directory_validates_same_inputs_as_derive_run_id(
    tmp_path: Path,
) -> None:
    """``derive_run_directory`` rejects the same invalid identities."""
    with pytest.raises(ValueError):
        derive_run_directory(
            **_valid_identity_kwargs(model="banana"),
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError):
        derive_run_directory(
            **_valid_identity_kwargs(seed_replicate_index=-1),
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError):
        derive_run_directory(
            **_valid_identity_kwargs(configuration_hash="short"),
            base_dir=tmp_path,
        )


# --------------------------------------------------------------------------- #
# Filesystem semantics
# --------------------------------------------------------------------------- #


def test_create_run_directory_on_fresh_path(tmp_path: Path) -> None:
    """A brand-new deep path is created successfully and stays empty."""
    target = tmp_path / "fresh" / "deep" / "path"
    result = create_run_directory(target)
    assert result == target
    assert target.exists()
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_create_run_directory_on_existing_empty_dir_succeeds(
    tmp_path: Path,
) -> None:
    """An existing empty directory is accepted unchanged."""
    target = tmp_path / "empty"
    target.mkdir()
    result = create_run_directory(target)
    assert result == target
    assert target.exists()
    assert list(target.iterdir()) == []


def test_create_run_directory_on_populated_dir_raises(
    tmp_path: Path,
) -> None:
    """A populated directory is refused; the file inside is untouched."""
    target = tmp_path / "populated"
    target.mkdir()
    sentinel = target / "existing_file.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError) as excinfo:
        create_run_directory(target)
    assert "populated" in str(excinfo.value).lower()
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserved"


def test_second_write_for_same_run_id_raises(tmp_path: Path) -> None:
    """Calling ``create_run_directory`` twice on a populated run fails."""
    base = tmp_path / "model_selection"
    target = derive_run_directory(
        **_valid_identity_kwargs(), base_dir=base
    )
    create_run_directory(target)
    sentinel = target / "run.json"
    sentinel.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run_directory(target)
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "{}"


def test_derive_run_directory_does_not_create_directory(
    tmp_path: Path,
) -> None:
    """``derive_run_directory`` is pure; it does not touch the disk."""
    base = tmp_path / "model_selection"
    target = derive_run_directory(
        **_valid_identity_kwargs(), base_dir=base
    )
    assert not target.exists()
    assert not base.exists()


def test_invalid_identity_leaves_no_partial_directory(
    tmp_path: Path,
) -> None:
    """An invalid input raises before any filesystem operation runs."""
    base = tmp_path / "model_selection_root"
    assert not base.exists()
    with pytest.raises(ValueError):
        derive_run_directory(
            **_valid_identity_kwargs(model="banana"),
            base_dir=base,
        )
    assert not base.exists()


# --------------------------------------------------------------------------- #
# Identity-consistency check
# --------------------------------------------------------------------------- #


def test_assert_run_id_matches_directory_passes_on_match(
    tmp_path: Path,
) -> None:
    """Matching ``run_id`` and directory pass without raising."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    directory = derive_run_directory(**kwargs, base_dir=tmp_path)
    assert_run_id_matches_directory(run_id, directory)


def test_assert_run_id_matches_directory_raises_on_model_mismatch(
    tmp_path: Path,
) -> None:
    """A model mismatch is reported."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "model": "dcdi"}, base_dir=tmp_path
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    assert "model" in str(excinfo.value)


def test_assert_run_id_matches_directory_raises_on_condition_mismatch(
    tmp_path: Path,
) -> None:
    """A condition mismatch is reported."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "condition": "standardised"}, base_dir=tmp_path
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    assert "condition" in str(excinfo.value)


def test_assert_run_id_matches_directory_raises_on_seed_population_mismatch(
    tmp_path: Path,
) -> None:
    """A ``seed_population`` mismatch is reported."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "seed_population": "held_out_evaluation"},
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    assert "seed_population" in str(excinfo.value)


def test_assert_run_id_matches_directory_raises_on_seed_replicate_index_mismatch(
    tmp_path: Path,
) -> None:
    """A ``seed_replicate_index`` mismatch is reported."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "seed_replicate_index": 99},
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    assert "seed_replicate_index" in str(excinfo.value)


def test_assert_run_id_matches_directory_raises_on_hash_prefix_mismatch(
    tmp_path: Path,
) -> None:
    """A ``configuration_hash`` prefix mismatch is reported."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "configuration_hash": _OTHER_HASH},
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    assert "configuration_hash" in str(excinfo.value)


def test_assert_run_id_matches_directory_reports_all_mismatching_components(
    tmp_path: Path,
) -> None:
    """Multiple mismatches are all named in the error message."""
    kwargs = _valid_identity_kwargs()
    run_id = derive_run_id(**kwargs)
    bad_directory = derive_run_directory(
        **{**kwargs, "model": "dcdi", "condition": "standardised"},
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_run_id_matches_directory(run_id, bad_directory)
    message = str(excinfo.value)
    assert "model" in message
    assert "condition" in message


# --------------------------------------------------------------------------- #
# Single-source-of-truth contract
# --------------------------------------------------------------------------- #


def test_identity_imports_valid_models_from_config() -> None:
    """``identity.VALID_MODELS`` is the same object as in ``config``."""
    assert (
        identity_module.VALID_MODELS is config_module.VALID_MODELS
    )


def test_identity_imports_valid_conditions_from_config() -> None:
    """``identity.VALID_CONDITIONS`` is the same object as in ``config``."""
    assert (
        identity_module.VALID_CONDITIONS
        is config_module.VALID_CONDITIONS
    )


def test_identity_imports_valid_seed_populations_from_config() -> None:
    """``identity.VALID_SEED_POPULATIONS`` is the same object as in ``config``."""
    assert (
        identity_module.VALID_SEED_POPULATIONS
        is config_module.VALID_SEED_POPULATIONS
    )
