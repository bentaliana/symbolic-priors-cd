"""Run identity and directory creation for the selection-study runner.

This module derives the canonical ``run_id`` string from an identity
tuple, derives the corresponding run-directory path, parses a
``run_id`` back into its components, creates run directories with
no-overwrite semantics, and asserts that a ``run_id`` and a
directory encode the same identity.

The run_id format is

    "<model>__<condition>__<seed_population>__seed<seed_replicate_index>__cfg<configuration_hash>"

The directory layout is

    <base_dir>/<model>/<condition>/<seed_population>/seed<seed_replicate_index>/<configuration_hash_prefix>/

where ``configuration_hash_prefix`` is the first 12 characters of the
full 64-character ``configuration_hash`` digest. The full digest is
the canonical identifier; the 12-character prefix is only the
filesystem path component.

The module imports ``VALID_MODELS``, ``VALID_CONDITIONS``, and
``VALID_SEED_POPULATIONS`` directly from
``experiments.selection_study.config`` so the enumerations have a
single source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from experiments.selection_study.config import (
    VALID_CONDITIONS,
    VALID_MODELS,
    VALID_SEED_POPULATIONS,
)


_HASH_PREFIX_LENGTH = 12
_CONFIGURATION_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_HASH_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{12}$")
_DEFAULT_BASE_DIR = Path("results/model_selection")


@dataclass(frozen=True)
class RunIdentity:
    """Identity tuple decoded from a ``run_id``.

    Attributes
    ----------
    model : str
        One of ``VALID_MODELS``.
    condition : str
        One of ``VALID_CONDITIONS``.
    seed_population : str
        One of ``VALID_SEED_POPULATIONS``.
    seed_replicate_index : int
        Within-population replicate index; non-negative.
    configuration_hash : str
        Full 64-character lowercase hexadecimal SHA-256 digest of the
        canonical JSON encoding of the resolved configuration.
    """

    model: str
    condition: str
    seed_population: str
    seed_replicate_index: int
    configuration_hash: str


def _validate_identity_inputs(
    *,
    model: str,
    condition: str,
    seed_population: str,
    seed_replicate_index: int,
    configuration_hash: str,
) -> None:
    """Validate the five identity components.

    Each invalid input raises ``ValueError`` naming the offending
    parameter and value before any string formatting or filesystem
    operation occurs.
    """
    if model not in VALID_MODELS:
        raise ValueError(
            f"parameter 'model' must be one of {VALID_MODELS}; "
            f"got {model!r}"
        )
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            "parameter 'condition' must be one of "
            f"{VALID_CONDITIONS}; got {condition!r}"
        )
    if seed_population not in VALID_SEED_POPULATIONS:
        raise ValueError(
            "parameter 'seed_population' must be one of "
            f"{VALID_SEED_POPULATIONS}; got {seed_population!r}"
        )
    if isinstance(seed_replicate_index, bool):
        raise ValueError(
            "parameter 'seed_replicate_index' must be int (not "
            f"bool); got {seed_replicate_index!r}"
        )
    if not isinstance(seed_replicate_index, int):
        raise ValueError(
            "parameter 'seed_replicate_index' must be int; got "
            f"{type(seed_replicate_index).__name__}"
        )
    if seed_replicate_index < 0:
        raise ValueError(
            "parameter 'seed_replicate_index' must be >= 0; got "
            f"{seed_replicate_index}"
        )
    if not isinstance(configuration_hash, str) or not _CONFIGURATION_HASH_PATTERN.fullmatch(
        configuration_hash
    ):
        raise ValueError(
            "parameter 'configuration_hash' must be a 64-character "
            f"lowercase hexadecimal string; got {configuration_hash!r}"
        )


def derive_run_id(
    model: str,
    condition: str,
    seed_population: str,
    seed_replicate_index: int,
    configuration_hash: str,
) -> str:
    """Return the canonical ``run_id`` string.

    Parameters
    ----------
    model : str
        One of ``VALID_MODELS``.
    condition : str
        One of ``VALID_CONDITIONS``.
    seed_population : str
        One of ``VALID_SEED_POPULATIONS``.
    seed_replicate_index : int
        Within-population replicate index; non-negative.
    configuration_hash : str
        Full 64-character lowercase hexadecimal SHA-256 digest.

    Returns
    -------
    str
        Run identifier formatted as
        ``"<model>__<condition>__<seed_population>__seed<idx>__cfg<hash>"``.
        The ``configuration_hash`` component is the FULL 64-character
        digest, not the 12-character prefix.

    Raises
    ------
    ValueError
        On any invalid component, before any string formatting.
    """
    _validate_identity_inputs(
        model=model,
        condition=condition,
        seed_population=seed_population,
        seed_replicate_index=seed_replicate_index,
        configuration_hash=configuration_hash,
    )
    return (
        f"{model}__{condition}__{seed_population}"
        f"__seed{seed_replicate_index}"
        f"__cfg{configuration_hash}"
    )


def derive_run_directory(
    model: str,
    condition: str,
    seed_population: str,
    seed_replicate_index: int,
    configuration_hash: str,
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> Path:
    """Return the run-directory path for an identity tuple.

    The layout is

        ``<base_dir>/<model>/<condition>/<seed_population>/seed<idx>/<prefix>/``

    where ``<prefix>`` is the first 12 characters of the full
    64-character ``configuration_hash``.

    Parameters
    ----------
    model, condition, seed_population, seed_replicate_index : same as
        ``derive_run_id``.
    configuration_hash : str
        Full 64-character lowercase hexadecimal SHA-256 digest.
    base_dir : pathlib.Path, optional
        Root of the run-storage tree. Defaults to
        ``Path("results/model_selection")``. The parameter exists to
        support hermetic testing under pytest's ``tmp_path`` fixture.

    Returns
    -------
    pathlib.Path
        Path representing the run directory. The directory is NOT
        created; ``derive_run_directory`` is a pure computation.

    Raises
    ------
    ValueError
        On any invalid component, before any path computation.
    """
    _validate_identity_inputs(
        model=model,
        condition=condition,
        seed_population=seed_population,
        seed_replicate_index=seed_replicate_index,
        configuration_hash=configuration_hash,
    )
    hash_prefix = configuration_hash[:_HASH_PREFIX_LENGTH]
    return (
        base_dir
        / model
        / condition
        / seed_population
        / f"seed{seed_replicate_index}"
        / hash_prefix
    )


def parse_run_id(run_id: str) -> RunIdentity:
    """Inverse of ``derive_run_id``.

    The function splits ``run_id`` on the ``"__"`` separator into
    five components, strips the ``"seed"`` and ``"cfg"`` prefixes
    from the fourth and fifth components, and validates the
    recovered values against the module's identity enumerations.

    Parameters
    ----------
    run_id : str
        Run identifier produced by ``derive_run_id``.

    Returns
    -------
    RunIdentity
        Frozen dataclass containing the recovered identity. The
        ``configuration_hash`` attribute is the FULL 64-character
        digest, not the 12-character prefix.

    Raises
    ------
    ValueError
        On any malformed component or wrong-shaped ``run_id`` string.
    """
    if not isinstance(run_id, str):
        raise ValueError(
            f"run_id must be str; got {type(run_id).__name__}"
        )
    parts = run_id.split("__")
    if len(parts) != 5:
        raise ValueError(
            "run_id must have 5 '__'-separated components "
            "(model, condition, seed_population, seed<idx>, "
            "cfg<hash>); "
            f"got {len(parts)} component(s): {run_id!r}"
        )
    model, condition, seed_population, seed_part, cfg_part = parts

    if not seed_part.startswith("seed"):
        raise ValueError(
            "run_id seed component must start with 'seed'; "
            f"got {seed_part!r}"
        )
    index_str = seed_part[len("seed"):]
    if not index_str or not index_str.isdigit():
        raise ValueError(
            "run_id seed component must be 'seed<non-negative-"
            f"integer>'; got {seed_part!r}"
        )
    seed_replicate_index = int(index_str)

    if not cfg_part.startswith("cfg"):
        raise ValueError(
            "run_id cfg component must start with 'cfg'; "
            f"got {cfg_part!r}"
        )
    configuration_hash = cfg_part[len("cfg"):]
    if not _CONFIGURATION_HASH_PATTERN.fullmatch(configuration_hash):
        raise ValueError(
            "run_id configuration_hash must be a 64-character "
            "lowercase hexadecimal string; "
            f"got {configuration_hash!r}"
        )

    if model not in VALID_MODELS:
        raise ValueError(
            "run_id 'model' component must be one of "
            f"{VALID_MODELS}; got {model!r}"
        )
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            "run_id 'condition' component must be one of "
            f"{VALID_CONDITIONS}; got {condition!r}"
        )
    if seed_population not in VALID_SEED_POPULATIONS:
        raise ValueError(
            "run_id 'seed_population' component must be one of "
            f"{VALID_SEED_POPULATIONS}; got {seed_population!r}"
        )

    return RunIdentity(
        model=model,
        condition=condition,
        seed_population=seed_population,
        seed_replicate_index=seed_replicate_index,
        configuration_hash=configuration_hash,
    )


def create_run_directory(path: Path) -> Path:
    """Create a run directory with no-overwrite semantics.

    Behaviour:

    - if ``path`` does not exist, create it (with missing parents)
      and return it;
    - if ``path`` exists and is an empty directory, return it
      unchanged;
    - if ``path`` exists and is a non-empty directory, or exists
      and is a file, raise ``FileExistsError``.

    ``create_run_directory`` does not infer or validate run identity
    from the path; identity validation is performed by
    ``derive_run_id``, ``derive_run_directory``, and
    ``assert_run_id_matches_directory``. This function is responsible
    only for no-overwrite filesystem semantics.

    Parameters
    ----------
    path : pathlib.Path
        Target directory path.

    Returns
    -------
    pathlib.Path
        The same ``path`` argument, now guaranteed to exist as a
        directory.

    Raises
    ------
    FileExistsError
        If ``path`` exists and is a non-empty directory, or exists
        and is a file.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    if not path.is_dir():
        raise FileExistsError(
            f"path exists but is not a directory: {path}"
        )
    contents = list(path.iterdir())
    if contents:
        raise FileExistsError(
            "run directory is already populated; refusing to "
            f"overwrite: {path}"
        )
    return path


def assert_run_id_matches_directory(
    run_id: str, directory: Path
) -> None:
    """Assert that ``run_id`` and ``directory`` encode the same identity.

    Parses the ``run_id`` via ``parse_run_id`` and reads the
    directory's final five path components as the identity encoding.
    Compares every component; reports every disagreement in the
    error message, not only the first one. The hash comparison is
    exact and direction-aware: the directory's final component must
    be exactly 12 lowercase hex characters and must equal the first
    12 characters of the run_id's full configuration_hash.

    Parameters
    ----------
    run_id : str
        Run identifier produced by ``derive_run_id``.
    directory : pathlib.Path
        Directory path whose final five components encode the
        identity.

    Returns
    -------
    None
        On agreement.

    Raises
    ------
    ValueError
        On any disagreement. The message names every disagreeing
        component.
    """
    identity = parse_run_id(run_id)
    parts = directory.parts
    if len(parts) < 5:
        raise ValueError(
            "directory must encode at least five identity "
            f"components; got parts={parts!r}"
        )
    dir_model = parts[-5]
    dir_condition = parts[-4]
    dir_seed_population = parts[-3]
    dir_seed_segment = parts[-2]
    dir_hash_prefix = parts[-1]

    mismatches: list[str] = []
    if dir_model != identity.model:
        mismatches.append(
            "model "
            f"(directory={dir_model!r}, run_id={identity.model!r})"
        )
    if dir_condition != identity.condition:
        mismatches.append(
            "condition "
            f"(directory={dir_condition!r}, "
            f"run_id={identity.condition!r})"
        )
    if dir_seed_population != identity.seed_population:
        mismatches.append(
            "seed_population "
            f"(directory={dir_seed_population!r}, "
            f"run_id={identity.seed_population!r})"
        )

    expected_seed_segment = f"seed{identity.seed_replicate_index}"
    if dir_seed_segment != expected_seed_segment:
        mismatches.append(
            "seed_replicate_index "
            f"(directory={dir_seed_segment!r}, "
            f"run_id={expected_seed_segment!r})"
        )

    if not _DIRECTORY_HASH_PREFIX_PATTERN.fullmatch(dir_hash_prefix):
        mismatches.append(
            "configuration_hash (directory final component="
            f"{dir_hash_prefix!r} is not a 12-character lowercase "
            "hexadecimal string)"
        )
    else:
        expected_prefix = identity.configuration_hash[
            :_HASH_PREFIX_LENGTH
        ]
        if dir_hash_prefix != expected_prefix:
            mismatches.append(
                "configuration_hash (directory prefix="
                f"{dir_hash_prefix!r}, run_id prefix="
                f"{expected_prefix!r})"
            )

    if mismatches:
        raise ValueError(
            "run_id and directory disagree on: "
            + "; ".join(mismatches)
        )
