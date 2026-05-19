"""Minimal loader for selection-study run records.

Reads a ``run.json`` file from disk, validates its schema version
and the presence and types of every mandatory field, and returns a
frozen ``RunRecord`` dataclass carrying the parsed dictionary and
the source path. The multi-run filtering API ``load_runs`` is a
placeholder and raises ``NotImplementedError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SUPPORTED_SCHEMA_VERSION = 1


# Mandatory field name -> tuple of accepted Python types. Bool is
# rejected for any int field; if bool is acceptable it must appear
# explicitly in the tuple. The set mirrors the schema's mandatory
# fields and applies a permissive but order-of-magnitude-correct
# type check (e.g. floats may also load as ints from JSON ``0``;
# nullable fields include ``type(None)``).
_MANDATORY_FIELDS: dict[str, tuple] = {
    "run_id": (str,),
    "schema_version": (int,),
    "model": (str,),
    "condition": (str,),
    "seed_population": (str,),
    "seed_replicate_index": (int,),
    "configuration_hash": (str,),
    "graph_seed": (int,),
    "git_hash": (str,),
    "env_snapshot": (str,),
    "config_resolved": (dict, str),
    "seed_torch": (int, type(None)),
    "seed_numpy": (int, type(None)),
    "seed_dagma": (int, type(None)),
    "model_sampling_seed_base": (int,),
    "model_sampling_seed_derivation_rule": (str,),
    "train_data_seed": (int,),
    "validation_data_seed": (int, type(None)),
    "intervention_ground_truth_seed_base": (int,),
    "training_status": (str,),
    "n_iterations": (int, type(None)),
    "runtime_seconds": (int, float),
    "loss_history": (str, type(None)),
    "loss_history_status": (str,),
    "graph_status": (str,),
    "graph_status_reason": (str, type(None)),
    "thresholded_adjacency": (str,),
    "continuous_edge_object": (str,),
    "shd": (int,),
    "sid": (int,),
    "mmd_primary": (int, float, type(None)),
    "mmd_sensitivity_unit_variance": (int, float, type(None)),
    "mmd_bandwidth_sweep": (dict,),
    "validation_nll": (int, float, type(None)),
    "sampler_status": (str,),
    "sampler_status_reason": (str, type(None)),
    "sampler_policy_used": (str,),
    "mmd_available_count": (int,),
    "mmd_missing_count": (int,),
    "invalid_graph_for_this_run": (bool,),
    "shd_reversal_cost": (int,),
    "mmd_bandwidth_used_value": (dict,),
    "mmd_clip_policy": (str,),
    "sid_backend": (str,),
    "sid_backend_version": (str,),
    "sid_argument_order": (str,),
    "sid_return_value": (str,),
    "configuration_hash_algorithm": (str,),
    "wrapper_diagnostics": (dict,),
    "convergence_failure_notes": (str,),
    "wrapper_warnings": (list,),
    "interventions": (list,),
}


@dataclass(frozen=True)
class RunRecord:
    """Validated run record loaded from disk.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the ``run.json`` file the record was loaded
        from. The parent directory contains the sibling artefact
        files referenced from the record.
    data : Mapping[str, Any]
        Parsed JSON object after schema and mandatory-field
        validation.
    """

    path: Path
    data: Mapping[str, Any]


def _check_type(value: Any, accepted: tuple) -> bool:
    """Return True if ``value`` matches one of ``accepted`` types.

    ``bool`` is rejected when ``int`` is in ``accepted`` and ``bool``
    is not, because Python's ``bool`` is a subclass of ``int`` but
    booleans are not valid integer fields under the run-record
    schema.
    """
    if isinstance(value, bool) and bool not in accepted:
        return False
    return isinstance(value, accepted)


def load_run(run_path: Path | str) -> RunRecord:
    """Load and validate a run record.

    Parameters
    ----------
    run_path : pathlib.Path or str
        Either the path to a ``run.json`` file or the path to a
        directory containing one.

    Returns
    -------
    RunRecord
        Validated record.

    Raises
    ------
    FileNotFoundError
        If ``run_path`` does not resolve to an existing ``run.json``
        file.
    ValueError
        If the file is not valid JSON, the top-level value is not a
        JSON object, the ``schema_version`` field is missing or
        unsupported, or any mandatory field is missing.
    TypeError
        If any mandatory field has the wrong runtime type. The
        message names the offending field.
    """
    path = Path(run_path)
    if path.is_dir():
        json_path = path / "run.json"
    else:
        json_path = path
    if not json_path.exists():
        raise FileNotFoundError(f"run.json not found at {json_path}")
    if not json_path.is_file():
        raise FileNotFoundError(
            f"run.json path is not a regular file: {json_path}"
        )

    text = json_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"run.json at {json_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            "run.json top-level value must be a JSON object; "
            f"got {type(data).__name__}"
        )

    if "schema_version" not in data:
        raise ValueError(
            "run.json missing mandatory field 'schema_version'"
        )
    schema_version = data["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
    ):
        raise TypeError(
            "run.json field 'schema_version' must be an int (not bool); "
            f"got {type(schema_version).__name__}"
        )
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            "run.json schema_version is unsupported; "
            f"expected {_SUPPORTED_SCHEMA_VERSION}, got {schema_version}"
        )

    for name, accepted in _MANDATORY_FIELDS.items():
        if name not in data:
            raise ValueError(
                f"run.json missing mandatory field {name!r}"
            )
        if not _check_type(data[name], accepted):
            accepted_names = tuple(t.__name__ for t in accepted)
            raise TypeError(
                f"run.json field {name!r} has wrong type: expected one "
                f"of {accepted_names}, got "
                f"{type(data[name]).__name__}"
            )

    return RunRecord(path=json_path.resolve(), data=data)


def load_runs(filter_spec: Any) -> Any:
    """Load a collection of run records matching a filter.

    This API is not implemented in this commit.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "experiments.selection_study.loader.load_runs is not "
        "implemented yet."
    )
