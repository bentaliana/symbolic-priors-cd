"""Single-fit schema-conformance pipeline for the selection-study runner.

Drives one preflight-validated manifest entry through a toy SCM, a
wrapper fit, and the canonical run-record schema, then writes the
artefacts and the run.json file to disk. The toy constants defined
below exist only to exercise schema conformance; they are not the
selection-study constants and have no scientific meaning.

The wrapper class is resolved at runtime via importlib so that this
module does not directly import any wrapper class. Preprocessing
classes and the DCDI configuration dataclass are also imported
lazily through importlib for the same reason.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiments.selection_study.config import (
    SEED_DERIVATION_RULE_NAME,
)
from experiments.selection_study.identity import (
    assert_run_id_matches_directory,
    create_run_directory,
    derive_run_directory,
    derive_run_id,
)
from experiments.selection_study.preflight import Manifest
from experiments.selection_study.sampling import (
    compute_per_intervention_records,
)
from symbolic_priors_cd.data import (
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.metrics import shd, sid_score


# Schema-conformance gate constants. These exist only to drive the
# schema-conformance gate test pipeline; they are not the
# selection-study constants and have no scientific meaning.
SCHEMA_GATE_N_NODES = 3
SCHEMA_GATE_EXPECTED_EDGES = 3
SCHEMA_GATE_N_TRAIN = 64
SCHEMA_GATE_N_VAL_DCDI = 32
SCHEMA_GATE_DCDI_N_ITER = 30
SCHEMA_GATE_DCDI_CONFIG_KWARGS = {
    "stop_crit_win": 10,
    "train_batch_size": 8,
}

_SCHEMA_VERSION = 1
_SHD_REVERSAL_COST = 2
_SID_BACKEND = "gadjid"
_SID_BACKEND_VERSION = "0.1.0"
_SID_ARGUMENT_ORDER = "predicted_then_true"
_SID_RETURN_VALUE = "raw_mistake_count"
_MMD_CLIP_POLICY = "no_clip"
_CONFIGURATION_HASH_ALGORITHM = "sha256_canonical_json_sorted_keys"

_PREPROCESSING_MODULE = "symbolic_priors_cd.wrappers.preprocessing"
_DCDI_MODULE = "symbolic_priors_cd.wrappers.dcdi"


# ---------------------------------------------------------------------------
# Stop-condition exception types
# ---------------------------------------------------------------------------


class SchemaGateError(RuntimeError):
    """Base class for schema-conformance gate stop conditions."""


class InvalidGraphForSchemaGateError(SchemaGateError):
    """Raised when a toy fit returns a non-valid_dag thresholded graph.

    SID requires a valid DAG and the schema requires sid as a plain
    integer. The gate refuses to write a partial record (sid=None
    would violate the schema) and does not repair the graph.
    """


class DcdiSeedMismatchError(SchemaGateError):
    """Raised when DCDI seed_torch and seed_numpy are unequal or null.

    DCDIWrapper.fit accepts a single optimisation seed. Recording
    distinct seed_torch and seed_numpy values is dishonest because
    the training loop sets both global generators from the same
    integer.
    """


# ---------------------------------------------------------------------------
# Wrapper-reference resolution
# ---------------------------------------------------------------------------


def resolve_wrapper(reference: str) -> type:
    """Resolve a ``"module.path:ClassName"`` reference to the class object.

    Parameters
    ----------
    reference : str
        A string of the form ``"module.path:ClassName"``.

    Returns
    -------
    type
        The class object referenced.

    Raises
    ------
    TypeError
        If ``reference`` is not a string, or the resolved attribute
        is not a class.
    ValueError
        If ``reference`` is malformed (missing ``:`` separator, more
        than one ``:``, or empty module or class component).
    ImportError
        If the named module cannot be imported.
    AttributeError
        If the named module does not define the named attribute.
    """
    if not isinstance(reference, str):
        raise TypeError(
            "wrapper reference must be a string of the form "
            f"'module.path:ClassName'; got {type(reference).__name__}"
        )
    if reference.count(":") != 1:
        raise ValueError(
            "wrapper reference must contain exactly one ':' separating "
            f"the module path from the class name; got {reference!r}"
        )
    module_path, class_name = reference.split(":", 1)
    if not module_path:
        raise ValueError(
            "wrapper reference module-path component is empty; "
            f"got {reference!r}"
        )
    if not class_name:
        raise ValueError(
            "wrapper reference class-name component is empty; "
            f"got {reference!r}"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"could not import module {module_path!r} from wrapper "
            f"reference {reference!r}: {exc}"
        ) from exc
    if not hasattr(module, class_name):
        raise AttributeError(
            f"module {module_path!r} has no attribute {class_name!r} "
            f"(from wrapper reference {reference!r})"
        )
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(
            f"wrapper reference {reference!r} resolved to a "
            f"{type(cls).__name__}, not a class"
        )
    return cls


# ---------------------------------------------------------------------------
# Lazy access to wrapper-side preprocessing and DCDI configuration
# ---------------------------------------------------------------------------


def _make_preprocessor(condition: str) -> Any:
    """Instantiate the project preprocessor for a configuration condition."""
    pp_module = importlib.import_module(_PREPROCESSING_MODULE)
    if condition == "centred_only":
        return pp_module.CentredOnlyTransform()
    if condition == "standardised":
        return pp_module.StandardisedTransform()
    raise ValueError(
        f"condition must be 'centred_only' or 'standardised'; "
        f"got {condition!r}"
    )


def _resolve_dcdi_config_class() -> type:
    """Dynamically import the DCDIConfig dataclass."""
    dcdi_module = importlib.import_module(_DCDI_MODULE)
    return getattr(dcdi_module, "DCDIConfig")


# ---------------------------------------------------------------------------
# Reproducibility-field helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    """Return the repository HEAD commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _env_snapshot() -> str:
    """Return a short inline environment snapshot string."""
    parts = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
    ]
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert an object to JSON-compatible primitives.

    NumPy and torch array-like objects are replaced with a structural
    descriptor ``{"__type__": ..., "shape": ..., "dtype": ...}``. The
    actual binary data lives in sibling artefact files referenced
    from the run record by relative path.

    Unsupported object types raise ``TypeError`` naming the offending
    runtime type. Silent stringification is not performed; an
    unsupported type signals a schema bug or an un-vetted diagnostics
    key.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if (
        hasattr(obj, "detach")
        and hasattr(obj, "cpu")
        and hasattr(obj, "shape")
        and hasattr(obj, "dtype")
    ):
        return {
            "__type__": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    raise TypeError(
        "cannot serialise object of type "
        f"{type(obj).__name__!r} to JSON-compatible primitives; "
        "extend _to_jsonable if this is a legitimate diagnostics value"
    )


# ---------------------------------------------------------------------------
# Continuous edge object artefact
# ---------------------------------------------------------------------------


def _save_continuous_edge_object(
    diagnostics: dict, model: str, path: Path
) -> None:
    """Write the model's continuous edge representation as an npz file.

    DAGMA writes a single ``W_continuous`` array. DCDI writes two
    arrays named ``log_alpha`` and ``w_adj``.
    """
    model_specific = diagnostics["model_specific_diagnostics"]
    if model == "dagma":
        w_continuous = np.asarray(
            model_specific["continuous_w_pre_threshold"], dtype=np.float64
        )
        np.savez(path, W_continuous=w_continuous)
        return
    if model == "dcdi":
        log_alpha = model_specific["continuous_log_alpha_pre_threshold"]
        w_adj = model_specific["continuous_w_adj_pre_threshold"]
        log_alpha_np = log_alpha.detach().cpu().numpy()
        w_adj_np = w_adj.detach().cpu().numpy()
        np.savez(path, log_alpha=log_alpha_np, w_adj=w_adj_np)
        return
    raise ValueError(
        f"continuous-edge serialisation unsupported for model {model!r}"
    )


# ---------------------------------------------------------------------------
# Schema-facing sampling-policy mapping
# ---------------------------------------------------------------------------


# The preflight currently emits "native_conditionals" for DCDI runs,
# while the schema enumerates the DCDI policy as "dcdi_native". The
# mapping happens at the pipeline boundary; preflight is unchanged.
# This mismatch is a follow-up cleanup item to be resolved at the
# preflight surface.
_PLANNED_TO_SCHEMA_POLICY = {
    "native_conditionals": "dcdi_native",
}


def _map_planned_sampling_policy_to_schema(planned: str) -> str:
    """Map preflight's planned_sampling_policy to the schema enum value."""
    return _PLANNED_TO_SCHEMA_POLICY.get(planned, planned)


# ---------------------------------------------------------------------------
# Top-level pipeline entry
# ---------------------------------------------------------------------------


def run_single_fit(
    manifest: Manifest,
    entry_index: int,
    *,
    run_root: Path,
) -> Path:
    """Run a single toy fit and emit a schema-conforming run.json.

    Parameters
    ----------
    manifest : Manifest
        A preflight-validated manifest.
    entry_index : int
        Index of the manifest entry to fit. Must be a plain ``int``
        in ``[0, len(manifest.entries))``.
    run_root : pathlib.Path
        Root of the run-storage tree. The run directory is derived
        below this root via ``identity.derive_run_directory``.

    Returns
    -------
    pathlib.Path
        Path of the written ``run.json`` file.

    Raises
    ------
    TypeError
        If ``manifest`` is not a ``Manifest`` or ``entry_index`` is
        not a plain ``int``.
    IndexError
        If ``entry_index`` is outside ``[0, len(manifest.entries))``.
    FileExistsError
        If the target run directory already exists and is non-empty.
        Raised before any artefact is written.
    DcdiSeedMismatchError
        For DCDI entries when the resolved configuration's
        ``seed_torch`` and ``seed_numpy`` are unequal or null.
    InvalidGraphForSchemaGateError
        When the toy fit produces a thresholded graph that is not a
        valid DAG. The schema requires ``sid`` as a plain integer
        and the wrapper cannot compute SID on a non-DAG; the gate
        refuses to write a partial or repaired record.
    """
    if not isinstance(manifest, Manifest):
        raise TypeError(
            "manifest must be a Manifest instance; got "
            f"{type(manifest).__name__}"
        )
    if isinstance(entry_index, bool) or not isinstance(entry_index, int):
        raise TypeError(
            "entry_index must be a plain int (not bool); got "
            f"{type(entry_index).__name__}"
        )
    n_entries = len(manifest.entries)
    if entry_index < 0 or entry_index >= n_entries:
        raise IndexError(
            f"entry_index must be in [0, {n_entries}); got {entry_index}"
        )

    entry = manifest.entries[entry_index]
    resolved_config = manifest.resolved_config

    # Identity invariants. The run_id is derivable purely from
    # identity fields; the run directory adds run_root.
    run_id = derive_run_id(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
    )
    if run_id != entry.expected_run_id:
        raise RuntimeError(
            "derived run_id disagrees with manifest entry.expected_run_id; "
            f"derived={run_id!r}, entry={entry.expected_run_id!r}"
        )
    run_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=run_root,
    )
    assert_run_id_matches_directory(run_id, run_dir)

    # Static DCDI preconditions are validated before any filesystem
    # write so an unequal or null seed pair, or a missing validation
    # seed, cannot leave an empty run directory behind.
    dcdi_fit_seed: Optional[int] = None
    if entry.model == "dcdi":
        if entry.validation_data_seed is None:
            raise ValueError(
                "DCDI entry has validation_data_seed=None; expected a "
                "non-negative integer"
            )
        seed_torch_cfg = resolved_config.get("seed_torch")
        seed_numpy_cfg = resolved_config.get("seed_numpy")
        if seed_torch_cfg is None or seed_numpy_cfg is None:
            raise DcdiSeedMismatchError(
                "DCDI requires non-null seed_torch and seed_numpy in the "
                "resolved configuration; "
                f"got seed_torch={seed_torch_cfg!r}, "
                f"seed_numpy={seed_numpy_cfg!r}"
            )
        if seed_torch_cfg != seed_numpy_cfg:
            raise DcdiSeedMismatchError(
                "DCDIWrapper.fit accepts a single optimisation seed but the "
                "resolved configuration has unequal seed_torch and "
                f"seed_numpy: seed_torch={seed_torch_cfg!r}, "
                f"seed_numpy={seed_numpy_cfg!r}"
            )
        dcdi_fit_seed = int(seed_torch_cfg)

    # Reject a populated target directory before any data is generated
    # or any artefact is written. create_run_directory raises
    # FileExistsError if run_dir exists and is non-empty, or if the
    # path exists and is not a directory.
    create_run_directory(run_dir)

    # Build the toy SCM and sample training data.
    scm = generate_linear_gaussian_scm(
        n_nodes=SCHEMA_GATE_N_NODES,
        expected_edges=SCHEMA_GATE_EXPECTED_EDGES,
        seed=entry.graph_seed,
    )
    x_train_raw = sample_observational(
        scm, n_samples=SCHEMA_GATE_N_TRAIN, rng=entry.train_data_seed
    )

    # Sample validation data only when the candidate uses one. The
    # validation_data_seed presence was validated upfront.
    x_val_raw: Optional[np.ndarray] = None
    if entry.model == "dcdi":
        x_val_raw = sample_observational(
            scm,
            n_samples=SCHEMA_GATE_N_VAL_DCDI,
            rng=entry.validation_data_seed,
        )

    # Preprocess: fit on training data only and reuse for validation.
    preprocessor = _make_preprocessor(entry.condition)
    preprocessor.fit(x_train_raw)
    x_train_model = preprocessor.transform(x_train_raw)
    x_val_model = (
        preprocessor.transform(x_val_raw) if x_val_raw is not None else None
    )

    # Resolve the wrapper class and dispatch fit by entry.model.
    wrapper_cls = resolve_wrapper(entry.planned_wrapper)
    wrapper = wrapper_cls()

    seed_torch_recorded: Optional[int] = None
    seed_numpy_recorded: Optional[int] = None
    seed_dagma_recorded: Optional[int] = None

    t_start = time.perf_counter()
    if entry.model == "dagma":
        wrapper.fit(
            x_train_model,
            preprocessor=preprocessor,
            seed=entry.train_data_seed,
            config=None,
        )
        # DAGMA does not call torch / numpy / dagma global seed setters.
    elif entry.model == "dcdi":
        # Seed equality and presence were validated upfront; the
        # validated value is held in dcdi_fit_seed.
        assert dcdi_fit_seed is not None
        dcdi_config_cls = _resolve_dcdi_config_class()
        dcdi_config = dcdi_config_cls(**SCHEMA_GATE_DCDI_CONFIG_KWARGS)
        wrapper.fit(
            x_train_model,
            X_val=x_val_model,
            preprocessor=preprocessor,
            seed=dcdi_fit_seed,
            n_iter=SCHEMA_GATE_DCDI_N_ITER,
            config=dcdi_config,
        )
        seed_torch_recorded = dcdi_fit_seed
        seed_numpy_recorded = dcdi_fit_seed
    else:
        raise ValueError(
            f"entry.model must be 'dagma' or 'dcdi'; got {entry.model!r}"
        )
    runtime_seconds = float(time.perf_counter() - t_start)

    diagnostics = wrapper.get_diagnostics()
    graph_status = str(diagnostics["graph_status"])
    graph_invalid_reason = diagnostics.get("graph_invalid_reason")
    sampler_status = str(diagnostics["sampler_status"])
    sampler_unavailable_reason = diagnostics.get("sampler_unavailable_reason")
    training_status = str(diagnostics["training_status"])
    n_iterations_raw = diagnostics.get("n_iterations")
    loss_history_values = list(diagnostics.get("loss_history") or [])

    # The schema requires sid as a plain integer. A non-valid_dag
    # thresholded graph makes SID unrecoverable. The gate stops
    # before any artefact is written; the previously-created run
    # directory remains empty and is left in place for inspection.
    if graph_status != "valid_dag":
        raise InvalidGraphForSchemaGateError(
            "toy fit produced graph_status="
            f"{graph_status!r}; the schema requires sid as a plain int "
            "and SID cannot be computed on a non-valid_dag graph. "
            f"reason={graph_invalid_reason!r}"
        )

    predicted_adj = wrapper.thresholded_adjacency()
    if predicted_adj.dtype != bool:
        predicted_adj = predicted_adj.astype(bool)
    true_adj = np.asarray(scm.adjacency, dtype=bool)
    shd_value = int(
        shd(predicted_adj, true_adj, reversal_cost=_SHD_REVERSAL_COST)
    )
    sid_value = int(sid_score(predicted_adj, true_adj))

    # Write binary artefacts.
    thresholded_path = run_dir / "thresholded_adjacency.npz"
    continuous_path = run_dir / "continuous_edge_object.npz"
    np.savez(thresholded_path, thresholded_adjacency=predicted_adj)
    _save_continuous_edge_object(diagnostics, entry.model, continuous_path)

    if loss_history_values:
        loss_history_path = run_dir / "loss_history.npz"
        loss_array = np.asarray(loss_history_values, dtype=np.float64)
        np.savez(loss_history_path, loss_history=loss_array)
        loss_history_field: Optional[str] = "loss_history.npz"
        loss_history_status = "available"
    else:
        loss_history_field = None
        loss_history_status = "unavailable_no_api"

    per_intervention_seeds_map = dict(entry.per_intervention_seeds)
    sampler_policy_used = _map_planned_sampling_policy_to_schema(
        entry.planned_sampling_policy
    )
    mmd_result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=sampler_status,
        sampler_unavailable_reason=sampler_unavailable_reason,
        sampler_policy_used=sampler_policy_used,
        intervention_set=list(resolved_config.get("intervention_set", [])),
        per_intervention_seeds_map=per_intervention_seeds_map,
        preprocessor=preprocessor,
    )
    intervention_records = mmd_result["records"]
    mmd_primary = mmd_result["mmd_primary"]
    mmd_sensitivity_unit_variance = mmd_result[
        "mmd_sensitivity_unit_variance"
    ]
    mmd_bandwidth_sweep = mmd_result["mmd_bandwidth_sweep"]
    mmd_bandwidth_used_value = mmd_result["mmd_bandwidth_used_value"]
    mmd_available_count = mmd_result["mmd_available_count"]
    mmd_missing_count = mmd_result["mmd_missing_count"]
    invalid_graph_for_this_run = graph_status != "valid_dag"

    record: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": _SCHEMA_VERSION,
        "model": entry.model,
        "condition": entry.condition,
        "seed_population": entry.seed_population,
        "seed_replicate_index": int(entry.seed_replicate_index),
        "configuration_hash": entry.configuration_hash,
        "graph_seed": int(entry.graph_seed),
        "git_hash": _git_hash(),
        "env_snapshot": _env_snapshot(),
        "config_resolved": _to_jsonable(resolved_config),
        "seed_torch": seed_torch_recorded,
        "seed_numpy": seed_numpy_recorded,
        "seed_dagma": seed_dagma_recorded,
        "model_sampling_seed_base": int(entry.model_sampling_seed_base),
        "model_sampling_seed_derivation_rule": SEED_DERIVATION_RULE_NAME,
        "train_data_seed": int(entry.train_data_seed),
        "validation_data_seed": (
            None
            if entry.validation_data_seed is None
            else int(entry.validation_data_seed)
        ),
        "intervention_ground_truth_seed_base": int(
            entry.intervention_ground_truth_seed_base
        ),
        "training_status": training_status,
        "n_iterations": (
            None if n_iterations_raw is None else int(n_iterations_raw)
        ),
        "runtime_seconds": runtime_seconds,
        "loss_history": loss_history_field,
        "loss_history_status": loss_history_status,
        "graph_status": graph_status,
        "graph_status_reason": (
            None
            if graph_invalid_reason is None
            else str(graph_invalid_reason)
        ),
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": shd_value,
        "sid": sid_value,
        "mmd_primary": mmd_primary,
        "mmd_sensitivity_unit_variance": mmd_sensitivity_unit_variance,
        "mmd_bandwidth_sweep": mmd_bandwidth_sweep,
        "validation_nll": None,
        "sampler_status": sampler_status,
        "sampler_status_reason": (
            None
            if sampler_unavailable_reason is None
            else str(sampler_unavailable_reason)
        ),
        "sampler_policy_used": sampler_policy_used,
        "mmd_available_count": int(mmd_available_count),
        "mmd_missing_count": int(mmd_missing_count),
        "invalid_graph_for_this_run": bool(invalid_graph_for_this_run),
        "shd_reversal_cost": _SHD_REVERSAL_COST,
        "mmd_bandwidth_used_value": mmd_bandwidth_used_value,
        "mmd_clip_policy": _MMD_CLIP_POLICY,
        "sid_backend": _SID_BACKEND,
        "sid_backend_version": _SID_BACKEND_VERSION,
        "sid_argument_order": _SID_ARGUMENT_ORDER,
        "sid_return_value": _SID_RETURN_VALUE,
        "configuration_hash_algorithm": _CONFIGURATION_HASH_ALGORITHM,
        "wrapper_diagnostics": _to_jsonable(diagnostics),
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": intervention_records,
    }

    payload = json.dumps(
        record, sort_keys=True, ensure_ascii=True, indent=2
    )
    run_json_path = run_dir / "run.json"
    run_json_path.write_text(payload, encoding="utf-8")
    return run_json_path
