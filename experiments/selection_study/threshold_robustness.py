"""Offline threshold-robustness re-computation for selection-study runs.

Given an existing completed run directory, recompute SHD, SID,
edge count, and graph status at the saved per-model threshold
triple from the saved continuous-edge artefact. This is defensive
local sensitivity analysis only: no retraining, no wrapper
instantiation, no MMD recomputation, no threshold calibration,
no run.json mutation.

The triple is read from
``record.data["config_resolved"]["threshold_robustness_triple"]``
and validated against per-model protocol constants in order
(low, primary, high), not as a permutation. DAGMA thresholds
operate on ``abs(W_continuous)``; DCDI thresholds operate on
``w_adj`` (not ``log_alpha``). SHD is computed for every
threshold; SID is computed only for thresholds whose adjacency
classifies as ``"valid_dag"`` and is ``None`` otherwise. The
re-computation never imports or instantiates a wrapper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiments.selection_study.loader import load_run
from symbolic_priors_cd.data import generate_linear_gaussian_scm
from symbolic_priors_cd.metrics import shd, sid_score


PROTOCOL_THRESHOLD_TRIPLES: dict[str, tuple[float, float, float]] = {
    "dagma": (0.2, 0.3, 0.4),
    "dcdi": (0.4, 0.5, 0.6),
}

_THRESHOLD_ROLES: tuple[str, str, str] = ("low", "primary", "high")
_TRIPLE_TOLERANCE = 1e-9


_VALID_GRAPH_STATUSES: tuple[str, ...] = (
    "valid_dag",
    "cyclic",
    "bidirected",
    "self_loop",
    "invalid_shape",
)


# ---------------------------------------------------------------------------
# Graph-status helper (kept self-contained so this module does not import
# from the wrappers package; the wrappers package owns the canonical
# implementation in wrappers/_graph_status.py and this helper mirrors its
# priority order: invalid_shape -> self_loop -> bidirected -> cyclic ->
# valid_dag).
# ---------------------------------------------------------------------------


def _is_acyclic(adjacency: np.ndarray) -> bool:
    """Return True when the boolean adjacency contains no directed cycle."""
    d = adjacency.shape[0]
    a = adjacency.astype(np.int64)
    prod = np.eye(d, dtype=np.int64)
    for _ in range(d):
        prod = prod @ a
        if np.trace(prod) != 0:
            return False
    return True


def _classify_graph_status(
    adjacency: np.ndarray,
) -> tuple[str, Optional[str]]:
    """Classify a boolean adjacency under the project status taxonomy.

    The priority order matches ``wrappers._graph_status.classify_graph_status``:
    ``invalid_shape`` -> ``self_loop`` -> ``bidirected`` -> ``cyclic`` ->
    ``valid_dag``. The adjacency is never modified.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return (
            "invalid_shape",
            f"Adjacency must be square 2D, got shape {adjacency.shape}.",
        )
    if adjacency.dtype != bool:
        raise TypeError(
            f"adjacency must have dtype bool, got {adjacency.dtype}."
        )
    if np.any(np.diag(adjacency)):
        return (
            "self_loop",
            "Adjacency has at least one self-loop on the diagonal.",
        )
    if np.any(adjacency & adjacency.T):
        return (
            "bidirected",
            "Adjacency has at least one bidirected edge pair.",
        )
    if not _is_acyclic(adjacency):
        return "cyclic", "Adjacency contains a directed cycle."
    return "valid_dag", None


# ---------------------------------------------------------------------------
# Triple validation
# ---------------------------------------------------------------------------


def _read_saved_triple(
    record_data: dict, model: str
) -> tuple[float, float, float]:
    """Read and ordered-validate the saved threshold triple.

    The triple lives in
    ``record.data["config_resolved"]["threshold_robustness_triple"]``
    and must match the model's protocol triple element-by-element
    in order (low, primary, high). Permutation matches are
    rejected because the ordering encodes the role of each
    threshold.
    """
    config_resolved = record_data.get("config_resolved")
    if not isinstance(config_resolved, dict):
        raise ValueError(
            "run record is missing 'config_resolved' as a JSON "
            "object; cannot read threshold_robustness_triple."
        )
    saved = config_resolved.get("threshold_robustness_triple")
    if saved is None:
        raise ValueError(
            "config_resolved is missing 'threshold_robustness_triple'; "
            "the selection-study protocol requires an explicit triple "
            "in the configuration."
        )
    if not isinstance(saved, (list, tuple)) or len(saved) != 3:
        raise ValueError(
            "threshold_robustness_triple must be a length-3 sequence "
            f"of numbers; got {saved!r}"
        )
    for index, value in enumerate(saved):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "threshold_robustness_triple entries must be plain "
                "numbers (int or float, not bool); "
                f"got entry at index {index}: {value!r}"
            )
    if model not in PROTOCOL_THRESHOLD_TRIPLES:
        raise ValueError(
            "model must be one of "
            f"{tuple(PROTOCOL_THRESHOLD_TRIPLES)}; got {model!r}"
        )
    expected = PROTOCOL_THRESHOLD_TRIPLES[model]
    saved_floats = (
        float(saved[0]),
        float(saved[1]),
        float(saved[2]),
    )
    for index in range(3):
        if not math.isclose(
            saved_floats[index],
            expected[index],
            abs_tol=_TRIPLE_TOLERANCE,
        ):
            raise ValueError(
                f"threshold_robustness_triple mismatch for model "
                f"{model!r}: saved triple "
                f"{saved_floats!r} does not match expected protocol "
                f"triple {expected!r}; ordering is "
                "(low, primary, high) and a permutation is not "
                "accepted."
            )
    return saved_floats


# ---------------------------------------------------------------------------
# Continuous-edge artefact loading
# ---------------------------------------------------------------------------


def _load_threshold_target(
    run_dir: Path, record_data: dict, model: str
) -> tuple[np.ndarray, str]:
    """Return the float64 array to threshold and its artefact filename.

    For DAGMA the array is ``abs(W_continuous)``. For DCDI the
    array is ``w_adj`` cast to float64; ``log_alpha`` is preserved
    in the artefact but never thresholded by this module.
    """
    artefact_name = record_data.get("continuous_edge_object")
    if not isinstance(artefact_name, str) or not artefact_name:
        raise ValueError(
            "run record 'continuous_edge_object' must be a non-empty "
            f"string filename; got {artefact_name!r}"
        )
    artefact_path = run_dir / artefact_name
    if not artefact_path.is_file():
        raise FileNotFoundError(
            "continuous_edge_object artefact not found at "
            f"{artefact_path}"
        )
    with np.load(artefact_path) as data:
        keys = set(data.files)
        if model == "dagma":
            if "W_continuous" not in keys:
                raise ValueError(
                    "DAGMA continuous_edge_object artefact at "
                    f"{artefact_path} must contain 'W_continuous'; "
                    f"available keys: {sorted(keys)!r}"
                )
            w_continuous = np.asarray(
                data["W_continuous"], dtype=np.float64
            )
            return np.abs(w_continuous), artefact_name
        if model == "dcdi":
            if "w_adj" not in keys:
                raise ValueError(
                    "DCDI continuous_edge_object artefact at "
                    f"{artefact_path} must contain 'w_adj'; "
                    f"available keys: {sorted(keys)!r}"
                )
            w_adj = np.asarray(data["w_adj"], dtype=np.float64)
            return w_adj, artefact_name
        raise ValueError(
            "model in run record must be 'dagma' or 'dcdi'; "
            f"got {model!r}"
        )


# ---------------------------------------------------------------------------
# True-graph reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_true_adjacency(
    record_data: dict,
) -> np.ndarray:
    """Reconstruct the true adjacency from saved run/config fields.

    Reads ``graph_seed`` and the four SCM-generation parameters
    from ``config_resolved``; calls
    ``generate_linear_gaussian_scm`` with those values; returns
    the SCM's ``adjacency`` array. The function does not
    re-sample data and does not instantiate any wrapper.
    """
    graph_seed_raw = record_data.get("graph_seed")
    if isinstance(graph_seed_raw, bool) or not isinstance(
        graph_seed_raw, int
    ):
        raise ValueError(
            "run record 'graph_seed' must be an int (not bool); "
            f"got {graph_seed_raw!r}"
        )
    config_resolved = record_data.get("config_resolved")
    if not isinstance(config_resolved, dict):
        raise ValueError(
            "run record is missing 'config_resolved' as a JSON "
            "object; cannot reconstruct true adjacency."
        )
    for field_name in (
        "n_nodes",
        "expected_edges",
        "noise_scale",
        "weight_magnitude_range",
    ):
        if field_name not in config_resolved:
            raise ValueError(
                "config_resolved is missing SCM-generation field "
                f"{field_name!r}; cannot reconstruct true graph "
                "from saved fields."
            )
    n_nodes_raw = config_resolved["n_nodes"]
    expected_edges_raw = config_resolved["expected_edges"]
    noise_scale_raw = config_resolved["noise_scale"]
    weight_range_raw = config_resolved["weight_magnitude_range"]
    if isinstance(n_nodes_raw, bool) or not isinstance(
        n_nodes_raw, int
    ):
        raise ValueError(
            "config_resolved 'n_nodes' must be an int (not bool); "
            f"got {n_nodes_raw!r}"
        )
    if isinstance(expected_edges_raw, bool) or not isinstance(
        expected_edges_raw, int
    ):
        raise ValueError(
            "config_resolved 'expected_edges' must be an int (not "
            f"bool); got {expected_edges_raw!r}"
        )
    if isinstance(noise_scale_raw, bool) or not isinstance(
        noise_scale_raw, (int, float)
    ):
        raise ValueError(
            "config_resolved 'noise_scale' must be a number (not "
            f"bool); got {noise_scale_raw!r}"
        )
    if (
        not isinstance(weight_range_raw, (list, tuple))
        or len(weight_range_raw) != 2
    ):
        raise ValueError(
            "config_resolved 'weight_magnitude_range' must be a "
            f"length-2 sequence; got {weight_range_raw!r}"
        )
    scm = generate_linear_gaussian_scm(
        n_nodes=int(n_nodes_raw),
        expected_edges=int(expected_edges_raw),
        seed=int(graph_seed_raw),
        noise_scale=float(noise_scale_raw),
        weight_magnitude_range=(
            float(weight_range_raw[0]),
            float(weight_range_raw[1]),
        ),
    )
    return np.asarray(scm.adjacency, dtype=bool)


# ---------------------------------------------------------------------------
# Per-threshold record assembly
# ---------------------------------------------------------------------------


def _record_for_threshold(
    *,
    threshold: float,
    role: str,
    threshold_target: np.ndarray,
    true_adjacency: np.ndarray,
    shd_reversal_cost: int,
) -> dict:
    """Compute the per-threshold record for one threshold value."""
    predicted = threshold_target >= float(threshold)
    predicted = np.asarray(predicted, dtype=bool)
    graph_status, graph_status_reason = _classify_graph_status(predicted)
    if graph_status not in _VALID_GRAPH_STATUSES:
        raise RuntimeError(
            "internal error: graph status classifier returned a "
            f"value outside the documented taxonomy: {graph_status!r}"
        )
    # edge_count counts every True entry in the thresholded
    # boolean adjacency, including any diagonal entries. Invalid
    # self-loops are recorded rather than silently hidden.
    edge_count = int(predicted.sum())

    shd_value: Optional[int]
    shd_unavailable_reason: Optional[str]
    if graph_status == "invalid_shape":
        shd_value = None
        shd_unavailable_reason = (
            "predicted adjacency shape is invalid; SHD is not "
            "computed"
        )
    elif graph_status == "self_loop":
        shd_value = None
        shd_unavailable_reason = (
            "predicted adjacency contains a self-loop; project "
            "SHD primitive rejects non-zero predicted diagonals"
        )
    else:
        shd_value = int(
            shd(predicted, true_adjacency, reversal_cost=shd_reversal_cost)
        )
        shd_unavailable_reason = None

    sid_value: Optional[int]
    sid_unavailable_reason: Optional[str]
    if graph_status == "valid_dag":
        sid_value = int(sid_score(predicted, true_adjacency))
        sid_unavailable_reason = None
    else:
        sid_value = None
        sid_unavailable_reason = (
            f"graph_status is {graph_status}; SID is computed "
            "only for valid DAGs"
        )

    # Structured unavailability contract: each metric is either
    # a concrete int with a None reason, or None with a non-empty
    # reason string. The two forms are mutually exclusive.
    if (shd_value is None) != (shd_unavailable_reason is not None):
        raise RuntimeError(
            "internal error: SHD unavailability invariant violated; "
            f"shd={shd_value!r}, "
            f"shd_unavailable_reason={shd_unavailable_reason!r}"
        )
    if (sid_value is None) != (sid_unavailable_reason is not None):
        raise RuntimeError(
            "internal error: SID unavailability invariant violated; "
            f"sid={sid_value!r}, "
            f"sid_unavailable_reason={sid_unavailable_reason!r}"
        )

    return {
        "threshold": float(threshold),
        "threshold_role": role,
        "edge_count": int(edge_count),
        "graph_status": graph_status,
        "graph_status_reason": (
            None if graph_status_reason is None else str(graph_status_reason)
        ),
        "shd": (None if shd_value is None else int(shd_value)),
        "shd_unavailable_reason": shd_unavailable_reason,
        "sid": (None if sid_value is None else int(sid_value)),
        "sid_unavailable_reason": sid_unavailable_reason,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def recompute_at_thresholds(
    run_dir: Path | str,
    *,
    write_sibling: bool = True,
) -> dict:
    """Recompute structural metrics at the saved threshold triple.

    Parameters
    ----------
    run_dir : pathlib.Path or str
        Either the path to a run directory or the path to a
        ``run.json`` file inside one. The continuous-edge artefact
        and (if ``write_sibling`` is True) the
        ``threshold_robustness.json`` sibling are resolved
        relative to this directory.
    write_sibling : bool, optional
        When ``True`` (the default), write
        ``threshold_robustness.json`` next to ``run.json``. When
        ``False``, the record is returned without writing any
        file. ``run.json`` is never mutated regardless of this
        flag; a byte-immutability guard verifies it.

    Returns
    -------
    dict
        JSON-safe record carrying ``run_id``, ``model``,
        ``condition``, ``configuration_hash``,
        ``continuous_edge_object_artefact``, ``threshold_triple``,
        ``primary_threshold``, ``primary_threshold_index``,
        ``shd_reversal_cost``, and a length-3 ``records`` list of
        per-threshold records.

    Raises
    ------
    FileNotFoundError
        If the run directory or the continuous-edge artefact does
        not exist on disk.
    ValueError
        If the saved triple is missing, has the wrong length, has
        non-numeric entries, or does not match the per-model
        protocol triple in order; or if the run record is missing
        any field required to reconstruct the true graph.
    RuntimeError
        If ``run.json`` bytes are observed to change between the
        first read and a final read after the sibling write.
    """
    if isinstance(run_dir, str):
        path = Path(run_dir)
    elif isinstance(run_dir, Path):
        path = run_dir
    else:
        raise TypeError(
            "run_dir must be a Path or str; "
            f"got {type(run_dir).__name__}"
        )
    if path.is_file():
        directory = path.parent
        json_path = path
    else:
        directory = path
        json_path = path / "run.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"run.json not found under {path}"
        )

    record = load_run(json_path)
    record_data = dict(record.data)
    model = str(record_data["model"])
    condition = str(record_data["condition"])
    run_id = str(record_data["run_id"])
    configuration_hash = str(record_data["configuration_hash"])
    shd_reversal_cost_raw = record_data.get("shd_reversal_cost")
    if isinstance(shd_reversal_cost_raw, bool) or not isinstance(
        shd_reversal_cost_raw, int
    ):
        raise ValueError(
            "run record 'shd_reversal_cost' must be an int (not "
            f"bool); got {shd_reversal_cost_raw!r}"
        )
    shd_reversal_cost = int(shd_reversal_cost_raw)
    saved_triple = _read_saved_triple(record_data, model)
    threshold_target, artefact_name = _load_threshold_target(
        directory, record_data, model
    )
    true_adjacency = _reconstruct_true_adjacency(record_data)
    if true_adjacency.shape != threshold_target.shape:
        raise ValueError(
            "reconstructed true adjacency shape "
            f"{true_adjacency.shape} does not match threshold "
            f"target shape {threshold_target.shape}; the saved "
            "SCM-generation parameters disagree with the saved "
            "continuous-edge artefact."
        )

    run_json_bytes_before = json_path.read_bytes()

    threshold_records: list[dict] = []
    for index, threshold_value in enumerate(saved_triple):
        threshold_records.append(
            _record_for_threshold(
                threshold=threshold_value,
                role=_THRESHOLD_ROLES[index],
                threshold_target=threshold_target,
                true_adjacency=true_adjacency,
                shd_reversal_cost=shd_reversal_cost,
            )
        )

    primary_index = 1
    output: dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "configuration_hash": configuration_hash,
        "continuous_edge_object_artefact": artefact_name,
        "threshold_triple": [float(t) for t in saved_triple],
        "primary_threshold": float(saved_triple[primary_index]),
        "primary_threshold_index": int(primary_index),
        "shd_reversal_cost": int(shd_reversal_cost),
        "records": threshold_records,
    }

    if write_sibling:
        sibling_path = directory / "threshold_robustness.json"
        payload = json.dumps(
            output,
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        )
        sibling_path.write_text(payload, encoding="utf-8")

    run_json_bytes_after = json_path.read_bytes()
    if run_json_bytes_before != run_json_bytes_after:
        raise RuntimeError(
            "run.json bytes changed during threshold-robustness "
            "recomputation; the recomputation must be read-only with "
            "respect to run.json."
        )

    return output


__all__ = [
    "PROTOCOL_THRESHOLD_TRIPLES",
    "recompute_at_thresholds",
]
