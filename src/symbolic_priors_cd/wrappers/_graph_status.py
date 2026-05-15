"""Shared graph-status classification for causal discovery wrappers.

Pure-bool-adjacency utilities. They classify a candidate thresholded
adjacency matrix and map a graph status to a sampler-availability
status. Inputs are never modified.

The classification priority is:
    invalid_shape -> self_loop -> bidirected -> cyclic -> valid_dag.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from symbolic_priors_cd.wrappers.status import GraphStatus, SamplerStatus


def _is_acyclic_adjacency(adjacency: np.ndarray) -> bool:
    """Return True if the boolean adjacency matrix is acyclic.

    Builds successive matrix powers up to size d; a non-zero trace at
    any power indicates a directed cycle.

    Same semantic role as metrics._graph_validation._is_acyclic_adjacency;
    the two helpers must be kept consistent.  Wrappers must not import from
    metrics, so the implementation is duplicated rather than shared.

    Parameters
    ----------
    adjacency : np.ndarray
        Boolean adjacency matrix, shape (d, d).

    Returns
    -------
    bool
    """
    d = adjacency.shape[0]
    a = adjacency.astype(np.int64)
    prod = np.eye(d, dtype=np.int64)
    for _ in range(d):
        prod = prod @ a
        if np.trace(prod) != 0:
            return False
    return True


def classify_graph_status(
    adjacency: np.ndarray,
) -> tuple[GraphStatus, Optional[str]]:
    """Classify the structural status of a boolean adjacency matrix.

    Checks are applied in priority order: invalid_shape -> self_loop ->
    bidirected -> cyclic -> valid_dag. The first matching condition is
    returned; the adjacency is never modified.

    Parameters
    ----------
    adjacency : np.ndarray
        Candidate adjacency matrix, dtype bool, row-source /
        column-destination convention.

    Returns
    -------
    status : GraphStatus
        One of "valid_dag", "cyclic", "bidirected", "self_loop",
        "invalid_shape".
    reason : Optional[str]
        Human-readable description for non-valid-dag statuses;
        None when status is "valid_dag".

    Raises
    ------
    TypeError
        If adjacency dtype is not bool (checked after shape validation).
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
        return "self_loop", "Adjacency has at least one self-loop on the diagonal."

    # After the self-loop check the diagonal is all False, so any True
    # entry in adjacency & adjacency.T is an off-diagonal bidirected pair.
    if np.any(adjacency & adjacency.T):
        return "bidirected", "Adjacency has at least one bidirected edge pair."

    if not _is_acyclic_adjacency(adjacency):
        return "cyclic", "Adjacency contains a directed cycle."

    return "valid_dag", None


def _topological_order(adjacency: np.ndarray) -> list[int]:
    """Return node indices in topological order for a boolean DAG adjacency.

    Uses Kahn's algorithm. Ties in the ready set are broken by ascending
    node index so the output is deterministic for a given adjacency.

    Parameters
    ----------
    adjacency : np.ndarray
        Boolean adjacency matrix, shape (d, d), row-source /
        column-destination convention. adjacency[i, j] = True means
        edge i -> j. Must be a valid DAG.

    Returns
    -------
    list[int]
        Node indices in topological order (parents before children).

    Raises
    ------
    TypeError
        If adjacency.dtype is not bool.
    ValueError
        If adjacency is not a valid DAG (invalid shape, self-loop,
        bidirected edge, or cycle).
    """
    graph_status, reason = classify_graph_status(adjacency)
    if graph_status != "valid_dag":
        raise ValueError(
            f"Topological sort requires a valid DAG; got '{graph_status}'. "
            f"Reason: {reason}"
        )
    d = adjacency.shape[0]
    in_degree = adjacency.sum(axis=0).astype(int)
    ready = sorted(i for i in range(d) if in_degree[i] == 0)
    order: list[int] = []
    while ready:
        j = ready.pop(0)
        order.append(j)
        for k in np.nonzero(adjacency[j, :])[0]:
            in_degree[k] -= 1
            if in_degree[k] == 0:
                ready.append(k)
        ready.sort()
    return order


def infer_sampler_status(
    graph_status: GraphStatus,
) -> tuple[SamplerStatus, Optional[str]]:
    """Map a graph status to the corresponding sampler availability.

    Parameters
    ----------
    graph_status : GraphStatus
        Output of classify_graph_status.

    Returns
    -------
    status : SamplerStatus
        "available" when graph_status is "valid_dag"; otherwise
        "unavailable_invalid_graph".
    reason : Optional[str]
        None for "available"; descriptive reason otherwise.
    """
    if graph_status == "valid_dag":
        return "available", None
    return (
        "unavailable_invalid_graph",
        f"Graph status is '{graph_status}', not 'valid_dag'.",
    )


__all__ = [
    "_is_acyclic_adjacency",
    "_topological_order",
    "classify_graph_status",
    "infer_sampler_status",
]
