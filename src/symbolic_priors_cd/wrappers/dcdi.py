"""DCDI-G wrapper: thresholding helpers and graph/sampler status machinery.

Standalone utilities used by DCDIWrapper. The class itself and its fit,
sample, and diagnostics methods are completed in later commits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from symbolic_priors_cd.wrappers.status import GraphStatus, SamplerStatus


def _predict_adjacency_at(
    continuous_w_adj: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Apply a threshold to the continuous edge-probability matrix.

    Parameters
    ----------
    continuous_w_adj : torch.Tensor
        Continuous edge-probability matrix, shape (d, d). Off-diagonal
        entries are in [0, 1]; diagonal is exactly zero.
    threshold : float
        Entries >= threshold map to True; all others map to False.

    Returns
    -------
    np.ndarray
        Boolean adjacency of shape (d, d), dtype bool,
        row-source / column-destination convention.
    """
    p = continuous_w_adj.detach().cpu().numpy()
    return (p >= threshold).astype(bool)


def _is_acyclic_adjacency(adjacency: np.ndarray) -> bool:
    """Return True if the boolean adjacency matrix is acyclic.

    Builds successive matrix powers up to size d; a non-zero trace at
    any power indicates a directed cycle.

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
