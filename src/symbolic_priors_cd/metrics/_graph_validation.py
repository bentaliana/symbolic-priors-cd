"""Shared internal adjacency-matrix validation for metrics modules.

Both structural.py and interventional.py import from here so validation
logic for DAG adjacency inputs lives in exactly one place.
"""

from __future__ import annotations

import numpy as np


def _validate_adjacency(matrix: np.ndarray, name: str) -> None:
    """Raise informatively if ``matrix`` is not a valid boolean DAG adjacency.

    Checks strict bool dtype, 2D square shape, and no diagonal self-loops.
    Does not verify acyclicity or the absence of bidirected edges.
    """
    if matrix.dtype != bool:
        raise TypeError(
            f"{name} must have dtype bool, got {matrix.dtype}"
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{name} must be a square 2D matrix, got shape {matrix.shape}"
        )
    if np.any(np.diag(matrix)):
        raise ValueError(
            f"{name} must have no self-loops (diagonal must be all False)"
        )


def _is_acyclic_adjacency(adjacency: np.ndarray) -> bool:
    """Return True if the boolean adjacency matrix contains no directed cycle.

    Builds successive matrix powers up to size d; a non-zero trace at any
    power indicates a directed cycle.  Assumes the input is already a square
    bool matrix (checked by _validate_adjacency before this is called).

    Same semantic role as wrappers._graph_status._is_acyclic_adjacency; the
    two helpers must be kept consistent.  Metrics must not import from wrappers,
    so the implementation is duplicated rather than shared.

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
