"""Structural graph metrics for causal discovery evaluation.

All functions in this module operate on already-validated DAG adjacency
matrices. They do not perform acyclicity checking or thresholding.
"""

from __future__ import annotations

import numpy as np

from symbolic_priors_cd.metrics._graph_validation import _validate_adjacency


def shd(
    predicted: np.ndarray,
    true: np.ndarray,
    reversal_cost: int = 2,
) -> int:
    """Compute the Structural Hamming Distance between two DAG adjacency matrices.

    SHD counts the minimum number of edge operations required to convert
    ``predicted`` into ``true``. Each insertion or deletion costs 1.
    A reversal (an edge present in opposite direction) costs ``reversal_cost``.

    With the default ``reversal_cost=2``, a reversal is counted as two
    independent operations (one deletion and one insertion), which is the
    stricter convention adopted for this project. Setting ``reversal_cost=1``
    treats a reversal as a single cheaper operation.

    Inputs must be strict boolean DAG adjacency matrices with no self-loops.
    The function does not verify acyclicity or the absence of bidirected edges;
    behaviour on such malformed inputs is undefined.

    Parameters
    ----------
    predicted : np.ndarray, square, dtype bool
        Estimated DAG adjacency matrix. ``predicted[i, j] = True`` means
        directed edge i->j.
    true : np.ndarray, square, dtype bool
        Ground-truth DAG adjacency matrix, same shape as ``predicted``.
    reversal_cost : int
        Cost charged for each reversed edge. Must be a positive integer
        (bool values are rejected). Default is 2.

    Returns
    -------
    int
        SHD score. Zero means the graphs are identical.

    Raises
    ------
    TypeError
        If ``predicted`` or ``true`` is not dtype bool, or if
        ``reversal_cost`` is not a plain ``int`` (bool is rejected).
    ValueError
        If shapes differ, inputs are not square, self-loops are present,
        or ``reversal_cost`` is not a positive integer.
    """
    _validate_adjacency(predicted, "predicted")
    _validate_adjacency(true, "true")
    if predicted.shape != true.shape:
        raise ValueError(
            f"predicted and true must have the same shape, "
            f"got {predicted.shape} and {true.shape}"
        )
    if isinstance(reversal_cost, bool):
        raise TypeError(
            f"reversal_cost must be a plain int, not bool"
        )
    if not isinstance(reversal_cost, int):
        raise TypeError(
            f"reversal_cost must be a positive int, got {type(reversal_cost).__name__}"
        )
    if reversal_cost <= 0:
        raise ValueError(
            f"reversal_cost must be a positive int, got {reversal_cost}"
        )

    diff = predicted != true
    # reversal_mask[i, j] is True when both (i,j) and (j,i) are wrong,
    # meaning one direction is present in true and the opposite in predicted.
    reversal_mask = diff & diff.T
    n_reversals = int(reversal_mask.sum()) // 2
    n_other = int(diff.sum()) - 2 * n_reversals
    return n_reversals * reversal_cost + n_other


