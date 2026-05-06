"""Interventional adequacy metrics for causal discovery evaluation.

Contains sample-based metrics (MMD) and graph-based metrics (SID stub).
Does not perform thresholding or graph inference; those steps belong
upstream in the evaluation harness.
"""

from __future__ import annotations

import numpy as np

from symbolic_priors_cd.metrics._graph_validation import _validate_adjacency


def _validate_sample_matrix(arr: np.ndarray, name: str) -> None:
    """Raise informatively if ``arr`` is not a valid MMD sample matrix."""
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array of shape (n_samples, n_features), "
            f"got {arr.ndim}D"
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"{name} must have at least 2 samples (the unbiased estimator "
            f"requires m*(m-1) > 0), got {arr.shape[0]}"
        )


def _validate_mmd_pair(x: np.ndarray, y: np.ndarray) -> None:
    """Raise informatively if x and y are not compatible MMD inputs."""
    _validate_sample_matrix(x, "x")
    _validate_sample_matrix(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must have the same number of features, "
            f"got {x.shape[1]} and {y.shape[1]}"
        )


def _squared_pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (n_a, n_b) matrix of squared Euclidean distances.

    Uses the identity ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i·b_j.
    """
    return np.maximum(
        np.sum(a ** 2, axis=1, keepdims=True)
        + np.sum(b ** 2, axis=1, keepdims=True).T
        - 2.0 * (a @ b.T),
        0.0,
    )


def _median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate RBF bandwidth via the median heuristic on concatenated samples.

    Computes the median of squared pairwise distances over all pairs i < j
    in the concatenated set (self-distances excluded). The result is used
    as the bandwidth in k(a, b) = exp(-||a-b||^2 / bandwidth).
    """
    z = np.vstack([x, y])
    sq_dists = _squared_pairwise_distances(z, z)
    upper_idx = np.triu_indices(z.shape[0], k=1)
    return float(np.median(sq_dists[upper_idx]))


def mmd_rbf_unbiased(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """Compute the unbiased MMD squared between two sample sets using an RBF kernel.

    Uses the kernel k(a, b) = exp(-||a - b||^2 / bandwidth). With
    ``bandwidth=None``, the bandwidth is set to the median of all pairwise
    squared distances across the concatenated samples (upper triangle only,
    self-distances excluded).

    The estimator is unbiased: within-group sums exclude the diagonal and
    use denominators m*(m-1) and n*(n-1). The return value can be negative,
    which is expected for the unbiased estimator and is never clipped.

    Parameters
    ----------
    x : np.ndarray of shape (m, d), m >= 2
        First sample set.
    y : np.ndarray of shape (n, d), n >= 2
        Second sample set. Must have the same number of features as x.
    bandwidth : float or None
        RBF kernel bandwidth (strictly positive). ``None`` uses the median
        heuristic computed from pairwise squared distances on the concatenated
        samples.

    Returns
    -------
    float
        Unbiased MMD squared. Negative values are valid and are not clipped.

    Raises
    ------
    ValueError
        If inputs are not 2D, have fewer than 2 samples, differ in feature
        dimension, bandwidth is not strictly positive, or the median heuristic
        yields a non-positive bandwidth (degenerate samples).
    """
    _validate_mmd_pair(x, y)
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError(
            f"bandwidth must be strictly positive, got {bandwidth}"
        )
    if bandwidth is None:
        bandwidth = _median_bandwidth(x, y)
        if bandwidth <= 0:
            raise ValueError(
                f"median heuristic produced a non-positive bandwidth "
                f"({bandwidth:.6g}); samples may be degenerate (all identical)"
            )
    m, n = x.shape[0], y.shape[0]
    Kxx = np.exp(-_squared_pairwise_distances(x, x) / bandwidth)
    Kyy = np.exp(-_squared_pairwise_distances(y, y) / bandwidth)
    Kxy = np.exp(-_squared_pairwise_distances(x, y) / bandwidth)
    # Diagonal of the RBF kernel is always exp(0) = 1, so trace equals the
    # sample count. Subtract to exclude self-pairs from within-group sums.
    xx_term = (Kxx.sum() - m) / (m * (m - 1))
    yy_term = (Kyy.sum() - n) / (n * (n - 1))
    xy_term = 2.0 * Kxy.sum() / (m * n)
    return float(xx_term - xy_term + yy_term)


def mmd_sensitivity_sweep(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[float, float]:
    """Compute MMD at several bandwidth scales relative to the median heuristic.

    The median bandwidth is computed once from the concatenated samples, then
    each multiplier scales it: bandwidth_i = median_bandwidth * multiplier_i.
    This ensures all sweep values share the same baseline.

    Parameters
    ----------
    x : np.ndarray of shape (m, d), m >= 2
        First sample set.
    y : np.ndarray of shape (n, d), n >= 2
        Second sample set.
    bandwidth_multipliers : tuple of positive floats
        Scale factors applied to the median bandwidth. Must be non-empty;
        all entries must be strictly positive.

    Returns
    -------
    dict[float, float]
        Mapping from each multiplier to the corresponding unbiased MMD squared.

    Raises
    ------
    ValueError
        If inputs are invalid, multipliers are empty or non-positive, or the
        median heuristic yields a non-positive bandwidth.
    """
    _validate_mmd_pair(x, y)
    if len(bandwidth_multipliers) == 0:
        raise ValueError("bandwidth_multipliers must not be empty")
    for mult in bandwidth_multipliers:
        if mult <= 0:
            raise ValueError(
                f"all bandwidth_multipliers must be strictly positive, got {mult}"
            )
    base_bandwidth = _median_bandwidth(x, y)
    if base_bandwidth <= 0:
        raise ValueError(
            f"median heuristic produced a non-positive bandwidth "
            f"({base_bandwidth:.6g}); samples may be degenerate (all identical)"
        )
    return {
        mult: mmd_rbf_unbiased(x, y, bandwidth=base_bandwidth * mult)
        for mult in bandwidth_multipliers
    }


def sid_score(predicted_dag: np.ndarray, true_dag: np.ndarray) -> int:
    """Compute the Structural Intervention Distance between two DAG adjacency matrices.

    SID measures how many interventional distributions are incorrect under
    the predicted graph relative to the true graph. Unlike SHD, SID directly
    quantifies intervention mistakes rather than edge-edit distance.

    Inputs must be strict boolean DAG adjacency matrices with no self-loops.
    The function does not verify acyclicity or the absence of bidirected edges;
    behaviour on such malformed inputs is undefined.

    Parameters
    ----------
    predicted_dag : np.ndarray, square, dtype bool
        Estimated DAG adjacency matrix. ``predicted_dag[i, j] = True`` means
        directed edge i->j.
    true_dag : np.ndarray, square, dtype bool
        Ground-truth DAG adjacency matrix, same shape as ``predicted_dag``.

    Returns
    -------
    int
        SID score. Zero means every interventional distribution is correct.

    Raises
    ------
    TypeError
        If either input is not dtype bool.
    ValueError
        If inputs are not square, shapes differ, or self-loops are present.
    NotImplementedError
        Always , SID computation is deferred pending explicit verification.
    """
    _validate_adjacency(predicted_dag, "predicted_dag")
    _validate_adjacency(true_dag, "true_dag")
    if predicted_dag.shape != true_dag.shape:
        raise ValueError(
            f"predicted_dag and true_dag must have the same shape, "
            f"got {predicted_dag.shape} and {true_dag.shape}"
        )
    raise NotImplementedError(
        "SID implementation is deferred pending explicit verification."
    )
