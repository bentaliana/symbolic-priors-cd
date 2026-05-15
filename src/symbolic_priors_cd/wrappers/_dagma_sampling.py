"""DAGMA residual noise estimation and model-frame ancestral sampler.

Provides helpers for:
- estimating per-node residual standard deviations from model-frame
  training data and a thresholded sampling weight matrix;
- drawing model-frame samples via linear-Gaussian ancestral sampling
  conditioned on a thresholded adjacency.

Raw-unit roundtrip (preprocessor inverse-transform) is handled by the
wrapper layer, not here.
"""

from __future__ import annotations

import numpy as np

from symbolic_priors_cd.wrappers._graph_status import (
    classify_graph_status,
    _topological_order,
)


def estimate_residual_sigmas(
    X_model_frame: np.ndarray,
    W_continuous: np.ndarray,
    A_thresh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-node residual standard deviations in the model frame.

    Computes the sampling weight matrix from surviving thresholded edges,
    evaluates residuals against that matrix, and returns per-node
    standard deviations with ddof=0. The caller is responsible for
    validating the returned sigma values before treating the sampler
    as available.

    Parameters
    ----------
    X_model_frame : np.ndarray
        Model-frame training data, shape (n_samples, n_vars). Must be
        the stored copy that was not passed to DagmaLinear; DAGMA
        mean-centres its input in place, so the passed copy is mutated.
    W_continuous : np.ndarray
        Pre-threshold continuous weight matrix, shape (n_vars, n_vars).
        Never modified by this function.
    A_thresh : np.ndarray
        Boolean thresholded adjacency, shape (n_vars, n_vars), dtype bool.

    Returns
    -------
    w_sample : np.ndarray
        Sampling weight matrix of shape (n_vars, n_vars); equals
        W_continuous * A_thresh, zeroing sub-threshold entries.
    sigma_vector : np.ndarray
        Per-node residual standard deviations, shape (n_vars,), ddof=0.
        May contain zeros or non-finite values for degenerate inputs.
    """
    if X_model_frame.ndim != 2:
        raise ValueError(
            f"X_model_frame must be 2D, got ndim={X_model_frame.ndim}."
        )
    if W_continuous.ndim != 2 or W_continuous.shape[0] != W_continuous.shape[1]:
        raise ValueError(
            f"W_continuous must be square 2D, got shape {W_continuous.shape}."
        )
    if A_thresh.shape != W_continuous.shape:
        raise ValueError(
            f"A_thresh shape {A_thresh.shape} does not match "
            f"W_continuous shape {W_continuous.shape}."
        )
    if A_thresh.dtype != bool:
        raise TypeError(
            f"A_thresh must have dtype bool, got {A_thresh.dtype}."
        )
    if X_model_frame.shape[1] != W_continuous.shape[0]:
        raise ValueError(
            f"X_model_frame has {X_model_frame.shape[1]} columns but "
            f"W_continuous has {W_continuous.shape[0]} rows."
        )
    w_sample = W_continuous * A_thresh.astype(W_continuous.dtype)
    r = X_model_frame - X_model_frame @ w_sample
    sigma_vector = r.std(axis=0, ddof=0)
    return w_sample, sigma_vector


def sample_linear_gaussian_model_frame(
    A_thresh: np.ndarray,
    W_sample: np.ndarray,
    sigma_vector: np.ndarray,
    *,
    target: int,
    value_model: float,
    n_samples: int,
    sample_seed: int,
) -> np.ndarray:
    """Draw model-frame samples via linear-Gaussian ancestral sampling.

    Traverses nodes in topological order. At the intervention target,
    the column is clamped to value_model. For all other nodes, the
    conditional mean is computed from parent values using the row-source /
    column-destination weight convention, and Gaussian noise is added.

    No inverse transform is applied. The caller is responsible for
    converting raw-unit intervention values to model frame before calling
    this function, and for inverse-transforming the returned samples if
    raw-unit outputs are required.

    Parameters
    ----------
    A_thresh : np.ndarray
        Boolean thresholded adjacency, shape (n_vars, n_vars), dtype bool.
        Must be a valid DAG.
    W_sample : np.ndarray
        Sampling weight matrix, shape (n_vars, n_vars). Typically
        W_continuous * A_thresh; sub-threshold entries are zero.
    sigma_vector : np.ndarray
        Per-node residual standard deviations, shape (n_vars,). All
        entries must be finite and strictly positive.
    target : int
        Index of the intervened variable (0-indexed). The target column
        is set to value_model for every sample.
    value_model : float
        Model-frame intervention value written into the target column.
        Must be finite.
    n_samples : int
        Number of samples to draw. Must be at least 1.
    sample_seed : int
        Seed for np.random.default_rng. Identical arguments produce
        identical output. No global RNG state is mutated.

    Returns
    -------
    np.ndarray
        Float64 array of shape (n_samples, n_vars) in model frame.

    Raises
    ------
    TypeError
        If A_thresh.dtype is not bool.
    ValueError
        If any input fails validation (shape mismatch, invalid graph,
        non-positive sigma, out-of-range target, non-positive n_samples,
        non-finite value_model).
    """
    # Validate A_thresh: shape, dtype, and graph validity via classify_graph_status.
    graph_status, reason = classify_graph_status(A_thresh)
    if graph_status != "valid_dag":
        raise ValueError(
            f"A_thresh is not a valid DAG; got '{graph_status}'. Reason: {reason}"
        )

    n_vars = A_thresh.shape[0]

    try:
        w_arr = np.asarray(W_sample, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"W_sample could not be converted to a float array: {exc}"
        ) from exc
    if w_arr.ndim != 2 or w_arr.shape != A_thresh.shape:
        raise ValueError(
            f"W_sample must be 2D with shape {A_thresh.shape}, "
            f"got shape {w_arr.shape}."
        )
    if not np.all(np.isfinite(w_arr)):
        bad = np.argwhere(~np.isfinite(w_arr)).tolist()
        raise ValueError(
            f"W_sample must contain only finite values. "
            f"Non-finite entries at positions {bad}."
        )

    sigma_arr = np.asarray(sigma_vector, dtype=float)
    if sigma_arr.shape != (n_vars,):
        raise ValueError(
            f"sigma_vector must have shape ({n_vars},), got {sigma_arr.shape}."
        )
    bad = np.where(~(np.isfinite(sigma_arr) & (sigma_arr > 0)))[0]
    if bad.size > 0:
        raise ValueError(
            f"sigma_vector must be finite and strictly positive. "
            f"Invalid at indices {bad.tolist()}."
        )

    if not isinstance(target, (int, np.integer)) or not (0 <= int(target) < n_vars):
        raise ValueError(
            f"target must be an integer in [0, {n_vars}), got {target!r}."
        )
    target = int(target)

    if not isinstance(n_samples, (int, np.integer)) or int(n_samples) < 1:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples!r}."
        )
    n_samples = int(n_samples)

    if not np.isfinite(value_model):
        raise ValueError(
            f"value_model must be finite, got {value_model!r}."
        )

    order = _topological_order(A_thresh)
    rng = np.random.default_rng(sample_seed)
    X = np.zeros((n_samples, n_vars), dtype=float)

    for j in order:
        if j == target:
            X[:, j] = float(value_model)
        else:
            parents = np.where(A_thresh[:, j])[0]
            if parents.size > 0:
                mean = X[:, parents] @ w_arr[parents, j]
            else:
                mean = 0.0
            X[:, j] = mean + rng.normal(0.0, sigma_arr[j], size=n_samples)

    return X


__all__ = ["estimate_residual_sigmas", "sample_linear_gaussian_model_frame"]
