"""DAGMA residual noise estimation for the model-frame sampler.

Provides the helper that estimates per-node residual standard deviations
from model-frame training data and a thresholded sampling weight matrix.
Sampling helpers (ancestral sampler, raw-unit roundtrip) will be added
in later commits.
"""

from __future__ import annotations

import numpy as np


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


__all__ = ["estimate_residual_sigmas"]
