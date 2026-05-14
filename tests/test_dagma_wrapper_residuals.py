"""Tests for DAGMA residual sigma estimation.

Verifies model-frame X storage, W_sample construction, hand-computed
sigma correctness, degenerate sigma handling, no-floor policy, and
that invalid graphs skip residual estimation entirely.

All tests use a monkeypatched DagmaLinear so they run in milliseconds
and deterministically exercise the wrapper's residual path.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_priors_cd.wrappers.dagma import (
    DAGMAWrapper,
    _threshold_continuous_w,
)
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform


# ---------------------------------------------------------------------------
# Helpers: fake DagmaLinear and fixture builder
# ---------------------------------------------------------------------------


def _patch_dagma_with_W(monkeypatch, W: np.ndarray) -> None:
    """Replace DagmaLinear with a fake that returns the given W on fit."""
    W_fixed = W.astype(float).copy()

    class _InjectedFake:
        h_final = 1e-7
        score_final = -1.0

        def __init__(self, loss_type: str = "l2") -> None:
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            return W_fixed.copy()

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _InjectedFake,
    )


def _build_fitted_wrapper(
    monkeypatch,
    W: np.ndarray,
    X: np.ndarray | None = None,
) -> tuple[DAGMAWrapper, np.ndarray]:
    """Return a fitted DAGMAWrapper and the X used to fit it."""
    _patch_dagma_with_W(monkeypatch, W)
    d = W.shape[0]
    if X is None:
        X = np.random.default_rng(0).standard_normal((20, d))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)
    return wrapper, X


# ---------------------------------------------------------------------------
# Known W matrices used across tests
# ---------------------------------------------------------------------------

# Valid DAG: chain 0 -> 1 -> 2.
_W_VALID = np.array(
    [
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Cyclic: 0 -> 1 -> 2 -> 0 (all weights above 0.3 threshold).
_W_CYCLIC = np.array(
    [
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 0.0],
    ],
    dtype=float,
)

# Valid DAG that perfectly predicts column 1 from column 0.
# With X[:, 1] == X[:, 0] and W[0, 1] = 1.0, R[:, 1] = 0 exactly,
# so sigma[1] = 0 -- a degenerate (non-positive) value.
_W_DEGEN = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)

_N_DEGEN = 5
_X_DEGEN = np.zeros((_N_DEGEN, 3), dtype=float)
_X_DEGEN[:, 0] = np.arange(_N_DEGEN, dtype=float)       # [0, 1, 2, 3, 4]
_X_DEGEN[:, 1] = np.arange(_N_DEGEN, dtype=float)       # == X[:, 0]
_X_DEGEN[:, 2] = np.arange(_N_DEGEN, dtype=float) + 1.0  # [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 1. fit stores model-frame X for residual estimation
# ---------------------------------------------------------------------------


def test_fit_stores_x_train_model_frame(monkeypatch):
    """After a successful fit, _X_train_model_frame equals the input X_train."""
    wrapper, X = _build_fitted_wrapper(monkeypatch, _W_VALID)
    assert wrapper._X_train_model_frame is not None
    np.testing.assert_array_equal(
        wrapper._X_train_model_frame, X.astype(float)
    )


# ---------------------------------------------------------------------------
# 2. stored model-frame X is not the DAGMA-mutated local array
# ---------------------------------------------------------------------------


def test_stored_x_not_mutated_by_dagma(monkeypatch):
    """_X_train_model_frame must retain original values despite DAGMA mutating
    its own local copy during mean-centering."""
    W_fixed = _W_VALID.copy()

    class _MutatingFake:
        h_final = 1e-7
        score_final = -1.0

        def __init__(self, loss_type: str = "l2") -> None:
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            X[:] = 0.0  # mutate the copy DAGMA received
            return W_fixed.copy()

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear", _MutatingFake
    )
    X = np.random.default_rng(1).standard_normal((10, 3))
    X_before = X.astype(float).copy()
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)

    np.testing.assert_array_equal(
        wrapper._X_train_model_frame,
        X_before,
        err_msg="_X_train_model_frame was corrupted by DAGMA's internal mutation.",
    )


# ---------------------------------------------------------------------------
# 3. W_sample equals W_continuous * A_thresh
# ---------------------------------------------------------------------------


def test_w_sample_equals_continuous_times_adjacency(monkeypatch):
    """_w_sample_residual_fitted must equal W_continuous * A_thresh elementwise."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    a_thresh = _threshold_continuous_w(wrapper._continuous_w_pre_threshold, 0.3)
    expected = wrapper._continuous_w_pre_threshold * a_thresh.astype(float)
    np.testing.assert_array_equal(wrapper._w_sample_residual_fitted, expected)


# ---------------------------------------------------------------------------
# 4. W_sample uses thresholded surviving edges only
# ---------------------------------------------------------------------------


def test_w_sample_zeros_sub_threshold_entries(monkeypatch):
    """Sub-threshold entries in W_continuous must be zero in W_sample."""
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.9  # above threshold; survives
    W[1, 2] = 0.1  # below threshold; must be zeroed
    wrapper, _ = _build_fitted_wrapper(monkeypatch, W)

    assert wrapper._w_sample_residual_fitted is not None
    assert wrapper._w_sample_residual_fitted[0, 1] == pytest.approx(0.9)
    assert wrapper._w_sample_residual_fitted[1, 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. residual sigma matches hand computation
# ---------------------------------------------------------------------------


def test_residual_sigma_matches_hand_computation(monkeypatch):
    """sigma must equal R.std(axis=0, ddof=0) where R = X - X @ W_sample."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 3))
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, X=X)

    w_sample = wrapper._w_sample_residual_fitted
    r = X - X @ w_sample
    sigma_expected = r.std(axis=0, ddof=0)

    np.testing.assert_allclose(
        wrapper._sigma_vector_residual_fitted,
        sigma_expected,
        atol=1e-12,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 6. sigma vector has shape (d,)
# ---------------------------------------------------------------------------


def test_sigma_vector_shape(monkeypatch):
    """_sigma_vector_residual_fitted must have shape (n_vars,)."""
    d = 4
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 0.9
    W[1, 2] = 0.8
    W[2, 3] = 0.7
    wrapper, _ = _build_fitted_wrapper(monkeypatch, W)

    assert wrapper._sigma_vector_residual_fitted is not None
    assert wrapper._sigma_vector_residual_fitted.shape == (d,)


# ---------------------------------------------------------------------------
# 7. sigma is finite and positive on a normal random fixture
# ---------------------------------------------------------------------------


def test_sigma_finite_and_positive_on_normal_fixture(monkeypatch):
    """On a well-conditioned random dataset, all sigmas are finite and > 0."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    sigma = wrapper._sigma_vector_residual_fitted

    assert sigma is not None
    assert np.all(np.isfinite(sigma)), f"sigma contains non-finite values: {sigma}"
    assert np.all(sigma > 0), f"sigma contains non-positive values: {sigma}"


# ---------------------------------------------------------------------------
# 8. sigma computation uses model-frame data, not raw or re-transformed data
# ---------------------------------------------------------------------------


def test_sigma_uses_stored_model_frame_x(monkeypatch):
    """sigma recomputed from _X_train_model_frame must match the stored sigma."""
    X = np.random.default_rng(7).standard_normal((20, 3))
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, X=X)

    # Verify _X_train_model_frame equals what was passed to fit.
    np.testing.assert_array_equal(
        wrapper._X_train_model_frame, X.astype(float)
    )
    # Recompute sigma from the stored model-frame X; must match stored sigma.
    w_sample = wrapper._w_sample_residual_fitted
    r = wrapper._X_train_model_frame - wrapper._X_train_model_frame @ w_sample
    sigma_from_stored = r.std(axis=0, ddof=0)
    np.testing.assert_allclose(
        wrapper._sigma_vector_residual_fitted,
        sigma_from_stored,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 9. invalid graph does not compute W_sample or sigma
# ---------------------------------------------------------------------------


def test_invalid_graph_w_sample_and_sigma_remain_none(monkeypatch):
    """For a cyclic thresholded graph, W_sample and sigma must remain None."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_CYCLIC)

    assert wrapper._graph_status == "cyclic"
    assert wrapper._w_sample_residual_fitted is None
    assert wrapper._sigma_vector_residual_fitted is None


# ---------------------------------------------------------------------------
# 10. invalid graph keeps sampler_status = unavailable_invalid_graph
# ---------------------------------------------------------------------------


def test_invalid_graph_keeps_sampler_unavailable_invalid_graph(monkeypatch):
    """Residual estimation must not override sampler_status for invalid graphs."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_CYCLIC)

    assert wrapper._sampler_status == "unavailable_invalid_graph"
    assert wrapper._sampler_unavailable_reason is not None
    assert "cyclic" in wrapper._sampler_unavailable_reason


# ---------------------------------------------------------------------------
# 11. non-finite sigma sets unavailable_unresolved_noise_policy
# ---------------------------------------------------------------------------


def test_non_finite_sigma_sets_unavailable_unresolved_noise_policy(monkeypatch):
    """sigma containing inf or NaN triggers unavailable_unresolved_noise_policy."""
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.9  # valid DAG edge; survives threshold

    X_inf = np.ones((5, 3), dtype=float)
    X_inf[0, 0] = np.inf  # forces inf in residuals -> inf sigma

    wrapper, _ = _build_fitted_wrapper(monkeypatch, W, X=X_inf)

    assert wrapper._graph_status == "valid_dag"
    assert wrapper._sampler_status == "unavailable_unresolved_noise_policy"
    assert wrapper._sampler_unavailable_reason is not None
    assert wrapper._sigma_vector_residual_fitted is not None
    assert not np.all(np.isfinite(wrapper._sigma_vector_residual_fitted))


# ---------------------------------------------------------------------------
# 12. zero sigma sets unavailable_unresolved_noise_policy
# ---------------------------------------------------------------------------


def test_zero_sigma_sets_unavailable_unresolved_noise_policy(monkeypatch):
    """sigma[j] == 0 for any j triggers unavailable_unresolved_noise_policy."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_DEGEN, X=_X_DEGEN.copy())

    assert wrapper._graph_status == "valid_dag"
    assert wrapper._sampler_status == "unavailable_unresolved_noise_policy"
    assert wrapper._sampler_unavailable_reason is not None


# ---------------------------------------------------------------------------
# 13. no sigma floor or clamping
# ---------------------------------------------------------------------------


def test_degenerate_sigma_stored_without_floor(monkeypatch):
    """Zero sigma must be stored verbatim; no variance floor is applied."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_DEGEN, X=_X_DEGEN.copy())

    sigma = wrapper._sigma_vector_residual_fitted
    assert sigma is not None
    # Column 1 residuals are exactly zero: sigma[1] must be 0.0, not eps.
    assert sigma[1] == 0.0, (
        f"Expected sigma[1] == 0.0 (no floor) but got {sigma[1]}."
    )


# ---------------------------------------------------------------------------
# 14. thresholding and continuous W are not mutated by residual estimation
# ---------------------------------------------------------------------------


def test_continuous_w_unchanged_after_residual_estimation(monkeypatch):
    """_continuous_w_pre_threshold must not be modified by residual estimation."""
    wrapper, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    W_snap = wrapper._continuous_w_pre_threshold.copy()

    # Accessing thresholded adjacency at multiple thresholds recomputes
    # from _continuous_w_pre_threshold; verify the underlying field is intact.
    _ = wrapper.thresholded_adjacency(0.3)
    _ = wrapper.thresholded_adjacency(0.2)

    np.testing.assert_array_equal(wrapper._continuous_w_pre_threshold, W_snap)


# ---------------------------------------------------------------------------
# 15. failed fit does not leave usable residual-estimation fields
# ---------------------------------------------------------------------------


def test_failed_fit_leaves_residual_fields_none(monkeypatch):
    """After a failed fit, all residual-estimation fields remain None."""

    class _RaisingFake:
        h_final = None
        score_final = None

        def __init__(self, loss_type: str = "l2") -> None:
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear", _RaisingFake
    )
    X = np.random.default_rng(5).standard_normal((10, 3))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="simulated fit failure"):
        wrapper.fit(X, preprocessor=pre, seed=0)

    assert wrapper._X_train_model_frame is None
    assert wrapper._w_sample_residual_fitted is None
    assert wrapper._sigma_vector_residual_fitted is None


# ---------------------------------------------------------------------------
# estimate_residual_sigmas input validation
# ---------------------------------------------------------------------------

from symbolic_priors_cd.wrappers._dagma_sampling import estimate_residual_sigmas  # noqa: E402


def _valid_inputs(n: int = 5, d: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid (X, W, A) inputs for estimate_residual_sigmas."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((n, d))
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 0.8
    A = (np.abs(W) >= 0.3).astype(bool)
    return X, W, A


def test_validate_x_not_2d_raises():
    """X_model_frame that is not 2D must raise ValueError."""
    _, W, A = _valid_inputs()
    X_1d = np.ones(5)
    with pytest.raises(ValueError, match="2D"):
        estimate_residual_sigmas(X_1d, W, A)

    X_3d = np.ones((5, 3, 1))
    with pytest.raises(ValueError, match="2D"):
        estimate_residual_sigmas(X_3d, W, A)


def test_validate_w_not_2d_raises():
    """W_continuous that is not 2D must raise ValueError."""
    X, _, A = _valid_inputs()
    W_1d = np.ones(3)
    with pytest.raises(ValueError, match="square 2D"):
        estimate_residual_sigmas(X, W_1d, A)


def test_validate_w_not_square_raises():
    """W_continuous that is 2D but not square must raise ValueError."""
    X, _, A = _valid_inputs()
    W_rect = np.zeros((3, 4), dtype=float)
    A_wrong = np.zeros((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="square 2D"):
        estimate_residual_sigmas(X, W_rect, A_wrong)


def test_validate_a_thresh_shape_mismatch_raises():
    """A_thresh with a different shape than W_continuous must raise ValueError."""
    X, W, _ = _valid_inputs()
    A_wrong = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        estimate_residual_sigmas(X, W, A_wrong)


def test_validate_a_thresh_non_bool_raises():
    """A_thresh with dtype other than bool must raise TypeError."""
    X, W, _ = _valid_inputs()
    A_int = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(TypeError, match="bool"):
        estimate_residual_sigmas(X, W, A_int)

    A_float = np.zeros((3, 3), dtype=float)
    with pytest.raises(TypeError, match="bool"):
        estimate_residual_sigmas(X, W, A_float)


def test_validate_x_column_count_mismatch_raises():
    """X_model_frame with column count != W_continuous rows must raise ValueError."""
    _, W, A = _valid_inputs(d=3)
    X_wrong = np.ones((5, 4))  # 4 columns, W is 3x3
    with pytest.raises(ValueError, match="column"):
        estimate_residual_sigmas(X_wrong, W, A)


def test_valid_inputs_do_not_raise():
    """A consistent set of valid inputs must not raise any error."""
    X, W, A = _valid_inputs()
    w_sample, sigma = estimate_residual_sigmas(X, W, A)
    assert w_sample.shape == W.shape
    assert sigma.shape == (W.shape[0],)
