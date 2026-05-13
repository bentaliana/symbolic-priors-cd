"""Tests for DAGMA thresholding, graph-status classification, and the
no-silent-repair guarantee.

Uses a monkeypatched DagmaLinear so each scenario can inject a known
continuous W and exercise the wrapper's threshold-and-classify path
deterministically without a full DAGMA run.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_priors_cd.wrappers._graph_status import (
    classify_graph_status,
    infer_sampler_status,
)
from symbolic_priors_cd.wrappers.dagma import (
    DAGMAConfig,
    DAGMAWrapper,
    _threshold_continuous_w,
)
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform


# ---------------------------------------------------------------------------
# Helpers: monkeypatched DagmaLinear that returns an injected W matrix
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


def _build_fitted_wrapper(monkeypatch, W: np.ndarray) -> DAGMAWrapper:
    """Return a fitted DAGMAWrapper whose continuous W equals the given matrix."""
    _patch_dagma_with_W(monkeypatch, W)
    d = W.shape[0]
    X = np.random.default_rng(0).standard_normal((20, d))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)
    return wrapper


# A signed, sub-threshold-rich W used across many tests.
_KNOWN_W = np.array(
    [
        [0.0,   0.5,  -0.25],
        [-0.4,  0.0,   0.05],
        [0.7,  -0.1,   0.0 ],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Standalone _threshold_continuous_w helper
# ---------------------------------------------------------------------------


def test_threshold_continuous_w_returns_bool_array():
    """_threshold_continuous_w returns a strict bool ndarray of the same shape."""
    result = _threshold_continuous_w(_KNOWN_W, 0.3)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert result.shape == _KNOWN_W.shape


def test_threshold_continuous_w_uses_abs_value():
    """Negative weights with abs above the threshold map to True."""
    result = _threshold_continuous_w(_KNOWN_W, 0.3)
    # _KNOWN_W[1, 0] = -0.4, abs = 0.4 >= 0.3 -> True.
    assert result[1, 0]
    # _KNOWN_W[0, 1] = 0.5, abs = 0.5 >= 0.3 -> True.
    assert result[0, 1]
    # _KNOWN_W[2, 0] = 0.7, abs = 0.7 >= 0.3 -> True.
    assert result[2, 0]


def test_threshold_continuous_w_drops_sub_threshold_entries():
    """Entries with abs below the threshold map to False."""
    result = _threshold_continuous_w(_KNOWN_W, 0.3)
    # _KNOWN_W[0, 2] = -0.25, abs = 0.25 < 0.3 -> False.
    assert not result[0, 2]
    # _KNOWN_W[1, 2] = 0.05, abs = 0.05 < 0.3 -> False.
    assert not result[1, 2]
    # _KNOWN_W[2, 1] = -0.1, abs = 0.1 < 0.3 -> False.
    assert not result[2, 1]


def test_threshold_continuous_w_does_not_mutate_input():
    """The helper must not modify its input array."""
    W = _KNOWN_W.copy()
    before = W.copy()
    _ = _threshold_continuous_w(W, 0.3)
    np.testing.assert_array_equal(W, before)


# ---------------------------------------------------------------------------
# DAGMAWrapper.thresholded_adjacency: unfitted, defaults, abs, shape
# ---------------------------------------------------------------------------


def test_thresholded_adjacency_raises_before_fit():
    """thresholded_adjacency on an unfitted wrapper raises RuntimeError."""
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="unfitted"):
        wrapper.thresholded_adjacency()


def test_thresholded_adjacency_default_threshold_is_0_3(monkeypatch):
    """The default threshold is 0.3 and matches the explicit value."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    a_default = wrapper.thresholded_adjacency()
    a_explicit = wrapper.thresholded_adjacency(threshold=0.3)
    np.testing.assert_array_equal(a_default, a_explicit)


def test_thresholded_adjacency_returns_bool_dtype_and_shape(monkeypatch):
    """The returned array has dtype bool and shape (d, d)."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    A = wrapper.thresholded_adjacency()
    assert A.dtype == bool
    assert A.shape == _KNOWN_W.shape


def test_thresholded_adjacency_uses_abs_w(monkeypatch):
    """Negative weights above threshold map to True (abs is applied)."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    A = wrapper.thresholded_adjacency(0.3)
    # _KNOWN_W[1, 0] = -0.4 -> True after abs threshold.
    assert A[1, 0]


def test_thresholded_adjacency_drops_sub_threshold_in_adjacency(monkeypatch):
    """Sub-threshold weights are preserved in continuous W but False in adjacency."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    W = wrapper.native_edge_continuous()
    A = wrapper.thresholded_adjacency(0.3)
    # Continuous W retains the small value verbatim.
    assert W[0, 2] == pytest.approx(-0.25)
    # Adjacency at 0.3 drops it.
    assert not A[0, 2]


# ---------------------------------------------------------------------------
# Orientation, defensive copy, monotonicity
# ---------------------------------------------------------------------------


def test_thresholded_adjacency_preserves_row_source_orientation(monkeypatch):
    """Edge (i, j) at row i, column j is True when abs(W[i, j]) >= threshold."""
    # Place a single signed edge from row 0 (source) to column 1 (destination).
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = -0.9
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    A = wrapper.thresholded_adjacency(0.3)
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 1] = True
    np.testing.assert_array_equal(A, expected)


def test_thresholded_adjacency_returns_copy_not_internal_buffer(monkeypatch):
    """Mutating the returned adjacency must not change the internal continuous W
    or subsequent threshold results."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    A = wrapper.thresholded_adjacency(0.3)
    W_before = wrapper.native_edge_continuous()
    A[:] = False
    A_again = wrapper.thresholded_adjacency(0.3)
    # Continuous W is unchanged.
    np.testing.assert_array_equal(wrapper.native_edge_continuous(), W_before)
    # The fresh thresholded adjacency is unaffected by mutating an earlier
    # return value (proves each call returns an independent array).
    assert A_again[1, 0]


def test_thresholding_does_not_mutate_continuous_w(monkeypatch):
    """A thresholded_adjacency call must not modify _continuous_w_pre_threshold."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    W_before = wrapper._continuous_w_pre_threshold.copy()
    _ = wrapper.thresholded_adjacency(0.3)
    _ = wrapper.thresholded_adjacency(0.4)
    _ = wrapper.thresholded_adjacency(0.2)
    np.testing.assert_array_equal(wrapper._continuous_w_pre_threshold, W_before)


def test_threshold_monotonicity_across_0_2_0_3_0_4(monkeypatch):
    """Edge count is weakly non-increasing as the threshold increases."""
    wrapper = _build_fitted_wrapper(monkeypatch, _KNOWN_W)
    c_020 = int(wrapper.thresholded_adjacency(0.2).sum())
    c_030 = int(wrapper.thresholded_adjacency(0.3).sum())
    c_040 = int(wrapper.thresholded_adjacency(0.4).sum())
    assert c_020 >= c_030 >= c_040


# ---------------------------------------------------------------------------
# Graph status: classification branches and no silent repair
# ---------------------------------------------------------------------------


def test_fit_classifies_valid_dag(monkeypatch):
    """A clean DAG yields graph_status = valid_dag and sampler_status = available."""
    # Chain 0 -> 1 -> 2 with strong weights.
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.9
    W[1, 2] = 0.8
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    assert wrapper._graph_status == "valid_dag"
    assert wrapper._graph_invalid_reason is None
    assert wrapper._sampler_status == "available"
    assert wrapper._sampler_unavailable_reason is None


def test_fit_classifies_cyclic_without_repair(monkeypatch):
    """A 3-cycle yields graph_status = cyclic and is not repaired."""
    W = np.array(
        [
            [0.0,  0.9,  0.0],
            [0.0,  0.0,  0.8],
            [0.7,  0.0,  0.0],
        ],
        dtype=float,
    )
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    assert wrapper._graph_status == "cyclic"
    assert wrapper._graph_invalid_reason is not None
    A = wrapper.thresholded_adjacency(0.3)
    # All three edges of the cycle remain present; no silent repair.
    assert A[0, 1] and A[1, 2] and A[2, 0]


def test_fit_classifies_bidirected_without_repair(monkeypatch):
    """An opposing edge pair yields graph_status = bidirected and is preserved."""
    W = np.array(
        [
            [0.0,  0.9,  0.0],
            [0.8,  0.0,  0.0],
            [0.0,  0.0,  0.0],
        ],
        dtype=float,
    )
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    assert wrapper._graph_status == "bidirected"
    assert wrapper._graph_invalid_reason is not None
    A = wrapper.thresholded_adjacency(0.3)
    # Both opposing edges remain; no symmetrisation or larger-wins rule.
    assert A[0, 1] and A[1, 0]


def test_fit_classifies_self_loop_without_repair(monkeypatch):
    """A diagonal entry that crosses the threshold yields graph_status = self_loop."""
    W = np.array(
        [
            [0.9,  0.0,  0.0],
            [0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0],
        ],
        dtype=float,
    )
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    assert wrapper._graph_status == "self_loop"
    assert wrapper._graph_invalid_reason is not None
    A = wrapper.thresholded_adjacency(0.3)
    # The self-loop is still on the diagonal; no silent zeroing.
    assert A[0, 0]


def test_fit_invalid_graph_sets_sampler_unavailable(monkeypatch):
    """All non-valid-dag graph statuses produce sampler_status =
    unavailable_invalid_graph with a non-empty reason."""
    cyclic_W = np.array(
        [
            [0.0,  0.9,  0.0],
            [0.0,  0.0,  0.8],
            [0.7,  0.0,  0.0],
        ],
        dtype=float,
    )
    wrapper = _build_fitted_wrapper(monkeypatch, cyclic_W)
    assert wrapper._sampler_status == "unavailable_invalid_graph"
    assert wrapper._sampler_unavailable_reason is not None
    assert "cyclic" in wrapper._sampler_unavailable_reason


def test_valid_dag_sets_sampler_available(monkeypatch):
    """A valid DAG yields sampler_status = available with no reason."""
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.9
    W[1, 2] = 0.8
    wrapper = _build_fitted_wrapper(monkeypatch, W)
    assert wrapper._sampler_status == "available"
    assert wrapper._sampler_unavailable_reason is None


# ---------------------------------------------------------------------------
# Invalid-shape branch via the shared helper directly
# ---------------------------------------------------------------------------


def test_shared_helper_invalid_shape_branch():
    """Non-square bool arrays classify as invalid_shape with a reason."""
    non_square = np.zeros((2, 3), dtype=bool)
    status, reason = classify_graph_status(non_square)
    assert status == "invalid_shape"
    assert reason is not None
    # Sampler-status mapping for invalid_shape.
    sampler_status, sampler_reason = infer_sampler_status(status)
    assert sampler_status == "unavailable_invalid_graph"
    assert sampler_reason is not None


def test_classify_graph_status_rejects_non_bool_square_input():
    """A square but non-bool adjacency raises TypeError; shape passes the
    invalid_shape check first, so the dtype guard fires next."""
    int_square = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(TypeError, match="bool"):
        classify_graph_status(int_square)

    float_square = np.zeros((3, 3), dtype=float)
    with pytest.raises(TypeError, match="bool"):
        classify_graph_status(float_square)
