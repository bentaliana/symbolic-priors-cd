"""Tests for canonical continuous-W preservation and native_edge_continuous().

Verifies that the wrapper:
- raises RuntimeError when native_edge_continuous is called before fit;
- stores the continuous W matrix without thresholding, sign changes, or
  transposition;
- always returns defensive copies so callers cannot corrupt internal state;
- does not create the continuous-W field when a fit fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform

# ---------------------------------------------------------------------------
# A known W matrix with signed and sub-threshold entries used throughout.
# Positive entries: 0.7, 0.8.
# Negative entries: -0.25, -0.4, -0.1.
# Sub-threshold by abs value (< 0.3): 0.05, -0.1, -0.25.
# ---------------------------------------------------------------------------
_KNOWN_W = np.array(
    [
        [0.0,   0.7,  -0.25],
        [-0.4,  0.0,   0.05],
        [0.8,  -0.1,   0.0 ],
    ],
    dtype=float,
)


class _KnownWFake:
    """Fake DagmaLinear that always returns _KNOWN_W."""

    h_final = 1.5e-7
    score_final = -5.0

    def __init__(self, loss_type: str = "l2") -> None:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return _KNOWN_W.copy()


class _RaisingFake:
    """Fake DagmaLinear whose fit always raises."""

    h_final = None
    score_final = None

    def __init__(self, loss_type: str = "l2") -> None:
        pass

    def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
        raise RuntimeError("simulated fit failure")


@pytest.fixture
def fitted_wrapper(monkeypatch) -> DAGMAWrapper:
    """Return a DAGMAWrapper that has been fitted with the _KnownWFake."""
    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear", _KnownWFake
    )
    X = np.random.default_rng(0).standard_normal((20, 3))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)
    return wrapper


# ---------------------------------------------------------------------------
# Pre-fit and failed-fit semantics
# ---------------------------------------------------------------------------


def test_native_edge_continuous_raises_before_fit():
    """Calling native_edge_continuous on an unfitted wrapper raises RuntimeError."""
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="unfitted"):
        wrapper.native_edge_continuous()


def test_native_edge_continuous_raises_after_failed_fit(monkeypatch):
    """After a failed fit, native_edge_continuous still raises RuntimeError."""
    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear", _RaisingFake
    )
    X = np.random.default_rng(1).standard_normal((10, 3))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="simulated fit failure"):
        wrapper.fit(X, preprocessor=pre, seed=0)
    with pytest.raises(RuntimeError, match="unfitted"):
        wrapper.native_edge_continuous()


def test_failed_fit_does_not_create_continuous_w_field(monkeypatch):
    """A failed fit must not leave _continuous_w_pre_threshold on the wrapper."""
    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear", _RaisingFake
    )
    X = np.random.default_rng(2).standard_normal((10, 3))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError):
        wrapper.fit(X, preprocessor=pre, seed=0)
    assert not hasattr(wrapper, "_continuous_w_pre_threshold"), (
        "_continuous_w_pre_threshold must not be set after a failed fit."
    )


# ---------------------------------------------------------------------------
# Correctness of stored W
# ---------------------------------------------------------------------------


def test_native_edge_continuous_returns_fitted_w(fitted_wrapper):
    """native_edge_continuous must return an array equal to the fitted W."""
    W = fitted_wrapper.native_edge_continuous()
    np.testing.assert_array_equal(W, _KNOWN_W)


def test_continuous_w_pre_threshold_is_copy_of_fit_result_w(fitted_wrapper):
    """_continuous_w_pre_threshold must equal _fit_result.W but be a distinct object."""
    assert fitted_wrapper._continuous_w_pre_threshold is not fitted_wrapper._fit_result.W
    np.testing.assert_array_equal(
        fitted_wrapper._continuous_w_pre_threshold,
        fitted_wrapper._fit_result.W,
    )


def test_signed_weights_are_preserved(fitted_wrapper):
    """Negative entries in W must not be zeroed or sign-flipped."""
    W = fitted_wrapper.native_edge_continuous()
    assert W[1, 0] == pytest.approx(-0.4)
    assert W[0, 2] == pytest.approx(-0.25)
    assert W[2, 1] == pytest.approx(-0.1)


def test_sub_threshold_weights_are_preserved(fitted_wrapper):
    """Entries with abs(W) < 0.3 must not be zeroed by the wrapper."""
    W = fitted_wrapper.native_edge_continuous()
    assert W[1, 2] == pytest.approx(0.05)
    assert W[0, 2] == pytest.approx(-0.25)
    assert W[2, 1] == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Defensive copy semantics
# ---------------------------------------------------------------------------


def test_native_edge_continuous_returns_copy_not_internal(fitted_wrapper):
    """The returned array must be a distinct object from the internal field."""
    W = fitted_wrapper.native_edge_continuous()
    assert W is not fitted_wrapper._continuous_w_pre_threshold


def test_mutating_returned_w_does_not_affect_internal_state(fitted_wrapper):
    """Mutating the returned array must not change the wrapper's stored W."""
    W = fitted_wrapper.native_edge_continuous()
    sentinel = 999.0
    W[0, 1] = sentinel
    W_after = fitted_wrapper.native_edge_continuous()
    assert W_after[0, 1] != sentinel, (
        "Mutating the returned W changed the wrapper's internal "
        "_continuous_w_pre_threshold."
    )
    np.testing.assert_array_equal(W_after, _KNOWN_W)


def test_repeated_calls_return_distinct_objects_with_equal_values(fitted_wrapper):
    """Each call to native_edge_continuous must return a distinct object."""
    W1 = fitted_wrapper.native_edge_continuous()
    W2 = fitted_wrapper.native_edge_continuous()
    assert W1 is not W2
    np.testing.assert_array_equal(W1, W2)
