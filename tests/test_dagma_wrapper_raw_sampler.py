"""Tests for DAGMAWrapper.sample_interventional raw-unit sampling.

Covers pre-fit guard, raw-unit roundtrip under both preprocessors,
intervention-value transformation, inverse-transform application,
sigma-policy selection, unavailable-sampler cases, input validation,
determinism, no-mutation guarantees, and no global RNG mutation.

All tests use monkeypatched DagmaLinear so no real DAGMA fits are run.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers.dagma import DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)


# ---------------------------------------------------------------------------
# Helpers: fake DagmaLinear and fitted-wrapper builders
# ---------------------------------------------------------------------------


def _patch_dagma_with_W(monkeypatch, W: np.ndarray) -> None:
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
    preprocessor_cls=CentredOnlyTransform,
) -> tuple[DAGMAWrapper, np.ndarray, CentredOnlyTransform | StandardisedTransform]:
    """Return a fitted DAGMAWrapper, the training X, and the fitted preprocessor."""
    _patch_dagma_with_W(monkeypatch, W)
    d = W.shape[0]
    if X is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, d)) + np.arange(d) * 0.5
    pre = preprocessor_cls().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)
    return wrapper, X, pre


# ---------------------------------------------------------------------------
# Known W matrices
# ---------------------------------------------------------------------------

# Valid DAG: chain 0->1->2 with strong weights.
_W_VALID = np.array(
    [[0.0, 0.8, 0.0],
     [0.0, 0.0, 0.6],
     [0.0, 0.0, 0.0]],
    dtype=float,
)

# Valid DAG where W[0,1]=1.0 exactly predicts X[:,1] from X[:,0] when
# X[:,0]==X[:,1], so sigma[1]=0 -> unavailable_unresolved_noise_policy.
_W_DEGEN = np.array(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    dtype=float,
)
_N_DEGEN = 6
_X_DEGEN = np.zeros((_N_DEGEN, 3), dtype=float)
_X_DEGEN[:, 0] = np.arange(_N_DEGEN, dtype=float)
_X_DEGEN[:, 1] = np.arange(_N_DEGEN, dtype=float)   # X[:,1] == X[:,0]
_X_DEGEN[:, 2] = np.arange(_N_DEGEN, dtype=float) + 2.0

# Cyclic W: 3-cycle 0->1->2->0.
_W_CYCLIC = np.array(
    [[0.0, 0.9, 0.0],
     [0.0, 0.0, 0.8],
     [0.7, 0.0, 0.0]],
    dtype=float,
)

_INTERV = Intervention(target=0, value=2.0)

_SAMPLER_PATH = "symbolic_priors_cd.wrappers.dagma.sample_linear_gaussian_model_frame"


def _make_mock_sampler(return_value: np.ndarray | None = None, capture: dict | None = None):
    """Return a mock replacement for sample_linear_gaussian_model_frame."""
    def _mock(A_thresh, W_sample, sigma_vector, *, target, value_model, n_samples, sample_seed):
        if capture is not None:
            capture["A_thresh"] = A_thresh.copy()
            capture["sigma_vector"] = sigma_vector.copy()
            capture["value_model"] = value_model
            capture["target"] = target
            capture["n_samples"] = n_samples
            capture["sample_seed"] = sample_seed
            capture["called"] = True
        n_vars = A_thresh.shape[0]
        if return_value is not None:
            return return_value.copy()
        return np.zeros((n_samples, n_vars), dtype=float)
    return _mock


# ---------------------------------------------------------------------------
# 1. Pre-fit guard
# ---------------------------------------------------------------------------


def test_sample_interventional_raises_before_fit():
    """sample_interventional on an unfitted wrapper must raise RuntimeError."""
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="unfitted"):
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0)


# ---------------------------------------------------------------------------
# 2-4. Basic sampling shape and raw-unit target clamping
# ---------------------------------------------------------------------------


def test_residual_fitted_returns_correct_shape(monkeypatch):
    """residual_fitted sampling returns array of shape (n_samples, n_vars)."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    out = wrapper.sample_interventional(_INTERV, n_samples=8, sample_seed=0)
    assert out is not None
    assert out.shape == (8, 3)
    assert out.dtype == np.float64


def test_target_column_equals_raw_value_centred_only(monkeypatch):
    """Under CentredOnlyTransform, target column must equal raw intervention
    value within atol=1e-12 after the model-frame clamp and inverse-transform."""
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID)
    raw_value = 3.5
    out = wrapper.sample_interventional(
        Intervention(target=1, value=raw_value), n_samples=20, sample_seed=0
    )
    assert out is not None
    np.testing.assert_allclose(out[:, 1], raw_value, atol=1e-12)


def test_target_column_equals_raw_value_standardised(monkeypatch):
    """Under StandardisedTransform, target column must equal raw intervention
    value within atol=1e-12 after the model-frame clamp and inverse-transform."""
    wrapper, _, pre = _build_fitted_wrapper(
        monkeypatch, _W_VALID, preprocessor_cls=StandardisedTransform
    )
    raw_value = -1.5
    out = wrapper.sample_interventional(
        Intervention(target=0, value=raw_value), n_samples=15, sample_seed=0
    )
    assert out is not None
    np.testing.assert_allclose(out[:, 0], raw_value, atol=1e-12)


# ---------------------------------------------------------------------------
# 5-8. Preprocessor roundtrip semantics
# ---------------------------------------------------------------------------


def test_intervention_value_is_transformed_to_model_frame(monkeypatch):
    """The value_model passed to the sampler must be the model-frame value,
    not the raw intervention value."""
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID)
    raw_value = 4.0
    target = 1
    expected_model_value = pre.transform_intervention_value(raw_value, target)
    cap = {}
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(capture=cap)):
        wrapper.sample_interventional(
            Intervention(target=target, value=raw_value), n_samples=5, sample_seed=0
        )
    assert cap["value_model"] == pytest.approx(expected_model_value)
    assert cap["value_model"] != pytest.approx(raw_value)


def test_inverse_transform_applied_to_model_frame_output(monkeypatch):
    """Returned samples must be inverse-transformed from model frame to raw units."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 3)) + np.array([5.0, 2.0, -3.0])
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID, X=X)

    # Mock the inner sampler to return a known model-frame array of zeros.
    mock_model_frame = np.zeros((5, 3), dtype=float)
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(return_value=mock_model_frame)):
        out = wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0)

    # inverse_transform(zeros) = zeros + mean = training mean broadcast over rows.
    expected = pre.inverse_transform(mock_model_frame)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # Verify the output is not model-frame (non-zero mean columns).
    assert not np.allclose(out, mock_model_frame)


def test_preprocessor_fit_not_called_inside_sample_interventional(monkeypatch):
    """sample_interventional must not call preprocessor.fit."""
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID)
    with patch.object(pre, "fit", wraps=pre.fit) as mock_fit:
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0)
    mock_fit.assert_not_called()


def test_preprocessor_statistics_unchanged_after_sampling(monkeypatch):
    """Preprocessor fitted statistics must not be modified by sample_interventional."""
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID)
    mean_before = pre._mean.copy()
    wrapper.sample_interventional(_INTERV, n_samples=10, sample_seed=0)
    np.testing.assert_array_equal(pre._mean, mean_before)


def test_standardised_preprocessor_statistics_unchanged_after_sampling(monkeypatch):
    """StandardisedTransform mean and std must not be modified by sample_interventional."""
    wrapper, _, pre = _build_fitted_wrapper(
        monkeypatch, _W_VALID, preprocessor_cls=StandardisedTransform
    )
    mean_before = pre._mean.copy()
    std_before = pre._std.copy()
    wrapper.sample_interventional(_INTERV, n_samples=10, sample_seed=0)
    np.testing.assert_array_equal(pre._mean, mean_before)
    np.testing.assert_array_equal(pre._std, std_before)


# ---------------------------------------------------------------------------
# 9-12. Sigma-policy selection
# ---------------------------------------------------------------------------


def test_residual_fitted_uses_stored_sigma(monkeypatch):
    """residual_fitted must pass _sigma_vector_residual_fitted to the sampler."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    cap = {}
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(capture=cap)):
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                      noise_policy="residual_fitted")
    np.testing.assert_array_equal(cap["sigma_vector"], wrapper._sigma_vector_residual_fitted)


def test_unit_variance_uses_ones_not_stored_sigma(monkeypatch):
    """unit_variance must pass np.ones(n_vars) to the sampler, not stored sigma."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    n_vars = wrapper._continuous_w_pre_threshold.shape[0]
    cap = {}
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(capture=cap)):
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                      noise_policy="unit_variance")
    np.testing.assert_array_equal(cap["sigma_vector"], np.ones(n_vars))
    # Must differ from residual sigma.
    assert not np.allclose(cap["sigma_vector"], wrapper._sigma_vector_residual_fitted)


def test_unit_variance_does_not_mutate_stored_sigma(monkeypatch):
    """unit_variance must not overwrite _sigma_vector_residual_fitted."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    sigma_before = wrapper._sigma_vector_residual_fitted.copy()
    wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                  noise_policy="unit_variance")
    np.testing.assert_array_equal(wrapper._sigma_vector_residual_fitted, sigma_before)


def test_unit_variance_does_not_overwrite_sampler_status(monkeypatch):
    """unit_variance must not change _sampler_status."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    status_before = wrapper._sampler_status
    wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                  noise_policy="unit_variance")
    assert wrapper._sampler_status == status_before


# ---------------------------------------------------------------------------
# 13-15. Unavailable-sampler cases
# ---------------------------------------------------------------------------


def test_invalid_graph_returns_none_without_calling_sampler(monkeypatch):
    """An invalid thresholded graph must return None and not invoke the sampler."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_CYCLIC)
    assert wrapper._graph_status == "cyclic"
    cap = {"called": False}
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(capture=cap)):
        result = wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0)
    assert result is None
    assert not cap["called"]


def test_residual_fitted_returns_none_when_unresolved_noise_policy(monkeypatch):
    """residual_fitted returns None when _sampler_status is
    unavailable_unresolved_noise_policy (degenerate sigma)."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_DEGEN, X=_X_DEGEN.copy())
    assert wrapper._graph_status == "valid_dag"
    assert wrapper._sampler_status == "unavailable_unresolved_noise_policy"
    result = wrapper.sample_interventional(
        Intervention(target=2, value=1.0), n_samples=5, sample_seed=0,
        noise_policy="residual_fitted",
    )
    assert result is None


def test_unit_variance_runs_when_residual_fitted_unavailable(monkeypatch):
    """unit_variance can sample when graph_status is valid_dag and W_sample
    exists, even when residual_fitted is blocked by degenerate sigma."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_DEGEN, X=_X_DEGEN.copy())
    assert wrapper._sampler_status == "unavailable_unresolved_noise_policy"
    assert wrapper._w_sample_residual_fitted is not None

    out = wrapper.sample_interventional(
        Intervention(target=2, value=1.0), n_samples=5, sample_seed=0,
        noise_policy="unit_variance",
    )
    assert out is not None
    assert out.shape == (5, 3)
    assert np.all(np.isfinite(out))


def test_residual_fitted_raises_if_sigma_none_despite_available_status(monkeypatch):
    """If sampler_status is 'available' but _sigma_vector_residual_fitted is None,
    residual_fitted must raise RuntimeError (internal inconsistency guard)."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    assert wrapper._graph_status == "valid_dag"
    assert wrapper._sampler_status == "available"
    assert wrapper._w_sample_residual_fitted is not None

    # Simulate an internal inconsistency: sigma is missing despite available status.
    wrapper._sigma_vector_residual_fitted = None

    with pytest.raises(RuntimeError, match="inconsistency"):
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                      noise_policy="residual_fitted")


# ---------------------------------------------------------------------------
# 16-18. Input validation
# ---------------------------------------------------------------------------


def test_unsupported_noise_policy_raises_value_error(monkeypatch):
    """An unsupported noise_policy string must raise ValueError."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    with pytest.raises(ValueError, match="noise_policy"):
        wrapper.sample_interventional(_INTERV, n_samples=5, sample_seed=0,
                                      noise_policy="unknown_policy")


def test_invalid_n_samples_raises_value_error(monkeypatch):
    """n_samples < 1 must raise ValueError through the public wrapper path."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    with pytest.raises(ValueError):
        wrapper.sample_interventional(_INTERV, n_samples=0, sample_seed=0)
    with pytest.raises(ValueError):
        wrapper.sample_interventional(_INTERV, n_samples=-1, sample_seed=0)


def test_invalid_target_raises_value_error(monkeypatch):
    """target outside [0, n_vars) must raise ValueError through the public path."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    with pytest.raises(ValueError):
        wrapper.sample_interventional(
            Intervention(target=99, value=1.0), n_samples=5, sample_seed=0
        )


# ---------------------------------------------------------------------------
# 19-22. Determinism and no-mutation
# ---------------------------------------------------------------------------


def test_deterministic_output_for_same_seed(monkeypatch):
    """Two calls with the same seed must produce identical raw-unit samples."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    kwargs = dict(n_samples=15, sample_seed=42)
    out1 = wrapper.sample_interventional(_INTERV, **kwargs)
    out2 = wrapper.sample_interventional(_INTERV, **kwargs)
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_produce_different_stochastic_columns(monkeypatch):
    """Different seeds must produce different non-target columns."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    out1 = wrapper.sample_interventional(
        Intervention(target=0, value=0.0), n_samples=20, sample_seed=1
    )
    out2 = wrapper.sample_interventional(
        Intervention(target=0, value=0.0), n_samples=20, sample_seed=2
    )
    # Node 1 is stochastic; outputs must differ.
    assert not np.allclose(out1[:, 1], out2[:, 1])


def test_no_np_random_seed_call(monkeypatch):
    """sample_interventional must not call np.random.seed."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    calls: list = []
    original = np.random.seed
    np.random.seed = lambda *args: calls.append(args)  # type: ignore[assignment]
    try:
        wrapper.sample_interventional(_INTERV, n_samples=10, sample_seed=5)
    finally:
        np.random.seed = original  # type: ignore[assignment]
    assert calls == [], f"np.random.seed was called with: {calls}"


def test_sample_interventional_does_not_mutate_internal_fields(monkeypatch):
    """sample_interventional must not modify continuous W, W_sample, or sigma."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    W_before = wrapper._continuous_w_pre_threshold.copy()
    W_samp_before = wrapper._w_sample_residual_fitted.copy()
    sigma_before = wrapper._sigma_vector_residual_fitted.copy()

    wrapper.sample_interventional(_INTERV, n_samples=10, sample_seed=0)

    np.testing.assert_array_equal(wrapper._continuous_w_pre_threshold, W_before)
    np.testing.assert_array_equal(wrapper._w_sample_residual_fitted, W_samp_before)
    np.testing.assert_array_equal(wrapper._sigma_vector_residual_fitted, sigma_before)


# ---------------------------------------------------------------------------
# 23. Raw-unit output (not model-frame)
# ---------------------------------------------------------------------------


def test_returned_samples_are_raw_unit_not_model_frame(monkeypatch):
    """Returned samples must be in raw SCM units, not model-frame units.

    Use a CentredOnlyTransform with non-zero training mean. Mock the inner
    sampler to return zeros. inverse_transform(zeros) = training mean, which
    differs from the model-frame zeros, confirming the roundtrip was applied.
    """
    rng = np.random.default_rng(99)
    X = rng.standard_normal((30, 3)) + np.array([10.0, -5.0, 3.0])
    wrapper, _, pre = _build_fitted_wrapper(monkeypatch, _W_VALID, X=X)

    n = 4
    mock_model_frame = np.zeros((n, 3), dtype=float)
    with patch(_SAMPLER_PATH, side_effect=_make_mock_sampler(return_value=mock_model_frame)):
        out = wrapper.sample_interventional(_INTERV, n_samples=n, sample_seed=0)

    # Model-frame zeros != raw-unit output (mean is non-trivial).
    assert not np.allclose(out, mock_model_frame), (
        "Output appears to be model-frame samples rather than raw-unit samples."
    )
    # Raw-unit output equals inverse_transform of zeros = training mean.
    np.testing.assert_allclose(out, pre.inverse_transform(mock_model_frame), atol=1e-12)
