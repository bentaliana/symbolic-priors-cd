"""Tests for the DAGMA soft-prior variant.

Covers prior_gradient correctness, validation, prior-free equivalence
under zero lambda or zero mask, post-hoc masking detection, shrinkage
direction and softness, monotonic suppression under increasing penalty,
and integration through run_soft_prior_dagma_fit.

All tests use small fits to keep runtime bounded. Tested edge indices
are derived dynamically from the baseline W; no hardcoded indices and
no reference to the data-generating graph are used in test logic.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from symbolic_priors_cd.wrappers._dagma_fit import (
    run_dagma_fit,
    run_soft_prior_dagma_fit,
)
from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear
from symbolic_priors_cd.wrappers._soft_prior_dagma import (
    SoftPriorDagmaLinear,
    prior_gradient,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fast-test configuration and deterministic data
# ---------------------------------------------------------------------------

_D = 4
_N = 500
_X_SEED = 12345

_FAST_CFG_KWARGS: dict = dict(
    lambda1=0.05,
    w_threshold=0.0,
    T=4,
    mu_init=1.0,
    mu_factor=0.1,
    s=[1.0, 0.9, 0.8, 0.7],
    warm_iter=2000,
    max_iter=4000,
    lr=3e-4,
    checkpoint=1000,
    beta_1=0.99,
    beta_2=0.999,
)


def _generate_chain_sem_data(d: int, n: int, seed: int) -> np.ndarray:
    """Return observational data drawn from a hidden linear-Gaussian SCM.

    The SCM structure and weights are intentionally encapsulated inside
    this helper and are not returned: callers only see ``X``. Tests
    must not depend on the data-generating graph.
    """
    rng = np.random.default_rng(seed)
    edge_weight = 0.9
    noise_scale = 0.3
    w_internal = np.zeros((d, d), dtype=float)
    for i in range(d - 1):
        w_internal[i, i + 1] = edge_weight
    noise = rng.standard_normal((n, d)) * noise_scale
    return noise @ np.linalg.inv(np.eye(d) - w_internal)


def _find_off_diagonal_entry_at_or_above(
    W: np.ndarray, threshold: float
) -> tuple[int, int] | None:
    """Return the first off-diagonal ``(i, j)`` with ``abs(W) >= threshold``.

    Returns ``None`` if no off-diagonal entry meets the threshold.
    """
    d = W.shape[0]
    abs_w = np.abs(W)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if abs_w[i, j] >= threshold:
                return (i, j)
    return None


def _make_mask_with_single_entry(d: int, i: int, j: int) -> np.ndarray:
    mask = np.zeros((d, d), dtype=float)
    mask[i, j] = 1.0
    return mask


def _fit_soft_prior(
    x_data: np.ndarray,
    lambda_prior: float,
    confidence_mask: np.ndarray,
) -> np.ndarray:
    """Fit a SoftPriorDagmaLinear on a copy of ``x_data`` and return W."""
    model = SoftPriorDagmaLinear(
        loss_type="l2",
        lambda_prior=lambda_prior,
        confidence_mask=confidence_mask,
    )
    return model.fit(X=x_data.copy(), **_FAST_CFG_KWARGS)


@pytest.fixture(scope="module")
def x_data() -> np.ndarray:
    return _generate_chain_sem_data(_D, _N, _X_SEED)


@pytest.fixture(scope="module")
def w_prior_free(x_data: np.ndarray) -> np.ndarray:
    """Baseline prior-free DagmaLinear fit on the test data."""
    model = DagmaLinear(loss_type="l2")
    return model.fit(
        X=x_data.copy(),
        exclude_edges=None,
        include_edges=None,
        **_FAST_CFG_KWARGS,
    )


# ---------------------------------------------------------------------------
# Test 1: exact prior-gradient formula
# ---------------------------------------------------------------------------


def test_prior_gradient_exact_formula():
    """prior_gradient must equal 2 * lambda_prior * C * W element-wise."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3, 3))
    C = np.array(
        [
            [0.0, 0.5, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    lambda_prior = 0.7
    expected = 2.0 * lambda_prior * C * W
    actual = prior_gradient(W, C, lambda_prior)
    np.testing.assert_array_equal(actual, expected)


def test_prior_gradient_zero_mask_positions_are_exactly_zero():
    """Positions where C is zero must produce exactly zero output."""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((4, 4)) * 5.0
    C = np.zeros((4, 4), dtype=float)
    C[0, 1] = 1.0
    C[2, 3] = 0.3
    out = prior_gradient(W, C, lambda_prior=0.5)
    zero_positions = C == 0.0
    assert np.all(out[zero_positions] == 0.0)


def test_prior_gradient_does_not_accept_mu_argument():
    """The prior gradient must not depend on mu or any path parameter."""
    sig = inspect.signature(prior_gradient)
    assert "mu" not in sig.parameters
    W = np.eye(3)
    C = np.zeros((3, 3))
    with pytest.raises(TypeError):
        prior_gradient(W, C, 0.1, mu=1.0)


# ---------------------------------------------------------------------------
# Test 2: validation
# ---------------------------------------------------------------------------


def _ok_mask(d: int = 3) -> np.ndarray:
    return np.zeros((d, d), dtype=float)


def test_validation_negative_lambda_raises():
    with pytest.raises(ValueError, match="non-negative"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=-0.1, confidence_mask=_ok_mask()
        )


def test_validation_nan_lambda_raises():
    with pytest.raises(ValueError, match="NaN"):
        SoftPriorDagmaLinear(
            loss_type="l2",
            lambda_prior=float("nan"),
            confidence_mask=_ok_mask(),
        )


def test_validation_infinite_lambda_raises():
    with pytest.raises(ValueError, match="finite"):
        SoftPriorDagmaLinear(
            loss_type="l2",
            lambda_prior=float("inf"),
            confidence_mask=_ok_mask(),
        )


def test_validation_nonsquare_mask_raises():
    bad = np.zeros((3, 4), dtype=float)
    with pytest.raises(ValueError, match="square"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=0.1, confidence_mask=bad
        )


def test_validation_wrong_shape_mask_raises():
    """The mask must match the variable count of X at fit time."""
    mask_4 = np.zeros((4, 4), dtype=float)
    model = SoftPriorDagmaLinear(
        loss_type="l2", lambda_prior=0.1, confidence_mask=mask_4
    )
    X_three_vars = np.zeros((5, 3), dtype=float)
    with pytest.raises(ValueError, match="match"):
        model.fit(X=X_three_vars, **_FAST_CFG_KWARGS)


def test_validation_negative_mask_entry_raises():
    bad = np.zeros((3, 3), dtype=float)
    bad[0, 1] = -0.5
    with pytest.raises(ValueError, match="negative"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=0.1, confidence_mask=bad
        )


def test_validation_nan_mask_entry_raises():
    bad = np.zeros((3, 3), dtype=float)
    bad[0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=0.1, confidence_mask=bad
        )


def test_validation_infinite_mask_entry_raises():
    bad = np.zeros((3, 3), dtype=float)
    bad[0, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=0.1, confidence_mask=bad
        )


def test_validation_nonzero_diagonal_raises():
    bad = np.zeros((3, 3), dtype=float)
    bad[1, 1] = 0.5
    with pytest.raises(ValueError, match="diagonal"):
        SoftPriorDagmaLinear(
            loss_type="l2", lambda_prior=0.1, confidence_mask=bad
        )


# ---------------------------------------------------------------------------
# Test 3: zero-lambda equivalence
# ---------------------------------------------------------------------------


def test_zero_lambda_w_matches_prior_free(x_data, w_prior_free):
    """With lambda_prior=0 and a nonzero mask, W must match the prior-free fit."""
    mask = np.ones((_D, _D), dtype=float)
    np.fill_diagonal(mask, 0.0)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.0, confidence_mask=mask)
    delta = float(np.max(np.abs(w_soft - w_prior_free)))
    assert delta < 1e-10, (
        f"max |W_soft - W_prior_free| = {delta:.3e} (expected < 1e-10)"
    )


def test_zero_lambda_thresholded_matches_prior_free(x_data, w_prior_free):
    """Thresholded adjacency at 0.3 must match between the two fits."""
    mask = np.ones((_D, _D), dtype=float)
    np.fill_diagonal(mask, 0.0)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.0, confidence_mask=mask)
    a_soft = np.abs(w_soft) >= 0.3
    a_pf = np.abs(w_prior_free) >= 0.3
    np.testing.assert_array_equal(a_soft, a_pf)


# ---------------------------------------------------------------------------
# Test 4: zero-mask equivalence
# ---------------------------------------------------------------------------


def test_zero_mask_w_matches_prior_free(x_data, w_prior_free):
    """With lambda_prior>0 and an all-zero mask, W must match the prior-free fit."""
    mask = np.zeros((_D, _D), dtype=float)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.5, confidence_mask=mask)
    delta = float(np.max(np.abs(w_soft - w_prior_free)))
    assert delta < 1e-10, (
        f"max |W_soft - W_prior_free| = {delta:.3e} (expected < 1e-10)"
    )


def test_zero_mask_thresholded_matches_prior_free(x_data, w_prior_free):
    """Thresholded adjacency at 0.3 must match between the two fits."""
    mask = np.zeros((_D, _D), dtype=float)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.5, confidence_mask=mask)
    a_soft = np.abs(w_soft) >= 0.3
    a_pf = np.abs(w_prior_free) >= 0.3
    np.testing.assert_array_equal(a_soft, a_pf)


# ---------------------------------------------------------------------------
# Test 5: post-hoc masking detector
# ---------------------------------------------------------------------------


def test_post_hoc_masking_detector_under_zero_lambda(x_data, w_prior_free):
    """A 1-entry mask at lambda_prior=0 must not alter W at that entry.

    A post-hoc masking implementation would change ``W[i, j]`` because
    ``mask[i, j] = 1.0``. A gradient-only implementation produces no
    change when ``lambda_prior = 0`` because the prior gradient is the
    zero matrix.
    """
    entry = _find_off_diagonal_entry_at_or_above(w_prior_free, 0.1)
    if entry is None:
        pytest.skip(
            "No off-diagonal baseline entry with abs(W) >= 0.1; "
            "cannot exercise the detector."
        )
    i, j = entry
    mask = _make_mask_with_single_entry(_D, i, j)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.0, confidence_mask=mask)
    diff = float(abs(w_soft[i, j] - w_prior_free[i, j]))
    assert diff < 1e-10, (
        f"|W_soft[{i},{j}] - W_pf[{i},{j}]| = {diff:.3e} "
        "(expected < 1e-10 under a gradient-only implementation)"
    )


# ---------------------------------------------------------------------------
# Test 6: shrinkage direction and softness
# ---------------------------------------------------------------------------


def test_shrinkage_direction_and_softness(x_data, w_prior_free):
    """A nonzero penalty must shrink the targeted entry without clamping."""
    entry = None
    for th in (0.3, 0.2, 0.1):
        entry = _find_off_diagonal_entry_at_or_above(w_prior_free, th)
        if entry is not None:
            break
    if entry is None:
        pytest.skip(
            "No off-diagonal baseline entry with abs(W) >= 0.1 even at "
            "the smallest threshold; cannot exercise shrinkage direction."
        )
    i, j = entry
    mask = _make_mask_with_single_entry(_D, i, j)
    w_soft = _fit_soft_prior(x_data, lambda_prior=0.1, confidence_mask=mask)
    base_abs = float(abs(w_prior_free[i, j]))
    soft_abs = float(abs(w_soft[i, j]))
    # Shrinkage direction: soft must be smaller in magnitude.
    assert soft_abs < base_abs, (
        f"Expected |W_soft[{i},{j}]| < |W_base[{i},{j}]|; "
        f"got soft={soft_abs:.6f}, base={base_abs:.6f}."
    )
    # Soft, not a hard clamp.
    assert soft_abs > 1e-6, (
        f"|W_soft[{i},{j}]| = {soft_abs:.3e} <= 1e-6 looks like a hard clamp."
    )


# ---------------------------------------------------------------------------
# Test 7: monotonic suppression under increasing penalty
# ---------------------------------------------------------------------------


def test_monotonic_suppression_under_increasing_penalty(x_data, w_prior_free):
    """abs(W[i,j]) must strictly decrease as lambda_prior increases."""
    entry = None
    for th in (0.3, 0.2, 0.1):
        entry = _find_off_diagonal_entry_at_or_above(w_prior_free, th)
        if entry is not None:
            break
    if entry is None:
        pytest.skip(
            "No off-diagonal baseline entry; cannot exercise monotonicity."
        )
    i, j = entry
    mask = _make_mask_with_single_entry(_D, i, j)
    lambdas = [0.01, 0.05, 0.2]
    abs_values: list[float] = []
    for lam in lambdas:
        w_soft = _fit_soft_prior(
            x_data, lambda_prior=lam, confidence_mask=mask
        )
        abs_values.append(float(abs(w_soft[i, j])))
    assert (
        abs_values[0] > abs_values[1] > abs_values[2]
    ), (
        f"Expected strict monotonic decrease across lambdas {lambdas}; "
        f"observed abs(W[{i},{j}]) = {abs_values}."
    )


# ---------------------------------------------------------------------------
# Test 8: direct gradient-mask correctness
# ---------------------------------------------------------------------------


def test_prior_gradient_zero_at_masked_zero_positions_isolated():
    """Output is zero at every position where the mask is zero."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((5, 5)) * 10.0
    C = np.zeros((5, 5), dtype=float)
    C[0, 1] = 1.0
    C[1, 2] = 0.5
    C[3, 4] = 2.0
    out = prior_gradient(W, C, lambda_prior=0.4)
    zero_positions = C == 0.0
    nonzero_positions = ~zero_positions
    assert np.all(out[zero_positions] == 0.0)
    # At nonzero positions the output equals 2 * lambda * C * W exactly.
    np.testing.assert_array_equal(
        out[nonzero_positions], (2.0 * 0.4 * C * W)[nonzero_positions]
    )


# ---------------------------------------------------------------------------
# Test 9: fit-helper integration
# ---------------------------------------------------------------------------


def _fast_cfg_for_helper_integration() -> DAGMAConfig:
    return DAGMAConfig(
        T=4,
        lambda1=0.05,
        s=(1.0, 0.9, 0.8, 0.7),
        mu_init=1.0,
        mu_factor=0.1,
        w_threshold_internal=0.0,
        lr=3e-4,
        warm_iter=2000,
        max_iter=4000,
        beta_1=0.99,
        beta_2=0.999,
        loss_type="l2",
    )


def test_fit_helper_agrees_with_prior_free_under_zero_inputs(x_data):
    """run_soft_prior_dagma_fit must match run_dagma_fit at lambda=0, mask=0."""
    cfg = _fast_cfg_for_helper_integration()
    mask = np.zeros((_D, _D), dtype=float)
    res_pf = run_dagma_fit(x_data.copy(), cfg)
    res_soft = run_soft_prior_dagma_fit(
        x_data.copy(), cfg, lambda_prior=0.0, confidence_mask=mask
    )
    delta = float(np.max(np.abs(res_soft.W - res_pf.W)))
    assert delta < 1e-10, (
        f"max |W_soft - W_pf| = {delta:.3e} (expected < 1e-10)"
    )


def test_fit_helper_returns_identical_fields(x_data):
    """The returned dataclasses must expose identical field names."""
    cfg = _fast_cfg_for_helper_integration()
    mask = np.zeros((_D, _D), dtype=float)
    res_pf = run_dagma_fit(x_data.copy(), cfg)
    res_soft = run_soft_prior_dagma_fit(
        x_data.copy(), cfg, lambda_prior=0.0, confidence_mask=mask
    )
    fields_pf = {f.name for f in dataclasses.fields(res_pf)}
    fields_soft = {f.name for f in dataclasses.fields(res_soft)}
    assert fields_pf == fields_soft, (
        f"prior-free fields = {fields_pf}, soft-prior fields = {fields_soft}"
    )
