"""Source-faithfulness tests for the DAGMA wrapper.

Verifies that DAGMAWrapper.fit produces the same continuous W matrix
as a direct call to DagmaLinear.fit under the same input, source path,
and hyperparameters. The binding gate is
``np.allclose(W_wrapper, W_direct, atol=1e-12, rtol=1e-12)``.

Two fit pairs are built once per module to keep test runtime bounded:
- ``same_input_fits``: both paths see the same raw input X.
- ``centred_input_fits``: both paths see the same X already centred
  by a CentredOnlyTransform.

The DAGMA fit is deterministic for fixed inputs and hyperparameters,
so the wrapper-vs-direct W comparison is expected to match bitwise on
the project's CPU build; the 1e-12 tolerance is conservative headroom.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from symbolic_priors_cd.wrappers import _dagma_utils
from symbolic_priors_cd.wrappers._dagma_utils import DAGMA_SOURCE_PATH, DagmaLinear
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform


# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

_N_SAMPLES = 50
_N_VARS = 5
_X_SEED = 0
_FIT_SEED = 0
_ATOL = 1e-12
_RTOL = 1e-12


def _make_input_X() -> np.ndarray:
    """Deterministic (n_samples, n_vars) Gaussian matrix."""
    rng = np.random.default_rng(_X_SEED)
    return rng.standard_normal((_N_SAMPLES, _N_VARS))


def _direct_dagma_fit(X: np.ndarray, cfg: DAGMAConfig) -> np.ndarray:
    """Run DagmaLinear.fit directly with the same explicit hyperparameters
    used by DAGMAWrapper.fit."""
    model = DagmaLinear(loss_type=cfg.loss_type)
    return model.fit(
        X=X.copy(),
        lambda1=cfg.lambda1,
        w_threshold=cfg.w_threshold_internal,
        T=cfg.T,
        mu_init=cfg.mu_init,
        mu_factor=cfg.mu_factor,
        s=list(cfg.s),
        warm_iter=cfg.warm_iter,
        max_iter=cfg.max_iter,
        lr=cfg.lr,
        beta_1=cfg.beta_1,
        beta_2=cfg.beta_2,
        exclude_edges=None,
        include_edges=None,
    )


# ---------------------------------------------------------------------------
# Same-input fits: both paths see the raw X
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def same_input_fits():
    """Run a direct fit and a wrapper fit on the same raw X.

    DAGMA centres internally in both paths, so the two fits process
    bit-identical inputs once their internal mean-centring runs.
    """
    X = _make_input_X()
    cfg = DAGMAConfig()

    X_before = X.copy()
    W_direct = _direct_dagma_fit(X, cfg)

    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=_FIT_SEED, config=cfg)
    W_wrapper = wrapper.native_edge_continuous()

    return {
        "W_direct": W_direct,
        "W_wrapper": W_wrapper,
        "X_caller": X,
        "X_caller_before": X_before,
    }


def test_wrapper_w_matches_direct_w_within_tolerance(same_input_fits):
    """Wrapper W must match direct DAGMA W within atol=1e-12, rtol=1e-12."""
    np.testing.assert_allclose(
        same_input_fits["W_wrapper"],
        same_input_fits["W_direct"],
        atol=_ATOL,
        rtol=_RTOL,
    )


def test_wrapper_fit_does_not_mutate_caller_x(same_input_fits):
    """The caller's X array must be bitwise unchanged after fit."""
    np.testing.assert_array_equal(
        same_input_fits["X_caller"],
        same_input_fits["X_caller_before"],
    )


def test_source_faithfulness_comparison_is_pre_threshold(same_input_fits):
    """w_threshold=0.0 is used, so both W matrices contain sub-threshold
    non-zero entries (entries with 0 < |W| < project_threshold=0.3).

    If the comparison were post-threshold, sub-threshold entries would be
    zeroed and this assertion would still trivially hold; but the same
    matrices must also satisfy the 1e-12 tolerance gate above, which
    proves the wrapper preserves the pre-threshold continuous values
    that DAGMA would otherwise zero at its default 0.3 threshold.
    """
    W_direct = same_input_fits["W_direct"]
    W_wrapper = same_input_fits["W_wrapper"]
    off_diag_mask = ~np.eye(_N_VARS, dtype=bool)

    abs_off_direct = np.abs(W_direct[off_diag_mask])
    abs_off_wrapper = np.abs(W_wrapper[off_diag_mask])

    has_sub_threshold_direct = np.any(
        (abs_off_direct > 0.0) & (abs_off_direct < 0.3)
    )
    has_sub_threshold_wrapper = np.any(
        (abs_off_wrapper > 0.0) & (abs_off_wrapper < 0.3)
    )
    assert has_sub_threshold_direct, (
        "Direct DAGMA fit produced no off-diagonal entries with "
        "0 < |W| < 0.3; cannot confirm comparison is pre-threshold."
    )
    assert has_sub_threshold_wrapper, (
        "Wrapper fit produced no off-diagonal entries with "
        "0 < |W| < 0.3; cannot confirm comparison is pre-threshold."
    )


# ---------------------------------------------------------------------------
# Identical input verification at the DAGMA boundary (fast, no real fit)
# ---------------------------------------------------------------------------


def test_direct_and_wrapper_paths_pass_identical_x_to_dagma(monkeypatch):
    """The X passed to DagmaLinear.fit by the wrapper must be value-equal
    to the X that a direct caller would pass (after the same X.copy())."""
    captured: dict[str, np.ndarray] = {}

    class _CapturingFake:
        h_final = 1e-7
        score_final = -1.0

        def __init__(self, loss_type: str = "l2") -> None:
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            captured["X_into_dagma"] = X.copy()
            d = X.shape[1]
            return np.zeros((d, d), dtype=float)

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _CapturingFake,
    )

    X = _make_input_X()
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=_FIT_SEED)
    X_via_wrapper = captured["X_into_dagma"]

    # The direct-call path applies the same defensive X.copy() before
    # passing to DagmaLinear.fit.
    X_via_direct = X.copy()

    np.testing.assert_array_equal(X_via_wrapper, X_via_direct)


# ---------------------------------------------------------------------------
# Centred-input fits: both paths see centred X
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def centred_input_fits():
    """Both paths see X already transformed by CentredOnlyTransform.

    The wrapper's preprocessor is fitted on raw X; the wrapper then
    receives ``pre.transform(X)``. The direct path receives the same
    centred array. This confirms source-faithfulness when the wrapper
    is given model-frame input, which is the production calling pattern.
    """
    X_raw = _make_input_X()
    cfg = DAGMAConfig()

    pre = CentredOnlyTransform().fit(X_raw)
    X_centred = pre.transform(X_raw)

    W_direct = _direct_dagma_fit(X_centred, cfg)

    wrapper = DAGMAWrapper()
    wrapper.fit(X_centred, preprocessor=pre, seed=_FIT_SEED, config=cfg)
    W_wrapper = wrapper.native_edge_continuous()

    return {"W_direct": W_direct, "W_wrapper": W_wrapper}


def test_source_faithfulness_with_centred_only_preprocessor(centred_input_fits):
    """Source-faithfulness holds when the wrapper consumes already-centred
    data via CentredOnlyTransform."""
    np.testing.assert_allclose(
        centred_input_fits["W_wrapper"],
        centred_input_fits["W_direct"],
        atol=_ATOL,
        rtol=_RTOL,
    )


# ---------------------------------------------------------------------------
# Pinned source resolution
# ---------------------------------------------------------------------------


def test_direct_call_dagma_linear_resolves_to_pinned_source():
    """DagmaLinear used by the direct call must come from the inspected
    source clone, not from site-packages."""
    module_file = Path(inspect.getfile(DagmaLinear)).resolve()
    assert module_file.is_relative_to(_dagma_utils._DAGMA_SRC), (
        f"DagmaLinear was loaded from '{module_file}', not from the "
        f"pinned source at '{_dagma_utils._DAGMA_SRC}'."
    )
    assert module_file == DAGMA_SOURCE_PATH
