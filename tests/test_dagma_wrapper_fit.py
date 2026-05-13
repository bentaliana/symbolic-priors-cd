"""Tests for the DAGMA wrapper fit path.

Covers input validation, the defensive X.copy() guarantee, explicit
hyperparameter forwarding to DagmaLinear, RNG non-mutation, fitted-
state management, h_final / score_final capture, and the alignment of
DAGMAConfig library-default values against the inspected DagmaLinear
signature.

Most tests use a lightweight fake DagmaLinear so they run in
milliseconds and directly inspect the call boundary.
"""

from __future__ import annotations

import inspect
from unittest import mock

import numpy as np
import pytest

from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_X_SMALL = _RNG.standard_normal((20, 3))
_PRE = CentredOnlyTransform().fit(_X_SMALL)


def _make_wrapper_and_pre(n=20, d=3, rng_seed=0):
    """Return a fresh wrapper and fitted CentredOnlyTransform for d-column data."""
    X = np.random.default_rng(rng_seed).standard_normal((n, d))
    pre = CentredOnlyTransform().fit(X)
    return DAGMAWrapper(), pre, X


class _CapturingFake:
    """Fake DagmaLinear that records the fit call and returns zeros.

    Attributes
    ----------
    instances : list
        Module-level list populated whenever __init__ is called.
    """

    instances: list[_CapturingFake] = []

    def __init__(self, loss_type: str = "l2") -> None:
        self.loss_type_used = loss_type
        self.h_final: float = 7.5e-7
        self.score_final: float = -99.0
        self.fit_call_args: dict = {}
        _CapturingFake.instances.append(self)

    def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.fit_call_args = {"X": X, **kwargs}
        d = X.shape[1]
        return np.zeros((d, d), dtype=float)


@pytest.fixture(autouse=False)
def capturing_fake(monkeypatch):
    """Patch DagmaLinear with _CapturingFake; yield (instances list,).

    Clears _CapturingFake.instances before each test.
    """
    _CapturingFake.instances.clear()
    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _CapturingFake,
    )
    return _CapturingFake.instances


# ---------------------------------------------------------------------------
# Defensive copy: caller's array is never mutated
# ---------------------------------------------------------------------------


def test_fit_does_not_mutate_caller_x_train(monkeypatch):
    """The wrapper must pass a copy to DagmaLinear, not the caller's array.

    A mutating fake is used so that any failure to copy would corrupt
    the original array to a sentinel value of 999.0.
    """

    class _MutatingFake:
        h_final = 1e-7
        score_final = -1.0

        def __init__(self, loss_type="l2"):
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            X[:] = 999.0
            return np.zeros((X.shape[1], X.shape[1]))

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _MutatingFake,
    )
    wrapper, pre, X_original = _make_wrapper_and_pre()
    X_before = X_original.copy()
    wrapper.fit(X_original, preprocessor=pre, seed=0)
    np.testing.assert_array_equal(
        X_original, X_before,
        err_msg="DAGMAWrapper.fit mutated the caller's X_train array.",
    )


def test_fit_passes_copy_not_original_to_dagma(capturing_fake):
    """The array passed to DagmaLinear.fit must be a distinct object from
    the caller's X_train."""
    wrapper, pre, X_original = _make_wrapper_and_pre()
    wrapper.fit(X_original, preprocessor=pre, seed=0)
    fake = capturing_fake[0]
    assert fake.fit_call_args["X"] is not X_original, (
        "DagmaLinear received the original X_train array, not a copy."
    )


# ---------------------------------------------------------------------------
# Explicit hyperparameters
# ---------------------------------------------------------------------------


def test_fit_passes_explicit_hyperparameters(capturing_fake):
    """All DAGMAConfig values must be forwarded explicitly to DagmaLinear.fit."""
    wrapper, pre, X = _make_wrapper_and_pre()
    cfg = DAGMAConfig()
    wrapper.fit(X, preprocessor=pre, seed=0, config=cfg)

    kw = capturing_fake[0].fit_call_args
    assert kw["lambda1"] == cfg.lambda1
    assert kw["w_threshold"] == cfg.w_threshold_internal
    assert kw["T"] == cfg.T
    assert kw["mu_init"] == cfg.mu_init
    assert kw["mu_factor"] == cfg.mu_factor
    assert kw["s"] == list(cfg.s)
    assert kw["warm_iter"] == cfg.warm_iter
    assert kw["max_iter"] == cfg.max_iter
    assert kw["lr"] == cfg.lr
    assert kw["beta_1"] == cfg.beta_1
    assert kw["beta_2"] == cfg.beta_2


def test_fit_passes_exclude_edges_none(capturing_fake):
    """exclude_edges must be passed as explicit None, not left to the default."""
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert capturing_fake[0].fit_call_args["exclude_edges"] is None


def test_fit_passes_include_edges_none(capturing_fake):
    """include_edges must be passed as explicit None, not left to the default."""
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert capturing_fake[0].fit_call_args["include_edges"] is None


# ---------------------------------------------------------------------------
# Fitted-state management
# ---------------------------------------------------------------------------


def test_fit_sets_fitted_true_on_success(capturing_fake):
    """_fitted becomes True only after a successful fit."""
    wrapper, pre, X = _make_wrapper_and_pre()
    assert not wrapper._fitted
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert wrapper._fitted


def test_fit_leaves_fitted_false_when_dagma_raises(monkeypatch):
    """If DagmaLinear.fit raises, _fitted must remain False."""

    class _RaisingFake:
        h_final = None
        score_final = None

        def __init__(self, loss_type="l2"):
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            raise RuntimeError("simulated DagmaLinear failure")

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _RaisingFake,
    )
    wrapper, pre, X = _make_wrapper_and_pre()
    with pytest.raises(RuntimeError, match="simulated DagmaLinear failure"):
        wrapper.fit(X, preprocessor=pre, seed=0)
    assert not wrapper._fitted, "_fitted must stay False after a failed fit."


def test_fit_exception_propagates_unchanged(monkeypatch):
    """Exceptions from DagmaLinear.fit are not swallowed or wrapped."""

    class _RaisingFake:
        h_final = None
        score_final = None

        def __init__(self, loss_type="l2"):
            pass

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            raise ValueError("bad convergence sentinel")

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _RaisingFake,
    )
    wrapper, pre, X = _make_wrapper_and_pre()
    with pytest.raises(ValueError, match="bad convergence sentinel"):
        wrapper.fit(X, preprocessor=pre, seed=0)


# ---------------------------------------------------------------------------
# h_final and score_final capture
# ---------------------------------------------------------------------------


def test_fit_captures_h_final_as_python_float(capturing_fake):
    """h_final from the fitted DagmaLinear model must be stored as a Python float."""
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert isinstance(wrapper._fit_result.h_final, float)
    assert wrapper._fit_result.h_final == pytest.approx(7.5e-7)


def test_fit_captures_score_final_as_python_float(capturing_fake):
    """score_final from the fitted DagmaLinear model must be stored as a Python float."""
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert isinstance(wrapper._fit_result.score_final, float)
    assert wrapper._fit_result.score_final == pytest.approx(-99.0)


# ---------------------------------------------------------------------------
# config=None resolves to DAGMAConfig defaults
# ---------------------------------------------------------------------------


def test_fit_config_none_resolves_to_default_config(capturing_fake):
    """Passing config=None must resolve to a DAGMAConfig() with default values."""
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=0, config=None)
    kw = capturing_fake[0].fit_call_args
    default = DAGMAConfig()
    assert kw["lambda1"] == default.lambda1
    assert kw["T"] == default.T
    assert kw["s"] == list(default.s)


def test_fit_forwards_custom_config_including_loss_type(capturing_fake):
    """A non-default DAGMAConfig must be forwarded verbatim, including loss_type
    which is passed to the DagmaLinear constructor, not to fit()."""
    custom = DAGMAConfig(
        T=2,
        lambda1=0.1,
        s=(1.0, 0.8),
        mu_init=0.5,
        mu_factor=0.05,
        w_threshold_internal=0.0,
        lr=1e-3,
        warm_iter=5000,
        max_iter=10000,
        beta_1=0.9,
        beta_2=0.99,
        loss_type="logistic",
    )
    wrapper, pre, X = _make_wrapper_and_pre()
    wrapper.fit(X, preprocessor=pre, seed=7, config=custom)

    fake = capturing_fake[0]
    # loss_type is passed to the DagmaLinear constructor, captured on the instance.
    assert fake.loss_type_used == "logistic"
    kw = fake.fit_call_args
    assert kw["lambda1"] == pytest.approx(0.1)
    assert kw["T"] == 2
    assert kw["s"] == [1.0, 0.8]
    assert kw["mu_init"] == pytest.approx(0.5)
    assert kw["mu_factor"] == pytest.approx(0.05)
    assert kw["w_threshold"] == pytest.approx(0.0)
    assert kw["lr"] == pytest.approx(1e-3)
    assert kw["warm_iter"] == 5000
    assert kw["max_iter"] == 10000
    assert kw["beta_1"] == pytest.approx(0.9)
    assert kw["beta_2"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# RNG non-mutation
# ---------------------------------------------------------------------------


def test_fit_does_not_call_np_random_seed(capturing_fake):
    """fit must not call np.random.seed during the fit path."""
    np_seed_calls: list[int] = []
    original = np.random.seed

    import numpy as _np

    _np.random.seed = lambda s: np_seed_calls.append(s)  # type: ignore[assignment]
    try:
        wrapper, pre, X = _make_wrapper_and_pre()
        wrapper.fit(X, preprocessor=pre, seed=42)
    finally:
        _np.random.seed = original  # type: ignore[assignment]

    assert np_seed_calls == [], (
        f"np.random.seed was called with: {np_seed_calls}"
    )


def test_fit_does_not_call_dagma_utils_set_random_seed(capturing_fake):
    """fit must not call dagma.utils.set_random_seed during the fit path."""
    # Import dagma.utils via the pinned path (already on sys.path from
    # _dagma_utils initialisation) and then intercept the function.
    import symbolic_priors_cd.wrappers._dagma_utils  # noqa: F401 (ensures path)
    import dagma.utils as _du

    with mock.patch.object(_du, "set_random_seed") as mock_seed:
        wrapper, pre, X = _make_wrapper_and_pre()
        wrapper.fit(X, preprocessor=pre, seed=42)
        mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_1d_input(capturing_fake):
    """A 1D array must raise ValueError."""
    wrapper, pre, _ = _make_wrapper_and_pre()
    with pytest.raises(ValueError, match="2D"):
        wrapper.fit(np.ones(5), preprocessor=pre, seed=0)


def test_fit_rejects_3d_input(capturing_fake):
    """A 3D array must raise ValueError."""
    wrapper, pre, _ = _make_wrapper_and_pre()
    with pytest.raises(ValueError, match="2D"):
        wrapper.fit(np.ones((2, 3, 4)), preprocessor=pre, seed=0)


def test_fit_rejects_zero_row_input(capturing_fake):
    """An array with zero rows must raise ValueError."""
    wrapper, pre, _ = _make_wrapper_and_pre()
    with pytest.raises(ValueError, match="row"):
        wrapper.fit(np.zeros((0, 3)), preprocessor=pre, seed=0)


def test_fit_rejects_one_column_input(capturing_fake):
    """An array with fewer than two columns must raise ValueError."""
    wrapper, pre, _ = _make_wrapper_and_pre()
    with pytest.raises(ValueError, match="column"):
        wrapper.fit(np.zeros((5, 1)), preprocessor=pre, seed=0)


def test_fit_rejects_bool_input(capturing_fake):
    """A boolean array must raise ValueError."""
    wrapper, pre, _ = _make_wrapper_and_pre()
    X_bool = np.ones((5, 3), dtype=bool)
    with pytest.raises(ValueError, match="bool"):
        wrapper.fit(X_bool, preprocessor=pre, seed=0)


# ---------------------------------------------------------------------------
# DAGMAConfig library defaults match DagmaLinear.fit signature
# ---------------------------------------------------------------------------


def test_dagmaconfig_lr_matches_dagmalinear_default():
    """DAGMAConfig.lr must equal the default in DagmaLinear.fit."""
    from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear as _DL

    default = inspect.signature(_DL.fit).parameters["lr"].default
    assert DAGMAConfig().lr == pytest.approx(default)


def test_dagmaconfig_warm_iter_matches_dagmalinear_default():
    """DAGMAConfig.warm_iter must equal the default in DagmaLinear.fit."""
    from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear as _DL

    default = inspect.signature(_DL.fit).parameters["warm_iter"].default
    assert DAGMAConfig().warm_iter == pytest.approx(default)


def test_dagmaconfig_max_iter_matches_dagmalinear_default():
    """DAGMAConfig.max_iter must equal the default in DagmaLinear.fit."""
    from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear as _DL

    default = inspect.signature(_DL.fit).parameters["max_iter"].default
    assert DAGMAConfig().max_iter == pytest.approx(default)


def test_dagmaconfig_beta_1_matches_dagmalinear_default():
    """DAGMAConfig.beta_1 must equal the default in DagmaLinear.fit."""
    from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear as _DL

    default = inspect.signature(_DL.fit).parameters["beta_1"].default
    assert DAGMAConfig().beta_1 == pytest.approx(default)


def test_dagmaconfig_beta_2_matches_dagmalinear_default():
    """DAGMAConfig.beta_2 must equal the default in DagmaLinear.fit."""
    from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear as _DL

    default = inspect.signature(_DL.fit).parameters["beta_2"].default
    assert DAGMAConfig().beta_2 == pytest.approx(default)
