"""Tests for the DAGMA hard-exclusion plumbing.

These tests verify that ``cfg.exclude_edges`` flows through the
project's DAGMA fit call site, that wrapper-level validation rejects
every malformed input form, that the prior-free baseline is
recovered when no exclusion is configured, and that
``run_soft_prior_dagma_fit`` ignores the field entirely.

A small deterministic linear-Gaussian chain SCM provides the data;
the baseline fit identifies a true-positive edge with strong
recovered magnitude as the hard-exclusion target.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_priors_cd.wrappers._dagma_fit import (
    run_dagma_fit,
    run_soft_prior_dagma_fit,
)
from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fast-test data and config
# ---------------------------------------------------------------------------

_D = 5
_N = 500
_X_SEED = 12345

_FAST_FIT_KWARGS: dict = dict(
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


def _make_fast_cfg(
    exclude_edges: tuple[tuple[int, int], ...] | None = None,
) -> DAGMAConfig:
    """Build a DAGMAConfig that mirrors ``_FAST_FIT_KWARGS`` exactly."""
    return DAGMAConfig(
        T=_FAST_FIT_KWARGS["T"],
        lambda1=_FAST_FIT_KWARGS["lambda1"],
        s=tuple(_FAST_FIT_KWARGS["s"]),
        mu_init=_FAST_FIT_KWARGS["mu_init"],
        mu_factor=_FAST_FIT_KWARGS["mu_factor"],
        w_threshold_internal=_FAST_FIT_KWARGS["w_threshold"],
        lr=_FAST_FIT_KWARGS["lr"],
        warm_iter=_FAST_FIT_KWARGS["warm_iter"],
        max_iter=_FAST_FIT_KWARGS["max_iter"],
        beta_1=_FAST_FIT_KWARGS["beta_1"],
        beta_2=_FAST_FIT_KWARGS["beta_2"],
        loss_type="l2",
        exclude_edges=exclude_edges,
    )


def _generate_chain_sem_data(d: int, n: int, seed: int) -> np.ndarray:
    """Generate observational data from a hidden linear-Gaussian chain SCM."""
    rng = np.random.default_rng(seed)
    edge_weight = 0.9
    noise_scale = 0.3
    w_internal = np.zeros((d, d), dtype=float)
    for i in range(d - 1):
        w_internal[i, i + 1] = edge_weight
    noise = rng.standard_normal((n, d)) * noise_scale
    return noise @ np.linalg.inv(np.eye(d) - w_internal)


def _true_adjacency_chain(d: int) -> np.ndarray:
    """The boolean adjacency of the chain SCM used by this test module."""
    a = np.zeros((d, d), dtype=bool)
    for i in range(d - 1):
        a[i, i + 1] = True
    return a


def _find_target_true_positive(
    W_baseline: np.ndarray, true_adj: np.ndarray
) -> tuple[int, int] | None:
    """Return the strongest true-positive edge above one of the thresholds."""
    abs_w = np.abs(W_baseline)
    for th in (0.3, 0.2, 0.1):
        best: tuple[int, int, float] | None = None
        d = W_baseline.shape[0]
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if not bool(true_adj[i, j]):
                    continue
                if abs_w[i, j] < th:
                    continue
                if best is None or abs_w[i, j] > best[2]:
                    best = (int(i), int(j), float(abs_w[i, j]))
        if best is not None:
            return (best[0], best[1])
    return None


def _find_non_excluded_strong_edge(
    W_baseline: np.ndarray, exclude: tuple[int, int], threshold: float
) -> tuple[int, int] | None:
    """Find an off-diagonal entry distinct from ``exclude`` with ``abs >= threshold``."""
    abs_w = np.abs(W_baseline)
    d = W_baseline.shape[0]
    best: tuple[int, int, float] | None = None
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if (i, j) == exclude:
                continue
            if abs_w[i, j] < threshold:
                continue
            if best is None or abs_w[i, j] > best[2]:
                best = (int(i), int(j), float(abs_w[i, j]))
    if best is None:
        return None
    return (best[0], best[1])


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def x_data() -> np.ndarray:
    return _generate_chain_sem_data(_D, _N, _X_SEED)


@pytest.fixture(scope="module")
def true_adj() -> np.ndarray:
    return _true_adjacency_chain(_D)


@pytest.fixture(scope="module")
def w_baseline(x_data: np.ndarray) -> np.ndarray:
    """Prior-free DagmaLinear fit on the test data."""
    model = DagmaLinear(loss_type="l2")
    return model.fit(
        X=x_data.copy(),
        exclude_edges=None,
        include_edges=None,
        **_FAST_FIT_KWARGS,
    )


@pytest.fixture(scope="module")
def target_edge(
    w_baseline: np.ndarray, true_adj: np.ndarray
) -> tuple[int, int]:
    """The true-positive edge with the largest recovered baseline magnitude."""
    target = _find_target_true_positive(w_baseline, true_adj)
    if target is None:
        pytest.skip(
            "No true-positive edge with abs(W_baseline) >= 0.1 found; "
            "cannot exercise hard-exclusion tests."
        )
    return target


@pytest.fixture(scope="module")
def w_excluded(
    x_data: np.ndarray, target_edge: tuple[int, int]
) -> np.ndarray:
    """run_dagma_fit result with the target edge hard-excluded."""
    i, j = target_edge
    cfg = _make_fast_cfg(exclude_edges=((i, j),))
    return run_dagma_fit(x_data.copy(), cfg).W


# ---------------------------------------------------------------------------
# T-1: Excluded edge is machine-zero in native continuous W
# ---------------------------------------------------------------------------


def test_excluded_edge_is_machine_zero_in_continuous_w(
    target_edge, w_excluded
):
    i, j = target_edge
    value = float(abs(w_excluded[i, j]))
    assert value < 1e-6, (
        f"|W_excluded[{i},{j}]| = {value:.3e}; expected < 1e-6."
    )


def test_excluded_fit_preserves_non_excluded_strong_edge(
    target_edge, w_baseline, w_excluded
):
    other = _find_non_excluded_strong_edge(
        w_baseline, exclude=target_edge, threshold=0.2
    )
    if other is None:
        pytest.skip(
            "No second off-diagonal baseline edge with abs(W) >= 0.2 "
            "distinct from the excluded edge; cannot verify locality."
        )
    k, l = other
    value = float(abs(w_excluded[k, l]))
    assert value > 0.1, (
        f"|W_excluded[{k},{l}]| = {value:.3e}; "
        "hard exclusion appears to suppress edges globally, not locally."
    )


# ---------------------------------------------------------------------------
# T-2: Excluded edge absent from thresholded adjacency at 0.3
# ---------------------------------------------------------------------------


def test_excluded_edge_falls_below_project_threshold(
    target_edge, w_excluded
):
    i, j = target_edge
    value = float(abs(w_excluded[i, j]))
    assert value < 0.3, (
        f"|W_excluded[{i},{j}]| = {value:.3e}; expected < 0.3."
    )


# ---------------------------------------------------------------------------
# T-3: None passes through unchanged
# ---------------------------------------------------------------------------


def test_none_exclude_edges_passes_through_to_baseline(
    x_data, w_baseline
):
    cfg = _make_fast_cfg(exclude_edges=None)
    result = run_dagma_fit(x_data.copy(), cfg).W
    delta = float(np.max(np.abs(result - w_baseline)))
    assert delta < 1e-10, (
        f"max |W_returned - W_baseline| = {delta:.3e}; expected < 1e-10."
    )


# ---------------------------------------------------------------------------
# T-4: Malformed exclude_edges raises ValueError before calling DAGMA
# ---------------------------------------------------------------------------


def _make_cfg_with_bad_exclude_edges(
    bad_value: object,
) -> DAGMAConfig:
    """Build a DAGMAConfig with an intentionally malformed exclude_edges.

    The DAGMAConfig dataclass does not validate ``exclude_edges`` at
    construction time; validation runs inside ``run_dagma_fit``. To
    exercise that path with malformed values that violate the
    declared type, we bypass ``__init__`` and use
    ``object.__setattr__``.
    """
    cfg = _make_fast_cfg(exclude_edges=None)
    object.__setattr__(cfg, "exclude_edges", bad_value)
    return cfg


def _run_with_bad(x_data: np.ndarray, bad_value: object) -> None:
    cfg = _make_cfg_with_bad_exclude_edges(bad_value)
    run_dagma_fit(x_data.copy(), cfg)


def test_exclude_edges_rejects_list_instead_of_tuple(x_data):
    with pytest.raises(ValueError, match="tuple"):
        _run_with_bad(x_data, [(0, 1)])


def test_exclude_edges_rejects_tuple_of_lists(x_data):
    with pytest.raises(ValueError, match="tuple"):
        _run_with_bad(x_data, ([0, 1],))


def test_exclude_edges_rejects_wrong_length_inner(x_data):
    with pytest.raises(ValueError, match="length"):
        _run_with_bad(x_data, ((0, 1, 2),))


def test_exclude_edges_rejects_out_of_range_index(x_data):
    # n_vars = _D = 5, so 999 is out of range.
    with pytest.raises(ValueError, match="out of range"):
        _run_with_bad(x_data, ((0, 999),))


def test_exclude_edges_rejects_negative_index(x_data):
    with pytest.raises(ValueError, match="non-negative"):
        _run_with_bad(x_data, ((-1, 0),))


def test_exclude_edges_rejects_boolean_true_index(x_data):
    with pytest.raises(ValueError, match="bool"):
        _run_with_bad(x_data, ((True, 1),))


def test_exclude_edges_rejects_boolean_false_index(x_data):
    with pytest.raises(ValueError, match="bool"):
        _run_with_bad(x_data, ((1, False),))


def test_exclude_edges_rejects_float_index_first(x_data):
    with pytest.raises(ValueError, match="plain int"):
        _run_with_bad(x_data, ((0.0, 1),))


def test_exclude_edges_rejects_float_index_second(x_data):
    with pytest.raises(ValueError, match="plain int"):
        _run_with_bad(x_data, ((0, 1.5),))


def test_exclude_edges_rejects_string_index(x_data):
    with pytest.raises(ValueError, match="plain int"):
        _run_with_bad(x_data, (("0", 1),))


def test_exclude_edges_rejects_self_loop(x_data):
    with pytest.raises(ValueError, match="self-loop"):
        _run_with_bad(x_data, ((2, 2),))


def test_exclude_edges_rejects_duplicate_edges(x_data):
    with pytest.raises(ValueError, match="duplicate"):
        _run_with_bad(x_data, ((0, 1), (0, 1)))


# ---------------------------------------------------------------------------
# T-5: Configuration hash changes when exclude_edges changes
# ---------------------------------------------------------------------------


def test_configuration_hash_differs_for_none_vs_excluded():
    cfg_none = _make_fast_cfg(exclude_edges=None)
    cfg_excl = _make_fast_cfg(exclude_edges=((0, 1),))
    assert cfg_none != cfg_excl
    assert hash(cfg_none) != hash(cfg_excl)


def test_configuration_hash_differs_for_different_excluded_edges():
    cfg_a = _make_fast_cfg(exclude_edges=((0, 1),))
    cfg_b = _make_fast_cfg(exclude_edges=((0, 2),))
    assert cfg_a != cfg_b
    assert hash(cfg_a) != hash(cfg_b)


# ---------------------------------------------------------------------------
# T-6: run_soft_prior_dagma_fit ignores exclude_edges in cfg
# ---------------------------------------------------------------------------


def test_soft_prior_path_ignores_exclude_edges_in_config(
    x_data, target_edge, w_baseline
):
    i, j = target_edge
    cfg = _make_fast_cfg(exclude_edges=((i, j),))
    zero_mask = np.zeros((_D, _D), dtype=float)
    soft = run_soft_prior_dagma_fit(
        x_data.copy(),
        cfg,
        lambda_prior=0.0,
        confidence_mask=zero_mask,
    )
    delta_full = float(np.max(np.abs(soft.W - w_baseline)))
    assert delta_full < 1e-10, (
        f"max |W_soft - W_baseline| = {delta_full:.3e}; expected < 1e-10. "
        "Soft-prior path appears to apply hard exclusion."
    )
    delta_target = float(abs(soft.W[i, j] - w_baseline[i, j]))
    assert delta_target < 1e-10, (
        f"|W_soft[{i},{j}] - W_baseline[{i},{j}]| = {delta_target:.3e}; "
        "expected < 1e-10."
    )
