"""Tests for the DAGMA model-frame ancestral sampler and topological-order helper.

Covers:
- topological order determinism and stability;
- topological order rejection of invalid graphs;
- sampler output shape, intervention clamping, parent-contribution
  convention, root-node noise, and non-root-node mean computation;
- sampler determinism for fixed seed, seed independence, and no-mutation
  guarantees;
- sampler rejection of invalid inputs.

All tests use small hand-constructed DAGs and operate in model frame
only. No preprocessor transform or inverse-transform is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_priors_cd.wrappers._graph_status import _topological_order
from symbolic_priors_cd.wrappers._dagma_sampling import (
    sample_linear_gaussian_model_frame,
)


# ---------------------------------------------------------------------------
# Small fixed adjacency matrices used across tests
# ---------------------------------------------------------------------------

# 3-node chain 0 -> 1 -> 2.
_A_CHAIN = np.array(
    [[False, True, False],
     [False, False, True],
     [False, False, False]],
    dtype=bool,
)

# 3-node graph: both 0 and 1 point to 2 (two source nodes).
_A_TWO_SOURCES = np.array(
    [[False, False, True],
     [False, False, True],
     [False, False, False]],
    dtype=bool,
)

# 2-node graph with no edges (both nodes are roots).
_A_NO_EDGES = np.array(
    [[False, False],
     [False, False]],
    dtype=bool,
)

# 3-node cycle 0 -> 1 -> 2 -> 0.
_A_CYCLE = np.array(
    [[False, True, False],
     [False, False, True],
     [True, False, False]],
    dtype=bool,
)

# 3-node graph with a self-loop on node 0.
_A_SELF_LOOP = np.array(
    [[True, True, False],
     [False, False, False],
     [False, False, False]],
    dtype=bool,
)

# 3-node graph with a bidirected pair (0 <-> 1).
_A_BIDIRECTED = np.array(
    [[False, True, False],
     [True, False, False],
     [False, False, False]],
    dtype=bool,
)


# ---------------------------------------------------------------------------
# Helper: build a consistent (W_sample, sigma) for a given adjacency
# ---------------------------------------------------------------------------

def _make_w_sigma(A: np.ndarray, weight: float = 1.0, sigma: float = 1.0):
    """Return W_sample and sigma_vector compatible with A."""
    W = A.astype(float) * weight
    s = np.full(A.shape[0], sigma, dtype=float)
    return W, s


# ---------------------------------------------------------------------------
# Tests 1-4: topological-order helper
# ---------------------------------------------------------------------------


def test_topological_order_chain():
    """3-node chain 0->1->2 must produce order [0, 1, 2]."""
    order = _topological_order(_A_CHAIN)
    assert order == [0, 1, 2]


def test_topological_order_stable_ascending_for_tied_sources():
    """When nodes 0 and 1 are both sources, ascending index is preserved."""
    order = _topological_order(_A_TWO_SOURCES)
    # 0 and 1 are both in-degree-0; ascending sort puts 0 before 1.
    assert order == [0, 1, 2]


def test_topological_order_rejects_cycle():
    """A cyclic adjacency must raise ValueError."""
    with pytest.raises(ValueError, match="valid DAG"):
        _topological_order(_A_CYCLE)


def test_topological_order_rejects_self_loop_and_bidirected():
    """Self-loop and bidirected adjacencies must raise ValueError."""
    with pytest.raises(ValueError, match="valid DAG"):
        _topological_order(_A_SELF_LOOP)

    with pytest.raises(ValueError, match="valid DAG"):
        _topological_order(_A_BIDIRECTED)


# ---------------------------------------------------------------------------
# Tests 5-9: sampler correctness
# ---------------------------------------------------------------------------


def test_sampler_output_shape():
    """Output shape must be (n_samples, n_vars)."""
    W, s = _make_w_sigma(_A_CHAIN)
    out = sample_linear_gaussian_model_frame(
        _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=7, sample_seed=0
    )
    assert out.shape == (7, 3)
    assert out.dtype == np.float64


def test_sampler_intervention_target_clamped_exactly():
    """The target column must equal value_model for every sample."""
    W, s = _make_w_sigma(_A_CHAIN)
    out = sample_linear_gaussian_model_frame(
        _A_CHAIN, W, s, target=1, value_model=-3.7, n_samples=20, sample_seed=0
    )
    np.testing.assert_array_equal(out[:, 1], -3.7)


def test_sampler_parent_contribution_row_source_col_dest():
    """Parent mean uses W_sample[parent_idx, j], not W_sample[j, parent_idx].

    Chain 0->1 with W[0,1]=2.0 and target=0 clamped to 3.0.
    Node 1 mean = X[:,0] @ W[[0],1] = 3.0 * 2.0 = 6.0.
    W[1,0] = 5.0 is a decoy: the wrong convention would produce 15.0.

    The full column is verified exactly by reproducing the RNG noise.
    Topological order is [0, 1]; node 0 is clamped (no RNG call);
    node 1 is the first (and only) RNG call.
    """
    A = np.array([[False, True], [False, False]], dtype=bool)
    W = np.array([[0.0, 2.0], [5.0, 0.0]], dtype=float)  # W[1,0]=5.0 decoy
    s = np.array([1.0, 1.0], dtype=float)
    n = 50
    seed = 7

    out = sample_linear_gaussian_model_frame(
        A, W, s, target=0, value_model=3.0, n_samples=n, sample_seed=seed
    )
    # Node 0: clamped to 3.0 for every sample.
    np.testing.assert_array_equal(out[:, 0], 3.0)
    # Node 1: mean = 3.0 * 2.0 = 6.0 (row-source/col-dest) + first RNG draw.
    # Reproduce exactly: node 0 is clamped so the first normal() call is for node 1.
    expected_noise = np.random.default_rng(seed).normal(0.0, s[1], n)
    np.testing.assert_allclose(out[:, 1], 6.0 + expected_noise, atol=1e-12)


def test_sampler_root_node_draws_from_normal_zero_mean():
    """A root node (no parents, not the target) draws from N(0, sigma_j).

    2-node no-edge graph, target=0 clamped; node 1 is a root with sigma=2.0.
    The noise vector equals rng.normal(0, 2.0, n) for the same seed.
    """
    W, _ = _make_w_sigma(_A_NO_EDGES)
    sigma = np.array([1.0, 2.0], dtype=float)
    n = 50
    seed = 13

    out = sample_linear_gaussian_model_frame(
        _A_NO_EDGES, W, sigma, target=0, value_model=0.0, n_samples=n, sample_seed=seed
    )
    # Topological order is [0, 1]. Node 0 is clamped (no RNG call).
    # Node 1 is the first (and only) RNG call.
    rng = np.random.default_rng(seed)
    expected = rng.normal(0.0, 2.0, n)
    np.testing.assert_allclose(out[:, 1], expected, atol=1e-12)


def test_sampler_non_root_uses_parent_mean_plus_noise():
    """Non-root nodes use X[:,parents] @ W_sample[parents,j] + noise.

    Chain 0->1->2, target=0 clamped to 2.0, W[0,1]=0.5, W[1,2]=0.8.
    Exact noise vectors are predicted from the same default_rng seed.
    """
    A = _A_CHAIN.copy()
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.5
    W[1, 2] = 0.8
    sigma = np.array([1.0, 0.1, 0.2], dtype=float)
    n = 5
    seed = 42

    out = sample_linear_gaussian_model_frame(
        A, W, sigma, target=0, value_model=2.0, n_samples=n, sample_seed=seed
    )

    # Replicate RNG calls: node 0 is clamped (no call), node 1 is first,
    # node 2 uses X[:,1] (already drawn).
    rng = np.random.default_rng(seed)
    noise_1 = rng.normal(0.0, sigma[1], n)  # first RNG call
    noise_2 = rng.normal(0.0, sigma[2], n)  # second RNG call

    expected_1 = 2.0 * W[0, 1] + noise_1   # 2.0 * 0.5 = 1.0 + noise
    expected_2 = expected_1 * W[1, 2] + noise_2

    np.testing.assert_allclose(out[:, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(out[:, 1], expected_1, atol=1e-12)
    np.testing.assert_allclose(out[:, 2], expected_2, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests 10-12: determinism, seed independence, no mutation
# ---------------------------------------------------------------------------


def test_sampler_deterministic_for_same_seed():
    """Two calls with the same seed must produce identical output."""
    W, s = _make_w_sigma(_A_CHAIN, weight=0.7)
    kwargs = dict(target=0, value_model=1.0, n_samples=30, sample_seed=99)
    out1 = sample_linear_gaussian_model_frame(_A_CHAIN, W, s, **kwargs)
    out2 = sample_linear_gaussian_model_frame(_A_CHAIN, W, s, **kwargs)
    np.testing.assert_array_equal(out1, out2)


def test_sampler_different_seeds_give_different_output():
    """Two calls with different seeds must produce different stochastic columns."""
    W, s = _make_w_sigma(_A_CHAIN, weight=0.5)
    out1 = sample_linear_gaussian_model_frame(
        _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=20, sample_seed=1
    )
    out2 = sample_linear_gaussian_model_frame(
        _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=20, sample_seed=2
    )
    # Node 1 is stochastic; outputs must differ.
    assert not np.allclose(out1[:, 1], out2[:, 1])


def test_sampler_does_not_mutate_inputs():
    """A_thresh, W_sample, and sigma_vector must be unchanged after sampling."""
    W, s = _make_w_sigma(_A_CHAIN, weight=0.9)
    A_copy = _A_CHAIN.copy()
    W_copy = W.copy()
    s_copy = s.copy()

    sample_linear_gaussian_model_frame(
        _A_CHAIN, W, s, target=0, value_model=1.5, n_samples=10, sample_seed=0
    )

    np.testing.assert_array_equal(_A_CHAIN, A_copy)
    np.testing.assert_array_equal(W, W_copy)
    np.testing.assert_array_equal(s, s_copy)


# ---------------------------------------------------------------------------
# Tests 13-17: sampler input validation
# ---------------------------------------------------------------------------


def test_sampler_rejects_invalid_sigma():
    """sigma_vector with zero, negative, NaN, or inf entries must raise ValueError."""
    W, _ = _make_w_sigma(_A_CHAIN)
    n_vars = _A_CHAIN.shape[0]

    for bad_sigma in [
        np.array([1.0, 0.0, 1.0]),   # zero
        np.array([1.0, -0.5, 1.0]),  # negative
        np.array([1.0, np.nan, 1.0]),  # NaN
        np.array([1.0, np.inf, 1.0]),  # inf
    ]:
        with pytest.raises(ValueError, match="sigma_vector"):
            sample_linear_gaussian_model_frame(
                _A_CHAIN, W, bad_sigma,
                target=0, value_model=0.0, n_samples=5, sample_seed=0,
            )


def test_sampler_rejects_w_sample_with_nan():
    """W_sample containing NaN must raise ValueError."""
    W, s = _make_w_sigma(_A_CHAIN)
    W_nan = W.copy()
    W_nan[0, 1] = np.nan
    with pytest.raises(ValueError, match="W_sample"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W_nan, s, target=0, value_model=0.0, n_samples=5, sample_seed=0
        )


def test_sampler_rejects_w_sample_with_inf():
    """W_sample containing inf must raise ValueError."""
    W, s = _make_w_sigma(_A_CHAIN)
    W_inf = W.copy()
    W_inf[1, 2] = np.inf
    with pytest.raises(ValueError, match="W_sample"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W_inf, s, target=0, value_model=0.0, n_samples=5, sample_seed=0
        )


def test_sampler_rejects_invalid_target():
    """target outside [0, n_vars) must raise ValueError."""
    W, s = _make_w_sigma(_A_CHAIN)
    with pytest.raises(ValueError, match="target"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W, s, target=-1, value_model=0.0, n_samples=5, sample_seed=0
        )
    with pytest.raises(ValueError, match="target"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W, s, target=3, value_model=0.0, n_samples=5, sample_seed=0
        )


def test_sampler_rejects_invalid_n_samples():
    """n_samples < 1 must raise ValueError."""
    W, s = _make_w_sigma(_A_CHAIN)
    with pytest.raises(ValueError, match="n_samples"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=0, sample_seed=0
        )
    with pytest.raises(ValueError, match="n_samples"):
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=-1, sample_seed=0
        )


def test_sampler_rejects_invalid_graph():
    """Non-DAG adjacency must raise ValueError."""
    W_cyc = _A_CYCLE.astype(float)
    s = np.ones(3)
    with pytest.raises(ValueError, match="valid DAG"):
        sample_linear_gaussian_model_frame(
            _A_CYCLE, W_cyc, s, target=0, value_model=0.0, n_samples=5, sample_seed=0
        )


def test_sampler_does_not_call_np_random_seed():
    """The sampler must not call np.random.seed."""
    W, s = _make_w_sigma(_A_CHAIN)
    calls: list = []
    original = np.random.seed

    np.random.seed = lambda x=None: calls.append(x)  # type: ignore[assignment]
    try:
        sample_linear_gaussian_model_frame(
            _A_CHAIN, W, s, target=0, value_model=0.0, n_samples=10, sample_seed=5
        )
    finally:
        np.random.seed = original  # type: ignore[assignment]

    assert calls == [], f"np.random.seed was called with: {calls}"
