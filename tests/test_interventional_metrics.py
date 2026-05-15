"""Tests for interventional distribution metrics (MMD).

All deterministic tests use small fixed sample arrays so results can be
verified by hand or by inline manual computation. Probabilistic tests use
fixed seeds and sufficient samples to keep failure probability negligible.
"""

import importlib.metadata

import gadjid
import numpy as np
import pytest

from symbolic_priors_cd.data.scm_generator import generate_linear_gaussian_scm
from symbolic_priors_cd.metrics import mmd_rbf_unbiased, mmd_sensitivity_sweep, sid_score


# ---------------------------------------------------------------------------
# Correctness: same distribution
# ---------------------------------------------------------------------------


def test_mmd_same_large_samples_near_zero():
    """Two large draws from the same distribution must yield |MMD^2| near zero."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 3))
    y = rng.standard_normal((1000, 3))
    result = mmd_rbf_unbiased(x, y)
    assert abs(result) < 0.05


# ---------------------------------------------------------------------------
# Correctness: different distributions
# ---------------------------------------------------------------------------


def test_mmd_different_distributions_positive():
    """Clearly separated distributions must give a large positive MMD^2."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 2))
    y = rng.standard_normal((500, 2)) + 5.0
    result = mmd_rbf_unbiased(x, y)
    assert result > 0.5


# ---------------------------------------------------------------------------
# Correctness: unbiased estimator can return negative values (not clipped)
# ---------------------------------------------------------------------------


def test_mmd_unbiased_can_be_negative():
    """The unbiased estimator must return the raw value and never clip to zero.

    Construction: x=[[0],[2]], y=[[1],[3]], bandwidth=0.5.
    Cross-group samples are close (distance 1); within-group samples are far
    (distance 2). This makes cross-kernel values dominate and drives MMD^2 < 0.

    Manual computation:
      k(x_0, x_1) = k(y_0, y_1) = exp(-4 / 0.5) = exp(-8)
      k(x_0, y_0) = k(x_1, y_0) = k(x_1, y_1) = exp(-1 / 0.5) = exp(-2)
      k(x_0, y_1) = exp(-9 / 0.5) = exp(-18)  [negligible]

      xx_term = 2*exp(-8) / (2*1) = exp(-8)
      yy_term = exp(-8)
      xy_term = 2*(3*exp(-2) + exp(-18)) / (2*2) = (3*exp(-2) + exp(-18)) / 2

      MMD^2 = 2*exp(-8) - (3*exp(-2) + exp(-18)) / 2  [clearly negative]
    """
    x = np.array([[0.0], [2.0]])
    y = np.array([[1.0], [3.0]])
    bw = 0.5

    expected = float(
        2.0 * np.exp(-4.0 / bw)
        - (3.0 * np.exp(-1.0 / bw) + np.exp(-9.0 / bw)) / 2.0
    )
    assert expected < 0, "test setup error: expected value should be negative"

    result = mmd_rbf_unbiased(x, y, bandwidth=bw)
    assert result == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Correctness: median heuristic
# ---------------------------------------------------------------------------


def test_mmd_median_heuristic_small_example():
    """Median heuristic must use pairwise squared distances on concatenated samples.

    x = [[0], [1]], y = [[2], [3]].
    Concatenated: [[0], [1], [2], [3]].
    Upper-triangle squared distances: 1, 4, 9, 1, 4, 1.
    Sorted: [1, 1, 1, 4, 4, 9] -> median of 6 values = (1 + 4) / 2 = 2.5.
    """
    x = np.array([[0.0], [1.0]])
    y = np.array([[2.0], [3.0]])
    result_auto = mmd_rbf_unbiased(x, y)
    result_explicit = mmd_rbf_unbiased(x, y, bandwidth=2.5)
    assert result_auto == pytest.approx(result_explicit, abs=1e-14)


# ---------------------------------------------------------------------------
# Correctness: reproducibility
# ---------------------------------------------------------------------------


def test_mmd_fixed_bandwidth_reproducible():
    """Identical inputs with an explicit bandwidth must return the same value."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 2))
    y = rng.standard_normal((50, 2))
    r1 = mmd_rbf_unbiased(x, y, bandwidth=1.0)
    r2 = mmd_rbf_unbiased(x, y, bandwidth=1.0)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Correctness: sensitivity sweep
# ---------------------------------------------------------------------------


def test_mmd_sensitivity_sweep_keys():
    """Default sweep must return a dict with keys 0.5, 1.0, 2.0."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 2))
    y = rng.standard_normal((50, 2))
    result = mmd_sensitivity_sweep(x, y)
    assert set(result.keys()) == {0.5, 1.0, 2.0}


def test_mmd_sensitivity_sweep_1x_matches_primitive():
    """sweep[1.0] must equal the primitive called with the same median bandwidth.

    The median bandwidth is computed manually so the comparison is not circular.
    """
    x = np.array([[0.0], [1.0]])
    y = np.array([[2.0], [3.0]])

    # Compute median bandwidth manually (same data as median-heuristic test).
    z = np.array([[0.0], [1.0], [2.0], [3.0]])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    sq_dists = [(z[i, 0] - z[j, 0]) ** 2 for i, j in pairs]
    median_bw = float(np.median(sq_dists))   # expected: 2.5

    sweep = mmd_sensitivity_sweep(x, y)
    primitive = mmd_rbf_unbiased(x, y, bandwidth=median_bw)
    assert sweep[1.0] == pytest.approx(primitive, abs=1e-14)


def test_mmd_sensitivity_sweep_all_values_finite():
    """All values in the sweep must be finite floats."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((80, 2))
    y = rng.standard_normal((80, 2)) + 1.0
    result = mmd_sensitivity_sweep(x, y)
    for mult, val in result.items():
        assert np.isfinite(val), f"non-finite MMD at multiplier {mult}"


def test_mmd_sensitivity_sweep_custom_multipliers():
    """A custom multiplier tuple must produce exactly those keys."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal((40, 2))
    result = mmd_sensitivity_sweep(x, y, bandwidth_multipliers=(0.1, 1.0, 10.0))
    assert set(result.keys()) == {0.1, 1.0, 10.0}


# ---------------------------------------------------------------------------
# Validation: mmd_rbf_unbiased inputs
# ---------------------------------------------------------------------------


def test_mmd_validation_x_not_2d():
    with pytest.raises(ValueError, match="2D"):
        mmd_rbf_unbiased(np.ones(10), np.ones((10, 1)))


def test_mmd_validation_y_not_2d():
    with pytest.raises(ValueError, match="2D"):
        mmd_rbf_unbiased(np.ones((10, 1)), np.ones(10))


def test_mmd_validation_feature_dim_mismatch():
    with pytest.raises(ValueError, match="features"):
        mmd_rbf_unbiased(np.ones((5, 3)), np.ones((5, 2)))


def test_mmd_validation_empty_x():
    with pytest.raises(ValueError, match="at least 2 samples"):
        mmd_rbf_unbiased(np.ones((0, 2)), np.ones((5, 2)))


def test_mmd_validation_empty_y():
    with pytest.raises(ValueError, match="at least 2 samples"):
        mmd_rbf_unbiased(np.ones((5, 2)), np.ones((0, 2)))


def test_mmd_validation_single_sample_x():
    """1-sample x raises because m*(m-1) = 0 in the unbiased estimator."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        mmd_rbf_unbiased(np.ones((1, 2)), np.ones((5, 2)))


def test_mmd_validation_single_sample_y():
    with pytest.raises(ValueError, match="at least 2 samples"):
        mmd_rbf_unbiased(np.ones((5, 2)), np.ones((1, 2)))


def test_mmd_validation_bandwidth_zero():
    with pytest.raises(ValueError, match="strictly positive"):
        mmd_rbf_unbiased(np.ones((5, 2)), np.ones((5, 2)), bandwidth=0.0)


def test_mmd_validation_bandwidth_negative():
    with pytest.raises(ValueError, match="strictly positive"):
        mmd_rbf_unbiased(np.ones((5, 2)), np.ones((5, 2)), bandwidth=-1.0)


# ---------------------------------------------------------------------------
# Validation: mmd_sensitivity_sweep inputs
# ---------------------------------------------------------------------------


def test_mmd_sensitivity_sweep_empty_multipliers():
    with pytest.raises(ValueError, match="must not be empty"):
        mmd_sensitivity_sweep(np.ones((5, 2)), np.ones((5, 2)), bandwidth_multipliers=())


def test_mmd_sensitivity_sweep_zero_multiplier():
    with pytest.raises(ValueError, match="strictly positive"):
        mmd_sensitivity_sweep(np.ones((5, 2)), np.ones((5, 2)), bandwidth_multipliers=(0.0,))


def test_mmd_sensitivity_sweep_negative_multiplier():
    with pytest.raises(ValueError, match="strictly positive"):
        mmd_sensitivity_sweep(np.ones((5, 2)), np.ones((5, 2)), bandwidth_multipliers=(-1.0,))


# ---------------------------------------------------------------------------
# Validation: degenerate median heuristic (all samples identical)
# ---------------------------------------------------------------------------


def test_mmd_rbf_unbiased_degenerate_median_raises():
    """When all concatenated samples are identical the median of squared
    pairwise distances is 0, so the median heuristic yields a non-positive
    bandwidth and must raise ValueError."""
    identical = np.ones((5, 2))
    with pytest.raises(ValueError, match="non-positive bandwidth"):
        mmd_rbf_unbiased(identical, identical)


def test_mmd_sensitivity_sweep_degenerate_median_raises():
    """Same degenerate case for mmd_sensitivity_sweep."""
    identical = np.ones((5, 2))
    with pytest.raises(ValueError, match="non-positive bandwidth"):
        mmd_sensitivity_sweep(identical, identical)


# ---------------------------------------------------------------------------
# SID identity on fixed DAGs, input validation, and cyclic-input rejection
# ---------------------------------------------------------------------------


def _empty_dag(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def test_sid_score_identity_simple():
    """sid_score(G, G) must be 0 for any valid DAG."""
    chain = np.array(
        [[False, True, False],
         [False, False, True],
         [False, False, False]]
    )
    assert sid_score(chain, chain) == 0


def test_sid_score_rejects_cyclic_predicted():
    """A cyclic predicted DAG must raise ValueError containing 'cycle'."""
    cycle = np.array(
        [[False, True, False],
         [False, False, True],
         [True, False, False]]
    )
    with pytest.raises(ValueError, match="cycle"):
        sid_score(cycle, _empty_dag(3))


def test_sid_validation_shape_mismatch():
    """Shape mismatch must raise ValueError."""
    A = _empty_dag(3)
    B = _empty_dag(4)
    with pytest.raises(ValueError, match="same shape"):
        sid_score(A, B)


def test_sid_validation_non_square():
    with pytest.raises(ValueError, match="square"):
        sid_score(np.zeros((3, 4), dtype=bool), _empty_dag(3))


def test_sid_validation_non_bool_predicted():
    with pytest.raises(TypeError, match="bool"):
        sid_score(np.zeros((3, 3), dtype=np.uint8), _empty_dag(3))


def test_sid_validation_non_bool_true():
    with pytest.raises(TypeError, match="bool"):
        sid_score(_empty_dag(3), np.zeros((3, 3), dtype=np.uint8))


def test_sid_validation_self_loop_in_predicted():
    A = _empty_dag(3)
    A[1, 1] = True
    with pytest.raises(ValueError, match="self-loops"):
        sid_score(A, _empty_dag(3))


def test_sid_validation_self_loop_in_true():
    B = _empty_dag(3)
    B[0, 0] = True
    with pytest.raises(ValueError, match="self-loops"):
        sid_score(_empty_dag(3), B)


# ---------------------------------------------------------------------------
# SID backend availability
# ---------------------------------------------------------------------------


def test_sid_backend_gadjid_importable():
    """gadjid must be importable and expose a callable sid attribute."""
    assert callable(gadjid.sid)


def test_sid_backend_gadjid_parent_aid_callable():
    """gadjid must expose a callable parent_aid attribute."""
    assert callable(gadjid.parent_aid)


def test_sid_backend_gadjid_pinned_version():
    """The installed gadjid version must match the pinned project dependency."""
    assert importlib.metadata.version("gadjid") == "0.1.0"


# ---------------------------------------------------------------------------
# SID identity on fixed DAGs
# ---------------------------------------------------------------------------


def _chain_dag(n: int) -> np.ndarray:
    """Return the n-node chain DAG 0->1->...->n-1 as a bool adjacency."""
    A = _empty_dag(n)
    for i in range(n - 1):
        A[i, i + 1] = True
    return A


def _fork_dag() -> np.ndarray:
    """Return the 3-node fork DAG 0->{1,2} as a bool adjacency."""
    A = _empty_dag(3)
    A[0, 1] = True
    A[0, 2] = True
    return A


def _collider_dag() -> np.ndarray:
    """Return the 3-node collider DAG {0,1}->2 as a bool adjacency."""
    A = _empty_dag(3)
    A[0, 2] = True
    A[1, 2] = True
    return A


def test_sid_score_identity_fixed_dags():
    """sid_score(G, G) must be 0 for each fixed DAG structure."""
    fixed_dags = [
        ("empty n=3",      _empty_dag(3)),
        ("chain 0->1->2",  _chain_dag(3)),
        ("fork 0->{1,2}",  _fork_dag()),
        ("collider {0,1}->2", _collider_dag()),
        ("chain n=5",      _chain_dag(5)),
    ]
    for name, G in fixed_dags:
        assert sid_score(G, G) == 0, f"identity failed for {name}"


# ---------------------------------------------------------------------------
# SID identity on generated DAGs
# ---------------------------------------------------------------------------

# 20 (n_nodes, expected_edges, seed) cases covering sparse, ER2, and dense
# density regimes across n in {3, 5, 8}.
_RANDOM_DAG_CASES = (
    (3, 2, 0), (3, 2, 1), (3, 3, 2), (3, 3, 3), (3, 1, 4), (3, 2, 5), (3, 3, 6),
    (5, 5, 0), (5, 5, 1), (5, 10, 2), (5, 10, 3), (5, 3, 4), (5, 8, 5), (5, 10, 6),
    (8, 8, 0), (8, 8, 1), (8, 16, 2), (8, 16, 3), (8, 4, 4), (8, 10, 5),
)


def test_sid_score_identity_random_dags():
    """sid_score(G, G) must be 0 for every generated valid DAG."""
    assert len(_RANDOM_DAG_CASES) >= 20, "test setup: need at least 20 cases"
    for n_nodes, expected_edges, seed in _RANDOM_DAG_CASES:
        scm = generate_linear_gaussian_scm(n_nodes, expected_edges, seed=seed)
        G = scm.adjacency
        result = sid_score(G, G)
        assert result == 0, (
            f"identity failed for n={n_nodes} expected_edges={expected_edges} "
            f"seed={seed}: sid_score(G, G) = {result}"
        )


# ---------------------------------------------------------------------------
# SID raw-count extraction
# ---------------------------------------------------------------------------


def test_sid_score_returns_int_mistake_count():
    """sid_score must return the raw integer mistake count, not the normalised float.

    Fixture: predicted=empty 3x3, true=chain 0->1->2.
    The backend returns (normalised_distance, mistake_count). The wrapper must
    return only the int mistake_count and discard the float normalised score.
    """
    predicted = _empty_dag(3)
    true = _chain_dag(3)

    result = sid_score(predicted, true)

    # Must be a Python int.
    assert type(result) is int, f"expected int, got {type(result)}"

    # Must match the backend mistake count directly.
    true_int8 = true.astype(np.int8)
    pred_int8 = predicted.astype(np.int8)
    backend = gadjid.sid(true_int8, pred_int8, edge_direction="from row to column")
    assert result == backend[1], "wrapper must return backend[1] (mistake_count)"

    # Must not be the normalised float.
    assert result != backend[0], (
        "wrapper must not return the normalised distance (backend[0])"
    )


# ---------------------------------------------------------------------------
# SID argument order and asymmetry
# ---------------------------------------------------------------------------


def test_sid_score_argument_order_asymmetric():
    """sid_score(predicted, true) and sid_score(true, predicted) must differ.

    Fixture: predicted=empty 3x3, true=chain 0->1->2. The project-facing
    argument order is (predicted, true); the backend order is (true, predicted).
    The two calls return different counts because SID is asymmetric.
    """
    predicted = _empty_dag(3)
    true = _chain_dag(3)
    assert sid_score(predicted, true) == 3
    assert sid_score(true, predicted) == 0


# ---------------------------------------------------------------------------
# SID backend-call mapping
# ---------------------------------------------------------------------------


def test_sid_score_wrapper_calls_gadjid_with_flipped_args_and_pinned_edge_direction(
    monkeypatch,
):
    """sid_score must flip argument order and pin edge_direction when calling gadjid.sid.

    This test is the primary safeguard against accidental argument-order flips
    or edge_direction regressions. It does not depend on any specific DAG pair.
    """
    import symbolic_priors_cd.metrics.interventional as interventional_module

    predicted = _empty_dag(3)
    true = _chain_dag(3)

    calls = []

    def fake_sid(*args, **kwargs):
        calls.append((args, kwargs))
        return (0.123, 7)

    monkeypatch.setattr(interventional_module.gadjid, "sid", fake_sid)

    result = sid_score(predicted, true)

    assert len(calls) == 1, "gadjid.sid must be called exactly once"
    pos_args, kw_args = calls[0]

    # First positional arg: true cast to int8.
    np.testing.assert_array_equal(pos_args[0], true.astype(np.int8))
    assert pos_args[0].dtype == np.int8, "first backend arg must be int8"

    # Second positional arg: predicted cast to int8.
    np.testing.assert_array_equal(pos_args[1], predicted.astype(np.int8))
    assert pos_args[1].dtype == np.int8, "second backend arg must be int8"

    # edge_direction must be pinned.
    assert kw_args.get("edge_direction") == "from row to column"

    # Return value must be int(fake_return[1]).
    assert result == 7
    assert type(result) is int


# ---------------------------------------------------------------------------
# SID parent_aid agreement
# ---------------------------------------------------------------------------


def test_gadjid_sid_matches_parent_aid_on_fixed_dags():
    """gadjid.sid and gadjid.parent_aid must return identical tuples on DAG inputs.

    This test locks the documented identity parent_aid == sid on DAG inputs.
    A divergence would indicate a gadjid release change that invalidates the
    upstream R-SID transitive cross-validation chain.
    """
    chain = _chain_dag(3).astype(np.int8)
    empty = _empty_dag(3).astype(np.int8)
    fork = _fork_dag().astype(np.int8)
    collider = _collider_dag().astype(np.int8)
    chain5 = _chain_dag(5).astype(np.int8)

    pairs = [
        (chain,    empty,    "chain vs empty"),
        (empty,    chain,    "empty vs chain"),
        (chain,    fork,     "chain vs fork"),
        (fork,     collider, "fork vs collider"),
        (collider, chain5[:3, :3], "collider vs 3-node subchain"),
    ]

    # Use a shared 5-node pair too.
    pairs.append((chain5, np.zeros((5, 5), dtype=np.int8), "chain5 vs empty5"))

    for g_true, g_guess, label in pairs:
        assert g_true.shape == g_guess.shape, f"shape mismatch in pair {label}"
        sid_result = gadjid.sid(
            g_true, g_guess, edge_direction="from row to column"
        )
        aid_result = gadjid.parent_aid(
            g_true, g_guess, edge_direction="from row to column"
        )
        # Assert full tuple equality: mistake counts must be identical,
        # normalised distances must be identical.
        assert sid_result[1] == aid_result[1], (
            f"sid != parent_aid mistake_count for pair {label}: "
            f"sid={sid_result[1]}, parent_aid={aid_result[1]}"
        )
        assert sid_result[0] == pytest.approx(aid_result[0], abs=1e-12), (
            f"sid != parent_aid normalised distance for pair {label}: "
            f"sid={sid_result[0]}, parent_aid={aid_result[0]}"
        )


# ---------------------------------------------------------------------------
# SID edge-direction sensitivity
# ---------------------------------------------------------------------------


def test_sid_backend_edge_direction_sensitivity_witness():
    """gadjid.sid must be sensitive to edge_direction on at least one DAG pair.

    Fixture: true=fork 0->{1,2}, predicted={1->2} (single edge). Under
    'from row to column' the int8 matrices code different DAGs than under
    'from column to row', so the mistake counts must differ.

    This test confirms the two edge_direction values are semantically distinct
    and provides a concrete witness for the project convention choice.
    """
    # true: fork 0 -> {1, 2}
    true_int8 = np.zeros((3, 3), dtype=np.int8)
    true_int8[0, 1] = 1
    true_int8[0, 2] = 1

    # predicted: single edge 1 -> 2
    pred_int8 = np.zeros((3, 3), dtype=np.int8)
    pred_int8[1, 2] = 1

    r2c = gadjid.sid(true_int8, pred_int8, edge_direction="from row to column")
    c2r = gadjid.sid(true_int8, pred_int8, edge_direction="from column to row")

    assert r2c != c2r, (
        "edge_direction sensitivity witness failed: r2c and c2r are equal "
        "on this pair, which means the pair does not distinguish the two "
        "conventions; choose a different witness pair."
    )


# ---------------------------------------------------------------------------
# SID dtype contract
# ---------------------------------------------------------------------------


def test_sid_score_rejects_int8_input_from_caller():
    """sid_score must raise TypeError when the caller passes int8 predicted."""
    with pytest.raises(TypeError, match="bool"):
        sid_score(np.zeros((3, 3), dtype=np.int8), _empty_dag(3))


def test_sid_score_rejects_int64_input():
    """sid_score must raise TypeError when the caller passes int64 predicted."""
    with pytest.raises(TypeError, match="bool"):
        sid_score(np.zeros((3, 3), dtype=np.int64), _empty_dag(3))


def test_sid_score_rejects_uint8_input():
    """sid_score must raise TypeError when the caller passes uint8 predicted."""
    with pytest.raises(TypeError, match="bool"):
        sid_score(np.zeros((3, 3), dtype=np.uint8), _empty_dag(3))


def test_sid_score_rejects_float64_input():
    """sid_score must raise TypeError when the caller passes float64 predicted."""
    with pytest.raises(TypeError, match="bool"):
        sid_score(np.zeros((3, 3), dtype=np.float64), _empty_dag(3))


# ---------------------------------------------------------------------------
# SID invalid-graph rejection
# ---------------------------------------------------------------------------


def test_sid_score_rejects_cyclic_true():
    """A cyclic true DAG must raise ValueError containing 'cycle'."""
    cycle = np.array(
        [[False, True, False],
         [False, False, True],
         [True, False, False]]
    )
    with pytest.raises(ValueError, match="cycle"):
        sid_score(_empty_dag(3), cycle)


def test_sid_score_rejects_bidirected_pair():
    """A bidirected pair in predicted (a directed 2-cycle) must raise ValueError.

    A bidirected pair A[i,j]=True and A[j,i]=True is a directed 2-cycle.
    The acyclicity check catches it and raises ValueError with 'cycle'.
    The rejection message says 'cycle' rather than 'bidirected' because the
    project checks acyclicity, not symmetry, to classify this invalid pattern.
    """
    bidirected = np.array(
        [[False, True, False],
         [True, False, False],
         [False, False, False]]
    )
    with pytest.raises(ValueError, match="cycle"):
        sid_score(bidirected, _empty_dag(3))


# ---------------------------------------------------------------------------
# SID invalid-input no-number behaviour
# ---------------------------------------------------------------------------


def test_sid_score_raises_on_invalid_input_not_a_number():
    """sid_score must raise ValueError on a cyclic predicted graph.

    This test asserts that the function raises an exception rather than
    returning a numeric fallback (such as an SHD value or zero).
    No assertion is made about any numeric return value.
    """
    cycle = np.array(
        [[False, True, False],
         [False, False, True],
         [True, False, False]]
    )
    with pytest.raises(ValueError, match="cycle"):
        sid_score(cycle, _empty_dag(3))


def test_sid_score_empty_predicted_vs_true_chain_returns_backend_reference_count():
    """sid_score(empty 3x3, chain 0->1->2) must return the backend-confirmed count.

    Fixture: predicted = empty 3-node graph, true = chain 0->1->2.
    The raw SID mistake count for this pair is 3, confirmed by the gadjid backend.
    This test pins that count as a regression anchor.
    """
    true = np.array([[False, True,  False],
                     [False, False, True],
                     [False, False, False]])
    predicted = _empty_dag(3)
    assert sid_score(predicted, true) == 3
