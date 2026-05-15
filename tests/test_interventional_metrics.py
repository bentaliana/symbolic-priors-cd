"""Tests for interventional distribution metrics (MMD).

All deterministic tests use small fixed sample arrays so results can be
verified by hand or by inline manual computation. Probabilistic tests use
fixed seeds and sufficient samples to keep failure probability negligible.
"""

import numpy as np
import pytest

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
# sid_score: identity, input validation, and cyclic-input rejection
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


@pytest.mark.skip(reason="expected_sid is not yet set; unskip once the expected value is confirmed")
def test_sid_preregistered_hand_computed():
    """SID for predicted=empty 3x3 against true=chain 0->1->2.

    true: 0->1->2 (chain)
    predicted: empty (no edges)

    Set expected_sid to the confirmed integer value before unskipping.
    """
    true = np.array([[False, True,  False],
                     [False, False, True],
                     [False, False, False]])
    predicted = _empty_dag(3)
    expected_sid: int | None = None
    result = sid_score(predicted, true)
    assert result == expected_sid
