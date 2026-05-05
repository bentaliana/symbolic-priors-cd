"""Tests for structural graph metrics (SHD).

All tests use small hand-constructed boolean adjacency matrices.
No generated SCMs are used because the metric must be correct independently
of the data generation layer.
"""

import numpy as np
import pytest

from symbolic_priors_cd.metrics import shd


# ---------------------------------------------------------------------------
# Helpers: small hand-crafted adjacency matrices
# ---------------------------------------------------------------------------

def _empty(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def _chain_adj(n: int) -> np.ndarray:
    """Chain 0->1->2->…->(n-1)."""
    A = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        A[i, i + 1] = True
    return A


# ---------------------------------------------------------------------------
# Correctness: identical / empty graphs
# ---------------------------------------------------------------------------


def test_shd_identical_graphs_zero():
    """Identical non-empty graphs must give SHD = 0."""
    A = _chain_adj(4)
    assert shd(A, A) == 0


def test_shd_both_empty_zero():
    A = _empty(5)
    assert shd(A, A) == 0


def test_shd_single_graph_vs_itself_zero():
    A = np.array([[False, True, False],
                  [False, False, True],
                  [False, False, False]])
    assert shd(A, A) == 0


# ---------------------------------------------------------------------------
# Correctness: symmetry
# ---------------------------------------------------------------------------


def test_shd_symmetry():
    """SHD(A, B) must equal SHD(B, A)."""
    A = np.array([[False, True, False],
                  [False, False, True],
                  [False, False, False]])
    B = np.array([[False, False, True],
                  [False, False, True],
                  [False, False, False]])
    assert shd(A, B) == shd(B, A)


# ---------------------------------------------------------------------------
# Correctness: insertions and deletions (no reversals)
# ---------------------------------------------------------------------------


def test_shd_empty_vs_one_edge():
    """Empty predicted vs one-edge true: one missing edge, SHD = 1."""
    predicted = _empty(3)
    true = np.array([[False, True, False],
                     [False, False, False],
                     [False, False, False]])
    assert shd(predicted, true) == 1


def test_shd_empty_vs_k_edges():
    """Empty predicted vs k-edge true: SHD = k (all are insertions)."""
    k = 3
    true = _chain_adj(k + 1)   # k-edge chain on k+1 nodes
    predicted = _empty(k + 1)
    assert shd(predicted, true) == k


def test_shd_one_extra_edge():
    """predicted has one edge true does not: SHD = 1."""
    true = _empty(3)
    predicted = np.array([[False, True, False],
                          [False, False, False],
                          [False, False, False]])
    assert shd(predicted, true) == 1


# ---------------------------------------------------------------------------
# Correctness: reversals with reversal_cost=2 (default)
# ---------------------------------------------------------------------------


def test_shd_single_reversal_default_cost():
    """One reversal with reversal_cost=2 (default) must give SHD = 2.

    true  : 0->1  (edge 0->1 present, edge 1->0 absent)
    predicted: 1->0  (edge 1->0 present, edge 0->1 absent)
    """
    true = np.array([[False, True],
                     [False, False]])
    predicted = np.array([[False, False],
                          [True, False]])
    assert shd(predicted, true) == 2
    assert shd(predicted, true, reversal_cost=2) == 2


def test_shd_two_reversals_default_cost():
    """Two independent reversals: SHD = 4."""
    #  true: 0->1, 2->3
    true = np.array([
        [False, True,  False, False],
        [False, False, False, False],
        [False, False, False, True],
        [False, False, False, False],
    ])
    # predicted: 1->0, 3->2
    predicted = np.array([
        [False, False, False, False],
        [True,  False, False, False],
        [False, False, False, False],
        [False, False, True,  False],
    ])
    assert shd(predicted, true) == 4


# ---------------------------------------------------------------------------
# Correctness: reversals with reversal_cost=1
# ---------------------------------------------------------------------------


def test_shd_single_reversal_cost_one():
    """One reversal with reversal_cost=1 must give SHD = 1."""
    true = np.array([[False, True],
                     [False, False]])
    predicted = np.array([[False, False],
                          [True, False]])
    assert shd(predicted, true, reversal_cost=1) == 1


def test_shd_two_reversals_cost_one():
    """Two reversals with reversal_cost=1: SHD = 2."""
    true = np.array([
        [False, True,  False, False],
        [False, False, False, False],
        [False, False, False, True],
        [False, False, False, False],
    ])
    predicted = np.array([
        [False, False, False, False],
        [True,  False, False, False],
        [False, False, False, False],
        [False, False, True,  False],
    ])
    assert shd(predicted, true, reversal_cost=1) == 2


# ---------------------------------------------------------------------------
# Correctness: mixed reversals and other differences
# ---------------------------------------------------------------------------


def test_shd_reversal_plus_deletion_default_cost():
    """One reversal (cost 2) plus one deletion (cost 1) = SHD 3.

    3-node graph.
    true     : 0->1, 1->2
    predicted: 1->0, (1->2 absent)
    Differences: (0,1)/(1,0) reversal (cost 2) + (1,2) deletion (cost 1).
    """
    true = np.array([[False, True,  False],
                     [False, False, True],
                     [False, False, False]])
    predicted = np.array([[False, False, False],
                          [True,  False, False],
                          [False, False, False]])
    assert shd(predicted, true) == 3


def test_shd_reversal_plus_deletion_cost_one():
    """Same case with reversal_cost=1: SHD = 2."""
    true = np.array([[False, True,  False],
                     [False, False, True],
                     [False, False, False]])
    predicted = np.array([[False, False, False],
                          [True,  False, False],
                          [False, False, False]])
    assert shd(predicted, true, reversal_cost=1) == 2


# ---------------------------------------------------------------------------
# Validation: shape and dtype
# ---------------------------------------------------------------------------


def test_shd_validation_shape_mismatch():
    A = _empty(3)
    B = _empty(4)
    with pytest.raises(ValueError, match="same shape"):
        shd(A, B)


def test_shd_validation_non_square_predicted():
    predicted = np.zeros((3, 4), dtype=bool)
    true = _empty(3)
    with pytest.raises(ValueError, match="square"):
        shd(predicted, true)


def test_shd_validation_non_square_true():
    predicted = _empty(3)
    true = np.zeros((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="square"):
        shd(predicted, true)


def test_shd_validation_non_bool_predicted():
    predicted = np.zeros((3, 3), dtype=np.uint8)
    true = _empty(3)
    with pytest.raises(TypeError, match="bool"):
        shd(predicted, true)


def test_shd_validation_non_bool_true():
    predicted = _empty(3)
    true = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="bool"):
        shd(predicted, true)


def test_shd_validation_self_loop_in_predicted():
    predicted = _empty(3)
    predicted[1, 1] = True
    with pytest.raises(ValueError, match="self-loops"):
        shd(predicted, _empty(3))


def test_shd_validation_self_loop_in_true():
    true = _empty(3)
    true[0, 0] = True
    with pytest.raises(ValueError, match="self-loops"):
        shd(_empty(3), true)


# ---------------------------------------------------------------------------
# Validation: reversal_cost
# ---------------------------------------------------------------------------


def test_shd_validation_reversal_cost_zero():
    with pytest.raises(ValueError, match="positive"):
        shd(_empty(3), _empty(3), reversal_cost=0)


def test_shd_validation_reversal_cost_negative():
    with pytest.raises(ValueError, match="positive"):
        shd(_empty(3), _empty(3), reversal_cost=-1)


def test_shd_validation_reversal_cost_float():
    with pytest.raises(TypeError, match="int"):
        shd(_empty(3), _empty(3), reversal_cost=1.5)  # type: ignore[arg-type]


def test_shd_validation_reversal_cost_bool():
    """True is a bool subclass of int and must be explicitly rejected."""
    with pytest.raises(TypeError, match="bool"):
        shd(_empty(3), _empty(3), reversal_cost=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_shd_returns_plain_int():
    """shd must return a Python int, not np.int64 or similar."""
    result = shd(_chain_adj(3), _empty(3))
    assert type(result) is int
