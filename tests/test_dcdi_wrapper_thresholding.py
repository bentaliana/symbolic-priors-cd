"""Tests for the DCDI thresholding helper and graph/sampler status machinery.

Covers _predict_adjacency_at, classify_graph_status, and infer_sampler_status.
All inputs are constructed directly; no training run is needed.
"""

from __future__ import annotations

import numpy as np
import torch

from symbolic_priors_cd.wrappers.dcdi import (
    _predict_adjacency_at,
    classify_graph_status,
    infer_sampler_status,
)


# ---------------------------------------------------------------------------
# Thresholding helper
# ---------------------------------------------------------------------------


def test_thresholded_adjacency_default_05():
    """Threshold at 0.5 matches a direct >= comparison on the numpy array."""
    w_adj = torch.tensor([
        [0.0, 0.8, 0.3],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ])
    result = _predict_adjacency_at(w_adj, 0.5)
    expected = w_adj.numpy() >= 0.5
    assert result.dtype == bool
    assert np.array_equal(result, expected)


def test_mock_orientation():
    """Known continuous tensor produces the expected boolean adjacency.

    Row-source / column-destination convention: result[i, j] = True means
    edge i -> j. Entry (0,1)=0.9 and (1,2)=0.6 cross the 0.5 threshold;
    entry (0,2)=0.1 does not.
    """
    w_adj = torch.tensor([
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
    ])
    result = _predict_adjacency_at(w_adj, 0.5)
    expected = np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ])
    assert np.array_equal(result, expected)


def test_threshold_monotonicity():
    """Edge count is weakly non-increasing as the threshold increases."""
    # Entry (0,1)=0.45 drops out at 0.5; entry (0,2)=0.55 drops out at 0.6.
    w_adj = torch.tensor([
        [0.0, 0.45, 0.55],
        [0.0, 0.0, 0.65],
        [0.0, 0.0, 0.0],
    ])
    count_040 = int(_predict_adjacency_at(w_adj, 0.4).sum())
    count_050 = int(_predict_adjacency_at(w_adj, 0.5).sum())
    count_060 = int(_predict_adjacency_at(w_adj, 0.6).sum())
    assert count_040 >= count_050 >= count_060


# ---------------------------------------------------------------------------
# Graph status classification
# ---------------------------------------------------------------------------


def _chain_dag() -> np.ndarray:
    """Return a 3x3 chain DAG adjacency: 0->1->2."""
    return np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ])


def test_graph_status_valid_dag():
    """A chain DAG 0->1->2 is classified as valid_dag with reason None."""
    status, reason = classify_graph_status(_chain_dag())
    assert status == "valid_dag"
    assert reason is None


def test_graph_status_cyclic():
    """A directed cycle 0->1->2->0 is classified as cyclic."""
    adj = np.array([
        [False, True, False],
        [False, False, True],
        [True,  False, False],
    ])
    status, reason = classify_graph_status(adj)
    assert status == "cyclic"
    assert reason is not None


def test_graph_status_bidirected():
    """Both 0->1 and 1->0 present yields status bidirected."""
    adj = np.array([
        [False, True,  False],
        [True,  False, False],
        [False, False, False],
    ])
    status, reason = classify_graph_status(adj)
    assert status == "bidirected"
    assert reason is not None


def test_graph_status_self_loop():
    """A graph with a True diagonal entry is classified as self_loop."""
    adj = np.array([
        [True,  False, False],
        [False, False, False],
        [False, False, False],
    ])
    status, reason = classify_graph_status(adj)
    assert status == "self_loop"
    assert reason is not None


def test_invalid_graph_no_silent_repair():
    """classify_graph_status reports broken graphs without modifying the input.

    Checks three cases: cyclic, bidirected, and invalid shape. In each
    case the returned status matches the defect and the input array is
    bitwise unchanged after the call.
    """
    # Cyclic
    cyclic = np.array([
        [False, True, False],
        [False, False, True],
        [True,  False, False],
    ])
    before = cyclic.copy()
    status, _ = classify_graph_status(cyclic)
    assert status == "cyclic"
    assert np.array_equal(cyclic, before)

    # Bidirected
    bidir = np.array([
        [False, True,  False],
        [True,  False, False],
        [False, False, False],
    ])
    before = bidir.copy()
    status, _ = classify_graph_status(bidir)
    assert status == "bidirected"
    assert np.array_equal(bidir, before)

    # Non-square -> invalid_shape (no mutation risk, but still checked)
    non_square = np.zeros((2, 3), dtype=bool)
    status, reason = classify_graph_status(non_square)
    assert status == "invalid_shape"
    assert reason is not None


def test_sampler_status_invalid_graph():
    """All non-valid-dag graph statuses map to unavailable_invalid_graph.

    Also verifies that the reason string names the offending graph status.
    """
    invalid_statuses = ("cyclic", "bidirected", "self_loop", "invalid_shape")
    for gs in invalid_statuses:
        sampler_status, reason = infer_sampler_status(gs)
        assert sampler_status == "unavailable_invalid_graph", (
            f"Expected unavailable_invalid_graph for graph_status={gs!r}, "
            f"got {sampler_status!r}"
        )
        assert reason is not None
        assert gs in reason

    # valid_dag maps to available with no reason
    sampler_status, reason = infer_sampler_status("valid_dag")
    assert sampler_status == "available"
    assert reason is None
