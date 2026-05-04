"""Scientific-invariant tests for scm_generator.

Tests verify correctness of the SCM contract and the sampling semantics,
not just that the code runs. Each test corresponds to a property that must
hold for the evaluator to be trusted downstream.
"""

import dataclasses

import numpy as np
import pytest

from symbolic_priors_cd.data import (
    GenerationSpec,
    LinearGaussianSCM,
    generate_er_dag,
    generate_linear_gaussian_scm,
    sample_edge_weights,
    sample_observational,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_scm(
    n: int = 3,
    weight: float = 1.0,
    noise_scale: float = 1.0,
) -> LinearGaussianSCM:
    """Build an n-node chain X0 -> X1 -> … -> X(n-1) with identical weights."""
    adjacency = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        adjacency[i, i + 1] = True
        weights[i, i + 1] = weight
    n_possible = n * (n - 1) // 2
    spec = GenerationSpec(
        graph_family="ER",
        mechanism_family="linear_gaussian",
        n_nodes=n,
        expected_edges=n - 1,
        edge_probability=(n - 1) / n_possible if n_possible > 0 else 0.0,
        weight_magnitude_range=(0.5, 2.0),
        noise_scale=noise_scale,
        generation_seed=0,
    )
    return LinearGaussianSCM(
        n_nodes=n,
        adjacency=adjacency,
        weights=weights,
        noise_scale=noise_scale,
        topological_order=tuple(range(n)),
        spec=spec,
    )


# ---------------------------------------------------------------------------
# generate_er_dag
# ---------------------------------------------------------------------------


def test_generate_er_dag_topological_order_consistent_with_edges():
    """Every edge must go from a lower-ranked to a higher-ranked node."""
    for seed in range(30):
        adjacency, topo = generate_er_dag(10, 20, seed)
        rank = {node: pos for pos, node in enumerate(topo)}
        rows, cols = np.where(adjacency)
        for u, v in zip(rows.tolist(), cols.tolist()):
            assert rank[u] < rank[v], (
                f"seed={seed}: edge {u}->{v} violates topological order "
                f"(rank[{u}]={rank[u]}, rank[{v}]={rank[v]})"
            )


def test_generate_er_dag_adjacency_upper_triangular_in_topo_order():
    """Reordering rows/cols by topological order must give a strictly upper-triangular matrix."""
    adjacency, topo = generate_er_dag(10, 20, rng=42)
    perm = np.array(topo)
    adj_reordered = adjacency[np.ix_(perm, perm)]
    assert np.all(np.tril(adj_reordered) == 0), (
        "adjacency reordered by topological order is not strictly upper triangular"
    )


def test_generate_er_dag_no_self_loops():
    for seed in range(10):
        adjacency, _ = generate_er_dag(10, 20, rng=seed)
        assert not np.any(np.diag(adjacency)), f"self-loop found at seed={seed}"


def test_generate_er_dag_topological_order_is_permutation():
    adjacency, topo = generate_er_dag(10, 20, rng=0)
    assert sorted(topo) == list(range(10))


def test_generate_er_dag_expected_edge_count():
    """Mean edge count across seeds should be close to expected_edges."""
    counts = [generate_er_dag(10, 20, seed)[0].sum() for seed in range(200)]
    assert abs(np.mean(counts) - 20) < 1.5, (
        f"mean edge count {np.mean(counts):.2f} too far from expected 20"
    )


def test_generate_er_dag_zero_edges():
    adjacency, topo = generate_er_dag(5, 0, rng=0)
    assert adjacency.sum() == 0
    assert sorted(topo) == list(range(5))


def test_generate_er_dag_single_node():
    adjacency, topo = generate_er_dag(1, 0, rng=0)
    assert adjacency.shape == (1, 1)
    assert topo == (0,)


def test_generate_er_dag_validation_negative_nodes():
    with pytest.raises(ValueError, match="n_nodes must be positive"):
        generate_er_dag(0, 0, rng=0)


def test_generate_er_dag_validation_too_many_edges():
    with pytest.raises(ValueError, match="expected_edges"):
        generate_er_dag(5, 100, rng=0)


# ---------------------------------------------------------------------------
# sample_edge_weights
# ---------------------------------------------------------------------------


def test_sample_edge_weights_in_valid_range():
    """All nonzero weights must have absolute value in [0.5, 2.0]."""
    adjacency, _ = generate_er_dag(10, 40, rng=0)
    weights = sample_edge_weights(adjacency, rng=0)
    nonzero = weights[adjacency]
    assert np.all(np.abs(nonzero) >= 0.5), "weight magnitude below lower bound"
    assert np.all(np.abs(nonzero) <= 2.0), "weight magnitude above upper bound"


def test_sample_edge_weights_zero_where_no_edge():
    adjacency, _ = generate_er_dag(10, 20, rng=1)
    weights = sample_edge_weights(adjacency, rng=1)
    assert np.all(weights[~adjacency] == 0.0)


def test_sample_edge_weights_nonzero_on_every_edge():
    adjacency, _ = generate_er_dag(10, 40, rng=2)
    weights = sample_edge_weights(adjacency, rng=2)
    assert np.all(weights[adjacency] != 0.0)


def test_sample_edge_weights_no_self_weights():
    adjacency, _ = generate_er_dag(10, 20, rng=3)
    weights = sample_edge_weights(adjacency, rng=3)
    assert np.all(np.diag(weights) == 0.0)


def test_sample_edge_weights_sign_balance():
    """Sign distribution should be approximately 50/50 over many edges."""
    all_weights: list[float] = []
    for seed in range(100):
        adjacency, _ = generate_er_dag(10, 20, seed)
        weights = sample_edge_weights(adjacency, rng=seed)
        all_weights.extend(weights[adjacency].tolist())
    arr = np.array(all_weights)
    frac_neg = np.mean(arr < 0)
    assert abs(frac_neg - 0.5) < 0.08, (
        f"sign balance {frac_neg:.3f} is far from 0.5"
    )


def test_sample_edge_weights_custom_magnitude_range():
    adjacency, _ = generate_er_dag(10, 40, rng=0)
    weights = sample_edge_weights(adjacency, rng=0, magnitude_range=(1.0, 1.5))
    nonzero = weights[adjacency]
    assert np.all(np.abs(nonzero) >= 1.0)
    assert np.all(np.abs(nonzero) <= 1.5)


def test_sample_edge_weights_validation_non_square():
    with pytest.raises(ValueError, match="square"):
        sample_edge_weights(np.zeros((3, 4), dtype=bool), rng=0)


def test_sample_edge_weights_validation_bad_magnitude_range():
    adjacency = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="magnitude_range"):
        sample_edge_weights(adjacency, rng=0, magnitude_range=(2.0, 0.5))


# ---------------------------------------------------------------------------
# LinearGaussianSCM construction and immutability
# ---------------------------------------------------------------------------


def test_scm_construction_valid():
    scm = generate_linear_gaussian_scm(10, 20, seed=0)
    assert scm.n_nodes == 10
    assert scm.adjacency.shape == (10, 10)
    assert scm.weights.shape == (10, 10)
    assert scm.adjacency.dtype == bool
    assert scm.weights.dtype == np.float64


def test_scm_true_immutability_element():
    """Array element assignment must raise ValueError after construction."""
    scm = generate_linear_gaussian_scm(5, 5, seed=0)
    with pytest.raises(ValueError, match="read-only"):
        scm.weights[0, 1] = 99.0  # type: ignore[index]


def test_scm_true_immutability_adjacency_element():
    scm = generate_linear_gaussian_scm(5, 5, seed=0)
    with pytest.raises(ValueError, match="read-only"):
        scm.adjacency[0, 1] = True  # type: ignore[index]


def test_scm_true_immutability_attribute():
    """Field assignment must raise FrozenInstanceError."""
    scm = generate_linear_gaussian_scm(5, 5, seed=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scm.weights = np.zeros((5, 5))  # type: ignore[misc]


def test_scm_post_init_rejects_self_loop_in_adjacency():
    n = 3
    adjacency = np.eye(n, dtype=bool)  # diagonal = True
    weights = np.zeros((n, n))
    spec = _make_chain_scm(n).spec
    with pytest.raises(ValueError, match="diagonal"):
        LinearGaussianSCM(
            n_nodes=n,
            adjacency=adjacency,
            weights=weights,
            noise_scale=1.0,
            topological_order=(0, 1, 2),
            spec=spec,
        )


def test_scm_post_init_rejects_weight_where_no_edge():
    n = 3
    adjacency = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n))
    weights[0, 1] = 1.0  # weight with no corresponding edge
    spec = _make_chain_scm(n).spec
    with pytest.raises(ValueError, match="zero wherever adjacency is False"):
        LinearGaussianSCM(
            n_nodes=n,
            adjacency=adjacency,
            weights=weights,
            noise_scale=1.0,
            topological_order=(0, 1, 2),
            spec=spec,
        )


def test_scm_post_init_rejects_bad_topological_order():
    scm = _make_chain_scm(3)
    spec = scm.spec
    with pytest.raises(ValueError, match="permutation"):
        LinearGaussianSCM(
            n_nodes=3,
            adjacency=scm.adjacency,
            weights=scm.weights,
            noise_scale=1.0,
            topological_order=(0, 0, 1),  # not a permutation
            spec=spec,
        )


def test_scm_post_init_rejects_non_positive_noise_scale():
    scm = _make_chain_scm(3)
    with pytest.raises(ValueError, match="noise_scale"):
        LinearGaussianSCM(
            n_nodes=3,
            adjacency=scm.adjacency,
            weights=scm.weights,
            noise_scale=0.0,
            topological_order=(0, 1, 2),
            spec=scm.spec,
        )


# ---------------------------------------------------------------------------
# generate_linear_gaussian_scm
# ---------------------------------------------------------------------------


def test_generate_linear_gaussian_scm_spec_fields():
    scm = generate_linear_gaussian_scm(10, 20, seed=7)
    assert scm.spec.graph_family == "ER"
    assert scm.spec.mechanism_family == "linear_gaussian"
    assert scm.spec.n_nodes == 10
    assert scm.spec.expected_edges == 20
    assert scm.spec.generation_seed == 7
    assert scm.spec.noise_scale == 1.0
    expected_p = 20 / (10 * 9 // 2)
    assert abs(scm.spec.edge_probability - expected_p) < 1e-12


def test_generate_linear_gaussian_scm_reproducible():
    scm_a = generate_linear_gaussian_scm(10, 20, seed=42)
    scm_b = generate_linear_gaussian_scm(10, 20, seed=42)
    np.testing.assert_array_equal(scm_a.adjacency, scm_b.adjacency)
    np.testing.assert_array_equal(scm_a.weights, scm_b.weights)
    assert scm_a.topological_order == scm_b.topological_order


def test_generate_linear_gaussian_scm_different_seeds_differ():
    scm_a = generate_linear_gaussian_scm(10, 20, seed=0)
    scm_b = generate_linear_gaussian_scm(10, 20, seed=1)
    # Vanishingly unlikely to be equal across seeds
    assert not np.array_equal(scm_a.adjacency, scm_b.adjacency) or (
        not np.array_equal(scm_a.weights, scm_b.weights)
    )


def test_generate_linear_gaussian_scm_validation_non_int_seed():
    with pytest.raises(TypeError, match="seed must be an int"):
        generate_linear_gaussian_scm(10, 20, seed=0.0)  # type: ignore[arg-type]


def test_generate_linear_gaussian_scm_validation_bad_noise_scale():
    with pytest.raises(ValueError, match="noise_scale"):
        generate_linear_gaussian_scm(10, 20, seed=0, noise_scale=-1.0)


# ---------------------------------------------------------------------------
# sample_observational
# ---------------------------------------------------------------------------


def test_sample_observational_shape():
    scm = generate_linear_gaussian_scm(10, 20, seed=0)
    X = sample_observational(scm, n_samples=500, rng=1)
    assert X.shape == (500, 10)
    assert X.dtype == np.float64


def test_sample_observational_reproducible_with_seed():
    scm = generate_linear_gaussian_scm(10, 20, seed=0)
    X1 = sample_observational(scm, 100, rng=42)
    X2 = sample_observational(scm, 100, rng=42)
    np.testing.assert_array_equal(X1, X2)


def test_sample_observational_chain_moments():
    """3-node chain X0→X1→X2 with unit weights and noise has known moments."""
    scm = _make_chain_scm(n=3, weight=1.0, noise_scale=1.0)
    X = sample_observational(scm, n_samples=50_000, rng=0)

    # X0 ~ N(0, 1)
    assert abs(X[:, 0].mean()) < 0.05
    assert abs(X[:, 0].std() - 1.0) < 0.05

    # X1 = X0 + noise  => X1 ~ N(0, sqrt(2))
    assert abs(X[:, 1].mean()) < 0.05
    assert abs(X[:, 1].std() - np.sqrt(2.0)) < 0.05

    # X2 = X1 + noise  => X2 ~ N(0, sqrt(3))
    assert abs(X[:, 2].mean()) < 0.05
    assert abs(X[:, 2].std() - np.sqrt(3.0)) < 0.05


def test_sample_observational_root_node_is_gaussian():
    """In any graph, a node with no parents should be standard Gaussian."""
    scm = generate_linear_gaussian_scm(10, 20, seed=5)
    X = sample_observational(scm, n_samples=50_000, rng=0)
    for node in range(scm.n_nodes):
        if not np.any(scm.adjacency[:, node]):  # node has no parents
            assert abs(X[:, node].mean()) < 0.05
            assert abs(X[:, node].std() - scm.noise_scale) < 0.05
            break  # one root is enough for a sanity check


def test_sample_observational_rejects_non_positive_n_samples():
    scm = generate_linear_gaussian_scm(5, 4, seed=0)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        sample_observational(scm, n_samples=0, rng=0)


# ---------------------------------------------------------------------------
# LinearGaussianSCM — new validations
# ---------------------------------------------------------------------------


def test_scm_post_init_rejects_topological_order_violating_edge():
    """An edge that points backward in the supplied topological_order must raise.

    Build adjacency with edge 1->0, then declare topological_order=(0,1,2),
    which assigns rank 0 to node 0 and rank 1 to node 1. The edge 1->0 is
    a backward edge and should be caught.
    """
    n = 3
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[1, 0] = True          # backward edge: rank[1]=1, rank[0]=0
    weights = np.zeros((n, n))
    weights[1, 0] = 1.0
    spec = _make_chain_scm(n).spec
    with pytest.raises(ValueError, match="topological order violation"):
        LinearGaussianSCM(
            n_nodes=n,
            adjacency=adjacency,
            weights=weights,
            noise_scale=1.0,
            topological_order=(0, 1, 2),
            spec=spec,
        )


def test_scm_post_init_rejects_inconsistent_spec_n_nodes():
    """spec.n_nodes that disagrees with the SCM's n_nodes must raise."""
    scm = _make_chain_scm(3)
    bad_spec = GenerationSpec(
        graph_family=scm.spec.graph_family,
        mechanism_family=scm.spec.mechanism_family,
        n_nodes=4,                          # mismatch: SCM has 3 nodes
        expected_edges=scm.spec.expected_edges,
        edge_probability=scm.spec.edge_probability,
        weight_magnitude_range=scm.spec.weight_magnitude_range,
        noise_scale=scm.spec.noise_scale,
        generation_seed=scm.spec.generation_seed,
    )
    with pytest.raises(ValueError, match="spec.n_nodes mismatch"):
        LinearGaussianSCM(
            n_nodes=3,
            adjacency=scm.adjacency,
            weights=scm.weights,
            noise_scale=scm.noise_scale,
            topological_order=scm.topological_order,
            spec=bad_spec,
        )


def test_scm_post_init_rejects_inconsistent_spec_noise_scale():
    """spec.noise_scale that disagrees with the SCM's noise_scale must raise."""
    scm = _make_chain_scm(3, noise_scale=1.0)
    bad_spec = GenerationSpec(
        graph_family=scm.spec.graph_family,
        mechanism_family=scm.spec.mechanism_family,
        n_nodes=scm.spec.n_nodes,
        expected_edges=scm.spec.expected_edges,
        edge_probability=scm.spec.edge_probability,
        weight_magnitude_range=scm.spec.weight_magnitude_range,
        noise_scale=0.5,                    # mismatch: SCM has 1.0
        generation_seed=scm.spec.generation_seed,
    )
    with pytest.raises(ValueError, match="spec.noise_scale mismatch"):
        LinearGaussianSCM(
            n_nodes=3,
            adjacency=scm.adjacency,
            weights=scm.weights,
            noise_scale=1.0,
            topological_order=scm.topological_order,
            spec=bad_spec,
        )


def test_sample_observational_recovers_non_unit_coefficient():
    """Ancestral sampling must reflect the actual structural coefficient.

    Build a 2-node SCM X0 -> X1 with weight 2.5 and estimate the slope via
    cov(X0, X1) / var(X0). A transposition bug in the weight indexing (e.g.
    weights[j, i] instead of weights[i, j]) would produce slope ≈ 0 because
    weights[1, 0] = 0 in this SCM. Unit-weight chain tests cannot catch this.
    """
    n = 2
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[0, 1] = True
    weights = np.zeros((n, n))
    weights[0, 1] = 2.5
    noise_scale = 1.0
    spec = GenerationSpec(
        graph_family="ER",
        mechanism_family="linear_gaussian",
        n_nodes=n,
        expected_edges=1,
        edge_probability=1.0,
        weight_magnitude_range=(0.5, 2.0),
        noise_scale=noise_scale,
        generation_seed=0,
    )
    scm = LinearGaussianSCM(
        n_nodes=n,
        adjacency=adjacency,
        weights=weights,
        noise_scale=noise_scale,
        topological_order=(0, 1),
        spec=spec,
    )
    X = sample_observational(scm, n_samples=50_000, rng=0)
    # slope = cov(X0, X1) / var(X0); for X1 = 2.5*X0 + noise, this equals 2.5.
    cov_matrix = np.cov(X[:, 0], X[:, 1])
    slope = cov_matrix[0, 1] / cov_matrix[0, 0]
    assert abs(slope - 2.5) < 0.05, (
        f"recovered slope {slope:.4f} is too far from structural coefficient 2.5"
    )
