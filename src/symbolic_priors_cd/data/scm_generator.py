"""SCM data contract, generation, and observational sampling for linear-Gaussian SCMs.

Canonical representation: adjacency matrix (bool), weight matrix (float64),
topological ordering tuple. Not networkx for now, all downstream metrics and model
wrappers operate on matrices directly. A networkx adapter may be added later
as a convenience helper if a particular tool requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from symbolic_priors_cd.data._sampling import _ancestral_sample, _coerce_rng


@dataclass(frozen=True)
class GenerationSpec:
    """Generation-only provenance for a LinearGaussianSCM.

    Contains exactly the parameters used to create the SCM. 
    """

    graph_family: Literal["ER"]
    mechanism_family: Literal["linear_gaussian"]
    n_nodes: int
    expected_edges: int
    edge_probability: float
    weight_magnitude_range: tuple[float, float]
    noise_scale: float
    generation_seed: int


@dataclass(frozen=True)
class LinearGaussianSCM:
    """An immutable linear-Gaussian structural causal model.

    Canonical representation is adjacency + weight matrices plus a topological
    ordering tuple. DAGMA, DCDI, and all evaluator metrics consume matrices
    directly, so networkx is not the canonical form.

    Immutability is enforced in __post_init__: each ndarray is copied into a
    fresh owned buffer and marked read-only, so both
    ``scm.weights[i, j] = x`` (raises ValueError) and
    ``scm.weights = arr`` (raises FrozenInstanceError) are blocked.
    """

    n_nodes: int
    adjacency: np.ndarray   # shape (n, n), bool
    weights: np.ndarray     # shape (n, n), float64
    noise_scale: float
    topological_order: tuple[int, ...]
    spec: GenerationSpec

    def __post_init__(self) -> None:
        adj_bool = np.asarray(self.adjacency, dtype=bool)

        if adj_bool.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"adjacency shape {adj_bool.shape} does not match "
                f"(n_nodes, n_nodes)=({self.n_nodes}, {self.n_nodes})"
            )
        if self.weights.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match "
                f"(n_nodes, n_nodes)=({self.n_nodes}, {self.n_nodes})"
            )
        if np.any(np.diag(adj_bool)):
            raise ValueError("adjacency diagonal must be all False (no self-loops)")
        if np.any(np.diag(self.weights) != 0.0):
            raise ValueError("weights diagonal must be all zero (no self-loops)")
        if sorted(self.topological_order) != list(range(self.n_nodes)):
            raise ValueError(
                f"topological_order must be a permutation of range({self.n_nodes}), "
                f"got {self.topological_order}"
            )
        if np.any(self.weights[~adj_bool] != 0.0):
            raise ValueError("weights must be zero wherever adjacency is False")
        if not (np.isfinite(self.noise_scale) and self.noise_scale > 0):
            raise ValueError(
                f"noise_scale must be a positive finite float, got {self.noise_scale}"
            )

        # Every edge must point forward in the supplied topological order.
        rank = {node: pos for pos, node in enumerate(self.topological_order)}
        edge_rows, edge_cols = np.where(adj_bool)
        for u, v in zip(edge_rows.tolist(), edge_cols.tolist()):
            if rank[u] >= rank[v]:
                raise ValueError(
                    f"topological order violation: edge {u}->{v} points backward "
                    f"(rank[{u}]={rank[u]}, rank[{v}]={rank[v]})"
                )

        # Spec must be coherent with the SCM fields it mirrors.
        if self.spec.n_nodes != self.n_nodes:
            raise ValueError(
                f"spec.n_nodes mismatch: spec has {self.spec.n_nodes}, "
                f"SCM has {self.n_nodes}"
            )
        if self.spec.noise_scale != self.noise_scale:
            raise ValueError(
                f"spec.noise_scale mismatch: spec has {self.spec.noise_scale}, "
                f"SCM has {self.noise_scale}"
            )

        # Copy into owned read-only buffers so frozen-dataclass immutability
        # extends to the array contents, not just the field references.
        adj_copy = adj_bool.copy()
        adj_copy.flags.writeable = False
        w_copy = np.array(self.weights, dtype=np.float64)
        w_copy.flags.writeable = False
        object.__setattr__(self, "adjacency", adj_copy)
        object.__setattr__(self, "weights", w_copy)


def generate_er_dag(
    n_nodes: int,
    expected_edges: int,
    rng: np.random.Generator | int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Generate an Erdos–Renyi random DAG via an acyclic ordered-edge procedure.

    Acyclicity is guaranteed by construction: a uniformly random permutation
    defines a topological order, and each candidate forward edge is included
    independently with probability ``p = expected_edges / C(n, 2)``.
    No rejection sampling is required.

    Parameters
    ----------
    n_nodes : int
        Number of nodes. Must be positive.
    expected_edges : int
        Target expected number of edges. Must be in ``[0, n*(n-1)//2]``.
    rng : np.random.Generator or int
        Random generator or integer seed.

    Returns
    -------
    adjacency : np.ndarray of shape (n_nodes, n_nodes), dtype bool
        Directed adjacency matrix; ``adjacency[i, j] = True`` means edge i→j.
    topological_order : tuple[int, ...]
        Valid topological ordering of all nodes.

    Raises
    ------
    ValueError
        If ``n_nodes <= 0`` or ``expected_edges`` is outside ``[0, n*(n-1)//2]``.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    n_possible = n_nodes * (n_nodes - 1) // 2
    if not (0 <= expected_edges <= n_possible):
        raise ValueError(
            f"expected_edges={expected_edges} is outside [0, {n_possible}] "
            f"for n_nodes={n_nodes}"
        )
    rng = _coerce_rng(rng)
    p = expected_edges / n_possible if n_possible > 0 else 0.0
    perm = rng.permutation(n_nodes)
    topo = tuple(int(x) for x in perm)
    # Upper triangle of permuted-position space: entry (i, j) with i < j
    # represents a potential forward edge perm[i] -> perm[j].
    upper = np.triu(rng.random((n_nodes, n_nodes)) < p, k=1)
    src_pos, dst_pos = np.where(upper)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    if src_pos.size > 0:
        adjacency[perm[src_pos], perm[dst_pos]] = True
    return adjacency, topo


def sample_edge_weights(
    adjacency: np.ndarray,
    rng: np.random.Generator | int,
    magnitude_range: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Assign signed weights to existing edges from a bounded symmetric distribution.

    Each present edge gets sign ~ Uniform{-1, +1} and magnitude ~
    Uniform(low, high), which is equivalent to sampling from
    Uniform(-high, -low) ∪ Uniform(low, high) under the default symmetric
    range.

    Parameters
    ----------
    adjacency : np.ndarray, 2D square, bool-compatible
        Directed adjacency matrix.
    rng : np.random.Generator or int
        Random generator or integer seed.
    magnitude_range : tuple[float, float]
        Absolute-value range ``(low, high)`` with ``0 <= low < high``.

    Returns
    -------
    np.ndarray of shape ``adjacency.shape``, dtype float64
        Weight matrix; zero wherever ``adjacency`` is False.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square or ``magnitude_range`` is invalid.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be a square 2D array, got shape {adjacency.shape}"
        )
    low, high = magnitude_range
    if not (0.0 <= low < high):
        raise ValueError(
            f"magnitude_range must satisfy 0 <= low < high, got ({low}, {high})"
        )
    rng = _coerce_rng(rng)
    n = adjacency.shape[0]
    edge_rows, edge_cols = np.where(adjacency)
    n_edges = edge_rows.size
    magnitudes = rng.uniform(low, high, size=n_edges)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_edges)
    weights = np.zeros((n, n), dtype=np.float64)
    weights[edge_rows, edge_cols] = signs * magnitudes
    return weights


def generate_linear_gaussian_scm(
    n_nodes: int,
    expected_edges: int,
    seed: int,
    noise_scale: float = 1.0,
    weight_magnitude_range: tuple[float, float] = (0.5, 2.0),
) -> LinearGaussianSCM:
    """Generate a fully immutable linear-Gaussian SCM with a provenance record.

    Requires an integer ``seed`` (not a Generator) because the returned SCM
    carries a ``GenerationSpec`` with a non-optional ``generation_seed`` field,
    ensuring every SCM is reproducible from its spec alone without relying on
    opaque Generator state.

    Parameters
    ----------
    n_nodes : int
        Number of variables.
    expected_edges : int
        Target expected number of edges (ER convention: ER2 = 2 * n_nodes).
    seed : int
        Integer seed for reproducible generation.
    noise_scale : float
        Gaussian noise standard deviation, identical across all nodes.
    weight_magnitude_range : tuple[float, float]
        Absolute-value range ``(low, high)`` for edge-weight magnitudes.

    Returns
    -------
    LinearGaussianSCM
        Fully immutable SCM with arrays frozen at construction time.

    Raises
    ------
    TypeError
        If ``seed`` is not an int.
    ValueError
        If ``noise_scale <= 0`` or ``weight_magnitude_range`` is invalid.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not (np.isfinite(noise_scale) and noise_scale > 0):
        raise ValueError(
            f"noise_scale must be a positive finite float, got {noise_scale}"
        )
    low, high = weight_magnitude_range
    if not (0.0 <= low < high):
        raise ValueError(
            f"weight_magnitude_range must satisfy 0 <= low < high, "
            f"got {weight_magnitude_range}"
        )
    rng = np.random.default_rng(seed)
    adjacency, topological_order = generate_er_dag(n_nodes, expected_edges, rng)
    weights = sample_edge_weights(adjacency, rng, magnitude_range=weight_magnitude_range)
    n_possible = n_nodes * (n_nodes - 1) // 2
    edge_probability = expected_edges / n_possible if n_possible > 0 else 0.0
    spec = GenerationSpec(
        graph_family="ER",
        mechanism_family="linear_gaussian",
        n_nodes=n_nodes,
        expected_edges=expected_edges,
        edge_probability=edge_probability,
        weight_magnitude_range=weight_magnitude_range,
        noise_scale=noise_scale,
        generation_seed=seed,
    )
    return LinearGaussianSCM(
        n_nodes=n_nodes,
        adjacency=adjacency,
        weights=weights,
        noise_scale=noise_scale,
        topological_order=topological_order,
        spec=spec,
    )


def sample_observational(
    scm: LinearGaussianSCM,
    n_samples: int,
    rng: np.random.Generator | int,
) -> np.ndarray:
    """Draw i.i.d. observational samples from the SCM via ancestral sampling.

    Parameters
    ----------
    scm : LinearGaussianSCM
        The SCM to sample from.
    n_samples : int
        Number of samples to draw. Must be positive.
    rng : np.random.Generator or int
        Source of randomness. Passing an ``int`` constructs a fresh
        ``np.random.default_rng(rng)`` on each call, so identical integer
        seeds reproduce identical sample matrices. Passing an existing
        ``Generator`` consumes its state in place; subsequent calls advance
        the same stream.

    Returns
    -------
    np.ndarray of shape (n_samples, n_nodes), dtype float64
        Observational data matrix; rows are samples, columns are variables.
    """
    return _ancestral_sample(scm, n_samples, rng, clamp=None)
