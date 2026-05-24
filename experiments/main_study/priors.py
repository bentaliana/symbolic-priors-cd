"""Prior generation and corruption for the main study.

Provides pure, deterministic utilities for generating clean
forbidden-edge priors from a true adjacency and constructing
controlled corrupted-prior variants at fixed corruption fractions.

The scientific core is adjacency-first: every clean prior is
derived only from the true adjacency, never from training data,
learned weighted-adjacency matrices, or model outputs. A
convenience wrapper generates the project ER DAG/SCM from a single
integer seed and then delegates to the pure adjacency-only core.

This module produces immutable spec objects and a confidence mask.
It does not run any causal-discovery model, does not compute any
performance metric, and does not write to disk.

Edge representation follows the project's row-source /
column-destination convention: ``true_adjacency[i, j]`` is True
when there is an edge ``i -> j``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIOR_SEED_BASE: int = 9000
CORRUPTION_SEED_BASE: int = 9100
PRIOR_K: int = 10
CORRUPTION_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)

EDGE_LABEL_TRUE_NEGATIVE_RETAINED: str = "true_negative_retained"
EDGE_LABEL_TRUE_POSITIVE_CORRUPTED_REPLACEMENT: str = (
    "true_positive_corrupted_replacement"
)


# ---------------------------------------------------------------------------
# Frozen specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorSpec:
    """A clean forbidden-edge prior derived from a true adjacency.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the underlying graph.
    scm_seed : int or None
        Integer seed identifying the SCM that produced
        ``true_adjacency``, if known. ``None`` is permitted when the
        clean prior was derived from a hand-built adjacency or another
        graph source that has no scalar seed.
    prior_selection_seed : int
        Seed passed to ``np.random.default_rng`` to draw the clean
        forbidden edges from the true-negative pool.
    forbidden_edges : tuple of (int, int)
        Clean forbidden edges, sorted lexicographically by
        ``(row, col)``. Each pair is a true-negative directed edge
        under the supplied ``true_adjacency``.
    """

    n_nodes: int
    scm_seed: Optional[int]
    prior_selection_seed: int
    forbidden_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CorruptedPriorSpec:
    """A controlled corrupted variant of a clean forbidden-edge prior.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the underlying graph.
    scm_seed : int or None
        Integer seed identifying the source SCM. ``corruption_seed``
        is derived from ``scm_seed`` and ``corruption_index``, so the
        corruption stage requires ``scm_seed`` to be set.
    corruption_fraction : float
        The corruption fraction value from ``CORRUPTION_GRID``.
    corruption_index : int
        Index of ``corruption_fraction`` within ``CORRUPTION_GRID``.
    corruption_seed : int
        Seed passed to ``np.random.default_rng`` to draw the
        removed-clean and added-true-positive edges. Derived from
        ``corruption_seed_base + scm_seed + corruption_index`` and
        populated even at ``corruption_fraction == 0.0``.
    forbidden_edges : tuple of (int, int)
        Final forbidden edges after corruption, sorted
        lexicographically. Length equals ``len(prior.forbidden_edges)``.
    n_correct : int
        Number of clean prior edges retained.
    n_corrupted : int
        Number of clean prior edges replaced by true-positive
        replacements.
    removed_clean_edges : tuple of (int, int)
        Clean prior edges removed from the final forbidden set,
        sorted lexicographically.
    added_true_positive_edges : tuple of (int, int)
        True-positive edges added to the final forbidden set, sorted
        lexicographically.
    edge_labels : dict[str, str]
        Mapping from ``"i,j"`` string keys to provenance labels for
        every edge in ``forbidden_edges``. Retained clean edges carry
        ``"true_negative_retained"``; added true-positive replacements
        carry ``"true_positive_corrupted_replacement"``.
    """

    n_nodes: int
    scm_seed: Optional[int]
    corruption_fraction: float
    corruption_index: int
    corruption_seed: int
    forbidden_edges: tuple[tuple[int, int], ...]
    n_correct: int
    n_corrupted: int
    removed_clean_edges: tuple[tuple[int, int], ...]
    added_true_positive_edges: tuple[tuple[int, int], ...]
    edge_labels: dict[str, str]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_plain_int(value: object) -> bool:
    """True when ``value`` is a non-bool ``int``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_edge_pair(
    edge: object, n_nodes: Optional[int] = None
) -> tuple[int, int]:
    """Validate that ``edge`` is a directed edge tuple ``(i, j)``.

    When ``n_nodes`` is provided, also requires ``0 <= i, j < n_nodes``.
    Booleans are rejected even though ``bool`` is a subclass of ``int``.
    """
    if not (isinstance(edge, tuple) and len(edge) == 2):
        raise ValueError(
            f"edge must be a length-2 tuple; got {edge!r}."
        )
    i, j = edge
    if not (_is_plain_int(i) and _is_plain_int(j)):
        raise ValueError(
            f"edge indices must be non-bool ints; got {edge!r}."
        )
    if i < 0 or j < 0:
        raise ValueError(
            f"edge indices must be non-negative; got {edge!r}."
        )
    if i == j:
        raise ValueError(
            f"edge indices must differ (no self-loops); got {edge!r}."
        )
    if n_nodes is not None and (i >= n_nodes or j >= n_nodes):
        raise ValueError(
            f"edge {edge!r} is out of range for n_nodes={n_nodes}."
        )
    return (int(i), int(j))


def validate_adjacency(true_adjacency: object) -> np.ndarray:
    """Validate ``true_adjacency`` and return a fresh boolean copy.

    Accepts any 2D square array whose entries can be coerced to
    ``bool``. Rejects 1D/3D inputs, non-square shapes, object
    dtypes, non-finite numeric entries, and any non-False diagonal
    entry (which would imply a self-loop).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_nodes, n_nodes)``. Independent
        copy; mutating it does not affect the caller's input.
    """
    arr = np.asarray(true_adjacency)
    if arr.ndim != 2:
        raise ValueError(
            f"true_adjacency must be 2D; got ndim={arr.ndim}."
        )
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"true_adjacency must be square; got shape {arr.shape}."
        )
    if arr.dtype.kind == "O":
        raise ValueError(
            "true_adjacency must not be of object dtype."
        )
    if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
        raise ValueError(
            "true_adjacency must contain only finite values "
            "(no NaN, no infinite entries)."
        )
    try:
        bool_arr = arr.astype(bool, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"true_adjacency could not be coerced to bool: {exc}"
        ) from exc
    if np.any(np.diag(bool_arr)):
        raise ValueError(
            "true_adjacency must have an all-False diagonal "
            "(no self-loops)."
        )
    return bool_arr


# ---------------------------------------------------------------------------
# Edge enumeration
# ---------------------------------------------------------------------------


def true_negative_edges(
    true_adjacency: object,
) -> tuple[tuple[int, int], ...]:
    """Return all off-diagonal absent directed edges, row-major order."""
    arr = validate_adjacency(true_adjacency)
    d = int(arr.shape[0])
    out: list[tuple[int, int]] = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if not bool(arr[i, j]):
                out.append((int(i), int(j)))
    return tuple(out)


def true_positive_edges(
    true_adjacency: object,
) -> tuple[tuple[int, int], ...]:
    """Return all off-diagonal present directed edges, row-major order."""
    arr = validate_adjacency(true_adjacency)
    d = int(arr.shape[0])
    out: list[tuple[int, int]] = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if bool(arr[i, j]):
                out.append((int(i), int(j)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Edge-key helpers
# ---------------------------------------------------------------------------


def edge_tuple_to_key(edge: tuple[int, int]) -> str:
    """Convert ``(i, j)`` to the canonical key string ``"i,j"``.

    Validates that ``edge`` is a length-2 tuple of non-bool ints with
    distinct non-negative indices.
    """
    i, j = _validate_edge_pair(edge)
    return f"{i},{j}"


def edge_key_to_tuple(key: str) -> tuple[int, int]:
    """Inverse of :func:`edge_tuple_to_key`.

    Rejects strings that are not ``"i,j"`` with two non-negative
    distinct integer fields.
    """
    if not isinstance(key, str):
        raise ValueError(
            f"edge key must be a string; got {type(key).__name__}."
        )
    parts = key.split(",")
    if len(parts) != 2:
        raise ValueError(f"malformed edge key {key!r}.")
    a, b = parts[0].strip(), parts[1].strip()
    if a == "" or b == "":
        raise ValueError(f"malformed edge key {key!r}.")
    # Reject leading "+" or whitespace-only contents that int would accept.
    for part in (a, b):
        # int() accepts leading "+" / "-" plus digits; we additionally
        # reject any leading "+" because it is not a canonical form.
        if part.startswith("+"):
            raise ValueError(f"malformed edge key {key!r}.")
    try:
        i = int(a)
        j = int(b)
    except ValueError as exc:
        raise ValueError(f"malformed edge key {key!r}: {exc}") from exc
    if i < 0 or j < 0:
        raise ValueError(
            f"edge indices in key must be non-negative; got {key!r}."
        )
    if i == j:
        raise ValueError(
            f"edge indices in key must differ; got {key!r}."
        )
    return (i, j)


# ---------------------------------------------------------------------------
# Clean prior sampling
# ---------------------------------------------------------------------------


def sample_clean_forbidden_edges(
    true_adjacency: object,
    prior_k: int,
    prior_selection_seed: int,
    scm_seed: Optional[int] = None,
) -> PriorSpec:
    """Sample ``prior_k`` true-negative directed edges without replacement.

    Parameters
    ----------
    true_adjacency : np.ndarray-like
        Boolean-coercible 2D square adjacency. Row-source /
        column-destination.
    prior_k : int
        Number of forbidden edges to sample. Must be a positive int.
    prior_selection_seed : int
        Integer seed for ``np.random.default_rng``.
    scm_seed : int or None
        Optional source-SCM seed recorded on the returned ``PriorSpec``.

    Returns
    -------
    PriorSpec
        Spec with the sampled edges sorted lexicographically.

    Raises
    ------
    ValueError
        If validation fails or there are fewer than ``prior_k``
        true-negative candidate edges.
    """
    bool_arr = validate_adjacency(true_adjacency)
    if not _is_plain_int(prior_k):
        raise ValueError(
            f"prior_k must be a non-bool int; got {prior_k!r}."
        )
    if prior_k <= 0:
        raise ValueError(
            f"prior_k must be positive; got {prior_k}."
        )
    if not _is_plain_int(prior_selection_seed):
        raise ValueError(
            "prior_selection_seed must be a non-bool int; "
            f"got {prior_selection_seed!r}."
        )
    if scm_seed is not None and not _is_plain_int(scm_seed):
        raise ValueError(
            f"scm_seed must be None or a non-bool int; got {scm_seed!r}."
        )

    candidates = true_negative_edges(bool_arr)
    if len(candidates) < prior_k:
        raise ValueError(
            f"Not enough true-negative edges to sample {prior_k}; "
            f"only {len(candidates)} available."
        )

    rng = np.random.default_rng(int(prior_selection_seed))
    indices = rng.choice(len(candidates), size=int(prior_k), replace=False)
    selected = sorted(candidates[int(i)] for i in indices)
    return PriorSpec(
        n_nodes=int(bool_arr.shape[0]),
        scm_seed=None if scm_seed is None else int(scm_seed),
        prior_selection_seed=int(prior_selection_seed),
        forbidden_edges=tuple(selected),
    )


def generate_prior_for_scm_seed(
    scm_seed: int,
    n_nodes: int = 10,
    expected_edges: int = 20,
    prior_k: int = 10,
    prior_seed_base: int = 9000,
) -> PriorSpec:
    """Generate a clean prior from the project ER DAG/SCM for a seed.

    Convenience wrapper around :func:`sample_clean_forbidden_edges`.
    The SCM's true adjacency is extracted from the project's
    deterministic SCM generator and used as the sole input to the
    pure adjacency-only sampler. No training data is sampled here and
    no wrapper/metric code is invoked.

    The prior-selection seed is derived as
    ``prior_selection_seed = prior_seed_base + scm_seed``.
    """
    if not _is_plain_int(scm_seed):
        raise ValueError(
            f"scm_seed must be a non-bool int; got {scm_seed!r}."
        )
    if not _is_plain_int(n_nodes) or n_nodes <= 0:
        raise ValueError(
            f"n_nodes must be a positive int; got {n_nodes!r}."
        )
    if not _is_plain_int(expected_edges) or expected_edges < 0:
        raise ValueError(
            f"expected_edges must be a non-negative int; got {expected_edges!r}."
        )
    if not _is_plain_int(prior_k):
        raise ValueError(
            f"prior_k must be a non-bool int; got {prior_k!r}."
        )
    if not _is_plain_int(prior_seed_base):
        raise ValueError(
            f"prior_seed_base must be a non-bool int; got {prior_seed_base!r}."
        )

    scm = generate_linear_gaussian_scm(
        n_nodes=int(n_nodes),
        expected_edges=int(expected_edges),
        seed=int(scm_seed),
        noise_scale=1.0,
    )
    true_adjacency = np.asarray(scm.adjacency, dtype=bool)
    prior_selection_seed = int(prior_seed_base) + int(scm_seed)
    return sample_clean_forbidden_edges(
        true_adjacency=true_adjacency,
        prior_k=int(prior_k),
        prior_selection_seed=prior_selection_seed,
        scm_seed=int(scm_seed),
    )


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------


def corruption_index_for_fraction(corruption_fraction: float) -> int:
    """Return the index of ``corruption_fraction`` in ``CORRUPTION_GRID``.

    Uses ``math.isclose(..., abs_tol=1e-12, rel_tol=0.0)`` to admit
    numerically equivalent float representations. Off-grid values
    raise ``ValueError``; no bypass is provided.
    """
    try:
        value = float(corruption_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "corruption_fraction must be a real number; "
            f"got {corruption_fraction!r}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            "corruption_fraction must be finite; "
            f"got {corruption_fraction!r}."
        )
    for idx, grid_value in enumerate(CORRUPTION_GRID):
        if math.isclose(
            value, grid_value, abs_tol=1e-12, rel_tol=0.0
        ):
            return int(idx)
    raise ValueError(
        f"corruption_fraction {corruption_fraction!r} is not in "
        f"CORRUPTION_GRID {CORRUPTION_GRID}."
    )


def corrupt_prior(
    prior_spec: PriorSpec,
    true_adjacency: object,
    corruption_fraction: float,
    corruption_seed_base: int = CORRUPTION_SEED_BASE,
) -> CorruptedPriorSpec:
    """Build a :class:`CorruptedPriorSpec` at a fixed corruption fraction.

    The corruption fraction is mapped internally to its
    ``CORRUPTION_GRID`` index via
    :func:`corruption_index_for_fraction`; callers do not supply the
    index. The corruption seed is derived deterministically as
    ``corruption_seed_base + prior_spec.scm_seed + corruption_index``
    and is populated even when ``corruption_fraction == 0.0``.

    At ``corruption_fraction == 0.0`` the final forbidden edge set is
    identical to ``prior_spec.forbidden_edges`` and every edge label
    is ``"true_negative_retained"``.

    Otherwise the routine samples ``round(corruption_fraction *
    prior_k)`` clean prior edges to remove and the same number of
    true-positive edges to add, both without replacement, using the
    derived corruption seed.

    Raises
    ------
    ValueError
        On any validation failure listed in the docstring of each
        validation helper, including: ``prior_spec`` not a
        :class:`PriorSpec`; ``prior_spec.scm_seed`` is ``None``;
        ``true_adjacency`` shape does not match ``prior_spec.n_nodes``;
        any edge in ``prior_spec.forbidden_edges`` is malformed or is
        not a true-negative edge under ``true_adjacency``; fewer
        than ``n_corrupt`` true-positive edges are available for
        replacement.
    """
    if not isinstance(prior_spec, PriorSpec):
        raise ValueError(
            f"prior_spec must be a PriorSpec; "
            f"got {type(prior_spec).__name__}."
        )
    if prior_spec.scm_seed is None:
        raise ValueError(
            "prior_spec.scm_seed must not be None; the corruption "
            "seed cannot be derived without it."
        )
    if not _is_plain_int(corruption_seed_base):
        raise ValueError(
            "corruption_seed_base must be a non-bool int; "
            f"got {corruption_seed_base!r}."
        )
    bool_arr = validate_adjacency(true_adjacency)
    n = int(bool_arr.shape[0])
    if n != int(prior_spec.n_nodes):
        raise ValueError(
            f"true_adjacency shape {bool_arr.shape} does not match "
            f"prior_spec.n_nodes={prior_spec.n_nodes}."
        )

    true_negs = set(true_negative_edges(bool_arr))
    for edge in prior_spec.forbidden_edges:
        valid = _validate_edge_pair(edge, n_nodes=n)
        if valid not in true_negs:
            raise ValueError(
                f"prior_spec.forbidden_edges contains edge {edge!r} "
                "which is not a true-negative edge under the supplied "
                "true_adjacency."
            )

    corruption_index = corruption_index_for_fraction(corruption_fraction)
    corruption_seed = (
        int(corruption_seed_base)
        + int(prior_spec.scm_seed)
        + int(corruption_index)
    )
    prior_k = len(prior_spec.forbidden_edges)
    n_corrupt = int(round(float(corruption_fraction) * prior_k))

    if n_corrupt == 0:
        forbidden = tuple(prior_spec.forbidden_edges)
        edge_labels: dict[str, str] = {
            edge_tuple_to_key(edge): EDGE_LABEL_TRUE_NEGATIVE_RETAINED
            for edge in forbidden
        }
        return CorruptedPriorSpec(
            n_nodes=n,
            scm_seed=int(prior_spec.scm_seed),
            corruption_fraction=float(corruption_fraction),
            corruption_index=int(corruption_index),
            corruption_seed=int(corruption_seed),
            forbidden_edges=forbidden,
            n_correct=int(prior_k),
            n_corrupted=0,
            removed_clean_edges=tuple(),
            added_true_positive_edges=tuple(),
            edge_labels=edge_labels,
        )

    true_positives = true_positive_edges(bool_arr)
    if len(true_positives) < n_corrupt:
        raise ValueError(
            f"Not enough true-positive edges to add {n_corrupt} "
            "corrupted replacements; only "
            f"{len(true_positives)} available."
        )

    rng = np.random.default_rng(int(corruption_seed))
    clean_edges = list(prior_spec.forbidden_edges)
    removed_idx = rng.choice(
        len(clean_edges), size=n_corrupt, replace=False
    )
    removed_set: set[tuple[int, int]] = {
        clean_edges[int(i)] for i in removed_idx
    }
    added_idx = rng.choice(
        len(true_positives), size=n_corrupt, replace=False
    )
    added_set: set[tuple[int, int]] = {
        true_positives[int(i)] for i in added_idx
    }
    retained = [e for e in clean_edges if e not in removed_set]
    final = tuple(sorted(retained + sorted(added_set)))

    # Sanity: no duplicates can appear because added_set is drawn from
    # true positives and retained is a subset of true negatives.
    edge_labels = {}
    for edge in retained:
        edge_labels[edge_tuple_to_key(edge)] = (
            EDGE_LABEL_TRUE_NEGATIVE_RETAINED
        )
    for edge in added_set:
        edge_labels[edge_tuple_to_key(edge)] = (
            EDGE_LABEL_TRUE_POSITIVE_CORRUPTED_REPLACEMENT
        )

    return CorruptedPriorSpec(
        n_nodes=n,
        scm_seed=int(prior_spec.scm_seed),
        corruption_fraction=float(corruption_fraction),
        corruption_index=int(corruption_index),
        corruption_seed=int(corruption_seed),
        forbidden_edges=final,
        n_correct=int(prior_k - n_corrupt),
        n_corrupted=int(n_corrupt),
        removed_clean_edges=tuple(sorted(removed_set)),
        added_true_positive_edges=tuple(sorted(added_set)),
        edge_labels=edge_labels,
    )


# ---------------------------------------------------------------------------
# Confidence mask
# ---------------------------------------------------------------------------


def build_confidence_mask(
    corrupted_prior_spec: CorruptedPriorSpec,
    confidence: float,
) -> np.ndarray:
    """Return an ``n x n`` float mask carrying ``confidence`` at forbidden positions.

    All non-forbidden positions are exactly ``0.0`` and the diagonal
    is exactly zero. ``confidence == 0.0`` therefore produces an
    all-zero mask, which downstream callers can use to gate the
    soft-prior penalty.

    Raises
    ------
    ValueError
        If ``corrupted_prior_spec`` is not a
        :class:`CorruptedPriorSpec`, if ``confidence`` is non-finite
        or outside ``[0.0, 1.0]``, or if any edge in the spec is
        malformed.
    """
    if not isinstance(corrupted_prior_spec, CorruptedPriorSpec):
        raise ValueError(
            "corrupted_prior_spec must be a CorruptedPriorSpec; "
            f"got {type(corrupted_prior_spec).__name__}."
        )
    try:
        c = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"confidence must be a real number; got {confidence!r}: {exc}"
        ) from exc
    if not math.isfinite(c):
        raise ValueError(
            f"confidence must be finite; got {confidence!r}."
        )
    if c < 0.0 or c > 1.0:
        raise ValueError(
            f"confidence must satisfy 0.0 <= confidence <= 1.0; got {c}."
        )
    n = int(corrupted_prior_spec.n_nodes)
    mask = np.zeros((n, n), dtype=float)
    for edge in corrupted_prior_spec.forbidden_edges:
        i, j = _validate_edge_pair(edge, n_nodes=n)
        mask[i, j] = c
    # The diagonal is already zero because _validate_edge_pair
    # rejects i == j and no diagonal entry was written.
    return mask


__all__ = [
    "PRIOR_SEED_BASE",
    "CORRUPTION_SEED_BASE",
    "PRIOR_K",
    "CORRUPTION_GRID",
    "EDGE_LABEL_TRUE_NEGATIVE_RETAINED",
    "EDGE_LABEL_TRUE_POSITIVE_CORRUPTED_REPLACEMENT",
    "PriorSpec",
    "CorruptedPriorSpec",
    "validate_adjacency",
    "true_negative_edges",
    "true_positive_edges",
    "edge_tuple_to_key",
    "edge_key_to_tuple",
    "sample_clean_forbidden_edges",
    "generate_prior_for_scm_seed",
    "corruption_index_for_fraction",
    "corrupt_prior",
    "build_confidence_mask",
]
