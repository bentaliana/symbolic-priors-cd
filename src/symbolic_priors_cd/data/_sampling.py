"""Shared low-level sampling kernel for the data layer.

Both scm_generator (observational path) and interventions (interventional
path) import from here so the two paths cannot diverge silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from symbolic_priors_cd.data.scm_generator import LinearGaussianSCM


def _coerce_rng(rng: np.random.Generator | int) -> np.random.Generator:
    """Return a Generator from either a Generator or an int seed.

    Passing the same int seed twice yields two independent generators that
    start at identical states. Pass an existing Generator when continuity of
    the stochastic sequence across calls matters.
    """
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    if isinstance(rng, np.random.Generator):
        return rng
    raise TypeError(
        f"rng must be np.random.Generator or int, got {type(rng).__name__}"
    )


def _ancestral_sample(
    scm: LinearGaussianSCM,
    n_samples: int,
    rng: np.random.Generator | int,
    clamp: tuple[int, float] | None = None,
) -> np.ndarray:
    """Shared sampling kernel for observational and interventional paths.

    Iterates nodes in topological order. If ``clamp=(j, v)`` is given, node j
    is set to constant v with no noise and no parent contribution, implementing
    do(X_j = v). Descendants receive the clamped value through the structural
    equations as normal.

    This single kernel prevents semantic drift between the two sampling paths —
    any change to ancestral-sampling semantics happens in exactly one place.

    Parameters
    ----------
    scm : LinearGaussianSCM
    n_samples : int
        Must be positive.
    rng : np.random.Generator or int
    clamp : (node_index, value) or None
        If not None, fixes the named node to the given constant.

    Returns
    -------
    np.ndarray of shape (n_samples, n_nodes), dtype float64
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    rng = _coerce_rng(rng)
    X = np.zeros((n_samples, scm.n_nodes), dtype=np.float64)
    for j in scm.topological_order:
        if clamp is not None and clamp[0] == j:
            X[:, j] = clamp[1]
        else:
            parents = np.where(scm.adjacency[:, j])[0]
            parent_contrib = X[:, parents] @ scm.weights[parents, j]
            X[:, j] = parent_contrib + scm.noise_scale * rng.standard_normal(n_samples)
    return X
