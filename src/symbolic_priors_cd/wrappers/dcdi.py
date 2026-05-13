"""DCDI-G wrapper utilities: thresholding helper plus graph and sampler status.

The DCDI thresholding helper (``_predict_adjacency_at``) lives here because
it consumes a torch.Tensor of edge probabilities and applies a direct ``>=``
comparison, which is DCDI-specific. The graph and sampler status helpers
are model-agnostic and live in ``_graph_status``; they are re-exported here
so existing callers that import them from ``wrappers.dcdi`` keep working.
"""

from __future__ import annotations

import numpy as np
import torch

from symbolic_priors_cd.wrappers._graph_status import (
    _is_acyclic_adjacency,
    classify_graph_status,
    infer_sampler_status,
)


def _predict_adjacency_at(
    continuous_w_adj: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Apply a threshold to the continuous edge-probability matrix.

    Parameters
    ----------
    continuous_w_adj : torch.Tensor
        Continuous edge-probability matrix, shape (d, d). Off-diagonal
        entries are in [0, 1]; diagonal is exactly zero.
    threshold : float
        Entries >= threshold map to True; all others map to False.

    Returns
    -------
    np.ndarray
        Boolean adjacency of shape (d, d), dtype bool,
        row-source / column-destination convention.
    """
    p = continuous_w_adj.detach().cpu().numpy()
    return (p >= threshold).astype(bool)


__all__ = [
    "_predict_adjacency_at",
    "_is_acyclic_adjacency",
    "classify_graph_status",
    "infer_sampler_status",
]
