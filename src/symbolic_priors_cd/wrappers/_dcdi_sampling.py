"""DCDI structural-mask context for safe temporary mask enforcement.

Implements the save/mutate/restore pattern that enforces a thresholded
structural mask on a fitted DCDI model during sampling. The context
manager mutates only the live model's adjacency buffer and the
gumbel_adjacency log_alpha parameter, and always restores their
pre-context values on exit, including when an exception is raised
inside the context.

Detached continuous-edge snapshots held by the caller (for example the
tensors stored on TrainingResult) are independent of the live model
state and are therefore unaffected by this context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

from symbolic_priors_cd.wrappers._dcdi_utils import LearnableModel_NonLinGaussANM


@contextmanager
def _structural_mask_context(
    model: LearnableModel_NonLinGaussANM,
    a_thresh: np.ndarray,
) -> Iterator[None]:
    """Temporarily force a structural mask onto a fitted DCDI model.

    On entry, the current ``model.adjacency`` and
    ``model.gumbel_adjacency.log_alpha`` are captured as detached clones.
    The mask ``a_thresh`` is then written into ``model.adjacency`` as a
    0/1 float tensor, and ``log_alpha`` is set to a saturated tensor
    with +100.0 on entries where ``a_thresh`` is True and -100.0 elsewhere.
    These two writes together enforce structural masking through DCDI's
    layer-0 einsum, which multiplies the Gumbel mask M by
    ``model.adjacency`` so that excluded parents have no contribution to
    a target's predicted density parameters.

    On exit, including when an exception is raised inside the with-block,
    the original ``model.adjacency`` and ``log_alpha`` are restored
    bitwise from the detached clones captured on entry. The new mask
    tensor and saturated log_alpha are built on the same device and
    dtype as ``model.gumbel_adjacency.log_alpha`` so the wrapper works
    on non-default placement and precision configurations.

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        A DCDI model whose adjacency buffer and gumbel_adjacency
        log_alpha parameter are mutated in place for the duration of
        the context.
    a_thresh : np.ndarray
        Boolean adjacency mask of shape (num_vars, num_vars) in the
        project's row-source / column-destination convention. Entries
        equal to True denote retained edges; False denotes excluded
        edges.

    Yields
    ------
    None
    """
    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()
    try:
        log_alpha_param = model.gumbel_adjacency.log_alpha
        device = log_alpha_param.device
        dtype = log_alpha_param.dtype
        mask_tensor = torch.as_tensor(a_thresh, dtype=dtype, device=device)
        saturated = mask_tensor * 100.0 + (1.0 - mask_tensor) * -100.0
        with torch.no_grad():
            model.adjacency.copy_(mask_tensor)
            model.gumbel_adjacency.log_alpha.copy_(saturated)
        yield
    finally:
        with torch.no_grad():
            model.adjacency.copy_(saved_adj)
            model.gumbel_adjacency.log_alpha.copy_(saved_log_alpha)
