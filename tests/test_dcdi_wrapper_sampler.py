"""Tests for the DCDI structural-mask context manager.

Covers in-context mask enforcement, restoration after normal exit and
after an induced exception, the C-P9 structural-masking invariant
(excluded parents do not influence a target's density parameters while
included parents do), and preservation of detached continuous-edge
snapshots across an in-context exception.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from symbolic_priors_cd.wrappers._dcdi_sampling import _structural_mask_context
from symbolic_priors_cd.wrappers._dcdi_utils import (
    make_dcdi_model,
    snapshot_log_alpha,
    snapshot_w_adj,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _build_model_and_mask():
    """Construct a 3-variable DCDI model and a mask with one excluded edge.

    The chosen mask has edges 0->1 and 0->2, no other edges. For target 2,
    parent 0 is included (mask[0, 2] = True) and parent 1 is excluded
    (mask[1, 2] = False). This matches the structure used by the C-P9 probe.
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()
    mask = np.array([
        [False, True,  True],
        [False, False, False],
        [False, False, False],
    ])
    return model, mask


# ---------------------------------------------------------------------------
# In-context mask + restoration after normal exit
# ---------------------------------------------------------------------------


def test_structural_mask_context_sets_expected_mask_inside_context():
    """Inside the context, adjacency equals the mask and log_alpha is saturated.

    After normal exit, the model's adjacency and log_alpha are restored
    bitwise to their pre-context values.
    """
    model, mask = _build_model_and_mask()

    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()

    with _structural_mask_context(model, mask):
        adj_inside = model.adjacency.detach().clone()
        log_alpha_inside = model.gumbel_adjacency.log_alpha.detach().clone()

    expected_adj = torch.as_tensor(
        mask, dtype=saved_adj.dtype, device=saved_adj.device,
    )
    assert torch.equal(adj_inside, expected_adj), (
        f"Inside context adjacency was {adj_inside.tolist()}, "
        f"expected {expected_adj.tolist()}"
    )

    mask_tensor = torch.as_tensor(
        mask, dtype=log_alpha_inside.dtype, device=log_alpha_inside.device,
    )
    expected_log_alpha = torch.where(
        mask_tensor.bool(),
        torch.full_like(log_alpha_inside, 100.0),
        torch.full_like(log_alpha_inside, -100.0),
    )
    assert torch.equal(log_alpha_inside, expected_log_alpha), (
        "log_alpha inside the context is not saturated to +/-100 on the mask."
    )

    assert torch.equal(model.adjacency, saved_adj), (
        "model.adjacency was not restored after normal exit."
    )
    assert torch.equal(model.gumbel_adjacency.log_alpha, saved_log_alpha), (
        "model.gumbel_adjacency.log_alpha was not restored after normal exit."
    )


# ---------------------------------------------------------------------------
# Restoration after exception
# ---------------------------------------------------------------------------


def test_restoration_after_induced_exception():
    """An exception inside the context still triggers full restoration.

    The finally block of the context manager must run regardless of how
    the with-block exits. After catching the induced exception, the
    model's adjacency and log_alpha must equal their pre-context values
    bitwise.
    """
    model, mask = _build_model_and_mask()

    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()

    class _InducedError(RuntimeError):
        pass

    with pytest.raises(_InducedError):
        with _structural_mask_context(model, mask):
            raise _InducedError("induced for restoration test")

    assert torch.equal(model.adjacency, saved_adj), (
        "model.adjacency was not restored after an in-context exception."
    )
    assert torch.equal(model.gumbel_adjacency.log_alpha, saved_log_alpha), (
        "model.gumbel_adjacency.log_alpha was not restored after an "
        "in-context exception."
    )


# ---------------------------------------------------------------------------
# C-P9 invariant: structural masking through adjacency + log_alpha
# ---------------------------------------------------------------------------


def test_cp9_structural_masking():
    """Reproduce the C-P9 invariant inside the structural-mask context.

    With parent 1 excluded for target 2, Varying column 1 of the input batch by a
    large amount leaves the target's density parameters unchanged within 
    numerical tolerance. Varying column 0 (the included parent for target 2)
    by the same amount produces a visible change in the target's density parameters. 
    The RNG seed is reset before each forward pass so the
    Gumbel draw is identical across paired calls.
    """
    model, mask = _build_model_and_mask()

    bs = 4
    torch.manual_seed(123)
    x_a = torch.randn(bs, 3)

    x_excluded = x_a.clone()
    x_excluded[:, 1] = x_a[:, 1] + 5.0

    x_included = x_a.clone()
    x_included[:, 0] = x_a[:, 0] + 5.0

    with _structural_mask_context(model, mask):
        weights, biases, _extra = model.get_parameters(mode="wbx")

        torch.manual_seed(42)
        with torch.no_grad():
            dp_a = model.forward_given_params(x_a, weights, biases)

        torch.manual_seed(42)
        with torch.no_grad():
            dp_excluded = model.forward_given_params(x_excluded, weights, biases)

        torch.manual_seed(42)
        with torch.no_grad():
            dp_included = model.forward_given_params(x_included, weights, biases)

    excluded_diff = float((dp_a[2] - dp_excluded[2]).abs().max().item())
    included_diff = float((dp_a[2] - dp_included[2]).abs().max().item())

    assert excluded_diff < 1e-6, (
        f"Excluded parent (column 1) still influenced target 2: "
        f"max |delta| = {excluded_diff:.6e}; structural masking failed."
    )
    assert included_diff > 1e-3, (
        f"Included parent (column 0) did not influence target 2: "
        f"max |delta| = {included_diff:.6e}; the included-parent sensitivity "
        "check is unexpectedly weak on this seed."
    )


# ---------------------------------------------------------------------------
# Preserved continuous-edge snapshots are independent of the live state
# ---------------------------------------------------------------------------


def test_preserved_continuous_edges_unchanged_after_context_exception():
    """Detached continuous-edge snapshots stay bitwise stable across the context.

    snapshot_log_alpha and snapshot_w_adj return detached CPU clones
    held by the caller. Neither the in-context mutation of the live
    model state nor an induced exception inside the with-block should
    affect those snapshots. This is distinct from restoring the live
    model state, which is covered by test_restoration_after_induced_exception.
    """
    model, mask = _build_model_and_mask()

    preserved_log_alpha = snapshot_log_alpha(model)
    preserved_w_adj = snapshot_w_adj(model)

    log_alpha_baseline = preserved_log_alpha.clone()
    w_adj_baseline = preserved_w_adj.clone()

    class _InducedError(RuntimeError):
        pass

    with pytest.raises(_InducedError):
        with _structural_mask_context(model, mask):
            raise _InducedError("induced for snapshot-stability test")

    assert torch.equal(preserved_log_alpha, log_alpha_baseline), (
        "Preserved log_alpha snapshot changed after an in-context exception."
    )
    assert torch.equal(preserved_w_adj, w_adj_baseline), (
        "Preserved w_adj snapshot changed after an in-context exception."
    )
