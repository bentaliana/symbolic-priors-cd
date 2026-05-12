"""Tests for _dcdi_utils: model instantiation and parameter snapshot helpers.

Covers make_dcdi_model correctness (num_vars, observational mode, tensor
shapes, diagonal masking) and the snapshot helpers (detachment, CPU
placement, independence from the model parameter buffer).
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """A 3-variable DCDI model for read-only structural checks."""
    from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model
    return make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)


# ---------------------------------------------------------------------------
# make_dcdi_model
# ---------------------------------------------------------------------------


def test_make_dcdi_model_num_vars(tiny_model):
    """make_dcdi_model sets num_vars to the requested value."""
    assert tiny_model.num_vars == 3


def test_make_dcdi_model_observational_mode(tiny_model):
    """make_dcdi_model creates a model with intervention=False."""
    assert tiny_model.intervention is False


def test_make_dcdi_model_log_alpha_shape(tiny_model):
    """gumbel_adjacency.log_alpha has shape (num_vars, num_vars) after instantiation."""
    d = tiny_model.num_vars
    assert tiny_model.gumbel_adjacency.log_alpha.shape == (d, d)


def test_make_dcdi_model_w_adj_shape(tiny_model):
    """get_w_adj() has shape (num_vars, num_vars)."""
    d = tiny_model.num_vars
    assert tiny_model.get_w_adj().shape == (d, d)


def test_make_dcdi_model_w_adj_diagonal_zero(tiny_model):
    """get_w_adj() has exactly zero diagonal, enforced by the (1 - I) mask."""
    diagonal = torch.diagonal(tiny_model.get_w_adj())
    assert torch.all(diagonal == 0.0), (
        f"Expected zero diagonal, got {diagonal.tolist()}"
    )


# ---------------------------------------------------------------------------
# snapshot_log_alpha
# ---------------------------------------------------------------------------


def test_snapshot_log_alpha_requires_grad_false(tiny_model):
    """snapshot_log_alpha returns a tensor with requires_grad=False."""
    from symbolic_priors_cd.wrappers._dcdi_utils import snapshot_log_alpha

    snap = snapshot_log_alpha(tiny_model)
    assert not snap.requires_grad


def test_snapshot_log_alpha_is_cpu(tiny_model):
    """snapshot_log_alpha returns a CPU tensor."""
    from symbolic_priors_cd.wrappers._dcdi_utils import snapshot_log_alpha

    snap = snapshot_log_alpha(tiny_model)
    assert snap.device.type == "cpu"


def test_snapshot_log_alpha_is_independent_clone():
    """Mutating the snapshot does not affect the model parameter."""
    from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model, snapshot_log_alpha

    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    original = model.gumbel_adjacency.log_alpha[0, 1].item()

    snap = snapshot_log_alpha(model)
    snap[0, 1] = snap[0, 1] + 999.0

    after = model.gumbel_adjacency.log_alpha[0, 1].item()
    assert after == original, (
        "Mutating the snapshot changed the model parameter; the clone is not independent."
    )


# ---------------------------------------------------------------------------
# snapshot_w_adj
# ---------------------------------------------------------------------------


def test_snapshot_w_adj_requires_grad_false(tiny_model):
    """snapshot_w_adj returns a tensor with requires_grad=False."""
    from symbolic_priors_cd.wrappers._dcdi_utils import snapshot_w_adj

    snap = snapshot_w_adj(tiny_model)
    assert not snap.requires_grad


def test_snapshot_w_adj_is_cpu(tiny_model):
    """snapshot_w_adj returns a CPU tensor."""
    from symbolic_priors_cd.wrappers._dcdi_utils import snapshot_w_adj

    snap = snapshot_w_adj(tiny_model)
    assert snap.device.type == "cpu"


def test_snapshot_w_adj_is_independent_clone():
    """Mutating the snapshot does not affect subsequent get_w_adj() calls."""
    from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model, snapshot_w_adj

    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    original = model.get_w_adj()[0, 1].item()

    snap = snapshot_w_adj(model)
    snap[0, 1] = snap[0, 1] + 999.0

    after = model.get_w_adj()[0, 1].item()
    assert after == original, (
        "Mutating the snapshot changed model.get_w_adj(); the clone is not independent."
    )
