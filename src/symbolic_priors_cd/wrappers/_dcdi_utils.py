"""Low-level DCDI model helpers for the wrapper layer.

All DCDI imports are targeted low-level imports. dcdi.train is never
imported; importing it would pull in cdt and its R dependency chain, which
are not required for wrapper functionality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Resolve the DCDI source path relative to this file.
# This file lives at src/symbolic_priors_cd/wrappers/_dcdi_utils.py,
# so four parent steps reach the project root.
_DCDI_SRC = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "external"
    / "source_inspection"
    / "dcdi"
)

_learnables_path = _DCDI_SRC / "dcdi" / "models" / "learnables.py"
if not _DCDI_SRC.exists() or not _learnables_path.exists():
    raise ImportError(
        f"Pinned DCDI source not found at '{_DCDI_SRC}'. "
        "The inspected DCDI source must be present at "
        "external/source_inspection/dcdi relative to the project root."
    )
del _learnables_path

if str(_DCDI_SRC) not in sys.path:
    sys.path.insert(0, str(_DCDI_SRC))

from dcdi.models.learnables import LearnableModel_NonLinGaussANM  # noqa: E402

_actual = Path(sys.modules["dcdi.models.learnables"].__file__).resolve()
if not _actual.is_relative_to(_DCDI_SRC):
    raise ImportError(
        f"dcdi.models.learnables was imported from '{_actual}', "
        f"not from the expected source at '{_DCDI_SRC}'. "
        "A different DCDI installation may be shadowing the pinned source. "
        "Check sys.path ordering."
    )
del _actual


def make_dcdi_model(
    num_vars: int,
    num_layers: int = 2,
    hid_dim: int = 16,
    nonlin: str = "leaky-relu",
) -> LearnableModel_NonLinGaussANM:
    """Instantiate a DCDI-G model in observational-only mode.

    The model is configured with intervention=False, intervention_type="perfect",
    intervention_knowledge="known", and num_regimes=1. This matches the
    observational-only training path used by the wrapper.

    Parameters
    ----------
    num_vars : int
        Number of variables in the causal graph.
    num_layers : int
        Number of hidden layers in each per-variable conditional MLP.
    hid_dim : int
        Hidden units per layer in each conditional MLP.
    nonlin : str
        Nonlinearity for each hidden layer, e.g. "leaky-relu".

    Returns
    -------
    LearnableModel_NonLinGaussANM
        A freshly initialised model on the default device (CPU by default).
    """
    return LearnableModel_NonLinGaussANM(
        num_vars=num_vars,
        num_layers=num_layers,
        hid_dim=hid_dim,
        nonlin=nonlin,
        intervention=False,
        intervention_type="perfect",
        intervention_knowledge="known",
        num_regimes=1,
    )


def snapshot_log_alpha(model: LearnableModel_NonLinGaussANM) -> torch.Tensor:
    """Return a detached CPU clone of model.gumbel_adjacency.log_alpha.

    The returned tensor is fully independent of the model parameter buffer.
    Mutating it does not affect the model's parameters or gradients.

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        A DCDI model instance, fitted or freshly initialised.

    Returns
    -------
    torch.Tensor
        CPU float tensor of shape (num_vars, num_vars) with requires_grad=False.
    """
    return model.gumbel_adjacency.log_alpha.detach().cpu().clone()


def snapshot_w_adj(model: LearnableModel_NonLinGaussANM) -> torch.Tensor:
    """Return a detached CPU clone of model.get_w_adj().

    get_w_adj() evaluates sigmoid(log_alpha) * (1 - I), so the returned
    tensor has an exactly zero diagonal and off-diagonal entries in [0, 1].

    The returned tensor is fully independent of the model's computation graph.
    Mutating it does not affect any future call to get_w_adj().

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        A DCDI model instance, fitted or freshly initialised.

    Returns
    -------
    torch.Tensor
        CPU float tensor of shape (num_vars, num_vars) with requires_grad=False.
    """
    return model.get_w_adj().detach().cpu().clone()
