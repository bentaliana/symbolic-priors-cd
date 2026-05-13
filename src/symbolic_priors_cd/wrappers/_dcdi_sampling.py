"""DCDI structural-mask context and model-frame ancestral sampler.

Implements the save/mutate/restore pattern that enforces a thresholded
structural mask on a fitted DCDI model during sampling. The context
manager mutates only the live model's adjacency buffer and the
gumbel_adjacency log_alpha parameter, and always restores their
pre-context values on exit, including when an exception is raised
inside the context.

Detached continuous-edge snapshots held by the caller (for example the
tensors stored on TrainingResult) are independent of the live model
state and are therefore unaffected by this context manager.

sample_model_frame_dcdi performs ancestral sampling inside the
structural-mask context. It produces samples in model frame.

sample_raw_units_dcdi wraps sample_model_frame_dcdi with preprocessor
support: it transforms the raw-unit intervention value to model frame,
samples, then applies inverse_transform to return raw-unit samples.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

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


def _topological_order(adjacency: np.ndarray) -> list[int]:
    """Return node indices in topological order for a boolean DAG adjacency.

    Uses Kahn's algorithm. The initial ready set is sorted so the output
    is deterministic for a given adjacency.

    Parameters
    ----------
    adjacency : np.ndarray
        Boolean adjacency matrix of shape (d, d), row-source /
        column-destination convention. adjacency[i, j] = True means
        edge i -> j.

    Returns
    -------
    list[int]
        Node indices in topological order (parents before children).

    Raises
    ------
    ValueError
        If fewer than d nodes are processed, indicating a cycle.
    """
    d = adjacency.shape[0]
    in_degree = adjacency.sum(axis=0).astype(int)
    ready = sorted(i for i in range(d) if in_degree[i] == 0)
    order: list[int] = []
    while ready:
        j = ready.pop(0)
        order.append(j)
        children = np.nonzero(adjacency[j, :])[0]
        for k in children:
            in_degree[k] -= 1
            if in_degree[k] == 0:
                ready.append(k)
        ready.sort()
    if len(order) != d:
        raise ValueError(
            f"Topological sort incomplete: processed {len(order)} of {d} nodes. "
            "The adjacency may contain a cycle."
        )
    return order


def sample_model_frame_dcdi(
    model: LearnableModel_NonLinGaussANM,
    a_thresh: np.ndarray,
    target: int,
    intervention_value: float,
    n_samples: int,
    sample_seed: int,
) -> np.ndarray:
    """Sample from DCDI in model frame using ancestral sampling with target clamping.

    Traverses nodes in topological order. At the intervention target, all
    n_samples values in that column are set to intervention_value. For all
    other nodes, density parameters are obtained by calling
    forward_given_params on the current sample matrix x, and one sample
    per row is drawn from the resulting conditional Normal.

    Structural masking is enforced via _structural_mask_context for the
    duration of sampling: model.adjacency and gumbel_adjacency.log_alpha
    are set to the thresholded mask and are restored on exit even if an
    exception is raised. The Gumbel draws inside forward_given_params are
    effectively deterministic with saturated log_alpha (+100 or -100),
    so excluded parents contribute zero to any conditional.

    The model is placed in eval mode for the duration of the call and
    restored to its previous training mode on exit.

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        A fitted DCDI model.
    a_thresh : np.ndarray
        Boolean adjacency matrix of shape (num_vars, num_vars),
        row-source / column-destination convention. Must be a valid DAG.
    target : int
        Index of the intervened variable (0-indexed).
    intervention_value : float
        Model-frame value written into the target column for all samples.
    n_samples : int
        Number of samples to draw. Must be at least 1.
    sample_seed : int
        Seed for torch.manual_seed before sampling begins. Identical
        arguments with the same sample_seed produce identical output.

    Returns
    -------
    np.ndarray
        Floating array of shape (n_samples, num_vars) containing samples
        in model frame. The dtype matches model.gumbel_adjacency.log_alpha.

    Raises
    ------
    ValueError
        If a_thresh is not a valid DAG, target is out of range, or
        n_samples is less than 1.
    """
    from symbolic_priors_cd.wrappers.dcdi import classify_graph_status

    graph_status, reason = classify_graph_status(a_thresh)
    if graph_status != "valid_dag":
        raise ValueError(
            f"Cannot sample: graph status is '{graph_status}'. Reason: {reason}"
        )

    num_vars = model.num_vars
    if not (0 <= target < num_vars):
        raise ValueError(
            f"target={target} is out of range for a model with {num_vars} variables."
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")

    order = _topological_order(a_thresh)
    torch.manual_seed(sample_seed)

    was_training = model.training
    model.eval()

    try:
        log_alpha_param = model.gumbel_adjacency.log_alpha
        x = torch.zeros(
            n_samples, num_vars,
            dtype=log_alpha_param.dtype,
            device=log_alpha_param.device,
        )

        with _structural_mask_context(model, a_thresh):
            weights, biases, raw_extra = model.get_parameters(mode="wbx")
            transformed_extra = (
                model.transform_extra_params(model.extra_params)
                if len(raw_extra) != 0
                else []
            )

            with torch.no_grad():
                for j in order:
                    if j == target:
                        x[:, j] = float(intervention_value)
                    else:
                        all_dp = model.forward_given_params(x, weights, biases)
                        dp_j = list(torch.unbind(all_dp[j], 1))
                        if transformed_extra:
                            dp_j.extend(
                                list(torch.unbind(transformed_extra[j], 0))
                            )
                        conditional = model.get_distribution(dp_j)
                        x[:, j] = conditional.sample()
    finally:
        if was_training:
            model.train()

    return x.detach().cpu().numpy()


def sample_raw_units_dcdi(
    model: LearnableModel_NonLinGaussANM,
    a_thresh: np.ndarray,
    target: int,
    raw_intervention_value: float,
    n_samples: int,
    sample_seed: int,
    preprocessor: Union[
        "CentredOnlyTransform", "StandardisedTransform"
    ],
) -> np.ndarray:
    """Sample from DCDI and return results in raw SCM units.

    Converts the raw-unit intervention value to model frame using the
    fitted preprocessor, delegates all sampling to sample_model_frame_dcdi,
    then applies inverse_transform to the returned model-frame samples.

    The preprocessor must already be fitted. This function does not call
    preprocessor.fit and does not modify any stored preprocessor statistics.

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        A fitted DCDI model.
    a_thresh : np.ndarray
        Boolean adjacency matrix of shape (num_vars, num_vars),
        row-source / column-destination convention. Must be a valid DAG.
    target : int
        Index of the intervened variable (0-indexed).
    raw_intervention_value : float
        Intervention value in raw SCM units. Converted to model frame
        internally via preprocessor.transform_intervention_value.
    n_samples : int
        Number of samples to draw. Must be at least 1.
    sample_seed : int
        Seed for torch.manual_seed. Identical arguments produce identical
        output for the same fitted model and preprocessor.
    preprocessor : CentredOnlyTransform or StandardisedTransform
        A fitted preprocessor used to translate between raw and model frame.

    Returns
    -------
    np.ndarray
        Float64 array of shape (n_samples, num_vars) in raw SCM units.
        The target column equals raw_intervention_value up to float32
        precision of the sample tensor.
    """
    from symbolic_priors_cd.wrappers.preprocessing import (
        CentredOnlyTransform,
        StandardisedTransform,
    )

    if not isinstance(preprocessor, (CentredOnlyTransform, StandardisedTransform)):
        raise TypeError(
            f"preprocessor must be a CentredOnlyTransform or StandardisedTransform, "
            f"got {type(preprocessor).__name__}."
        )

    model_frame_value = preprocessor.transform_intervention_value(
        raw_intervention_value, target
    )
    model_frame_samples = sample_model_frame_dcdi(
        model, a_thresh, target, model_frame_value, n_samples, sample_seed,
    )
    return preprocessor.inverse_transform(model_frame_samples)
