"""Status taxonomy and WrapperDiagnostics for causal discovery wrappers.

The three status axes are independent. For example, training_status can be
"converged" while graph_status is "cyclic", or graph_status can be
"valid_dag" while sampler_status is "unavailable_unresolved_noise_policy".

WrapperDiagnostics is a TypedDict returned by get_diagnostics() after a fit.
Every key is present; Optional fields are explicitly None when not applicable.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Status Literals
# ---------------------------------------------------------------------------

TrainingStatus = Literal[
    "converged",
    "max_iter",
    "diverged",
    "wrapper_error",
]

GraphStatus = Literal[
    "valid_dag",
    "cyclic",
    "bidirected",
    "self_loop",
    "invalid_shape",
]

SamplerStatus = Literal[
    "available",
    "unavailable_invalid_graph",
    "unavailable_no_api",
    "unavailable_unresolved_noise_policy",
]


# ---------------------------------------------------------------------------
# Structured diagnostics
# ---------------------------------------------------------------------------


class WrapperDiagnostics(TypedDict):
    """Structured diagnostic output from a fitted wrapper.

    Every key is present after a fit. Optional keys are None when the
    associated feature is absent or unknown.
    """

    training_status: TrainingStatus
    graph_status: GraphStatus
    sampler_status: SamplerStatus
    seed: int
    n_iterations: int
    config_snapshot: dict[str, object]
    """Resolved DCDIConfig values for this run; always populated even when
    config=None was passed to fit (records the applied defaults)."""
    loss_history: list[float]
    loss_decomposition_final: dict[str, float]
    """Keys: nll, reg, prior, gamma, mu, h at the final training iteration."""
    convergence_info: dict[str, object]
    """Keys: first_stop (int or None), final_iter (int), converged (bool)."""
    continuous_log_alpha_pre_threshold: np.ndarray
    continuous_w_adj_pre_threshold: np.ndarray
    thresholded_adjacency: np.ndarray
    graph_invalid_reason: Optional[str]
    sampler_unavailable_reason: Optional[str]
    mmd_sampling_metadata: dict[str, object]
    """Keys: sample_seed, transform_mode, scaler_stats."""
    loss_hook_name: Optional[str]
    numerical_tolerances: dict[str, float]
