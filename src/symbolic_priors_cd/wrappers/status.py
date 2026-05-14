"""Status taxonomy and WrapperDiagnostics for causal discovery wrappers.

The three status axes are independent. For example, training_status can be
"converged" while graph_status is "cyclic", or graph_status can be
"valid_dag" while sampler_status is "unavailable_unresolved_noise_policy".

WrapperDiagnostics is a TypedDict returned by get_diagnostics() after a fit.
Every key is present; Optional fields are explicitly None when not applicable.
Fields unique to a single wrapper live inside model_specific_diagnostics so
the top-level schema stays generic across implementations.
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
    associated feature is absent or unknown. Wrapper-specific fields
    are placed inside model_specific_diagnostics rather than as
    top-level keys.
    """

    training_status: TrainingStatus
    graph_status: GraphStatus
    sampler_status: SamplerStatus
    seed: int
    n_iterations: Optional[int]
    """Observed iteration count, or None when unavailable."""
    config_snapshot: dict[str, object]
    """Resolved configuration values for this run; always populated even when
    config=None was passed to fit (records the applied defaults)."""
    loss_history: list[float]
    loss_decomposition_final: dict[str, float]
    """Final-iteration loss breakdown. Wrappers without a meaningful
    breakdown populate an empty dict."""
    convergence_info: dict[str, object]
    """Per-wrapper convergence information. Keys vary by implementation."""
    thresholded_adjacency: np.ndarray
    graph_invalid_reason: Optional[str]
    sampler_unavailable_reason: Optional[str]
    mmd_sampling_metadata: dict[str, object]
    """Sampling-policy metadata; per-call records are optional."""
    loss_hook_name: Optional[str]
    numerical_tolerances: dict[str, float]
    model_specific_diagnostics: dict[str, object]
    """Wrapper-specific fields not shared across implementations."""
