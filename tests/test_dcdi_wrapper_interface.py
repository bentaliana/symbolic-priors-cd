"""Tests for the DCDI wrapper public interface.

Covers package importability, status Literal values, and the
WrapperDiagnostics key contract.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from symbolic_priors_cd.wrappers import (
    GraphStatus,
    SamplerStatus,
    TrainingStatus,
    WrapperDiagnostics,
)
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics as _WDiag


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_wrappers_package_importable():
    """The wrappers package and its public names import without error."""
    import symbolic_priors_cd.wrappers as wrappers  # noqa: F401

    assert hasattr(wrappers, "TrainingStatus")
    assert hasattr(wrappers, "GraphStatus")
    assert hasattr(wrappers, "SamplerStatus")
    assert hasattr(wrappers, "WrapperDiagnostics")


# ---------------------------------------------------------------------------
# TrainingStatus
# ---------------------------------------------------------------------------


def test_training_status_values():
    """TrainingStatus Literal contains exactly the four documented values."""
    import typing

    args = typing.get_args(TrainingStatus)
    assert set(args) == {"converged", "max_iter", "diverged", "wrapper_error"}


# ---------------------------------------------------------------------------
# GraphStatus
# ---------------------------------------------------------------------------


def test_graph_status_values():
    """GraphStatus Literal contains exactly the five documented values."""
    import typing

    args = typing.get_args(GraphStatus)
    assert set(args) == {"valid_dag", "cyclic", "bidirected", "self_loop", "invalid_shape"}


# ---------------------------------------------------------------------------
# SamplerStatus
# ---------------------------------------------------------------------------


def test_sampler_status_values():
    """SamplerStatus Literal contains exactly the four documented values."""
    import typing

    args = typing.get_args(SamplerStatus)
    assert set(args) == {
        "available",
        "unavailable_invalid_graph",
        "unavailable_no_api",
        "unavailable_unresolved_noise_policy",
    }


# ---------------------------------------------------------------------------
# WrapperDiagnostics
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "training_status",
    "graph_status",
    "sampler_status",
    "seed",
    "n_iterations",
    "config_snapshot",
    "loss_history",
    "loss_decomposition_final",
    "convergence_info",
    "continuous_log_alpha_pre_threshold",
    "continuous_w_adj_pre_threshold",
    "thresholded_adjacency",
    "graph_invalid_reason",
    "sampler_unavailable_reason",
    "mmd_sampling_metadata",
    "loss_hook_name",
    "numerical_tolerances",
}


def test_wrapper_diagnostics_has_expected_keys():
    """WrapperDiagnostics defines all expected keys."""
    annotations = _WDiag.__annotations__
    assert set(annotations.keys()) == _EXPECTED_KEYS


def test_wrapper_diagnostics_key_count():
    """WrapperDiagnostics has exactly the expected number of keys."""
    assert len(_WDiag.__annotations__) == len(_EXPECTED_KEYS)
