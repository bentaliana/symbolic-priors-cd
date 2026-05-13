"""Tests for the DAGMA wrapper public interface.

Covers DAGMAConfig defaults, the pinned DAGMA source import boundary,
DAGMAWrapper construction and stub behaviour, the public re-exports
from the wrappers package, and the shared WrapperDiagnostics schema.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import typing

import numpy as np
import pytest

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers import (
    DAGMAConfig,
    DAGMAWrapper,
    WrapperDiagnostics,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig as _CfgDirect
from symbolic_priors_cd.wrappers.dagma import DAGMAWrapper as _WrapperDirect
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


def test_dagma_public_reexports():
    """DAGMAConfig and DAGMAWrapper are re-exported from the wrappers package."""
    import symbolic_priors_cd.wrappers as wrappers

    assert hasattr(wrappers, "DAGMAConfig")
    assert hasattr(wrappers, "DAGMAWrapper")
    assert wrappers.DAGMAConfig is _CfgDirect
    assert wrappers.DAGMAWrapper is _WrapperDirect


# ---------------------------------------------------------------------------
# DAGMAConfig defaults
# ---------------------------------------------------------------------------


def test_dagmaconfig_project_required_overrides():
    """The project-required DAGMA override values are present verbatim."""
    cfg = DAGMAConfig()
    assert cfg.T == 4
    assert cfg.lambda1 == 0.05
    assert cfg.s == (1.0, 0.9, 0.8, 0.7)
    assert cfg.mu_init == 1.0
    assert cfg.mu_factor == 0.1
    assert cfg.w_threshold_internal == 0.0


def test_dagmaconfig_recorded_library_defaults():
    """DAGMA library defaults are pinned so they appear in the run record."""
    cfg = DAGMAConfig()
    assert cfg.lr == 3e-4
    assert cfg.warm_iter == 30000
    assert cfg.max_iter == 60000
    assert cfg.beta_1 == 0.99
    assert cfg.beta_2 == 0.999
    assert cfg.loss_type == "l2"


def test_dagmaconfig_wrapper_constants():
    """Wrapper-level constants carry the expected default values."""
    cfg = DAGMAConfig()
    assert cfg.project_threshold == 0.3
    assert cfg.h_diagnostic_threshold == 1e-5


def test_dagmaconfig_is_frozen_dataclass():
    """DAGMAConfig is a frozen dataclass so instances are immutable."""
    assert dataclasses.is_dataclass(DAGMAConfig)
    cfg = DAGMAConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.T = 99  # type: ignore[misc]


def test_dagmaconfig_s_is_immutable_tuple():
    """The s sequence is a tuple of floats so it survives frozen-dataclass semantics."""
    cfg = DAGMAConfig()
    assert isinstance(cfg.s, tuple)
    assert all(isinstance(x, float) for x in cfg.s)


# ---------------------------------------------------------------------------
# DAGMA source import path
# ---------------------------------------------------------------------------


def test_dagma_imported_from_pinned_source():
    """_dagma_utils must load dagma.linear from the project's pinned source.

    Uses a subprocess so module caching from earlier tests cannot mask
    a misconfigured import path.
    """
    code = (
        "import sys; "
        "from pathlib import Path; "
        "import symbolic_priors_cd.wrappers._dagma_utils as u; "
        "f = Path(sys.modules['dagma.linear'].__file__).resolve(); "
        "ok = f.is_relative_to(u._DAGMA_SRC); "
        "sys.exit(0 if ok else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "dagma.linear was not imported from the pinned source.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_dagma_source_path_attribute_exposed():
    """_dagma_utils exposes DAGMA_SOURCE_PATH pointing inside the pinned source."""
    from symbolic_priors_cd.wrappers import _dagma_utils

    assert _dagma_utils.DAGMA_SOURCE_PATH.exists()
    assert _dagma_utils.DAGMA_SOURCE_PATH.is_relative_to(_dagma_utils._DAGMA_SRC)
    assert _dagma_utils.DAGMA_SOURCE_PATH.name == "linear.py"


def test_dagma_utils_does_not_import_dagma_utils_submodule():
    """_dagma_utils must not pull in dagma.utils as a side effect.

    Uses a subprocess so the check is not polluted by modules already
    loaded in the current pytest session.
    """
    code = (
        "import sys; "
        "import symbolic_priors_cd.wrappers._dagma_utils; "
        "assert 'dagma.utils' not in sys.modules, "
        "'dagma.utils was imported as a side effect of _dagma_utils'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "dagma.utils was imported as a side effect of importing _dagma_utils.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# DAGMAWrapper construction and implemented methods
# ---------------------------------------------------------------------------


def test_dagma_wrapper_constructable():
    """DAGMAWrapper can be instantiated with no arguments."""
    wrapper = DAGMAWrapper()
    assert wrapper is not None


def test_dagma_wrapper_fit_runs_on_valid_input():
    """fit completes without error on a valid 2D float array."""
    wrapper = DAGMAWrapper()
    pre = CentredOnlyTransform()
    X = np.random.default_rng(5).standard_normal((20, 3))
    pre.fit(X)
    wrapper.fit(X, preprocessor=pre, seed=0)
    assert wrapper._fitted


def test_dagma_wrapper_native_edge_continuous_stub_raises():
    """native_edge_continuous raises NotImplementedError until implemented."""
    wrapper = DAGMAWrapper()
    with pytest.raises(NotImplementedError):
        wrapper.native_edge_continuous()


def test_dagma_wrapper_thresholded_adjacency_stub_raises():
    """thresholded_adjacency raises NotImplementedError until implemented."""
    wrapper = DAGMAWrapper()
    with pytest.raises(NotImplementedError):
        wrapper.thresholded_adjacency()


def test_dagma_wrapper_sample_interventional_stub_raises():
    """sample_interventional raises NotImplementedError until implemented."""
    wrapper = DAGMAWrapper()
    intervention = Intervention(target=0, value=1.0)
    with pytest.raises(NotImplementedError):
        wrapper.sample_interventional(intervention, n_samples=10, sample_seed=0)


def test_dagma_wrapper_get_diagnostics_stub_raises():
    """get_diagnostics raises NotImplementedError until implemented."""
    wrapper = DAGMAWrapper()
    with pytest.raises(NotImplementedError):
        wrapper.get_diagnostics()


def test_dagma_wrapper_sample_interventional_signature():
    """sample_interventional declares noise_policy as a keyword-only argument
    with the expected default and Literal options."""
    sig = inspect.signature(DAGMAWrapper.sample_interventional)
    assert "noise_policy" in sig.parameters
    param = sig.parameters["noise_policy"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default == "residual_fitted"
    # Annotations are deferred via "from __future__ import annotations", so
    # resolve them with typing.get_type_hints before extracting Literal args.
    hints = typing.get_type_hints(DAGMAWrapper.sample_interventional)
    annotation_args = typing.get_args(hints["noise_policy"])
    assert set(annotation_args) == {"residual_fitted", "unit_variance"}


def test_dagma_wrapper_fit_signature_has_keyword_only_args():
    """fit declares preprocessor, seed, and config as keyword-only arguments."""
    sig = inspect.signature(DAGMAWrapper.fit)
    assert sig.parameters["preprocessor"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["seed"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["config"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["config"].default is None


# ---------------------------------------------------------------------------
# Shared WrapperDiagnostics schema
# ---------------------------------------------------------------------------


def test_wrapper_diagnostics_has_model_specific_diagnostics():
    """WrapperDiagnostics exposes a model_specific_diagnostics field for
    wrapper-specific records."""
    annotations = WrapperDiagnostics.__annotations__
    assert "model_specific_diagnostics" in annotations


def test_wrapper_diagnostics_no_top_level_continuous_log_alpha():
    """continuous_log_alpha_pre_threshold is not a top-level WrapperDiagnostics
    key; it belongs inside model_specific_diagnostics."""
    annotations = WrapperDiagnostics.__annotations__
    assert "continuous_log_alpha_pre_threshold" not in annotations


def test_wrapper_diagnostics_no_top_level_continuous_w_adj():
    """continuous_w_adj_pre_threshold is not a top-level WrapperDiagnostics
    key; it belongs inside model_specific_diagnostics."""
    annotations = WrapperDiagnostics.__annotations__
    assert "continuous_w_adj_pre_threshold" not in annotations
