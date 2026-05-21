"""Assembly tests for the public DCDIWrapper.

These tests verify the public surface of ``DCDIWrapper``:
importability and dynamic resolution via ``importlib`` plus
``getattr``; fit completion on a small input; native edge,
thresholded adjacency, and interventional sampling shape/type
contracts; argument-validation rejection of bool and non-int
``n_samples`` and ``sample_seed``; the ``WrapperDiagnostics``
schema (every mandatory key present, ``loss_hook_name is None``,
continuous tensors exposed as defensive clones); and the
sampler-status discipline that mechanical availability is
determined by ``graph_status`` alone, not by the quality of the
learned structure.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np
import pytest
import torch

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers import (
    DCDIConfig,
    DCDIWrapper,
    WrapperDiagnostics,
)
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _tiny_train_val(
    n_train: int = 32, n_val: int = 16, num_vars: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Build small fixed-seed training and validation arrays."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((n_train, num_vars)).astype(np.float64)
    X_val = rng.standard_normal((n_val, num_vars)).astype(np.float64)
    return X_train, X_val


def _fit_tiny_wrapper(seed: int = 0, n_iter: int = 20) -> DCDIWrapper:
    """Fit a DCDIWrapper on a tiny fixed-seed input and return it."""
    X_train, X_val = _tiny_train_val()
    preprocessor = CentredOnlyTransform().fit(X_train)
    X_train_model = preprocessor.transform(X_train)
    X_val_model = preprocessor.transform(X_val)
    config = DCDIConfig(stop_crit_win=10, train_batch_size=8)
    wrapper = DCDIWrapper()
    wrapper.fit(
        X_train_model,
        X_val=X_val_model,
        preprocessor=preprocessor,
        seed=seed,
        n_iter=n_iter,
        config=config,
    )
    return wrapper


def _force_valid_dag_state(wrapper: DCDIWrapper) -> None:
    """Mutate a fitted wrapper to a known valid-DAG cached state.

    A 3-node DCDI fit on random Gaussian input typically saturates
    to a bidirected adjacency because DCDI initialises edge
    probabilities near 1.0. To exercise the ``valid_dag`` branch of
    the wrapper's public surface in tests, this helper replaces the
    cached continuous matrix with a known upper-triangular tensor
    (edges 0->1 and 0->2, arbitrary weights) and re-derives the
    cached graph and sampler statuses through the public
    ``_graph_status`` helpers. The underlying
    ``model.gumbel_adjacency`` is unchanged, so the sampler
    exercises the model state actually learned by the tiny fit; the
    mutation only changes the cached structural mask through which
    sampling is gated.
    """
    from symbolic_priors_cd.wrappers._graph_status import (
        classify_graph_status,
        infer_sampler_status,
    )

    n = wrapper._continuous_w_adj_pre_threshold.shape[0]
    dtype = wrapper._continuous_w_adj_pre_threshold.dtype
    forced = torch.zeros((n, n), dtype=dtype)
    forced[0, 1] = 0.9
    if n >= 3:
        forced[0, 2] = 0.9
    wrapper._continuous_w_adj_pre_threshold = forced
    a_thresh = (forced.numpy() >= 0.5).astype(bool)
    graph_status, graph_reason = classify_graph_status(a_thresh)
    sampler_status, sampler_reason = infer_sampler_status(graph_status)
    assert graph_status == "valid_dag", (
        "test fixture invariant: forced continuous matrix must threshold "
        f"to a valid DAG; got {graph_status!r}."
    )
    wrapper._graph_status = graph_status
    wrapper._graph_invalid_reason = graph_reason
    wrapper._sampler_status = sampler_status
    wrapper._sampler_unavailable_reason = sampler_reason


# ---------------------------------------------------------------------------
# Importability and dynamic resolution
# ---------------------------------------------------------------------------


def test_dcdi_wrapper_importable_from_wrappers_package() -> None:
    """DCDIWrapper is reachable through the public wrappers package."""
    import symbolic_priors_cd.wrappers as wrappers

    assert hasattr(wrappers, "DCDIWrapper")
    assert wrappers.DCDIWrapper is DCDIWrapper


def test_dcdi_wrapper_dynamic_resolution_via_importlib_and_getattr() -> None:
    """Dynamic resolution of ``module:ClassName`` returns the class.

    Mirrors how a runner that resolves wrapper references by string
    (``importlib.import_module(module).<ClassName>``) would obtain the
    class.
    """
    module = importlib.import_module("symbolic_priors_cd.wrappers.dcdi")
    cls = getattr(module, "DCDIWrapper")
    assert cls is DCDIWrapper
    instance = cls()
    assert isinstance(instance, DCDIWrapper)


def test_dcdi_config_dynamic_resolution_via_importlib_and_getattr() -> None:
    """DCDIConfig is reachable via lazy ``__getattr__`` on the module."""
    module = importlib.import_module("symbolic_priors_cd.wrappers.dcdi")
    cfg_cls = getattr(module, "DCDIConfig")
    instance = cfg_cls()
    assert instance.h_threshold == DCDIConfig().h_threshold


def test_dcdi_config_default_reg_coeff_matches_reproduction_anchor() -> None:
    """DCDIConfig().reg_coeff must equal the reproduction-pass anchor value 0.1.

    The selection-study Configuration does not carry ``reg_coeff``
    as a top-level field; the reproduction-pass anchor for DCDI is
    therefore delivered through the DCDIConfig default. This
    regression test pins the default so a future wrapper-default
    change cannot silently move the reproduction-pass anchor.
    """
    cfg = DCDIConfig()
    assert cfg.reg_coeff == 0.1


# ---------------------------------------------------------------------------
# fit completion
# ---------------------------------------------------------------------------


def test_fit_runs_to_completion_on_small_input() -> None:
    """fit runs to completion on a small input and enters fitted state."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    # The wrapper is fitted; methods requiring a fit must no longer
    # raise RuntimeError.
    _ = wrapper.native_edge_continuous()
    _ = wrapper.thresholded_adjacency()
    _ = wrapper.get_diagnostics()


def test_fit_requires_X_val() -> None:
    """fit raises TypeError when X_val is omitted; X_val is keyword-only."""
    X_train, _ = _tiny_train_val()
    preprocessor = CentredOnlyTransform().fit(X_train)
    wrapper = DCDIWrapper()
    with pytest.raises(TypeError):
        wrapper.fit(  # type: ignore[call-arg]
            X_train,
            preprocessor=preprocessor,
            seed=0,
            n_iter=10,
        )


def test_fit_requires_n_iter() -> None:
    """fit raises TypeError when n_iter is omitted; n_iter is keyword-only."""
    X_train, X_val = _tiny_train_val()
    preprocessor = CentredOnlyTransform().fit(X_train)
    wrapper = DCDIWrapper()
    with pytest.raises(TypeError):
        wrapper.fit(  # type: ignore[call-arg]
            X_train,
            X_val=X_val,
            preprocessor=preprocessor,
            seed=0,
        )


def test_fit_rejects_bool_seed() -> None:
    """fit rejects bool seed even though bool is a subclass of int."""
    X_train, X_val = _tiny_train_val()
    preprocessor = CentredOnlyTransform().fit(X_train)
    wrapper = DCDIWrapper()
    with pytest.raises(ValueError, match="seed"):
        wrapper.fit(
            X_train,
            X_val=X_val,
            preprocessor=preprocessor,
            seed=True,  # type: ignore[arg-type]
            n_iter=10,
        )


def test_fit_rejects_non_positive_n_iter() -> None:
    """fit rejects n_iter < 1."""
    X_train, X_val = _tiny_train_val()
    preprocessor = CentredOnlyTransform().fit(X_train)
    wrapper = DCDIWrapper()
    with pytest.raises(ValueError, match="n_iter"):
        wrapper.fit(
            X_train,
            X_val=X_val,
            preprocessor=preprocessor,
            seed=0,
            n_iter=0,
        )


# ---------------------------------------------------------------------------
# native_edge_continuous and thresholded_adjacency
# ---------------------------------------------------------------------------


def test_native_edge_continuous_returns_2d_ndarray_of_expected_shape() -> None:
    """native_edge_continuous returns a float 2D ndarray of shape (d, d)."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    edges = wrapper.native_edge_continuous()
    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 2
    assert edges.shape == (3, 3)
    assert edges.dtype.kind == "f"


def test_native_edge_continuous_off_diagonal_in_unit_interval() -> None:
    """Off-diagonal continuous entries lie in [0, 1] per DCDI's get_w_adj."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    edges = wrapper.native_edge_continuous()
    off = ~np.eye(edges.shape[0], dtype=bool)
    assert np.all(edges[off] >= 0.0)
    assert np.all(edges[off] <= 1.0)


def test_native_edge_continuous_diagonal_is_zero() -> None:
    """DCDI's get_w_adj has an exactly zero diagonal."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    edges = wrapper.native_edge_continuous()
    assert np.all(np.diag(edges) == 0.0)


def test_native_edge_continuous_returns_independent_copy() -> None:
    """Mutating the returned array does not affect subsequent calls."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    edges_a = wrapper.native_edge_continuous()
    edges_a[0, 0] = 99.0
    edges_b = wrapper.native_edge_continuous()
    assert edges_b[0, 0] == 0.0


def test_thresholded_adjacency_returns_strict_bool_array() -> None:
    """Default threshold 0.5 produces a strict bool ndarray of shape (d, d)."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    adj = wrapper.thresholded_adjacency()
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (3, 3)
    assert adj.dtype == bool


def test_thresholded_adjacency_default_matches_05_threshold() -> None:
    """thresholded_adjacency() equals (native_edge_continuous() >= 0.5)."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    adj_default = wrapper.thresholded_adjacency()
    expected = wrapper.native_edge_continuous() >= 0.5
    assert np.array_equal(adj_default, expected)


def test_thresholded_adjacency_custom_threshold_is_monotone() -> None:
    """Higher thresholds weakly decrease the edge count."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    counts = [
        int(wrapper.thresholded_adjacency(threshold=t).sum())
        for t in (0.4, 0.5, 0.6)
    ]
    assert counts[0] >= counts[1] >= counts[2]


# ---------------------------------------------------------------------------
# sample_interventional
# ---------------------------------------------------------------------------


def test_sample_interventional_returns_expected_shape_for_valid_dag() -> None:
    """sample_interventional returns (n_samples, n_nodes) when graph is valid_dag."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    _force_valid_dag_state(wrapper)
    samples = wrapper.sample_interventional(
        Intervention(target=0, value=1.5),
        n_samples=10,
        sample_seed=0,
    )
    assert samples is not None
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (10, 3)


def test_sample_interventional_returns_none_for_non_valid_dag() -> None:
    """sample_interventional returns None when graph_status != valid_dag.

    After a normal fit, the cached ``_graph_status`` is mutated to
    ``"cyclic"`` to simulate a degenerate fit. The wrapper's
    contract is that the sampler is gated by ``graph_status`` and
    returns ``None`` for any non-valid_dag state.
    """
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    # Simulate a degenerate graph_status; the wrapper reads only this
    # cached attribute in sample_interventional's early-return branch.
    wrapper._graph_status = "cyclic"
    wrapper._sampler_status = "unavailable_invalid_graph"
    wrapper._sampler_unavailable_reason = "Test injection: simulated cyclic graph."
    out = wrapper.sample_interventional(
        Intervention(target=0, value=1.0),
        n_samples=10,
        sample_seed=0,
    )
    assert out is None


def test_sample_interventional_rejects_out_of_range_target() -> None:
    """sample_interventional validates intervention.target against n_vars."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    _force_valid_dag_state(wrapper)
    with pytest.raises(ValueError, match="target"):
        wrapper.sample_interventional(
            Intervention(target=99, value=1.0),
            n_samples=10,
            sample_seed=0,
        )


# ---------------------------------------------------------------------------
# Pre-fit error semantics
# ---------------------------------------------------------------------------


def test_methods_raise_runtime_error_on_unfitted_wrapper() -> None:
    """Methods that require a successful fit raise RuntimeError before fit is called."""
    wrapper = DCDIWrapper()
    with pytest.raises(RuntimeError):
        wrapper.native_edge_continuous()
    with pytest.raises(RuntimeError):
        wrapper.thresholded_adjacency()
    with pytest.raises(RuntimeError):
        wrapper.sample_interventional(
            Intervention(target=0, value=1.0),
            n_samples=10,
            sample_seed=0,
        )
    with pytest.raises(RuntimeError):
        wrapper.get_diagnostics()


# ---------------------------------------------------------------------------
# get_diagnostics
# ---------------------------------------------------------------------------


_MANDATORY_DIAG_KEYS = {
    "training_status",
    "graph_status",
    "sampler_status",
    "seed",
    "n_iterations",
    "config_snapshot",
    "loss_history",
    "loss_decomposition_final",
    "convergence_info",
    "thresholded_adjacency",
    "graph_invalid_reason",
    "sampler_unavailable_reason",
    "mmd_sampling_metadata",
    "loss_hook_name",
    "numerical_tolerances",
    "model_specific_diagnostics",
}


def test_get_diagnostics_returns_all_mandatory_keys() -> None:
    """get_diagnostics populates every top-level WrapperDiagnostics key."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    assert set(diag.keys()) == _MANDATORY_DIAG_KEYS


def test_get_diagnostics_loss_hook_name_is_none() -> None:
    """loss_hook_name is None because the wrapper does not register a loss hook."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    assert diag["loss_hook_name"] is None


def test_get_diagnostics_thresholded_adjacency_is_strict_bool() -> None:
    """The top-level thresholded_adjacency entry is a strict bool ndarray."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    a = diag["thresholded_adjacency"]
    assert isinstance(a, np.ndarray)
    assert a.dtype == bool
    assert a.shape == (3, 3)


def test_get_diagnostics_status_values_are_in_taxonomy() -> None:
    """training/graph/sampler statuses are members of their Literal taxonomies."""
    from symbolic_priors_cd.wrappers.status import (
        GraphStatus,
        SamplerStatus,
        TrainingStatus,
    )

    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    assert diag["training_status"] in typing.get_args(TrainingStatus)
    assert diag["graph_status"] in typing.get_args(GraphStatus)
    assert diag["sampler_status"] in typing.get_args(SamplerStatus)


def test_get_diagnostics_exposes_validation_nll_history() -> None:
    """convergence_info carries a validation-NLL trajectory.

    The trajectory is the per-evaluation list collected by the
    augmented-Lagrangian loop at the same cadence the loop already
    uses (one pre-training baseline plus one value every
    stop_crit_win iterations). The diagnostic field is type-stable:
    a list of finite floats whose length is at least one (the
    pre-training baseline).
    """
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    convergence_info = diag["convergence_info"]
    assert "validation_nll_history" in convergence_info
    assert "validation_nll_stop_crit_win" in convergence_info
    trajectory = convergence_info["validation_nll_history"]
    assert isinstance(trajectory, list)
    assert len(trajectory) >= 1
    for value in trajectory:
        assert isinstance(value, float)
        assert not isinstance(value, bool)
    cadence = convergence_info["validation_nll_stop_crit_win"]
    assert isinstance(cadence, int) and not isinstance(cadence, bool)
    assert cadence > 0


def test_get_diagnostics_seed_and_n_iterations_are_int() -> None:
    """seed and n_iterations are plain Python ints, not numpy scalars."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    assert isinstance(diag["seed"], int) and not isinstance(diag["seed"], bool)
    assert (
        isinstance(diag["n_iterations"], int)
        and not isinstance(diag["n_iterations"], bool)
    )


def test_get_diagnostics_model_specific_has_continuous_tensors() -> None:
    """model_specific_diagnostics carries the preserved continuous tensors."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag = wrapper.get_diagnostics()
    ms = diag["model_specific_diagnostics"]
    assert "continuous_log_alpha_pre_threshold" in ms
    assert "continuous_w_adj_pre_threshold" in ms
    assert isinstance(ms["continuous_log_alpha_pre_threshold"], torch.Tensor)
    assert isinstance(ms["continuous_w_adj_pre_threshold"], torch.Tensor)


# ---------------------------------------------------------------------------
# sampler_status discipline (CRITICAL)
# ---------------------------------------------------------------------------


def test_sampler_status_is_available_for_any_valid_dag_post_fit() -> None:
    """sampler_status is 'available' on any valid_dag, regardless of structure quality.

    Poor learned structure must surface in downstream metrics
    (SHD, SID, MMD), not in sampler availability. This test runs a
    tiny fit, forces a known valid-DAG cached state with arbitrary
    weights, asserts that ``sampler_status`` becomes ``"available"``,
    and confirms the sampler is mechanically callable.

    Regression intent: protect against the case where a learned
    graph missed strong true edges under a valid thresholded DAG
    with a callable sampling API; that structure-quality failure
    must remain visible in SHD / SID / MMD and must not be hidden
    behind a sampler_status flag.
    """
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    _force_valid_dag_state(wrapper)
    diag = wrapper.get_diagnostics()
    assert diag["graph_status"] == "valid_dag"
    assert diag["sampler_status"] == "available"
    assert diag["sampler_unavailable_reason"] is None
    samples = wrapper.sample_interventional(
        Intervention(target=0, value=1.0),
        n_samples=8,
        sample_seed=0,
    )
    assert samples is not None
    assert samples.shape == (8, 3)


def test_sampler_status_does_not_react_to_simulated_structure_quality() -> None:
    """Mutating the continuous matrix to a poor-structure-but-valid DAG must not flip sampler_status to an unavailable value.

    The cached ``_continuous_w_adj_pre_threshold`` is replaced with
    a known valid-DAG tensor whose values are arbitrary, and the
    cached graph and sampler statuses are re-derived. The expected
    outcome is ``sampler_status == "available"`` because the graph
    is a valid DAG and the sampling API remains callable; structure
    quality is not part of the mechanical availability decision.
    """
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    n = 3
    # Upper-triangular: edges 0->1 and 0->2; valid DAG; values are
    # arbitrary placeholders. The wrapper must not inspect values
    # beyond the >= 0.5 threshold check.
    bad_quality_continuous = torch.tensor(
        [[0.0, 0.9, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=wrapper._continuous_w_adj_pre_threshold.dtype,
    )
    wrapper._continuous_w_adj_pre_threshold = bad_quality_continuous

    a_thresh = (bad_quality_continuous.numpy() >= 0.5).astype(bool)
    from symbolic_priors_cd.wrappers._graph_status import (
        classify_graph_status,
        infer_sampler_status,
    )
    graph_status, graph_reason = classify_graph_status(a_thresh)
    sampler_status, sampler_reason = infer_sampler_status(graph_status)
    wrapper._graph_status = graph_status
    wrapper._graph_invalid_reason = graph_reason
    wrapper._sampler_status = sampler_status
    wrapper._sampler_unavailable_reason = sampler_reason

    assert graph_status == "valid_dag", (
        "test setup error: the chosen continuous matrix did not "
        "threshold to a valid DAG."
    )
    diag = wrapper.get_diagnostics()
    assert diag["sampler_status"] == "available"
    assert diag["sampler_unavailable_reason"] is None


# ---------------------------------------------------------------------------
# sample_interventional argument validation
# ---------------------------------------------------------------------------


def test_sample_interventional_rejects_bool_n_samples() -> None:
    """sample_interventional rejects bool n_samples even though bool is a subclass of int."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    with pytest.raises(ValueError, match="n_samples"):
        wrapper.sample_interventional(
            Intervention(target=0, value=1.0),
            n_samples=True,  # type: ignore[arg-type]
            sample_seed=0,
        )


def test_sample_interventional_rejects_non_int_n_samples() -> None:
    """sample_interventional rejects float n_samples (e.g. 1.5)."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    with pytest.raises(ValueError, match="n_samples"):
        wrapper.sample_interventional(
            Intervention(target=0, value=1.0),
            n_samples=1.5,  # type: ignore[arg-type]
            sample_seed=0,
        )


def test_sample_interventional_rejects_bool_sample_seed() -> None:
    """sample_interventional rejects bool sample_seed even though bool is a subclass of int."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    with pytest.raises(ValueError, match="sample_seed"):
        wrapper.sample_interventional(
            Intervention(target=0, value=1.0),
            n_samples=10,
            sample_seed=True,  # type: ignore[arg-type]
        )


def test_sample_interventional_rejects_string_sample_seed() -> None:
    """sample_interventional rejects string sample_seed."""
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    with pytest.raises(ValueError, match="sample_seed"):
        wrapper.sample_interventional(
            Intervention(target=0, value=1.0),
            n_samples=10,
            sample_seed="0",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Diagnostic tensor cloning
# ---------------------------------------------------------------------------


def test_get_diagnostics_returns_cloned_continuous_tensors() -> None:
    """Diagnostic continuous tensors are detached CPU clones, not references.

    Mutating the tensors returned inside ``model_specific_diagnostics``
    must not affect subsequent calls to ``get_diagnostics`` nor the
    array returned by ``native_edge_continuous``.
    """
    wrapper = _fit_tiny_wrapper(seed=0, n_iter=20)
    diag_first = wrapper.get_diagnostics()
    ms_first = diag_first["model_specific_diagnostics"]
    log_alpha_first = ms_first["continuous_log_alpha_pre_threshold"]
    w_adj_first = ms_first["continuous_w_adj_pre_threshold"]

    assert isinstance(log_alpha_first, torch.Tensor)
    assert isinstance(w_adj_first, torch.Tensor)

    # Mutate the returned tensors in place.
    log_alpha_first.fill_(999.0)
    w_adj_first.fill_(999.0)

    # Subsequent diagnostics must not see the mutation.
    diag_second = wrapper.get_diagnostics()
    ms_second = diag_second["model_specific_diagnostics"]
    log_alpha_second = ms_second["continuous_log_alpha_pre_threshold"]
    w_adj_second = ms_second["continuous_w_adj_pre_threshold"]
    assert not torch.all(log_alpha_second == 999.0)
    assert not torch.all(w_adj_second == 999.0)

    # Each call returns a fresh clone; the two clones are distinct
    # objects with independent storage.
    assert log_alpha_second is not log_alpha_first
    assert w_adj_second is not w_adj_first

    # native_edge_continuous must also be unaffected by the mutation.
    edges = wrapper.native_edge_continuous()
    assert isinstance(edges, np.ndarray)
    assert not np.all(edges == 999.0)
