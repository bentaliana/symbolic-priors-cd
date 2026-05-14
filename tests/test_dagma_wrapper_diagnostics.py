"""Tests for DAGMAWrapper.get_diagnostics().

Covers pre-fit guard, top-level WrapperDiagnostics key contract,
training-status mapping over h_final, n_iterations policy, configured-
budget routing into model_specific_diagnostics, config_snapshot
serialisation, convergence_info content, threshold-count correctness,
defensive copies, invalid-graph and degenerate-sigma branches, and
no-mutation / no-sampler-call guarantees.

All tests use monkeypatched DagmaLinear so no real DAGMA fit is run.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics as _WDiag


# ---------------------------------------------------------------------------
# Helpers: fake DagmaLinear and fitted-wrapper builders
# ---------------------------------------------------------------------------


def _patch_dagma(monkeypatch, W: np.ndarray, h_final: float = 1e-7,
                 score_final: float = -1.0) -> None:
    """Replace DagmaLinear with a fake returning the given W and scalars."""
    W_fixed = W.astype(float).copy()
    h_val = float(h_final)
    s_val = float(score_final)

    class _InjectedFake:
        def __init__(self, loss_type: str = "l2") -> None:
            self.h_final = h_val
            self.score_final = s_val

        def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
            return W_fixed.copy()

    monkeypatch.setattr(
        "symbolic_priors_cd.wrappers._dagma_fit.DagmaLinear",
        _InjectedFake,
    )


def _build_fitted_wrapper(
    monkeypatch,
    W: np.ndarray,
    X: np.ndarray | None = None,
    h_final: float = 1e-7,
    score_final: float = -1.0,
    config: DAGMAConfig | None = None,
    seed: int = 0,
) -> tuple[DAGMAWrapper, np.ndarray, CentredOnlyTransform]:
    """Return a fitted DAGMAWrapper plus the training X and fitted preprocessor."""
    _patch_dagma(monkeypatch, W, h_final=h_final, score_final=score_final)
    d = W.shape[0]
    if X is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, d))
    pre = CentredOnlyTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=seed, config=config)
    return wrapper, X, pre


# ---------------------------------------------------------------------------
# Known W matrices
# ---------------------------------------------------------------------------

_W_VALID = np.array(
    [[0.0, 0.8, 0.0],
     [0.0, 0.0, 0.6],
     [0.0, 0.0, 0.0]],
    dtype=float,
)

# Cyclic 0->1->2->0 above threshold.
_W_CYCLIC = np.array(
    [[0.0, 0.9, 0.0],
     [0.0, 0.0, 0.8],
     [0.7, 0.0, 0.0]],
    dtype=float,
)

# Valid DAG that forces sigma[1] = 0 with the matching X fixture.
_W_DEGEN = np.array(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],
    dtype=float,
)
_N_DEGEN = 6
_X_DEGEN = np.zeros((_N_DEGEN, 3), dtype=float)
_X_DEGEN[:, 0] = np.arange(_N_DEGEN, dtype=float)
_X_DEGEN[:, 1] = np.arange(_N_DEGEN, dtype=float)   # X[:,1] == X[:,0]
_X_DEGEN[:, 2] = np.arange(_N_DEGEN, dtype=float) + 2.0

_EXPECTED_TOP_LEVEL_KEYS = set(_WDiag.__annotations__.keys())

_REQUIRED_MSD_KEYS = {
    "model_name",
    "dagma_source_path",
    "continuous_w_pre_threshold",
    "thresholded_adjacency_project",
    "project_threshold",
    "w_threshold_internal",
    "h_final",
    "score_final",
    "residual_sigma_vector",
    "w_sample",
    "residual_noise_available",
    "unit_variance_available",
    "x_train_model_frame_shape",
    "graph_status",
    "sampler_status",
    "sampler_unavailable_reason",
    "threshold_grid_edge_counts",
    "near_threshold_entry_count",
    "sub_threshold_nonzero_count",
    "iterations_configured_upper_bound",
}


# ---------------------------------------------------------------------------
# 1. Pre-fit guard
# ---------------------------------------------------------------------------


def test_get_diagnostics_raises_before_fit():
    """get_diagnostics on an unfitted wrapper raises RuntimeError."""
    wrapper = DAGMAWrapper()
    with pytest.raises(RuntimeError, match="unfitted"):
        wrapper.get_diagnostics()


# ---------------------------------------------------------------------------
# 2. Top-level key contract
# ---------------------------------------------------------------------------


def test_get_diagnostics_top_level_keys(monkeypatch):
    """The returned record has exactly the WrapperDiagnostics top-level keys."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    diag = wrapper.get_diagnostics()
    assert set(diag.keys()) == _EXPECTED_TOP_LEVEL_KEYS


# ---------------------------------------------------------------------------
# 3. training_status mapping over h_final
# ---------------------------------------------------------------------------


def test_training_status_converged(monkeypatch):
    """h_final <= h_diagnostic_threshold => training_status == 'converged'."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, h_final=1e-8)
    diag = wrapper.get_diagnostics()
    assert diag["training_status"] == "converged"
    assert diag["convergence_info"]["converged"] is True


def test_training_status_max_iter(monkeypatch):
    """Finite h_final above h_diagnostic_threshold => 'max_iter'."""
    cfg = DAGMAConfig()
    wrapper, _, _ = _build_fitted_wrapper(
        monkeypatch, _W_VALID, h_final=cfg.h_diagnostic_threshold * 10.0
    )
    diag = wrapper.get_diagnostics()
    assert diag["training_status"] == "max_iter"
    assert diag["convergence_info"]["converged"] is False


def test_training_status_diverged(monkeypatch):
    """Non-finite h_final => training_status == 'diverged'."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, h_final=np.inf)
    diag = wrapper.get_diagnostics()
    assert diag["training_status"] == "diverged"
    assert diag["convergence_info"]["converged"] is False


def test_training_status_independent_of_graph_status(monkeypatch):
    """A cyclic learned graph must not override training_status; it is set
    from h_final independently of graph_status."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_CYCLIC, h_final=1e-8)
    diag = wrapper.get_diagnostics()
    assert diag["graph_status"] == "cyclic"
    assert diag["training_status"] == "converged"


# ---------------------------------------------------------------------------
# 4. n_iterations is None for DAGMA
# ---------------------------------------------------------------------------


def test_n_iterations_is_none_for_dagma(monkeypatch):
    """DAGMA does not expose actual iterations; top-level n_iterations is None."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    diag = wrapper.get_diagnostics()
    assert diag["n_iterations"] is None


# ---------------------------------------------------------------------------
# 5. Configured budget is not reported as actual n_iterations
# ---------------------------------------------------------------------------


def test_configured_budget_matches_dagma_source_schedule(monkeypatch):
    """iterations_configured_upper_bound equals the source-verified DAGMA
    schedule (T-1)*warm_iter + max_iter, and never appears as top-level
    n_iterations.

    The pinned DAGMA source at external/source_inspection/dagma/src/dagma/
    linear.py drives the path-following loop with tqdm(total=
    (T-1)*warm_iter + max_iter): stages 0..T-2 use warm_iter inner steps
    and stage T-1 uses max_iter inner steps.
    """
    cfg = DAGMAConfig()
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, config=cfg)
    diag = wrapper.get_diagnostics()
    expected_budget = (cfg.T - 1) * cfg.warm_iter + cfg.max_iter
    msd = diag["model_specific_diagnostics"]
    assert msd["iterations_configured_upper_bound"] == expected_budget
    # For DAGMA defaults (T=4, warm_iter=30000, max_iter=60000) this is 150000.
    assert msd["iterations_configured_upper_bound"] == 150_000
    assert diag["n_iterations"] is None
    assert diag["n_iterations"] != expected_budget


def test_iterations_configured_formula_documents_schedule(monkeypatch):
    """iterations_configured_formula is a string explaining the schedule
    and explicitly marks it as a configured budget, not a measured count."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    msd = wrapper.get_diagnostics()["model_specific_diagnostics"]
    formula = msd["iterations_configured_formula"]
    assert isinstance(formula, str)
    assert "(T - 1) * warm_iter + max_iter" in formula
    assert "configured" in formula.lower()
    assert "not" in formula.lower() and "observed" in formula.lower()


# ---------------------------------------------------------------------------
# 6. config_snapshot matches default and custom DAGMAConfig values
# ---------------------------------------------------------------------------


def test_config_snapshot_matches_default_config(monkeypatch):
    """With config=None, config_snapshot matches DAGMAConfig() defaults."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, config=None)
    default = DAGMAConfig()
    snap = wrapper.get_diagnostics()["config_snapshot"]
    assert snap["T"] == default.T
    assert snap["lambda1"] == default.lambda1
    assert snap["s"] == list(default.s)
    assert isinstance(snap["s"], list)  # serialisability: tuple -> list
    assert snap["mu_init"] == default.mu_init
    assert snap["mu_factor"] == default.mu_factor
    assert snap["w_threshold_internal"] == default.w_threshold_internal
    assert snap["lr"] == default.lr
    assert snap["warm_iter"] == default.warm_iter
    assert snap["max_iter"] == default.max_iter
    assert snap["beta_1"] == default.beta_1
    assert snap["beta_2"] == default.beta_2
    assert snap["loss_type"] == default.loss_type
    assert snap["project_threshold"] == default.project_threshold
    assert snap["h_diagnostic_threshold"] == default.h_diagnostic_threshold


def test_config_snapshot_matches_custom_config(monkeypatch):
    """A non-default DAGMAConfig is reflected verbatim in config_snapshot."""
    custom = DAGMAConfig(
        T=2,
        lambda1=0.1,
        s=(1.0, 0.8),
        mu_init=0.5,
        mu_factor=0.05,
        w_threshold_internal=0.0,
        lr=1e-3,
        warm_iter=1000,
        max_iter=2000,
        beta_1=0.9,
        beta_2=0.99,
        loss_type="logistic",
    )
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, config=custom)
    snap = wrapper.get_diagnostics()["config_snapshot"]
    assert snap["T"] == 2
    assert snap["lambda1"] == pytest.approx(0.1)
    assert snap["s"] == [1.0, 0.8]
    assert snap["lr"] == pytest.approx(1e-3)
    assert snap["warm_iter"] == 1000
    assert snap["max_iter"] == 2000
    assert snap["loss_type"] == "logistic"


# ---------------------------------------------------------------------------
# 7. loss_decomposition_final contains h_final and score_final
# ---------------------------------------------------------------------------


def test_loss_decomposition_final_contents(monkeypatch):
    """loss_decomposition_final exposes h_final and score_final."""
    wrapper, _, _ = _build_fitted_wrapper(
        monkeypatch, _W_VALID, h_final=2.5e-7, score_final=-3.14
    )
    diag = wrapper.get_diagnostics()
    ldf = diag["loss_decomposition_final"]
    assert ldf["h_final"] == pytest.approx(2.5e-7)
    assert ldf["score_final"] == pytest.approx(-3.14)


# ---------------------------------------------------------------------------
# 8. convergence_info contents
# ---------------------------------------------------------------------------


def test_convergence_info_contents(monkeypatch):
    """convergence_info contains h_final, threshold, converged, and an
    actual-iteration-unavailable note."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, h_final=1e-7)
    info = wrapper.get_diagnostics()["convergence_info"]
    assert "h_final" in info
    assert "h_diagnostic_threshold" in info
    assert "converged" in info
    assert isinstance(info["converged"], bool)
    # The note must explicitly state that the actual iteration count is unavailable.
    note = str(info.get("actual_iterations_note", ""))
    assert "iteration" in note.lower()
    assert "does not" in note.lower() or "unavailable" in note.lower()


# ---------------------------------------------------------------------------
# 9. thresholded_adjacency equals thresholded_adjacency(config.project_threshold)
# ---------------------------------------------------------------------------


def test_top_level_thresholded_adjacency_matches_helper(monkeypatch):
    """The top-level thresholded_adjacency equals wrapper.thresholded_adjacency
    at config.project_threshold."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    expected = wrapper.thresholded_adjacency(wrapper._config.project_threshold)
    diag = wrapper.get_diagnostics()
    np.testing.assert_array_equal(diag["thresholded_adjacency"], expected)
    assert diag["thresholded_adjacency"].dtype == bool


# ---------------------------------------------------------------------------
# 10. model_specific_diagnostics contains all required DAGMA keys
# ---------------------------------------------------------------------------


def test_model_specific_diagnostics_required_keys(monkeypatch):
    """model_specific_diagnostics contains every required DAGMA-specific key."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    msd = wrapper.get_diagnostics()["model_specific_diagnostics"]
    missing = _REQUIRED_MSD_KEYS - set(msd.keys())
    assert not missing, f"model_specific_diagnostics missing keys: {missing}"
    assert msd["model_name"] == "DAGMA-linear"


# ---------------------------------------------------------------------------
# 11. dagma_source_path points to pinned inspected source
# ---------------------------------------------------------------------------


def test_dagma_source_path_points_to_pinned_source(monkeypatch):
    """dagma_source_path resolves under external/source_inspection/dagma/src."""
    from symbolic_priors_cd.wrappers._dagma_utils import _DAGMA_SRC
    from pathlib import Path

    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    diag = wrapper.get_diagnostics()
    path = Path(diag["model_specific_diagnostics"]["dagma_source_path"])
    assert path.is_relative_to(_DAGMA_SRC)
    assert path.name == "linear.py"


# ---------------------------------------------------------------------------
# 12. Defensive copies of arrays
# ---------------------------------------------------------------------------


def test_arrays_are_defensive_copies(monkeypatch):
    """All numpy arrays in diagnostics are defensive copies of wrapper state."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    diag = wrapper.get_diagnostics()
    msd = diag["model_specific_diagnostics"]

    assert (
        diag["thresholded_adjacency"]
        is not wrapper._continuous_w_pre_threshold
    )
    assert msd["continuous_w_pre_threshold"] is not wrapper._continuous_w_pre_threshold
    assert msd["thresholded_adjacency_project"] is not diag["thresholded_adjacency"]
    assert msd["w_sample"] is not wrapper._w_sample_residual_fitted
    assert msd["residual_sigma_vector"] is not wrapper._sigma_vector_residual_fitted

    # Mutating returned arrays must not affect wrapper state.
    msd["continuous_w_pre_threshold"][0, 1] = 999.0
    msd["w_sample"][0, 1] = 999.0
    msd["residual_sigma_vector"][0] = 999.0
    diag["thresholded_adjacency"][0, 1] = False

    assert wrapper._continuous_w_pre_threshold[0, 1] != 999.0
    assert wrapper._w_sample_residual_fitted[0, 1] != 999.0
    assert wrapper._sigma_vector_residual_fitted[0] != 999.0
    # Re-derived thresholded adjacency must still reflect the true edge.
    assert wrapper.thresholded_adjacency(wrapper._config.project_threshold)[0, 1]


# ---------------------------------------------------------------------------
# 13. Repeated calls return fresh arrays
# ---------------------------------------------------------------------------


def test_repeated_calls_return_fresh_arrays(monkeypatch):
    """Two get_diagnostics calls return distinct numpy objects with equal content."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    d1 = wrapper.get_diagnostics()
    d2 = wrapper.get_diagnostics()

    assert d1["thresholded_adjacency"] is not d2["thresholded_adjacency"]
    np.testing.assert_array_equal(
        d1["thresholded_adjacency"], d2["thresholded_adjacency"]
    )

    a1 = d1["model_specific_diagnostics"]["continuous_w_pre_threshold"]
    a2 = d2["model_specific_diagnostics"]["continuous_w_pre_threshold"]
    assert a1 is not a2
    np.testing.assert_array_equal(a1, a2)

    sigma1 = d1["model_specific_diagnostics"]["residual_sigma_vector"]
    sigma2 = d2["model_specific_diagnostics"]["residual_sigma_vector"]
    assert sigma1 is not sigma2
    np.testing.assert_array_equal(sigma1, sigma2)


# ---------------------------------------------------------------------------
# 14. Invalid graph: continuous W and adjacency present; W_sample and sigma None
# ---------------------------------------------------------------------------


def test_invalid_graph_diagnostics(monkeypatch):
    """Cyclic learned graph: continuous W and adjacency preserved; w_sample
    and residual_sigma_vector are None."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_CYCLIC)
    diag = wrapper.get_diagnostics()
    msd = diag["model_specific_diagnostics"]

    assert diag["graph_status"] == "cyclic"
    assert diag["sampler_status"] == "unavailable_invalid_graph"
    assert diag["sampler_unavailable_reason"] is not None

    np.testing.assert_array_equal(
        msd["continuous_w_pre_threshold"], _W_CYCLIC
    )
    expected_a = wrapper.thresholded_adjacency(wrapper._config.project_threshold)
    np.testing.assert_array_equal(msd["thresholded_adjacency_project"], expected_a)

    assert msd["w_sample"] is None
    assert msd["residual_sigma_vector"] is None
    assert msd["residual_noise_available"] is False
    assert msd["unit_variance_available"] is False


# ---------------------------------------------------------------------------
# 15. Degenerate sigma: sigma vector retained; residual_noise_available False
# ---------------------------------------------------------------------------


def test_degenerate_sigma_diagnostics(monkeypatch):
    """Valid-DAG but degenerate sigma: sigma vector is exposed for debugging
    and residual_noise_available is False."""
    wrapper, _, _ = _build_fitted_wrapper(
        monkeypatch, _W_DEGEN, X=_X_DEGEN.copy()
    )
    diag = wrapper.get_diagnostics()
    msd = diag["model_specific_diagnostics"]

    assert diag["graph_status"] == "valid_dag"
    assert diag["sampler_status"] == "unavailable_unresolved_noise_policy"
    assert msd["residual_sigma_vector"] is not None
    assert msd["residual_sigma_vector"].shape == (3,)
    assert msd["residual_noise_available"] is False


# ---------------------------------------------------------------------------
# 16. unit_variance_available is True for valid graph with W_sample
# ---------------------------------------------------------------------------


def test_unit_variance_available_true_when_residual_fails(monkeypatch):
    """unit_variance_available is True even when residual_noise_available is
    False, provided graph_status is valid_dag and W_sample exists."""
    wrapper, _, _ = _build_fitted_wrapper(
        monkeypatch, _W_DEGEN, X=_X_DEGEN.copy()
    )
    msd = wrapper.get_diagnostics()["model_specific_diagnostics"]
    assert msd["residual_noise_available"] is False
    assert msd["unit_variance_available"] is True
    assert msd["w_sample"] is not None


# ---------------------------------------------------------------------------
# 17. Hand-checkable threshold counts
# ---------------------------------------------------------------------------


def test_threshold_counts_on_hand_constructed_w(monkeypatch):
    """threshold_grid_edge_counts, sub_threshold_nonzero_count, and
    near_threshold_entry_count match a hand-constructed W with known abs values.

    Off-diagonal magnitudes (3x3 layout):
      [_, 0.5,  0.0]
      [0.25, _, 0.35]
      [0.15, 0.0, _ ]
    """
    W = np.zeros((3, 3), dtype=float)
    W[0, 1] = 0.5     # >= 0.4
    W[1, 0] = -0.25   # in [0.2, 0.4), below project_threshold 0.3
    W[1, 2] = 0.35    # >= 0.3, in [0.2, 0.4]
    W[2, 0] = 0.15    # below 0.2, below project_threshold

    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, W)
    msd = wrapper.get_diagnostics()["model_specific_diagnostics"]
    counts = msd["threshold_grid_edge_counts"]
    assert counts["0.2"] == 3   # 0.5, 0.25, 0.35
    assert counts["0.3"] == 2   # 0.5, 0.35
    assert counts["0.4"] == 1   # 0.5
    # 0 < abs(W) < 0.3 on off-diag: 0.25 and 0.15
    assert msd["sub_threshold_nonzero_count"] == 2
    # 0.2 <= abs(W) <= 0.4 on off-diag: 0.25 and 0.35
    assert msd["near_threshold_entry_count"] == 2


# ---------------------------------------------------------------------------
# 18. get_diagnostics does not call the sampler
# ---------------------------------------------------------------------------


def test_get_diagnostics_does_not_call_model_frame_sampler(monkeypatch):
    """get_diagnostics must not invoke sample_linear_gaussian_model_frame."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    with patch(
        "symbolic_priors_cd.wrappers.dagma.sample_linear_gaussian_model_frame"
    ) as mock_sampler:
        wrapper.get_diagnostics()
    mock_sampler.assert_not_called()


# ---------------------------------------------------------------------------
# 19. get_diagnostics does not mutate wrapper state
# ---------------------------------------------------------------------------


def test_get_diagnostics_does_not_mutate_wrapper_state(monkeypatch):
    """Internal arrays and status fields are unchanged after a diagnostics call."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)

    W_before = wrapper._continuous_w_pre_threshold.copy()
    W_samp_before = wrapper._w_sample_residual_fitted.copy()
    sigma_before = wrapper._sigma_vector_residual_fitted.copy()
    X_before = wrapper._X_train_model_frame.copy()
    graph_status_before = wrapper._graph_status
    sampler_status_before = wrapper._sampler_status

    _ = wrapper.get_diagnostics()

    np.testing.assert_array_equal(wrapper._continuous_w_pre_threshold, W_before)
    np.testing.assert_array_equal(wrapper._w_sample_residual_fitted, W_samp_before)
    np.testing.assert_array_equal(wrapper._sigma_vector_residual_fitted, sigma_before)
    np.testing.assert_array_equal(wrapper._X_train_model_frame, X_before)
    assert wrapper._graph_status == graph_status_before
    assert wrapper._sampler_status == sampler_status_before


# ---------------------------------------------------------------------------
# 20. Top-level static fields (smoke test for fixed values)
# ---------------------------------------------------------------------------


def test_top_level_static_field_values(monkeypatch):
    """loss_history, loss_hook_name, numerical_tolerances, and
    mmd_sampling_metadata have the documented static values."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    diag = wrapper.get_diagnostics()
    cfg = wrapper._config

    assert diag["loss_history"] == []
    assert diag["loss_hook_name"] is None
    assert diag["numerical_tolerances"]["h_diagnostic_threshold"] == (
        cfg.h_diagnostic_threshold
    )
    metadata = diag["mmd_sampling_metadata"]
    assert metadata["noise_policy_default"] == "residual_fitted"
    assert set(metadata["supported_noise_policies"]) == {
        "residual_fitted", "unit_variance"
    }


# ---------------------------------------------------------------------------
# Enriched mmd_sampling_metadata
# ---------------------------------------------------------------------------


def test_mmd_sampling_metadata_required_keys(monkeypatch):
    """mmd_sampling_metadata contains every required static-policy key."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    metadata = wrapper.get_diagnostics()["mmd_sampling_metadata"]
    required = {
        "primary_noise_policy",
        "sensitivity_noise_policy",
        "supported_noise_policies",
        "project_threshold",
        "preprocessor_class",
        "residual_fitted_available",
        "unit_variance_available",
    }
    missing = required - set(metadata.keys())
    assert not missing, f"mmd_sampling_metadata missing keys: {missing}"


def test_mmd_sampling_metadata_static_policy_values(monkeypatch):
    """The static policy fields carry the documented values."""
    cfg = DAGMAConfig()
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID, config=cfg)
    metadata = wrapper.get_diagnostics()["mmd_sampling_metadata"]
    assert metadata["primary_noise_policy"] == "residual_fitted"
    assert metadata["sensitivity_noise_policy"] == "unit_variance"
    assert metadata["project_threshold"] == cfg.project_threshold
    assert metadata["preprocessor_class"] == "CentredOnlyTransform"
    assert metadata["residual_fitted_available"] is True
    assert metadata["unit_variance_available"] is True


def test_mmd_sampling_metadata_preprocessor_class_standardised(monkeypatch):
    """preprocessor_class records the actual fitted preprocessor type."""
    from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform

    _patch_dagma(monkeypatch, _W_VALID)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3)) + np.arange(3) * 0.5
    pre = StandardisedTransform().fit(X)
    wrapper = DAGMAWrapper()
    wrapper.fit(X, preprocessor=pre, seed=0)
    metadata = wrapper.get_diagnostics()["mmd_sampling_metadata"]
    assert metadata["preprocessor_class"] == "StandardisedTransform"


def test_mmd_sampling_metadata_availability_under_degenerate_sigma(monkeypatch):
    """When residual sigma is degenerate, residual_fitted_available is False
    but unit_variance_available is True (graph is valid, W_sample exists)."""
    wrapper, _, _ = _build_fitted_wrapper(
        monkeypatch, _W_DEGEN, X=_X_DEGEN.copy()
    )
    metadata = wrapper.get_diagnostics()["mmd_sampling_metadata"]
    assert metadata["residual_fitted_available"] is False
    assert metadata["unit_variance_available"] is True


def test_mmd_sampling_metadata_no_per_call_records(monkeypatch):
    """No sample-seed or per-call record keys leak into mmd_sampling_metadata."""
    wrapper, _, _ = _build_fitted_wrapper(monkeypatch, _W_VALID)
    metadata = wrapper.get_diagnostics()["mmd_sampling_metadata"]
    forbidden = {"sample_seed", "calls", "per_call_records", "sigma_vector_used"}
    leaked = forbidden & set(metadata.keys())
    assert not leaked, f"mmd_sampling_metadata leaked per-call keys: {leaked}"
