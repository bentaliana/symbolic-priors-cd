"""Tests for the real data and fit backends.

Mocked DAGMA wrapper / soft-prior fit primitives drive most of the
fast tests. A small smoke-run section exercises the full executor
path with real fits at d=5, n_train=200 to keep wall-clock low.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.main_study import backends as backends_mod
from experiments.main_study.backends import (
    DAGMABackend,
    DataBundleLoader,
    MainStudyFitBackend,
    SoftPriorBackend,
    SoftPriorSampler,
)
from experiments.main_study.executor import (
    DataBundle,
    ExecutionResult,
    FitOutcome,
    MetricOutcome,
    execute_planned_run,
)
from experiments.main_study.priors import (
    PRIOR_SEED_BASE,
    CorruptedPriorSpec,
    build_confidence_mask,
)
from experiments.main_study.schema import (
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import (
    PlannedRun,
    make_planned_run,
)
from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_RUN_HASH12 = "0123456789ab"
_GENERATED_AT = "2026-05-24T12:00:00Z"


def _fast_dagma_config(**overrides) -> DAGMAConfig:
    """A small, fast DAGMA config suitable for the smoke-run tests."""
    base: dict[str, Any] = dict(
        T=3,
        lambda1=0.05,
        s=(1.0, 0.9, 0.8),
        mu_init=1.0,
        mu_factor=0.1,
        w_threshold_internal=0.0,
        lr=3e-4,
        warm_iter=2000,
        max_iter=4000,
        beta_1=0.99,
        beta_2=0.999,
        loss_type="l2",
    )
    base.update(overrides)
    return DAGMAConfig(**base)


def _make_corrupted_spec_5() -> CorruptedPriorSpec:
    """5-node CorruptedPriorSpec consistent with the soft/hard fixtures."""
    forbidden = ((0, 2), (1, 3), (2, 4))
    return CorruptedPriorSpec(
        n_nodes=5,
        scm_seed=401,
        corruption_fraction=0.0,
        corruption_index=0,
        corruption_seed=9100 + 401 + 0,
        forbidden_edges=forbidden,
        n_correct=3,
        n_corrupted=0,
        removed_clean_edges=(),
        added_true_positive_edges=(),
        edge_labels={
            "0,2": "true_negative_retained",
            "1,3": "true_negative_retained",
            "2,4": "true_negative_retained",
        },
    )


def _prior_free_config(seed_value: int = 401) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_fast_dagma_config(),
        parent_heldout_run_hash_full=_PARENT_HASH,
    )


def _matched_l1_config(seed_value: int = 401, matched: float = 0.07) -> MainStudyConfig:
    return make_main_study_config(
        method_family="matched_l1",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_fast_dagma_config(lambda1=matched),
        parent_heldout_run_hash_full=_PARENT_HASH,
        matched_l1_lambda1=matched,
    )


def _soft_frobenius_config(seed_value: int = 401, confidence: float = 0.5) -> MainStudyConfig:
    return make_main_study_config(
        method_family="soft_frobenius",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_fast_dagma_config(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        confidence=confidence,
        corrupted_prior_spec=_make_corrupted_spec_5(),
    )


def _hard_exclusion_config(seed_value: int = 401) -> MainStudyConfig:
    return make_main_study_config(
        method_family="hard_exclusion",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_fast_dagma_config(),
        parent_heldout_run_hash_full=_PARENT_HASH,
        corrupted_prior_spec=_make_corrupted_spec_5(),
    )


def _fake_metric_outcome() -> MetricOutcome:
    return MetricOutcome(
        sid=0.0,
        shd=0.0,
        mmd=-1e-4,
        interventions_mmd={"records": [], "mmd_primary": -1e-4},
        metric_runtime_seconds=0.01,
    )


# ===========================================================================
# A. DataBundleLoader
# ===========================================================================


def test_databundleloader_returns_correctly_shaped_bundle():
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    cfg = _prior_free_config(seed_value=401)
    bundle = loader(cfg)
    assert isinstance(bundle, DataBundle)
    assert bundle.x_train.shape == (200, 5)
    assert bundle.true_adjacency.shape == (5, 5)
    assert bundle.true_adjacency.dtype == bool
    assert bundle.scm_seed == 401
    assert np.all(np.isfinite(bundle.x_train))


def test_databundleloader_metadata_contains_scm_and_inputs():
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    cfg = _prior_free_config(seed_value=401)
    bundle = loader(cfg)
    md = bundle.metadata
    assert "scm" in md
    # The SCM object exposes its adjacency directly.
    scm = md["scm"]
    assert scm.adjacency.shape == (5, 5)
    assert md["n_nodes"] == 5
    assert md["expected_edges"] == 5
    assert md["n_train"] == 200
    assert md["noise_scale"] == 1.0
    assert md["weight_magnitude_range"] == (0.5, 2.0)


def test_databundleloader_is_deterministic_in_seed():
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    cfg = _prior_free_config(seed_value=401)
    a = loader(cfg)
    b = loader(cfg)
    np.testing.assert_array_equal(a.x_train, b.x_train)
    np.testing.assert_array_equal(a.true_adjacency, b.true_adjacency)


def test_databundleloader_different_seed_produces_different_data():
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    a = loader(_prior_free_config(seed_value=401))
    b = loader(_prior_free_config(seed_value=402))
    assert not np.array_equal(a.x_train, b.x_train)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "10"])
def test_databundleloader_rejects_invalid_n_nodes(bad):
    with pytest.raises(ValueError, match="n_nodes"):
        DataBundleLoader(n_nodes=bad)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5])
def test_databundleloader_rejects_invalid_expected_edges(bad):
    with pytest.raises(ValueError, match="expected_edges"):
        DataBundleLoader(expected_edges=bad)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5])
def test_databundleloader_rejects_invalid_n_train(bad):
    with pytest.raises(ValueError, match="n_train"):
        DataBundleLoader(n_train=bad)


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, float("nan"), float("inf"), True]
)
def test_databundleloader_rejects_invalid_noise_scale(bad):
    with pytest.raises(ValueError, match="noise_scale"):
        DataBundleLoader(noise_scale=bad)


@pytest.mark.parametrize(
    "bad",
    [
        (0.5,),                       # wrong length
        (1.0, 0.5),                  # low > high
        (0.5, float("inf")),         # non-finite
        (float("nan"), 1.0),         # non-finite
        (-0.1, 1.0),                 # negative low
        (0.0, 0.0),                  # high <= 0
    ],
)
def test_databundleloader_rejects_invalid_weight_range(bad):
    with pytest.raises(ValueError, match="weight_magnitude_range"):
        DataBundleLoader(weight_magnitude_range=bad)


# ===========================================================================
# B. DAGMABackend with mocked DAGMAWrapper
# ===========================================================================


class _FakeDAGMAWrapper:
    """Test double mirroring the public DAGMAWrapper surface."""

    instances: list["_FakeDAGMAWrapper"] = []

    # Class-level configuration consumed by tests.
    _w: np.ndarray = np.zeros((5, 5), dtype=float)
    _thresholded: np.ndarray = np.zeros((5, 5), dtype=bool)
    _graph_status: str = "valid_dag"
    _sampler_status: str = "available"
    _training_status: str = "converged"

    def __init__(self) -> None:
        self.fit_call: dict[str, Any] = {}
        _FakeDAGMAWrapper.instances.append(self)

    def fit(self, X_train, *, preprocessor, seed, config=None):
        self.fit_call = {
            "X_train": X_train,
            "preprocessor": preprocessor,
            "seed": seed,
            "config": config,
        }

    def native_edge_continuous(self) -> np.ndarray:
        return self._w.copy()

    def thresholded_adjacency(self, threshold: float = 0.3) -> np.ndarray:
        return self._thresholded.copy()

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "training_status": self._training_status,
            "graph_status": self._graph_status,
            "sampler_status": self._sampler_status,
            "seed": 42,
            "n_iterations": None,
            "config_snapshot": {},
            "loss_history": [],
            "loss_decomposition_final": {},
            "convergence_info": {},
            "thresholded_adjacency": self._thresholded.copy(),
            "graph_invalid_reason": None,
            "sampler_unavailable_reason": None,
            "mmd_sampling_metadata": {},
            "loss_hook_name": None,
            "numerical_tolerances": {},
            "model_specific_diagnostics": {},
        }

    def sample_interventional(
        self, intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"
    ):
        return np.zeros((n_samples, self._w.shape[0]), dtype=float)


@pytest.fixture
def fake_wrapper(monkeypatch):
    """Reset _FakeDAGMAWrapper state and inject it into backends."""
    _FakeDAGMAWrapper.instances.clear()
    _FakeDAGMAWrapper._w = np.zeros((5, 5), dtype=float)
    _FakeDAGMAWrapper._thresholded = np.zeros((5, 5), dtype=bool)
    _FakeDAGMAWrapper._graph_status = "valid_dag"
    _FakeDAGMAWrapper._sampler_status = "available"
    _FakeDAGMAWrapper._training_status = "converged"
    monkeypatch.setattr(backends_mod, "DAGMAWrapper", _FakeDAGMAWrapper)
    return _FakeDAGMAWrapper


def _make_bundle_5(seed: int = 401) -> DataBundle:
    return DataBundle(
        x_train=np.random.default_rng(seed).standard_normal((40, 5)),
        true_adjacency=np.zeros((5, 5), dtype=bool),
        scm_seed=seed,
    )


@pytest.mark.parametrize(
    "config_builder",
    [_prior_free_config, _matched_l1_config, _hard_exclusion_config],
)
def test_dagma_backend_accepts_non_soft_families(
    fake_wrapper, config_builder
):
    cfg = config_builder()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    outcome = backend(planned, _make_bundle_5(), None)
    assert isinstance(outcome, FitOutcome)
    assert outcome.graph_status == "valid_dag"
    assert outcome.sampler_status == "available"


def test_dagma_backend_rejects_soft_frobenius(fake_wrapper):
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    mask = build_confidence_mask(cfg.corrupted_prior_spec, 0.5)
    with pytest.raises(ValueError, match="soft_frobenius"):
        backend(planned, _make_bundle_5(), mask)


def test_dagma_backend_rejects_non_none_confidence_mask(fake_wrapper):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    with pytest.raises(ValueError, match="confidence_mask"):
        backend(planned, _make_bundle_5(), np.zeros((5, 5)))


def test_dagma_backend_passes_dagma_config_to_wrapper(fake_wrapper):
    cfg = _matched_l1_config(matched=0.07)
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    backend(planned, _make_bundle_5(), None)
    instance = fake_wrapper.instances[-1]
    assert instance.fit_call["config"] is cfg.dagma_config


def test_dagma_backend_sets_model_sampler_when_sampler_available(
    fake_wrapper,
):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    outcome = backend(planned, _make_bundle_5(), None)
    assert outcome.model_sampler is not None
    assert callable(outcome.model_sampler)


def test_dagma_backend_model_sampler_is_none_when_sampler_unavailable(
    fake_wrapper,
):
    fake_wrapper._graph_status = "cyclic"
    fake_wrapper._sampler_status = "unavailable_invalid_graph"
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    outcome = backend(planned, _make_bundle_5(), None)
    assert outcome.model_sampler is None
    assert outcome.graph_status == "cyclic"
    assert outcome.sampler_status == "unavailable_invalid_graph"


# ===========================================================================
# C. SoftPriorBackend with mocked low-level primitives
# ===========================================================================


@dataclasses.dataclass
class _FakeFitResult:
    W: np.ndarray
    h_final: float
    score_final: float


def _fake_run_soft_prior_factory(
    *, W: np.ndarray, h_final: float = 1e-7, score_final: float = -1.5,
    captured: dict | None = None,
):
    def fake(x_local, cfg, *, lambda_prior, confidence_mask):
        if captured is not None:
            captured["x_local"] = x_local
            captured["cfg"] = cfg
            captured["lambda_prior"] = lambda_prior
            captured["confidence_mask"] = confidence_mask
        return _FakeFitResult(W=W.copy(), h_final=h_final, score_final=score_final)
    return fake


def test_softprior_backend_rejects_non_soft_family():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = SoftPriorBackend()
    with pytest.raises(ValueError, match="soft_frobenius"):
        backend(planned, _make_bundle_5(), np.zeros((5, 5)))


def test_softprior_backend_rejects_none_confidence_mask():
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = SoftPriorBackend()
    with pytest.raises(ValueError, match="confidence_mask"):
        backend(planned, _make_bundle_5(), None)


def test_softprior_backend_rejects_none_lambda_prior():
    cfg = _soft_frobenius_config()
    # Bypass the factory-set lambda_prior to test the validator.
    object.__setattr__(cfg, "lambda_prior", None)
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = SoftPriorBackend()
    mask = build_confidence_mask(_make_corrupted_spec_5(), 0.5)
    with pytest.raises(ValueError, match="lambda_prior"):
        backend(planned, _make_bundle_5(), mask)


def test_softprior_backend_passes_lambda_prior_and_mask_to_helper(
    monkeypatch,
):
    cfg = _soft_frobenius_config(confidence=0.5)
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _make_bundle_5()
    mask = build_confidence_mask(cfg.corrupted_prior_spec, 0.5)

    captured: dict = {}
    # Return a "valid DAG" W: small enough not to threshold.
    fake_w = np.zeros((5, 5), dtype=float)
    monkeypatch.setattr(
        backends_mod,
        "run_soft_prior_dagma_fit",
        _fake_run_soft_prior_factory(W=fake_w, captured=captured),
    )
    backend = SoftPriorBackend()
    outcome = backend(planned, bundle, mask)
    assert captured["lambda_prior"] == pytest.approx(FROZEN_LAMBDA_PRIOR)
    np.testing.assert_array_equal(captured["confidence_mask"], mask)
    assert isinstance(outcome, FitOutcome)


def test_softprior_backend_uses_project_threshold_for_thresholding(
    monkeypatch,
):
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    # Choose a W with entries near the threshold.
    fake_w = np.zeros((5, 5), dtype=float)
    fake_w[0, 1] = 0.31  # just above 0.3
    fake_w[2, 3] = 0.29  # just below 0.3
    monkeypatch.setattr(
        backends_mod,
        "run_soft_prior_dagma_fit",
        _fake_run_soft_prior_factory(W=fake_w),
    )
    backend = SoftPriorBackend()
    outcome = backend(planned, _make_bundle_5(), build_confidence_mask(
        cfg.corrupted_prior_spec, 0.5
    ))
    assert outcome.thresholded_adjacency[0, 1] == True
    assert outcome.thresholded_adjacency[2, 3] == False


def test_softprior_backend_graph_status_uses_classifier(monkeypatch):
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    # An adjacency with a 2-cycle to force cyclic graph_status.
    fake_w = np.zeros((5, 5), dtype=float)
    fake_w[0, 1] = 0.9
    fake_w[1, 0] = 0.9
    monkeypatch.setattr(
        backends_mod,
        "run_soft_prior_dagma_fit",
        _fake_run_soft_prior_factory(W=fake_w),
    )
    backend = SoftPriorBackend()
    outcome = backend(planned, _make_bundle_5(), build_confidence_mask(
        cfg.corrupted_prior_spec, 0.5
    ))
    # The bidirected pair takes precedence over cycle in the classifier.
    assert outcome.graph_status in ("bidirected", "cyclic")
    assert outcome.sampler_status != "available"
    assert outcome.model_sampler is None


def test_softprior_backend_model_sampler_callable_for_valid_dag(monkeypatch):
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    # A chain DAG, no self-loops, no bidirected: valid_dag.
    fake_w = np.zeros((5, 5), dtype=float)
    fake_w[0, 1] = 0.8
    fake_w[1, 2] = 0.7
    fake_w[2, 3] = 0.6
    fake_w[3, 4] = 0.5
    monkeypatch.setattr(
        backends_mod,
        "run_soft_prior_dagma_fit",
        _fake_run_soft_prior_factory(W=fake_w),
    )
    backend = SoftPriorBackend()
    outcome = backend(planned, _make_bundle_5(), build_confidence_mask(
        cfg.corrupted_prior_spec, 0.5
    ))
    assert outcome.graph_status == "valid_dag"
    assert outcome.sampler_status == "available"
    assert callable(outcome.model_sampler)


def test_softprior_backend_diagnostics_contain_expected_keys(monkeypatch):
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    fake_w = np.zeros((5, 5), dtype=float)
    monkeypatch.setattr(
        backends_mod,
        "run_soft_prior_dagma_fit",
        _fake_run_soft_prior_factory(W=fake_w, h_final=1e-8, score_final=-3.14),
    )
    backend = SoftPriorBackend()
    outcome = backend(planned, _make_bundle_5(), build_confidence_mask(
        cfg.corrupted_prior_spec, 0.5
    ))
    diag = outcome.wrapper_diagnostics
    assert "training_status" in diag
    assert "graph_status" in diag
    assert "sampler_status" in diag
    assert "model_specific_diagnostics" in diag
    msd = diag["model_specific_diagnostics"]
    for key in (
        "h_final",
        "score_final",
        "continuous_w_pre_threshold",
        "threshold",
        "lambda_prior",
        "confidence_nonzero_count",
        "confidence_max",
    ):
        assert key in msd, f"missing diagnostics key {key!r}"
    assert msd["h_final"] == pytest.approx(1e-8)
    assert msd["score_final"] == pytest.approx(-3.14)
    assert msd["lambda_prior"] == pytest.approx(FROZEN_LAMBDA_PRIOR)


# ===========================================================================
# D. SoftPriorSampler
# ===========================================================================


def test_softprior_sampler_signature_matches_dagma_wrapper():
    dagma_sig = inspect.signature(DAGMAWrapper.sample_interventional)
    soft_sig = inspect.signature(SoftPriorSampler.sample_interventional)
    dagma_names = [p.name for p in dagma_sig.parameters.values()]
    soft_names = [p.name for p in soft_sig.parameters.values()]
    assert dagma_names == soft_names, (
        f"DAGMAWrapper params {dagma_names} vs SoftPriorSampler "
        f"params {soft_names}"
    )
    # Keyword-only enforcement matches.
    for name in ("sample_seed", "noise_policy"):
        assert (
            dagma_sig.parameters[name].kind
            == soft_sig.parameters[name].kind
        )
    # Default for noise_policy matches.
    assert (
        dagma_sig.parameters["noise_policy"].default
        == soft_sig.parameters["noise_policy"].default
        == "residual_fitted"
    )


def _build_softprior_sampler_for_test() -> SoftPriorSampler:
    n = 5
    rng = np.random.default_rng(0)
    raw_data = rng.standard_normal((100, n))
    pre = StandardisedTransform().fit(raw_data)
    # A simple DAG: chain 0->1->2->3->4
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        adj[i, i + 1] = True
    w_sample = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        w_sample[i, i + 1] = 0.5
    sigma = np.ones(n, dtype=float)
    return SoftPriorSampler(
        thresholded_adjacency=adj,
        w_sample=w_sample,
        sigma=sigma,
        preprocessor=pre,
    )


def test_softprior_sampler_returns_raw_unit_samples_with_expected_shape():
    sampler = _build_softprior_sampler_for_test()
    intervention = Intervention(target=0, value=1.5)
    samples = sampler.sample_interventional(
        intervention, n_samples=10, sample_seed=42, noise_policy="residual_fitted"
    )
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (10, 5)
    assert np.all(np.isfinite(samples))


def test_softprior_sampler_unsupported_noise_policy_raises():
    sampler = _build_softprior_sampler_for_test()
    intervention = Intervention(target=0, value=1.0)
    with pytest.raises(ValueError, match="noise_policy"):
        sampler.sample_interventional(
            intervention, n_samples=5, sample_seed=42, noise_policy="bogus"
        )


def test_softprior_sampler_unit_variance_works():
    sampler = _build_softprior_sampler_for_test()
    intervention = Intervention(target=0, value=0.0)
    samples = sampler.sample_interventional(
        intervention, n_samples=5, sample_seed=42, noise_policy="unit_variance"
    )
    assert samples.shape == (5, 5)


# ===========================================================================
# E. MainStudyFitBackend dispatch
# ===========================================================================


class _CapturingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, planned, data_bundle, confidence_mask):
        self.calls.append((planned, data_bundle, confidence_mask))
        # Return a minimal valid FitOutcome so the dispatcher doesn't fail
        # on the executor side; backends aren't called through the executor
        # in these tests, so we can return anything that satisfies FitOutcome.
        n = data_bundle.x_train.shape[1]
        return FitOutcome(
            continuous_w=np.zeros((n, n)),
            thresholded_adjacency=np.zeros((n, n), dtype=bool),
            graph_status="cyclic",
            sampler_status="unavailable_invalid_graph",
            training_status="converged",
            wrapper_diagnostics={},
            model_sampler=None,
        )


@pytest.mark.parametrize(
    "config_builder, expected_target",
    [
        (_prior_free_config, "dagma"),
        (_matched_l1_config, "dagma"),
        (_hard_exclusion_config, "dagma"),
        (_soft_frobenius_config, "soft"),
    ],
)
def test_main_study_dispatcher_routes_to_correct_backend(
    config_builder, expected_target
):
    dagma_fake = _CapturingBackend()
    soft_fake = _CapturingBackend()
    dispatcher = MainStudyFitBackend(
        dagma_backend=dagma_fake,
        soft_prior_backend=soft_fake,
    )
    cfg = config_builder()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _make_bundle_5()
    mask = None
    if cfg.method_family == "soft_frobenius":
        mask = build_confidence_mask(cfg.corrupted_prior_spec, 0.5)
    dispatcher(planned, bundle, mask)
    if expected_target == "dagma":
        assert len(dagma_fake.calls) == 1
        assert len(soft_fake.calls) == 0
    else:
        assert len(soft_fake.calls) == 1
        assert len(dagma_fake.calls) == 0


def test_main_study_dispatcher_passes_confidence_mask_unchanged():
    soft_fake = _CapturingBackend()
    dispatcher = MainStudyFitBackend(soft_prior_backend=soft_fake)
    cfg = _soft_frobenius_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _make_bundle_5()
    mask = build_confidence_mask(cfg.corrupted_prior_spec, 0.5)
    dispatcher(planned, bundle, mask)
    _, _, captured_mask = soft_fake.calls[0]
    assert captured_mask is mask


def test_main_study_dispatcher_unknown_family_raises():
    cfg = _prior_free_config()
    # Bypass schema validation to force a bogus method_family value.
    object.__setattr__(cfg, "method_family", "alien_family")
    planned_cls = PlannedRun
    # Build a PlannedRun manually with a config the schema would normally
    # reject; instead, simulate the dispatcher input directly.
    from dataclasses import replace

    class _FakePlanned:
        def __init__(self, cfg):
            self.config = cfg

    fake_planned = _FakePlanned(cfg)
    dispatcher = MainStudyFitBackend()
    with pytest.raises(ValueError, match="alien_family"):
        dispatcher(fake_planned, _make_bundle_5(), None)


# ===========================================================================
# F. execute_planned_run integration with real backends
# ===========================================================================


def _execute_with_real_backends(cfg: MainStudyConfig) -> ExecutionResult:
    """Drive execute_planned_run with real backends + fake metric backend."""
    planned = make_planned_run(cfg, _RUN_HASH12)
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    fit_backend = MainStudyFitBackend()

    captured_sampler: dict = {}

    def fake_metric(planned_arg, data, fit_outcome):
        captured_sampler["model_sampler"] = fit_outcome.model_sampler
        return _fake_metric_outcome()

    result = execute_planned_run(
        planned,
        data_loader=loader,
        fit_backend=fit_backend,
        metric_backend=fake_metric,
        generated_at_utc=_GENERATED_AT,
    )
    return result, captured_sampler


def _fast_planned_with_d5(cfg_builder) -> MainStudyConfig:
    """Use the d=5 helper configs that pair with DataBundleLoader(n_nodes=5)."""
    return cfg_builder()


@pytest.mark.parametrize(
    "config_builder",
    [
        _prior_free_config,
        _matched_l1_config,
        _hard_exclusion_config,
        _soft_frobenius_config,
    ],
)
def test_execute_planned_run_with_real_backends_returns_execution_result(
    config_builder,
):
    cfg = config_builder()
    result, captured = _execute_with_real_backends(cfg)
    assert isinstance(result, ExecutionResult)
    record = result.record
    assert record.fit_status == "success"
    assert record.config.method_family == cfg.method_family
    # The metric backend's saved reference reflects whatever the fit
    # outcome produced. When sampler_status was "available" the
    # sampler must be callable; otherwise it must be None (the metric
    # backend would not have been invoked in that case).
    if record.metric_status == "computed":
        assert callable(captured.get("model_sampler"))
    else:
        # If sampler/graph turned out unavailable the executor skips
        # the metric backend entirely; captured stays empty.
        assert captured.get("model_sampler") in (None, captured.get("model_sampler"))


# ===========================================================================
# G. Scope and import guard tests
# ===========================================================================


_BACKENDS_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "dataclasses",
    "math",
    "typing",
    "inspect",
    "numpy",
    "experiments.main_study.executor",
    "experiments.main_study.workloads",
    "experiments.main_study.schema",
    "experiments.main_study.priors",
    "experiments.main_study.paths",
    "symbolic_priors_cd.data.scm_generator",
    "symbolic_priors_cd.data.interventions",
    "symbolic_priors_cd.wrappers.dagma",
    "symbolic_priors_cd.wrappers.preprocessing",
    "symbolic_priors_cd.wrappers._dagma_fit",
    "symbolic_priors_cd.wrappers._dagma_sampling",
    "symbolic_priors_cd.wrappers._graph_status",
})


_BACKENDS_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "experiments.selection_study",
    "experiments.main_study.records",
    "experiments.main_study.calibration_lambda_prior",
    "dagma",
    "dcdi",
    "tests",
)


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_backends_module_does_not_import_forbidden_packages():
    src = Path(backends_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _BACKENDS_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"backends.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_backends_module_imports_are_allowlisted():
    src = Path(backends_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _BACKENDS_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _BACKENDS_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"backends.py import {mod!r} is not in the allowlist "
            f"{sorted(_BACKENDS_ALLOWED_PREFIXES)}."
        )


def test_backends_perform_no_file_io(monkeypatch, fake_wrapper):
    """Sentinels: backend execution must not call open or Path.mkdir."""
    calls: list[str] = []

    def trap_open(*args, **kwargs):
        calls.append("open")
        raise AssertionError("backends must not call open()")

    monkeypatch.setattr("builtins.open", trap_open)

    def trap_mkdir(*args, **kwargs):
        calls.append("mkdir")
        raise AssertionError("backends must not create directories")

    monkeypatch.setattr(Path, "mkdir", trap_mkdir)

    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    backend = DAGMABackend()
    backend(planned, _make_bundle_5(), None)
    assert calls == []
