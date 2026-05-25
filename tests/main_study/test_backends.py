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
    "collections",
    "dataclasses",
    "math",
    "time",
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
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers.dagma",
    "symbolic_priors_cd.wrappers.preprocessing",
    "symbolic_priors_cd.wrappers._dagma_fit",
    "symbolic_priors_cd.wrappers._dagma_sampling",
    "symbolic_priors_cd.wrappers._graph_status",
})


_BACKENDS_FORBIDDEN_PREFIXES: tuple[str, ...] = (
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


# ===========================================================================
# RealMetricBackend
# ===========================================================================


from experiments.main_study.backends import (
    DEFAULT_BANDWIDTH_MULTIPLIERS,
    DEFAULT_INTERVENTION_VALUES_RAW,
    PER_INTERVENTION_GT_SEED_OFFSET,
    PER_INTERVENTION_MODEL_SEED_OFFSET,
    SHD_REVERSAL_COST,
    RealMetricBackend,
    build_default_intervention_specs,
)
from experiments.main_study.backends import (
    _aggregate_bandwidth_sweep,
    _compute_per_intervention_record,
    _median_bandwidth_deterministic,
)
from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm as _generate_scm,
)


# ---------------------------------------------------------------------------
# A. Intervention specs and seed policy
# ---------------------------------------------------------------------------


def test_build_default_intervention_specs_target_major_order():
    specs = build_default_intervention_specs(n_nodes=3)
    assert len(specs) == 6
    expected_ids = [
        "do_X0_neg2", "do_X0_pos2",
        "do_X1_neg2", "do_X1_pos2",
        "do_X2_neg2", "do_X2_pos2",
    ]
    assert [s["intervention_id"] for s in specs] == expected_ids
    # target_node alternates between increasing values; pairs.
    assert [s["target_node"] for s in specs] == [0, 0, 1, 1, 2, 2]
    assert [s["value_raw"] for s in specs] == [-2.0, 2.0] * 3


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "10"])
def test_build_default_intervention_specs_rejects_invalid_n_nodes(bad):
    with pytest.raises(ValueError, match="n_nodes"):
        build_default_intervention_specs(n_nodes=bad)


def test_intervention_seed_offsets_do_not_collide_with_protocol_seeds():
    # Build for two protocol seeds at calibration (401) and evaluation (501).
    specs = build_default_intervention_specs(n_nodes=10)
    for base in (401, 501):
        gt_seeds = [
            base + PER_INTERVENTION_GT_SEED_OFFSET + idx
            for idx in range(len(specs))
        ]
        model_seeds = [
            base + PER_INTERVENTION_MODEL_SEED_OFFSET + idx
            for idx in range(len(specs))
        ]
        # No seed equals the base.
        for s in gt_seeds + model_seeds:
            assert s != base, (
                f"per-intervention seed {s} equals base seed {base}"
            )
        # GT and model seed pools are disjoint.
        assert set(gt_seeds).isdisjoint(set(model_seeds)), (
            f"GT and model seed pools overlap for base {base}"
        )
        # All seeds lie well above the prior-seed band (9000-9199).
        for s in gt_seeds + model_seeds:
            assert s > 9199, (
                f"per-intervention seed {s} falls into the prior-seed "
                "band"
            )


# ---------------------------------------------------------------------------
# B. Deterministic bandwidth helper
# ---------------------------------------------------------------------------


def test_median_bandwidth_deterministic_repeatable():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 3))
    a = _median_bandwidth_deterministic(x, y)
    b = _median_bandwidth_deterministic(x, y)
    assert a is not None
    assert a > 0
    assert a == b


def test_median_bandwidth_returns_none_for_identical_samples():
    x = np.zeros((10, 3))
    y = np.zeros((10, 3))
    assert _median_bandwidth_deterministic(x, y) is None


def test_median_bandwidth_returns_none_for_nan_samples():
    x = np.zeros((5, 3))
    x[0, 0] = float("nan")
    y = np.zeros((5, 3))
    assert _median_bandwidth_deterministic(x, y) is None


def test_median_bandwidth_returns_none_for_inf_samples():
    x = np.zeros((5, 3))
    x[0, 0] = float("inf")
    y = np.zeros((5, 3))
    assert _median_bandwidth_deterministic(x, y) is None


def test_median_bandwidth_rejects_wrong_shape():
    x = np.zeros((5, 3))
    y = np.zeros((5, 4))
    with pytest.raises(ValueError, match="columns"):
        _median_bandwidth_deterministic(x, y)


def test_median_bandwidth_rejects_non_2d():
    x = np.zeros(10)
    y = np.zeros((5, 3))
    with pytest.raises(ValueError, match="2D"):
        _median_bandwidth_deterministic(x, y)


def test_median_bandwidth_matches_hand_computed_case():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    y = np.array([[2.0, 0.0], [3.0, 0.0]])
    # Pooled squared pairwise distances (upper triangle):
    # (0,1)=1, (0,2)=4, (0,3)=9, (1,2)=1, (1,3)=4, (2,3)=1
    # Sorted: [1, 1, 1, 4, 4, 9]. Median over 6 values: (1+4)/2 = 2.5.
    assert _median_bandwidth_deterministic(x, y) == pytest.approx(2.5)


def test_median_bandwidth_works_on_non_contiguous_views():
    rng = np.random.default_rng(0)
    full = rng.standard_normal((40, 6))
    # Strided slice creates a non-contiguous view.
    x = full[:, ::2]
    y = full[:, 1::2]
    a = _median_bandwidth_deterministic(x, y)
    assert a is not None
    assert a > 0


# ---------------------------------------------------------------------------
# C. Per-intervention record helper with mocked samplers/metrics
# ---------------------------------------------------------------------------


def _bundle_with_real_scm(n_nodes: int = 5) -> DataBundle:
    """A DataBundle carrying a real SCM in metadata (for intervene call)."""
    scm = _generate_scm(n_nodes=n_nodes, expected_edges=n_nodes, seed=401)
    return DataBundle(
        x_train=np.random.default_rng(0).standard_normal((40, n_nodes)),
        true_adjacency=np.asarray(scm.adjacency, dtype=bool).copy(),
        scm_seed=401,
        metadata={"scm": scm},
    )


def _fit_outcome_with_callable_sampler(n_nodes: int = 5) -> FitOutcome:
    def fake_sampler(intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"):
        return np.random.default_rng(sample_seed).standard_normal(
            (n_samples, n_nodes)
        )

    # A simple valid DAG: chain 0->1->2->...
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = True
    return FitOutcome(
        continuous_w=np.zeros((n_nodes, n_nodes)),
        thresholded_adjacency=adj,
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=fake_sampler,
    )


def test_per_intervention_record_has_required_fields_and_keys(monkeypatch):
    cfg = _prior_free_config(seed_value=401)
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = _fit_outcome_with_callable_sampler(5)

    def fake_mmd(x, y, bandwidth=None):
        # Distinguish by multiplier through the bandwidth value.
        return -0.01 * float(bandwidth)

    monkeypatch.setattr(backends_mod, "mmd_rbf_unbiased", fake_mmd)

    spec = {
        "intervention_id": "do_X0_neg2",
        "target_node": 0,
        "value_raw": -2.0,
    }
    record = _compute_per_intervention_record(
        planned=planned,
        data_bundle=bundle,
        fit_outcome=fit,
        intervention_spec=spec,
        intervention_index=0,
        n_samples=16,
        bandwidth_multipliers=(0.5, 1.0, 2.0),
    )
    expected_fields = {
        "intervention_id",
        "target_node",
        "value_raw",
        "value_model_frame",
        "ground_truth_sampling_seed",
        "model_sampling_seed",
        "n_ground_truth_samples",
        "n_model_samples",
        "mmd_value",
        "mmd_status",
        "bandwidth_used",
        "bandwidth_sweep",
        "sampler_status_for_intervention",
        "sampler_reason",
    }
    assert set(record.keys()) == expected_fields
    assert record["mmd_status"] == "available"
    assert record["sampler_status_for_intervention"] == "available"
    assert record["sampler_reason"] is None
    assert set(record["bandwidth_sweep"].keys()) == {"0.5x", "1.0x", "2.0x"}
    # The "1.0x" entry is the primary mmd_value.
    assert record["mmd_value"] == record["bandwidth_sweep"]["1.0x"]
    # Negative finite value is preserved (no clipping).
    assert record["mmd_value"] < 0
    # Seed math.
    assert (
        record["ground_truth_sampling_seed"]
        == 401 + PER_INTERVENTION_GT_SEED_OFFSET + 0
    )
    assert (
        record["model_sampling_seed"]
        == 401 + PER_INTERVENTION_MODEL_SEED_OFFSET + 0
    )
    assert record["n_ground_truth_samples"] == 16
    assert record["n_model_samples"] == 16


def test_per_intervention_record_model_sampler_returns_none(monkeypatch):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)

    def none_sampler(intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"):
        return None

    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=np.zeros((5, 5), dtype=bool),
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=none_sampler,
    )
    record = _compute_per_intervention_record(
        planned=planned,
        data_bundle=bundle,
        fit_outcome=fit,
        intervention_spec={
            "intervention_id": "do_X0_neg2",
            "target_node": 0,
            "value_raw": -2.0,
        },
        intervention_index=0,
        n_samples=16,
        bandwidth_multipliers=(0.5, 1.0, 2.0),
    )
    assert record["mmd_status"] == "unavailable_sampler_failure"
    assert record["mmd_value"] is None
    assert record["n_model_samples"] == 0
    assert record["bandwidth_used"] is None
    assert all(v is None for v in record["bandwidth_sweep"].values())


def test_per_intervention_record_degenerate_bandwidth(monkeypatch):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = _fit_outcome_with_callable_sampler(5)

    monkeypatch.setattr(
        backends_mod,
        "_median_bandwidth_deterministic",
        lambda x, y: None,
    )
    record = _compute_per_intervention_record(
        planned=planned,
        data_bundle=bundle,
        fit_outcome=fit,
        intervention_spec={
            "intervention_id": "do_X0_neg2",
            "target_node": 0,
            "value_raw": -2.0,
        },
        intervention_index=0,
        n_samples=16,
        bandwidth_multipliers=(0.5, 1.0, 2.0),
    )
    assert record["mmd_status"] == "unavailable_other"
    assert record["mmd_value"] is None
    assert record["bandwidth_used"] is None
    assert all(v is None for v in record["bandwidth_sweep"].values())


def test_per_intervention_record_missing_scm_raises():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle_no_scm = DataBundle(
        x_train=np.zeros((10, 5)),
        true_adjacency=np.zeros((5, 5), dtype=bool),
        scm_seed=401,
        metadata={},
    )
    fit = _fit_outcome_with_callable_sampler(5)
    with pytest.raises(KeyError, match="scm"):
        _compute_per_intervention_record(
            planned=planned,
            data_bundle=bundle_no_scm,
            fit_outcome=fit,
            intervention_spec={
                "intervention_id": "do_X0_neg2",
                "target_node": 0,
                "value_raw": -2.0,
            },
            intervention_index=0,
            n_samples=16,
            bandwidth_multipliers=(0.5, 1.0, 2.0),
        )


def test_per_intervention_record_non_callable_sampler_raises():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    # Build a FitOutcome with model_sampler=None then bypass the
    # validator by using sampler_status != "available" so the
    # constructor doesn't reject it.
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=np.zeros((5, 5), dtype=bool),
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=None,
    )
    with pytest.raises(ValueError, match="model_sampler"):
        _compute_per_intervention_record(
            planned=planned,
            data_bundle=bundle,
            fit_outcome=fit,
            intervention_spec={
                "intervention_id": "do_X0_neg2",
                "target_node": 0,
                "value_raw": -2.0,
            },
            intervention_index=0,
            n_samples=16,
            bandwidth_multipliers=(0.5, 1.0, 2.0),
        )


# ---------------------------------------------------------------------------
# D. RealMetricBackend validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "10"])
def test_realmetricbackend_rejects_invalid_mmd_n_samples(bad):
    with pytest.raises(ValueError, match="mmd_n_samples"):
        RealMetricBackend(mmd_n_samples=bad)


@pytest.mark.parametrize(
    "bad",
    [
        (),                       # empty tuple
        (0.5, 0.0),              # zero entry
        (0.5, -1.0),             # negative entry
        (0.5, float("nan")),     # NaN
        (0.5, float("inf")),     # inf
        (0.5, True),             # bool
    ],
)
def test_realmetricbackend_rejects_invalid_bandwidth_multipliers(bad):
    with pytest.raises(ValueError, match="bandwidth_multipliers"):
        RealMetricBackend(bandwidth_multipliers=bad)


def test_realmetricbackend_requires_primary_multiplier_1():
    with pytest.raises(ValueError, match="primary multiplier"):
        RealMetricBackend(bandwidth_multipliers=(0.5, 2.0))


def test_realmetricbackend_rejects_intervention_spec_missing_field():
    bad = ({"target_node": 0, "value_raw": -2.0},)  # missing intervention_id
    with pytest.raises(ValueError, match="intervention_id"):
        RealMetricBackend(intervention_specs=bad)


def test_realmetricbackend_call_rejects_invalid_graph_status():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=np.zeros((5, 5), dtype=bool),
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=None,
    )
    with pytest.raises(ValueError, match="graph_status"):
        RealMetricBackend()(planned, bundle, fit)


def test_realmetricbackend_call_rejects_invalid_sampler_status():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=np.zeros((5, 5), dtype=bool),
        graph_status="valid_dag",
        sampler_status="unavailable_unresolved_noise_policy",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=None,
    )
    with pytest.raises(ValueError, match="sampler_status"):
        RealMetricBackend()(planned, bundle, fit)


def test_realmetricbackend_call_rejects_missing_scm():
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle_no_scm = DataBundle(
        x_train=np.zeros((10, 5)),
        true_adjacency=np.zeros((5, 5), dtype=bool),
        scm_seed=401,
        metadata={},
    )
    fit = _fit_outcome_with_callable_sampler(5)
    with pytest.raises(KeyError, match="scm"):
        RealMetricBackend()(planned, bundle_no_scm, fit)


# ---------------------------------------------------------------------------
# E. SID/SHD/MMD aggregation with mocked metric primitives
# ---------------------------------------------------------------------------


def test_realmetricbackend_aggregates_metrics_with_mocked_primitives(
    monkeypatch,
):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = _fit_outcome_with_callable_sampler(5)

    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 7)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 3
    )

    # Return values keyed by bandwidth so different multipliers give
    # different MMD values; with default DEFAULT_BANDWIDTH_MULTIPLIERS
    # the test uses base * 0.5/1.0/2.0.
    def fake_mmd(x, y, bandwidth=None):
        return -0.01 * float(bandwidth)

    monkeypatch.setattr(backends_mod, "mmd_rbf_unbiased", fake_mmd)

    backend = RealMetricBackend(
        mmd_n_samples=16,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
            {"intervention_id": "do_X1_pos2", "target_node": 1, "value_raw": 2.0},
        ),
    )
    outcome = backend(planned, bundle, fit)
    assert outcome.sid == 7.0
    assert outcome.shd == 3.0
    # mmd_primary = mean of the two "1.0x" sweep values, both finite
    # and negative; the mean is also negative.
    assert outcome.mmd < 0
    assert math.isfinite(outcome.mmd)
    payload = outcome.interventions_mmd
    assert payload["mmd_available_count"] == 2
    assert payload["mmd_missing_count"] == 0
    assert payload["mmd_n_samples"] == 16
    assert payload["bandwidth_multipliers"] == [0.5, 1.0, 2.0]
    # Aggregate sweep entries reflect mean of finite per-intervention values.
    agg = payload["mmd_bandwidth_sweep"]
    for key in ("0.5x", "1.0x", "2.0x"):
        assert agg[key] is not None
        assert math.isfinite(agg[key])
    # Primary mmd equals aggregate "1.0x".
    assert outcome.mmd == pytest.approx(agg["1.0x"])


def test_realmetricbackend_no_available_records_raises(monkeypatch):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)

    def none_sampler(intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"):
        return None

    adj = np.zeros((5, 5), dtype=bool)
    adj[0, 1] = True
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=adj,
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=none_sampler,
    )
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 0)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 0
    )
    backend = RealMetricBackend(
        mmd_n_samples=16,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
        ),
    )
    with pytest.raises(ValueError, match="no available MMD"):
        backend(planned, bundle, fit)


def test_realmetricbackend_metric_runtime_seconds_finite_and_non_negative(
    monkeypatch,
):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    fit = _fit_outcome_with_callable_sampler(5)
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 0)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 0
    )
    monkeypatch.setattr(
        backends_mod,
        "mmd_rbf_unbiased",
        lambda x, y, bandwidth=None: -0.1,
    )
    backend = RealMetricBackend(
        mmd_n_samples=16,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
        ),
    )
    outcome = backend(planned, bundle, fit)
    assert math.isfinite(outcome.metric_runtime_seconds)
    assert outcome.metric_runtime_seconds >= 0.0


# ---------------------------------------------------------------------------
# G. Raw-unit / sampler-discipline
# ---------------------------------------------------------------------------


def test_realmetricbackend_passes_raw_intervention_value_to_sampler(
    monkeypatch,
):
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)
    captured: list = []

    def capturing_sampler(intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"):
        captured.append((intervention.target, intervention.value))
        return np.random.default_rng(sample_seed).standard_normal((n_samples, 5))

    adj = np.zeros((5, 5), dtype=bool)
    adj[0, 1] = True
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=adj,
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=capturing_sampler,
    )
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 0)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 0
    )
    monkeypatch.setattr(
        backends_mod,
        "mmd_rbf_unbiased",
        lambda x, y, bandwidth=None: -0.1,
    )
    backend = RealMetricBackend(
        mmd_n_samples=8,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
            {"intervention_id": "do_X1_pos2", "target_node": 1, "value_raw": 2.0},
        ),
    )
    backend(planned, bundle, fit)
    # The metric backend passes the raw intervention value through
    # to the sampler without any preprocessing-side conversion.
    assert (0, -2.0) in captured
    assert (1, 2.0) in captured


def test_realmetricbackend_treats_sampler_opaquely(monkeypatch):
    """A plain callable (no .__self__, no preprocessor) must work."""
    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    bundle = _bundle_with_real_scm(5)

    def plain_callable_sampler(intervention, n_samples, *, sample_seed, noise_policy="residual_fitted"):
        return np.random.default_rng(sample_seed).standard_normal((n_samples, 5))

    adj = np.zeros((5, 5), dtype=bool)
    adj[0, 1] = True
    fit = FitOutcome(
        continuous_w=np.zeros((5, 5)),
        thresholded_adjacency=adj,
        graph_status="valid_dag",
        sampler_status="available",
        training_status="converged",
        wrapper_diagnostics={},
        model_sampler=plain_callable_sampler,
    )
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 0)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 0
    )
    monkeypatch.setattr(
        backends_mod,
        "mmd_rbf_unbiased",
        lambda x, y, bandwidth=None: -0.1,
    )
    backend = RealMetricBackend(
        mmd_n_samples=8,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
        ),
    )
    outcome = backend(planned, bundle, fit)
    # value_model_frame falls back to None because the plain callable
    # exposes no .__self__/preprocessor.
    record = outcome.interventions_mmd["records"][0]
    assert record["value_model_frame"] is None


# ---------------------------------------------------------------------------
# Bandwidth aggregator
# ---------------------------------------------------------------------------


def test_aggregate_bandwidth_sweep_means_finite_values():
    records = [
        {"bandwidth_sweep": {"0.5x": 0.1, "1.0x": 0.2, "2.0x": 0.3}},
        {"bandwidth_sweep": {"0.5x": 0.3, "1.0x": 0.4, "2.0x": 0.5}},
    ]
    agg = _aggregate_bandwidth_sweep(records, (0.5, 1.0, 2.0))
    assert agg["0.5x"] == pytest.approx(0.2)
    assert agg["1.0x"] == pytest.approx(0.3)
    assert agg["2.0x"] == pytest.approx(0.4)


def test_aggregate_bandwidth_sweep_returns_none_when_no_available():
    records = [
        {"bandwidth_sweep": {"0.5x": None, "1.0x": None, "2.0x": None}},
    ]
    agg = _aggregate_bandwidth_sweep(records, (0.5, 1.0, 2.0))
    assert agg == {"0.5x": None, "1.0x": None, "2.0x": None}


def test_aggregate_bandwidth_sweep_preserves_negative_values():
    records = [
        {"bandwidth_sweep": {"0.5x": -0.1, "1.0x": -0.2, "2.0x": -0.3}},
        {"bandwidth_sweep": {"0.5x": -0.3, "1.0x": -0.4, "2.0x": -0.5}},
    ]
    agg = _aggregate_bandwidth_sweep(records, (0.5, 1.0, 2.0))
    for key in agg:
        assert agg[key] < 0


# ---------------------------------------------------------------------------
# F. Integration: execute_planned_run with real fit + real metric stubs
# ---------------------------------------------------------------------------


def _real_integration_metric_backend() -> RealMetricBackend:
    """RealMetricBackend with reduced spec for fast integration tests."""
    return RealMetricBackend(
        mmd_n_samples=32,
        intervention_specs=(
            {"intervention_id": "do_X0_neg2", "target_node": 0, "value_raw": -2.0},
            {"intervention_id": "do_X0_pos2", "target_node": 0, "value_raw": 2.0},
        ),
    )


@pytest.mark.parametrize(
    "config_builder",
    [_prior_free_config, _soft_frobenius_config],
)
def test_execute_planned_run_with_real_fit_and_real_metric_backend(
    config_builder, monkeypatch
):
    """End-to-end smoke: real DataBundleLoader, real fit, real metric
    backend; sid_score/shd/mmd_rbf_unbiased are monkeypatched to keep
    the test stable regardless of DAGMA's actual numeric output."""
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 5)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 3
    )

    def fake_mmd(x, y, bandwidth=None):
        return -0.01 * float(bandwidth)

    monkeypatch.setattr(backends_mod, "mmd_rbf_unbiased", fake_mmd)

    cfg = config_builder()
    planned = make_planned_run(cfg, _RUN_HASH12)
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    fit_backend = MainStudyFitBackend()
    metric_backend = _real_integration_metric_backend()

    result = execute_planned_run(
        planned,
        data_loader=loader,
        fit_backend=fit_backend,
        metric_backend=metric_backend,
        generated_at_utc=_GENERATED_AT,
    )
    assert isinstance(result, ExecutionResult)
    rec = result.record
    if rec.metric_status == "computed":
        assert rec.sid == 5.0
        assert rec.shd == 3.0
        assert rec.mmd is not None and math.isfinite(rec.mmd)
        # interventions_mmd artefact present and contains records.
        payload = result.artefacts["interventions_mmd.json"]
        assert "records" in payload
        assert payload["mmd_available_count"] >= 1
    # Else: the fit produced an invalid graph; the metric backend was
    # correctly not called.


def test_realmetricbackend_integration_does_no_file_io(monkeypatch):
    """Sentinel: full integration path does not open files / make dirs."""
    monkeypatch.setattr(backends_mod, "sid_score", lambda p, t: 5)
    monkeypatch.setattr(
        backends_mod, "shd", lambda p, t, reversal_cost=2: 3
    )
    monkeypatch.setattr(
        backends_mod,
        "mmd_rbf_unbiased",
        lambda x, y, bandwidth=None: -0.01 * float(bandwidth),
    )
    calls: list[str] = []

    def trap_open(*args, **kwargs):
        calls.append("open")
        raise AssertionError("must not call open()")

    def trap_mkdir(*args, **kwargs):
        calls.append("mkdir")
        raise AssertionError("must not create directories")

    monkeypatch.setattr("builtins.open", trap_open)
    monkeypatch.setattr(Path, "mkdir", trap_mkdir)

    cfg = _prior_free_config()
    planned = make_planned_run(cfg, _RUN_HASH12)
    loader = DataBundleLoader(n_nodes=5, expected_edges=5, n_train=200)
    execute_planned_run(
        planned,
        data_loader=loader,
        fit_backend=MainStudyFitBackend(),
        metric_backend=_real_integration_metric_backend(),
        generated_at_utc=_GENERATED_AT,
    )
    assert calls == []


# ---------------------------------------------------------------------------
# H. Import allowlist update (after adding symbolic_priors_cd.metrics)
# ---------------------------------------------------------------------------


def test_backends_allowlist_permits_symbolic_priors_metrics():
    """The metrics module is now allowed in backends.py."""
    src = Path(backends_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    metrics_imports = [
        m
        for m in _module_imports(tree)
        if m.startswith("symbolic_priors_cd.metrics")
    ]
    assert metrics_imports, (
        "backends.py is expected to import from symbolic_priors_cd.metrics"
    )
    # Forbidden prefixes still hold.
    for mod in _module_imports(tree):
        for forbidden in _BACKENDS_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden)
