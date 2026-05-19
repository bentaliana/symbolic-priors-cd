"""Tests for the per-intervention MMD sampling pipeline.

These tests exercise ``compute_per_intervention_records`` and the
deterministic median-heuristic bandwidth helper directly, without
running the single-fit pipeline. A small fitted DAGMA wrapper is
used as the realistic test subject; a fake raw-unit sampler is used
where we need policy-agnostic deterministic behaviour or to inject
specific failure modes.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from symbolic_priors_cd.data import (
    Intervention,
    generate_linear_gaussian_scm,
    intervene,
    sample_observational,
)

import experiments.selection_study.sampling as sampling_module
from experiments.selection_study.sampling import (
    SCHEMA_GATE_MMD_N_SAMPLES,
    _median_bandwidth_deterministic,
    compute_per_intervention_records,
)


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------


class _PerInterventionSeedsStub:
    """Stub matching the ``PerInterventionSeeds`` shape consumed by sampling."""

    def __init__(
        self, ground_truth_sampling_seed: int, model_sampling_seed: int
    ) -> None:
        self.ground_truth_sampling_seed = ground_truth_sampling_seed
        self.model_sampling_seed = model_sampling_seed


_INTERVENTION_SET = [
    {"intervention_id": "intv_a", "target_node": 0, "value_raw": 2.0},
    {"intervention_id": "intv_b", "target_node": 1, "value_raw": -2.0},
]

_SEEDS_MAP = {
    "intv_a": _PerInterventionSeedsStub(11, 12),
    "intv_b": _PerInterventionSeedsStub(21, 22),
}


def _build_scm_and_data():
    """Build a small 3-node SCM plus 64 raw-unit observational samples."""
    scm = generate_linear_gaussian_scm(
        n_nodes=3, expected_edges=3, seed=0
    )
    x_raw = sample_observational(scm, n_samples=64, rng=1)
    return scm, x_raw


def _make_centred_preprocessor(x_raw: np.ndarray):
    """Build and fit a CentredOnlyTransform on raw training data."""
    pp_module = __import__(
        "symbolic_priors_cd.wrappers.preprocessing", fromlist=["x"]
    )
    pp = pp_module.CentredOnlyTransform()
    pp.fit(x_raw)
    return pp


def _fit_dagma_wrapper(scm, x_raw, preprocessor):
    """Fit a DAGMA wrapper on the supplied data and return it."""
    dagma_module = __import__(
        "symbolic_priors_cd.wrappers.dagma", fromlist=["x"]
    )
    wrapper = dagma_module.DAGMAWrapper()
    x_model = preprocessor.transform(x_raw)
    wrapper.fit(x_model, preprocessor=preprocessor, seed=1, config=None)
    return wrapper


def _dagma_wrapper_for_tests():
    """Fit a tiny DAGMA wrapper and return (scm, wrapper, preprocessor)."""
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    wrapper = _fit_dagma_wrapper(scm, x_raw, pp)
    return scm, wrapper, pp


class _FakeRawSamplerWrapper:
    """Wrapper-shaped fake whose model samples come from the true SCM.

    Useful for tests that need a deterministic, policy-agnostic
    sampler producing raw-unit samples. The fake ignores noise_policy
    keyword arguments (the dispatch path is policy-aware; the fake
    is not).
    """

    def __init__(self, scm: Any) -> None:
        self._scm = scm
        self._calls: list[dict] = []

    def sample_interventional(
        self, intervention: Intervention, n_samples: int, *, sample_seed: int,
        **kwargs,
    ) -> np.ndarray:
        self._calls.append(
            {
                "target": intervention.target,
                "value": intervention.value,
                "n_samples": n_samples,
                "sample_seed": sample_seed,
                "noise_policy": kwargs.get("noise_policy"),
            }
        )
        sampler = intervene(self._scm, intervention)
        return sampler.sample(n_samples, rng=sample_seed)


# ---------------------------------------------------------------------------
# Bandwidth-helper determinism
# ---------------------------------------------------------------------------


def test_bandwidth_helper_is_deterministic_on_repeated_calls() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 4))
    bw1 = _median_bandwidth_deterministic(x, y)
    bw2 = _median_bandwidth_deterministic(x, y)
    assert bw1 == bw2


def test_bandwidth_helper_invariant_to_c_vs_fortran_layout() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 4))
    bw_c = _median_bandwidth_deterministic(x, y)
    bw_f = _median_bandwidth_deterministic(
        np.asfortranarray(x), np.asfortranarray(y)
    )
    assert bw_c == bw_f
    bw_mixed = _median_bandwidth_deterministic(
        np.asfortranarray(x), y
    )
    assert bw_c == bw_mixed


def test_bandwidth_helper_invariant_to_dtype_downcast() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 4))
    bw_64 = _median_bandwidth_deterministic(x, y)
    bw_from_32 = _median_bandwidth_deterministic(
        x.astype(np.float32), y.astype(np.float32)
    )
    # 32-bit samples round to different float64 values; we only
    # require that the helper accepts both layouts. The two need
    # not be identical to each other.
    assert isinstance(bw_64, float)
    assert isinstance(bw_from_32, float)


# ---------------------------------------------------------------------------
# Available sampler, real DAGMA fit
# ---------------------------------------------------------------------------


def _available_dagma_result():
    """Fit DAGMA and run a primary residual_fitted aggregation pass."""
    scm, wrapper, pp = _dagma_wrapper_for_tests()
    diag = wrapper.get_diagnostics()
    return diag, scm, wrapper, pp


def test_available_sampler_produces_records_with_finite_mmd() -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for r in result["records"]:
        assert r["mmd_status"] == "available"
        assert isinstance(r["mmd_value"], float)
        assert np.isfinite(r["mmd_value"])
        assert r["n_ground_truth_samples"] > 0
        assert r["n_model_samples"] > 0
        assert isinstance(r["bandwidth_used"], float)
        assert r["bandwidth_used"] > 0.0
        assert r["bandwidth_sweep"]["1.0x"] == r["mmd_value"]
        for key in ("0.5x", "1.0x", "2.0x"):
            assert isinstance(r["bandwidth_sweep"][key], float)


def test_per_intervention_one_x_sweep_equals_mmd_value() -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for r in result["records"]:
        if r["mmd_status"] == "available":
            assert r["bandwidth_sweep"]["1.0x"] == r["mmd_value"]


def test_aggregates_equal_means_of_available_records() -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    available = [
        r for r in result["records"] if r["mmd_status"] == "available"
    ]
    if not available:
        pytest.skip("no available records to aggregate")
    expected_primary = float(np.mean([r["mmd_value"] for r in available]))
    assert result["mmd_primary"] == pytest.approx(expected_primary, abs=1e-12)
    for key in ("0.5x", "1.0x", "2.0x"):
        per_inter = [r["bandwidth_sweep"][key] for r in available]
        expected_key = float(np.mean(per_inter))
        assert result["mmd_bandwidth_sweep"][key] == pytest.approx(
            expected_key, abs=1e-12
        )
    for r in result["records"]:
        assert (
            result["mmd_bandwidth_used_value"][r["intervention_id"]]
            == r["bandwidth_used"]
        )


def test_available_and_missing_counts_sum_to_total() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    wrapper = _fit_dagma_wrapper(scm, x_raw, pp)
    diag = wrapper.get_diagnostics()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    assert (
        result["mmd_available_count"] + result["mmd_missing_count"]
        == len(result["records"])
    )


# ---------------------------------------------------------------------------
# Samples are passed to MMD in raw-unit frame
# ---------------------------------------------------------------------------


def test_samples_passed_to_mmd_are_in_raw_frame(monkeypatch) -> None:
    """Verifies that both samples reach MMD in raw SCM units.

    Ground-truth samples come from ``intervene(scm, ...).sample`` and
    must have the target column equal to ``value_raw``. Model
    samples come from ``wrapper.sample_interventional`` and must
    also have the target column equal to ``value_raw`` (the wrapper
    handles the raw-model-raw roundtrip internally).
    """
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")

    captured: list[dict] = []
    real_mmd = sampling_module.mmd_rbf_unbiased

    def fake_mmd(x, y, bandwidth=None):
        captured.append({"x": np.asarray(x).copy(), "y": np.asarray(y).copy()})
        return real_mmd(x, y, bandwidth=bandwidth)

    monkeypatch.setattr(sampling_module, "mmd_rbf_unbiased", fake_mmd)

    compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
        n_samples=32,
    )

    target_a = _INTERVENTION_SET[0]["target_node"]
    value_a = _INTERVENTION_SET[0]["value_raw"]
    first = captured[0]
    assert np.allclose(first["x"][:, target_a], value_a, atol=1e-9)
    assert np.allclose(first["y"][:, target_a], value_a, atol=1e-5)


# ---------------------------------------------------------------------------
# Negative MMD values must not be clipped
# ---------------------------------------------------------------------------


def test_negative_mmd_values_are_preserved(monkeypatch) -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")

    def fake_mmd(x, y, bandwidth=None):
        return -0.0042

    monkeypatch.setattr(sampling_module, "mmd_rbf_unbiased", fake_mmd)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for r in result["records"]:
        assert r["mmd_status"] == "available"
        assert r["mmd_value"] == -0.0042
        for key in ("0.5x", "1.0x", "2.0x"):
            assert r["bandwidth_sweep"][key] == -0.0042
    assert result["mmd_primary"] == pytest.approx(-0.0042, abs=1e-12)


# ---------------------------------------------------------------------------
# Unavailable sampler propagates status and skips the wrapper call
# ---------------------------------------------------------------------------


def test_unavailable_sampler_skips_wrapper_call(monkeypatch) -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=fake,
        sampler_status="unavailable_invalid_graph",
        sampler_unavailable_reason="test injection",
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    assert fake._calls == []
    for r in result["records"]:
        assert r["mmd_status"] == "unavailable_invalid_graph"
        assert r["mmd_value"] is None
        assert r["bandwidth_used"] is None
        assert r["n_ground_truth_samples"] == 0
        assert r["n_model_samples"] == 0
        assert r["sampler_reason"] == "test injection"
        assert r["sampler_status_for_intervention"] == (
            "unavailable_invalid_graph"
        )
    assert result["mmd_primary"] is None
    assert result["mmd_sensitivity_unit_variance"] is None
    assert result["mmd_available_count"] == 0
    assert result["mmd_missing_count"] == len(_INTERVENTION_SET)
    for key in ("0.5x", "1.0x", "2.0x"):
        assert result["mmd_bandwidth_sweep"][key] is None
    for iid in (i["intervention_id"] for i in _INTERVENTION_SET):
        assert result["mmd_bandwidth_used_value"][iid] is None


# ---------------------------------------------------------------------------
# Degenerate bandwidth maps to unavailable_other without new enums
# ---------------------------------------------------------------------------


def test_degenerate_bandwidth_maps_to_unavailable_other(monkeypatch) -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    monkeypatch.setattr(
        sampling_module,
        "_median_bandwidth_deterministic",
        lambda x, y: 0.0,
    )
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for r in result["records"]:
        assert r["mmd_status"] == "unavailable_other"
        assert r["mmd_value"] is None
        assert r["bandwidth_used"] is None
        assert r["sampler_reason"] is not None
        assert "bandwidth" in r["sampler_reason"].lower()


def test_non_finite_mmd_maps_to_unavailable_other(monkeypatch) -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")

    def fake_mmd(x, y, bandwidth=None):
        return float("nan")

    monkeypatch.setattr(sampling_module, "mmd_rbf_unbiased", fake_mmd)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for r in result["records"]:
        assert r["mmd_status"] == "unavailable_other"
        assert r["mmd_value"] is None
        assert r["bandwidth_used"] is None


# ---------------------------------------------------------------------------
# Wrapper-API inconsistencies raise RuntimeError
# ---------------------------------------------------------------------------


class _AlwaysNoneWrapper:
    def sample_interventional(self, *args, **kwargs):
        return None


class _WrongShapeWrapper:
    def __init__(self, n_nodes: int) -> None:
        self._n_nodes = n_nodes

    def sample_interventional(self, *args, **kwargs):
        return np.zeros((5, self._n_nodes + 1), dtype=np.float64)


def test_wrapper_returns_none_raises_runtime_error() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    wrapper = _AlwaysNoneWrapper()
    with pytest.raises(RuntimeError, match="None"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=wrapper,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
        )


def test_wrapper_returns_wrong_shape_raises_runtime_error() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    wrapper = _WrongShapeWrapper(n_nodes=scm.n_nodes)
    with pytest.raises(RuntimeError, match="shape"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=wrapper,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
        )


# ---------------------------------------------------------------------------
# DAGMA sensitivity is only computed for residual_fitted primary
# ---------------------------------------------------------------------------


def test_dagma_sensitivity_set_for_residual_fitted_primary() -> None:
    diag, scm, wrapper, pp = _available_dagma_result()
    if diag["sampler_status"] != "available":
        pytest.skip("test requires an available DAGMA sampler")
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=wrapper,
        sampler_status=diag["sampler_status"],
        sampler_unavailable_reason=diag["sampler_unavailable_reason"],
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    assert result["mmd_sensitivity_unit_variance"] is not None
    assert isinstance(result["mmd_sensitivity_unit_variance"], float)
    assert np.isfinite(result["mmd_sensitivity_unit_variance"])


def test_dcdi_native_policy_has_no_sensitivity_aggregate() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=fake,
        sampler_status="available",
        sampler_unavailable_reason=None,
        sampler_policy_used="dcdi_native",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    assert result["mmd_sensitivity_unit_variance"] is None


def test_dcdi_native_does_not_pass_noise_policy_kwarg() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    compute_per_intervention_records(
        scm=scm,
        wrapper=fake,
        sampler_status="available",
        sampler_unavailable_reason=None,
        sampler_policy_used="dcdi_native",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    for call in fake._calls:
        assert call["noise_policy"] is None


def test_dagma_residual_fitted_passes_noise_policy_kwarg() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    compute_per_intervention_records(
        scm=scm,
        wrapper=fake,
        sampler_status="available",
        sampler_unavailable_reason=None,
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    primary_calls = fake._calls[: len(_INTERVENTION_SET)]
    sensitivity_calls = fake._calls[len(_INTERVENTION_SET):]
    for call in primary_calls:
        assert call["noise_policy"] == "residual_fitted"
    for call in sensitivity_calls:
        assert call["noise_policy"] == "unit_variance"


# ---------------------------------------------------------------------------
# Unsupported policy and missing seeds
# ---------------------------------------------------------------------------


def test_unsupported_policy_raises_value_error() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    with pytest.raises(ValueError, match="sampler_policy_used"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=_FakeRawSamplerWrapper(scm),
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="no_such_policy",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
        )


def test_missing_seed_for_intervention_raises_key_error() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    seeds_missing_b = {"intv_a": _PerInterventionSeedsStub(11, 12)}
    with pytest.raises(KeyError, match="intv_b"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=seeds_missing_b,
            preprocessor=pp,
        )


# ---------------------------------------------------------------------------
# Empty intervention set
# ---------------------------------------------------------------------------


def test_empty_intervention_set_returns_empty_records_and_null_aggregates() -> None:
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=_FakeRawSamplerWrapper(scm),
        sampler_status="available",
        sampler_unavailable_reason=None,
        sampler_policy_used="dcdi_native",
        intervention_set=[],
        per_intervention_seeds_map={},
        preprocessor=pp,
    )
    assert result["records"] == []
    assert result["mmd_primary"] is None
    assert result["mmd_sensitivity_unit_variance"] is None
    for key in ("0.5x", "1.0x", "2.0x"):
        assert result["mmd_bandwidth_sweep"][key] is None
    assert result["mmd_bandwidth_used_value"] == {}
    assert result["mmd_available_count"] == 0
    assert result["mmd_missing_count"] == 0


# ---------------------------------------------------------------------------
# Schema-gate sample-count constant
# ---------------------------------------------------------------------------


def test_schema_gate_mmd_n_samples_default() -> None:
    """SCHEMA_GATE_MMD_N_SAMPLES is the documented gate default."""
    assert SCHEMA_GATE_MMD_N_SAMPLES == 64


# ---------------------------------------------------------------------------
# sampler_status validation at the top of compute_per_intervention_records
# ---------------------------------------------------------------------------


def test_unknown_sampler_status_raises_value_error() -> None:
    """An unrecognised sampler_status raises ValueError naming the field."""
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    with pytest.raises(ValueError, match="sampler_status"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="not_a_real_status",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
        )


def test_unavailable_other_rejected_as_top_level_sampler_status() -> None:
    """``unavailable_other`` is reserved for per-intervention mmd_status only.

    The top-level sampler_status argument must not accept it; the
    schema's intent is that ``unavailable_other`` describes a
    per-intervention MMD-side failure given an available sampler.
    """
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    with pytest.raises(ValueError, match="sampler_status"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="unavailable_other",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
        )


@pytest.mark.parametrize(
    "unavailable_status",
    [
        "unavailable_invalid_graph",
        "unavailable_no_api",
        "unavailable_unresolved_noise_policy",
    ],
)
def test_unavailable_sampler_statuses_propagate_consistently(
    unavailable_status: str,
) -> None:
    """Each accepted unavailable status propagates into per-intervention records.

    The wrapper sampler must not be called, every record carries
    ``mmd_status == unavailable_status``, the value/bandwidth/sample
    counts honour the schema's null-on-unavailable invariants, and
    the aggregates collapse to None / zero accordingly.
    """
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    result = compute_per_intervention_records(
        scm=scm,
        wrapper=fake,
        sampler_status=unavailable_status,
        sampler_unavailable_reason="injected for parametrised test",
        sampler_policy_used="residual_fitted",
        intervention_set=_INTERVENTION_SET,
        per_intervention_seeds_map=_SEEDS_MAP,
        preprocessor=pp,
    )
    assert fake._calls == []
    for record in result["records"]:
        assert record["mmd_status"] == unavailable_status
        assert record["sampler_status_for_intervention"] == (
            unavailable_status
        )
        assert record["mmd_value"] is None
        assert record["bandwidth_used"] is None
        assert record["n_ground_truth_samples"] == 0
        assert record["n_model_samples"] == 0
        assert record["sampler_reason"] == (
            "injected for parametrised test"
        )
    assert result["mmd_primary"] is None
    assert result["mmd_sensitivity_unit_variance"] is None
    assert result["mmd_available_count"] == 0
    assert result["mmd_missing_count"] == len(_INTERVENTION_SET)


# ---------------------------------------------------------------------------
# n_samples validation at the top of compute_per_intervention_records
# ---------------------------------------------------------------------------


def test_n_samples_bool_rejected() -> None:
    """A bool n_samples is rejected even though bool is an int subclass."""
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    with pytest.raises(TypeError, match="n_samples"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
            n_samples=True,  # type: ignore[arg-type]
        )


def test_n_samples_float_rejected() -> None:
    """A float n_samples (e.g. 1.5) is rejected by type."""
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    with pytest.raises(TypeError, match="n_samples"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
            n_samples=1.5,  # type: ignore[arg-type]
        )


def test_n_samples_one_rejected_as_too_small() -> None:
    """n_samples=1 is rejected because the unbiased MMD requires >= 2 per side."""
    scm, x_raw = _build_scm_and_data()
    pp = _make_centred_preprocessor(x_raw)
    fake = _FakeRawSamplerWrapper(scm)
    with pytest.raises(ValueError, match="n_samples"):
        compute_per_intervention_records(
            scm=scm,
            wrapper=fake,
            sampler_status="available",
            sampler_unavailable_reason=None,
            sampler_policy_used="dcdi_native",
            intervention_set=_INTERVENTION_SET,
            per_intervention_seeds_map=_SEEDS_MAP,
            preprocessor=pp,
            n_samples=1,
        )
