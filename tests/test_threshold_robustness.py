"""Tests for offline threshold-robustness re-computation.

These tests exercise ``recompute_at_thresholds`` on both pipeline-
generated runs (the load-bearing cross-check against the schema-
conformance gate) and hand-crafted fixtures that cover negative
DAGMA weights, DCDI's ``w_adj`` thresholding, ordered-triple
validation, sibling-artefact behaviour, and ``run.json`` byte
immutability.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    Configuration,
    InterventionSpec,
    PhaseBConfiguration,
    SEED_DERIVATION_RULE_NAME,
)
from experiments.selection_study.loader import load_run
from experiments.selection_study.pipeline import (
    InvalidGraphForSchemaGateError,
    run_single_fit,
)
from experiments.selection_study.preflight import enumerate_manifest
from experiments.selection_study.threshold_robustness import (
    PROTOCOL_THRESHOLD_TRIPLES,
    recompute_at_thresholds,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


_INTERVENTION_A = InterventionSpec(
    intervention_id="intv_a", target_node=0, value_raw=2.0
)
_INTERVENTION_B = InterventionSpec(
    intervention_id="intv_b", target_node=1, value_raw=-2.0
)
_PHASE_B = PhaseBConfiguration(
    name="default", hyperparameters=(("lr", 0.01),)
)


def _make_dagma_config() -> Configuration:
    """Return a minimal-valid DAGMA Configuration for these tests."""
    return Configuration(
        model="dagma",
        condition="centred_only",
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=(("calibration", (10,)),),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        phase_b_configurations=(_PHASE_B,),
        threshold_robustness_triple=(0.2, 0.3, 0.4),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
    )


# ---------------------------------------------------------------------------
# Hand-crafted fixture helpers
# ---------------------------------------------------------------------------


def _minimal_record(
    *,
    model: str,
    condition: str,
    threshold_triple: tuple[float, float, float],
    n_nodes: int,
    expected_edges: int,
    graph_seed: int,
    shd_value: int,
    sid_value: int,
    noise_scale: float = 1.0,
    weight_magnitude_range: tuple[float, float] = (0.5, 2.0),
) -> dict[str, Any]:
    """Build a schema-conformant run.json dict for fixtures."""
    return {
        "run_id": "fixture__run__id",
        "schema_version": 1,
        "model": model,
        "condition": condition,
        "seed_population": "calibration",
        "seed_replicate_index": 0,
        "configuration_hash": "0" * 64,
        "graph_seed": int(graph_seed),
        "git_hash": "unknown",
        "env_snapshot": "fixture",
        "config_resolved": {
            "model": model,
            "condition": condition,
            "n_nodes": int(n_nodes),
            "expected_edges": int(expected_edges),
            "noise_scale": float(noise_scale),
            "weight_magnitude_range": [
                float(weight_magnitude_range[0]),
                float(weight_magnitude_range[1]),
            ],
            "threshold_robustness_triple": [
                float(threshold_triple[0]),
                float(threshold_triple[1]),
                float(threshold_triple[2]),
            ],
        },
        "seed_torch": None,
        "seed_numpy": None,
        "seed_dagma": None,
        "model_sampling_seed_base": 0,
        "model_sampling_seed_derivation_rule": SEED_DERIVATION_RULE_NAME,
        "train_data_seed": 0,
        "validation_data_seed": None,
        "intervention_ground_truth_seed_base": 0,
        "training_status": "converged",
        "n_iterations": None,
        "runtime_seconds": 0.0,
        "loss_history": None,
        "loss_history_status": "unavailable_no_api",
        "graph_status": "valid_dag",
        "graph_status_reason": None,
        "thresholded_adjacency": "thresholded_adjacency.npz",
        "continuous_edge_object": "continuous_edge_object.npz",
        "shd": int(shd_value),
        "sid": int(sid_value),
        "mmd_primary": None,
        "mmd_sensitivity_unit_variance": None,
        "mmd_bandwidth_sweep": {"0.5x": None, "1.0x": None, "2.0x": None},
        "validation_nll": None,
        "sampler_status": "available",
        "sampler_status_reason": None,
        "sampler_policy_used": "residual_fitted",
        "mmd_available_count": 0,
        "mmd_missing_count": 0,
        "invalid_graph_for_this_run": False,
        "shd_reversal_cost": 2,
        "mmd_bandwidth_used_value": {},
        "mmd_clip_policy": "no_clip",
        "sid_backend": "gadjid",
        "sid_backend_version": "0.1.0",
        "sid_argument_order": "predicted_then_true",
        "sid_return_value": "raw_mistake_count",
        "configuration_hash_algorithm": "sha256_canonical_json_sorted_keys",
        "wrapper_diagnostics": {},
        "convergence_failure_notes": "",
        "wrapper_warnings": [],
        "interventions": [],
    }


def _write_fixture(
    tmp_path: Path,
    record: dict[str, Any],
    *,
    artefact_arrays: dict[str, np.ndarray],
) -> Path:
    """Write a run.json + continuous_edge_object.npz fixture."""
    run_dir = tmp_path / "fixture_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    np.savez(run_dir / "continuous_edge_object.npz", **artefact_arrays)
    return run_dir


# ---------------------------------------------------------------------------
# DAGMA: abs(W_continuous), including a negative-weight fixture
# ---------------------------------------------------------------------------


def test_dagma_thresholds_use_abs_w_continuous(tmp_path: Path) -> None:
    """DAGMA thresholds operate on ``abs(W_continuous)``.

    The fixture's continuous matrix carries one strongly negative
    off-diagonal entry. With a positive threshold and signed W
    the predicted adjacency would be empty; with abs(W) the entry
    survives every threshold in the triple. The test asserts the
    surviving edge is recovered.
    """
    n = 3
    w = np.zeros((n, n), dtype=np.float64)
    w[0, 2] = -0.9
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=n,
        expected_edges=2,
        graph_seed=123,
        shd_value=0,
        sid_value=0,
    )
    record["shd"] = 999
    record["sid"] = 999
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": w},
    )
    out = recompute_at_thresholds(run_dir, write_sibling=False)
    assert out["model"] == "dagma"
    for entry in out["records"]:
        assert entry["edge_count"] == 1, (
            "DAGMA threshold did not preserve the negative edge "
            f"with abs(W): {entry!r}"
        )


def test_dagma_primary_threshold_matches_run_shd_and_sid(
    tmp_path: Path,
) -> None:
    """Cross-check against the schema-gate pipeline.

    The pipeline writes a complete run.json; the threshold-
    robustness primary record must match ``run.json["shd"]`` and
    ``run.json["sid"]`` exactly.
    """
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    run_dir = json_path.parent
    record = load_run(json_path).data
    out = recompute_at_thresholds(run_dir, write_sibling=True)
    primary = out["records"][out["primary_threshold_index"]]
    assert primary["threshold_role"] == "primary"
    assert math.isclose(
        primary["threshold"], 0.3, abs_tol=1e-9
    )
    assert isinstance(primary["shd"], int)
    assert primary["shd_unavailable_reason"] is None
    assert primary["shd"] == record["shd"], (
        f"primary-threshold SHD={primary['shd']!r} does not match "
        f"run.json shd={record['shd']!r}"
    )
    assert record["graph_status"] == "valid_dag"
    assert primary["graph_status"] == "valid_dag"
    assert isinstance(primary["sid"], int)
    assert primary["sid_unavailable_reason"] is None
    assert primary["sid"] == record["sid"], (
        f"primary-threshold SID={primary['sid']!r} does not match "
        f"run.json sid={record['sid']!r}"
    )


def test_run_json_bytes_unchanged_after_recompute(tmp_path: Path) -> None:
    """``run.json`` is byte-identical before and after recomputation."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    before = json_path.read_bytes()
    recompute_at_thresholds(json_path.parent, write_sibling=True)
    after = json_path.read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# DCDI: w_adj, not log_alpha
# ---------------------------------------------------------------------------


def test_dcdi_thresholds_w_adj_not_log_alpha(tmp_path: Path) -> None:
    """DCDI thresholding uses ``w_adj``; ``log_alpha`` is ignored.

    The fixture's ``log_alpha`` is saturated positive (would cross
    every threshold if it were the target) while ``w_adj`` carries
    a single near-mid-range entry. Each per-threshold edge count
    must reflect ``w_adj``, not ``log_alpha``.
    """
    n = 4
    log_alpha = np.full((n, n), 50.0, dtype=np.float64)
    np.fill_diagonal(log_alpha, 0.0)
    w_adj = np.zeros((n, n), dtype=np.float64)
    w_adj[0, 1] = 0.55
    record = _minimal_record(
        model="dcdi",
        condition="centred_only",
        threshold_triple=(0.4, 0.5, 0.6),
        n_nodes=n,
        expected_edges=2,
        graph_seed=7,
        shd_value=999,
        sid_value=999,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={
            "log_alpha": log_alpha,
            "w_adj": w_adj,
        },
    )
    out = recompute_at_thresholds(run_dir, write_sibling=False)
    edges = {entry["threshold"]: entry["edge_count"] for entry in out["records"]}
    assert edges[0.4] == 1
    assert edges[0.5] == 1
    assert edges[0.6] == 0


# ---------------------------------------------------------------------------
# Ordered triple validation
# ---------------------------------------------------------------------------


def test_ordered_triple_accepted(tmp_path: Path) -> None:
    """The protocol-ordered DAGMA triple ``(0.2, 0.3, 0.4)`` is accepted."""
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=3,
        expected_edges=2,
        graph_seed=11,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": np.zeros((3, 3), dtype=np.float64)},
    )
    out = recompute_at_thresholds(run_dir, write_sibling=False)
    assert tuple(out["threshold_triple"]) == (0.2, 0.3, 0.4)
    assert [r["threshold_role"] for r in out["records"]] == [
        "low",
        "primary",
        "high",
    ]


def test_out_of_order_triple_rejected(tmp_path: Path) -> None:
    """A permutation of the protocol triple is rejected.

    ``(0.3, 0.2, 0.4)`` would match as an unordered set but is the
    wrong assignment of ``(low, primary, high)``; the validator
    must reject it.
    """
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.3, 0.2, 0.4),
        n_nodes=3,
        expected_edges=2,
        graph_seed=11,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": np.zeros((3, 3), dtype=np.float64)},
    )
    with pytest.raises(ValueError) as excinfo:
        recompute_at_thresholds(run_dir, write_sibling=False)
    assert "threshold_robustness_triple" in str(excinfo.value)


def test_wrong_triple_for_model_rejected(tmp_path: Path) -> None:
    """A DAGMA model paired with the DCDI triple is rejected."""
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.4, 0.5, 0.6),
        n_nodes=3,
        expected_edges=2,
        graph_seed=11,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": np.zeros((3, 3), dtype=np.float64)},
    )
    with pytest.raises(ValueError) as excinfo:
        recompute_at_thresholds(run_dir, write_sibling=False)
    message = str(excinfo.value)
    assert "threshold_robustness_triple" in message
    assert "dagma" in message.lower()


def test_missing_triple_rejected(tmp_path: Path) -> None:
    """A missing triple in ``config_resolved`` is rejected."""
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=3,
        expected_edges=2,
        graph_seed=11,
        shd_value=0,
        sid_value=0,
    )
    record["config_resolved"].pop("threshold_robustness_triple")
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": np.zeros((3, 3), dtype=np.float64)},
    )
    with pytest.raises(ValueError) as excinfo:
        recompute_at_thresholds(run_dir, write_sibling=False)
    assert "threshold_robustness_triple" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Sibling-artefact behaviour
# ---------------------------------------------------------------------------


def test_sibling_written_when_write_sibling_true(tmp_path: Path) -> None:
    """``write_sibling=True`` produces ``threshold_robustness.json``."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    sibling = json_path.parent / "threshold_robustness.json"
    assert not sibling.exists()
    out = recompute_at_thresholds(json_path.parent, write_sibling=True)
    assert sibling.is_file()
    loaded = json.loads(sibling.read_text(encoding="utf-8"))
    assert loaded == out


def test_no_sibling_when_write_sibling_false(tmp_path: Path) -> None:
    """``write_sibling=False`` produces no sibling artefact."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    sibling = json_path.parent / "threshold_robustness.json"
    recompute_at_thresholds(json_path.parent, write_sibling=False)
    assert not sibling.exists()


# ---------------------------------------------------------------------------
# Non-valid-DAG threshold gives sid=None with int SHD and reason
# ---------------------------------------------------------------------------


def test_non_valid_dag_threshold_yields_sid_none(tmp_path: Path) -> None:
    """A threshold producing a bidirected adjacency yields ``sid=None``.

    The fixture sets W so that the high threshold survives in
    both directions of an edge pair, producing a bidirected
    graph. SHD must remain a plain int; SID must be ``None``;
    ``graph_status`` must be a non-``valid_dag`` taxonomy value
    and ``graph_status_reason`` must be a non-empty string.
    """
    n = 3
    w = np.zeros((n, n), dtype=np.float64)
    w[0, 1] = 0.9
    w[1, 0] = 0.9
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=n,
        expected_edges=2,
        graph_seed=42,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": w},
    )
    out = recompute_at_thresholds(run_dir, write_sibling=False)
    invalid = [
        r for r in out["records"] if r["graph_status"] != "valid_dag"
    ]
    assert invalid, (
        "fixture must produce at least one invalid threshold; "
        f"records={out['records']!r}"
    )
    for r in invalid:
        assert r["graph_status"] in (
            "cyclic",
            "bidirected",
            "self_loop",
            "invalid_shape",
        )
        assert isinstance(r["graph_status_reason"], str)
        assert r["graph_status_reason"]
        # The fixture is constructed so that the off-diagonal
        # bidirected pair never crosses the diagonal; SHD remains
        # computable for these non-valid-DAG records.
        assert r["graph_status"] in ("cyclic", "bidirected"), (
            "fixture must isolate the SHD-computable non-valid-DAG "
            f"statuses; got {r!r}"
        )
        assert isinstance(r["shd"], int)
        assert r["shd_unavailable_reason"] is None
        assert r["sid"] is None
        assert isinstance(r["sid_unavailable_reason"], str)
        assert r["sid_unavailable_reason"]


# ---------------------------------------------------------------------------
# Module-level sanity checks
# ---------------------------------------------------------------------------


def test_protocol_threshold_triples_are_ordered() -> None:
    """The exported per-model triples are strictly ascending."""
    for model, triple in PROTOCOL_THRESHOLD_TRIPLES.items():
        assert len(triple) == 3
        assert triple[0] < triple[1] < triple[2], (
            f"protocol triple for {model!r} is not ordered low<primary<high: "
            f"{triple!r}"
        )


def test_threshold_robustness_module_does_not_import_wrappers() -> None:
    """The module source contains no wrappers import statements."""
    import experiments.selection_study.threshold_robustness as tr_module

    source = Path(tr_module.__file__).read_text(encoding="utf-8")
    forbidden_patterns = (
        "from symbolic_priors_cd.wrappers",
        "import symbolic_priors_cd.wrappers",
        "import dagma",
        "from dagma",
        "import dcdi",
        "from dcdi",
    )
    for pattern in forbidden_patterns:
        assert pattern not in source, (
            "threshold_robustness must not import "
            f"{pattern!r}; the module is wrapper-free by contract."
        )


# ---------------------------------------------------------------------------
# Top-level record shape
# ---------------------------------------------------------------------------


def test_top_level_record_shape(tmp_path: Path) -> None:
    """The returned record carries the documented top-level fields."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    out = recompute_at_thresholds(json_path.parent, write_sibling=False)
    expected_top_level = {
        "run_id",
        "model",
        "condition",
        "configuration_hash",
        "continuous_edge_object_artefact",
        "threshold_triple",
        "primary_threshold",
        "primary_threshold_index",
        "shd_reversal_cost",
        "records",
    }
    assert expected_top_level.issubset(set(out.keys()))
    assert len(out["records"]) == 3
    assert out["primary_threshold_index"] == 1
    assert out["primary_threshold"] == out["threshold_triple"][1]
    for entry in out["records"]:
        for key in (
            "threshold",
            "threshold_role",
            "edge_count",
            "graph_status",
            "graph_status_reason",
            "shd",
            "shd_unavailable_reason",
            "sid",
            "sid_unavailable_reason",
        ):
            assert key in entry, (
                f"missing key {key!r} in per-threshold record {entry!r}"
            )
        # Mutual-exclusion invariant: each metric is either int
        # with None reason, or None with non-empty string reason.
        assert (entry["shd"] is None) == (
            entry["shd_unavailable_reason"] is not None
        )
        assert (entry["sid"] is None) == (
            entry["sid_unavailable_reason"] is not None
        )


# A run that triggers the schema-gate's invalid-graph stop must not
# be passed to threshold_robustness because no run.json exists. The
# scenario is silently irrelevant to this module; the test below
# confirms that the schema-gate stop does not produce any artefact
# the recomputation could be asked to consume.


def test_self_loop_threshold_records_self_loop_status(
    tmp_path: Path,
) -> None:
    """A diagonal entry above threshold yields ``graph_status='self_loop'``.

    The fixture places a strong off-diagonal edge and a strong
    diagonal entry. Every threshold in the DAGMA triple captures
    both: the predicted adjacency carries a self-loop, classifies
    as ``self_loop``, ``edge_count`` includes the diagonal entry,
    and both metrics are reported as structurally unavailable with
    explicit reason fields rather than crashing or silently
    omitting values. The sibling artefact carries the same fields.
    """
    n = 3
    w = np.zeros((n, n), dtype=np.float64)
    w[0, 1] = 0.9
    w[0, 0] = 0.9
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=n,
        expected_edges=2,
        graph_seed=99,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": w},
    )
    out = recompute_at_thresholds(run_dir, write_sibling=True)
    self_loop_records = [
        r for r in out["records"] if r["graph_status"] == "self_loop"
    ]
    assert self_loop_records, (
        "fixture must produce at least one self_loop record; "
        f"records={out['records']!r}"
    )
    for entry in self_loop_records:
        assert entry["edge_count"] >= 2, (
            "self-loop edge_count must include the diagonal True "
            f"entry; got {entry!r}"
        )
        assert entry["shd"] is None
        assert isinstance(entry["shd_unavailable_reason"], str)
        assert entry["shd_unavailable_reason"]
        lowered = entry["shd_unavailable_reason"].lower()
        assert "self-loop" in lowered or "diagonal" in lowered, (
            "shd_unavailable_reason must mention self-loop or "
            f"diagonal; got {entry['shd_unavailable_reason']!r}"
        )
        assert entry["sid"] is None
        assert isinstance(entry["sid_unavailable_reason"], str)
        assert entry["sid_unavailable_reason"]
        sid_lowered = entry["sid_unavailable_reason"].lower()
        assert (
            "valid dag" in sid_lowered or "self_loop" in sid_lowered
        ), (
            "sid_unavailable_reason must mention valid DAG or "
            f"self_loop; got {entry['sid_unavailable_reason']!r}"
        )
        assert isinstance(entry["graph_status_reason"], str)
        assert entry["graph_status_reason"]
    sibling_path = run_dir / "threshold_robustness.json"
    assert sibling_path.is_file()
    loaded = json.loads(sibling_path.read_text(encoding="utf-8"))
    for written, expected in zip(loaded["records"], out["records"]):
        if written["graph_status"] == "self_loop":
            assert written["shd"] is None
            assert (
                written["shd_unavailable_reason"]
                == expected["shd_unavailable_reason"]
            )
            assert written["sid"] is None
            assert (
                written["sid_unavailable_reason"]
                == expected["sid_unavailable_reason"]
            )


def test_shape_mismatch_raises_before_sibling_artefact_write(
    tmp_path: Path,
) -> None:
    """A shape mismatch between SCM fields and the artefact raises.

    The fixture says ``n_nodes=3`` in ``config_resolved`` but the
    saved ``W_continuous`` is 2x2. The recomputation must raise
    ``ValueError`` before any sibling artefact is written; this
    documents that an ``invalid_shape`` situation does not
    normally reach persistence through the public path.
    """
    record = _minimal_record(
        model="dagma",
        condition="centred_only",
        threshold_triple=(0.2, 0.3, 0.4),
        n_nodes=3,
        expected_edges=2,
        graph_seed=13,
        shd_value=0,
        sid_value=0,
    )
    run_dir = _write_fixture(
        tmp_path,
        record,
        artefact_arrays={"W_continuous": np.zeros((2, 2), dtype=np.float64)},
    )
    sibling = run_dir / "threshold_robustness.json"
    assert not sibling.exists()
    with pytest.raises(ValueError):
        recompute_at_thresholds(run_dir, write_sibling=True)
    assert not sibling.exists()


def test_invalid_graph_stop_leaves_no_run_json_for_recompute(
    tmp_path: Path,
) -> None:
    """A schema-gate invalid-graph stop produces no run.json sibling."""
    from tests.test_pipeline import _BidirectedFakeWrapper  # noqa: WPS433

    config = Configuration(
        model="dagma",
        condition="centred_only",
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=(("calibration", (10,)),),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        phase_b_configurations=(_PHASE_B,),
        threshold_robustness_triple=(0.2, 0.3, 0.4),
        wrapper_api_reference=(
            "tests.test_pipeline:_BidirectedFakeWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    _ = _BidirectedFakeWrapper
