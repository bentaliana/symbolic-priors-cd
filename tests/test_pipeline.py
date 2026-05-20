"""Tests for the schema-conformance gate pipeline.

Exercises ``resolve_wrapper`` and ``run_single_fit`` end-to-end on a
toy SCM for both DAGMA and DCDI candidates, plus the explicit stop
conditions: non-Manifest input, invalid entry_index, populated
target directory, DCDI seed mismatch, and invalid-graph-makes-SID-
unrecoverable. The invalid-graph stop is exercised symmetrically
via a deliberately-broken fake wrapper that yields a non-valid_dag
thresholded adjacency.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from experiments.selection_study.config import (
    CONFIGURATION_HASH_ALGORITHM_NAME,
    SEED_DERIVATION_RULE_NAME,
    Configuration,
    InterventionSpec,
    PhaseBConfiguration,
)
from experiments.selection_study.identity import (
    derive_run_directory,
    derive_run_id,
)
from experiments.selection_study.loader import load_run
from experiments.selection_study.pipeline import (
    DcdiSeedMismatchError,
    InvalidGraphForSchemaGateError,
    resolve_wrapper,
    run_single_fit,
)
from experiments.selection_study.preflight import (
    Manifest,
    enumerate_manifest,
)


# ---------------------------------------------------------------------------
# Fake wrappers used to test the invalid-graph stop symmetrically
# ---------------------------------------------------------------------------


class _BidirectedFakeWrapper:
    """Fake wrapper whose thresholded adjacency is bidirected.

    Accepts both the DAGMA fit signature
    (``X_train, *, preprocessor, seed, config``) and the DCDI fit
    signature (which adds ``X_val``, ``n_iter``) via ``**kwargs``.
    The diagnostics record carries DAGMA-shaped
    ``model_specific_diagnostics`` keys; that part is unreached
    because the pipeline raises before any continuous-edge artefact
    is written when the graph is invalid.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._n_vars = None
        self._preprocessor = None

    def fit(self, X_train, **kwargs):
        self._fitted = True
        self._n_vars = int(X_train.shape[1])
        self._preprocessor = kwargs.get("preprocessor")

    def native_edge_continuous(self):
        n = self._n_vars
        out = np.zeros((n, n), dtype=np.float64)
        out[0, 1] = 0.9
        out[1, 0] = 0.9
        return out

    def thresholded_adjacency(self, threshold: float = 0.5) -> np.ndarray:
        n = self._n_vars
        out = np.zeros((n, n), dtype=bool)
        out[0, 1] = True
        out[1, 0] = True
        return out

    def sample_interventional(self, intervention, n_samples, *, sample_seed):
        return None

    def get_diagnostics(self):
        adj = self.thresholded_adjacency()
        return {
            "training_status": "max_iter",
            "graph_status": "bidirected",
            "sampler_status": "unavailable_invalid_graph",
            "seed": 0,
            "n_iterations": None,
            "config_snapshot": {},
            "loss_history": [],
            "loss_decomposition_final": {},
            "convergence_info": {},
            "thresholded_adjacency": adj,
            "graph_invalid_reason": "deliberately bidirected for stop test",
            "sampler_unavailable_reason": (
                "deliberately bidirected for stop test"
            ),
            "mmd_sampling_metadata": {},
            "loss_hook_name": None,
            "numerical_tolerances": {},
            "model_specific_diagnostics": {
                "continuous_w_pre_threshold": self.native_edge_continuous(),
                "model_name": "FAKE",
            },
        }


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


_DAGMA_SCHEMA_GATE_FIELDS = dict(
    dagma_warm_iter=20000,
    dagma_max_iter=70000,
    dagma_lr=3e-4,
    dagma_beta_1=0.99,
    dagma_beta_2=0.999,
)
_DCDI_SCHEMA_GATE_FIELDS = dict(
    n_val_dcdi=32,
    dcdi_num_train_iter=30,
    dcdi_stop_crit_win=10,
    dcdi_train_patience=5,
    dcdi_train_batch_size=8,
    dcdi_lr=1e-3,
    dcdi_h_threshold=1e-8,
    dcdi_hidden_units=16,
    dcdi_hidden_layers=2,
)


def _make_dagma_config(
    *,
    wrapper_reference: str = "symbolic_priors_cd.wrappers.dagma:DAGMAWrapper",
    seed_population: str = "calibration",
    seeds: tuple[int, ...] = (10,),
    condition: str = "centred_only",
) -> Configuration:
    return Configuration(
        model="dagma",
        condition=condition,
        seed_torch=None,
        seed_numpy=None,
        seed_dagma=None,
        seed_populations=((seed_population, seeds),),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        phase_b_configurations=(_PHASE_B,),
        threshold_robustness_triple=(0.2, 0.3, 0.4),
        wrapper_api_reference=wrapper_reference,
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **_DAGMA_SCHEMA_GATE_FIELDS,
    )


def _make_dcdi_config(
    *,
    wrapper_reference: str = "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper",
    seed_torch: int = 42,
    seed_numpy: int = 42,
    seed_population: str = "calibration",
    seeds: tuple[int, ...] = (10,),
    condition: str = "centred_only",
) -> Configuration:
    return Configuration(
        model="dcdi",
        condition=condition,
        seed_torch=seed_torch,
        seed_numpy=seed_numpy,
        seed_dagma=None,
        seed_populations=((seed_population, seeds),),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        phase_b_configurations=(_PHASE_B,),
        threshold_robustness_triple=(0.4, 0.5, 0.6),
        wrapper_api_reference=wrapper_reference,
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **_DCDI_SCHEMA_GATE_FIELDS,
    )


# ---------------------------------------------------------------------------
# resolve_wrapper
# ---------------------------------------------------------------------------


def test_resolve_wrapper_resolves_dagma_dynamically() -> None:
    cls = resolve_wrapper("symbolic_priors_cd.wrappers.dagma:DAGMAWrapper")
    assert isinstance(cls, type)
    assert cls.__name__ == "DAGMAWrapper"


def test_resolve_wrapper_resolves_dcdi_dynamically() -> None:
    cls = resolve_wrapper("symbolic_priors_cd.wrappers.dcdi:DCDIWrapper")
    assert isinstance(cls, type)
    assert cls.__name__ == "DCDIWrapper"


def test_resolve_wrapper_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        resolve_wrapper(123)  # type: ignore[arg-type]


def test_resolve_wrapper_rejects_missing_colon() -> None:
    with pytest.raises(ValueError, match="exactly one ':'"):
        resolve_wrapper("symbolic_priors_cd.wrappers.dagma.DAGMAWrapper")


def test_resolve_wrapper_rejects_multiple_colons() -> None:
    with pytest.raises(ValueError, match="exactly one ':'"):
        resolve_wrapper("a:b:c")


def test_resolve_wrapper_rejects_empty_module() -> None:
    with pytest.raises(ValueError, match="module-path component is empty"):
        resolve_wrapper(":DAGMAWrapper")


def test_resolve_wrapper_rejects_empty_class_name() -> None:
    with pytest.raises(ValueError, match="class-name component is empty"):
        resolve_wrapper("symbolic_priors_cd.wrappers.dagma:")


def test_resolve_wrapper_rejects_missing_module() -> None:
    with pytest.raises(ImportError):
        resolve_wrapper("no_such_module_for_test:Anything")


def test_resolve_wrapper_rejects_missing_attribute() -> None:
    with pytest.raises(AttributeError):
        resolve_wrapper("symbolic_priors_cd.wrappers.dagma:NoSuchClass")


def test_resolve_wrapper_rejects_non_class_attribute() -> None:
    # SEED_DERIVATION_RULE_NAME is a str constant, not a class.
    with pytest.raises(TypeError, match="not a class"):
        resolve_wrapper(
            "experiments.selection_study.config:SEED_DERIVATION_RULE_NAME"
        )


# ---------------------------------------------------------------------------
# run_single_fit input validation
# ---------------------------------------------------------------------------


def test_run_single_fit_rejects_non_manifest(tmp_path) -> None:
    with pytest.raises(TypeError, match="Manifest"):
        run_single_fit(
            object(),  # type: ignore[arg-type]
            0,
            run_root=tmp_path,
        )


def test_run_single_fit_rejects_bool_entry_index(tmp_path) -> None:
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(TypeError, match="entry_index"):
        run_single_fit(
            manifest,
            True,  # type: ignore[arg-type]
            run_root=tmp_path / "runs",
        )


def test_run_single_fit_rejects_negative_entry_index(tmp_path) -> None:
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(IndexError):
        run_single_fit(manifest, -1, run_root=tmp_path / "runs")


def test_run_single_fit_rejects_out_of_range_entry_index(tmp_path) -> None:
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(IndexError):
        run_single_fit(
            manifest, len(manifest.entries), run_root=tmp_path / "runs"
        )


# ---------------------------------------------------------------------------
# DAGMA toy run end-to-end
# ---------------------------------------------------------------------------


def _verify_run_json_against_schema(
    record: dict,
    expected_run_id: str,
    expected_configuration_hash: str,
) -> None:
    """Assert the record contains every required field with sensible types."""
    assert record["schema_version"] == 1
    assert record["run_id"] == expected_run_id
    assert record["configuration_hash"] == expected_configuration_hash
    assert record["graph_status"] == "valid_dag"
    assert record["graph_status_reason"] is None
    assert isinstance(record["shd"], int)
    assert isinstance(record["sid"], int)
    assert isinstance(record["runtime_seconds"], float)
    assert isinstance(record["wrapper_diagnostics"], dict)
    assert record["mmd_clip_policy"] == "no_clip"
    assert record["sid_backend"] == "gadjid"
    assert record["sid_backend_version"] == "0.1.0"
    assert record["sid_argument_order"] == "predicted_then_true"
    assert record["sid_return_value"] == "raw_mistake_count"
    assert record["configuration_hash_algorithm"] == (
        "sha256_canonical_json_sorted_keys"
    )
    assert record["thresholded_adjacency"] == "thresholded_adjacency.npz"
    assert record["continuous_edge_object"] == "continuous_edge_object.npz"
    assert record["shd_reversal_cost"] == 2
    # mmd_primary and mmd_sensitivity_unit_variance may be float or
    # None depending on sampler_status and whether the primary policy
    # is residual_fitted.
    assert (
        record["mmd_primary"] is None
        or isinstance(record["mmd_primary"], float)
    )
    assert (
        record["mmd_sensitivity_unit_variance"] is None
        or isinstance(record["mmd_sensitivity_unit_variance"], float)
    )
    assert record["validation_nll"] is None
    assert isinstance(record["mmd_bandwidth_sweep"], dict)
    assert set(record["mmd_bandwidth_sweep"].keys()) == {"0.5x", "1.0x", "2.0x"}
    assert isinstance(record["mmd_bandwidth_used_value"], dict)
    assert isinstance(record["wrapper_warnings"], list)
    assert isinstance(record["convergence_failure_notes"], str)


def test_dagma_toy_run_produces_loadable_run_json(tmp_path) -> None:
    """A DAGMA toy fit writes run.json that the minimal loader accepts."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]

    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    assert json_path.name == "run.json"
    assert json_path.exists()

    record_obj = load_run(json_path)
    record = record_obj.data
    _verify_run_json_against_schema(
        record,
        expected_run_id=entry.expected_run_id,
        expected_configuration_hash=entry.configuration_hash,
    )
    assert record["model"] == "dagma"
    assert record["seed_torch"] is None
    assert record["seed_numpy"] is None
    assert record["seed_dagma"] is None
    assert record["n_iterations"] is None
    assert record["loss_history_status"] == "unavailable_no_api"
    assert record["loss_history"] is None
    assert record["sampler_policy_used"] == "residual_fitted"


def test_dagma_run_records_scm_generation_fields_in_config_resolved(
    tmp_path,
) -> None:
    """``config_resolved`` carries the SCM-generation fields end-to-end."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    record = load_run(json_path).data
    resolved = record["config_resolved"]
    assert resolved["n_nodes"] == config.n_nodes
    assert resolved["expected_edges"] == config.expected_edges
    assert resolved["noise_scale"] == float(config.noise_scale)
    assert resolved["weight_magnitude_range"] == [
        float(config.weight_magnitude_range[0]),
        float(config.weight_magnitude_range[1]),
    ]


def test_dagma_run_artefacts_exist_and_load(tmp_path) -> None:
    """thresholded_adjacency.npz and continuous_edge_object.npz exist and load."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    run_dir = json_path.parent

    adj_path = run_dir / "thresholded_adjacency.npz"
    cont_path = run_dir / "continuous_edge_object.npz"
    assert adj_path.exists() and cont_path.exists()

    expected_n_nodes = config.n_nodes
    with np.load(adj_path) as f:
        adj = f["thresholded_adjacency"]
    assert adj.dtype == bool
    assert adj.shape == (expected_n_nodes, expected_n_nodes)

    with np.load(cont_path) as f:
        w = f["W_continuous"]
    assert w.shape == (expected_n_nodes, expected_n_nodes)
    assert w.dtype == np.float64


def test_dagma_run_id_matches_directory(tmp_path) -> None:
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    json_path = run_single_fit(manifest, 0, run_root=tmp_path / "runs")

    expected_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )
    assert json_path.parent == expected_dir

    record = load_run(json_path).data
    expected_run_id = derive_run_id(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
    )
    assert record["run_id"] == expected_run_id


def test_dagma_intervention_records_match_section_6_10(tmp_path) -> None:
    """Per-intervention records honour the schema's consistency rules."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    json_path = run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    record = load_run(json_path).data

    interventions = record["interventions"]
    assert len(interventions) == len(config.intervention_set)
    available_records = [
        r for r in interventions if r["mmd_status"] == "available"
    ]
    available = len(available_records)
    missing = len(interventions) - available
    assert record["mmd_available_count"] == available
    assert record["mmd_missing_count"] == missing
    assert record["mmd_available_count"] + record["mmd_missing_count"] == (
        len(interventions)
    )
    allowed_mmd_statuses = {
        "available",
        "unavailable_invalid_graph",
        "unavailable_no_api",
        "unavailable_unresolved_noise_policy",
        "unavailable_other",
    }
    for r in interventions:
        assert r["mmd_status"] in allowed_mmd_statuses
        assert set(r["bandwidth_sweep"].keys()) == {"0.5x", "1.0x", "2.0x"}
        # Consistency rules between sampler_status_for_intervention,
        # mmd_status, and mmd_value.
        if r["sampler_status_for_intervention"] != "available":
            assert r["mmd_status"] == r["sampler_status_for_intervention"]
            assert r["mmd_value"] is None
            assert r["bandwidth_used"] is None
        elif r["mmd_status"] == "available":
            assert isinstance(r["mmd_value"], float)
            assert np.isfinite(r["mmd_value"])
            assert r["bandwidth_sweep"]["1.0x"] == r["mmd_value"]
            assert isinstance(r["bandwidth_used"], float)
            assert r["bandwidth_used"] > 0.0
            assert r["n_ground_truth_samples"] > 0
            assert r["n_model_samples"] > 0
        else:
            assert r["mmd_status"] == "unavailable_other"
            assert r["mmd_value"] is None
    # Seeds match the manifest entry.
    seeds_map = dict(entry.per_intervention_seeds)
    for r in interventions:
        s = seeds_map[r["intervention_id"]]
        assert r["ground_truth_sampling_seed"] == s.ground_truth_sampling_seed
        assert r["model_sampling_seed"] == s.model_sampling_seed
    # mmd_bandwidth_used_value mirrors each record's bandwidth_used.
    for r in interventions:
        assert (
            record["mmd_bandwidth_used_value"][r["intervention_id"]]
            == r["bandwidth_used"]
        )


# ---------------------------------------------------------------------------
# DCDI toy run end-to-end (acceptable either path)
# ---------------------------------------------------------------------------


def test_dcdi_toy_run_either_succeeds_or_triggers_invalid_graph_stop(
    tmp_path,
) -> None:
    """DCDI at toy constants may or may not produce a valid DAG.

    Either outcome is acceptable for the schema gate. On a valid_dag,
    the run.json must round-trip through the loader. On a non-
    valid_dag, the pipeline must raise InvalidGraphForSchemaGateError
    without writing run.json.
    """
    config = _make_dcdi_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    try:
        json_path = run_single_fit(
            manifest, 0, run_root=tmp_path / "runs"
        )
    except InvalidGraphForSchemaGateError as exc:
        assert "graph_status" in str(exc)
        # Confirm no run.json was written.
        expected_dir = derive_run_directory(
            model=entry.model,
            condition=entry.condition,
            seed_population=entry.seed_population,
            seed_replicate_index=entry.seed_replicate_index,
            configuration_hash=entry.configuration_hash,
            base_dir=tmp_path / "runs",
        )
        assert not (expected_dir / "run.json").exists()
        return

    record = load_run(json_path).data
    _verify_run_json_against_schema(
        record,
        expected_run_id=entry.expected_run_id,
        expected_configuration_hash=entry.configuration_hash,
    )
    assert record["model"] == "dcdi"
    assert record["seed_torch"] == 42
    assert record["seed_numpy"] == 42
    assert record["seed_dagma"] is None
    assert record["sampler_policy_used"] == "dcdi_native"
    assert isinstance(record["n_iterations"], int)
    # DCDI loss history is non-empty for a non-trivial n_iter.
    assert record["loss_history_status"] == "available"
    assert record["loss_history"] == "loss_history.npz"
    assert (json_path.parent / "loss_history.npz").exists()
    with np.load(json_path.parent / "loss_history.npz") as f:
        lh = f["loss_history"]
    assert lh.ndim == 1 and lh.shape[0] > 0


# ---------------------------------------------------------------------------
# DCDI seed-mismatch stop
# ---------------------------------------------------------------------------


def test_dcdi_seed_mismatch_raises(tmp_path) -> None:
    """Unequal seed_torch and seed_numpy trigger the DCDI seed-mismatch stop."""
    config = _make_dcdi_config(seed_torch=7, seed_numpy=8)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(DcdiSeedMismatchError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")


# ---------------------------------------------------------------------------
# Invalid-graph stop, tested symmetrically for DAGMA and DCDI dispatch
# ---------------------------------------------------------------------------


_FAKE_WRAPPER_REFERENCE = (
    "tests.test_pipeline:_BidirectedFakeWrapper"
)


def test_invalid_graph_stop_for_dagma_dispatch(tmp_path) -> None:
    """A fake wrapper returning a bidirected graph triggers the SID stop."""
    config = _make_dagma_config(wrapper_reference=_FAKE_WRAPPER_REFERENCE)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")


def test_invalid_graph_stop_for_dcdi_dispatch(tmp_path) -> None:
    """The same fake wrapper triggers the stop when entered via the DCDI dispatch."""
    config = _make_dcdi_config(wrapper_reference=_FAKE_WRAPPER_REFERENCE)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")


def test_invalid_graph_stop_does_not_write_run_json(tmp_path) -> None:
    """When the stop fires, no run.json is written."""
    config = _make_dagma_config(wrapper_reference=_FAKE_WRAPPER_REFERENCE)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    expected_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )
    assert not (expected_dir / "run.json").exists()


# ---------------------------------------------------------------------------
# Populated target directory rejection
# ---------------------------------------------------------------------------


def test_populated_target_directory_raises_before_destructive_writes(
    tmp_path,
) -> None:
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]

    run_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("preserved", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")

    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert not (run_dir / "run.json").exists()
    assert not (run_dir / "thresholded_adjacency.npz").exists()
    assert not (run_dir / "continuous_edge_object.npz").exists()


# ---------------------------------------------------------------------------
# Configuration hash round-trip
# ---------------------------------------------------------------------------


def test_configuration_hash_in_record_matches_entry(tmp_path) -> None:
    """The configuration_hash recorded matches the manifest entry's hash."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    json_path = run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    record = load_run(json_path).data
    assert record["configuration_hash"] == entry.configuration_hash
    assert record["configuration_hash"] == manifest.configuration_hash


def test_run_json_payload_is_ascii_safe(tmp_path) -> None:
    """run.json must be sorted and ASCII-safe to round-trip cleanly."""
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    raw = json_path.read_text(encoding="utf-8")
    # ascii-only check
    raw.encode("ascii")
    # sorted-keys check via re-parse + re-dump
    parsed = json.loads(raw)
    redumped = json.dumps(parsed, sort_keys=True, ensure_ascii=True, indent=2)
    assert redumped == raw


# ---------------------------------------------------------------------------
# _to_jsonable: unsupported types must raise rather than silently stringify
# ---------------------------------------------------------------------------


def test_to_jsonable_raises_typeerror_for_unsupported_type() -> None:
    """An unsupported object type must raise TypeError naming the type."""
    from experiments.selection_study.pipeline import _to_jsonable

    class _NotSerialisable:
        pass

    with pytest.raises(TypeError, match="_NotSerialisable"):
        _to_jsonable(_NotSerialisable())


def test_to_jsonable_raises_typeerror_for_nested_unsupported_type() -> None:
    """A nested unsupported value triggers the TypeError too."""
    from experiments.selection_study.pipeline import _to_jsonable

    class _NotSerialisable:
        pass

    payload = {"outer": {"inner": [_NotSerialisable()]}}
    with pytest.raises(TypeError, match="_NotSerialisable"):
        _to_jsonable(payload)


# ---------------------------------------------------------------------------
# Static DCDI precondition stops must not create the run directory
# ---------------------------------------------------------------------------


def test_dcdi_seed_mismatch_does_not_create_run_directory(tmp_path) -> None:
    """When seed_torch != seed_numpy the run directory must not exist."""
    config = _make_dcdi_config(seed_torch=7, seed_numpy=8)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    run_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )
    assert not run_dir.exists()
    with pytest.raises(DcdiSeedMismatchError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    assert not run_dir.exists()


# ---------------------------------------------------------------------------
# Invalid-graph stop empty-directory invariant
# ---------------------------------------------------------------------------


def test_invalid_graph_stop_leaves_empty_run_directory(tmp_path) -> None:
    """The invalid-graph stop leaves the run directory existing and empty.

    The run directory must exist (so a future re-fit at the same
    identity sees an empty slot to write into) and contain no
    artefacts: no run.json, no thresholded_adjacency.npz, no
    continuous_edge_object.npz, no loss_history.npz, no other files.
    """
    config = _make_dagma_config(wrapper_reference=_FAKE_WRAPPER_REFERENCE)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    run_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )

    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")

    assert run_dir.exists()
    assert run_dir.is_dir()
    contents = list(run_dir.iterdir())
    assert contents == [], (
        f"invalid-graph stop left artefacts in run_dir: {contents!r}"
    )


def test_pipeline_dagma_toy_run_writes_non_null_mmd_primary(tmp_path) -> None:
    """A DAGMA toy fit with an available sampler writes a real mmd_primary.

    The pipeline must produce a non-null arithmetic-mean mmd_primary,
    a non-null mmd_sensitivity_unit_variance (because DAGMA's primary
    policy is residual_fitted), per-intervention records with
    mmd_status='available' and finite mmd_value, and aggregate
    fields equal to the means of the available per-intervention
    values.
    """
    config = _make_dagma_config()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    record = load_run(json_path).data

    if record["sampler_status"] != "available":
        pytest.skip(
            "test requires an available DAGMA sampler from the toy fit"
        )

    assert record["mmd_primary"] is not None
    assert isinstance(record["mmd_primary"], float)
    assert np.isfinite(record["mmd_primary"])
    assert record["mmd_available_count"] > 0
    assert record["mmd_available_count"] == len(record["interventions"])
    assert record["mmd_missing_count"] == 0

    assert record["mmd_sensitivity_unit_variance"] is not None
    assert isinstance(record["mmd_sensitivity_unit_variance"], float)
    assert np.isfinite(record["mmd_sensitivity_unit_variance"])

    available = [
        r for r in record["interventions"] if r["mmd_status"] == "available"
    ]
    assert len(available) == len(record["interventions"])
    expected_primary = float(
        np.mean([r["mmd_value"] for r in available])
    )
    assert record["mmd_primary"] == pytest.approx(expected_primary, abs=1e-12)
    for key in ("0.5x", "1.0x", "2.0x"):
        expected_sweep = float(
            np.mean([r["bandwidth_sweep"][key] for r in available])
        )
        assert record["mmd_bandwidth_sweep"][key] == pytest.approx(
            expected_sweep, abs=1e-12
        )

    for r in available:
        assert record["mmd_bandwidth_used_value"][r["intervention_id"]] == (
            r["bandwidth_used"]
        )


def test_invalid_graph_stop_leaves_empty_run_directory_dcdi(tmp_path) -> None:
    """Same empty-directory invariant on the DCDI dispatch path."""
    config = _make_dcdi_config(wrapper_reference=_FAKE_WRAPPER_REFERENCE)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    entry = manifest.entries[0]
    run_dir = derive_run_directory(
        model=entry.model,
        condition=entry.condition,
        seed_population=entry.seed_population,
        seed_replicate_index=entry.seed_replicate_index,
        configuration_hash=entry.configuration_hash,
        base_dir=tmp_path / "runs",
    )

    with pytest.raises(InvalidGraphForSchemaGateError):
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")

    assert run_dir.exists()
    assert run_dir.is_dir()
    contents = list(run_dir.iterdir())
    assert contents == [], (
        f"invalid-graph stop left artefacts in run_dir: {contents!r}"
    )



# ---------------------------------------------------------------------------
# Configuration-driven pipeline behaviour
# ---------------------------------------------------------------------------


def _make_dagma_config_with(**overrides) -> Configuration:
    """Return a DAGMA Configuration with the listed field overrides."""
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
        **{**_DAGMA_SCHEMA_GATE_FIELDS, **overrides},
    )


def _make_dcdi_config_with(**overrides) -> Configuration:
    """Return a DCDI Configuration with the listed field overrides."""
    return Configuration(
        model="dcdi",
        condition="centred_only",
        seed_torch=42,
        seed_numpy=42,
        seed_dagma=None,
        seed_populations=(("calibration", (10,)),),
        intervention_set=(_INTERVENTION_A, _INTERVENTION_B),
        phase_b_configurations=(_PHASE_B,),
        threshold_robustness_triple=(0.4, 0.5, 0.6),
        wrapper_api_reference=(
            "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper"
        ),
        seed_derivation_rule=SEED_DERIVATION_RULE_NAME,
        configuration_hash_algorithm=CONFIGURATION_HASH_ALGORITHM_NAME,
        **{**_DCDI_SCHEMA_GATE_FIELDS, **overrides},
    )


def test_pipeline_uses_config_n_train_for_observational_sampling(
    tmp_path, monkeypatch,
) -> None:
    """``run_single_fit`` requests ``config.n_train`` SCM samples."""
    import experiments.selection_study.pipeline as pipeline_module

    config = _make_dagma_config_with(n_train=37)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    captured: list[int] = []
    original_sample = pipeline_module.sample_observational

    def spy_sample(*args, **kwargs):
        captured.append(int(kwargs.get("n_samples")))
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "sample_observational", spy_sample
    )
    run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    assert captured, "sample_observational was not invoked"
    assert captured[0] == 37, (
        f"first observational draw used {captured[0]}, expected 37"
    )


def test_pipeline_passes_config_mmd_n_samples_into_mmd(
    tmp_path, monkeypatch,
) -> None:
    """The MMD call receives ``config.mmd_n_samples`` as ``n_samples``."""
    import experiments.selection_study.pipeline as pipeline_module

    config = _make_dagma_config_with(mmd_n_samples=23)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    captured: list[int] = []
    original = pipeline_module.compute_per_intervention_records

    def spy(*args, **kwargs):
        captured.append(int(kwargs.get("n_samples")))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "compute_per_intervention_records",
        spy,
    )
    run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    assert captured == [23], (
        f"compute_per_intervention_records received n_samples={captured!r}"
    )


def test_pipeline_constructs_dagma_config_from_configuration(
    tmp_path, monkeypatch,
) -> None:
    """DAGMAConfig is built from the resolved DAGMA-only fields."""
    import experiments.selection_study.pipeline as pipeline_module

    config = _make_dagma_config_with(
        dagma_warm_iter=25000,
        dagma_max_iter=65000,
        dagma_lr=4e-4,
        dagma_beta_1=0.95,
        dagma_beta_2=0.995,
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    captured: list[dict] = []
    real_resolver = pipeline_module._resolve_dagma_config_class

    def spy_resolver():
        cls = real_resolver()

        class _SpyDAGMAConfig(cls):
            def __init__(self, *args, **kwargs):
                captured.append(dict(kwargs))
                super().__init__(*args, **kwargs)

        return _SpyDAGMAConfig

    monkeypatch.setattr(
        pipeline_module,
        "_resolve_dagma_config_class",
        spy_resolver,
    )
    run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    assert len(captured) == 1, (
        f"DAGMAConfig was instantiated {len(captured)} times"
    )
    kwargs = captured[0]
    assert kwargs["warm_iter"] == 25000
    assert kwargs["max_iter"] == 65000
    assert kwargs["lr"] == 4e-4
    assert kwargs["beta_1"] == 0.95
    assert kwargs["beta_2"] == 0.995


def test_pipeline_passes_dcdi_fields_to_fit_and_config(
    tmp_path, monkeypatch,
) -> None:
    """DCDIWrapper.fit and DCDIConfig receive every DCDI-only field."""
    import experiments.selection_study.pipeline as pipeline_module

    config = _make_dcdi_config_with()
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")

    dcdi_config_kwargs: list[dict] = []
    fit_kwargs: list[dict] = []
    real_resolver = pipeline_module._resolve_dcdi_config_class
    real_resolve_wrapper = pipeline_module.resolve_wrapper

    def spy_dcdi_config_resolver():
        cls = real_resolver()

        class _SpyDCDIConfig(cls):
            def __init__(self, *args, **kwargs):
                dcdi_config_kwargs.append(dict(kwargs))
                super().__init__(*args, **kwargs)

        return _SpyDCDIConfig

    def spy_resolve_wrapper(reference):
        wrapper_cls = real_resolve_wrapper(reference)
        original_fit = wrapper_cls.fit

        def fit(self, X_train, **kwargs):
            fit_kwargs.append(dict(kwargs))
            return original_fit(self, X_train, **kwargs)

        return type(
            "_SpyDCDIWrapper",
            (wrapper_cls,),
            {"fit": fit},
        )

    monkeypatch.setattr(
        pipeline_module,
        "_resolve_dcdi_config_class",
        spy_dcdi_config_resolver,
    )
    monkeypatch.setattr(
        pipeline_module,
        "resolve_wrapper",
        spy_resolve_wrapper,
    )
    try:
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    except InvalidGraphForSchemaGateError:
        pass

    assert len(dcdi_config_kwargs) == 1
    dcdi_kwargs = dcdi_config_kwargs[0]
    assert dcdi_kwargs["h_threshold"] == config.dcdi_h_threshold
    assert dcdi_kwargs["lr"] == config.dcdi_lr
    assert dcdi_kwargs["train_batch_size"] == config.dcdi_train_batch_size
    assert dcdi_kwargs["train_patience"] == config.dcdi_train_patience
    assert dcdi_kwargs["stop_crit_win"] == config.dcdi_stop_crit_win
    assert dcdi_kwargs["num_layers"] == config.dcdi_hidden_layers
    assert dcdi_kwargs["hid_dim"] == config.dcdi_hidden_units

    assert len(fit_kwargs) == 1
    fk = fit_kwargs[0]
    assert fk["n_iter"] == config.dcdi_num_train_iter


def test_dcdi_validation_is_split_from_n_train_observational_batch(
    tmp_path, monkeypatch,
) -> None:
    """DCDI draws exactly one ``n_train`` SCM batch and splits it."""
    import experiments.selection_study.pipeline as pipeline_module

    config = _make_dcdi_config_with(n_train=80, n_val_dcdi=20)
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    captured: list[int] = []
    original_sample = pipeline_module.sample_observational

    def spy_sample(*args, **kwargs):
        captured.append(int(kwargs.get("n_samples")))
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "sample_observational", spy_sample
    )
    try:
        run_single_fit(manifest, 0, run_root=tmp_path / "runs")
    except InvalidGraphForSchemaGateError:
        pass
    assert captured == [80], (
        "DCDI pipeline drew more than one observational batch: "
        f"calls={captured!r}"
    )


def test_dcdi_split_helper_partitions_rows_deterministically() -> None:
    """``_split_train_validation`` splits a batch by deterministic perm."""
    import experiments.selection_study.pipeline as pipeline_module

    x_raw = np.arange(20, dtype=np.float64).reshape(10, 2)
    fit_a, val_a = pipeline_module._split_train_validation(
        x_raw, permutation_seed=7, n_val=3
    )
    fit_b, val_b = pipeline_module._split_train_validation(
        x_raw, permutation_seed=7, n_val=3
    )
    assert fit_a.shape == (7, 2)
    assert val_a.shape == (3, 2)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(val_a, val_b)
    fit_c, val_c = pipeline_module._split_train_validation(
        x_raw, permutation_seed=8, n_val=3
    )
    assert not (
        np.array_equal(fit_a, fit_c) and np.array_equal(val_a, val_c)
    )
    union = np.concatenate([fit_a, val_a], axis=0)
    assert union.shape == x_raw.shape
    sorted_union = union[np.argsort(union[:, 0])]
    assert np.array_equal(sorted_union, x_raw)


def test_pipeline_run_json_config_resolved_matches_configuration(
    tmp_path,
) -> None:
    """``config_resolved`` in run.json equals the Configuration values."""
    config = _make_dagma_config_with(
        n_train=33,
        mmd_n_samples=27,
        dagma_warm_iter=15000,
        dagma_max_iter=55000,
    )
    manifest = enumerate_manifest(config, base_dir=tmp_path / "runs")
    json_path = run_single_fit(
        manifest, 0, run_root=tmp_path / "runs"
    )
    record = load_run(json_path).data
    resolved = record["config_resolved"]
    assert resolved["n_train"] == 33
    assert resolved["mmd_n_samples"] == 27
    assert resolved["dagma_warm_iter"] == 15000
    assert resolved["dagma_max_iter"] == 55000
    assert resolved["dagma_lr"] == config.dagma_lr
    assert resolved["dagma_beta_1"] == config.dagma_beta_1
    assert resolved["dagma_beta_2"] == config.dagma_beta_2
