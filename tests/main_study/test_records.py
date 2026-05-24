"""Tests for the main-study post-run record schema and validators.

Synthetic configs and synthetic record payloads only; no fits, no
metric computation, no file I/O. Builder helpers return valid
records, and parametrised tests override individual fields to
exercise each validation rule.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from experiments.main_study import records as records_mod
from experiments.main_study.priors import (
    CorruptedPriorSpec,
    edge_tuple_to_key,
)
from experiments.main_study.records import (
    FAILURE_KINDS,
    FIT_STATUSES,
    GRAPH_STATUS_VALUES,
    METRIC_STATUSES,
    SAMPLER_STATUS_VALUES,
    MainStudyRunRecord,
    derive_metric_status_for_failure,
    diagnostics_to_canonical,
    make_failure_record,
)
from experiments.main_study.schema import (
    FROZEN_LAMBDA_PRIOR,
    SCHEMA_VERSION,
    MainStudyConfig,
    compute_configuration_hash,
    configuration_hash_prefix,
    make_run_id,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig
from symbolic_priors_cd.wrappers.status import GraphStatus, SamplerStatus


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


_VALID_PARENT_HASH = "a" * 64
_OTHER_PARENT_HASH = "b" * 64
_VALID_GENERATED_AT = "2026-05-25T12:00:00Z"


def _make_corrupted_prior(
    *,
    forbidden_edges: tuple[tuple[int, int], ...] = (
        (0, 2), (1, 3), (2, 4)
    ),
    corruption_fraction: float = 0.0,
    corruption_index: int = 0,
    corruption_seed: int = 9100 + 42,
    n_correct: int | None = None,
    n_corrupted: int = 0,
    removed_clean_edges: tuple[tuple[int, int], ...] = (),
    added_true_positive_edges: tuple[tuple[int, int], ...] = (),
    edge_labels: dict[str, str] | None = None,
) -> CorruptedPriorSpec:
    if n_correct is None:
        n_correct = len(forbidden_edges) - n_corrupted
    if edge_labels is None:
        edge_labels = {
            edge_tuple_to_key(e): "true_negative_retained"
            for e in forbidden_edges
        }
    return CorruptedPriorSpec(
        n_nodes=5,
        scm_seed=42,
        corruption_fraction=corruption_fraction,
        corruption_index=corruption_index,
        corruption_seed=corruption_seed,
        forbidden_edges=tuple(forbidden_edges),
        n_correct=n_correct,
        n_corrupted=n_corrupted,
        removed_clean_edges=removed_clean_edges,
        added_true_positive_edges=added_true_positive_edges,
        edge_labels=edge_labels,
    )


def _build_prior_free_config() -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )


def _build_soft_frobenius_config() -> MainStudyConfig:
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.4,
        corruption_index=2,
    )
    return MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=0.5,
        corrupted_prior_spec=cp,
    )


def _build_hard_exclusion_config() -> MainStudyConfig:
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.4,
        corruption_index=2,
    )
    sorted_forbidden = tuple(sorted(cp.forbidden_edges))
    return MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(exclude_edges=sorted_forbidden),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )


def _build_matched_l1_config() -> MainStudyConfig:
    return MainStudyConfig(
        method_family="matched_l1",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        matched_l1_lambda1=0.07,
    )


def _success_paths_for_prior_free(prefix: str, run_id: str) -> dict:
    return {
        "continuous_w_path": f"results/main_study/{prefix}/artefacts/{run_id}/continuous_w.npz",
        "thresholded_adjacency_path": f"results/main_study/{prefix}/artefacts/{run_id}/thresholded_adjacency.npz",
        "true_adjacency_path": f"results/main_study/{prefix}/artefacts/{run_id}/true_adjacency.npz",
        "interventions_mmd_path": f"results/main_study/{prefix}/artefacts/{run_id}/interventions_mmd.json",
    }


def _success_paths_for_soft_frobenius(prefix: str, run_id: str) -> dict:
    base = _success_paths_for_prior_free(prefix, run_id)
    base.update(
        confidence_mask_path=f"results/main_study/{prefix}/artefacts/{run_id}/confidence_mask.npz",
        prior_edge_set_clean_path=f"results/main_study/{prefix}/artefacts/{run_id}/prior_edge_set_clean.json",
        prior_edge_set_corrupted_path=f"results/main_study/{prefix}/artefacts/{run_id}/prior_edge_set_corrupted.json",
        per_edge_labels_path=f"results/main_study/{prefix}/artefacts/{run_id}/per_edge_labels.json",
    )
    return base


def _success_paths_for_hard_exclusion(prefix: str, run_id: str) -> dict:
    base = _success_paths_for_prior_free(prefix, run_id)
    base.update(
        prior_edge_set_clean_path=f"results/main_study/{prefix}/artefacts/{run_id}/prior_edge_set_clean.json",
        prior_edge_set_corrupted_path=f"results/main_study/{prefix}/artefacts/{run_id}/prior_edge_set_corrupted.json",
        per_edge_labels_path=f"results/main_study/{prefix}/artefacts/{run_id}/per_edge_labels.json",
    )
    return base


def _build_success_record(
    *,
    config: MainStudyConfig | None = None,
    **overrides,
) -> MainStudyRunRecord:
    cfg = config if config is not None else _build_prior_free_config()
    cfh = compute_configuration_hash(cfg)
    prefix = configuration_hash_prefix(cfg)
    rid = make_run_id(cfg)

    family = cfg.method_family
    if family == "soft_frobenius":
        path_kwargs = _success_paths_for_soft_frobenius(prefix, rid)
    elif family == "hard_exclusion":
        path_kwargs = _success_paths_for_hard_exclusion(prefix, rid)
    else:
        path_kwargs = _success_paths_for_prior_free(prefix, rid)

    base: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        configuration_hash_full=cfh,
        configuration_hash_prefix=prefix,
        run_id=rid,
        n_nodes=10,
        fit_status="success",
        graph_status="valid_dag",
        sampler_status="available",
        metric_status="computed",
        failure_kind=None,
        failure_message="",
        sid=12.0,
        shd=3.0,
        mmd=0.05,
        runtime_seconds=120.0,
        fit_runtime_seconds=100.0,
        metric_runtime_seconds=20.0,
        wrapper_diagnostics={"training_status": "converged", "n_iterations": 5000},
        parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
        generated_at_utc=_VALID_GENERATED_AT,
        code_version=None,
    )
    base.update(path_kwargs)
    base.update(overrides)
    return MainStudyRunRecord(**base)


def _build_failure_record(
    *,
    config: MainStudyConfig | None = None,
    **overrides,
) -> MainStudyRunRecord:
    cfg = config if config is not None else _build_prior_free_config()
    cfh = compute_configuration_hash(cfg)
    prefix = configuration_hash_prefix(cfg)
    rid = make_run_id(cfg)

    base: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        configuration_hash_full=cfh,
        configuration_hash_prefix=prefix,
        run_id=rid,
        n_nodes=10,
        fit_status="model_fit_failure",
        graph_status=None,
        sampler_status=None,
        metric_status="not_computed_due_to_fit_failure",
        failure_kind="non_convergence",
        failure_message="DAGMA stage 1 diverged",
        sid=None,
        shd=None,
        mmd=None,
        runtime_seconds=15.0,
        fit_runtime_seconds=15.0,
        metric_runtime_seconds=None,
        wrapper_diagnostics={"training_status": "diverged"},
        parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
        generated_at_utc=_VALID_GENERATED_AT,
        code_version=None,
    )
    base.update(overrides)
    return MainStudyRunRecord(**base)


# ---------------------------------------------------------------------------
# T-1: status constants
# ---------------------------------------------------------------------------


def test_fit_statuses_contents():
    assert FIT_STATUSES == (
        "success",
        "model_fit_failure",
        "infrastructure_failure_during_fit",
    )


def test_metric_statuses_contents_include_not_computed_due_to_fit_failure():
    assert METRIC_STATUSES == (
        "computed",
        "unavailable_graph_invalid",
        "unavailable_sampler_failure",
        "unavailable_dependency_missing",
        "not_computed_due_to_fit_failure",
    )
    assert "not_computed_due_to_fit_failure" in METRIC_STATUSES


def test_failure_kinds_contents():
    assert FAILURE_KINDS == (
        None,
        "non_convergence",
        "invalid_graph",
        "sampler_unavailable",
        "metric_unavailable",
        "infrastructure",
    )


# ---------------------------------------------------------------------------
# T-2: graph/sampler statuses imported from wrappers
# ---------------------------------------------------------------------------


def test_graph_status_values_match_wrapper_literal():
    assert GRAPH_STATUS_VALUES == tuple(get_args(GraphStatus))


def test_sampler_status_values_match_wrapper_literal():
    assert SAMPLER_STATUS_VALUES == tuple(get_args(SamplerStatus))


# ---------------------------------------------------------------------------
# T-3: frozen and keyword-only
# ---------------------------------------------------------------------------


def test_record_is_frozen():
    record = _build_success_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.sid = 99.0  # type: ignore[misc]


def test_record_is_keyword_only():
    """A positional argument must be rejected at construction time."""
    with pytest.raises(TypeError):
        MainStudyRunRecord(SCHEMA_VERSION)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# T-4: invalid schema_version
# ---------------------------------------------------------------------------


def test_invalid_schema_version_raises():
    with pytest.raises(ValueError, match="schema_version"):
        _build_success_record(schema_version=1)


# ---------------------------------------------------------------------------
# T-5: configuration_hash_full mismatch
# ---------------------------------------------------------------------------


def test_configuration_hash_full_mismatch_raises_with_both_values():
    cfg = _build_prior_free_config()
    real = compute_configuration_hash(cfg)
    fake = "f" * 64
    with pytest.raises(ValueError) as exc:
        _build_success_record(
            config=cfg,
            configuration_hash_full=fake,
        )
    msg = str(exc.value)
    assert fake in msg, msg
    assert real in msg, msg


# ---------------------------------------------------------------------------
# T-6: configuration_hash_prefix mismatch
# ---------------------------------------------------------------------------


def test_configuration_hash_prefix_mismatch_raises():
    with pytest.raises(ValueError, match="configuration_hash_prefix"):
        _build_success_record(configuration_hash_prefix="0123456789ab")


# ---------------------------------------------------------------------------
# T-7: invalid hash format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "g" * 64,
        "A" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_invalid_hash_format_raises(bad):
    with pytest.raises(ValueError, match="configuration_hash_full"):
        _build_success_record(configuration_hash_full=bad)


# ---------------------------------------------------------------------------
# T-8: run_id mismatch
# ---------------------------------------------------------------------------


def test_run_id_mismatch_raises():
    with pytest.raises(ValueError, match="run_id"):
        _build_success_record(run_id="wrong_run_id")


# ---------------------------------------------------------------------------
# T-9: parent hash mismatch
# ---------------------------------------------------------------------------


def test_parent_hash_mismatch_raises():
    with pytest.raises(ValueError, match="parent_heldout_run_hash_full"):
        _build_success_record(
            parent_heldout_run_hash_full=_OTHER_PARENT_HASH
        )


# ---------------------------------------------------------------------------
# T-10: invalid enum values
# ---------------------------------------------------------------------------


def test_invalid_fit_status_raises():
    with pytest.raises(ValueError, match="fit_status"):
        _build_success_record(fit_status="invalid_status")


def test_invalid_metric_status_raises():
    with pytest.raises(ValueError, match="metric_status"):
        _build_success_record(metric_status="invalid_metric_status")


def test_invalid_failure_kind_raises():
    with pytest.raises(ValueError, match="failure_kind"):
        _build_failure_record(failure_kind="unknown_failure")


def test_invalid_graph_status_raises():
    with pytest.raises(ValueError, match="graph_status"):
        _build_success_record(graph_status="not_a_graph_status")


def test_invalid_sampler_status_raises():
    with pytest.raises(ValueError, match="sampler_status"):
        _build_success_record(sampler_status="not_a_sampler_status")


# ---------------------------------------------------------------------------
# T-11: success record with computed metrics
# ---------------------------------------------------------------------------


def test_success_record_with_computed_metrics_validates():
    record = _build_success_record()
    assert record.fit_status == "success"
    assert record.metric_status == "computed"
    assert record.sid == pytest.approx(12.0)
    assert record.shd == pytest.approx(3.0)
    assert record.mmd == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# T-12: success with failure_kind or failure_message raises
# ---------------------------------------------------------------------------


def test_success_with_failure_kind_raises():
    with pytest.raises(ValueError, match="failure_kind"):
        _build_success_record(failure_kind="non_convergence")


def test_success_with_nonempty_failure_message_raises():
    with pytest.raises(ValueError, match="failure_message"):
        _build_success_record(failure_message="something went wrong")


# ---------------------------------------------------------------------------
# T-13: success with graph_status=None or sampler_status=None raises
# ---------------------------------------------------------------------------


def test_success_with_graph_status_none_raises():
    with pytest.raises(ValueError, match="graph_status"):
        _build_success_record(graph_status=None)


def test_success_with_sampler_status_none_raises():
    with pytest.raises(ValueError, match="sampler_status"):
        _build_success_record(sampler_status=None)


# ---------------------------------------------------------------------------
# T-14: failure with metric_status=computed raises
# ---------------------------------------------------------------------------


def test_failure_with_metric_status_computed_raises():
    with pytest.raises(ValueError, match="metric_status"):
        _build_failure_record(metric_status="computed")


# ---------------------------------------------------------------------------
# T-15: failure record requires failure_kind or failure_message
# ---------------------------------------------------------------------------


def test_failure_record_without_failure_kind_or_message_raises():
    with pytest.raises(ValueError, match="failure"):
        _build_failure_record(failure_kind=None, failure_message="")


# ---------------------------------------------------------------------------
# T-16: valid failure record with nullable metrics validates
# ---------------------------------------------------------------------------


def test_valid_failure_record_with_nullable_metrics_validates():
    record = _build_failure_record()
    assert record.fit_status == "model_fit_failure"
    assert record.sid is None
    assert record.shd is None
    assert record.mmd is None
    assert record.metric_runtime_seconds is None


def test_failure_record_accepts_failure_message_alone():
    record = _build_failure_record(
        failure_kind=None,
        failure_message="some boundary failure",
    )
    assert record.failure_kind is None
    assert record.failure_message == "some boundary failure"


# ---------------------------------------------------------------------------
# T-17: computed metric record with bad sid/shd/mmd raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.5])
def test_computed_record_rejects_bad_sid(bad):
    with pytest.raises(ValueError, match="sid"):
        _build_success_record(sid=bad)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.5])
def test_computed_record_rejects_bad_shd(bad):
    with pytest.raises(ValueError, match="shd"):
        _build_success_record(shd=bad)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.5])
def test_computed_record_rejects_bad_mmd(bad):
    with pytest.raises(ValueError, match="mmd"):
        _build_success_record(mmd=bad)


# ---------------------------------------------------------------------------
# T-18: non-computed metric record with any non-None sid/shd/mmd raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field, value", [
    ("sid", 1.0),
    ("shd", 0.0),
    ("mmd", 0.001),
])
def test_failure_record_with_nonNone_metric_raises(field, value):
    with pytest.raises(ValueError, match=field):
        _build_failure_record(**{field: value})


# ---------------------------------------------------------------------------
# T-19: timing validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.001, -10.0, float("nan"), float("inf"), float("-inf")])
def test_runtime_seconds_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="runtime_seconds"):
        _build_success_record(runtime_seconds=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_fit_runtime_seconds_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="fit_runtime_seconds"):
        _build_success_record(fit_runtime_seconds=bad)


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_metric_runtime_seconds_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="metric_runtime_seconds"):
        _build_success_record(metric_runtime_seconds=bad)


def test_metric_runtime_seconds_accepts_none_for_failure_record():
    record = _build_failure_record(metric_runtime_seconds=None)
    assert record.metric_runtime_seconds is None


# ---------------------------------------------------------------------------
# T-20: n_nodes type and positivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "10"])
def test_n_nodes_rejects_invalid(bad):
    with pytest.raises(ValueError, match="n_nodes"):
        _build_success_record(n_nodes=bad)


# ---------------------------------------------------------------------------
# T-21: generated_at_utc parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "2026-05-25T12:00:00Z",
        "2026-05-25T12:00:00+00:00",
        "2026-05-25T12:00:00-05:00",
        "2026-05-25T00:00:00.123456+00:00",
    ],
)
def test_generated_at_utc_accepts_timezone_aware_iso8601(good):
    record = _build_success_record(generated_at_utc=good)
    assert record.generated_at_utc == good


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "2026-05-25T12:00:00",
        "2026-05-25",
        "not a date",
        "2026/05/25 12:00",
    ],
)
def test_generated_at_utc_rejects_naive_or_malformed(bad):
    with pytest.raises(ValueError, match="generated_at_utc"):
        _build_success_record(generated_at_utc=bad)


# ---------------------------------------------------------------------------
# T-22: invalid artefact paths raise
# ---------------------------------------------------------------------------


def test_invalid_artefact_path_raises():
    with pytest.raises(ValueError, match="continuous_w_path"):
        _build_success_record(continuous_w_path="/absolute/forbidden.npz")


def test_artefact_path_with_dotdot_raises():
    with pytest.raises(ValueError, match="thresholded_adjacency_path"):
        _build_success_record(
            thresholded_adjacency_path="results/../etc/passwd"
        )


# ---------------------------------------------------------------------------
# T-23: success record requires three core artefact paths
# ---------------------------------------------------------------------------


def test_success_record_requires_continuous_w_path():
    with pytest.raises(ValueError, match="continuous_w_path"):
        _build_success_record(continuous_w_path=None)


def test_success_record_requires_thresholded_adjacency_path():
    with pytest.raises(ValueError, match="thresholded_adjacency_path"):
        _build_success_record(thresholded_adjacency_path=None)


def test_success_record_requires_true_adjacency_path():
    with pytest.raises(ValueError, match="true_adjacency_path"):
        _build_success_record(true_adjacency_path=None)


def test_computed_metric_record_requires_interventions_mmd_path():
    with pytest.raises(ValueError, match="interventions_mmd_path"):
        _build_success_record(interventions_mmd_path=None)


# ---------------------------------------------------------------------------
# T-24: soft_frobenius success requires prior paths and confidence_mask_path
# ---------------------------------------------------------------------------


def test_soft_frobenius_success_validates():
    cfg = _build_soft_frobenius_config()
    record = _build_success_record(config=cfg)
    assert record.confidence_mask_path is not None
    assert record.prior_edge_set_clean_path is not None
    assert record.prior_edge_set_corrupted_path is not None
    assert record.per_edge_labels_path is not None


@pytest.mark.parametrize(
    "missing",
    [
        "confidence_mask_path",
        "prior_edge_set_clean_path",
        "prior_edge_set_corrupted_path",
        "per_edge_labels_path",
    ],
)
def test_soft_frobenius_success_requires_each_prior_or_confidence_path(missing):
    cfg = _build_soft_frobenius_config()
    with pytest.raises(ValueError, match=missing):
        _build_success_record(config=cfg, **{missing: None})


# ---------------------------------------------------------------------------
# T-25: hard_exclusion success requires prior paths but rejects confidence_mask_path
# ---------------------------------------------------------------------------


def test_hard_exclusion_success_validates():
    cfg = _build_hard_exclusion_config()
    record = _build_success_record(config=cfg)
    assert record.confidence_mask_path is None
    assert record.prior_edge_set_clean_path is not None
    assert record.prior_edge_set_corrupted_path is not None
    assert record.per_edge_labels_path is not None


@pytest.mark.parametrize(
    "missing",
    [
        "prior_edge_set_clean_path",
        "prior_edge_set_corrupted_path",
        "per_edge_labels_path",
    ],
)
def test_hard_exclusion_success_requires_each_prior_path(missing):
    cfg = _build_hard_exclusion_config()
    with pytest.raises(ValueError, match=missing):
        _build_success_record(config=cfg, **{missing: None})


def test_hard_exclusion_rejects_confidence_mask_path():
    cfg = _build_hard_exclusion_config()
    bad_path = "results/main_study/abcdef012345/artefacts/r/confidence_mask.npz"
    with pytest.raises(ValueError, match="confidence_mask_path"):
        _build_success_record(config=cfg, confidence_mask_path=bad_path)


def test_hard_exclusion_failure_also_rejects_confidence_mask_path():
    cfg = _build_hard_exclusion_config()
    bad_path = "results/main_study/abcdef012345/artefacts/r/confidence_mask.npz"
    with pytest.raises(ValueError, match="confidence_mask_path"):
        _build_failure_record(config=cfg, confidence_mask_path=bad_path)


# ---------------------------------------------------------------------------
# T-26: prior_free / matched_l1 reject prior paths and confidence_mask_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family_cfg_builder", [
    _build_prior_free_config,
    _build_matched_l1_config,
])
@pytest.mark.parametrize("bad_field", [
    "confidence_mask_path",
    "prior_edge_set_clean_path",
    "prior_edge_set_corrupted_path",
    "per_edge_labels_path",
])
def test_prior_free_and_matched_l1_reject_priorbacked_paths(
    family_cfg_builder, bad_field
):
    cfg = family_cfg_builder()
    bad_value = "results/main_study/abcdef012345/artefacts/r/x.json"
    with pytest.raises(ValueError, match=bad_field):
        _build_success_record(config=cfg, **{bad_field: bad_value})


# ---------------------------------------------------------------------------
# T-27: wrapper_diagnostics must be dict
# ---------------------------------------------------------------------------


def test_wrapper_diagnostics_rejects_non_dict_mapping():
    """A custom mapping subclass must be rejected even if it
    quacks like a dict."""
    from collections import OrderedDict

    od = OrderedDict({"k": "v"})
    with pytest.raises(TypeError, match="wrapper_diagnostics"):
        _build_success_record(wrapper_diagnostics=od)


def test_wrapper_diagnostics_rejects_list():
    with pytest.raises(TypeError, match="wrapper_diagnostics"):
        _build_success_record(wrapper_diagnostics=[("k", "v")])


# ---------------------------------------------------------------------------
# T-28: wrapper_diagnostics is deep-copied
# ---------------------------------------------------------------------------


def test_wrapper_diagnostics_is_deep_copied():
    payload = {"nested": {"loss_history": [1.0, 2.0, 3.0]}}
    record = _build_success_record(wrapper_diagnostics=payload)
    # Mutate the caller's dict and the nested list after construction.
    payload["nested"]["loss_history"].append(99.0)
    payload["new_key"] = "should_not_appear"
    # Record's diagnostics must be unaffected.
    assert record.wrapper_diagnostics["nested"]["loss_history"] == [
        1.0,
        2.0,
        3.0,
    ]
    assert "new_key" not in record.wrapper_diagnostics


# ---------------------------------------------------------------------------
# T-29: non-canonicalisable diagnostics raise TypeError
# ---------------------------------------------------------------------------


def test_non_canonicalisable_diagnostics_raise_typeerror():
    arr = np.array([[True, False], [False, True]])
    with pytest.raises(TypeError):
        _build_success_record(wrapper_diagnostics={"adjacency": arr})


class _CustomObject:
    pass


def test_diagnostics_with_custom_object_raise_typeerror():
    with pytest.raises(TypeError):
        _build_success_record(wrapper_diagnostics={"obj": _CustomObject()})


# ---------------------------------------------------------------------------
# T-30: import allowlist for records.py
# ---------------------------------------------------------------------------


_RECORDS_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "experiments.selection_study",
    "experiments.main_study.calibration_lambda_prior",
    "dagma",
    "dcdi",
    "tests",
)


_RECORDS_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "copy",
    "dataclasses",
    "math",
    "re",
    "typing",
    "datetime",
    "numpy",
    "experiments.main_study.paths",
    "experiments.main_study.schema",
    "symbolic_priors_cd.wrappers.status",
})


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_records_module_does_not_import_forbidden_packages():
    src = Path(records_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _RECORDS_FORBIDDEN_IMPORT_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"records.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_records_module_imports_are_allowlisted():
    src = Path(records_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _RECORDS_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _RECORDS_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"records.py import {mod!r} is not in the allowlist "
            f"{sorted(_RECORDS_ALLOWED_PREFIXES)}."
        )


# ===========================================================================
# diagnostics_to_canonical
# ===========================================================================


def test_diagnostics_to_canonical_converts_ndarray_float():
    diag = {"arr": np.array([1.5, 2.5, 3.5], dtype=np.float64)}
    out = diagnostics_to_canonical(diag)
    assert out["arr"] == [1.5, 2.5, 3.5]
    assert all(type(v) is float for v in out["arr"])


def test_diagnostics_to_canonical_converts_ndarray_bool_to_python_bool():
    diag = {"m": np.array([[True, False], [False, True]], dtype=bool)}
    out = diagnostics_to_canonical(diag)
    assert out["m"] == [[True, False], [False, True]]
    # Python bool, not int.
    for row in out["m"]:
        for v in row:
            assert type(v) is bool, (
                f"expected Python bool, got {type(v).__name__}"
            )


def test_diagnostics_to_canonical_converts_numpy_scalars():
    diag = {
        "n_iter": np.int64(1234),
        "tol": np.float64(1e-7),
        "flag": np.bool_(True),
    }
    out = diagnostics_to_canonical(diag)
    assert out["n_iter"] == 1234
    assert type(out["n_iter"]) is int
    assert out["tol"] == pytest.approx(1e-7)
    assert type(out["tol"]) is float
    assert out["flag"] is True
    assert type(out["flag"]) is bool


def test_diagnostics_to_canonical_recurses_into_nested_structures():
    diag = {
        "outer": {
            "loss_history": [
                np.float64(0.5),
                np.float64(0.3),
                np.float64(0.1),
            ],
            "model_specific": {
                "thresholded_adjacency": np.array(
                    [[True, False], [False, True]]
                ),
                "n_iterations": np.int32(2500),
            },
        },
        "training_status": "converged",
        "loss_decomposition_final": (np.float64(0.01), np.float64(0.5)),
    }
    out = diagnostics_to_canonical(diag)
    assert out["outer"]["loss_history"] == [0.5, 0.3, 0.1]
    assert out["outer"]["model_specific"]["thresholded_adjacency"] == [
        [True, False],
        [False, True],
    ]
    assert out["outer"]["model_specific"]["n_iterations"] == 2500
    assert out["training_status"] == "converged"
    # Tuple gets converted to list.
    assert out["loss_decomposition_final"] == [0.01, 0.5]


def test_diagnostics_to_canonical_rejects_non_string_dict_key():
    diag = {"nested": {1: "value"}}
    with pytest.raises(TypeError, match="must be a string"):
        diagnostics_to_canonical(diag)


def test_diagnostics_to_canonical_rejects_unsupported_object_with_path():
    class _Custom:
        pass

    diag = {"a": {"b": [1, 2, _Custom()]}}
    with pytest.raises(TypeError) as exc:
        diagnostics_to_canonical(diag)
    msg = str(exc.value)
    # Path-aware error must locate the offending position.
    assert "diag" in msg
    assert "'a'" in msg
    assert "'b'" in msg
    assert "[2]" in msg
    assert "_Custom" in msg


def test_diagnostics_to_canonical_rejects_top_level_non_dict():
    with pytest.raises(TypeError, match="top level"):
        diagnostics_to_canonical(["not", "a", "dict"])  # type: ignore[arg-type]


def test_diagnostics_to_canonical_output_passes_canonicalize_for_json():
    """Output of diagnostics_to_canonical must be canonicalisable."""
    from experiments.main_study.schema import canonicalize_for_json

    diag = {
        "training_status": "converged",
        "loss_history": [np.float64(0.5), np.float64(0.1)],
        "thresholded_adjacency": np.array([[True]]),
    }
    out = diagnostics_to_canonical(diag)
    # canonicalize_for_json raises if anything is unsupported.
    canonicalize_for_json(out)


def test_diagnostics_to_canonical_does_not_mutate_input():
    arr = np.array([1.0, 2.0])
    diag = {"a": arr, "b": [arr]}
    snapshot_arr_data = arr.copy()
    snapshot_keys = list(diag.keys())

    _ = diagnostics_to_canonical(diag)

    # Original ndarray is unchanged.
    assert np.array_equal(arr, snapshot_arr_data)
    # Original dict still has the same keys.
    assert list(diag.keys()) == snapshot_keys
    # Original dict still holds the same arr object.
    assert diag["a"] is arr


def test_diagnostics_to_canonical_returns_new_dict():
    diag: dict = {}
    out = diagnostics_to_canonical(diag)
    assert out is not diag


# ===========================================================================
# derive_metric_status_for_failure
# ===========================================================================


@pytest.mark.parametrize(
    "fit_status",
    ["model_fit_failure", "infrastructure_failure_during_fit"],
)
def test_derive_metric_status_non_success_returns_not_computed(fit_status):
    out = derive_metric_status_for_failure(
        fit_status=fit_status,
        graph_status=None,
        sampler_status=None,
    )
    assert out == "not_computed_due_to_fit_failure"


def test_derive_metric_status_non_success_takes_precedence_over_graph():
    """Even if graph_status looks invalid, fit failure dominates."""
    out = derive_metric_status_for_failure(
        fit_status="model_fit_failure",
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
    )
    assert out == "not_computed_due_to_fit_failure"


@pytest.mark.parametrize(
    "graph_status",
    ["cyclic", "bidirected", "self_loop", "invalid_shape"],
)
def test_derive_metric_status_success_plus_invalid_graph(graph_status):
    out = derive_metric_status_for_failure(
        fit_status="success",
        graph_status=graph_status,
        sampler_status=None,
    )
    assert out == "unavailable_graph_invalid"


@pytest.mark.parametrize(
    "sampler_status",
    [
        "unavailable_invalid_graph",
        "unavailable_no_api",
        "unavailable_unresolved_noise_policy",
    ],
)
def test_derive_metric_status_success_valid_dag_sampler_unavailable(
    sampler_status,
):
    out = derive_metric_status_for_failure(
        fit_status="success",
        graph_status="valid_dag",
        sampler_status=sampler_status,
    )
    assert out == "unavailable_sampler_failure"


def test_derive_metric_status_success_valid_dag_available_raises():
    with pytest.raises(ValueError, match="success-with-available"):
        derive_metric_status_for_failure(
            fit_status="success",
            graph_status="valid_dag",
            sampler_status="available",
        )


def test_derive_metric_status_success_none_graph_none_sampler_raises():
    with pytest.raises(ValueError, match="success-with-available"):
        derive_metric_status_for_failure(
            fit_status="success",
            graph_status=None,
            sampler_status=None,
        )


def test_derive_metric_status_rejects_invalid_fit_status_field():
    with pytest.raises(ValueError, match="fit_status"):
        derive_metric_status_for_failure(
            fit_status="bogus_fit",
            graph_status=None,
            sampler_status=None,
        )


def test_derive_metric_status_rejects_invalid_graph_status_field():
    with pytest.raises(ValueError, match="graph_status"):
        derive_metric_status_for_failure(
            fit_status="success",
            graph_status="not_a_graph_status",
            sampler_status=None,
        )


def test_derive_metric_status_rejects_invalid_sampler_status_field():
    with pytest.raises(ValueError, match="sampler_status"):
        derive_metric_status_for_failure(
            fit_status="success",
            graph_status="valid_dag",
            sampler_status="not_a_sampler_status",
        )


# ===========================================================================
# make_failure_record
# ===========================================================================


def _failure_kwargs(
    *,
    config: MainStudyConfig | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else _build_prior_free_config()
    return dict(
        config=cfg,
        n_nodes=10,
        fit_status="model_fit_failure",
        failure_kind="non_convergence",
        failure_message="DAGMA stage 1 diverged",
        runtime_seconds=15.0,
        fit_runtime_seconds=15.0,
        wrapper_diagnostics={"training_status": "diverged"},
        generated_at_utc=_VALID_GENERATED_AT,
    )


def test_make_failure_record_model_fit_failure_returns_valid_record():
    cfg = _build_prior_free_config()
    record = make_failure_record(**_failure_kwargs(config=cfg))
    assert record.fit_status == "model_fit_failure"
    assert record.metric_status == "not_computed_due_to_fit_failure"
    assert record.sid is None
    assert record.shd is None
    assert record.mmd is None
    assert record.metric_runtime_seconds is None


def test_make_failure_record_derives_graph_invalid_downstream_metric_status():
    cfg = _build_prior_free_config()
    record = make_failure_record(
        config=cfg,
        n_nodes=10,
        fit_status="success",
        failure_kind=None,
        failure_message="",
        runtime_seconds=120.0,
        fit_runtime_seconds=100.0,
        wrapper_diagnostics={"training_status": "converged"},
        generated_at_utc=_VALID_GENERATED_AT,
        graph_status="cyclic",
        sampler_status="unavailable_invalid_graph",
        # interventions_mmd_path can be omitted because metric_status
        # is derived to "unavailable_graph_invalid".
        continuous_w_path="results/main_study/abcdef012345/artefacts/r/continuous_w.npz",
        thresholded_adjacency_path="results/main_study/abcdef012345/artefacts/r/thresholded_adjacency.npz",
        true_adjacency_path="results/main_study/abcdef012345/artefacts/r/true_adjacency.npz",
    )
    assert record.metric_status == "unavailable_graph_invalid"
    assert record.fit_status == "success"
    assert record.graph_status == "cyclic"
    assert record.sampler_status == "unavailable_invalid_graph"
    assert record.sid is None
    assert record.shd is None
    assert record.mmd is None


def test_make_failure_record_success_equivalent_with_metric_status_none_raises():
    cfg = _build_prior_free_config()
    with pytest.raises(ValueError, match="noncomputed condition"):
        make_failure_record(
            config=cfg,
            n_nodes=10,
            fit_status="success",
            failure_kind=None,
            failure_message="",
            runtime_seconds=1.0,
            fit_runtime_seconds=1.0,
            wrapper_diagnostics={},
            generated_at_utc=_VALID_GENERATED_AT,
            graph_status="valid_dag",
            sampler_status="available",
        )


def test_make_failure_record_prior_free_with_prior_path_raises_through_record():
    """prior_free record must reject prior artefact paths even on failure."""
    cfg = _build_prior_free_config()
    bad_path = "results/main_study/abcdef012345/artefacts/r/prior_edge_set_clean.json"
    with pytest.raises(ValueError, match="prior_edge_set_clean_path"):
        make_failure_record(
            **_failure_kwargs(config=cfg),
            prior_edge_set_clean_path=bad_path,
        )


def test_make_failure_record_soft_frobenius_failure_may_omit_prior_paths():
    """soft_frobenius failure records do not require prior paths."""
    cfg = _build_soft_frobenius_config()
    kwargs = _failure_kwargs(config=cfg)
    # No prior paths supplied; the record must still validate.
    record = make_failure_record(**kwargs)
    assert record.config.method_family == "soft_frobenius"
    assert record.fit_status == "model_fit_failure"
    assert record.confidence_mask_path is None
    assert record.prior_edge_set_clean_path is None
    assert record.prior_edge_set_corrupted_path is None
    assert record.per_edge_labels_path is None


def test_make_failure_record_derives_hashes_and_run_id_from_config():
    cfg = _build_soft_frobenius_config()
    record = make_failure_record(**_failure_kwargs(config=cfg))
    assert record.configuration_hash_full == compute_configuration_hash(cfg)
    assert record.configuration_hash_prefix == configuration_hash_prefix(cfg)
    assert record.run_id == make_run_id(cfg)
    assert (
        record.parent_heldout_run_hash_full
        == cfg.parent_heldout_run_hash_full
    )


def test_make_failure_record_canonicalises_and_deep_copies_diagnostics():
    """Diagnostics with numpy arrays must be canonicalised, and the
    caller's input must not survive into the record after the helper
    completes."""
    cfg = _build_prior_free_config()
    payload = {
        "training_status": "diverged",
        "loss_history": np.array([0.5, 0.3, 0.1]),
        "model_specific": {
            "n_iterations": np.int32(2500),
        },
    }
    record = make_failure_record(
        config=cfg,
        n_nodes=10,
        fit_status="model_fit_failure",
        failure_kind="non_convergence",
        failure_message="diverged",
        runtime_seconds=1.0,
        fit_runtime_seconds=1.0,
        wrapper_diagnostics=payload,
        generated_at_utc=_VALID_GENERATED_AT,
    )
    # Canonicalised: ndarray -> list, np.int32 -> int.
    assert record.wrapper_diagnostics["loss_history"] == [0.5, 0.3, 0.1]
    assert isinstance(
        record.wrapper_diagnostics["loss_history"], list
    )
    assert record.wrapper_diagnostics["model_specific"]["n_iterations"] == 2500
    assert type(
        record.wrapper_diagnostics["model_specific"]["n_iterations"]
    ) is int

    # Mutating the caller's input does not change the record.
    payload["new_top_key"] = "should_not_appear"
    payload["model_specific"]["n_iterations"] = 999999
    assert "new_top_key" not in record.wrapper_diagnostics
    assert (
        record.wrapper_diagnostics["model_specific"]["n_iterations"] == 2500
    )


def test_make_failure_record_same_inputs_produce_equal_records():
    cfg = _build_prior_free_config()
    kwargs = _failure_kwargs(config=cfg)
    a = make_failure_record(**copy.deepcopy(kwargs))
    b = make_failure_record(**copy.deepcopy(kwargs))
    assert a == b


def test_make_failure_record_infrastructure_failure_path():
    cfg = _build_prior_free_config()
    record = make_failure_record(
        config=cfg,
        n_nodes=10,
        fit_status="infrastructure_failure_during_fit",
        failure_kind="infrastructure",
        failure_message="disk write failed",
        runtime_seconds=0.5,
        fit_runtime_seconds=0.5,
        wrapper_diagnostics={},
        generated_at_utc=_VALID_GENERATED_AT,
    )
    assert record.fit_status == "infrastructure_failure_during_fit"
    assert record.metric_status == "not_computed_due_to_fit_failure"
    assert record.failure_kind == "infrastructure"
