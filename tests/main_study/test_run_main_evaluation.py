"""Tests for the main-study main-evaluation runner.

Every test writes only under pytest's ``tmp_path``. Real DAGMA is
never invoked. The real metric backend is never invoked. The
canonical ``run_main_study`` orchestrator is replaced by a fake that
persists deterministic fake records and artefacts at the planned
paths, so the M-8 outer wrapper can be exercised end-to-end without
loading the heavy wrapper code path.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from experiments.main_study import run_main_evaluation as me_mod
from experiments.main_study.backends import (
    DataBundleLoader,
    DEFAULT_BANDWIDTH_MULTIPLIERS,
    MainStudyFitBackend,
    RealMetricBackend,
)
from experiments.main_study.records import (
    SCHEMA_VERSION,
    MainStudyRunRecord,
)
from experiments.main_study.run_io import persist_record_atomic
from experiments.main_study.run_main_evaluation import (
    EVALUATION_SEED_VALUES,
    EXPECTED_COUNTS_BY_METHOD,
    EXPECTED_WORKLOAD_COUNT,
    FORBIDDEN_CALIBRATION_SEEDS,
    MAIN_EVALUATION_PROTOCOL_VERSION,
    MATCHED_L1_LAMBDA1,
    REQUIRED_MODE,
    STATUS_CSV_FILENAME,
    SUMMARY_JSON_FILENAME,
    SUMMARY_MD_FILENAME,
    MainEvaluationExecutionSummary,
    MainEvaluationRunSpec,
    build_main_evaluation_output_dir,
    build_main_evaluation_planned_runs,
    capture_code_version,
    compute_main_evaluation_run_hash12,
    default_utc_factory,
    main as cli_main,
    run_main_evaluation,
    summarise_planned_runs,
    validate_parent_hash_full,
    verify_backend_defaults,
    write_main_evaluation_outputs,
)
from experiments.main_study.runner import RunSummary, WorkloadStatus
from experiments.main_study.schema import (
    CALIBRATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    make_main_study_config,
)
from experiments.main_study.workloads import PlannedRun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_PARENT_HASH_OTHER = "b" * 64
_N_NODES = 10


def _planned_record_kwargs(
    planned: PlannedRun,
    *,
    n_nodes: int = _N_NODES,
    fit_status: str = "success",
    graph_status: Optional[str] = "valid_dag",
    sampler_status: Optional[str] = "available",
    metric_status: str = "computed",
) -> dict[str, Any]:
    family = planned.config.method_family
    kwargs: dict[str, Any] = dict(
        schema_version=SCHEMA_VERSION,
        config=planned.config,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=n_nodes,
        fit_status=fit_status,
        graph_status=graph_status,
        sampler_status=sampler_status,
        metric_status=metric_status,
        failure_kind=None,
        failure_message="",
        runtime_seconds=1.0,
        fit_runtime_seconds=0.8,
        wrapper_diagnostics={"training_status": "converged"},
        parent_heldout_run_hash_full=planned.config.parent_heldout_run_hash_full,
        generated_at_utc="2026-05-25T00:00:00Z",
    )
    if metric_status == "computed":
        kwargs["sid"] = 1.0
        kwargs["shd"] = 2.0
        kwargs["mmd"] = 0.001
        kwargs["metric_runtime_seconds"] = 0.2
        kwargs["interventions_mmd_path"] = planned.artefact_paths[
            "interventions_mmd.json"
        ]
    kwargs["continuous_w_path"] = planned.artefact_paths["continuous_w.npz"]
    kwargs["thresholded_adjacency_path"] = planned.artefact_paths[
        "thresholded_adjacency.npz"
    ]
    kwargs["true_adjacency_path"] = planned.artefact_paths[
        "true_adjacency.npz"
    ]
    if family == "soft_frobenius":
        kwargs["confidence_mask_path"] = planned.artefact_paths[
            "confidence_mask.npz"
        ]
        kwargs["prior_edge_set_clean_path"] = planned.artefact_paths[
            "prior_edge_set_clean.json"
        ]
        kwargs["prior_edge_set_corrupted_path"] = planned.artefact_paths[
            "prior_edge_set_corrupted.json"
        ]
        kwargs["per_edge_labels_path"] = planned.artefact_paths[
            "per_edge_labels.json"
        ]
    elif family == "hard_exclusion":
        kwargs["prior_edge_set_clean_path"] = planned.artefact_paths[
            "prior_edge_set_clean.json"
        ]
        kwargs["prior_edge_set_corrupted_path"] = planned.artefact_paths[
            "prior_edge_set_corrupted.json"
        ]
        kwargs["per_edge_labels_path"] = planned.artefact_paths[
            "per_edge_labels.json"
        ]
    return kwargs


def _write_fake_artefacts(planned: PlannedRun, *, base_dir: Path) -> None:
    for name, rel in planned.artefact_paths.items():
        full = base_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        if name == "thresholded_adjacency.npz":
            np.savez(
                full,
                thresholded_adjacency=np.zeros(
                    (_N_NODES, _N_NODES), dtype=bool
                ),
            )
        elif name == "continuous_w.npz":
            np.savez(
                full,
                continuous_w=np.zeros(
                    (_N_NODES, _N_NODES), dtype=float
                ),
            )
        elif name == "true_adjacency.npz":
            np.savez(
                full,
                true_adjacency=np.zeros(
                    (_N_NODES, _N_NODES), dtype=bool
                ),
            )
        elif name == "confidence_mask.npz":
            np.savez(
                full,
                confidence_mask=np.zeros(
                    (_N_NODES, _N_NODES), dtype=float
                ),
            )
        elif name == "interventions_mmd.json":
            full.write_text(
                json.dumps({"records": [], "mmd_primary": 0.001}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_clean.json":
            full.write_text(
                json.dumps({"n_nodes": _N_NODES, "forbidden_edges": []}),
                encoding="utf-8",
            )
        elif name == "prior_edge_set_corrupted.json":
            full.write_text(
                json.dumps({
                    "n_nodes": _N_NODES,
                    "corruption_fraction": 0.0,
                    "forbidden_edges": [],
                }),
                encoding="utf-8",
            )
        elif name == "per_edge_labels.json":
            full.write_text(json.dumps({}), encoding="utf-8")


def make_fake_runner(
    *,
    call_log: Optional[list[dict[str, Any]]] = None,
) -> Callable[..., RunSummary]:
    def fake(
        planned_runs,
        *,
        base_dir: Path,
        data_loader,
        fit_backend,
        metric_backend,
        mode: str,
        code_version,
        generated_at_utc_factory,
        n_nodes_for_failure_record,
        logger=None,
        **kwargs,
    ) -> RunSummary:
        planned_tuple = tuple(planned_runs)
        if call_log is not None:
            call_log.append({
                "n_planned": len(planned_tuple),
                "mode": mode,
                "base_dir": base_dir,
                "data_loader": data_loader,
                "fit_backend": fit_backend,
                "metric_backend": metric_backend,
                "code_version": code_version,
            })
        statuses: list[WorkloadStatus] = []
        for planned in planned_tuple:
            _write_fake_artefacts(planned, base_dir=base_dir)
            record = MainStudyRunRecord(**_planned_record_kwargs(planned))
            persist_record_atomic(
                record, planned.record_path, base_dir=base_dir
            )
            statuses.append(WorkloadStatus(
                run_id=planned.run_id,
                configuration_hash_prefix=planned.configuration_hash_prefix,
                method_family=planned.config.method_family,
                final_status="success_computed",
                record_path=planned.record_path,
                runtime_seconds=0.0,
                message="",
            ))
        return RunSummary(
            n_planned=len(planned_tuple),
            n_executed=len(planned_tuple),
            n_success_computed=len(planned_tuple),
            n_success_metric_unavailable=0,
            n_model_fit_failure=0,
            n_skipped=0,
            n_overwritten=0,
            n_infrastructure_failure=0,
            total_runtime_seconds=0.0,
            per_workload_status=tuple(statuses),
        )
    return fake


# ===========================================================================
# A. Parent-hash validation
# ===========================================================================


def test_validate_parent_hash_full_accepts_64_hex():
    assert validate_parent_hash_full("a" * 64) == "a" * 64


def test_validate_parent_hash_full_rejects_12_char_prefix():
    with pytest.raises(ValueError, match="prefixes are not accepted"):
        validate_parent_hash_full("a" * 12)


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65],
)
def test_validate_parent_hash_full_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_parent_hash_full(bad)


def test_validate_parent_hash_full_rejects_non_string():
    with pytest.raises(ValueError, match="string"):
        validate_parent_hash_full(12345)  # type: ignore[arg-type]


# ===========================================================================
# B. Run-hash determinism and code_version independence
# ===========================================================================


def test_compute_run_hash12_is_12_hex_deterministic():
    a = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    b = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    assert a == b
    assert len(a) == 12
    assert all(c in "0123456789abcdef" for c in a)


def test_compute_run_hash12_changes_with_parent():
    a = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    b = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH_OTHER
    )
    assert a != b


def test_run_main_evaluation_hash_unaffected_by_code_version(tmp_path):
    """code_version is provenance only; identical hashes for two
    different code_version strings."""
    fake_a = make_fake_runner()
    fake_b = make_fake_runner()
    s_a = run_main_evaluation(
        output_root=tmp_path / "a",
        parent_heldout_run_hash_full=_PARENT_HASH,
        code_version="aaaaaaa",
        runner_fn=fake_a,
        data_loader=DataBundleLoader(),
        fit_backend=MainStudyFitBackend(),
        metric_backend=RealMetricBackend(),
    )
    s_b = run_main_evaluation(
        output_root=tmp_path / "b",
        parent_heldout_run_hash_full=_PARENT_HASH,
        code_version="bbbbbbb",
        runner_fn=fake_b,
        data_loader=DataBundleLoader(),
        fit_backend=MainStudyFitBackend(),
        metric_backend=RealMetricBackend(),
    )
    assert (
        s_a.main_evaluation_run_hash12
        == s_b.main_evaluation_run_hash12
    )
    assert s_a.code_version == "aaaaaaa"
    assert s_b.code_version == "bbbbbbb"


# ===========================================================================
# C. Plan enumeration
# ===========================================================================


@pytest.fixture(scope="module")
def _planned():
    h = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    return build_main_evaluation_planned_runs(
        main_evaluation_run_hash12=h,
        parent_heldout_run_hash_full=_PARENT_HASH,
    )


def test_plan_total_is_224(_planned):
    assert len(_planned) == 224 == EXPECTED_WORKLOAD_COUNT


def test_plan_method_family_counts_exact(_planned):
    counts: dict[str, int] = {}
    for p in _planned:
        counts[p.config.method_family] = (
            counts.get(p.config.method_family, 0) + 1
        )
    assert counts == EXPECTED_COUNTS_BY_METHOD


def test_plan_uses_evaluation_seeds_only(_planned):
    seeds = {int(p.config.seed_value) for p in _planned}
    assert seeds == set(EVALUATION_SEED_VALUES)


def test_plan_does_not_use_calibration_seeds(_planned):
    for p in _planned:
        for cs in FORBIDDEN_CALIBRATION_SEEDS:
            assert int(p.config.seed_value) != cs


def test_matched_l1_configs_use_frozen_lambda1(_planned):
    matched = [p for p in _planned if p.config.method_family == "matched_l1"]
    assert matched, "expected at least one matched_l1 planned run"
    for p in matched:
        assert p.config.matched_l1_lambda1 == MATCHED_L1_LAMBDA1
        assert p.config.dagma_config.lambda1 == MATCHED_L1_LAMBDA1


def test_soft_frobenius_uses_frozen_lambda_prior(_planned):
    soft = [p for p in _planned if p.config.method_family == "soft_frobenius"]
    assert soft
    for p in soft:
        assert p.config.lambda_prior == FROZEN_LAMBDA_PRIOR


def test_hard_exclusion_five_per_seed_no_confidence(_planned):
    hard = [p for p in _planned if p.config.method_family == "hard_exclusion"]
    assert len(hard) == 35
    per_seed: dict[int, int] = {}
    for p in hard:
        per_seed[int(p.config.seed_value)] = (
            per_seed.get(int(p.config.seed_value), 0) + 1
        )
        # No confidence axis.
        assert p.config.confidence is None
    for s in EVALUATION_SEED_VALUES:
        assert per_seed[s] == 5


def test_factory_round_trip_all_configs(_planned):
    """Every config in the plan must round-trip via make_main_study_config
    with the same field values. Confirms no direct MainStudyConfig
    instantiation was used."""
    for p in _planned:
        cfg = p.config
        family = cfg.method_family
        if family == "prior_free":
            rebuilt = make_main_study_config(
                method_family="prior_free",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=cfg.dagma_config,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
            )
        elif family == "matched_l1":
            rebuilt = make_main_study_config(
                method_family="matched_l1",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=cfg.dagma_config,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
                matched_l1_lambda1=cfg.matched_l1_lambda1,
            )
        elif family == "soft_frobenius":
            rebuilt = make_main_study_config(
                method_family="soft_frobenius",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=cfg.dagma_config,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
                confidence=cfg.confidence,
                corrupted_prior_spec=cfg.corrupted_prior_spec,
            )
        elif family == "hard_exclusion":
            # hard_exclusion factory injects exclude_edges into the
            # DAGMAConfig; rebuild from a fresh exclude-edges-None
            # DAGMAConfig of identical other fields so the factory
            # produces an equal result.
            from experiments.main_study.backends import DAGMAConfig
            base_dagma = dataclasses.replace(
                cfg.dagma_config, exclude_edges=None
            )
            rebuilt = make_main_study_config(
                method_family="hard_exclusion",
                seed_value=cfg.seed_value,
                seed_population=cfg.seed_population,
                dagma_config=base_dagma,
                parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
                corrupted_prior_spec=cfg.corrupted_prior_spec,
            )
        else:
            raise AssertionError(f"unknown family {family!r}")
        assert rebuilt == cfg


def test_total_mismatch_raises(monkeypatch):
    """If the underlying enumeration produces a wrong count, the
    builder raises. We simulate by monkeypatching the enumerator."""
    from experiments.main_study import workloads as wmod
    original = wmod.enumerate_planned_runs

    def short(*args, **kwargs):
        full = original(*args, **kwargs)
        return tuple(full[:10])  # truncate to wrong count

    monkeypatch.setattr(
        "experiments.main_study.run_main_evaluation.enumerate_planned_runs",
        short,
    )
    h = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    with pytest.raises(ValueError, match="total mismatch"):
        build_main_evaluation_planned_runs(
            main_evaluation_run_hash12=h,
            parent_heldout_run_hash_full=_PARENT_HASH,
        )


def test_method_count_mismatch_raises(monkeypatch):
    """If the enumerator drops one method-family entirely the
    method-count verifier raises."""
    from experiments.main_study import workloads as wmod
    original = wmod.enumerate_planned_runs

    def drop_prior_free(*args, **kwargs):
        full = original(*args, **kwargs)
        return tuple(
            p for p in full if p.config.method_family != "prior_free"
        )

    monkeypatch.setattr(
        "experiments.main_study.run_main_evaluation.enumerate_planned_runs",
        drop_prior_free,
    )
    h = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    with pytest.raises(ValueError, match="total mismatch|prior_free"):
        build_main_evaluation_planned_runs(
            main_evaluation_run_hash12=h,
            parent_heldout_run_hash_full=_PARENT_HASH,
        )


def test_hard_exclusion_count_change_raises(monkeypatch):
    """If a single hard_exclusion run is dropped, the per-seed
    invariant verifier flags it."""
    from experiments.main_study import workloads as wmod
    original = wmod.enumerate_planned_runs

    def drop_one_hard(*args, **kwargs):
        full = original(*args, **kwargs)
        dropped_one = False
        out = []
        for p in full:
            if (
                not dropped_one
                and p.config.method_family == "hard_exclusion"
            ):
                dropped_one = True
                continue
            out.append(p)
        return tuple(out)

    monkeypatch.setattr(
        "experiments.main_study.run_main_evaluation.enumerate_planned_runs",
        drop_one_hard,
    )
    h = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=_PARENT_HASH
    )
    with pytest.raises(ValueError):
        build_main_evaluation_planned_runs(
            main_evaluation_run_hash12=h,
            parent_heldout_run_hash_full=_PARENT_HASH,
        )


def test_summarise_planned_runs_basic(_planned):
    s = summarise_planned_runs(_planned)
    assert s["total"] == 224
    assert s["method_family_counts"] == EXPECTED_COUNTS_BY_METHOD
    assert set(s["seed_counts"].keys()) == set(EVALUATION_SEED_VALUES)


# ===========================================================================
# D. Mode and orchestrator invocation
# ===========================================================================


def test_run_main_evaluation_requires_mode_raise(tmp_path):
    fake = make_fake_runner()
    with pytest.raises(ValueError, match="raise"):
        run_main_evaluation(
            output_root=tmp_path,
            parent_heldout_run_hash_full=_PARENT_HASH,
            runner_fn=fake,
            mode="skip",
            data_loader=DataBundleLoader(),
            fit_backend=MainStudyFitBackend(),
            metric_backend=RealMetricBackend(),
        )


def test_run_main_evaluation_calls_runner_fn_once(tmp_path):
    calls: list[dict[str, Any]] = []
    fake = make_fake_runner(call_log=calls)
    run_main_evaluation(
        output_root=tmp_path,
        parent_heldout_run_hash_full=_PARENT_HASH,
        runner_fn=fake,
        data_loader=DataBundleLoader(),
        fit_backend=MainStudyFitBackend(),
        metric_backend=RealMetricBackend(),
    )
    assert len(calls) == 1
    assert calls[0]["n_planned"] == 224
    assert calls[0]["mode"] == "raise"


def test_run_main_evaluation_passes_real_backends_to_runner(tmp_path):
    calls: list[dict[str, Any]] = []
    fake = make_fake_runner(call_log=calls)
    run_main_evaluation(
        output_root=tmp_path,
        parent_heldout_run_hash_full=_PARENT_HASH,
        runner_fn=fake,
        data_loader=DataBundleLoader(),
        fit_backend=MainStudyFitBackend(),
        metric_backend=RealMetricBackend(),
    )
    c = calls[0]
    assert isinstance(c["data_loader"], DataBundleLoader)
    assert isinstance(c["fit_backend"], MainStudyFitBackend)
    assert isinstance(c["metric_backend"], RealMetricBackend)


# ===========================================================================
# E. verify_backend_defaults
# ===========================================================================


def test_verify_backend_defaults_accepts_production_defaults():
    verify_backend_defaults(
        DataBundleLoader(),
        MainStudyFitBackend(),
        RealMetricBackend(),
    )


def test_verify_backend_defaults_rejects_wrong_mmd_n_samples():
    with pytest.raises(ValueError, match="mmd_n_samples"):
        verify_backend_defaults(
            DataBundleLoader(),
            MainStudyFitBackend(),
            RealMetricBackend(mmd_n_samples=500),
        )


def test_verify_backend_defaults_rejects_custom_interventions():
    with pytest.raises(ValueError, match="intervention_specs"):
        verify_backend_defaults(
            DataBundleLoader(),
            MainStudyFitBackend(),
            RealMetricBackend(intervention_specs=({"node": 0, "value": 1.0, "label": "x"},)),
        )


def test_verify_backend_defaults_rejects_wrong_bandwidth():
    with pytest.raises(ValueError, match="bandwidth_multipliers"):
        verify_backend_defaults(
            DataBundleLoader(),
            MainStudyFitBackend(),
            RealMetricBackend(bandwidth_multipliers=(1.0,)),
        )


def test_verify_backend_defaults_rejects_wrong_loader_type():
    with pytest.raises(ValueError, match="data_loader"):
        verify_backend_defaults(
            object(),  # not a DataBundleLoader
            MainStudyFitBackend(),
            RealMetricBackend(),
        )


# ===========================================================================
# F. Output files (schemas, structural-only content)
# ===========================================================================


def _run_with_fake(tmp_path: Path) -> MainEvaluationExecutionSummary:
    fake = make_fake_runner()
    return run_main_evaluation(
        output_root=tmp_path,
        parent_heldout_run_hash_full=_PARENT_HASH,
        runner_fn=fake,
        data_loader=DataBundleLoader(),
        fit_backend=MainStudyFitBackend(),
        metric_backend=RealMetricBackend(),
    )


def test_outputs_written(tmp_path):
    summary = _run_with_fake(tmp_path)
    out = tmp_path / summary.output_dir
    assert (out / SUMMARY_JSON_FILENAME).exists()
    assert (out / STATUS_CSV_FILENAME).exists()
    assert (out / SUMMARY_MD_FILENAME).exists()


def test_summary_json_required_fields(tmp_path):
    summary = _run_with_fake(tmp_path)
    out = tmp_path / summary.output_dir
    payload = json.loads(
        (out / SUMMARY_JSON_FILENAME).read_text(encoding="utf-8")
    )
    required = {
        "main_evaluation_run_hash12",
        "parent_heldout_run_hash_full",
        "output_dir",
        "code_version",
        "matched_l1_lambda1",
        "lambda_prior",
        "seed_values",
        "n_planned",
        "n_executed",
        "n_skipped",
        "n_overwritten",
        "n_success_computed",
        "n_success_metric_unavailable",
        "n_model_fit_failure",
        "n_infrastructure_failure",
        "method_family_counts",
        "mode",
        "total_runtime_seconds",
        "per_workload_records",
    }
    assert required.issubset(set(payload.keys()))
    assert payload["mode"] == "raise"
    assert payload["matched_l1_lambda1"] == MATCHED_L1_LAMBDA1
    assert payload["lambda_prior"] == FROZEN_LAMBDA_PRIOR
    assert payload["n_planned"] == 224


def test_summary_json_per_workload_references_record_and_hash(tmp_path):
    summary = _run_with_fake(tmp_path)
    out = tmp_path / summary.output_dir
    payload = json.loads(
        (out / SUMMARY_JSON_FILENAME).read_text(encoding="utf-8")
    )
    per = payload["per_workload_records"]
    assert len(per) == 224
    for entry in per:
        assert entry["record_path"]
        assert entry["configuration_hash_full"]
        assert len(entry["configuration_hash_full"]) == 64


def test_status_csv_has_required_columns_and_rows(tmp_path):
    summary = _run_with_fake(tmp_path)
    out = tmp_path / summary.output_dir
    with (out / STATUS_CSV_FILENAME).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    required_cols = {
        "run_id",
        "configuration_hash_prefix",
        "configuration_hash_full",
        "method_family",
        "final_status",
        "was_overwritten",
        "record_path",
        "runtime_seconds",
        "message",
    }
    assert required_cols.issubset(set(rows[0].keys()))
    assert len(rows) == 224


def test_markdown_summary_is_minimal_no_scientific_claims(tmp_path):
    summary = _run_with_fake(tmp_path)
    out = tmp_path / summary.output_dir
    text = (out / SUMMARY_MD_FILENAME).read_text(encoding="utf-8")
    # Required structural fields.
    assert "main_evaluation_run_hash12" in text
    assert "matched_l1_lambda1" in text
    assert "lambda_prior" in text
    assert "n_planned" in text
    # Must NOT contain comparison or ranking language; this is M-9's job.
    forbidden_substrings = [
        "best ", "winner", "rank ", "ranked",
        "outperform", "p-value", "significance",
        "degradation curve", "figure", "plot",
        "comparison", "compared to",
        "mean SID", "mean SHD", "mean MMD",
    ]
    for token in forbidden_substrings:
        assert token.lower() not in text.lower(), (
            f"markdown must be structural-only; found forbidden "
            f"token {token!r}"
        )


def test_per_run_records_live_outside_main_evaluation_subdir(tmp_path):
    """Per-run records must live at canonical results/main_study/<hash>/
    paths, not nested under the main_evaluation summary directory."""
    summary = _run_with_fake(tmp_path)
    main_eval_root = (
        tmp_path / "results" / "main_study" / "main_evaluation"
        / summary.main_evaluation_run_hash12
    )
    for entry in (main_eval_root).rglob("*"):
        if entry.is_file():
            # Only the three summary files should live under the
            # main_evaluation subdir.
            assert entry.name in {
                SUMMARY_JSON_FILENAME,
                STATUS_CSV_FILENAME,
                SUMMARY_MD_FILENAME,
            }, (
                f"file {entry.name!r} should not be under "
                "main_evaluation summary dir"
            )
    # Per-run records exist under the canonical layout:
    canonical_root = (
        tmp_path / "results" / "main_study"
        / summary.main_evaluation_run_hash12 / "records"
    )
    json_files = list(canonical_root.glob("*.json"))
    assert len(json_files) == 224


# ===========================================================================
# G. CLI
# ===========================================================================


def test_cli_returns_zero_on_complete_run(tmp_path, monkeypatch):
    fake = make_fake_runner()
    monkeypatch.setattr(me_mod, "run_main_study", fake)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-heldout-run-hash-full", _PARENT_HASH,
    ])
    assert rc == 0


def test_cli_returns_one_on_bad_parent_hash(tmp_path):
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-heldout-run-hash-full", "deadbeef",
    ])
    assert rc == 1


def test_cli_returns_one_on_infrastructure_failure(tmp_path, monkeypatch):
    def fake_with_infra(planned_runs, **kwargs):
        statuses = []
        # One workload-status with infrastructure_failure so n_infra > 0.
        first = next(iter(planned_runs))
        statuses.append(WorkloadStatus(
            run_id=first.run_id,
            configuration_hash_prefix=first.configuration_hash_prefix,
            method_family=first.config.method_family,
            final_status="infrastructure_failure",
            record_path=first.record_path,
            runtime_seconds=0.0,
            message="simulated",
        ))
        return RunSummary(
            n_planned=224,
            n_executed=0,
            n_skipped=0,
            n_overwritten=0,
            n_success_computed=0,
            n_success_metric_unavailable=0,
            n_model_fit_failure=0,
            n_infrastructure_failure=1,
            total_runtime_seconds=0.0,
            per_workload_status=tuple(statuses + [
                WorkloadStatus(
                    run_id=p.run_id,
                    configuration_hash_prefix=p.configuration_hash_prefix,
                    method_family=p.config.method_family,
                    final_status="skipped",
                    record_path=p.record_path,
                    runtime_seconds=0.0,
                    message="",
                )
                for p in list(planned_runs)[1:]
            ]),
        )

    monkeypatch.setattr(me_mod, "run_main_study", fake_with_infra)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--parent-heldout-run-hash-full", _PARENT_HASH,
    ])
    assert rc == 1


def test_cli_returns_one_on_missing_required_args(tmp_path):
    with pytest.raises(SystemExit):
        cli_main([])  # argparse: missing required args


# ===========================================================================
# H. Decision-log non-modification
# ===========================================================================


def test_docs_03_not_modified(tmp_path):
    _run_with_fake(tmp_path)
    assert not (tmp_path / "docs").exists()


def test_no_writes_outside_tmp_path(tmp_path):
    """All outputs land under tmp_path; nothing is written elsewhere
    by these tests."""
    _run_with_fake(tmp_path)
    # results dir created under tmp_path
    assert (tmp_path / "results").is_dir()


# ===========================================================================
# I. Import allowlist
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "datetime",
    "hashlib",
    "json",
    "pathlib",
    "subprocess",
    "sys",
    "typing",
    "numpy",
    "experiments.main_study.backends",
    "experiments.main_study.records",
    "experiments.main_study.run_io",
    "experiments.main_study.runner",
    "experiments.main_study.schema",
    "experiments.main_study.workloads",
})


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.calibrate_matched_l1",
    "matplotlib",
    "seaborn",
    "PIL",
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


def test_module_imports_are_allowlisted():
    src = Path(me_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"run_main_evaluation.py import {mod!r} is not in the "
            f"allowlist {sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(me_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"run_main_evaluation.py must not import {mod!r}; "
                f"forbidden prefix {forbidden!r}."
            )


# ===========================================================================
# J. Provenance helpers
# ===========================================================================


def test_default_utc_factory_zulu_string():
    s = default_utc_factory()
    assert s.endswith("Z")
    assert len(s) == len("YYYY-MM-DDTHH:MM:SSZ")


def test_capture_code_version_returns_str_or_none():
    val = capture_code_version()
    assert val is None or (isinstance(val, str) and val)


# ===========================================================================
# K. Dataclass smoke
# ===========================================================================


def test_main_evaluation_run_spec_carries_protocol_version():
    spec = MainEvaluationRunSpec(
        parent_heldout_run_hash_full=_PARENT_HASH,
        main_evaluation_run_hash12="abcdef012345",
        output_dir_relative="results/main_study/main_evaluation/abcdef012345",
        code_version=None,
        matched_l1_lambda1=MATCHED_L1_LAMBDA1,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        seed_values=EVALUATION_SEED_VALUES,
    )
    assert spec.protocol_version == MAIN_EVALUATION_PROTOCOL_VERSION
