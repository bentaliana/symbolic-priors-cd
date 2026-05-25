"""Main-study main-evaluation runner.

Executes the full 224-workload main-study evaluation grid on
evaluation seeds 501-507 only. This module is a thin orchestration
script over the existing planning/enumeration/runner/run-I/O stack;
it neither implements fit or metric logic, nor analyses results
scientifically. Aggregate comparisons, method rankings, plots, and
final readout figures are deferred to a follow-up readout stage.

The runner refuses calibration seeds, refuses 12-character parent
prefixes, refuses non-``raise`` modes, and asserts the production
backend defaults before invoking the underlying orchestrator. The
underlying orchestrator's ``"raise"`` mode causes pre-existing
record paths to fail loudly before any model fit is attempted, so
incomplete prior runs cannot silently corrupt the headline plan.

Output layout
-------------
- Execution summary outputs are written under
  ``<output_root>/results/main_study/main_evaluation/<main_evaluation_run_hash12>/``.
- Per-run records and artefacts live at the standard main-study
  layout under ``<output_root>/results/main_study/<main_evaluation_run_hash12>/``;
  the summary references each per-run record by its ``record_path``
  and ``configuration_hash_full``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from experiments.main_study.backends import (
    DAGMAConfig,
    DataBundleLoader,
    DEFAULT_BANDWIDTH_MULTIPLIERS,
    MainStudyFitBackend,
    RealMetricBackend,
)
from experiments.main_study.records import MainStudyRunRecord
from experiments.main_study.run_io import load_existing_record
from experiments.main_study.runner import (
    RunSummary,
    WorkloadStatus,
    run_main_study,
)
from experiments.main_study.schema import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    MainStudyConfig,
    METHOD_FAMILIES,
)
from experiments.main_study.workloads import (
    PlannedRun,
    enumerate_planned_runs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


MAIN_EVALUATION_PROTOCOL_VERSION: str = "main_study_main_evaluation_v1"

EVALUATION_SEED_VALUES: tuple[int, ...] = (
    501, 502, 503, 504, 505, 506, 507,
)
FORBIDDEN_CALIBRATION_SEEDS: tuple[int, ...] = (401, 402)

MATCHED_L1_LAMBDA1: float = 0.0625

EXPECTED_WORKLOAD_COUNT: int = 224
EXPECTED_COUNTS_BY_METHOD: dict[str, int] = {
    "prior_free": 7,
    "matched_l1": 7,
    "soft_frobenius": 175,
    "hard_exclusion": 35,
}

DEFAULT_N_NODES: int = 10
DEFAULT_EXPECTED_EDGES: int = 20

REQUIRED_MODE: str = "raise"

MAIN_EVALUATION_SUMMARY_SUBDIR: tuple[str, ...] = (
    "results",
    "main_study",
    "main_evaluation",
)

SUMMARY_JSON_FILENAME: str = "main_evaluation_execution_summary.json"
STATUS_CSV_FILENAME: str = "main_evaluation_workload_status.csv"
SUMMARY_MD_FILENAME: str = "main_evaluation_execution_summary.md"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MainEvaluationRunSpec:
    """Identity of one main-evaluation run."""

    parent_heldout_run_hash_full: str
    main_evaluation_run_hash12: str
    output_dir_relative: str
    code_version: Optional[str]
    matched_l1_lambda1: float
    lambda_prior: float
    seed_values: tuple[int, ...]
    protocol_version: str = MAIN_EVALUATION_PROTOCOL_VERSION


@dataclass(frozen=True, kw_only=True)
class MainEvaluationExecutionSummary:
    """End-state summary of one main-evaluation invocation."""

    main_evaluation_run_hash12: str
    parent_heldout_run_hash_full: str
    output_dir: str
    code_version: Optional[str]
    matched_l1_lambda1: float
    lambda_prior: float
    seed_values: tuple[int, ...]
    n_planned: int
    n_executed: int
    n_skipped: int
    n_overwritten: int
    n_success_computed: int
    n_success_metric_unavailable: int
    n_model_fit_failure: int
    n_infrastructure_failure: int
    method_family_counts: dict[str, int]
    mode: str
    total_runtime_seconds: float


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def default_utc_factory() -> str:
    """Return the current UTC instant as an ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_code_version() -> Optional[str]:
    """Return the current git commit hash, or ``None`` on failure.

    Provenance only; the value never participates in the scientific
    identity hash and is not used to gate selection anywhere.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out if out else None


_HEX64_CHARS: frozenset[str] = frozenset("0123456789abcdef")


def validate_parent_hash_full(parent_hash: object) -> str:
    """Validate a 64-character lowercase hex parent hash.

    Rejects 12-character prefixes explicitly: the main-evaluation
    runner refuses to resolve prefixes so the operator cannot
    accidentally tie the headline plan to an ambiguous parent
    artefact.
    """
    if not isinstance(parent_hash, str):
        raise ValueError(
            "parent_heldout_run_hash_full must be a string; got "
            f"{type(parent_hash).__name__}."
        )
    if len(parent_hash) == 12:
        raise ValueError(
            "parent_heldout_run_hash_full must be the full 64-character "
            "lowercase hex hash; 12-character prefixes are not accepted "
            f"for main-evaluation runs. got {parent_hash!r}."
        )
    if len(parent_hash) != 64 or not all(
        c in _HEX64_CHARS for c in parent_hash
    ):
        raise ValueError(
            "parent_heldout_run_hash_full must be exactly 64 lowercase "
            f"hex characters; got {parent_hash!r}."
        )
    return parent_hash


def compute_main_evaluation_run_hash12(
    *,
    parent_heldout_run_hash_full: str,
    protocol_version: str = MAIN_EVALUATION_PROTOCOL_VERSION,
    seed_values: tuple[int, ...] = EVALUATION_SEED_VALUES,
    matched_l1_lambda1: float = MATCHED_L1_LAMBDA1,
    lambda_prior: float = FROZEN_LAMBDA_PRIOR,
    method_families: tuple[str, ...] = tuple(METHOD_FAMILIES),
    expected_counts_by_method: dict[str, int] = EXPECTED_COUNTS_BY_METHOD,
    expected_total: int = EXPECTED_WORKLOAD_COUNT,
) -> str:
    """Deterministic 12-char hex hash over scientific protocol identity.

    Inputs covered: protocol version, parent provenance, evaluation
    seeds, frozen matched-L1 lambda1, frozen lambda_prior, method-
    family grid definition, and the expected workload counts.
    ``code_version`` is intentionally excluded.
    """
    payload = {
        "protocol_version": protocol_version,
        "parent_heldout_run_hash_full": parent_heldout_run_hash_full,
        "seed_values": [int(s) for s in seed_values],
        "matched_l1_lambda1": float(matched_l1_lambda1),
        "lambda_prior": float(lambda_prior),
        "method_families": list(method_families),
        "expected_counts_by_method": {
            k: int(v) for k, v in expected_counts_by_method.items()
        },
        "expected_total": int(expected_total),
    }
    serialised = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:12]


def build_main_evaluation_output_dir(
    output_root: Path, main_evaluation_run_hash12: str
) -> Path:
    """Return the (un-created) summary output directory under output_root."""
    return (
        output_root
        / MAIN_EVALUATION_SUMMARY_SUBDIR[0]
        / MAIN_EVALUATION_SUMMARY_SUBDIR[1]
        / MAIN_EVALUATION_SUMMARY_SUBDIR[2]
        / main_evaluation_run_hash12
    )


# ---------------------------------------------------------------------------
# Workload enumeration
# ---------------------------------------------------------------------------


def build_main_evaluation_planned_runs(
    *,
    main_evaluation_run_hash12: str,
    parent_heldout_run_hash_full: str,
    base_dagma_config: Optional[DAGMAConfig] = None,
    n_nodes: int = DEFAULT_N_NODES,
    expected_edges: int = DEFAULT_EXPECTED_EDGES,
    matched_l1_lambda1: float = MATCHED_L1_LAMBDA1,
    seed_values: tuple[int, ...] = EVALUATION_SEED_VALUES,
) -> tuple[PlannedRun, ...]:
    """Build the full 224-workload plan via the canonical factory path.

    All :class:`MainStudyConfig` instances are produced via
    ``make_main_study_config`` and all :class:`PlannedRun` instances
    via ``enumerate_planned_runs``; this module never instantiates
    either dataclass directly. The result is verified against the
    expected counts (total and per method family); any mismatch
    raises ``ValueError`` naming the offending family or total.
    """
    if base_dagma_config is None:
        base_dagma_config = DAGMAConfig()
    for s in seed_values:
        if s in FORBIDDEN_CALIBRATION_SEEDS:
            raise ValueError(
                "main-evaluation seed_values must not include "
                f"calibration seed {s!r}; got {seed_values!r}."
            )
    planned = enumerate_planned_runs(
        main_study_run_hash12=main_evaluation_run_hash12,
        seed_population="main_evaluation",
        seed_values=tuple(seed_values),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=parent_heldout_run_hash_full,
        method_families=(
            "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
        ),
        n_nodes=int(n_nodes),
        expected_edges=int(expected_edges),
        matched_l1_lambda1=float(matched_l1_lambda1),
    )
    _verify_plan_counts(planned)
    _verify_no_calibration_seeds(planned)
    _verify_method_specific_invariants(planned)
    return planned


def _verify_plan_counts(planned: tuple[PlannedRun, ...]) -> None:
    if len(planned) != EXPECTED_WORKLOAD_COUNT:
        raise ValueError(
            "main-evaluation plan total mismatch: expected "
            f"{EXPECTED_WORKLOAD_COUNT} planned runs, got {len(planned)}."
        )
    counts: dict[str, int] = {k: 0 for k in EXPECTED_COUNTS_BY_METHOD}
    for p in planned:
        family = p.config.method_family
        if family not in counts:
            raise ValueError(
                "main-evaluation plan contains unexpected method_family "
                f"{family!r}; allowed: {sorted(EXPECTED_COUNTS_BY_METHOD)}."
            )
        counts[family] += 1
    for family, expected in EXPECTED_COUNTS_BY_METHOD.items():
        if counts[family] != expected:
            raise ValueError(
                "main-evaluation plan method-family count mismatch: "
                f"family {family!r} expected {expected}, got "
                f"{counts[family]}."
            )


def _verify_no_calibration_seeds(planned: tuple[PlannedRun, ...]) -> None:
    for p in planned:
        seed = int(p.config.seed_value)
        if seed in FORBIDDEN_CALIBRATION_SEEDS:
            raise ValueError(
                "main-evaluation plan contains calibration seed "
                f"{seed!r}; only evaluation seeds 501-507 are allowed."
            )


def _verify_method_specific_invariants(
    planned: tuple[PlannedRun, ...]
) -> None:
    """Check matched-L1 / hard-exclusion / soft-Frobenius axis rules."""
    hard_per_seed: dict[int, int] = {}
    hard_confidences: set[Optional[float]] = set()
    for p in planned:
        cfg = p.config
        family = cfg.method_family
        if family == "matched_l1":
            if cfg.matched_l1_lambda1 is None or float(
                cfg.matched_l1_lambda1
            ) != MATCHED_L1_LAMBDA1:
                raise ValueError(
                    "matched_l1 configs must use matched_l1_lambda1="
                    f"{MATCHED_L1_LAMBDA1!r}; got "
                    f"{cfg.matched_l1_lambda1!r} on run "
                    f"{p.run_id!r}."
                )
        elif family == "soft_frobenius":
            if cfg.lambda_prior is None or float(
                cfg.lambda_prior
            ) != float(FROZEN_LAMBDA_PRIOR):
                raise ValueError(
                    "soft_frobenius configs must use frozen "
                    f"lambda_prior={FROZEN_LAMBDA_PRIOR!r}; got "
                    f"{cfg.lambda_prior!r} on run {p.run_id!r}."
                )
        elif family == "hard_exclusion":
            hard_per_seed[int(cfg.seed_value)] = (
                hard_per_seed.get(int(cfg.seed_value), 0) + 1
            )
            hard_confidences.add(cfg.confidence)
    for seed in EVALUATION_SEED_VALUES:
        if hard_per_seed.get(int(seed), 0) != 5:
            raise ValueError(
                "hard_exclusion must have exactly 5 configs per "
                f"evaluation seed (one per corruption level); seed "
                f"{seed!r} got {hard_per_seed.get(int(seed), 0)}."
            )
    if hard_confidences != {None}:
        raise ValueError(
            "hard_exclusion configs must not carry a confidence value; "
            f"got confidence set {sorted(c for c in hard_confidences if c is not None)!r}."
        )


# ---------------------------------------------------------------------------
# Plan summary (counts only; no scientific aggregation)
# ---------------------------------------------------------------------------


def summarise_planned_runs(
    planned_runs: tuple[PlannedRun, ...],
) -> dict[str, Any]:
    """Return total, per-method-family, and per-seed counts.

    This is a structural summary used to populate the execution-summary
    outputs. No SID/SHD/MMD value is read; no comparison is made.
    """
    family_counts: dict[str, int] = {}
    seed_counts: dict[int, int] = {}
    for p in planned_runs:
        f = p.config.method_family
        family_counts[f] = family_counts.get(f, 0) + 1
        s = int(p.config.seed_value)
        seed_counts[s] = seed_counts.get(s, 0) + 1
    return {
        "total": len(planned_runs),
        "method_family_counts": dict(
            sorted(family_counts.items(), key=lambda kv: kv[0])
        ),
        "seed_counts": dict(sorted(seed_counts.items())),
    }


# ---------------------------------------------------------------------------
# Backend defaults verification
# ---------------------------------------------------------------------------


def verify_backend_defaults(
    data_loader: object,
    fit_backend: object,
    metric_backend: object,
) -> None:
    """Refuse to proceed unless production backend defaults are in use.

    Specifically: ``RealMetricBackend`` must have
    ``mmd_n_samples == 1000``, ``intervention_specs is None``, and
    ``bandwidth_multipliers == DEFAULT_BANDWIDTH_MULTIPLIERS``.
    DataBundleLoader / MainStudyFitBackend identity is required so a
    silently-substituted custom backend cannot reach the headline run.
    """
    if not isinstance(data_loader, DataBundleLoader):
        raise ValueError(
            "data_loader must be a DataBundleLoader instance; got "
            f"{type(data_loader).__name__}."
        )
    if not isinstance(fit_backend, MainStudyFitBackend):
        raise ValueError(
            "fit_backend must be a MainStudyFitBackend instance; got "
            f"{type(fit_backend).__name__}."
        )
    if not isinstance(metric_backend, RealMetricBackend):
        raise ValueError(
            "metric_backend must be a RealMetricBackend instance; got "
            f"{type(metric_backend).__name__}."
        )
    if int(metric_backend.mmd_n_samples) != 1000:
        raise ValueError(
            "RealMetricBackend.mmd_n_samples must equal 1000 for the "
            f"headline plan; got {metric_backend.mmd_n_samples!r}."
        )
    if metric_backend.intervention_specs is not None:
        raise ValueError(
            "RealMetricBackend.intervention_specs must be None so the "
            "default intervention specs are used; got "
            f"{metric_backend.intervention_specs!r}."
        )
    expected_bw = tuple(DEFAULT_BANDWIDTH_MULTIPLIERS)
    actual_bw = tuple(metric_backend.bandwidth_multipliers)
    if actual_bw != expected_bw:
        raise ValueError(
            "RealMetricBackend.bandwidth_multipliers must equal "
            f"{expected_bw!r}; got {actual_bw!r}."
        )


# ---------------------------------------------------------------------------
# Output writers (structural only; no scientific comparison)
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary_to_json_dict(
    s: MainEvaluationExecutionSummary,
    *,
    planned_summary: dict[str, Any],
    workload_statuses: tuple[WorkloadStatus, ...],
    planned_runs: tuple[PlannedRun, ...],
) -> dict[str, Any]:
    per_workload: list[dict[str, Any]] = []
    by_run_id: dict[str, PlannedRun] = {p.run_id: p for p in planned_runs}
    for ws in workload_statuses:
        planned = by_run_id.get(ws.run_id)
        per_workload.append({
            "run_id": ws.run_id,
            "configuration_hash_prefix": ws.configuration_hash_prefix,
            "method_family": ws.method_family,
            "final_status": ws.final_status,
            "was_overwritten": bool(ws.was_overwritten),
            "record_path": ws.record_path,
            "configuration_hash_full": (
                planned.configuration_hash_full
                if planned is not None else None
            ),
        })
    return {
        "main_evaluation_run_hash12": s.main_evaluation_run_hash12,
        "parent_heldout_run_hash_full": s.parent_heldout_run_hash_full,
        "output_dir": s.output_dir,
        "code_version": s.code_version,
        "matched_l1_lambda1": float(s.matched_l1_lambda1),
        "lambda_prior": float(s.lambda_prior),
        "seed_values": list(s.seed_values),
        "n_planned": int(s.n_planned),
        "n_executed": int(s.n_executed),
        "n_skipped": int(s.n_skipped),
        "n_overwritten": int(s.n_overwritten),
        "n_success_computed": int(s.n_success_computed),
        "n_success_metric_unavailable": int(s.n_success_metric_unavailable),
        "n_model_fit_failure": int(s.n_model_fit_failure),
        "n_infrastructure_failure": int(s.n_infrastructure_failure),
        "method_family_counts": dict(
            sorted(s.method_family_counts.items())
        ),
        "mode": s.mode,
        "total_runtime_seconds": float(s.total_runtime_seconds),
        "planned_summary": planned_summary,
        "per_workload_records": per_workload,
    }


def _write_summary_json(
    *,
    summary: MainEvaluationExecutionSummary,
    planned_summary: dict[str, Any],
    workload_statuses: tuple[WorkloadStatus, ...],
    planned_runs: tuple[PlannedRun, ...],
    output_dir: Path,
) -> Path:
    payload = _summary_to_json_dict(
        summary,
        planned_summary=planned_summary,
        workload_statuses=workload_statuses,
        planned_runs=planned_runs,
    )
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path = output_dir / SUMMARY_JSON_FILENAME
    _write_text(path, text)
    return path


def _write_workload_status_csv(
    *,
    workload_statuses: tuple[WorkloadStatus, ...],
    planned_runs: tuple[PlannedRun, ...],
    output_dir: Path,
) -> Path:
    by_run_id: dict[str, PlannedRun] = {p.run_id: p for p in planned_runs}
    path = output_dir / STATUS_CSV_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "configuration_hash_prefix",
        "configuration_hash_full",
        "method_family",
        "final_status",
        "was_overwritten",
        "record_path",
        "runtime_seconds",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ws in workload_statuses:
            planned = by_run_id.get(ws.run_id)
            writer.writerow({
                "run_id": ws.run_id,
                "configuration_hash_prefix": ws.configuration_hash_prefix,
                "configuration_hash_full": (
                    planned.configuration_hash_full
                    if planned is not None else ""
                ),
                "method_family": ws.method_family,
                "final_status": ws.final_status,
                "was_overwritten": bool(ws.was_overwritten),
                "record_path": ws.record_path,
                "runtime_seconds": float(ws.runtime_seconds),
                "message": ws.message,
            })
    return path


def _write_summary_markdown(
    *,
    summary: MainEvaluationExecutionSummary,
    output_dir: Path,
) -> Path:
    lines: list[str] = []
    lines.append("# Main-evaluation execution summary")
    lines.append("")
    lines.append(
        f"- main_evaluation_run_hash12: {summary.main_evaluation_run_hash12}"
    )
    lines.append(
        f"- parent_heldout_run_hash_full: "
        f"{summary.parent_heldout_run_hash_full}"
    )
    lines.append(f"- code_version: {summary.code_version}")
    lines.append(
        f"- matched_l1_lambda1: {summary.matched_l1_lambda1}"
    )
    lines.append(f"- lambda_prior: {summary.lambda_prior}")
    lines.append("")
    lines.append("## Workload counts")
    lines.append("")
    lines.append(f"- n_planned: {summary.n_planned}")
    for family, count in sorted(summary.method_family_counts.items()):
        lines.append(f"- {family}: {count}")
    lines.append("")
    lines.append("## Execution status counts")
    lines.append("")
    lines.append(f"- n_executed: {summary.n_executed}")
    lines.append(f"- n_skipped: {summary.n_skipped}")
    lines.append(f"- n_overwritten: {summary.n_overwritten}")
    lines.append(
        f"- n_success_computed: {summary.n_success_computed}"
    )
    lines.append(
        f"- n_success_metric_unavailable: "
        f"{summary.n_success_metric_unavailable}"
    )
    lines.append(
        f"- n_model_fit_failure: {summary.n_model_fit_failure}"
    )
    lines.append(
        f"- n_infrastructure_failure: "
        f"{summary.n_infrastructure_failure}"
    )
    lines.append(f"- mode: {summary.mode}")
    lines.append("")
    lines.append("## Output paths")
    lines.append("")
    lines.append(f"- summary_dir: {summary.output_dir}")
    lines.append(
        f"- per_workload_record_paths: see "
        f"{STATUS_CSV_FILENAME} for record_path and "
        f"configuration_hash_full per run"
    )
    path = output_dir / SUMMARY_MD_FILENAME
    _write_text(path, "\n".join(lines))
    return path


def write_main_evaluation_outputs(
    summary: MainEvaluationExecutionSummary,
    workload_statuses: tuple[WorkloadStatus, ...],
    *,
    planned_runs: tuple[PlannedRun, ...],
    planned_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Persist the three execution-summary files. No scientific content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = _write_summary_json(
        summary=summary,
        planned_summary=planned_summary,
        workload_statuses=workload_statuses,
        planned_runs=planned_runs,
        output_dir=output_dir,
    )
    csv_path = _write_workload_status_csv(
        workload_statuses=workload_statuses,
        planned_runs=planned_runs,
        output_dir=output_dir,
    )
    md_path = _write_summary_markdown(
        summary=summary, output_dir=output_dir,
    )
    return {
        "summary_json": json_path,
        "status_csv": csv_path,
        "summary_md": md_path,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_main_evaluation(
    *,
    output_root: Path,
    parent_heldout_run_hash_full: str,
    code_version: Optional[str] = None,
    generated_at_utc_factory: Callable[[], str] = default_utc_factory,
    runner_fn: Optional[Callable[..., RunSummary]] = None,
    mode: str = REQUIRED_MODE,
    data_loader: Optional[Any] = None,
    fit_backend: Optional[Any] = None,
    metric_backend: Optional[Any] = None,
    n_nodes: int = DEFAULT_N_NODES,
    expected_edges: int = DEFAULT_EXPECTED_EDGES,
    n_nodes_for_failure_record: Optional[int] = None,
    logger: Optional[Any] = None,
) -> MainEvaluationExecutionSummary:
    """Drive the 224-workload main-evaluation through the M-6 stack.

    Production defaults are enforced:

    - ``mode`` must equal ``"raise"``;
    - parent hash must be the full 64-character form;
    - all 224 planned runs are enumerated through the canonical
      factory/enumeration path;
    - real :class:`DataBundleLoader`, :class:`MainStudyFitBackend`,
      and :class:`RealMetricBackend` are constructed at protocol
      defaults unless the caller injects equivalents that pass
      :func:`verify_backend_defaults`.

    The function returns a structural execution summary; aggregate
    method comparisons, rankings, plots, and the decision log are
    explicitly out of scope.
    """
    if mode != REQUIRED_MODE:
        raise ValueError(
            "run_main_evaluation requires mode='raise'; got "
            f"{mode!r}."
        )
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    parent_full = validate_parent_hash_full(parent_heldout_run_hash_full)
    if runner_fn is None:
        runner_fn = run_main_study
    if code_version is None:
        code_version = capture_code_version()

    if data_loader is None:
        data_loader = DataBundleLoader(
            n_nodes=int(n_nodes), expected_edges=int(expected_edges)
        )
    if fit_backend is None:
        fit_backend = MainStudyFitBackend()
    if metric_backend is None:
        metric_backend = RealMetricBackend()
    verify_backend_defaults(data_loader, fit_backend, metric_backend)

    main_evaluation_run_hash12 = compute_main_evaluation_run_hash12(
        parent_heldout_run_hash_full=parent_full,
    )
    output_dir = build_main_evaluation_output_dir(
        output_root, main_evaluation_run_hash12
    )

    planned_runs = build_main_evaluation_planned_runs(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        parent_heldout_run_hash_full=parent_full,
        n_nodes=int(n_nodes),
        expected_edges=int(expected_edges),
    )
    planned_summary = summarise_planned_runs(planned_runs)

    if n_nodes_for_failure_record is None:
        n_nodes_for_failure_record = int(n_nodes)

    run_summary: RunSummary = runner_fn(
        planned_runs,
        base_dir=output_root,
        data_loader=data_loader,
        fit_backend=fit_backend,
        metric_backend=metric_backend,
        mode=REQUIRED_MODE,
        code_version=code_version,
        generated_at_utc_factory=generated_at_utc_factory,
        n_nodes_for_failure_record=int(n_nodes_for_failure_record),
        logger=logger,
    )
    if not isinstance(run_summary, RunSummary):
        raise TypeError(
            "runner_fn must return a RunSummary; got "
            f"{type(run_summary).__name__}."
        )

    summary = MainEvaluationExecutionSummary(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        parent_heldout_run_hash_full=parent_full,
        output_dir=str(
            output_dir.relative_to(output_root)
        ).replace("\\", "/"),
        code_version=code_version,
        matched_l1_lambda1=MATCHED_L1_LAMBDA1,
        lambda_prior=float(FROZEN_LAMBDA_PRIOR),
        seed_values=EVALUATION_SEED_VALUES,
        n_planned=int(run_summary.n_planned),
        n_executed=int(run_summary.n_executed),
        n_skipped=int(run_summary.n_skipped),
        n_overwritten=int(run_summary.n_overwritten),
        n_success_computed=int(run_summary.n_success_computed),
        n_success_metric_unavailable=int(
            run_summary.n_success_metric_unavailable
        ),
        n_model_fit_failure=int(run_summary.n_model_fit_failure),
        n_infrastructure_failure=int(
            run_summary.n_infrastructure_failure
        ),
        method_family_counts=dict(
            planned_summary["method_family_counts"]
        ),
        mode=REQUIRED_MODE,
        total_runtime_seconds=float(
            run_summary.total_runtime_seconds
        ),
    )
    write_main_evaluation_outputs(
        summary,
        run_summary.per_workload_status,
        planned_runs=planned_runs,
        planned_summary=planned_summary,
        output_dir=output_dir,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK: int = 0
_EXIT_ERROR: int = 1


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_main_evaluation",
        description=(
            "Execute the main-study main-evaluation grid (224 workloads "
            "on evaluation seeds 501-507). Writes structural execution "
            "outputs only; does not analyse results or modify the "
            "decision log."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Root directory under which results/main_study/... will be "
            "created."
        ),
    )
    parser.add_argument(
        "--parent-heldout-run-hash-full",
        type=str,
        required=True,
        help=(
            "Full 64-character lowercase hex held-out parent run hash. "
            "12-character prefixes are not accepted for main-evaluation."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_main_evaluation(
            output_root=args.output_root,
            parent_heldout_run_hash_full=args.parent_heldout_run_hash_full,
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            f"run_main_evaluation: error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return _EXIT_ERROR
    if summary.n_planned != EXPECTED_WORKLOAD_COUNT:
        sys.stderr.write(
            f"run_main_evaluation: expected {EXPECTED_WORKLOAD_COUNT} "
            f"planned runs, got {summary.n_planned}.\n"
        )
        return _EXIT_ERROR
    persisted = (
        summary.n_executed + summary.n_skipped + summary.n_overwritten
    )
    if persisted != EXPECTED_WORKLOAD_COUNT:
        sys.stderr.write(
            f"run_main_evaluation: expected {EXPECTED_WORKLOAD_COUNT} "
            f"records persisted, observed n_executed+n_skipped+"
            f"n_overwritten={persisted}.\n"
        )
        return _EXIT_ERROR
    if summary.n_infrastructure_failure > 0:
        sys.stderr.write(
            "run_main_evaluation: at least one infrastructure failure "
            "was recorded; treating as non-zero exit.\n"
        )
        return _EXIT_ERROR
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EVALUATION_SEED_VALUES",
    "EXPECTED_COUNTS_BY_METHOD",
    "EXPECTED_WORKLOAD_COUNT",
    "FORBIDDEN_CALIBRATION_SEEDS",
    "MAIN_EVALUATION_PROTOCOL_VERSION",
    "MATCHED_L1_LAMBDA1",
    "MainEvaluationExecutionSummary",
    "MainEvaluationRunSpec",
    "REQUIRED_MODE",
    "build_main_evaluation_output_dir",
    "build_main_evaluation_planned_runs",
    "capture_code_version",
    "compute_main_evaluation_run_hash12",
    "default_utc_factory",
    "main",
    "run_main_evaluation",
    "summarise_planned_runs",
    "verify_backend_defaults",
    "write_main_evaluation_outputs",
]
