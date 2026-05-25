"""Single-run executor core for the main-study pipeline.

Dispatches one :class:`PlannedRun` through caller-supplied
``data_loader`` / ``fit_backend`` / ``metric_backend`` callables,
assembles the post-run record, and returns the in-memory artefact
payloads. The executor itself performs no I/O, never invokes DAGMA
directly, never imports any metric module, and never creates a
directory or writes a file.

Backends are dependency-injected so tests can drive the dispatch,
record construction, and artefact assembly deterministically. Real
DAGMA, real metrics, and real path persistence belong to follow-up
modules.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from experiments.main_study.priors import (
    PRIOR_SEED_BASE,
    CorruptedPriorSpec,
    build_confidence_mask,
)
from experiments.main_study.records import (
    GRAPH_STATUS_VALUES,
    SAMPLER_STATUS_VALUES,
    MainStudyRunRecord,
    derive_metric_status_for_failure,
    diagnostics_to_canonical,
    make_failure_record,
)
from experiments.main_study.schema import (
    SCHEMA_VERSION,
    MainStudyConfig,
    canonicalize_for_json,
)
from experiments.main_study.workloads import (
    PlannedRun,
    expected_artefact_names_for_method,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelFitFailure(Exception):
    """Signal a recoverable model-side failure from a fit backend.

    The executor catches :class:`ModelFitFailure` and converts it
    into a :class:`MainStudyRunRecord` via :func:`make_failure_record`
    with ``fit_status='model_fit_failure'``. Any other exception
    raised by a backend is treated as infrastructure failure and
    propagates unchanged.
    """


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _is_plain_int(value: Any) -> bool:
    """True when ``value`` is an ``int`` and not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_real(value: Any) -> bool:
    """True when ``value`` is a non-bool real number that ``isfinite``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _empty_dict_factory() -> dict:
    return {}


@dataclass(frozen=True, kw_only=True)
class DataBundle:
    """Synthetic training data plus the true adjacency for one run.

    ``x_train`` is a 2D numeric ndarray; ``true_adjacency`` is a
    square bool-like ndarray whose dimension matches the column count
    of ``x_train``. ``scm_seed`` is a non-bool int recorded for
    provenance; ``metadata`` is a free-form caller-supplied dict.
    """

    x_train: np.ndarray
    true_adjacency: np.ndarray
    scm_seed: int
    metadata: dict[str, object] = field(default_factory=_empty_dict_factory)

    def __post_init__(self) -> None:
        if not isinstance(self.x_train, np.ndarray):
            raise TypeError(
                "DataBundle.x_train must be a numpy ndarray; got "
                f"{type(self.x_train).__name__}."
            )
        if self.x_train.ndim != 2:
            raise ValueError(
                "DataBundle.x_train must be 2D; got ndim="
                f"{self.x_train.ndim}."
            )
        if self.x_train.dtype.kind not in "fiu":
            raise ValueError(
                "DataBundle.x_train must have a numeric dtype "
                f"(float/int); got {self.x_train.dtype}."
            )
        if not np.all(np.isfinite(self.x_train)):
            raise ValueError(
                "DataBundle.x_train must contain only finite values."
            )
        if not isinstance(self.true_adjacency, np.ndarray):
            raise TypeError(
                "DataBundle.true_adjacency must be a numpy ndarray; "
                f"got {type(self.true_adjacency).__name__}."
            )
        if self.true_adjacency.ndim != 2:
            raise ValueError(
                "DataBundle.true_adjacency must be 2D; got ndim="
                f"{self.true_adjacency.ndim}."
            )
        if (
            self.true_adjacency.shape[0]
            != self.true_adjacency.shape[1]
        ):
            raise ValueError(
                "DataBundle.true_adjacency must be square; got shape "
                f"{self.true_adjacency.shape}."
            )
        if self.true_adjacency.shape[0] != self.x_train.shape[1]:
            raise ValueError(
                "DataBundle.true_adjacency dimension "
                f"{self.true_adjacency.shape[0]} does not match "
                f"x_train.shape[1]={self.x_train.shape[1]}."
            )
        if not _is_plain_int(self.scm_seed):
            raise ValueError(
                "DataBundle.scm_seed must be a non-bool int; got "
                f"{self.scm_seed!r}."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError(
                "DataBundle.metadata must be a dict; got "
                f"{type(self.metadata).__name__}."
            )


@dataclass(frozen=True, kw_only=True)
class FitOutcome:
    """Result of a single fit attempt.

    Carries the continuous and thresholded weight matrices, the
    graph/sampler/training status strings the runner needs to decide
    whether to compute metrics, the wrapper diagnostics dict that
    will be canonicalised before being stored on the record, and an
    optional in-memory ``model_sampler`` callable that downstream
    metric computation can use to draw interventional samples from
    the learned model.

    ``model_sampler`` is required to be callable when
    ``sampler_status == "available"`` and may be ``None`` otherwise.
    It is never serialised, recorded, canonicalised, or written into
    any artefact; the executor only forwards it on the
    ``FitOutcome`` instance passed to the metric backend.
    """

    continuous_w: np.ndarray
    thresholded_adjacency: np.ndarray
    graph_status: str
    sampler_status: str
    training_status: str
    wrapper_diagnostics: dict[str, object]
    model_sampler: Optional[Callable[..., object]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.continuous_w, np.ndarray):
            raise TypeError(
                "FitOutcome.continuous_w must be a numpy ndarray; got "
                f"{type(self.continuous_w).__name__}."
            )
        if (
            self.continuous_w.ndim != 2
            or self.continuous_w.shape[0] != self.continuous_w.shape[1]
        ):
            raise ValueError(
                "FitOutcome.continuous_w must be a 2D square array; "
                f"got shape {self.continuous_w.shape}."
            )
        if self.continuous_w.dtype.kind not in "fiu":
            raise ValueError(
                "FitOutcome.continuous_w must have a numeric dtype; "
                f"got {self.continuous_w.dtype}."
            )
        if not isinstance(self.thresholded_adjacency, np.ndarray):
            raise TypeError(
                "FitOutcome.thresholded_adjacency must be a numpy "
                f"ndarray; got {type(self.thresholded_adjacency).__name__}."
            )
        if self.thresholded_adjacency.shape != self.continuous_w.shape:
            raise ValueError(
                "FitOutcome.thresholded_adjacency shape "
                f"{self.thresholded_adjacency.shape} does not match "
                f"continuous_w shape {self.continuous_w.shape}."
            )
        if self.thresholded_adjacency.dtype.kind not in "bi":
            raise ValueError(
                "FitOutcome.thresholded_adjacency must be a bool-like "
                f"array; got dtype {self.thresholded_adjacency.dtype}."
            )
        if self.graph_status not in GRAPH_STATUS_VALUES:
            raise ValueError(
                "FitOutcome.graph_status must be one of "
                f"{GRAPH_STATUS_VALUES}; got {self.graph_status!r}."
            )
        if self.sampler_status not in SAMPLER_STATUS_VALUES:
            raise ValueError(
                "FitOutcome.sampler_status must be one of "
                f"{SAMPLER_STATUS_VALUES}; got {self.sampler_status!r}."
            )
        if not isinstance(self.training_status, str) or not self.training_status:
            raise ValueError(
                "FitOutcome.training_status must be a non-empty "
                f"string; got {self.training_status!r}."
            )
        if not isinstance(self.wrapper_diagnostics, dict):
            raise TypeError(
                "FitOutcome.wrapper_diagnostics must be a dict; got "
                f"{type(self.wrapper_diagnostics).__name__}."
            )
        # model_sampler is required to be callable only when the
        # sampler is reported as available. For invalid-graph or
        # unavailable-sampler outcomes the metric backend will not
        # be invoked, so model_sampler may be absent (None).
        if self.sampler_status == "available":
            if self.model_sampler is None:
                raise ValueError(
                    "FitOutcome.model_sampler must be a callable when "
                    "sampler_status == 'available'; got None."
                )
            if not callable(self.model_sampler):
                raise ValueError(
                    "FitOutcome.model_sampler must be callable when "
                    "sampler_status == 'available'; got "
                    f"{type(self.model_sampler).__name__}."
                )


@dataclass(frozen=True, kw_only=True)
class MetricOutcome:
    """Result of metric computation for one run.

    ``sid`` and ``shd`` must be finite and non-negative; ``mmd`` must
    be finite but may take negative finite values because the raw
    unbiased RBF MMD estimator can be negative in finite samples.
    """

    sid: float
    shd: float
    mmd: float
    interventions_mmd: dict[str, object]
    metric_runtime_seconds: float

    def __post_init__(self) -> None:
        # sid and shd: finite non-negative.
        for label, value in (("sid", self.sid), ("shd", self.shd)):
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"MetricOutcome.{label} must be a finite "
                    f"non-negative number; got {value!r}."
                )
            v = float(value)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"MetricOutcome.{label} must be finite and "
                    f"non-negative; got {value!r}."
                )
        # mmd: finite real, may be negative.
        if (
            isinstance(self.mmd, bool)
            or not isinstance(self.mmd, (int, float))
        ):
            raise ValueError(
                "MetricOutcome.mmd must be a finite real number; got "
                f"{self.mmd!r}."
            )
        if not math.isfinite(float(self.mmd)):
            raise ValueError(
                "MetricOutcome.mmd must be finite (negative finite "
                "values are accepted because the raw unbiased RBF MMD "
                f"estimator can be negative in finite samples); got "
                f"{self.mmd!r}."
            )
        if not isinstance(self.interventions_mmd, dict):
            raise TypeError(
                "MetricOutcome.interventions_mmd must be a dict; got "
                f"{type(self.interventions_mmd).__name__}."
            )
        # metric_runtime_seconds: finite non-negative.
        if (
            isinstance(self.metric_runtime_seconds, bool)
            or not isinstance(self.metric_runtime_seconds, (int, float))
        ):
            raise ValueError(
                "MetricOutcome.metric_runtime_seconds must be a "
                f"finite non-negative number; got {self.metric_runtime_seconds!r}."
            )
        v = float(self.metric_runtime_seconds)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(
                "MetricOutcome.metric_runtime_seconds must be finite "
                f"and non-negative; got {self.metric_runtime_seconds!r}."
            )


@dataclass(frozen=True, kw_only=True)
class ExecutionResult:
    """Bundle of (record, artefacts) returned by :func:`execute_planned_run`."""

    record: MainStudyRunRecord
    artefacts: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.record, MainStudyRunRecord):
            raise TypeError(
                "ExecutionResult.record must be a MainStudyRunRecord; "
                f"got {type(self.record).__name__}."
            )
        if not isinstance(self.artefacts, dict):
            raise TypeError(
                "ExecutionResult.artefacts must be a dict; got "
                f"{type(self.artefacts).__name__}."
            )


# ---------------------------------------------------------------------------
# Type aliases for injected backends
# ---------------------------------------------------------------------------


DataLoader = Callable[[MainStudyConfig], DataBundle]
FitBackend = Callable[[PlannedRun, DataBundle, Optional[np.ndarray]], FitOutcome]
MetricBackend = Callable[[PlannedRun, DataBundle, FitOutcome], MetricOutcome]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Mapping from artefact filename -> MainStudyRunRecord path field.
_ARTEFACT_NAME_TO_RECORD_FIELD: dict[str, str] = {
    "continuous_w.npz": "continuous_w_path",
    "thresholded_adjacency.npz": "thresholded_adjacency_path",
    "true_adjacency.npz": "true_adjacency_path",
    "confidence_mask.npz": "confidence_mask_path",
    "interventions_mmd.json": "interventions_mmd_path",
    "prior_edge_set_clean.json": "prior_edge_set_clean_path",
    "prior_edge_set_corrupted.json": "prior_edge_set_corrupted_path",
    "per_edge_labels.json": "per_edge_labels_path",
}


def _should_compute_metrics(fit_outcome: FitOutcome) -> bool:
    """True when graph_status is ``valid_dag`` and sampler is ``available``."""
    return (
        fit_outcome.graph_status == "valid_dag"
        and fit_outcome.sampler_status == "available"
    )


def _metric_status_for_successful_fit(fit_outcome: FitOutcome) -> str:
    """Derive metric_status assuming the fit succeeded.

    Returns ``"computed"`` when both graph and sampler are healthy;
    otherwise delegates to :func:`derive_metric_status_for_failure`
    with ``fit_status='success'`` so the right unavailable bucket is
    selected.
    """
    if _should_compute_metrics(fit_outcome):
        return "computed"
    return derive_metric_status_for_failure(
        fit_status="success",
        graph_status=fit_outcome.graph_status,
        sampler_status=fit_outcome.sampler_status,
    )


def _npz_payload(array_name: str, array: np.ndarray) -> dict[str, np.ndarray]:
    """Wrap an ndarray in the npz-friendly mapping ``{name: array}``."""
    return {array_name: array}


def _clean_prior_payload_from_corrupted_spec(
    corrupted: CorruptedPriorSpec,
) -> dict[str, Any]:
    """Reconstruct the pre-corruption clean prior from a corrupted spec.

    Set arithmetic:
        clean_forbidden = (corrupted.forbidden_edges
                           - corrupted.added_true_positive_edges)
                          | corrupted.removed_clean_edges

    Returned fields:
        n_nodes, scm_seed, forbidden_edges, and (when scm_seed is not
        None) prior_selection_seed derived from the frozen rule
        ``PRIOR_SEED_BASE + scm_seed``. The derived seed is protocol
        provenance, not an invented value: it reproduces exactly the
        seed the canonical clean-prior helper would have used for the
        same scm_seed.

    No call to any prior-regeneration helper is made; the clean edge
    set is recovered purely from the spec's bookkeeping fields.
    """
    forbidden_set = {tuple(e) for e in corrupted.forbidden_edges}
    added_set = {tuple(e) for e in corrupted.added_true_positive_edges}
    removed_set = {tuple(e) for e in corrupted.removed_clean_edges}
    clean_set = (forbidden_set - added_set) | removed_set
    sorted_clean = sorted(clean_set)
    payload: dict[str, Any] = {
        "n_nodes": int(corrupted.n_nodes),
        "scm_seed": (
            None if corrupted.scm_seed is None else int(corrupted.scm_seed)
        ),
        "forbidden_edges": [
            [int(i), int(j)] for (i, j) in sorted_clean
        ],
    }
    if corrupted.scm_seed is not None:
        payload["prior_selection_seed"] = (
            int(PRIOR_SEED_BASE) + int(corrupted.scm_seed)
        )
    return payload


def _build_artefacts(
    *,
    planned: PlannedRun,
    data_bundle: DataBundle,
    fit_outcome: FitOutcome,
    metric_outcome: Optional[MetricOutcome],
    confidence_mask: Optional[np.ndarray],
    metric_status: str,
) -> dict[str, object]:
    """Assemble the in-memory artefact mapping for a successful fit.

    The artefact keys are the filenames returned by
    :func:`expected_artefact_names_for_method` for the run's
    method_family, with two filtering rules: ``interventions_mmd.json``
    is included only when ``metric_status == "computed"``;
    method-family-specific files (confidence mask, prior set, edge
    labels) are gated by the same expected-name table.
    """
    family = planned.config.method_family
    expected_names = set(expected_artefact_names_for_method(family))
    artefacts: dict[str, object] = {}

    if "continuous_w.npz" in expected_names:
        artefacts["continuous_w.npz"] = _npz_payload(
            "continuous_w", fit_outcome.continuous_w.copy()
        )
    if "thresholded_adjacency.npz" in expected_names:
        artefacts["thresholded_adjacency.npz"] = _npz_payload(
            "thresholded_adjacency",
            fit_outcome.thresholded_adjacency.copy(),
        )
    if "true_adjacency.npz" in expected_names:
        artefacts["true_adjacency.npz"] = _npz_payload(
            "true_adjacency",
            np.asarray(data_bundle.true_adjacency, dtype=bool).copy(),
        )

    if "confidence_mask.npz" in expected_names:
        if confidence_mask is None:
            raise RuntimeError(
                "method_family expects confidence_mask.npz but no "
                "confidence_mask was built by the executor."
            )
        artefacts["confidence_mask.npz"] = _npz_payload(
            "confidence_mask", confidence_mask.copy()
        )

    if "prior_edge_set_clean.json" in expected_names:
        cp = planned.config.corrupted_prior_spec
        if cp is None:
            raise RuntimeError(
                "method_family expects prior_edge_set_clean.json but "
                "config.corrupted_prior_spec is None."
            )
        artefacts["prior_edge_set_clean.json"] = (
            _clean_prior_payload_from_corrupted_spec(cp)
        )
    if "prior_edge_set_corrupted.json" in expected_names:
        cp = planned.config.corrupted_prior_spec
        if cp is None:
            raise RuntimeError(
                "method_family expects prior_edge_set_corrupted.json "
                "but config.corrupted_prior_spec is None."
            )
        artefacts["prior_edge_set_corrupted.json"] = canonicalize_for_json(cp)
    if "per_edge_labels.json" in expected_names:
        cp = planned.config.corrupted_prior_spec
        if cp is None:
            raise RuntimeError(
                "method_family expects per_edge_labels.json but "
                "config.corrupted_prior_spec is None."
            )
        artefacts["per_edge_labels.json"] = dict(cp.edge_labels)

    if (
        metric_status == "computed"
        and "interventions_mmd.json" in expected_names
    ):
        if metric_outcome is None:
            raise RuntimeError(
                "metric_status='computed' but no MetricOutcome was "
                "returned by the metric backend."
            )
        artefacts["interventions_mmd.json"] = dict(
            metric_outcome.interventions_mmd
        )

    return artefacts


def _build_record(
    *,
    planned: PlannedRun,
    n_nodes: int,
    fit_outcome: FitOutcome,
    metric_outcome: Optional[MetricOutcome],
    metric_status: str,
    canonical_diag: dict[str, object],
    runtime_seconds: float,
    fit_runtime_seconds: float,
    generated_at_utc: str,
    code_version: Optional[str],
) -> MainStudyRunRecord:
    """Construct the ``MainStudyRunRecord`` for a successful-fit run."""
    cfg = planned.config
    expected_names = set(expected_artefact_names_for_method(cfg.method_family))

    def _path_or_none(
        artefact_name: str, include: bool = True
    ) -> Optional[str]:
        if include and artefact_name in expected_names:
            return planned.artefact_paths[artefact_name]
        return None

    return MainStudyRunRecord(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        configuration_hash_full=planned.configuration_hash_full,
        configuration_hash_prefix=planned.configuration_hash_prefix,
        run_id=planned.run_id,
        n_nodes=int(n_nodes),
        fit_status="success",
        graph_status=fit_outcome.graph_status,
        sampler_status=fit_outcome.sampler_status,
        metric_status=metric_status,
        failure_kind=None,
        failure_message="",
        sid=(metric_outcome.sid if metric_outcome is not None else None),
        shd=(metric_outcome.shd if metric_outcome is not None else None),
        mmd=(metric_outcome.mmd if metric_outcome is not None else None),
        runtime_seconds=float(runtime_seconds),
        fit_runtime_seconds=float(fit_runtime_seconds),
        metric_runtime_seconds=(
            metric_outcome.metric_runtime_seconds
            if metric_outcome is not None
            else None
        ),
        wrapper_diagnostics=canonical_diag,
        continuous_w_path=_path_or_none("continuous_w.npz"),
        thresholded_adjacency_path=_path_or_none(
            "thresholded_adjacency.npz"
        ),
        confidence_mask_path=_path_or_none("confidence_mask.npz"),
        interventions_mmd_path=_path_or_none(
            "interventions_mmd.json",
            include=(metric_status == "computed"),
        ),
        prior_edge_set_clean_path=_path_or_none(
            "prior_edge_set_clean.json"
        ),
        prior_edge_set_corrupted_path=_path_or_none(
            "prior_edge_set_corrupted.json"
        ),
        per_edge_labels_path=_path_or_none("per_edge_labels.json"),
        true_adjacency_path=_path_or_none("true_adjacency.npz"),
        parent_heldout_run_hash_full=cfg.parent_heldout_run_hash_full,
        generated_at_utc=generated_at_utc,
        code_version=code_version,
    )


def _validate_artefact_keys_against_record(
    artefacts: dict[str, object],
    record: MainStudyRunRecord,
) -> None:
    """Cross-check artefact keys against the record's non-None paths.

    Every artefact key must correspond to a non-None record path, and
    every non-None record artefact path must have a matching artefact
    key. Mismatches raise ``RuntimeError`` because they indicate an
    internal bug in the executor's record/artefact assembly.
    """
    paths_present = {
        name: getattr(record, field_name) is not None
        for name, field_name in _ARTEFACT_NAME_TO_RECORD_FIELD.items()
    }
    for name in artefacts:
        if not paths_present.get(name, False):
            raise RuntimeError(
                f"Executor produced artefact {name!r} but the record "
                "has no corresponding non-None path."
            )
    for name, has_path in paths_present.items():
        if has_path and name not in artefacts:
            raise RuntimeError(
                f"Record references {name!r} but the executor did not "
                "produce a matching artefact key."
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_planned_run(
    planned: PlannedRun,
    *,
    data_loader: DataLoader,
    fit_backend: FitBackend,
    metric_backend: MetricBackend,
    generated_at_utc: str,
    code_version: Optional[str] = None,
) -> ExecutionResult:
    """Execute one planned run through injected backends.

    Order of operations:

    1. Validate ``planned`` is a :class:`PlannedRun`.
    2. Call ``data_loader(planned.config)`` to obtain training data
       and the true adjacency. Infrastructure exceptions propagate.
    3. For ``soft_frobenius``, build the confidence mask via
       :func:`build_confidence_mask`; otherwise pass ``None``.
    4. Call ``fit_backend(planned, data_bundle, confidence_mask)``.
       :class:`ModelFitFailure` is caught and converted to a failure
       record; any other exception propagates.
    5. Canonicalise wrapper diagnostics, decide metric_status, and
       call ``metric_backend`` only when metric_status would be
       ``"computed"``.
    6. Assemble artefact payloads and the
       :class:`MainStudyRunRecord`. Return both.
    """
    if not isinstance(planned, PlannedRun):
        raise TypeError(
            "execute_planned_run requires a PlannedRun; got "
            f"{type(planned).__name__}."
        )

    t_total_start = time.perf_counter()
    cfg = planned.config

    # Infrastructure: any exception from data loading propagates.
    data_bundle = data_loader(cfg)
    if not isinstance(data_bundle, DataBundle):
        raise TypeError(
            "data_loader must return a DataBundle; got "
            f"{type(data_bundle).__name__}."
        )

    n_nodes = int(data_bundle.x_train.shape[1])

    confidence_mask: Optional[np.ndarray] = None
    if cfg.method_family == "soft_frobenius":
        if cfg.corrupted_prior_spec is None:
            raise RuntimeError(
                "soft_frobenius configuration requires a "
                "corrupted_prior_spec; config has None."
            )
        if cfg.confidence is None:
            raise RuntimeError(
                "soft_frobenius configuration requires a confidence "
                "value; config has None."
            )
        confidence_mask = build_confidence_mask(
            cfg.corrupted_prior_spec, float(cfg.confidence)
        )

    t_fit_start = time.perf_counter()
    try:
        fit_outcome = fit_backend(planned, data_bundle, confidence_mask)
    except ModelFitFailure as exc:
        t_end = time.perf_counter()
        fit_runtime = float(t_end - t_fit_start)
        total_runtime = float(t_end - t_total_start)
        failure_msg = str(exc) if str(exc) else type(exc).__name__
        record = make_failure_record(
            config=cfg,
            n_nodes=n_nodes,
            fit_status="model_fit_failure",
            failure_kind=None,
            failure_message=failure_msg,
            runtime_seconds=total_runtime,
            fit_runtime_seconds=fit_runtime,
            wrapper_diagnostics={},
            generated_at_utc=generated_at_utc,
            code_version=code_version,
        )
        return ExecutionResult(record=record, artefacts={})
    if not isinstance(fit_outcome, FitOutcome):
        raise TypeError(
            "fit_backend must return a FitOutcome; got "
            f"{type(fit_outcome).__name__}."
        )
    t_fit_end = time.perf_counter()
    fit_runtime = float(t_fit_end - t_fit_start)

    canonical_diag = diagnostics_to_canonical(
        fit_outcome.wrapper_diagnostics
    )

    metric_status = _metric_status_for_successful_fit(fit_outcome)
    metric_outcome: Optional[MetricOutcome] = None
    if metric_status == "computed":
        metric_outcome = metric_backend(
            planned, data_bundle, fit_outcome
        )
        if not isinstance(metric_outcome, MetricOutcome):
            raise TypeError(
                "metric_backend must return a MetricOutcome; got "
                f"{type(metric_outcome).__name__}."
            )

    t_total_end = time.perf_counter()
    total_runtime = float(t_total_end - t_total_start)

    artefacts = _build_artefacts(
        planned=planned,
        data_bundle=data_bundle,
        fit_outcome=fit_outcome,
        metric_outcome=metric_outcome,
        confidence_mask=confidence_mask,
        metric_status=metric_status,
    )
    record = _build_record(
        planned=planned,
        n_nodes=n_nodes,
        fit_outcome=fit_outcome,
        metric_outcome=metric_outcome,
        metric_status=metric_status,
        canonical_diag=canonical_diag,
        runtime_seconds=total_runtime,
        fit_runtime_seconds=fit_runtime,
        generated_at_utc=generated_at_utc,
        code_version=code_version,
    )
    _validate_artefact_keys_against_record(artefacts, record)

    return ExecutionResult(record=record, artefacts=artefacts)


__all__ = [
    "ModelFitFailure",
    "DataBundle",
    "FitOutcome",
    "MetricOutcome",
    "ExecutionResult",
    "DataLoader",
    "FitBackend",
    "MetricBackend",
    "execute_planned_run",
]
