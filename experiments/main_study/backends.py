"""Real data and fit backends for the main-study pipeline.

Concrete backends conforming to the executor-core ``DataLoader`` /
``FitBackend`` interfaces. The data loader generates observational
training samples and the SCM's true adjacency from the project's SCM
utility. The DAGMA backend drives :class:`DAGMAWrapper` for the
prior-free / matched_l1 / hard_exclusion families. The soft-prior
backend drives :func:`run_soft_prior_dagma_fit` and composes the
post-fit graph/sampler state from the same wrapper-side helpers that
:class:`DAGMAWrapper` uses internally.

No metric module is imported here and no metric is computed. No file
is written and no directory is created. All wrappers and helpers are
the project's existing primitives; no SCM, preprocessing, graph-status,
or sampling logic is duplicated.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from experiments.main_study.executor import (
    DataBundle,
    FitOutcome,
    MetricOutcome,
)
from experiments.main_study.schema import MainStudyConfig
from experiments.main_study.workloads import PlannedRun

from symbolic_priors_cd.data.interventions import Intervention, intervene
from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.metrics import mmd_rbf_unbiased, shd, sid_score
from symbolic_priors_cd.wrappers._dagma_fit import run_soft_prior_dagma_fit
from symbolic_priors_cd.wrappers._dagma_sampling import (
    estimate_residual_sigmas,
    sample_linear_gaussian_model_frame,
)
from symbolic_priors_cd.wrappers._graph_status import (
    classify_graph_status,
    infer_sampler_status,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_plain_int(value: Any) -> bool:
    """True when ``value`` is an ``int`` and not a ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_positive(value: Any) -> bool:
    """True when ``value`` is a finite positive real number (no bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _training_status_from_h_final(
    h_final: float, h_diagnostic_threshold: float
) -> str:
    """Map the DAGMA ``h_final`` scalar to a training_status string.

    Mirrors :meth:`DAGMAWrapper.get_diagnostics` so the soft-prior
    path emits the same vocabulary the rest of the project uses.
    """
    if not math.isfinite(float(h_final)):
        return "diverged"
    if float(h_final) <= float(h_diagnostic_threshold):
        return "converged"
    return "max_iter"


# ---------------------------------------------------------------------------
# DataBundleLoader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DataBundleLoader:
    """Generate the project's standard SCM and observational training data.

    Deterministic in ``config.seed_value``: the SCM and the
    observational samples reproduce bit-identically for the same
    integer seed. The loader performs no file I/O and never samples
    interventions; it returns a :class:`DataBundle` whose metadata
    field carries the live SCM object plus the loader configuration.
    """

    n_nodes: int = 10
    expected_edges: int = 20
    n_train: int = 1000
    noise_scale: float = 1.0
    weight_magnitude_range: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self) -> None:
        for label, value in (
            ("n_nodes", self.n_nodes),
            ("expected_edges", self.expected_edges),
            ("n_train", self.n_train),
        ):
            if not _is_plain_int(value):
                raise ValueError(
                    f"DataBundleLoader.{label} must be a non-bool int; "
                    f"got {value!r}."
                )
            if value <= 0:
                raise ValueError(
                    f"DataBundleLoader.{label} must be positive; "
                    f"got {value}."
                )
        if not _is_finite_positive(self.noise_scale):
            raise ValueError(
                "DataBundleLoader.noise_scale must be a finite "
                f"positive number; got {self.noise_scale!r}."
            )
        if (
            not isinstance(self.weight_magnitude_range, tuple)
            or len(self.weight_magnitude_range) != 2
        ):
            raise ValueError(
                "DataBundleLoader.weight_magnitude_range must be a "
                "length-2 tuple; got "
                f"{self.weight_magnitude_range!r}."
            )
        low, high = self.weight_magnitude_range
        for label, value in (
            ("weight_magnitude_range[0]", low),
            ("weight_magnitude_range[1]", high),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"DataBundleLoader.{label} must be a real number; "
                    f"got {value!r}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"DataBundleLoader.{label} must be finite; got "
                    f"{value!r}."
                )
        if float(low) < 0.0 or float(high) <= 0.0 or float(low) > float(high):
            raise ValueError(
                "DataBundleLoader.weight_magnitude_range must satisfy "
                f"0 <= low <= high and high > 0; got {low!r}, {high!r}."
            )

    def __call__(self, config: MainStudyConfig) -> DataBundle:
        if not isinstance(config, MainStudyConfig):
            raise TypeError(
                "DataBundleLoader requires a MainStudyConfig; got "
                f"{type(config).__name__}."
            )
        seed = int(config.seed_value)
        scm = generate_linear_gaussian_scm(
            n_nodes=int(self.n_nodes),
            expected_edges=int(self.expected_edges),
            seed=seed,
            noise_scale=float(self.noise_scale),
            weight_magnitude_range=(
                float(self.weight_magnitude_range[0]),
                float(self.weight_magnitude_range[1]),
            ),
        )
        x_train = sample_observational(
            scm,
            n_samples=int(self.n_train),
            rng=np.random.default_rng(seed),
        )
        true_adjacency = np.asarray(scm.adjacency, dtype=bool).copy()
        metadata: dict[str, object] = {
            "scm": scm,
            "n_nodes": int(self.n_nodes),
            "expected_edges": int(self.expected_edges),
            "n_train": int(self.n_train),
            "noise_scale": float(self.noise_scale),
            "weight_magnitude_range": (
                float(self.weight_magnitude_range[0]),
                float(self.weight_magnitude_range[1]),
            ),
        }
        return DataBundle(
            x_train=np.asarray(x_train, dtype=float),
            true_adjacency=true_adjacency,
            scm_seed=seed,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# DAGMABackend (prior_free / matched_l1 / hard_exclusion)
# ---------------------------------------------------------------------------


_DAGMA_FAMILIES: frozenset[str] = frozenset(
    {"prior_free", "matched_l1", "hard_exclusion"}
)


@dataclass(frozen=True, kw_only=True)
class DAGMABackend:
    """Real DAGMA fit backend for non-soft-prior method families.

    Wraps :class:`DAGMAWrapper`; the wrapper handles preprocessing
    integration, threshold-driven graph classification, residual-
    sigma sampler bookkeeping, and diagnostics composition. The
    backend only adapts inputs and outputs to the executor's
    ``FitBackend`` contract.
    """

    preprocessor_factory: Callable[[], Any] = StandardisedTransform
    threshold: Optional[float] = None

    def __call__(
        self,
        planned: PlannedRun,
        data_bundle: DataBundle,
        confidence_mask: Optional[np.ndarray],
    ) -> FitOutcome:
        family = planned.config.method_family
        if family not in _DAGMA_FAMILIES:
            raise ValueError(
                f"DAGMABackend does not handle method_family "
                f"{family!r}; expected one of {sorted(_DAGMA_FAMILIES)}."
            )
        if confidence_mask is not None:
            raise ValueError(
                "DAGMABackend does not accept a confidence_mask; "
                "the soft-prior path is handled by SoftPriorBackend. "
                f"got confidence_mask of type "
                f"{type(confidence_mask).__name__}."
            )

        preprocessor = self.preprocessor_factory()
        preprocessor.fit(data_bundle.x_train)
        x_model = preprocessor.transform(data_bundle.x_train)

        cfg = planned.config.dagma_config
        wrapper = DAGMAWrapper()
        wrapper.fit(
            x_model,
            preprocessor=preprocessor,
            seed=int(planned.config.seed_value),
            config=cfg,
        )

        diagnostics = wrapper.get_diagnostics()
        continuous_w = wrapper.native_edge_continuous()
        threshold_value = (
            float(self.threshold)
            if self.threshold is not None
            else float(cfg.project_threshold)
        )
        thresholded_adjacency = wrapper.thresholded_adjacency(
            threshold_value
        )

        graph_status = str(diagnostics["graph_status"])
        sampler_status = str(diagnostics["sampler_status"])
        training_status = str(diagnostics["training_status"])

        model_sampler: Optional[Callable[..., object]] = None
        if sampler_status == "available":
            model_sampler = wrapper.sample_interventional

        return FitOutcome(
            continuous_w=np.asarray(continuous_w, dtype=float),
            thresholded_adjacency=np.asarray(
                thresholded_adjacency, dtype=bool
            ),
            graph_status=graph_status,
            sampler_status=sampler_status,
            training_status=training_status,
            wrapper_diagnostics=dict(diagnostics),
            model_sampler=model_sampler,
        )


# ---------------------------------------------------------------------------
# SoftPriorSampler (in-memory adapter mirroring DAGMAWrapper.sample_interventional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SoftPriorSampler:
    """In-memory sampler adapter for the soft-prior fit path.

    Mirrors :meth:`DAGMAWrapper.sample_interventional` in signature
    and semantics: it transforms the raw-unit intervention value to
    model frame via the fitted preprocessor, calls
    :func:`sample_linear_gaussian_model_frame`, and applies
    ``preprocessor.inverse_transform`` to return raw-unit samples.

    The adapter is constructed only when the soft-prior fit yields a
    valid DAG with a well-formed residual-sigma estimate; the soft-
    prior backend never builds this sampler when the wrapper-side
    helpers would report ``sampler_status != "available"``.
    """

    thresholded_adjacency: np.ndarray
    w_sample: np.ndarray
    sigma: np.ndarray
    preprocessor: Any

    def sample_interventional(
        self,
        intervention: Intervention,
        n_samples: int,
        *,
        sample_seed: int,
        noise_policy: str = "residual_fitted",
    ) -> Optional[np.ndarray]:
        if noise_policy not in ("residual_fitted", "unit_variance"):
            raise ValueError(
                f"Unsupported noise_policy: {noise_policy!r}. "
                "Must be 'residual_fitted' or 'unit_variance'."
            )
        n_vars = int(self.thresholded_adjacency.shape[0])
        if not (0 <= int(intervention.target) < n_vars):
            raise ValueError(
                f"intervention.target must be in [0, {n_vars}); got "
                f"{intervention.target!r}."
            )
        if noise_policy == "residual_fitted":
            sigma = self.sigma
        else:
            sigma = np.ones(n_vars, dtype=float)
        value_model = self.preprocessor.transform_intervention_value(
            float(intervention.value),
            int(intervention.target),
        )
        model_frame_samples = sample_linear_gaussian_model_frame(
            self.thresholded_adjacency,
            self.w_sample,
            sigma,
            target=int(intervention.target),
            value_model=float(value_model),
            n_samples=int(n_samples),
            sample_seed=int(sample_seed),
        )
        return self.preprocessor.inverse_transform(model_frame_samples)


# ---------------------------------------------------------------------------
# SoftPriorBackend (soft_frobenius)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SoftPriorBackend:
    """Real soft-prior fit backend for the ``soft_frobenius`` method family.

    Drives :func:`run_soft_prior_dagma_fit`, then composes graph/
    sampler status with the same wrapper-side helpers that
    :class:`DAGMAWrapper` uses. Builds an in-memory
    :class:`SoftPriorSampler` when the sampler is available so the
    metric backend can draw interventional samples through the same
    interface as the prior-free path.
    """

    preprocessor_factory: Callable[[], Any] = StandardisedTransform
    threshold: Optional[float] = None

    def __call__(
        self,
        planned: PlannedRun,
        data_bundle: DataBundle,
        confidence_mask: Optional[np.ndarray],
    ) -> FitOutcome:
        if planned.config.method_family != "soft_frobenius":
            raise ValueError(
                "SoftPriorBackend only handles method_family="
                f"'soft_frobenius'; got {planned.config.method_family!r}."
            )
        if confidence_mask is None:
            raise ValueError(
                "SoftPriorBackend requires a non-None confidence_mask."
            )
        if planned.config.lambda_prior is None:
            raise ValueError(
                "SoftPriorBackend requires planned.config.lambda_prior "
                "to be set; got None."
            )

        preprocessor = self.preprocessor_factory()
        preprocessor.fit(data_bundle.x_train)
        x_model = preprocessor.transform(data_bundle.x_train)
        # Keep an independent copy so the underlying fit's in-place
        # mean-centring of its X does not perturb the model-frame copy
        # we use for residual-sigma estimation later.
        x_model_for_sigma = np.asarray(x_model, dtype=float).copy()

        cfg = planned.config.dagma_config
        result = run_soft_prior_dagma_fit(
            np.asarray(x_model, dtype=float).copy(),
            cfg,
            lambda_prior=float(planned.config.lambda_prior),
            confidence_mask=confidence_mask,
        )
        continuous_w = np.asarray(result.W, dtype=float)
        h_final = float(result.h_final)
        score_final = float(result.score_final)

        threshold_value = (
            float(self.threshold)
            if self.threshold is not None
            else float(cfg.project_threshold)
        )
        thresholded_adjacency = (
            np.abs(continuous_w) >= threshold_value
        ).astype(bool)

        graph_status, graph_reason = classify_graph_status(
            thresholded_adjacency
        )
        sampler_status, sampler_reason = infer_sampler_status(graph_status)

        training_status = _training_status_from_h_final(
            h_final, cfg.h_diagnostic_threshold
        )

        model_sampler: Optional[Callable[..., object]] = None
        sampler_obj: Optional[SoftPriorSampler] = None
        residual_sigma_vector: Optional[np.ndarray] = None
        w_sample: Optional[np.ndarray] = None
        if sampler_status == "available":
            w_sample, sigma_vec = estimate_residual_sigmas(
                x_model_for_sigma,
                continuous_w,
                thresholded_adjacency,
            )
            if not np.all(np.isfinite(sigma_vec)) or not np.all(
                sigma_vec > 0
            ):
                sampler_status = "unavailable_unresolved_noise_policy"
                sampler_reason = (
                    "Residual-fitted sigma estimate non-finite or "
                    "non-positive."
                )
                w_sample = None
                residual_sigma_vector = None
            else:
                residual_sigma_vector = sigma_vec
                sampler_obj = SoftPriorSampler(
                    thresholded_adjacency=thresholded_adjacency.copy(),
                    w_sample=np.asarray(w_sample, dtype=float).copy(),
                    sigma=np.asarray(sigma_vec, dtype=float).copy(),
                    preprocessor=preprocessor,
                )
                model_sampler = sampler_obj.sample_interventional

        confidence_nonzero_count = int(
            np.count_nonzero(confidence_mask)
        )
        confidence_max = float(np.max(confidence_mask))

        diagnostics: dict[str, object] = {
            "training_status": training_status,
            "graph_status": graph_status,
            "sampler_status": sampler_status,
            "seed": int(planned.config.seed_value),
            "config_snapshot": dataclasses.asdict(cfg),
            "thresholded_adjacency": thresholded_adjacency.copy(),
            "graph_invalid_reason": graph_reason,
            "sampler_unavailable_reason": sampler_reason,
            "model_specific_diagnostics": {
                "h_final": h_final,
                "score_final": score_final,
                "continuous_w_pre_threshold": continuous_w.copy(),
                "threshold": threshold_value,
                "lambda_prior": float(planned.config.lambda_prior),
                "confidence_nonzero_count": confidence_nonzero_count,
                "confidence_max": confidence_max,
                "residual_sigma_available": (
                    residual_sigma_vector is not None
                ),
            },
        }

        return FitOutcome(
            continuous_w=continuous_w.copy(),
            thresholded_adjacency=thresholded_adjacency,
            graph_status=graph_status,
            sampler_status=sampler_status,
            training_status=training_status,
            wrapper_diagnostics=diagnostics,
            model_sampler=model_sampler,
        )


# ---------------------------------------------------------------------------
# MainStudyFitBackend (dispatcher)
# ---------------------------------------------------------------------------


def _default_dagma_backend() -> DAGMABackend:
    return DAGMABackend()


def _default_soft_prior_backend() -> SoftPriorBackend:
    return SoftPriorBackend()


@dataclass(frozen=True, kw_only=True)
class MainStudyFitBackend:
    """Dispatch fits by ``method_family`` to the appropriate backend."""

    dagma_backend: DAGMABackend = dataclasses.field(
        default_factory=_default_dagma_backend
    )
    soft_prior_backend: SoftPriorBackend = dataclasses.field(
        default_factory=_default_soft_prior_backend
    )

    def __call__(
        self,
        planned: PlannedRun,
        data_bundle: DataBundle,
        confidence_mask: Optional[np.ndarray],
    ) -> FitOutcome:
        family = planned.config.method_family
        if family in _DAGMA_FAMILIES:
            return self.dagma_backend(planned, data_bundle, confidence_mask)
        if family == "soft_frobenius":
            return self.soft_prior_backend(
                planned, data_bundle, confidence_mask
            )
        raise ValueError(
            f"MainStudyFitBackend cannot dispatch method_family "
            f"{family!r}; expected one of {sorted(_DAGMA_FAMILIES) + ['soft_frobenius']}."
        )


# ---------------------------------------------------------------------------
# Metric backend constants
# ---------------------------------------------------------------------------


# Raw-unit intervention values used by ``build_default_intervention_specs``.
DEFAULT_INTERVENTION_VALUES_RAW: tuple[float, ...] = (-2.0, 2.0)

# Bandwidth-sweep multipliers, applied to the median-heuristic bandwidth.
DEFAULT_BANDWIDTH_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)

# Seed offsets for per-intervention ground-truth and model sampling.
# Both offsets are deliberately large so they cannot collide with the
# protocol's run seeds (4xx calibration, 5xx evaluation) or prior /
# corruption seed bands (9000-9099, 9100-9199).
PER_INTERVENTION_GT_SEED_OFFSET: int = 10000
PER_INTERVENTION_MODEL_SEED_OFFSET: int = 20000

# Project convention for the SHD reversal cost.
SHD_REVERSAL_COST: int = 2


# ---------------------------------------------------------------------------
# Intervention specs
# ---------------------------------------------------------------------------


def _label_for_value(value_raw: float) -> str:
    """Return a stable label for a raw intervention value.

    The two canonical project values get short labels; any other value
    gets a deterministic safe label of the form ``value_<...>`` with no
    spaces or decimal points.
    """
    v = float(value_raw)
    if v == -2.0:
        return "neg2"
    if v == 2.0:
        return "pos2"
    # Build a deterministic, filename-safe label.
    text = repr(v)
    safe = text.replace(" ", "_").replace(".", "p").replace("-", "neg")
    return f"value_{safe}"


def build_default_intervention_specs(
    n_nodes: int,
    values_raw: tuple[float, ...] = DEFAULT_INTERVENTION_VALUES_RAW,
) -> tuple[dict[str, object], ...]:
    """Build the per-node intervention specs in target-major order.

    For each ``target_node`` in ``range(n_nodes)`` and each
    ``value_raw`` in ``values_raw``, emit a dict with stable keys
    ``intervention_id``, ``target_node``, ``value_raw``. The
    ``intervention_id`` carries a deterministic label (``"neg2"`` and
    ``"pos2"`` for the canonical +/- 2.0 values).
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int):
        raise ValueError(
            "build_default_intervention_specs: n_nodes must be a "
            f"non-bool int; got {n_nodes!r}."
        )
    if n_nodes <= 0:
        raise ValueError(
            "build_default_intervention_specs: n_nodes must be "
            f"positive; got {n_nodes}."
        )
    specs: list[dict[str, object]] = []
    for target_node in range(int(n_nodes)):
        for value_raw in values_raw:
            v = float(value_raw)
            label = _label_for_value(v)
            specs.append(
                {
                    "intervention_id": f"do_X{target_node}_{label}",
                    "target_node": int(target_node),
                    "value_raw": v,
                }
            )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Deterministic median-heuristic bandwidth
# ---------------------------------------------------------------------------


def _median_bandwidth_deterministic(
    x: np.ndarray, y: np.ndarray
) -> Optional[float]:
    """Deterministic median pairwise squared distance over the pooled samples.

    Mirrors the recipe used by the selection-study sampling helper:
    convert both inputs to ``float64`` C-contiguous arrays, build the
    pooled matrix ``vstack([x, y])``, compute squared pairwise
    distances via the norm-expansion identity, clamp at zero to absorb
    finite-precision sign noise, take the upper triangle (excluding
    self-pairs), sort it explicitly, and return the median as a Python
    ``float``.

    Returns ``None`` when the resulting bandwidth is non-finite or
    not strictly positive (degenerate or all-identical samples).
    Raises ``ValueError`` when input shapes are inconsistent; that
    is a programmer error, not a degenerate-data condition.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(
            "_median_bandwidth_deterministic: x must be a numpy "
            f"ndarray; got {type(x).__name__}."
        )
    if not isinstance(y, np.ndarray):
        raise ValueError(
            "_median_bandwidth_deterministic: y must be a numpy "
            f"ndarray; got {type(y).__name__}."
        )
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "_median_bandwidth_deterministic: x and y must be 2D; "
            f"got ndim={x.ndim} and ndim={y.ndim}."
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "_median_bandwidth_deterministic: x and y must have the "
            f"same number of columns; got {x.shape[1]} vs {y.shape[1]}."
        )
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    z = np.vstack([x_c, y_c])
    # ``invalid="ignore"`` suppresses spurious RuntimeWarnings from
    # arithmetic on non-finite inputs (NaN/inf). Degenerate inputs are
    # caught explicitly below by the isfinite check, which returns
    # None rather than propagating a non-finite bandwidth.
    with np.errstate(invalid="ignore"):
        norms = np.sum(z ** 2, axis=1, keepdims=True)
        sq = norms + norms.T - 2.0 * (z @ z.T)
        np.maximum(sq, 0.0, out=sq)
    iu_rows, iu_cols = np.triu_indices(z.shape[0], k=1)
    upper = sq[iu_rows, iu_cols]
    if not np.all(np.isfinite(upper)):
        return None
    upper_sorted = np.sort(upper)
    median_val = float(np.median(upper_sorted))
    if not math.isfinite(median_val) or median_val <= 0.0:
        return None
    return median_val


# ---------------------------------------------------------------------------
# Per-intervention record helper
# ---------------------------------------------------------------------------


def _multiplier_label(multiplier: float) -> str:
    """Stable label for a bandwidth multiplier (e.g. 0.5 -> "0.5x")."""
    return f"{float(multiplier)}x"


def _empty_sweep(
    bandwidth_multipliers: tuple[float, ...],
) -> dict[str, Optional[float]]:
    return {_multiplier_label(m): None for m in bandwidth_multipliers}


def _try_value_model_frame(
    model_sampler: Callable[..., object],
    value_raw: float,
    target_node: int,
) -> Optional[float]:
    """Best-effort lookup of the model-frame intervention value.

    Inspects the sampler callable's ``__self__`` (when bound) for a
    preprocessor exposing ``transform_intervention_value``. Returns
    ``None`` on any failure rather than fabricating a value.
    """
    try:
        sampler_self = getattr(model_sampler, "__self__", None)
        if sampler_self is None:
            return None
        preproc = getattr(sampler_self, "preprocessor", None)
        if preproc is None:
            preproc = getattr(sampler_self, "_preprocessor", None)
        if preproc is None:
            return None
        if not hasattr(preproc, "transform_intervention_value"):
            return None
        return float(
            preproc.transform_intervention_value(
                float(value_raw), int(target_node)
            )
        )
    except Exception:
        return None


def _compute_per_intervention_record(
    *,
    planned: PlannedRun,
    data_bundle: DataBundle,
    fit_outcome: FitOutcome,
    intervention_spec: dict[str, object],
    intervention_index: int,
    n_samples: int,
    bandwidth_multipliers: tuple[float, ...],
) -> dict[str, object]:
    """Build one per-intervention MMD record.

    Validates that the data bundle carries an SCM under
    ``metadata["scm"]`` and that ``fit_outcome.model_sampler`` is
    callable. Draws ground-truth samples via the project
    :func:`intervene` API and model samples via the supplied sampler
    callable. Computes the deterministic median bandwidth and the
    multiplicative bandwidth sweep; the primary ``mmd_value`` is the
    ``"1.0x"`` sweep entry. Negative finite MMD values are preserved.
    """
    if "scm" not in data_bundle.metadata:
        raise KeyError(
            "_compute_per_intervention_record: data_bundle.metadata "
            "is missing the 'scm' key required to draw ground-truth "
            "interventional samples."
        )
    model_sampler = fit_outcome.model_sampler
    if model_sampler is None or not callable(model_sampler):
        raise ValueError(
            "_compute_per_intervention_record: fit_outcome.model_sampler "
            "must be callable; got "
            f"{type(model_sampler).__name__}."
        )

    target_node = int(intervention_spec["target_node"])
    value_raw = float(intervention_spec["value_raw"])
    intervention_id = str(intervention_spec["intervention_id"])

    scm = data_bundle.metadata["scm"]
    intervention = Intervention(target=target_node, value=value_raw)

    base_seed = int(planned.config.seed_value)
    gt_seed = (
        base_seed
        + PER_INTERVENTION_GT_SEED_OFFSET
        + int(intervention_index)
    )
    model_seed = (
        base_seed
        + PER_INTERVENTION_MODEL_SEED_OFFSET
        + int(intervention_index)
    )
    sampler_status_top = str(fit_outcome.sampler_status)
    value_model_frame = _try_value_model_frame(
        model_sampler, value_raw, target_node
    )

    record: dict[str, object] = {
        "intervention_id": intervention_id,
        "target_node": target_node,
        "value_raw": value_raw,
        "value_model_frame": value_model_frame,
        "ground_truth_sampling_seed": int(gt_seed),
        "model_sampling_seed": int(model_seed),
        "n_ground_truth_samples": 0,
        "n_model_samples": 0,
        "mmd_value": None,
        "mmd_status": "available",
        "bandwidth_used": None,
        "bandwidth_sweep": _empty_sweep(bandwidth_multipliers),
        "sampler_status_for_intervention": sampler_status_top,
        "sampler_reason": None,
    }

    gt_samples = intervene(scm, intervention).sample(
        int(n_samples), rng=np.random.default_rng(gt_seed)
    )
    record["n_ground_truth_samples"] = int(gt_samples.shape[0])

    model_samples = model_sampler(
        intervention,
        int(n_samples),
        sample_seed=int(model_seed),
        noise_policy="residual_fitted",
    )
    if model_samples is None:
        record["mmd_status"] = "unavailable_sampler_failure"
        record["sampler_reason"] = (
            "model_sampler returned None for this intervention."
        )
        record["n_model_samples"] = 0
        return record
    record["n_model_samples"] = int(model_samples.shape[0])

    base_bandwidth = _median_bandwidth_deterministic(
        gt_samples, model_samples
    )
    if base_bandwidth is None:
        record["mmd_status"] = "unavailable_other"
        record["sampler_reason"] = (
            "Median-heuristic bandwidth is degenerate or unavailable."
        )
        return record

    sweep: dict[str, Optional[float]] = {}
    primary_label = _multiplier_label(1.0)
    for mult in bandwidth_multipliers:
        bandwidth = float(base_bandwidth) * float(mult)
        try:
            value = float(
                mmd_rbf_unbiased(
                    gt_samples, model_samples, bandwidth=bandwidth
                )
            )
        except Exception as exc:
            record["mmd_status"] = "unavailable_other"
            record["sampler_reason"] = (
                f"MMD computation failed at multiplier {mult}: {exc}"
            )
            record["bandwidth_used"] = None
            record["bandwidth_sweep"] = _empty_sweep(
                bandwidth_multipliers
            )
            return record
        if not math.isfinite(value):
            record["mmd_status"] = "unavailable_other"
            record["sampler_reason"] = (
                f"MMD value at multiplier {mult} is non-finite "
                f"({value})."
            )
            record["bandwidth_used"] = None
            record["bandwidth_sweep"] = _empty_sweep(
                bandwidth_multipliers
            )
            return record
        sweep[_multiplier_label(mult)] = value

    record["bandwidth_used"] = float(base_bandwidth)
    record["bandwidth_sweep"] = sweep
    record["mmd_value"] = sweep.get(primary_label)
    if record["mmd_value"] is None:
        record["mmd_status"] = "unavailable_other"
        record["sampler_reason"] = (
            "bandwidth_multipliers does not include the primary "
            "multiplier 1.0; cannot determine mmd_value."
        )
    return record


# ---------------------------------------------------------------------------
# Bandwidth-sweep aggregation
# ---------------------------------------------------------------------------


def _aggregate_bandwidth_sweep(
    records: list[dict[str, object]],
    bandwidth_multipliers: tuple[float, ...],
) -> dict[str, Optional[float]]:
    """Average finite per-intervention sweep values for each multiplier.

    Returns ``None`` for any multiplier whose available values are
    empty or all non-finite. Negative finite values are preserved
    without clipping or transformation.
    """
    out: dict[str, Optional[float]] = {}
    for mult in bandwidth_multipliers:
        key = _multiplier_label(mult)
        values: list[float] = []
        for r in records:
            sweep = r.get("bandwidth_sweep")
            if not isinstance(sweep, dict):
                continue
            v = sweep.get(key)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                values.append(f)
        out[key] = float(np.mean(values)) if values else None
    return out


# ---------------------------------------------------------------------------
# RealMetricBackend
# ---------------------------------------------------------------------------


def _default_intervention_specs_none() -> Optional[tuple]:
    return None


@dataclass(frozen=True, kw_only=True)
class RealMetricBackend:
    """Real metric backend computing SID, SHD, and per-intervention MMD.

    SID and SHD are computed against ``data_bundle.true_adjacency``.
    Per-intervention MMD is the unbiased RBF estimator with the
    deterministic median-heuristic bandwidth and an explicit
    multiplicative sweep; the aggregate ``mmd`` is the arithmetic
    mean of the ``"1.0x"`` per-intervention values across
    intervention records whose ``mmd_status == "available"``. The
    aggregate may be negative.
    """

    mmd_n_samples: int = 1000
    bandwidth_multipliers: tuple[float, ...] = DEFAULT_BANDWIDTH_MULTIPLIERS
    intervention_specs: Optional[tuple[dict[str, object], ...]] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.mmd_n_samples, bool)
            or not isinstance(self.mmd_n_samples, int)
        ):
            raise ValueError(
                "RealMetricBackend.mmd_n_samples must be a non-bool "
                f"int; got {self.mmd_n_samples!r}."
            )
        if self.mmd_n_samples <= 0:
            raise ValueError(
                "RealMetricBackend.mmd_n_samples must be positive; "
                f"got {self.mmd_n_samples}."
            )
        if (
            not isinstance(self.bandwidth_multipliers, tuple)
            or len(self.bandwidth_multipliers) == 0
        ):
            raise ValueError(
                "RealMetricBackend.bandwidth_multipliers must be a "
                "non-empty tuple of finite positive floats; got "
                f"{self.bandwidth_multipliers!r}."
            )
        for mult in self.bandwidth_multipliers:
            if (
                isinstance(mult, bool)
                or not isinstance(mult, (int, float))
            ):
                raise ValueError(
                    "RealMetricBackend.bandwidth_multipliers entries "
                    f"must be finite positive floats; got {mult!r}."
                )
            f = float(mult)
            if not math.isfinite(f) or f <= 0.0:
                raise ValueError(
                    "RealMetricBackend.bandwidth_multipliers entries "
                    "must be finite and strictly positive; got "
                    f"{mult!r}."
                )
        if not any(
            math.isclose(float(m), 1.0, abs_tol=1e-12)
            for m in self.bandwidth_multipliers
        ):
            raise ValueError(
                "RealMetricBackend.bandwidth_multipliers must "
                "include the primary multiplier 1.0; got "
                f"{self.bandwidth_multipliers!r}."
            )
        if self.intervention_specs is not None:
            if not isinstance(self.intervention_specs, tuple):
                raise ValueError(
                    "RealMetricBackend.intervention_specs must be None "
                    f"or a tuple; got {type(self.intervention_specs).__name__}."
                )
            for idx, spec in enumerate(self.intervention_specs):
                if not isinstance(spec, dict):
                    raise ValueError(
                        "RealMetricBackend.intervention_specs entries "
                        f"must be dicts; entry {idx} is "
                        f"{type(spec).__name__}."
                    )
                for required_key in (
                    "intervention_id",
                    "target_node",
                    "value_raw",
                ):
                    if required_key not in spec:
                        raise ValueError(
                            "RealMetricBackend.intervention_specs entry "
                            f"{idx} is missing required key "
                            f"{required_key!r}."
                        )

    def __call__(
        self,
        planned: PlannedRun,
        data_bundle: DataBundle,
        fit_outcome: FitOutcome,
    ) -> MetricOutcome:
        if not isinstance(planned, PlannedRun):
            raise TypeError(
                "RealMetricBackend requires a PlannedRun; got "
                f"{type(planned).__name__}."
            )
        if not isinstance(data_bundle, DataBundle):
            raise TypeError(
                "RealMetricBackend requires a DataBundle; got "
                f"{type(data_bundle).__name__}."
            )
        if not isinstance(fit_outcome, FitOutcome):
            raise TypeError(
                "RealMetricBackend requires a FitOutcome; got "
                f"{type(fit_outcome).__name__}."
            )
        if fit_outcome.graph_status != "valid_dag":
            raise ValueError(
                "RealMetricBackend requires fit_outcome.graph_status="
                f"'valid_dag'; got {fit_outcome.graph_status!r}."
            )
        if fit_outcome.sampler_status != "available":
            raise ValueError(
                "RealMetricBackend requires fit_outcome.sampler_status="
                f"'available'; got {fit_outcome.sampler_status!r}."
            )
        if fit_outcome.model_sampler is None or not callable(
            fit_outcome.model_sampler
        ):
            raise ValueError(
                "RealMetricBackend requires fit_outcome.model_sampler "
                "to be a callable; got "
                f"{type(fit_outcome.model_sampler).__name__}."
            )
        if "scm" not in data_bundle.metadata:
            raise KeyError(
                "RealMetricBackend requires data_bundle.metadata to "
                "contain the 'scm' key for ground-truth interventional "
                "sampling."
            )

        t_start = time.perf_counter()

        predicted = np.asarray(
            fit_outcome.thresholded_adjacency, dtype=bool
        )
        true_adj = np.asarray(data_bundle.true_adjacency, dtype=bool)
        sid_value = float(sid_score(predicted, true_adj))
        shd_value = float(
            shd(predicted, true_adj, reversal_cost=SHD_REVERSAL_COST)
        )

        if self.intervention_specs is None:
            specs = build_default_intervention_specs(
                n_nodes=int(true_adj.shape[0])
            )
        else:
            specs = self.intervention_specs

        records: list[dict[str, object]] = []
        for idx, spec in enumerate(specs):
            record = _compute_per_intervention_record(
                planned=planned,
                data_bundle=data_bundle,
                fit_outcome=fit_outcome,
                intervention_spec=dict(spec),
                intervention_index=int(idx),
                n_samples=int(self.mmd_n_samples),
                bandwidth_multipliers=self.bandwidth_multipliers,
            )
            records.append(record)

        available = [
            r
            for r in records
            if r["mmd_status"] == "available"
            and r["mmd_value"] is not None
            and isinstance(r["mmd_value"], (int, float))
            and math.isfinite(float(r["mmd_value"]))
        ]
        if not available:
            raise ValueError(
                "RealMetricBackend: no available MMD interventions; "
                "every per-intervention record reported "
                "mmd_status != 'available' or a non-finite mmd_value."
            )
        mmd_primary = float(
            np.mean([float(r["mmd_value"]) for r in available])
        )

        aggregate_sweep = _aggregate_bandwidth_sweep(
            records, self.bandwidth_multipliers
        )

        payload: dict[str, object] = {
            "records": records,
            "mmd_primary": mmd_primary,
            "mmd_available_count": len(available),
            "mmd_missing_count": len(records) - len(available),
            "mmd_bandwidth_sweep": aggregate_sweep,
            "mmd_n_samples": int(self.mmd_n_samples),
            "bandwidth_multipliers": [
                float(m) for m in self.bandwidth_multipliers
            ],
            "ground_truth_seed_offset": int(
                PER_INTERVENTION_GT_SEED_OFFSET
            ),
            "model_seed_offset": int(
                PER_INTERVENTION_MODEL_SEED_OFFSET
            ),
        }

        t_end = time.perf_counter()
        metric_runtime = float(t_end - t_start)

        return MetricOutcome(
            sid=sid_value,
            shd=shd_value,
            mmd=mmd_primary,
            interventions_mmd=payload,
            metric_runtime_seconds=metric_runtime,
        )


__all__ = [
    "DataBundleLoader",
    "DAGMABackend",
    "SoftPriorBackend",
    "SoftPriorSampler",
    "MainStudyFitBackend",
    "RealMetricBackend",
    "build_default_intervention_specs",
    "DEFAULT_INTERVENTION_VALUES_RAW",
    "DEFAULT_BANDWIDTH_MULTIPLIERS",
    "PER_INTERVENTION_GT_SEED_OFFSET",
    "PER_INTERVENTION_MODEL_SEED_OFFSET",
    "SHD_REVERSAL_COST",
]
