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
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from experiments.main_study.executor import (
    DataBundle,
    FitOutcome,
)
from experiments.main_study.schema import MainStudyConfig
from experiments.main_study.workloads import PlannedRun

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
    sample_observational,
)
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


__all__ = [
    "DataBundleLoader",
    "DAGMABackend",
    "SoftPriorBackend",
    "SoftPriorSampler",
    "MainStudyFitBackend",
]
