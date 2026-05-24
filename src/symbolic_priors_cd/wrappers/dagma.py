"""DAGMA-linear wrapper: public surface and frozen configuration.

Defines the ``DAGMAConfig`` dataclass and the ``DAGMAWrapper`` class
that exposes DAGMA-linear behind a project-level API. The wrapper
calls into the pinned DAGMA source and exposes fit, native continuous
edge access, thresholded adjacency, raw-unit interventional sampling,
and structured diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers._dagma_sampling import (
    estimate_residual_sigmas,
    sample_linear_gaussian_model_frame,
)
from symbolic_priors_cd.wrappers._graph_status import (
    classify_graph_status,
    infer_sampler_status,
)
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics


# ---------------------------------------------------------------------------
# DAGMA thresholding helper
# ---------------------------------------------------------------------------


def _threshold_continuous_w(
    continuous_w: np.ndarray, threshold: float
) -> np.ndarray:
    """Apply the DAGMA project threshold to a continuous W matrix.

    Returns ``abs(continuous_w) >= threshold`` as a strict bool array.
    The DAGMA edge object is signed, so the threshold is applied to
    absolute values. The function never mutates its input.

    Parameters
    ----------
    continuous_w : np.ndarray
        Float array of shape (d, d). Row-source / column-destination
        convention.
    threshold : float
        Edges with ``abs(continuous_w) >= threshold`` survive.

    Returns
    -------
    np.ndarray
        Boolean adjacency of shape (d, d), dtype bool.
    """
    return (np.abs(continuous_w) >= threshold).astype(bool)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAGMAConfig:
    """Frozen DAGMA-linear hyperparameters for the project.

    ``T``, ``lambda1``, ``s``, ``mu_init``, ``mu_factor``, and
    ``w_threshold_internal`` carry the project's required override
    values: they must be passed at every fit call rather than left to
    DAGMA's library defaults, which differ.

    ``lr``, ``warm_iter``, ``max_iter``, ``beta_1``, ``beta_2``, and
    ``loss_type`` mirror DAGMA's library defaults; they are recorded
    explicitly so every run is fully reproducible from its
    configuration snapshot.

    ``project_threshold`` is the wrapper-level threshold applied to
    ``abs(W_continuous)`` to derive the boolean adjacency. It is
    parameterised so threshold-robustness reports can sweep
    alternative values without retraining.

    ``h_diagnostic_threshold`` is used to derive ``training_status``
    from DAGMA's ``h_final`` at the wrapper boundary. It is a
    reporting threshold, not a graph-repair mechanism.
    """

    # Project-required overrides
    T: int = 4
    lambda1: float = 0.05
    s: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
    mu_init: float = 1.0
    mu_factor: float = 0.1
    w_threshold_internal: float = 0.0

    # DAGMA library defaults, recorded explicitly
    lr: float = 3e-4
    warm_iter: int = 30000
    max_iter: int = 60000
    beta_1: float = 0.99
    beta_2: float = 0.999
    loss_type: str = "l2"

    # Wrapper-level constants
    project_threshold: float = 0.3
    h_diagnostic_threshold: float = 1e-5

    # Hard-exclusion mask passed through to DagmaLinear.fit. When
    # ``None``, no exclusion is applied and DagmaLinear behaves as in
    # the prior-free baseline. When a tuple of (row, col) integer
    # pairs, the wrapper validates the value before the fit call and
    # forwards it as-is to DagmaLinear, which re-zeros the listed
    # entries after every Adam step. The soft-prior fit path
    # ignores this field.
    exclude_edges: Optional[tuple[tuple[int, int], ...]] = None


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class DAGMAWrapper:
    """Public DAGMA-linear wrapper.

    Exposes ``fit``, ``native_edge_continuous``, ``thresholded_adjacency``,
    ``sample_interventional`` (raw-unit roundtrip via the caller's fitted
    preprocessor), and ``get_diagnostics`` (structured ``WrapperDiagnostics``
    record). Methods that depend on a successful fit raise ``RuntimeError``
    when called on an unfitted wrapper.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._graph_status: Optional[str] = None
        self._graph_invalid_reason: Optional[str] = None
        self._sampler_status: Optional[str] = None
        self._sampler_unavailable_reason: Optional[str] = None
        self._X_train_model_frame: Optional[np.ndarray] = None
        self._w_sample_residual_fitted: Optional[np.ndarray] = None
        self._sigma_vector_residual_fitted: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        *,
        preprocessor: Union[CentredOnlyTransform, StandardisedTransform],
        seed: int,
        config: Optional[DAGMAConfig] = None,
    ) -> None:
        """Fit DAGMA-linear on observational training data in model frame.

        X_train is expected already in model frame (transformed by the
        caller's preprocessor before this call). The wrapper passes a
        defensive copy to DagmaLinear so the caller's array is never
        mutated, even though DagmaLinear mean-centres its input in place.

        The fit path does not call ``np.random.seed``,
        ``torch.manual_seed``, or ``dagma.utils.set_random_seed``.
        DagmaLinear.fit is deterministic for fixed input and
        hyperparameters. The ``seed`` argument is stored for
        traceability only.

        Parameters
        ----------
        X_train : np.ndarray
            2D float array of shape (n_samples, n_vars), already in
            model frame. Minimum shape: (1, 2).
        preprocessor : CentredOnlyTransform or StandardisedTransform
            Fitted preprocessor used by later sampling calls for
            intervention-value transforms and inverse transforms.
        seed : int
            Run identifier recorded for reproducibility. Not used to
            seed any random number generator.
        config : DAGMAConfig or None
            Hyperparameter configuration. Defaults to ``DAGMAConfig()``
            when None.

        Raises
        ------
        ValueError
            If X_train is not a valid 2D numeric array with at least
            one row and two columns.
        Any exception raised by DagmaLinear.fit propagates unchanged.
        """
        # Import lazily to avoid pulling in the DAGMA source at
        # wrappers package import time.
        from symbolic_priors_cd.wrappers._dagma_fit import run_dagma_fit

        # --- input validation ---
        X_arr = np.asarray(X_train)
        if X_arr.dtype.kind == "b":
            raise ValueError(
                "X_train has dtype bool; provide a numeric float array."
            )
        try:
            X_float = X_arr.astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"X_train could not be converted to a float array: {exc}"
            ) from exc
        if X_float.ndim != 2:
            raise ValueError(
                f"X_train must be a 2D array, got ndim={X_float.ndim}."
            )
        if X_float.shape[0] < 1:
            raise ValueError(
                f"X_train must have at least one row, got shape {X_float.shape}."
            )
        if X_float.shape[1] < 2:
            raise ValueError(
                f"X_train must have at least two columns (variables), "
                f"got shape {X_float.shape}."
            )

        # Defensive copy: DagmaLinear mutates its input during
        # mean-centering; passing a copy keeps the caller's array
        # unchanged.
        X_local = X_float.copy()

        cfg = config if config is not None else DAGMAConfig()

        # run_dagma_fit raises if DagmaLinear.fit raises; all
        # assignments below only execute on a successful return, so
        # _fitted and _continuous_w_pre_threshold are never set in
        # the error case.
        fit_result = run_dagma_fit(X_local, cfg)

        self._seed = seed
        self._config = cfg
        self._preprocessor = preprocessor
        self._fit_result = fit_result
        # Canonical continuous-W field: a copy so that future callers
        # cannot mutate the internal record via _fit_result.W.
        self._continuous_w_pre_threshold: np.ndarray = fit_result.W.copy()

        # Classify the boolean adjacency at the project threshold and
        # store the resulting graph and sampler status. The continuous
        # W matrix is never modified; the threshold is applied to a
        # fresh array. Invalid graphs are reported, not repaired.
        a_thresh = _threshold_continuous_w(
            self._continuous_w_pre_threshold, cfg.project_threshold
        )
        graph_status, graph_reason = classify_graph_status(a_thresh)
        sampler_status, sampler_reason = infer_sampler_status(graph_status)
        self._graph_status = graph_status
        self._graph_invalid_reason = graph_reason
        self._sampler_status = sampler_status
        self._sampler_unavailable_reason = sampler_reason

        # Store model-frame training data for residual estimation.
        # X_float was not passed to DagmaLinear and is not mutated by
        # DAGMA's internal mean-centering (only X_local is mutated).
        self._X_train_model_frame = X_float.copy()

        # Residual sigma estimation is performed only for valid thresholded
        # DAGs. Invalid graphs leave _w_sample_residual_fitted and
        # _sigma_vector_residual_fitted as None.
        if self._graph_status == "valid_dag":
            w_sample, sigma_vec = estimate_residual_sigmas(
                self._X_train_model_frame,
                self._continuous_w_pre_threshold,
                a_thresh,
            )
            self._w_sample_residual_fitted = w_sample
            self._sigma_vector_residual_fitted = sigma_vec
            if not np.all(np.isfinite(sigma_vec)) or not np.all(sigma_vec > 0):
                self._sampler_status = "unavailable_unresolved_noise_policy"
                self._sampler_unavailable_reason = (
                    "Residual-fitted sigma estimate non-finite or non-positive."
                )

        self._fitted = True

    def native_edge_continuous(self) -> np.ndarray:
        """Return the canonical pre-threshold continuous ``W`` matrix.

        Returns a defensive copy so that mutating the returned array
        does not affect the wrapper's internal state.

        The matrix is the continuous DAGMA edge output captured with
        ``w_threshold=0.0``. Signs and sub-threshold values are
        preserved exactly as returned by DagmaLinear.

        Returns
        -------
        np.ndarray
            Float array of shape (n_vars, n_vars).

        Raises
        ------
        RuntimeError
            If called before a successful fit.
        """
        if not self._fitted:
            raise RuntimeError(
                "native_edge_continuous called on an unfitted DAGMAWrapper. "
                "Call fit() first."
            )
        return self._continuous_w_pre_threshold.copy()

    def thresholded_adjacency(self, threshold: float = 0.3) -> np.ndarray:
        """Return ``abs(W_continuous) >= threshold`` as a boolean adjacency.

        The default threshold matches the project default. Custom
        thresholds are supported without retraining. The returned array
        is a fresh boolean array; the wrapper's internal continuous
        ``W`` is never modified. No silent repair is applied: invalid
        graph patterns (self-loops, bidirected pairs, cycles) are
        preserved in the returned adjacency exactly as they appear.

        Parameters
        ----------
        threshold : float
            Threshold applied to ``abs(W_continuous)``. Defaults to 0.3.

        Returns
        -------
        np.ndarray
            Boolean adjacency of shape ``(n_vars, n_vars)``,
            row-source / column-destination convention.

        Raises
        ------
        RuntimeError
            If called before a successful fit.
        """
        if not self._fitted:
            raise RuntimeError(
                "thresholded_adjacency called on an unfitted DAGMAWrapper. "
                "Call fit() first."
            )
        return _threshold_continuous_w(
            self._continuous_w_pre_threshold, threshold
        )

    def sample_interventional(
        self,
        intervention: Intervention,
        n_samples: int,
        *,
        sample_seed: int,
        noise_policy: Literal["residual_fitted", "unit_variance"] = "residual_fitted",
    ) -> Optional[np.ndarray]:
        """Draw interventional samples in raw SCM units.

        Transforms the raw-unit intervention value to model frame, samples
        via the linear-Gaussian ancestral sampler in topological order, then
        applies inverse_transform to return raw-unit samples.

        Returns None when the sampler is unavailable (invalid graph or
        unresolved noise policy for residual_fitted).

        Parameters
        ----------
        intervention : Intervention
            Target variable index and raw-unit intervention value.
        n_samples : int
            Number of samples to draw. Must be at least 1.
        sample_seed : int
            Seed for np.random.default_rng. Identical arguments produce
            identical output. No global RNG state is mutated.
        noise_policy : {"residual_fitted", "unit_variance"}
            "residual_fitted" uses per-node residual standard deviations
            estimated during fit. "unit_variance" replaces sigma with
            np.ones(n_vars) as a sensitivity check.

        Returns
        -------
        np.ndarray or None
            Float64 array of shape (n_samples, n_vars) in raw SCM units,
            or None when the sampler is unavailable.

        Raises
        ------
        RuntimeError
            If called before a successful fit, or if graph_status is
            valid_dag but required internal fields are missing.
        ValueError
            If noise_policy is not supported, or if n_samples or target
            fail validation inside the model-frame sampler.
        """
        if not self._fitted:
            raise RuntimeError(
                "sample_interventional called on an unfitted DAGMAWrapper. "
                "Call fit() first."
            )

        if noise_policy not in ("residual_fitted", "unit_variance"):
            raise ValueError(
                f"Unsupported noise_policy: {noise_policy!r}. "
                "Must be 'residual_fitted' or 'unit_variance'."
            )

        # Both policies require a valid thresholded graph.
        if self._graph_status != "valid_dag":
            return None

        n_vars = self._continuous_w_pre_threshold.shape[0]

        # Under a normal valid-DAG fit, W_sample must exist.
        # A missing W_sample at this point indicates an internal inconsistency.
        if self._w_sample_residual_fitted is None:
            raise RuntimeError(
                "Internal inconsistency: graph_status is 'valid_dag' but "
                "_w_sample_residual_fitted is None."
            )

        if not (0 <= intervention.target < n_vars):
            raise ValueError(
                f"intervention.target must be in [0, {n_vars}), "
                f"got {intervention.target!r}."
            )

        # _sampler_status reflects the primary residual_fitted policy.
        # unit_variance is gated independently on graph validity (checked
        # above) and W_sample availability (checked above), so unit_variance
        # may run even when _sampler_status is
        # unavailable_unresolved_noise_policy (e.g., degenerate residual sigma).
        if noise_policy == "residual_fitted":
            if self._sampler_status != "available":
                return None
            if self._sigma_vector_residual_fitted is None:
                raise RuntimeError(
                    "Internal inconsistency: sampler_status is 'available' but "
                    "_sigma_vector_residual_fitted is None."
                )
            sigma = self._sigma_vector_residual_fitted
        else:
            sigma = np.ones(n_vars, dtype=float)

        value_model = self._preprocessor.transform_intervention_value(
            intervention.value, intervention.target
        )

        a_thresh = _threshold_continuous_w(
            self._continuous_w_pre_threshold, self._config.project_threshold
        )
        model_frame_samples = sample_linear_gaussian_model_frame(
            a_thresh,
            self._w_sample_residual_fitted,
            sigma,
            target=intervention.target,
            value_model=value_model,
            n_samples=n_samples,
            sample_seed=sample_seed,
        )
        return self._preprocessor.inverse_transform(model_frame_samples)

    def get_diagnostics(self) -> WrapperDiagnostics:
        """Return the structured diagnostics record after a fit.

        Populates the shared ``WrapperDiagnostics`` schema with DAGMA's
        recorded fit and sampler state. Wrapper-specific entries live
        inside ``model_specific_diagnostics``; the top level matches the
        cross-wrapper schema.

        DAGMA does not expose an actual inner-loop iteration count, so
        top-level ``n_iterations`` is ``None``. The configured
        optimisation budget is recorded only inside
        ``model_specific_diagnostics`` as
        ``iterations_configured_upper_bound``.

        All numpy arrays in the returned record are defensive copies;
        mutating them does not affect wrapper state.

        Returns
        -------
        WrapperDiagnostics
            Dictionary populating every top-level key of the schema.

        Raises
        ------
        RuntimeError
            If called before a successful fit.
        """
        if not self._fitted:
            raise RuntimeError(
                "get_diagnostics called on an unfitted DAGMAWrapper. "
                "Call fit() first."
            )

        # Import lazily so importing this module does not pull in the
        # DAGMA source at wrappers package import time.
        from symbolic_priors_cd.wrappers._dagma_utils import DAGMA_SOURCE_PATH

        cfg = self._config
        continuous_w = self._continuous_w_pre_threshold
        a_thresh = _threshold_continuous_w(continuous_w, cfg.project_threshold)

        # Training-status mapping from h_final.
        h_final = float(self._fit_result.h_final)
        score_final = float(self._fit_result.score_final)
        if not np.isfinite(h_final):
            training_status = "diverged"
            converged = False
        elif h_final <= cfg.h_diagnostic_threshold:
            training_status = "converged"
            converged = True
        else:
            training_status = "max_iter"
            converged = False

        config_snapshot: dict[str, object] = {
            "T": cfg.T,
            "lambda1": cfg.lambda1,
            "s": list(cfg.s),
            "mu_init": cfg.mu_init,
            "mu_factor": cfg.mu_factor,
            "w_threshold_internal": cfg.w_threshold_internal,
            "lr": cfg.lr,
            "warm_iter": cfg.warm_iter,
            "max_iter": cfg.max_iter,
            "beta_1": cfg.beta_1,
            "beta_2": cfg.beta_2,
            "loss_type": cfg.loss_type,
            "project_threshold": cfg.project_threshold,
            "h_diagnostic_threshold": cfg.h_diagnostic_threshold,
            "exclude_edges": (
                None
                if cfg.exclude_edges is None
                else [list(edge) for edge in cfg.exclude_edges]
            ),
        }

        loss_decomposition_final: dict[str, float] = {
            "h_final": h_final,
            "score_final": score_final,
        }

        convergence_info: dict[str, object] = {
            "h_final": h_final,
            "h_diagnostic_threshold": cfg.h_diagnostic_threshold,
            "converged": converged,
            "actual_iterations_note": (
                "DAGMA does not expose an actual inner-loop iteration count; "
                "see model_specific_diagnostics.iterations_configured_upper_bound "
                "for the configured budget."
            ),
        }

        numerical_tolerances: dict[str, float] = {
            "h_diagnostic_threshold": cfg.h_diagnostic_threshold,
        }

        # Sigma / W_sample availability flags are also surfaced under
        # mmd_sampling_metadata so callers can read sampling policy state
        # without inspecting model_specific_diagnostics.
        residual_fitted_available = (
            self._sampler_status == "available"
            and self._sigma_vector_residual_fitted is not None
        )
        unit_variance_available = (
            self._graph_status == "valid_dag"
            and self._w_sample_residual_fitted is not None
        )

        mmd_sampling_metadata: dict[str, object] = {
            "primary_noise_policy": "residual_fitted",
            "sensitivity_noise_policy": "unit_variance",
            "noise_policy_default": "residual_fitted",
            "supported_noise_policies": ["residual_fitted", "unit_variance"],
            "project_threshold": cfg.project_threshold,
            "preprocessor_class": type(self._preprocessor).__name__,
            "residual_fitted_available": bool(residual_fitted_available),
            "unit_variance_available": bool(unit_variance_available),
        }

        # Threshold-related counts on the continuous W matrix.
        abs_w = np.abs(continuous_w)
        n = continuous_w.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        threshold_grid_edge_counts: dict[str, int] = {
            "0.2": int(((abs_w >= 0.2) & off_diag).sum()),
            "0.3": int(((abs_w >= 0.3) & off_diag).sum()),
            "0.4": int(((abs_w >= 0.4) & off_diag).sum()),
        }
        sub_threshold_nonzero_count = int(
            ((abs_w > 0.0) & (abs_w < cfg.project_threshold) & off_diag).sum()
        )
        near_threshold_entry_count = int(
            ((abs_w >= 0.2) & (abs_w <= 0.4) & off_diag).sum()
        )

        # Sigma / W_sample availability flags reuse the values computed
        # above for mmd_sampling_metadata. residual_noise_available is the
        # spec-required key under model_specific_diagnostics; it tracks
        # the same condition as residual_fitted_available.
        residual_noise_available = residual_fitted_available

        x_train_shape: Optional[tuple[int, int]] = (
            tuple(self._X_train_model_frame.shape)
            if self._X_train_model_frame is not None
            else None
        )

        sigma_copy = (
            self._sigma_vector_residual_fitted.copy()
            if self._sigma_vector_residual_fitted is not None
            else None
        )
        w_sample_copy = (
            self._w_sample_residual_fitted.copy()
            if self._w_sample_residual_fitted is not None
            else None
        )

        model_specific: dict[str, object] = {
            "model_name": "DAGMA-linear",
            "dagma_source_path": str(DAGMA_SOURCE_PATH),
            "continuous_w_pre_threshold": continuous_w.copy(),
            "thresholded_adjacency_project": a_thresh.copy(),
            "project_threshold": cfg.project_threshold,
            "w_threshold_internal": cfg.w_threshold_internal,
            "h_final": h_final,
            "score_final": score_final,
            "residual_sigma_vector": sigma_copy,
            "w_sample": w_sample_copy,
            "residual_noise_available": bool(residual_noise_available),
            "unit_variance_available": bool(unit_variance_available),
            "x_train_model_frame_shape": x_train_shape,
            "graph_status": self._graph_status,
            "sampler_status": self._sampler_status,
            "sampler_unavailable_reason": self._sampler_unavailable_reason,
            "threshold_grid_edge_counts": threshold_grid_edge_counts,
            "near_threshold_entry_count": near_threshold_entry_count,
            "sub_threshold_nonzero_count": sub_threshold_nonzero_count,
            "iterations_configured_upper_bound": int(
                (cfg.T - 1) * cfg.warm_iter + cfg.max_iter
            ),
            "iterations_configured_formula": (
                "(T - 1) * warm_iter + max_iter; DAGMA's path-following loop "
                "runs T stages with warm_iter inner steps for stages 0..T-2 "
                "and max_iter inner steps for stage T-1, matching the tqdm "
                "total at dagma/linear.py. This is a configured optimisation "
                "budget, not an observed iteration count."
            ),
        }

        diagnostics: WrapperDiagnostics = {
            "training_status": training_status,
            "graph_status": self._graph_status,
            "sampler_status": self._sampler_status,
            "seed": int(self._seed),
            "n_iterations": None,
            "config_snapshot": config_snapshot,
            "loss_history": [],
            "loss_decomposition_final": loss_decomposition_final,
            "convergence_info": convergence_info,
            "thresholded_adjacency": a_thresh.copy(),
            "graph_invalid_reason": self._graph_invalid_reason,
            "sampler_unavailable_reason": self._sampler_unavailable_reason,
            "mmd_sampling_metadata": mmd_sampling_metadata,
            "loss_hook_name": None,
            "numerical_tolerances": numerical_tolerances,
            "model_specific_diagnostics": model_specific,
        }
        return diagnostics


__all__ = ["DAGMAConfig", "DAGMAWrapper"]
