"""DCDI-G wrapper: public surface and assembly over existing private helpers.

Defines the ``DCDIWrapper`` class and the thresholding helper
``_predict_adjacency_at``. The wrapper consumes the private DCDI
infrastructure verbatim:

- model instantiation via ``_dcdi_utils.make_dcdi_model``;
- augmented-Lagrangian training via
  ``_dcdi_training.run_dcdi_training_loop``;
- thresholding via the in-module ``_predict_adjacency_at`` helper;
- structural-mask-aware interventional sampling via
  ``_dcdi_sampling.sample_raw_units_dcdi``;
- graph and sampler status classification via the module-level
  helpers re-exported from ``_graph_status``.

``sampler_status`` reports mechanical availability only: a valid
thresholded DAG with a callable sampling API maps to
``"available"``, and any non-``valid_dag`` thresholded adjacency
maps to ``"unavailable_invalid_graph"``. The wrapper does not
degrade ``sampler_status`` on the basis of how well the learned
structure matches the data. Poor learned structure surfaces in
downstream metrics (SHD, SID, MMD), not in ``sampler_status``.

No loss hook is registered; ``loss_hook_name`` in the diagnostics
record is always ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers._graph_status import (
    _is_acyclic_adjacency,
    classify_graph_status,
    infer_sampler_status,
)
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics


if TYPE_CHECKING:
    # Lazy: the type alias is needed only for static checkers.
    # ``DCDIConfig`` lives in ``_dcdi_training``; importing that module
    # eagerly triggers the pinned DCDI source-import chain via
    # ``_dcdi_utils``. We re-export ``DCDIConfig`` lazily via module
    # ``__getattr__`` so importing this module alone does not load the
    # DCDI source.
    from symbolic_priors_cd.wrappers._dcdi_training import DCDIConfig


# ---------------------------------------------------------------------------
# DCDI thresholding helper
# ---------------------------------------------------------------------------


def _predict_adjacency_at(
    continuous_w_adj: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    """Apply a threshold to the continuous edge-probability matrix.

    Parameters
    ----------
    continuous_w_adj : torch.Tensor
        Continuous edge-probability matrix, shape (d, d). Off-diagonal
        entries are in [0, 1]; diagonal is exactly zero.
    threshold : float
        Entries >= threshold map to True; all others map to False.

    Returns
    -------
    np.ndarray
        Boolean adjacency of shape (d, d), dtype bool,
        row-source / column-destination convention.
    """
    p = continuous_w_adj.detach().cpu().numpy()
    return (p >= threshold).astype(bool)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class DCDIWrapper:
    """Public DCDI-G wrapper.

    Exposes ``fit``, ``native_edge_continuous``,
    ``thresholded_adjacency``, ``sample_interventional`` (raw-unit
    roundtrip via the caller's fitted preprocessor), and
    ``get_diagnostics`` returning the shared ``WrapperDiagnostics``
    record. Methods that depend on a successful fit raise
    ``RuntimeError`` when called on an unfitted wrapper.

    The wrapper assembles existing private DCDI infrastructure; it
    does not reimplement training, sampling, or thresholding. No
    loss hook is registered; ``loss_hook_name`` in the diagnostics
    record is always ``None``.

    Notes
    -----
    The fit signature includes two keyword-only parameters absent
    from the DAGMA wrapper pattern: ``X_val`` and ``n_iter``. Both
    are required because the consumed private training function
    ``_dcdi_training.run_dcdi_training_loop`` requires them. The
    wrapper does not create an internal train/validation split;
    ``X_val`` is caller-supplied validation data.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._model = None
        self._config = None
        self._preprocessor: Optional[
            Union[CentredOnlyTransform, StandardisedTransform]
        ] = None
        self._seed: Optional[int] = None
        self._n_iter_configured: Optional[int] = None
        self._training_result = None
        self._continuous_log_alpha_pre_threshold: Optional[torch.Tensor] = None
        self._continuous_w_adj_pre_threshold: Optional[torch.Tensor] = None
        self._graph_status: Optional[str] = None
        self._graph_invalid_reason: Optional[str] = None
        self._sampler_status: Optional[str] = None
        self._sampler_unavailable_reason: Optional[str] = None
        self._project_threshold: float = 0.5

    def fit(
        self,
        X_train: np.ndarray,
        *,
        X_val: np.ndarray,
        preprocessor: Union[CentredOnlyTransform, StandardisedTransform],
        seed: int,
        n_iter: int,
        config: "Optional[DCDIConfig]" = None,
    ) -> None:
        """Fit DCDI-G in observational-only mode on model-frame data.

        Both ``X_train`` and ``X_val`` are expected already in the
        candidate's model frame; the caller is responsible for
        preprocessing. The wrapper consumes the existing private
        training loop ``_dcdi_training.run_dcdi_training_loop``
        verbatim. The training loop sets ``torch.manual_seed`` and
        ``np.random.seed`` from ``seed``.

        Parameters
        ----------
        X_train : np.ndarray
            2D float array of shape ``(n_train, n_vars)`` in model
            frame. Minimum shape ``(1, 2)``.
        X_val : np.ndarray
            Caller-supplied validation data of shape
            ``(n_val, n_vars)`` in model frame. The wrapper does not
            create an internal train/validation split. The
            augmented-Lagrangian schedule uses validation NLL
            trajectories.
        preprocessor : CentredOnlyTransform or StandardisedTransform
            Fitted preprocessor used inside ``sample_interventional``
            for raw-to-model-frame intervention-value transforms and
            for inverse transforms of generated samples.
        seed : int
            Seed for ``torch.manual_seed`` and ``np.random.seed``
            inside the training loop; also stored on the wrapper for
            run records.
        n_iter : int
            Maximum number of training iterations. Required because
            the consumed private training function takes this value
            explicitly.
        config : DCDIConfig or None, optional
            Hyperparameter dataclass from
            ``_dcdi_training.DCDIConfig``. Defaults to
            ``DCDIConfig()``.

        Raises
        ------
        ValueError
            If ``X_train`` or ``X_val`` is not a numeric 2D array
            with at least one row and two columns, or if ``seed`` /
            ``n_iter`` is not a plain ``int``, or if ``n_iter`` is
            not positive.
        Any exception raised by
        ``_dcdi_training.run_dcdi_training_loop`` propagates
        unchanged.
        """
        from symbolic_priors_cd.wrappers._dcdi_training import (
            DCDIConfig as _DCDIConfig,
            run_dcdi_training_loop,
        )
        from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model

        X_train_float = self._validate_2d_float(X_train, "X_train")
        X_val_float = self._validate_2d_float(X_val, "X_val")
        if X_train_float.shape[1] != X_val_float.shape[1]:
            raise ValueError(
                "X_train and X_val must have the same number of variables; "
                f"got X_train.shape={X_train_float.shape} and "
                f"X_val.shape={X_val_float.shape}."
            )

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                "seed must be a plain int (not bool); "
                f"got {type(seed).__name__}={seed!r}."
            )
        if isinstance(n_iter, bool) or not isinstance(n_iter, int):
            raise ValueError(
                "n_iter must be a plain int (not bool); "
                f"got {type(n_iter).__name__}={n_iter!r}."
            )
        if n_iter < 1:
            raise ValueError(
                f"n_iter must be at least 1; got {n_iter}."
            )

        cfg = config if config is not None else _DCDIConfig()
        num_vars = X_train_float.shape[1]

        model = make_dcdi_model(
            num_vars=num_vars,
            num_layers=cfg.num_layers,
            hid_dim=cfg.hid_dim,
            nonlin=cfg.nonlin,
        )

        result = run_dcdi_training_loop(
            model,
            X_train_float,
            X_val_float,
            config=cfg,
            seed=seed,
            n_iter=n_iter,
        )

        self._seed = int(seed)
        self._n_iter_configured = int(n_iter)
        self._config = cfg
        self._preprocessor = preprocessor
        self._model = model
        self._training_result = result
        self._continuous_log_alpha_pre_threshold = (
            result.continuous_log_alpha_pre_threshold
        )
        self._continuous_w_adj_pre_threshold = (
            result.continuous_w_adj_pre_threshold
        )

        a_thresh = _predict_adjacency_at(
            self._continuous_w_adj_pre_threshold, self._project_threshold
        )
        graph_status, graph_reason = classify_graph_status(a_thresh)
        sampler_status, sampler_reason = infer_sampler_status(graph_status)
        self._graph_status = graph_status
        self._graph_invalid_reason = graph_reason
        self._sampler_status = sampler_status
        self._sampler_unavailable_reason = sampler_reason

        self._fitted = True

    @staticmethod
    def _validate_2d_float(X: np.ndarray, name: str) -> np.ndarray:
        """Validate that ``X`` is a numeric 2D array with shape ``(>=1, >=2)``.

        Coerces to float, rejects bool arrays, non-numeric data,
        non-2D shapes, and arrays with too few rows or columns.
        """
        arr = np.asarray(X)
        if arr.dtype.kind == "b":
            raise ValueError(
                f"{name} has dtype bool; provide a numeric float array."
            )
        try:
            X_float = arr.astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{name} could not be converted to a float array: {exc}"
            ) from exc
        if X_float.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array, got ndim={X_float.ndim}."
            )
        if X_float.shape[0] < 1:
            raise ValueError(
                f"{name} must have at least one row, got shape {X_float.shape}."
            )
        if X_float.shape[1] < 2:
            raise ValueError(
                f"{name} must have at least two columns (variables), "
                f"got shape {X_float.shape}."
            )
        return X_float

    def native_edge_continuous(self) -> np.ndarray:
        """Return the preserved continuous edge-probability matrix.

        The returned array is a fresh NumPy copy of the detached CPU
        clone captured at the end of training by
        ``_dcdi_training.run_dcdi_training_loop`` (via
        ``_dcdi_utils.snapshot_w_adj``). Off-diagonal entries are in
        ``[0, 1]``; the diagonal is exactly zero.

        Returns
        -------
        np.ndarray
            Float array of shape ``(n_vars, n_vars)`` in
            row-source / column-destination orientation.

        Raises
        ------
        RuntimeError
            If called before a successful fit.
        """
        if not self._fitted:
            raise RuntimeError(
                "native_edge_continuous called on an unfitted DCDIWrapper. "
                "Call fit() first."
            )
        return (
            self._continuous_w_adj_pre_threshold.detach().cpu().numpy().copy()
        )

    def thresholded_adjacency(self, threshold: float = 0.5) -> np.ndarray:
        """Return ``continuous_w_adj >= threshold`` as a boolean adjacency.

        The default threshold is ``0.5``, the DCDI threshold applied
        to the continuous edge-probability matrix. Custom thresholds
        are supported without retraining. The wrapper's internal
        continuous tensor is never modified. No silent repair is
        applied: invalid graph patterns are preserved in the
        returned adjacency exactly as they appear.

        Parameters
        ----------
        threshold : float, optional
            Threshold applied to the continuous edge-probability
            matrix. Defaults to ``0.5``.

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
                "thresholded_adjacency called on an unfitted DCDIWrapper. "
                "Call fit() first."
            )
        return _predict_adjacency_at(
            self._continuous_w_adj_pre_threshold, threshold
        )

    def sample_interventional(
        self,
        intervention: Intervention,
        n_samples: int,
        *,
        sample_seed: int,
    ) -> Optional[np.ndarray]:
        """Draw interventional samples in raw SCM units.

        Computes the thresholded adjacency at the wrapper default of
        ``0.5``. Returns ``None`` when
        ``graph_status != "valid_dag"``; sampler availability is
        gated mechanically by ``graph_status`` and not by the
        quality of the learned structure. Delegates the
        structural-mask save-mutate-sample-restore pattern to
        ``_dcdi_sampling.sample_raw_units_dcdi``, which handles the
        raw-unit roundtrip through the stored preprocessor.

        Parameters
        ----------
        intervention : Intervention
            Target variable index and raw-unit intervention value.
        n_samples : int
            Number of samples to draw. Must be a plain ``int``
            (``bool`` is rejected) and at least ``1``.
        sample_seed : int
            Seed for ``torch.manual_seed`` inside the sampler. Must
            be a plain ``int`` (``bool`` is rejected). Identical
            arguments produce identical output for the same fitted
            model and preprocessor.

        Returns
        -------
        np.ndarray or None
            Float array of shape ``(n_samples, n_vars)`` in raw SCM
            units, or ``None`` when the sampler is unavailable
            (``graph_status != "valid_dag"``).

        Raises
        ------
        RuntimeError
            If called before a successful fit.
        ValueError
            If ``n_samples`` is not a plain ``int`` or is less than
            ``1``; if ``sample_seed`` is not a plain ``int``; or if
            ``intervention.target`` is outside ``[0, n_vars)``.
        """
        from symbolic_priors_cd.wrappers._dcdi_sampling import (
            sample_raw_units_dcdi,
        )

        if not self._fitted:
            raise RuntimeError(
                "sample_interventional called on an unfitted DCDIWrapper. "
                "Call fit() first."
            )

        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise ValueError(
                "n_samples must be a plain int (not bool); "
                f"got {type(n_samples).__name__}={n_samples!r}."
            )
        if n_samples < 1:
            raise ValueError(
                f"n_samples must be at least 1, got {n_samples}."
            )
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
            raise ValueError(
                "sample_seed must be a plain int (not bool); "
                f"got {type(sample_seed).__name__}={sample_seed!r}."
            )

        if self._graph_status != "valid_dag":
            return None

        n_vars = self._continuous_w_adj_pre_threshold.shape[0]
        if not (0 <= intervention.target < n_vars):
            raise ValueError(
                f"intervention.target must be in [0, {n_vars}), "
                f"got {intervention.target!r}."
            )

        a_thresh = _predict_adjacency_at(
            self._continuous_w_adj_pre_threshold, self._project_threshold
        )
        return sample_raw_units_dcdi(
            self._model,
            a_thresh,
            target=intervention.target,
            raw_intervention_value=float(intervention.value),
            n_samples=n_samples,
            sample_seed=sample_seed,
            preprocessor=self._preprocessor,
        )

    def get_diagnostics(self) -> WrapperDiagnostics:
        """Return the structured diagnostics record after a fit.

        Populates every top-level key of the shared
        ``WrapperDiagnostics`` schema. Wrapper-specific entries
        (continuous tensors, gamma/mu update iteration lists, the
        configured iteration bound, and thresholding metadata) live
        inside ``model_specific_diagnostics``. ``loss_hook_name`` is
        ``None`` because the wrapper does not register a loss hook.

        The continuous tensors placed inside
        ``model_specific_diagnostics`` are fresh detached CPU
        clones, so mutating them does not affect subsequent calls
        to ``get_diagnostics`` or ``native_edge_continuous``.

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
                "get_diagnostics called on an unfitted DCDIWrapper. "
                "Call fit() first."
            )

        cfg = self._config
        result = self._training_result
        continuous_w_adj = self._continuous_w_adj_pre_threshold
        continuous_log_alpha = self._continuous_log_alpha_pre_threshold
        a_thresh = _predict_adjacency_at(continuous_w_adj, self._project_threshold)

        final_h = float(result.final_h)
        if not np.isfinite(final_h):
            training_status = "diverged"
        elif result.converged:
            training_status = "converged"
        else:
            training_status = "max_iter"

        loss_history = list(result.loss_history)
        final_aug = (
            float(loss_history[-1]) if loss_history else 0.0
        )

        # The training loop captures the total augmented-Lagrangian
        # value per step plus the final Lagrangian state; the
        # per-component nll / reg / prior decomposition is not
        # preserved, so the final breakdown is restricted to the
        # scalar quantities recorded by the loop.
        loss_decomposition_final: dict[str, float] = {
            "final_aug": final_aug,
            "final_h": final_h,
            "final_gamma": float(result.final_gamma),
            "final_mu": float(result.final_mu),
        }

        convergence_info: dict[str, object] = {
            "converged": bool(result.converged),
            "first_stop": (
                None if result.first_stop is None else int(result.first_stop)
            ),
            "final_h": final_h,
            "h_threshold": float(cfg.h_threshold),
            "gamma_update_iters": list(result.gamma_update_iters),
            "mu_update_iters": list(result.mu_update_iters),
            "n_iterations_completed": int(result.n_iterations),
            "n_iter_configured": int(self._n_iter_configured),
            "validation_nll_history": [
                float(v) for v in result.validation_nll_history
            ],
            "validation_nll_stop_crit_win": int(cfg.stop_crit_win),
        }

        numerical_tolerances: dict[str, float] = {
            "h_threshold": float(cfg.h_threshold),
        }

        n = continuous_w_adj.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        p_np = continuous_w_adj.detach().cpu().numpy()
        threshold_grid_edge_counts: dict[str, int] = {
            "0.4": int(((p_np >= 0.4) & off_diag).sum()),
            "0.5": int(((p_np >= 0.5) & off_diag).sum()),
            "0.6": int(((p_np >= 0.6) & off_diag).sum()),
        }

        config_snapshot: dict[str, object] = {
            "h_threshold": cfg.h_threshold,
            "mu_init": cfg.mu_init,
            "mu_mult_factor": cfg.mu_mult_factor,
            "gamma_init": cfg.gamma_init,
            "omega_gamma": cfg.omega_gamma,
            "omega_mu": cfg.omega_mu,
            "lr": cfg.lr,
            "train_batch_size": cfg.train_batch_size,
            "train_patience": cfg.train_patience,
            "stop_crit_win": cfg.stop_crit_win,
            "reg_coeff": cfg.reg_coeff,
            "num_layers": cfg.num_layers,
            "hid_dim": cfg.hid_dim,
            "nonlin": cfg.nonlin,
            "project_threshold": self._project_threshold,
        }

        mmd_sampling_metadata: dict[str, object] = {
            "primary_noise_policy": "dcdi_native",
            "noise_policy_default": "dcdi_native",
            "supported_noise_policies": ["dcdi_native"],
            "project_threshold": self._project_threshold,
            "preprocessor_class": type(self._preprocessor).__name__,
            "sampler_available": bool(self._sampler_status == "available"),
        }

        model_specific: dict[str, object] = {
            "model_name": "DCDI-G",
            "continuous_log_alpha_pre_threshold": (
                continuous_log_alpha.detach().cpu().clone()
            ),
            "continuous_w_adj_pre_threshold": (
                continuous_w_adj.detach().cpu().clone()
            ),
            "thresholded_adjacency_project": a_thresh.copy(),
            "project_threshold": self._project_threshold,
            "graph_status": self._graph_status,
            "sampler_status": self._sampler_status,
            "graph_invalid_reason": self._graph_invalid_reason,
            "sampler_unavailable_reason": self._sampler_unavailable_reason,
            "threshold_grid_edge_counts": threshold_grid_edge_counts,
            "gamma_update_iters": list(result.gamma_update_iters),
            "mu_update_iters": list(result.mu_update_iters),
            "n_iterations_completed": int(result.n_iterations),
            "n_iter_configured": int(self._n_iter_configured),
            "first_stop": (
                None if result.first_stop is None else int(result.first_stop)
            ),
            "final_h": final_h,
            "final_gamma": float(result.final_gamma),
            "final_mu": float(result.final_mu),
        }

        diagnostics: WrapperDiagnostics = {
            "training_status": training_status,
            "graph_status": self._graph_status,
            "sampler_status": self._sampler_status,
            "seed": int(self._seed),
            "n_iterations": int(result.n_iterations),
            "config_snapshot": config_snapshot,
            "loss_history": [float(v) for v in loss_history],
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


# ---------------------------------------------------------------------------
# Lazy re-export of DCDIConfig
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Lazy module-level attribute access for ``DCDIConfig``.

    Importing ``_dcdi_training`` triggers the pinned DCDI source-import
    chain via ``_dcdi_utils``. We defer that import until
    ``DCDIConfig`` is actually requested, so importing this module
    alone does not load the DCDI source. ``from .dcdi import
    DCDIConfig`` and attribute access through the package work
    identically.
    """
    if name == "DCDIConfig":
        from symbolic_priors_cd.wrappers._dcdi_training import (
            DCDIConfig as _DCDIConfig,
        )
        return _DCDIConfig
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "DCDIConfig",
    "DCDIWrapper",
    "_predict_adjacency_at",
    "_is_acyclic_adjacency",
    "classify_graph_status",
    "infer_sampler_status",
]
