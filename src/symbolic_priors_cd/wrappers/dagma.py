"""DAGMA-linear wrapper: public surface and frozen configuration.

Defines the ``DAGMAConfig`` dataclass and the ``DAGMAWrapper`` class
that exposes DAGMA-linear behind a project-level API. Methods that
are not yet implemented raise ``NotImplementedError`` so the class
can be imported, instantiated, and type-checked at any stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from symbolic_priors_cd.data.interventions import Intervention
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics


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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class DAGMAWrapper:
    """Public DAGMA-linear wrapper.

    Methods raise ``NotImplementedError`` until their implementations
    land. The class is otherwise constructable so downstream code,
    type checkers, and import-level smoke tests can already depend on
    its public surface.
    """

    def __init__(self) -> None:
        self._fitted: bool = False

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

        Not implemented yet.
        """
        raise NotImplementedError(
            "DAGMAWrapper.thresholded_adjacency is not implemented yet."
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

        Returns ``None`` when the sampler is unavailable. The
        ``noise_policy`` argument selects between residual-fitted
        per-node noise and a unit-variance sensitivity policy. Not
        implemented yet.
        """
        raise NotImplementedError(
            "DAGMAWrapper.sample_interventional is not implemented yet."
        )

    def get_diagnostics(self) -> WrapperDiagnostics:
        """Return the structured diagnostics record after a fit.

        Not implemented yet.
        """
        raise NotImplementedError(
            "DAGMAWrapper.get_diagnostics is not implemented yet."
        )


__all__ = ["DAGMAConfig", "DAGMAWrapper"]
