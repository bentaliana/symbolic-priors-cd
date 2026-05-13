"""DAGMA-linear wrapper: public surface and frozen configuration.

Defines the ``DAGMAConfig`` dataclass and the ``DAGMAWrapper`` class
that exposes DAGMA-linear behind a project-level API. Method
implementations are added incrementally; methods that are not yet
implemented raise ``NotImplementedError`` so the class can already
be imported, instantiated, and type-checked.
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

        Not implemented yet.
        """
        raise NotImplementedError("DAGMAWrapper.fit is not implemented yet.")

    def native_edge_continuous(self) -> np.ndarray:
        """Return the preserved pre-threshold continuous ``W`` matrix.

        Not implemented yet.
        """
        raise NotImplementedError(
            "DAGMAWrapper.native_edge_continuous is not implemented yet."
        )

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
