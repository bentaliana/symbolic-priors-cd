"""DAGMA fit call path.

Provides the function that drives a single DagmaLinear.fit call and
returns a provisional result record. All relevant hyperparameters are
passed explicitly so the wrapper never relies on DagmaLinear's library
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear

if TYPE_CHECKING:
    from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


@dataclass
class _DagmaFitResult:
    """Provisional result from a single DagmaLinear.fit call.

    W is a copy of the continuous edge matrix returned by DagmaLinear
    with w_threshold=0.0 so all pre-threshold values are preserved.
    h_final and score_final are Python floats captured from the
    fitted model's attributes.
    """

    W: np.ndarray
    h_final: float
    score_final: float


def run_dagma_fit(X_local: np.ndarray, cfg: "DAGMAConfig") -> _DagmaFitResult:
    """Call DagmaLinear.fit with all hyperparameters from cfg and return
    a provisional result record.

    X_local must already be a defensive copy of the caller's data;
    this function passes it directly to DagmaLinear without additional
    copying.

    Parameters
    ----------
    X_local : np.ndarray
        Model-frame training data, shape (n_samples, n_vars), float.
        The caller is responsible for ensuring this is an independent
        copy so that DAGMA's in-place mean-centering does not affect
        upstream data.
    cfg : DAGMAConfig
        Resolved configuration whose values are passed explicitly to
        DagmaLinear.fit. No argument is left to the library default.

    Returns
    -------
    _DagmaFitResult
        Contains a copy of the returned continuous W matrix and the
        h_final / score_final scalars captured from the fitted model.

    Raises
    ------
    Any exception raised by DagmaLinear.fit propagates unchanged.
    """
    if cfg.exclude_edges is not None:
        _validate_exclude_edges(cfg.exclude_edges, int(X_local.shape[1]))
    model = DagmaLinear(loss_type=cfg.loss_type)
    W = model.fit(
        X=X_local,
        lambda1=cfg.lambda1,
        w_threshold=cfg.w_threshold_internal,
        T=cfg.T,
        mu_init=cfg.mu_init,
        mu_factor=cfg.mu_factor,
        s=list(cfg.s),
        warm_iter=cfg.warm_iter,
        max_iter=cfg.max_iter,
        lr=cfg.lr,
        beta_1=cfg.beta_1,
        beta_2=cfg.beta_2,
        exclude_edges=cfg.exclude_edges,
        include_edges=None,
    )
    return _DagmaFitResult(
        W=W.copy(),
        h_final=float(model.h_final),
        score_final=float(model.score_final),
    )


def _validate_exclude_edges(
    exclude_edges: object, n_vars: int
) -> None:
    """Validate an exclude_edges tuple before passing it to DagmaLinear.

    DagmaLinear's own validation creates a ``ValueError`` instance but
    never raises it, so malformed input would silently produce an
    empty mask. This helper raises a descriptive ``ValueError`` on
    every malformed-input case the project supports.

    Rules
    -----
    - ``exclude_edges`` must be an actual ``tuple`` (not ``list`` or
      other sequence type).
    - Each element must be a ``tuple`` of exactly two items.
    - Each index must satisfy ``type(idx) is int``; booleans are
      explicitly rejected even though ``bool`` is a subclass of ``int``.
    - Each index must satisfy ``0 <= idx < n_vars``.
    - No self-loops: ``i != j``.
    - No duplicate edges.
    """
    if type(exclude_edges) is not tuple:
        raise ValueError(
            "exclude_edges must be a tuple; "
            f"got {type(exclude_edges).__name__}."
        )
    seen: set[tuple[int, int]] = set()
    for idx, item in enumerate(exclude_edges):
        if type(item) is not tuple:
            raise ValueError(
                f"exclude_edges[{idx}] must be a tuple; "
                f"got {type(item).__name__}: {item!r}."
            )
        if len(item) != 2:
            raise ValueError(
                f"exclude_edges[{idx}] must have length 2; "
                f"got length {len(item)}: {item!r}."
            )
        i, j = item
        if type(i) is not int:
            raise ValueError(
                f"exclude_edges[{idx}][0] must be a plain int "
                f"(no bool, no float, no str); "
                f"got {type(i).__name__}: {i!r}."
            )
        if type(j) is not int:
            raise ValueError(
                f"exclude_edges[{idx}][1] must be a plain int "
                f"(no bool, no float, no str); "
                f"got {type(j).__name__}: {j!r}."
            )
        if i < 0 or j < 0:
            raise ValueError(
                f"exclude_edges[{idx}] = ({i}, {j}) must have "
                "non-negative indices."
            )
        if i >= n_vars or j >= n_vars:
            raise ValueError(
                f"exclude_edges[{idx}] = ({i}, {j}) is out of range "
                f"for n_vars={n_vars}."
            )
        if i == j:
            raise ValueError(
                f"exclude_edges[{idx}] is a self-loop ({i}, {j})."
            )
        if (i, j) in seen:
            raise ValueError(
                f"exclude_edges contains duplicate edge ({i}, {j}) "
                f"at index {idx}."
            )
        seen.add((i, j))


def run_soft_prior_dagma_fit(
    X_local: np.ndarray,
    cfg: "DAGMAConfig",
    *,
    lambda_prior: float,
    confidence_mask: np.ndarray,
) -> _DagmaFitResult:
    """Call SoftPriorDagmaLinear.fit and return a provisional result record.

    Mirrors :func:`run_dagma_fit` but instantiates
    ``SoftPriorDagmaLinear`` so the targeted Frobenius prior gradient
    is added to each Adam iteration. The returned record uses the same
    ``_DagmaFitResult`` dataclass so callers can treat both helpers
    uniformly.

    Parameters
    ----------
    X_local : np.ndarray
        Model-frame training data, shape ``(n_samples, n_vars)``,
        float. The caller is responsible for passing an independent
        copy: the parent mean-centres its input in place.
    cfg : DAGMAConfig
        Resolved configuration whose values are passed explicitly to
        the underlying fit call. No argument is left to the library
        default.
    lambda_prior : float
        Non-negative penalty scale for the prior gradient.
    confidence_mask : np.ndarray
        Square non-negative matrix with zero diagonal, shape
        ``(n_vars, n_vars)``. Per-entry weighting for the targeted
        Frobenius prior gradient.

    Returns
    -------
    _DagmaFitResult
        Contains a copy of the returned continuous W matrix and the
        ``h_final`` / ``score_final`` scalars captured from the fitted
        model.

    Raises
    ------
    Any exception raised by ``SoftPriorDagmaLinear.fit`` propagates
    unchanged. Construction-time validation errors from
    ``SoftPriorDagmaLinear`` propagate unchanged.
    """
    from symbolic_priors_cd.wrappers._soft_prior_dagma import (
        SoftPriorDagmaLinear,
    )

    model = SoftPriorDagmaLinear(
        loss_type=cfg.loss_type,
        lambda_prior=lambda_prior,
        confidence_mask=confidence_mask,
    )
    W = model.fit(
        X=X_local,
        lambda1=cfg.lambda1,
        w_threshold=cfg.w_threshold_internal,
        T=cfg.T,
        mu_init=cfg.mu_init,
        mu_factor=cfg.mu_factor,
        s=list(cfg.s),
        warm_iter=cfg.warm_iter,
        max_iter=cfg.max_iter,
        lr=cfg.lr,
        beta_1=cfg.beta_1,
        beta_2=cfg.beta_2,
    )
    return _DagmaFitResult(
        W=W.copy(),
        h_final=float(model.h_final),
        score_final=float(model.score_final),
    )
