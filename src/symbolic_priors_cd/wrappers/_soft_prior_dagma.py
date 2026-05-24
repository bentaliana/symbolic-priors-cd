"""Soft-prior DAGMA-linear variant.

Adds a fixed targeted Frobenius prior gradient to DAGMA's hand-coded
gradient assembly. The prior gradient is added once per Adam iteration,
immediately after the parent's ``Gobj`` assembly and before the Adam
update inside ``minimize``. It is not scaled by ``mu`` or by any
path-following parameter.

The prior loss is

    L_prior = lambda_prior * sum_ij C_ij * W_ij**2

and its gradient is

    G_prior = 2 * lambda_prior * (C * W)

where ``C`` is a fixed confidence mask. ``include_edges`` and
``exclude_edges`` are never used by this variant: both are forwarded
to the parent as ``None``.
"""

from __future__ import annotations

import math
import typing

import numpy as np
import scipy.linalg as sla
from scipy.special import expit as sigmoid

from symbolic_priors_cd.wrappers._dagma_utils import DagmaLinear


def prior_gradient(
    W: np.ndarray,
    confidence_mask: np.ndarray,
    lambda_prior: float,
) -> np.ndarray:
    """Return the targeted Frobenius prior gradient.

    The returned array equals ``2.0 * lambda_prior * confidence_mask * W``
    element-wise. Entries where ``confidence_mask`` is zero are exactly
    zero in the output regardless of the value of ``W`` at those
    positions.

    The function deliberately takes no ``mu`` or other path-following
    argument: the prior gradient is not scaled by ``mu``.

    Parameters
    ----------
    W : np.ndarray
        Continuous weighted adjacency matrix, shape ``(d, d)``.
    confidence_mask : np.ndarray
        Per-entry penalty weights, same shape as ``W``. Non-negative.
    lambda_prior : float
        Global penalty scale. Non-negative.

    Returns
    -------
    np.ndarray
        Gradient of the prior loss, same shape as ``W``.
    """
    return 2.0 * lambda_prior * confidence_mask * W


def _validate_lambda_prior(lambda_prior: typing.Any) -> float:
    """Validate ``lambda_prior`` and return it as a Python float.

    Raises ``ValueError`` if the value is not a finite, non-negative
    real number.
    """
    try:
        value = float(lambda_prior)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"lambda_prior must be a real number; got {lambda_prior!r}."
        ) from exc
    if math.isnan(value):
        raise ValueError("lambda_prior must not be NaN.")
    if math.isinf(value):
        raise ValueError(
            f"lambda_prior must be finite; got {lambda_prior!r}."
        )
    if value < 0.0:
        raise ValueError(
            f"lambda_prior must be non-negative; got {value}."
        )
    return value


def _validate_confidence_mask(
    confidence_mask: typing.Any,
) -> np.ndarray:
    """Validate ``confidence_mask`` shape, finiteness, sign, and diagonal.

    The shape-vs-variable-count check is performed separately at fit
    time. Returns a fresh float64 copy of the validated mask.
    """
    if not isinstance(confidence_mask, np.ndarray):
        raise ValueError(
            "confidence_mask must be a numpy ndarray; "
            f"got {type(confidence_mask).__name__}."
        )
    if confidence_mask.ndim != 2:
        raise ValueError(
            "confidence_mask must be a 2D array; "
            f"got ndim={confidence_mask.ndim}."
        )
    if confidence_mask.shape[0] != confidence_mask.shape[1]:
        raise ValueError(
            "confidence_mask must be square; "
            f"got shape {confidence_mask.shape}."
        )
    if not np.all(np.isfinite(confidence_mask)):
        raise ValueError(
            "confidence_mask must contain only finite values "
            "(no NaN, no infinite entries)."
        )
    if np.any(np.asarray(confidence_mask) < 0):
        raise ValueError(
            "confidence_mask must not contain negative values."
        )
    if np.any(np.diag(confidence_mask) != 0):
        raise ValueError(
            "confidence_mask must have zero on the diagonal "
            "(no self-loop priors)."
        )
    return np.asarray(confidence_mask, dtype=float).copy()


class SoftPriorDagmaLinear(DagmaLinear):
    """DAGMA-linear with a fixed targeted Frobenius prior gradient.

    The prior gradient
    ``G_prior = 2 * lambda_prior * (confidence_mask * W)`` is added to
    the parent's hand-coded ``Gobj`` immediately before the Adam update
    inside ``minimize``. It is not scaled by ``mu`` or by any other
    path-following parameter.

    Both ``lambda_prior`` and ``confidence_mask`` are fixed for the
    lifetime of the instance. ``confidence_mask`` is validated at
    construction except for the shape-vs-variable-count check, which
    runs at ``fit`` time when the variable count becomes available.
    ``exclude_edges`` and ``include_edges`` are never used; they are
    forwarded to the parent as ``None``.
    """

    def __init__(
        self,
        loss_type: str,
        *,
        lambda_prior: float,
        confidence_mask: np.ndarray,
        verbose: bool = False,
        dtype: type = np.float64,
    ) -> None:
        super().__init__(loss_type=loss_type, verbose=verbose, dtype=dtype)
        self._lambda_prior: float = _validate_lambda_prior(lambda_prior)
        self._confidence_mask: np.ndarray = _validate_confidence_mask(
            confidence_mask
        )

    def fit(
        self,
        X: np.ndarray,
        lambda1: float = 0.03,
        w_threshold: float = 0.3,
        T: int = 5,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: typing.Union[typing.List[float], float] = [1.0, 0.9, 0.8, 0.7, 0.6],
        warm_iter: int = 30000,
        max_iter: int = 60000,
        lr: float = 3e-4,
        checkpoint: int = 1000,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
    ) -> np.ndarray:
        """Fit DAGMA-linear with the configured prior gradient.

        Validates that ``X`` is 2D with at least two columns and that
        the stored ``confidence_mask`` matches the variable count of
        ``X``. ``exclude_edges`` and ``include_edges`` are passed to the
        parent as ``None``.

        Parameters
        ----------
        X : np.ndarray
            ``(n, d)`` training data. Mean-centred in place by the
            parent for ``loss_type='l2'``.
        lambda1 : float
            Coefficient of the parent's L1 penalty.
        w_threshold : float
            Post-fit threshold applied by the parent. Use ``0.0`` to
            preserve the continuous W matrix.
        T : int
            Number of DAGMA stages.
        mu_init, mu_factor : float
            Path-following parameters.
        s : list of float or float
            Domain control parameters per stage.
        warm_iter, max_iter : int
            Inner Adam budget for warm (t < T-1) and final (t = T-1) stages.
        lr : float
            Adam learning rate.
        checkpoint : int
            Inner iteration spacing for convergence checks.
        beta_1, beta_2 : float
            Adam hyperparameters.

        Returns
        -------
        np.ndarray
            The final continuous ``W`` matrix returned by the parent.

        Raises
        ------
        ValueError
            If ``X`` is not 2D with at least two columns, or if
            ``confidence_mask`` shape does not match the variable count.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a 2D array; got ndim={X_arr.ndim}."
            )
        if X_arr.shape[1] < 2:
            raise ValueError(
                f"X must have at least two columns (variables); "
                f"got shape {X_arr.shape}."
            )
        if self._confidence_mask.shape[0] != X_arr.shape[1]:
            raise ValueError(
                f"confidence_mask shape {self._confidence_mask.shape} "
                f"does not match the variable count {X_arr.shape[1]}."
            )
        return super().fit(
            X=X,
            lambda1=lambda1,
            w_threshold=w_threshold,
            T=T,
            mu_init=mu_init,
            mu_factor=mu_factor,
            s=s,
            warm_iter=warm_iter,
            max_iter=max_iter,
            lr=lr,
            checkpoint=checkpoint,
            beta_1=beta_1,
            beta_2=beta_2,
            exclude_edges=None,
            include_edges=None,
        )

    def minimize(
        self,
        W: np.ndarray,
        mu: float,
        max_iter: int,
        s: float,
        lr: float,
        tol: float = 1e-6,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        pbar=None,
    ):
        """Override of ``DagmaLinear.minimize`` injecting the prior gradient.

        The body mirrors the parent verbatim. The only addition is one
        accumulation of ``G_prior`` into ``Gobj`` immediately after the
        parent's ``Gobj`` assembly and before the Adam update. The
        accumulation is unconditional: when ``lambda_prior`` is zero or
        ``confidence_mask`` is the zero matrix the prior gradient is
        the zero matrix and the addition is a no-op.
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- "
            f"l1: {self.lambda1} for {max_iter} max iterations"
        )
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0

        for iter in range(1, max_iter + 1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):
                if iter == 1 or s <= 0.9:
                    self.vprint(
                        f"W went out of domain for s={s} at iteration {iter}"
                    )
                    return W, False
                else:
                    W += lr * grad
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f"Learning rate decreased to lr: {lr}")

            if self.loss_type == "l2":
                G_score = -mu * self.cov @ (self.Id - W)
            elif self.loss_type == "logistic":
                G_score = (
                    mu / self.n * self.X.T @ sigmoid(self.X @ W)
                    - mu * self.cov
                )

            Gobj = (
                G_score
                + mu * self.lambda1 * np.sign(W)
                + 2 * W * M.T
                + mask_inc * np.sign(W)
            )

            # Prior-gradient injection. Added once per iteration, before
            # the Adam update. Not scaled by mu.
            Gobj = Gobj + prior_gradient(
                W, self._confidence_mask, self._lambda_prior
            )

            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc

            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f"\nInner iteration {iter}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True


__all__ = [
    "SoftPriorDagmaLinear",
    "prior_gradient",
]
