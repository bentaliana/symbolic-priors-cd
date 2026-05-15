"""Intervention description and interventional sampling for LinearGaussianSCM.

Owns the Intervention dataclass, InterventionalSampler, and the intervene
factory. Sampling delegates to the shared _ancestral_sample kernel in
_sampling.py so observational and interventional semantics are unified by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symbolic_priors_cd.data._sampling import _ancestral_sample
from symbolic_priors_cd.data.scm_generator import LinearGaussianSCM


@dataclass(frozen=True)
class Intervention:
    """A single-node hard intervention: do(X_target = value).

    Only single-node hard interventions are supported.
    """

    target: int
    value: float


@dataclass(frozen=True)
class InterventionalSampler:
    """Sampler for the post-intervention distribution under a hard do-intervention.

    Holds a reference to the original immutable SCM (no copy needed because the SCM
    is already deeply immutable) plus the intervention spec. Sampling delegates
    to _ancestral_sample with a clamp, so do-calculus semantics and the
    observational path share one implementation.
    """

    scm: LinearGaussianSCM
    intervention: Intervention

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | int,
    ) -> np.ndarray:
        """Sample from the interventional distribution do(X_target = value).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw. Must be positive.
        rng : np.random.Generator or int
            Source of randomness. Passing an ``int`` constructs a fresh
            ``np.random.default_rng(rng)`` on each call, so identical integer
            seeds reproduce identical sample matrices. Passing an existing
            ``Generator`` consumes its state in place; subsequent calls advance
            the same stream.

        Returns
        -------
        np.ndarray of shape (n_samples, n_nodes), dtype float64
            Column ``intervention.target`` is exactly ``intervention.value``
            in every row; all other columns follow the original structural
            equations, using the clamped value as a parent where applicable.
        """
        return _ancestral_sample(
            self.scm,
            n_samples,
            rng,
            clamp=(self.intervention.target, self.intervention.value),
        )


def intervene(
    scm: LinearGaussianSCM,
    intervention: Intervention,
) -> InterventionalSampler:
    """Create an interventional sampler for a hard do-intervention.

    Non-mutating: the original SCM is not modified. The returned sampler
    holds a reference to the original SCM plus the intervention spec.

    Parameters
    ----------
    scm : LinearGaussianSCM
        The pre-intervention system.
    intervention : Intervention
        The hard intervention to apply at sample time.

    Returns
    -------
    InterventionalSampler

    Raises
    ------
    ValueError
        If ``intervention.target`` is outside ``[0, scm.n_nodes)``.
    """
    if not (0 <= intervention.target < scm.n_nodes):
        raise ValueError(
            f"intervention.target={intervention.target} is outside valid range "
            f"[0, {scm.n_nodes})"
        )
    return InterventionalSampler(scm=scm, intervention=intervention)
