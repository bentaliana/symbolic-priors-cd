"""Ground-truth compatibility checks for the evaluator.

All four checks must pass before any model comparison begins:

1. SID of the true graph against itself is exactly zero.
2. MMD between two independent same-intervention batches is near zero.
3. MMD between two independent observational batches is near zero.
4. do(X_j = x) clamps the target column exactly.

When no SID backend is wired in, ``sid_score`` raises
``NotImplementedError`` and ``check_sid_self_zero`` returns ``None``.
The assert-wrapper gate handles this via an explicit ``require_sid`` flag.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from symbolic_priors_cd.data import (
    Intervention,
    LinearGaussianSCM,
    intervene,
    sample_observational,
)
from symbolic_priors_cd.metrics.interventional import mmd_rbf_unbiased, sid_score


class CompatibilityReport(TypedDict):
    """Structured output from :func:`run_ground_truth_compatibility_checks`."""

    sid_self_zero_status: Literal["passed", "deferred", "failed"]
    sid_self_zero_value: int | None
    mmd_same_intervention: float
    mmd_same_observational: float
    do_clamping_max_deviation: float


def _derive_sid_status(
    value: int | None,
) -> Literal["passed", "deferred", "failed"]:
    if value is None:
        return "deferred"
    return "passed" if value == 0 else "failed"


def check_sid_self_zero(true_dag: np.ndarray) -> int | None:
    """Call ``sid_score(true_dag, true_dag)`` and return the result.

    Returns ``None`` if SID is not yet implemented (``NotImplementedError``).
    Any input-validation error from ``sid_score`` propagates unchanged.

    Parameters
    ----------
    true_dag : np.ndarray, square, dtype bool
        Ground-truth DAG adjacency matrix.

    Returns
    -------
    int or None
        ``0`` if the check passes; any other integer if it fails; ``None`` if
        no SID backend is available.
    """
    try:
        return sid_score(true_dag, true_dag)
    except NotImplementedError:
        return None


def check_mmd_same_intervention(
    scm: LinearGaussianSCM,
    intervention: Intervention,
    n_samples: int = 1000,
    seed_a: int = 0,
    seed_b: int = 1,
) -> float:
    """MMD between two independent interventional batches from the same SCM.

    ``seed_a`` and ``seed_b`` are distinct fixed seeds ensuring the two batches
    are statistically independent. With ``n_samples=1000`` the estimator
    variance is small enough that the result is reliably near zero when both
    batches come from the same distribution.

    Parameters
    ----------
    scm : LinearGaussianSCM
    intervention : Intervention
    n_samples : int
        Samples drawn per batch. Default 1000.
    seed_a : int
        Seed for the first batch.
    seed_b : int
        Seed for the second batch.

    Returns
    -------
    float
        Unbiased MMD squared. Expected to be near zero for a correct SCM.
    """
    sampler = intervene(scm, intervention)
    x_a = sampler.sample(n_samples, rng=seed_a)
    x_b = sampler.sample(n_samples, rng=seed_b)
    return mmd_rbf_unbiased(x_a, x_b)


def check_mmd_same_observational(
    scm: LinearGaussianSCM,
    n_samples: int = 1000,
    seed_a: int = 0,
    seed_b: int = 1,
) -> float:
    """MMD between two independent observational batches from the same SCM.

    ``seed_a`` and ``seed_b`` are distinct fixed seeds ensuring independence.
    ``n_samples=1000`` default balances variance reduction against runtime;
    the exact clamping check uses a much smaller default because it is
    deterministic.

    Parameters
    ----------
    scm : LinearGaussianSCM
    n_samples : int
        Samples drawn per batch. Default 1000.
    seed_a : int
        Seed for the first batch.
    seed_b : int
        Seed for the second batch.

    Returns
    -------
    float
        Unbiased MMD squared. Expected to be near zero for a correct SCM.
    """
    x_a = sample_observational(scm, n_samples, rng=seed_a)
    x_b = sample_observational(scm, n_samples, rng=seed_b)
    return mmd_rbf_unbiased(x_a, x_b)


def check_do_clamping(
    scm: LinearGaussianSCM,
    intervention: Intervention,
    n_samples: int = 100,
    seed: int = 0,
) -> float:
    """Maximum absolute deviation of the target column from the intervention value.

    ``n_samples=100`` is sufficient because clamping is exact and deterministic;
    larger samples add no diagnostic value here.

    Parameters
    ----------
    scm : LinearGaussianSCM
    intervention : Intervention
    n_samples : int
        Number of samples to draw. Default 100.
    seed : int
        RNG seed.

    Returns
    -------
    float
        Maximum absolute deviation; zero means the target column is perfectly
        clamped.
    """
    sampler = intervene(scm, intervention)
    X = sampler.sample(n_samples, rng=seed)
    return float(np.abs(X[:, intervention.target] - intervention.value).max())


def run_ground_truth_compatibility_checks(
    scm: LinearGaussianSCM,
    intervention: Intervention,
    n_samples: int = 1000,
) -> CompatibilityReport:
    """Run all four compatibility checks and return a structured report.

    ``n_samples`` is forwarded to the MMD checks. The clamping check uses its
    own default of 100, since clamping is exact and does not benefit from
    larger samples.

    Parameters
    ----------
    scm : LinearGaussianSCM
    intervention : Intervention
    n_samples : int
        Sample count for MMD checks. Default 1000.

    Returns
    -------
    CompatibilityReport
        Dict with keys ``sid_self_zero_status``, ``sid_self_zero_value``,
        ``mmd_same_intervention``, ``mmd_same_observational``, and
        ``do_clamping_max_deviation``.
    """
    sid_value = check_sid_self_zero(scm.adjacency)
    return CompatibilityReport(
        sid_self_zero_status=_derive_sid_status(sid_value),
        sid_self_zero_value=sid_value,
        mmd_same_intervention=check_mmd_same_intervention(
            scm, intervention, n_samples
        ),
        mmd_same_observational=check_mmd_same_observational(scm, n_samples),
        do_clamping_max_deviation=check_do_clamping(scm, intervention),
    )


def assert_ground_truth_compatibility(
    scm: LinearGaussianSCM,
    intervention: Intervention,
    *,
    mmd_tolerance: float = 0.01,
    clamp_tolerance: float = 1e-12,
    require_sid: bool = False,
) -> None:
    """Assert that the evaluator passes all ground-truth compatibility gates.

    Raises ``AssertionError`` with the full structured report if any gate
    fails. Tolerances live only here, not in the primitive check functions.

    Gate logic:

    - ``sid_self_zero_status == "failed"`` is always a hard error.
    - ``sid_self_zero_status == "deferred"`` fails only when ``require_sid=True``.
    - Both MMD checks use ``abs(value) < mmd_tolerance`` (unbiased MMD can be
      slightly negative).
    - Clamping uses ``value < clamp_tolerance``.

    Parameters
    ----------
    scm : LinearGaussianSCM
    intervention : Intervention
    mmd_tolerance : float
        Upper bound on ``abs(MMD^2)`` for both same-distribution MMD checks.
    clamp_tolerance : float
        Upper bound on the maximum absolute clamping deviation.
    require_sid : bool
        If ``True``, a deferred SID check is treated as a failure.

    Raises
    ------
    AssertionError
        If any gate fails. Message includes the list of failures and the full
        compatibility report.
    """
    report = run_ground_truth_compatibility_checks(scm, intervention)
    failures: list[str] = []

    status = report["sid_self_zero_status"]
    if status == "failed":
        failures.append(
            f"SID self-zero check failed (returned {report['sid_self_zero_value']})"
        )
    elif status == "deferred" and require_sid:
        failures.append("SID is deferred but require_sid=True")

    mmd_iv = report["mmd_same_intervention"]
    if abs(mmd_iv) >= mmd_tolerance:
        failures.append(
            f"MMD (same intervention) = {mmd_iv:.6g} is not < {mmd_tolerance}"
        )

    mmd_obs = report["mmd_same_observational"]
    if abs(mmd_obs) >= mmd_tolerance:
        failures.append(
            f"MMD (same observational) = {mmd_obs:.6g} is not < {mmd_tolerance}"
        )

    clamp_dev = report["do_clamping_max_deviation"]
    if clamp_dev >= clamp_tolerance:
        failures.append(
            f"do-clamping max deviation = {clamp_dev:.6g} is not < {clamp_tolerance}"
        )

    if failures:
        raise AssertionError(
            f"Ground truth compatibility check failed.\n"
            f"Failures: {failures}\n"
            f"Full report: {dict(report)}"
        )
