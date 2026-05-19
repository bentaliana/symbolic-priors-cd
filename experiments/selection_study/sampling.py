"""Per-intervention MMD sampling and aggregation.

Implements ``compute_per_intervention_records``, the function that
takes a fitted wrapper plus a configured intervention set and
returns per-intervention MMD records and the run-level MMD
aggregates the run-record schema requires.

The function is intentionally narrow. It does not build SCMs, fit
wrappers, write files, derive identity strings, or mutate the
manifest. The caller (the single-fit pipeline) is responsible for
all of those things.

Bandwidths use a local deterministic median-heuristic helper so the
recorded bandwidth value is exactly the bandwidth passed to the MMD
estimator. The unbiased RBF MMD is consumed unchanged from the
metrics package; negative values are preserved verbatim.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from symbolic_priors_cd.data import Intervention, intervene
from symbolic_priors_cd.metrics import mmd_rbf_unbiased


# Sample count for the schema-conformance gate only. Selection-study
# MMD sample sizes live elsewhere; this constant is intentionally
# small to keep the gate quick.
SCHEMA_GATE_MMD_N_SAMPLES = 64


_POLICY_RESIDUAL_FITTED = "residual_fitted"
_POLICY_UNIT_VARIANCE = "unit_variance"
_POLICY_DCDI_NATIVE = "dcdi_native"


# Allowed values for the top-level ``sampler_status`` argument. The
# value ``"unavailable_other"`` is intentionally NOT in this set: it
# is reserved for the per-intervention ``mmd_status`` field when the
# top-level sampler is available but the MMD itself cannot be
# computed (degenerate bandwidth, non-finite estimator output, etc.).
_VALID_TOP_LEVEL_SAMPLER_STATUSES = (
    "available",
    "unavailable_invalid_graph",
    "unavailable_no_api",
    "unavailable_unresolved_noise_policy",
)

_BANDWIDTH_SWEEP_MULTIPLIERS = (
    ("0.5x", 0.5),
    ("1.0x", 1.0),
    ("2.0x", 2.0),
)
_BANDWIDTH_SWEEP_KEYS = tuple(k for k, _ in _BANDWIDTH_SWEEP_MULTIPLIERS)


def _empty_bandwidth_sweep() -> dict[str, Optional[float]]:
    """Return a fresh bandwidth-sweep mapping with all values None."""
    return {k: None for k in _BANDWIDTH_SWEEP_KEYS}


def _median_bandwidth_deterministic(
    x: np.ndarray, y: np.ndarray
) -> float:
    """Median pairwise squared distance over the concatenated samples.

    The two inputs are coerced to float64 C-contiguous arrays before
    any computation, so the result is bitwise-identical across
    input memory layouts on a fixed BLAS backend. Concatenation
    order is x first, then y. Squared distances are computed by the
    squared-norm expansion and clamped at zero to absorb sign
    rounding from finite-precision arithmetic. The upper triangle
    (k=1, excluding self-pairs) is flattened in row-major order and
    the bandwidth is ``np.median`` over that flat vector.
    """
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    z = np.vstack([x_c, y_c])
    norms = np.sum(z ** 2, axis=1, keepdims=True)
    sq = norms + norms.T - 2.0 * (z @ z.T)
    np.maximum(sq, 0.0, out=sq)
    iu_rows, iu_cols = np.triu_indices(z.shape[0], k=1)
    return float(np.median(sq[iu_rows, iu_cols]))


def _call_wrapper_sampler(
    wrapper: Any,
    intervention: Intervention,
    n_samples: int,
    sample_seed: int,
    sampler_policy_used: str,
    policy_for_call: str,
) -> Optional[np.ndarray]:
    """Dispatch ``sample_interventional`` according to the schema policy.

    Wrappers whose policy is a DAGMA noise policy accept a
    ``noise_policy`` keyword. The DCDI native sampling path does
    not. Passing ``noise_policy`` to a DCDI-shaped wrapper would be
    a programming error and is suppressed here.
    """
    if sampler_policy_used in (_POLICY_RESIDUAL_FITTED, _POLICY_UNIT_VARIANCE):
        return wrapper.sample_interventional(
            intervention,
            n_samples,
            sample_seed=sample_seed,
            noise_policy=policy_for_call,
        )
    if sampler_policy_used == _POLICY_DCDI_NATIVE:
        return wrapper.sample_interventional(
            intervention,
            n_samples,
            sample_seed=sample_seed,
        )
    raise ValueError(
        f"unsupported sampler_policy_used={sampler_policy_used!r}"
    )


def _compute_one_intervention(
    *,
    scm: Any,
    wrapper: Any,
    inter: dict,
    seeds: Any,
    sampler_status: str,
    sampler_unavailable_reason: Optional[str],
    sampler_policy_used: str,
    policy_for_call: str,
    preprocessor: Any,
    n_samples: int,
) -> dict:
    """Compute one per-intervention record.

    The within-record consistency rules between
    ``sampler_status_for_intervention``, ``mmd_status``, and
    ``mmd_value`` are enforced here: a mechanically-unavailable
    sampler propagates its status into ``mmd_status``; an available
    sampler whose MMD cannot be computed yields ``"unavailable_other"``;
    a successful path yields ``mmd_status == "available"`` with a
    finite ``mmd_value`` (which may be negative because the unbiased
    estimator is not clipped).
    """
    iid = str(inter["intervention_id"])
    target = int(inter["target_node"])
    value_raw = float(inter["value_raw"])
    value_model = float(
        preprocessor.transform_intervention_value(value_raw, target)
    )
    record: dict = {
        "intervention_id": iid,
        "target_node": target,
        "value_raw": value_raw,
        "value_model_frame": value_model,
        "ground_truth_sampling_seed": int(seeds.ground_truth_sampling_seed),
        "model_sampling_seed": int(seeds.model_sampling_seed),
        "n_ground_truth_samples": 0,
        "n_model_samples": 0,
        "mmd_value": None,
        "mmd_status": None,
        "bandwidth_used": None,
        "bandwidth_sweep": _empty_bandwidth_sweep(),
        "sampler_status_for_intervention": sampler_status,
        "sampler_reason": None,
    }

    if sampler_status != "available":
        record["mmd_status"] = sampler_status
        record["sampler_reason"] = sampler_unavailable_reason
        return record

    intervention_obj = Intervention(target=target, value=value_raw)
    ground_truth_samples = intervene(scm, intervention_obj).sample(
        n_samples, rng=int(seeds.ground_truth_sampling_seed)
    )
    model_samples = _call_wrapper_sampler(
        wrapper=wrapper,
        intervention=intervention_obj,
        n_samples=n_samples,
        sample_seed=int(seeds.model_sampling_seed),
        sampler_policy_used=sampler_policy_used,
        policy_for_call=policy_for_call,
    )
    if model_samples is None:
        raise RuntimeError(
            "wrapper.sample_interventional returned None for intervention "
            f"{iid!r} despite sampler_status='available'"
        )
    if not isinstance(model_samples, np.ndarray):
        raise RuntimeError(
            "wrapper.sample_interventional returned a non-ndarray of type "
            f"{type(model_samples).__name__} for intervention {iid!r}"
        )
    expected_shape = (n_samples, scm.n_nodes)
    if model_samples.shape != expected_shape:
        raise RuntimeError(
            "wrapper.sample_interventional returned shape "
            f"{model_samples.shape} for intervention {iid!r}; "
            f"expected {expected_shape}"
        )

    try:
        bandwidth = _median_bandwidth_deterministic(
            ground_truth_samples, model_samples
        )
    except Exception as exc:
        record["mmd_status"] = "unavailable_other"
        record["sampler_reason"] = (
            f"bandwidth computation failed: {exc}"
        )
        record["n_ground_truth_samples"] = int(n_samples)
        record["n_model_samples"] = int(n_samples)
        return record

    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        record["mmd_status"] = "unavailable_other"
        record["sampler_reason"] = (
            f"median-heuristic bandwidth is {bandwidth}; expected a "
            "positive finite value"
        )
        record["n_ground_truth_samples"] = int(n_samples)
        record["n_model_samples"] = int(n_samples)
        return record

    sweep: dict[str, Optional[float]] = _empty_bandwidth_sweep()
    failed_reason: Optional[str] = None
    for key, mult in _BANDWIDTH_SWEEP_MULTIPLIERS:
        try:
            value = float(
                mmd_rbf_unbiased(
                    ground_truth_samples,
                    model_samples,
                    bandwidth=bandwidth * mult,
                )
            )
        except Exception as exc:
            failed_reason = f"MMD computation failed at {key}: {exc}"
            break
        if not np.isfinite(value):
            failed_reason = (
                f"MMD at {key} is non-finite ({value})"
            )
            break
        sweep[key] = value

    if failed_reason is not None:
        record["mmd_status"] = "unavailable_other"
        record["sampler_reason"] = failed_reason
        record["bandwidth_used"] = None
        record["bandwidth_sweep"] = _empty_bandwidth_sweep()
        record["n_ground_truth_samples"] = int(n_samples)
        record["n_model_samples"] = int(n_samples)
        return record

    record["mmd_status"] = "available"
    record["mmd_value"] = sweep["1.0x"]
    record["bandwidth_used"] = float(bandwidth)
    record["bandwidth_sweep"] = sweep
    record["n_ground_truth_samples"] = int(n_samples)
    record["n_model_samples"] = int(n_samples)
    return record


def compute_per_intervention_records(
    *,
    scm: Any,
    wrapper: Any,
    sampler_status: str,
    sampler_unavailable_reason: Optional[str],
    sampler_policy_used: str,
    intervention_set: list,
    per_intervention_seeds_map: dict,
    preprocessor: Any,
    n_samples: int = SCHEMA_GATE_MMD_N_SAMPLES,
) -> dict:
    """Compute per-intervention MMD records and run-level aggregates.

    Returns a dictionary with these keys:

    - ``records``: list of per-intervention records in the order of
      ``intervention_set``; each record carries every field
      required by the per-intervention schema.
    - ``mmd_primary``: arithmetic mean of ``mmd_value`` across
      records with ``mmd_status == "available"``, or ``None`` when
      none are available.
    - ``mmd_sensitivity_unit_variance``: same kind of mean for a
      second pass under the DAGMA unit-variance noise policy. Set
      only when ``sampler_policy_used == "residual_fitted"`` and
      ``sampler_status == "available"``; otherwise ``None``.
    - ``mmd_bandwidth_sweep``: mapping from ``"0.5x"``, ``"1.0x"``,
      ``"2.0x"`` to the arithmetic mean of the corresponding per-
      intervention sweep value across available records, or
      ``None`` per key when no available record produced a value at
      that multiplier.
    - ``mmd_bandwidth_used_value``: mapping from intervention_id to
      the per-intervention median-heuristic bandwidth, or ``None``
      when the intervention's MMD is unavailable.
    - ``mmd_available_count`` and ``mmd_missing_count``: integers
      whose sum equals ``len(records)``.

    Raises
    ------
    KeyError
        If an entry in ``intervention_set`` references an
        ``intervention_id`` not present in
        ``per_intervention_seeds_map``.
    RuntimeError
        If ``sampler_status == "available"`` but the wrapper
        returns ``None``, the wrong runtime type, or the wrong
        shape from ``sample_interventional``. Those are wrapper-API
        inconsistencies, not valid MMD-unavailable outcomes.
    ValueError
        If ``sampler_policy_used`` is not one of the supported
        schema values.
    """
    if sampler_policy_used not in (
        _POLICY_RESIDUAL_FITTED,
        _POLICY_UNIT_VARIANCE,
        _POLICY_DCDI_NATIVE,
    ):
        raise ValueError(
            f"unsupported sampler_policy_used={sampler_policy_used!r}"
        )
    if sampler_status not in _VALID_TOP_LEVEL_SAMPLER_STATUSES:
        raise ValueError(
            "sampler_status must be one of "
            f"{_VALID_TOP_LEVEL_SAMPLER_STATUSES}; "
            f"got sampler_status={sampler_status!r}. The value "
            "'unavailable_other' is reserved for per-intervention "
            "mmd_status when the top-level sampler is available but "
            "MMD cannot be computed."
        )
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError(
            "n_samples must be a plain int (not bool); "
            f"got n_samples={n_samples!r} of type "
            f"{type(n_samples).__name__}"
        )
    if n_samples < 2:
        raise ValueError(
            "n_samples must be at least 2 because the unbiased MMD "
            "estimator requires two or more samples per side; "
            f"got n_samples={n_samples}"
        )

    primary_records: list[dict] = []
    for inter in intervention_set:
        iid = str(inter["intervention_id"])
        if iid not in per_intervention_seeds_map:
            raise KeyError(
                "per_intervention_seeds_map is missing intervention_id "
                f"{iid!r}"
            )
        seeds = per_intervention_seeds_map[iid]
        primary_records.append(
            _compute_one_intervention(
                scm=scm,
                wrapper=wrapper,
                inter=inter,
                seeds=seeds,
                sampler_status=sampler_status,
                sampler_unavailable_reason=sampler_unavailable_reason,
                sampler_policy_used=sampler_policy_used,
                policy_for_call=sampler_policy_used,
                preprocessor=preprocessor,
                n_samples=n_samples,
            )
        )

    available_primary = [
        r for r in primary_records if r["mmd_status"] == "available"
    ]
    mmd_available_count = len(available_primary)
    mmd_missing_count = len(primary_records) - mmd_available_count

    if available_primary:
        mmd_primary: Optional[float] = float(
            np.mean([r["mmd_value"] for r in available_primary])
        )
    else:
        mmd_primary = None

    bandwidth_sweep_aggregate: dict[str, Optional[float]] = {}
    for key in _BANDWIDTH_SWEEP_KEYS:
        values = [
            r["bandwidth_sweep"][key]
            for r in available_primary
            if r["bandwidth_sweep"][key] is not None
        ]
        bandwidth_sweep_aggregate[key] = (
            float(np.mean(values)) if values else None
        )

    bandwidth_used_value: dict[str, Optional[float]] = {}
    for r in primary_records:
        bandwidth_used_value[r["intervention_id"]] = r["bandwidth_used"]

    mmd_sensitivity_unit_variance: Optional[float] = None
    if (
        sampler_policy_used == _POLICY_RESIDUAL_FITTED
        and sampler_status == "available"
    ):
        sensitivity_records: list[dict] = []
        for inter in intervention_set:
            iid = str(inter["intervention_id"])
            seeds = per_intervention_seeds_map[iid]
            sensitivity_records.append(
                _compute_one_intervention(
                    scm=scm,
                    wrapper=wrapper,
                    inter=inter,
                    seeds=seeds,
                    sampler_status=sampler_status,
                    sampler_unavailable_reason=sampler_unavailable_reason,
                    sampler_policy_used=_POLICY_RESIDUAL_FITTED,
                    policy_for_call=_POLICY_UNIT_VARIANCE,
                    preprocessor=preprocessor,
                    n_samples=n_samples,
                )
            )
        available_sensitivity = [
            r for r in sensitivity_records if r["mmd_status"] == "available"
        ]
        if available_sensitivity:
            mmd_sensitivity_unit_variance = float(
                np.mean([r["mmd_value"] for r in available_sensitivity])
            )

    return {
        "records": primary_records,
        "mmd_primary": mmd_primary,
        "mmd_sensitivity_unit_variance": mmd_sensitivity_unit_variance,
        "mmd_bandwidth_sweep": bandwidth_sweep_aggregate,
        "mmd_bandwidth_used_value": bandwidth_used_value,
        "mmd_available_count": int(mmd_available_count),
        "mmd_missing_count": int(mmd_missing_count),
    }
