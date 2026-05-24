"""Lambda_prior smoke calibration on main-study calibration seeds.

Runs a production-scale lambda_prior pilot using calibration seeds 401
and 402 only. For each calibration seed the script first runs two
consistency checks (zero-mask gate and zero-lambda gate) against a
prior-free DAGMA baseline; only when both checks pass does it run the
soft-prior DAGMA fit at each candidate lambda_prior. A single
true-positive learned edge is selected per seed as the probe target.
The script then applies the joint per-seed minimum-passing rule to
recommend a smallest lambda_prior that achieves the configured soft-
suppression window on every calibration seed.

The script writes a JSON summary and a flat CSV table under
``inspection/probes/output/lambda_prior_calibration/``. It does not
write under ``results/`` and does not run any headline evaluation.
The recommended lambda_prior is a probe output intended for human
review; it is not auto-applied anywhere.

Executable form
---------------
    python experiments/main_study/calibration_lambda_prior.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.wrappers._dagma_fit import (
    run_dagma_fit,
    run_soft_prior_dagma_fit,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig
from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALIBRATION_SEEDS: tuple[int, ...] = (401, 402)
EVALUATION_SEEDS: tuple[int, ...] = (501, 502, 503, 504, 505, 506, 507)
LAMBDA_PRIOR_CANDIDATES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)

N_NODES = 10
EXPECTED_EDGES = 20
N_TRAIN = 1000
CONDITION = "standardised"

DAGMA_LAMBDA1 = 0.1
DAGMA_THRESHOLD = 0.3
DAGMA_WARM_ITER = 20000
DAGMA_MAX_ITER = 70000
DAGMA_LR = 3e-4
DAGMA_BETAS: tuple[float, float] = (0.99, 0.999)
W_THRESHOLD_INTERNAL = 0.0

# T, s, mu_init, mu_factor, loss_type mirror the project's frozen
# DAGMAConfig defaults so the calibration probe runs at the same
# DAGMA operating point as the selection-study production fits.
DAGMA_T = 4
DAGMA_S: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
DAGMA_MU_INIT = 1.0
DAGMA_MU_FACTOR = 0.1
DAGMA_LOSS_TYPE = "l2"

# Target selection thresholds, tried in order.
TARGET_THRESHOLDS: tuple[float, ...] = (0.3, 0.2, 0.1)

# Acceptance window for the soft-suppression diagnosis.
RATIO_MIN_PASS = 0.05
RATIO_MAX_PASS = 0.5
SOFT_ABS_FLOOR = 0.01
SOFT_ABS_HARD_FLOOR = 1e-6

# Step A tolerance for the consistency-check deltas.
STEP_A_DELTA_TOL = 1e-10

# Data-seed derivation: a single integer drives both SCM generation
# and the observational-sampling Generator. The SCM generator and the
# observational sampler construct independent np.random.default_rng
# instances internally, so identical integer seeds reproduce identical
# (SCM, X) pairs without cross-call state leakage.
DATA_SEED_DERIVATION_LABEL = (
    "graph_seed = train_data_seed = calibration_seed; "
    "independent np.random.default_rng(int) per call"
)

# Output paths anchored to the project root so the script works from
# any current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    _PROJECT_ROOT
    / "inspection"
    / "probes"
    / "output"
    / "lambda_prior_calibration"
)

JSON_OUTPUT_NAME = "lambda_prior_calibration.json"
CSV_OUTPUT_NAME = "lambda_prior_calibration.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "seed",
    "train_data_seed",
    "target_i",
    "target_j",
    "target_threshold_used",
    "lambda_prior",
    "base_abs",
    "soft_abs",
    "ratio",
    "reduction_percent",
    "candidate_status",
    "passes",
    "step_a_zero_mask_delta",
    "step_a_zero_lambda_delta",
)


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def validate_calibration_seeds(
    calibration_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
) -> None:
    """Raise ``ValueError`` if calibration and evaluation seed sets overlap.

    The check is symmetric: any non-empty intersection is rejected,
    including the case where a single calibration seed happens to
    appear in the evaluation seed pool.

    Parameters
    ----------
    calibration_seeds : iterable of int
        Seeds used by this calibration script.
    evaluation_seeds : iterable of int
        Headline evaluation seed pool that must remain untouched.
    """
    cal_set = set(int(s) for s in calibration_seeds)
    eval_set = set(int(s) for s in evaluation_seeds)
    overlap = cal_set & eval_set
    if overlap:
        raise ValueError(
            "Calibration seeds must not overlap with evaluation seeds. "
            f"Overlap: {sorted(overlap)}."
        )


def select_true_positive_target(
    true_adjacency: np.ndarray,
    W_prior_free: np.ndarray,
    thresholds: Iterable[float],
) -> Optional[tuple[int, int, float]]:
    """Select the strongest true-positive edge above one of the thresholds.

    For each threshold in order, collect off-diagonal positions
    ``(i, j)`` where ``true_adjacency[i, j]`` is True and
    ``abs(W_prior_free[i, j]) >= threshold``. Returns the ``(i, j)``
    with the largest absolute value and the threshold under which it
    was selected. Returns ``None`` when no off-diagonal true-positive
    edge meets any of the supplied thresholds.

    Parameters
    ----------
    true_adjacency : np.ndarray
        Boolean adjacency of the data-generating SCM.
    W_prior_free : np.ndarray
        Continuous adjacency from a prior-free DAGMA fit.
    thresholds : iterable of float
        Threshold tiers, tried in order.

    Returns
    -------
    tuple[int, int, float] or None
        ``(i, j, threshold_used)`` or ``None``.
    """
    d = W_prior_free.shape[0]
    abs_w = np.abs(W_prior_free)
    for th in thresholds:
        best: Optional[tuple[int, int, float]] = None
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if not bool(true_adjacency[i, j]):
                    continue
                if abs_w[i, j] < th:
                    continue
                if best is None or abs_w[i, j] > best[2]:
                    best = (int(i), int(j), float(abs_w[i, j]))
        if best is not None:
            return (best[0], best[1], float(th))
    return None


def evaluate_candidate(
    base_abs: float, soft_abs: float
) -> tuple[str, bool, float]:
    """Classify a ``(base_abs, soft_abs)`` pair into the candidate window.

    Acceptance window (all conditions required):

    - ``RATIO_MIN_PASS <= soft_abs / base_abs <= RATIO_MAX_PASS``
    - ``soft_abs >= SOFT_ABS_FLOOR``
    - ``soft_abs > SOFT_ABS_HARD_FLOOR``

    Otherwise the result is classified as ``too_weak`` when the ratio
    exceeds ``RATIO_MAX_PASS`` and ``too_strong`` in every other case.

    Parameters
    ----------
    base_abs : float
        ``abs(W_prior_free[target_i, target_j])``.
    soft_abs : float
        ``abs(W_soft[target_i, target_j])`` at the candidate
        ``lambda_prior``.

    Returns
    -------
    tuple[str, bool, float]
        ``(candidate_status, passes, ratio)``.
    """
    if base_abs <= 0.0 or not math.isfinite(base_abs):
        return ("too_strong", False, 0.0)
    ratio = float(soft_abs) / float(base_abs)
    if ratio > RATIO_MAX_PASS:
        return ("too_weak", False, ratio)
    if (
        ratio < RATIO_MIN_PASS
        or soft_abs < SOFT_ABS_FLOOR
        or soft_abs <= SOFT_ABS_HARD_FLOOR
    ):
        return ("too_strong", False, ratio)
    return ("passed", True, ratio)


def recommend_lambda(
    candidate_rows: list[dict],
    calibration_seeds: Iterable[int],
    per_seed_error: Optional[dict[int, str]] = None,
) -> tuple[Optional[float], bool, Optional[str]]:
    """Apply the joint per-seed minimum-passing recommendation rule.

    Returns ``(recommended_lambda_prior, grid_needs_review,
    grid_review_reason)``. The recommendation is the larger of the
    two per-seed minimum-passing candidates. The recommendation is
    ``None`` when any seed has no passing candidate or when any
    upstream per-seed error blocks candidate evaluation.

    Parameters
    ----------
    candidate_rows : list of dict
        Per-(seed, lambda_prior) records. Each row must include
        ``seed``, ``lambda_prior``, ``passes``, and
        ``candidate_status``.
    calibration_seeds : iterable of int
        Seeds expected to be present in ``candidate_rows``.
    per_seed_error : dict[int, str] or None
        Per-seed error labels for seeds that failed before candidate
        evaluation. Recognised labels are ``"infrastructure_failure"``
        and ``"target_selection_failed"``.

    Returns
    -------
    tuple[Optional[float], bool, Optional[str]]
        ``(recommended_lambda_prior, grid_needs_review,
        grid_review_reason)``.
    """
    per_seed_error = per_seed_error or {}
    seeds = tuple(int(s) for s in calibration_seeds)

    if per_seed_error:
        reasons = list(per_seed_error.values())
        if "infrastructure_failure" in reasons:
            return (None, True, "infrastructure_failure")
        if "target_selection_failed" in reasons:
            return (None, True, "target_selection_failed")
        return (None, True, "mixed_or_seed_inconsistent")

    per_seed_min: dict[int, Optional[float]] = {}
    for s in seeds:
        rows = [r for r in candidate_rows if int(r["seed"]) == s]
        if not rows:
            return (None, True, "infrastructure_failure")
        passing = sorted(
            float(r["lambda_prior"]) for r in rows if bool(r["passes"])
        )
        per_seed_min[s] = passing[0] if passing else None

    if all(v is not None for v in per_seed_min.values()):
        return (max(per_seed_min.values()), False, None)

    statuses = [r["candidate_status"] for r in candidate_rows]
    if statuses and all(st == "too_strong" for st in statuses):
        return (None, True, "all_too_strong")
    if statuses and all(st == "too_weak" for st in statuses):
        return (None, True, "all_too_weak")
    return (None, True, "mixed_or_seed_inconsistent")


def write_calibration_outputs(
    summary: dict,
    rows: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the JSON summary and the flat CSV rows under ``output_dir``.

    Creates ``output_dir`` if missing. Returns the JSON path and the
    CSV path. CSV columns are restricted to ``CSV_COLUMNS``; row keys
    outside that set are silently ignored, and missing keys are
    written as empty strings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_OUTPUT_NAME
    csv_path = output_dir / CSV_OUTPUT_NAME
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)
        fh.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {col: row.get(col, "") for col in CSV_COLUMNS}
            )
    return json_path, csv_path


# ---------------------------------------------------------------------------
# Internal: production-scale fit utilities
# ---------------------------------------------------------------------------


def _build_dagma_config() -> DAGMAConfig:
    """Return a DAGMAConfig at the calibration probe's production point."""
    return DAGMAConfig(
        T=DAGMA_T,
        lambda1=DAGMA_LAMBDA1,
        s=DAGMA_S,
        mu_init=DAGMA_MU_INIT,
        mu_factor=DAGMA_MU_FACTOR,
        w_threshold_internal=W_THRESHOLD_INTERNAL,
        lr=DAGMA_LR,
        warm_iter=DAGMA_WARM_ITER,
        max_iter=DAGMA_MAX_ITER,
        beta_1=DAGMA_BETAS[0],
        beta_2=DAGMA_BETAS[1],
        loss_type=DAGMA_LOSS_TYPE,
    )


def _prepare_training_data(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate SCM, draw observational data, return model-frame X.

    Returns
    -------
    tuple
        ``(X_model_frame, true_adjacency, train_data_seed)``.
    """
    train_data_seed = int(seed)
    scm = generate_linear_gaussian_scm(
        n_nodes=N_NODES,
        expected_edges=EXPECTED_EDGES,
        seed=int(seed),
        noise_scale=1.0,
    )
    X_raw = sample_observational(
        scm, n_samples=N_TRAIN, rng=train_data_seed
    )
    transform = StandardisedTransform().fit(X_raw)
    X_model = transform.transform(X_raw)
    true_adjacency = np.asarray(scm.adjacency, dtype=bool).copy()
    return X_model, true_adjacency, train_data_seed


def _fit_prior_free(X_model: np.ndarray, cfg: DAGMAConfig) -> np.ndarray:
    res = run_dagma_fit(X_model.copy(), cfg)
    return res.W


def _fit_soft_prior(
    X_model: np.ndarray,
    cfg: DAGMAConfig,
    lambda_prior: float,
    confidence_mask: np.ndarray,
) -> np.ndarray:
    res = run_soft_prior_dagma_fit(
        X_model.copy(),
        cfg,
        lambda_prior=float(lambda_prior),
        confidence_mask=confidence_mask,
    )
    return res.W


def _run_step_a_checks(
    X_model: np.ndarray,
    cfg: DAGMAConfig,
    W_prior_free: np.ndarray,
) -> dict:
    """Run Step A consistency checks. Raise ``RuntimeError`` on failure."""
    d = W_prior_free.shape[0]

    # A-1: zero mask + max-candidate lambda must reproduce the prior-free fit.
    zero_mask = np.zeros((d, d), dtype=float)
    W_a1 = _fit_soft_prior(
        X_model,
        cfg,
        lambda_prior=max(LAMBDA_PRIOR_CANDIDATES),
        confidence_mask=zero_mask,
    )
    delta_a1 = float(np.max(np.abs(W_a1 - W_prior_free)))
    a_thresh_pf = (np.abs(W_prior_free) >= DAGMA_THRESHOLD).astype(bool)
    a_thresh_a1 = (np.abs(W_a1) >= DAGMA_THRESHOLD).astype(bool)
    if delta_a1 > STEP_A_DELTA_TOL:
        raise RuntimeError(
            "Step A-1 zero-mask gate failed: max |W_soft - W_prior_free| "
            f"= {delta_a1:.3e} exceeds tolerance {STEP_A_DELTA_TOL:.3e}."
        )
    if not np.array_equal(a_thresh_pf, a_thresh_a1):
        raise RuntimeError(
            "Step A-1 thresholded adjacency at "
            f"{DAGMA_THRESHOLD} differs from the prior-free baseline."
        )

    # A-2: nonzero mask + zero lambda must also reproduce the prior-free fit.
    abs_pf = np.abs(W_prior_free)
    best: Optional[tuple[int, int, float]] = None
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if abs_pf[i, j] >= 0.1:
                if best is None or abs_pf[i, j] > best[2]:
                    best = (i, j, float(abs_pf[i, j]))
    a2_fallback = False
    if best is None:
        a2_fallback = True
        off_diag_mask = ~np.eye(d, dtype=bool)
        flat_idx = int(
            np.argmax(np.where(off_diag_mask, abs_pf, -np.inf))
        )
        i, j = int(flat_idx // d), int(flat_idx % d)
        best = (i, j, float(abs_pf[i, j]))
    p, q = int(best[0]), int(best[1])
    a2_mask = np.zeros((d, d), dtype=float)
    a2_mask[p, q] = 1.0
    W_a2 = _fit_soft_prior(
        X_model, cfg, lambda_prior=0.0, confidence_mask=a2_mask
    )
    delta_a2 = float(np.max(np.abs(W_a2 - W_prior_free)))
    delta_a2_entry = float(abs(W_a2[p, q] - W_prior_free[p, q]))
    if delta_a2 > STEP_A_DELTA_TOL:
        raise RuntimeError(
            "Step A-2 zero-lambda gate failed: max |W_soft - W_prior_free| "
            f"= {delta_a2:.3e} exceeds tolerance {STEP_A_DELTA_TOL:.3e}."
        )
    if delta_a2_entry > STEP_A_DELTA_TOL:
        raise RuntimeError(
            f"Step A-2 entry delta failed at ({p}, {q}): "
            f"{delta_a2_entry:.3e} exceeds tolerance "
            f"{STEP_A_DELTA_TOL:.3e}."
        )

    return {
        "step_a_zero_mask_delta": delta_a1,
        "step_a_zero_lambda_delta": delta_a2,
        "step_a_zero_lambda_entry_delta": delta_a2_entry,
        "step_a_zero_lambda_entry": [int(p), int(q)],
        "step_a2_used_fallback_threshold": bool(a2_fallback),
    }


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_lambda_prior_calibration(
    output_dir: Optional[Path] = None,
) -> dict:
    """Execute the calibration probe and write JSON/CSV outputs.

    Parameters
    ----------
    output_dir : pathlib.Path or None
        Where to write the JSON and CSV artefacts. ``None`` uses
        :data:`DEFAULT_OUTPUT_DIR`.

    Returns
    -------
    dict
        The full summary dictionary written to JSON.
    """
    validate_calibration_seeds(CALIBRATION_SEEDS, EVALUATION_SEEDS)
    out_dir = (
        Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    )

    cfg = _build_dagma_config()
    candidate_rows: list[dict] = []
    per_seed_summary: list[dict] = []
    per_seed_error: dict[int, str] = {}

    for seed in CALIBRATION_SEEDS:
        seed_info: dict[str, Any] = {
            "seed": int(seed),
            "train_data_seed": int(seed),
        }
        try:
            X_model, true_adj, train_data_seed = _prepare_training_data(seed)
            seed_info["train_data_seed"] = int(train_data_seed)
            print(
                f"[seed {seed}] fitting prior-free DAGMA on n={N_TRAIN} "
                f"standardised samples ...",
                flush=True,
            )
            W_pf = _fit_prior_free(X_model, cfg)
            print(
                f"[seed {seed}] running Step A consistency checks ...",
                flush=True,
            )
            step_a = _run_step_a_checks(X_model, cfg, W_pf)
        except Exception as exc:
            tb = traceback.format_exc()
            print(
                f"[seed {seed}] infrastructure failure: {exc}\n{tb}",
                flush=True,
            )
            per_seed_error[int(seed)] = "infrastructure_failure"
            seed_info["error"] = "infrastructure_failure"
            seed_info["error_detail"] = str(exc)
            per_seed_summary.append(seed_info)
            continue

        seed_info.update(
            {
                "step_a_zero_mask_delta": step_a["step_a_zero_mask_delta"],
                "step_a_zero_lambda_delta": step_a[
                    "step_a_zero_lambda_delta"
                ],
                "step_a_zero_lambda_entry": step_a[
                    "step_a_zero_lambda_entry"
                ],
                "step_a_zero_lambda_entry_delta": step_a[
                    "step_a_zero_lambda_entry_delta"
                ],
                "step_a2_used_fallback_threshold": step_a[
                    "step_a2_used_fallback_threshold"
                ],
            }
        )

        target = select_true_positive_target(
            true_adj, W_pf, TARGET_THRESHOLDS
        )
        if target is None:
            print(
                f"[seed {seed}] no true-positive learned edge at any of "
                f"{TARGET_THRESHOLDS}. Target selection failed.",
                flush=True,
            )
            per_seed_error[int(seed)] = "target_selection_failed"
            seed_info["target_selection_failed"] = True
            per_seed_summary.append(seed_info)
            continue

        target_i, target_j, threshold_used = target
        base_abs = float(np.abs(W_pf[target_i, target_j]))
        seed_info.update(
            {
                "target_i": int(target_i),
                "target_j": int(target_j),
                "target_threshold_used": float(threshold_used),
                "base_abs": base_abs,
            }
        )

        d = W_pf.shape[0]
        mask = np.zeros((d, d), dtype=float)
        mask[target_i, target_j] = 1.0

        for lam in LAMBDA_PRIOR_CANDIDATES:
            print(
                f"[seed {seed}] candidate lambda_prior={lam} ...",
                flush=True,
            )
            W_soft = _fit_soft_prior(X_model, cfg, lam, mask)
            soft_abs = float(np.abs(W_soft[target_i, target_j]))
            status, passes, ratio = evaluate_candidate(base_abs, soft_abs)
            reduction_percent = (
                (1.0 - ratio) * 100.0 if base_abs > 0 else 0.0
            )
            candidate_rows.append(
                {
                    "seed": int(seed),
                    "train_data_seed": int(train_data_seed),
                    "target_i": int(target_i),
                    "target_j": int(target_j),
                    "target_threshold_used": float(threshold_used),
                    "lambda_prior": float(lam),
                    "base_abs": base_abs,
                    "soft_abs": soft_abs,
                    "ratio": ratio,
                    "reduction_percent": reduction_percent,
                    "candidate_status": status,
                    "passes": bool(passes),
                    "step_a_zero_mask_delta": step_a[
                        "step_a_zero_mask_delta"
                    ],
                    "step_a_zero_lambda_delta": step_a[
                        "step_a_zero_lambda_delta"
                    ],
                }
            )

        per_seed_summary.append(seed_info)

    recommended, needs_review, reason = recommend_lambda(
        candidate_rows,
        CALIBRATION_SEEDS,
        per_seed_error=per_seed_error or None,
    )

    summary: dict[str, Any] = {
        "calibration_seeds": list(int(s) for s in CALIBRATION_SEEDS),
        "evaluation_seeds": list(int(s) for s in EVALUATION_SEEDS),
        "lambda_prior_candidates": list(
            float(lam) for lam in LAMBDA_PRIOR_CANDIDATES
        ),
        "constants": {
            "n_nodes": N_NODES,
            "expected_edges": EXPECTED_EDGES,
            "n_train": N_TRAIN,
            "condition": CONDITION,
            "dagma_lambda1": DAGMA_LAMBDA1,
            "dagma_threshold": DAGMA_THRESHOLD,
            "dagma_warm_iter": DAGMA_WARM_ITER,
            "dagma_max_iter": DAGMA_MAX_ITER,
            "dagma_lr": DAGMA_LR,
            "dagma_betas": list(DAGMA_BETAS),
            "w_threshold_internal": W_THRESHOLD_INTERNAL,
            "dagma_T": DAGMA_T,
            "dagma_s": list(DAGMA_S),
            "dagma_mu_init": DAGMA_MU_INIT,
            "dagma_mu_factor": DAGMA_MU_FACTOR,
            "dagma_loss_type": DAGMA_LOSS_TYPE,
            "ratio_min_pass": RATIO_MIN_PASS,
            "ratio_max_pass": RATIO_MAX_PASS,
            "soft_abs_floor": SOFT_ABS_FLOOR,
            "soft_abs_hard_floor": SOFT_ABS_HARD_FLOOR,
            "step_a_delta_tol": STEP_A_DELTA_TOL,
            "target_thresholds": list(TARGET_THRESHOLDS),
        },
        "data_seed_derivation": DATA_SEED_DERIVATION_LABEL,
        "selection_rule": (
            "For each calibration seed, take the minimum lambda_prior "
            "that passes the soft-suppression acceptance window. The "
            "recommended lambda_prior is the larger of the two per-seed "
            "minima. If any calibration seed has no passing candidate, "
            "the recommendation is null and grid_needs_review is true."
        ),
        "recommended_lambda_prior": (
            None if recommended is None else float(recommended)
        ),
        "grid_needs_review": bool(needs_review),
        "grid_review_reason": reason,
        "per_seed_summary": per_seed_summary,
        "candidate_rows": candidate_rows,
        "no_evaluation_seeds_used_confirmation": (
            "calibration_seeds="
            f"{list(int(s) for s in CALIBRATION_SEEDS)} have empty "
            "intersection with evaluation_seeds="
            f"{list(int(s) for s in EVALUATION_SEEDS)}."
        ),
    }

    json_path, csv_path = write_calibration_outputs(
        summary, candidate_rows, out_dir
    )
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return summary


def main() -> int:
    summary = run_lambda_prior_calibration()
    print("")
    print(
        f"recommended_lambda_prior: {summary['recommended_lambda_prior']}"
    )
    if summary["grid_needs_review"]:
        print(f"grid_review_reason: {summary['grid_review_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
