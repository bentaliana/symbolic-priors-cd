"""Helper-level tests for the lambda_prior calibration probe.

These tests exercise the pure helper functions only and never run a
production DAGMA fit. Synthetic candidate rows are used for the
recommendation tests; ``select_true_positive_target`` is tested with
hand-built adjacency and weight matrices.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.main_study.calibration_lambda_prior import (
    CSV_COLUMNS,
    CSV_OUTPUT_NAME,
    EVALUATION_SEEDS,
    JSON_OUTPUT_NAME,
    LAMBDA_PRIOR_CANDIDATES,
    evaluate_candidate,
    recommend_lambda,
    select_true_positive_target,
    validate_calibration_seeds,
    write_calibration_outputs,
)


# ---------------------------------------------------------------------------
# T-1: criterion function
# ---------------------------------------------------------------------------


def test_criterion_passes_in_window():
    status, passes, ratio = evaluate_candidate(base_abs=1.0, soft_abs=0.3)
    assert status == "passed"
    assert passes is True
    assert ratio == pytest.approx(0.3)


def test_criterion_too_strong_when_soft_abs_below_floor():
    status, passes, _ = evaluate_candidate(base_abs=0.0166666666, soft_abs=0.005)
    # ratio = 0.005 / 0.01666... = 0.3 (in ratio window) but soft_abs < 0.01.
    assert status == "too_strong"
    assert passes is False


def test_criterion_too_strong_when_ratio_below_lower_bound():
    status, passes, ratio = evaluate_candidate(base_abs=2.0, soft_abs=0.02)
    # ratio = 0.01, below the 0.05 lower bound.
    assert status == "too_strong"
    assert passes is False
    assert ratio == pytest.approx(0.01)


def test_criterion_too_weak_when_ratio_above_upper_bound():
    status, passes, ratio = evaluate_candidate(base_abs=1.0 / 6.0, soft_abs=0.1)
    # ratio = 0.6, above the 0.5 upper bound.
    assert status == "too_weak"
    assert passes is False
    assert ratio == pytest.approx(0.6)


def test_criterion_too_strong_when_base_abs_is_zero():
    status, passes, ratio = evaluate_candidate(base_abs=0.0, soft_abs=0.1)
    assert status == "too_strong"
    assert passes is False
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# T-2: true-positive target selection
# ---------------------------------------------------------------------------


def test_target_selection_picks_strongest_true_positive_at_first_threshold():
    """Among true-positive edges, the strongest one wins at the first
    threshold that admits at least one of them."""
    d = 4
    true_adj = np.zeros((d, d), dtype=bool)
    true_adj[0, 1] = True
    true_adj[1, 2] = True
    true_adj[2, 3] = True
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 0.35   # true positive, above 0.3
    W[1, 2] = 0.55   # true positive, above 0.3, strongest TP
    W[2, 3] = 0.10   # true positive, below 0.3
    W[3, 0] = 0.80   # false positive (true_adj is False at (3,0))
    result = select_true_positive_target(true_adj, W, thresholds=(0.3, 0.2, 0.1))
    assert result == (1, 2, 0.3)


def test_target_selection_does_not_select_false_positive_edges():
    """A learned edge that is not in the true adjacency must be ignored."""
    d = 3
    true_adj = np.zeros((d, d), dtype=bool)
    true_adj[0, 1] = True  # only true edge
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 0.15    # true positive, below 0.3 and below 0.2, above 0.1
    W[1, 2] = 0.95    # false positive (would dominate any selection)
    W[2, 0] = 0.85    # false positive
    result = select_true_positive_target(
        true_adj, W, thresholds=(0.3, 0.2, 0.1)
    )
    # Only the (0, 1) edge is a true positive and it qualifies at 0.1 only.
    assert result == (0, 1, 0.1)


def test_target_selection_falls_back_through_thresholds():
    d = 3
    true_adj = np.zeros((d, d), dtype=bool)
    true_adj[0, 2] = True
    W = np.zeros((d, d), dtype=float)
    W[0, 2] = 0.22  # true positive, qualifies at 0.2 not 0.3
    result = select_true_positive_target(true_adj, W, thresholds=(0.3, 0.2, 0.1))
    assert result == (0, 2, 0.2)


def test_target_selection_returns_none_when_no_qualifying_true_positive():
    d = 3
    true_adj = np.zeros((d, d), dtype=bool)
    true_adj[0, 1] = True
    W = np.zeros((d, d), dtype=float)
    W[0, 1] = 0.05  # below smallest threshold (0.1)
    W[1, 0] = 0.90  # false positive but high magnitude
    result = select_true_positive_target(true_adj, W, thresholds=(0.3, 0.2, 0.1))
    assert result is None


def test_target_selection_skips_diagonal_entries():
    d = 3
    true_adj = np.eye(d, dtype=bool)  # only diagonal is "true"
    W = np.eye(d, dtype=float) * 0.95  # large diagonal magnitudes
    result = select_true_positive_target(true_adj, W, thresholds=(0.3, 0.2, 0.1))
    assert result is None


# ---------------------------------------------------------------------------
# T-3: joint selection rule
# ---------------------------------------------------------------------------


def _row(seed: int, lam: float, status: str, passes: bool) -> dict:
    return {
        "seed": int(seed),
        "lambda_prior": float(lam),
        "passes": bool(passes),
        "candidate_status": str(status),
    }


def test_joint_rule_recommends_larger_per_seed_minimum():
    """Seed 401 min passes at 0.05; seed 402 min at 0.1; recommend 0.1."""
    rows = [
        _row(401, 0.01, "too_weak", False),
        _row(401, 0.05, "passed", True),
        _row(401, 0.10, "passed", True),
        _row(401, 0.50, "too_strong", False),
        _row(402, 0.01, "too_weak", False),
        _row(402, 0.05, "too_weak", False),
        _row(402, 0.10, "passed", True),
        _row(402, 0.50, "too_strong", False),
    ]
    rec, needs_review, reason = recommend_lambda(rows, (401, 402))
    assert rec == pytest.approx(0.1)
    assert needs_review is False
    assert reason is None


def test_joint_rule_handles_all_pass_uses_larger_minimum():
    """If both seeds pass at every candidate, recommend the larger of two
    per-seed minima (here both are the smallest candidate)."""
    rows = [
        _row(401, 0.01, "passed", True),
        _row(401, 0.05, "passed", True),
        _row(402, 0.01, "passed", True),
        _row(402, 0.05, "passed", True),
    ]
    rec, needs_review, reason = recommend_lambda(rows, (401, 402))
    assert rec == pytest.approx(0.01)
    assert needs_review is False
    assert reason is None


# ---------------------------------------------------------------------------
# T-4: null recommendation when one seed has no passing candidate
# ---------------------------------------------------------------------------


def test_joint_rule_returns_null_when_one_seed_has_no_passing_candidate():
    rows = [
        _row(401, 0.01, "too_weak", False),
        _row(401, 0.05, "passed", True),
        _row(401, 0.10, "passed", True),
        _row(401, 0.50, "too_strong", False),
        _row(402, 0.01, "too_strong", False),
        _row(402, 0.05, "too_strong", False),
        _row(402, 0.10, "too_strong", False),
        _row(402, 0.50, "too_strong", False),
    ]
    rec, needs_review, reason = recommend_lambda(rows, (401, 402))
    assert rec is None
    assert needs_review is True
    assert reason in (
        "all_too_strong",
        "mixed_or_seed_inconsistent",
    )


def test_joint_rule_marks_infrastructure_failure_label():
    rec, needs_review, reason = recommend_lambda(
        candidate_rows=[],
        calibration_seeds=(401, 402),
        per_seed_error={401: "infrastructure_failure"},
    )
    assert rec is None
    assert needs_review is True
    assert reason == "infrastructure_failure"


def test_joint_rule_marks_target_selection_failed_label():
    rec, needs_review, reason = recommend_lambda(
        candidate_rows=[],
        calibration_seeds=(401, 402),
        per_seed_error={402: "target_selection_failed"},
    )
    assert rec is None
    assert needs_review is True
    assert reason == "target_selection_failed"


# ---------------------------------------------------------------------------
# T-5: seed guard
# ---------------------------------------------------------------------------


def test_validate_calibration_seeds_rejects_any_evaluation_seed():
    """Each evaluation seed individually must be rejected if present."""
    for evil in EVALUATION_SEEDS:
        with pytest.raises(ValueError, match="overlap"):
            validate_calibration_seeds((401, evil), EVALUATION_SEEDS)


def test_validate_calibration_seeds_rejects_set_overlap():
    """Any non-empty intersection with the evaluation pool is rejected."""
    with pytest.raises(ValueError, match="overlap"):
        validate_calibration_seeds((501, 502), EVALUATION_SEEDS)


def test_validate_calibration_seeds_passes_when_disjoint():
    # 401, 402 are disjoint from 501..507.
    validate_calibration_seeds((401, 402), EVALUATION_SEEDS)


# ---------------------------------------------------------------------------
# T-6: output schema
# ---------------------------------------------------------------------------


def _mock_summary_and_rows() -> tuple[dict, list[dict]]:
    rows = [
        {
            "seed": 401,
            "train_data_seed": 401,
            "target_i": 1,
            "target_j": 2,
            "target_threshold_used": 0.3,
            "lambda_prior": 0.05,
            "base_abs": 0.42,
            "soft_abs": 0.15,
            "ratio": 0.15 / 0.42,
            "reduction_percent": (1.0 - 0.15 / 0.42) * 100.0,
            "candidate_status": "passed",
            "passes": True,
            "step_a_zero_mask_delta": 0.0,
            "step_a_zero_lambda_delta": 0.0,
        },
        {
            "seed": 402,
            "train_data_seed": 402,
            "target_i": 0,
            "target_j": 3,
            "target_threshold_used": 0.2,
            "lambda_prior": 0.1,
            "base_abs": 0.31,
            "soft_abs": 0.08,
            "ratio": 0.08 / 0.31,
            "reduction_percent": (1.0 - 0.08 / 0.31) * 100.0,
            "candidate_status": "passed",
            "passes": True,
            "step_a_zero_mask_delta": 0.0,
            "step_a_zero_lambda_delta": 0.0,
        },
    ]
    summary = {
        "calibration_seeds": [401, 402],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "lambda_prior_candidates": list(LAMBDA_PRIOR_CANDIDATES),
        "constants": {"n_train": 1000},
        "data_seed_derivation": "graph_seed = train_data_seed = calibration_seed",
        "selection_rule": "per-seed minimum; recommend max",
        "recommended_lambda_prior": 0.1,
        "grid_needs_review": False,
        "grid_review_reason": None,
        "per_seed_summary": [
            {"seed": 401, "target_i": 1, "target_j": 2, "target_threshold_used": 0.3},
            {"seed": 402, "target_i": 0, "target_j": 3, "target_threshold_used": 0.2},
        ],
        "candidate_rows": rows,
        "no_evaluation_seeds_used_confirmation": (
            "no overlap between {401, 402} and the evaluation pool"
        ),
    }
    return summary, rows


def test_outputs_are_written(tmp_path: Path):
    summary, rows = _mock_summary_and_rows()
    json_path, csv_path = write_calibration_outputs(summary, rows, tmp_path)
    assert json_path.exists()
    assert csv_path.exists()
    assert json_path.name == JSON_OUTPUT_NAME
    assert csv_path.name == CSV_OUTPUT_NAME


def test_outputs_create_missing_directory(tmp_path: Path):
    summary, rows = _mock_summary_and_rows()
    nested = tmp_path / "nested" / "subdir"
    json_path, csv_path = write_calibration_outputs(summary, rows, nested)
    assert json_path.exists()
    assert csv_path.exists()


def test_outputs_contain_required_json_fields(tmp_path: Path):
    summary, rows = _mock_summary_and_rows()
    json_path, _ = write_calibration_outputs(summary, rows, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    required = {
        "calibration_seeds",
        "evaluation_seeds",
        "lambda_prior_candidates",
        "constants",
        "data_seed_derivation",
        "selection_rule",
        "recommended_lambda_prior",
        "grid_needs_review",
        "grid_review_reason",
        "per_seed_summary",
        "candidate_rows",
        "no_evaluation_seeds_used_confirmation",
    }
    missing = required - set(payload.keys())
    assert not missing, f"missing JSON fields: {sorted(missing)}"


def test_outputs_contain_required_csv_columns(tmp_path: Path):
    summary, rows = _mock_summary_and_rows()
    _, csv_path = write_calibration_outputs(summary, rows, tmp_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == CSV_COLUMNS
        records = list(reader)
    assert len(records) == len(rows)


def test_outputs_contain_no_evaluation_seed(tmp_path: Path):
    """Neither JSON nor CSV must reference any seed in 501..507 in a
    non-disclaimer position. The evaluation_seeds list and the
    no-eval-confirmation message are allowed listings; the seed and
    train_data_seed columns of the CSV must never contain them."""
    summary, rows = _mock_summary_and_rows()
    json_path, csv_path = write_calibration_outputs(summary, rows, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # JSON: candidate_rows and per_seed_summary entries must use only
    # calibration seeds.
    for entry in payload.get("candidate_rows", []):
        assert int(entry["seed"]) not in EVALUATION_SEEDS, (
            f"candidate row uses evaluation seed: {entry['seed']}"
        )
        assert int(entry["train_data_seed"]) not in EVALUATION_SEEDS, (
            "candidate row uses evaluation train_data_seed: "
            f"{entry['train_data_seed']}"
        )
    for entry in payload.get("per_seed_summary", []):
        if "seed" in entry:
            assert int(entry["seed"]) not in EVALUATION_SEEDS

    # CSV: seed and train_data_seed columns must not contain any
    # evaluation seed.
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            assert int(r["seed"]) not in EVALUATION_SEEDS
            assert int(r["train_data_seed"]) not in EVALUATION_SEEDS
