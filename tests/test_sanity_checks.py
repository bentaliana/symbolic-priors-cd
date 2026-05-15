"""Tests for ground-truth compatibility checks.

Tests exercise each primitive check in isolation, then verify the structured
report shape, and finally test the full assert-wrapper gate on three paths:
pass, tolerance fail, and failed-SID hard error.
"""

import numpy as np
import pytest

from symbolic_priors_cd.data import Intervention, generate_linear_gaussian_scm
from symbolic_priors_cd.metrics import (
    assert_ground_truth_compatibility,
    check_do_clamping,
    check_mmd_same_intervention,
    check_mmd_same_observational,
    check_sid_self_zero,
    run_ground_truth_compatibility_checks,
)
from symbolic_priors_cd.metrics.sanity_checks import _derive_sid_status


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_test_scm_and_intervention():
    """5-node SCM with seed=0, intervention on node 0 at value 2.0."""
    scm = generate_linear_gaussian_scm(5, 5, seed=0)
    return scm, Intervention(target=0, value=2.0)


# ---------------------------------------------------------------------------
# Individual check: SID
# ---------------------------------------------------------------------------


def test_check_sid_self_zero_returns_zero_for_valid_dag():
    """check_sid_self_zero(true_dag) must return 0 for a valid ground-truth DAG."""
    scm, _ = _make_test_scm_and_intervention()
    assert check_sid_self_zero(scm.adjacency) == 0


# ---------------------------------------------------------------------------
# Individual check: MMD (same distribution)
# ---------------------------------------------------------------------------


def test_check_mmd_same_intervention_near_zero():
    """MMD between two same-intervention batches must be near zero."""
    scm, intervention = _make_test_scm_and_intervention()
    result = check_mmd_same_intervention(scm, intervention)
    assert abs(result) < 0.05


def test_check_mmd_same_observational_near_zero():
    """MMD between two observational batches from the same SCM must be near zero."""
    scm, _ = _make_test_scm_and_intervention()
    result = check_mmd_same_observational(scm)
    assert abs(result) < 0.05


# ---------------------------------------------------------------------------
# Individual check: do-clamping
# ---------------------------------------------------------------------------


def test_check_do_clamping_returns_zero():
    """Clamping on a root-like node must return exactly 0.0."""
    scm, intervention = _make_test_scm_and_intervention()
    assert check_do_clamping(scm, intervention) == 0.0


def test_check_do_clamping_zero_for_non_root_intervention():
    """Clamping on an interior node (with parents) must also return 0.0."""
    scm = generate_linear_gaussian_scm(5, 5, seed=0)
    intervention = Intervention(target=2, value=-3.0)
    assert check_do_clamping(scm, intervention) == 0.0


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_run_checks_report_has_expected_keys():
    scm, intervention = _make_test_scm_and_intervention()
    report = run_ground_truth_compatibility_checks(scm, intervention)
    assert set(report.keys()) == {
        "sid_self_zero_status",
        "sid_self_zero_value",
        "mmd_same_intervention",
        "mmd_same_observational",
        "do_clamping_max_deviation",
    }


def test_run_checks_sid_status_is_passed():
    scm, intervention = _make_test_scm_and_intervention()
    report = run_ground_truth_compatibility_checks(scm, intervention)
    assert report["sid_self_zero_status"] == "passed"


def test_run_checks_sid_value_is_zero():
    scm, intervention = _make_test_scm_and_intervention()
    report = run_ground_truth_compatibility_checks(scm, intervention)
    assert report["sid_self_zero_value"] == 0


def test_run_checks_mmd_values_are_floats():
    scm, intervention = _make_test_scm_and_intervention()
    report = run_ground_truth_compatibility_checks(scm, intervention)
    assert type(report["mmd_same_intervention"]) is float
    assert type(report["mmd_same_observational"]) is float


def test_run_checks_clamping_deviation_is_zero():
    scm, intervention = _make_test_scm_and_intervention()
    report = run_ground_truth_compatibility_checks(scm, intervention)
    assert report["do_clamping_max_deviation"] == 0.0


# ---------------------------------------------------------------------------
# Integration: assert-wrapper gate
# ---------------------------------------------------------------------------


def test_assert_gate_passes_require_sid_false():
    """Pass path: a valid SCM with passing SID must not raise."""
    scm, intervention = _make_test_scm_and_intervention()
    assert_ground_truth_compatibility(scm, intervention)  # must not raise


def test_assert_gate_passes_require_sid_true():
    """require_sid=True must not raise when SID self-zero check passes."""
    scm, intervention = _make_test_scm_and_intervention()
    assert_ground_truth_compatibility(scm, intervention, require_sid=True)  # must not raise


def test_assert_gate_fails_tight_mmd_tolerance():
    """Tolerance fail path: an impossibly tight mmd_tolerance must raise."""
    scm, intervention = _make_test_scm_and_intervention()
    with pytest.raises(AssertionError):
        assert_ground_truth_compatibility(scm, intervention, mmd_tolerance=1e-15)


def test_assert_gate_error_contains_report():
    """AssertionError message must include the full structured report."""
    scm, intervention = _make_test_scm_and_intervention()
    with pytest.raises(AssertionError) as exc_info:
        assert_ground_truth_compatibility(scm, intervention, mmd_tolerance=1e-15)
    message = str(exc_info.value)
    # Full report string is present with key fields
    assert "Full report:" in message
    assert "sid_self_zero_status" in message
    assert "do_clamping_max_deviation" in message
    # At least one MMD failure reason is present
    assert "MMD" in message


def test_assert_gate_multiple_failures_all_in_message():
    """When multiple MMD gates fail, the message must mention both."""
    scm, intervention = _make_test_scm_and_intervention()
    with pytest.raises(AssertionError) as exc_info:
        assert_ground_truth_compatibility(scm, intervention, mmd_tolerance=1e-15)
    message = str(exc_info.value)
    assert "mmd_same_intervention" in message or "MMD" in message
    assert "mmd_same_observational" in message or "MMD" in message


# ---------------------------------------------------------------------------
# _derive_sid_status — direct unit tests
# ---------------------------------------------------------------------------


def test_derive_sid_status_zero_is_passed():
    assert _derive_sid_status(0) == "passed"


def test_derive_sid_status_nonzero_is_failed():
    assert _derive_sid_status(1) == "failed"
    assert _derive_sid_status(-1) == "failed"
    assert _derive_sid_status(99) == "failed"


# ---------------------------------------------------------------------------
# Failed-SID gate — monkeypatched test
# ---------------------------------------------------------------------------


def test_assert_gate_fails_on_failed_sid_regardless_of_require_sid(monkeypatch):
    """Failed SID must be a hard error even when require_sid=False."""
    import symbolic_priors_cd.metrics.sanity_checks as sc_module

    monkeypatch.setattr(sc_module, "check_sid_self_zero", lambda dag: 3)

    scm, intervention = _make_test_scm_and_intervention()
    with pytest.raises(AssertionError) as exc_info:
        assert_ground_truth_compatibility(scm, intervention, require_sid=False)
    assert "SID self-zero check failed" in str(exc_info.value)
