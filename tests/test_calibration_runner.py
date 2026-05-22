"""Tests for the calibration workload enumeration.

These tests exercise the calibration runner's enumeration and
executable-candidate expansion layer. They verify the frozen
workload arithmetic (20 executable candidates, 40 fit jobs), the
per-candidate configuration_hash distinctness, the calibration
seed contract, the DCDI fit-RNG convention, and the absence of any
model-fit invocation during enumeration. The tests are designed to
pin the workload contract independently of which calibration grid
point a candidate carries.

The selected-configurations artefact writer, the within-model
ranking logic, and any held-out execution live in separate
components and are not exercised here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from experiments.selection_study.calibration import (
    CalibrationCandidate,
    CalibrationFitJob,
    CalibrationWorkload,
    enumerate_calibration_workload,
    expand_calibration_candidates,
)
from experiments.selection_study.config import (
    Configuration,
    configuration_hash,
    load_config,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CALIBRATION_CONFIG_DIR = (
    _PROJECT_ROOT
    / "experiments"
    / "selection_study"
    / "configs"
    / "calibration"
)
_DAGMA_CENTRED_PATH = (
    _CALIBRATION_CONFIG_DIR / "dagma_calibration_centred_only.json"
)
_DAGMA_STANDARDISED_PATH = (
    _CALIBRATION_CONFIG_DIR / "dagma_calibration_standardised.json"
)
_DCDI_CENTRED_PATH = (
    _CALIBRATION_CONFIG_DIR / "dcdi_calibration_centred_only.json"
)
_DCDI_STANDARDISED_PATH = (
    _CALIBRATION_CONFIG_DIR / "dcdi_calibration_standardised.json"
)


_EXPECTED_CALIBRATION_SEEDS: tuple[int, ...] = (201, 202)
_FORBIDDEN_HELD_OUT_SEEDS: frozenset[int] = frozenset(
    {301, 302, 303, 304, 305}
)
_EXPECTED_CANDIDATE_COUNT = 20
_EXPECTED_FIT_JOB_COUNT = 40
_EXPECTED_GROUPS: tuple[tuple[str, str], ...] = (
    ("dagma", "centred_only"),
    ("dagma", "standardised"),
    ("dcdi", "centred_only"),
    ("dcdi", "standardised"),
)
_EXPECTED_DAGMA_GRID: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.1, 0.25,
)
_EXPECTED_DCDI_GRID: tuple[float, ...] = (
    0.01, 0.03, 0.1, 0.3, 1.0,
)
_DCDI_FIT_RNG_VALUE: int = 42


def _load_all_calibration_parents() -> tuple[Configuration, ...]:
    """Load the four on-disk calibration parent configurations."""
    return (
        load_config(_DAGMA_CENTRED_PATH),
        load_config(_DAGMA_STANDARDISED_PATH),
        load_config(_DCDI_CENTRED_PATH),
        load_config(_DCDI_STANDARDISED_PATH),
    )


# ---------------------------------------------------------------------------
# Per-parent expansion tests
# ---------------------------------------------------------------------------


def test_expand_calibration_candidates_dagma_centred_yields_five() -> None:
    """A DAGMA centred-only parent expands to five executable candidates."""
    parent = load_config(_DAGMA_CENTRED_PATH)
    candidates = expand_calibration_candidates(parent)
    assert len(candidates) == 5
    for candidate in candidates:
        assert candidate.model == "dagma"
        assert candidate.condition == "centred_only"


def test_expand_calibration_candidates_carry_single_grid_point() -> None:
    """Each executable candidate carries exactly one calibration grid point.

    Bundling all five grid points into a single Configuration would
    collapse five distinct candidates into one configuration_hash. The
    expansion is responsible for reducing the parent's five-entry
    calibration_configurations to a single-entry tuple per candidate.
    """
    parent = load_config(_DAGMA_CENTRED_PATH)
    candidates = expand_calibration_candidates(parent)
    for candidate in candidates:
        executable = candidate.configuration
        assert len(executable.calibration_configurations) == 1


def test_expand_preserves_grid_point_hyperparameter_dagma() -> None:
    """DAGMA candidates carry one lambda1 value from the frozen grid."""
    parent = load_config(_DAGMA_CENTRED_PATH)
    candidates = expand_calibration_candidates(parent)
    observed = tuple(
        sorted(
            float(dict(c.grid_point_hyperparameter)["lambda1"])
            for c in candidates
        )
    )
    assert observed == _EXPECTED_DAGMA_GRID


def test_expand_preserves_grid_point_hyperparameter_dcdi() -> None:
    """DCDI candidates carry one reg_coeff value from the frozen grid."""
    parent = load_config(_DCDI_CENTRED_PATH)
    candidates = expand_calibration_candidates(parent)
    observed = tuple(
        sorted(
            float(dict(c.grid_point_hyperparameter)["reg_coeff"])
            for c in candidates
        )
    )
    assert observed == _EXPECTED_DCDI_GRID


def test_expand_yields_distinct_configuration_hashes_within_parent() -> None:
    """Five executable candidates from one parent have distinct hashes.

    Two grid points that differ only in their hyperparameter value
    must produce distinct executable configuration_hash values because
    the calibration_configurations field participates in the hash.
    """
    parent = load_config(_DAGMA_CENTRED_PATH)
    candidates = expand_calibration_candidates(parent)
    hashes = {c.configuration_hash_full for c in candidates}
    assert len(hashes) == 5


# ---------------------------------------------------------------------------
# Whole-workload enumeration tests
# ---------------------------------------------------------------------------


def test_workload_yields_twenty_executable_candidates() -> None:
    """Enumeration over the four parents yields 20 executable candidates."""
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    assert isinstance(workload, CalibrationWorkload)
    assert len(workload.candidates) == _EXPECTED_CANDIDATE_COUNT


def test_workload_yields_forty_fit_jobs_after_seed_expansion() -> None:
    """Combining 20 candidates with two calibration seeds yields 40 jobs."""
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    assert len(workload.fit_jobs) == _EXPECTED_FIT_JOB_COUNT


def test_workload_has_four_groups_of_five() -> None:
    """The 20 executable candidates split into four (model, condition) groups.

    Each group corresponds to one parent calibration file and contains
    exactly five candidates (one per grid point).
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    group_counts: dict[tuple[str, str], int] = {}
    for candidate in workload.candidates:
        key = (candidate.model, candidate.condition)
        group_counts[key] = group_counts.get(key, 0) + 1
    assert set(group_counts.keys()) == set(_EXPECTED_GROUPS)
    for group_key, count in group_counts.items():
        assert count == 5, (
            f"group {group_key!r} has {count} candidates; expected 5"
        )


def test_workload_candidate_hashes_are_globally_distinct() -> None:
    """All 20 executable candidates have distinct full configuration_hashes.

    A genuine SHA-256 collision is treated as an error; the
    enumeration is required to surface it explicitly rather than
    silently merge rows.
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    hashes = [c.configuration_hash_full for c in workload.candidates]
    assert len(set(hashes)) == _EXPECTED_CANDIDATE_COUNT


def test_workload_calibration_seeds_match_frozen_pair() -> None:
    """The workload pins calibration seeds to the frozen (201, 202) pair."""
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    assert tuple(workload.calibration_seeds) == _EXPECTED_CALIBRATION_SEEDS


def test_workload_does_not_reference_held_out_seeds() -> None:
    """No held-out seed appears anywhere in the workload's fit jobs.

    Held-out seeds [301, 302, 303, 304, 305] are reserved for the
    held-out evaluation runner and must not enter calibration.
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    seen_seed_values = {job.seed_value for job in workload.fit_jobs}
    forbidden = seen_seed_values & _FORBIDDEN_HELD_OUT_SEEDS
    assert not forbidden, (
        "calibration workload referenced held-out seed(s): "
        f"{sorted(forbidden)!r}"
    )


def test_workload_fit_job_seeds_are_exactly_calibration_pool() -> None:
    """Every fit-job seed value is drawn from the calibration pool."""
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    expected_seed_set = set(_EXPECTED_CALIBRATION_SEEDS)
    for job in workload.fit_jobs:
        assert job.seed_value in expected_seed_set, (
            f"fit job seed_value={job.seed_value!r} is not in the "
            f"calibration pool {expected_seed_set!r}"
        )


def test_workload_fit_jobs_cover_each_candidate_with_both_seeds() -> None:
    """For each candidate, both calibration seed indices appear once.

    The 40 fit jobs decompose into 20 candidates times 2 seed
    replicate indices, with each (candidate, seed_replicate_index)
    pair appearing exactly once.
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    coverage: dict[tuple[str, int], int] = {}
    for job in workload.fit_jobs:
        key = (job.candidate.configuration_hash_full, job.seed_replicate_index)
        coverage[key] = coverage.get(key, 0) + 1
    assert len(coverage) == _EXPECTED_FIT_JOB_COUNT
    for key, count in coverage.items():
        assert count == 1, (
            f"(candidate_hash, seed_replicate_index)={key!r} appears "
            f"{count} times; expected exactly 1"
        )


# ---------------------------------------------------------------------------
# Seed convention per model
# ---------------------------------------------------------------------------


def test_dcdi_candidates_use_fixed_fit_rng_42() -> None:
    """DCDI calibration candidates pin seed_torch and seed_numpy to 42.

    The fit-RNG convention for every DCDI fit at calibration is the
    single integer 42 for both torch and numpy global RNG setters.
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    dcdi_candidates = [
        c for c in workload.candidates if c.model == "dcdi"
    ]
    assert len(dcdi_candidates) == 10
    for candidate in dcdi_candidates:
        executable = candidate.configuration
        assert executable.seed_torch == _DCDI_FIT_RNG_VALUE
        assert executable.seed_numpy == _DCDI_FIT_RNG_VALUE
        assert executable.seed_dagma is None


def test_dagma_candidates_keep_all_seed_fields_none() -> None:
    """DAGMA calibration candidates carry no global RNG setter values.

    The DAGMA fit path is deterministic by construction and does not
    call torch.manual_seed, np.random.seed, or
    dagma.utils.set_random_seed; the corresponding seed fields remain
    None in every DAGMA executable Configuration.
    """
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    dagma_candidates = [
        c for c in workload.candidates if c.model == "dagma"
    ]
    assert len(dagma_candidates) == 10
    for candidate in dagma_candidates:
        executable = candidate.configuration
        assert executable.seed_torch is None
        assert executable.seed_numpy is None
        assert executable.seed_dagma is None


# ---------------------------------------------------------------------------
# Intervention policy coverage
# ---------------------------------------------------------------------------


def test_calibration_intervention_set_covers_all_nodes_both_signs() -> None:
    """Each calibration parent encodes 20 interventions over 10 nodes.

    The eligible-nodes policy for calibration intervenes on every
    node of the 10-node selection cell with both signs of magnitude
    2, yielding 20 intervention conditions per seed.
    """
    parents = _load_all_calibration_parents()
    for parent in parents:
        intervention_set = parent.intervention_set
        assert len(intervention_set) == 20
        observed = {
            (intervention.target_node, intervention.value_raw)
            for intervention in intervention_set
        }
        expected = {
            (node, sign)
            for node in range(10)
            for sign in (-2.0, 2.0)
        }
        assert observed == expected


# ---------------------------------------------------------------------------
# No fit invocation during enumeration
# ---------------------------------------------------------------------------


def test_enumeration_does_not_invoke_run_single_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enumerating the calibration workload does not call run_single_fit.

    The enumeration step is pure: it loads configs, validates them,
    expands per-grid-point Configurations, and reports the workload
    arithmetic without invoking any model fit.
    """
    from experiments.selection_study import pipeline

    fit_invocations = {"count": 0}

    def fake_run_single_fit(*args: Any, **kwargs: Any) -> Path:
        fit_invocations["count"] += 1
        return Path("/tmp/never-reached")

    monkeypatch.setattr(
        pipeline, "run_single_fit", fake_run_single_fit
    )
    parents = _load_all_calibration_parents()
    workload = enumerate_calibration_workload(parents)
    assert fit_invocations["count"] == 0
    assert len(workload.fit_jobs) == _EXPECTED_FIT_JOB_COUNT


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_accepts_phase_calibration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runner CLI accepts --phase calibration and runs enumeration.

    The implementation of --phase calibration loads the
    parent configs from the supplied directory, validates them under
    the calibration-stage real-study guard, and prints the workload
    summary. It does not invoke any model fit.
    """
    from experiments.selection_study import pipeline, run as run_module

    fit_invocations = {"count": 0}

    def fake_run_single_fit(*args: Any, **kwargs: Any) -> Path:
        fit_invocations["count"] += 1
        return Path("/tmp/never-reached")

    monkeypatch.setattr(
        pipeline, "run_single_fit", fake_run_single_fit
    )

    config_dir_argument = str(_CALIBRATION_CONFIG_DIR)
    run_module.main(
        [
            "--phase",
            "calibration",
            "--config",
            config_dir_argument,
        ]
    )
    assert fit_invocations["count"] == 0


def test_cli_phase_calibration_requires_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--phase calibration without --config is rejected with a clear error."""
    from experiments.selection_study import run as run_module

    with pytest.raises(ValueError) as excinfo:
        run_module.main(["--phase", "calibration"])
    assert "--phase calibration" in str(excinfo.value)
    assert "--config" in str(excinfo.value)
