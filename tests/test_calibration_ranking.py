"""Tests for the within-model calibration ranking module.

These tests exercise the deterministic lexicographic ranker that
turns calibration records into the candidate_ranking and selections
structure consumed by the selected-configurations artefact writer.
The tests use synthetic records only and do not invoke any model fit
or any orchestration code. The ranker is a pure data transformation
and must produce the same output for the same input regardless of
input order.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any

import pytest

from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
    rank_condition_model_cell,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CANDIDATES_PER_CONDITION_PER_MODEL,
    CONDITIONS,
    FULL_HASH_LENGTH,
    HASH_PREFIX_LENGTH,
    MODELS,
    SCHEMA_VERSION,
    SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
    validate_selected_configurations_artefact,
)


# ---------------------------------------------------------------------------
# Synthetic record factories
# ---------------------------------------------------------------------------


def _synthetic_hash(seed: str) -> str:
    """Return a deterministic 64-character lowercase hex string."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _per_seed_extras() -> dict[str, Any]:
    """Return the auxiliary per-seed fields the schema preserves."""
    threshold_metrics = [
        {
            "threshold": float(threshold),
            "shd": 5,
            "sid": 10,
            "mmd_primary": 0.05,
        }
        for threshold in (0.2, 0.3, 0.4)
    ]
    mmd_by_intervention = [
        {
            "intervention_target": node,
            "intervention_value": float(sign),
            "mmd_primary": 0.05,
        }
        for node in range(10)
        for sign in (-2, 2)
    ]
    return {
        "threshold_metrics": threshold_metrics,
        "mmd_by_intervention": mmd_by_intervention,
        "bandwidth_summaries": {"median_heuristic": 1.0},
    }


def _make_record(
    *,
    model: str,
    condition: str,
    seed_value: int,
    hyperparameter_name: str,
    hyperparameter_value: float,
    sid: Any,
    mmd_primary: Any,
    shd: Any,
    config_hash_full: str,
    run_id_suffix: str = "",
) -> dict[str, Any]:
    """Build one synthetic calibration record with all required fields."""
    if hyperparameter_name == "lambda1":
        hyperparameter_label = f"lambda1_{hyperparameter_value}"
    else:
        hyperparameter_label = f"reg_coeff_{hyperparameter_value}"
    extras = _per_seed_extras()
    return {
        "model": model,
        "condition": condition,
        "configuration_hash_full": config_hash_full,
        "configuration_hash_prefix": config_hash_full[:HASH_PREFIX_LENGTH],
        "hyperparameters": {hyperparameter_name: hyperparameter_value},
        "seed_value": seed_value,
        "shd": shd,
        "sid": sid,
        "mmd_primary": mmd_primary,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "runtime_seconds": 12.5,
        "n_iterations": None,
        "threshold_metrics": extras["threshold_metrics"],
        "mmd_by_intervention": extras["mmd_by_intervention"],
        "bandwidth_summaries": extras["bandwidth_summaries"],
        "run_id": (
            f"{model}__{condition}__calibration__seed"
            f"{seed_value}__cfg{config_hash_full}{run_id_suffix}"
        ),
    }


def _make_two_seed_records(
    *,
    model: str,
    condition: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    sid_pair: tuple[Any, Any],
    mmd_pair: tuple[Any, Any],
    shd_pair: tuple[Any, Any],
    config_hash_full: str,
) -> list[dict[str, Any]]:
    """Build two records (seed 201 and seed 202) for one candidate hash."""
    records = []
    for seed_value, sid_value, mmd_value, shd_value in zip(
        CALIBRATION_SEEDS, sid_pair, mmd_pair, shd_pair
    ):
        records.append(
            _make_record(
                model=model,
                condition=condition,
                seed_value=seed_value,
                hyperparameter_name=hyperparameter_name,
                hyperparameter_value=hyperparameter_value,
                sid=sid_value,
                mmd_primary=mmd_value,
                shd=shd_value,
                config_hash_full=config_hash_full,
            )
        )
    return records


def _make_cell_records(
    *,
    model: str,
    condition: str,
    hyperparameter_name: str,
    candidates: list[tuple[float, tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]],
) -> list[dict[str, Any]]:
    """Build the 10 records of one (model, condition) cell.

    ``candidates`` is a list of 5 tuples
    ``(hyperparameter_value, sid_pair, mmd_pair, shd_pair)``.
    """
    if len(candidates) != CANDIDATES_PER_CONDITION_PER_MODEL:
        raise AssertionError(
            "test helper requires exactly "
            f"{CANDIDATES_PER_CONDITION_PER_MODEL} candidate entries"
        )
    records: list[dict[str, Any]] = []
    for hyperparameter_value, sid_pair, mmd_pair, shd_pair in candidates:
        seed = (
            f"{model}|{condition}|{hyperparameter_name}|{hyperparameter_value}"
        )
        config_hash_full = _synthetic_hash(seed)
        records.extend(
            _make_two_seed_records(
                model=model,
                condition=condition,
                hyperparameter_name=hyperparameter_name,
                hyperparameter_value=hyperparameter_value,
                sid_pair=sid_pair,
                mmd_pair=mmd_pair,
                shd_pair=shd_pair,
                config_hash_full=config_hash_full,
            )
        )
    return records


def _identity_cell_records(
    *,
    model: str,
    condition: str,
    grid_values: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Build a cell with distinct SID values so the ranking is unambiguous."""
    hyperparameter_name = "lambda1" if model == "dagma" else "reg_coeff"
    candidates = []
    for index, value in enumerate(grid_values):
        sid_value = 10.0 + 5.0 * index
        mmd_value = 0.10 - 0.01 * index
        shd_value = 5
        candidates.append(
            (
                value,
                (sid_value, sid_value),
                (mmd_value, mmd_value),
                (shd_value, shd_value),
            )
        )
    return _make_cell_records(
        model=model,
        condition=condition,
        hyperparameter_name=hyperparameter_name,
        candidates=candidates,
    )


def _full_set_records() -> list[dict[str, Any]]:
    """Build the 40-record full calibration set with distinct SIDs."""
    records: list[dict[str, Any]] = []
    for model in MODELS:
        if model == "dagma":
            grid = (0.01, 0.025, 0.05, 0.1, 0.25)
        else:
            grid = (0.01, 0.03, 0.1, 0.3, 1.0)
        for condition in CONDITIONS:
            records.extend(
                _identity_cell_records(
                    model=model,
                    condition=condition,
                    grid_values=grid,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Cell-level: SID ordering
# ---------------------------------------------------------------------------


def test_distinct_sid_values_rank_ascending() -> None:
    """With distinct SID values and no band overlaps, ranking is by SID."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    candidate_ranking, selection = rank_condition_model_cell(cell)
    sids = [c["aggregate_metrics"]["mean_sid"] for c in candidate_ranking]
    assert sids == sorted(sids)
    assert candidate_ranking[0]["rank"] == 1
    assert (
        selection["selected_configuration_hash_full"]
        == candidate_ranking[0]["configuration_hash_full"]
    )


def test_sid_value_exactly_at_band_threshold_is_inside() -> None:
    """A candidate whose SID equals best * 1.10 is inside the band (inclusive).

    The test pins band-membership directly rather than rank order:
    with equal MMD on both candidates, the in-band MMD tiebreaker is
    a tie and SHD/hash decide; the load-bearing assertion is that
    both candidates carry ``sid_band_eligible == True``.
    """
    best_sid = 100.0
    boundary_sid = best_sid * 1.10
    candidates = [
        (0.01, (best_sid, best_sid), (0.1, 0.1), (5, 5)),
        (0.025, (boundary_sid, boundary_sid), (0.1, 0.1), (5, 5)),
        (0.05, (best_sid + 50.0, best_sid + 50.0), (0.1, 0.1), (5, 5)),
        (0.1, (best_sid + 60.0, best_sid + 60.0), (0.1, 0.1), (5, 5)),
        (0.25, (best_sid + 70.0, best_sid + 70.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    in_band_sids = sorted(
        c["aggregate_metrics"]["mean_sid"]
        for c in candidate_ranking
        if c["aggregate_metrics"]["sid_band_eligible"]
    )
    assert in_band_sids == [best_sid, boundary_sid]


def test_sid_value_just_inside_band_is_inside() -> None:
    """A candidate whose SID is slightly below best * 1.10 is inside."""
    best_sid = 100.0
    inside_sid = best_sid * 1.10 - 1e-9
    candidates = [
        (0.01, (best_sid, best_sid), (0.5, 0.5), (5, 5)),
        (0.025, (inside_sid, inside_sid), (0.1, 0.1), (5, 5)),
        (0.05, (best_sid + 50.0, best_sid + 50.0), (0.1, 0.1), (5, 5)),
        (0.1, (best_sid + 60.0, best_sid + 60.0), (0.1, 0.1), (5, 5)),
        (0.25, (best_sid + 70.0, best_sid + 70.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    in_band = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["sid_band_eligible"]
    ]
    assert len(in_band) == 2


def test_sid_value_just_outside_band_is_outside() -> None:
    """A candidate whose SID is slightly above best * 1.10 is outside."""
    best_sid = 100.0
    outside_sid = best_sid * 1.10 + 1e-6
    candidates = [
        (0.01, (best_sid, best_sid), (0.5, 0.5), (5, 5)),
        (0.025, (outside_sid, outside_sid), (0.01, 0.01), (5, 5)),
        (0.05, (best_sid + 50.0, best_sid + 50.0), (0.1, 0.1), (5, 5)),
        (0.1, (best_sid + 60.0, best_sid + 60.0), (0.1, 0.1), (5, 5)),
        (0.25, (best_sid + 70.0, best_sid + 70.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    in_band = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["sid_band_eligible"]
    ]
    assert len(in_band) == 1
    assert in_band[0]["aggregate_metrics"]["mean_sid"] == best_sid


def test_best_sid_zero_admits_only_zero_into_band() -> None:
    """When best_finite_mean_sid == 0 only zero-SID candidates are inside."""
    candidates = [
        (0.01, (0.0, 0.0), (0.05, 0.05), (5, 5)),
        (0.025, (1.0, 1.0), (0.01, 0.01), (5, 5)),
        (0.05, (10.0, 10.0), (0.1, 0.1), (5, 5)),
        (0.1, (20.0, 20.0), (0.1, 0.1), (5, 5)),
        (0.25, (30.0, 30.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    in_band = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["sid_band_eligible"]
    ]
    assert len(in_band) == 1
    assert in_band[0]["aggregate_metrics"]["mean_sid"] == 0.0


# ---------------------------------------------------------------------------
# Cell-level: MMD inside the band
# ---------------------------------------------------------------------------


def test_mmd_reorders_candidates_inside_sid_band() -> None:
    """MMD breaks the SID tie inside the band, ranking lower MMD first."""
    inside_sid = 100.0
    # All five candidates share the same SID (inside the band) but
    # differ on MMD.
    candidates = [
        (0.01, (inside_sid, inside_sid), (0.50, 0.50), (5, 5)),
        (0.025, (inside_sid, inside_sid), (0.10, 0.10), (5, 5)),
        (0.05, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.1, (inside_sid, inside_sid), (0.20, 0.20), (5, 5)),
        (0.25, (inside_sid, inside_sid), (0.30, 0.30), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    mmds = [
        c["aggregate_metrics"]["mean_mmd_primary"] for c in candidate_ranking
    ]
    assert mmds == sorted(mmds)
    assert candidate_ranking[0]["aggregate_metrics"]["mean_mmd_primary"] == 0.05


def test_mmd_cannot_promote_candidate_outside_band_above_in_band() -> None:
    """An out-of-band candidate cannot beat an in-band one on MMD alone.

    The in-band candidate has the worst MMD in the cell; the
    out-of-band candidate has the best MMD in the cell; yet the
    in-band candidate ranks above the out-of-band one because MMD
    cannot promote across the band boundary.
    """
    best_sid = 100.0
    in_band_sid = best_sid * 1.05
    out_of_band_sid = best_sid * 1.5
    candidates = [
        (0.01, (best_sid, best_sid), (0.5, 0.5), (5, 5)),
        (0.025, (in_band_sid, in_band_sid), (0.99, 0.99), (5, 5)),
        (0.05, (out_of_band_sid, out_of_band_sid), (0.01, 0.01), (5, 5)),
        (0.1, (best_sid + 100.0, best_sid + 100.0), (0.1, 0.1), (5, 5)),
        (0.25, (best_sid + 200.0, best_sid + 200.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    in_band_at_99 = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["mean_mmd_primary"] == 0.99
    ]
    out_of_band_at_01 = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["mean_mmd_primary"] == 0.01
    ]
    assert len(in_band_at_99) == 1
    assert len(out_of_band_at_01) == 1
    assert in_band_at_99[0]["rank"] < out_of_band_at_01[0]["rank"]


# ---------------------------------------------------------------------------
# Cell-level: SHD and configuration_hash tiebreakers
# ---------------------------------------------------------------------------


def test_shd_breaks_tie_after_sid_and_mmd() -> None:
    """When SID and MMD are equal inside the band, SHD breaks the tie."""
    inside_sid = 100.0
    candidates = [
        (0.01, (inside_sid, inside_sid), (0.05, 0.05), (10, 10)),
        (0.025, (inside_sid, inside_sid), (0.05, 0.05), (3, 3)),
        (0.05, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.1, (inside_sid, inside_sid), (0.05, 0.05), (7, 7)),
        (0.25, (inside_sid, inside_sid), (0.05, 0.05), (12, 12)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    shds = [c["aggregate_metrics"]["mean_shd"] for c in candidate_ranking]
    assert shds == sorted(shds)


def test_configuration_hash_breaks_final_ties() -> None:
    """When SID, MMD, and SHD are all equal, hash decides ordering."""
    inside_sid = 100.0
    candidates = [
        (0.01, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.025, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.05, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.1, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
        (0.25, (inside_sid, inside_sid), (0.05, 0.05), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    hashes = [c["configuration_hash_full"] for c in candidate_ranking]
    assert hashes == sorted(hashes)


def test_shuffled_input_gives_identical_output() -> None:
    """Input order does not affect the final ranking."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    candidate_ranking_a, selection_a = rank_condition_model_cell(cell)
    shuffled = list(cell)
    random.Random(0xC0FFEE).shuffle(shuffled)
    assert shuffled != cell
    candidate_ranking_b, selection_b = rank_condition_model_cell(shuffled)
    assert candidate_ranking_b == candidate_ranking_a
    assert selection_b == selection_a


# ---------------------------------------------------------------------------
# Non-finite handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "non_finite_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(None, id="none"),
        pytest.param(float("inf"), id="positive_inf"),
        pytest.param(float("-inf"), id="negative_inf"),
    ],
)
def test_non_finite_sid_ranks_below_finite_sid(
    non_finite_value: Any,
) -> None:
    """A candidate with non-finite mean_sid ranks below any finite-SID candidate.

    Negative infinity must not be treated as a winning low SID; the
    non-finite layer is strictly below all finite-SID layers.
    """
    candidates = [
        (0.01, (10.0, 10.0), (0.5, 0.5), (5, 5)),
        (0.025, (20.0, 20.0), (0.1, 0.1), (5, 5)),
        (0.05, (30.0, 30.0), (0.1, 0.1), (5, 5)),
        (0.1, (40.0, 40.0), (0.1, 0.1), (5, 5)),
        (
            0.25,
            (non_finite_value, non_finite_value),
            (0.001, 0.001),
            (1, 1),
        ),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    last = candidate_ranking[-1]
    assert last["aggregate_metrics"]["mean_sid"] is None
    assert last["aggregate_metrics"]["has_non_finite_seed_metric"] is True


def test_all_non_finite_sid_falls_through_to_mmd() -> None:
    """If every candidate has non-finite SID, MMD orders the cell."""
    candidates = [
        (0.01, (float("nan"), float("nan")), (0.5, 0.5), (5, 5)),
        (0.025, (float("nan"), float("nan")), (0.1, 0.1), (5, 5)),
        (0.05, (None, None), (0.05, 0.05), (5, 5)),
        (0.1, (float("inf"), float("inf")), (0.2, 0.2), (5, 5)),
        (0.25, (float("-inf"), float("-inf")), (0.3, 0.3), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    for c in candidate_ranking:
        assert c["aggregate_metrics"]["mean_sid"] is None
    mmds = [c["aggregate_metrics"]["mean_mmd_primary"] for c in candidate_ranking]
    assert mmds == sorted(mmds)


def test_all_non_finite_sid_and_mmd_falls_through_to_shd() -> None:
    """If SID and MMD are non-finite for all candidates, SHD orders the cell."""
    candidates = [
        (0.01, (float("nan"), float("nan")), (float("nan"), float("nan")), (10, 10)),
        (0.025, (None, None), (None, None), (3, 3)),
        (0.05, (float("inf"), float("inf")), (float("inf"), float("inf")), (5, 5)),
        (0.1, (float("-inf"), float("-inf")), (float("-inf"), float("-inf")), (7, 7)),
        (0.25, (None, None), (None, None), (12, 12)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    shds = [c["aggregate_metrics"]["mean_shd"] for c in candidate_ranking]
    assert shds == sorted(shds)


def test_all_non_finite_sid_mmd_shd_falls_through_to_configuration_hash() -> None:
    """When every ranking metric is non-finite, hash decides the order."""
    candidates = [
        (0.01, (float("nan"), float("nan")), (float("nan"), float("nan")), (float("nan"), float("nan"))),
        (0.025, (None, None), (None, None), (None, None)),
        (0.05, (float("inf"), float("inf")), (float("inf"), float("inf")), (float("inf"), float("inf"))),
        (0.1, (float("-inf"), float("-inf")), (float("-inf"), float("-inf")), (float("-inf"), float("-inf"))),
        (0.25, (None, None), (None, None), (None, None)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    hashes = [c["configuration_hash_full"] for c in candidate_ranking]
    assert hashes == sorted(hashes)


def test_non_finite_sets_has_non_finite_seed_metric_flag() -> None:
    """Any non-finite per-seed metric sets has_non_finite_seed_metric."""
    candidates = [
        (0.01, (10.0, 10.0), (0.5, 0.5), (5, 5)),
        (0.025, (20.0, float("nan")), (0.1, 0.1), (5, 5)),
        (0.05, (30.0, 30.0), (0.1, 0.1), (5, 5)),
        (0.1, (40.0, 40.0), (0.1, 0.1), (5, 5)),
        (0.25, (50.0, 50.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    flags = {
        c["configuration_hash_full"]: c["aggregate_metrics"][
            "has_non_finite_seed_metric"
        ]
        for c in candidate_ranking
    }
    truthy = [k for k, v in flags.items() if v]
    assert len(truthy) == 1


def test_degenerate_metric_names_identify_affected_aggregates() -> None:
    """degenerate_metric_names lists the affected aggregate field names."""
    candidates = [
        (0.01, (10.0, 10.0), (0.5, 0.5), (5, 5)),
        (
            0.025,
            (float("nan"), 20.0),
            (None, 0.1),
            (5, 5),
        ),
        (0.05, (30.0, 30.0), (0.1, 0.1), (5, 5)),
        (0.1, (40.0, 40.0), (0.1, 0.1), (5, 5)),
        (0.25, (50.0, 50.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    affected = [
        c
        for c in candidate_ranking
        if c["aggregate_metrics"]["has_non_finite_seed_metric"]
    ]
    assert len(affected) == 1
    names = set(affected[0]["aggregate_metrics"]["degenerate_metric_names"])
    assert names == {"mean_sid", "mean_mmd_primary"}


def test_rank_one_degeneracy_propagates_to_selection_flag() -> None:
    """If the rank-1 candidate is degenerate the selection records it."""
    # All five candidates have non-finite SID; the rank-1 candidate
    # therefore has has_non_finite_seed_metric=True.
    candidates = [
        (0.01, (None, None), (0.5, 0.5), (5, 5)),
        (0.025, (None, None), (0.1, 0.1), (5, 5)),
        (0.05, (None, None), (0.05, 0.05), (5, 5)),
        (0.1, (None, None), (0.2, 0.2), (5, 5)),
        (0.25, (None, None), (0.3, 0.3), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, selection = rank_condition_model_cell(records)
    assert candidate_ranking[0]["aggregate_metrics"][
        "has_non_finite_seed_metric"
    ] is True
    assert selection["degeneracy_flag"] is True


def test_no_raw_non_finite_values_in_aggregate_metrics() -> None:
    """Aggregate metrics must report None instead of NaN, inf, or -inf."""
    candidates = [
        (0.01, (float("nan"), float("nan")), (float("inf"), float("inf")), (float("-inf"), float("-inf"))),
        (0.025, (10.0, 10.0), (0.1, 0.1), (5, 5)),
        (0.05, (20.0, 20.0), (0.1, 0.1), (5, 5)),
        (0.1, (30.0, 30.0), (0.1, 0.1), (5, 5)),
        (0.25, (40.0, 40.0), (0.1, 0.1), (5, 5)),
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    for candidate in candidate_ranking:
        for value in candidate["aggregate_metrics"].values():
            if isinstance(value, float):
                assert math.isfinite(value), (
                    "aggregate_metrics must not carry non-finite floats; "
                    f"got {value!r} in candidate "
                    f"{candidate['configuration_hash_prefix']!r}"
                )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_input_is_rejected() -> None:
    """An empty record sequence is rejected by both rankers."""
    with pytest.raises(ValueError):
        rank_condition_model_cell([])
    with pytest.raises(ValueError):
        rank_calibration_records([])


def test_full_set_rejects_short_input() -> None:
    """Fewer than 40 records are rejected by the full-set ranker."""
    records = _full_set_records()[:-1]
    with pytest.raises(ValueError) as excinfo:
        rank_calibration_records(records)
    assert "40" in str(excinfo.value)


def test_full_set_rejects_extra_records() -> None:
    """More than 40 records are rejected by the full-set ranker."""
    records = _full_set_records()
    records.append(records[0])
    with pytest.raises(ValueError) as excinfo:
        rank_calibration_records(records)
    assert "40" in str(excinfo.value)


def test_missing_required_field_is_rejected() -> None:
    """A record missing one required field raises a precise ValueError."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    del cell[0]["mmd_primary"]
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell)
    assert "mmd_primary" in str(excinfo.value)


def test_unexpected_model_is_rejected() -> None:
    """A record carrying an unknown model is rejected."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    cell[0]["model"] = "notch"
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell)
    assert "model" in str(excinfo.value).lower()


def test_unexpected_condition_is_rejected() -> None:
    """A record carrying an unknown condition is rejected."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    cell[0]["condition"] = "raw"
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell)
    assert "condition" in str(excinfo.value).lower()


def test_hash_prefix_full_mismatch_is_rejected() -> None:
    """A configuration_hash_prefix that does not match the full hash fails."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    cell[0]["configuration_hash_prefix"] = "deadbeefdead"
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell)
    assert "configuration_hash_prefix" in str(excinfo.value)


def test_wrong_seed_value_is_rejected() -> None:
    """A held-out seed value in calibration input is rejected."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    cell[0]["seed_value"] = 301
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell)
    assert "301" in str(excinfo.value) or "seed_value" in str(excinfo.value)


def test_fewer_than_five_candidate_hashes_in_cell_is_rejected() -> None:
    """A cell with fewer than 5 distinct candidate hashes is rejected."""
    cell = _identity_cell_records(
        model="dagma",
        condition="centred_only",
        grid_values=(0.01, 0.025, 0.05, 0.1, 0.25),
    )
    # Drop one candidate's two records and duplicate another candidate's
    # records to keep the total at 10.
    target_hash = cell[0]["configuration_hash_full"]
    cell_without_one = [r for r in cell if r["configuration_hash_full"] != target_hash]
    duplicated = cell[2:4]  # next candidate's two records
    cell_modified = cell_without_one + duplicated
    assert len(cell_modified) == 10
    with pytest.raises(ValueError) as excinfo:
        rank_condition_model_cell(cell_modified)
    assert (
        "candidate hashes" in str(excinfo.value)
        or "seed_value" in str(excinfo.value)
        or "records" in str(excinfo.value)
    )


def test_three_seeds_per_candidate_is_rejected() -> None:
    """A candidate with three seed records is rejected by cell-shape check.

    The full-set ranker requires exactly 40 records, so adding a
    third seed for one candidate alongside removing one record from
    another candidate keeps the total at 40 but produces a cell-
    shape mismatch.
    """
    records = _full_set_records()
    # Remove one record from the first candidate of cell
    # (centred_only, dagma) and duplicate another record so the
    # total record count remains 40 but one candidate now has three
    # records on its hash.
    first_centred_dagma = next(
        r for r in records
        if r["condition"] == "centred_only" and r["model"] == "dagma"
    )
    target_hash = first_centred_dagma["configuration_hash_full"]
    target_records = [
        r for r in records
        if r["configuration_hash_full"] == target_hash
    ]
    # Find a second candidate in the same cell to duplicate from.
    other_candidate = next(
        r for r in records
        if (
            r["condition"] == "centred_only"
            and r["model"] == "dagma"
            and r["configuration_hash_full"] != target_hash
        )
    )
    duplicate = copy.deepcopy(other_candidate)
    # Drop one record from the first candidate, push two records
    # from another candidate onto it so the first candidate ends up
    # with three records of the same hash (and the cell shape now
    # has only four candidate hashes).
    records_modified = [
        r for r in records if r is not target_records[0]
    ]
    duplicate2 = copy.deepcopy(other_candidate)
    duplicate2["configuration_hash_full"] = target_hash
    duplicate2["configuration_hash_prefix"] = target_hash[:HASH_PREFIX_LENGTH]
    duplicate2["seed_value"] = (
        201 if other_candidate["seed_value"] == 202 else 202
    )
    records_modified.append(duplicate2)
    assert len(records_modified) == 40
    with pytest.raises(ValueError) as excinfo:
        rank_calibration_records(records_modified)
    assert "records" in str(excinfo.value) or "candidate" in str(
        excinfo.value
    )


# ---------------------------------------------------------------------------
# Full-set output structure
# ---------------------------------------------------------------------------


def test_full_set_output_has_candidate_ranking_and_selections() -> None:
    """The full-set output dict carries candidate_ranking and selections keys."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    assert set(output.keys()) == {"candidate_ranking", "selections"}


def test_full_set_output_covers_four_condition_model_groups() -> None:
    """Every (condition, model) pair has 5 ranked candidates and a selection."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    for condition in CONDITIONS:
        assert condition in output["candidate_ranking"]
        assert condition in output["selections"]
        for model in MODELS:
            ranking = output["candidate_ranking"][condition][model]
            assert len(ranking) == CANDIDATES_PER_CONDITION_PER_MODEL
            assert [c["rank"] for c in ranking] == [1, 2, 3, 4, 5]
            selection = output["selections"][condition][model]
            assert selection["selected_rank"] == 1
            assert (
                selection["selected_configuration_hash_full"]
                == ranking[0]["configuration_hash_full"]
            )


def test_full_set_output_per_seed_metrics_is_list_shaped() -> None:
    """Every candidate's per_seed_metrics is a list, never a dict."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    for condition in CONDITIONS:
        for model in MODELS:
            for candidate in output["candidate_ranking"][condition][model]:
                assert isinstance(candidate["per_seed_metrics"], list)
                assert len(candidate["per_seed_metrics"]) == len(
                    CALIBRATION_SEEDS
                )


def test_full_set_output_source_run_ids_populated() -> None:
    """Every candidate carries a non-empty source_run_ids list."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    for condition in CONDITIONS:
        for model in MODELS:
            for candidate in output["candidate_ranking"][condition][model]:
                assert candidate["source_run_ids"]
                assert all(
                    isinstance(rid, str)
                    for rid in candidate["source_run_ids"]
                )


def test_full_set_output_aggregate_metrics_has_required_fields() -> None:
    """Every candidate's aggregate_metrics contains the six required fields."""
    required = {
        "mean_sid",
        "mean_mmd_primary",
        "mean_shd",
        "std_sid",
        "std_mmd_primary",
        "std_shd",
    }
    records = _full_set_records()
    output = rank_calibration_records(records)
    for condition in CONDITIONS:
        for model in MODELS:
            for candidate in output["candidate_ranking"][condition][model]:
                assert required.issubset(
                    candidate["aggregate_metrics"].keys()
                )


# ---------------------------------------------------------------------------
# Selection-artefact compatibility
# ---------------------------------------------------------------------------


def _envelope_around(output: dict[str, Any]) -> dict[str, Any]:
    """Wrap a ranker output in a minimal valid artefact envelope."""
    full_hash = _synthetic_hash("calibration_run_envelope_seed")
    return {
        "schema_version": SCHEMA_VERSION,
        "artefact_type": SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
        "decision_scope": "within_model_configuration_selection",
        "base_model_decision_made": False,
        "selected_configuration_semantics": (
            "rank_1_within_model_and_condition"
        ),
        "calibration_run_hash_prefix": full_hash[:HASH_PREFIX_LENGTH],
        "calibration_run_hash_full": full_hash,
        "selection_rule_id": "within_model_sid_first_lex",
        "selection_rule_ref": (
            "Lexicographic rank: mean SID, then mean MMD inside the "
            "SID tie margin, then mean SHD, then configuration_hash."
        ),
        "seed_population": "calibration",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "intervention_policy_ref": "all_nodes_both_signs_v1",
        "fit_rng_policy_ref": "dcdi_torch_numpy_42_v1",
        "selections": output["selections"],
        "candidate_ranking": output["candidate_ranking"],
        "generated_at_utc": "2026-05-22T12:00:00Z",
    }


def test_full_set_output_passes_artefact_validator() -> None:
    """A ranker-built envelope passes the selected-configurations validator."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    envelope = _envelope_around(output)
    validate_selected_configurations_artefact(envelope)


# ---------------------------------------------------------------------------
# Scope confirmations
# ---------------------------------------------------------------------------


def test_module_does_not_invoke_run_single_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking does not invoke pipeline.run_single_fit under any input."""
    from experiments.selection_study import pipeline

    fit_invocations = {"count": 0}

    def fake_run_single_fit(*args: Any, **kwargs: Any):
        fit_invocations["count"] += 1
        return None

    monkeypatch.setattr(pipeline, "run_single_fit", fake_run_single_fit)
    records = _full_set_records()
    rank_calibration_records(records)
    assert fit_invocations["count"] == 0


def test_module_exposes_no_final_winner_public_name() -> None:
    """The module exposes no public name resembling a final-winner field."""
    from experiments.selection_study import calibration_ranking

    public_names = [
        name for name in dir(calibration_ranking) if not name.startswith("_")
    ]
    forbidden_substrings = (
        "winner",
        "WINNER",
        "model_winner",
        "MODEL_WINNER",
        "base_model_winner",
        "BASE_MODEL_WINNER",
        "recommended_model",
        "RECOMMENDED_MODEL",
        "final_decision",
        "FINAL_DECISION",
    )
    for name in public_names:
        for substring in forbidden_substrings:
            assert substring not in name, (
                f"public name {name!r} contains forbidden substring "
                f"{substring!r}"
            )


def test_adversarial_layer_mix_orders_correctly() -> None:
    """A cell with two in-band, one out-of-band, and two non-finite-SID
    candidates ranks in the expected B, A, C, E, D order.

    The cell tests every layer-boundary interaction in one fixture:

    - A and B are both inside the 10 percent SID band (best=10,
      candidate B = 11 = 10 * 1.10). Inside the band, MMD orders
      ascending: B (0.50) below A (0.99). B therefore ranks above A.
    - C has finite SID = 12, which is outside the 0.10-band (12 >
      11.0). C cannot be promoted by its low MMD = 0.01 because MMD
      does not cross the band boundary.
    - D has SID = NaN and E has SID = -inf. Both are non-finite SID
      and rank below all finite-SID candidates. -inf must not be
      treated as a winning low value.
    - Inside the non-finite-SID layer, MMD orders ascending: E
      (0.0001) below D (0.001). E therefore ranks above D.

    The expected final rank order is B, A, C, E, D.
    """
    candidates = [
        # (hyperparameter_value, sid_pair, mmd_pair, shd_pair)
        (0.01, (10.0, 10.0), (0.99, 0.99), (5, 5)),  # A
        (0.025, (11.0, 11.0), (0.50, 0.50), (5, 5)),  # B
        (0.05, (12.0, 12.0), (0.01, 0.01), (5, 5)),  # C
        (
            0.1,
            (float("nan"), float("nan")),
            (0.001, 0.001),
            (5, 5),
        ),  # D
        (
            0.25,
            (float("-inf"), float("-inf")),
            (0.0001, 0.0001),
            (5, 5),
        ),  # E
    ]
    records = _make_cell_records(
        model="dagma",
        condition="centred_only",
        hyperparameter_name="lambda1",
        candidates=candidates,
    )
    candidate_ranking, _ = rank_condition_model_cell(records)
    mean_sids = [
        c["aggregate_metrics"]["mean_sid"] for c in candidate_ranking
    ]
    mean_mmds = [
        c["aggregate_metrics"]["mean_mmd_primary"]
        for c in candidate_ranking
    ]
    # Rank 1 is B (in-band, lower MMD between A and B).
    assert mean_sids[0] == 11.0
    assert mean_mmds[0] == 0.50
    # Rank 2 is A (in-band, higher MMD).
    assert mean_sids[1] == 10.0
    assert mean_mmds[1] == 0.99
    # Rank 3 is C (out-of-band, finite SID; cannot be promoted by MMD).
    assert mean_sids[2] == 12.0
    assert mean_mmds[2] == 0.01
    # Rank 4 is E (-inf SID is non-finite; MMD 0.0001 is lower than D).
    assert mean_sids[3] is None
    assert mean_mmds[3] == 0.0001
    # Rank 5 is D (non-finite SID; MMD 0.001 is higher than E).
    assert mean_sids[4] is None
    assert mean_mmds[4] == 0.001
    # All non-finite SID candidates carry the degeneracy flag.
    for candidate in candidate_ranking[3:]:
        assert candidate["aggregate_metrics"][
            "has_non_finite_seed_metric"
        ] is True


def test_no_raw_non_finite_anywhere_in_full_ranker_output() -> None:
    """The full ranker output carries no raw NaN, +inf, or -inf at any depth.

    Builds a 40-record full-set input where every (model, condition)
    cell contains at least one candidate whose per-seed metrics are
    non-finite. Walks the entire rank_calibration_records output
    (candidate_ranking, selections, per_seed_metrics, threshold_metrics,
    mmd_by_intervention, bandwidth_summaries) and asserts that every
    float encountered is finite. Non-finite per-seed inputs must be
    converted to ``None`` in the output, never propagated as raw
    NaN/inf floats.
    """
    records: list[dict[str, Any]] = []
    for model in MODELS:
        if model == "dagma":
            grid = (0.01, 0.025, 0.05, 0.1, 0.25)
            hyperparameter_name = "lambda1"
        else:
            grid = (0.01, 0.03, 0.1, 0.3, 1.0)
            hyperparameter_name = "reg_coeff"
        for condition in CONDITIONS:
            candidates_for_cell = []
            for index, value in enumerate(grid):
                if index == 0:
                    # First candidate has every per-seed metric
                    # set to a non-finite value, plus injected
                    # non-finite values inside threshold_metrics,
                    # mmd_by_intervention, and bandwidth_summaries.
                    sid_pair: tuple[Any, Any] = (
                        float("nan"),
                        float("-inf"),
                    )
                    mmd_pair: tuple[Any, Any] = (
                        float("inf"),
                        float("nan"),
                    )
                    shd_pair: tuple[Any, Any] = (5, 5)
                else:
                    sid_pair = (10.0 + 5.0 * index, 10.0 + 5.0 * index)
                    mmd_pair = (0.10, 0.10)
                    shd_pair = (5, 5)
                candidates_for_cell.append(
                    (value, sid_pair, mmd_pair, shd_pair)
                )
            cell_records = _make_cell_records(
                model=model,
                condition=condition,
                hyperparameter_name=hyperparameter_name,
                candidates=candidates_for_cell,
            )
            # Inject non-finite values inside threshold_metrics,
            # mmd_by_intervention, and bandwidth_summaries of the
            # first candidate's records.
            for record in cell_records[:2]:
                record["threshold_metrics"][0]["mmd_primary"] = (
                    float("nan")
                )
                record["mmd_by_intervention"][0]["mmd_primary"] = (
                    float("-inf")
                )
                record["bandwidth_summaries"]["median_heuristic"] = (
                    float("inf")
                )
            records.extend(cell_records)

    output = rank_calibration_records(records)

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                _walk(item, f"{path}[{index}]")
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, float):
            assert math.isfinite(obj), (
                f"non-finite float {obj!r} at {path}; the ranker must "
                "convert non-finite values to None before they reach "
                "artefact-facing output"
            )

    _walk(output, "$")


def test_ranker_output_does_not_contain_forbidden_field_names() -> None:
    """No forbidden winner field name appears at any depth of the output."""
    records = _full_set_records()
    output = rank_calibration_records(records)
    forbidden = {
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    }

    def walk(obj: Any, path: str = "$") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert key not in forbidden, (
                    f"forbidden field name {key!r} found at {path}"
                )
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")

    walk(output)
