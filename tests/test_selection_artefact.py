"""Tests for the selected-configurations artefact module.

These tests exercise the calibration-run identity hash, the on-disk
path layout, the artefact schema validator, and the atomic writer.
They use synthetic data only and do not invoke any model fit or any
calibration runner. Forbidden-field-name rejection at every nesting
depth is exercised explicitly.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CANDIDATES_PER_CONDITION_PER_MODEL,
    CONDITIONS,
    FULL_HASH_LENGTH,
    HASH_PREFIX_LENGTH,
    MODELS,
    SCHEMA_VERSION,
    SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
    SELECTED_CONFIGURATIONS_FILENAME,
    build_calibration_run_identity_payload,
    compute_calibration_run_hash12,
    compute_calibration_run_hash_full,
    selected_configurations_path,
    validate_selected_configurations_artefact,
    write_selected_configurations,
)


# ---------------------------------------------------------------------------
# Synthetic input factories
# ---------------------------------------------------------------------------


def _synthetic_hash(seed: str) -> str:
    """Return a deterministic 64-character lowercase hex string."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _synthetic_executable_candidate_identities() -> list[dict[str, Any]]:
    """Return 20 synthetic candidate identity records.

    The records cover both models, both conditions, and five grid
    points each. Hashes are computed deterministically from a seed
    string so the same factory call yields the same records across
    test runs.
    """
    records: list[dict[str, Any]] = []
    for model in MODELS:
        for condition in CONDITIONS:
            for grid_point_order in range(
                CANDIDATES_PER_CONDITION_PER_MODEL
            ):
                seed = (
                    f"{model}|{condition}|grid{grid_point_order}"
                )
                records.append(
                    {
                        "model": model,
                        "condition": condition,
                        "grid_point_order": grid_point_order,
                        "configuration_hash_full": _synthetic_hash(seed),
                    }
                )
    return records


def _identity_kwargs(
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the common kwargs accepted by the identity helpers."""
    return {
        "executable_candidate_identities": (
            candidates
            if candidates is not None
            else _synthetic_executable_candidate_identities()
        ),
        "selection_rule_id": "within_model_sid_first_lex",
        "selection_rule_ref": (
            "Lexicographic rank: mean SID, then mean MMD inside the "
            "SID tie margin, then mean SHD, then configuration_hash."
        ),
        "intervention_policy_ref": "all_nodes_both_signs_v1",
        "fit_rng_policy_ref": "dcdi_torch_numpy_42_v1",
    }


def _make_per_seed_record(
    seed_value: int,
    *,
    candidate_seed: str,
) -> dict[str, Any]:
    """Build one per-seed metrics record with the realistic shape.

    Includes 3 threshold_metrics rows and 20 mmd_by_intervention
    rows (10 nodes x 2 intervention signs). Bandwidth summaries are
    preserved as a small mapping carrying the heuristic value.
    """
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
            "mmd_primary": 0.05 + 0.001 * node,
        }
        for node in range(10)
        for sign in (-2, 2)
    ]
    return {
        "seed_value": seed_value,
        "shd": 5,
        "sid": 10,
        "mmd_primary": 0.05,
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "training_status": "converged",
        "runtime_seconds": 12.5,
        "n_iterations": None,
        "threshold_metrics": threshold_metrics,
        "mmd_by_intervention": mmd_by_intervention,
        "bandwidth_summaries": {
            "median_heuristic": 1.0,
            "scaled_0p5x": 0.5,
            "scaled_1p0x": 1.0,
            "scaled_2p0x": 2.0,
        },
    }


def _make_candidate(
    *,
    rank: int,
    model: str,
    condition: str,
    grid_point_value: float,
    candidate_seed: str,
) -> dict[str, Any]:
    """Build one candidate_ranking entry."""
    full_hash = _synthetic_hash(candidate_seed)
    hyperparameter_name = (
        "lambda1" if model == "dagma" else "reg_coeff"
    )
    return {
        "rank": rank,
        "configuration_hash_prefix": full_hash[:HASH_PREFIX_LENGTH],
        "configuration_hash_full": full_hash,
        "hyperparameters": {hyperparameter_name: grid_point_value},
        "aggregate_metrics": {
            "mean_sid": 10.0,
            "mean_mmd_primary": 0.05,
            "mean_shd": 5.0,
            "std_sid": 0.5,
            "std_mmd_primary": 0.005,
            "std_shd": 0.2,
        },
        "per_seed_metrics": [
            _make_per_seed_record(
                seed_value=seed_value,
                candidate_seed=f"{candidate_seed}_seed{seed_value}",
            )
            for seed_value in CALIBRATION_SEEDS
        ],
        "source_run_ids": [
            f"{model}__{condition}__calibration__seed{idx}__"
            f"cfg{full_hash}"
            for idx in range(len(CALIBRATION_SEEDS))
        ],
        "n_calibration_records": len(CALIBRATION_SEEDS),
    }


def _make_candidates_for_group(
    *, model: str, condition: str
) -> list[dict[str, Any]]:
    """Build five candidate_ranking entries for one (model, condition) pair."""
    if model == "dagma":
        grid_values = (0.01, 0.025, 0.05, 0.1, 0.25)
    else:
        grid_values = (0.01, 0.03, 0.1, 0.3, 1.0)
    return [
        _make_candidate(
            rank=rank,
            model=model,
            condition=condition,
            grid_point_value=value,
            candidate_seed=f"{model}|{condition}|rank{rank}|value{value}",
        )
        for rank, value in enumerate(grid_values, start=1)
    ]


def _make_artefact(
    *,
    calibration_run_hash_full: str | None = None,
    extra_top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid synthetic selected-configurations artefact.

    The candidate_ranking lists are constructed first; the rank-1
    candidate in each group is reused as the selections entry, so
    the rank-1-equals-selected invariant is satisfied by
    construction.
    """
    if calibration_run_hash_full is None:
        calibration_run_hash_full = _synthetic_hash(
            "synthetic_calibration_run"
        )
    candidate_ranking: dict[str, dict[str, list[dict[str, Any]]]] = {}
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        candidate_ranking[condition] = {}
        selections[condition] = {}
        for model in MODELS:
            candidates = _make_candidates_for_group(
                model=model, condition=condition
            )
            candidate_ranking[condition][model] = candidates
            rank_one = candidates[0]
            selections[condition][model] = {
                "selected_configuration_hash_prefix": (
                    rank_one["configuration_hash_prefix"]
                ),
                "selected_configuration_hash_full": (
                    rank_one["configuration_hash_full"]
                ),
                "selected_hyperparameters": dict(
                    rank_one["hyperparameters"]
                ),
                "selected_rank": 1,
                "selection_metrics": dict(rank_one["aggregate_metrics"]),
                "source_run_ids": list(rank_one["source_run_ids"]),
            }
    artefact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artefact_type": SELECTED_CONFIGURATIONS_ARTEFACT_TYPE,
        "decision_scope": "within_model_configuration_selection",
        "base_model_decision_made": False,
        "selected_configuration_semantics": (
            "rank_1_within_model_and_condition"
        ),
        "calibration_run_hash_prefix": (
            calibration_run_hash_full[:HASH_PREFIX_LENGTH]
        ),
        "calibration_run_hash_full": calibration_run_hash_full,
        "selection_rule_id": "within_model_sid_first_lex",
        "selection_rule_ref": (
            "Lexicographic rank: mean SID, then mean MMD inside the "
            "SID tie margin, then mean SHD, then configuration_hash."
        ),
        "seed_population": "calibration",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "intervention_policy_ref": "all_nodes_both_signs_v1",
        "fit_rng_policy_ref": "dcdi_torch_numpy_42_v1",
        "selections": selections,
        "candidate_ranking": candidate_ranking,
        "generated_at_utc": "2026-05-22T12:00:00Z",
    }
    if extra_top_level is not None:
        artefact.update(extra_top_level)
    return artefact


# ---------------------------------------------------------------------------
# Hash tests
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_across_repeated_calls() -> None:
    """Two identical inputs produce the same full hash."""
    kwargs = _identity_kwargs()
    first = compute_calibration_run_hash_full(**kwargs)
    second = compute_calibration_run_hash_full(**kwargs)
    assert first == second


def test_hash_is_invariant_under_input_shuffling() -> None:
    """A shuffled candidate list produces the same hash."""
    candidates = _synthetic_executable_candidate_identities()
    shuffled = list(candidates)
    rng = random.Random(0xC0FFEE)
    rng.shuffle(shuffled)
    assert shuffled != candidates
    original_hash = compute_calibration_run_hash_full(
        **_identity_kwargs(candidates=candidates)
    )
    shuffled_hash = compute_calibration_run_hash_full(
        **_identity_kwargs(candidates=shuffled)
    )
    assert original_hash == shuffled_hash


def test_hash_changes_when_one_candidate_hash_changes() -> None:
    """A single changed candidate hash changes the calibration_run_hash."""
    candidates = _synthetic_executable_candidate_identities()
    original_hash = compute_calibration_run_hash_full(
        **_identity_kwargs(candidates=candidates)
    )
    modified = deepcopy(candidates)
    modified[0]["configuration_hash_full"] = _synthetic_hash(
        "perturbed_candidate"
    )
    perturbed_hash = compute_calibration_run_hash_full(
        **_identity_kwargs(candidates=modified)
    )
    assert original_hash != perturbed_hash


def test_full_hash_is_64_lowercase_hex_characters() -> None:
    """The full hash is a 64-character lowercase hex string."""
    digest = compute_calibration_run_hash_full(**_identity_kwargs())
    assert len(digest) == FULL_HASH_LENGTH
    assert digest == digest.lower()
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_hash_prefix_is_first_12_characters() -> None:
    """The 12-character prefix equals the first 12 chars of the full hash."""
    digest = compute_calibration_run_hash_full(**_identity_kwargs())
    prefix = compute_calibration_run_hash12(**_identity_kwargs())
    assert prefix == digest[:HASH_PREFIX_LENGTH]
    assert len(prefix) == HASH_PREFIX_LENGTH


def test_identity_payload_stores_full_hashes_only() -> None:
    """The identity payload's executable hash list is full hashes only."""
    payload = build_calibration_run_identity_payload(**_identity_kwargs())
    hashes = payload["executable_configuration_hashes_full"]
    assert isinstance(hashes, list)
    assert len(hashes) == 20
    for entry in hashes:
        assert isinstance(entry, str)
        assert len(entry) == FULL_HASH_LENGTH
    # No key anywhere in the payload contains "prefix": prefixes are
    # not stored in the identity payload by design.
    serialised = json.dumps(payload)
    assert "prefix" not in serialised


def test_identity_payload_does_not_record_metadata_per_candidate() -> None:
    """The payload's hash list contains plain strings, not objects.

    Two distinct callers that happen to produce the same set of full
    hashes (with the same sort order) yield identical payloads even
    if their candidate metadata differs.
    """
    payload = build_calibration_run_identity_payload(**_identity_kwargs())
    hashes = payload["executable_configuration_hashes_full"]
    for entry in hashes:
        assert isinstance(entry, str)


def test_hash_payload_includes_required_identity_fields() -> None:
    """The identity payload carries the fields the hash is defined over."""
    payload = build_calibration_run_identity_payload(**_identity_kwargs())
    required_keys = {
        "schema_version",
        "artefact_type",
        "stage",
        "models",
        "conditions",
        "calibration_seeds",
        "executable_configuration_hashes_full",
        "selection_rule_id",
        "selection_rule_ref",
        "intervention_policy_ref",
        "fit_rng_policy_ref",
    }
    assert set(payload.keys()) == required_keys
    assert payload["schema_version"] == 1
    assert payload["artefact_type"] == "calibration_run_identity"
    assert payload["stage"] == "calibration"
    assert payload["models"] == list(MODELS)
    assert payload["conditions"] == list(CONDITIONS)
    assert payload["calibration_seeds"] == list(CALIBRATION_SEEDS)


def test_hash_payload_excludes_timestamps_and_local_paths() -> None:
    """The payload contains nothing that would make the hash unstable."""
    payload = build_calibration_run_identity_payload(**_identity_kwargs())
    serialised = json.dumps(payload)
    for forbidden in (
        "generated_at",
        "timestamp",
        "/results/",
        "/tmp/",
        "\\Users\\",
    ):
        assert forbidden not in serialised


def test_identity_input_with_invalid_hash_is_rejected() -> None:
    """Malformed candidate hashes raise ValueError at input validation."""
    candidates = _synthetic_executable_candidate_identities()
    candidates[3]["configuration_hash_full"] = "not-a-hash"
    with pytest.raises(ValueError) as excinfo:
        compute_calibration_run_hash_full(
            **_identity_kwargs(candidates=candidates)
        )
    assert "configuration_hash_full" in str(excinfo.value)


def test_empty_candidate_list_is_rejected() -> None:
    """An empty candidate sequence is rejected by the identity helper."""
    with pytest.raises(ValueError):
        compute_calibration_run_hash_full(**_identity_kwargs(candidates=[]))


# ---------------------------------------------------------------------------
# Path tests
# ---------------------------------------------------------------------------


def test_selected_configurations_path_layout(tmp_path: Path) -> None:
    """The path resolves to .../model_selection/calibration/<hash12>/<file>."""
    digest = compute_calibration_run_hash_full(**_identity_kwargs())
    hash12 = digest[:HASH_PREFIX_LENGTH]
    path = selected_configurations_path(
        calibration_run_hash12=hash12,
        results_root=tmp_path,
    )
    assert path == (
        tmp_path
        / "model_selection"
        / "calibration"
        / hash12
        / SELECTED_CONFIGURATIONS_FILENAME
    )


def test_selected_configurations_path_does_not_create_directory(
    tmp_path: Path,
) -> None:
    """Building the path does not create any filesystem entry."""
    digest = compute_calibration_run_hash_full(**_identity_kwargs())
    hash12 = digest[:HASH_PREFIX_LENGTH]
    path = selected_configurations_path(
        calibration_run_hash12=hash12,
        results_root=tmp_path,
    )
    assert not path.exists()
    assert not path.parent.exists()


def test_selected_configurations_path_rejects_bad_prefix(
    tmp_path: Path,
) -> None:
    """A non-hex or wrong-length prefix raises ValueError."""
    with pytest.raises(ValueError):
        selected_configurations_path(
            calibration_run_hash12="not-hex-1234",
            results_root=tmp_path,
        )
    with pytest.raises(ValueError):
        selected_configurations_path(
            calibration_run_hash12="ab",
            results_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


def test_valid_artefact_passes_validation() -> None:
    """The factory-constructed artefact passes the schema validator."""
    artefact = _make_artefact()
    validate_selected_configurations_artefact(artefact)


def test_validation_rejects_missing_top_level_field() -> None:
    """A missing top-level field is reported with the field name."""
    artefact = _make_artefact()
    del artefact["selection_rule_id"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "selection_rule_id" in str(excinfo.value)


def test_validation_rejects_unknown_top_level_field() -> None:
    """An unknown top-level field is rejected."""
    artefact = _make_artefact()
    artefact["unexpected_field"] = "value"
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "unexpected_field" in str(excinfo.value)


def test_validation_rejects_missing_condition() -> None:
    """A missing condition key under selections is reported."""
    artefact = _make_artefact()
    del artefact["selections"]["centred_only"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "centred_only" in str(excinfo.value)


def test_validation_rejects_missing_model() -> None:
    """A missing model key under a condition is reported."""
    artefact = _make_artefact()
    del artefact["selections"]["standardised"]["dcdi"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "dcdi" in str(excinfo.value)


def test_validation_rejects_base_model_decision_made_true() -> None:
    """A True value for base_model_decision_made is rejected."""
    artefact = _make_artefact()
    artefact["base_model_decision_made"] = True
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "base_model_decision_made" in str(excinfo.value)


def test_validation_rejects_selected_hash_prefix_mismatch() -> None:
    """A prefix that does not match the full hash is rejected."""
    artefact = _make_artefact()
    artefact["selections"]["centred_only"]["dagma"][
        "selected_configuration_hash_prefix"
    ] = "deadbeefdead"
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "selected_configuration_hash_prefix" in str(excinfo.value)


def test_validation_rejects_calibration_run_hash_prefix_mismatch() -> None:
    """The calibration_run_hash_prefix must match the full hash."""
    artefact = _make_artefact()
    artefact["calibration_run_hash_prefix"] = "0" * HASH_PREFIX_LENGTH
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "calibration_run_hash_prefix" in str(excinfo.value)


def test_validation_rejects_missing_candidate_ranking() -> None:
    """A missing candidate_ranking top-level field is rejected."""
    artefact = _make_artefact()
    del artefact["candidate_ranking"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "candidate_ranking" in str(excinfo.value)


def test_validation_rejects_candidate_ranking_with_wrong_count() -> None:
    """A candidate_ranking group with fewer than 5 candidates is rejected."""
    artefact = _make_artefact()
    artefact["candidate_ranking"]["centred_only"]["dagma"] = (
        artefact["candidate_ranking"]["centred_only"]["dagma"][:4]
    )
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "5" in str(excinfo.value) or "exactly" in str(excinfo.value)


def test_validation_rejects_rank_1_not_matching_selection() -> None:
    """The rank-1 candidate's hash must equal the selections entry."""
    artefact = _make_artefact()
    different_hash = _synthetic_hash("different_rank_1")
    artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "configuration_hash_full"
    ] = different_hash
    artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "configuration_hash_prefix"
    ] = different_hash[:HASH_PREFIX_LENGTH]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    message = str(excinfo.value)
    assert "rank" in message.lower() or "selections" in message


def test_validation_rejects_missing_aggregate_metric() -> None:
    """Removing one aggregate metric causes validation to fail."""
    artefact = _make_artefact()
    del artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "aggregate_metrics"
    ]["mean_sid"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "mean_sid" in str(excinfo.value)


def test_validation_rejects_missing_per_seed_metric() -> None:
    """Removing one per-seed metric field causes validation to fail."""
    artefact = _make_artefact()
    del artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "per_seed_metrics"
    ][0]["mmd_primary"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "mmd_primary" in str(excinfo.value)


def test_validation_rejects_missing_threshold_metrics_field() -> None:
    """Removing a threshold-metric subfield is reported with the path."""
    artefact = _make_artefact()
    del artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "per_seed_metrics"
    ][0]["threshold_metrics"][0]["sid"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "sid" in str(excinfo.value)


def test_validation_rejects_missing_mmd_by_intervention_field() -> None:
    """A missing mmd_by_intervention subfield is reported with the path."""
    artefact = _make_artefact()
    del artefact["candidate_ranking"]["centred_only"]["dagma"][0][
        "per_seed_metrics"
    ][0]["mmd_by_intervention"][0]["intervention_target"]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "intervention_target" in str(excinfo.value)


def test_validation_rejects_selected_rank_not_one() -> None:
    """selected_rank must equal 1 because the artefact stores rank-1."""
    artefact = _make_artefact()
    artefact["selections"]["centred_only"]["dagma"]["selected_rank"] = 2
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "selected_rank" in str(excinfo.value)


def test_validation_rejects_wrong_calibration_seeds() -> None:
    """calibration_seeds must equal [201, 202]."""
    artefact = _make_artefact()
    artefact["calibration_seeds"] = [201, 999]
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert "calibration_seeds" in str(excinfo.value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    ],
)
def test_validation_rejects_forbidden_winner_field_at_top_level(
    forbidden_field: str,
) -> None:
    """A forbidden field at the top level is rejected by name."""
    artefact = _make_artefact()
    artefact[forbidden_field] = "dagma"
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert forbidden_field in str(excinfo.value)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "winner",
        "model_winner",
        "base_model_winner",
        "recommended_model",
        "final_decision",
        "decision",
    ],
)
def test_validation_rejects_forbidden_winner_field_when_nested(
    forbidden_field: str,
) -> None:
    """A forbidden field nested anywhere is still rejected by name."""
    artefact = _make_artefact()
    artefact["selections"]["centred_only"]["dagma"][forbidden_field] = (
        "dcdi"
    )
    with pytest.raises(ValueError) as excinfo:
        validate_selected_configurations_artefact(artefact)
    assert forbidden_field in str(excinfo.value)


def test_validation_allows_decision_scope_and_decision_made_flag() -> None:
    """decision_scope and base_model_decision_made are allowed and required."""
    artefact = _make_artefact()
    assert artefact["decision_scope"] == (
        "within_model_configuration_selection"
    )
    assert artefact["base_model_decision_made"] is False
    validate_selected_configurations_artefact(artefact)


def test_validation_allows_optional_code_version_field() -> None:
    """code_version is permitted at the top level when present."""
    artefact = _make_artefact(
        extra_top_level={"code_version": "abc1234"}
    )
    validate_selected_configurations_artefact(artefact)


# ---------------------------------------------------------------------------
# Visual-readiness tests
# ---------------------------------------------------------------------------


def test_artefact_can_be_flattened_to_visualisation_rows() -> None:
    """The artefact carries every field a flat-row visualisation needs.

    For each (condition, model, candidate_rank, seed_value) the
    flattened row contains: condition, model, rank, hyperparameter,
    mean_sid, mean_mmd_primary, mean_shd, seed_value, shd, sid,
    mmd_primary. The test produces 80 rows total
    (2 conditions x 2 models x 5 candidates x 2 seeds).
    """
    artefact = _make_artefact()
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model in MODELS:
            for candidate in (
                artefact["candidate_ranking"][condition][model]
            ):
                hyperparameter_value = next(
                    iter(candidate["hyperparameters"].values())
                )
                aggregate = candidate["aggregate_metrics"]
                for per_seed in candidate["per_seed_metrics"]:
                    rows.append(
                        {
                            "condition": condition,
                            "model": model,
                            "rank": candidate["rank"],
                            "hyperparameter": hyperparameter_value,
                            "mean_sid": aggregate["mean_sid"],
                            "mean_mmd_primary": aggregate[
                                "mean_mmd_primary"
                            ],
                            "mean_shd": aggregate["mean_shd"],
                            "seed_value": per_seed["seed_value"],
                            "shd": per_seed["shd"],
                            "sid": per_seed["sid"],
                            "mmd_primary": per_seed["mmd_primary"],
                        }
                    )
    assert len(rows) == (
        len(CONDITIONS)
        * len(MODELS)
        * CANDIDATES_PER_CONDITION_PER_MODEL
        * len(CALIBRATION_SEEDS)
    )


def test_artefact_preserves_threshold_metrics_lists() -> None:
    """threshold_metrics is preserved as a list with the expected fields."""
    artefact = _make_artefact()
    one_per_seed = (
        artefact["candidate_ranking"]["centred_only"]["dagma"][0][
            "per_seed_metrics"
        ][0]
    )
    assert isinstance(one_per_seed["threshold_metrics"], list)
    assert len(one_per_seed["threshold_metrics"]) == 3
    for entry in one_per_seed["threshold_metrics"]:
        assert set(entry.keys()) == {
            "threshold",
            "shd",
            "sid",
            "mmd_primary",
        }


def test_artefact_preserves_mmd_by_intervention_lists() -> None:
    """mmd_by_intervention is preserved as a 20-element list."""
    artefact = _make_artefact()
    one_per_seed = (
        artefact["candidate_ranking"]["centred_only"]["dagma"][0][
            "per_seed_metrics"
        ][0]
    )
    assert isinstance(one_per_seed["mmd_by_intervention"], list)
    assert len(one_per_seed["mmd_by_intervention"]) == 20
    for entry in one_per_seed["mmd_by_intervention"]:
        assert set(entry.keys()) == {
            "intervention_target",
            "intervention_value",
            "mmd_primary",
        }


def test_artefact_preserves_bandwidth_summaries() -> None:
    """bandwidth_summaries is preserved on every per-seed record."""
    artefact = _make_artefact()
    one_per_seed = (
        artefact["candidate_ranking"]["centred_only"]["dagma"][0][
            "per_seed_metrics"
        ][0]
    )
    assert "bandwidth_summaries" in one_per_seed
    assert isinstance(one_per_seed["bandwidth_summaries"], dict)


# ---------------------------------------------------------------------------
# Writer / round-trip tests
# ---------------------------------------------------------------------------


def _write_path(tmp_path: Path, digest: str | None = None) -> Path:
    """Build a writable selected_configurations.json path under tmp_path."""
    if digest is None:
        digest = _synthetic_hash("default_calibration_run")
    return selected_configurations_path(
        calibration_run_hash12=digest[:HASH_PREFIX_LENGTH],
        results_root=tmp_path,
    )


def test_writer_refuses_overwrite_by_default(tmp_path: Path) -> None:
    """A second write without force raises FileExistsError."""
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_selected_configurations(artefact, path)


def test_writer_force_true_allows_overwrite(tmp_path: Path) -> None:
    """A second write with force=True overwrites the canonical file."""
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    # Mutate a non-identity field for a visible difference.
    artefact_v2 = deepcopy(artefact)
    artefact_v2["generated_at_utc"] = "2026-05-22T13:00:00Z"
    write_selected_configurations(artefact_v2, path, force=True)
    with path.open(encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk["generated_at_utc"] == "2026-05-22T13:00:00Z"


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    """The writer creates intermediate directories under results_root."""
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    assert not path.parent.exists()
    write_selected_configurations(artefact, path)
    assert path.is_file()


def test_writer_uses_indent_2_sort_keys_utf8(tmp_path: Path) -> None:
    """The on-disk JSON is formatted with indent=2 and sorted keys, UTF-8."""
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    raw = path.read_bytes()
    # ensure_ascii=True is ASCII-only by construction.
    assert all(byte < 128 for byte in raw)
    text = raw.decode("utf-8")
    # indent=2 produces a leading "{\n  "schema_version": ..."
    assert text.startswith("{\n  ")
    # sort_keys=True puts artefact_type before base_model_decision_made
    assert text.index('"artefact_type"') < text.index(
        '"base_model_decision_made"'
    )


def test_writer_does_not_write_invalid_artefact(tmp_path: Path) -> None:
    """An invalid artefact never reaches the canonical path."""
    artefact = _make_artefact()
    del artefact["selection_rule_id"]
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    with pytest.raises(ValueError):
        write_selected_configurations(artefact, path)
    assert not path.exists()


def test_writer_leaves_no_temp_file_after_success(tmp_path: Path) -> None:
    """No temporary artefact remains in the target directory after success."""
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    siblings = list(path.parent.iterdir())
    assert siblings == [path], (
        f"unexpected sibling files left in target directory: "
        f"{siblings!r}"
    )


def test_writer_leaves_no_temp_file_after_invalid_artefact(
    tmp_path: Path,
) -> None:
    """An invalid artefact leaves no temporary file in the target directory.

    The validator runs before any temporary file is opened, so no
    temp file is created and the parent directory is not even
    populated.
    """
    artefact = _make_artefact()
    del artefact["selections"]
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    with pytest.raises(ValueError):
        write_selected_configurations(artefact, path)
    assert not path.exists()
    # The parent directory should not be created since validation
    # fails before any I/O occurs.
    if path.parent.exists():
        # If the parent exists at all (it should not under the
        # current implementation), it must at least be empty.
        assert list(path.parent.iterdir()) == []


def test_writer_round_trip_preserves_realistic_artefact_shape(
    tmp_path: Path,
) -> None:
    """A realistic-shape artefact round-trips through the writer.

    Builds an artefact with 2 conditions, 2 models, 5 candidates per
    condition/model, 2 per-seed metric records per candidate, 3
    threshold_metrics rows per per-seed record, and 20
    mmd_by_intervention rows per per-seed record. Writes the
    artefact, re-reads it, and asserts equality with the in-memory
    structure.
    """
    artefact = _make_artefact()
    # Smoke-check the shape before writing.
    assert len(artefact["candidate_ranking"]) == len(CONDITIONS)
    for condition in CONDITIONS:
        assert len(artefact["candidate_ranking"][condition]) == (
            len(MODELS)
        )
        for model in MODELS:
            candidates = artefact["candidate_ranking"][condition][model]
            assert len(candidates) == (
                CANDIDATES_PER_CONDITION_PER_MODEL
            )
            for candidate in candidates:
                assert len(candidate["per_seed_metrics"]) == 2
                for per_seed in candidate["per_seed_metrics"]:
                    assert len(per_seed["threshold_metrics"]) == 3
                    assert len(per_seed["mmd_by_intervention"]) == 20
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    with path.open(encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk == artefact


# ---------------------------------------------------------------------------
# Scope confirmations
# ---------------------------------------------------------------------------


def test_module_does_not_invoke_pipeline_run_single_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The artefact module does not invoke pipeline.run_single_fit.

    Monkeypatches pipeline.run_single_fit with a call counter and
    exercises the full happy-path write of a synthetic artefact.
    The counter must remain at zero.
    """
    from experiments.selection_study import pipeline

    fit_invocations = {"count": 0}

    def fake_run_single_fit(*args: Any, **kwargs: Any) -> Path:
        fit_invocations["count"] += 1
        return Path("/tmp/never-reached")

    monkeypatch.setattr(
        pipeline, "run_single_fit", fake_run_single_fit
    )
    artefact = _make_artefact()
    path = _write_path(tmp_path, artefact["calibration_run_hash_full"])
    write_selected_configurations(artefact, path)
    assert fit_invocations["count"] == 0


def test_module_exposes_no_final_winner_constant() -> None:
    """The module exports no constant resembling a final winner."""
    from experiments.selection_study import selection_artefact

    public_names = [
        name for name in dir(selection_artefact) if not name.startswith("_")
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
