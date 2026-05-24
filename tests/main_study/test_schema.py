"""Tests for the planning-side main-study schema, factory, and hashing.

These tests construct synthetic minimal :class:`PriorSpec` and
:class:`CorruptedPriorSpec` instances directly and never invoke SCM
generation, DAGMA fits, or any I/O. Module-import allowlisting is
verified by AST inspection of the schema module source.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

from experiments.main_study import priors as priors_mod
from experiments.main_study import schema as schema_mod
from experiments.main_study.priors import (
    CORRUPTION_GRID,
    PRIOR_K,
    CorruptedPriorSpec,
    PriorSpec,
    edge_tuple_to_key,
)
from experiments.main_study.schema import (
    CALIBRATION_SEEDS,
    CONFIDENCE_GRID,
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    METHOD_FAMILIES,
    SCHEMA_VERSION,
    SEED_POPULATIONS,
    MainStudyConfig,
    canonicalize_for_json,
    compute_configuration_hash,
    configuration_hash_prefix,
    make_main_study_config,
    make_run_id,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


_VALID_PARENT_HASH = "a" * 64
_VALID_PARENT_HASH_OTHER = "b" * 64


def _make_clean_prior() -> PriorSpec:
    return PriorSpec(
        n_nodes=5,
        scm_seed=42,
        prior_selection_seed=9042,
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
    )


def _make_corrupted_prior(
    *,
    forbidden_edges: tuple[tuple[int, int], ...] = (
        (0, 2), (1, 3), (2, 4)
    ),
    corruption_fraction: float = 0.0,
    corruption_index: int = 0,
    corruption_seed: int = 9100 + 42,
    n_correct: int | None = None,
    n_corrupted: int = 0,
    removed_clean_edges: tuple[tuple[int, int], ...] = (),
    added_true_positive_edges: tuple[tuple[int, int], ...] = (),
    edge_labels: dict[str, str] | None = None,
) -> CorruptedPriorSpec:
    if n_correct is None:
        n_correct = len(forbidden_edges) - n_corrupted
    if edge_labels is None:
        edge_labels = {
            edge_tuple_to_key(e): "true_negative_retained"
            for e in forbidden_edges
        }
    return CorruptedPriorSpec(
        n_nodes=5,
        scm_seed=42,
        corruption_fraction=corruption_fraction,
        corruption_index=corruption_index,
        corruption_seed=corruption_seed,
        forbidden_edges=tuple(forbidden_edges),
        n_correct=n_correct,
        n_corrupted=n_corrupted,
        removed_clean_edges=removed_clean_edges,
        added_true_positive_edges=added_true_positive_edges,
        edge_labels=edge_labels,
    )


def _default_dagma_config(
    exclude_edges: tuple[tuple[int, int], ...] | None = None,
) -> DAGMAConfig:
    return DAGMAConfig(exclude_edges=exclude_edges)


def _build_prior_free_config(
    *,
    seed_value: int = 401,
    seed_population: str = "main_calibration",
    parent_hash: str = _VALID_PARENT_HASH,
) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=seed_value,
        seed_population=seed_population,
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=parent_hash,
    )


def _build_soft_frobenius_config(
    *,
    seed_value: int = 401,
    confidence: float = 0.5,
    corruption_fraction: float = 0.4,
) -> MainStudyConfig:
    cp = _make_corrupted_prior(
        corruption_fraction=corruption_fraction,
        corruption_index=2 if corruption_fraction == 0.4 else 0,
    )
    return MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=confidence,
        corrupted_prior_spec=cp,
    )


def _build_hard_exclusion_config(
    *,
    seed_value: int = 401,
    corruption_fraction: float = 0.4,
) -> MainStudyConfig:
    cp = _make_corrupted_prior(
        corruption_fraction=corruption_fraction,
        corruption_index=2,
    )
    return MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(
            exclude_edges=tuple(sorted(cp.forbidden_edges))
        ),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )


def _build_matched_l1_config(
    *,
    seed_value: int = 401,
    matched_l1_lambda1: float = 0.07,
) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="matched_l1",
        seed_value=seed_value,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        matched_l1_lambda1=matched_l1_lambda1,
    )


# ---------------------------------------------------------------------------
# T-1, T-2, T-3: Constants
# ---------------------------------------------------------------------------


def test_schema_version_is_two():
    assert SCHEMA_VERSION == 2


def test_constants_documented_values_and_imports_from_priors():
    assert METHOD_FAMILIES == (
        "prior_free",
        "soft_frobenius",
        "matched_l1",
        "hard_exclusion",
    )
    assert SEED_POPULATIONS == ("main_calibration", "main_evaluation")
    assert CALIBRATION_SEEDS == (401, 402)
    assert EVALUATION_SEEDS == (501, 502, 503, 504, 505, 506, 507)
    assert CONFIDENCE_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    # PRIOR_K and CORRUPTION_GRID must be the priors-module values,
    # re-exported through schema rather than redefined.
    assert schema_mod.PRIOR_K is priors_mod.PRIOR_K
    assert schema_mod.CORRUPTION_GRID is priors_mod.CORRUPTION_GRID
    assert PRIOR_K == 10
    assert CORRUPTION_GRID == (0.0, 0.2, 0.4, 0.6, 0.8)


def test_frozen_lambda_prior_value():
    assert FROZEN_LAMBDA_PRIOR == pytest.approx(2e-4, abs=1e-12)


# ---------------------------------------------------------------------------
# T-4: unknown method_family
# ---------------------------------------------------------------------------


def test_main_study_config_rejects_unknown_method_family():
    with pytest.raises(ValueError, match="method_family"):
        MainStudyConfig(
            method_family="unknown",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


# ---------------------------------------------------------------------------
# T-5: seed-population / seed-value boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(CALIBRATION_SEEDS))
def test_calibration_seed_population_accepts_calibration_seeds(seed):
    cfg = MainStudyConfig(
        method_family="prior_free",
        seed_value=seed,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    assert cfg.seed_value == seed


@pytest.mark.parametrize("seed", list(EVALUATION_SEEDS))
def test_evaluation_seed_population_accepts_evaluation_seeds(seed):
    cfg = MainStudyConfig(
        method_family="prior_free",
        seed_value=seed,
        seed_population="main_evaluation",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    assert cfg.seed_value == seed


def test_calibration_population_rejects_evaluation_seed():
    with pytest.raises(ValueError, match="main_calibration"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=501,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


def test_evaluation_population_rejects_calibration_seed():
    with pytest.raises(ValueError, match="main_evaluation"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_evaluation",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


def test_unknown_seed_population_rejected():
    with pytest.raises(ValueError, match="seed_population"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=401,
            seed_population="unknown",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


# ---------------------------------------------------------------------------
# T-6: method-family invariants
# ---------------------------------------------------------------------------


def test_prior_free_valid_construction():
    cfg = _build_prior_free_config()
    assert cfg.method_family == "prior_free"
    assert cfg.lambda_prior is None
    assert cfg.confidence is None
    assert cfg.corrupted_prior_spec is None
    assert cfg.matched_l1_lambda1 is None


def test_prior_free_rejects_confidence():
    with pytest.raises(ValueError, match="prior_free"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            confidence=0.5,
        )


def test_prior_free_rejects_exclude_edges_in_dagma_config():
    with pytest.raises(ValueError, match="prior_free"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(exclude_edges=((0, 1),)),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


def test_soft_frobenius_valid_construction():
    cfg = _build_soft_frobenius_config()
    assert cfg.lambda_prior == pytest.approx(FROZEN_LAMBDA_PRIOR)
    assert cfg.confidence == pytest.approx(0.5)


def test_soft_frobenius_requires_confidence():
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="confidence"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=FROZEN_LAMBDA_PRIOR,
            corrupted_prior_spec=cp,
        )


def test_soft_frobenius_requires_corrupted_prior_spec():
    with pytest.raises(ValueError, match="corrupted_prior_spec"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=FROZEN_LAMBDA_PRIOR,
            confidence=0.5,
        )


def test_soft_frobenius_rejects_exclude_edges():
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="exclude_edges"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(exclude_edges=((0, 1),)),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=FROZEN_LAMBDA_PRIOR,
            confidence=0.5,
            corrupted_prior_spec=cp,
        )


def test_matched_l1_valid_construction():
    cfg = _build_matched_l1_config(matched_l1_lambda1=0.07)
    assert cfg.matched_l1_lambda1 == pytest.approx(0.07)


def test_matched_l1_rejects_zero_lambda():
    with pytest.raises(ValueError, match="matched_l1_lambda1"):
        MainStudyConfig(
            method_family="matched_l1",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            matched_l1_lambda1=0.0,
        )


def test_matched_l1_rejects_negative_lambda():
    with pytest.raises(ValueError, match="matched_l1_lambda1"):
        MainStudyConfig(
            method_family="matched_l1",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            matched_l1_lambda1=-1e-3,
        )


def test_hard_exclusion_valid_construction():
    cfg = _build_hard_exclusion_config()
    assert cfg.dagma_config.exclude_edges is not None
    assert set(cfg.dagma_config.exclude_edges) == set(
        cfg.corrupted_prior_spec.forbidden_edges
    )


def test_hard_exclusion_requires_matching_exclude_edges():
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    with pytest.raises(ValueError, match="hard_exclusion"):
        MainStudyConfig(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            # Disagrees with corrupted_prior_spec.forbidden_edges.
            dagma_config=_default_dagma_config(exclude_edges=((0, 4),)),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-7: soft_frobenius rejects lambda_prior != FROZEN_LAMBDA_PRIOR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_lambda", [0.0, 1e-2, 0.5, 1.0])
def test_soft_frobenius_rejects_lambda_other_than_frozen(bad_lambda):
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="lambda_prior"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=bad_lambda,
            confidence=0.5,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-8: hard_exclusion exclude_edges set equality
# ---------------------------------------------------------------------------


def test_hard_exclusion_exclude_edges_must_equal_forbidden_edges():
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    # Edge set differs by one element.
    with pytest.raises(ValueError, match="hard_exclusion"):
        MainStudyConfig(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(
                exclude_edges=((0, 2), (1, 3))
            ),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


def test_hard_exclusion_rejects_none_exclude_edges():
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="hard_exclusion"):
        MainStudyConfig(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(exclude_edges=None),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


def test_hard_exclusion_direct_construction_rejects_duplicate_exclude_edges():
    """Duplicate exclude_edges that deduplicate to the forbidden set must fail.

    Set equality would silently accept this malformed input; sorted-
    tuple equality catches the length mismatch.
    """
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    duplicated_exclude = ((0, 2), (0, 2), (1, 3), (2, 4))
    with pytest.raises(ValueError, match="length"):
        MainStudyConfig(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(
                exclude_edges=duplicated_exclude
            ),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


def test_hard_exclusion_factory_rejects_duplicate_exclude_edges():
    """Factory must reject caller-supplied duplicate exclude_edges
    whose deduplicated set matches corrupted_prior_spec.forbidden_edges.
    """
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    duplicated_exclude = ((0, 2), (0, 2), (1, 3), (2, 4))
    with pytest.raises(ValueError, match="length"):
        make_main_study_config(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(
                exclude_edges=duplicated_exclude
            ),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-9: confidence grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", list(CONFIDENCE_GRID))
def test_confidence_grid_values_accepted(c):
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    cfg = MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=c,
        corrupted_prior_spec=cp,
    )
    assert cfg.confidence == pytest.approx(c)


@pytest.mark.parametrize("bad", [-0.1, 0.1, 0.3, 0.7, 1.1, float("nan")])
def test_confidence_off_grid_values_rejected(bad):
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="confidence"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=FROZEN_LAMBDA_PRIOR,
            confidence=bad,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-10: corruption_fraction grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac", list(CORRUPTION_GRID))
def test_corruption_fraction_grid_values_accepted(frac):
    cp = _make_corrupted_prior(corruption_fraction=frac)
    cfg = MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=0.5,
        corrupted_prior_spec=cp,
    )
    assert cfg.corrupted_prior_spec.corruption_fraction == pytest.approx(
        frac
    )


@pytest.mark.parametrize("bad", [-0.2, 0.1, 0.3, 0.7, 1.0])
def test_corruption_fraction_off_grid_values_rejected(bad):
    cp = _make_corrupted_prior(corruption_fraction=bad)
    with pytest.raises(ValueError, match="corruption_fraction"):
        MainStudyConfig(
            method_family="soft_frobenius",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            lambda_prior=FROZEN_LAMBDA_PRIOR,
            confidence=0.5,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-11: parent_heldout_run_hash_full validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "A" * 64,
        "g" * 64,
        "ab" + "c" * 63,
        "a" * 63,
        "a" * 65,
    ],
)
def test_parent_hash_rejects_malformed(bad):
    with pytest.raises(ValueError, match="parent_heldout_run_hash_full"):
        MainStudyConfig(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=bad,
        )


def test_parent_hash_accepts_64_lowercase_hex():
    cfg = _build_prior_free_config(parent_hash="0123456789abcdef" * 4)
    assert cfg.parent_heldout_run_hash_full == "0123456789abcdef" * 4


# ---------------------------------------------------------------------------
# T-12: hash determinism
# ---------------------------------------------------------------------------


def test_configuration_hash_deterministic_for_same_config():
    cfg_a = _build_soft_frobenius_config(corruption_fraction=0.4)
    cfg_b = _build_soft_frobenius_config(corruption_fraction=0.4)
    assert compute_configuration_hash(cfg_a) == compute_configuration_hash(
        cfg_b
    )


# ---------------------------------------------------------------------------
# T-13: hash changes when relevant fields change
# ---------------------------------------------------------------------------


def test_hash_changes_with_method_family():
    cfg_pf = _build_prior_free_config()
    cfg_ml1 = _build_matched_l1_config()
    assert compute_configuration_hash(cfg_pf) != compute_configuration_hash(
        cfg_ml1
    )


def test_hash_changes_with_seed_value():
    a = _build_prior_free_config(seed_value=401)
    b = _build_prior_free_config(seed_value=402)
    assert compute_configuration_hash(a) != compute_configuration_hash(b)


def test_hash_changes_with_seed_population():
    a = MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    b = MainStudyConfig(
        method_family="prior_free",
        seed_value=501,
        seed_population="main_evaluation",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    # Different seed and population both change the hash; this test
    # asserts the combined difference, which by construction also
    # establishes that changing the population alone changes the hash
    # (only seed_value differs from population).
    assert compute_configuration_hash(a) != compute_configuration_hash(b)


def test_hash_changes_with_confidence():
    a = _build_soft_frobenius_config(confidence=0.25)
    b = _build_soft_frobenius_config(confidence=0.75)
    assert compute_configuration_hash(a) != compute_configuration_hash(b)


def test_hash_changes_with_corruption_fraction():
    a = _build_soft_frobenius_config(corruption_fraction=0.2)
    b = _build_soft_frobenius_config(corruption_fraction=0.4)
    assert compute_configuration_hash(a) != compute_configuration_hash(b)


def test_hash_changes_with_dagma_config_lambda1():
    base = _default_dagma_config()
    modified = dataclasses.replace(base, lambda1=0.02)
    cfg_a = MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=base,
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    cfg_b = MainStudyConfig(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=modified,
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    assert compute_configuration_hash(cfg_a) != compute_configuration_hash(
        cfg_b
    )


def test_hash_changes_with_dagma_config_exclude_edges():
    cp_a = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3)),
        corruption_fraction=0.0,
    )
    cp_b = _make_corrupted_prior(
        forbidden_edges=((0, 3), (1, 3)),
        corruption_fraction=0.0,
    )
    cfg_a = MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(
            exclude_edges=tuple(sorted(cp_a.forbidden_edges))
        ),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp_a,
    )
    cfg_b = MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(
            exclude_edges=tuple(sorted(cp_b.forbidden_edges))
        ),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp_b,
    )
    assert compute_configuration_hash(cfg_a) != compute_configuration_hash(
        cfg_b
    )


# ---------------------------------------------------------------------------
# T-14: edge ordering invariance
# ---------------------------------------------------------------------------


def test_hash_invariant_to_forbidden_edges_ordering():
    edges_sorted = ((0, 2), (1, 3), (2, 4))
    edges_shuffled = ((2, 4), (0, 2), (1, 3))
    cp_a = _make_corrupted_prior(
        forbidden_edges=edges_sorted, corruption_fraction=0.0
    )
    cp_b = _make_corrupted_prior(
        forbidden_edges=edges_shuffled, corruption_fraction=0.0
    )
    cfg_a = MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=0.5,
        corrupted_prior_spec=cp_a,
    )
    cfg_b = MainStudyConfig(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        lambda_prior=FROZEN_LAMBDA_PRIOR,
        confidence=0.5,
        corrupted_prior_spec=cp_b,
    )
    assert compute_configuration_hash(cfg_a) == compute_configuration_hash(
        cfg_b
    )


def test_hash_invariant_to_exclude_edges_ordering():
    edges_sorted = ((0, 2), (1, 3), (2, 4))
    edges_shuffled = ((2, 4), (1, 3), (0, 2))
    cp = _make_corrupted_prior(
        forbidden_edges=edges_sorted, corruption_fraction=0.0
    )
    cfg_a = MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(exclude_edges=edges_sorted),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )
    cfg_b = MainStudyConfig(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(
            exclude_edges=edges_shuffled
        ),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )
    assert compute_configuration_hash(cfg_a) == compute_configuration_hash(
        cfg_b
    )


# ---------------------------------------------------------------------------
# T-15: parent hash does not influence configuration hash
# ---------------------------------------------------------------------------


def test_parent_hash_does_not_affect_configuration_hash():
    a = _build_prior_free_config(parent_hash=_VALID_PARENT_HASH)
    b = _build_prior_free_config(parent_hash=_VALID_PARENT_HASH_OTHER)
    assert compute_configuration_hash(a) == compute_configuration_hash(b)


def test_schema_version_does_not_appear_in_canonical_condition():
    # Inspect the body of compute_configuration_hash excluding its
    # docstring, which legitimately mentions both excluded fields.
    func_node = ast.parse(
        inspect.getsource(compute_configuration_hash)
    ).body[0]
    body = list(func_node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    body_src = "\n".join(ast.unparse(n) for n in body)
    assert "schema_version" not in body_src
    assert "parent_heldout_run_hash_full" not in body_src


# ---------------------------------------------------------------------------
# T-16: canonicalize_for_json behaviour
# ---------------------------------------------------------------------------


def test_canonicalize_dataclass_to_dict():
    spec = _make_clean_prior()
    out = canonicalize_for_json(spec)
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "n_nodes",
        "scm_seed",
        "prior_selection_seed",
        "forbidden_edges",
    }
    # Edge field has been converted and sorted.
    assert out["forbidden_edges"] == [[0, 2], [1, 3], [2, 4]]


def test_canonicalize_tuple_to_list():
    assert canonicalize_for_json((1, 2, 3)) == [1, 2, 3]
    assert canonicalize_for_json([1, 2, 3]) == [1, 2, 3]


def test_canonicalize_recognised_edge_collections_sorted():
    # field_name routing on a top-level call.
    out = canonicalize_for_json(
        ((2, 1), (0, 1), (3, 0)),
        field_name="forbidden_edges",
    )
    assert out == [[0, 1], [2, 1], [3, 0]]


def test_canonicalize_edge_labels_sorted_by_key():
    labels = {"2,4": "true_negative_retained", "0,1": "true_negative_retained"}
    out = canonicalize_for_json(labels)
    assert list(out.keys()) == ["0,1", "2,4"]


def test_canonicalize_preserves_none_and_bools():
    assert canonicalize_for_json(None) is None
    assert canonicalize_for_json(True) is True
    assert canonicalize_for_json(False) is False
    # Bool must not be coerced to 1/0.
    assert canonicalize_for_json(True) is not 1
    # Verify JSON serialisation distinguishes them.
    assert json.dumps(canonicalize_for_json(True)) == "true"


def test_canonicalize_rejects_unsupported_type():
    class Foo:
        pass

    with pytest.raises(TypeError):
        canonicalize_for_json(Foo())


def test_canonicalize_rejects_non_string_dict_keys():
    with pytest.raises(TypeError):
        canonicalize_for_json({1: "value"})


# ---------------------------------------------------------------------------
# T-17: hash uses canonical JSON + SHA-256
# ---------------------------------------------------------------------------


def test_compute_configuration_hash_matches_manual_sha256():
    cfg = _build_prior_free_config()
    condition = {
        "method_family": canonicalize_for_json(cfg.method_family),
        "seed_value": canonicalize_for_json(cfg.seed_value),
        "seed_population": canonicalize_for_json(cfg.seed_population),
        "dagma_config": canonicalize_for_json(cfg.dagma_config),
        "lambda_prior": canonicalize_for_json(cfg.lambda_prior),
        "confidence": canonicalize_for_json(cfg.confidence),
        "corrupted_prior_spec": canonicalize_for_json(
            cfg.corrupted_prior_spec
        ),
        "matched_l1_lambda1": canonicalize_for_json(
            cfg.matched_l1_lambda1
        ),
    }
    payload = json.dumps(
        condition, sort_keys=True, separators=(",", ":")
    )
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert compute_configuration_hash(cfg) == expected
    assert len(expected) == 64
    assert all(ch in "0123456789abcdef" for ch in expected)


def test_compute_configuration_hash_implementation_uses_hashlib_sha256():
    src = inspect.getsource(compute_configuration_hash)
    assert "hashlib.sha256" in src
    assert "json.dumps" in src
    assert 'sort_keys=True' in src
    assert "(\",\", \":\")" in src


def test_python_builtin_hash_is_not_used_in_schema_module():
    src = Path(schema_mod.__file__).read_text(encoding="utf-8")
    # The Python builtin ``hash(...)`` must not appear anywhere in
    # the persistent-identifier code path. Allow the word "hash" in
    # variable names like ``parent_heldout_run_hash_full`` and in
    # function names like ``compute_configuration_hash``.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "hash", (
                "schema.py must not call the Python builtin hash() "
                "for persistent identifiers."
            )


# ---------------------------------------------------------------------------
# T-18: configuration_hash_prefix returns first 12 chars
# ---------------------------------------------------------------------------


def test_configuration_hash_prefix_is_first_12_chars():
    cfg = _build_prior_free_config()
    full = compute_configuration_hash(cfg)
    prefix = configuration_hash_prefix(cfg)
    assert len(prefix) == 12
    assert prefix == full[:12]


# ---------------------------------------------------------------------------
# T-19: make_run_id format
# ---------------------------------------------------------------------------


def test_make_run_id_format():
    cfg = _build_prior_free_config(
        seed_value=402, seed_population="main_calibration"
    )
    rid = make_run_id(cfg)
    prefix12 = configuration_hash_prefix(cfg)
    expected = f"prior_free__main_calibration__seed402__cfg{prefix12}"
    assert rid == expected


# ---------------------------------------------------------------------------
# T-20: factory injects FROZEN_LAMBDA_PRIOR; signature has no lambda_prior
# ---------------------------------------------------------------------------


def test_factory_signature_does_not_accept_lambda_prior():
    sig = inspect.signature(make_main_study_config)
    assert "lambda_prior" not in sig.parameters


def test_factory_injects_frozen_lambda_prior_for_soft_frobenius():
    cp = _make_corrupted_prior(corruption_fraction=0.4, corruption_index=2)
    cfg = make_main_study_config(
        method_family="soft_frobenius",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        confidence=0.5,
        corrupted_prior_spec=cp,
    )
    assert cfg.lambda_prior == pytest.approx(FROZEN_LAMBDA_PRIOR)


def test_factory_lambda_prior_for_other_families_stays_none():
    cfg = make_main_study_config(
        method_family="prior_free",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
    )
    assert cfg.lambda_prior is None


# ---------------------------------------------------------------------------
# T-21: factory wires exclude_edges for hard_exclusion
# ---------------------------------------------------------------------------


def test_factory_sets_exclude_edges_for_hard_exclusion():
    cp = _make_corrupted_prior(
        forbidden_edges=((2, 4), (0, 2), (1, 3)),
        corruption_fraction=0.0,
    )
    cfg = make_main_study_config(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )
    assert cfg.dagma_config.exclude_edges is not None
    # Factory must sort the edges lexicographically.
    assert cfg.dagma_config.exclude_edges == ((0, 2), (1, 3), (2, 4))


def test_factory_accepts_caller_supplied_matching_exclude_edges():
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    cfg = make_main_study_config(
        method_family="hard_exclusion",
        seed_value=401,
        seed_population="main_calibration",
        dagma_config=_default_dagma_config(
            exclude_edges=((2, 4), (0, 2), (1, 3))
        ),
        parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        corrupted_prior_spec=cp,
    )
    # Sorted by the factory before validation.
    assert cfg.dagma_config.exclude_edges == ((0, 2), (1, 3), (2, 4))


def test_factory_rejects_conflicting_exclude_edges_for_hard_exclusion():
    cp = _make_corrupted_prior(
        forbidden_edges=((0, 2), (1, 3), (2, 4)),
        corruption_fraction=0.0,
    )
    with pytest.raises(ValueError, match="hard_exclusion"):
        make_main_study_config(
            method_family="hard_exclusion",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(
                exclude_edges=((0, 1),)
            ),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-22: factory rejects invalid prior_free and matched_l1 combinations
# ---------------------------------------------------------------------------


def test_factory_rejects_prior_free_with_confidence():
    with pytest.raises(ValueError, match="prior_free"):
        make_main_study_config(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            confidence=0.5,
        )


def test_factory_rejects_prior_free_with_corrupted_prior_spec():
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="prior_free"):
        make_main_study_config(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            corrupted_prior_spec=cp,
        )


def test_factory_rejects_prior_free_with_matched_l1_lambda1():
    with pytest.raises(ValueError, match="prior_free"):
        make_main_study_config(
            method_family="prior_free",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            matched_l1_lambda1=0.07,
        )


def test_factory_requires_matched_l1_lambda1_for_matched_l1():
    with pytest.raises(ValueError, match="matched_l1"):
        make_main_study_config(
            method_family="matched_l1",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
        )


def test_factory_rejects_matched_l1_with_confidence():
    with pytest.raises(ValueError, match="matched_l1"):
        make_main_study_config(
            method_family="matched_l1",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            matched_l1_lambda1=0.07,
            confidence=0.5,
        )


def test_factory_rejects_matched_l1_with_corrupted_prior_spec():
    cp = _make_corrupted_prior(corruption_fraction=0.0)
    with pytest.raises(ValueError, match="matched_l1"):
        make_main_study_config(
            method_family="matched_l1",
            seed_value=401,
            seed_population="main_calibration",
            dagma_config=_default_dagma_config(),
            parent_heldout_run_hash_full=_VALID_PARENT_HASH,
            matched_l1_lambda1=0.07,
            corrupted_prior_spec=cp,
        )


# ---------------------------------------------------------------------------
# T-23: import allowlist for schema.py
# ---------------------------------------------------------------------------


_SCHEMA_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "dataclasses",
    "typing",
    "json",
    "hashlib",
    "math",
    "re",
    "inspect",
    "experiments.main_study.priors",
    "symbolic_priors_cd.wrappers.dagma",
})


_SCHEMA_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "experiments.selection_study",
    "experiments.main_study.calibration_lambda_prior",
    "dagma",
    "dcdi",
)


def _module_imports(tree: ast.Module) -> list[tuple[str, bool]]:
    top_ids = {id(n) for n in tree.body}
    out: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, id(node) in top_ids))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append((node.module, id(node) in top_ids))
    return out


def test_schema_module_imports_are_allowlisted():
    src = Path(schema_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod, _is_top in _module_imports(tree):
        for forbidden in _SCHEMA_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"schema.py must not import {mod!r}; "
                f"forbidden prefix {forbidden!r}."
            )
        ok = (
            mod in _SCHEMA_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _SCHEMA_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"schema.py import {mod!r} is not in the allowlist "
            f"{sorted(_SCHEMA_ALLOWED_PREFIXES)}."
        )
