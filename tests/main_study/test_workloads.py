"""Tests for the main-study workload and artefact-plan enumeration.

These tests construct planned configurations and verify shape,
ordering, uniqueness, validation, and the import allowlist. No
DAGMA fit is invoked, no metric is computed, no file or directory
is created.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from experiments.main_study import workloads as workloads_mod
from experiments.main_study.paths import (
    ARTEFACT_NAMES,
    artefact_path,
    record_filename,
    records_dir,
    validate_relative_posix_path,
)
from experiments.main_study.priors import (
    CORRUPTION_GRID,
    PRIOR_K,
    CorruptedPriorSpec,
)
from experiments.main_study.schema import (
    CALIBRATION_SEEDS,
    CONFIDENCE_GRID,
    EVALUATION_SEEDS,
    FROZEN_LAMBDA_PRIOR,
    METHOD_FAMILIES,
    SEED_POPULATIONS,
    MainStudyConfig,
    compute_configuration_hash,
    configuration_hash_prefix,
    make_main_study_config,
    make_run_id,
)
from experiments.main_study.workloads import (
    PlannedRun,
    build_corrupted_prior_specs_for_seed,
    enumerate_main_study_configs,
    enumerate_planned_runs,
    expected_artefact_names_for_method,
    make_planned_run,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_PARENT_HASH = "a" * 64
_RUN_HASH12 = "0123456789ab"
_N_NODES = 10
_EXPECTED_EDGES = 20


@pytest.fixture(scope="module")
def base_dagma_config() -> DAGMAConfig:
    return DAGMAConfig()


def _prior_free_config(seed: int = 401) -> MainStudyConfig:
    return MainStudyConfig(
        method_family="prior_free",
        seed_value=seed,
        seed_population="main_calibration",
        dagma_config=DAGMAConfig(),
        parent_heldout_run_hash_full=_PARENT_HASH,
    )


# ---------------------------------------------------------------------------
# T-1: expected_artefact_names_for_method
# ---------------------------------------------------------------------------


def test_prior_free_artefact_names_are_base_success_set():
    arts = expected_artefact_names_for_method("prior_free")
    assert set(arts) == {
        "continuous_w.npz",
        "thresholded_adjacency.npz",
        "true_adjacency.npz",
        "interventions_mmd.json",
    }


def test_matched_l1_artefact_names_match_prior_free():
    assert expected_artefact_names_for_method(
        "matched_l1"
    ) == expected_artefact_names_for_method("prior_free")


def test_soft_frobenius_artefact_names_include_confidence_and_prior_files():
    arts = set(expected_artefact_names_for_method("soft_frobenius"))
    base = set(expected_artefact_names_for_method("prior_free"))
    assert base.issubset(arts)
    assert "confidence_mask.npz" in arts
    assert "prior_edge_set_clean.json" in arts
    assert "prior_edge_set_corrupted.json" in arts
    assert "per_edge_labels.json" in arts


def test_hard_exclusion_artefact_names_have_prior_files_but_no_confidence():
    arts = set(expected_artefact_names_for_method("hard_exclusion"))
    base = set(expected_artefact_names_for_method("prior_free"))
    assert base.issubset(arts)
    assert "confidence_mask.npz" not in arts
    assert "prior_edge_set_clean.json" in arts
    assert "prior_edge_set_corrupted.json" in arts
    assert "per_edge_labels.json" in arts


def test_unknown_method_family_raises():
    with pytest.raises(ValueError, match="unknown method_family"):
        expected_artefact_names_for_method("nonexistent")


def test_artefact_names_are_subset_of_paths_module_artefact_names():
    """Every artefact name we plan for must be a known artefact in paths."""
    for mf in METHOD_FAMILIES:
        for name in expected_artefact_names_for_method(mf):
            assert name in ARTEFACT_NAMES


# ---------------------------------------------------------------------------
# T-2: make_planned_run
# ---------------------------------------------------------------------------


def test_make_planned_run_record_path_uses_records_dir_and_filename():
    cfg = _prior_free_config(seed=401)
    planned = make_planned_run(cfg, _RUN_HASH12)
    expected_record = (
        f"{records_dir(_RUN_HASH12)}/{record_filename(planned.run_id)}"
    )
    assert planned.record_path == expected_record


def test_make_planned_run_artefact_paths_use_artefact_path_helper():
    cfg = _prior_free_config(seed=401)
    planned = make_planned_run(cfg, _RUN_HASH12)
    for name, path in planned.artefact_paths.items():
        assert path == artefact_path(_RUN_HASH12, planned.run_id, name)


def test_make_planned_run_hash_prefix_runid_match_schema_helpers():
    cfg = _prior_free_config(seed=402)
    planned = make_planned_run(cfg, _RUN_HASH12)
    assert planned.configuration_hash_full == compute_configuration_hash(cfg)
    assert planned.configuration_hash_prefix == configuration_hash_prefix(cfg)
    assert planned.run_id == make_run_id(cfg)


def test_make_planned_run_no_trailing_slashes():
    cfg = _prior_free_config(seed=401)
    planned = make_planned_run(cfg, _RUN_HASH12)
    assert not planned.record_path.endswith("/")
    for path in planned.artefact_paths.values():
        assert not path.endswith("/")


def test_make_planned_run_paths_pass_relative_posix_validation():
    cfg = _prior_free_config(seed=401)
    planned = make_planned_run(cfg, _RUN_HASH12)
    assert validate_relative_posix_path(planned.record_path) == planned.record_path
    for path in planned.artefact_paths.values():
        assert validate_relative_posix_path(path) == path


def test_planned_run_rejects_wrong_artefact_set():
    cfg = _prior_free_config(seed=401)
    full = compute_configuration_hash(cfg)
    prefix = configuration_hash_prefix(cfg)
    rid = make_run_id(cfg)
    record_path = f"{records_dir(_RUN_HASH12)}/{record_filename(rid)}"
    # Drop one required artefact name.
    incomplete = {
        "continuous_w.npz": artefact_path(_RUN_HASH12, rid, "continuous_w.npz"),
    }
    with pytest.raises(ValueError, match="missing artefact"):
        PlannedRun(
            config=cfg,
            configuration_hash_full=full,
            configuration_hash_prefix=prefix,
            run_id=rid,
            record_path=record_path,
            artefact_paths=incomplete,
        )


def test_planned_run_rejects_unknown_artefact_name():
    cfg = _prior_free_config(seed=401)
    full = compute_configuration_hash(cfg)
    prefix = configuration_hash_prefix(cfg)
    rid = make_run_id(cfg)
    record_path = f"{records_dir(_RUN_HASH12)}/{record_filename(rid)}"
    bad_paths = {
        name: artefact_path(_RUN_HASH12, rid, name)
        for name in expected_artefact_names_for_method("prior_free")
    }
    bad_paths["confidence_mask.npz"] = (
        f"results/main_study/{_RUN_HASH12}/artefacts/{rid}/confidence_mask.npz"
    )
    with pytest.raises(ValueError, match="unknown artefact"):
        PlannedRun(
            config=cfg,
            configuration_hash_full=full,
            configuration_hash_prefix=prefix,
            run_id=rid,
            record_path=record_path,
            artefact_paths=bad_paths,
        )


def test_planned_run_rejects_bad_hash():
    cfg = _prior_free_config(seed=401)
    prefix = configuration_hash_prefix(cfg)
    rid = make_run_id(cfg)
    record_path = f"{records_dir(_RUN_HASH12)}/{record_filename(rid)}"
    paths = {
        name: artefact_path(_RUN_HASH12, rid, name)
        for name in expected_artefact_names_for_method("prior_free")
    }
    with pytest.raises(ValueError, match="configuration_hash_full"):
        PlannedRun(
            config=cfg,
            configuration_hash_full="f" * 64,
            configuration_hash_prefix=prefix,
            run_id=rid,
            record_path=record_path,
            artefact_paths=paths,
        )


# ---------------------------------------------------------------------------
# T-3: prior_free enumeration
# ---------------------------------------------------------------------------


def test_prior_free_enumeration_one_config_per_seed(base_dagma_config):
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("prior_free",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
    )
    assert len(configs) == len(CALIBRATION_SEEDS)
    for cfg in configs:
        assert cfg.method_family == "prior_free"
        assert cfg.confidence is None
        assert cfg.corrupted_prior_spec is None
        assert cfg.lambda_prior is None
        assert cfg.matched_l1_lambda1 is None
        assert cfg.dagma_config.exclude_edges is None


# ---------------------------------------------------------------------------
# T-4: matched_l1 enumeration
# ---------------------------------------------------------------------------


def test_matched_l1_enumeration_one_config_per_seed_with_replaced_lambda(
    base_dagma_config,
):
    matched_lambda = 0.07
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("matched_l1",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        matched_l1_lambda1=matched_lambda,
    )
    assert len(configs) == len(CALIBRATION_SEEDS)
    for cfg in configs:
        assert cfg.method_family == "matched_l1"
        assert cfg.matched_l1_lambda1 == pytest.approx(matched_lambda)
        assert cfg.dagma_config.lambda1 == pytest.approx(matched_lambda)
        assert cfg.dagma_config.exclude_edges is None
        assert cfg.confidence is None
        assert cfg.corrupted_prior_spec is None
        assert cfg.lambda_prior is None


# ---------------------------------------------------------------------------
# T-5: soft_frobenius counts and invariants
# ---------------------------------------------------------------------------


def test_soft_frobenius_counts_and_invariants(base_dagma_config):
    confidences = (0.0, 0.5)
    corruptions = (0.0, 0.4)
    seed_values = (401, 402)
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=seed_values,
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("soft_frobenius",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        confidence_grid=confidences,
        corruption_grid=corruptions,
    )
    assert len(configs) == 2 * 2 * 2  # 8

    for cfg in configs:
        assert cfg.method_family == "soft_frobenius"
        assert cfg.lambda_prior == pytest.approx(FROZEN_LAMBDA_PRIOR)
        assert cfg.confidence in confidences
        assert isinstance(cfg.corrupted_prior_spec, CorruptedPriorSpec)
        assert cfg.corrupted_prior_spec.corruption_fraction in corruptions
        assert cfg.dagma_config.exclude_edges is None


# ---------------------------------------------------------------------------
# T-6: hard_exclusion counts and invariants
# ---------------------------------------------------------------------------


def test_hard_exclusion_counts_and_invariants(base_dagma_config):
    corruptions = (0.0, 0.4)
    seed_values = (401, 402)
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=seed_values,
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("hard_exclusion",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        corruption_grid=corruptions,
    )
    assert len(configs) == 2 * 2  # 4

    for cfg in configs:
        assert cfg.method_family == "hard_exclusion"
        assert isinstance(cfg.corrupted_prior_spec, CorruptedPriorSpec)
        assert cfg.confidence is None
        assert cfg.lambda_prior is None
        # exclude_edges equals corrupted_prior_spec.forbidden_edges
        # under sorted-tuple equality (enforced by MainStudyConfig).
        excl_sorted = tuple(sorted(cfg.dagma_config.exclude_edges))
        forb_sorted = tuple(
            sorted(cfg.corrupted_prior_spec.forbidden_edges)
        )
        assert excl_sorted == forb_sorted


# ---------------------------------------------------------------------------
# T-7: no fake axes for prior_free / matched_l1
# ---------------------------------------------------------------------------


def test_prior_free_and_matched_l1_never_carry_confidence_or_corruption(
    base_dagma_config,
):
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("prior_free", "matched_l1"),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        matched_l1_lambda1=0.07,
    )
    for cfg in configs:
        assert cfg.confidence is None, (
            f"{cfg.method_family} must not carry confidence; got "
            f"{cfg.confidence!r}"
        )
        assert cfg.corrupted_prior_spec is None, (
            f"{cfg.method_family} must not carry corrupted_prior_spec"
        )


# ---------------------------------------------------------------------------
# T-8: deterministic ordering
# ---------------------------------------------------------------------------


def test_enumeration_is_deterministic(base_dagma_config):
    kwargs = dict(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("prior_free", "soft_frobenius", "hard_exclusion"),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        confidence_grid=(0.0, 0.5),
        corruption_grid=(0.0, 0.4),
    )
    a = enumerate_main_study_configs(**kwargs)
    b = enumerate_main_study_configs(**kwargs)
    hashes_a = tuple(compute_configuration_hash(c) for c in a)
    hashes_b = tuple(compute_configuration_hash(c) for c in b)
    assert hashes_a == hashes_b


def test_enumeration_order_follows_seed_method_corruption_confidence(
    base_dagma_config,
):
    """Order: seed ascending, then method order, then corruption asc,
    then confidence asc within soft_frobenius."""
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=(402, 401),  # caller order reversed; helper sorts asc
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("prior_free", "soft_frobenius"),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        confidence_grid=(0.5, 0.0),
        corruption_grid=(0.4, 0.0),
    )
    # Per seed: 1 prior_free + 2 corruption x 2 confidence = 5. Two seeds = 10.
    assert len(configs) == 10
    seeds_in_order = [cfg.seed_value for cfg in configs]
    # seeds ascending, each repeated 5 times.
    assert seeds_in_order == [401] * 5 + [402] * 5

    # Within seed 401: prior_free, then 4 soft_frobenius runs.
    first_block = configs[:5]
    assert first_block[0].method_family == "prior_free"
    for cfg in first_block[1:]:
        assert cfg.method_family == "soft_frobenius"

    # Soft_frobenius inner order: corruption asc, then confidence asc.
    soft_block = first_block[1:]
    cf_pairs = [
        (
            cfg.corrupted_prior_spec.corruption_fraction,
            cfg.confidence,
        )
        for cfg in soft_block
    ]
    assert cf_pairs == [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.4, 0.0),
        (0.4, 0.5),
    ]


# ---------------------------------------------------------------------------
# T-9: uniqueness across planned runs
# ---------------------------------------------------------------------------


def test_enumerate_planned_runs_no_duplicates(base_dagma_config):
    planned = enumerate_planned_runs(
        main_study_run_hash12=_RUN_HASH12,
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=(
            "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
        ),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        matched_l1_lambda1=0.07,
        confidence_grid=(0.0, 0.5),
        corruption_grid=(0.0, 0.4),
    )
    hashes = [p.configuration_hash_full for p in planned]
    assert len(set(hashes)) == len(hashes)
    run_ids = [p.run_id for p in planned]
    assert len(set(run_ids)) == len(run_ids)
    record_paths = [p.record_path for p in planned]
    assert len(set(record_paths)) == len(record_paths)


# ---------------------------------------------------------------------------
# T-10: full headline plan count
# ---------------------------------------------------------------------------


def test_full_evaluation_plan_count_is_224(base_dagma_config):
    configs = enumerate_main_study_configs(
        seed_population="main_evaluation",
        seed_values=tuple(EVALUATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=(
            "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
        ),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        matched_l1_lambda1=0.07,
    )
    assert len(configs) == 224
    counts: dict[str, int] = {mf: 0 for mf in METHOD_FAMILIES}
    for cfg in configs:
        counts[cfg.method_family] += 1
    assert counts["prior_free"] == 7
    assert counts["matched_l1"] == 7
    assert counts["soft_frobenius"] == 7 * 5 * 5
    assert counts["hard_exclusion"] == 7 * 5


# ---------------------------------------------------------------------------
# T-11: matched_l1 absent value
# ---------------------------------------------------------------------------


def test_matched_l1_without_lambda_raises(base_dagma_config):
    with pytest.raises(ValueError, match="matched_l1_lambda1"):
        enumerate_main_study_configs(
            seed_population="main_calibration",
            seed_values=tuple(CALIBRATION_SEEDS),
            base_dagma_config=base_dagma_config,
            parent_heldout_run_hash_full=_PARENT_HASH,
            method_families=("matched_l1",),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
        )


def test_omitting_matched_l1_works_without_lambda(base_dagma_config):
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=("prior_free",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
    )
    assert all(cfg.method_family == "prior_free" for cfg in configs)


# ---------------------------------------------------------------------------
# T-12: corrupted prior determinism
# ---------------------------------------------------------------------------


def test_build_corrupted_prior_specs_deterministic():
    a = build_corrupted_prior_specs_for_seed(
        seed_value=401, n_nodes=_N_NODES, expected_edges=_EXPECTED_EDGES,
    )
    b = build_corrupted_prior_specs_for_seed(
        seed_value=401, n_nodes=_N_NODES, expected_edges=_EXPECTED_EDGES,
    )
    assert a == b


def test_corrupted_prior_specs_match_supplied_grid():
    specs = build_corrupted_prior_specs_for_seed(
        seed_value=401, n_nodes=_N_NODES, expected_edges=_EXPECTED_EDGES,
        corruption_grid=(0.0, 0.4, 0.8),
    )
    fractions = [s.corruption_fraction for s in specs]
    assert fractions == [0.0, 0.4, 0.8]


def test_n_corrupted_matches_round_fraction_times_prior_k():
    specs = build_corrupted_prior_specs_for_seed(
        seed_value=401, n_nodes=_N_NODES, expected_edges=_EXPECTED_EDGES,
    )
    for spec in specs:
        assert spec.n_corrupted == int(round(spec.corruption_fraction * PRIOR_K))


# ---------------------------------------------------------------------------
# T-13: seed boundary
# ---------------------------------------------------------------------------


def test_main_evaluation_rejects_calibration_seed(base_dagma_config):
    with pytest.raises(ValueError, match="main_evaluation"):
        enumerate_main_study_configs(
            seed_population="main_evaluation",
            seed_values=(401,),
            base_dagma_config=base_dagma_config,
            parent_heldout_run_hash_full=_PARENT_HASH,
            method_families=("prior_free",),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
        )


def test_main_calibration_rejects_evaluation_seed(base_dagma_config):
    with pytest.raises(ValueError, match="main_calibration"):
        enumerate_main_study_configs(
            seed_population="main_calibration",
            seed_values=(501,),
            base_dagma_config=base_dagma_config,
            parent_heldout_run_hash_full=_PARENT_HASH,
            method_families=("prior_free",),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
        )


# ---------------------------------------------------------------------------
# T-14: base config contamination
# ---------------------------------------------------------------------------


def test_base_config_with_exclude_edges_raises():
    bad_base = DAGMAConfig(exclude_edges=((0, 1),))
    with pytest.raises(ValueError, match="exclude_edges"):
        enumerate_main_study_configs(
            seed_population="main_calibration",
            seed_values=tuple(CALIBRATION_SEEDS),
            base_dagma_config=bad_base,
            parent_heldout_run_hash_full=_PARENT_HASH,
            method_families=("prior_free",),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
        )


# ---------------------------------------------------------------------------
# T-15: path validity for every PlannedRun
# ---------------------------------------------------------------------------


def test_every_planned_run_path_is_valid_relative_posix(base_dagma_config):
    planned = enumerate_planned_runs(
        main_study_run_hash12=_RUN_HASH12,
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=_PARENT_HASH,
        method_families=(
            "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
        ),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
        matched_l1_lambda1=0.07,
        confidence_grid=(0.0, 0.5),
        corruption_grid=(0.0, 0.4),
    )
    for plan in planned:
        # validate_relative_posix_path raises if any rule is violated.
        validate_relative_posix_path(plan.record_path)
        assert not plan.record_path.endswith("/")
        for path in plan.artefact_paths.values():
            validate_relative_posix_path(path)
            assert not path.endswith("/")
            parts = path.split("/")
            assert "." not in parts
            assert ".." not in parts
            assert "\\" not in path


# ---------------------------------------------------------------------------
# T-16: parent hash
# ---------------------------------------------------------------------------


def test_short_parent_hash_raises(base_dagma_config):
    with pytest.raises(ValueError, match="parent_heldout_run_hash_full"):
        enumerate_main_study_configs(
            seed_population="main_calibration",
            seed_values=tuple(CALIBRATION_SEEDS),
            base_dagma_config=base_dagma_config,
            parent_heldout_run_hash_full="0123456789ab",
            method_families=("prior_free",),
            n_nodes=_N_NODES,
            expected_edges=_EXPECTED_EDGES,
        )


def test_full_parent_hash_accepted(base_dagma_config):
    configs = enumerate_main_study_configs(
        seed_population="main_calibration",
        seed_values=tuple(CALIBRATION_SEEDS),
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full="0123456789abcdef" * 4,
        method_families=("prior_free",),
        n_nodes=_N_NODES,
        expected_edges=_EXPECTED_EDGES,
    )
    assert len(configs) == len(CALIBRATION_SEEDS)


# ---------------------------------------------------------------------------
# T-17: import allowlist
# ---------------------------------------------------------------------------


_WORKLOADS_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers.dcdi",
    "symbolic_priors_cd.wrappers._dcdi",
    "experiments.selection_study",
    "experiments.main_study.records",
    "experiments.main_study.calibration_lambda_prior",
    "dagma",
    "dcdi",
    "tests",
)


_WORKLOADS_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "dataclasses",
    "typing",
    "collections",
    "experiments.main_study.schema",
    "experiments.main_study.paths",
    "experiments.main_study.priors",
    "symbolic_priors_cd.wrappers.dagma",
    "symbolic_priors_cd.data.scm_generator",
})


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_workloads_module_does_not_import_forbidden_packages():
    src = Path(workloads_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _WORKLOADS_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"workloads.py must not import {mod!r}; forbidden "
                f"prefix {forbidden!r}."
            )


def test_workloads_module_imports_are_allowlisted():
    src = Path(workloads_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _WORKLOADS_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _WORKLOADS_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"workloads.py import {mod!r} is not in the allowlist "
            f"{sorted(_WORKLOADS_ALLOWED_PREFIXES)}."
        )
