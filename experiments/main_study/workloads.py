"""Dry-run workload enumeration for the main-study pipeline.

Provides the planning side of the runner: enumerate which
configurations will be executed and where each run's record and
artefacts will live on disk. Side-effect free: no file or directory
is created, no model is fitted, no metric is computed.

Edge representation follows the project's row-source /
column-destination convention.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from experiments.main_study.paths import (
    artefact_path,
    record_filename,
    records_dir,
    validate_relative_posix_path,
)
from experiments.main_study.priors import (
    CORRUPTION_GRID,
    PRIOR_K,
    CorruptedPriorSpec,
    PriorSpec,
    corrupt_prior,
    generate_prior_for_scm_seed,
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
from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
)
from symbolic_priors_cd.wrappers.dagma import DAGMAConfig


# ---------------------------------------------------------------------------
# Artefact-name tables
# ---------------------------------------------------------------------------


_BASE_SUCCESS_ARTEFACTS: tuple[str, ...] = (
    "continuous_w.npz",
    "thresholded_adjacency.npz",
    "true_adjacency.npz",
    "interventions_mmd.json",
)


_PRIOR_BACKED_ARTEFACTS: tuple[str, ...] = (
    "prior_edge_set_clean.json",
    "prior_edge_set_corrupted.json",
    "per_edge_labels.json",
)


_CONFIDENCE_MASK_ARTEFACT: str = "confidence_mask.npz"


def expected_artefact_names_for_method(method_family: str) -> tuple[str, ...]:
    """Return the artefact filenames a metric-computed success run must
    write for ``method_family``.

    The returned tuple is the set of keys that :class:`PlannedRun`'s
    ``artefact_paths`` mapping must contain. Method families that do
    not produce a prior or a confidence mask do not list those names.

    Unknown ``method_family`` raises ``ValueError``.
    """
    if method_family == "prior_free":
        return _BASE_SUCCESS_ARTEFACTS
    if method_family == "matched_l1":
        return _BASE_SUCCESS_ARTEFACTS
    if method_family == "soft_frobenius":
        return (
            _BASE_SUCCESS_ARTEFACTS
            + (_CONFIDENCE_MASK_ARTEFACT,)
            + _PRIOR_BACKED_ARTEFACTS
        )
    if method_family == "hard_exclusion":
        return _BASE_SUCCESS_ARTEFACTS + _PRIOR_BACKED_ARTEFACTS
    raise ValueError(
        f"unknown method_family {method_family!r}; expected one of "
        f"{METHOD_FAMILIES}."
    )


# ---------------------------------------------------------------------------
# PlannedRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class PlannedRun:
    """A single planned run: the executable configuration plus where
    its on-disk record and artefacts will go.

    ``record_path`` and every value in ``artefact_paths`` is a
    relative POSIX path under ``results/main_study/<prefix>/`` and is
    validated via :func:`validate_relative_posix_path` at
    construction. ``artefact_paths`` contains exactly the artefact
    names returned by :func:`expected_artefact_names_for_method` for
    ``config.method_family``.
    """

    config: MainStudyConfig
    configuration_hash_full: str
    configuration_hash_prefix: str
    run_id: str
    record_path: str
    artefact_paths: dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.config, MainStudyConfig):
            raise TypeError(
                "PlannedRun.config must be a MainStudyConfig; got "
                f"{type(self.config).__name__}."
            )
        expected_full = compute_configuration_hash(self.config)
        if self.configuration_hash_full != expected_full:
            raise ValueError(
                "PlannedRun.configuration_hash_full does not match "
                "compute_configuration_hash(config). got "
                f"{self.configuration_hash_full!r}, expected "
                f"{expected_full!r}."
            )
        if self.configuration_hash_prefix != self.configuration_hash_full[:12]:
            raise ValueError(
                "PlannedRun.configuration_hash_prefix must equal the "
                "first 12 characters of configuration_hash_full. got "
                f"prefix {self.configuration_hash_prefix!r}, full "
                f"{self.configuration_hash_full!r}."
            )
        expected_run_id = make_run_id(self.config)
        if self.run_id != expected_run_id:
            raise ValueError(
                "PlannedRun.run_id does not match make_run_id(config). "
                f"got {self.run_id!r}, expected {expected_run_id!r}."
            )

        # Path validation: each path must be a relative POSIX path
        # with no trailing slash. validate_relative_posix_path already
        # rejects trailing slashes; the explicit endswith check is a
        # defensive guard.
        validate_relative_posix_path(self.record_path)
        if self.record_path.endswith("/"):
            raise ValueError(
                f"PlannedRun.record_path must not end with '/'; got "
                f"{self.record_path!r}."
            )

        if not isinstance(self.artefact_paths, dict):
            raise TypeError(
                "PlannedRun.artefact_paths must be a dict; got "
                f"{type(self.artefact_paths).__name__}."
            )
        expected_names = set(
            expected_artefact_names_for_method(self.config.method_family)
        )
        actual_names = set(self.artefact_paths.keys())
        unknown = actual_names - expected_names
        missing = expected_names - actual_names
        if unknown:
            raise ValueError(
                f"PlannedRun.artefact_paths contains unknown artefact "
                f"name(s) {sorted(unknown)} for method_family="
                f"{self.config.method_family!r}; allowed: "
                f"{sorted(expected_names)}."
            )
        if missing:
            raise ValueError(
                f"PlannedRun.artefact_paths is missing artefact name(s) "
                f"{sorted(missing)} for method_family="
                f"{self.config.method_family!r}."
            )
        for name, path in self.artefact_paths.items():
            if not isinstance(path, str):
                raise TypeError(
                    f"PlannedRun.artefact_paths[{name!r}] must be a "
                    f"string; got {type(path).__name__}."
                )
            validate_relative_posix_path(path)
            if path.endswith("/"):
                raise ValueError(
                    f"PlannedRun.artefact_paths[{name!r}] must not end "
                    f"with '/'; got {path!r}."
                )


def make_planned_run(
    config: MainStudyConfig, main_study_run_hash12: str
) -> PlannedRun:
    """Wrap ``config`` into a :class:`PlannedRun` with computed paths.

    The record path goes under
    ``results/main_study/<main_study_run_hash12>/records/``; per-
    artefact paths go under
    ``results/main_study/<main_study_run_hash12>/artefacts/<run_id>/``.
    No directory is created and no file is written.
    """
    if not isinstance(config, MainStudyConfig):
        raise TypeError(
            "make_planned_run requires a MainStudyConfig; got "
            f"{type(config).__name__}."
        )
    full_hash = compute_configuration_hash(config)
    prefix = configuration_hash_prefix(config)
    run_id = make_run_id(config)
    record_path = (
        f"{records_dir(main_study_run_hash12)}/"
        f"{record_filename(run_id)}"
    )
    artefact_names = expected_artefact_names_for_method(config.method_family)
    artefact_paths = {
        name: artefact_path(main_study_run_hash12, run_id, name)
        for name in artefact_names
    }
    return PlannedRun(
        config=config,
        configuration_hash_full=full_hash,
        configuration_hash_prefix=prefix,
        run_id=run_id,
        record_path=record_path,
        artefact_paths=artefact_paths,
    )


# ---------------------------------------------------------------------------
# Corrupted-prior spec construction (shared by soft_frobenius and hard_exclusion)
# ---------------------------------------------------------------------------


def build_corrupted_prior_specs_for_seed(
    *,
    seed_value: int,
    n_nodes: int,
    expected_edges: int,
    prior_k: int = PRIOR_K,
    corruption_grid: tuple[float, ...] = CORRUPTION_GRID,
) -> tuple[CorruptedPriorSpec, ...]:
    """Build one :class:`CorruptedPriorSpec` per ``corruption_grid`` entry.

    The clean forbidden-edge prior is generated by the canonical
    :func:`generate_prior_for_scm_seed` helper (which internally uses
    the project SCM utility). The true adjacency needed by
    :func:`corrupt_prior` is obtained by calling the same SCM utility
    a second time with the same integer seed, which produces a
    bit-identical SCM. No observational training data is sampled.

    The returned tuple is sorted by ``corruption_fraction`` ascending
    and is deterministic for identical inputs.
    """
    clean_prior = generate_prior_for_scm_seed(
        scm_seed=int(seed_value),
        n_nodes=int(n_nodes),
        expected_edges=int(expected_edges),
        prior_k=int(prior_k),
    )
    scm = generate_linear_gaussian_scm(
        n_nodes=int(n_nodes),
        expected_edges=int(expected_edges),
        seed=int(seed_value),
        noise_scale=1.0,
    )
    true_adjacency = scm.adjacency
    specs = [
        corrupt_prior(clean_prior, true_adjacency, float(fraction))
        for fraction in corruption_grid
    ]
    specs.sort(key=lambda s: s.corruption_fraction)
    return tuple(specs)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def _validate_caller_inputs(
    *,
    seed_population: str,
    method_families: tuple[str, ...],
    base_dagma_config: DAGMAConfig,
    matched_l1_lambda1: Optional[float],
) -> None:
    if seed_population not in SEED_POPULATIONS:
        raise ValueError(
            f"seed_population must be one of {SEED_POPULATIONS}; got "
            f"{seed_population!r}."
        )
    if not isinstance(method_families, tuple) or not method_families:
        raise ValueError(
            "method_families must be a non-empty tuple; got "
            f"{method_families!r}."
        )
    seen: set[str] = set()
    for mf in method_families:
        if mf not in METHOD_FAMILIES:
            raise ValueError(
                f"unknown method_family {mf!r}; expected one of "
                f"{METHOD_FAMILIES}."
            )
        if mf in seen:
            raise ValueError(
                f"method_families contains duplicate entry {mf!r}."
            )
        seen.add(mf)
    if not isinstance(base_dagma_config, DAGMAConfig):
        raise TypeError(
            "base_dagma_config must be a DAGMAConfig; got "
            f"{type(base_dagma_config).__name__}."
        )
    if base_dagma_config.exclude_edges is not None:
        raise ValueError(
            "base_dagma_config.exclude_edges must be None; "
            "hard_exclusion configs set exclude_edges via "
            "make_main_study_config. got "
            f"{base_dagma_config.exclude_edges!r}."
        )
    if "matched_l1" in method_families:
        if matched_l1_lambda1 is None:
            raise ValueError(
                "matched_l1 was requested but matched_l1_lambda1 is "
                "None; supply the calibrated value."
            )
        if isinstance(matched_l1_lambda1, bool) or not isinstance(
            matched_l1_lambda1, (int, float)
        ):
            raise ValueError(
                "matched_l1_lambda1 must be a positive finite number; "
                f"got {matched_l1_lambda1!r}."
            )
        if float(matched_l1_lambda1) <= 0.0:
            raise ValueError(
                "matched_l1_lambda1 must be > 0; got "
                f"{matched_l1_lambda1!r}."
            )


def enumerate_main_study_configs(
    *,
    seed_population: str,
    seed_values: tuple[int, ...],
    base_dagma_config: DAGMAConfig,
    parent_heldout_run_hash_full: str,
    method_families: tuple[str, ...],
    n_nodes: int,
    expected_edges: int,
    matched_l1_lambda1: Optional[float] = None,
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID,
    corruption_grid: tuple[float, ...] = CORRUPTION_GRID,
) -> tuple[MainStudyConfig, ...]:
    """Enumerate the full :class:`MainStudyConfig` plan for one population.

    Ordering is fully deterministic: seed ascending, then method
    family in caller-provided order, then (for grid-bearing families)
    corruption ascending, then confidence ascending. Within
    ``prior_free`` and ``matched_l1`` exactly one config per seed is
    produced; no fake confidence or corruption axes are added.
    """
    _validate_caller_inputs(
        seed_population=seed_population,
        method_families=method_families,
        base_dagma_config=base_dagma_config,
        matched_l1_lambda1=matched_l1_lambda1,
    )

    sorted_seeds = sorted(int(s) for s in seed_values)
    sorted_confidences = sorted(float(c) for c in confidence_grid)
    sorted_corruptions = sorted(float(f) for f in corruption_grid)

    needs_corrupted = any(
        mf in ("soft_frobenius", "hard_exclusion")
        for mf in method_families
    )

    configs: list[MainStudyConfig] = []

    for seed in sorted_seeds:
        if needs_corrupted:
            corrupted_specs = build_corrupted_prior_specs_for_seed(
                seed_value=seed,
                n_nodes=n_nodes,
                expected_edges=expected_edges,
                corruption_grid=tuple(sorted_corruptions),
            )
            # The helper returns specs in ascending corruption order
            # already, but resort defensively so the iteration order
            # below matches the public contract regardless of helper
            # behaviour.
            sorted_specs = sorted(
                corrupted_specs, key=lambda s: s.corruption_fraction
            )
        else:
            sorted_specs = []

        for mf in method_families:
            if mf == "prior_free":
                cfg = make_main_study_config(
                    method_family="prior_free",
                    seed_value=seed,
                    seed_population=seed_population,
                    dagma_config=base_dagma_config,
                    parent_heldout_run_hash_full=parent_heldout_run_hash_full,
                )
                configs.append(cfg)
            elif mf == "matched_l1":
                replacement_dagma = dataclasses.replace(
                    base_dagma_config,
                    lambda1=float(matched_l1_lambda1),
                )
                cfg = make_main_study_config(
                    method_family="matched_l1",
                    seed_value=seed,
                    seed_population=seed_population,
                    dagma_config=replacement_dagma,
                    parent_heldout_run_hash_full=parent_heldout_run_hash_full,
                    matched_l1_lambda1=float(matched_l1_lambda1),
                )
                configs.append(cfg)
            elif mf == "soft_frobenius":
                for spec in sorted_specs:
                    for conf in sorted_confidences:
                        cfg = make_main_study_config(
                            method_family="soft_frobenius",
                            seed_value=seed,
                            seed_population=seed_population,
                            dagma_config=base_dagma_config,
                            parent_heldout_run_hash_full=parent_heldout_run_hash_full,
                            confidence=float(conf),
                            corrupted_prior_spec=spec,
                        )
                        configs.append(cfg)
            elif mf == "hard_exclusion":
                for spec in sorted_specs:
                    cfg = make_main_study_config(
                        method_family="hard_exclusion",
                        seed_value=seed,
                        seed_population=seed_population,
                        dagma_config=base_dagma_config,
                        parent_heldout_run_hash_full=parent_heldout_run_hash_full,
                        corrupted_prior_spec=spec,
                    )
                    configs.append(cfg)

    return tuple(configs)


def enumerate_planned_runs(
    *,
    main_study_run_hash12: str,
    seed_population: str,
    seed_values: tuple[int, ...],
    base_dagma_config: DAGMAConfig,
    parent_heldout_run_hash_full: str,
    method_families: tuple[str, ...],
    n_nodes: int,
    expected_edges: int,
    matched_l1_lambda1: Optional[float] = None,
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID,
    corruption_grid: tuple[float, ...] = CORRUPTION_GRID,
) -> tuple[PlannedRun, ...]:
    """Enumerate planned runs and verify cross-run path uniqueness.

    Wraps :func:`enumerate_main_study_configs` and converts each
    config to a :class:`PlannedRun` via :func:`make_planned_run`. The
    final result is asserted to contain no duplicate configuration
    hashes, run-ids, or record paths.
    """
    configs = enumerate_main_study_configs(
        seed_population=seed_population,
        seed_values=seed_values,
        base_dagma_config=base_dagma_config,
        parent_heldout_run_hash_full=parent_heldout_run_hash_full,
        method_families=method_families,
        n_nodes=n_nodes,
        expected_edges=expected_edges,
        matched_l1_lambda1=matched_l1_lambda1,
        confidence_grid=confidence_grid,
        corruption_grid=corruption_grid,
    )
    planned = tuple(
        make_planned_run(cfg, main_study_run_hash12) for cfg in configs
    )
    _assert_no_duplicate_paths(planned)
    return planned


def _assert_no_duplicate_paths(planned: tuple[PlannedRun, ...]) -> None:
    hashes = [p.configuration_hash_full for p in planned]
    if len(set(hashes)) != len(hashes):
        seen: dict[str, int] = {}
        for h in hashes:
            seen[h] = seen.get(h, 0) + 1
        dupes = sorted(h for h, count in seen.items() if count > 1)
        raise ValueError(
            f"duplicate configuration_hash_full in plan: {dupes}."
        )
    run_ids = [p.run_id for p in planned]
    if len(set(run_ids)) != len(run_ids):
        seen_ids: dict[str, int] = {}
        for r in run_ids:
            seen_ids[r] = seen_ids.get(r, 0) + 1
        dupes = sorted(r for r, count in seen_ids.items() if count > 1)
        raise ValueError(
            f"duplicate run_id in plan: {dupes}."
        )
    record_paths = [p.record_path for p in planned]
    if len(set(record_paths)) != len(record_paths):
        seen_paths: dict[str, int] = {}
        for rp in record_paths:
            seen_paths[rp] = seen_paths.get(rp, 0) + 1
        dupes = sorted(p for p, count in seen_paths.items() if count > 1)
        raise ValueError(
            f"duplicate record_path in plan: {dupes}."
        )


__all__ = [
    "PlannedRun",
    "expected_artefact_names_for_method",
    "make_planned_run",
    "build_corrupted_prior_specs_for_seed",
    "enumerate_main_study_configs",
    "enumerate_planned_runs",
]
