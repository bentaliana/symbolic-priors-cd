# 08_base_model_selection_plan

## Status

Planning artefact only. No runner code, test code, configuration
file, or external repository file is created by this document.
Implementation does not begin until this plan is reviewed and
Commit 1 is approved.

Version 1.0.

---

## Change log

- v1.0 (this document): initial commit-structured plan for the
  base-model selection-study runner. Freezes the runner
  architecture, the 14-commit sequence, the four named gate
  commits, the assignment of the Section 16.1 seed-discipline
  resolution to Commit 2, the calibration ranking rule, the
  compute budget, and the scope-cut hierarchy.

---

## 1. Authoritative inputs and current project state

### Inputs read for this plan

This plan is subordinate to the following documents in priority
order:

1. `docs/01_research_question_and_commitments.md` (frozen
   scientific contract).
2. `docs/02_base_model_selection.md` v1.4 (frozen selection-study
   protocol; v1.4 amended only the SID-deferred phrasing and
   explicitly left the Section 3.5 seed-discipline statement
   unchanged).
3. `docs/03_decision_log.md` (evaluator conventions, wrapper-phase
   conventions, 13/05/2026 DAGMA wrapper plan acceptance,
   15/05/2026 SID closure entry, 18/05/2026 schema-freeze entry).
4. `docs/04_wrapper_api_contract.md` (status taxonomy, no-silent-
   repair policy, sampler-status taxonomy).
5. `docs/08a_experiment_tracking_and_results_schema.md` v1
   (run-record schema, identity rule, directory layout, MMD
   reliability rule, loader interface, smoke-check requirement,
   Section 16 open issues).
6. `docs/05_dcdi_wrapper_implementation_plan.md` and
   `docs/06_dagma_wrapper_implementation_plan.md` (commit-
   structured planning pattern).
7. `docs/phase_2b_dcdi_readout.md`,
   `docs/phase_2c_dagma_readout.md`, and
   `docs/phase_2d_sid_readout.md` (current wrapper and metric
   state).

### Project state at the start of this plan

- Phase 1 evaluator foundation complete.
- DCDI wrapper Commits 1 through 9 implemented and green; DCDI
  Commit 10 sampler-quality validation did not pass and is
  recorded as the diagnostic artefact C-P11; DCDI Commit 11
  (loss-hook injection) paused.
- DAGMA wrapper Commits 1 through 10 implemented and green; C-P13
  diagnostic recorded as fixture-specific evidence; DAGMA loss-
  hook work deferred.
- SID implemented and verified via `gadjid==0.1.0`; the SID
  verification gate is closed.
- MMD and SHD have independent cross-check evidence (`docs/04j`).
- `docs/08a` v1 frozen as the run-record schema.
- One open issue carried by this plan: `docs/08a` Section 16.1
  records a seed-discipline conflict between
  `docs/02_base_model_selection.md` Section 3.5 and the
  `docs/03_decision_log.md` 13/05/2026 DAGMA wrapper acceptance
  entry. This plan assigns the resolution to Commit 2 and does
  not pre-pick.
- No selection-study runner, no soft-prior layer, no main-study
  runner exists yet.

---

## 2. Purpose and scope

### Purpose

Specify the base-model selection-study runner at the level of
detail required for a reviewable, commit-by-commit implementation.
After this plan is approved, implementation may proceed Commit 1
at a time, in the order documented in this document, with each
commit gated on its acceptance criterion before the next is
started. Four named gate commits constrain the order further.

### Scope of this plan

In scope:

- the runner that executes the protocol in
  `docs/02_base_model_selection.md`;
- the configuration schema, seed-derivation policy, identity rule,
  and directory layout the runner must honour;
- the preflight manifest, schema-conformance, calibration vs
  held-out evaluation non-overlap, and end-to-end smoke-check
  gates;
- the Phase A reproduction pass, Phase B calibration, and held-out
  evaluation phases;
- the threshold-robustness offline re-computation procedure;
- the selection-study report generator;
- the halt-and-resume mechanism that supports the scope-cut
  hierarchy;
- the resolution path for the `docs/08a` Section 16.1 seed-
  discipline conflict.

Out of scope of this plan:

- the wrapper layer beyond what the runner imports (the wrappers
  are already implemented per
  `docs/05_dcdi_wrapper_implementation_plan.md` and
  `docs/06_dagma_wrapper_implementation_plan.md`);
- soft-prior / loss-hook injection on either wrapper;
- DCDI Commit 11 resumption;
- hard-constraint baseline implementation;
- main-study runner architecture (introduced through a later
  `docs/08a` `schema_version` bump);
- W&B integration code (the runner is functional without W&B);
- the base-model decision itself (produced by the runner's
  report, not by this plan).

---

## 3. Runner architecture overview

### Placement

The runner lives under `experiments/`. No training logic is added
to `src/symbolic_priors_cd/` beyond the existing wrapper layer.
No training logic lives in `notebooks/`. The runner imports the
existing project metrics and wrapper APIs and orchestrates them;
it does not duplicate them.

### Entry point shape

A single CLI entry point is provided. The CLI accepts:

- `--help`: prints usage and exits.
- `--config <path>`: path to a configuration file consumed by the
  runner.
- `--dry-run`: runs preflight only (Commit 4); no fits are
  invoked.
- `--resume`: resumes a halted run from the existing
  `results/model_selection/` tree (Commit 11).

The exact argument-parser library and the resolved argument names
are delegated to Commit 1. No flag beyond those listed above is
introduced without a contemporaneous entry in
`docs/03_decision_log.md`.

### Logging discipline

The canonical record for any run is the per-run JSON file written
under the `docs/08a` directory layout. The runner additionally
emits a structured local log (file under the run directory or a
dedicated `logs/` sibling, decided by Commit 1) for human
inspection. Library code under `src/symbolic_priors_cd/` does not
contain `print` statements. The CLI may write to
stdout for progress reporting only; stdout is not a record of
results.

### Progress reporting

Human-visible progress reporting (for example via `tqdm`) is
permitted but is never load-bearing for results. If a progress
indicator and a JSON record disagree, the JSON record wins. The
report generator (Commit 12) reads only local files; it never
reads from a progress display or from W&B.

### Failure handling

NaN, divergence, and `wrapper_error` outcomes produce a complete
run record per `docs/08a`, classified through the
`docs/04_wrapper_api_contract.md` taxonomy. No silent omission.
No run deletion. No graph repair. Invalid graphs are classified
through `graph_status` and retained.

### W&B mirror

W&B is an optional mirror only, per `docs/08a` Section 12. The
runner is fully functional without W&B. W&B integration is not
implemented by this plan.

---

## 4. Module and file structure

### New paths to create

```
experiments/selection_study/__init__.py
experiments/selection_study/run.py
experiments/selection_study/config.py
experiments/selection_study/identity.py
experiments/selection_study/preflight.py
experiments/selection_study/pipeline.py
experiments/selection_study/sampling.py
experiments/selection_study/threshold_robustness.py
experiments/selection_study/reproduction_pass.py
experiments/selection_study/calibration.py
experiments/selection_study/held_out.py
experiments/selection_study/resume.py
experiments/selection_study/report.py
experiments/selection_study/loader.py
tests/test_selection_runner_config.py
tests/test_selection_runner_identity.py
tests/test_selection_runner_preflight.py
tests/test_selection_runner_pipeline.py
tests/test_selection_runner_sampling.py
tests/test_selection_runner_threshold_robustness.py
tests/test_reproduction_pass_runner.py
tests/test_calibration_runner.py
tests/test_selection_runner_held_out.py
tests/test_selection_runner_resume.py
tests/test_selection_runner_report.py
tests/test_selection_runner_end_to_end.py
```

The submodule name `experiments/selection_study/` is chosen for
clarity against the `results/model_selection/` output tree, which
already uses `model_selection` as its top-level subdirectory. The
two names are deliberately distinct: `experiments/selection_study/`
is the runner; `results/model_selection/` is the runner's output.

### Existing modules touched

- `src/symbolic_priors_cd/wrappers/__init__.py`: read-only;
  imported by the runner.
- `src/symbolic_priors_cd/metrics/__init__.py`: read-only;
  imported by the runner.
- `src/symbolic_priors_cd/data/`: read-only; imported by the
  runner for SCM generation and ground-truth interventional
  sampling.

No source file under `src/symbolic_priors_cd/` is modified by
this plan unless Commit 2's resolution of the `docs/08a` Section
16.1 conflict selects the option that requires it (see Section 7
of this document).

### What is not added

- No new dependency.
- No new evaluator metric.
- No notebook.
- No W&B integration code.
- No soft-prior or loss-hook code.

---

## 5. Commit sequence

Commits are conservative and reviewable. Each commit must pass
its acceptance criterion before the next is started. The four
gate commits (Commits 4, 5, 10, and 13) impose additional
discipline: no work on commits beyond a gate may proceed until
the gate's acceptance criteria are met. A failed gate is recorded
as a diagnostic artefact in `docs/04*` and `inspection/probes/`,
analogous to C-P11 in `docs/04f_dcdi_sampler_quality_diagnostic.md`.

### Commit 1: Runner scaffolding

Purpose: establish the module structure, the CLI entry point, and
the import surface, with no fit logic in place.

What the commit does:

- Creates `experiments/selection_study/` with `__init__.py` and
  `run.py`.
- Implements the CLI with `--help`, `--config`, `--dry-run`, and
  `--resume` flags. All flags are wired to no-op handlers in this
  commit; no fits are invoked from any flag.
- Adds stubbed modules `config.py`, `identity.py`, `preflight.py`,
  `pipeline.py`, `sampling.py`, `threshold_robustness.py`,
  `reproduction_pass.py`, `calibration.py`, `held_out.py`,
  `resume.py`, `report.py`, `loader.py`. Stubs raise
  `NotImplementedError` with explicit messages.

Acceptance criteria:

- The runner module imports cleanly under the project Python
  version.
- `python -m experiments.selection_study.run --help` returns a
  usage string.
- Neither a DAGMA nor a DCDI fit is reachable from any code path
  in this commit.
- No `print` statement in library code; stdout from the CLI is
  limited to the help text.

Files touched (allowed): the paths listed above under "New paths
to create" for this commit. Files forbidden: any file under
`src/symbolic_priors_cd/`, any test of metrics or wrappers, any
file under `docs/`.

Tests required: `test_selection_runner_help_exits_cleanly`,
`test_selection_runner_no_fit_reachable_from_help_or_dry_run_in_commit_1`,
`test_stub_modules_raise_notimplementederror`. Invariant:
"importing the runner does not trigger any model fit and does not
touch global RNG state".

Dependencies: none.

### Commit 2: Configuration schema and seed-derivation policy

Purpose: freeze the runner's configuration object, the canonical
JSON serialisation, the SHA-256 `configuration_hash` per
`docs/08a` Section 4, and the seed-derivation rule for the per-
purpose seeds declared in `docs/08a` Sections 6.1, 6.2, and 6.10.
This commit ALSO resolves the `docs/08a` Section 16.1 seed-
discipline conflict.

What the commit does:

- Implements `config.py`. The configuration object is a frozen,
  serialisable record covering: model (`dagma` or `dcdi`),
  condition (`centred_only` or `standardised`), seed populations
  and their seed lists, intervention set, hyperparameter grids
  for Phase B, threshold values for offline robustness, and
  references to the wrapper APIs.
- Implements the canonical JSON serialisation: `json.dumps(...,
  sort_keys=True, separators=(",", ":"))` with deterministic
  float repr.
- Implements the SHA-256 hash of the canonical JSON. The full
  64-character lowercase hex digest is stored; the first 12
  characters are exposed as the directory-path prefix.
- Implements the per-purpose seed-derivation rule: for each run,
  `graph_seed`, `train_data_seed`, `validation_data_seed` (when
  applicable), `intervention_ground_truth_seed_base`, and
  `model_sampling_seed_base` are derived deterministically from
  the run's identity tuple and the configuration. Per-
  intervention `ground_truth_sampling_seed` and
  `model_sampling_seed` are derived from the corresponding
  bases and the `intervention_id`. The derivation rule is
  recorded in the run record under
  `model_sampling_seed_derivation_rule`.
- Resolves `docs/08a` Section 16.1. Two options are eligible:
  - Option A: amend `docs/02_base_model_selection.md` to v1.5 to
    relax the uniform seed-setter requirement for candidates
    that are deterministic by construction. The DAGMA wrapper
    keeps its current behaviour of not calling
    `torch.manual_seed`, `np.random.seed`, or
    `dagma.utils.set_random_seed`. `seed_torch`,
    `seed_numpy`, and `seed_dagma` may be null in DAGMA runs.
  - Option B: amend the DAGMA wrapper and
    `docs/03_decision_log.md` to make the DAGMA fit call the
    listed seed setters at fit time. `seed_torch`,
    `seed_numpy`, and `seed_dagma` are non-null for every
    DAGMA run.
- Commit 2 picks one option, records the choice in a
  contemporaneous `docs/03_decision_log.md` entry citing this
  plan and `docs/08a` Section 16.1, and produces the
  corresponding documentation amendment (either
  `docs/02_base_model_selection.md` v1.5 in Option A, or the
  DAGMA wrapper source change plus a `docs/03_decision_log.md`
  entry in Option B) in the same commit.
- This plan does not pre-pick.

Acceptance criteria:

- The canonical JSON serialisation of a resolved configuration
  is byte-stable across two consecutive serialisations of the
  same in-memory object.
- The SHA-256 hash is identical across two consecutive
  serialisations of the same resolved configuration.
- Each per-purpose seed is deterministic given the run identity
  and the configuration; a second call with the same inputs
  returns the same seed.
- The `docs/08a` Section 16.1 conflict is closed: exactly one of
  Option A or Option B is in effect; the choice is traceable
  through `docs/03_decision_log.md`; and the affected document
  (either `docs/02_base_model_selection.md` v1.5 or the DAGMA
  wrapper source) reflects the resolution.
- No fit is invoked from this commit.

Files touched (allowed): `experiments/selection_study/config.py`,
`docs/03_decision_log.md` (new entry), and ONE of either
`docs/02_base_model_selection.md` (Option A) or
`src/symbolic_priors_cd/wrappers/dagma.py` and its tests
(Option B). Files forbidden: anything else.

Tests required:
`test_canonical_json_byte_stable`,
`test_configuration_hash_sha256_deterministic`,
`test_per_purpose_seeds_deterministic`,
`test_per_intervention_seeds_derived_from_bases_and_intervention_id`,
plus the test corresponding to the chosen option (a regression
test that DAGMA fit either does or does not call the listed seed
setters, matching the resolution). Invariant: "the seed
derivation rule is total, deterministic, and consistent with the
documentation chosen by Commit 2's resolution".

Dependencies: Commit 1.

### Commit 3: Run identity and directory creation

Purpose: implement the `docs/08a` Section 3 directory layout, the
`run_id` derivation per Section 4, the `configuration_hash_prefix`
path component, and the no-overwrite rule.

What the commit does:

- Implements `identity.py`. Provides functions to derive
  `run_id` from `(model, condition, seed_population,
  seed_replicate_index, configuration_hash)` and to derive the
  run directory path from the same tuple plus the
  `configuration_hash_prefix`.
- Implements an atomic directory-creation helper that refuses to
  overwrite an existing populated run directory.
- Implements an identity-consistency check: the `run_id`
  embedded in a written record must match the directory path
  the record lives in.

Acceptance criteria:

- A second write for the same `run_id` raises an explicit error
  before any data is written.
- A `run_id` whose components do not match the directory path
  is rejected.
- Directory creation is atomic in the sense that a partial
  directory is never left behind by a failed identity-check.

Files touched (allowed):
`experiments/selection_study/identity.py`. Files forbidden:
anything else.

Tests required:
`test_run_id_format_matches_docs_08a_section_4`,
`test_directory_path_encodes_same_identity_as_run_id`,
`test_overwrite_existing_run_raises`,
`test_identity_mismatch_rejected_before_write`. Invariant: "the
run_id and the directory path are two encodings of the same
identity tuple; mismatches are errors, not warnings".

Dependencies: Commits 1, 2.

### Commit 4 (PREFLIGHT MANIFEST GATE): Dry-run manifest

Purpose: make the `--dry-run` flag functional. The runner reads
the configuration, enumerates every planned run, and validates
the manifest before any fit can be invoked. This commit is a
gate: no work on commits beyond it may proceed until its
acceptance criteria are met.

What the commit does:

- Implements `preflight.py`. The manifest is a list of records,
  one per planned run, each containing:
  - `model`, `condition`, `seed_population`,
    `seed_replicate_index`,
  - `graph_seed`, `train_data_seed`, `validation_data_seed`,
    `intervention_ground_truth_seed_base`,
    `model_sampling_seed_base`,
  - the resolved configuration object,
  - `configuration_hash`,
  - `expected_run_id`,
  - `expected_output_directory`,
  - `planned_wrapper` (the wrapper class or factory the runner
    will invoke),
  - `planned_sampling_policy` (`residual_fitted`,
    `unit_variance`, or `dcdi_native`).
- Validates the manifest before any fit can be invoked. The
  preflight validations are:
  - no calibration seed appears in the held-out evaluation
    population, and vice versa;
  - no duplicate `run_id` appears in the manifest;
  - every resolved configuration hashes deterministically (the
    hash is stable across a second call inside the same
    preflight invocation);
  - every `expected_output_directory` is creatable, has no
    pre-existing populated run, and is writable;
  - every `docs/08a` Section 6 mandatory field can in principle
    be populated for the planned run (a schema-level pre-check
    against the run record's required keys; no actual values
    are computed);
  - no `wandb` import is reachable from the preflight code
    path.
- Saves the validated manifest as a JSON artefact next to the
  `results/model_selection/` tree (path determined by Commit 4).
- Exits with non-zero status if any validation fails. The
  failure reason is reported as a structured error.

Acceptance criteria:

- `--dry-run` produces a saved manifest file and a non-zero
  exit status if any validation fails.
- Every preflight validation listed above runs.
- No fit is invoked from any path in this commit.
- No commit after this one may invoke a fit unless the saved
  manifest has been preflight-validated for that run.

Files touched (allowed):
`experiments/selection_study/preflight.py`, the CLI handler in
`experiments/selection_study/run.py`. Files forbidden: anything
else.

Tests required:
`test_preflight_rejects_seed_population_overlap`,
`test_preflight_rejects_duplicate_run_id`,
`test_preflight_rejects_unhashable_or_non_deterministic_config`,
`test_preflight_rejects_pre_populated_output_directory`,
`test_preflight_schema_field_precheck_passes_for_valid_manifest`,
`test_preflight_does_not_import_wandb`,
`test_dry_run_does_not_invoke_any_fit`. Invariant: "every fit
the runner ever invokes corresponds to a preflight-validated
manifest entry; orphan fits are unreachable".

Dependencies: Commits 1, 2, 3.

### Commit 5 (SCHEMA-CONFORMANCE GATE): Single-fit pipeline

Purpose: emit one complete `docs/08a`-conforming `run.json`
from a single wrapper fit on a small SCM. This commit is a gate:
no work on commits beyond it may proceed until its acceptance
criteria are met.

What the commit does:

- Implements `pipeline.py`. The pipeline takes a wrapper (DAGMA
  or DCDI) and a preflight-validated manifest entry, runs one
  fit, and emits the complete `run.json` with every `docs/08a`
  Section 6 mandatory field populated.
- Writes the binary artefacts referenced from `run.json`:
  `continuous_edge_object.npz` and `thresholded_adjacency.npz`,
  plus `loss_history.npz` when `loss_history_status ==
  "available"`.
- Reads back the written `run.json` through a basic loader
  call and asserts every mandatory field is present and of the
  correct type.

Acceptance criteria:

- The written `run.json` contains every mandatory field defined
  in `docs/08a` Section 6, with types matching the schema.
- The record round-trips through a basic loader call (Commit 5
  introduces the minimal loader needed for this check; the full
  loader interface is sketched in `docs/08a` Section 13).
- The `configuration_hash` recorded in the record matches the
  hash recomputed from the resolved configuration on disk.
- The `run_id` recorded in the record matches the directory
  path the record lives in.
- The fit corresponds to a preflight-validated manifest entry;
  fits without a manifest entry raise an explicit error.
- No commit after this one until this gate passes.

Files touched (allowed):
`experiments/selection_study/pipeline.py`,
`experiments/selection_study/loader.py` (minimal). Files
forbidden: anything else.

Tests required:
`test_single_fit_emits_full_docs_08a_record`,
`test_loader_roundtrip_for_single_fit_record`,
`test_configuration_hash_in_record_matches_recomputed_hash`,
`test_run_id_in_record_matches_directory_path`,
`test_fit_without_preflight_manifest_entry_raises`. Invariant:
"every persisted run is a complete docs/08a record; partial or
schema-divergent records are errors, not warnings".

Dependencies: Commits 1, 2, 3, 4.

### Commit 6: Per-intervention MMD sampling pipeline

Purpose: implement the per-intervention records and the MMD
computation that drives `docs/08a` Section 6.10.

What the commit does:

- Implements `sampling.py`. Consumes the wrappers'
  `sample_interventional` API. For each intervention in the
  configured intervention set:
  - draws ground-truth interventional samples from the SCM,
  - draws model-generated interventional samples from the
    wrapper (or records the unavailable reason from the
    wrapper),
  - computes MMD per intervention with the median heuristic
    resolved per-pair,
  - computes the three-point bandwidth sweep at
    `{"0.5x", "1.0x", "2.0x"}` of the resolved bandwidth,
  - populates the per-intervention record per `docs/08a`
    Section 6.10.
- Enforces the within-record consistency rules from `docs/08a`
  Section 6.10:
  - if `sampler_status_for_intervention != "available"`, then
    `mmd_status` equals `sampler_status_for_intervention` and
    `mmd_value` is null;
  - if the sampler is available but MMD cannot be computed for
    any other reason, `mmd_status == "unavailable_other"` and
    `mmd_value` is null, with the reason in `sampler_reason`;
  - if `mmd_status == "available"`, `mmd_value` is a finite
    float; negative values are permitted per `mmd_clip_policy`.
- Persists the per-intervention list as the source of truth for
  MMD, then computes the derived aggregates `mmd_primary`,
  `mmd_sensitivity_unit_variance`, `mmd_bandwidth_sweep`,
  `mmd_available_count`, `mmd_missing_count`, and
  `mmd_bandwidth_used_value` (`docs/08a` Sections 6.5, 6.6,
  6.7).

Acceptance criteria:

- Every per-intervention record honours the `docs/08a` Section
  6.10 consistency rules.
- The derived aggregates match the per-intervention list bit-
  for-bit; a programmatic check is included in the runner.
- The per-intervention list is the source of truth: if a
  derived aggregate disagrees with the list, the runner raises
  rather than persisting the inconsistency.
- The sampling RNG state never leaks into other interventions;
  per-intervention seeds are derived per Commit 2's rule.

Files touched (allowed):
`experiments/selection_study/sampling.py`. Files forbidden:
anything else.

Tests required:
`test_per_intervention_consistency_rule_unavailable_sampler_implies_unavailable_mmd`,
`test_per_intervention_consistency_rule_available_sampler_mmd_unavailable_other`,
`test_per_intervention_consistency_rule_available_mmd_is_finite_float`,
`test_derived_aggregates_match_per_intervention_list`,
`test_per_intervention_rng_isolation`. Invariant: "the per-
intervention list is the source of truth for MMD and the derived
aggregates are exact functions of it".

Dependencies: Commits 1, 2, 3, 4, 5.

### Commit 7: Threshold-robustness offline re-computation

Purpose: implement the `docs/08a` Section 8 procedure for
recomputing structural metrics at the threshold triples in
`docs/02_base_model_selection.md` from saved continuous edge
objects, without retraining.

What the commit does:

- Implements `threshold_robustness.py`. Given a `run_id`, loads
  the run record through the loader, reads the
  `continuous_edge_object` artefact, re-thresholds at the
  per-model triple (DAGMA `{0.2, 0.3, 0.4}` applied to
  `abs(W_continuous)`; DCDI `{0.4, 0.5, 0.6}` applied to
  `w_adj`), and recomputes SHD, SID, and edge counts against
  the true graph for each threshold.
- Validates that re-thresholding produces no retraining call,
  no wrapper instantiation, and no SCM regeneration beyond
  reading the saved ground-truth adjacency.
- Persists the per-threshold metrics as a sibling artefact in
  the run directory (file name decided by Commit 7).

Acceptance criteria:

- No retraining is invoked.
- The recomputed boolean adjacency at each threshold matches a
  direct reference computation (load continuous edge object,
  apply threshold, classify edges) bitwise.
- The recomputed SHD, SID, and edge counts at each threshold
  match the direct reference computation exactly. All three
  metrics are integer-valued, so no float tolerance is
  required; the binding gate is exact integer equality.

Files touched (allowed):
`experiments/selection_study/threshold_robustness.py`. Files
forbidden: anything else.

Tests required:
`test_threshold_robustness_no_retraining_invoked`,
`test_threshold_robustness_boolean_adjacency_bitwise_match`,
`test_threshold_robustness_shd_exact_match`,
`test_threshold_robustness_sid_exact_match`,
`test_threshold_robustness_edge_count_exact_match`. Invariant:
"offline re-thresholding is a pure function of the saved
continuous edge object and the threshold value".

Dependencies: Commits 1, 2, 3, 4, 5.

### Commit 8: Phase A reproduction-pass runner

Purpose: implement the Phase A reproduction pass under the
protocol in `docs/02_base_model_selection.md`. Each model runs
under paper-grounded defaults on its paper-aligned reference
cell.

What the commit does:

- Implements `reproduction_pass.py`. Reads the configuration,
  enumerates the reproduction-pass runs, drives the pipeline
  (Commit 5) for each run, and persists each record under
  `seed_population = "reproduction"`.
- Records, per run, whether the result meets the disqualification
  thresholds defined in `docs/02_base_model_selection.md`.

Acceptance criteria:

- Each model either produces records that meet the
  disqualification thresholds defined in
  `docs/02_base_model_selection.md` or is disqualified
  explicitly. Disqualification reasons are recorded in the run
  record's `convergence_failure_notes` and
  `wrapper_warnings` fields.
- Reproduction-pass records are logged under
  `seed_population = "reproduction"`; they do not mix with
  calibration or held-out evaluation records.
- No mid-run threshold or criterion adjustment.

Files touched (allowed):
`experiments/selection_study/reproduction_pass.py`. Files
forbidden: anything else.

Tests required:
`test_reproduction_records_use_reproduction_seed_population`,
`test_reproduction_disqualification_recorded_in_record`,
`test_reproduction_does_not_overlap_with_calibration_or_held_out`.
Invariant: "reproduction-pass records are separable from
calibration and held-out evaluation by their `seed_population`
value alone".

Dependencies: Commits 1, 2, 3, 4, 5, 6, 7.

### Commit 9: Phase B calibration runner

Purpose: implement the Phase B equal-budget local calibration
under the protocol in `docs/02_base_model_selection.md` (5
configurations per model per condition times 2 calibration seeds
per configuration), then select the best configuration per model
per condition under a frozen ranking rule.

What the commit does:

- Implements `calibration.py`. Enumerates the calibration runs from
  the configuration, drives the pipeline (Commit 5) for each
  run, and persists each record under
  `seed_population = "calibration"`.
- Implements the calibration ranking rule. The rule is frozen
  by this plan, not chosen at runtime, and mirrors
  `docs/02_base_model_selection.md` Section 2 applied within-
  model:
  - Primary: mean SID across calibration seeds and
    interventions.
  - Tiebreaker inside the `docs/02_base_model_selection.md`
    10 percent SID tie margin: mean MMD over available cells.
  - Final tiebreaker: mean SHD.
  - Deterministic fallback: lexicographic order over
    `configuration_hash`.
- Held-out evaluation records are not consulted by the ranking.
  The runner enforces this by reading only records with
  `seed_population == "calibration"` when computing the
  ranking.
- Records the selected `configuration_hash` per model per
  condition for consumption by Commit 10.

Acceptance criteria:

- Calibration records are logged under
  `seed_population = "calibration"`.
- The ranking is mechanical, deterministic, and reproducible
  from the saved calibration records. Two independent
  invocations of the ranking on the same records produce the
  same selected `configuration_hash`.
- The selected `configuration_hash` per model per condition is
  persisted as a sibling artefact (path decided by Commit 9).
- Held-out evaluation records cannot influence the ranking; a
  test forces this by injecting a held-out record into the
  read set and asserting the ranking output is unchanged.

Files touched (allowed):
`experiments/selection_study/calibration.py`. Files forbidden:
anything else.

Tests required:
`test_calibration_records_use_calibration_seed_population`,
`test_calibration_ranking_lexicographic_order_matches_doc02_section_2`,
`test_calibration_ranking_deterministic_across_two_invocations`,
`test_calibration_ranking_ignores_held_out_records`,
`test_calibration_ranking_final_fallback_is_configuration_hash_order`.
Invariant: "the calibration ranking is a pure function of the
calibration records under the frozen lexicographic rule".

Dependencies: Commits 1, 2, 3, 4, 5, 6, 7, 8.

### Commit 10 (CALIBRATION / EVALUATION SEED-POPULATION NON-OVERLAP GATE): Held-out evaluation runner

Purpose: run the selected configuration from Commit 9 on the
held-out evaluation seeds (5 per model per condition per
`docs/02_base_model_selection.md`). This commit is a gate: no
work on commits beyond it may proceed until its acceptance
criteria are met.

What the commit does:

- Implements `held_out.py`. Reads the selected
  `configuration_hash` per model per condition from Commit 9,
  enumerates the held-out evaluation runs, drives the pipeline
  (Commit 5) for each run, and persists each record under
  `seed_population = "held_out_evaluation"`.
- Enforces seed-population non-overlap at run time, not by
  convention. Before any fit is started, the runner reads the
  existing `results/model_selection/` tree, identifies every
  prior run's `(model, condition, seed_replicate_index)`
  under `seed_population = "calibration"`, and rejects any
  held-out evaluation run that collides with a calibration
  seed. The error is explicit and the run is not started.

Acceptance criteria:

- Held-out evaluation records are logged under
  `seed_population = "held_out_evaluation"`.
- Non-overlap with calibration is enforced by a runtime
  assertion that reads the prior records and raises on
  collision before any fit is started.
- The runtime assertion is exercised by a test that constructs
  an overlapping pair and confirms the runner raises.
- No commit after this one may consume held-out records until
  this gate passes.

Files touched (allowed):
`experiments/selection_study/held_out.py`. Files forbidden:
anything else.

Tests required:
`test_held_out_records_use_held_out_evaluation_seed_population`,
`test_held_out_runtime_non_overlap_assertion_raises_on_collision`,
`test_held_out_does_not_invoke_fit_on_collision`. Invariant:
"calibration and held-out evaluation seed populations are
mechanically disjoint; the runner cannot produce a held-out
record using a calibration seed".

Dependencies: Commits 1, 2, 3, 4, 5, 6, 7, 8, 9.

### Commit 11: Halt-and-resume semantics

Purpose: enable a controlled halt and a clean resume without
losing progress or duplicating fits. This is the implementation
lever for the scope-cut hierarchy defined in Section 8 of this
document.

What the commit does:

- Implements `resume.py`. The `--resume` flag reads the existing
  `results/model_selection/` tree, identifies completed
  `run_id` values, and skips those entries on the next pass.
- Defines what counts as "completed": a run directory is
  complete only when its `run.json` contains every `docs/08a`
  Section 6 mandatory field, the configuration hash matches,
  and the loader accepts the record. Partial directories are
  classified explicitly and are not silently treated as
  complete.
- Resume must produce no duplicate `run_id`. A repeat
  enumeration over the same configuration must yield the same
  identity tuple set.

Acceptance criteria:

- A controlled halt (sent during a fit, between fits, or
  between phases) followed by `--resume` produces the same
  final record set as an uninterrupted run.
- Resume never produces a duplicate `run_id`.
- Partial-fit directories are classified through an explicit
  status (recorded in the resume code's report), not silently
  treated as complete.

Files touched (allowed):
`experiments/selection_study/resume.py`. Files forbidden:
anything else.

Tests required:
`test_resume_skips_completed_runs`,
`test_resume_does_not_produce_duplicate_run_ids`,
`test_resume_classifies_partial_directories_explicitly`,
`test_halt_then_resume_matches_uninterrupted_run`. Invariant:
"halt-then-resume is idempotent on the final record set".

Dependencies: Commits 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

### Commit 12: Selection-study report generator

Purpose: produce the one-page selection-study report defined in
`docs/02_base_model_selection.md` from saved run records.

What the commit does:

- Implements `report.py`. Reads only local files through the
  loader. Produces:
  - a per-criterion summary table across both candidates,
    covering SID, MMD primary, SHD, and the
    disqualification/availability status for each
    (`model`, `condition`, `seed_population`) cell;
  - a threshold-robustness table at the three threshold values
    per candidate, derived from saved continuous edge objects
    through Commit 7;
  - an MMD missingness disclosure: `mmd_available_count`,
    `mmd_missing_count`, invalid-graph rate from
    `invalid_graph_for_this_run`, and `sampler_status` reasons
    disaggregated by taxonomy value;
  - the declared base-model decision, produced mechanically
    from the records under the
    `docs/02_base_model_selection.md` Section 2 lexicographic
    rule.

Acceptance criteria:

- The report reads only local files. W&B is not consulted; a
  test asserts no `wandb` import is reachable from the report
  code path.
- The lexicographic decision is produced mechanically from the
  records, not by human interpretation.
- The report is reproducible: two runs of the report generator
  on the same record set produce byte-identical output.

Files touched (allowed):
`experiments/selection_study/report.py`. Files forbidden:
anything else.

Tests required:
`test_report_reads_only_local_files`,
`test_report_does_not_import_wandb`,
`test_report_lexicographic_decision_matches_docs_02_section_2`,
`test_report_byte_reproducible_on_same_record_set`. Invariant:
"the selection-study report is a pure function of the saved run
records under the docs/02 lexicographic rule".

Dependencies: Commits 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.

### Commit 13 (END-TO-END SMOKE-CHECK GATE): End-to-end smoke run

Purpose: run the full pipeline on a tiny SCM with reduced seed
counts, to validate the runner end to end before any real
selection-study run is started. This commit is a gate: no real
selection-study run is invoked until this gate passes.

What the commit does:

- Configures a tiny SCM (for example 3 nodes, 2 configurations
  in Phase B, 1 calibration seed per configuration, 2 held-out
  seeds per model per condition).
- Drives the full pipeline: preflight, Phase A, Phase B,
  calibration ranking, held-out evaluation, report generation.
- Reads every `run.json` produced and confirms `docs/08a`
  schema conformance through the loader.

Acceptance criteria:

- The full pipeline completes without crash.
- Every produced `run.json` conforms to `docs/08a`
  (`schema_version` present, every mandatory field present,
  types match, derived aggregates consistent with the per-
  intervention list).
- The report generator produces a complete one-page report on
  the tiny SCM record set.
- No silent failures. Every fit that did not converge is
  reflected in `training_status` and the report's MMD
  missingness disclosure.
- No scope deviation from the configuration used by the smoke
  run.

Files touched (allowed):
`tests/test_selection_runner_end_to_end.py`. Files forbidden:
anything that changes the runner code (this commit only
exercises the pipeline).

Tests required: the end-to-end test itself, which is the
acceptance gate. Invariant: "every component the real selection
study will use is exercised at least once on a tiny SCM, with
every persisted record schema-conforming".

Dependencies: Commits 1 through 12.

### Commit 14: Runner readout and phase handoff

Purpose: administrative final commit, analogous to Commit 11 of
`docs/06_dagma_wrapper_implementation_plan.md`. Record the state
at runner completion and the artefacts produced.

What the commit does:

- Writes a phase readout (for example `docs/phase_3_readout.md`
  or an equivalent named artefact) summarising runner
  mechanics, gates passed, deferred items, and any open issues
  surfaced during implementation.
- Records the final state of `docs/03_decision_log.md` with any
  contemporaneous entries added during implementation.

Acceptance criteria:

- The phase readout exists and references this plan, the
  authoritative documents in Section 1 of this document, and
  the gate commits.
- No new source file is added.

Files touched (allowed): the new phase readout and
`docs/03_decision_log.md` (final entry). Files forbidden:
anything else.

Tests required: none.

Dependencies: Commits 1 through 13.

---

## 6. Gate-commit discipline

Four commits in this plan are gates. The gate label is part of
the commit's title:

- Commit 4: PREFLIGHT MANIFEST GATE.
- Commit 5: SCHEMA-CONFORMANCE GATE.
- Commit 10: CALIBRATION / EVALUATION SEED-POPULATION NON-OVERLAP
  GATE.
- Commit 13: END-TO-END SMOKE-CHECK GATE.

Rules:

- No work on commits beyond a gate may proceed until the gate's
  acceptance criteria are met.
- A gate failure is not silently retried. The failure is
  recorded as a diagnostic artefact in
  `inspection/probes/` and `docs/04*`, analogous to C-P11 in
  `docs/04f_dcdi_sampler_quality_diagnostic.md`. The runner
  development pauses until the failure is reviewed at the
  project level.
- No acceptance threshold inside a gate is weakened in response
  to a failed gate.

---

## 7. Section 16.1 resolution path

`docs/08a` Section 16.1 records a conflict between
`docs/02_base_model_selection.md` Section 3.5 (which mandates
that all runs set `torch.manual_seed`, `np.random.seed`, and
`dagma.utils.set_random_seed` even when these would be no-ops)
and `docs/03_decision_log.md` 13/05/2026 (which records that the
DAGMA fit does not call any of those setters).

This conflict is resolved as part of Commit 2. It is not a
standalone prerequisite, and it is not deferred. Commit 2 must:

1. pick exactly one of the two eligible options:
   - Option A: amend `docs/02_base_model_selection.md` to v1.5
     to relax the uniform seed-setter requirement for
     candidates that are deterministic by construction; or
   - Option B: amend the DAGMA wrapper source and
     `docs/03_decision_log.md` to make the DAGMA fit call the
     listed seed setters at fit time;
2. record the choice in a contemporaneous
   `docs/03_decision_log.md` entry citing this plan and
   `docs/08a` Section 16.1;
3. produce the corresponding documentation amendment in the
   same commit (either `docs/02_base_model_selection.md` v1.5
   for Option A, or the DAGMA wrapper source change plus
   `docs/03_decision_log.md` entry for Option B).

This plan does not pre-pick the option. The choice is recorded
at Commit 2 review time.

---

## 8. Compute budget and scope-cut hierarchy

This plan adopts the budget defined in
`docs/02_base_model_selection.md` Section 8 verbatim:

- 7 working days from first executable run to declared winner;
- GBP 50 out-of-pocket spend ceiling;
- 30 GPU-hours ceiling;
- effective limit of the available university compute allocation.

This plan also adopts the scope-cut hierarchy defined in
`docs/02_base_model_selection.md` Section 8 verbatim:

1. reduce Phase B calibration breadth (configurations or
   calibration seeds) BEFORE reducing the 5 held-out
   evaluation seeds;
2. reduce non-essential logging visualisations before reducing
   evaluation metrics;
3. reduce held-out evaluation seed count from 5 to 3 only if
   absolutely necessary, and only after step 1 has already
   exhausted calibration-side cuts;
4. do not allow calibration seeds and held-out evaluation
   seeds to overlap as a way to save compute;
5. do not expand the model shortlist or the synthetic cell.

The halt-and-resume mechanism delivered by Commit 11 is the
implementation lever for the scope-cut hierarchy. When a budget
ceiling threatens to be hit, the runner is halted, scope is cut
per the order above (encoded by editing the configuration
between halt and resume), and the runner is resumed. The runner
does not auto-cut scope; the cut is a human decision recorded in
`docs/03_decision_log.md`.

---

## 9. Test discipline

Tests verify scientific invariants and
correctness, not merely that code executes. Each commit's
acceptance criteria are exercised by at least one test whose
failure indicates a violation of a named invariant. The
invariant for each commit is restated here for reference:

- Commit 1: "importing the runner does not trigger any model fit
  and does not touch global RNG state".
- Commit 2: "the seed derivation rule is total, deterministic,
  and consistent with the documentation chosen by Commit 2's
  resolution".
- Commit 3: "the run_id and the directory path are two encodings
  of the same identity tuple; mismatches are errors, not
  warnings".
- Commit 4: "every fit the runner ever invokes corresponds to a
  preflight-validated manifest entry; orphan fits are
  unreachable".
- Commit 5: "every persisted run is a complete docs/08a record;
  partial or schema-divergent records are errors, not warnings".
- Commit 6: "the per-intervention list is the source of truth
  for MMD and the derived aggregates are exact functions of it".
- Commit 7: "offline re-thresholding is a pure function of the
  saved continuous edge object and the threshold value".
- Commit 8: "Phase A records are separable from Phase B and
  held-out evaluation by their `seed_population` value alone".
- Commit 9: "the calibration ranking is a pure function of the
  calibration records under the frozen lexicographic rule".
- Commit 10: "calibration and held-out evaluation seed
  populations are mechanically disjoint; the runner cannot
  produce a held-out record using a calibration seed".
- Commit 11: "halt-then-resume is idempotent on the final record
  set".
- Commit 12: "the selection-study report is a pure function of
  the saved run records under the docs/02 lexicographic rule".
- Commit 13: "every component the real selection study will use
  is exercised at least once on a tiny SCM, with every
  persisted record schema-conforming".
- Commit 14: not test-bearing; administrative.

---

## 10. What this document does not commit to

The following items are out of scope for this plan and are
deferred:

- prior-loss / soft-prior implementation on either wrapper,
  deferred and the main-study contract in
  `docs/01_research_question_and_commitments.md`;
- DCDI Commit 11 resumption, paused per
  `docs/03_decision_log.md` 13/05/2026;
- hard-constraint baseline implementation, deferred per
  `docs/03_decision_log.md` 13/05/2026 and the hard-constraint
  warning in `docs/02_base_model_selection.md`;
- main-study runner architecture, introduced through a later
  `docs/08a` `schema_version` bump;
- W&B integration code; the runner is functional without W&B;
- the base-model decision itself, produced by Commit 12's report
  generator, not by this plan;
- amendments to `docs/01_research_question_and_commitments.md`
  or `docs/02_base_model_selection.md` unless triggered by
  Commit 2's resolution of `docs/08a` Section 16.1.

---

## 11. Immediate next step

Begin Commit 1: runner scaffolding under
`experiments/selection_study/`. Commit 1 is approved for
implementation after this plan is reviewed and accepted; no
prior commit exists.
