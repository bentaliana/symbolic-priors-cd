# 08a_experiment_tracking_and_results_schema

## Status

Frozen contract for local experiment tracking and run-record schema.
Version 1.0.

This document defines the local result-storage layout, the canonical
run-record schema, the run identity rule, the schema versioning rule,
the W&B mirror policy, the notebook consumption interface, and the
visualisation requirements that the base-model selection-study runner
must satisfy. It is consumed by the runner, by offline threshold-
robustness re-computation, by the MMD-unavailable / reliability-
limited rule defined in docs/02_base_model_selection.md, by notebook
analysis, and by the eventual main-study runner via a schema_version
bump.

This document does not implement the runner, the loader, the smoke-
check, or any visualisation. It freezes the contract that those
components must satisfy.

---

## Change log

- v1.0 (this document): initial schema freeze.

---

## 1. Purpose

The selection-study runner produces multiple fits per candidate base
model across seeds, preprocessing conditions, and seed populations.
Without a frozen result schema, downstream analysis would either
depend on private wrapper attributes or risk silently dropping
required fields. This document fixes the contract so that:

- every run produces a self-describing record;
- every reportable selection-study value traces to a saved file;
- the MMD-unavailable / reliability-limited rule defined in
  docs/02_base_model_selection.md can be evaluated mechanically
  across runs without re-running fits or sampling;
- the threshold-robustness reporting required by
  docs/02_base_model_selection.md can be performed offline from the
  saved continuous edge objects without retraining;
- notebooks consume a stable interface rather than ad-hoc paths.

The schema is intentionally narrow. It freezes only what the runner
and the selection-study report require. Main-study extensions are
deferred to a schema_version bump.

---

## 2. Scope

In scope:

- local directory layout for selection-study runs;
- canonical run identifier;
- schema versioning rule;
- mandatory fields per run record;
- continuous edge object storage policy;
- threshold-robustness support;
- WrapperDiagnostics consumption policy;
- invalid-graph reporting at the schema level;
- mechanical evaluability of the MMD-unavailable rule;
- W&B mirror policy;
- notebook consumption interface;
- smoke-check requirement before any selection-study run;
- visualisation requirements for the selection-study report.

Out of scope (see Section 17):

- runner architecture;
- prior-loss logging fields;
- main-study extensions including corruption-grid fields;
- hard-constraint baseline schema;
- W&B integration code;
- loader implementation;
- smoke-check implementation;
- visualisation implementation.

---

## 3. Local directory layout

Local files are the authoritative experiment record. W&B and any
other external dashboards are optional mirrors only (see Section 12).

The selection-study runner writes one run directory per fit. The
layout extends the convention recorded in
docs/02_base_model_selection.md:

```
results/
  model_selection/
    <model>/
      <condition>/
        <seed_population>/
          seed<seed_replicate_index>/
            <configuration_hash_prefix>/
              run.json
              continuous_edge_object.npz
              thresholded_adjacency.npz
              loss_history.npz
              env_snapshot.txt          (optional inline; see Section 6)
              config_resolved.json      (optional inline; see Section 6)
```

Path components:

- `<model>`: `dagma` or `dcdi`.
- `<condition>`: `centred_only` or `standardised`.
- `<seed_population>`: `calibration`, `held_out_evaluation`, or
  `reproduction`.
- `<seed_replicate_index>`: integer.
- `<configuration_hash_prefix>`: the first 12 characters of the full
  content hash of the resolved configuration object. The prefix is
  used as a directory component for filesystem-friendly length; the
  full hash is recorded in the run record (see Section 6).

Each run directory contains exactly one canonical JSON record named
`run.json`. Binary artefacts referenced by the JSON record live as
sibling files in the same directory and are referenced by relative
path. The JSON record itself must be small enough to load in
milliseconds, not seconds; large arrays do not live inline in the
JSON.

The path layout uses POSIX-style forward slashes inside the JSON
record. The runner is responsible for translating to the platform
path separator when reading from disk.

Later studies (for example a main-study runner) use their own top-
level subdirectory under `results/` (for example `results/main_study/
...`) so that selection-study results are not mixed with downstream
study results.

---

## 4. Canonical run identifier

The canonical run identifier `run_id` is derived deterministically
from the tuple:

```
(model, condition, seed_population, seed_replicate_index, configuration_hash)
```

with the format:

```
run_id = "<model>__<condition>__<seed_population>__seed<seed_replicate_index>__cfg<configuration_hash>"
```

Rules:

- `seed_population` is part of the identity, not metadata, because
  docs/02_base_model_selection.md requires non-overlapping
  calibration and held-out evaluation populations and the schema
  must make leakage detectable at the identifier level.
- `configuration_hash` is the deterministic content hash of the
  resolved configuration object actually used by the code, not of
  the user-requested configuration. The hashing target is the
  canonical JSON serialisation of `config_resolved`, produced with
  sorted keys, deterministic float repr, and no non-deterministic
  object reprs (for example `json.dumps(..., sort_keys=True,
  separators=(",", ":"))`). The hash function is SHA-256. The full
  64-character lowercase hex digest is stored in
  `configuration_hash`. The first 12 characters of that digest are
  used as the `<configuration_hash_prefix>` directory component;
  the prefix is never used as the canonical identifier. The
  algorithm name is recorded per run via
  `configuration_hash_algorithm` (Section 6.7).
- The `run_id` and the directory path encode the same identity. If
  they disagree, the run record is invalid and must be rejected by
  the loader.
- Two runs may share `(model, condition, seed_population, seed_replicate_index)`
  only if they have different `configuration_hash` values. Two runs
  with identical `run_id` represent the same logical run and the
  later write must not silently overwrite the earlier one.

---

## 5. Schema versioning

The schema is versioned by an integer field `schema_version`. The
initial version is `1`.

Rules:

- Every run record contains `schema_version` as a top-level integer
  field.
- The loader inspects `schema_version` first. The loader either
  accepts the record at a known version or rejects it with an
  explicit reason. There is no silent migration.
- Adding, removing, or changing the type of any mandatory field
  defined in Section 6 requires a `schema_version` bump.
- Adding optional fields beneath `wrapper_diagnostics` does not
  require a `schema_version` bump because the wrapper diagnostics
  record is consumed as-is (see Section 9).
- Future studies that need additional mandatory fields (for
  example main-study corruption fields) must introduce a new
  `schema_version` and the corresponding loader path. Records at the
  earlier version remain readable under the earlier schema.

---

## 6. Mandatory fields per run

Every run record contains all of the fields listed below. The list
is a freeze, not a recommendation. Adding fields requires a
`schema_version` bump per Section 5. Field types are described in
plain Python terms; the runner serialises them as JSON-compatible
values, with binary artefacts referenced by relative path.

### 6.1 Identity

- `run_id`: string. Derived per Section 4.
- `schema_version`: integer. Initial value `1`.
- `model`: string. One of `dagma`, `dcdi`.
- `condition`: string. One of `centred_only`, `standardised`.
- `seed_population`: string. One of `calibration`,
  `held_out_evaluation`, `reproduction`.
- `seed_replicate_index`: integer. Within-population replicate
  index.
- `configuration_hash`: string. Full content hash of the resolved
  configuration object. Algorithm frozen in Section 4 and recorded
  per run via `configuration_hash_algorithm` (Section 6.7).
- `graph_seed`: integer. SCM construction seed.

### 6.2 Reproducibility

- `git_hash`: string. The repository HEAD commit hash at run time.
- `env_snapshot`: string. Either an inline serialised environment
  summary, or a relative path (for example `env_snapshot.txt`) to a
  sibling file in the run directory.
- `config_resolved`: object. The configuration as the code used it,
  not as the user requested it. May be stored inline as a JSON
  object or referenced as a relative path (for example
  `config_resolved.json`). When referenced as a path, the loader is
  responsible for resolving it before returning a record.
- `seed_torch`: integer or null. The value passed to
  `torch.manual_seed`. Null when the candidate's fit does not call
  `torch.manual_seed` (see Section 16 for the open conflict on this
  rule).
- `seed_numpy`: integer or null. The value passed to
  `np.random.seed`. Null when the candidate's fit does not call
  `np.random.seed` (see Section 16).
- `seed_dagma`: integer or null. The value passed to
  `dagma.utils.set_random_seed`. Null when the candidate's fit does
  not call this setter (see Section 16).
- `model_sampling_seed_base`: integer. The seed used to construct
  the local `np.random.default_rng` instance for model-generated
  interventional sampling.
- `model_sampling_seed_derivation_rule`: string. A human-readable
  identifier for the derivation rule used to produce
  `model_sampling_seed_base` from upstream seed material (for
  example `"hash(seed, intervention_id, sampling_round)"`).
  Required by docs/02_base_model_selection.md so model-generated
  draws are reproducible independently of training.
- `train_data_seed`: integer. Observational training data sampling
  seed.
- `validation_data_seed`: integer or null. Validation-split
  sampling seed. Null when the candidate uses no validation split
  (for example DAGMA in the current configuration).
- `intervention_ground_truth_seed_base`: integer. Base seed for
  ground-truth interventional sampling. Per-intervention
  ground-truth seeds are derived deterministically and recorded in
  the per-intervention list (Section 6.10).

### 6.3 Training

- `training_status`: string. One of the values defined by the
  status taxonomy in docs/04_wrapper_api_contract.md
  (`converged`, `max_iter`, `diverged`, `wrapper_error`).
- `n_iterations`: integer or null. The observed iteration count
  when the wrapper exposes one. DCDI records an integer; DAGMA
  records null because DAGMA does not expose an observable inner-
  loop iteration counter. The configured upper bound for DAGMA is
  recorded inside `wrapper_diagnostics.model_specific_diagnostics`
  and is not re-emitted at this level.
- `runtime_seconds`: float. Wall-clock fit time in seconds.
- `loss_history`: string or null. Relative path to a binary
  artefact (for example `loss_history.npz`) containing the loss
  trajectory or a downsampled loss history. Must be non-null when
  `loss_history_status == "available"`. Must be null otherwise;
  loaders must not attempt to read the artefact when
  `loss_history_status != "available"`.
- `loss_history_status`: string. One of `available`,
  `unavailable_no_api`, `unavailable_not_recorded`.

### 6.4 Graph

- `graph_status`: string. One of the values defined by the status
  taxonomy in docs/04_wrapper_api_contract.md (`valid_dag`,
  `cyclic`, `bidirected`, `self_loop`, `invalid_shape`).
- `graph_status_reason`: string or null. Null when
  `graph_status == "valid_dag"`. Otherwise a short human-readable
  reason taken from the wrapper diagnostics. The reason is
  informational; the status string is the load-bearing field.
- `thresholded_adjacency`: string. Relative path to a binary
  artefact (for example `thresholded_adjacency.npz`) containing the
  thresholded boolean adjacency in the project row-source /
  column-destination orientation.
- `continuous_edge_object`: string. Relative path to a binary
  artefact (for example `continuous_edge_object.npz`) containing the
  native continuous edge representation at full precision (see
  Section 7).

### 6.5 Metrics

- `shd`: integer. Computed using the project SHD primitive against
  the ground-truth adjacency. See `shd_reversal_cost` in Section 6.7
  for the reversal convention used.
- `sid`: integer. The raw mistake count returned by the project
  `sid_score` primitive. SID is active in this schema; there is no
  deferred-SID branch.
- `mmd_primary`: float or null. Derived aggregate over the
  per-intervention records in `interventions` (Section 6.10);
  mean of `mmd_value` across present interventions under the
  candidate's primary sampling policy. Null when no intervention
  produced an MMD value for this run. The aggregation rule is the
  mean across present intervention conditions, per
  docs/02_base_model_selection.md.
- `mmd_sensitivity_unit_variance`: float or null. Derived
  aggregate over the per-intervention records in `interventions`
  (Section 6.10) for the unit-variance sensitivity policy,
  applicable for DAGMA. Null when not applicable (for example DCDI
  runs) or when no intervention produced a value.
- `mmd_bandwidth_sweep`: object with three nullable float fields
  keyed by multiplier. Derived aggregate over `interventions`
  (Section 6.10):
  - `"0.5x"`: float or null
  - `"1.0x"`: float or null
  - `"2.0x"`: float or null
  Each entry is the run-level mean MMD under the corresponding
  bandwidth multiplier of the median heuristic. Each entry is
  independently nullable.
- `validation_nll`: float or null. The validation negative-log-
  likelihood where the candidate exposes one. Null when not
  applicable.

### 6.6 Sampler and MMD accounting

These per-run fields are required by the mechanical evaluability of
the MMD-unavailable / reliability-limited rule (see Section 11).

- `sampler_status`: string. One of the values defined by the
  status taxonomy in docs/04_wrapper_api_contract.md
  (`available`, `unavailable_invalid_graph`, `unavailable_no_api`,
  `unavailable_unresolved_noise_policy`).
- `sampler_status_reason`: string or null. Short human-readable
  reason taken from the wrapper diagnostics; null when
  `sampler_status == "available"`. The taxonomy enum continues to
  live on `sampler_status`.
- `sampler_policy_used`: string. One of `residual_fitted`,
  `unit_variance`, `dcdi_native`. Identifies the sampling policy
  that produced `mmd_primary` for this run.
- `mmd_available_count`: integer. Derived aggregate over
  `interventions` (Section 6.10); number of intervention records
  with `mmd_status == "available"`.
- `mmd_missing_count`: integer. Derived aggregate over
  `interventions` (Section 6.10); number of intervention records
  with `mmd_status != "available"`. The sum
  `mmd_available_count + mmd_missing_count` equals the length of
  `interventions` and the size of the configured intervention set
  for this run.
- `invalid_graph_for_this_run`: boolean. Derived from
  `graph_status != "valid_dag"`. Stored explicitly because the
  invalid-graph rate is a load-bearing aggregate in
  docs/02_base_model_selection.md and the schema must make it
  recoverable without re-deriving from the status string at every
  consumer site.

### 6.7 Convention

- `shd_reversal_cost`: integer. The project default is `2`. Recorded
  per run because docs/04j_mmd_shd_reference_crosscheck.md
  documents a convention difference between the project default
  and `gadjid.shd`. Any future report that mixes project SHD with
  `gadjid.shd` must read this field and account for the difference.
- `mmd_bandwidth_used_value`: object mapping intervention
  identifier (string) to the bandwidth value resolved by the
  median heuristic for that intervention condition (float).
  Derived aggregate over `interventions` (Section 6.10); each
  entry mirrors the corresponding `bandwidth_used` field on the
  per-intervention record. Stored per intervention because the
  median heuristic is per-pair, not per-run; aggregating across
  interventions before recording would lose reproducibility
  information. Empty object when no intervention produced a
  usable MMD value.
- `mmd_clip_policy`: string. Always `"no_clip"`. The raw unbiased
  MMD estimator may return negative values and is preserved
  verbatim. Recorded per run so any future change is detectable
  through the run record.
- `sid_backend`: string. Always `"gadjid"` in this schema version.
- `sid_backend_version`: string. The pinned version of the SID
  backend (currently `"0.1.0"`).
- `sid_argument_order`: string. Always `"predicted_then_true"` in
  this schema version. The internal backend call flips order once
  at the wrapper boundary; the project-facing convention is
  predicted first, true second.
- `sid_return_value`: string. Always `"raw_mistake_count"` in this
  schema version. The normalised SID score is discarded.
- `configuration_hash_algorithm`: string. Always
  `"sha256_canonical_json_sorted_keys"` at this schema_version.
  Future hash changes require a schema_version bump.

### 6.8 Wrapper diagnostics

- `wrapper_diagnostics`: object. The full `WrapperDiagnostics`
  record from the wrapper, consumed as-is. Includes
  `model_specific_diagnostics` and every other top-level key the
  wrapper emits. The schema does not redesign, reshape, or
  selectively re-emit fields (see Section 9).

### 6.9 Notes

- `convergence_failure_notes`: string. May be empty. A short free-
  form note recording any convergence anomaly or wrapper-level
  warning that did not result in a `wrapper_error` training_status.
- `wrapper_warnings`: list of strings. May be empty. Non-fatal
  warnings emitted by the wrapper during fit or sampling, for
  example any non-fatal source patches applied. Required by
  docs/04_wrapper_api_contract.md.

### 6.10 Interventions

- `interventions`: list of structured per-intervention records,
  stored inline in `run.json`. Each record contains:
  - `intervention_id`: string.
  - `target_node`: integer.
  - `value_raw`: float. Intervention value in raw SCM units.
  - `value_model_frame`: float. Intervention value transformed
    into the candidate's model frame by the project preprocessing
    rules defined in docs/02_base_model_selection.md.
  - `ground_truth_sampling_seed`: integer. Per-intervention seed
    used for ground-truth interventional sampling. Derived
    deterministically from `intervention_ground_truth_seed_base`
    and `intervention_id`.
  - `model_sampling_seed`: integer. Per-intervention seed used for
    model-generated interventional sampling. Derived
    deterministically from `model_sampling_seed_base` and
    `intervention_id` per
    `model_sampling_seed_derivation_rule`.
  - `n_ground_truth_samples`: integer.
  - `n_model_samples`: integer.
  - `mmd_value`: float or null. The unbiased RBF MMD between the
    ground-truth and model-generated samples for this
    intervention condition. Null when not available; reason
    carried in `mmd_status`.
  - `mmd_status`: string. One of `available`,
    `unavailable_invalid_graph`, `unavailable_no_api`,
    `unavailable_unresolved_noise_policy`, `unavailable_other`.
    The first four values mirror the `sampler_status` taxonomy in
    docs/04_wrapper_api_contract.md; `unavailable_other` extends
    that taxonomy to cover MMD-specific failures not attributable
    to sampler state (for example a degenerate median-heuristic
    bandwidth per docs/04j_mmd_shd_reference_crosscheck.md).
    Future alignment between this taxonomy and docs/04 is a
    docs/04 amendment concern, not a `schema_version` concern.
  - `bandwidth_used`: float or null. The bandwidth resolved by
    the median heuristic for this intervention condition. Null
    when `mmd_status != "available"`.
  - `bandwidth_sweep`: object with keys `"0.5x"`, `"1.0x"`, and
    `"2.0x"`, each float or null. MMD value at the corresponding
    bandwidth multiplier of the median heuristic for this
    intervention condition. Each entry is independently nullable.
  - `sampler_status_for_intervention`: string from the status
    taxonomy in docs/04_wrapper_api_contract.md
    (`available`, `unavailable_invalid_graph`,
    `unavailable_no_api`, `unavailable_unresolved_noise_policy`).
  - `sampler_reason`: string or null. Short human-readable reason
    taken from the wrapper diagnostics when the sampler is
    unavailable; null when
    `sampler_status_for_intervention == "available"`.

The `interventions` list is the source of truth for MMD. The
run-level fields `mmd_primary`,
`mmd_sensitivity_unit_variance`, and `mmd_bandwidth_sweep`
(Section 6.5), `mmd_available_count` and `mmd_missing_count`
(Section 6.6), and `mmd_bandwidth_used_value` (Section 6.7) are
derived convenience aggregates that the runner persists for fast
loading. Derived aggregates must be consistent with the
`interventions` list. If a consumer detects an inconsistency, the
per-intervention list wins and the disagreement is a bug to be
reported.

Within each intervention record, the runner and any consumer must
honour the following consistency rules between
`sampler_status_for_intervention`, `mmd_status`, and `mmd_value`:

- If `sampler_status_for_intervention != "available"`, then
  `mmd_status` equals `sampler_status_for_intervention` and
  `mmd_value` is null.
- If `sampler_status_for_intervention == "available"` and MMD
  cannot be computed for any other reason, `mmd_status` is
  `"unavailable_other"` and `mmd_value` is null. The explanatory
  reason is carried in `sampler_reason`; this `schema_version`
  does not expose a separate `mmd_reason` field. If a dedicated
  MMD-reason field is needed later, it goes through a
  `schema_version` bump.
- If `mmd_status == "available"`, `mmd_value` must be a finite
  float. Negative values are permitted per the unbiased MMD
  estimator (see `mmd_clip_policy` in Section 6.7).

Raw intervention samples are not required as mandatory artefacts
at this `schema_version`.

---

## 7. Continuous edge object storage

The continuous native edge object is stored at full precision as a
binary artefact in the run directory and referenced from the JSON
record by relative path (Section 6.4).

Format: `npz`. Justification: `npz` is the project default array
container, supports multiple named arrays in one file (DCDI exports
both `log_alpha` and `get_w_adj()`), is `numpy`-native, and round-
trips losslessly at full float64 precision.

Per-model content:

- DAGMA: one array named `W_continuous`, shape `(n_nodes, n_nodes)`,
  dtype `float64`. The matrix returned by `DagmaLinear.fit` with
  `w_threshold = 0.0`.
- DCDI: two arrays. `log_alpha`, the value of
  `model.gumbel_adjacency.log_alpha` at training exit. `w_adj`, the
  value of `model.get_w_adj()` at training exit. Both at full
  precision, both in the project row-source / column-destination
  orientation.

The thresholded boolean adjacency referenced in Section 6.4 is
stored in a separate `npz` artefact (for example
`thresholded_adjacency.npz`) containing one array named
`thresholded_adjacency`, dtype `bool`, shape `(n_nodes, n_nodes)`.

The schema does not require additional artefacts beyond those
referenced from the mandatory fields in Section 6. Additional
artefacts may be introduced in a future `schema_version` if the
runner needs them.

---

## 8. Threshold-robustness support

The schema enables offline re-computation of structural metrics at
the threshold triples defined in
docs/02_base_model_selection.md for the threshold-robustness report.
Re-computation is performed by:

1. Loading the run record via the loader (see Section 13).
2. Reading `continuous_edge_object` from the referenced path.
3. Re-thresholding the continuous edge object at each value of the
   per-model threshold triple. For DAGMA the triple is applied to
   `abs(W_continuous)`. For DCDI the triple is applied to the
   `w_adj` array.
4. Re-evaluating SHD, SID, and edge counts on each resulting
   boolean adjacency.

No retraining is required. No wrapper state beyond the saved
continuous edge object is consulted. The runner must therefore
ensure the continuous edge object is saved at the precision and
orientation specified in Section 7.

If a candidate's thresholded graph at a non-default threshold is
not a valid DAG, the same status-taxonomy rules apply: SID is not
silently computed on a non-DAG, SHD may still be computed but is
flagged as structurally invalid in the report. See
docs/04_wrapper_api_contract.md.

---

## 9. WrapperDiagnostics consumption policy

The schema consumes `wrapper_diagnostics` as-is from the wrapper.

Rules:

- The schema does not redesign, reshape, or selectively re-emit
  fields from `WrapperDiagnostics`.
- The schema does not introduce a private schema-side field as a
  substitute for a missing or under-populated `WrapperDiagnostics`
  field. If a needed field is missing, the right response is to
  widen the `WrapperDiagnostics` TypedDict in the wrapper layer
  deliberately, with a corresponding `schema_version` bump if the
  missing field would be promoted to a top-level mandatory field
  in Section 6.
- The schema does not enforce internal structure of
  `model_specific_diagnostics`. Each wrapper is free to populate
  this nested object with its own model-native fields.
- Loaders may project `wrapper_diagnostics` into a narrower view
  for notebook consumption, but no information may be discarded
  during persistence.

---

## 10. No silent graph repair at the schema level

The schema does not filter, drop, or repair runs based on graph
validity. Every fit attempt produces a run record regardless of
`graph_status`.

Rules:

- Invalid-graph runs are classified through `graph_status` and
  retained in the run directory.
- The schema does not provide an "exclude invalid graphs" toggle in
  the run record itself. Filtering invalid runs is a consumer-side
  decision and is performed by the loader filter or by notebook
  analysis using the explicit `graph_status` and
  `invalid_graph_for_this_run` fields.
- Aggregation pipelines that compute the invalid-graph rate
  defined in docs/02_base_model_selection.md must read
  `invalid_graph_for_this_run` (or equivalently `graph_status`) from
  every run record in the relevant population. Silent omission is
  not permitted.

This matches the no-silent-repair policy of the wrapper layer
documented in docs/04_wrapper_api_contract.md and the decision-log
entries in docs/03_decision_log.md.

---

## 11. MMD reliability rule mechanical evaluability

The MMD-unavailable / reliability-limited rule defined in
docs/02_base_model_selection.md must be evaluable mechanically
across the run records produced by the runner, without re-running
fits or sampling.

The per-run fields required for that evaluation are:

- `sampler_status` (Section 6.6) for the unavailable-reasons
  disaggregation.
- `sampler_status_reason` (Section 6.6) as a human-readable
  annotation accompanying `sampler_status`. The taxonomy enum
  driving the disaggregation lives on `sampler_status`; this
  field is informational only.
- `mmd_available_count` (Section 6.6).
- `mmd_missing_count` (Section 6.6).
- `invalid_graph_for_this_run` (Section 6.6) for the invalid-graph
  rate aggregation.
- `mmd_primary` (Section 6.5) as the value contributed by the run
  when available.
- `sampler_policy_used` (Section 6.6) so policy-conditioned
  aggregation is possible.

Together these fields are sufficient to compute, per candidate and
per condition:

- the mean MMD over available cells;
- the available count;
- the missing count;
- the invalid-graph rate;
- the disaggregated unavailable reasons from the
  docs/04_wrapper_api_contract.md taxonomy.

These are the four reporting requirements stated in the MMD-
unavailable rule of docs/02_base_model_selection.md. The schema
records all of them as load-bearing per-run fields rather than as
post-hoc derivations.

---

## 12. W&B mirror policy

Local files are the authoritative experiment record.

Rules:

- W&B (or any equivalent external tracking tool) may be used only
  as an optional mirror or dashboard.
- Every reportable thesis value traces to a local file. No
  reportable value may exist only in W&B.
- W&B does not compute load-bearing metrics. Metrics are computed
  by the runner and persisted locally; W&B receives them after
  computation.
- W&B and the local record share the same `run_id`. If they
  disagree on any field, the local record wins and the gap is a
  bug.
- Integration is one-way: the runner emits to W&B at run
  completion. The runner does not read state from W&B.
- W&B integration is optional and disabled by default. Local
  recording is mandatory; W&B mirroring is not.

No W&B-specific fields are introduced into the run record. The
canonical run record is identical whether W&B is enabled or not.

---

## 13. Notebook consumption

Notebooks consume the schema through a loader function pair.
Implementation is deferred (see Section 17); this document
specifies only the interface.

Loader signature sketch (single line per function, plain Python
type hints):

```
load_run(run_id: str) -> RunRecord
load_runs(filter: Mapping[str, Any]) -> Iterable[RunRecord]
```

`RunRecord` is a typed record carrying every mandatory field
defined in Section 6 plus a resolved view of any path-referenced
artefacts (for example the `continuous_edge_object` loaded as a
`numpy` array). The exact type is a loader-implementation concern.

Rules:

- Notebooks never read raw files. Notebooks call `load_run` or
  `load_runs` and receive a typed record.
- Notebooks never contain training logic. Notebooks consume and
  analyse; the runner trains.
- The `filter` argument to `load_runs` accepts a mapping over
  identity fields (`model`, `condition`, `seed_population`,
  `seed_replicate_index`, `configuration_hash`) and may be
  extended in a backward-
  compatible way without a `schema_version` bump because the
  filter shape is a loader-implementation concern, not a record
  schema concern.
- The loader is responsible for rejecting records whose
  `schema_version` is unsupported, with an explicit reason.

---

## 14. Smoke-check requirement

Before any selection-study run consumes this schema, one toy fit
must produce a run record that round-trips through the loader with
every mandatory field defined in Section 6 populated and
`schema_version` present.

Rules:

- The smoke-check exercises both wrappers (`dagma`, `dcdi`) and
  both conditions (`centred_only`, `standardised`) if both are in
  scope at smoke-check time.
- The smoke-check writes one full run directory per case, including
  every binary artefact referenced from the run record.
- The smoke-check reads each written record back via `load_run`
  and asserts that every mandatory field in Section 6 is present
  and of the expected type.
- The smoke-check does not exercise the selection-study criteria;
  it exercises only the schema and the loader.
- The smoke-check is not part of normal `pytest` collection unless
  the runner explicitly registers it. Its purpose is one-time
  validation that the schema is implementable before the runner is
  written.

Implementation of the smoke-check is deferred (Section 17).

---

## 15. Visualisation requirements

The selection-study report defined in docs/02_base_model_selection.md
is a one-page artefact summarising the selection outcome. The
schema supports the following report contents and no more:

- A per-criterion summary across both candidates, covering SID,
  MMD primary, and SHD under each condition and each seed
  population.
- A threshold-robustness table at the three threshold values
  per candidate (the threshold triples are defined in
  docs/02_base_model_selection.md), derived from the saved
  continuous edge objects per Section 8. The table shows SHD,
  SID, and edge counts at each threshold.
- An MMD missingness disclosure showing:
  - `mmd_available_count` summed across the relevant population;
  - `mmd_missing_count` summed across the relevant population;
  - the invalid-graph rate computed from
    `invalid_graph_for_this_run`;
  - `sampler_status` reasons disaggregated by the taxonomy values
    defined in docs/04_wrapper_api_contract.md.
- A declared base-model decision and the justification text
  required by docs/02_base_model_selection.md.

Out of scope for this schema:

- Main-study corruption-degradation visuals.
- Per-hypothesis test outputs (H1, H2, H3, H4 of
  docs/01_research_question_and_commitments.md).
- Cross-seed instability visuals.

These visuals are introduced through a future `schema_version` bump
when the main-study runner consumes a wider schema.

---

## 16. Open issues and conflicts

The following items were surfaced while drafting this schema and
are recorded here rather than smoothed over. They require review
before the selection-study runner is written.

### 16.1 Seed-discipline conflict between docs/02 and docs/03

docs/02_base_model_selection.md mandates, for all selection-study
runs:

> All runs MUST set fixed seeds for both `torch.manual_seed` and
> `np.random.seed` (DCDI) and for `dagma.utils.set_random_seed`
> (DAGMA), even when these would be no-ops, so the seed discipline
> is uniform across candidates.

docs/03_decision_log.md, in the entry dated 13/05/2026 recording
the DAGMA wrapper plan acceptance, states:

> DAGMA fit will not call `dagma.utils.set_random_seed`,
> `np.random.seed`, or `torch.manual_seed`; the fit is deterministic
> for fixed input and hyperparameters, and sampler randomness is
> handled through local `np.random.default_rng(sample_seed)`.

These two statements directly conflict on whether the DAGMA fit
path is required to call the listed seed setters. This schema
cannot resolve the conflict because the answer changes the meaning
of the `seed_torch`, `seed_numpy`, and `seed_dagma` fields in
Section 6.2: those fields are non-null when the corresponding
setter is called and null otherwise.

This schema currently allows both seed fields to be null, and
records the resolution as pending. The conflict must be resolved
before the runner is written, by either:

- amending docs/02_base_model_selection.md to relax the uniform
  seed-setter requirement for candidates that are deterministic by
  construction; or
- amending the DAGMA wrapper to call the listed seed setters at
  fit time, and updating docs/03_decision_log.md accordingly.

The schema does not pick between these options.

### 16.2 SID gating language in docs/02 versus current SID status

docs/02_base_model_selection.md retains language in its logging
schema and outputs section treating SID as "deferred" and
describing an "explicit gating statement on SID" before the
selection study can be declared scientifically complete. SID is
now implemented and verified per docs/phase_2d_sid_readout.md and
the SID closure entry in docs/03_decision_log.md. The deferred-SID
code path no longer exists.

This schema treats SID as active. The `sid` field in Section 6.5
is mandatory and typed `int`. There is no deferred-SID branch.

If docs/02_base_model_selection.md is later amended to remove the
deferred-SID gating language, no change to this schema is
required. The mismatch is between docs/02 and the current code
state; the schema follows the current code state.

### 16.3 Intervention outputs are not a mandatory top-level field

docs/02_base_model_selection.md lists "intervention outputs used in
Criterion 1" among required per-run log fields. This schema does
not include `intervention_outputs` as a mandatory top-level field
in Section 6. The runner may persist intervention outputs as
additional binary artefacts in the run directory, but at
`schema_version = 1` these artefacts are not referenced from the
canonical record and are not required for the MMD reliability rule
(Section 11) or the threshold-robustness report (Section 8).

If a future analysis requires intervention outputs to be loaded by
`load_run` or `load_runs`, an explicit mandatory field referencing
the artefact must be added under a new `schema_version`.

### 16.4 Doc 02 deferred-SID language is stale

docs/02_base_model_selection.md retains deferred-SID phrasing in
Section 3.4 ("before then, SID is logged as deferred") and Section
7 item 6 ("explicit gating statement on SID... SID-dependent
claims remain deferred"). Phase 2d closed the SID verification
gate and rendered that phrasing obsolete; see
docs/phase_2d_sid_readout.md and the SID closure entry in
docs/03_decision_log.md.

Recommendation: a docs/02 v1.4 editorial amendment should remove
the deferred-SID language before the selection-study runner is
written. The schema itself requires no change.

This entry is the editorial recommendation; Section 16.2 describes
how the schema itself handles the mismatch.

---

## 17. What this document does not commit to

The following items are deferred to later artefacts:

- selection-runner architecture, configuration shape, and entry
  point;
- prior-loss logging fields, including any field describing the
  prior penalty term, the confidence weight schedule, the prior
  family, or the corruption level;
- main-study extension fields including any corruption-grid
  field, any per-hypothesis aggregation field, and any field
  required only for the main-study reporting schema;
- hard-constraint baseline schema, including any field describing
  exclude-edges or include-edges sets;
- W&B integration code;
- loader implementation;
- smoke-check implementation;
- visualisation implementation.

These items are introduced through later documents and, where they
require schema changes, through a `schema_version` bump.

---

## 18. Immediate next step

The next required document is:

`docs/08_base_model_selection_plan.md`

Its role is to plan the base-model selection-study runner under the
protocol in docs/02_base_model_selection.md, consuming the run
record schema and the directory layout frozen by this document.

docs/08_base_model_selection_plan.md will follow the commit-
structured planning pattern of
docs/05_dcdi_wrapper_implementation_plan.md and
docs/06_dagma_wrapper_implementation_plan.md, with a numbered
commit sequence, per-commit acceptance criteria, and explicit gate
commits (for example a schema-conformance gate and an end-to-end
smoke-check gate) before the runner can be treated as ready for the
selection study. This is a forward-looking expectation, not a
freeze on the internal structure of
docs/08_base_model_selection_plan.md.
