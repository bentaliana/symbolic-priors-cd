# 08g_file_exists_error_incident

## Status

The incident is repaired and closed for calibration. The
calibration artefact at
`results/model_selection/calibration/4a67117a10b1/selected_configurations.json`
is now consistent with the 40 on-disk per-fit records and is the
valid input to held-out evaluation. No further action is required
at the calibration stage; held-out evaluation remains pending as
the normal downstream step.

## 1. Context

The real calibration run was executed against the four
calibration configs at
`experiments/selection_study/configs/calibration/` covering
model in `{dagma, dcdi}` and condition in
`{centred_only, standardised}`. The run produced
`calibration_run_hash_prefix = 4a67117a10b1` and completed
mechanically:

- 40 per-fit JSON records in `records/`.
- 40 `START` and 40 `END` lines in `calibration_run.log`.
- A `selected_configurations.json` artefact that passed
  structural validation (artefact type, schema version, seed
  population, identity payload hash, absence of forbidden
  field names).

Calibration is a within-model and within-condition
hyperparameter selection stage. It ranks the five-point
sparsity grid per (model, condition) and reports the rank-1
configuration for each cell. It does not perform base-model
selection; that decision happens at held-out evaluation.

## 2. Incident

One of the 40 per-fit records was initially degenerate rather
than a converged fit. The contaminated record was:

- file:
  `records/dagma_centred_only_06ee98d13852_seed201.json`
- model: `dagma`
- condition: `centred_only`
- seed_value: `201`
- hyperparameters: `lambda1 = 0.25`
- failure_type: `FileExistsError`

The associated stale raw output directory was:

`results/model_selection/dagma/centred_only/calibration/seed0/06ee98d13852`

The contamination was filesystem state, not model behaviour:
the underlying DAGMA fit was never executed by the current
calibration run for this job, because the per-fit pipeline
refused to start once it observed the pre-existing per-run
directory.

## 3. Diagnosis

The stale directory contained partial binary artefacts
(`continuous_edge_object.npz`, `thresholded_adjacency.npz`)
but no `run.json`, which is what a converged fit writes
last. The directory's filesystem timestamps predated the
successful current calibration run, indicating residue from
an earlier interrupted attempt rather than a fault in the
current run.

A scan of the rest of the calibration tree found no other
contaminated raw output directories and no other degenerate
records. The remaining 39 per-fit records were all clean
converged DAGMA or DCDI fits with `graph_status = valid_dag`
and `sampler_status = available`.

The initial `selected_configurations.json` selected
`06ee98d13852` (`lambda1 = 0.25`) as the rank-1 candidate for
the centred_only DAGMA cell. This selection was driven by a
single finite seed metric (`seed_value = 202`); the
`seed_value = 201` row was the degenerate
`FileExistsError` record and therefore contributed
non-finite per-seed metrics. Structural validation did not
catch this, because absence of a per-seed metric is
schema-legal for failed fits.

## 4. Fix implemented

Two changes were made in
`experiments/selection_study/calibration.py`:

1. `FileExistsError` raised by `pipeline.run_single_fit`,
   which means a pre-existing per-run directory was
   detected, is now caught at the calibration job loop and
   re-raised as `_CalibrationInfrastructureError`. The
   calibration runner treats this as fail-fast and aborts
   the run, rather than recording a degenerate calibration
   record. Genuine model-fit failures, for example a graph
   schema failure produced by a converged but invalid fit,
   are unaffected and may still be recorded as degenerate
   records.
2. A new entry point
   `repair_single_calibration_job(...)` was added. It
   validates identity inputs (model, condition,
   configuration_hash_prefix, seed_value), locates exactly
   one matching calibration fit job, re-runs only that fit,
   atomically overwrites the target per-fit record, re-reads
   all 40 records, re-runs the within-model ranking, and
   rewrites `selected_configurations.json`.

A regression test for the `FileExistsError` classification
was added in
`tests/test_calibration_fit_adapter.py`
(`test_file_exists_error_from_pipeline_is_infrastructure_failure`).
A dedicated test module `tests/test_calibration_repair.py`
covers `repair_single_calibration_job` for happy-path
behaviour, identity validation, on-disk precondition checks,
and infrastructure-failure propagation.

## 5. Live repair

The live repair was carried out against the real calibration
tree at
`results/model_selection/calibration/4a67117a10b1/`.

- The stale partial directory was moved out of the live
  results tree before repair to
  `archive/incidents/file_exists_error_2026-05-22/06ee98d13852_seed0_partial/`,
  preserving the residue for audit without contaminating
  any subsequent fit.
- `repair_single_calibration_job` was invoked for
  `model = dagma`, `condition = centred_only`,
  `configuration_hash_prefix = 06ee98d13852`,
  `seed_value = 201`.
- `previous_record_status = failed`.
- `new_record_status = converged`.
- The target per-fit record was atomically overwritten with
  the converged fit's metrics.
- `selected_configurations.json` was rewritten in place and
  re-validated structurally.
- All four `selected_degeneracy_flags_after_repair` are
  `false`.

## 6. Corrected selected configurations after repair

| condition       | model | selected hash | hyperparameters | mean SID | mean MMD              | mean SHD | degenerate? |
| --------------- | ----- | ------------- | --------------- | -------- | --------------------- | -------- | ----------- |
| centred_only    | dagma | 06ee98d13852  | lambda1 = 0.25  | 0.0      | 0.0059540658211281626 | 0.0      | false       |
| centred_only    | dcdi  | dd39d6325e7d  | reg_coeff = 0.1 | 60.0     | 0.08875084654991505   | 30.5     | false       |
| standardised    | dagma | 7b345b1b2e85  | lambda1 = 0.1   | 46.0     | 0.09699846001057562   | 18.0     | false       |
| standardised    | dcdi  | 16f92df3d6af  | reg_coeff = 0.3 | 46.0     | 0.1028451515945459    | 25.0     | false       |

These values are read directly from the post-repair
`selections` block of
`results/model_selection/calibration/4a67117a10b1/selected_configurations.json`.

## 7. Methodological implication

Structural validation of a calibration artefact is necessary
but not sufficient. Semantically contaminated records, such
as a degenerate `FileExistsError` record from a stale
filesystem directory, can pass schema and identity checks
because the schema permits non-finite per-seed metrics for
failed fits. The runner now distinguishes filesystem and
state contamination from genuine model-fit failure: a
pre-existing per-run output directory is treated as
infrastructure failure and aborts the run, instead of being
absorbed silently into the calibration record set.

The post-repair `selected_configurations.json` is the valid
input to held-out evaluation. The pre-repair version, in
which the `06ee98d13852` centred_only DAGMA candidate was
selected on the basis of a single finite seed, is not.
Recording this incident in the documentation tree makes it
auditable rather than hidden.

## 8. Scope limits

- This incident report does not perform base-model
  selection.
- It does not claim DAGMA is the final selected base model.
  Centred_only DAGMA recovered the ground-truth graph
  perfectly on this seed pair, but base-model selection is
  decided by held-out evaluation, not by calibration.
- It does not report held-out evaluation.
- It does not change the within-model ranking rule, the
  intervention policy, or the DCDI fit-RNG policy.
- It does not reopen the calibration protocol.
- The next stage (held-out evaluation) must read the
  repaired `selected_configurations.json` at
  `results/model_selection/calibration/4a67117a10b1/`, not
  any pre-repair copy.
