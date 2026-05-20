# 08d. DCDI training-budget pilot (C-P15)

## Status

Diagnostic readout for the C-P15 measurement probe required by
`docs/08c_real_run_constants_and_training_budget_audit.md` Section 2.
The probe itself is measurement-only: re-running it does not, by
itself, change `num_train_iter`. The adopted project decision
freezing `dcdi_num_train_iter = 300000` lives later in this
document under "Adopted project decision: DCDI training-budget
ceiling" and is consequence-tracked in `docs/03_decision_log.md`.
The probe does NOT count as Phase A reproduction-pass evidence,
and running it does not amend any planning document.

Version 1.0. Date: 2026-05-20.

## Purpose

`docs/08c` Section 2 concluded that the DCDI training budget
ceiling cannot be set from the DCDI paper alone (the paper does
not pin a `num_train_iter` value; it specifies a patience-based
stop schedule). C-P15 produces a small per-seed table of DCDI
stopping behaviour on the real 10-node ER2 selection-study cell
so the user can choose the ceiling on the basis of observed
evidence rather than guesswork.

## Probe

- Script: `inspection/probes/c_p15_dcdi_training_budget_pilot.py`
- Environment: project `.venv`, CPU only.
- No source, test, configuration, or external repository file is
  modified by running this probe.

### Exact configuration

SCM cell:

- `n_nodes = 10`
- `expected_edges = 20` (ER2 per `docs/02` Section 9)
- `noise_scale = 1.0`
- `weight_magnitude_range = (0.5, 2.0)`

DCDI configuration:

- `stop_crit_win = 100` [SOURCE: DCDI argparse default; PROJECT DOC: `docs/08c` Section 2]
- `train_patience = 5` [SOURCE: DCDI argparse default]
- `train_patience_post = 5` [SOURCE: DCDI argparse default; documented for paper/source context and NOT exercised by the project wrapper; the probe does not pass it to `DCDIConfig`; see Field exposure below]
- `train_batch_size = 64` [PAPER: DCDI Table 2]
- `h_threshold = 1e-8` [PAPER: DCDI Table 2]
- `lr = 1e-3` [PAPER: DCDI Table 2]
- hidden units 16, hidden layers 2, leaky-ReLU, Xavier init [PAPER: DCDI Table 2]
- RMSprop optimiser [PAPER: DCDI Table 2]

Validation split:

- 80/20 split from the observational sample [PAPER: DCDI Appendix B.5].
- The split is drawn from a `numpy.random.default_rng` seeded by
  `seed + 4111`. The seed value is recorded in the output row
  under `validation_data_seed`.

Seeds:

- Pilot seeds are drawn from the "reproduction" seed population.
  Calibration and held-out evaluation seed pools are NOT consulted.
- Default smoke seeds: `(101, 102)`.
- Default full seeds: `(101, 102, 103)`.
- The same integer seed is used as the DCDI fit seed and as the
  graph seed. The validation split seed and train-data sampling
  seed are derived deterministically from it via additive offsets.

## Smoke command

Smoke mode exercises the 10-node code path at a tiny iteration
cap on a small observational sample. It exists to verify the
probe works before running the full pilot.

```
python -m inspection.probes.c_p15_dcdi_training_budget_pilot --mode smoke
```

or equivalently from the project root:

```
python inspection/probes/c_p15_dcdi_training_budget_pilot.py --mode smoke
```

Smoke parameters (frozen in the script):

- `n_total = 200` observational samples, split 80/20.
- `num_train_iter_cap = 200`.
- Seeds: `(101, 102)`.

Smoke mode will not converge under the patience gate at 200
iterations; the smoke check passes if every row writes correctly
and the `final_iteration` equals the cap.

## Full-pilot command

Do NOT run full mode unless authorised. The full pilot runs on
the real selection-study cell and may take time on CPU.

```
python -m inspection.probes.c_p15_dcdi_training_budget_pilot --mode full
```

Optional flags:

- `--full-num-train-iter-cap <int>` to override the default cap
  (default `300_000`).
- `--seeds 101,102,103` to override the pilot seed list. Seeds
  must be drawn from the reproduction pool; calibration and
  held-out seeds must not appear.

Full mode reads:

- `n_total = 1000` observational samples [PROJECT DOC: `docs/02` Section 9].
- `num_train_iter_cap` defaults to `300_000` (override permitted
  with `--full-num-train-iter-cap`).

## Output

The probe writes one CSV row per (mode, seed) pair to
`inspection/probes/output/c_p15_dcdi_training_budget_pilot.csv`.
The file is created with a header if missing; otherwise rows are
appended. Existing rows are not rewritten. If the existing file's
header does not match the current schema, the probe raises
`ValueError` and refuses to append; delete or rename the file
before re-running.

### Frozen output schema (same columns in the same order in both modes)

| Column | Type | Meaning |
|---|---|---|
| `mode` | `smoke` or `full` | mode this row was produced under |
| `seed` | int | DCDI fit seed |
| `graph_seed` | int | SCM generation seed (same int as `seed`) |
| `validation_data_seed` | int | seed driving the 80/20 split RNG |
| `train_data_seed` | int | seed passed to `sample_observational` to draw the observational batch before the split |
| `n_total` | int | observational batch size drawn before the split |
| `n_fit_samples` | int | number of rows in `X_train_model` after the 80/20 split |
| `n_val_samples` | int | number of rows in `X_val_model` after the 80/20 split |
| `num_train_iter_cap` | int | iteration ceiling for this run |
| `final_iteration` | int or `not_exposed` | actual iteration reached (the wrapper exposes this) |
| `first_stop_iteration` | int or `not_exposed` | first iteration where the acyclicity-and-patience gate fired |
| `second_stop_iteration` | int or `not_exposed` | iteration where permanent thresholding fires (see Field exposure) |
| `final_h` | float or `not_exposed` | acyclicity surrogate at exit |
| `final_gamma` | float or `not_exposed` | augmented-Lagrangian multiplier at exit |
| `final_mu` | float or `not_exposed` | augmented-Lagrangian penalty coefficient at exit |
| `gamma_update_count` | int | number of in-loop gamma updates recorded by the training loop |
| `mu_update_count` | int | number of in-loop mu updates recorded by the training loop |
| `last_gamma_update_iteration` | int or `not_exposed` | iteration index of the final gamma update, or `not_exposed` when none occurred |
| `last_mu_update_iteration` | int or `not_exposed` | iteration index of the final mu update, or `not_exposed` when none occurred |
| `graph_status` | one of the wrapper-API taxonomy values | structural status of the thresholded adjacency |
| `sampler_status` | one of the wrapper-API taxonomy values | mechanical sampler availability |
| `training_status` | one of the wrapper-API taxonomy values | training-loop status (`converged`, `max_iter`, `diverged`, `wrapper_error`) |
| `runtime_seconds` | float | wall-clock fit runtime |
| `validation_nll_trajectory_summary` | compact summary string or `not_exposed` | semicolon-delimited per-stop-check validation-NLL trajectory summary (see "Validation-NLL summary format" below) |

The probe enforces this header at append time: if the CSV exists
and its first line does not match `CSV_COLUMNS` exactly, the
probe raises `ValueError` rather than appending incompatible
rows. Deleting or renaming the file is required after any schema
change.

### Field exposure

- `second_stop_iteration` is recorded as `not_exposed` because
  the project's `_dcdi_training.run_dcdi_training_loop`
  implements the first patience gate (acyclicity is reached,
  validation-NLL is evaluated, and patience counts down) but
  does NOT implement the DCDI upstream second stage (permanent
  thresholding and `train_patience_post`). Permanent
  thresholding therefore never fires inside the project
  wrapper, and there is no observable "second stop" iteration
  to record. The probe does not pass `train_patience_post`
  to `DCDIConfig`; the knob is documented in this readout for
  paper/source context only.
- `validation_nll_trajectory_summary` is populated from the
  validation-NLL list collected at the existing stop-check
  cadence; the trajectory now lives on
  `TrainingResult.validation_nll_history` and is surfaced via
  `DCDIWrapper.get_diagnostics()["convergence_info"][
  "validation_nll_history"]`. See the "Validation-NLL summary
  format" subsection for the exact string layout. The summary
  falls back to `not_exposed` only when a future wrapper
  version omits the field.
- `final_gamma`, `final_mu`, `gamma_update_count`,
  `mu_update_count`, `last_gamma_update_iteration`, and
  `last_mu_update_iteration` are populated from
  `diagnostics["model_specific_diagnostics"]`. When the
  training loop made no update of a given kind, the count is
  `0` and the corresponding `last_*_update_iteration` is
  `not_exposed`.

`first_stop_iteration` IS exposed (the wrapper diagnostics
record reports `model_specific_diagnostics.first_stop`); it is
populated whenever the acyclicity-and-patience gate fires before
the cap is hit. If the cap is hit before the gate fires, the
field is `not_exposed` (the wrapper returns `None`).

### Validation-NLL summary format

The training loop already evaluates the validation NLL once
before training and then once at every `stop_crit_win` iterations
after a step. The values are returned via
`get_diagnostics()["convergence_info"]["validation_nll_history"]`
together with the cadence
`get_diagnostics()["convergence_info"]["validation_nll_stop_crit_win"]`.
The probe reads both fields and emits a compact semicolon-delimited
string of the form:

```
count=<N>;nonfinite_count=<NF>;first=<F>;last=<L>;min=<M>;argmin=<I>;tail=[v1,v2,...];stop_crit_win=<W>
```

where:

- `count = N` is the number of validation evaluations recorded
  (one pre-training baseline plus one per stop-check window). For
  a full-mode run with `num_train_iter_cap = 300000` and
  `stop_crit_win = 100`, `N` is at most `3001`.
- `nonfinite_count = NF` is the number of entries in the
  trajectory that are NaN or `+/-inf`. Finite-only summary
  statistics (`min`, `argmin`) are computed over the remaining
  `N - NF` finite values.
- `first` is the pre-training baseline; `last` is the final
  recorded value; `min` is the minimum over the finite values.
- `argmin` is the integer index into the history list at which
  the finite minimum occurred. Multiply by `stop_crit_win` to
  get the approximate training iteration; index `0` is the
  pre-training baseline. When no finite values exist `min` and
  `argmin` are both `not_exposed`.
- `tail = [...]` is the final up to five values (finite or
  not), useful for visually inspecting the recent plateau /
  drift pattern without dumping the full trajectory into the
  CSV.
- `stop_crit_win` records the cadence so consumers can reconstruct
  approximate iteration counts from indices.

The string is plain ASCII and never embeds commas inside numeric
fields. The probe records `not_exposed` if the diagnostics
dictionary lacks the expected fields.

### Interpretation patterns for the validation-NLL trajectory

- A decreasing trajectory across many stop checks indicates the
  augmented-Lagrangian objective is still improving the validation
  fit; the budget is plausibly insufficient.
- A flat or oscillating trajectory after an initial drop indicates
  a plateau; the patience gate may be approaching firing.
  Comparing `argmin` against `count - 1` reveals whether the
  minimum is recent or in the distant past.
- A worsening trajectory (`last > min` by a large margin)
  indicates the optimisation is now degrading the validation fit,
  often because `gamma` and `mu` have been pushed hard. This is
  evidence that the patience gate will eventually fire, but the
  current state may have already overshot the best validation
  point.

### Joint interpretation of validation NLL and the gamma/mu state

The validation-NLL trajectory is a signal about score quality;
the gamma/mu state is a signal about acyclicity pressure. Read
them together when deciding whether a run is budget-limited,
schedule-limited, or infeasible. In this wrapper, gamma and mu
updates can occur both before and after `first_stop`:
`gamma`/`mu` are updated at stop-check windows whenever the
acyclicity threshold has not yet been met and the validation-NLL
plateau test fires, so update counts are NOT a proxy for
"post-first-stop activity".

- If validation NLL plateaus but `final_h` remains far above
  `h_threshold`, inspect `gamma_update_count` and
  `mu_update_count` before concluding the issue is budget. A
  high update count plus a stalled `final_h` suggests
  optimisation difficulty or near-infeasibility under the
  current architecture, not a budget shortfall; a low update
  count plus a stalled `final_h` suggests the coefficient
  schedule has not yet pushed enough acyclicity pressure and a
  longer budget could help.
- If neither gamma nor mu has updated (counts equal `0`), the
  run is still in the early score-fitting regime under weak
  acyclicity pressure; an empty `last_*_update_iteration`
  consistent with the count is expected.
- If gamma/mu have updated repeatedly but `final_h` is still
  high, the run is likely compute-envelope-limited or facing
  hard optimisation difficulty. A longer budget may help but is
  not guaranteed to.

The 10k smoke-of-scale row should not be over-interpreted: the
patience gate has not yet fired and the augmented-Lagrangian
schedule may not have stabilised. The 300k full run is the
budget-evidence pilot.

The pilot remains diagnostic-only. The validation-NLL summary is
evidence the user reads when choosing `num_train_iter`; it does
not freeze the ceiling.

## Interpretation rules

The probe is diagnostic-only. The output table is evidence for
the user's eventual choice of real-study `num_train_iter`. The
following interpretation rules apply:

1. **The pilot does not freeze `num_train_iter`.** The user
   reads the table and makes the choice; the choice is recorded
   in a `docs/02` v1.6 amendment and a contemporaneous `docs/03`
   entry per `docs/08c` Section 10.
2. **The pilot does not count as Phase A reproduction
   evidence.** Phase A reproduction uses the paper-aligned cell
   defined in `docs/02` Section 3.3 with its own seed pool. The
   pilot's reproduction-pool seeds are pilot-only artefacts.
3. **The pilot does not consult calibration or held-out
   evaluation seeds.** Any future `--seeds` override must
   honour this constraint.
4. **C-P11 rerun policy is unchanged.** Per `docs/08c`
   Section 3, the C-P11 sampler-quality diagnostic must be
   rerun on a 10-node fixture at the chosen real-study
   `num_train_iter` ceiling before held-out evaluation begins,
   unless `docs/03` explicitly scopes the original C-P11 result
   to its 30 000-iteration 3-node fixture.
5. **Smoke-mode rows must not be used as evidence for the
   real-study ceiling.** Smoke mode uses 200 observational
   samples and a 200-iteration cap; that is far below the
   selection-study cell. Smoke rows verify the code path only.

## Caveats

- The validation-NLL trajectory is now surfaced on
  `TrainingResult.validation_nll_history` and read by the probe
  through `DCDIWrapper.get_diagnostics()`. The summary string
  in `validation_nll_trajectory_summary` is the compact view;
  the full trajectory is recoverable through the wrapper
  diagnostics if a notebook or follow-up probe needs every
  per-stop-check value.
- The second-stage patience-post mechanism (permanent
  thresholding plus `train_patience_post`) is not exercised by
  the project wrapper. Consumers of the pilot output should
  read `first_stop_iteration` and `final_iteration` together;
  if the two are equal modulo `stop_crit_win`, the run
  converged under the first patience gate. If
  `final_iteration` equals `num_train_iter_cap` and
  `first_stop_iteration` is `not_exposed`, the cap was hit
  without convergence and the budget should likely be raised.
- The 10k full-mode row is a runtime/scaling diagnostic rather
  than budget evidence. Comparing it against the eventual 300k
  row is the intended use; it is not safe to extrapolate a
  ceiling from the 10k row alone.
- C-P11 was run at `n_iter = 30_000` on a 3-node fixture
  [PROJECT DOC: `docs/04f`]. The pilot's results on the 10-node
  ER2 cell do not retroactively validate or invalidate C-P11.
  C-P11 reapplication is a separate decision per `docs/08c`
  Section 3.

## Completed 300k full-pilot results

The 300 000-iteration full pilot ran on three reproduction-pool
seeds (101, 102, 103). All three reached the patience gate
under the cap. Exact values are reproduced from
`inspection/probes/output/c_p15_dcdi_training_budget_pilot.csv`.

| Seed | num_train_iter_cap | final_iteration | first_stop_iteration | final_h | final_gamma | final_mu | gamma_update_count | mu_update_count | last_gamma_update_iteration | last_mu_update_iteration | graph_status | sampler_status | training_status | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 300000 | 118900 | 118500 | 8.834e-09 | 7.967e+04 | 3.689e+11 | 114 | 65 | 117200 | 114400 | valid_dag | available | converged | 833.12 |
| 102 | 300000 | 75300 | 74900 | 9.188e-09 | 4.856e+04 | 2.252e+07 | 80 | 51 | 73800 | 61800 | valid_dag | available | converged | 738.18 |
| 103 | 300000 | 86700 | 86300 | 7.892e-09 | 5.115e+04 | 2.252e+07 | 79 | 51 | 86200 | 72200 | valid_dag | available | converged | 707.77 |

### Validation-NLL summaries

Per-seed compact summary (cadence is `stop_crit_win = 100`, so
the approximate iteration of the finite minimum is
`argmin * 100`):

| Seed | first | min | argmin | approx argmin iteration | last |
|---|---|---|---|---|---|
| 101 | 65.018 | 0.9793 | 69 | ~6900 | 1.8874 |
| 102 | 76.07 | 0.9093 | 75 | ~7500 | 1.7973 |
| 103 | 108.95 | 0.8676 | 95 | ~9500 | 1.9797 |

The minimum validation NLL in every case is recorded near the
start of training (well below 10 000 iterations); the last value
is much higher than the minimum. Validation NLL is therefore
not monotonically decreasing across the run: the augmented-
Lagrangian schedule pushes acyclicity pressure (gamma and mu
grow markedly across the run, with `mu` reaching values from
about 2e+07 to about 4e+11 at exit) at the cost of validation
fit. This is the documented DCDI dynamic and is the reason the
patience gate is acyclicity-driven, not validation-NLL-driven.

## Why this evidence resolves the budget question

The DCDI training-budget question was audited
([`docs/08c`](08c_real_run_constants_and_training_budget_audit.md)
Section 2) before Commit 8 because `docs/02` Section 9 did not
freeze a numerical `num_train_iter` ceiling, and the only DCDI
iteration count in the project code was the schema-gate constant
`SCHEMA_GATE_DCDI_N_ITER = 30`. Thirty iterations with
`stop_crit_win = 10` makes the patience gate structurally
incapable of firing; the schema-gate value cannot inform a real
training-budget decision.

The earlier C-P11 sampler-quality diagnostic ran at
`n_iter = 30 000` on a 3-node fixture, never reached the
acyclicity gate (`final_h = 5.31e-04`), and was not binding
evidence about DCDI behaviour at the real-study budget. The
10-node 300k pilot is what the audit's option (b)+(d) called
for: paper-aligned hyperparameters from DCDI Table 2 plus a
high iteration ceiling whose actual stop iteration is decided
by the patience gate.

Before running the expensive 300k pilot, the validation-NLL
trajectory and the augmented-Lagrangian state (`final_gamma`,
`final_mu`, `gamma_update_iters`, `mu_update_iters`) were
exposed on the wrapper's existing diagnostics dict (see the
20/05/2026 `docs/03` entry "Validation NLL trajectory exposed
on DCDI TrainingResult"). The 10k row was a runtime/scaling
diagnostic: `gamma_update_count = 8`, `mu_update_count = 7`,
`final_h = 0.953`, `training_status = max_iter`, validation
NLL plateaued in the low single digits. That row showed DCDI
in mid-schedule rather than converged; it did not by itself
justify a budget ceiling.

The 300k row resolves the question. All three pilot seeds
converged via the first patience gate well below the cap with
`graph_status = valid_dag` and
`sampler_status = available`. The slowest seed (101) stopped
at `final_iteration = 118900`; the fastest (102) at 75 300.
The seed-to-seed variance in `final_mu` is large
(approximately 2.25e+07 on seeds 102 and 103, approximately
3.69e+11 on seed 101): different ER2 graph realisations
require very different amounts of acyclicity pressure before
the constraint is satisfied. This is the heterogeneity the
budget ceiling must absorb.

What the 300k pilot does NOT say:

- `valid_dag` plus `sampler_status = available` does NOT imply
  the learned graph is structurally correct. SHD, SID, and MMD
  remain the selection-study metrics; the pilot reports neither.
- The pilot does NOT count as Phase A reproduction evidence.
  Phase A uses paper-aligned reproduction seeds and a separate
  acceptance protocol per `docs/02` Section 3.3.
- A high `final_mu` does not by itself indicate a problem; in
  DCDI's augmented-Lagrangian schedule it is the price paid for
  driving `h` below the threshold on a hard graph.

## Adopted project decision: DCDI training-budget ceiling

`dcdi_num_train_iter = 300000` is adopted as the hard maximum
iteration ceiling for DCDI in the selection study, with the
existing patience-based early stopping enabled. The patience
parameters (`stop_crit_win = 100`, `train_patience = 5`,
`h_threshold = 1e-8`) and RMSprop / batch / architecture
values are unchanged from DCDI Table 2 (see `docs/08c` Section 2
and the wrapper defaults in
`src/symbolic_priors_cd/wrappers/_dcdi_training.py`).

Justification:

- All three C-P15 pilot seeds (101, 102, 103) converged below
  300 000. The worst observed `final_iteration` was 118 900 on
  seed 101.
- Phase B configurations may converge more slowly than the
  pilot's default `reg_coeff = 0.1` because sparsity-coefficient
  variation alters the loss landscape and the acyclicity
  trajectory. The ceiling leaves substantial headroom (factor
  of about 2.5 on the observed worst case) without growing the
  per-run worst-case runtime catastrophically.
- Early stopping means converged runs use only the iterations
  they need, regardless of the ceiling. The ceiling is a safety
  net, not a target.
- The value is pilot-derived. The DCDI paper does not specify a
  numerical `num_train_iter` ceiling (it specifies the patience
  schedule); the upstream argparse default is `1_000_000`, which
  exceeds the project compute envelope.
- `dcdi_num_train_iter` is a training budget, not a tunable
  hyperparameter. It MUST NOT be varied by held-out evaluation
  records, and it MUST NOT be one of the five Phase B
  configurations. Variation requires a contemporaneous
  `docs/03_decision_log.md` entry.

This decision is recorded in `docs/03_decision_log.md` as the
adopted DCDI training-budget ceiling. The corresponding
`docs/02` v1.6 amendment that frames the ceiling inside the
selection-study protocol is a separate, user-adjudicated step
and is not made here.

## Items NOT addressed in this commit

The following remain open for the separate user-adjudicated
`docs/02` v1.6 amendment and any associated `docs/03` entries:

- DAGMA paper-vs-library budget choice (`warm_iter` paper
  20 000 vs library 30 000; `max_iter` paper 70 000 vs
  library 60 000). See `docs/08c` Section 4.
- Phase B sparsity policy and grid endpoints (symmetrise-upward
  vs symmetrise-downward; exact half-decade endpoints around
  paper anchors). See `docs/08c` Section 5.
- C-P11 reapplication policy before held-out evaluation (rerun
  at the new budget, or explicit scope-statement in `docs/03`).
  See `docs/08c` Section 3.
- `mmd_n_samples = 1000` elevation to a top-level
  `Configuration` field. See `docs/08c` Section 6.
- `n_train` and DCDI validation-split `n_val_dcdi`
  `Configuration` fields. See `docs/08c` Section 8.
- Visual / reporting artefact requirements for notebook
  inspection (e.g. learning-curve and gamma/mu trajectory
  plots from the saved diagnostics).

## What this readout does NOT change

- `docs/02_base_model_selection.md` and `docs/08c` are not
  modified by this readout.
- No file under `experiments/selection_study/` is modified.
- No notebook, configuration file, results directory, or
  dependency manifest is modified.
- The pilot itself remains diagnostic-only; the budget ceiling
  is the user-adjudicated decision recorded above and
  consequence-tracked in `docs/03_decision_log.md`.
