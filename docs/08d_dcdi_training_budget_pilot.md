# 08d. DCDI training-budget pilot (C-P15)

## Status

Diagnostic readout for the C-P15 measurement probe required by
`docs/08c_real_run_constants_and_training_budget_audit.md` Section 2.
The probe is measurement-only. It does NOT freeze
`num_train_iter`, does NOT count as Phase A reproduction-pass
evidence, and does NOT amend any planning document.

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
| `graph_status` | one of the wrapper-API taxonomy values | structural status of the thresholded adjacency |
| `sampler_status` | one of the wrapper-API taxonomy values | mechanical sampler availability |
| `training_status` | one of the wrapper-API taxonomy values | training-loop status (`converged`, `max_iter`, `diverged`, `wrapper_error`) |
| `runtime_seconds` | float | wall-clock fit runtime |
| `validation_nll_trajectory_summary` | string or `not_exposed` | per-iteration validation-NLL trajectory summary (see Field exposure) |

The probe enforces this header at append time: if the CSV exists
and its first line does not match `CSV_COLUMNS` exactly, the
probe raises `ValueError` rather than appending incompatible
rows. Deleting or renaming the file is required after any schema
change.

### Field exposure

Two fields are recorded as `not_exposed` because the current
project wrapper / training-loop infrastructure does not expose
them, and the probe is forbidden from modifying `src/` to expose
them:

- `second_stop_iteration`. The project's
  `_dcdi_training.run_dcdi_training_loop` implements the first
  patience gate (acyclicity is reached, validation-NLL is
  evaluated, and patience counts down) but does NOT implement
  the DCDI upstream second stage (permanent thresholding and
  `train_patience_post`). Permanent thresholding therefore
  never fires inside the project wrapper, and there is no
  observable "second stop" iteration to record. The probe
  does not pass `train_patience_post` to `DCDIConfig`; the knob
  is documented in this readout for paper/source context only.
- `validation_nll_trajectory_summary`. The training loop
  computes validation NLLs internally but the public
  `TrainingResult` does not return the trajectory. The probe
  records `not_exposed` rather than reimplement the trajectory
  collection outside `src/`.

`first_stop_iteration` IS exposed (the wrapper diagnostics
record reports `model_specific_diagnostics.first_stop`); it is
populated whenever the acyclicity-and-patience gate fires before
the cap is hit. If the cap is hit before the gate fires, the
field is `not_exposed` (the wrapper returns `None`).

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

- The validation-NLL trajectory is computed inside the wrapper
  but not returned; the project does not currently surface it.
  If the user needs the trajectory to make the ceiling choice,
  a small follow-up to `src/symbolic_priors_cd/wrappers/_dcdi_training.py`
  exposing `val_history` on `TrainingResult` would be the
  minimal additional surface; that follow-up is out of scope
  for this probe.
- The second-stage patience-post mechanism (permanent
  thresholding plus `train_patience_post`) is not exercised by
  the project wrapper. Consumers of the pilot output should
  read `first_stop_iteration` and `final_iteration` together;
  if the two are equal modulo `stop_crit_win`, the run
  converged under the first patience gate. If
  `final_iteration` equals `num_train_iter_cap` and
  `first_stop_iteration` is `not_exposed`, the cap was hit
  without convergence and the budget should likely be raised.
- C-P11 was run at `n_iter = 30_000` on a 3-node fixture
  [PROJECT DOC: `docs/04f`]. The pilot's results on the 10-node
  ER2 cell do not retroactively validate or invalidate C-P11.
  C-P11 reapplication is a separate decision per `docs/08c`
  Section 3.

## What this readout does NOT change

- No file under `src/` is modified.
- No file under `tests/` is modified.
- No file under `experiments/selection_study/` is modified.
- No notebook is modified.
- No configuration file is modified.
- No results directory is modified.
- `docs/02_base_model_selection.md` and
  `docs/03_decision_log.md` are not modified.
- `docs/08c_real_run_constants_and_training_budget_audit.md` is
  not modified.
- No dependency is added or removed.
