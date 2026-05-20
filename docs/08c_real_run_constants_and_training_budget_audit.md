# 08c. Real-run constants and training-budget audit

## Status

Audit document only. No source, test, configuration, or other docs
change as a result of this file. Proposals here are for user review
prior to amending `docs/02_base_model_selection.md` and
`docs/03_decision_log.md` and prior to implementing Commit 8 of
`docs/08_base_model_selection_plan.md`.

Version 1.1. Date: 2026-05-20.

### Change log

- v1.0 (2026-05-20): initial audit drafted before paper re-read.
- v1.1 (2026-05-20): paper re-read for both DAGMA and DCDI; paper
  evidence added for DCDI sparsity selection (held-out NLL), DCDI
  sparsity grid width (10 values), DCDI hyperparameter table,
  DAGMA `warm_iter` and `max_iter` (paper values 20000 and 70000),
  DAGMA Adam learning rate and betas, DAGMA loss-convergence
  criterion (relative error < 1e-6), DAGMA "no hyperparameter
  optimization" remark (single L1 value across graphs). Phase B
  sparsity-fairness analysis strengthened; DCDI pilot policy
  refined into a measurement-only diagnostic; C-P11 reapplication
  policy split by phase; Configuration/hash storage decisions
  itemised per knob; flat vs nested layout compared; DCDI
  validation policy clarified as a split from `n_train`; MMD
  `n_samples` cost discussion expanded with kernel-pair cost.

### Evidence labels

- [PAPER]
- [SOURCE]
- [PROJECT DOC]
- [RUNTIME PROBE]
- [TEST]
- [UNSUPPORTED / PROJECT DECISION]

For this revision both PDFs were re-read at the relevant
appendices; specific page citations appear inline.

---

## 1. Executive verdict

**More source/paper inspection has now been performed, but a small
runtime pilot is still needed before Commit 8.** Three classes of
real-run constants remain unfrozen, and the paper re-read changed
three values relative to v1.0:

- DCDI `num_train_iter` is not paper-pinned. The DCDI paper does
  not specify a numerical iteration ceiling; it specifies a
  patience-based stopping rule and shows a single learning-curve
  figure where convergence on a 10-node perfect-interventions
  graph occurred at "around iteration 62 000" [PAPER: DCDI Figure
  6 and the surrounding "Learning dynamics" paragraph, page 30].
  62 000 is an observation on one graph, not a frozen budget.
  Pilot evidence is required.
- DAGMA per-stage inner iteration counts in the paper are
  `2 x 10^4` for `t in {0, 1, 2}` and `7 x 10^4` for `t = 3`
  [PAPER: DAGMA Section C.1.1, page 21]. The current project
  wrapper carries `warm_iter = 30000` and `max_iter = 60000` from
  the DAGMA library default, NOT from the paper. The library and
  paper disagree on these values; the project decision must be
  recorded explicitly.
- Real-run MMD `n_samples` is named in `docs/02` Section 4.2
  ("defaults to 1000") but is not stored on `Configuration` today
  and the runner is reading the schema-gate value
  `SCHEMA_GATE_MMD_N_SAMPLES = 64` [SOURCE:
  `experiments/selection_study/sampling.py:32`].

Recommended order of operations remains:

1. Run a measurement-only DCDI training-budget pilot on the
   reproduction-pass cell. Output is diagnostic, not a budget
   freeze.
2. Decide DAGMA `warm_iter`, `max_iter`, `lr`, `beta_1`, `beta_2`
   per Section 4 below (the paper-explicit values, or the library
   defaults, with a recorded reason).
3. Decide Phase B sparsity-fairness policy per Section 5.
4. Decide C-P11 reapplication per Section 3.
5. Amend `docs/02` to v1.6 and add a contemporaneous `docs/03`
   entry.
6. Extend `Configuration` per Section 8.
7. Open Commit 8.

No DCDI Commit 11 / loss-hook work should resume in this commit.

---

## 2. DCDI real-study training budget

### Schema-gate value and why it is toy-only

[SOURCE: `experiments/selection_study/pipeline.py`]
`SCHEMA_GATE_DCDI_N_ITER = 30` and
`SCHEMA_GATE_DCDI_CONFIG_KWARGS = {"stop_crit_win": 10,
"train_batch_size": 8}`. With `stop_crit_win = 10` and
`n_iter = 30`, at most three convergence checks fire and the
patience gate cannot meaningfully trigger. The schema-gate value
must not propagate into Phase A/B.

### Paper / source evidence

- DCDI Appendix B.5 specifies the augmented-Lagrangian
  hyperparameters [PAPER: DCDI Table 2, page 32]:
  `mu_0 = 1e-8`, `gamma_0 = 0`, `eta = 2` (mu mult factor),
  `delta = 0.9` (omega_mu), augmented-Lagrangian constraint
  threshold `1e-8`, learning rate `1e-3`, 16 hidden units,
  2 hidden layers, RMSprop, minibatch size 64, leaky-ReLU
  activation, Xavier initialisation.
- DCDI paper does NOT pin a `num_train_iter` value. The
  Appendix B.5 narrative says "DCDI-G and DCDI-DSF used exactly
  the same default hyperparameters that are summarized in Table
  2." Table 2 contains no iteration count.
- DCDI paper does pin the patience-based stop schedule
  conceptually; the precise patience parameters
  (`train_patience = 5`, `train_patience_post = 5`,
  `stop_crit_win = 100`) come from the DCDI repository argparse
  defaults [SOURCE:
  `external/source_inspection/dcdi/main.py:107, 127, 129, 131`].
- DCDI paper Figure 6 + accompanying "Learning dynamics"
  paragraph [PAPER: page 29-30]: on a sparse 10-node graph
  with perfect interventions, the patience gate first fired
  "around iteration 20 000" (acyclicity first reached) and the
  acyclicity-plus-validation stopping criterion was satisfied
  "around iteration 62 000". This is one fixture, one paper,
  shown to illustrate dynamics, not a frozen ceiling.
- DCDI upstream `--num-train-iter` default is `1_000_000`
  [SOURCE: `external/source_inspection/dcdi/main.py:60`].
- DCDI training data per supplementary experiments: "Each data
  set has 10 000 samples uniformly distributed in the different
  interventional settings" [PAPER: DCDI Section 4, page 7]. The
  selection-study cell uses 1 000 observational samples, an
  order of magnitude smaller, which is relevant for budget
  scaling.

### Runtime probe evidence already available

- C-P8 [PROJECT DOC: `docs/04c_runtime_probe_results.md`]:
  five-step CPU runs are bit-identical with seeds set. Says
  nothing about how many steps are needed.
- C-P11 [PROJECT DOC: `docs/04f_dcdi_sampler_quality_diagnostic.md`]:
  one diagnostic at `n_iter = 30 000` on a 3-node fixture,
  `converged = False`, `final_h = 5.31e-04`. Not directly
  applicable to the 10-node cell.

### Real-study `num_train_iter` policy options

| Option | Description | Fairness | Compute | Leakage |
|---|---|---|---|---|
| (a) Upstream default | `num_train_iter = 1_000_000` with patience gate enabled | Symmetric across DCDI runs | High; the worst-case run on the 10-node cell could absorb a non-trivial fraction of the 30 GPU-hour ceiling [PROJECT DOC: `docs/02` Section 8] | Low |
| (b) Pilot-measured ceiling | Run a pilot, observe stopping behaviour, set `num_train_iter` to a value comfortably above the observed worst-case stopping iteration | Symmetric across DCDI runs | Moderate; pilot uses reproduction or pilot-only seeds | Moderate; pilot evidence must not consult calibration or held-out seeds |
| (c) Phase B calibration | Make `num_train_iter` one of the five Phase B configurations | High asymmetry; DAGMA's iteration budget is not tuned in Phase B | Moderate | High; tuning iteration budget on calibration seeds is a capacity sweep |
| (d) Paper-anchored ceiling | Use Figure 6's observed ~62 000 as a starting anchor, multiplied by a safety factor to cover variability | Symmetric across DCDI runs | Low to moderate | Low; the anchor is paper-derived, not project-data-derived |

### Recommendation (proposal for user review)

Combine (b) and (d). Run a measurement-only pilot per Section 2's
"Pilot policy" subsection below; if the pilot's worst-case
stopping iteration is within a small factor of the paper's
~62 000 figure (page 30 of the DCDI paper), adopt that observed
value plus a project-decided safety factor; if the pilot's
worst-case stopping iteration is materially larger, adopt the
larger value with reasoning recorded in the `docs/03` entry.
Do NOT include `num_train_iter` in Phase B calibration; it is a
training-budget knob, not a calibrated hyperparameter, and
DAGMA's budget is not similarly tunable.

### Pilot policy (measurement-only)

The pilot must measure, not freeze. For every pilot run:

- iteration index at which the augmented-Lagrangian acyclicity
  threshold is first satisfied (DCDI calls this the
  `first_stop` event in `TrainingResult`)
  [SOURCE: `src/symbolic_priors_cd/wrappers/_dcdi_training.py:71`];
- iteration index at which permanent thresholding fires (the
  "second stop" in the DCDI upstream loop; not currently
  exposed by the project wrapper, but observable from the loss
  history)
  [SOURCE: `external/source_inspection/dcdi/dcdi/train.py:320-333`];
- final stopping iteration (the wrapper's `n_iterations`)
  [SOURCE: `_dcdi_training.py:68`];
- final value of the acyclicity surrogate (`final_h`)
  [SOURCE: `_dcdi_training.py:74`];
- thresholded graph status (`graph_status` taxonomy from
  `wrappers/_graph_status.py`)
  [SOURCE: `src/symbolic_priors_cd/wrappers/_graph_status.py`];
- sampler status (`sampler_status` taxonomy)
  [SOURCE: same];
- validation-NLL plateau pattern, if observable from the
  validation history exposed by the training loop
  [SOURCE: `_dcdi_training.py:276, 307`];
- training status taxonomy value (`converged`, `max_iter`,
  `diverged`, `wrapper_error`)
  [PROJECT DOC: `docs/04_wrapper_api_contract.md` Section 7];
- wall-clock runtime;
- seed used (must be a reproduction-pass seed; calibration and
  held-out seeds are off-limits per Section 9 below).

The pilot is diagnostic-only. Its output is budget-setting
evidence, NOT paper-reproduction evidence. If reproduction
seeds are used, the pilot does not count as the Phase A
reproduction pass under `docs/02` Section 3.3; the reproduction
pass remains a separate, documented run.

A pilot should consist of two or three runs across distinct
reproduction seeds, on the 10-node ER2 selection-study cell,
with `num_train_iter` set generously (for example `300_000` or
upstream default `1_000_000`) and the patience gate enabled.
The pilot output table is what determines the eventual ceiling
in the `docs/02` v1.6 amendment; the audit does not pre-pick
the value.

---

## 3. C-P11 / DCDI sampler-quality interaction

### What budget was C-P11 run at

[PROJECT DOC: `docs/04f_dcdi_sampler_quality_diagnostic.md`
lines 72, 99, 103-104]

- `n_iter = 30 000`
- `n_iterations` observed: `30 000`
- `final_h = 5.31e-04`, well above `h_threshold = 1e-8`
- `converged = False`; patience never elapsed
- Fixture: 3 nodes (not 10)

### Will the real-study budget materially differ

Almost certainly yes. The DCDI paper's own Figure 6 example
needed ~62 000 iterations on a 10-node graph to reach the
patience-and-acyclicity stop. The selection study's 10-node
ER2 cell is at the same scale; even under favourable
optimisation behaviour, the real-study `num_train_iter`
ceiling will likely be at least 100 000, possibly 300 000 or
more. C-P11's 30 000 is materially below this.

### Could C-P11's conclusion change

Plausibly yes. C-P11's negative finding is "DCDI's learned
structure on this fixture at this budget did not produce a
sampler that hit the MMD floor against the true DAG even after
wrapper-side saturation" [PROJECT DOC: `docs/04f`]. The
underlying chain is "structure quality limits sampler quality";
structure quality is a function of training budget plus
optimisation behaviour, and the DCDI paper's own learning
curves show structure substantially improving between 20 000
and 62 000 iterations [PAPER: DCDI Figure 6, page 29-30].

Two paths are consistent with the available evidence:

- DCDI converges to a thresholded valid DAG at the real-study
  budget that approximates the true structure. The C-P11
  sampler-quality concern is then mitigated.
- DCDI still produces a poor or non-valid-DAG structure at the
  real-study budget. The MMD-unavailable / reliability-limited
  rule from `docs/02` Section 6 Case 6 governs reporting.

### Reapplication options

| Option | When required | Cost | Cost mitigation |
|---|---|---|---|
| (A1) Required before Phase A starts | Phase A cannot proceed until C-P11 is rerun on a 10-node ER2 fixture at the proposed `num_train_iter` ceiling | One additional DCDI fit at the new budget plus the existing C-P11 sampler analysis script | The fit overlaps with the pilot from Section 2 if planned together |
| (A2) Required before held-out evaluation | Phase A reproduction and Phase B calibration may proceed; the C-P11 rerun must be complete before held-out evaluation reads sampler-quality signal | Same as (A1) but later in the timeline | Allows pilot, Phase A, and the C-P11 rerun to happen in parallel branches if compute permits |
| (B) Optional but recommended | C-P11 is explicitly scoped to its 30 000-iteration 3-node fixture in `docs/03`; the selection-study report's MMD-missingness disclosure carries the sampler-quality signal | Zero | None needed |

### Recommendation (proposal for user review)

Option (A2): require the C-P11 rerun before held-out evaluation,
not before Phase A starts. This allows Phase A reproduction and
Phase B calibration to proceed while the rerun is queued, and
ensures the held-out evaluation is interpreted under
budget-matched evidence. If compute is severely constrained,
option (B) is acceptable provided the `docs/03` entry is
explicit and the selection-study report names C-P11's scope
limitation.

Doing nothing (treating C-P11 as binding without budget match)
is not defensible.

---

## 4. Symmetric DAGMA training-budget audit

[SOURCE: `src/symbolic_priors_cd/wrappers/dagma.py:93-110`,
`src/symbolic_priors_cd/wrappers/_dagma_fit.py:66-82`,
`external/source_inspection/dagma/src/dagma/linear.py:234-244`;
PAPER: DAGMA Section C.1.1, page 21]

### DAGMA's training-budget surface

Paper-explicit values from DAGMA Section C.1.1 (verbatim
restatement):

- `T = 4`, `mu_0 = 1`, `alpha = 0.1`, `beta_1 = 0.05`,
  `s = {1, .9, .8, .7}`.
- Adam learning rate `3 x 10^-4`, Adam betas `(0.99, 0.999)`.
- Per-stage iterations: `2 x 10^4` for `t in {0, 1, 2}`,
  `7 x 10^4` for `t = 3`, "or until the loss converges,
  whichever comes first."
- Loss convergence: relative error between subsequent iterations
  less than `10^-6`.
- Final threshold of `0.3` on `abs(W)`.

Paper Remark 5 explicitly states: "we have not performed any
hyperparameter optimization ... for each of the following SEM
settings, we simply chose a reasonable value for the L1 penalty
coefficient and used that same value for all ER and SF graphs
across many different numbers of nodes." So DAGMA's paper L1
treatment is a single fixed value, not a sweep.

| Knob | Wrapper default | Paper value | docs/02 frozen? | Evidence |
|---|---|---|---|---|
| `T` | 4 | 4 | Yes | [PAPER] |
| `s` | `(1.0, 0.9, 0.8, 0.7)` | `{1, .9, .8, .7}` | Yes | [PAPER] |
| `lambda1` (`beta_1`) | 0.05 | 0.05 | Yes | [PAPER] |
| `mu_init` | 1.0 | 1 | Yes | [PAPER] |
| `mu_factor` (`alpha`) | 0.1 | 0.1 | Yes | [PAPER] |
| `w_threshold_internal` | 0.0 | n/a (project wrapper convention) | Yes | [PROJECT DOC] |
| `project_threshold` | 0.3 | 0.3 | Yes | [PAPER] |
| `warm_iter` | 30000 | **20000** | No | [SOURCE / PAPER conflict] |
| `max_iter` | 60000 | **70000** | No | [SOURCE / PAPER conflict] |
| Adam `lr` | 3e-4 | 3e-4 | No | [PAPER] |
| Adam `beta_1` (project) | 0.99 | 0.99 | No | [PAPER] |
| Adam `beta_2` (project) | 0.999 | 0.999 | No | [PAPER] |
| Loss-convergence rule | not implemented in wrapper | relative error < 1e-6 | No | [PAPER] |
| `loss_type` | `"l2"` | n/a (paper Section C.1) | Implied | [PAPER / PROJECT DOC] |
| `h_diagnostic_threshold` | 1e-5 | n/a | No | [UNSUPPORTED / PROJECT DECISION] |

Two paper-vs-wrapper discrepancies must be resolved:

- `warm_iter`: paper `20000`, wrapper `30000`. The library
  default is `30000` so the wrapper inherited the library
  value [SOURCE: `dagma/src/dagma/linear.py`]. The project
  must decide whether to follow the paper or the library.
- `max_iter`: paper `70000`, wrapper `60000`. Same source of
  conflict.

Either choice is defensible; the project must record the
choice. The audit recommends paper values to align with the
"DAGMA-linear paper-grounded supplementary defaults" framing in
`docs/02` Section 3.3, but the library values were the
wrapper's working defaults before this audit. A two-line note
in the `docs/02` v1.6 amendment plus the `docs/03` entry is
sufficient.

The DAGMA paper's loss-convergence rule (relative error < 1e-6)
is currently not implemented in the wrapper. The wrapper runs
the full `warm_iter` / `max_iter` budget on every stage. This is
a separate departure from the paper; the audit does not
recommend implementing the rule before Commit 8 unless a
training-cost concern forces it. Note in `docs/02` recommended.

### Side-by-side DAGMA vs DCDI

| Concept | DAGMA | DCDI | Evidence | Affects | In Configuration? |
|---|---|---|---|---|---|
| Outer / stage loop | `T = 4` | n/a (single augmented Lagrangian) | [PAPER both] | training time | DAGMA only; already in `Configuration` indirectly |
| Per-stage inner iteration count | `warm_iter`, `max_iter` (paper `20000` / `70000`) | n/a | [PAPER DAGMA; SOURCE wrapper] | training time | DAGMA-only field |
| Total iteration ceiling | `(T - 1) * warm_iter + max_iter` (paper: 130 000; wrapper: 150 000) | `num_train_iter` (paper-unset; upstream `1_000_000`) | [PAPER / SOURCE] | training time, graph validity | Both, model-specific |
| Convergence rule | relative loss < 1e-6 (paper; wrapper does not enforce) | h <= 1e-8 + patience | [PAPER both; partial SOURCE] | actual training length | DCDI patience values; DAGMA convergence rule not currently implemented |
| Optimiser | Adam (lr 3e-4, betas 0.99/0.999) | RMSprop (lr 1e-3, batch 64) | [PAPER both] | fairness | Implicit in wrapper choice |
| Sparsity regulariser | `lambda1 = 0.05`, fixed | `reg_coeff`, grid-searched over 10 values | [PAPER both] | structure, edge count | Per-model; see Section 5 |

The asymmetry in budget structure is intrinsic to the
algorithms. The asymmetry in sparsity treatment is also paper
behaviour (DAGMA uses one L1 value; DCDI sweeps 10 values), so
the project policy must decide whether to follow paper
asymmetry or to symmetrise within the selection study (see
Section 5).

### Recommendation (proposal for user review)

Adopt paper-explicit values for DAGMA `warm_iter = 20000` and
`max_iter = 70000`, paper-explicit `lr = 3e-4`,
`beta_1 = 0.99`, `beta_2 = 0.999`. Freeze them in
`docs/02` Section 3.3 alongside the existing DAGMA bullets.
Record `h_diagnostic_threshold = 1e-5` as
`[UNSUPPORTED / PROJECT DECISION]` with a one-line reason.

---

## 5. DCDI sparsity and calibration fairness

### Paper evidence

- DCDI uses held-out NLL on an 80/20 train/test split to
  select the sparsity coefficient: "the models were trained on
  80% examples and evaluated on the 20% remaining examples.
  The hyperparameter combination chosen was the one that
  induced the lowest negative log-likelihood on the held-out
  examples" [PAPER: DCDI Appendix B.5, page 31].
- DCDI sparsity grid: `log10(lambda) in {-7, -6, -5, -4, -3,
  -2, -1, 0, 1, 2}`, i.e. **10 values** spanning 9 orders of
  magnitude [PAPER: DCDI Table 1, page 31].
- DCDI "default hyperparameters were chosen using small-scale
  experiments on perfect-known interventions data sets in
  order to have a small SHD" [PAPER: DCDI Appendix B.5, end of
  paragraph]. Architecture parameters (hidden units, hidden
  layers) were not grid-searched in the main experiments;
  only the regularisation coefficient was.
- DAGMA Remark 5 [PAPER: DAGMA Section C, page 20]: "we have
  not performed any hyperparameter optimization ... we simply
  chose a reasonable value for the L1 penalty coefficient and
  used that same value for all ER and SF graphs". DAGMA's
  paper L1 treatment is a single fixed value (`beta_1 = 0.05`
  in Section C.1.1).

### Is it defensible for DCDI to receive a 5-value sweep while DAGMA receives one fixed value

This is a load-bearing fairness question. Three positions are
defensible and one is not.

- **Defensible (i): Mirror paper asymmetry.** DAGMA paper does
  not sweep; DCDI paper sweeps 10 values. Selection study
  preserves the paper-default treatment of each model. Risk:
  the selection study is a head-to-head bake-off, and giving
  DCDI a hyperparameter sweep that DAGMA does not get advantages
  DCDI's reported scores. Mirroring paper behaviour is
  defensible only if the bake-off explicitly acknowledges this
  asymmetry as "each model evaluated under its native
  hyperparameter discipline".
- **Defensible (ii): Symmetrise upward.** Both candidates
  receive a Phase B sweep with the same number of
  configurations. DAGMA gets a `lambda1` sweep; DCDI gets a
  `reg_coeff` sweep. Number of configurations matches.
  Native ranges differ. The current `docs/02` Section 3.3
  posture (DCDI: 5-value sweep; DAGMA: 1 value) is partway
  here for DCDI but unsymmetric for DAGMA.
- **Defensible (iii): Symmetrise downward.** Both candidates
  pin their sparsity at the paper default. DCDI uses
  `reg_coeff = 0.1` (the DCDI argparse default); DAGMA uses
  `lambda1 = 0.05` (the paper Section C.1.1 value). No
  Phase B sweep for either. Phase B then sweeps other knobs
  if any, or is collapsed.
- **Not defensible: Current `docs/02` Section 3.3 as written.**
  DCDI receives a 5-value sparsity sweep; DAGMA receives one
  pinned `lambda1`. This is asymmetric and is neither (i)
  (because docs/02 cut DCDI from 10 to 5 values, departing from
  paper) nor (ii) (because DAGMA gets no sweep) nor (iii)
  (because DCDI still sweeps). The asymmetry is not justified
  in `docs/02` itself.

### Same number of Phase B configurations

Adopt the principle "same number of Phase B configurations per
model, in each model's native parameterisation". This rules out
mirroring paper asymmetry (option (i)) unless `docs/02` is
explicit about the asymmetry. Between (ii) and (iii), the audit
recommends:

- **(ii) Symmetrise upward** at the level of number of
  configurations. The exact grid values for both candidates are
  a separate grid-freeze step that should follow a paper
  re-read; the audit does not finalise grid values here.

### Grid-width fairness

The three grid-width principles in the prompt:

- (a) Same number of configurations plus comparable
  order-of-magnitude exploration in each model's native
  parameterisation.
- (b) Narrower, equally conservative local sweeps around
  paper-supported defaults for both models.
- (c) Accept different grid widths because parameterisations
  differ, document the asymmetry honestly.

DCDI's `log10(reg_coeff)` from `-7` to `+2` is 9 orders of
magnitude; that range is what the DCDI paper used and was tuned
on a 80/20 split. DAGMA's `lambda1 = 0.05` is one value; a
naive 9-orders-of-magnitude DAGMA sweep would include values
like `1e2` that collapse the graph to near-empty structure (the
L1 penalty dominates the score) and values like `1e-7` that
effectively turn off regularisation. Mechanically copying the
DCDI span to DAGMA is not appropriate because the two penalties
are not on the same scale (DCDI L1 penalty divides by `d^2`
inside the loss; DAGMA L1 penalty does not).

The audit recommends **principle (b) for Phase B**: narrower,
equally conservative local sweeps anchored on paper defaults
for both candidates, with the same number of grid points and a
recorded grid-spacing rule. A defensible starting design:

- DAGMA Phase B grid: 5 values of `lambda1` anchored on `0.05`,
  e.g., `{0.01, 0.025, 0.05, 0.1, 0.25}` (half-decade spacing
  around the paper anchor).
- DCDI Phase B grid: 5 values of `reg_coeff` anchored on `0.1`,
  e.g., `{0.01, 0.03, 0.1, 0.3, 1.0}` (half-decade spacing
  around the upstream default).

The audit does NOT recommend freezing these specific values in
this commit. Both grids are draft suggestions whose exact
endpoints must be reviewed at the grid-freeze step before
Phase B configuration construction. The principle to freeze
now is: same number of grid points, half-decade or
third-decade spacing, anchored on the paper default,
documented in `docs/02` Section 3.3.

If compute is constrained, fall back to principle (b)
collapsed to 3 configurations per model. Do not adopt
principle (c) (different grid widths) without a strong
written justification.

Principle (a) (same number of configurations, mechanically
similar order-of-magnitude exploration in each model's native
parameterisation) is also defensible but allows wider variance
in interpretation than (b). The audit recommends (b) for
selection-study tightness.

### DCDI sparsity-selection criterion

Per paper, DCDI sparsity is selected by held-out NLL on an
80/20 train/test split [PAPER: DCDI Appendix B.5]. This is the
same criterion that the project's Phase B ranking implicitly
uses if validation-NLL is one of the ranking dimensions. The
current `docs/08` Commit 9 ranking is lexicographic over SID,
MMD, SHD, with `configuration_hash` as a final tiebreaker;
validation NLL is not a primary ranking criterion. The audit
notes this divergence but does not recommend changing the
ranking rule; it does recommend stating in `docs/02` v1.6 that
the Phase B ranking is selection-task-driven (SID and MMD)
rather than NLL-driven (the DCDI paper's choice), with reason.

### What must not be tuned using held-out evaluation

[PROJECT DOC: `docs/02` Section 3.3, `docs/08a` Section 6,
`docs/08` Section 9]

- Sparsity coefficients (`lambda1`, `reg_coeff`).
- Training budget (`num_train_iter`, `warm_iter`, `max_iter`).
- MMD `n_samples`.
- Threshold values (`0.3` DAGMA, `0.5` DCDI, robustness
  triples).
- Any wrapper hyperparameter.

### Recommendation (proposal for user review)

Adopt symmetrise-upward (Section 5 option (ii)) with grid-width
principle (b) (narrower, conservative local sweeps). Both
candidates receive 5 Phase B configurations centred on their
paper defaults with half-decade spacing. The exact endpoints
are a separate grid-freeze step. `docs/02` Section 3.3 amended
to document that DAGMA's `lambda1` is no longer pinned for
Phase B.

---

## 6. Real MMD sampling policy

[SOURCE: `experiments/selection_study/sampling.py:32`;
PROJECT DOC: `docs/02` Section 4.2, `docs/04j`]

### Current state in code

`SCHEMA_GATE_MMD_N_SAMPLES = 64` is the default in
`compute_per_intervention_records`. The pipeline calls this
function without an explicit `n_samples` override; the
schema-gate run therefore currently uses `n = 64`.

### `docs/02` proposed value

`docs/02` Section 4.2: "Sample size for MMD comparison defaults
to 1000 model-generated samples per intervention condition,
matching the observational sample size frozen in Section 9."
This is `[PROJECT DOC]` only.

### Metric validation evidence

[PROJECT DOC: `docs/04j_mmd_shd_reference_crosscheck.md`]
MMD primitive cross-checked against a loop reference; the
cross-check confirms implementation correctness, not
sample-count sufficiency.

### Paper guidance on sample counts

The DCDI paper does not use MMD for selection; DCDI uses
held-out NLL [PAPER: DCDI Appendix B.5]. The DAGMA paper does
not use MMD; DAGMA uses SHD and runtime [PAPER: DAGMA Section
C.1.1, Table 1]. Neither paper provides MMD sample-count
guidance for selection. Selection-study MMD is therefore a
project decision, not a paper-anchored choice.

### Cost / variance tradeoff

Unbiased RBF MMD with `n` samples per side has two cost
components per intervention per bandwidth multiplier:

- **Model sampling cost**: `n` calls to the wrapper sampler
  (`forward_given_params` for DCDI, ancestral pass for DAGMA),
  plus `n` calls to the ground-truth `intervene().sample`.
  Cost is linear in `n` per intervention per bandwidth.
- **Kernel / MMD computation cost**: pairwise RBF kernel
  matrix of shape `(2n, 2n)`, then the unbiased estimator
  formula. Cost is `O(n^2)` per bandwidth per intervention.
  At `n = 1000`, the kernel matrix is `2000 x 2000` per
  intervention per bandwidth; with three bandwidths in the
  sweep and (say) 20 interventions, the per-run kernel cost is
  on the order of `3 * 20 * 4e6 = 2.4e8` kernel evaluations.
  On the project hardware (CPU, vectorised numpy), this is
  small relative to training cost but not free.

Variance of the unbiased estimator decreases at roughly
`O(1/n)` rate for moderate sample sizes. At `n = 64` the
standard error is approximately four times what it is at
`n = 1000`; at `n = 1000` it is about three times what it is at
`n = 10 000`. The marginal variance reduction beyond `n = 1000`
is small; the marginal cost is linear in `n` plus quadratic in
`n^2` for the kernel matrix, which makes `n = 10 000` an order
of magnitude more expensive on the kernel side.

`n = 1000` is a defensible operating point: it matches the
training sample count frozen in `docs/02` Section 9, the
per-side sample is the same order of magnitude as the
training-data sample, and the cost is bounded.

The recommendation does NOT claim paper support; selection-
study MMD is project-decided in scope and sample count.

### Should real MMD `n_samples` enter `Configuration` / hash

Yes. MMD `n_samples` is part of the experimental configuration.
It enters the run record via `config_resolved` and participates
in `configuration_hash`. The argument is identical to the SCM-A
argument from `docs/03` 19/05/2026 entry.

### Recommendation (proposal for user review)

- Add `mmd_n_samples: int = 1000` as a top-level `Configuration`
  field.
- Plumb it through `pipeline.run_single_fit` so
  `compute_per_intervention_records` uses the configured value.
- Keep `SCHEMA_GATE_MMD_N_SAMPLES = 64` as the
  `compute_per_intervention_records` keyword default for the
  function's own tests; the pipeline stops relying on it.

---

## 7. Other schema-gate constants that must not leak

[SOURCE: `experiments/selection_study/pipeline.py`,
`experiments/selection_study/sampling.py`,
`experiments/selection_study/config.py`,
`experiments/selection_study/threshold_robustness.py`]

| Name | Current value | Purpose | Class | Affects | Become `Configuration`? | Action |
|---|---|---|---|---|---|---|
| `SCHEMA_GATE_N_TRAIN` | 64 | schema-gate training-sample count | toy-only | training, evaluation | Yes (`n_train`) | Default 1000 for real runs; pin at `Configuration` level. |
| `SCHEMA_GATE_N_VAL_DCDI` | 32 | DCDI validation split size in schema gate | toy-only | DCDI training, patience gate | Yes (`n_val_dcdi`; DCDI-only) | See Section 8 for split-vs-additional policy. |
| `SCHEMA_GATE_DCDI_N_ITER` | 30 | schema-gate DCDI iteration ceiling | toy-only | DCDI training | Yes (`dcdi_num_train_iter`; DCDI-only) | Pilot then freeze per Section 2. |
| `SCHEMA_GATE_DCDI_CONFIG_KWARGS` | `{"stop_crit_win": 10, "train_batch_size": 8}` | schema-gate DCDI patience cadence and batch | toy-only | DCDI patience cadence, batch size | Yes (separate DCDI-only fields) | Phase A/B values: `stop_crit_win = 100`, `train_batch_size = 64` per paper. |
| `SCHEMA_GATE_MMD_N_SAMPLES` | 64 | schema-gate MMD model-batch size | toy-only | MMD primary, sensitivity, sweep | Yes (`mmd_n_samples`) | See Section 6. |
| `n_nodes` default (Configuration) | 3 | schema-gate SCM size | toy-only | training, evaluation | Already in `Configuration` (19/05/2026) | Phase A/B configs must set `n_nodes = 10`. |
| `expected_edges` default | 3 | schema-gate SCM density | toy-only | training, evaluation | Already in `Configuration` | Phase A/B configs must set `expected_edges = 20` (ER2). |
| `noise_scale` default | 1.0 | SCM noise | real-run | training, evaluation | Already in `Configuration` | Keep `1.0` (`docs/02` Section 3.1). |
| `weight_magnitude_range` default | `(0.5, 2.0)` | SCM edge weights | real-run | training, evaluation | Already in `Configuration` | Keep `(0.5, 2.0)` unless project decides otherwise. |
| `threshold_robustness_triple` per model | DAGMA `(0.2, 0.3, 0.4)`, DCDI `(0.4, 0.5, 0.6)` | offline robustness | real-run | report only | Already in `Configuration` | Substantiated by `docs/08b`. |
| `_SHD_REVERSAL_COST` | 2 | SHD reversal-cost convention | real-run | SHD value | Currently runner-side | Consider promoting (`gadjid` default is 1). |
| `_SID_BACKEND`, `_SID_BACKEND_VERSION`, `_SID_ARGUMENT_ORDER`, `_SID_RETURN_VALUE` | gadjid 0.1.0, predicted_then_true, raw_mistake_count | SID provenance | real-run | reproducibility | Keep runner-side. |
| `_MMD_CLIP_POLICY` | `"no_clip"` | MMD value clipping policy | real-run | report | Currently runner-side | Keep. |
| `PROTOCOL_THRESHOLD_TRIPLES` | DAGMA `(0.2,0.3,0.4)`, DCDI `(0.4,0.5,0.6)` | per-model protocol triples | real-run | offline recomputation | Cross-validated against `Configuration` | No change. |
| `_BANDWIDTH_SWEEP_MULTIPLIERS` | `0.5x, 1.0x, 2.0x` | MMD bandwidth sensitivity | real-run | MMD sensitivity reporting | Currently runner-side | Per `docs/02` Section 4.2 frozen; no change. |

---

## 8. Real-run `Configuration` / `configuration_hash` policy

### Per-knob storage decisions

| Knob | Storage | Reason |
|---|---|---|
| `n_train` | top-level `Configuration` field | Affects every run; enters `configuration_hash` so a change creates a new run identity. |
| `n_val_dcdi` | top-level `Configuration` field, DCDI-only (None for DAGMA) | See Section 8 "validation policy" below. |
| `mmd_n_samples` | top-level `Configuration` field | Affects every per-intervention MMD record; enters `configuration_hash`. |
| `dcdi_num_train_iter` | top-level `Configuration` field, DCDI-only (None for DAGMA) | Affects every DCDI run; enters `configuration_hash` so post-hoc budget expansion forces a new identity. |
| `dcdi_stop_crit_win` | top-level `Configuration` field, DCDI-only | Affects patience cadence; project must record explicitly. |
| `dcdi_train_patience` | top-level `Configuration` field, DCDI-only | Affects actual stop iteration; record explicitly. |
| `dcdi_train_patience_post` | top-level `Configuration` field, DCDI-only | Same as above. |
| `dcdi_train_batch_size` | top-level `Configuration` field, DCDI-only | Affects training variance; record explicitly. |
| `dcdi_lr` | top-level `Configuration` field, DCDI-only | Affects convergence; paper-explicit value. |
| `dcdi_h_threshold` | top-level `Configuration` field, DCDI-only | Affects stopping criterion; paper-explicit. |
| `dagma_warm_iter` | top-level `Configuration` field, DAGMA-only (None for DCDI) | Affects every DAGMA run; paper-vs-library disagreement must be recorded. |
| `dagma_max_iter` | top-level `Configuration` field, DAGMA-only | Same as above. |
| `dagma_lr` | top-level `Configuration` field, DAGMA-only | Paper-explicit; record. |
| `dagma_beta_1` (Adam) | top-level `Configuration` field, DAGMA-only | Paper-explicit; record. |
| `dagma_beta_2` (Adam) | top-level `Configuration` field, DAGMA-only | Paper-explicit; record. |
| `dagma_lambda1` (L1 coefficient) | top-level `Configuration` field, DAGMA-only OR PhaseBConfiguration.hyperparameters | Top-level for Section 5 option (iii); PhaseBConfiguration for option (ii). |
| `dagma_T` | top-level `Configuration` field, DAGMA-only | Already implicitly in `Configuration` via the wrapper; record explicitly. |
| `dagma_s` | top-level `Configuration` field, DAGMA-only | Same as above. |
| `dagma_mu_init` | top-level `Configuration` field, DAGMA-only | Same as above. |
| `dagma_mu_factor` | top-level `Configuration` field, DAGMA-only | Same as above. |

Note that the existing wrapper carries `T`, `s`, `lambda1`,
`mu_init`, `mu_factor` on its own `DAGMAConfig` dataclass; the
`Configuration` extension does not need to duplicate them as
separate top-level fields if the wrapper-side dataclass is
already consumed via `resolved_config`. The decision is a
trade-off: duplication risks drift between
`Configuration` and `DAGMAConfig`; relying on
`DAGMAConfig` alone means the Configuration's
`configuration_hash` does not reflect wrapper-side knobs.
Recommendation: top-level fields on `Configuration` for all
selection-study-relevant DAGMA knobs, plumbed through to
`DAGMAConfig` at fit time. The same applies to
`DCDIConfig`.

### Flat vs nested layout

| Aspect | Flat fields with None-for-other-model | Nested model-specific config object |
|---|---|---|
| Maintainability | Adding a new DCDI knob = one new flat field plus a None branch for DAGMA in `__post_init__` validation. Lots of None branches accumulate. | Adding a new DCDI knob = one new field on the DCDI sub-object. No DAGMA-side change. |
| Hash stability | A `None` on the unused model still enters `canonical_json` and the hash, but a `None` value hashes the same on both sides; backward compatibility requires careful default management. | A nested object hashes its own contents; adding a knob to one model does not change the other model's hash. |
| Readability | Top-level fields are visible at a glance; the model-specific naming convention (`dcdi_*` / `dagma_*`) makes ownership clear. | Nested object groups related knobs; reduces the top-level field count. |
| Implementation risk | Minimal; mirrors the SCM-A flat-field pattern from 19/05/2026 (n_nodes, expected_edges, noise_scale, weight_magnitude_range are all top-level flat fields today). | Higher; introduces a second pattern. Would push back toward refactoring SCM-A into a nested object for consistency. |
| Consistency with current Configuration design | High; the dataclass already uses flat fields throughout. | Low; introduces intra-`Configuration` inconsistency unless SCM-A is refactored to nested as well. |

**Recommendation: flat fields with None-for-other-model**,
matching the SCM-A pattern from 19/05/2026. The consistency
argument outweighs the readability argument: a future refactor
to nested objects can be a single decision once and for all,
not partial. The flat pattern's accumulation of None branches
is a real cost but is bounded by the small number of
selection-study knobs.

### How Phase A/B ensures `n_nodes = 10` and `expected_edges = 20`

By explicit configuration construction. The 19/05/2026
defaults are schema-gate-only (`n_nodes = 3`,
`expected_edges = 3`). Phase A/B configuration JSON files must
set `n_nodes = 10` and `expected_edges = 20` explicitly.
Optional regression test: pin
`(n_nodes, expected_edges) == (10, 20)` for selection-study
configs in a Phase A artefact.

### MMD `n_samples` and training sample counts

In `config_resolved`. They enter `configuration_hash` via
`to_canonical_dict`. Any change in sample size changes
`configuration_hash`, `run_id`, and the run-directory path,
forcing a fresh run identity rather than silently overlaying.

### Exact `docs/02` amendments needed

Section 3.3 (DAGMA-linear starting point): freeze
`warm_iter = 20000`, `max_iter = 70000`, `lr = 3e-4`,
`beta_1 = 0.99`, `beta_2 = 0.999` per paper; record
`h_diagnostic_threshold = 1e-5` as project decision; either
keep `lambda1 = 0.05` pinned (option (iii)) or introduce a
Phase B `lambda1` grid (option (ii)). Optionally note that the
wrapper does not currently enforce the paper's loss-convergence
rule of relative error < 1e-6.

Section 3.3 (DCDI-G starting point): freeze
`num_train_iter` (pilot-derived per Section 2),
`stop_crit_win = 100`, `train_patience = 5`,
`train_patience_post = 5`, `h_threshold = 1e-8`, RMSprop
`lr = 1e-3`, `train_batch_size = 64`, hidden units `16`,
hidden layers `2`, leaky-ReLU, Xavier init [PAPER]. Decide
whether the 5-value Phase B sparsity sweep is retained as in
current `docs/02`, narrowed under principle (b), or
symmetrised against a matched DAGMA `lambda1` sweep.

Section 4.2 (MMD rule): replace "defaults to 1000" with
"`mmd_n_samples = 1000` is a top-level `Configuration` field
that enters `configuration_hash`."

Section 9 (Tactical constants frozen for the selection study
only): add bullets for `n_train = 1000`, `mmd_n_samples = 1000`,
`dcdi_num_train_iter = <pilot-derived>`, DAGMA inner-iter
budgets, DAGMA Adam values, DCDI patience values, DCDI MLP
architecture values.

A minor `docs/02` version (v1.6) is appropriate.

---

## 9. Leakage and fairness policy

### Permitted evidence for choosing DCDI `num_train_iter`

[PROJECT DOC: `docs/02` Section 3.3 and Section 8 scope-cut
hierarchy]

Allowed:

- DCDI paper convergence behaviour (Figure 6 ~62 000 example)
  [PAPER].
- Upstream argparse default `1_000_000` [SOURCE].
- Pilot runs on `seed_population = "reproduction"` per
  Section 2.

Forbidden:

- Calibration records (`seed_population = "calibration"`).
- Held-out evaluation records.

### Permitted evidence for DAGMA training-budget values

Allowed:

- DAGMA paper Section C.1.1 values [PAPER]: `warm_iter = 20000`,
  `max_iter = 70000`, Adam `(lr, beta_1, beta_2) =
  (3e-4, 0.99, 0.999)`.
- DAGMA library defaults [SOURCE]: `warm_iter = 30000`,
  `max_iter = 60000`.
- Pilot evidence (rare; DAGMA does not need a budget pilot
  because its schedule is deterministic).

Forbidden: calibration / held-out records.

### Permitted evidence for MMD `n_samples`

Allowed:

- `docs/02` Section 4.2 default (1000) [PROJECT DOC].
- Section 9 observational sample size frozen [PROJECT DOC].
- A short variance vs cost analysis if needed.

Forbidden: tuning `n_samples` to maximise the
selection-relevant gap (metric tuning).

### Permitted evidence for Phase B calibration

Allowed: `seed_population = "calibration"` records for the
model under calibration.

Forbidden: held-out evaluation records; records produced after
Phase B ranking has been frozen.

### What must remain held-out

`seed_population = "held_out_evaluation"` records, until the
selection-study report.

### How to prevent post-hoc DCDI training-budget expansion

- `docs/02` Section 3.3 (after amendment) MUST name a specific
  `dcdi_num_train_iter` value.
- `configuration_hash` MUST include `dcdi_num_train_iter`.
- Optional: preflight MUST reject manifests whose
  `dcdi_num_train_iter` differs from the `docs/02`-frozen value
  without a contemporaneous `docs/03` entry.

### Comparable opportunity for DAGMA and DCDI

Per Section 5 recommendation (option (ii) plus grid-width
principle (b)): both candidates receive 5 Phase B
configurations centred on paper defaults with half-decade
spacing in their native parameterisation. Both candidates have
explicit training-budget ceilings in `docs/02`. Same
`n_train`, same `mmd_n_samples`, same `n_nodes = 10`, same
`expected_edges = 20`, same intervention set, same seed
derivation rule.

---

## 10. Recommended amendments

The following are draft texts for user review. Not applied to
`docs/02` or `docs/03` in this commit.

### Draft `docs/02` amendment (v1.6)

Two changes in Section 3.3 (DAGMA-linear starting point).

Replace the existing bullet list with:

```
- number of iterations: T = 4
- L1 coefficient: lambda1 = 0.05
- log-det parameter sequence: s = [1.0, 0.9, 0.8, 0.7]
- initial central-path coefficient: mu_init = 1.0
- decay factor: mu_factor = 0.1
- in-library threshold passed to DagmaLinear.fit: w_threshold = 0.0
- project-level threshold applied externally to abs(W_continuous):
  0.3
- per-stage inner Adam iterations (non-final): warm_iter = 20000  [PAPER, Section C.1.1]
- final-stage inner Adam iterations: max_iter = 70000           [PAPER, Section C.1.1]
- Adam learning rate: lr = 3e-4                                 [PAPER]
- Adam betas: beta_1 = 0.99, beta_2 = 0.999                     [PAPER]
- (optional) loss-convergence rule from DAGMA Section C.1.1:
  relative error between subsequent iterations < 1e-6; not
  currently enforced by the project wrapper; project decision
  recorded in docs/03.
- if Phase B sparsity policy adopted (Section 5 option (ii) in
  docs/08c): Phase B sweeps 5 values of lambda1 centred on 0.05
  with half-decade spacing; grid endpoints frozen in a separate
  grid-freeze step.
```

Two changes in Section 3.3 (DCDI-G starting point).

Replace / extend the existing bullets with:

```
- Lagrangian multiplier: gamma_0 = 0                           [PAPER, Table 2]
- penalty coefficient: mu_0 = 1e-8                             [PAPER, Table 2]
- penalty update factor: n = 2; decrease threshold: 0.9        [PAPER, Table 2]
- stopping criterion: h(Lambda) < 1e-8                         [PAPER, Table 2]
- learning rate: lr = 1e-3                                     [PAPER, Table 2]
- RMSprop optimiser; minibatch size = 64                       [PAPER, Section B.5]
- hidden units = 16, hidden layers = 2, leaky-ReLU             [PAPER, Table 2]
- Xavier initialisation; adjacency entries near 1.0            [PAPER, Section B.5]
- training-budget ceiling: num_train_iter = <pilot-derived>     [PROJECT DECISION; pilot per docs/08c Section 2]
- stop-check window: stop_crit_win = 100                       [SOURCE]
- training patience: train_patience = 5                        [SOURCE]
- post-threshold patience: train_patience_post = 5             [SOURCE]
- DCDI validation split: 80/20 split from n_train (n_val_dcdi = 200 when n_train = 1000)  [PAPER, Section B.5]
- sparsity coefficient: <selected policy per docs/08c Section 5> values
```

Section 4.2 (MMD rule), final paragraph replacement:

```
Sample size for MMD comparison is mmd_n_samples = 1000
model-generated samples per intervention condition, matching
the observational sample size frozen in Section 9. The value
is a top-level Configuration field that participates in
configuration_hash.
```

Section 9 (Tactical constants frozen for the selection study
only), additional bullets:

```
- training sample count: n_train = 1000
- DCDI validation split: 80/20 fraction of n_train
- MMD sample count: mmd_n_samples = 1000
- DCDI training-budget ceiling: num_train_iter = <pilot-derived>
- DCDI patience values: stop_crit_win = 100, train_patience = 5,
  train_patience_post = 5
- DAGMA per-stage inner iterations: warm_iter = 20000 (paper)
  or 30000 (library); project decision: <paper / library>
- DAGMA final-stage inner iterations: max_iter = 70000 (paper)
  or 60000 (library); project decision: <paper / library>
- DAGMA Adam: lr = 3e-4, beta_1 = 0.99, beta_2 = 0.999
```

### Draft `docs/03_decision_log.md` entry

```
## YYYY-MM-DD -- Real-run training-budget, sparsity, and MMD constants

### Decision

docs/02 amended to v1.6:
- DAGMA per-stage iteration budgets, Adam values, and threshold
  fixed at paper-explicit values from DAGMA Section C.1.1
  (warm_iter = 20000, max_iter = 70000, lr = 3e-4, betas
  0.99 / 0.999, threshold 0.3). Wrapper defaults that did not
  match paper (warm_iter = 30000, max_iter = 60000) are
  superseded by paper values; the wrapper will be updated in a
  separate Commit-8 prerequisite commit if needed.
- DCDI hyperparameters fixed at DCDI Table 2 values
  (mu_0 = 1e-8, gamma_0 = 0, eta = 2, delta = 0.9,
  h_threshold = 1e-8, lr = 1e-3, batch 64, hidden 16, layers 2,
  leaky-ReLU, Xavier init). num_train_iter ceiling is
  pilot-derived per docs/08c Section 2; the pilot is
  diagnostic-only and uses reproduction seeds.
- DCDI validation split fixed as 80/20 of n_train, matching
  DCDI Appendix B.5. n_val_dcdi enters Configuration as a
  DCDI-only field with None for DAGMA.
- Phase B sparsity treatment: <option (ii) symmetrised at 5
  configurations per model centred on paper defaults with
  half-decade spacing, per docs/08c Section 5; exact endpoints
  frozen in a separate grid-freeze step>.
- mmd_n_samples = 1000 elevated to a top-level Configuration
  field that enters configuration_hash.
- n_train = 1000 elevated to a top-level Configuration field.

Configuration extended; canonical_json, configuration_hash,
and load_config carry the new fields. Preflight unchanged in
behaviour. Pipeline reads new fields from resolved_config in
place of schema-gate constants for any real-run configuration.

### Reason

Commit 7 surfaced the SCM-regime gap (resolved 19/05/2026 as
SCM-A) and the training-budget / MMD-sample analogues
(resolved by this entry). Paper re-read for DAGMA and DCDI
substantiated specific values for budget knobs that were
project-decided-by-omission in the wrapper. Phase B sparsity
fairness asymmetry surfaced; symmetrise-upward chosen.

### What does NOT change

- No selection criterion, no evaluation rule, no metric
  primitive.
- No src/ change unless DAGMA wrapper warm_iter / max_iter
  defaults are updated in a separate commit before Commit 8.
- No run.json schema_version bump; new fields enter via
  config_resolved.

### Consequence

- DAGMA and DCDI receive paper-anchored training-budget
  treatments and symmetric Phase B sparsity opportunity.
- Phase A/B configurations must explicitly set all new fields;
  defaults preserve the schema-gate cell.
- Existing schema-gate hashes change; no persistent run records
  are being migrated.
- C-P11 reapplication policy fixed per docs/08c Section 3
  (rerun required before held-out evaluation; explicit scope
  if not rerun).
```

### Recommended code / configuration extension before Commit 8

Mirror the 19/05/2026 SCM-A pattern:

- Add the proposed top-level fields to `Configuration` with
  sensible defaults.
- Add validation rules in `__post_init__` (positive integers,
  no bool, DCDI-only fields None for DAGMA, DAGMA-only fields
  None for DCDI, paper values where applicable).
- Plumb the new fields from `resolved_config` into
  `pipeline.run_single_fit` and
  `compute_per_intervention_records`.
- Optionally update `DAGMAConfig` defaults to paper values
  (`warm_iter = 20000`, `max_iter = 70000`) if the project
  decision in `docs/02` v1.6 elects paper-anchored defaults.
- Add tests mirroring the SCM-field tests in
  `tests/test_config_schema.py` and `tests/test_pipeline.py`.

### Whether Commit 8 may proceed

Not yet. Prerequisites in order:

1. DCDI training-budget pilot complete (reproduction seeds
   only).
2. DAGMA paper-vs-library decision recorded.
3. Phase B sparsity policy decided.
4. C-P11 reapplication policy recorded.
5. `docs/02` v1.6 amendment reviewed and committed.
6. `docs/03` entry committed.
7. `Configuration` extension and tests committed.

Then Commit 8 may proceed.

---

## Appendix A: Inputs to this audit

Documents read:

- `docs/01_research_question_and_commitments.md`
- `docs/02_base_model_selection.md` (v1.5)
- `docs/03_decision_log.md` (through 20/05/2026)
- `docs/04b_source_inspection.md`
- `docs/04c_runtime_probe_results.md`
- `docs/04f_dcdi_sampler_quality_diagnostic.md`
- `docs/04j_mmd_shd_reference_crosscheck.md` (referenced)
- `docs/08_base_model_selection_plan.md`
- `docs/08a_experiment_tracking_and_results_schema.md`
- `docs/08b_selection_study_constants_and_fairness_audit.md`

Source files read:

- `experiments/selection_study/config.py`
- `experiments/selection_study/pipeline.py`
- `experiments/selection_study/sampling.py`
- `experiments/selection_study/threshold_robustness.py`
- `experiments/selection_study/preflight.py`
- `src/symbolic_priors_cd/wrappers/dagma.py` (`DAGMAConfig`)
- `src/symbolic_priors_cd/wrappers/_dagma_fit.py`
- `src/symbolic_priors_cd/wrappers/_dcdi_training.py`
- `external/source_inspection/dcdi/main.py` (argparse)
- `external/source_inspection/dcdi/dcdi/train.py` (stopping)
- `external/source_inspection/dagma/src/dagma/linear.py`

Papers re-read (v1.1 of this audit):

- `papers/DCDI.pdf`, pages 1-3, 7-9, 17-19, 23-25, 29-33.
  Citations used: Section 4 (page 7) sample sizes; Section 4
  + Appendix B.5 (page 8, 31) sparsity-selection criterion;
  Appendix B.5 (page 31) DCDI sparsity grid; Table 2 (page 32)
  DCDI hyperparameters; Appendix B.3 (page 29-30) Figure 6
  learning dynamics with ~62 000 iterations on 10-node graph.
- `papers/DAGMA.pdf`, pages 1-3, 19-21. Citations used:
  Section C.1.1 (page 21) `T`, `mu_0`, `alpha`, `beta_1`, `s`,
  `warm_iter`, `max_iter`, Adam values, loss convergence
  criterion, final threshold; Section C, Remark 5 (page 20)
  "no hyperparameter optimization" note.

`docs/04d_dcdi_mask_probe_results.md` was referenced via
`docs/02` v1.3 but not re-read for this audit; the
structural-mask question is orthogonal to the training-budget
topics here.

---

## Appendix B: What this document does NOT change

- No file under `src/` is modified.
- No file under `tests/` is modified.
- No file under `experiments/selection_study/` is modified.
- No notebook is modified.
- No configuration file is modified.
- No results directory is modified.
- No dependency is added or removed.
- `docs/02_base_model_selection.md` and
  `docs/03_decision_log.md` are not modified by this commit.
  The draft amendment texts above are proposals for user review.
- `docs/04*`, `docs/08`, `docs/08a`, `docs/08b` are not modified.
- No `pyproject.toml`, `requirements*.txt`, or environment file
  is modified.
