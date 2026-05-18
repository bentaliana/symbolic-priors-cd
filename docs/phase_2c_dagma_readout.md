# Phase 2c readout: DAGMA wrapper subphase handoff

## 1. Title, date, author, purpose

- Title: Phase 2c readout - DAGMA wrapper subphase handoff.
- Date: 2026-05-14.
- Author: project record, written at the close of the DAGMA wrapper
  subphase.
- Purpose: orient a future contributor to the state of the DAGMA
  wrapper, the empirical results recorded so far, what is still
  deferred, and what the next critical path is. This document closes
  the DAGMA wrapper subphase by summarising what was implemented and
  what was empirically validated.

This readout is an orientation layer only. The authoritative documents
listed in section 2 take priority if any conflict arises with this
summary. Any claim here that disagrees with `docs/01`, `docs/02`,
`docs/03`, or the C-P13 report is a defect in this readout, not in the
authoritative source.

## 2. Canonical documents, read order

Read these in the listed order before making non-trivial changes to the
project:

1. `docs/01_research_question_and_commitments.md` - frozen scientific
   contract for the main study.
2. `docs/02_base_model_selection.md` - frozen selection-study protocol.
3. `docs/03_decision_log.md` - implemented design decisions, refinements,
   and evaluator conventions.
4. `docs/phase_1_readout.md` - evaluator-foundation handoff.
5. `docs/phase_2b_dcdi_readout.md` - DCDI wrapper subphase handoff
   (paired counterpart to this document).
6. `docs/04_wrapper_api_contract.md` - wrapper capabilities, status
   taxonomy, invalid-output policy.
7. `docs/04b_source_inspection.md` - DAGMA and DCDI source inspection.
8. `docs/04c_runtime_probe_results.md` - DAGMA continuous-W
   preservation, DAGMA input mutation, DCDI imports and sampling
   feasibility.
9. `docs/04f_dcdi_sampler_quality_diagnostic.md` - C-P11 DCDI
   sampler-quality diagnostic.
10. `docs/04g_equal_variance_identifiability_check.md` - C-P12
    equal-variance identifiability sanity check.
11. `docs/06_dagma_wrapper_implementation_plan.md` - DAGMA wrapper
    plan, including the source-verified iteration schedule and
    diagnostics schema.
12. `docs/04h_dagma_sampler_quality_diagnostic.md` - C-P13 DAGMA
    sampler-quality diagnostic.

## 3. DAGMA wrapper commit status

All of Commits 1 through 10 of
`docs/06_dagma_wrapper_implementation_plan.md` are implemented and
green in the normal pytest collection:

- Commit 1: scaffolding, `DAGMAConfig`, `DAGMAWrapper` skeleton, DAGMA
  source import boundary in `_dagma_utils.py`,
  `WrapperDiagnostics.model_specific_diagnostics` schema in place.
- Commit 2: fit path with defensive `X.copy()` for DAGMA's in-place
  mean-centering, explicit hyperparameter forwarding to
  `DagmaLinear.fit`, no global RNG mutation.
- Commit 3: canonical pre-threshold continuous `W` boundary;
  `native_edge_continuous()` with defensive copies.
- Commit 4: source-faithfulness gate against a direct
  `DagmaLinear.fit` call, `allclose(atol=1e-12, rtol=1e-12)`.
- Commit 5: external thresholding at `abs(W) >= 0.3`, graph-status
  classification (`invalid_shape -> self_loop -> bidirected -> cyclic
-> valid_dag`) via the shared `_graph_status.py` helper; no silent
  repair.
- Commit 6: residual-fitted per-node sigma estimation with no
  variance floor; degenerate sigma maps to
  `unavailable_unresolved_noise_policy`.
- Commit 7: model-frame linear-Gaussian ancestral sampler with
  topological ordering and intervention clamping.
- Commit 8: raw-unit `sample_interventional` with preprocessor
  intervention-value transform and inverse-transform roundtrip;
  `residual_fitted` and `unit_variance` noise policies.
- Commit 9: C-P13 sampler-quality diagnostic probe and report
  (`inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py`,
  `docs/04h_dagma_sampler_quality_diagnostic.md`).
- Commit 10: `get_diagnostics()` returning the shared
  `WrapperDiagnostics` schema. Configured iteration budget
  recorded under `model_specific_diagnostics`; top-level
  `n_iterations` is `None` because DAGMA does not expose an
  observable inner-loop iteration counter.

Commit 11 (final readout / public API stabilisation) is the present
readout itself. Doc 06 does not gate this commit on additional
implementation; it is administrative.

## 4. Test-suite state

Latest known full pytest run at the close of the DAGMA wrapper
subphase: `367 passed, 1 skipped, 2 expected warnings`.

- The skipped test is the verified-SID scaffold from Phase 1.
- The two warnings are the expected `RuntimeWarning`s from the
  inf-sigma case in `tests/test_dagma_wrapper_residuals.py`
  (matmul/subtract on a deliberately non-finite fixture).

These numbers are the latest known counts at this readout, not a
permanent invariant. Future commits will change them.

## 5. Key DAGMA wrapper outcomes

- The wrapper loads `DagmaLinear` from the pinned inspected source at
  `external/source_inspection/dagma/src/dagma/linear.py` and never
  from a `site-packages` DAGMA. The import boundary verifies the
  resolved path at module load and flushes cached `dagma.*` entries
  that resolve outside the pinned source.
- `DAGMAWrapper.fit` calls `DagmaLinear.fit` directly with every
  hyperparameter passed explicitly from `DAGMAConfig`. Project-required
  override values (`T`, `lambda1`, `s`, `mu_init`, `mu_factor`,
  `w_threshold_internal`) are never left to the library default.
- The wrapper passes `X.copy()` into `DagmaLinear.fit`. DAGMA mutates
  its input array in place during L2 mean-centering at
  `linear.py:307`; the defensive copy keeps the caller's array
  unchanged.
- The wrapper calls DAGMA with `w_threshold_internal = 0.0` to
  preserve the continuous `W` matrix verbatim. Project-level
  thresholding is applied externally at the wrapper boundary using
  `abs(W) >= 0.3` (`thresholded_adjacency`).
- Adjacency orientation is row-source / column-destination
  (`adjacency[i, j] = True` means edge `i -> j`); DAGMA shares this
  convention so no transposition is applied at the boundary.
- Invalid thresholded graphs are classified, not repaired. The
  classification priority is `invalid_shape -> self_loop ->
bidirected -> cyclic -> valid_dag`. The continuous `W`, the
  thresholded adjacency, and the invalid-graph reason are all
  exposed in diagnostics for inspection.
- Residual-fitted per-node sigma is estimated in the model frame
  as `R = X_model_frame - X_model_frame @ W_sample`, then
  `sigma_j = R[:, j].std(ddof=0)`, where `W_sample =
W_continuous * A_thresh.astype(W_continuous.dtype)`. No sigma
  floor or clamping is applied. A non-finite or non-positive sigma
  routes the wrapper into `sampler_status =
unavailable_unresolved_noise_policy` and `sample_interventional`
  returns `None` under the `residual_fitted` policy.
- Two noise policies are exposed at sample time. `residual_fitted`
  is the primary policy and uses the stored sigma vector.
  `unit_variance` overrides sigma with `np.ones(n_vars)` as a
  sensitivity check; it can still run when `residual_fitted` is
  blocked by degenerate sigma, provided `graph_status` is
  `valid_dag` and `W_sample` exists.
- `get_diagnostics()` returns the shared `WrapperDiagnostics`
  TypedDict with all 16 top-level keys populated. DAGMA-specific
  fields live under `model_specific_diagnostics` and include the
  continuous `W`, the thresholded adjacency at the project
  threshold, the residual sigma vector, `W_sample`, availability
  flags, threshold-grid edge counts, sub-threshold and
  near-threshold counts, the pinned DAGMA source path, and the
  configured iteration upper bound.
- Top-level `n_iterations` is `None` for DAGMA because DAGMA does
  not expose an observable inner-loop iteration counter. The
  configured optimisation budget is recorded only under
  `model_specific_diagnostics.iterations_configured_upper_bound`,
  derived from the source-verified DAGMA schedule
  `(T - 1) * warm_iter + max_iter` (the path-following loop uses
  `warm_iter` inner steps for stages `0..T-2` and `max_iter` for
  stage `T-1`, matching the `tqdm(total=...)` total in the pinned
  source). For `DAGMAConfig()` defaults the value is `150000`. This
  is a configured upper bound, not an observed iteration count.
- `get_diagnostics()` is now the official selection-study logging
  surface for DAGMA. Future selection-study runners should consume
  this record rather than read wrapper private fields directly. If
  a missing field is needed for a downstream report, the right
  response is to widen `get_diagnostics()` (or
  `model_specific_diagnostics`) deliberately, not to depend on a
  private attribute.

## 6. Key C-P13 findings

See `docs/04h_dagma_sampler_quality_diagnostic.md` for the full
report. Summary on the same frozen fixture used by C-P11:

- True adjacency recovered exactly. `SHD = 0`.
- `graph_status = valid_dag`.
- `sampler_status = available`.
- Residual sigma vector `[0.9971, 0.9930, 1.0006]`, consistent with
  the SCM's unit-variance Gaussian noise.
- `wrapper_vs_truth_mmd = +8.439610e-03`.
- `floor_mmd = -1.756334e-04` (negative; valid finite-sample
  behaviour of the unbiased MMD estimator on same-distribution
  comparisons).
- Literal primary threshold `wrapper_vs_truth_mmd <= 3 * floor_mmd`
  evaluates as `FAIL` on this run because the floor is negative;
  this is the documented C-P11 negative-floor caveat and the
  literal inequality is therefore non-informative on this run.
- Wrong-structure sanity check (delete the strongest downstream
  true edge `2 -> 0`, `|true_w| = 1.7861`): `correct_mmd =
+7.025987e-03`, `wrong_mmd = +6.417106e-01`, ratio `91.334`.
  The fail-safe inequality `correct * 1.5 <= wrong` passes by a
  large margin.
- Diagnostic A (true adjacency, DAGMA learned continuous-W
  coefficients): MMD `+5.332152e-03`. Because the learned
  thresholded adjacency equals the true adjacency on this fixture,
  Diagnostic A's `W_sample` and sigma vector are bitwise identical
  to the learned wrapper's; it re-samples the same generative
  process under a different seed lane, so it is not an independent
  structural-localisation signal on this fixture.
- Diagnostic B1 and B2 (learned-weight and oracle-weight
  augmentation with the strongest missing true edge): not
  applicable. No true edges were missing from DAGMA's thresholded
  adjacency on this fixture.
- Unit-variance sensitivity: `+8.432217e-03`, essentially
  indistinguishable from `residual_fitted` (delta `-7.39e-06`).
  This is consistent with the SCM having unit-variance noise.

The C-P13 result is positive on this fixture but explicitly
fixture-specific. The negative-floor caveat means the literal
primary inequality should not be reported as a pass. The 91x
wrong/correct ratio is a sanity check that the sampler responds to
a known-bad structural perturbation; it is not a general
quality metric and is amplified by deletion of the dominant
downstream true edge.

## 7. Comparison to DCDI

On the same frozen fixture, same training data, same intervention,
same batch sizes, same seed protocol, same MMD function and
bandwidth, and the same median aggregation rule, the same-fixture
DAGMA vs DCDI numbers are:

|                              | DAGMA (C-P13)  | DCDI (C-P11)                |
| ---------------------------- | -------------- | --------------------------- |
| `graph_status`               | `valid_dag`    | `valid_dag`                 |
| SHD to true                  | `0`            | strongest true edge missing |
| `wrapper_vs_truth_mmd`       | `+8.44e-03`    | `+6.28e-01`                 |
| true-structure (Diag A)      | `+5.33e-03`    | `+5.26e-02`                 |
| augmented-structure (Diag B) | not applicable | `+4.23e-02`                 |

The dominant driver of the same-fixture gap is **structure
recovery**, not sampler mechanics. DAGMA recovered the true adjacency
exactly here; DCDI-G missed the strongest true edge `2 -> 0` in
C-P11. C-P11's own Diagnostic A on DCDI showed that MMD drops by
roughly an order of magnitude when the true adjacency is forced.
The same-fixture DAGMA-vs-DCDI gap is therefore consistent with
"DAGMA found the right structure here and DCDI did not", multiplied
through the same downstream ancestral-sampling and MMD pipeline.

This is a controlled same-fixture diagnostic comparison, not a
general model ranking. A single 3-node fixture cannot decide the
base model.

## 8. What the DAGMA result supports

- DAGMA's wrapper and sampler are empirically credible on this small
  fixture: raw-unit clamping is exact, the wrong-structure sanity
  check responds, residual sigma is sensible, and unit-variance is a
  cheap sensitivity check that confirms the policy.
- DAGMA is a promising candidate for the formal base-model selection
  study described in `docs/02_base_model_selection.md`.
- The evaluation pipeline reflects structural quality through interventional MMD: a learned adjacency that matches the true adjacency produces wrapper-vs-truth MMD about 75x smaller than an adjacency missing a dominant true edge under the same protocol. 

## 9. What it does not support

- DAGMA is **not** selected as the base model. C-P13 is a single
  small-fixture diagnostic and is not equivalent to the
  multi-seed selection study.
- C-P13 does **not** replace `docs/02_base_model_selection.md`.
- C-P13 does **not** prove DAGMA robustness on 10-node ER2 graphs,
  multi-seed regimes, or non-Gaussian noise.
- C-P13 did **not** exercise DAGMA's likely weak-edge and
  thresholding-suppression failure modes. Diagnostics B1 and B2
  are exactly the diagnostics that would test those failure modes,
  and they were not applicable on this fixture because no true edges
  were missing. Those failure modes are deferred to the full
  selection study and any future fixture that exposes them.
- DAGMA's symbolic-prior / loss-hook injection is **not**
  implemented. The main-study soft-prior penalty term does not
  exist in code yet on the DAGMA side.
- Prior-loss implementation should **not** begin before verified SID
  integration and formal base-model selection.
- DCDI loss-hook work remains paused. C-P13 does not justify
  unpausing it; DCDI Commit 11 is gated on a project-level review of
  C-P11 / C-P12 plus the selection-study outcome, not on DAGMA's
  C-P13 result.
- C-P11 and C-P13 used a small 3-node linear-Gaussian diagnostic
  fixture. The upcoming base-model selection study is conditioned on
  the thesis's intended 10-node ER2 linear-Gaussian synthetic SCM
  regime. That regime is deliberate: the downstream prior-corruption
  experiments are planned within the same synthetic linear-Gaussian
  setting. C-P11/C-P13 and the selection-study results should not be
  read as general claims about DAGMA or DCDI performance in nonlinear,
  non-Gaussian, or interventional-training settings.
- The formal base-model decision depends on the documented selection
  study under `docs/02_base_model_selection.md` and has not yet been
  made.

## 10. Status of loss-hook / prior-penalty work

- No DAGMA symbolic-prior loss term has been implemented. Doc 06
  Section 18 records that DAGMA loss-hook integration is invasive
  because DAGMA uses a hand-coded optimisation step (`minimize` in
  `linear.py`) rather than an autograd-friendly loss function. A
  DAGMA prior-penalty integration will need its own plan and its own
  source-faithfulness validation; it is not a one-line autograd
  hook.
- No DCDI loss-hook work has been resumed. DCDI Commit 11 remains
  paused per `docs/03_decision_log.md` pending a project-level
  review of the C-P11 / C-P12 findings and the full base-model
  selection study.
- Loss-hook or prior-penalty work should wait until the base-model
  selection study has produced a defensible decision. Starting
  prior-penalty work before then would risk locking effort into a
  candidate that the selection study has not endorsed.

## 11. Remaining blockers and deferred work

- SID is now closed as a blocker. Implementation, verification, and
  regression test suite are complete (see `docs/phase_2d_sid_readout.md`).
  MMD and SHD have also been cross-checked against independent references
  (see `docs/04j_mmd_shd_reference_crosscheck.md`).
- Experiment tracking / results schema. A results schema and logging
  layer are needed before the selection-study runner can be written.
  This is the next immediate step.
- Selection-study runner / execution. No runner has been written yet.
  The runner must invoke DAGMA (and DCDI, if still in scope) fits at
  multi-seed scale, log diagnostics, and apply the criterion ordering
  frozen in `docs/02`.
- Final base-model decision. Cannot be made before the selection study
  is run.
- Selected-model loss/prior injection. Deferred to after selection.
- Main prior-corruption experiments. Frozen in `docs/01` but blocked
  on selection plus the loss-hook implementation.
- Thesis results analysis and write-up. Downstream.

## 12. Next critical path

SID implementation and verification are now closed (see
`docs/phase_2d_sid_readout.md`). MMD and SHD cross-checks are also
complete (see `docs/04j_mmd_shd_reference_crosscheck.md`). The metric
layer is verified.

The next critical path is:

1. Experiment tracking and results schema. Design and implement a
   logging/results schema for the selection-study runner before writing
   the runner itself.
2. Base-model selection planning and execution. Write the runner,
   execute the selection study under the protocol in
   `docs/02_base_model_selection.md`, and record the outcome.

Prior-loss implementation remains deferred until after base-model
selection produces a defensible decision.

## 13. Handoff instructions

- Future conversations should read `docs/phase_2b_dcdi_readout.md`
  first, then this readout, then the authoritative documents from
  section 2. Treat the readouts as orientation, not as the contract.
- Maintain the no-silent-repair policy across both wrappers. Invalid
  graphs are reported through the status taxonomy and exposed in
  diagnostics; they are not deleted, symmetrised, or replaced.
- Do not revise thresholds, MMD aggregation rules, or selection
  criteria post hoc to make a result look better. Threshold and
  protocol revisions go through `docs/02` amendments and are
  recorded in `docs/03`.
- Do not treat C-P13 as model selection. C-P13 is a fixture-specific
  diagnostic. The Doc 02 selection study is the only mechanism that
  produces a base-model decision.
- Use `DAGMAWrapper.get_diagnostics()` for selection-study logging
  rather than reading the wrapper's private fields. If a needed
  diagnostic is missing, widen the returned record deliberately
  rather than introducing private-attribute access.
- Preserve decision-log discipline in `docs/03`. Every architectural
  decision that affects scientific outputs gets a contemporaneous
  entry there.
- Do not unpause DCDI Commit 11 unless explicitly instructed after a
  project-level review of the DCDI sampler-quality findings.
- Do not begin DAGMA prior-penalty integration before a defensible
  selection result. If DAGMA is later selected, the loss-hook
  integration plan is a new artefact and must come before any source
  patch.

## Interpretation guardrails

These constraints apply to any future summary or aggregation that
cites C-P13:

- C-P13 is not a proof that DAGMA is in general better than DCDI; it
  is a same-fixture diagnostic comparison.
- DAGMA is not selected. No claim in this readout, in
  `docs/04h_dagma_sampler_quality_diagnostic.md`, or anywhere else
  should be read as a selection decision.
- The 91x wrong/correct ratio is a structure-sensitivity sanity
  check produced by deleting the dominant downstream true edge, not
  a general quality metric. Treat the number with appropriate
  context.
- The literal primary inequality
  `wrapper_vs_truth_mmd <= 3 * floor_mmd` did not pass on this run;
  the floor MMD is negative and the literal inequality is therefore
  non-informative. The substantive read is the absolute-scale gap.
- Diagnostic A is not an independent structural localisation signal
  on this fixture, because DAGMA's learned thresholded adjacency
  equals the true adjacency; Diagnostic A re-samples the same
  generative process under a different seed lane.
- Diagnostics B1 and B2 did not test DAGMA's weak-edge or
  thresholding-suppression failure modes on this fixture; they were
  not applicable because no true edges were missing. Those failure
  modes remain untested here.
- SID verification is now closed. The next implementation step on the
  project critical path is base-model selection execution, not
  loss-term injection.
