# Phase 2b Readout - DCDI Wrapper Subphase Complete (Paused for Review)

**Date:** 2026-05-13
**Author:** Ben Taliana
**Purpose:** Orient any new collaborator to the project state at the end of the DCDI wrapper subphase. The DCDI wrapper implementation has been carried up to and including the sampler-quality validation step. The validation step failed and has been converted into a diagnostic artefact. This readout is the handover point before DAGMA wrapper planning begins.

This readout is an orientation layer only. The authoritative documents listed below take priority if there is any conflict.

---

## 1. Canonical documents (read in this order)

1. `docs/01_research_question_and_commitments.md` - frozen scientific contract.
2. `docs/02_base_model_selection.md` - frozen base-model selection protocol.
3. `docs/03_decision_log.md` - tactical decisions, evaluator conventions, wrapper-phase conventions.
4. `docs/04b_source_inspection.md` and `docs/04c_runtime_probe_plan.md` / `docs/04c_runtime_probe_results.md` / `docs/04d_dcdi_mask_probe_results.md` - inspection and probe trail for the DCDI wrapper.
5. `docs/04e_equivalence_calibration_results.md` - calibration record for the wrapper-vs-reference training equivalence test.
6. `docs/04f_dcdi_sampler_quality_diagnostic.md` - failure record for the DCDI sampler-quality validation step (originally Commit 10).
7. `docs/04g_equal_variance_identifiability_check.md` - follow-up sanity check showing the C-P11 fixture is recoverable in principle.
8. `docs/05_dcdi_wrapper_implementation_plan.md` - DCDI wrapper plan with execution status now folded in.
9. `docs/phase_1_readout.md` - evaluator-foundation handover from the previous phase.

---

## 2. DCDI wrapper commit status

The DCDI wrapper implementation plan in `docs/05_dcdi_wrapper_implementation_plan.md` is a 14-commit sequence with Commit 5 as the behavioural-equivalence gate and Commit 10 as the sampler-quality gate.

### Completed (green in normal pytest collection)

- Commit 1 - wrappers scaffolding, status taxonomy, `WrapperDiagnostics` TypedDict.
- Commit 2 - `CentredOnlyTransform` and `StandardisedTransform` preprocessing with intervention-value transform.
- Commit 3 - DCDI low-level helpers (pinned-source import, `make_dcdi_model`, parameter snapshot helpers) with import-isolation tests.
- Commit 4 - augmented-Lagrangian training loop with continuous-edge preservation (no second-stop `log_alpha` saturation).
- Commit 5 - behavioural-equivalence gate against a hand-replicated reference loop, plus the C-P10 calibration probe and `docs/04e_equivalence_calibration_results.md`.
- Commit 6 - thresholding helper and graph-status machinery; no silent graph repair.
- Commit 7 - structural-mask context manager that enforces a thresholded mask during sampling and restores live model state on exit (including on exception).
- Commit 8 - model-frame ancestral sampler with target clamping, deterministic seeding, structural-mask enforcement, and refusal on invalid graph.
- Commit 9 - raw-unit intervention roundtrip via `sample_raw_units_dcdi`, using the supplied preprocessor without mutating it.

### Paused

- **Commit 10 - sampler-quality validation: diagnostic failed.** Both original acceptance thresholds (`wrapper_vs_truth <= 3 * floor` and `correct * 1.5 <= wrong`) were missed on the project SCM fixture. The failing pytest module has been removed from normal collection and converted into the diagnostic probe `inspection/probes/c_p11_dcdi_sampler_quality_diagnostic.py` and the report `docs/04f_dcdi_sampler_quality_diagnostic.md`. No threshold has been weakened.

- **Commit 11 - loss-hook injection: paused pending project-level review** of the Commit 10 diagnostic.

- Commits 12, 13, and 14 remain blocked behind Commit 11.

### Test-suite state

Normal pytest collection currently reports **190 passed, 1 skipped**. The single skipped test is the SID scaffold that was already deferred at the end of Phase 1.

---

## 3. Key C-P11 diagnostic findings

Setup: 3-node ER2 SCM at seed 0; 5000 observational training samples (seed 1); 500 validation samples (seed 2); `CentredOnlyTransform`; `make_dcdi_model(num_layers=2, hid_dim=8)`; `DCDIConfig()` paper defaults; DCDI seed 0; `n_iter = 30000`; intervention `do(X_2 = 2.0)`.

- DCDI training reached `final_h = 5.3e-4` but did not satisfy the `h_threshold = 1e-8` convergence test. `converged = False`.
- Learned thresholded adjacency at 0.5: edges `0 -> 1` and `2 -> 1`. The strongest true edge `2 -> 0` (true weight magnitude 1.7861) was **missing** from the learned structure (its continuous `w_adj` value was 0.23, below the 0.5 threshold).
- `graph_status` for the learned adjacency: `valid_dag`.
- Monte Carlo floor MMD (median pairwise across 5 ground-truth batches): `-1.756e-4`. The unbiased estimator can be negative when both samples come from the same distribution, so `3 * floor` is not a usable positive acceptance threshold here; the substantive comparison is the order-of-magnitude gap between floor scale (`~1e-4`) and wrapper-vs-truth MMD.
- Wrapper-vs-truth median MMD: `+6.275e-1`. Three orders of magnitude above the floor scale.
- Correct-structure vs wrong-structure ratio: `1.199`, below the required `1.5`.
- Diagnostic A (MMD under the TRUE adjacency, same DCDI conditionals): `+5.26e-2`. About 12x reduction from the wrapper-vs-truth result.
- Diagnostic B (MMD under learned + strongest missing true edge): `+4.23e-2`. About 15x reduction.

**Localisation.** The two added diagnostics localise the failure to learned **structure quality**. DCDI's per-node conditional distributions are usable; sampler mechanics tests (clamping, structural masking, restoration on exception, deterministic seeding, raw-unit roundtrip) all remain green in normal pytest collection.

---

## 4. Key C-P12 follow-up finding

Setup: same SCM fixture as C-P11; same 5000-sample observational training data; raw SCM units; all 25 DAGs on 3 nodes enumerated.

Scoring: for each candidate DAG, OLS regression of each node on its candidate parents with intercept; per-node residual sums of squares pooled into a single shared-variance MLE; Gaussian BIC under the equal-error-variance assumption with `k_params = n_edges + d + 1`.

- Total DAGs enumerated: **25** (matches the known count for labelled DAGs on 3 nodes).
- True DAG rank: **1 of 25**. BIC = 42502.18. `sigma2_hat = 0.9917` (matches the data-generating `noise_scale = 1.0`).
- Rank-1 to rank-2 BIC margin: **+232.22**. Well above any conventional "very strong evidence" BIC gap.
- DCDI-learned DAG rank: **19 of 25**. BIC = 53136.49. BIC delta from top = `+10634.31`. `sigma2_hat = 2.0161` (about twice the true variance, because the missing strong edge `2 -> 0` dumps its variance into the residual).

**Interpretation.** The C-P11 fixture is recoverable in principle when the equal-error-variance assumption is encoded, despite the general Markov-equivalence caveat for arbitrary linear-Gaussian SCMs. The project SCM uses homogeneous noise across all nodes, so equal-variance applies on this data. DCDI-G's failure on this fixture is therefore best read as a base-model **inductive-bias / optimisation / model-mismatch** issue (per-node nonlinear MLPs with augmented-Lagrangian DAG penalty, no equal-variance assumption, no second-stop `log_alpha` saturation under the wrapper's preservation policy), not as data impossibility.

C-P12 is a **sanity check, not a baseline**. Exhaustive enumeration does not scale beyond 5-6 nodes, and equal-variance Gaussian BIC carries strong assumptions that the project main study has not committed to.

---

## 5. Project decision and posture

- DCDI loss-hook injection (Commit 11) and all downstream DCDI wrapper commits remain **paused**. No DCDI Commit 11 work begins without a project-level review of the C-P11 and C-P12 evidence.
- Sampler mechanics for DCDI are considered validated by the green unit-test layer (Commits 1 through 9). Continuous-edge preservation, structural-mask context, sampler clamping, structural-mask enforcement, raw-unit roundtrip, and behavioural equivalence against a hand-replicated reference loop all hold.
- The next priority is the DAGMA wrapper design and implementation. The base-model selection study cannot run until both candidate wrappers exist and verified SID is integrated.
- Verified SID integration remains a **load-bearing requirement** for the selection study and is unchanged by this subphase. The SID scaffold test in the normal suite stays intentionally skipped until verified SID is integrated and tested against a hand-computed small case.
- No acceptance threshold has been weakened anywhere as part of this subphase. No silent graph repair has been introduced. No source file under `src/` has been edited to make the sampler-quality test pass.

---

## 6. Next recommended artefact

`docs/06_dagma_wrapper_implementation_plan.md`

It should follow the same plan-then-implement discipline used for `docs/05_dcdi_wrapper_implementation_plan.md`: a frozen commit sequence, an acceptance criterion per commit, a behavioural-equivalence gate against the inspected DAGMA source, an explicit sampler-quality validation step against the project MMD primitive, and a documented stance on any deviation from the DAGMA paper or library.

Specific points the plan must address up front:

- the DAGMA-linear API, expected adjacency convention, and continuous-W extraction.
- preservation policy for the continuous `W` matrix at thresholding time, paralleling the DCDI continuous-edge preservation policy in `docs/05_dcdi_wrapper_implementation_plan.md`.
- whether DAGMA mean-centring side effects on input arrays require defensive `X.copy()` calls (the convention in this project's wrapper-phase guidance).
- how DAGMA's continuous output integrates with the project's row-source / column-destination adjacency convention.
- a sampler-quality validation step calibrated on the same kind of fixture used in C-P11, so DAGMA and DCDI sampler-quality outcomes can be compared on equal footing.

---

## 7. Non-negotiable conventions carried forward

These hold from earlier phases and continue to apply.

- Metric argument order is `(predicted, true)`.
- `sid_score` lives in `metrics/interventional.py`.
- Shared adjacency validation lives in `metrics/_graph_validation.py`.
- SHD default reversal cost is `2`.
- MMD uses the unbiased RBF estimator; the median heuristic is computed on concatenated samples; the primitive does not clip negative values.
- Evaluator gate defaults: `mmd_tolerance = 0.01`, `clamp_tolerance = 1e-12`, `require_sid = False`. Failed SID is always a hard error.
- Project adjacency convention is row-source / column-destination: `adjacency[i, j] = True` means edge `i -> j`. Wrappers must convert external model outputs into this convention before passing adjacency matrices to evaluator metrics or downstream sampling logic.
- DCDI sampling enforces the thresholded structure through `model.adjacency` plus saturated `log_alpha`, transient for the duration of sampling and reverted in a `finally` block. The `mask=` argument of `forward_given_params` is not used as a structural mask.
- No silent graph repair anywhere in the wrapper layer.

---

## 8. Repo state at this readout

- Normal pytest collection: **190 passed, 1 skipped**.
- The skipped test is the SID scaffold from Phase 1.
- New artefacts introduced during this subphase that live outside normal pytest collection: `inspection/probes/c_p10_equivalence_calibration.py`, `inspection/probes/c_p11_dcdi_sampler_quality_diagnostic.py`, `inspection/probes/c_p12_equal_variance_identifiability_check.py`, `docs/04e_equivalence_calibration_results.md`, `docs/04f_dcdi_sampler_quality_diagnostic.md`, `docs/04g_equal_variance_identifiability_check.md`.
- Source files under `src/symbolic_priors_cd/wrappers/` cover the wrapper status taxonomy, preprocessing, DCDI low-level helpers, training loop, thresholding and graph-status machinery, and the structural-mask context plus model-frame and raw-unit samplers. All are exercised by green unit tests.
- No DAGMA wrapper code, no selection-study runner, no soft-prior layer, and no verified SID implementation has been added in this subphase.

This readout is the handover point at the end of the DCDI wrapper subphase and the start of DAGMA wrapper planning.
