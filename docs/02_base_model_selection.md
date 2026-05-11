# 02_base_model_selection

## Status

Frozen protocol for base-model selection.  
Version 1.3.  
This document defines how the thesis chooses between DAGMA-linear and DCDI-G as the base differentiable causal discovery model for the main study.

---

## Change log

- **v1.1** 22/04/2026: Initial selection study protocol created.
- **v1.2** (this amendment): froze selection-study tactical details supported by `docs/04a_orientation_audit.md`, `docs/04b_source_inspection.md`, and `docs/04c_runtime_probe_results.md`. Changes include: ER2 wording clarification; data-variant rename to centred-only and standardised; explicit DAGMA hyperparameter overrides plus wrapper preconditions (`X.copy()` rule, continuous-`W` preservation, internal `w_threshold=0.0` with project threshold 0.3 applied externally); DCDI wrapper preconditions (`normalize=False`, low-level imports only, no `dcdi.train`); wrapper-side MMD sampling policy with DAGMA residual-fitted noise as primary and unit-variance as sensitivity check; project-owned preprocessing with explicit transform equations; expanded run logging including continuous edge objects and sampler statuses; MMD-unavailable tie rule with quantitative thresholds; calibration-versus-evaluation seed-split fix preventing leakage; threshold robustness reporting at neighbouring thresholds; SID gate before scientific completeness; determinism scope statement; hard-constraint warning; and Section 9 update reflecting the new tactical constants.
- **v1.3**: integrated the DCDI structural-mask probe from `docs/04d_dcdi_mask_probe_results.md`; clarified that DCDI MMD sampling must enforce the thresholded graph through `model.adjacency` plus saturated `log_alpha`; fixed internal consistency issues in Criterion 3 wording, intervention-unit wording, Section 6 tie-breaker logic, Phase B seed accounting, paper-aligned reproduction-cell specification, and minor editorial issues.

---

## 1. Purpose

This document selects the base model for the thesis by comparing **DAGMA-linear** and **DCDI-G** on a fixed selection study cell and freezing only the tactical constants required for that selection. It does **not** define the full main-study experiment. Main-study tactical constants are deferred to Document 03 after the selection study winner is chosen.

The selection study is not intended to produce final thesis results. Its purpose is to choose, scientifically and transparently, the model best aligned with the thesis question: **observational-only training with evaluation on unseen interventions under uncertain symbolic priors**.

---

## 2. Decision rule

Base-model selection is determined by **lexicographic ordering** over three criteria.

### Criterion 1: Interventional adequacy

This is the dominant criterion because the thesis is primarily about unseen-intervention behaviour rather than structure recovery alone.

#### Criterion 1a: SID-based interventional adequacy

For each model, compute mean SID across the selection study intervention set and seed set.  
**Winner:** the model with lower mean SID.

#### Criterion 1b: MMD-based distributional fidelity

If mean SID differs by **10% or less**, compare mean MMD across the same intervention set and seed set.  
**Winner:** the model with lower mean MMD.

#### Rule for Criterion 1

- If SID differs by **more than 10%**, Criterion 1 is decided by SID alone.
- If SID differs by **10% or less**, MMD is used as the tiebreaker inside Criterion 1.

### Criterion 2: Prior-injection ergonomics

If Criterion 1 does not decisively separate the models, compare how cleanly each model supports the thesis method.

This criterion evaluates:

1. how many lines of code must be added to integrate the soft-prior penalty;
2. whether the penalty integrates **without changing the optimiser scheme**;
3. whether penalised edges actually shrink as intended in a smoke test.

**Winner:** the model that requires less architectural surgery, ideally preserves the original optimisation pipeline, and demonstrates correct penalised-edge behaviour in the smoke test.

### Criterion 3: Standardisation robustness

If Criteria 1 and 2 remain effectively tied, compare structural recovery under the centred-only and standardised data variants.

For each model, compute mean SHD in both settings and measure the relative increase after standardisation.

A model is considered **catastrophically dependent on scale artefacts** if mean SHD increases by **more than 50%** after standardisation.

**Winner:** the model with the smaller SHD degradation under standardisation.

### Final tie principle

If all criteria remain tied within the pre-registered margins, select the model **better aligned with the main-study interventional evaluation pipeline**, with engineering reliability treated as a secondary consideration. The final decision and justification must be written explicitly in the selection study report.

This lexicographic ordering is deliberate. SID is prioritised because the thesis is primarily concerned with interventional adequacy. MMD is secondary but still useful as a distributional check when SID is effectively tied. Weighting the criteria equally would risk selecting a base model for reasons misaligned with the main research question.

---

## 3. Selection study setup

### 3.1 Synthetic data specification

The selection study uses one thesis-aligned synthetic cell:

- **Graph family:** Erdős–Rényi DAG
- **Number of nodes:** 10
- **Edge density (ER2):** `expected_edges = 2 * n_nodes`, so the 10-node cell uses `expected_edges = 20`. Under the project's ordered-DAG construction this corresponds to an expected out-degree of approximately 2.
- **Mechanism family:** linear Gaussian SCM
- **Observational sample size:** 1000
- **Noise model:** fixed-variance Gaussian noise (unit variance per node, per Doc 01 and the decision log)
- **Seed count:** 5 independent graph/data seeds
- **Data variants:** **centred-only** and **standardised**. The previously named _unstandardised_ condition is operationalised as centred-only and unscaled (per-variable mean subtracted, no variance scaling). This is an operational clarification of the previous wording, not a change to the thesis-level commitment. The standardised condition additionally divides each variable by its training standard deviation. Preprocessing is owned by the project, not by the wrappers; see Section 4.4 for the transform equations.

### Justification

This cell is used because Document 01 already commits the thesis to a 10-node ER2 primary benchmark logic, observational-only training, and unseen-intervention evaluation. The selection study therefore uses the thesis-aligned decision cell rather than reenacting the original papers’ preferred reporting setups.

### 3.2 Models under test

The selection study compares exactly two candidates:

1. **DAGMA-linear**
2. **DCDI-G in observational-only mode**

#### DAGMA-linear

Implementation source: official DAGMA codebase.  
Use the linear Gaussian setting and start from the paper-grounded supplementary defaults.

#### DCDI-G

Implementation source: official DCDI codebase.  
Run DCDI-G in the observational setting by using the formulation where the observational environment is represented by **\(I_1 := \emptyset\)**. This does **not** violate the thesis commitment to observational-only training. The supplement’s **DCD-no-interv** result is treated as confirmation that DCDI-G is well-defined under purely observational data.

### 3.3 Calibration runs

The held-out evaluation phase uses **5 independent evaluation seeds** per model per condition. Phase B additionally uses **2 calibration seeds per configuration**, as described below.

#### Justification

The selection study is a selection procedure designed to detect large practical differences, not to estimate final effect sizes precisely. Five seeds provide enough information for a defensible selection while keeping compute manageable.

Calibration is run in **two phases**.

#### Phase A: reproduction pass

Each model is first run under a paper-grounded/default configuration to verify that:

- the code runs correctly;
- the implementation is compatible with the selection study cell;
- one result from the paper or supplement can be approximately reproduced.

The paper-aligned reproduction cell must be declared before running the reproduction pass.

For DAGMA-linear, the reproduction pass uses the closest paper-reported small linear-Gaussian ER cell available in the DAGMA supplementary material, rather than treating the 10-node thesis cell as paper-reported if no such 10-node DAGMA result exists.

For DCDI-G, the reproduction pass uses the closest reported DCD-no-interv linear-Gaussian cell from the DCDI supplementary material.

Any mismatch between the paper-reported cell and the project reproduction cell must be recorded before the reproduction result is interpreted.

#### Phase B: equal-budget local calibration

After the reproduction pass, each model is given a small equal-budget local calibration opportunity on the selection study cell. The budget is **5 configurations per model and 2 calibration seeds per configuration**. The selected configuration from Phase B is then evaluated on **5 held-out evaluation seeds** that do not overlap with the 2 calibration seeds. The total seed budget per model is therefore 2 calibration seeds plus 5 held-out evaluation seeds, drawn from independent populations and recorded.

Calibration and evaluation seeds must not overlap. This prevents the leakage path where the configuration is selected on the same seeds it is later evaluated on. If compute pressure forces reuse, the resulting Criterion 1 numbers must be reported as **descriptive** rather than as held-out evaluation, and the selection-study report must state explicitly that the seed pools overlap.

##### DAGMA-linear starting point

Use the following explicit configuration values for DAGMA-linear in the selection study. These override library defaults because library defaults diverge from the supplementary configuration referenced in the DAGMA paper:

- number of iterations: `T = 4`
- L1 coefficient: `lambda1 = 0.05`
- log-det parameter sequence: `s = [1.0, 0.9, 0.8, 0.7]`
- initial central-path coefficient: `mu_init = 1.0`
- decay factor: `mu_factor = 0.1`
- in-library threshold passed to `DagmaLinear.fit`: `w_threshold = 0.0`
- project-level threshold applied externally to `abs(W_continuous)` at the wrapper output boundary: `0.3`

Wrapper preconditions for DAGMA-linear:

- The wrapper MUST pass `X.copy()` to `DagmaLinear.fit` because DAGMA mutates its input array in place during L2 mean-centering (verified by probe D-P4 in `docs/04c_runtime_probe_results.md`).
- The wrapper MUST save the continuous `W_est` immediately after `fit` returns, before applying the project-level threshold. The continuous matrix is the native edge object required by the wrapper contract; it is used for threshold robustness reporting (Section 7) and for model-generated MMD sampling (Section 4.2).

Any local calibration in Phase B that changes a value above must be logged explicitly.

##### DCDI-G starting point

Use the official DCDI-G implementation with a small held-out validation split and a restricted local search around the paper-style sparsity regularisation setup.

- Lagrangian multiplier: `gamma_0 = 0`
- penalty coefficient: `mu_0 = 10^-8`
- Xavier initialization
- adjacency-matrix entries initialised at or near 1.0
- learning rate: `10^-3`
- penalty update factor: `n = 2`; decrease threshold: 0.9
- stopping criterion: `h(Lambda) < 10^-8`
- RMSprop optimiser; minibatch size = 64
- sparsity coefficient: 5 values spanning `10^-7` to `10^2`

Wrapper preconditions for DCDI-G:

- The wrapper MUST call DCDI's `DataManagerFile` (or the equivalent data path) with `normalize=False`. Project-level preprocessing is upstream of the wrapper, per Section 4.4.
- The wrapper MUST NOT import `dcdi.train`. That module pulls in `cdt.utils.R` and triggers an optional R dependency chain (see `docs/04b_source_inspection.md` item C-10).
- The wrapper uses the low-level imports confirmed by source inspection and runtime probes (`docs/04c_runtime_probe_results.md` C-P1 and C-P2): `dcdi.models.learnables.LearnableModel_NonLinGaussANM`, `dcdi.dag_optim.{GumbelAdjacency, compute_dag_constraint}`, and `dcdi.utils.penalty.compute_penalty`.

### 3.4 What gets logged

Each run must save:

- graph seed
- data seed
- preprocessing condition flag (centred-only or standardised)
- model configuration (including the explicit DAGMA overrides from Section 3.3 and the DCDI starting-point values)
- runtime
- SHD
- SID (once verified SID is integrated; before then, SID is logged as deferred, see Section 7)
- MMD primary value (residual-fitted DAGMA noise; native DCDI conditionals) and MMD sensitivity value (unit-variance DAGMA noise)
- bandwidth-sweep MMD values at 0.5x, 1.0x, and 2.0x of the median heuristic (per Section 4.2)
- validation NLL where applicable
- **thresholded boolean adjacency** in the project row-source / column-destination convention
- **continuous native edge object** at full precision: DAGMA `W_continuous` (the matrix returned by `fit` with `w_threshold=0.0`), DCDI `model.gumbel_adjacency.log_alpha` and `model.get_w_adj()`. Saved so threshold robustness can be assessed offline without retraining.
- **training_status, graph_status, sampler_status** values as defined in `docs/04_wrapper_api_contract.md` Section 7
- **MMD sampling policy used** for the run (residual-fitted DAGMA noise, unit-variance DAGMA noise, or DCDI native conditionals) and the **MMD sampling RNG seed plus derivation rule** so MMD draws are reproducible independently of training
- **DAGMA per-node sigma vector** when residual-fitted noise is used, so the noise estimate is auditable
- intervention outputs used in Criterion 1
- **MMD missingness counts**: number of seed-intervention cells where MMD could be computed, number where it could not, and the **invalid graph rate** (fraction of runs where `graph_status` was not `valid_dag`)
- **MMD unavailability reasons** when applicable, taken from the `sampler_status` taxonomy (`unavailable_invalid_graph`, `unavailable_no_api`, `unavailable_unresolved_noise_policy`)
- convergence / failure notes

### 3.5 Logging and reproducibility

All selection study runs are stored under:

`results/model_selection/<model>/<condition>/<seed>/`

Each run directory must include:

- config file
- git hash
- environment snapshot
- random seed
- raw outputs
- computed metrics
- notes on any failure mode

No selection study conclusion may rely on a result that cannot be traced to a saved run directory.

#### Determinism scope

- DAGMA appears deterministic on a fixed platform given fixed input data and explicit hyperparameters (DAGMA has no internal random initialisation in `fit`; W starts at zero and Adam is deterministic given the inputs). Wrapper-level repeatability must still be re-checked once the wrapper is implemented, because the wrapper introduces its own seed-handling code.
- DCDI was verified to produce bitwise-identical `log_alpha` on CPU under a 5-step controlled run with seeds set (probe C-P8 in `docs/04c_runtime_probe_results.md`). This is small-scale evidence only.
- The selection study MUST NOT claim hardware-independent or production-scale bitwise determinism. The wrapper documentation must instead state that reproducibility is guaranteed within a documented numerical tolerance on the project hardware.
- All runs MUST set fixed seeds for both `torch.manual_seed` and `np.random.seed` (DCDI) and for `dagma.utils.set_random_seed` (DAGMA), even when these would be no-ops, so the seed discipline is uniform across candidates. Numerical tolerances applied to any reproducibility check MUST be documented in the run record.

---

## 4. Evaluation criteria — operational definitions

### 4.0 Ground-truth compatibility check

Before any model comparison begins, the evaluator must pass the following sanity checks on at least one sampled SCM:

1. SID of the true graph against itself is exactly zero.
2. MMD between two independently sampled interventional batches from the same SCM is near zero up to Monte Carlo noise.
3. The SCM intervention sampler behaves correctly under a manual `do(X_j = x)` check.

If any of these checks fail, the selection study halts until the evaluator is fixed.

### 4.1 Criterion 1a: SID-based interventional adequacy

For each fitted model:

1. derive the predicted graph or thresholded native edge object;
2. compute SID against the ground-truth DAG;
3. repeat across all selection study interventions and seeds;
4. average over the full selection study intervention set.

This is the primary subcriterion because SID directly measures intervention mistakes implied by the learned structure.

### 4.2 Criterion 1b: MMD-based distributional fidelity

If Criterion 1a is effectively tied, compare the models using MMD between:

- ground-truth interventional samples from the SCM; and
- model-generated interventional samples.

#### Intervention set

The selection study uses **single-node hard interventions** at:

- \(do(X_j = -2)\)
- \(do(X_j = +2)\)

for eligible nodes in the selection study graph.

#### Justification

These are non-trivial interventions in raw SCM units. For each model, the raw intervention value is transformed into the relevant model frame using the preprocessing equations in Section 4.4 before clamping. The values remain symmetric around zero and provide a non-trivial perturbation for the linear-Gaussian selection cell. The choice of \(+2\) is also broadly consistent with the DCDI paper's perfect-intervention regime (marginal \(N(2,1)\) on targeted nodes) without making the intervention semantics depend on a standardised frame.

#### MMD rule

- default bandwidth initialisation: median heuristic
- mandatory sensitivity check: bandwidth sweep (0.5×, 1.0×, 2.0× median heuristic), ordering must be stable across all three to count

#### Wrapper-side sampling policy

Neither DAGMA nor DCDI exposes a built-in sampler under arbitrary `do(X_j = v)`. Wrappers implement model-generated interventional sampling using the verified mechanisms below. Notation: `mean_j` and `std_j` are the per-variable training mean and standard deviation fitted by project preprocessing (Section 4.4).

**DAGMA sampling, primary policy: residual-fitted per-node noise.**

1. Compute the thresholded boolean adjacency from the continuous matrix saved per Section 3.3: `A_thresh = abs(W_continuous) >= 0.3`.
2. If `A_thresh` is not a valid DAG, set `sampler_status = unavailable_invalid_graph` and contribute no MMD value for that run.
3. Build the sampling weight matrix from the continuous weights on surviving edges: `W_sample = W_continuous * A_thresh`.
4. Estimate per-node noise on the model-frame training data: `R = X_model_frame - X_model_frame @ W_sample`, then `sigma_j = std(R[:, j])` for each node `j`.
5. Sample ancestrally in topological order on `A_thresh`. At the intervention target, clamp the (model-frame) value. Other nodes are drawn as `X_j = X_parents @ W_sample[parents, j] + N(0, sigma_j)`.
6. Transform raw intervention values into the model frame BEFORE clamping using the equations in Section 4.4 (`v_model = v_raw - mean_j` for centred-only; `v_model = (v_raw - mean_j) / std_j` for standardised). Transform generated samples back to raw SCM units BEFORE MMD comparison.
7. **Log the per-node sigma vector** (Section 3.4).

**DAGMA sampling, sensitivity check: unit-variance noise.**

8. Repeat steps 1, 2, 3, 5, 6 with `sigma_j = 1.0` for every node. Report MMD under both noise policies. Unit-variance noise is a sensitivity check, not the primary policy. If the two policies disagree in selection-relevant direction, the selection-study report MUST flag the disagreement explicitly.

**DCDI sampling, verified API path.**

9. Compute the thresholded boolean adjacency from `A_thresh = model.get_w_adj() >= 0.5`. The thresholded valid DAG is the active sampling graph; soft edges in `P = sigmoid(log_alpha)` that do not survive the project threshold MUST NOT be used as parents during sampling.
10. If `A_thresh` is not a valid DAG, set `sampler_status = unavailable_invalid_graph` and contribute no MMD value for that run.
11. Before DCDI MMD sampling, the wrapper must enforce the thresholded structural mask by temporarily setting `model.adjacency` to a tensor copy of `A_thresh` and setting `model.gumbel_adjacency.log_alpha` to a saturated tensor consistent with `A_thresh` (large positive values on surviving edges, large negative values elsewhere). This is the project wrapper policy because DCDI's structural forward pass multiplies `M = gumbel_adjacency(bs)` by `model.adjacency`. Setting both objects makes the active parent set deterministic and prevents sub-threshold edges from contributing during sampling. The wrapper must restore the original `model.adjacency` and `log_alpha` after sampling, preferably through a save-mutate-sample-restore context. The `mask=` argument to `forward_given_params` is NOT used as a structural parent mask in observational mode; it is the intervention mask in DCDI's source convention (see `docs/04d_dcdi_mask_probe_results.md`).
12. For each batch, traverse nodes in topological order of `A_thresh`. For each non-target node, call `model.forward_given_params(x, weights, biases)` and select the density parameters for that node, then build the conditional Normal via `model.get_distribution(...)` and draw a sample. At the intervention target, clamp the (model-frame) value. The minimal call pattern (`mask=None, regime=None`) is sufficient in observational mode (verified by probes C-P5 and C-P7 in `docs/04c_runtime_probe_results.md`).
13. Transform raw intervention values into the model frame before clamping and generated samples back to raw SCM units before MMD comparison, using the equations in Section 4.4.

The MMD sampling RNG seed and the seed-derivation rule MUST be logged per Section 3.4. Sample size for MMD comparison defaults to 1000 model-generated samples per intervention condition, matching the observational sample size frozen in Section 9.

### 4.3 Criterion 2: Prior-injection ergonomics

This criterion tests how naturally each model supports the thesis method.

#### Smoke test

For each model:

1. select a small set of correct forbidden-edge priors;
2. add the soft-prior penalty;
3. fit on a small 10-node ER2 instance;
4. compare penalised-edge behaviour against an unpenalised run.

#### Scoring dimensions

- **Code change size:** smaller is better
- **Optimiser preservation:** no optimiser redesign is preferred
- **Penalty behaviour:** penalised edges must shrink as intended

For DAGMA, the relevant object is the weighted adjacency matrix \(W\).  
For DCDI, the relevant object is the relaxed edge-probability matrix \(P = \sigma(\Lambda)\).

#### Hard-constraint warning

The hard-constraint baseline in the main study is conceptually distinct from post-threshold masking. Setting selected entries of a boolean adjacency to False AFTER training is NOT a substitute for enforcing those constraints DURING training. Training-time hard constraints, if used in the main study, must be designed explicitly at the optimiser level (for example via DAGMA's `exclude_edges` and `include_edges` parameters, or via DCDI's mask multiplications inside the forward pass) and their design must be recorded in Doc 03 before any main-study run.

### 4.4 Criterion 3: Standardisation robustness

#### Project-owned preprocessing

Preprocessing is owned by the project, not by the wrappers. The two preprocessing conditions are produced upstream of the wrapper and the wrapper receives the already-transformed data. Let `mean` and `std` denote the per-variable training mean and standard deviation, fitted on the training data only. Define the conversions as follows.

**Centred-only condition:**

```
x_model = x_raw - mean
x_raw   = x_model + mean
v_model = v_raw   - mean_j     (intervention value on node j)
```

**Standardised condition:**

```
x_model = (x_raw - mean) / std
x_raw   = x_model * std + mean
v_model = (v_raw - mean_j) / std_j
```

Means and standard deviations are fitted on training data only and reused for test-time samples and for transforming intervention values. No test-set leakage is permitted.

Wrapper consequences (recorded here for completeness; the same statements appear in Section 3.3):

- DAGMA receives `X.copy()` of the already-centred (or already-standardised) data. DAGMA's internal mean-centering at `linear.py:307` is redundant under the centred-only condition and is consistent with the standardised condition (centring a zero-mean array is a no-op). The wrapper does not need to disable DAGMA's centring.
- DCDI receives data through `DataManagerFile(..., normalize=False)`. No internal DCDI normalisation runs.

#### Robustness comparison procedure

For each model:

1. fit on the centred-only condition and compute SHD;
2. fit on the standardised condition and compute SHD;
3. compute the relative change.

A model is considered catastrophically dependent on scale artefacts if mean SHD increases by **more than 50%** between the centred-only and the standardised condition. Phase B hyperparameters are used for both conditions.

---

## 5. Disqualification conditions

A model is disqualified if any of the following occurs:

1. it cannot be installed and run within **5 working days**;
2. it cannot reproduce one paper/supplement result within **20%** on the selection study cell or a closely aligned cell;
3. it produces NaN, divergence, or non-converged training in **more than 50%** of selection study seeds;
4. it cannot produce usable intervention outputs for Criterion 1 without ad hoc undocumented modifications;
5. it fails the prior-injection smoke test in a way that shows the penalty does not meaningfully act on the native edge object.

A disqualified model is removed from further comparison and the reason is recorded in writing.

The primary performance summaries for DAGMA in the supplementary are:
Table 1 (Page 21): Summarises the average SHD and runtime for small to moderate numbers of nodes (d∈{20,30,50,80,100}) across different graph and noise types.
Table 2 (Page 22): Summarises performance for large-scale graphs with d∈{200,300,500,800,1000}

For the DCDI-G model variant DCD-no-interv (which represents DCDI-G applied strictly to purely observational data), the specific results for 10-node linear models are located in the following parts of the supplementary material:
Specific Table and Location:
Table: Table 7.
Section: Appendix C.4.1 ("Perfect interventions").
Context: This table provides an ablation study comparing DCDI variants against purely observational baselines to demonstrate the benefit of interventional data.

---

## 6. Tiebreakers and edge-case rules

### Case 1: SID is decisive

If mean SID differs by more than 10%, Criterion 1 is decided by SID alone.

### Case 2: SID is within the 10% margin and MMD is available

If mean SID differs by 10% or less and MMD is available under Case 6, MMD decides Criterion 1.

### Case 3: Criterion 1 remains inconclusive

If Criterion 1 remains inconclusive after applying SID, MMD, and the MMD-unavailable rule, use Criterion 2.

### Case 4: Criteria 1 and 2 remain tied

If Criteria 1 and 2 remain tied, use Criterion 3.

### Case 5: Both models fail

If both models are disqualified or both fail to produce a defensible win under the pre-registered criteria, the selection study is declared inconclusive. In that case:

- the failure is documented honestly;
- the most scientifically defensible fallback is chosen based on the final tie principle;
- and the limitation is recorded in the thesis decision log.

### Case 6: MMD-unavailable rule

MMD may be unavailable for a subset of seed-intervention cells, for example because the thresholded graph from a candidate is not a valid DAG, the sampler returns an error, or the noise policy is not yet frozen for a given candidate. The selection-study report MUST then state:

- mean MMD over the available cells (with the available count noted),
- the missing count,
- the invalid graph rate (fraction of runs whose `graph_status` was not `valid_dag`),
- the unavailable reasons taken from the `sampler_status` taxonomy.

If the invalid-or-unavailable rate exceeds **20 percent** for a candidate, the MMD comparison MUST be flagged as **reliability-limited** and the candidate's MMD value MUST NOT be used as the sole basis for a positive Criterion 1b decision.

If MMD is reliability-limited but unavailable for 50% or less of the candidate's seed-intervention cells, the MMD value may be reported, but if it would determine the selected model, Criterion 2 must be reported as a corroborating decision factor. The selection report must state that the Criterion 1b result was reliability-limited.

If mean SID is within the 10 percent tie margin (Criterion 1b is in scope) AND MMD is unavailable for more than **50 percent** of one candidate's seed-intervention cells, that candidate FAILS Criterion 1b, unless both candidates fail similarly (both above 50 percent unavailable). In that joint-failure case, Criterion 1 is declared inconclusive, the final tie principle from Section 2 is applied, and the MMD failure is recorded explicitly in the selection-study report.

This rule is independent of Case 1 through Case 5; it constrains how MMD numbers are interpreted whenever they are partially available. It does not override the lexicographic structure of the decision rule.

---

## 7. Outputs

The selection study must produce:

1. a fully populated `results/model_selection/` directory;
2. a one-page selection study report summarising:
   - setup,
   - scores on all criteria,
   - disqualifications if any,
   - final decision,
   - ground truth compatibility check,
   - and justification;
3. archived trained models or saved native edge objects for each run;
4. a written declaration of the selected base model;
5. a **threshold robustness report**. Using the saved continuous edge objects, recompute headline boolean-adjacency metrics (SHD, SID once integrated, edge counts) at the project default threshold and at two neighbouring thresholds. The threshold triples are:
   - DAGMA: `{0.2, 0.3, 0.4}` applied to `abs(W_continuous)`;
   - DCDI: `{0.4, 0.5, 0.6}` applied to `model.get_w_adj()`.
     Report whether the selection-relevant ordering of the two candidates is stable across the three threshold values. The continuous edge objects saved per Section 3.4 make this re-computation possible without retraining;
6. an **explicit gating statement on SID**. The selection study CANNOT be declared scientifically complete until verified SID is integrated into the project's metrics layer and the SID pre-registered scaffold test in `tests/test_interventional_metrics.py` is unskipped. SID-dependent claims (including any conclusion that depends on Criterion 1a) remain **deferred** until the gate is satisfied. The selection-study report MUST state this gate explicitly whenever it is reporting SID-dependent results before SID is verified.

---

## 8. Timeline and budget

### Timeline

The selection study is allotted **7 working days** from first executable run to declared winner.

### Budget ceiling

The selection study uses the first binding limit among:

- **£50** out-of-pocket spend,
- **30 GPU-hours**,
- or the effective limit of the available university compute allocation.

### Scope-cut hierarchy if the selection study overruns

If the selection study threatens to exceed the time or compute ceiling, scope is reduced in the following order:

1. reduce **Phase B calibration breadth** (configurations or calibration seeds) BEFORE reducing the 5 held-out evaluation seeds. The held-out evaluation pool is the load-bearing population for Criterion 1 and MUST be protected first;
2. reduce non-essential logging visualisations before reducing evaluation metrics;
3. reduce held-out evaluation seed count from 5 to 3 only if absolutely necessary, and only after step 1 has already exhausted calibration-side cuts;
4. do **not** allow calibration seeds and held-out evaluation seeds to overlap as a way to save compute. If overlap is unavoidable, the result MUST be labelled descriptive rather than held-out evaluation, per Section 3.3 Phase B;
5. do **not** expand the model shortlist or the synthetic cell.

If the selection study is still inconclusive after the time/budget ceiling, apply the final tie principle and document the decision honestly.

---

## 9. Tactical constants frozen for the selection study only

The following constants are frozen for the selection study only and do **not** automatically propagate to the main study:

- selection study graph: 10-node ER2, with `expected_edges = 2 * n_nodes = 20`
- observational sample size: 1000
- Phase B seed split: **2 calibration seeds per configuration** plus **5 held-out evaluation seeds**, non-overlapping
- intervention values: \(\{-2, +2\}\)
- catastrophic SHD-degradation threshold: >50%
- SID tie margin inside Criterion 1: 10%
- DAGMA in-library threshold (passed to `DagmaLinear.fit`): **0.0**, so the continuous matrix is preserved
- DAGMA project-level threshold (applied externally to `abs(W_continuous)`): **0.3**
- DCDI threshold (applied to `model.get_w_adj()`): **0.5**
- DAGMA MMD-sampling noise policy: **residual-fitted per-node noise** is the primary policy; **unit-variance noise** is a sensitivity check
- DCDI MMD-sampling API: `forward_given_params` plus `get_distribution`, with the minimal call pattern (mask=None, regime=None) verified by probes C-P5 through C-P7
- DCDI data normalisation: `normalize=False` in `DataManagerFile`; project preprocessing is upstream of the wrapper
- threshold robustness triples: DAGMA `{0.2, 0.3, 0.4}` on `abs(W_continuous)`, DCDI `{0.4, 0.5, 0.6}` on `model.get_w_adj()`
- reproduction-then-equal-budget calibration structure

### Notes

- The DAGMA project-level threshold of **0.3** is paper-grounded from the supplementary material. The in-library `w_threshold=0.0` is a wrapper-internal convention so the continuous matrix is preserved for threshold robustness reporting and MMD sampling.
- The DCDI threshold of **0.5** is a selection study choice based on the natural midpoint of edge-existence probability.
- These constants may be changed later only by explicit revision in the relevant subsequent document.

---

## 10. What this document does not commit to

The following are deferred to Document 03 and the main-study protocol:

- main-study seed counts
- main-study corruption grid
- full main-study intervention grid
- full MMD bandwidth schedule for final reporting
- multiple-comparisons families for the main study
- nonlinear ablation design
- final hyperparameter grids for the chosen model
- main-study thresholding choices beyond the selection study

This document is intentionally narrow. Its only job is to select the base model and freeze the minimum tactical constants needed to do that defensibly.

---

## 11. Immediate next step after selection study

Once the selection study winner is declared, the next required document is:

**03_main_study_execution_protocol.md**

Its role is to freeze the main-study tactical constants for the chosen base model without reopening the structural commitments already fixed in Document 01.
