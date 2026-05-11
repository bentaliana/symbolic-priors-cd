# Phase 1 Readout — Evaluator Foundation Complete

**Date:** 2026-05-06  
**Author:** Ben Taliana  
**Purpose:** This document orients any new collaborator (human or AI) to the project state at the end of Phase 1 and the start of Phase 2. It is a high-level handoff document, not a substitute for the frozen planning documents or the decision log.

---

## 1. Project identity

This thesis investigates whether **confidence-weighted soft symbolic priors** can improve **generalisation to unseen interventions** in differentiable causal discovery when training uses **observational data only** and prior knowledge is systematically corrupted. The intended contribution is not the invention of symbolic priors themselves, but a controlled empirical study of whether uncertain, semantically meaningful priors improve interventional behaviour more robustly than prior-free, generic-regularised, or hard-constrained alternatives.

The current methodology and motivation are consistent with the progress report’s framing: causal discovery models may look good structurally or observationally while still failing under intervention, so the evaluation must centre interventional behaviour rather than structural fit alone.

---

## 2. Canonical documents (read in this order)

If any document conflicts with this readout, the documents below take priority in this order:

1. `docs/01_research_question_and_commitments.md`  
   Frozen scientific contract.

2. `docs/02_base_model_selection.md`  
   Frozen base-model selection protocol.

3. `docs/decision_log.md`  
   Running implementation and tactical decisions.

This readout is an orientation layer only.

---

## 3. What Phase 1 completed

Phase 1 completed the **evaluator foundation**.

### 3.1 Data layer complete

Implemented and tested:

- immutable `LinearGaussianSCM` representation with provenance
- ER DAG generation under the project’s `expected_edges` convention
- linear-Gaussian observational sampling
- hard intervention sampling via `intervene(...)` and `InterventionalSampler`
- constructor and sampling invariants tested thoroughly

### 3.2 Metrics layer complete except verified SID

Implemented and tested:

- `shd(predicted, true, reversal_cost=2)`
- `mmd_rbf_unbiased(x, y, bandwidth=None)`
- `mmd_sensitivity_sweep(x, y, bandwidth_multipliers=(0.5, 1.0, 2.0))`
- `sid_score(predicted, true)` **stub only**, intentionally deferred pending verified implementation

### 3.3 Sanity/gate layer complete

Implemented and tested:

- `check_sid_self_zero`
- `check_mmd_same_intervention`
- `check_mmd_same_observational`
- `check_do_clamping`
- `run_ground_truth_compatibility_checks`
- `assert_ground_truth_compatibility`

The evaluator can now produce a structured compatibility report and enforce a gate before model comparison begins.

### 3.4 Full regression status

As of the end of Phase 1:

- the full project test suite passes
- one SID scaffold test remains intentionally skipped because verified SID is not yet integrated

---

## 4. What is still explicitly deferred

The following is **not** complete yet:

### 4.1 Real SID computation

`sid_score(...)` is currently a stub that raises `NotImplementedError`.  
This is intentional. SID must be integrated only after an external implementation is explicitly verified against a hand-computed small case.

### 4.2 Base-model wrappers

No DAGMA or DCDI wrapper has been implemented yet.

### 4.3 Selection-study execution

No candidate-model comparison has been run yet.

### 4.4 Main-study symbolic-prior implementation

The actual thesis contribution layer — injecting soft symbolic priors into the selected base model and evaluating corruption robustness — has not started yet.

---

## 5. Immediate next step (Phase 2)

The immediate next step is:

**design the DAGMA and DCDI wrapper interfaces for the base-model selection study.**

That design work must be done with the current evaluator as the fixed downstream consumer.

Each wrapper should eventually:

- accept observational data in the project’s internal format
- run the model in the intended setting
- return a thresholded boolean adjacency matrix compatible with the evaluator
- isolate model-specific quirks inside the wrapper boundary

Before final selection-study execution, verified SID integration must also be completed, because SID is a load-bearing interventional criterion in the selection protocol.

---

## 6. Near-term roadmap

### Phase 2 — Base-model selection infrastructure

- read DAGMA paper and code/library carefully
- read DCDI paper and code/library carefully
- design wrapper interface
- implement DAGMA wrapper
- implement DCDI wrapper
- implement selection-study runner

### Phase 3 — Verified SID integration

- identify candidate implementation
- verify against hand-computed small example
- unskip the SID scaffold test only after verification

### Phase 4 — Base-model selection execution

- run the selection study per `docs/02_base_model_selection.md`
- generate results and declare the winning base model

### Phase 5 — Main-study protocol freeze

- write Document 03 only after the selected base model and tactical constants are known

### Phase 6 — Soft-prior implementation and corruption study

- implement confidence-weighted soft priors in the chosen base model
- compare against prior-free, L1, hard-constraint, and random-prior baselines
- evaluate degradation under corrupted priors

---

## 7. Non-negotiable current conventions

These are load-bearing current conventions. Do not silently change them.

- Metric argument order is `(predicted, true)`
- `sid_score` lives in `metrics/interventional.py`
- shared adjacency validation lives in `metrics/_graph_validation.py`
- SHD default reversal cost is `2`
- MMD uses the unbiased RBF estimator
- MMD median heuristic is computed on concatenated samples
- MMD primitive does not clip negative values
- evaluator gate defaults:
  - `mmd_tolerance = 0.01`
  - `clamp_tolerance = 1e-12`
  - `require_sid = False`
- failed SID is always a hard error
- `require_sid` only governs the deferred-SID case

See `docs/decision_log.md` for detailed rationale.

---

## 8. Known risks entering Phase 2

### 8.1 Wrapper fairness risk

DAGMA and DCDI expose different APIs and assumptions. Wrapper design must avoid introducing unfairness through inconsistent preprocessing, thresholding, or output adaptation.

### 8.2 Verified SID dependency

Selection-study results should not be treated as scientifically complete until SID is verified and unstubbed.

### 8.3 Thresholding risk

Thresholding from model output to boolean adjacency is a load-bearing choice and must remain consistent with the frozen protocol.

### 8.4 Phase 4 hook requirement

The selected wrapper architecture must leave room for the later soft-prior integration step. Do not design wrappers in a way that makes loss-level modification awkward or impossible.

---

## 9. Working mode reminder

This project uses:

- frozen planning documents for strategic commitments
- a decision log for tactical decisions
- plan-then-implement discipline for each commit
- atomic commits
- adversarial review across multiple AI collaborators before accepting important design changes

This readout is the orientation point at the end of evaluator construction and the start of wrapper design.

## Current repo state

As of the end of Phase 1:

- full project test suite passes
- current status: **129 passed, 1 skipped**
- the single skipped test is the intentionally deferred SID scaffold, which remains inactive until SID is verified against a hand-computed reference case
- the evaluator foundation is therefore complete enough to support the transition into wrapper design, while verified SID remains an explicit near-term dependency
