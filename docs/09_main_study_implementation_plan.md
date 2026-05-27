# 09 Main Study Implementation Plan

> **ADDENDUM (added 27/05/2026, post-implementation drift audit).** This document is the **frozen pre-implementation protocol** as it stood at version 1.0. Its body below is preserved as a historical record. The status updates and corrections listed here reflect what was actually implemented; they are formally recorded in docs/03 ("27/05/2026 — Drift audit and retrospective closure of docs/01 amendments"). Read this addendum together with the body.
>
> **Implementation status (M-1 through M-11 are all IMPLEMENTED):**
>
> - M-1 Soft-prior injection: IMPLEMENTED (`src/symbolic_priors_cd/wrappers/_soft_prior_dagma.py`).
> - M-2 Lambda_prior calibration: IMPLEMENTED. `lambda_prior = 2e-4`, frozen 24/05/2026 in docs/03. The Section 2 placeholder `lambda_prior = TBD` and the Section 7 candidate grid `(0.01, 0.05, 0.1, 0.5)` are superseded; see docs/09a for the supporting readout and docs/03 24/05/2026 for the frozen value and the rationale for the lower-grid revision `(2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)`.
> - M-3 Prior generation and corruption: IMPLEMENTED (`experiments/main_study/priors.py`).
> - M-4 Hard-exclusion plumbing: IMPLEMENTED. See docs/03 24/05/2026 for verification and interpretation note (hard exclusion may redistribute edge weights at non-excluded positions; it is a distinct constrained-optimisation regime, not "soft prior at infinite confidence").
> - M-5 Main-study schema and config factory: IMPLEMENTED. `schema_version = 2`.
> - M-6 Main-study runner: IMPLEMENTED. Parent reference `parent_heldout_run_hash_full` embeds the full 64-char hash of `88da382e8672...`.
> - M-7 Matched-L1 calibration: IMPLEMENTED. `matched_l1_lambda1 = 0.0625`, frozen in docs/03 from match-by-sparsity on main-calibration seeds 401, 402; target mean edge count 13.0; calibration hash `274cfe3fef32`. See docs/10 for the calibration protocol and `results/main_study/calibration/matched_l1/274cfe3fef32/` for the artefacts.
> - M-8 Main evaluation runs: IMPLEMENTED. Run hash `864fe6722256`, 224 records under `results/main_study/864fe6722256/records/` (7 prior_free + 7 matched_l1 + 35 hard_exclusion + 175 soft_frobenius).
> - M-9 Readout and figures: IMPLEMENTED. Outputs under `results/main_study/main_evaluation/864fe6722256/readout/` (8 figures plus `statistics_summary.json`, `paired_seed_comparisons.csv`, `baseline_comparison.csv`, `degradation_summary.csv`, `metric_correlations.csv`, `reference_forbidden_edge_comparison.csv`); notebook at `notebooks/main_evaluation.ipynb`.
>
> **Additions not in the original v1.0 plan (recorded retrospectively):**
>
> - **M-10 Prior structural relevance diagnostic.** Pre-registered in docs/11, completed under analysis hash `1b46785b59a4`, results recorded in docs/12. Offline structural diagnostic using saved adjacencies only; no model refitting, no MMD recomputation.
> - **M-11 Oracle prior relevance diagnostic.** Completed under analysis hash `1b95c563db88`, results recorded in docs/13. Five scenarios over the seven evaluation seeds: actual reference forbidden removal, exact budget-matched FP removal (k=10), full FP removal, greedy acyclicity-guarded FN addition (k=10), greedy full-candidate FN addition. Pre-declared as the final scheduled exploratory diagnostic.
> - **Presentation-only readout** for M-10 + M-11, under `results/main_study/exploratory/prior_relevance_diagnostics/`, with `notebooks/prior_relevance_diagnostics.ipynb`.
>
> **docs/01 amendments missed at Section 15 Step 0** (which said they should be recorded in docs/03 "before M-1 begins" but were not):
>
> All seven of the docs/01 amendments listed in Section 14 of this document, plus the H4 retirement, are now formally recorded in docs/03 under the 27/05/2026 drift-audit entry. The implementation proceeded without the procedurally-required record; this is a procedural-discipline finding only and does not affect any saved result.
>
> **The body below is the original v1.0 plan and is not edited by this addendum (except to substitute the frozen calibration values in Section 2 for clarity; the original `TBD` markers are noted in this addendum).**

## Status

Protocol document for the prior-loss main study.
Version 1.0.
Frozen before any main-study implementation begins.
Decisions herein are binding. Changes require an explicit version bump and a docs/03 entry.

---

## Change Log

- v1.0: Initial freeze. All decisions derived from main-study scoping conversation (May 2026),
  literature decision document (prior_loss_literature_review_and_decisions.md),
  and docs/01 v1.1 scientific contract.

---

## 1. Purpose and Scope

This document is the operative protocol for the prior-loss main study. It inherits all
non-negotiable commitments from docs/01 v1.1 and records the tactical decisions needed
to implement the main study after base-model selection is complete.

The base model is DAGMA-linear, selected by the held-out adjudication (docs/08h,
held-out hash 88da382e8672). That decision is closed.

### 1.1 In Scope

- Soft-prior DAGMA: targeted Frobenius penalty on forbidden edges with confidence weighting.
- Hard-exclusion DAGMA: DAGMA-native projected-Adam exclusion via exclude_edges.
- Matched-L1 DAGMA: global L1 regularisation tuned by match-by-sparsity rule.
- Prior-free DAGMA: DAGMA without any prior modification; baseline reference.
- Lambda_prior calibration on main-calibration seeds.
- Matched-L1 sparsity selection on main-calibration seeds.
- Main evaluation on main-evaluation seeds across the full confidence and corruption grids.
- Main-study artefact schema v2 (schema_version = 2).
- Main-study readout: static figures, per-seed traces, degradation curves.

### 1.2 Explicitly Out of Scope (Deferred)

- DCDI with priors (no DCDI in main study).
- Oracle-prior upper bound (methodological ambiguity; dropped).
- Ordering priors (deferred; only forbidden-edge priors in main study).
- 10-node ER4 and 20-node ER2 ablation cells (deferred; docs/01 amendment required).
- H4 formal instability hypothesis (retired as formal test; exploratory only if time remains).
- Learned confidence weights (outside fixed-confidence protocol boundary; see docs/01 v1.1).
- ATE-error training objective.
- Real-data evaluation as primary evidence.
- Random-prior sensitivity addendum: conditional on core pipeline completion.
- Global-Frobenius sensitivity addendum: conditional on core pipeline completion.

The sensitivity addenda, if implemented, are labelled exploratory and do not modify headline
results.

---

## 2. Frozen Constants

### 2.1 Inherited from Selection Study (Do Not Reopen)

These constants are carried forward from the selection study without change.

```
n_nodes              = 10
graph_type           = ER2
expected_edges       = 20
n_train              = 1000
mmd_n_samples        = 1000
condition            = standardised
dagma_lambda1        = 0.1           # calibration-selected for standardised condition
                                     # (hash 7b345b1b2e85)
dagma_threshold_primary   = 0.3
dagma_threshold_triple    = (0.2, 0.3, 0.4)
dagma_warm_iter      = 20000
dagma_max_iter       = 70000
dagma_lr             = 3e-4
dagma_betas          = (0.99, 0.999)
```

### 2.2 New Main-Study Constants (Frozen Here)

```
# Seed pools (disjoint from all selection-study pools)
# selection-study pools for reference: reproduction=(101,102,103),
# calibration=(201,202), held_out_evaluation=(301,302,303,304,305)
main_calibration_seeds    = (401, 402)
main_evaluation_seeds     = (501, 502, 503, 504, 505, 506, 507)

# Prior specification
prior_family              = "forbidden_edge"
prior_k                   = 10         # number of forbidden edges per SCM seed
PRIOR_SEED_BASE           = 9000       # prior_selection_seed = PRIOR_SEED_BASE + seed_value
CORRUPTION_SEED_BASE      = 9100       # corruption_seed = CORRUPTION_SEED_BASE
                                       #   + seed_value + corruption_fraction_index

# Confidence grid (includes 0.0 as internal consistency check for prior-free equivalence)
confidence_grid           = (0.0, 0.25, 0.5, 0.75, 1.0)

# Corruption grid (from docs/01 v1.1 Section 11; Elicit recommendation of 10/30/50 not adopted)
corruption_grid           = (0.0, 0.20, 0.40, 0.60, 0.80)

# Prior loss
# lambda_prior: FROZEN 24/05/2026 in docs/03 from the M-2 lower-grid calibration.
# Original Section 7 grid (0.01, 0.05, 0.1, 0.5) was rejected (all-too-strong); the
# lower-grid (2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3) selected 2e-4 as the smallest passing value.
lambda_prior              = 2e-4       # frozen; see docs/03 24/05/2026 and docs/09a

# Matched-L1
# matched_l1_lambda1: FROZEN in docs/03 from the M-7 match-by-sparsity calibration on
# main-calibration seeds 401, 402. Target mean edge count 13.0 (per-seed [14, 12] from
# soft_frobenius clean/conf=1.0); selected value produces mean edge count 12.5
# (absolute_gap = 0.5). Calibration hash: 274cfe3fef32; see docs/10.
matched_l1_lambda1        = 0.0625     # frozen; see docs/03 and results/main_study/calibration/matched_l1/274cfe3fef32/
```

---

## 3. Prior Specification

### 3.1 Prior Family

Forbidden-edge priors only. Ordering priors are deferred. This is a documented amendment
to the two-prior-family scope in docs/01 v1.1 Section 18; the rationale is recorded in the
prior-loss literature decision document (prior_loss_literature_review_and_decisions.md).

### 3.2 Prior Source

Ground-truth-derived true-negative edges. For each SCM seed, the true DAG is known.
The forbidden-edge prior set F_seed is sampled from the set of true-negative directed edges
in that SCM without replacement. These are edges that are genuinely absent from the true DAG;
the prior correctly identifies them as forbidden in the uncorrupted condition.

### 3.3 Prior Generation Procedure (per seed)

```
1. Generate the ER2 SCM for seed s (same procedure as selection study).
2. Identify all true-negative directed edges: edges (i,j) not in the true DAG.
3. Identify all true-positive directed edges: edges (i,j) that ARE in the true DAG.
4. Set prior_selection_seed_s = PRIOR_SEED_BASE + s.
5. Sample k=10 edges WITHOUT replacement from the true-negative set using
   prior_selection_seed_s. These form the clean forbidden-edge prior set F_clean_s.
6. For each corruption level c in corruption_grid (including c=0.0):
   a. Let corruption_fraction_index = index of c in corruption_grid (0,1,2,3,4).
   b. Set corruption_seed_s_c = CORRUPTION_SEED_BASE + s + corruption_fraction_index.
   c. Compute n_corrupt = round(c * k) edges to corrupt.
   d. Using corruption_seed_s_c, sample n_corrupt edges WITHOUT replacement from
      F_clean_s to designate as the edges to be replaced (the replaceable edges).
   e. Using corruption_seed_s_c + k, sample n_corrupt edges WITHOUT replacement from
      the true-positive edge set to serve as replacements.
   f. Construct F_corrupted_s_c by replacing the designated edges in F_clean_s with
      the true-positive replacements. Non-designated edges in F_clean_s are retained.
   g. At c=0.0: n_corrupt=0, so F_corrupted_s_0 == F_clean_s exactly.
7. Record and save: F_clean_s, F_corrupted_s_c (for each c),
   prior_selection_seed_s, corruption_seed_s_c (for each c, including c=0.0),
   corrupted_edge_count (= n_corrupt for each c).
```

corruption_seed is populated for ALL corruption levels including 0.0 for uniform
reproducibility. At c=0.0, n_corrupt=0, so the seed has no practical effect, but it
is recorded for auditability. The null value for corruption_seed in the schema applies
only to method families that do not use a prior (prior_free, matched_l1).

### 3.4 Confidence Mask

For a given forbidden-edge prior set F and confidence level conf, the confidence mask C is:

```
C[i,j] = conf   if (i,j) in F
C[i,j] = 0.0   otherwise
```

At conf = 0.0, C is the zero matrix and the model is mathematically identical to prior-free DAGMA.
This is used as an internal consistency check (see Section 6.1).

---

## 4. Prior-Loss Formulation

### 4.1 Penalty Term

The soft-prior penalty is:

```
L_prior = lambda_prior * sum_{(i,j) in F} c_ij * W_ij^2
```

where W is the continuous weighted adjacency matrix learned by DAGMA, F is the
forbidden-edge prior set, and c_ij in {0.0, 0.25, 0.5, 0.75, 1.0} is the per-edge
confidence weight (uniform across all edges in F for a given confidence level).

### 4.2 Gradient

The gradient added to DAGMA's hand-coded Gobj at each optimisation step is:

```
G_prior = 2 * lambda_prior * (C o W)
```

where C is the confidence mask and o denotes elementwise multiplication.
G_prior is added to Gobj BEFORE the Adam update. It is NOT multiplied by DAGMA's
augmented Lagrangian parameter mu. This ensures confidence strength remains constant
and interpretable across all optimisation stages (warm and final phases).

### 4.3 Implementation Requirement

The penalty is injected by subclassing DagmaLinear and overriding minimize() to
add G_prior to the gradient. This is a training-time modification. It is not
a post-hoc graph adjustment. The wrapper's public API (fit, native_edge_continuous,
thresholded_adjacency, sample_interventional, get_diagnostics) is unchanged.

The prior gradient acts on the same continuous W that DAGMA's score gradient acts on.
Both use the same single-threshold policy (w_threshold_internal = 0.0, external
abs(W) >= 0.3 thresholding).

### 4.4 Gaussian MAP Interpretation

The formulation corresponds to MAP estimation under a zero-centred Gaussian prior
on forbidden edge weights: W_ij ~ N(0, 1/(2*lambda_prior*c_ij)) for (i,j) in F.
High confidence corresponds to high prior precision around zero. Corrupted priors
impose a misspecified Gaussian that the observational score can partially overcome.
This interpretation is valid within DAGMA's continuous-weight setting.

---

## 5. Baseline Suite

### 5.1 Core Baselines (Required)

| Method | Varies by confidence? | Varies by corruption? | Fits (7 seeds) |
|---|---|---|---|
| Soft-prior DAGMA | yes | yes | 7 x 5 x 5 = 175 |
| Hard-exclusion DAGMA | no | yes | 7 x 5 = 35 |
| Prior-free DAGMA | no | no | 7 |
| Matched-L1 DAGMA | no | no | 7 |
| **Total core** | | | **224** |

At approximately 1.5 seconds per fit, total core runtime is approximately 6 minutes.

### 5.2 Soft-Prior at Confidence 0.0 (Internal Consistency Check)

Confidence = 0.0 is included in the soft-prior sweep. At this setting G_prior = 0
everywhere; the model should produce results indistinguishable from prior-free DAGMA
on the same seed and SCM. This check is the primary smoke gate for the injection
mechanism in production runs. If soft-prior at confidence 0.0 deviates materially
from prior-free DAGMA on the same seed, the run is flagged as an infrastructure failure
and the deviation is logged.

The 35 additional fits (7 seeds x 1 confidence level x 5 corruption levels) at
confidence = 0.0 cost approximately 52 seconds. They are included in the 175 soft-prior
fits above.

### 5.3 Hard-Exclusion Baseline

Uses DAGMA's native exclude_edges mechanism. The same forbidden-edge prior set F
is applied as a training-time projected-Adam exclusion: forbidden entries are
re-zeroed via W *= mask_exc after every Adam step in both warm and final phases.

exclude_edges is passed as a tuple of tuples: ((i1,j1), (i2,j2), ...).
The wrapper validates this format before calling DagmaLinear.fit.
include_edges is not used.
The mechanism is verified by the M-4 regression test before any production runs.

Hard-exclusion does not vary over confidence (there is no confidence concept for a
hard constraint). It varies over corruption only: at each corruption level, the
forbidden-edge set F_corrupted_s_c is applied as the exclusion set.

### 5.4 Matched-L1 Baseline

Uses standard DAGMA with an augmented lambda1 chosen by the match-by-sparsity rule
(docs/01 v1.1 Section 9). The target sparsity is the mean thresholded edge count
of soft-prior DAGMA at corruption 0% and confidence 1.0 on main-calibration seeds
(401, 402). The lambda1 value that most closely achieves this mean edge count
is selected from the augmented grid (existing grid plus intermediate points if
needed). This value is frozen in docs/03 after M-7 before any evaluation-seed runs.

### 5.5 Sensitivity Addenda (Conditional)

Implemented only after core pipeline is green and time remains.
Priority order: global-Frobenius first, random-prior second.

Global-Frobenius: lambda_F * sum_{i != j} W_ij^2, tuned to match soft-prior density.
Answers whether gains require semantic targeting vs. L2 shrinkage in general.

Random-prior: same Frobenius penalty on randomly selected edge directions (same count
and confidence schedule). Answers whether gains require semantic alignment.

Both are labelled exploratory in the artefact and thesis.

---

## 6. Seed Pools

```
main_calibration   = (401, 402)
main_evaluation    = (501, 502, 503, 504, 505, 506, 507)
```

Calibration seeds are used exclusively for:
- lambda_prior smoke calibration (M-2)
- Matched-L1 sparsity matching (M-7)

Headline results use only the seven main-evaluation seeds.
No calibration seed may appear in headline evaluation records.
No additional evaluation seeds may be added after inspecting headline results.
Any later seed expansion is labelled exploratory.

All seed integers are disjoint from all selection-study pools:
reproduction=(101,102,103), calibration=(201,202), held_out_evaluation=(301,302,303,304,305).

---

## 7. Lambda_Prior Calibration Procedure

Lambda_prior is NOT frozen at protocol time. It is set by M-2 calibration and frozen
in docs/03 before any evaluation-seed runs begin.

### 7.1 Smoke Test (M-2, calibration seeds)

Construct a calibration SCM from seed 401. Run prior-free DAGMA and identify a
true-positive edge (an edge that IS in the true DAG) with |W_ij| >= 0.3 in the
prior-free continuous W (i.e., one that DAGMA recovers with meaningful magnitude).
Select the strongest such edge if multiple qualify. If no true-positive edge on
seed 401 has |W_ij| >= 0.3, use seed 402 or the strongest recovered true-positive
edge available and record the fallback in the docs/03 entry. Using the minimum-
magnitude edge would produce a trivial pass.

Designate this edge as forbidden in the prior (a deliberately corrupted prior).
Run soft-prior DAGMA at confidence = 1.0 and lambda_prior = candidate value.

**Non-degenerate shrinkage criterion:**
The penalised true-positive edge's continuous W_ij must be at least 50% lower than
the same edge's prior-free continuous W_ij on the same seed, SCM, and data.
The penalised entry must not be below 1e-6 (which would indicate hard-clamping,
not soft suppression).

If this criterion passes, the lambda_prior candidate is accepted.
If the entry is near-zero (below 0.01), lambda_prior is too strong; try a smaller value.
If the reduction is less than 50%, lambda_prior is too weak; try a larger value.

### 7.2 Pilot Values

Try lambda_prior candidates from: {0.01, 0.05, 0.1, 0.5}.
Freeze the first value that satisfies the non-degenerate shrinkage criterion.
Record the frozen value and pilot evidence in docs/03 before M-8 begins.

### 7.3 Confidence-0 Consistency Check (Infrastructure Gate)

At confidence = 0.0, G_prior = 0 everywhere. The soft-prior path must produce numerically
identical continuous W to the prior-free path on the same SCM, data, hyperparameters, and
seed. DAGMA is verified deterministic by construction (selection-study diagnostics confirmed
zero variance across repeated fits with identical input). The check is:

```
max(|W_soft_confidence_0 - W_prior_free|) < 1e-10
```

Any deviation above this threshold is an infrastructure failure, not a warning. Identical
continuous W implies identical thresholded adjacency, SHD, and SID. A 5% SID tolerance
is not acceptable: SID may be 0 in both cases, making percentage tolerance ill-defined,
and the check is an infrastructure equivalence gate, not a scientific performance comparison.

---

## 8. Evaluation Hierarchy and Positive-Result Criteria

### 8.1 Metric Hierarchy (from docs/01 v1.1 Section 12)

- **Primary:** SID (interventional; gadjid backend, raw mistake count, predicted-then-true order)
- **Secondary:** MMD (unbiased RBF, median-heuristic bandwidth; reported with kernel-sensitivity)
- **Diagnostic:** SHD (reversal cost 2; structural proxy only)

### 8.2 Positive-Result Criteria (from docs/01 v1.1 Section 13)

A full positive result requires all three:
1. Soft-prior DAGMA outperforms prior-free DAGMA on mean SID at low-to-moderate corruption
   (0% and 20% corruption levels, at one or more confidence levels).
2. Soft-prior DAGMA outperforms matched-L1 DAGMA on mean SID in the same setting.
3. Soft-prior DAGMA shows lower degradation summary than hard-exclusion DAGMA across
   the corruption grid.

A partial positive result: a proper subset of the three conditions holds.
A negative result: none hold.

All outcomes are thesis-valid. The negative-result framing is specified in docs/01 v1.1
Section 14 and remains in force.

### 8.3 Degradation Summary Statistic

Area under the mean-SID-vs-corruption curve, computed by trapezoidal integration
across corruption_grid = (0.0, 0.2, 0.4, 0.6, 0.8). Lower area is better.
The degradation summary is computed per (method, confidence_level) cell and stored
in the main-study readout artefact. This is the pre-registered statistic for
positive-result criterion 3.

---

## 9. Main-Study Artefact Schema (v2)

### 9.1 Schema Version

schema_version = 2. Incompatible with selection-study schema_version = 1.
The main-study loader must check schema_version before reading any field.

### 9.2 Mandatory Fields per Run Record (in addition to selection-study fields)

```
method_family          : str          # "prior_free", "soft_frobenius", "matched_l1",
                                      #   "hard_exclusion", "global_frobenius", "random_prior"
prior_family           : str          # "forbidden_edge" or "none"
prior_k                : int          # 10 (or 0 for prior_free / matched_l1)
confidence             : float|null   # confidence level for soft_frobenius;
                                      # null for hard_exclusion, prior_free, matched_l1
                                      # (hard_exclusion is not a confidence-scaled method)
corruption_fraction    : float|null   # 0.0, 0.2, 0.4, 0.6, 0.8 for soft_frobenius
                                      # and hard_exclusion; null for prior_free and
                                      # matched_l1 run records (derived readout rows
                                      # may expand these for plotting but are not fits)
lambda_prior           : float|null   # frozen value for soft_frobenius;
                                      # null for all other method families
effective_lambda1      : float        # actual lambda1 used in this run (all methods)
prior_selection_seed   : int|null     # PRIOR_SEED_BASE + seed_value;
                                      # null for prior_free and matched_l1
corruption_seed        : int|null     # CORRUPTION_SEED_BASE + seed_value + corruption_idx;
                                      # populated for all corruption levels including 0.0
                                      # (see Section 3.3); null for prior_free and matched_l1
forbidden_edge_count   : int          # |F|; 10 for soft/hard; 0 for prior_free/matched_l1
corrupted_edge_count   : int          # number of corrupted edges in F_corrupted;
                                      # always 0 at corruption_fraction = 0.0
seed_population        : str          # "main_calibration" or "main_evaluation"
parent_heldout_run_hash_full : str    # full 64-character SHA-256 hash of the held-out
                                      # artefact (resolved before M-6; see Section 13)
```

**Method-family field policy:**
- soft_frobenius: confidence populated, corruption_fraction populated, lambda_prior populated.
- hard_exclusion: confidence null, corruption_fraction populated, lambda_prior null.
- prior_free: confidence null, corruption_fraction null, lambda_prior null.
- matched_l1: confidence null, corruption_fraction null, lambda_prior null.

Readout and plotting code creates derived comparison rows that expand prior_free and
matched_l1 values across corruption and confidence cells for figure generation. Those
derived rows are NOT stored as independent run records and must be clearly labelled
as derived references in the summary artefact.

### 9.3 Artefact Files Saved per Run

Required for thesis figures and optional animations. Saved alongside run.json.

```
continuous_w.npz              # native_edge_continuous() output at full precision
thresholded_adjacency.npz     # bool adjacency matrix at threshold 0.3
prior_edge_set_clean.json     # F_clean_s: the k=10 true-negative forbidden edges
prior_edge_set_corrupted.json # F_corrupted_s_c: the corrupted forbidden edges at this level
confidence_mask.npz           # C matrix (n x n float)
per_edge_labels.json          # for each edge in F: "true_negative_retained",
                               # "true_positive_corrupted_replacement"
interventions_mmd.json        # per-intervention MMD records (not only aggregate)
```

### 9.4 Derived Summary Fields (Computed by Readout, Not Runner)

These are NOT stored per-run. They are computed by the readout notebook/script from
the per-run records and stored in the summary artefact.

```
mean_sid, median_sid, std_sid, iq_range_sid
mean_mmd, median_mmd, std_mmd, iq_range_mmd
mean_shd, median_shd
pairwise_delta_soft_vs_priorfree_{sid,mmd,shd}
pairwise_delta_soft_vs_matchedl1_{sid,mmd,shd}
pairwise_delta_soft_vs_hardexclusion_{sid,mmd,shd}
degradation_summary_auc    # area under mean-SID-vs-corruption curve
first_corruption_advantage_lost  # first corruption level where soft-prior loses to matched-L1
per_seed_traces            # per-seed SID/MMD/SHD at every cell
bootstrap_ci_95_{sid,mmd}  # bootstrap 95% CI for mean differences
```

---

## 10. Results Path Structure

```
results/
  main_study/
    <main_study_run_hash12>/
      records/
        <method>_seed<s>_conf<c>_corr<r>.json
      summary/
        main_study_summary.json
        degradation_curves.csv
        per_seed_traces.csv
      readout/
        main_study_readout.md
      main_study.log
```

Selection-study results in results/model_selection/ are not touched.

---

## 11. Failure-Mode Taxonomy

### 11.1 Model-Fit Failure (catch, build degenerate record, continue)

- DAGMA fails to converge (h_final > convergence threshold)
- Thresholded adjacency is not a valid DAG (cyclic, bidirected, self-loop)
- Sampler unavailable (invalid graph, unresolved noise policy)
- MMD unavailable (sampler status not available)
- Soft-prior at confidence 0.0 deviates materially from prior-free (logged as warning,
  not failure; the run record is preserved and the deviation flagged in diagnostics)

### 11.2 Infrastructure Failure (raise, abort run with _MainStudyInfrastructureError)

- Malformed prior edge set (not a tuple of tuples, contains self-loops, contains
  edges outside [0, n_nodes) x [0, n_nodes))
- lambda_prior or matched_l1_lambda1 is TBD (not yet frozen by calibration)
- Configuration hash mismatch between run record and directory path
- Per-run file write failure
- exclude_edges validation failure (type not tuple-of-tuples, malformed pairs)
- Any condition that would make the artefact structurally untrustworthy

---

## 12. Implementation Commit Plan

Commit labels use M-1 through M-9 prefix. Each commit is atomic (green tests
before and after). No main-study code is written before this document is committed.

### M-1: Soft-Prior Injection Mechanism

**New:** Subclass DagmaLinear as SoftPriorDagmaLinear. Override minimize() to add
G_prior = 2 * lambda_prior * (C o W) to Gobj, before the Adam update.
Not scaled by mu. Confidence mask C passed as constructor argument.

**Tests required:**
- lambda_prior = 0.0 produces bitwise-identical output to prior-free DagmaLinear
  on the same input and hyperparameters.
- Penalised forbidden entry continuous W_ij is strictly lower than the same entry
  under prior-free DagmaLinear on the same seed and data.
- Unpenalised entries are not globally suppressed (spot-check a non-forbidden entry).
- native_edge_continuous() output is preserved and logged correctly.
- Graph-status and sampler-status logic is unaffected.

**Must not:**
- Modify any existing DAGMAWrapper, _dagma_fit.py, or selection-study code.
- Use include_edges.
- Multiply G_prior by mu.

**Readiness gate:** All existing 1229+ tests pass after M-1.

---

### M-2: Lambda_Prior Smoke Calibration

**New:** Calibration runner on seeds 401-402. Runs three sub-steps:
(a) Smoke test on deliberately corrupted true-positive edge (per Section 7.1).
(b) Consistency check: soft-prior at confidence 0.0 matches prior-free within 5% on SID.
(c) Lambda_prior pilot across {0.01, 0.05, 0.1, 0.5}; selects first candidate passing
    the non-degenerate shrinkage criterion.

**Output:** lambda_prior value, recorded in docs/03 entry before M-7 begins.

**Must not:** Use any evaluation-seed (501-507) data.

**Readiness gate:** lambda_prior is frozen in docs/03 before M-7 (calibration runs) and
before any M-8 (evaluation-seed runs) begin. It does NOT block M-3 prior-generation code
or M-4 hard-exclusion plumbing, which are independent of lambda_prior and may proceed
in parallel with M-2.

---

### M-3: Prior Generation and Corruption Mechanism

**New:** Module experiments/main_study/priors.py (or equivalent).
Implements per-seed prior generation per Section 3.3.
Saves F_clean, F_corrupted (per corruption level), prior_selection_seed, corruption_seed.

**Tests required:**
- F_clean contains exactly k=10 edges, all true-negative for the given SCM.
- F_corrupted at corruption 0.0 equals F_clean.
- F_corrupted at corruption 0.8 contains exactly round(0.8*10)=8 true-positive replacements.
- Prior generation is deterministic: same seed produces same F_clean and F_corrupted.
- PRIOR_SEED_BASE and CORRUPTION_SEED_BASE constants are present and not equal to any
  operational seed pool value.

**Independent of:** M-1, M-2. Can be developed in parallel.

---

### M-4: Hard-Exclusion Plumbing

**Modified:** _dagma_fit.py call site. Add exclude_edges field to DAGMAConfig:
Optional[Tuple[Tuple[int, int], ...]] = None.
Pass cfg.exclude_edges to DagmaLinear.fit() (currently hardcoded None).

**New wrapper-level validation:** Before calling DagmaLinear.fit, assert exclude_edges
is None or a tuple of tuples of length-2 integer pairs. Raise _MainStudyInfrastructureError
if malformed. Do not rely on DAGMA's internal validation (which is broken: no raise
keyword in its ValueError branch).

**Regression test:**
- Fit DAGMA on a small SCM with one strong true-positive edge excluded.
- Assert native_edge_continuous()[i,j] < 1e-6 for the excluded edge (machine-zero,
  not merely below threshold).
- Assert native_edge_continuous() is unchanged for a non-excluded strong edge.
- Assert thresholded_adjacency does not contain the excluded edge.
- Malformed exclude_edges input raises _MainStudyInfrastructureError before calling DAGMA.

**Must not:** Use include_edges. Must not modify any selection-study code or tests.

**Independent of:** M-1, M-2, M-3.

---

### M-5: Main-Study Artefact Schema and Config Factory

**New:** experiments/main_study/schema.py (or equivalent).
Implements schema_version = 2 record structure per Section 9.
MainStudyConfig: lambda_prior, confidence_level, corruption_fraction, method_family,
prior_forbidden_edges (as tuple-of-tuples), exclude_edges (for hard-exclusion).
Configuration hash includes all prior and method fields.

**Tests required:**
- Schema validation rejects records with missing mandatory fields.
- Configuration hash is deterministic for identical configs.
- Schema_version = 2 record is rejected by the selection-study schema_version = 1 loader.

**Depends on:** M-3 (to know the prior fields that must be hashed).

---

### M-6: Main-Study Runner

**New:** experiments/main_study/run.py (or equivalent).
Templates from run_held_out_evaluation. Implements:
- Workload enumeration: (seed, method_family, confidence_level, corruption_fraction)
- Dependency injection: production / mocked fit runner
- Preflight gate (dry-run mode)
- Per-fit atomic record persistence
- _MainStudyInfrastructureError taxonomy
- Provenance chain: embeds held-out artefact hash 88da382e8672 as parent reference

**Must not:**
- Touch results/model_selection/ or any selection-study artefact.
- Run any evaluation-seed fits before lambda_prior and matched_l1_lambda1 are frozen.
- Contain notebook logic, plot generation, or hidden training logic.

**Depends on:** M-1, M-3, M-4, M-5.

---

### M-7: Calibration Runs and Matched-L1 Selection

**Execution only:** Run soft-prior DAGMA on main-calibration seeds (401, 402) at
corruption 0.0 and confidence 1.0. Compute mean thresholded edge count.
Select lambda1 closest to that count from the augmented grid. Freeze matched_l1_lambda1
in docs/03 entry before any evaluation-seed runs.

**Output:** matched_l1_lambda1, recorded in docs/03 before M-8.

**Must not:** Use any evaluation-seed data. Must not run prior-free or matched-L1
against evaluation seeds before this decision is frozen.

**Depends on:** M-2 (lambda_prior frozen), M-6 (runner operational).

---

### M-8: Main Evaluation Runs

**Execution only:** Run all four core methods on main-evaluation seeds (501-507)
across the full confidence and corruption grids per Section 5.1.
Total: 224 core fits at approximately 6 minutes runtime.

**Checklist before running:**
- [ ] lambda_prior is frozen in docs/03.
- [ ] matched_l1_lambda1 is frozen in docs/03.
- [ ] M-4 hard-exclusion regression test is green.
- [ ] M-2 consistency check (confidence 0.0 ~ prior-free) is passing on calibration seeds.
- [ ] No evaluation-seed data has been inspected.

**Must not:** Re-open any frozen decision after inspecting results. If a run fails
infrastructure validation, classify as _MainStudyInfrastructureError, do not patch,
do not re-run silently.

---

### M-9: Readout and Figures

**New:** Reading from saved main-study records only. No fit logic inside the readout.

**Required outputs:**
- Main study summary JSON (per-cell means, medians, IQR, per-seed traces).
- Degradation curves: mean SID vs. corruption level, one line per confidence level,
  shaded IQR, reference lines for matched-L1 and prior-free.
- Comparison table: soft-prior vs. prior-free, vs. matched-L1, vs. hard-exclusion at
  each corruption level, primary metric SID.
- Per-seed traces for all corruption levels (visible in supplement or appendix).
- Degradation summary AUC and first-corruption-advantage-lost values per method.

**Optional (animation-ready frame data):**
- Frame-indexed CSVs: one frame per corruption level, showing confidence grid vs. mean SID
  with IQR bands and baseline references. These allow a defence-presentation animation to
  be generated without rerunning any fit.

**Must not:** Rerun any fit. Must not add new result cells post-hoc. The readout reads
what is saved; it does not produce new data.

---

## 13. Provenance Chain

Every main-study run record embeds the full 64-character SHA-256 hash of the held-out
artefact as `parent_heldout_run_hash_full`.

**Resolution gate (before M-6):** Before implementing the main-study runner, resolve
the full 64-character held-out artefact hash by reading:
`results/model_selection/held_out/88da382e8672/heldout_evaluation.json`
and extracting the `run_hash_full` field (or equivalent field containing the full hash).
The 12-character prefix `88da382e8672` is not sufficient for `parent_heldout_run_hash_full`.
The full hash is frozen in docs/03 before M-6 implementation begins.

---

## 14. Explicit Deferrals (docs/01 Amendment Record)

The following elements are deferred from the main study. Each deferral is documented
here and in the corresponding docs/03 entry.

| Deferred item | Original commitment | Reason for deferral |
|---|---|---|
| Ordering priors | docs/01 Section 18 two-prior-family scope | Implementation complexity; separate contribution |
| 10-node ER4 ablation | docs/01 Section 10.2 | Timeline; requires calibration sub-study |
| 20-node ER2 ablation | docs/01 Section 10.2 | Timeline; requires calibration sub-study |
| Oracle-prior upper bound | Progress report Objective 4(d) | Methodological ambiguity in continuous optimisation |
| Random-prior control | docs/01 included baselines | Accidental-signal confound; demoted to sensitivity addendum |
| H4 instability hypothesis | docs/01 Section 4 | Cross-seed graph-variability measure never frozen in docs/02; retired as formal test |
| ATE-error objective | docs/01 Section 12.3 | Not stable in this setting |
| DCDI prior-loss | docs/08h | DCDI not carried forward |
| Learned confidence weights | docs/01 Section 18 | Outside fixed-confidence protocol boundary |

---

## 15. Immediate Next Steps

### Step 0 (Before any code): docs/03 amendment entry

Before M-1 begins, commit a docs/03 entry recording the main-study scoping decision.
This entry must formally record the following docs/01 amendments, which are established
in docs/09 Section 14 but not yet reflected in docs/03:

- Ordering priors deferred from two-prior-family scope.
- Random-prior control demoted from core baseline to conditional sensitivity addendum.
- H4 instability hypothesis retired as formal test.
- 10-node ER4 and 20-node ER2 ablation cells deferred.
- Oracle-prior upper bound dropped.
- L1-style default soft-prior objective (docs/06 Section 18 example) superseded by
  targeted Frobenius (L2) form, as motivated by the prior-loss literature decision document.
- DAGMA selected as sole main-study model (DCDI not carried forward).

The docs/01 change-control rule requires these amendments to be recorded before
implementation proceeds. docs/09 is not a substitute for docs/03 entries.

### Step 1: Commit docs/09 to the repository.

### Step 2: Begin M-1, M-3, M-4 in parallel.

M-1 (soft-prior injection), M-3 (prior generation), and M-4 (hard-exclusion plumbing)
are independent of each other. M-2 (lambda_prior calibration) depends on M-1 and must
precede M-7. M-5 (schema) depends on M-3. M-6 (runner) depends on M-1, M-3, M-4, M-5.
M-7 (calibration runs) depends on M-2 and M-6. M-8 (evaluation) depends on M-7.
M-9 (readout) depends on M-8.