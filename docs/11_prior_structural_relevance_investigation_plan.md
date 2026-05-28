# Investigation Plan: Structural Relevance of Forbidden-Edge Priors to DAGMA Error Modes

> **ADDENDUM 2 (added 28/05/2026, DAGMA backbone correction).** The analysis hash `1b46785b59a4` quoted in Addendum 1 below was computed over the off-protocol-backbone M-8 run `864fe6722256`. The corrected M-8 run `166c792c43bc` was completed on 28/05/2026; the M-10 investigation has been re-run on the corrected records and now lives under analysis hash `6f660aaeef3d` at `results/main_study/exploratory/prior_structural_relevance/6f660aaeef3d/`. The pre-registered protocol in the body below is unchanged; only the underlying main-evaluation records and the resulting analysis hash differ. The Section-6 outcome category and the M-11 follow-up still hold under the corrected backbone (M-11 is now `079fda7ac4f4`).
>
> **ADDENDUM (added 27/05/2026, post-implementation drift audit).** The investigation pre-registered by this document has been completed. The plan body below is preserved as the pre-registered protocol.
>
> - **Analysis hash:** `1b46785b59a4`
> - **Output directory:** `results/main_study/exploratory/prior_structural_relevance/1b46785b59a4/`
> - **Readout document:** [docs/12_prior_relevance_findings_and_next_steps.md](docs/12_prior_relevance_findings_and_next_steps.md)
> - **Outcome category (Section 6 of this plan):** Outcome 2 (low coverage and small mean dSID under removal). Mean false-positive coverage ≈ 0.13; mean dSID under offline removal of reference forbidden edges ≈ −0.57; mean dSHD ≈ −1.29. The "Candidate C" structural-orthogonality explanation is partially supported.
> - **Follow-up (M-11, pre-registered in docs/12 Section "Next step"):** the oracle prior-relevance diagnostic was completed under analysis hash `1b95c563db88` and is recorded in [docs/13_prior_relevance_oracle_findings.md](docs/13_prior_relevance_oracle_findings.md). It refined the M-10 interpretation: false-positive removal carries substantially more SID leverage than false-negative addition, but the original prior targets did not consistently align with the SID-consequential false positives.
> - **Stop condition:** the M-10 + M-11 sequence is closed; the "Decision: Exploratory diagnostic sequence complete" entry in docs/03 sets the stop condition.

## Status

- Document version: 1.1 (revised draft, pre-investigation)
- Date created: 2026-05-26
- Scope: post-hoc exploratory analysis of existing M-8 records
- Pre-registration: this document is written *before* the investigation runs
- Headline result status: the pre-registered main-evaluation result (run hash `864fe6722256`) is unaffected by this document and remains the thesis's primary finding

## 1. Background

The pre-registered main evaluation (M-8) produced 224 fits across the frozen experimental grid. The M-9 readout established the following observations:

1. The targeted Frobenius prior engaged mechanically as designed: across all 7 evaluation seeds, soft_frobenius suppressed targeted forbidden edges to approximately 40% of the prior_free baseline weight (`reference_forbidden_edge_comparison.csv`).

2. This mechanical engagement did not translate to improvement on aggregate interventional metrics:
   - mean SID: prior_free = 66.1, soft_frobenius_clean_conf1 = 69.7 (mean_diff = +3.57, bootstrap CI [+1.29, ...])
   - mean MMD: prior_free = 0.113, soft_frobenius_clean_conf1 = 0.114 (mean_diff ~0, CI crosses zero)
   - hard_exclusion at clean prior outperforms soft_frobenius on every aggregate metric

3. Structural-interventional correlation across all 224 fits was moderate (SID-MMD Spearman = 0.55), indicating that SID/MMD and structural diagnostics are related but not interchangeable.

The pre-registered protocol treats this as a thesis-valid mixed result per docs/01 Section 14. The discussion section will address why the prior engaged without producing metric improvement.

## 2. The Suspicion

During post-hoc inspection, a structural question emerged about *why* the prior failed to improve metrics despite engaging mechanically. Three candidate explanations exist:

**Candidate explanation A:** `lambda_prior=2e-4` is too weak. The prior engages but with insufficient strength to produce metric-level effects.

**Candidate explanation B:** DAGMA's optimisation landscape is robust to soft prior perturbations at any practical lambda_prior. The acyclicity constraint and L1 regularisation dominate the loss surface; small Frobenius penalties on a subset of entries cannot shift the optimum meaningfully.

**Candidate explanation C:** The chosen prior class (forbidden edges over randomly-selected true-negatives) is *structurally orthogonal to DAGMA's error modes*. The prior suppresses entries that DAGMA was already suppressing through its own L1 regularisation, so the additional suppression has limited informational impact on the learned graph at positions where errors actually occur.

These three explanations are not mutually exclusive. All three could be partially true.

This investigation plan focuses on **Candidate C** because it is testable using existing M-8 data without any new fits, and because it has the strongest implications for thesis framing.

## 3. The Specific Claim Under Investigation

The claim to be investigated is:

> The forbidden-edge prior used in the main study may target positions where DAGMA does not typically make structural errors. DAGMA's structural errors on this benchmark may concentrate at *different* positions than those covered by the prior. If true, this would mean the prior is informationally orthogonal to DAGMA's empirical error structure, providing a mechanism-level explanation that contributes to (but does not necessarily fully explain) the observed result that mechanical engagement did not improve metrics.

This is not a claim that the prior is "bad" or that the experiment failed. It is a structural observation about the relationship between two distributions:
- The distribution of positions covered by the forbidden-edge prior (sampled from true-negative entries)
- The distribution of positions where DAGMA actually makes structural errors (which can be computed from existing records)

If these two distributions are largely disjoint, then the direct graph-level opportunity for this forbidden-edge prior to improve SID and SHD is limited. A stronger `lambda_prior` could still affect optimisation indirectly even when targeted positions are not thresholded false positives, but the direct mechanism would have limited room to operate.

## 4. How the Suspicion Arose from the Observed Results

The suspicion is anchored in three specific numerical observations from the M-9 readout, not in general dissatisfaction with the result:

**Observation 1 — Baseline SHD is high.** Mean SHD for prior_free is 24.0 across 7 seeds. The ground truth has approximately 20 directed edges. SHD of 24 against truth of 20 implies roughly half the edges in the learned graphs differ from the truth at any given fit. DAGMA is making many structural errors at the standardised condition.

**Observation 2 — The prior covers a small fraction of possible error positions.** The forbidden-edge prior specifies 10 entries per seed. The full d=10 adjacency matrix has 90 off-diagonal entries. The prior addresses at most 10/90 ≈ 11% of the possible error space. If DAGMA's errors are distributed across the matrix, the prior cannot directly address most of them.

**Observation 3 — DAGMA's L1 already produces low values at the targeted positions.** From `reference_forbidden_edge_comparison.csv`, prior_free's mean absolute weight at the targeted forbidden-edge positions ranges from 0.039 to 0.226 across seeds. For 5 of 7 seeds, the value is below 0.11. The thresholding rule converts weights ≥ 0.3 into edges, so most targeted positions are already below threshold in prior_free. The prior is reinforcing suppressions that DAGMA's L1 was already producing.

These three observations together suggest the prior may be suppressing positions that are not the bottleneck for DAGMA's structural accuracy. The errors that drive SID and MMD likely occur elsewhere — possibly at false-negative positions (missed true edges) or at false-positive positions outside the prior's coverage.

## 5. Investigation Methodology

The investigation uses **only existing M-8 records and artefacts**. The following constraints apply:

- No model fitting
- No MMD recomputation
- No new interventional sampling
- No replacement of saved metrics
- Only offline structural diagnostics over existing adjacency artefacts, including SID and SHD recomputation on edited adjacency matrices for the targeted-edge-removal diagnostic

All inputs derive from data already on disk under `results/main_study/864fe6722256/records/` and `results/main_study/864fe6722256/artefacts/`.

### 5.1 Data sources

For each of the 7 prior_free evaluation seeds (501-507):
- The learned thresholded adjacency matrix from `thresholded_adjacency.npz`
- The continuous weight matrix from `continuous_w.npz`
- The ground-truth adjacency matrix from `true_adjacency.npz`
- The forbidden-edge set from the corresponding soft_frobenius_clean_conf1 record's `corrupted_prior_spec.forbidden_edges`

### 5.2 Error position computation

For each prior_free fit, classify each off-diagonal entry (i, j) into one of four categories:

- **True positive (TP):** edge in truth AND edge in learned graph (correctly recovered)
- **True negative (TN):** no edge in truth AND no edge in learned graph (correctly absent)
- **False positive (FP):** no edge in truth AND edge in learned graph (spurious edge)
- **False negative (FN):** edge in truth AND no edge in learned graph (missed edge)

This produces a position-classified matrix per seed.

### 5.3 Coverage quantities to compute

For each seed:
- Total counts of TP, TN, FP, FN
- The set of FP positions (which entries DAGMA learned as spurious)
- The set of FN positions (which true edges DAGMA missed)
- The intersection of FP positions with forbidden-edge positions (where the prior could directly help)
- The intersection of FN positions with forbidden-edge positions (this is zero by construction — forbidden-edges are true-negatives in truth — but worth recording explicitly for the readout)

Two coverage ratios are computed per seed:

1. **False-positive coverage** (the primary ratio):
   ```
   |forbidden_positions ∩ FP_positions| / |FP_positions|
   ```
   This is the directly-addressable error denominator. Forbidden-edge priors can prevent false positives by suppressing entries that should be zero. They cannot fix false negatives by construction.

2. **Total-error coverage** (the secondary ratio):
   ```
   |forbidden_positions ∩ (FP ∪ FN)_positions| / |(FP ∪ FN)_positions|
   ```
   This is the broader denominator and is useful for context. If false negatives dominate the error budget, this ratio will appear small regardless of how well the prior targets false positives.

Aggregated across seeds:
- Mean false-positive coverage
- Mean total-error coverage
- Distribution of FP positions: are they concentrated at specific entries, or distributed across the matrix?
- Distribution of FN positions: which true edges does DAGMA tend to miss?
- Ratio of FN to FP in the total error budget

### 5.4 Offline targeted-edge removal effect

Coverage measures whether the prior overlaps DAGMA's errors. It does not measure whether removing those overlapping errors would actually improve structural metrics.

For each seed, compute the offline counterfactual:

```
For each seed in 501-507:
  W_thresholded = prior_free thresholded adjacency
  W_edited = copy of W_thresholded with the 10 reference forbidden edges set to 0
  SID_baseline = SID(W_thresholded vs true_adjacency)  [matches existing record]
  SHD_baseline = SHD(W_thresholded vs true_adjacency)  [matches existing record]
  SID_edited = SID(W_edited vs true_adjacency)
  SHD_edited = SHD(W_edited vs true_adjacency)
  
  delta_SID = SID_edited - SID_baseline
  delta_SHD = SHD_edited - SHD_baseline
```

This estimates the **maximum direct structural benefit** of perfectly removing the targeted edges, as if the prior had been a hard constraint applied as post-processing.

If the delta_SID and delta_SHD are near zero across all seeds, then even perfect targeted removal would not have improved structural metrics. This would substantially strengthen the case that the prior class is structurally orthogonal to where structural improvements are achievable on this benchmark.

If the deltas are meaningfully negative (improvement), then the prior had addressable opportunity that was not realised at `lambda_prior=2e-4`. This would shift weight toward Candidate A (lambda too weak) or Candidate B (optimisation robustness).

**Important caveat:** This diagnostic recomputes SID and SHD on edited adjacency matrices. It does not estimate counterfactual MMD. MMD depends on sampled interventional distributions from the learned model, not only the adjacency matrix. A pseudo-MMD computed from edited graphs would not correspond to any well-defined estimator. MMD is therefore interpreted only from existing saved M-8/M-9 outputs in this investigation; no MMD counterfactual is attempted.

### 5.5 Optional topological relevance

If straightforward, for each seed, examine whether the forbidden-edge positions tend to lie at topologically significant locations in the true causal graph (e.g., near the source nodes, near the sink nodes, on long causal paths). This is descriptive and not used to support any specific conclusion. It may be useful for the thesis discussion to note whether the prior tended to address structurally peripheral or structurally central positions.

### 5.6 Diagnostic outputs

The investigation should produce:

1. `prior_target_overlap.csv`: per-seed coverage ratios (false-positive coverage and total-error coverage) plus aggregate summaries.

2. `prior_free_error_decomposition.csv`: per-seed counts of TP/TN/FP/FN and the ratio of FN to FP.

3. `offline_forbidden_edge_removal_effect.csv`: per-seed SID_baseline, SHD_baseline, SID_edited, SHD_edited, delta_SID, delta_SHD.

4. `prior_edge_topological_relevance.csv` (optional): per-seed descriptors of where forbidden-edge positions sit in the causal structure.

5. `aggregated_error_heatmap.png` (if straightforward): a 10×10 heatmap showing aggregated error frequency across seeds, with the prior-covered positions overlaid.

6. `investigation_readout.md`: a concise narrative summary of the findings, written in cautious language consistent with the headline thesis discussion.

These outputs are diagnostic, not confirmatory. They describe structural properties of the existing data.

## 6. Pre-declared Interpretation of Possible Outcomes

To prevent post-hoc interpretation drift, this document pre-declares how each possible outcome will be interpreted.

**The following bands are heuristic decision aids for the diagnostic readout, not confirmatory statistical thresholds.** The exact percentages are pragmatic choices to anchor interpretation; an examiner asking "why 30% and not 25%?" should be answered with "the bands are heuristic; the actual interpretation depends on the full pattern of evidence, not a single threshold crossing."

The investigation has two primary diagnostics: false-positive coverage (ratio 1 in Section 5.3) and the offline targeted-edge removal effect (Section 5.4). Both must be considered jointly.

### 6.1 Outcome 1: High coverage AND meaningful removal effect

If mean false-positive coverage ≥ 30% AND mean delta_SID under perfect removal is meaningfully negative (improvement), then Candidate C is **not the dominant explanation**. The prior class is structurally relevant to the error modes and the targeted positions would have helped structural metrics if suppressed harder. The failure to improve metrics must be explained by Candidate A (`lambda_prior` too weak) or Candidate B (optimisation robustness) or some combination.

Implication: this would strengthen the case for an exploratory `lambda_prior` sensitivity analysis. The thesis discussion can frame the result around "the right prior class but at insufficient strength."

### 6.2 Outcome 2: Low coverage OR negligible removal effect

If mean false-positive coverage < 10%, OR if mean delta_SID under perfect removal is near zero (even with reasonable coverage), then Candidate C is **partially supported**. The prior class either does not overlap DAGMA's errors meaningfully, or it overlaps them but at positions that do not drive structural metrics.

Implication: this would not justify re-running the main experiment with different `lambda_prior` values, since the direct opportunity for this prior class to help structural metrics is limited. The thesis discussion can instead frame the result around "this prior class is informationally orthogonal to DAGMA's empirical error structure on this benchmark" and identify alternative prior classes (required-edge priors targeting false negatives, topological-order priors constraining the search space) as natural directions for future work.

### 6.3 Outcome 3: Mixed signals

If coverage is in the 10-30% range, or if the two diagnostics disagree (e.g., high coverage but negligible removal effect, or low coverage but meaningful removal effect at the few overlapping positions), the interpretation is ambiguous. Both Candidate A and Candidate C could be partially true.

Implication: the thesis discussion can acknowledge both explanations and note that disambiguation would require either a stronger-lambda exploratory analysis or a different-prior-class exploratory analysis. Neither will be conducted as part of this thesis unless explicitly pre-registered as a follow-up study.

### 6.4 What this investigation does NOT do

This investigation does not:

- Re-run the main experiment with different parameters
- Modify any frozen protocol constant
- Replace or alter the pre-registered headline result
- Constitute a confirmatory test of any hypothesis
- Conclude that the prior class is "wrong" or "right" — only whether it overlaps DAGMA's empirical error positions on this benchmark, and whether perfect overlap-removal would have helped structural metrics

The investigation produces a structural description of the data, which the thesis Discussion section may then interpret.

## 7. Implementation Path

The investigation will be implemented as a small analysis script under `experiments/main_study/exploratory/`, consuming the existing record JSONs and artefact NPZ files. No new modules in `src/` are required.

The SID computation for the offline removal diagnostic must use the same SID backend used by M-8 (the project's `gadjid` integration via `src/symbolic_priors_cd/metrics/sid_score.py`). The SHD computation must use the project's existing SHD function. This ensures the recomputed metrics on edited adjacencies are directly comparable to the saved baselines from M-8.

Estimated scope: 300-400 lines of analysis code plus tests plus a markdown readout. Estimated runtime: under one minute (no model fits, just inspection of existing arrays and 7 small SID/SHD recomputations).

Outputs will be saved to a clearly-labelled exploratory subdirectory:
```
results/main_study/exploratory/prior_structural_relevance/<investigation_hash12>/
  prior_target_overlap.csv
  prior_free_error_decomposition.csv
  offline_forbidden_edge_removal_effect.csv
  prior_edge_topological_relevance.csv          (optional)
  aggregated_error_heatmap.png                  (if straightforward)
  investigation_readout.md
```

The exploratory subdirectory naming makes clear that these outputs are not part of the pre-registered main study.

## 8. Documentation and Decision Trail

This investigation plan is committed to the repository before the investigation runs.

After the investigation completes, a brief decision-log entry will be added to `docs/03_decision_log.md` recording:
- That the investigation was conducted
- What it found (which outcome above applied)
- What the thesis discussion will say as a result
- Whether any further exploratory analysis is warranted

The investigation's outputs and readout are exploratory evidence. They inform the thesis Discussion section but do not alter the pre-registered Results.

## 9. What This Document Does Not Claim

To be explicit about scope:

- This document does **not** claim the main result is wrong or invalid
- This document does **not** claim the prior was incorrectly designed
- This document does **not** propose changes to the frozen protocol
- This document does **not** predict the investigation's outcome
- This document only states that a specific structural property of the data is worth examining, and pre-declares how the examination will be interpreted

The investigation may reveal that the prior was well-suited to DAGMA's errors, in which case `lambda_prior` strength becomes the more likely explanation. It may reveal that the prior was structurally orthogonal, in which case prior class becomes the more likely explanation. It may reveal an ambiguous mix. All three outcomes are scientifically valuable.

## 10. Next Step

Implementation of the investigation script. Before implementation, this plan document should be committed to the repository and reviewed.

Following commit, the investigation will be drafted as a small experiment under `experiments/main_study/exploratory/`, run on the existing M-8 data, and reported in a readout document. The findings will inform the thesis Discussion section's framing of the mechanism-without-metric-improvement observation.