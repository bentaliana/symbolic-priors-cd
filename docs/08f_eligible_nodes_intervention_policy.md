# Adjudication (a): Eligible-nodes intervention-set policy for calibration and held-out evaluation

**Status:** Decision-ready. Proposed policy with literature-grounded reasoning, awaiting integration into docs/02 as the next amendment and into docs/03 as a paired decision entry.

**Decision date:** 22/05/2026

**Author scope:** Internal project document. The synthesis below is the evidence base and reasoning trail for the docs/02 amendment that will follow. It is not itself the thesis methodology text; see §7 for the relationship between this document and the eventual thesis methodology section.

---

## 1. Question being decided

Docs/02 §4.2 names `do(X_j = ±2)` as the intervention magnitude convention for "eligible nodes" but does not pin which nodes are eligible, how many, how they are selected, or whether the policy is symmetric across calibration and held-out evaluation. Phase A (reproduction-pass) used the minimal `do(X_0 = ±2)` as smoke coverage — two intervention conditions on one node — explicitly as a runner-sanity check rather than as an evidence-generating policy.

For Commit 9 (calibration runner) and Commit 10 (held-out evaluation runner) to be implemented, the project needs a frozen answer to six sub-questions:

1. **Intervention magnitude.** Uniform constant across nodes, or node-adapted to natural variance.
2. **Intervention coverage.** How many target nodes per evaluation cell, relative to total nodes (d = 10 for the project's selection cell).
3. **Root-node policy.** Whether root-of-DAG nodes (nodes without parents in the ground-truth graph) are included as intervention targets.
4. **Topological stratification.** Whether target nodes are deliberately stratified by depth or other structural features.
5. **Stage symmetry.** Whether the policy is identical at calibration and held-out, or deliberately asymmetric.
6. **Cross-seed and cross-method consistency.** Whether target node-index sets are identical across replicate seeds and across competing models.

The decision below resolves all six.

---

## 2. What this document does and does not decide

**This document decides:**
- The eligible-nodes intervention-set policy for calibration (Commit 9) and held-out evaluation (Commit 10).
- The application of `do(X_j = ±2)` to the set of eligible nodes (the magnitude convention itself is preserved from §4.2 and not changed).
- The cross-seed and cross-method consistency discipline within calibration and within held-out.

**This document explicitly does not decide:**
- Adjudication (b): DCDI fit-RNG seed convention for stages beyond reproduction-pass. Closed by docs/02 v1.10 and the 22/05/2026 docs/03 entry "DCDI fit-RNG seed convention frozen for calibration and held-out evaluation"; the fit-RNG was frozen at `seed_torch = seed_numpy = 42` for every DCDI fit at Phase B calibration and held-out evaluation, with a small pre-declared post-selection sensitivity diagnostic.
- Adjudication (c): Commit-9 selected-configuration artefact path. Closed by the 22/05/2026 docs/03 entry "Selected-configuration artefact path and schema frozen for Commit 9 to Commit 10 handoff" together with the subsequent condition × model schema correction; the artefact is the JSON file `results/model_selection/calibration/<calibration_run_hash12>/selected_configurations.json` with top-level `selections[condition][model]`.
- Phase A (reproduction-pass) policy: unchanged. Phase A retains its minimal `do(X_0 = ±2)` smoke coverage as documented in docs/02 §3.3.
- Intervention magnitude convention: unchanged. The `do(X_j = ±2)` magnitude is preserved from docs/02 §4.2.
- The §2 lexicographic selection rule, seed pools, thresholds, budgets, metrics, wrapper APIs, configuration hash semantics, or any other frozen convention.

---

## 3. Evidence base

Three sources of unequal epistemic weight:

**Source 1 — Brouillard et al. (2020) DCDI paper [Tier 1: read directly].** One of the project's two candidate base models, making it the most directly relevant precedent. Read in full from `papers/DCDI.pdf`, with specific attention to Section 4 (Experiments), Appendix B.1 (Synthetic data sets), Appendix B.5 (Default hyperparameters and hyperparameter search), and Appendix C.6 (Evaluation on unseen interventional distributions).

**Source 2 — Q1 Elicit report (May 2026) [Internal synthesis; direct source for project reasoning, not primary literature].** Systematic review titled *"Intervention-Selection Policies in Causal Discovery"*, generated via Elicit semantic search across 500 candidate papers, 25 screened in, 20 included as eligible (publication years 2019–2025). Screening *considered* small-scale (10–30 nodes) settings, linear-Gaussian or additive-noise SEMs, sample-based distributional metrics, and ground-truth structure as inclusion-relevant criteria; the final synthesis found that only one of the 20 included studies (Chevalley et al. 2025 [Q1 ref 5]) actually used a sample-based distributional metric, while the remaining 19 relied primarily on structural accuracy metrics (SHD, SID, FPR, FNR). The full PDF is available in the project. We possess and have read the report directly, but the report is a synthesis document, not peer-reviewed primary literature; the 20 papers it cites remain Tier 3 unless separately promoted by direct reading (see §10).

**Source 3 — CausalBench (Chevalley et al. 2025, *Communications Biology*; arXiv preprint version 2022) [Tier 2: bibliographically verified, not read in full].** Author list and bibliographic details verified across five independent sources (arXiv listing, Semantic Scholar, PubMed, the CausalBench GitHub citation file, and the authors' own BibTeX file). Used as supplementary precedent only for the methodological choice of sample-based distributional metrics, not for intervention-target selection (CausalBench's interventions are determined by biological experiments, not design-time choices, and operate on 1000+ gene networks rather than 10-node SEMs). The companion CausalBench Challenge paper (Chevalley et al. 2025, PMLR) is also at Tier 2 and provides the methodological description of the mean Wasserstein distance metric used in this document.

---

## 4. Per-axis findings and decisions

Each sub-axis is presented in three parts: what the literature shows, what the decision is, and why.

### 4.1 Intervention magnitude

**Decision:** Retain the docs/02 §4.2 convention of uniform `do(X_j = ±2)` across all targeted nodes. No node-adapted scaling.

**Findings.**

*[Tier 1]* DCDI's perfect-intervention convention at Appendix B.1: "For perfect intervention, the distribution of intervened nodes is replaced by a marginal N(2, 1). This type of intervention, that produce a mean-shift, is similar to those used in [GIES (Hauser & Bühlmann 2012), Squires et al. 2020]." The DCDI convention is a stochastic mean-shift centred at 2 with unit variance; the project's deterministic `do(X_j = ±2)` is on the same magnitude scale, with the sign-symmetry adding the negative perturbation.

*[Tier 3 via Q1 report]* The Q1 Elicit report identifies a clean two-way split. Uniform-magnitude precedents include Xue et al. (2023) "applied uniform effects across all nodes"; Xue et al. (2024–2025) "uniform intervention magnitudes"; Peng et al. (2020) "uniform diagonal values of 0.5 for each node"; Sharifian et al. (2025) "uniform magnitudes across nodes by replacing causal mechanisms with exogenous noise"; Chevalley et al. (2025, ICLR) "drew fixed constants from a signed Uniform distribution between 1.0 and 5.0". Node-adapted precedents include Hauser & Bühlmann (2015, JRSS-B) "scaled magnitudes to observational standard deviations, with expectation values ranging from 1 to 50"; Reiser (2022) "bounded intervention magnitudes between the fifth and 95th percentile of each variable"; Chen et al. (2024) "node-adapted magnitudes by sampling new weights for shifted nodes from Unif[6,8]". *Note: Publication years here have been corrected against the Q1 report's tags where independent verification has established the correct year (see §8 reference list and §9 audit trail).*

**Reasoning.**

The literature splits on this axis. Both uniform and node-adapted approaches are defensible. The decision to retain uniform `do(X_j = ±2)` rests on three grounds:

(i) *Direct precedent from the most relevant base model.* DCDI's convention at d=10 is uniform mean-shift at magnitude 2, which is essentially the project's convention in deterministic form.

(ii) *Cleaner interpretation under MMD evaluation.* Node-adapted magnitudes introduce a confound: any cross-node variation in measured MMD reflects both (a) sampler quality differences and (b) intervention-magnitude differences. Uniform magnitudes isolate (a), which is what the H1–H4 hypotheses are designed to test.

(iii) *Preservation of the existing docs/02 §4.2 convention.* The amendment proposed by this document is additive (specifies which nodes, not what magnitude). Changing both axes simultaneously would expand scope without methodological gain.

### 4.2 Intervention coverage (number of target nodes)

**Decision:** All 10 nodes targeted at held-out evaluation, with both `do(X_j = +2)` and `do(X_j = -2)` — yielding 20 intervention conditions per seed.

**Findings.**

*[Tier 1]* DCDI's d=10 convention at Appendix B.1: "For 10-node graphs, single node interventions are performed on every node." Direct precedent for all-nodes coverage at the project's selection-cell size, from one of the project's two candidate base models. (DCDI's d=20 convention differs: "For 20-node graphs, interventions target 1 to 2 nodes chosen uniformly at random" — but the project's selection cell is d=10, so the d=10 convention is the relevant one.)

*[Tier 3 via Q1 report]* The Q1 report records no consensus on coverage. Densities span from sparse (Castelletti & Peluso 2023 at 2–4 nodes from 20–40 total; Chen et al. 2024 at 15%) through systematic ratios (Hauser & Bühlmann 2015 tests 20%/50%/100%; Chevalley et al. 2025 ICLR tests 25%/50%/75%/100%) to complete coverage (Xue et al. 2023 and 2024–2025 intervene on every node; Peng et al. 2020 designs experiments where each node receives one intervention covariate). The Q1 report frames the divide as application-driven: complete coverage suits high-throughput experimental settings, sparse coverage suits domains with constrained interventions.

*[Tier 2 via CausalBench]* CausalBench varies "fraction_partial_intervention" at 25%/50%/75%/100% as a *training-data* parameter for benchmark algorithms, not as an evaluation policy. This is precedent for systematic density variation but applies to a different question than the project's.

**Reasoning.**

(i) *Maximises evidence per held-out seed.* The thesis's primary scientific question concerns unseen-intervention generalisation; held-out evaluation needs sufficient intervention conditions to detect generalisation differences between prior-on and prior-off models. All-nodes-both-signs gives 20 conditions per seed.

(ii) *Direct alignment with DCDI's d=10 convention.* This is the strongest single-paper precedent for the project's setting. Within the literature's heterogeneous range, all-nodes coverage at d=10 is explicitly used by the project's primary candidate base model.

(iii) *Bounded compute cost.* See §4.5 for the calibration arithmetic and §6 for the implementation budget calculation. Compute is bounded by fit time, not intervention-sampling time, at the project's scale.

(iv) *Honest framing in the eventual selection-study report.* "All nodes" is a defensible policy that does not require the project to justify why specific nodes were chosen. The alternative ("we evaluated on nodes X, Y, Z") raises a viva question that "all nodes" does not.

### 4.3 Root-node policy

**Decision:** Include root nodes in the eligible-nodes set without exclusion.

**Findings.**

*[Tier 1]* DCDI's "every node" convention at d=10 implicitly includes root nodes; the paper does not discuss roots as a special case. This is the strongest single precedent for the project's setting and supports inclusion.

*[Tier 3 via Q1 report]* "No study explicitly documented whether root nodes were systematically included or excluded from targeting [Q1 refs 1, 11–17]." The literature is silent on this axis. Either choice is defensible against the field's current state.

**Reasoning.**

The argument for excluding roots — that root interventions are "trivially predictable" because the model has no parent structure to fail at — is considered and rejected on three grounds:

(i) Root interventions still test the model's ability to *propagate* the intervention forward through its learned structure to descendants. The downstream interventional distribution depends on the model's learned parent-structure for *all* descendants of the intervened root. Root interventions are not trivial; they test forward-propagation.

(ii) Root-set size varies across SCM realisations (each ER2 graph has a different number of roots, typically 1–3 nodes). Excluding roots would mean per-seed evaluation counts vary, complicating aggregation across seeds.

(iii) The DCDI precedent for the project's primary candidate base model is unambiguous and includes roots.

### 4.4 Topological stratification

**Decision:** Do not stratify target nodes by topological depth or other structural features.

**Findings.**

*[Tier 1]* DCDI does not stratify; its d=10 "every node" convention is uniform across the topological structure.

*[Tier 3 via Q1 report]* "Stratification by topological depth or graph distance from roots was not mentioned in any study's intervention selection protocol [Q1 refs 1, 11–17]." The literature is silent.

**Reasoning.**

At d=10 with all-nodes coverage, the stratification question is moot — every topological depth stratum is fully represented because every node is targeted. Stratification would only become meaningful under sparse-coverage policies (which the decision in §4.2 rejects).

If post-hoc analysis of held-out results suggests depth-specific patterns worth reporting, that becomes a finding for the selection-study report rather than a design choice baked in upfront.

### 4.5 Stage symmetry (calibration vs held-out)

**Decision:** Same all-nodes-both-signs coverage at calibration and held-out. Both stages evaluate on 20 intervention conditions per seed.

This yields:
- **Calibration:** 20 intervention conditions × 2 calibration-pool seeds (201, 202) = 40 intervention cells per candidate configuration. The full calibration grid evaluates 5 candidate configurations per model, producing 200 intervention cells per model across the grid.
- **Held-out evaluation:** 20 intervention conditions × 5 held-out-pool seeds (301, 302, 303, 304, 305) = 100 intervention cells per selected configuration per model.

**Findings.**

*[Tier 1]* DCDI's hyperparameter selection at Appendix B.5: "For score-based methods (GIES, CAM and DCDI), we select it by maximizing the held-out likelihood as explained in Appendix B.5 (without using the ground truth DAG)." DCDI uses held-out negative log-likelihood on a standard 80/20 validation split for hyperparameter selection. This is an asymmetry in *metric* (held-out NLL for model selection vs interventional NLL for generalisation evaluation in Appendix C.6), not in *intervention set*.

*[Tier 3 via Q1 report]* "Only one study explicitly documented different intervention policies across evaluation stages [Xue et al. 2023, Q1 ref 6]." Xue et al. (2023) "performed 5-fold cross-validation with separate data instances re-drawn for each DAG during model selection and hyperparameter tuning, then evaluated methods on the original simulated data after hyperparameter tuning. However, the study did not explicitly specify whether intervention sets differed between stages." The literature does not converge on stage-asymmetric intervention sets.

**Reasoning.**

The argument for asymmetric coverage — that calibration's purpose is ranking and held-out's is performance estimation, so narrower calibration coverage would save compute — is considered and rejected on four grounds:

(i) The literature does not converge on asymmetric coverage. Only one of 20 Q1-included studies addresses stage-specific protocols at all, and even that one does not differentiate intervention sets.

(ii) Symmetric coverage is methodologically simpler and avoids the design question "narrower-by-how-much."

(iii) Compute cost at the project's scale is bounded. For each model at calibration: 5 configurations × 2 calibration seeds × 20 intervention conditions × 1000 samples × 2 (model + ground-truth) = 400,000 sampled observations per model across the full calibration grid, excluding fit cost. This is comparable to reproduction-pass compute per seed and well within the 30 GPU-hour budget ceiling per docs/02 §8.

(iv) Symmetric coverage produces calibration evidence that is directly comparable to held-out evidence on the same intervention set — useful for diagnosing whether configurations selected at calibration generalise to held-out seeds.

### 4.6 Cross-seed and cross-method consistency

**Decision:** The same node-index policy is used across all replicate seeds within a seed-population (calibration seeds {201, 202}; held-out seeds {301, 302, 303, 304, 305}). The same node-index policy is used across DAGMA and DCDI within each stage.

**Precise wording matters here.** What is fixed across seeds is the *node-index policy* — the rule "intervene on all node-indices j ∈ {0, ..., 9}." What is *not* the same across seeds is the *causal role* of those indices, because each ER2 graph realisation produces a different random DAG. In seed 201, node-index 0 may be a root; in seed 202, node-index 0 may be a leaf. The node-index policy is index-stable; the causal-role distribution emerges from the SCM realisation under each seed.

**Findings.**

*[Tier 1]* DCDI's evaluation runs all methods on the same data per generated graph. Cross-method consistency is implicit in the experimental design.

*[Tier 3 via Q1 report]* "Only two studies explicitly addressed consistency across random seeds. Xue et al. (2024) used the same target-node sets across different random seeds and replicate runs [Q1 ref 7]." For cross-method consistency: "Xue et al. (2023) used the same target-node sets across different competing methods, evaluating all methods on the same original simulated data after cross-validation [Q1 ref 6]. Xue et al. (2024) similarly used pre-specified, consistent target-node sets across methods [Q1 ref 7]. Chen et al. used the same target-node sets across methods when comparing with DCI [Q1 ref 3]." Three of 20 included studies explicitly documented this.

*[Tier 1 — Q1 report's own methodological argument]* The Q1 report's Synthesis section explicitly argues for consistency as a discipline: "The absence of standardized intervention-selection protocols for cross-method benchmarking represents a critical gap... This creates potential for unfair comparisons where method A's superior performance may reflect fortuitous alignment between its inductive biases and the specific intervention targets chosen, rather than genuine algorithmic superiority." This argument supports adopting consistency as policy regardless of whether the field has converged on it.

**Reasoning.**

At d=10 with all-nodes both-signs coverage, "consistency" simplifies to "the policy is the same fixed set of 20 conditions for every replicate seed and for both candidate models within each stage." The set is fully determined by the policy and the node count; there is no random selection that could differ across runs. This trivially satisfies both consistency requirements without further design.

### 4.7 Supplementary methodological precedent for sample-based distributional evaluation

Independent of intervention-selection design, the project's use of sample-based distributional metrics (MMD) for evaluating learned causal models on unseen interventions is itself a methodological choice that benefits from precedent grounding.

**Findings.**

*[Tier 3 via Q1 report]* "Only one study employed sample-based distributional metrics matching the research question criteria. Chevalley et al. used Wasserstein distance computed via the SCIPY Python package, with 5000 observational samples and 100 samples per intervention." Per Q1's central finding: "19 studies relied exclusively on structural accuracy metrics (SHD, FPR, FNR)." Sample-based distributional evaluation is *uncommon* in the Q1 sample.

*[Tier 2]* CausalBench (Chevalley et al. 2025, *Communications Biology*; arXiv preprint 2022): per the CausalBench Challenge paper (Chevalley et al. 2025, PMLR 275:533–551), CausalBench "proposes to compute the empirical Wasserstein distance as a measure of distributional change under the effect of the intervention on the parent node of each predicted edge in the output graph, then taking the mean over the scores of all edges." Wasserstein-based distributional evaluation has documented precedent in causal discovery via the Chevalley line of work.

*[Tier 1]* DCDI Appendix C.6 uses log-likelihood on held-out interventional distributions as a generalisation measure: "we evaluate the likelihood of the fitted model on the remaining unseen interventional distribution." This is a different distributional metric (NLL vs MMD/Wasserstein) but in the same family.

**Honest framing.** Distribution-based evaluation of learned causal models is *uncommon* in the field per Q1 (1 of 20 included studies). The project's choice of MMD is *novel-but-precedented*: novel because MMD specifically does not appear in the Q1 sample; precedented because Wasserstein-based distributional evaluation has documented precedent in the Chevalley line of work (CausalBench in 2022/2025, "Deriving Causal Order" in ICLR 2025). MMD is itself well-established as a kernel-based distributional metric outside causal discovery, but I have not verified MMD-in-causal-discovery precedents specifically; that is open for further verification before thesis methodology writing.

---

## 5. Honest limitations and acknowledged uncertainties

**Limitation 1: Most cited papers are read via Q1 report summaries, not directly.** This document cites Hauser & Bühlmann (2015, JRSS-B), Castelletti & Peluso (2023, JASA), Xue et al. (2023, arXiv; 2024–2025, iScience), Peng et al. (2020), Chen et al. (2024), Chevalley et al. (2025, ICLR), Sharifian et al. (2025), and Reiser (2022) via the Q1 Elicit report's documented summaries (Tier 3). The Q1 report is itself a synthesis with its own LLM-driven extraction methodology; any inaccuracy in the Q1 summaries propagates here. Note also that the Q1 report tagged several of these papers with preprint years (2013, 2022, 2024) rather than journal-publication years (2015, 2023, 2025); the corrected years used in this document are recorded in §8 and §9. Where a citation becomes load-bearing for thesis methodology, the original paper should be promoted to Tier 1 by direct reading. See §10 for the thesis-readiness checklist.

**Limitation 2: The literature does not converge on a single canonical answer.** Per the Q1 report's central finding: "The field lacks standardized conventions for intervention-selection in held-out evaluation sets, with substantial heterogeneity across all six dimensions queried." This document identifies *defensible positions* from the documented range; it does not claim those positions are uniquely correct. The eventual selection-study report should frame the policy as "the literature does not converge; we chose from the defensible range with documented reasoning" rather than "following the literature's convention."

**Limitation 3: The project's evaluation setting is genuinely novel-by-omission.** Per Q1, only 1 of 20 included studies used a sample-based distributional metric (Wasserstein), and the project uses MMD specifically. CausalBench provides Wasserstein-based precedent but operates on real biological data with biology-determined interventions rather than synthetic SEMs with design-time intervention selection. The project is doing something the literature does not extensively do; some design choices around intervention selection for MMD-based evaluation on synthetic SEMs are genuinely constructive rather than purely precedent-following.

**Limitation 4: Hauser & Bühlmann (2015) and Hauser & Bühlmann (2012) are distinct works.** The Q1 report cites Hauser & Bühlmann with year 2013 (the arXiv preprint year), referring to *"Jointly interventional and observational data: estimation of interventional Markov equivalence classes of directed acyclic graphs"*, published in *Journal of the Royal Statistical Society: Series B*, vol. 77(1), pp. 291–318, January 2015 (DOI: 10.1111/rssb.12071; arXiv preprint 1303.3216 from March 2013). This is distinct from Hauser & Bühlmann (2012), the GIES paper in *Journal of Machine Learning Research* (*"Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs"*), which is cited by DCDI. These appear to be distinct works by the same authors; I have not verified directly whether they make identical or distinct substantive claims. This document treats them as distinct references and does not attribute claims from one to the other. The 2015 publication year is verified independently (see §9).

**Limitation 5: I have not searched for MMD-specific precedent in causal discovery.** The MMD-vs-Wasserstein methodological choice has not been literature-grounded in this document. If MMD has documented precedent in causal discovery evaluation, that precedent would strengthen §4.7. Open for verification before thesis methodology writing.

**Limitation 6: CausalBench is supplementary, not load-bearing.** CausalBench operates on real biological data (CRISPRi perturbation experiments on 1000+ genes), not synthetic SEMs. Its interventions are determined by biological experiments, not by a selection policy. CausalBench is therefore appropriate as supplementary precedent for sample-based distributional evaluation (it uses Wasserstein distance) but inappropriate as load-bearing precedent for intervention-target selection. This document treats it accordingly throughout.

---

## 6. Frozen policy for implementation

The following constitutes the proposed amendment to docs/02 (eligible-nodes intervention-set policy):

```
For both calibration and held-out evaluation, the eligible-nodes
intervention set is defined as: all nodes j ∈ {0, 1, ..., 9} of the
10-node ER2 selection cell, with both intervention signs (positive
and negative). Each evaluation stage thus produces 20 intervention
conditions per seed: {do(X_j = +2), do(X_j = -2) for j ∈ [0, 9]}.

The intervention magnitude is the docs/02 §4.2 frozen convention of
|X_j| = 2 (uniform across nodes, deterministic point intervention).

Root-of-DAG nodes are included as intervention targets without
exclusion. Topological-depth stratification is not applied; all-nodes
coverage at d = 10 obviates per-stratum selection.

The same node-index policy is applied across all replicate seeds
within a stage (calibration-pool seeds {201, 202}; held-out-pool
seeds {301, 302, 303, 304, 305}) and across both candidate base
models (DAGMA, DCDI) within each stage. Note: the node-index policy
is index-stable across seeds; the causal roles of those indices
(root, intermediate, leaf) vary per SCM realisation, as expected
under random ER2 graph generation.

The intervention-set policy is therefore identical across calibration
and held-out stages, the same across all replicate seeds within a
stage, and the same across both candidate base models.

Implementation arithmetic:
- Calibration: 20 intervention conditions × 2 calibration seeds = 40
  intervention cells per candidate configuration. The full calibration
  grid evaluates 5 candidate configurations per model, producing 200
  intervention cells per model across the grid.
- Held-out: 20 intervention conditions × 5 held-out seeds = 100
  intervention cells per selected configuration per model.

Phase A (reproduction-pass) retains its minimal do(X_0 = ±2) smoke
coverage as documented in docs/02 §3.3; this amendment does not
change Phase A.
```

A paired docs/03 entry should record:

- Date of decision (22/05/2026).
- Adjudication scope (eligible-nodes policy for calibration and held-out evaluation only).
- The six-axis per-axis position summarised.
- The evidence base (Q1 Elicit report; DCDI Appendix B.1 and B.5 read directly; CausalBench supplementary check verified via web search).
- Explicit statement of what does NOT change: intervention magnitude convention from §4.2; Phase A policy; selection rule from §2; seed pools; thresholds; budgets; metrics; wrapper APIs; configuration hash semantics.
- Explicit forward note: this policy will be implemented by the Commit 9 calibration runner and the Commit 10 held-out evaluation runner. Configuration hashes of those runners will incorporate the intervention-set policy via canonical JSON serialisation.
- Explicit acknowledgement that the literature does not converge on a single policy; this document records the project's reasoned choice from the documented defensible range.

---

## 7. What this means for the thesis

This document is the **evidence base and reasoning trail** that underwrites the eligible-nodes design decision. It is *not* the thesis methodology section, but it provides the citation map that section will draw on.

**Relationship to thesis writing:**

The thesis methodology section will be a compressed version of this document, with citations to the same primary sources but at higher confidence (Tier 1 readings of papers currently at Tier 3 here). The compression preserves:

1. The headline framing: "the literature does not converge on intervention-selection policies for held-out evaluation of learned SCMs; the project adopts all-nodes-both-signs coverage from the documented defensible range."

2. The two strongest precedents: DCDI's d=10 all-nodes convention (Brouillard et al. 2020, Appendix B.1, directly cited) and the Chevalley line of distributional-metric work (CausalBench Communications Biology 2025 + Chevalley et al. ICLR 2025, directly cited).

3. The methodological argument for cross-seed and cross-method consistency (Q1 Synthesis section's own framing, supported by Xue et al. 2023/2024 and Chen et al. 2024 documentation of this discipline).

4. The honest acknowledgement that the project's MMD-based evaluation is novel-but-precedented relative to the field.

The thesis methodology section will *not* compress out:

- The honest framing about field heterogeneity. "We chose from the defensible range" is stronger than "we followed convention" because the convention does not exist.
- The explicit per-axis reasoning for the six sub-axes. Each can be challenged in viva and each must be defensible.
- The cross-seed and cross-method consistency rationale. This is the methodological discipline the project follows that the field largely does not, per Q1.

**What this document is not:**

- It is not the docs/02 amendment text. The §6 frozen-policy block above is the *content* of the amendment; the actual amendment will be reformatted to match docs/02 voice and inserted in the appropriate section.
- It is not a literature review. It is a position document with sources, organised around the project's specific design question.
- It is not exhaustive. The limitations in §5 are real and should be transparent in the eventual selection-study report and thesis methodology section.

---

## 8. References

Citations are organised by verification tier. Each entry includes the tier marker, bibliographic details as substantiated, and notes on what verification was performed.

### Tier 1: Read directly

**Brouillard, P., Lachapelle, S., Lacoste, A., Lacoste-Julien, S., & Drouin, A. (2020).** *Differentiable Causal Discovery from Interventional Data.* 34th Conference on Neural Information Processing Systems (NeurIPS 2020). arXiv:2007.01754v2.
- Verification: Read directly from `papers/DCDI.pdf`. Specific sections cited: Section 4 (Experiments), Appendix B.1 (Synthetic data sets), Appendix B.5 (Default hyperparameters), Appendix C.6 (Evaluation on unseen interventional distributions).

**Q1 Elicit Report (May 2026).** *Intervention-Selection Policies in Causal Discovery.* Generated via Elicit semantic search across 500 candidate papers, 25 screened in, 20 included.
- Verification: PDF available in project (provided by user). Citations from the report are attributed to the report's specific findings; the 20 underlying papers are Tier 3 unless separately read.

### Tier 2: Bibliographically verified, not read in full

**Chevalley, M., Roohani, Y. H., Mehrjou, A., Leskovec, J., & Schwab, P. (2025).** CausalBench: A large-scale benchmark for network inference from single-cell perturbation data. *Communications Biology*, 8, Article 412. https://doi.org/10.1038/s42003-025-07764-y. arXiv preprint version: arXiv:2210.17283 (October 2022).
- Verification: Author list and bibliographic details verified across five independent sources — arXiv listing (arxiv.org/abs/2210.17283), Semantic Scholar paper page, PubMed (PMID associated with Communications Biology version), the CausalBench GitHub citation file (github.com/causalbench/causalbench), and the corresponding author's BibTeX file (schwabpatrick.com/bibtex/causalbench.txt).
- Author affiliations: M.C., Y.R., A.M. and P.S. associated with GSK plc; J.L. associated with Stanford University.

**Chevalley, M., Sackett-Sanders, J., Roohani, Y. H., Notin, P., Bakulin, A., Brzezinski, D., Deng, K., Guan, Y., Hong, J., Ibrahim, M., Kotlowski, W., Kowiel, M., Misiakos, P., Nazaret, A., Püschel, M., Wendler, C., Mehrjou, A., & Schwab, P. (2025).** The CausalBench challenge: A machine learning contest for gene network inference from single-cell perturbation data. *Proceedings of Machine Learning Research*, 275, 533–551. Proceedings of the Fourth Conference on Causal Learning and Reasoning (CLeaR 2025), Lausanne, 7–9 May 2025; published 15 June 2025. arXiv preprint version: arXiv:2308.15395 (August 2023, revised May 2025).
- Verification: Full 18-author list, venue, page range, and conference dates verified independently across four sources — PMLR proceedings index (proceedings.mlr.press/v275/), dblp Conference on Causal Learning and Reasoning 2025 listing, ADS Harvard listing, and Semantic Scholar paper page. The 2023 arXiv preprint has been revised; the 2025 PMLR version is the formal publication and should be cited in thesis methodology.

### Tier 3: Cited via Q1 Elicit Report's documented summaries

The following references are listed with corrections to the Q1 Elicit report's year tags where independent verification has established the correct publication year, and noted otherwise. **Each of these must be promoted to Tier 1 by direct reading before being cited in the thesis methodology section.**

- **[Q1 ref 1]** Castelletti, F., & Peluso, S. (2023). Network Structure Learning Under Uncertain Interventions. *Journal of the American Statistical Association*, 118(543), 2117–2128. DOI: 10.1080/01621459.2022.2037430. *Note: First online 2022 (the year Q1 records); issue publication 2023. Publication-status details sourced from ChatGPT's review of this document (May 2026); independent verification of specific issue and pagination pending before thesis use.*
- **[Q1 ref 3]** Chen, T., Bello, K., Locatello, F., et al. (2024). Identifying General Mechanism Shifts in Linear Causal Representations. *Neural Information Processing Systems*. arXiv:2410.24059.
- **[Q1 ref 4]** Hauser, A., & Bühlmann, P. (2015). Jointly interventional and observational data: estimation of interventional Markov equivalence classes of directed acyclic graphs. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 77(1), 291–318. DOI: 10.1111/rssb.12071. arXiv preprint: 1303.3216 (March 2013, the year Q1 records). *Note: This is distinct from Hauser & Bühlmann (2012), the GIES paper in JMLR, which is cited by DCDI. 2015 publication year verified independently via JRSS-B journal listing (see §9).*
- **[Q1 ref 5]** Chevalley, M., Schwab, P., & Mehrjou, A. (2025). Deriving Causal Order from Single-Variable Interventions: Guarantees & Algorithm. *International Conference on Learning Representations (ICLR 2025)*. arXiv preprint: 2405.18314 (May 2024, the year Q1 records). *Note: The Q1 Elicit report listed this as a 2024 paper (preprint year); the conference paper was presented at ICLR 2025 on Thursday 24 April 2025. ICLR 2025 venue verified independently via OpenReview submission record and Patrick Schwab's LinkedIn announcement (see §9).*
- **[Q1 ref 6]** Xue, A. Y., Rao, J., Sankararaman, S., & Pimentel, H. (2023). dotears: Scalable, consistent DAG estimation using observational and interventional data. arXiv preprint: 2305.19215.
- **[Q1 ref 7]** Xue, A., Rao, J., Sankararaman, S., & Pimentel, H. (2024–2025). dotears: Scalable and consistent directed acyclic graph estimation using observational and interventional data. *iScience*. DOI: 10.1016/j.isci.2024.111673. *Note: Publication timing around late 2024 / early 2025 depending on online vs issue date; specific issue and pagination should be verified before thesis use. Q1 Elicit report listed year 2024.*
- **[Q1 ref 8]** Peng, S., Shen, X., & Pan, W. (2020). Reconstruction of a directed acyclic graph with intervention. *Electronic Journal of Statistics*. DOI: 10.1214/20-ejs1767.
- **[Q1 ref 9]** Sharifian, E., Salehkaleybar, S., & Kiyavash, N. (2025). Near-Optimal Experiment Design in Linear non-Gaussian Cyclic Models. arXiv preprint: 2509.21423.
- **[Q1 ref 10]** Reiser, C. (2022). Observational and Interventional Causal Learning for Regret-Minimizing Control. arXiv preprint: 2212.02435.

---

## 9. Verification audit trail

This section records exactly what verification was performed for each citation, so the document is self-auditing.

**Verification history.** Initial document drafted 22/05/2026 with Tier 1 verification of DCDI (direct read) and Tier 2 verification of CausalBench benchmark (Chevalley et al. 2025, *Communications Biology*; cross-checked across five sources). A subsequent ChatGPT review (22/05/2026) flagged several Q1-derived citations as having year tags that corresponded to preprint years rather than journal-publication years. Three of those flagged items (CausalBench Challenge 2025 PMLR, Hauser & Bühlmann 2015 JRSS-B, Chevalley "Deriving Causal Order" ICLR 2025) were independently verified via web search before being applied; two (Castelletti & Peluso 2023 JASA, Xue et al. publication timing) remain at ChatGPT-sourced confidence and are flagged for independent verification before thesis use.

### Tier 1 verifications

**Brouillard et al. (2020) DCDI:**
- Source: `papers/DCDI.pdf` in project.
- Quotes cited in this document are taken directly from Appendix B.1 ("For 10-node graphs, single node interventions are performed on every node"; "For perfect intervention, the distribution of intervened nodes is replaced by a marginal N(2, 1)"), Appendix B.5 ("For score-based methods (GIES, CAM and DCDI), we select it by maximizing the held-out likelihood"), and Appendix C.6 ("we evaluate the likelihood of the fitted model on the remaining unseen interventional distribution").
- All quotes are direct from the paper; no paraphrasing without quotation marks.

**Q1 Elicit Report:**
- Source: PDF provided by user.
- Quotes in this document are from the report's Synthesis section, the per-axis Findings sections, and the Reference list.
- All Tier 3 citations in this document are anchored to the Q1 report's documented summaries.

### Tier 2 verifications

**CausalBench benchmark (Chevalley et al. 2025, Communications Biology; arXiv preprint 2022):**
- Author list verified across five sources:
  1. arXiv listing at arxiv.org/abs/2210.17283 ("Authors: Mathieu Chevalley, Yusuf Roohani, Arash Mehrjou, Jure Leskovec, Patrick Schwab").
  2. Semantic Scholar paper page (`semanticscholar.org/paper/CausalBench:-A-Large-scale-Benchmark-for-Network-Chevalley-Roohani/...`).
  3. PubMed listing for the Communications Biology version: "Mathieu Chevalley 1 2, Yusuf H Roohani 1 3, Arash Mehrjou 1, Jure Leskovec 3, Patrick Schwab 4".
  4. CausalBench GitHub repository citation file (`github.com/causalbench/causalbench`).
  5. Corresponding author Patrick Schwab's BibTeX file (`schwabpatrick.com/bibtex/causalbench.txt`).
- Publication details for Communications Biology version: volume 8, Article 412, DOI 10.1038/s42003-025-07764-y.
- Author affiliations verified via PubMed and Communications Biology metadata: M.C., Y.R., A.M., P.S. affiliated with GSK plc; J.L. affiliated with Stanford University.

**CausalBench Challenge (Chevalley et al. 2025, PMLR 275:533–551):**
- Full 18-author list verified across four independent sources:
  1. PMLR proceedings index for volume 275 (proceedings.mlr.press/v275/).
  2. dblp Conference on Causal Learning and Reasoning 2025 listing (dblp.org/db/conf/clear2/clear2025.html).
  3. ADS Harvard listing (ui.adsabs.harvard.edu/abs/2023arXiv230815395C/abstract).
  4. Semantic Scholar paper page.
- Authors: Chevalley M, Sackett-Sanders J, Roohani YH, Notin P, Bakulin A, Brzezinski D, Deng K, Guan Y, Hong J, Ibrahim M, Kotlowski W, Kowiel M, Misiakos P, Nazaret A, Püschel M, Wendler C, Mehrjou A, Schwab P.
- Venue: Proceedings of the Fourth Conference on Causal Learning and Reasoning (CLeaR 2025), held in Lausanne, Switzerland, 7–9 May 2025. Published as PMLR volume 275 on 15 June 2025. Series editors: Neil D. Lawrence. Volume editors: Biwei Huang, Mathias Drton.
- arXiv preprint (2308.15395) submitted August 2023, revised May 2025; the 2025 PMLR version is the formal publication.
- Methodological claim about Wasserstein distance verified via the CausalBench Challenge paper's direct text (verified via arXiv HTML version and ResearchGate PDF excerpt): "Chevalley et al. (2022) proposes to compute the empirical Wasserstein distance as a measure of distributional change under the effect of the intervention on the parent node of each predicted edge in the output graph, then taking the mean over the scores of all edges."

**Hauser & Bühlmann (2015), JRSS-B [promoted from Tier 3 to Tier 2 for bibliographic verification only; not read in full]:**
- Journal publication verified across multiple sources:
  1. RePEc ideas.repec.org listing (vol 77, issue 1, January 2015, pp 291–318).
  2. arXiv preprint listing (1303.3216) showing submission 13 March 2013.
  3. ResearchGate paper page citing "Hauser & Bühlmann (2015)" consistently.
- Distinct from the 2012 JMLR GIES paper "Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs" cited by DCDI.
- The Q1 Elicit report's year tag (2013) corresponds to the arXiv preprint year, not the journal publication year. This is bibliographic verification only; the paper's substantive claims have not been independently checked against the Q1 report's summary.

**Chevalley, Schwab & Mehrjou (2025), "Deriving Causal Order" [promoted from Tier 3 to Tier 2 for bibliographic verification only; not read in full]:**
- ICLR 2025 venue verified across three sources:
  1. OpenReview submission record (openreview.net/forum?id=u63OVngeSp) showing ICLR 2025 submission with explicit acknowledgement of "https://iclr.cc/Conferences/2025/AuthorGuide".
  2. Patrick Schwab's LinkedIn announcement listing presentation at ICLR 2025 ("Session: Hall 3 + Hall 2B #461 | Thursday, 24 April, 3:00 – 5:30 p.m").
  3. ETH/GSK affiliation cross-reference consistent with the 2024 arXiv preprint.
- The Q1 Elicit report's year tag (2024) corresponds to the arXiv preprint year (arXiv:2405.18314, May 2024), not the conference publication year. This is bibliographic verification only; the paper's substantive claims have not been independently checked against the Q1 report's summary.

### Tier 3 — what is NOT verified

For each Tier 3 citation, the following has *not* been verified by this document:
- Direct access to the paper.
- Direct verification that the claim attributed to the paper (via the Q1 report's summary) appears in the paper.
- Verification of specific page references, section numbers, or quotation accuracy.

The Q1 Elicit report's extraction methodology is documented in its own "Data extraction" section, which describes an LLM-driven extraction process applied to the 20 included papers. This is a reasonable methodology for synthesis work but does not substitute for direct reading where citations become load-bearing.

---

## 10. Thesis-readiness checklist

This checklist identifies which citations need to be promoted from their current tier to Tier 1 before the thesis methodology section is written.

### Must promote to Tier 1 (load-bearing for thesis methodology)

- **Xue et al. (2024–2025) "dotears" iScience paper [Q1 ref 7].** This is the strongest documented precedent for cross-method and cross-seed consistency. If the thesis methodology section cites the consistency discipline, this paper should be read directly. Publication timing (late 2024 vs early 2025) and specific issue/pagination should be verified at the time of direct reading. Expected effort: ~1 hour.

- **Chevalley et al. (2025) "Deriving Causal Order from Single-Variable Interventions" ICLR 2025 paper [Q1 ref 5].** This is the only Q1-included study using sample-based distributional metrics (Wasserstein), making it the closest methodological neighbour to the project's MMD-based approach. If the thesis methodology section frames the project's evaluation as in the Chevalley line, this paper should be read directly. Bibliographic verification (ICLR 2025 venue) already completed at Tier 2; substantive content still requires direct reading. Expected effort: ~1.5 hours.

- **CausalBench (Chevalley et al. 2022/2025).** Already Tier 2; should be promoted to Tier 1 by reading at least Section 4 (Evaluation methodology) and the supplementary intervention-evaluation details. Expected effort: ~1 hour given the paper is open-access at the Communications Biology version (DOI 10.1038/s42003-025-07764-y).

### Should promote to Tier 1 (supporting citations likely to appear in methodology)

- **Hauser & Bühlmann (2015), JRSS-B paper [Q1 ref 4].** Foundational work on intervention design; widely cited. If the thesis methodology section makes any general claim about intervention design conventions, this paper should be at Tier 1. Bibliographic verification (2015 journal publication) already completed at Tier 2; substantive content still requires direct reading. Expected effort: ~2 hours given its length and density.

- **Xue et al. (2023) "dotears" arXiv preprint [Q1 ref 6].** Predecessor to the iScience version; may not need independent reading if the iScience version is read. Verify whether the two versions differ substantively. Expected effort: ~0.5 hours if iScience version already read.

### May remain at Tier 3 (supporting reference only, not directly cited)

- Castelletti & Peluso (2023, JASA) [Q1 ref 1].
- Chen et al. (2024) [Q1 ref 3].
- Peng et al. (2020) [Q1 ref 8].
- Sharifian et al. (2025) [Q1 ref 9].
- Reiser (2022) [Q1 ref 10].

These can remain at Tier 3 if the thesis methodology section cites them only in passing or as part of the documented Q1 spread, without making specific claims that require direct verification.

### Total verification work for thesis-readiness

Approximately 5–6 hours of focused reading, distributed across the must-promote and should-promote items. This work can happen in parallel with the implementation work for Commits 9–10 and does not gate the docs/02 amendment proposed by §6 of this document.

---

## 11. Next actions

i. Apply the docs/02 amendment using the §6 frozen-policy block as the content, reformatted to match docs/02 voice. This is a doc-only commit; no source, test, config, or schema change. The amendment can be drafted as a Claude Code prompt analogous to the prior Path B amendment (docs/02 v1.8) and the subsequent path-unification commits.

ii. Add the paired docs/03 entry per §6 with the dated decision, the per-axis positions, and the explicit "does NOT change" list.

iii. Decide on the filename for this document. Suggested: `docs/08f_eligible_nodes_intervention_policy.md`, consistent with the docs/08 family naming pattern (08a–08e currently in use). This document is intended to be tracked in the repository as part of the project's methodology evidence base.

iv. Begin the Q2 Elicit run for adjudication (b) (DCDI fit-RNG convention) in parallel. Adjudication (b) is independent of (a) and can be processed in parallel without blocking.

v. Schedule the verification work from §10 (must-promote citations) to happen alongside Commit 9 implementation, so that the thesis methodology section has Tier 1 citations available when it is written.

vi. Adjudication (c) (Commit-9 selected-configuration artefact path) does not require literature work and can be decided on engineering grounds at any point before Commit 9 implementation begins.

---

**End of document.**

This synthesis is the evidence base for the eligible-nodes intervention-set policy decision. It documents what the literature shows, what the project decided, and why. It is structured for two downstream uses: immediate use as the basis for the docs/02 amendment, and later use as the citation map for the thesis methodology section. The verification audit trail (§9) and thesis-readiness checklist (§10) are designed to make the second use traceable and auditable.

Where the document acknowledges uncertainty, that acknowledgement is itself a methodological discipline: better to record what we have not verified than to imply certainty we do not have.
