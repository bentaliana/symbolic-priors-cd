# Prior Knowledge Integration in Differentiable Causal Discovery: Targeted Scoping Review and Prior-Loss Design Decision

**Document type:** Internal research record and design decision note.  
**Status:** Proposed for incorporation into docs/09. Decisions become binding when recorded in docs/09 and docs/03.  
**Date:** May 2026  
**Important:** This document is an internal research trail, not a final scholarly citation source. All factual claims attributed to individual papers must be verified against primary sources before use in the thesis literature review. The Elicit report guides source selection and positions the contribution; the thesis must cite primary papers directly.

---

## Search Methodology

A targeted scoping review was conducted using the Elicit semantic search engine (138 million papers indexed, covering Semantic Scholar and OpenAlex). The query targeted NOTEARS, DAGMA, GOLEM, and successor methods from 2018 to the present, with explicit focus on methods that learn a continuous weighted adjacency matrix and incorporate domain or symbolic prior knowledge beyond generic regularisation. Soft interventions and purely discrete Bayesian network structure learning were excluded unless explicitly connected to continuous differentiable optimisation.

500 papers were retrieved. 25 were included after holistic screening against nine criteria: differentiable objective, continuous graph representation, gradient-based optimisation, domain-specific prior knowledge, explicit prior-knowledge incorporation, mathematical formulation detail, foundational NOTEARS/DAGMA/GOLEM lineage, continuous optimisation framework, and methodological adequacy. 475 papers were excluded as below the screening threshold.

**Coverage limitation:** Of the 25 included papers, 8 had full text available for extraction; 17 were assessed from abstracts only. All claims about what individual papers do or do not implement should be read as "within the reviewed set, accounting for abstract-level coverage limitations," not as definitive statements about the entire literature.

**Post-Elicit concurrent papers:** Following the Elicit search, two additional close-neighbour papers were identified (Section 5). These papers were identified after the Elicit scoping review and are treated as concurrent close-neighbour work that refines the positioning of this thesis, rather than as evidence that the original Elicit review failed. Their implications for each finding are noted in the individual finding sections.

---

## Finding 1: No Included Paper Uses a Fixed Targeted Confidence-Weighted Frobenius Penalty

The most prevalent approaches to incorporating structural priors in continuous causal discovery are hard equality or inequality constraints enforced via augmented Lagrangian methods, and projection or masking operations applied at each gradient step. Chowdhury et al. (2023) implemented expert knowledge as hard constraints within a NOTEARS framework, demonstrating statistically significant structural accuracy improvements when constraints correctly identified active edges. Tran et al. (2024) used a mask-based implementation for topological ordering priors, reporting improvements in expected structural Hamming distance and AUROC relative to unconstrained baselines.

Soft targeted penalties on specific edges are considerably rarer. Within the reviewed set, only two papers were found to use targeted L1 penalties on individual edges: Waxman et al. (2024), who introduced DAGMA-DCE as an interpretable non-parametric extension of DAGMA with targeted L1 edge penalties, reporting improved SID and F1 over standard DAGMA and NOTEARS; and Zhang et al. (2026), who combined transfer-entropy-weighted ADMM with targeted L1 penalties on specific directed edges.

No included paper implemented a fixed targeted confidence-weighted L2 or Frobenius penalty of the form:

$$\mathcal{L}_{\text{prior}} = \lambda_{\text{prior}} \sum_{(i,j) \in \mathcal{F}} c_{ij} W_{ij}^2$$

where $W$ is the continuous weighted adjacency matrix, $\mathcal{F}$ is the set of prior-forbidden directed edges, and $c_{ij} \in [0, 1]$ are **pre-specified** per-edge confidence weights. The Elicit scoping review identifies this specific formulation as a gap: "No study implemented targeted L2/Frobenius penalties of the form $L_{\text{prior}} = \lambda \times \Sigma\, c_{ij} \times W_{ij}^2$" (Elicit Systematic Review, 2026). After the Elicit search, PRCD-MAP (Shan & Zhou, 2026) was found to use a "prior-weighted ℓ₂ regularizer" alongside an ℓ₁ penalty in a MAP objective — but with trust weights **learned from data** via empirical Bayes, not pre-specified and systematically corrupted. The isolated fixed-weight targeted Frobenius formulation in this thesis remains, to the best of this review, unreported in the literature as a standalone design choice with controlled corruption evaluation.

**Implication for this thesis:** To the best of this targeted review plus post-Elicit search, a fixed confidence-weighted targeted Frobenius penalty with systematic corruption evaluation appears to be a distinct and unreported contribution. PRCD-MAP's concurrent use of a prior-weighted ℓ₂ in a MAP framework provides supporting evidence that the MAP/L2 direction is principled, while the fixed-versus-learned distinction preserves the originality of the controlled experimental design.

---

## Finding 2: Principled Confidence-Weighted Uncertain Priors Are Rare in the Reviewed Literature

Within the Elicit-reviewed set, twenty-three of 25 included papers treated prior knowledge as equally reliable and certain, with no modelling of uncertainty, partial reliability, or differential confidence per edge or constraint (Elicit Systematic Review, 2026). Two exceptions existed but neither provided a principled confidence-weighting framework. Pal et al. (2025) included weights $\omega_{ij}$ in a soft bow-freeness penalty but did not specify how these weights are determined or related to confidence levels. Tran et al. (2024) implicitly associated confidence with topological ordering information without explicit mathematical representation.

After the Elicit search, PRCD-MAP (Shan & Zhou, 2026) emerged as a direct close-neighbour work that explicitly assigns per-edge trust to imperfect priors. PRCD-MAP's trust is calibrated by empirical Bayes on a Laplace-approximated marginal likelihood and propagated via an MLP. This represents a data-driven learned approach to the same fundamental challenge — heterogeneous reliability of prior knowledge — that this thesis addresses through fixed pre-specified confidence weights.

The distinction is material and pre-committed. The project protocol (docs/01 v1.1, non-negotiable structural commitments) explicitly specifies "fixed confidence weights in the main study." PRCD-MAP's empirical Bayes trust learning is not a different implementation choice; it is a fundamentally different research question: "how to learn which priors to trust from data" versus this thesis's question: "how does performance degrade as we systematically corrupt fixed-confidence priors." These questions are complementary. This thesis does not compete with PRCD-MAP; it occupies a controlled experimental niche that PRCD-MAP's adaptive approach cannot substitute.

**Implication for this thesis:** Principled confidence-weighted uncertain priors are rare but no longer absent from the literature. The field is emerging in this direction. This thesis contributes a controlled experimental study of fixed confidence weights under systematic corruption — a design that is deliberate, pre-registered, and distinct from data-driven trust learning.

---

## Finding 3: Systematic Prior Corruption Evaluation Is Largely Absent From the Reviewed Literature

Within the 25 Elicit-included papers, no study evaluated performance under deliberately corrupted, systematically incorrect, or reliably specified prior knowledge at known corruption levels (Elicit Systematic Review, 2026). The most relevant adjacent finding was Chowdhury et al. (2023), who found that constraints on active edges have a larger positive impact than constraints on inactive edges, but this asymmetry was not studied under deliberate corruption.

After the Elicit search, two concurrent works were found that directly address imperfect priors. PRCD-MAP (Shan & Zhou, 2026) explicitly includes corruption analyses — evaluating robustness under random, systematic, and adversarial prior corruption for its learned-trust mechanism. Wang et al. (2025) study performance degradation under flawed constraints of unknown location and type. Both papers confirm that imperfect-prior robustness is a live and open problem. The claim that prior corruption evaluation is entirely absent from the literature is therefore too strong; PRCD-MAP clearly conducts such analysis.

The specific unquantified gap that remains is narrower and more precise: the degradation behaviour of fixed pre-specified confidence-weighted priors in DAGMA, evaluated on interventional metrics (SID, MMD) under controlled corruption at specified levels, compared against matched-L1 and hard-exclusion baselines. PRCD-MAP's corruption analysis evaluates a learned-trust mechanism on structural metrics. Wang et al. address unknown-location constraint errors without a specified corruption grid or interventional evaluation. Neither paper isolates the fixed-confidence degradation question against matched baselines on interventional performance. That gap — not the broad absence of corruption evaluation — is what this thesis addresses.

**On the corruption grid:** The Elicit review recommended corruption levels of 10%, 30%, and 50%. That recommendation is noted but not adopted. The corruption grid is already specified in the frozen design contract (docs/01 v1.1, Section 11) as 0%, 20%, 40%, 60%, 80%, predating this review. The Elicit recommendation confirms the need for corruption evaluation; it does not override the pre-registered grid.

**Implication for this thesis:** Prior corruption evaluation is no longer entirely absent from the literature — PRCD-MAP conducts such analysis for its adaptive learned-trust mechanism. The unquantified gap this thesis addresses is specific: the degradation threshold for fixed confidence-weighted priors in DAGMA, measured on interventional metrics (SID, MMD), under controlled corruption at specified levels, compared against matched-L1 and hard-exclusion baselines. That precise gap remains unaddressed in all identified literature, including PRCD-MAP and Wang et al.

---

## Finding 4: Baseline Isolation for Semantic Content Is Inadequate in the Reviewed Literature

Within the reviewed set, no paper was found to implement random-prior or shuffled-prior controls (Elicit Systematic Review, 2026). No paper was found to use a matched-sparsity control specifically designed to test whether gains from a targeted confidence-weighted Frobenius prior exceed what an equivalently sparse global regulariser would achieve. The review states: "Only Zhang et al. (2026) compared L1 and L2 generic penalties, but not in a targeted context. This makes it impossible to determine from existing work whether improvements stem from the semantic correctness of priors or merely from the regularisation strength" (Elicit Systematic Review, 2026).

No included paper compared soft and hard constraint mechanisms applied to identical prior information under identical experimental conditions. The concurrent papers (PRCD-MAP, Wang et al.) also do not include matched-sparsity or hard-constraint comparison baselines designed to isolate semantic content from regularisation effects. This gap persists across all identified literature.

**Implication for this thesis:** The baseline suite here — prior-free DAGMA, matched-L1 DAGMA, hard-exclusion DAGMA, and random-prior sensitivity addendum — is designed to address control gaps that persist across the reviewed literature and the identified concurrent work. The matched-L1 baseline and hard-exclusion comparison are not standard in this space; including them strengthens the evidentiary value of the study.

---

## 5. Close-Neighbour Addendum: Concurrent Works on Imperfect Priors (Post-Elicit)

Following the Elicit scoping review, two close-neighbour papers were identified that directly address imperfect or uncertain priors in causal discovery. They are treated as concurrent close-neighbour work that refines the positioning of this thesis, rather than as evidence that the original Elicit review failed. By the time of thesis final submission, both constitute related work that must be engaged with directly in the literature review.

**PRCD-MAP: Learning How Much to Trust Imperfect Priors in Causal Discovery (Shan & Zhou, 2026).** This paper proposes a soft prior-consumption layer assigning per-edge trust to imperfect priors, modulating a prior-aware ℓ₁ penalty and prior-weighted ℓ₂ regularizer in a MAP objective. Trust is calibrated by empirical Bayes on a Laplace-approximated marginal likelihood and propagated by an MLP. PRCD-MAP provides three contributions to this thesis's positioning. First, its use of a prior-weighted ℓ₂ in a MAP framework provides neighbouring precedent for the Gaussian/MAP justification of the Frobenius form adopted here. Second, it confirms that per-edge trust/confidence is a live and independently motivated research direction. Third, it defines what this thesis is not: PRCD-MAP learns trust adaptively from data; this thesis studies fixed confidence under controlled corruption. That distinction is pre-committed in docs/01 as a structural protocol constraint.

**Robust Causal Discovery under Imperfect Structural Constraints (Wang et al., 2025).** This paper addresses performance degradation under flawed constraints of unknown location and type, using a surrogate model to assess constraint credibility and a sparse penalisation term measuring the loss between learned and constrained adjacency matrices, framed as a multi-objective Pareto problem. It explicitly motivates the imperfect-prior problem and confirms that existing methods fail under unknown constraint errors. Note: Wang et al.'s method does not appear to operate within the continuous differentiable adjacency-matrix optimisation lineage (NOTEARS/DAGMA); their framework is not a direct architectural peer of this thesis's approach. The contribution is relevant as problem motivation, not as a methodological comparator.

**The adaptive-versus-diagnostic distinction.** The cleanest single framing of what separates this thesis from PRCD-MAP is the following. PRCD-MAP asks: "When prior accuracy is unknown, can the model infer which priors to trust?" This thesis asks: "For fixed pre-specified confidence priors, how much corruption can the soft-prior mechanism tolerate before its advantage over matched-L1 and hard exclusion disappears, as measured on interventional metrics?" These are complementary scientific questions. Answering the adaptive question does not remove the need for the diagnostic one; controlled fixed-prior studies provide the reference baseline against which adaptive trust learning should be compared. The thesis design is simpler, but simpler in a scientifically useful way.

**Revised positioning after concurrent work:** This thesis should no longer claim that confidence-weighted or imperfect-prior methods are absent from the literature — PRCD-MAP clearly addresses both. The thesis contributes a controlled diagnostic study: fixed confidence-weighted Frobenius priors in DAGMA, evaluated under systematic corruption at specified levels against a known synthetic ground truth, compared with matched-L1 and native hard-exclusion baselines, assessed on interventional metrics (SID, MMD). That is a distinct and defensible position within an active emerging research area. The contribution is not adaptive trust learning but a controlled evaluation of how a targeted Frobenius soft prior behaves relative to prior-free DAGMA, matched-L1 regularisation, and DAGMA-native hard exclusion as prior reliability degrades.

---

## 6. Synthesis: Positioning of the Targeted Frobenius Formulation

The Elicit scoping review and subsequent concurrent paper search together position the targeted Frobenius prior as theoretically motivated, partially supported by convergent concurrent work, and empirically distinct from all identified papers. Three justifications motivate its adoption over the L1-style candidate previously described in docs/06 Section 18.

**Theoretical justification (Bayesian MAP, scoped to DAGMA's continuous-weight setting).** In the continuous weighted-adjacency framework used by DAGMA (Bello et al., 2022), a confidence-weighted zero-mean Gaussian prior over forbidden edge weights yields a Frobenius penalty under MAP estimation. A forbidden edge $(i, j)$ with confidence $c_{ij}$ receives prior $W_{ij} \sim \mathcal{N}(0,\, (2\lambda_{\text{prior}} c_{ij})^{-1})$; taking the negative log-likelihood gives exactly $\lambda_{\text{prior}} c_{ij} W_{ij}^2$. This interpretation holds within DAGMA's continuous-weight setting. PRCD-MAP's independent use of a prior-weighted ℓ₂ in a MAP framework provides concurrent convergent support for this theoretical framing.

**Methodological justification (graceful degradation).** The L2 gradient at a forbidden position is $2\lambda_{\text{prior}} c_{ij} W_{ij}$, proportional to the current weight magnitude. Near zero the penalty is weak; the opposing force grows as the edge weight grows. Under prior corruption, the data gradient can reach equilibrium with the prior gradient, yielding a non-zero continuous weight. Hard exclusion projects $W_{ij}$ to exactly zero after each Adam step regardless of data signal. The Frobenius form occupies the continuous middle ground required for the graceful-degradation comparison that forms the thesis's secondary objective.

**Empirical contribution justification (controlled experimental gap).** No identified paper — including PRCD-MAP and Wang et al. — implements a controlled corruption protocol with specified corruption levels, fixed pre-specified confidence weights, synthetic known ground truth, and interventional performance evaluation (SID, MMD). This controlled experimental design is the primary empirical contribution.

**Thesis methodology positioning language (copy-paste ready, updated to reflect concurrent works):**

> "Prior knowledge integration in differentiable causal structure learning has predominantly used hard constraints, projection-based masking, or untargeted augmented Lagrangian formulations. Targeted soft penalties for specific forbidden edges are rare: within a targeted scoping review of 25 papers, only two studies were found to use targeted L1 penalties on specific edges (Waxman et al., 2024; Zhang et al., 2026), and no included paper used a fixed targeted L2 penalty with pre-specified confidence weights. Very recent concurrent work has begun to address imperfect priors: PRCD-MAP (Shan & Zhou, 2026) learns per-edge trust via empirical Bayes in a MAP framework using a prior-weighted ℓ₂ term, and Wang et al. (2025) address structural constraint credibility through surrogate-model assessment. This thesis contributes a distinct controlled experimental study: a fixed confidence-weighted targeted Frobenius penalty applied to DAGMA, evaluated under systematic prior corruption at specified levels against a known synthetic ground truth, with matched-L1 and native hard-exclusion baselines, assessed on interventional generalisation metrics (SID, MMD). The controlled fixed-confidence design is pre-committed in the project protocol and is complementary to, not redundant with, adaptive trust-learning approaches."

---

## 7. Limitations of This Evidence

This targeted scoping review is an internal research trail and design input, not a final scholarly citation source. Several limitations apply.

First, 17 of 25 Elicit-included papers were assessed from abstracts only. Paper-level claims about what specific methods do or do not implement may be incomplete. Before any claim appears in the thesis, the relevant primary paper must be read in full and cited directly.

Second, Elicit's extraction was performed by a large language model. Extraction errors, misclassification of penalty forms, and omissions from abstract-only papers are possible.

Third, the two concurrent papers (PRCD-MAP, Wang et al.) were identified after the Elicit search and not included in the 25-paper set. They are assessed from abstracts and partial search results only. Full reading of both papers is required before thesis citation.

Fourth, 475 papers were excluded at the Elicit screening stage. Relevant work may exist in excluded papers.

Fifth, Wang et al.'s base framework may not be in the continuous differentiable optimisation lineage, which would make it a less direct comparator than the abstract suggests. This must be confirmed by full-paper reading.

These limitations do not undermine the findings as design inputs. They require that the thesis literature review is grounded in primary paper reading.

---

## 8. Thesis Design Decisions

The following decisions are proposed for incorporation into docs/09 and docs/03.

### D1 — Adopt targeted Frobenius (L2) as the primary soft-prior penalty form

$$\mathcal{L}_{\text{prior}} = \lambda_{\text{prior}} \sum_{(i,j) \in \mathcal{F}} c_{ij} W_{ij}^2$$

Gradient added to DAGMA's `Gobj`:

$$G_{\text{prior}} = 2\lambda_{\text{prior}} (C \odot W)$$

Not scaled by DAGMA's $\mu$. Confidence weights $c_{ij}$ are **pre-specified and fixed** before training, not learned from data. This is a structural protocol commitment (docs/01 v1.1).

### D2 — This decision supersedes the L1-style candidate form in docs/06

docs/06 Section 18 described L1 as an example implementation path. This document provides the literature and design rationale for adopting the Frobenius (L2) form. PRCD-MAP's concurrent use of a prior-weighted ℓ₂ in a MAP framework provides supporting evidence. docs/09 records the Frobenius form as the implemented method, superseding the L1-style example.

### D3 — Matched-L1 DAGMA is core; it controls for global sparsity pressure

Answers: does semantic targeting of the Frobenius penalty produce interventional gains beyond what an equivalently sparse global L1 regulariser achieves? Selected by the match-by-sparsity rule (docs/01 Section 9). Must be implemented after soft-prior runs at 0% corruption and before headline evaluation.

### D4 — Hard-exclusion DAGMA is core; it controls for constraint mechanism

Uses DAGMA's native `exclude_edges` projected-Adam mechanism. Answers: does soft L2 prior uncertainty degrade more gracefully than non-negotiable hard exclusion under prior corruption? Source inspection (see project docs) confirms the mechanism is clean and implementable.

### D5 — Global-Frobenius sensitivity addendum (conditional)

A global Frobenius regulariser across all edges, tuned to match graph density, would isolate whether gains require semantic targeting. Scientifically desirable. Conditional: implement only after core pipeline is complete and time remains.

### D6 — Random-prior sensitivity addendum (conditional)

Tests whether improvement requires semantic correctness. Conditional on core pipeline completion.

**Prioritisation note (D5 vs D6):** If only one sensitivity addendum is feasible given time, prioritise D5 (global-Frobenius) over D6 (random-prior). Global-Frobenius directly answers the sharpest examiner objection — "Is this just L2 shrinkage?" — by isolating the targeted-versus-global confound within the same penalty family. Random-prior tests semantic alignment, which is useful but addresses a secondary concern given the matched-L1 baseline already provides sparsity isolation. The ordering is: core baselines first, global-Frobenius second if time remains, random-prior third.

### D7 — Metric hierarchy is unchanged

docs/01 Section 12: SID primary, MMD secondary with kernel-sensitivity checks, SHD auxiliary. Not modified by this review or by the concurrent papers, which focus on structural metrics.

---

## 9. Source Annotations

**Waxman, D., Butler, K., & Djuric, P. (2024).** DAGMA-DCE: Interpretable, non-parametric differentiable causal discovery. *IEEE Open Journal of Signal Processing.* https://doi.org/10.1109/OJSP.2024.3351593  
→ One of two papers using targeted L1 penalties on specific edges; closest existing precedent in the Elicit set; performance evidence (lower SID, higher F1). Used in Finding 1 and thesis positioning.

**Zhang, J., Cao, J., Rutkowski, L., & Shi, X. (2026).** Accelerated transfer-entropy-weighted ADMM for learning causal directed acyclic graphs. *IEEE Transactions on Network Science and Engineering.* https://doi.org/10.1109/TNSE.2026.3685889  
→ Second paper using targeted L1 penalties; only paper comparing L1 and L2 generic penalties (not in targeted context). Used in Findings 1 and 4.

**Pal, S., O'Quinn, J., Aryan, K., et al. (2025).** DAG DECORation: Continuous optimisation for structure learning under hidden confounding. *arXiv.* https://doi.org/10.48550/arXiv.2510.02117  
→ One of two Elicit-set papers with confidence-adjacent weighting; implicit confidence via $\omega_{ij}$ weights. Used in Finding 2.

**Chowdhury, J., Rashid, R., & Terejanu, G. (2023).** Evaluation of induced expert knowledge in causal structure learning by NOTEARS. *International Conference on Pattern Recognition Applications and Methods.* https://doi.org/10.48550/arXiv.2301.01817  
→ Hard-constraint NOTEARS; 10–20% structural accuracy gains; asymmetric impact of active vs. inactive edge constraints; assumes correct priors throughout. Used in Findings 1, 3, and 4.

**Tran et al. (2024).** TOBAC. *(Venue to be confirmed from primary paper before thesis citation.)*  
→ Mask-based implementation for ordering priors; one of two Elicit-set papers with implicit confidence. Used in Findings 1 and 2.

**Bello, K., Aragam, B., & Ravikumar, P. (2022).** DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterisation. *NeurIPS.*  
→ Base model this thesis extends; continuous weighted-adjacency framework underpinning the Gaussian MAP interpretation. Used in Finding 1 and Section 6.

**Sun, X., Schulte, O., Liu, G., & Poupart, P. (2021).** NTS-NOTEARS: Learning nonparametric DBNs with prior knowledge. *AISTATS.*  
→ 10–20% structural accuracy improvement from correct priors; used to establish gains are possible while noting absence of adequate controls. Used in Finding 4.

**Shan, X., & Zhou, D. (2026).** PRCD-MAP: Learning how much to trust imperfect priors in causal discovery. *arXiv.* https://arxiv.org/abs/2605.01669  
→ **Post-Elicit concurrent work.** Per-edge trust for imperfect priors; prior-aware ℓ₁ and prior-weighted ℓ₂ in MAP objective; trust learned via empirical Bayes and MLP. Supports MAP/L2 direction; defines the learned-trust approach that docs/01 explicitly excludes from this thesis's main study. Used in Section 5 (addendum), Findings 1, 2, 3, and positioning language.

**Wang, Z., Lin, X., He, C., & Gao, X. (2025).** Robust causal discovery under imperfect structural constraints. *arXiv.* https://arxiv.org/abs/2511.06790  
→ **Post-Elicit concurrent work.** Surrogate model credibility assessment; multi-objective Pareto optimisation for imperfect structural constraints; studies unknown-location constraint errors. Confirms imperfect-prior robustness is live problem. Framework may not be in the continuous differentiable NOTEARS/DAGMA lineage — verify before thesis citation. Used in Section 5 (addendum) and Finding 3.

**Elicit Systematic Review (2026).** *Incorporating symbolic priors in causal learning.* Elicit AI semantic search, 138 million papers. *(Internal arXiv/Elicit report, on file.)*  
→ Source of all quantitative claims about the 25-paper reviewed set; direct quotations in Findings 1–4; gap taxonomy; primary thesis positioning language. **Not a primary citation for the thesis; use this document as a trail to primary papers only.**

---

*End of document. Next: docs/09 main-study protocol.*