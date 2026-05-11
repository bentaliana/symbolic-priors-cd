# 01_research_question_and_commitments

## Status

Frozen design contract for the main thesis study.  
Version 1.1.  
This document defines the non-negotiable scientific commitments for the main study before major implementation begins.

---

## Change log

- **v1.0 -> v1.1**:
  - replaced partially verbal hypotheses with operational versions;
  - merged the earlier observational–interventional gap and structural–interventional divergence into a single falsifiable divergence hypothesis;
  - split the soft-prior objective into DAGMA-specific and DCDI-specific candidate forms under a shared principle;
  - replaced the vague matched-L1 wording with an explicit match-by-sparsity rule;
  - tightened the random-prior control so it matches the semantic-prior setup more faithfully;
  - renamed "Success criteria" to "Positive-result criteria";
  - added explicit multiple-comparisons commitments;
  - added a dedicated subsection for deferred operational details that will be frozen in Document 02;
  - clarified which parts of the design are structural commitments and which are tactical commitments.

---

## 1. Scope and function

This document freezes the core research question, hypotheses, scope boundaries, benchmark commitments, evaluation hierarchy, and reporting rules for the thesis.

Its purpose is to prevent:

- scope drift,
- convenience-driven design changes,
- post-hoc rationalisation,
- and ambiguity in how results will later be interpreted.

This document is normative rather than historical. It defines the study contract for the main experiment. Historical reasoning, abandoned alternatives, and design rationale are recorded separately in the thesis decision log.

---

## 2. Research questions

### Primary research question

Do semantically meaningful but uncertain structural priors improve unseen-intervention generalisation in differentiable causal discovery more than matched non-semantic regularisation, when all models are trained on observational data only?

### Secondary research question

As structural priors become increasingly corrupted, do soft priors degrade more gracefully than hard constraints?

---

## 3. Headline empirical claim under test

The main study evaluates the following empirical claim:

> Semantically meaningful but uncertain structural priors can improve unseen-intervention performance more than matched generic regularisation under observational-only training, and soft priors may remain more robust than hard constraints as prior reliability degrades.

This is a testable claim, not an assumed conclusion.

---

## 4. Falsifiable hypotheses

### H1. Structural–interventional divergence

In the primary benchmark cell under observational-only training, the association between structural recovery and unseen-intervention performance is weak.

**Operationalisation:**  
The association between a structural metric and the primary interventional metric will be tested using a pre-registered rank-based association measure across methods and seeds in the primary benchmark cell. The exact threshold and test specification will be fixed in Document 02.

### H2. Semantic-prior advantage

A model augmented with confidence-weighted soft structural priors will outperform both:

1. an otherwise identical prior-free baseline, and
2. an otherwise identical model regularised with matched generic sparsity pressure,

on the primary interventional metric in the primary benchmark cell at low-to-moderate prior corruption.

**Default interpretation of low-to-moderate corruption:**  
0% to 40% corruption, unless revised through the change-log process after pilot evidence.

### H3. Graceful degradation

Across the pre-registered corruption grid, the soft-prior model will degrade more gracefully than the hard-constraint baseline on the primary interventional metric.

**Operationalisation:**  
Graceful degradation is defined by a lower degradation summary over the corruption grid, with the exact summary statistic fixed in Document 02.

### H4. Instability as diagnostic signal

Greater cross-seed structural instability will be associated with poorer unseen-intervention performance.

**Operationalisation:**  
For each experimental condition, instability is defined using a pre-registered cross-seed graph-variability measure. Its association with unseen-intervention error is then tested across conditions. The exact instability measure and test specification will be fixed in Document 02.

---

## 5. Claim-to-test mapping

The study tests each hypothesis using the following evidence:

- **H1** is tested by relating structural and interventional metrics under observational-only training in the primary benchmark cell.
- **H2** is tested by pairwise comparison of the soft-prior model against the prior-free and matched-L1 baselines in the primary benchmark cell.
- **H3** is tested through degradation analysis across the prior-corruption grid, with direct comparison against the hard-constraint baseline.
- **H4** is tested by relating cross-seed graph variability to unseen-intervention error.

This mapping is fixed before implementation of the full study.

---

## 6. Scope freeze

### Included in the main study

- observational-only training
- synthetic SCM benchmarks with known ground truth
- unseen-intervention evaluation
- two prior families only:
  - forbidden-edge priors
  - ordering / temporal priors
- four comparison baselines:
  - prior-free
  - matched L1
  - hard constraint
  - random-prior control
- corruption-based robustness analysis
- multiple random seeds
- primary emphasis on interventional evaluation

### Excluded from the main study

- training with interventional supervision
- ATE-based training regularisation
- monotonicity priors
- learned confidence weights in the main experiment
- real-data benchmarks as headline evidence
- benchmark expansion beyond the pre-specified cells unless explicitly documented as stretch work

---

## 7. Design commitments

### 7.1 Training regime

All main-study models are trained on observational data only.  
Interventional data is reserved exclusively for evaluation.

### 7.2 Base-model shortlist

The base learner for the main study will be chosen from the closed shortlist:

- DAGMA
- DCDI

No additional base models will be added unless both shortlisted models are disqualified under the bake-off protocol.

### 7.3 Confidence-weight policy

Confidence weights are fixed in the main study.

A small pre-specified grid over confidence strength will be evaluated. Learned confidence weights are excluded from the main study and may be considered only as an appendix-level exploratory ablation.

### 7.4 Prior families

The main study uses exactly two structural prior families:

1. forbidden edges
2. ordering constraints

Ordering constraints are operationalised as directional restrictions that induce sets of prohibited edge directions.

### 7.5 Monotonicity

Monotonicity priors are excluded from the main study because they add disproportionate formal and experimental complexity without strengthening the central comparison.

---

## 8. Main-study objective functions

### Shared principle

The prior penalty acts on the base model’s native continuous edge-representation object.  
The default prior penalty is L1-style because it directly encourages edge suppression. If pilot evidence justifies a different functional form, that change must be recorded explicitly through the change-log process.

### Candidate form if DAGMA is selected

Let:

- \(W\) denote DAGMA’s weighted adjacency matrix,
- \(\theta\) denote the remaining model parameters,
- \(\mathcal{L}\_{obs}(W,\theta)\) denote the observational fit term,
- \(h(W)\) denote DAGMA’s acyclicity term,
- \(\mathcal{F}\) denote the set of forbidden directed edges,
- \(\mathcal{O}\) denote the set of prohibited directed edges induced by ordering constraints,
- \(c\_{ij}\in[0,1]\) denote the fixed confidence weight for edge constraint \((i,j)\).

Then the default soft-prior objective is:

\[
\mathcal{L}_{\text{DAGMA}}(W,\theta)
=
\mathcal{L}_{obs}(W,\theta)

- \lambda\_{acyc} h(W)
- \lambda*{prior}
  \left(
  \sum*{(i,j)\in\mathcal{F}} c*{ij}\,|W*{ij}|
- \sum*{(i,j)\in\mathcal{O}} c*{ij}\,|W\_{ij}|
  \right).
  \]

### Candidate form if DCDI is selected

Let:

- \(\Lambda\) denote the matrix of relaxed edge parameters,
- \(P = \sigma(\Lambda)\) denote the continuous edge-existence matrix,
- \(\phi\) denote the remaining model parameters,
- \(\mathcal{L}\_{obs}(\phi,\Lambda)\) denote the observational fit term,
- \(h(P)\) denote the acyclicity term evaluated on the relaxed edge representation,
- \(\mathcal{F}\), \(\mathcal{O}\), and \(c\_{ij}\) be defined as above.

Then the default soft-prior objective is:

\[
\mathcal{L}_{\text{DCDI}}(\phi,\Lambda)
=
\mathcal{L}_{obs}(\phi,\Lambda)

- \lambda\_{acyc} h(P)
- \lambda*{prior}
  \left(
  \sum*{(i,j)\in\mathcal{F}} c*{ij}\,P*{ij}
- \sum*{(i,j)\in\mathcal{O}} c*{ij}\,P\_{ij}
  \right).
  \]

### Interpretation

- Forbidden-edge priors penalise nonzero support for edges that should be absent.
- Ordering priors penalise directed support that violates the specified order.
- The prior term is semantic and selective, not global.

### Matched-L1 baseline

The matched-L1 baseline uses the base model’s native generic sparsity term, with \(\lambda\_{L1}^{\*}\) selected by the match-by-sparsity rule defined in Section 9.

### Hard-constraint baseline

The hard-constraint baseline encodes the same prior information as non-negotiable structural restrictions rather than penalties.

### Random-prior control

The random-prior control matches the semantic-prior setup in:

- prior-family composition,
- number of constraints per family,
- corruption level,
- and confidence-weight schedule,

and differs only in semantic alignment with the ground-truth graph.

---

## 9. Matching rule for the generic regularisation baseline

The matched-L1 baseline exists to test whether any advantage of symbolic priors is due to semantic information rather than generic graph restriction.

### Primary fairness rule

The matched-L1 baseline is calibrated by match-by-sparsity in the primary benchmark cell at 0% corruption.

### Operational definition

For each candidate \(\lambda*{L1}\) in a pre-registered grid, compute the mean number of thresholded nonzero edges across the calibration seed set. Select \(\lambda*{L1}^{_}\) as the value whose mean edge count is closest to that of the soft-prior model on the same calibration seed set. The selected \(\lambda\_{L1}^{_}\) is then held fixed across all main-study runs in all benchmark cells and all corruption levels.

### Interpretation

This rule is intended to make the generic regularisation baseline comparable in effective sparsity pressure to the soft-prior method in the principal calibration setting. It does not imply that global sparsity pressure and semantically targeted prior penalties are perfectly equivalent.

### Robustness check

A secondary appendix-level comparison may report results under a more generous tuning protocol for the L1 baseline in order to test whether the main-study ordering survives stronger baseline tuning.

The exact thresholding rule, calibration grid, and calibration seed set are fixed in Document 02.

---

## 10. Benchmark commitments

### 10.1 Primary benchmark cell

The primary benchmark cell is:

- 10-node Erdős–Rényi DAG
- expected degree approximately 2
- synthetic SCM with known ground-truth graph
- observational training data only
- unseen single-node interventions at test time

This is the principal decision cell for the thesis.

### 10.2 Required ablation cells

At minimum, the main study includes:

- 20-node ER2
- 10-node ER4

These test whether conclusions survive changes in graph size and density.

### 10.3 Stretch cell

A 50-node setting is stretch work only and is not required for the main thesis claim.

### 10.4 Standardisation commitment

The bake-off will explicitly compare standardised and unstandardised synthetic settings.  
The main study will prioritise the setting judged more benchmark-valid after that comparison, with preference for the setting that reduces exploitable variance-order artefacts.

---

## 11. Corruption protocol commitments

Prior corruption is a central experimental axis.

The default corruption grid is:

- 0%
- 20%
- 40%
- 60%
- 80%

Corruption is applied before training.

### Forbidden-edge corruption

A proportion of forbidden-edge priors is made incorrect by flipping whether selected constraints are truly valid.

### Ordering corruption

A proportion of ordering constraints is corrupted by reversing or otherwise invalidating the corresponding directional restrictions.

### Additional note

If pilot experiments reveal that a corruption mechanism is ill-posed, the mechanism may be revised, but the corruption-based robustness design itself will not be abandoned.

---

## 12. Evaluation hierarchy

### 12.1 Primary metric

**SID** is the primary metric for the main study.

### 12.2 Secondary metric

**MMD** is the secondary metric and will be reported with kernel-sensitivity checks rather than as a single unqualified number.

### 12.3 Tertiary metric

**ATE error** is tertiary and will be reported only in synthetic settings where it is stable, meaningful, and non-redundant.

### 12.4 Structural metrics

Structural metrics may be reported as auxiliary diagnostics, but they are not the primary basis of the thesis claim.

---

## 13. Positive-result criteria

The method counts as a **full positive result** if, in the primary benchmark cell, all of the following hold:

1. the soft-prior model outperforms the prior-free baseline on the primary interventional metric at low-to-moderate corruption;
2. the soft-prior model outperforms the matched-L1 baseline in the same setting; and
3. the soft-prior model shows better degradation behaviour than the hard-constraint baseline across the corruption grid.

A **partial positive result** is obtained if only a proper subset of these conditions holds.

A **negative result** is obtained if none of these conditions is supported by the pre-registered evaluation and reporting rules.

Additional support is obtained if:

- the pattern replicates in the required ablation cells;
- the gains remain visible under the benchmark-valid standardisation choice; and
- structural gains do not fully account for the interventional gains.

---

## 14. Negative-result plan

A negative or mixed outcome remains thesis-valid.

The study will still count as scientifically successful if it establishes with clear evidence that:

- semantic priors do not outperform matched generic regularisation under the tested regime;
- soft priors help only when prior accuracy is high and lose value rapidly under corruption;
- hard constraints are brittle under even modest misspecification;
- structural improvements do not reliably translate into unseen-intervention gains; or
- cross-seed instability is a useful warning signal for interventional unreliability.

The thesis is therefore not contingent on proving universal superiority of soft priors. It is contingent on delivering a clean answer to the research question.

---

## 15. Statistical reporting commitments

The following rules are fixed in advance:

- all headline experiments use multiple random seeds;
- seed values are logged and retained;
- per-seed results are preserved, not only aggregate summaries;
- means and standard deviations are reported;
- medians may also be reported where distributions are skewed;
- paired comparisons are preferred where possible;
- 95% confidence intervals on key pairwise differences are reported using a justified resampling procedure;
- effect sizes and interval estimates are primary; p-values are secondary;
- Benjamini–Hochberg false discovery rate control at \(q = 0.05\) is applied to pre-declared families of comparisons;
- Bonferroni correction may be applied to a small number of headline claims where appropriate;
- no superiority claim will rely on a single seed or a single favourable figure;
- exploratory analyses are labelled explicitly as exploratory.

Formal hypothesis testing, if used, is secondary to effect sizes, interval estimates, and consistency across seeds and benchmark cells.

---

## 16. Reproducibility commitments

Every experimental run must save:

- configuration file
- random seed
- code version or git hash
- environment information
- learned graph or native edge outputs
- evaluation metrics
- corruption level
- benchmark cell identity

No headline result may be reported if it cannot be traced to a saved configuration and run record.

---

## 17. Real-data commitment

Sachs or another real-data dataset may be included only as a secondary case study or appendix-level demonstration.

Real-data evaluation will not serve as the primary evidence for the main thesis claim because the central claim depends on controlled ground-truth comparison under known prior corruption.

---

## 18. Change-control rule

After this document is frozen, the following may still change without redesign:

- wording refinements
- minor benchmark implementation details
- pilot-level tactical choices explicitly delegated to Document 02
- exact choice between DAGMA and DCDI, as determined by the bake-off protocol

The following may not change without explicit written justification and a version bump:

- observational-only training
- unseen-intervention evaluation as the main target
- two-prior-family scope
- fixed confidence weights in the main study
- matched baseline logic
- corruption-based robustness framing
- primary benchmark cell
- primary metric hierarchy

---

## 19. Deferred operational details fixed in Document 02

The following tactical commitments are deferred to `02_bakeoff_protocol.md` and the experimental protocol document, where they will be frozen before any main-study runs:

- base-model selection rule for DAGMA vs DCDI
- threshold for counting a DAGMA edge as nonzero
- threshold for counting a DCDI edge as nonzero
- L1 calibration grid
- intervention values for test-time do-operations
- MMD bandwidth grid for kernel-sensitivity analysis
- exact operational test specification for H1
- exact instability measure and test specification for H4
- multiple-comparisons correction families
- compute budget ceiling and scope-cut hierarchy
- standardisation decision rule

These are tactical rather than structural commitments. They may be revised before freezing in Document 02 without bumping the version of this contract. Once frozen in Document 02, they follow the same change-control discipline as this document.

---

## 20. Immediate next document

The next required document is:

**02_bakeoff_protocol.md**

Its role is to define how the base model will be chosen between DAGMA and DCDI without convenience-driven reasoning and to freeze the operational details delegated by this contract.
