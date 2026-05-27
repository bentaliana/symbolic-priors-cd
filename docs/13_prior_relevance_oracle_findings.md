# 13. Prior Relevance, Oracle Diagnostics, and Thesis Implications

> **ADDENDUM (added 27/05/2026, post-implementation drift audit).** This document remains the **operative final interpretive document** for the exploratory phase. It is consistent with the implementation: the analyses recorded under hashes `1b46785b59a4` (M-10) and `1b95c563db88` (M-11) match the saved artefacts on disk. The "Stop condition" in Section 13 is in force and is formally recorded in docs/03 ("Decision: Exploratory diagnostic sequence complete"). A presentation-only visual readout of M-10 + M-11 has been produced at `results/main_study/exploratory/prior_relevance_diagnostics/` with `notebooks/prior_relevance_diagnostics.ipynb`. No content in the body of this document requires correction. The next step (thesis writing) is unchanged.

## Status

This document records the logical development from the primary main-evaluation result to the exploratory prior-relevance diagnostics, and defines the resulting interpretation for the thesis.

The analyses described here are explanatory and exploratory. They do not replace the frozen primary main-evaluation result, do not change any protocol constant, and do not retroactively redefine the main study.

Primary main-evaluation run:

```text
main_evaluation_run_hash12 = 864fe6722256

Prior structural relevance analysis:

analysis_hash12 = 1b46785b59a4

Oracle prior-relevance diagnostic:

analysis_hash12 = 1b95c563db88

All exploratory diagnostics used existing saved artefacts only. No model was refit, no new interventional samples were generated, no MMD counterfactuals were computed, and no protocol constants were changed.

1. Primary result: what the main evaluation showed

The primary main-evaluation result was mixed.

The clean soft-prior condition did not clearly improve SID/MMD over the prior-free or matched-L1 baselines. This meant that the primary claim that confidence-weighted soft forbidden-edge priors would improve clean-prior interventional generalisation was not supported in the frozen main evaluation.

However, the result did not indicate an implementation failure. The soft prior did engage mechanically: it suppressed the targeted forbidden-edge weights and reduced thresholded predictions at those positions relative to prior-free and matched-L1.

The primary result therefore created a diagnostic question:

Why did mechanical prior engagement not translate into clear metric-level improvement?

At this point, two possible explanations were especially important:

the prior strength may have been too weak;
the selected prior targets may not have aligned with the structural errors that were most relevant to SID/SHD/MMD.

The subsequent exploratory analyses investigated the second explanation.

2. Motivation for the prior structural relevance analysis

The soft prior was designed to penalise a set of forbidden edges. These edges were semantically valid in the sense that they were absent from the ground-truth graph. However, a semantically valid forbidden-edge prior is not necessarily useful for improving SID or MMD.

A forbidden-edge prior can only directly suppress false positives. It cannot recover false negatives, and it may not improve SID if the targeted false positives are not interventionally consequential.

The prior structural relevance analysis therefore asked:

Did the selected forbidden-edge prior targets overlap with the actual structural errors made by the prior-free DAGMA baseline?

This analysis was conducted offline using existing saved records and adjacency artefacts only.

3. Prior structural relevance analysis: method

For each evaluation seed, the analysis compared:

the prior-free thresholded graph;
the true graph;
the clean soft-prior forbidden-edge set;
the baseline conditions:
prior-free;
matched-L1;
soft Frobenius clean/confidence=1;
hard exclusion clean.

The analysis computed:

the fraction of clean reference forbidden edges predicted as edges by each condition;
prior-free false-positive and false-negative counts;
how many prior-free false positives were covered by the original forbidden-edge prior;
the offline SID/SHD effect of removing the targeted forbidden edges from the prior-free graph;
minimal topological descriptors of the targeted forbidden edges.

MMD was not recomputed because editing a thresholded adjacency matrix after training is not equivalent to retraining the model or resampling from the learned interventional mechanisms.

4. Prior structural relevance analysis: findings

The original soft forbidden-edge prior was not inert. It substantially reduced prediction at the targeted forbidden-edge positions.

The fraction of reference forbidden edges predicted as edges was:

Condition	Mean fraction of reference forbidden edges predicted
prior-free	0.1286
matched-L1	0.1143
soft Frobenius clean/confidence=1	0.0286
hard exclusion clean	0.0000

Thus, the soft prior exerted measurable targeted structural pressure.

However, the selected prior targets covered only a limited part of the prior-free error pattern. Across the seven prior-free fits, the average error profile was approximately:

Quantity	Mean
False positives	10.1
False negatives	13.9
Targeted false positives	1.3

The targeted false-positive fraction ranged from approximately 0.077 to 0.273 across seeds.

The offline removal effect was also limited:

Metric	Mean change after removing original reference forbidden edges
ΔSID	-0.5714
ΔSHD	-1.286

The SHD improvement was small and expected, because removing spurious targeted edges reduces structural error. The SID effect was mixed across seeds: some seeds improved, some worsened, and some were unchanged.

This analysis showed that the original forbidden-edge prior did act on its intended targets, but those targets represented only a limited subset of the prior-free false positives and did not reliably affect SID.

At this stage, the evidence suggested that prior relevance and target selection were plausible limiting factors. However, this interpretation required further clarification, because false negatives were numerically more frequent than false positives. The next diagnostic therefore asked whether false negatives or false positives carried greater SID/SHD leverage when corrected offline.

5. Motivation for the oracle prior-relevance diagnostic

The prior structural relevance analysis showed that the original prior targets had limited overlap with prior-free false positives. However, that alone did not answer a deeper question:

Was there substantial SID/SHD improvement available if the prior had targeted more relevant structural errors?

This motivated a final offline diagnostic comparing the original forbidden-edge prior targets against better-aligned structural counterfactuals.

The goal was not to implement a new prior method. The goal was to estimate, using existing graphs only, what improvement was available under different target-selection assumptions.

This diagnostic was deliberately bounded:

no model fitting;
no new sampling;
no MMD recomputation;
no lambda_prior tuning;
no new main study;
no implementation of required-edge priors.

It was defined as the final scheduled exploratory diagnostic before thesis writing.

6. Oracle prior-relevance diagnostic: method

The oracle prior-relevance diagnostic considered five offline structural scenarios for each prior-free seed.

6.1 Actual reference forbidden-edge removal

This reproduced the prior structural relevance analysis by removing the original reference forbidden-edge targets from the prior-free graph and recomputing SID/SHD.

This provided the direct offline effect of the actual prior targets.

6.2 Exact budget-matched false-positive removal

For each seed, all prior-free false positives were identified. The diagnostic then exhaustively searched all subsets of false positives of size up to k=10, matching the original prior budget.

The selected subset was the one giving the lowest SID after removal, with SHD and deterministic lexicographic order used as tie-breakers.

This asked:

What could a perfectly targeted 10-edge forbidden-edge prior have achieved on the same learned graph?

Because this search was exhaustive over the available false-positive subsets up to the fixed budget, it is an exact budget-matched false-positive diagnostic.

6.3 Full false-positive removal

All prior-free false positives were removed and SID/SHD were recomputed.

This is not guaranteed to be SID-optimal. It is a structural full-correction diagnostic showing the effect of removing all false-positive errors.

6.4 Greedy acyclicity-guarded false-negative addition

False negatives were identified and then added back to the prior-free graph using a greedy SID-primary procedure with an acyclicity guard. Candidate additions that would create cycles were skipped.

Two variants were computed:

a budget-matched version with k=10;
a full-candidate greedy version without the k=10 cap.

These required-edge diagnostics are not global optima. They are greedy acyclicity-guarded approximations intended to estimate whether required-edge style information had direct structural leverage.

7. Oracle prior-relevance diagnostic: findings

The diagnostic produced the following mean changes across seven evaluation seeds:

Scenario	Mean ΔSID	Mean ΔSHD	Mean selected edges	Mean skipped cycle edges
Actual reference forbidden-edge removal	-0.57	-1.29	10.00	0.00
Exact budget-matched false-positive removal	-24.86	-8.86	8.86	0.00
Full false-positive removal	-23.43	-10.14	10.14	0.00
Greedy budget-matched false-negative addition	-5.29	-2.86	2.86	11.00
Greedy full-candidate false-negative addition	-5.29	-2.86	2.86	11.00

These are mean values across the seven evaluation seeds. Per-seed variability is non-trivial. For example, the actual reference forbidden-edge removal produced ΔSID values ranging from -6 on seed 502 to +3 on seed 504. The per-seed results are recorded in the saved diagnostic CSVs under analysis hash 1b95c563db88.

The actual reference forbidden-edge removal reproduced the earlier prior relevance diagnostic exactly, confirming consistency between the two analyses.

The exact budget-matched false-positive diagnostic produced a much larger SID improvement than the original prior targets. Its mean ΔSID was approximately -24.86, compared with -0.57 for the original reference-target removal. This indicates that, under the same 10-edge budget, there was substantial structural/SID leverage available if the forbidden-edge targets had been selected from the most relevant false positives.

The full false-positive removal diagnostic produced a slightly smaller SID improvement than the exact budget-matched subset. This is not a contradiction. SID is not necessarily monotonic under edge removal; removing all false positives is not guaranteed to produce the best SID. This justifies the methodological caution against calling full false-positive removal a SID-optimal ceiling.

The required-edge diagnostics produced moderate improvements, but far smaller than the exact budget-matched false-positive diagnostic. They also selected only about 2.86 edges on average, while skipping about 11 candidate false-negative additions per seed because they would have created cycles.

This shows that required-edge repair is structurally constrained when performed as a post-hoc graph edit.

8. Reconciling the M-10 and M-11 findings

The prior structural relevance analysis showed that false negatives outnumbered false positives in the prior-free graphs. At first glance, this might suggest that forbidden-edge priors were aimed at the wrong error type.

The oracle diagnostic refined this interpretation. Although false negatives were more frequent in count, false-positive correction carried substantially more SID leverage on this benchmark. The exact budget-matched false-positive diagnostic produced a much larger mean SID improvement than the greedy false-negative addition diagnostics.

These two observations are complementary:

numerically, false negatives were more frequent;
structurally, false positives were more consequential for SID under the offline diagnostics;
the original forbidden-edge prior class was therefore not targeting an irrelevant error type in principle;
the limitation was that the randomly sampled forbidden-edge targets did not consistently align with the SID-consequential false-positive errors in the learned graphs.

This reconciles the two explanatory analyses. The issue was not simply that forbidden-edge priors target false positives. In this benchmark, false-positive correction had substantial potential. The issue was that the specific false positives targeted by the original prior were not the ones with the most SID/SHD leverage.

9. Combined interpretation

Together, the primary result, prior relevance analysis, and oracle diagnostic support the following interpretation.

The soft prior did not fail because the prior-loss mechanism was inert. It exerted measurable targeted pressure on the intended forbidden-edge positions.

However, the randomly sampled forbidden-edge targets did not consistently align with the SID-consequential false-positive errors in the learned graphs. The prior selected a small subset of prior-free false positives, and offline removal of those targets produced only small SHD improvements and inconsistent SID changes.

The oracle diagnostic then showed that the forbidden-edge prior class itself had much greater potential if target selection were improved. A perfectly targeted 10-edge false-positive removal diagnostic produced much larger SID/SHD improvements than the original reference forbidden-edge targets.

Therefore, the most defensible conclusion is:

The primary limitation in the tested setup was not simply prior strength, but prior target selection. The randomly sampled forbidden-edge prior targets did not sufficiently align with the SID-relevant false positives in the learned graphs.

This conclusion is narrower and more precise than claiming that semantic priors are ineffective. It instead shows that the usefulness of a symbolic prior depends critically on whether the prior targets the model’s actual error modes.

10. What this means for the thesis

The final thesis should separate three levels of evidence.

10.1 Primary result

The frozen main evaluation remains the primary result.

The clean soft-prior condition did not clearly improve SID/MMD over prior-free or matched-L1.

10.2 Mechanistic result

The soft prior mechanism worked as designed. It suppressed the targeted forbidden-edge positions relative to prior-free and matched-L1.

10.3 Explanatory diagnostics

The exploratory diagnostics explain why the mechanism did not translate into metric-level improvement:

the original forbidden-edge targets covered only a limited subset of prior-free false positives;
the original targets had small and inconsistent SID effects when removed offline;
exact budget-matched false-positive target selection would have produced much larger SID/SHD improvements;
required-edge correction was possible only for a small number of false negatives because many candidate additions violated acyclicity.

This gives the thesis a stronger conclusion:

Prior knowledge is not useful merely because it is correct. It must also be relevant to the base learner’s error modes and to the evaluation metric.

11. What this does not show

These analyses do not show that:

semantic priors are ineffective in general;
soft priors cannot improve causal discovery;
the frozen primary experiment was invalid;
lambda_prior = 2e-4 was necessarily optimal or suboptimal;
an implemented better-targeted prior would necessarily reproduce the offline diagnostic gains;
required-edge priors would necessarily underperform forbidden-edge priors if imposed during training;
MMD would improve under the same structural edits.

The oracle diagnostics are offline structural counterfactuals. They estimate available SID/SHD leverage under graph edits; they are not trained models and they do not model interventional sample distributions.

12. Future work implications

The most important future-work direction is not simply to tune lambda_prior.

The results suggest that future work should investigate prior elicitation and prior target selection.

Promising directions include:

developing methods for selecting forbidden-edge priors that target high-impact false-positive patterns;
studying how domain experts can provide priors that are not only correct, but also interventionally relevant;
exploring methods that identify which edges are likely to be SID-consequential before prior specification, either through uncertainty quantification or through pilot model training;
evaluating required-edge priors during training, rather than only as post-hoc graph edits;
comparing forbidden-edge, required-edge, and topological-order priors under matched prior budgets;
testing whether target-selection findings generalise beyond the d=10 ER2 linear-Gaussian DAGMA setting.

These are future-work directions, not additional experiments within the current project.

13. Stop condition

The exploratory diagnostic sequence is complete.

The project should not continue into additional implementation experiments unless a genuine implementation bug is discovered that affects the existing evidence.

The next step is thesis writing.

The thesis Results chapter should report the frozen primary result. The Discussion chapter should integrate the prior relevance and oracle diagnostics to explain why the mechanism did not translate into metric improvement. The Future Work section should frame better-aligned prior target selection as the principal direction for follow-up research, distinct from the orthogonal direction of stronger regularisation pressure.