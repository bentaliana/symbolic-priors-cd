# Prior Relevance Findings and Next Steps

> **ADDENDUM 2 (added 28/05/2026, DAGMA backbone correction).** The analyses described in this document were originally produced on top of the M-8 main evaluation hash `864fe6722256`, which was generated on an off-protocol DAGMA backbone (`lambda1 = 0.05`, `warm_iter = 30000`, `max_iter = 60000` — the wrapper-level Phase-A defaults, not the protocol values `0.10 / 20000 / 70000`). The defect, its correction, and the new authoritative hashes are formally recorded in docs/03 ("28/05/2026 — DAGMA backbone default-leak defect ..."). The current operative analysis hashes are:
>
> - **M-10 prior structural relevance:** `6f660aaeef3d` (output directory `results/main_study/exploratory/prior_structural_relevance/6f660aaeef3d/`).
> - **M-11 oracle prior relevance:** `079fda7ac4f4` (output directory `results/main_study/exploratory/oracle_prior_relevance/079fda7ac4f4/`).
> - **Underlying M-8 main evaluation:** `166c792c43bc`.
>
> The qualitative findings recorded below — that mechanism-level forbidden-edge suppression engages but that the originally-sampled prior targets are not the dominant source of DAGMA's errors at the protocol operating point — are robust under the corrected backbone. The new artefacts should be read in place of the superseded hashes wherever they appear in the body and in Addendum 1.
>
> **ADDENDUM (added 27/05/2026, post-implementation drift audit).** The "next planned diagnostic step" described in Section "Next step" of this document — the alternative-prior relevance upper-bound diagnostic (forbidden-edge oracle + required-edge oracle) — has been completed.
>
> - **M-11 analysis hash:** `1b95c563db88`
> - **M-11 output directory:** `results/main_study/exploratory/oracle_prior_relevance/1b95c563db88/`
> - **M-11 readout:** [docs/13_prior_relevance_oracle_findings.md](docs/13_prior_relevance_oracle_findings.md)
> - **Key M-11 finding:** under the same k = 10 prior budget, exact false-positive removal yielded mean dSID ≈ −24.86 vs −0.57 for actual reference removal; greedy acyclicity-guarded false-negative addition yielded mean dSID ≈ −5.29 with only ≈ 2.86 selected per seed (≈ 11.0 skipped due to acyclicity). The forbidden-edge prior class had substantial structural leverage available; the original randomly-sampled targets did not consistently exploit it.
> - **Combined interpretation:** prior target selection, not prior strength, was the dominant limit in the tested setup.
> - **Stop condition:** the exploratory diagnostic sequence is now closed (per docs/03 "Decision: Exploratory diagnostic sequence complete"). No further investigations are scheduled within the project timeline.
>
> The body of this document is preserved as the M-10 readout and the pre-registration of the (now-completed) M-11 diagnostic.

## Status

This note records the outcome of the exploratory prior structural relevance analysis and defines the next planned diagnostic step.

This analysis is explanatory and exploratory. It does not replace the frozen primary main-evaluation result, does not change any protocol constant, and does not reinterpret the primary study as a different experiment.

**Primary main-evaluation run:** `864fe6722256`  
**Prior structural relevance analysis:** `1b46785b59a4`  
**Output directory:** `results/main_study/exploratory/prior_structural_relevance/1b46785b59a4/`

The analysis used existing saved artefacts only. No model was refit, no new interventional samples were generated, and no MMD counterfactuals were computed.

---

## Why this analysis was conducted

The primary M-8/M-9 result was mixed. The soft Frobenius prior behaved as intended at the mechanism level: it reduced the learned weights and thresholded predictions at the targeted forbidden-edge positions. However, this mechanical suppression did not translate into a clear improvement in the primary clean-prior SID/MMD comparison against prior-free or matched-L1 baselines.

This created a diagnostic question:

> Was the soft prior failing because the penalty strength was too weak, or because the selected forbidden-edge prior targeted positions that were not the dominant source of DAGMA’s errors?

The prior structural relevance analysis investigated the second possibility using existing outputs only.

---

## Main findings

### 1. The prior mechanism engaged

The fraction of reference forbidden edges predicted as edges was:

| Condition | Mean fraction of reference edges predicted |
|---|---:|
| prior-free | 0.1286 |
| matched-L1 | 0.1143 |
| soft Frobenius clean/confidence=1 | 0.0286 |
| hard exclusion clean | 0.0000 |

This confirms that the soft Frobenius prior substantially reduced predictions at the targeted forbidden-edge positions relative to prior-free and matched-L1, while hard exclusion removed them completely.

**Interpretation:** the soft prior was not inert; it exerted measurable targeted structural pressure.

---

### 2. The prior targeted only a limited subset of prior-free false positives

Across the seven prior-free fits, the average structural error profile was approximately:

| Quantity | Mean |
|---|---:|
| False positives | 10.1 |
| False negatives | 13.9 |
| Targeted false positives | 1.3 |

Per seed, the fraction of false positives covered by the prior ranged from approximately `0.077` to `0.273`.

This means the prior was not irrelevant, but it covered only a limited subset of the false-positive error space. It also could not address false negatives by construction, because forbidden-edge priors only suppress absent edges; they cannot add missing true edges.

---

### 3. Offline removal produced small SHD improvement but mixed SID changes

The analysis removed the seed-specific reference forbidden edges from the prior-free thresholded adjacency and recomputed SID and SHD.

| Metric | Mean change after removal |
|---|---:|
| ΔSID | -0.5714 |
| ΔSHD | -1.286 |

The SHD effect was consistently small and favourable, because removing spurious targeted edges reduces structural error. The SID effect was mixed: some seeds improved, some worsened, and some were unchanged.

**Interpretation:** even perfect removal of the selected forbidden-edge targets would not reliably improve the primary interventional metric.

---

## Interpretation for the thesis

The primary thesis result should now be framed as follows:

1. The clean soft-prior condition did not clearly improve SID/MMD over prior-free or matched-L1 in the frozen primary evaluation.
2. The soft prior did mechanically suppress the intended forbidden-edge targets.
3. The corruption grid shows soft constraints degrading more slowly than hard exclusions on MMD, with mean slopes around `0.019` versus `0.027` at confidence `1.0`, though substantial seed-level variability limits the strength of this comparative observation.
4. The prior relevance analysis explains why mechanism-level engagement did not become a clean metric-level improvement:
   - the prior targeted only a limited subset of false positives;
   - false negatives were more common than false positives;
   - forbidden-edge priors cannot correct missing true edges;
   - offline removal of the targeted edges produced small SHD gains but inconsistent SID changes.

A defensible thesis sentence is:

> The soft forbidden-edge prior exerted measurable targeted structural pressure on the predicted graph, reducing predicted edges at forbidden-edge positions from approximately 12.9% to 2.9%, but the selected prior targets covered only a limited subset of prior-free false positives, approximately 13% on average, and could not address the dominant false-negative component of the base learner’s error profile on this benchmark. This helps explain why mechanical prior engagement did not translate into a clear clean-prior SID/MMD improvement.

---

## What this analysis does not show

This analysis does not show that:

- semantic priors are ineffective in general;
- soft priors cannot improve differentiable causal discovery;
- `lambda_prior = 2e-4` was necessarily optimal or suboptimal;
- required-edge priors would definitely improve the result;
- topological-order priors would definitely improve the result;
- the primary experiment was invalid.

It shows only that, for this frozen main-evaluation run and this forbidden-edge prior construction, the selected prior targets had limited overlap with the dominant prior-free error modes.

---

## Next step: alternative-prior relevance upper-bound diagnostic

The next step should remain offline and exploratory.

The goal is to ask whether more relevant prior targets or different prior classes would have had greater direct leverage over SID/SHD.

### Forbidden-edge oracle diagnostic: two-tier comparison

For each prior-free seed:

#### Class ceiling

Identify all false positives, remove all of them, and recompute SID and SHD.

This gives the absolute upper bound on what any forbidden-edge prior could achieve by correcting all false-positive errors in the learned thresholded graph.

#### Budget-matched ceiling

Identify the 10 false positives whose removal produces the largest SID improvement, or all false positives if fewer than 10 exist, and recompute SID and SHD.

This gives the upper bound for a `k=10` forbidden-edge prior with perfect target selection.

The gap between the actual prior result and the budget-matched ceiling indicates how much may have been lost to suboptimal target selection. The gap between the budget-matched ceiling and the class ceiling indicates the cost of the `k=10` prior budget itself.

This diagnostic asks:

> If a forbidden-edge prior had targeted the right absent edges, what direct SID/SHD improvement was available?

---

### Required-edge oracle diagnostic: two-tier comparison with acyclicity guard

For each prior-free seed:

#### Class ceiling

Identify all false negatives. Add each missing true edge in turn, checking that the resulting graph remains acyclic before each addition. Skip any edge whose addition would create a cycle, and report the skipped count. Recompute SID and SHD on the resulting DAG.

This gives an upper-bound estimate for what a required-edge prior could achieve while respecting the DAG constraint.

#### Budget-matched ceiling

Identify the 10 false negatives whose addition, in greedy order and respecting acyclicity, produces the largest SID improvement. Recompute SID and SHD.

This gives the upper bound for a `k=10` required-edge prior with perfect target selection under the acyclicity constraint.

If a substantial fraction of false negatives create cycles when added, the required-edge prior class has its own structural limits. That would itself be informative about the difficulty of repairing DAGMA’s learned graphs after training.

This diagnostic asks:

> If a required-edge prior had targeted missed true edges, would it have had more direct leverage than forbidden-edge priors?

This is especially relevant because the prior structural relevance analysis found false negatives to be more common than false positives.

---

## Why this next step is justified

The current analysis indicates that the prior’s limitation may be one of information alignment, not simply regularisation strength.

If the budget-matched false-positive oracle produces much larger SID/SHD improvements than the original forbidden-edge prior, then the issue is likely target selection within the forbidden-edge class.

If the required-edge oracle produces much larger improvements, then required-edge priors may be a more relevant future direction.

If neither oracle produces meaningful SID improvement, then the structural errors being corrected may not be the dominant drivers of SID in this benchmark, strengthening the structural/interventional non-equivalence discussion.

The diagnostic should report three levels clearly:

| Diagnostic | What it tells us |
|---|---|
| Actual prior result | What the frozen forbidden-edge prior produced |
| Budget-matched oracle | What a perfectly targeted prior with the same `k=10` budget could produce |
| Class-ceiling oracle | What the full prior class could achieve if all relevant errors of that type were corrected |

---

## Reporting discipline

The next diagnostic must be reported as exploratory and offline.

It must not:

- replace the primary M-8/M-9 result;
- claim that oracle priors are realistic domain knowledge;
- recompute MMD from edited graphs;
- imply that an implemented prior-loss method would necessarily achieve the oracle effect;
- tune `lambda_prior`;
- present a new headline result.

It may support future-work claims such as:

- required-edge priors may better target false-negative-dominated error modes;
- prior usefulness depends on alignment with model error structure;
- randomly sampled true-negative forbidden-edge priors are not guaranteed to be interventionally useful;
- future work should compare prior classes under matched prior budgets.

---

## Summary

The exploratory prior structural relevance analysis clarified why the primary result was mixed.

The soft prior successfully suppressed its intended targets, but those targets represented only a limited subset of the prior-free false positives and could not address false negatives, which were more frequent. Offline removal of the targeted forbidden edges produced small SHD improvements and mixed SID changes.

The next step is therefore not to immediately tune `lambda_prior`, but to ask whether more relevant prior targets or different prior classes would have had greater direct leverage over SID/SHD. This should be done first through an offline upper-bound diagnostic before any new model-fitting study is considered.