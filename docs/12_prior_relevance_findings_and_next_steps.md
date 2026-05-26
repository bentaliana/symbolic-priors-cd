# Prior Relevance Findings and Next Steps

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

**Interpretation:** the soft prior was not inert; it exerted targeted structural pressure.

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
3. The corruption grid remains suggestive evidence that soft constraints may be less brittle than hard exclusions under prior corruption.
4. The prior relevance analysis explains why mechanism-level engagement did not become a clean metric-level improvement:
   - the prior targeted only a limited subset of false positives;
   - false negatives were more common than false positives;
   - forbidden-edge priors cannot correct missing true edges;
   - offline removal of the targeted edges produced small SHD gains but inconsistent SID changes.

A defensible thesis sentence is:

> The soft forbidden-edge prior successfully exerted targeted structural pressure, but the selected prior targets covered only a limited subset of prior-free false positives and could not address the larger false-negative component of the base learner’s error profile. This helps explain why mechanical prior engagement did not translate into a clear clean-prior SID/MMD improvement.

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

### Forbidden-edge oracle diagnostic

For each prior-free seed:

1. identify all false positives;
2. remove either all false positives or a budget-matched subset of `k=10` false positives;
3. recompute SID and SHD.

This asks:

> If a forbidden-edge prior had targeted the right absent edges, what direct SID/SHD improvement was available?

### Required-edge oracle diagnostic

For each prior-free seed:

1. identify all false negatives;
2. add either all false negatives or a budget-matched subset of `k=10` false negatives;
3. recompute SID and SHD.

This asks:

> If a required-edge prior had targeted missed true edges, would it have had more direct leverage than forbidden-edge priors?

This is especially relevant because the prior structural relevance analysis found false negatives to be more common than false positives.

---

## Why this next step is justified

The current analysis indicates that the prior’s limitation may be one of information alignment, not simply regularisation strength.

If oracle false-positive correction produces much larger SID/SHD improvements than the original forbidden-edge prior, then the issue is likely target selection within the forbidden-edge class.

If oracle false-negative correction produces much larger improvements, then required-edge priors may be a more relevant future direction.

If neither produces meaningful SID improvement, then the structural errors being corrected may not be the dominant drivers of SID in this benchmark, strengthening the structural/interventional non-equivalence discussion.

---

## Summary

The exploratory prior structural relevance analysis clarified why the primary result was mixed.

The soft prior successfully suppressed its intended targets, but those targets represented only a limited subset of the prior-free false positives and could not address false negatives, which were more frequent. Offline removal of the targeted forbidden edges produced small SHD improvements and mixed SID changes.

The next step is therefore not to immediately tune `lambda_prior`, but to ask whether more relevant prior targets or different prior classes would have had greater direct leverage over SID/SHD. This should be done first through an offline upper-bound diagnostic before any new model-fitting study is considered.