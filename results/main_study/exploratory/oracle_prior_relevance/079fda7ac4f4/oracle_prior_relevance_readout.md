# Oracle prior relevance: exploratory diagnostic

## Run identity

- `main_evaluation_run_hash12`: `166c792c43bc`
- `analysis_hash12`: `079fda7ac4f4`
- prior structural relevance analysis: `6f660aaeef3d`
- analysis protocol version: `oracle_prior_relevance_v1`
- output directory: `results/main_study/exploratory/oracle_prior_relevance/079fda7ac4f4`

Existing saved artefacts only were used. No new model fitting, no MMD recomputation, and no new interventional sampling were performed. This is an offline structural diagnostic. This is the final scheduled exploratory diagnostic before thesis writing.

## Evidence files used

- 7 prior_free records loaded from `results/main_study/166c792c43bc/records/`.
- 7 clean-soft reference records (soft_frobenius, corruption=0.0, confidence=1.0) loaded from the same directory.
- For each prior_free record, the persisted `thresholded_adjacency.npz` and `true_adjacency.npz` artefacts were read; the continuous-W artefact is not used by this diagnostic.

## Diagnostic scenarios

Five per-seed scenarios are computed for each of the seven evaluation seeds:
- `actual_reference_forbidden_removal`: remove the seed-specific clean reference forbidden-edge set.
- `fp_remove_budget10_exact`: exact exhaustive subset search over prior-free false positives, up to `budget_k = 10` removed edges. SID-primary selection with deterministic tie-breaks.
- `fp_remove_all_false_positives`: remove every prior-free false positive. Structural full-correction diagnostic; not claimed as a guaranteed SID ceiling.
- `fn_add_budget10_greedy_acyclic`: greedy SID-primary addition of up to `budget_k = 10` prior-free false negatives, guarded by acyclicity. Greedy approximation, not a global optimum.
- `fn_add_full_greedy_acyclic`: same greedy procedure without the `budget_k` cap. Continues until no beneficial valid addition remains.

## Budget convention

`budget_k = 10` matches the original forbidden-edge prior budget (10 edges per seed in the main evaluation). The comparison is "what could a 10-edge prior budget have achieved across different prior classes?". `budget_k = 10` is not claimed to be optimal for any required-edge prior.

## Aggregate summary (mean / median / min / max) by scenario

| scenario | n | mean dSID | median dSID | min dSID | max dSID | mean dSHD | median dSHD | min dSHD | max dSHD | mean n_cand | mean n_selected | mean n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `actual_reference_forbidden_removal` | 7 | -0.8571 | 0 | -6 | 0 | -0.7143 | -1 | -2 | 0 | 10 | 10 | 0 |
| `fp_remove_budget10_exact` | 7 | -22.71 | -23 | -33 | -13 | -6.714 | -7 | -9 | -4 | 6.857 | 6.714 | 0 |
| `fp_remove_all_false_positives` | 7 | -22.57 | -23 | -33 | -13 | -6.857 | -8 | -9 | -4 | 6.857 | 6.857 | 0 |
| `fn_add_budget10_greedy_acyclic` | 7 | -6.143 | -4 | -15 | -2 | -3.429 | -2 | -7 | -2 | 14 | 3.429 | 10.57 |
| `fn_add_full_greedy_acyclic` | 7 | -6.143 | -4 | -15 | -2 | -3.429 | -2 | -7 | -2 | 14 | 3.429 | 10.57 |

## Actual reference-forbidden removal

Removes the seed-specific clean reference forbidden-edge set from the prior-free thresholded adjacency and recomputes SID and SHD. Reproduces the prior structural relevance offline-removal diagnostic; deltas should match that earlier output within numerical precision.

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 78 | 0 | 28 | 26 | -2 | 10 | 10 | 0 |
| 502 | 50 | 44 | -6 | 15 | 14 | -1 | 10 | 10 | 0 |
| 503 | 79 | 79 | 0 | 23 | 23 | 0 | 10 | 10 | 0 |
| 504 | 44 | 44 | 0 | 13 | 13 | 0 | 10 | 10 | 0 |
| 505 | 79 | 79 | 0 | 22 | 22 | 0 | 10 | 10 | 0 |
| 506 | 74 | 74 | 0 | 21 | 20 | -1 | 10 | 10 | 0 |
| 507 | 56 | 56 | 0 | 24 | 23 | -1 | 10 | 10 | 0 |

## Exact budget-matched false-positive diagnostic

Exhaustive subset search over prior-free false positives up to `budget_k = 10`. Selection rule is SID-primary with deterministic tie-breaks. The empty subset is included; the selected result cannot be worse than the original under the selection rule.

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 52 | -26 | 28 | 19 | -9 | 9 | 9 | 0 |
| 502 | 50 | 37 | -13 | 15 | 11 | -4 | 4 | 4 | 0 |
| 503 | 79 | 50 | -29 | 23 | 14 | -9 | 9 | 9 | 0 |
| 504 | 44 | 25 | -19 | 13 | 8 | -5 | 5 | 5 | 0 |
| 505 | 79 | 46 | -33 | 22 | 14 | -8 | 8 | 8 | 0 |
| 506 | 74 | 51 | -23 | 21 | 16 | -5 | 5 | 5 | 0 |
| 507 | 56 | 40 | -16 | 24 | 17 | -7 | 8 | 7 | 0 |

## Full false-positive removal diagnostic

Removes every prior-free false positive and recomputes SID and SHD. This is a structural full-correction diagnostic; it is not a guaranteed SID ceiling.

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 52 | -26 | 28 | 19 | -9 | 9 | 9 | 0 |
| 502 | 50 | 37 | -13 | 15 | 11 | -4 | 4 | 4 | 0 |
| 503 | 79 | 50 | -29 | 23 | 14 | -9 | 9 | 9 | 0 |
| 504 | 44 | 25 | -19 | 13 | 8 | -5 | 5 | 5 | 0 |
| 505 | 79 | 46 | -33 | 22 | 14 | -8 | 8 | 8 | 0 |
| 506 | 74 | 51 | -23 | 21 | 16 | -5 | 5 | 5 | 0 |
| 507 | 56 | 41 | -15 | 24 | 16 | -8 | 8 | 8 | 0 |

## Greedy acyclicity-guarded false-negative diagnostic

Budget-matched variant with `budget_k = 10`:

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 76 | -2 | 28 | 26 | -2 | 19 | 2 | 17 |
| 502 | 50 | 36 | -14 | 15 | 9 | -6 | 11 | 6 | 5 |
| 503 | 79 | 75 | -4 | 23 | 20 | -3 | 14 | 3 | 11 |
| 504 | 44 | 40 | -4 | 13 | 11 | -2 | 8 | 2 | 6 |
| 505 | 79 | 77 | -2 | 22 | 20 | -2 | 14 | 2 | 12 |
| 506 | 74 | 59 | -15 | 21 | 14 | -7 | 16 | 7 | 9 |
| 507 | 56 | 54 | -2 | 24 | 22 | -2 | 16 | 2 | 14 |

Full-candidate variant (`budget_k = None`):

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 76 | -2 | 28 | 26 | -2 | 19 | 2 | 17 |
| 502 | 50 | 36 | -14 | 15 | 9 | -6 | 11 | 6 | 5 |
| 503 | 79 | 75 | -4 | 23 | 20 | -3 | 14 | 3 | 11 |
| 504 | 44 | 40 | -4 | 13 | 11 | -2 | 8 | 2 | 6 |
| 505 | 79 | 77 | -2 | 22 | 20 | -2 | 14 | 2 | 12 |
| 506 | 74 | 59 | -15 | 21 | 14 | -7 | 16 | 7 | 9 |
| 507 | 56 | 54 | -2 | 24 | 22 | -2 | 16 | 2 | 14 |

These are greedy diagnostic approximations. Subset interactions and acyclicity constraints mean the result is not a guaranteed global optimum.

## Acyclicity guard summary

Per-seed cycle-skip counts encountered during the greedy false-negative additions:

| seed | budget-k skipped | full-candidate skipped |
| --- | --- | --- |
| 501 | 17 | 17 |
| 502 | 5 | 5 |
| 503 | 11 | 11 |
| 504 | 6 | 6 |
| 505 | 12 | 12 |
| 506 | 9 | 9 |
| 507 | 14 | 14 |

## Comparison to original prior-target removal

The `actual_reference_forbidden_removal` rows reproduce the prior structural relevance offline removal output. Any byte-for-byte agreement on SID and SHD deltas is a consistency check, not new evidence.

## Limitations

- The exact false-positive budget diagnostic is exhaustive over FP subsets up to `budget_k = 10`; it does not model the optimisation-side relationship between forbidden-edge prior strength and DAGMA's learned graph.
- Removing all false positives is a full structural correction; it is not a guaranteed SID ceiling for any prior class.
- The false-negative diagnostics are greedy acyclicity-guarded approximations. They are not global optima.
- MMD counterfactuals are explicitly out of scope; saved MMD values are not modified.

## Implication for thesis discussion

The five scenarios characterise the maximum direct structural-metric improvement available under different offline prior-class proxies at the same 10-edge budget used in the frozen main evaluation. They are descriptive diagnostics intended to support cautious thesis discussion; they do not constitute a new headline comparison and do not replace the frozen primary result.

## Stop condition

This is the final scheduled exploratory diagnostic before thesis writing. Any idea emerging from this analysis (required-edge prior implementation, lambda_prior tuning, new main study) is recorded as future work rather than implemented within the current project timeline.

- oracle summary plot: generated at `oracle_summary_plot.png`.

