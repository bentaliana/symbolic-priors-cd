# Prior structural relevance: exploratory analysis

## Run identity

- `main_evaluation_run_hash12`: `864fe6722256`
- `analysis_hash12`: `1b46785b59a4`
- analysis protocol version: `prior_structural_relevance_v1`
- output directory: `results/main_study/exploratory/prior_structural_relevance/1b46785b59a4`

This analysis is exploratory. Existing saved artefacts only were used. No new model fitting, no MMD recomputation, and no new interventional sampling were performed. This analysis does not replace the frozen primary result.

## Evidence files used

- 28 records loaded from `results/main_study/864fe6722256/records/`: the 4 baseline conditions x 7 evaluation seeds.
- For each record, the persisted `thresholded_adjacency.npz`, `continuous_w.npz`, and `true_adjacency.npz` artefacts were read.

## Prior-target overlap summary

Per-condition mean fraction of the seed-specific clean-soft reference forbidden-edge set that the condition predicts as edges. Lower values mean the condition suppresses the reference forbidden edges more strongly.

| condition | mean fraction of reference edges predicted | n seeds |
| --- | --- | --- |
| `prior_free` | 0.1286 | 7 |
| `matched_l1` | 0.1143 | 7 |
| `soft_frobenius_clean_conf1` | 0.02857 | 7 |
| `hard_exclusion_clean` | 0 | 7 |

## Prior-free error decomposition summary

Off-diagonal TP / FP / FN counts for the prior-free baseline per seed, with targeted-false-positive counts and the primary relevance quantity `targeted_false_positive_fraction_of_fp`. SID, SHD, and MMD are read from the saved records.

| seed | n_true_edges | n_predicted | TP | FP | FN | total_error | targeted_FP | targeted_FP / FP | SID | SHD | MMD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 21 | 13 | 2 | 11 | 19 | 30 | 3 | 0.2727 | 77 | 30 | 0.1639 |
| 502 | 19 | 12 | 8 | 4 | 11 | 15 | 1 | 0.25 | 50 | 15 | 0.1016 |
| 503 | 18 | 17 | 4 | 13 | 14 | 27 | 1 | 0.07692 | 66 | 27 | 0.08279 |
| 504 | 12 | 13 | 4 | 9 | 8 | 17 | 1 | 0.1111 | 49 | 17 | 0.09419 |
| 505 | 17 | 16 | 3 | 13 | 14 | 27 | 1 | 0.07692 | 80 | 27 | 0.1109 |
| 506 | 20 | 14 | 5 | 9 | 15 | 24 | 1 | 0.1111 | 75 | 24 | 0.1316 |
| 507 | 22 | 18 | 6 | 12 | 16 | 28 | 1 | 0.08333 | 66 | 28 | 0.1028 |

## Offline SID/SHD removal summary

For each seed, the prior-free thresholded adjacency was edited offline by zeroing the seed-specific reference forbidden-edge positions, and SID and SHD were recomputed with the project's public metric functions. MMD is not recomputed; the column is intentionally omitted.

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_ref_edges_predicted_before | n_removed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 77 | 78 | 1 | 30 | 27 | -3 | 3 | 3 |
| 502 | 50 | 44 | -6 | 15 | 14 | -1 | 1 | 1 |
| 503 | 66 | 67 | 1 | 27 | 26 | -1 | 1 | 1 |
| 504 | 49 | 52 | 3 | 17 | 16 | -1 | 1 | 1 |
| 505 | 80 | 77 | -3 | 27 | 26 | -1 | 1 | 1 |
| 506 | 75 | 75 | 0 | 24 | 23 | -1 | 1 | 1 |
| 507 | 66 | 66 | 0 | 28 | 27 | -1 | 1 | 1 |

- Mean dSID across seeds: -0.5714 (after - original).
- Mean dSHD across seeds: -1.286 (after - original).

## Minimal topological relevance summary

Per reference forbidden edge `(source, target)`, descriptive topological properties over the true DAG: target descendant count, source ancestor count, and target/source in- and out-degrees. Path-length analysis, centrality measures, and intervention-effect computations are intentionally out of scope.
- Mean target descendant count across all reference edges: 3.257.
- Mean target in-degree across all reference edges: 1.371.
- Mean target out-degree across all reference edges: 2.043.

## Limitations

- This analysis is offline and exploratory; it cannot substitute for a pre-registered statistical test.
- Offline SID/SHD recomputation on edited adjacency matrices is a structural counterfactual; it does not estimate the downstream interventional-distribution effect.
- MMD counterfactuals are explicitly out of scope; the saved MMD values are read as-is.
- Coverage bands and topological summaries are heuristic diagnostic aids, not statistical thresholds.

## Implication for possible lambda_prior sensitivity

If the offline removal effect on SID and SHD is small in magnitude across seeds, then perfect targeted suppression of the reference forbidden edges would have produced only a small direct improvement on these structural metrics. A future sensitivity study at varied `lambda_prior` could examine indirect optimisation effects; such a study is out of scope here.

- aggregated error heatmap: generated at `aggregated_error_heatmap.png`.

