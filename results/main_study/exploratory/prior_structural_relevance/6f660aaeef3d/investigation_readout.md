# Prior structural relevance: exploratory analysis

## Run identity

- `main_evaluation_run_hash12`: `166c792c43bc`
- `analysis_hash12`: `6f660aaeef3d`
- analysis protocol version: `prior_structural_relevance_v1`
- output directory: `results/main_study/exploratory/prior_structural_relevance/6f660aaeef3d`

This analysis is exploratory. Existing saved artefacts only were used. No new model fitting, no MMD recomputation, and no new interventional sampling were performed. This analysis does not replace the frozen primary result.

## Evidence files used

- 28 records loaded from `results/main_study/166c792c43bc/records/`: the 4 baseline conditions x 7 evaluation seeds.
- For each record, the persisted `thresholded_adjacency.npz`, `continuous_w.npz`, and `true_adjacency.npz` artefacts were read.

## Prior-target overlap summary

Per-condition mean fraction of the seed-specific clean-soft reference forbidden-edge set that the condition predicts as edges. Lower values mean the condition suppresses the reference forbidden edges more strongly.

| condition | mean fraction of reference edges predicted | n seeds |
| --- | --- | --- |
| `prior_free` | 0.07143 | 7 |
| `matched_l1` | 0.07143 | 7 |
| `soft_frobenius_clean_conf1` | 0.01429 | 7 |
| `hard_exclusion_clean` | 0 | 7 |

## Prior-free error decomposition summary

Off-diagonal TP / FP / FN counts for the prior-free baseline per seed, with targeted-false-positive counts and the primary relevance quantity `targeted_false_positive_fraction_of_fp`. SID, SHD, and MMD are read from the saved records.

| seed | n_true_edges | n_predicted | TP | FP | FN | total_error | targeted_FP | targeted_FP / FP | SID | SHD | MMD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 21 | 11 | 2 | 9 | 19 | 28 | 2 | 0.2222 | 78 | 28 | 0.1662 |
| 502 | 19 | 12 | 8 | 4 | 11 | 15 | 1 | 0.25 | 50 | 15 | 0.1129 |
| 503 | 18 | 13 | 4 | 9 | 14 | 23 | 0 | 0 | 79 | 23 | 0.1022 |
| 504 | 12 | 9 | 4 | 5 | 8 | 13 | 0 | 0 | 44 | 13 | 0.09488 |
| 505 | 17 | 11 | 3 | 8 | 14 | 22 | 0 | 0 | 79 | 22 | 0.1156 |
| 506 | 20 | 9 | 4 | 5 | 16 | 21 | 1 | 0.2 | 74 | 21 | 0.1378 |
| 507 | 22 | 14 | 6 | 8 | 16 | 24 | 1 | 0.125 | 56 | 24 | 0.1162 |

## Offline SID/SHD removal summary

For each seed, the prior-free thresholded adjacency was edited offline by zeroing the seed-specific reference forbidden-edge positions, and SID and SHD were recomputed with the project's public metric functions. MMD is not recomputed; the column is intentionally omitted.

| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | dSHD | n_ref_edges_predicted_before | n_removed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 78 | 78 | 0 | 28 | 26 | -2 | 2 | 2 |
| 502 | 50 | 44 | -6 | 15 | 14 | -1 | 1 | 1 |
| 503 | 79 | 79 | 0 | 23 | 23 | 0 | 0 | 0 |
| 504 | 44 | 44 | 0 | 13 | 13 | 0 | 0 | 0 |
| 505 | 79 | 79 | 0 | 22 | 22 | 0 | 0 | 0 |
| 506 | 74 | 74 | 0 | 21 | 20 | -1 | 1 | 1 |
| 507 | 56 | 56 | 0 | 24 | 23 | -1 | 1 | 1 |

- Mean dSID across seeds: -0.8571 (after - original).
- Mean dSHD across seeds: -0.7143 (after - original).

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

