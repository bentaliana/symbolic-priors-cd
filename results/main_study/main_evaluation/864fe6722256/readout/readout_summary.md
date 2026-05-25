# Main-evaluation readout summary

## 1. Run identity

- `main_evaluation_run_hash12`: `864fe6722256`
- statistics input: `results/main_study/main_evaluation/864fe6722256/readout/main_evaluation_flat_records.csv`

## 2. Evidence files used

Inputs are upstream readout tables under `results/main_study/main_evaluation/864fe6722256/readout/`.

| input | rows |
| --- | --- |
| main_evaluation_flat_records.csv | 224 |
| baseline_comparison.csv | 16 |
| paired_seed_comparisons.csv | 20 |
| metric_correlations.csv | 20 |
| degradation_summary.csv | 24 |
| forbidden_edge_engagement_summary.csv | 32 |
| reference_forbidden_edge_comparison.csv | 28 |
| per_intervention_mmd_summary.csv | 640 |

## 3. Output files generated

### Figures

- `fig01_baseline_comparison_sid_shd_mmd`: `figures\fig01_baseline_comparison_sid_shd_mmd.png`
- `fig02_reference_forbidden_edge_suppression`: `figures\fig02_reference_forbidden_edge_suppression.png`
- `fig03_degradation_curves_sid`: `figures\fig03_degradation_curves_sid.png`
- `fig04_degradation_curves_mmd`: `figures\fig04_degradation_curves_mmd.png`
- `fig05_soft_frobenius_sid_heatmap`: `figures\fig05_soft_frobenius_sid_heatmap.png`
- `fig06_soft_frobenius_mmd_heatmap`: `figures\fig06_soft_frobenius_mmd_heatmap.png`
- `fig07_sid_vs_mmd_correlation`: `figures\fig07_sid_vs_mmd_correlation.png`
- `fig08_edge_count_and_engagement_diagnostic`: `figures\fig08_edge_count_and_engagement_diagnostic.png`

## 4. Key numerical descriptors

Tables only; thesis interpretation is separate.

### 4.1 Mean SID / SHD / MMD by baseline condition

| condition_label | n | mean_sid | mean_shd | mean_mmd |
| --- | --- | --- | --- | --- |
| prior_free | 7 | 66.14 | 24 | 0.1126 |
| matched_l1 | 7 | 67.57 | 24.14 | 0.1184 |
| soft_frobenius_clean_conf1 | 7 | 69.71 | 23.71 | 0.1137 |
| hard_exclusion_clean | 7 | 65 | 22.43 | 0.1022 |

### 4.2 Reference forbidden-edge engagement means by baseline condition

| condition_label | n_seeds | mean_abs_w_reference_forbidden_edges | mean_fraction_above_threshold |
| --- | --- | --- | --- |
| prior_free | 7 | 0.08883 | 0.1286 |
| matched_l1 | 7 | 0.07252 | 0.1143 |
| soft_frobenius_clean_conf1 | 7 | 0.0299 | 0.02857 |
| hard_exclusion_clean | 7 | 0 | 0 |

### 4.3 Selected overall correlation values (group_label = 'all')

| group_label | x_metric | y_metric | n | pearson | spearman | kendall_tau_b |
| --- | --- | --- | --- | --- | --- | --- |
| all | sid | mmd | 224 | 0.5006 | 0.5491 | 0.3681 |
| all | shd | mmd | 224 | 0.3398 | 0.3618 | 0.246 |
| all | edge_count_from_thresholded_adjacency | mmd | 224 | -0.391 | -0.2476 | -0.194 |

## 5. Caveats

- `matched_l1_lambda1 = 0.0625` is frozen via the matched-L1 calibration step.
- `lambda_prior = 0.0002` is frozen from earlier calibration.
- n = 7 evaluation seeds; the headline plan is paired by seed.
- Effect sizes and interval estimates are the primary evidence; p-values are secondary.
- No exploratory lambda_prior sensitivity is included in this readout.
- Any later M-10 sensitivity analysis is separate from the frozen primary result.
