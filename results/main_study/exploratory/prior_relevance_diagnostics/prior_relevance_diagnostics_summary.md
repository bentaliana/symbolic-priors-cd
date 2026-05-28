# Prior-relevance diagnostics: visual readout

## Run identity

- main_evaluation_run_hash12: `166c792c43bc`
- prior_relevance_analysis_hash12: `6f660aaeef3d`
- oracle_analysis_hash12: `079fda7ac4f4`

No model fits, no metric recomputation, no protocol change. Every figure is built from a single persisted artefact.

## Inputs read

| input | exists | path |
| --- | --- | --- |
| `aggregated_error_heatmap_png` | True | `results\main_study\exploratory\prior_structural_relevance\6f660aaeef3d\aggregated_error_heatmap.png` |
| `baseline_comparison_csv` | True | `results\main_study\main_evaluation\166c792c43bc\readout\baseline_comparison.csv` |
| `degradation_summary_csv` | True | `results\main_study\main_evaluation\166c792c43bc\readout\degradation_summary.csv` |
| `forbidden_engagement_summary_csv` | True | `results\main_study\main_evaluation\166c792c43bc\readout\forbidden_edge_engagement_summary.csv` |
| `offline_removal_effect_csv` | True | `results\main_study\exploratory\prior_structural_relevance\6f660aaeef3d\offline_forbidden_edge_removal_effect.csv` |
| `oracle_manifest_json` | True | `results\main_study\exploratory\oracle_prior_relevance\079fda7ac4f4\oracle_prior_relevance_manifest.json` |
| `oracle_per_seed_csv` | True | `results\main_study\exploratory\oracle_prior_relevance\079fda7ac4f4\oracle_diagnostics_per_seed.csv` |
| `oracle_summary_csv` | True | `results\main_study\exploratory\oracle_prior_relevance\079fda7ac4f4\oracle_diagnostics_summary.csv` |
| `prior_free_error_decomposition_csv` | True | `results\main_study\exploratory\prior_structural_relevance\6f660aaeef3d\prior_free_error_decomposition.csv` |
| `prior_relevance_manifest_json` | True | `results\main_study\exploratory\prior_structural_relevance\6f660aaeef3d\investigation_manifest.json` |
| `prior_target_overlap_csv` | True | `results\main_study\exploratory\prior_structural_relevance\6f660aaeef3d\prior_target_overlap.csv` |
| `reference_forbidden_comparison_csv` | True | `results\main_study\main_evaluation\166c792c43bc\readout\reference_forbidden_edge_comparison.csv` |

## Figures generated

- `fig01_main_result_clean_metrics.png` -> `figures/fig01_main_result_clean_metrics.png`
- `fig02_mechanism_engagement.png` -> `figures/fig02_mechanism_engagement.png`
- `fig03_corruption_degradation.png` -> `figures/fig03_corruption_degradation.png`
- `fig04_error_decomposition.png` -> `figures/fig04_error_decomposition.png`
- `fig05_prior_target_overlap.png` -> `figures/fig05_prior_target_overlap.png`
- `fig06_offline_removal_effect.png` -> `figures/fig06_offline_removal_effect.png`
- `fig07_aggregated_error_heatmap.png` -> `figures/fig07_aggregated_error_heatmap.png`
- `fig08_oracle_summary.png` -> `figures/fig08_oracle_summary.png`
- `fig09_oracle_per_seed_sid_delta.png` -> `figures/fig09_oracle_per_seed_sid_delta.png`
- `fig10_required_edge_acyclicity.png` -> `figures/fig10_required_edge_acyclicity.png`
- `fig11_fp_vs_fn_reconciliation.png` -> `figures/fig11_fp_vs_fn_reconciliation.png`

## Figures skipped

- (none)

## Investigative chain (labelling only)

1. Main result: the soft prior engaged mechanically on the targeted edges but the clean-grid SID / MMD did not show a clear improvement over the prior-free baseline.
2. Original forbidden-edge targets covered a small subset of the prior-free false positives.
3. Offline removal of those targets produced small SHD gains and mixed SID changes.
4. Exact budget-matched FP-targeted removal showed much larger available SID and SHD leverage.
5. Required-edge post-hoc repair was constrained by acyclicity: many beneficial candidates would have created cycles and were skipped.
6. Implication: future work points at improving target relevance / elicitation rather than at a stronger penalty weight.

