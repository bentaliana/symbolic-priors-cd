# Base-model selection visual readout

## Run identity

- calibration_run_hash_prefix: `4a67117a10b1`
- heldout_run_hash_prefix: `88da382e8672`
- base model carried forward: **DAGMA**

## Inputs read

| input | exists | path |
| --- | --- | --- |
| `adjudication_md` | True | `docs\08h_selection_study_adjudication.md` |
| `heldout_evaluation_json` | True | `results\model_selection\held_out\88da382e8672\heldout_evaluation.json` |
| `main_summary_csv` | True | `results\model_selection\held_out\88da382e8672\readout\main_summary.csv` |
| `per_seed_main_csv` | True | `results\model_selection\held_out\88da382e8672\readout\per_seed_main.csv` |
| `selected_configurations_json` | True | `results\model_selection\calibration\4a67117a10b1\selected_configurations.json` |
| `selection_doc_md` | True | `docs\02_base_model_selection.md` |
| `sensitivity_summary_csv` | True | `results\model_selection\held_out\88da382e8672\readout\sensitivity_summary.csv` |
| `status_summary_csv` | True | `results\model_selection\held_out\88da382e8672\readout\status_summary.csv` |

## Figures generated

- `fig02_heldout_metric_means.png` -> `base_model_selection_figures/fig02_heldout_metric_means.png`
- `fig02b_paired_model_differences.png` -> `base_model_selection_figures/fig02b_paired_model_differences.png`
- `fig03_heldout_sid_per_seed.png` -> `base_model_selection_figures/fig03_heldout_sid_per_seed.png`
- `fig05_runtime_log_scale.png` -> `base_model_selection_figures/fig05_runtime_log_scale.png`
- `fig06_dcdi_fit_rng_sensitivity.png` -> `base_model_selection_figures/fig06_dcdi_fit_rng_sensitivity.png`
- `fig07_dagma_ceiling_and_headroom.png` -> `base_model_selection_figures/fig07_dagma_ceiling_and_headroom.png`
- `fig_status_reliability.png` -> `base_model_selection_figures/fig_status_reliability.png`

## Figures skipped

- (none)

## Decision context

This readout summarises evidence already recorded by the frozen base-model selection adjudication. No new claim is introduced. The selection rule is SID primary; MMD is the tie-breaker under the documented SID-margin condition; SHD and runtime are advisory.

