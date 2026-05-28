# Matched-L1 Calibration Readout

- halt_status: completed
- parent_heldout_run_hash_full: 88da382e8672650e44f44e675011dda1a81868c9075acb86faef6c6caf23fd17
- calibration_run_hash12: 71bfe6629b9d
- output_dir: results/main_study/calibration/matched_l1/71bfe6629b9d
- code_version: f993c0422db85939481bcdea980537d145730fbe

## Target

- target_mean_edge_count: 11.5
- target_per_seed_edge_counts: [12, 11]

## Stage 1

- lambda1=0.025: mean_edge_count=18.0, valid_dag_count=2, absolute_gap=6.5, fragile=False
- lambda1=0.05: mean_edge_count=14.0, valid_dag_count=2, absolute_gap=2.5, fragile=False
- lambda1=0.075: mean_edge_count=12.5, valid_dag_count=2, absolute_gap=1.0, fragile=False
- lambda1=0.1: mean_edge_count=12.0, valid_dag_count=2, absolute_gap=0.5, fragile=False
- lambda1=0.15: mean_edge_count=9.5, valid_dag_count=2, absolute_gap=2.0, fragile=False
- lambda1=0.2: mean_edge_count=8.0, valid_dag_count=2, absolute_gap=3.5, fragile=False
- lambda1=0.25: mean_edge_count=8.0, valid_dag_count=2, absolute_gap=3.5, fragile=False

## Stage 2

Stage 2 generated 5 candidate values over the selected interval. 2 value(s) coincided with Stage 1 candidates and were skipped as duplicates, leaving 3 new candidate value(s) to evaluate.

- stage_2_interval: (0.075, 0.15)
- stage_2_generated_candidates: [0.075, 0.09375, 0.11249999999999999, 0.13124999999999998, 0.15]
- stage_2_skipped_duplicates: [0.075, 0.15]
- stage_2_candidates (new): [0.09375, 0.11249999999999999, 0.13124999999999998]
- lambda1=0.09375: mean_edge_count=12.5, valid_dag_count=2, absolute_gap=1.0, fragile=False
- lambda1=0.11249999999999999: mean_edge_count=12.0, valid_dag_count=2, absolute_gap=0.5, fragile=False
- lambda1=0.13124999999999998: mean_edge_count=9.0, valid_dag_count=2, absolute_gap=2.5, fragile=False

## Selection

- selected_lambda1: 0.1
- selected_candidate_mean_edge_count: 12.0
- selected_absolute_gap: 0.5
- selected_valid_dag_count: 2
- within_one_edge_tolerance: True

## Failure and fragility

- invalid_or_failed_fit_count: 0
- candidates_with_only_one_valid_dag_fit: []

## Diagnostic anomalies (advisory only)

- non_monotonic_confidence: corruption=0.0, mean_edge_counts_by_confidence=[12.0, 11.0, 11.0, 11.5, 11.5]
- non_monotonic_confidence: corruption=0.2, mean_edge_counts_by_confidence=[12.0, 11.0, 10.5, 11.0, 11.0]
- non_monotonic_confidence: corruption=0.6, mean_edge_counts_by_confidence=[12.0, 11.5, 11.5, 10.5, 11.5]

## Confirmations

- SID, SHD, and MMD were NOT used for selection of matched_l1_lambda1.
- Evaluation seeds were NOT used; only calibration seeds [401, 402] were used.
- Diagnostic-grid anomalies are advisory; they did not alter selected_lambda1.

halt_status: completed