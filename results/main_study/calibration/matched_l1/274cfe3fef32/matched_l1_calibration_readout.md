# Matched-L1 Calibration Readout

- halt_status: completed
- parent_heldout_run_hash_full: 88da382e8672650e44f44e675011dda1a81868c9075acb86faef6c6caf23fd17
- calibration_run_hash12: 274cfe3fef32
- output_dir: results/main_study/calibration/matched_l1/274cfe3fef32
- code_version: fd0d6619755d7e1e0e6c182c07d39e88a989887f

## Target

- target_mean_edge_count: 13.0
- target_per_seed_edge_counts: [14, 12]

## Stage 1

- lambda1=0.025: mean_edge_count=18.0, valid_dag_count=2, absolute_gap=5.0, fragile=False
- lambda1=0.05: mean_edge_count=14.0, valid_dag_count=2, absolute_gap=1.0, fragile=False
- lambda1=0.075: mean_edge_count=12.5, valid_dag_count=2, absolute_gap=0.5, fragile=False
- lambda1=0.1: mean_edge_count=12.0, valid_dag_count=2, absolute_gap=1.0, fragile=False
- lambda1=0.15: mean_edge_count=9.5, valid_dag_count=2, absolute_gap=3.5, fragile=False
- lambda1=0.2: mean_edge_count=8.0, valid_dag_count=2, absolute_gap=5.0, fragile=False
- lambda1=0.25: mean_edge_count=8.0, valid_dag_count=2, absolute_gap=5.0, fragile=False

## Stage 2

Stage 2 generated 5 candidate values over the selected interval. 3 value(s) coincided with Stage 1 candidates and were skipped as duplicates, leaving 2 new candidate value(s) to evaluate.

- stage_2_interval: (0.05, 0.1)
- stage_2_generated_candidates: [0.05, 0.0625, 0.07500000000000001, 0.08750000000000001, 0.1]
- stage_2_skipped_duplicates: [0.05, 0.07500000000000001, 0.1]
- stage_2_candidates (new): [0.0625, 0.08750000000000001]
- lambda1=0.0625: mean_edge_count=12.5, valid_dag_count=2, absolute_gap=0.5, fragile=False
- lambda1=0.08750000000000001: mean_edge_count=12.5, valid_dag_count=2, absolute_gap=0.5, fragile=False

## Selection

- selected_lambda1: 0.0625
- selected_candidate_mean_edge_count: 12.5
- selected_absolute_gap: 0.5
- selected_valid_dag_count: 2
- within_one_edge_tolerance: True

## Failure and fragility

- invalid_or_failed_fit_count: 0
- candidates_with_only_one_valid_dag_fit: []

## Diagnostic anomalies (advisory only)

- non_monotonic_confidence: corruption=0.0, mean_edge_counts_by_confidence=[14.0, 13.0, 13.5, 12.5, 13.0]
- non_monotonic_confidence: corruption=0.2, mean_edge_counts_by_confidence=[14.0, 12.5, 12.5, 13.5, 13.5]
- non_monotonic_confidence: corruption=0.4, mean_edge_counts_by_confidence=[14.0, 13.0, 13.5, 13.5, 14.0]
- non_monotonic_confidence: corruption=0.6, mean_edge_counts_by_confidence=[14.0, 12.5, 12.5, 12.5, 13.0]
- non_monotonic_confidence: corruption=0.8, mean_edge_counts_by_confidence=[14.0, 12.5, 11.0, 11.5, 11.5]

## Confirmations

- SID, SHD, and MMD were NOT used for selection of matched_l1_lambda1.
- Evaluation seeds were NOT used; only calibration seeds [401, 402] were used.
- Diagnostic-grid anomalies are advisory; they did not alter selected_lambda1.

halt_status: completed