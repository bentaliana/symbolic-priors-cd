    # Matched-L1 Calibration Plan

## Addendum (2026-05-28): re-calibration under the corrected DAGMA backbone

The first execution of this plan ran with a default DAGMA backbone
of `lambda1 = 0.05`, `warm_iter = 30000`, `max_iter = 60000`
(wrapper-level Phase-A defaults). The plan and the held-out
selection both fix the backbone at `lambda1 = 0.10`,
`warm_iter = 20000`, `max_iter = 70000`, so the original
calibration ran on a soft-prior target measured at off-protocol
backbone settings.

The calibration has been re-executed under the protocol backbone.
The earlier artefact tree at
`results/main_study/calibration/matched_l1/274cfe3fef32/` is
superseded; the new calibration is at
`results/main_study/calibration/matched_l1/71bfe6629b9d/`.

Outcome (new):

- soft-prior target mean edge count: `11.5` (per-seed `[12, 11]`)
- selected `matched_l1_lambda1`: `0.10`
- selected candidate mean edge count: `12.0`
- absolute gap: `0.5` (within the one-edge tolerance)
- valid-DAG count: `2`
- halt status: `completed`
- evaluation seeds used: `false`
- SID / SHD / MMD used for selection: `false`

The procedure described in Sections 1-9 below is unchanged; the
calibration script enforces every selection guardrail. The
selected value `0.10` carried forward into the main evaluation
plan supersedes the prior frozen value `0.0625`.

---

## 1. Purpose

The main study compares confidence-weighted soft forbidden-edge priors against prior-free DAGMA, hard exclusion, and a matched generic sparsity baseline. The matched-L1 baseline is included to control for generic regularisation pressure: it asks whether any observed benefit of the soft Frobenius prior is due to its targeted edge-specific structure, or whether a comparable amount of untargeted global L1 regularisation would produce similar behaviour.

This document defines how `matched_l1_lambda1` is selected before the main evaluation. The calibration uses only the main-calibration seeds and must be completed before any main-evaluation seed is run.

The calibration objective is sparsity matching, not performance optimisation. SID, SHD, MMD, runtime, or downstream performance must not be used to select `matched_l1_lambda1`.

---

## 2. Operational Definitions

### 2.1 Calibration Seeds

The calibration uses only the main-calibration seed pool:

```text
401, 402

Main-evaluation seeds must not be used before matched_l1_lambda1 is frozen.

2.2 Sparsity Measure

Sparsity is measured as the number of off-diagonal directed entries in the thresholded adjacency matrix:

edge_count = number of off-diagonal entries where abs(W) >= 0.3

where W is the learned continuous adjacency matrix and 0.3 is the frozen DAGMA threshold used throughout the main study.

The diagonal is explicitly excluded from edge_count. Self-loops are not counted as ordinary directed edges; they are captured separately through graph_status. This avoids double-penalising a run for self-loops: graph validity is handled by the validity criterion, while sparsity measures ordinary off-diagonal structural density.

Continuous nonzero counts are not used because DAGMA's L1 penalty does not necessarily produce exact numerical zeros, making such a measure sensitive to floating-point noise.

2.3 Target Soft-Prior Condition

The matched-L1 target is the mean thresholded edge count of the soft-prior model under the strongest clean-prior setting:

method_family = soft_frobenius
corruption_fraction = 0.0
confidence = 1.0
lambda_prior = 2e-4
seeds = 401, 402

This condition is used because it represents the strongest targeted soft-prior regularisation in the study. Matching against the full confidence/corruption grid would create a mixed target and would blur the interpretation of the matched-L1 baseline. The full grid is the object of the main study, not the definition of the baseline regularisation strength.

2.4 Matched-L1 Condition

For each candidate value of matched_l1_lambda1, DAGMA is run on the same calibration seeds with its global L1 regularisation set to that candidate value.

The selected value is a single global scalar. Per-seed lambda selection is not allowed.

3. Candidate Grid and Refinement Rule
3.1 Stage 1 Grid

The Stage-1 candidate grid is:

(0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25)

This grid spans weaker and stronger regularisation around the established DAGMA operating region while remaining small enough to avoid an unconstrained hyperparameter search.

3.2 Stage 2 Refinement

After Stage 1, candidates are ranked using the selection hierarchy in Section 4. The closest Stage-1 candidate under that hierarchy determines the Stage-2 refinement interval.

Final selection is made over the union of Stage-1 and Stage-2 evaluated candidates.

Internal Case

If the closest Stage-1 candidate has both a lower and an upper Stage-1 neighbour, Stage 2 refines over the interval between those neighbours.

Example:

closest Stage-1 candidate = 0.1
lower neighbour = 0.075
upper neighbour = 0.15
Stage-2 interval = [0.075, 0.15]
Boundary Case

If the closest Stage-1 candidate is the lower boundary, Stage 2 refines over:

[0.0125, 0.05]

If the closest Stage-1 candidate is the upper boundary, Stage 2 refines over:

[0.2, 0.3]

These boundary intervals provide one predeclared outward extension without allowing repeated grid expansion.

Candidate Generation

Within the selected Stage-2 interval, evenly spaced candidate values are generated. Values already evaluated in Stage 1 are skipped and must not be re-run. All generated Stage-2 values and all skipped duplicates must be recorded in the calibration readout.

No automatic third grid is allowed. If the selected final candidate remains at an outward boundary and the sparsity mismatch is poor, calibration must halt for human adjudication before M-8. Any further grid expansion requires an explicit decision-log entry before any evaluation seed is run.

4. Selection Rule and Failure Policy
4.1 Primary Selection Rule

For each candidate matched_l1_lambda1, compute:

candidate_mean_edge_count = mean edge_count across calibration seeds
target_mean_edge_count = mean edge_count of the soft_frobenius target across calibration seeds
absolute_gap = abs(candidate_mean_edge_count - target_mean_edge_count)

Candidates are ranked using the following hierarchy:

Exclude candidates with zero valid-DAG calibration fits.
Prefer candidates with the maximum number of valid-DAG calibration fits.
Among those, choose the candidate with the smallest absolute mean edge-count gap.
If still tied, choose the smaller lambda1 value to avoid unnecessary over-regularisation.

Valid-DAG count is computed from graph_status.

Candidates with only one valid-DAG fit out of two calibration seeds remain eligible, but this is fragile evidence and must be explicitly flagged in the readout. If no candidate has at least one valid-DAG calibration fit, calibration must stop for human review rather than freezing a value automatically.

4.2 Reporting Tolerance

A selected candidate is labelled a close sparsity match if:

absolute_gap <= 1.0 mean edge

This tolerance is a reporting label, not the selection rule. The selected value is always chosen by the hierarchy in Section 4.1. If the selected value has a gap greater than one mean edge, the value may still be frozen only if it is the best eligible candidate, and the residual mismatch must be reported explicitly.

4.3 Failed and Cyclic Runs

Thresholded edge count is defined whenever a learned continuous adjacency matrix or thresholded adjacency is available, including cyclic graphs. Cyclic graphs do not make sparsity undefined.

However, graph validity remains operationally important. A candidate that matches sparsity but produces invalid DAGs is less suitable as a stable baseline than a candidate that matches slightly less closely but produces valid DAGs. This is why valid-DAG count is prioritised in the selection hierarchy.

For every calibration run, the summary must report:

fit_status
graph_status
sampler_status
metric_status
edge_count

Failed, cyclic, sampler-unavailable, or metric-unavailable runs must not be hidden.

4.4 Metrics Are Diagnostic Only

SID, SHD, and MMD may be computed by the runner if available, but they must not be used to choose matched_l1_lambda1.

The calibration readout must explicitly state that metric values were diagnostic only and were not used for selection.

5. Required Calibration Runs
5.1 Soft-Prior Target Runs

The required target set is:

2 fits = 2 calibration seeds x 1 target condition

Target condition:

method_family = soft_frobenius
corruption_fraction = 0.0
confidence = 1.0
lambda_prior = 2e-4
seeds = 401, 402
5.2 Matched-L1 Stage 1

Stage 1 requires:

14 fits = 7 candidate lambda values x 2 calibration seeds
5.3 Matched-L1 Stage 2

Stage 2 generates exactly five evenly spaced candidate values within the selected Stage-2 interval (inclusive of the interval endpoints). Candidate values that coincide with values already evaluated in Stage 1 are skipped and recorded as skipped duplicates in the calibration readout; they must not be re-run. Each remaining new candidate is evaluated on both calibration seeds.

The five values are generated before duplicate filtering; the number of actual Stage-2 fits is therefore at most ten and depends only on how many of the five generated values coincide with Stage-1 candidates. Every newly generated Stage-2 candidate that is not a Stage-1 duplicate must be evaluated on both calibration seeds.

5.4 Soft-Prior Diagnostic Grid

The full soft-prior calibration grid is required as a diagnostic complement:

50 fits = 2 calibration seeds x 5 corruption levels x 5 confidence levels

These runs are not used to select matched_l1_lambda1. They are included to inspect whether soft-prior sparsity behaves anomalously across the calibration confidence/corruption grid before M-8.

The diagnostic grid exists to surface anomalous soft-prior sparsity behaviour before the headline evaluation. If it reveals an anomaly, such as sparsity collapse or unexpected non-monotonic behaviour across confidence or corruption settings, the response is to halt and adjudicate through the decision log, not to retune matched_l1_lambda1. The selected matched_l1_lambda1 is fixed by Sections 4 and 5.1 through 5.3 alone; diagnostic-grid findings can trigger adjudication or scope changes but must never silently shift the matched-L1 selection.

The diagnostic grid is required as non-selection evidence and must be clearly labelled as such in the calibration readout.

6. Outputs

The M-7 calibration stage should produce both machine-readable and human-readable outputs.

The calibration summary outputs are written under
`results/main_study/calibration/matched_l1/<calibration_run_hash12>/`.

The per-run records and artefacts produced during calibration follow the
standard main-study invocation layout under
`results/main_study/<calibration_run_hash12>/records/` and
`results/main_study/<calibration_run_hash12>/artifacts/`.

Each per-run record remains individually identifiable through its stored
`record_path`, `configuration_hash_full`, and `configuration_hash_prefix`.
The calibration summary references these per-run records by `record_path`
and `configuration_hash_full`.

6.1 Machine-Readable Outputs

The calibration should save:

matched_l1_calibration_summary.json
matched_l1_calibration_table.csv

The summary JSON must include at least:

target_mean_edge_count
target_per_seed_edge_counts
stage_1_candidates
stage_2_candidates
stage_2_interval
stage_2_skipped_duplicates
all_evaluated_candidates
selected_lambda1
selected_candidate_mean_edge_count
selected_absolute_gap
selected_valid_dag_count
within_one_edge_tolerance
selection_rule
diagnostic_metric_fields_used_for_selection = false
evaluation_seeds_used = false

Each candidate-row entry should include:

candidate_lambda1
seed
stage
edge_count
fit_status
graph_status
sampler_status
metric_status
record_path
configuration_hash_full
configuration_hash_prefix
6.2 Human-Readable Readout

The calibration should also produce:

matched_l1_calibration_readout.md

The readout must state:

the soft-prior target mean edge count;
the target per-seed edge counts;
the Stage-1 grid and result;
the Stage-2 interval and result;
the selected matched_l1_lambda1;
whether the selected value is within one mean edge of the target;
whether any candidate had invalid graphs, fit failures, sampler failures, or metric-unavailable runs;
whether any selected or near-selected candidate relied on only one valid-DAG fit;
confirmation that SID, SHD, and MMD were not used for selection;
confirmation that evaluation seeds were not used.
7. Decision-Log Update

After the calibration result is reviewed, docs/03_decision_log.md must be updated to freeze:

matched_l1_lambda1 = <selected value>

The decision-log entry should include:

calibration seeds used
soft-prior target condition
target mean edge count
candidate grids
Stage-2 refinement interval
selection rule
selected value
selected candidate mean edge count
absolute gap
valid-DAG count
whether the value is within one mean edge of the target
statement that evaluation seeds were not used
statement that SID, SHD, and MMD were not used for selection

The calibration script must not silently update docs/03_decision_log.md unless explicitly instructed. The selected value must be reviewed before freezing.

8. Guardrails

The following are prohibited during M-7:

using evaluation seeds;
selecting matched_l1_lambda1 by SID, SHD, MMD, runtime, or downstream performance;
per-seed lambda selection;
repeated grid expansion without a decision-log entry;
changing lambda_prior;
changing the soft-prior target condition after seeing results;
hiding failed, cyclic, sampler-unavailable, or metric-unavailable runs;
using the soft-prior diagnostic grid to alter the selected matched_l1_lambda1;
silently updating the decision log without human review.

If the calibration result is poor or unstable, the correct response is to document the instability and adjudicate before moving to M-8, not to silently search until a convenient value appears.

9. Planned Next Step

After this plan is accepted, implement the M-7 matched-L1 calibration script using the existing main-study workload planner, runner, real backends, and run I/O layer.

The script should run only calibration workloads, produce the outputs listed above, and report the selected candidate for review. It must not run any main-evaluation seed and must not freeze matched_l1_lambda1 in the decision log unless explicitly instructed.