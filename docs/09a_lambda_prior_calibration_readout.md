# Lambda Prior Calibration Readout

## Status

This is a documentation-only readout for the main-study
`lambda_prior` calibration. It summarises the evidence from the
two calibration probes that were run on the main-study calibration
seeds 401 and 402 and records the resulting selected calibration
value.

Selected calibration value:

    lambda_prior = 2e-4

This value must still be recorded in `docs/03_decision_log.md`
before any matched-L1 selection or headline evaluation begins.
Until that decision-log entry exists, `lambda_prior` is to be
treated as TBD by every downstream artefact and code path.

## 1. Purpose

The calibration determines a non-degenerate soft-prior scale for
the targeted Frobenius penalty added inside the soft-prior DAGMA
variant. The penalty has the form

    L_prior = lambda_prior * sum_ij C_ij * W_ij^2

and contributes the gradient

    G_prior = 2 * lambda_prior * (C * W)

to DAGMA's hand-coded gradient assembly. The calibration probe
fits the soft-prior DAGMA at each candidate `lambda_prior` against
a prior-free baseline on the same data and reads the resulting
continuous-W magnitude at a single deliberately corrupted target
edge per seed.

This calibration tests continuous-W behaviour on calibration seeds
only. It is not a performance result. It does not compute SID,
MMD, or SHD. It does not use any of the headline evaluation seeds
501 through 507.

## 2. Calibration Rule

For a single target edge `(i, j)` on a given seed, let
`base_abs = abs(W_prior_free[i, j])` and
`soft_abs = abs(W_soft[i, j])`. Define

    ratio = soft_abs / base_abs

A candidate `lambda_prior` is classified `passed` on a seed only
if all of the following hold:

    0.05 <= ratio <= 0.5
    soft_abs >= 0.01
    soft_abs > 1e-6

Otherwise it is `too_weak` (when `ratio > 0.5`) or `too_strong`
(in the remaining cases, including any case where `soft_abs` is
below the absolute floor).

Interpretation of the three conditions:

- The `ratio <= 0.5` upper bound requires that the penalty
  produces visible suppression at the target edge.
- The `ratio >= 0.05` lower bound, together with
  `soft_abs >= 0.01`, prevents a practical near-annihilation
  outcome from being accepted as soft suppression.
- The strict `soft_abs > 1e-6` floor prevents acceptance of
  effective hard-clamping behaviour.

Joint per-seed selection rule. For each calibration seed, the
minimum passing `lambda_prior` is identified. The selected
calibration value is the largest of the per-seed minima, i.e. the
smallest single candidate that passes simultaneously on every
calibration seed. If any seed has no passing candidate, the probe
returns null and the grid is flagged for review.

## 3. Probe 1: Initial Grid

- output directory:
  `inspection/probes/output/lambda_prior_calibration/`
- calibration seeds: 401 and 402
- candidates: `(0.01, 0.05, 0.1, 0.5)`
- outcome: no recommendation; every candidate classified
  `too_strong` on both seeds; `grid_review_reason = "all_too_strong"`

Step A consistency. Both Step A checks (zero-mask gate at the
maximum candidate, and zero-lambda gate with a nonzero one-entry
mask) passed on both seeds with bit-exact deltas (max absolute
W deviation = 0.0), so the rejection is a property of the grid
rather than of the soft-prior implementation.

Smallest-candidate behaviour at the target edge:

| seed | target | base_abs | smallest candidate | soft_abs at smallest | ratio at smallest | interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 401 | (4, 3) | 0.7549 | 0.01 | 0.003503 | 0.004640 | `too_strong`: ratio below 0.05 lower bound; `soft_abs` below 0.01 floor |
| 402 | (5, 2) | 0.8091 | 0.01 | 0.000968 | 0.001197 | `too_strong`: ratio below 0.05 lower bound; `soft_abs` below 0.01 floor |

At the smallest candidate `0.01` the targeted true-positive edge
is already reduced to approximately 0.5 percent of its prior-free
magnitude on seed 401 and to approximately 0.1 percent on seed
402. Every larger candidate on this grid produces stronger
suppression and remains classified `too_strong`. The first grid
was therefore rejected because even its smallest candidate
produced practical near-annihilation at the target edge.

## 4. Decision After Probe 1

The grid was rescaled downward rather than accepting a
`too_strong` value as the calibrated value, since accepting it
would have collapsed the targeted weight close to a hard
constraint and would have failed to test soft suppression.

The lower-grid candidates were:

    (2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)

Probe 1's artefacts were preserved as-is on disk; the lower-grid
run wrote to a separate output directory so the two probes
co-exist without overwrites.

## 5. Probe 2: Lower Grid

- output directory:
  `inspection/probes/output/lambda_prior_calibration_lower_grid/`
- `probe_run = "lower_grid"`
- same calibration seeds (401, 402), same production constants,
  same target edges, same acceptance rule, same joint
  recommendation rule
- outcome: selected calibration value `lambda_prior = 2e-4`;
  `grid_needs_review = false`

Both Step A checks again passed on both seeds with bit-exact
deltas (max absolute W deviation = 0.0).

Per-candidate values at the target edge (lower grid):

| seed | lambda_prior | base_abs | soft_abs | ratio | candidate_status | passes |
| ---: | ---: | ---: | ---: | ---: | --- | :---: |
| 401 | 2e-5  | 0.7549 | 0.72564 | 0.96123 | too_weak    | False |
| 401 | 5e-5  | 0.7549 | 0.68468 | 0.90697 | too_weak    | False |
| 401 | 1e-4  | 0.7549 | 0.62561 | 0.82871 | too_weak    | False |
| 401 | 2e-4  | 0.7549 | 0.12692 | 0.16812 | passed      | True  |
| 401 | 5e-4  | 0.7549 | 0.06116 | 0.08102 | passed      | True  |
| 401 | 1e-3  | 0.7549 | 0.03254 | 0.04311 | too_strong  | False |
| 402 | 2e-5  | 0.8091 | 0.77792 | 0.96149 | too_weak    | False |
| 402 | 5e-5  | 0.8091 | 0.68057 | 0.84117 | too_weak    | False |
| 402 | 1e-4  | 0.8091 | 0.53787 | 0.66479 | too_weak    | False |
| 402 | 2e-4  | 0.8091 | 0.35851 | 0.44311 | passed      | True  |
| 402 | 5e-4  | 0.8091 | 0.17675 | 0.21846 | passed      | True  |
| 402 | 1e-3  | 0.8091 | 0.07780 | 0.09616 | passed      | True  |

Per-seed minimum passing candidate:

- seed 401: `2e-4`
- seed 402: `2e-4`

Applying the joint per-seed rule (largest of the per-seed minima)
yields:

    lambda_prior = 2e-4

## 6. Interpretation

At `lambda_prior = 2e-4`, the targeted true-positive edge is
suppressed by the penalty but retains a clearly nonzero magnitude
on both calibration seeds:

- seed 401 retains approximately 16.8 percent of the prior-free
  target magnitude (`soft_abs` 0.127 from `base_abs` 0.755);
- seed 402 retains approximately 44.3 percent of the prior-free
  target magnitude (`soft_abs` 0.359 from `base_abs` 0.809).

On seed 402, the retained `soft_abs` 0.359 remains above the
DAGMA project threshold of 0.3. This shows that the calibrated
penalty does not necessarily eliminate a strongly data-supported
corrupted-prior edge: the data-fit gradient and the penalty
gradient reach an interior equilibrium rather than collapsing
the edge to zero. The same equilibrium logic produces a smaller
retained magnitude on seed 401, where the prior-free baseline
itself is smaller.

Seed 401 shows a sharp nonlinear response in this calibration
probe between `lambda_prior = 1e-4` (ratio 0.829, `too_weak`)
and `lambda_prior = 2e-4` (ratio 0.168, `passed`): the ratio
moves from approximately 0.83 to approximately 0.17 across a
single grid step. This is recorded here as an observed
calibration feature motivating the smoke-calibration step before
matched-L1 selection. No causal explanation is offered and no
broader claim is attached to this observation; the calibration
rule treats `2e-4` as a passing candidate on this seed.

## 7. Decision

The calibrated value for the main study is

    lambda_prior = 2e-4

This decision fixes the soft-prior penalty scale at the
configured operating point. It does not constitute evidence that
the soft-prior method improves SID, MMD, SHD, or any other
downstream performance metric. The calibration only fixes the
prior-loss scale before matched-L1 selection and headline
evaluation are run.

No evaluation seed (501 through 507) was used at any point in
either probe; both probes operated exclusively on the calibration
seeds 401 and 402.

## 8. Limitations

- Only two calibration seeds (401 and 402) were used. The
  calibration is a smoke check, not an estimate of cross-seed
  variance for the penalty scale.
- A single deliberately corrupted true-positive target edge was
  selected per seed via the target-selection rule (true-positive
  edge with the largest `abs(W_prior_free)` at the first
  threshold tier from `(0.3, 0.2, 0.1)` that admits one).
  Per-seed sensitivity to the choice of target edge is not
  measured.
- The probe characterises continuous-W shrinkage at the target
  edge only. It does not measure SID, MMD, or SHD; it does not
  measure thresholded-graph behaviour beyond the target entry;
  and it does not measure the effect of the penalty on other
  edges of the learned W.
- The acceptance window `0.05 <= ratio <= 0.5` with
  `soft_abs >= 0.01` is a calibration heuristic chosen to exclude
  near-annihilation and near-no-effect. It is not a downstream
  performance criterion.
- Final scientific claims must come from the main evaluation only.

## 9. Required Next Actions

1. Add a new entry to `docs/03_decision_log.md` that freezes
   `lambda_prior = 2e-4` for the main study and references this
   readout as the supporting evidence.
2. Proceed to the next implementation slices: M-3 prior
   generation and corruption, and M-4 hard-exclusion plumbing.
3. Do not run matched-L1 selection or any headline evaluation
   until `docs/03_decision_log.md` records the calibrated
   `lambda_prior` value.
