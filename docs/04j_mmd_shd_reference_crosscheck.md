# 04j: MMD and SHD reference cross-check

## Status

Verification probe artefact. Read-only with respect to project source,
tests, and dependency manifests. No project file was modified by this
probe.

## 1. Purpose

Verify the project mmd_rbf_unbiased and shd implementations against
independent references before selection-study execution, so that any
formula or convention error is caught before the selection-study runner
consumes the metric values.

This is not a metric redesign. No source or test changes are made.

## 2. Probe artefact

`inspection/probes/c_p14_mmd_shd_reference_crosscheck.py`

Written after source inspection of the project formula. The probe records
the inspected formula, implements an independent loop-based MMD reference
in plain Python (math.exp, no NumPy), runs both on the same inputs, and
compares to gadjid.shd for SHD.

## 3. Inspected MMD formula

The project mmd_rbf_unbiased function (verified by source inspection and
loop-reference agreement):

- Kernel: k(a, b) = exp(-||a - b||^2 / bandwidth).
- Within-sample XX term: (Kxx.sum() - m) / (m * (m - 1)).
  Diagonal self-pairs excluded; denominator m*(m-1).
- Within-sample YY term: (Kyy.sum() - n) / (n * (n - 1)).
  Same pattern.
- Cross-sample XY term: 2.0 * Kxy.sum() / (m * n).
  All m*n pairs included; factor of 2.
- MMD^2 = xx_term - xy_term + yy_term.
- Return value is raw and is never clipped. Negative values are valid
  for the unbiased estimator and are preserved.

Median heuristic (bandwidth=None path):
- Concatenate x and y into a single set of (m + n) samples.
- Compute pairwise squared distances over all pairs (i, j) with i < j
  (upper triangle only; self-distances excluded).
- Take np.median of those squared distances.
- Use this as the bandwidth parameter.
- If the result is <= 0 (degenerate case), raise ValueError.

## 4. MMD loop-reference comparison results

Loop reference implemented with plain Python loops and math.exp,
mirroring the inspected formula exactly. Cases A-C and E compare project
against the loop reference; tolerance 1e-10. Case D is a same-distribution
sanity check on the project implementation only (no loop reference).

| Case | Description | Metric | Result |
|---|---|---|---|
| A | fixed 3x2 vs 2x2 arrays, bw=1.0 | diff (abs) = 0.000e+00 | OK |
| B | negative unbiased-estimator example, bw=0.5 | diff (abs) = 2.220e-16 | OK |
| C | median-heuristic bandwidth (bw=2.5) | diff (abs) = 0.000e+00 | OK |
| D | same-dist seeded (n=200, seed=0) | project MMD = 0.000990 (< 0.05) | OK |
| E | all-identical samples (degenerate) | raised ValueError | OK |

Case B confirms the unbiased estimator returned -0.202332..., which is
correctly negative (cross-kernel dominates). The return value is not clipped.

Case C recovered the median bandwidth 2.5 from the reference independently,
confirming the concatenated-upper-triangle-squared-distance convention.

Case E confirms the degenerate-bandwidth path raises ValueError with the
message "median heuristic produced a non-positive bandwidth (0)".

The largest deviation across explicit-bandwidth cases A-C is 2.220e-16,
which is floating-point rounding at the last ULP. All loop-reference
comparisons pass within tolerance.

## 5. SHD hand cases

All calls use the project-facing convention shd(predicted, true, reversal_cost=...).

| Case | reversal_cost | Expected | Got | Result |
|---|---|---|---|---|
| identity empty 3x3 | 2 | 0 | 0 | OK |
| identity chain 0->1->2 | 2 | 0 | 0 | OK |
| empty predicted, single true edge | 2 | 1 | 1 | OK |
| single predicted edge, empty true | 2 | 1 | 1 | OK |
| two predicted edges, empty true | 2 | 2 | 2 | OK |
| reversed edge | 2 | 2 | 2 | OK |
| reversed edge | 1 | 1 | 1 | OK |

All seven SHD hand cases matched expected values.

## 6. gadjid.shd comparison

gadjid.shd signature: (g_true, g_guess) -- no edge_direction argument.
SHD is symmetric so no convention flip is required.
Return type: tuple (normalised_distance, count), same pattern as gadjid.sid.

gadjid.shd DAG cases (using int8 inputs, row-to-column reading):

| Case | Expected | gadjid count | Result |
|---|---|---|---|
| identity empty 2x2 | 0 | 0 | OK |
| identity chain 0->1->2 | 0 | 0 | OK |
| empty predicted, single true edge | 1 | 1 | OK |
| two predicted edges, empty true | 2 | 2 | OK |

### Reversal-cost convention comparison

For the case predicted=1->0, true=0->1 (single reversed edge, n=2):

    gadjid.shd count           = 1   (raw tuple = (1.0, 1))
    project shd reversal_cost=1 = 1
    project shd reversal_cost=2 = 2

gadjid.shd returns 1 for a reversed edge, matching the project's
reversal_cost=1 convention (cheaper: a reversal counts as one operation).
The project's default is reversal_cost=2 (stricter: a reversal counts as
two independent operations).

This is a documented convention difference, not a bug. The project's
reversal_cost=2 default is recorded in docs/03_decision_log.md (Metrics-
layer entry). For the selection study, SHD is a secondary metric and the
project convention is already frozen. gadjid.shd uses the cheaper
convention. If gadjid.shd is used in any future reporting step, the reversal
convention difference must be kept in mind.

No source change is recommended based on this finding.

### gadjid.shd invalid-input behaviour

Probe: gadjid.shd(empty 3x3, cyclic 3-node g_guess).

    RAISED RuntimeError: 'Errors occured when loading adjacency matrix...'

gadjid.shd raises RuntimeError on a cyclic g_guess input. This is consistent
with the invalid-input behaviour observed for gadjid.sid in the spike
(docs/04i). Project-side acyclicity validation would reject this input
before any gadjid call in a production code path.

## 7. Source and test change recommendation

No source or test changes are recommended. All MMD loop-reference
differences are at or below floating-point rounding (< 1e-10). All SHD
hand cases matched. gadjid.shd matched on all valid DAG cases. The
reversal-cost convention difference is documented and frozen.

## 8. What this verifies

- The project mmd_rbf_unbiased implementation matches the inspected formula
  and agrees with the independent loop reference on every tested case
  (kernel, diagonal exclusion, denominators, cross-term factor, no clipping).
- The median heuristic uses squared distances on the concatenated
  upper-triangle set, and the result is used directly as the bandwidth.
- The unbiased MMD estimator can correctly return negative values and
  does not clip them.
- The project shd implementation produces correct counts for all
  insertion, deletion, and reversal cases at both reversal_cost=1 and =2.
- gadjid.shd agrees with the project shd on all valid DAG inputs using
  the reversal_cost=1 convention.
- MMD and SHD have independent cross-check evidence before
  selection-study execution.

## 9. What this does not verify

- This does not replace SID verification (SID is already verified;
  see docs/phase_2d_sid_readout.md).
- This does not start the selection study.
- This does not validate any wrapper, sampler, or prior-loss component.
- This does not validate MMD or SHD on large graphs or model outputs;
  the cross-check is on the raw metric primitives only.

## 10. Test results from standard suite

pytest tests/test_interventional_metrics.py tests/test_structural_metrics.py:
48 passed, 0 skipped (interventional) + 26 passed (structural) = 74 passed.
Latest known full suite (from SID closure, not rerun during C-P14):
384 passed, 0 skipped, 2 warnings.
