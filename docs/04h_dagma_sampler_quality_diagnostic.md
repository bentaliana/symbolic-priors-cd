# C-P13: DAGMA Sampler-Quality Diagnostic

## 1. Status

Diagnostic probe artefact. Read-only with respect to project source and
external repositories. CPU only. No dependency was installed. This document
records the values produced by
`inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py` on the
project commit at which the probe was first run.

Date: 2026-05-14.

## 2. Purpose

Empirically test whether the DAGMA wrapper produces credible
interventional distributions on the same frozen diagnostic fixture used
by C-P11 for DCDI-G. Localise any failure to one of: learned structure,
coefficient quality, residual noise, invalid graph status, or sampler
mechanics.

## 3. Evidential scope

- This probe is reused verbatim (apart from model-specific arithmetic)
  from C-P11 in order to give a controlled DAGMA-vs-DCDI same-fixture
  comparison.
- It is fixture-specific evidence. A pass or fail does not replace the
  full Doc 02 base-model selection study.
- A pass licenses only the claim that DAGMA's wrapper and sampler
  behave credibly on this particular interventional-adequacy probe.
- The fixture was not designed to expose DAGMA's failure modes; it was
  designed to expose DCDI-G's failure modes in C-P11.
- The actual base-model decision still depends on multi-seed selection
  study results, criterion ordering, and verified SID integration.
- This document does not introduce a new main-study baseline and does
  not change the protocol.

## 4. Exact frozen setup

- SCM: `generate_linear_gaussian_scm(n_nodes=3, expected_edges=3, seed=0)`
- Training data: 5000 observational samples, `sample_observational`
  seed = 1
- Validation data: 500 observational samples, `sample_observational`
  seed = 2 (drawn for comparability with C-P11; not used in any MMD)
- Preprocessing: `CentredOnlyTransform` fitted on training data
- Intervention: `do(X_2 = +2.0)` in raw SCM units
- Wrapper: `DAGMAWrapper` with `DAGMAConfig()` defaults
- DAGMA internal threshold: `w_threshold_internal = 0.0`
- Project threshold (external): `abs(W_continuous) >= 0.3`
- Primary noise policy: `noise_policy = "residual_fitted"`
- Sensitivity noise policy: `noise_policy = "unit_variance"` (reported
  separately)
- Per-batch sample size: 1000
- `n_floor = 5`, `n_wrapper = 5`
- Aggregation statistic: median
- Floor aggregation: median over all `C(5, 2) = 10` pairwise
  ground-truth MMDs
- Paired aggregation: median over the 5 paired (wrapper_k, gt_k) MMDs
- MMD function: `mmd_rbf_unbiased` with default median-heuristic
  bandwidth (same call signature as C-P11)
- Seed bases:
  - `GT_FLOOR_SEED_BASE = 1000`
  - `GT_PAIRED_SEED_BASE = 1100`
  - `GT_WRONG_SEED_BASE = 1200`
  - `WRAPPER_SEED_BASE = 2000`
  - `WRAPPER_WRONG_SEED_BASE = 2100`
  - `WRAPPER_TRUE_SEED_BASE = 2200` (Diagnostic A)
  - `WRAPPER_LEARNED_AUG_SEED_BASE = 2300` (Diagnostic B1)
  - `WRAPPER_ORACLE_AUG_SEED_BASE = 2400` (Diagnostic B2)

The 2300 and 2400 seed bases are new relative to C-P11 because DAGMA
exposes an explicit per-edge weight, so B1 (learned weight) and B2
(oracle weight) are scientifically distinct; C-P11 used a single
augmented diagnostic at 2300 because DCDI's conditionals are not a
single coefficient per edge.

## 5. Reproducibility statement

Re-running the probe on the same project commit and the same Python
environment should reproduce the recorded scientific numerical values
bitwise: the learned continuous W matrix, `h_final`, `score_final`,
the thresholded adjacency, SHD, the residual sigma vector, every MMD
value (floor, wrapper-vs-truth, correct, wrong, Diagnostic A,
unit-variance), and the derived ratios.

Wall-clock fit time is environment-dependent and is **not** part of
the reproducibility contract. The "~0.8 s" value recorded in section 7
is illustrative of this hardware; reruns may report a different
duration without violating reproducibility.

If a future commit produces different scientific values, this document
must be updated together with the probe.

### Commands and verification

The recorded values were produced by:

```
python inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py
```

Full pytest run alongside this probe: `337 passed, 1 skipped,
2 expected warnings` (the skip is the verified-SID scaffold; the two
warnings are the expected `RuntimeWarning`s from the inf-sigma residual
test, not from this probe).

## 6. True adjacency and true weights

True adjacency (row-source / column-destination, 1 = edge present):

```
[[0 1 0]
 [0 0 0]
 [1 1 0]]
```

True weights:

```
[[0.0000 0.5041 0.0000]
 [0.0000 0.0000 0.0000]
 [1.7861 0.5504 0.0000]]
```

Topological order: `(2, 0, 1)`. Intervention target = `2`
(topological source), intervention value = `+2.0`.

## 7. DAGMA learned continuous W

```
[[ 0.0004  0.5322  0.0001]
 [ 0.0004  0.0003  0.0002]
 [ 1.7004  0.4821  0.0003]]
```

Continuous-W values at every TRUE-edge position:

- `(0 -> 1)`: continuous_W = `+0.5322`, true_w = `+0.5041`
- `(2 -> 0)`: continuous_W = `+1.7004`, true_w = `+1.7861`
- `(2 -> 1)`: continuous_W = `+0.4821`, true_w = `+0.5504`

All three true-edge magnitudes are well above the project threshold
`0.3`, so the diagnostic-A sampler uses learned coefficients that
mirror the true SCM weights closely.

Fit-side scalars:
- `h_final = 5.40e-07`
- `score_final = +1.4888`
- Fit time on this fixture: ~0.8 s

## 8. DAGMA thresholded adjacency

Threshold = 0.3, applied to `abs(W_continuous)`:

```
[[0 1 0]
 [0 0 0]
 [1 1 0]]
```

This matches the true adjacency exactly.

## 9. graph_status and sampler_status

- `graph_status = "valid_dag"`
- `sampler_status = "available"`
- `sampler_unavailable_reason = None`

## 10. Residual sigma and W_sample summary

Residual sigma vector (model frame, `ddof=0`):
`[0.9971, 0.9930, 1.0006]`. These are consistent with the SCM's
unit-variance Gaussian noise model.

Learned W_sample (continuous_W masked by thresholded adjacency):

```
[[0.0000 0.5322 0.0000]
 [0.0000 0.0000 0.0000]
 [1.7004 0.4821 0.0000]]
```

## 11. SHD to true graph

`SHD = 0`. DAGMA recovered the true adjacency exactly on this fixture
under the default project threshold.

## 12. MMD results

### Monte Carlo floor (ground-truth pairwise)

`floor_mmd = -1.756334e-04` (negative; valid finite-sample behaviour of
the unbiased estimator on same-distribution comparisons).

### Wrapper-vs-truth (residual_fitted, learned structure)

`wrapper_vs_truth_mmd = +8.439610e-03`

Primary threshold (`wrapper_vs_truth_mmd <= 3 * floor_mmd`):

- Literal evaluation: `+8.439610e-03 <= -5.269003e-04` -> **FAIL**.
- Caveat: `floor_mmd` is negative, so `3 * floor_mmd` is not a meaningful
  positive acceptance criterion (mirrors C-P11's documented caveat).
- Order-of-magnitude comparison:
  - `|floor_mmd| = 1.756334e-04`
  - `|wrapper_vs_truth_mmd| = 8.439610e-03`
  - Ratio of absolute scales: ~48.
- For context, the wrong-structure MMD reported below is `+6.42e-01`,
  about 76 times the wrapper-vs-truth value. The wrapper's
  same-structure MMD therefore sits much closer to the floor's
  absolute scale than to a meaningfully wrong distribution's MMD.

### Correct vs wrong structure (fail-safe)

Deleted edge: `2 -> 0` (true `|weight| = 1.7861`, the strongest true
edge that is downstream of the intervention target and present in the
learned adjacency).

- `correct_mmd = +7.025987e-03`
- `wrong_mmd   = +6.417106e-01`
- ratio `wrong / correct = 91.334`
- Fail-safe inequality: `correct * 1.5 <= wrong` -> **PASS** (with
  large margin).

The wrapper sampler is therefore sensitive to structural errors.

### Diagnostic A: MMD under TRUE adjacency, DAGMA learned continuous-W

- Diagnostic-A `W_sample` and sigma are bitwise identical to the
  learned `W_sample` and sigma, because DAGMA's thresholded learned
  adjacency equals the true adjacency on this fixture.
- All true edges have `|continuous_W| >= 0.3`, so Diagnostic A does
  not collapse onto B2 by way of a near-zero true-edge weight.
- `true_struct_mmd = +5.332152e-03`

Diagnostic A is paired against `gt_wrong_paired`, whereas
wrapper-vs-truth is paired against `gt_paired`. The two values are
therefore not directly comparable bitwise, but both sit on the same
order of magnitude and both reflect coefficient-quality residual
error rather than structural error.

### Diagnostic B1 / B2: missing-edge augmentation

`missing_true_edges = {}`. Every true edge is present in DAGMA's
learned thresholded adjacency. B1 and B2 are therefore **not
applicable** on this fixture. No replacement diagnostic was invented.

### Unit-variance sensitivity (separate from primary result)

- `unit_variance_mmd = +8.432217e-03`
- delta vs `residual_fitted`: `-7.39e-06`

The unit-variance result is essentially indistinguishable from the
residual-fitted result on this fixture. This is consistent with the
residual sigma vector being close to `1.0` because the underlying SCM
has unit-variance noise.

## 13. Diagnostic A interpretation

Because DAGMA's thresholded learned adjacency equals the true
adjacency on this fixture, Diagnostic A is **degenerate as a
localisation tool** here. Its `W_sample` and sigma vector are
bitwise identical to the learned wrapper's, so running it amounts to
re-sampling from the same learned-structure generative process under
a different seed lane (`WRAPPER_TRUE_SEED_BASE = 2200` paired against
`GT_WRONG_SEED_BASE = 1200`, versus the learned `wrapper_vs_truth`
which uses `WRAPPER_SEED_BASE = 2000` paired against
`GT_PAIRED_SEED_BASE = 1100`).

The numerical differences among `wrapper_vs_truth_mmd` (`+8.44e-03`),
`correct_mmd` (`+7.03e-03`), and `true_struct_mmd` (`+5.33e-03`) on
this fixture therefore reflect Monte Carlo / MMD seed-lane variability
across paired batches, not independent structural or coefficient
evidence. On a fixture where DAGMA had instead missed or reversed a
true edge, Diagnostic A would have been informative; here it is not.

## 14. Diagnostic B1 / B2 interpretation

Not applicable: there are no missing true edges to augment.

## 15. Comparison to C-P11 at a high level

On the **same fixture**, same training data, same intervention, same
batch sizes, same seed protocol, same MMD function and bandwidth
rule, same median aggregation:

|                                | DAGMA (this probe)   | DCDI (C-P11)         |
|--------------------------------|----------------------|----------------------|
| `graph_status`                 | `valid_dag`          | `valid_dag`          |
| SHD to true                    | `0`                  | non-zero (strongest true edge missing) |
| `wrapper_vs_truth_mmd`         | `+8.44e-03`          | `+6.28e-01`          |
| true-structure (Diag A)        | `+5.33e-03`          | `+5.26e-02`          |
| augmented-structure (Diag B)   | not applicable       | `+4.23e-02`          |
| fail-safe `wrong/correct`      | `91.3`               | not the load-bearing test in C-P11 |

This is a controlled same-fixture comparison, not a general ranking.
The conclusion supported here is **only**: on this particular
fixture, the DAGMA wrapper produced a wrapper-vs-truth MMD about 75x
smaller (nearly two orders of magnitude smaller) than DCDI-G under
the protocol that C-P11 reported as failing for DCDI-G. The DAGMA
Diagnostic-A value also sits about ten times smaller than DCDI's
Diagnostic-A value on the same fixture.

**Interpretation of the 75x gap.** The dominant driver of this gap is
**structure recovery**, not sampler mechanics. DAGMA recovered the
true adjacency exactly on this fixture (SHD = 0); DCDI-G missed the
strongest true edge `2 -> 0` in C-P11. C-P11's own Diagnostic-A on
DCDI showed that MMD dropped by roughly an order of magnitude when
the true adjacency was forced. The 75x same-fixture gap therefore
should not be read as evidence that DAGMA's sampler mechanics are
intrinsically ~75x better than DCDI's; it is consistent with
"DAGMA found the right structure here and DCDI did not", multiplied
through the same downstream ancestral-sampling and MMD pipeline.

These same-fixture margins are large, but a single 3-node fixture
cannot decide the base model.

## 16. Interpretation

- DAGMA recovered the true adjacency exactly (SHD = 0) and the
  wrapper produced credible raw-unit samples on this fixture.
- The wrong-structure ratio is `wrong / correct = 91.3`. This large
  ratio reflects deletion of the **dominant** downstream true edge
  `2 -> 0` (true `|weight| = 1.7861`, the largest weight in the SCM),
  so it should be read as a structure-sensitivity sanity check —
  evidence that the sampler responds to a known-bad structural
  perturbation — rather than as a general sampler-quality metric.
  A weaker deleted edge would have produced a much smaller ratio.
- The literal C-P11 primary inequality
  `wrapper_vs_truth_mmd <= 3 * floor_mmd` evaluates as FAIL on this
  run because `floor_mmd` is negative; this is the documented
  negative-floor caveat carried over from C-P11. The substantive
  read is the absolute-scale gap, which is small relative to the
  scale at which a clearly wrong structure produces MMD.
- DAGMA's wrapper-vs-truth MMD is on the same order of magnitude as
  Diagnostic A. As noted in section 13, Diagnostic A is degenerate
  as a localisation tool on this fixture (the learned adjacency
  equals the true adjacency), so the numerical gap between
  `wrapper_vs_truth_mmd`, `correct_mmd`, and `true_struct_mmd` is
  Monte Carlo / MMD seed-lane variability, not independent
  structural or coefficient evidence.
- Diagnostics B1 and B2 are not applicable on this fixture because
  there are no missing true edges. This is a positive result for
  DAGMA on this fixture, but it also means the fixture **did not
  exercise** DAGMA's likely weak-edge and thresholding failure modes
  (sub-threshold true edges that survive in continuous-W, oracle-
  weight gaps on missing edges). Those failure modes remain
  untested here and are deferred to the full Doc 02 selection study.
- The unit-variance sensitivity matches the residual-fitted result
  to within ~1e-5, consistent with the SCM's unit-variance noise.
- Compared to C-P11's DCDI result on the identical fixture, DAGMA's
  wrapper-vs-truth MMD is about 75x smaller. As argued in section
  15, the dominant driver of that gap is structure recovery (DAGMA
  found the right adjacency here; DCDI did not), not sampler
  mechanics. This is a same-fixture diagnostic comparison, not a
  general model ranking, and does not constitute a selection of
  DAGMA as the base model. The full base-model decision still rests
  on the Doc 02 selection study, multi-seed results, criterion
  ordering, and verified SID.

## 17. Caveat

- This is a diagnostic probe, not a new main-study baseline.
- It does not change the main protocol.
- It does not replace verified SID integration; the SID gate
  remains open.
- It does not promote the equal-variance exhaustive Gaussian-BIC
  score (C-P12) to a baseline.
- It does not weaken any evaluator standard: no acceptance threshold
  was relaxed and no silent graph repair was introduced.
- The negative-floor caveat means the literal C-P11 inequality is
  not by itself sufficient evidence either way on its own.

## 18. Next steps

Result summary:

- DAGMA recovered the true adjacency exactly on this fixture
  (SHD = 0).
- The wrong-structure sanity check shows `wrong / correct = 91.3`
  (`correct_mmd = +7.03e-03`, `wrong_mmd = +6.42e-01`). The large
  ratio reflects deletion of the dominant downstream true edge
  `2 -> 0` and should be read as evidence that the sampler responds
  to a known-bad structural perturbation, not as a general
  quality metric.
- The literal C-P11 primary threshold remains non-informative on
  this run because `floor_mmd` is negative; the substantive
  absolute-scale read is recorded in section 12.
- Diagnostic A is degenerate as a localisation tool here
  (learned adjacency equals true adjacency); B1/B2 are not
  applicable (no missing true edges), so the fixture did not
  exercise DAGMA's likely weak-edge / thresholding failure modes.
- This is fixture-specific diagnostic evidence only. It does not
  replace the Doc 02 selection study or verified SID, a single
  3-node fixture cannot decide the base model, and this report
  does not select DAGMA as the base model.

Actions:

- DAGMA wrapper mechanics (Commits 1 to 8 of Doc 06) plus this probe
  are sufficient to draft `docs/phase_2c_dagma_readout.md` as the
  paired counterpart to `docs/phase_2b_dcdi_readout.md`. The readout
  should record:
  - that DAGMA fit, thresholding, graph status, residual noise,
    model-frame sampler, and raw-unit sampler are all implemented
    and tested;
  - that C-P13 recovers the true adjacency exactly on the same
    fixture where DCDI-G failed under C-P11, and that the
    structure-sensitivity sanity check passed against deletion of
    the dominant downstream true edge;
  - that Diagnostic A is degenerate on this fixture and Diagnostics
    B1/B2 are not applicable, so weak-edge and thresholding failure
    modes were not exercised here;
  - that this is fixture-specific evidence only, that the result
    is consistent with structure recovery rather than intrinsic
    sampler-mechanics superiority over DCDI, and that the full
    base-model selection study is still required.
- Doc 06 Commit 10 (diagnostics assembly and `get_diagnostics`) and
  Commit 11 (final readout and public API stabilisation) can proceed
  on the wrapper side. They do not depend on resolving the DCDI
  loss-hook pause.
- Verified SID integration remains a parallel project-level
  blocker before selection-study conclusions can be treated as
  scientifically complete.
- DCDI Commit 11 remains paused. Nothing in C-P13 changes that
  decision.
