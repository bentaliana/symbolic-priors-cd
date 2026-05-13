# DCDI Sampler-Quality Diagnostic (C-P11)

## Status

- Commit 10 of `docs/05_dcdi_wrapper_implementation_plan.md`
  (sampler-quality validation) **did NOT pass**. The pytest tests that
  enforced the original acceptance thresholds have been removed from
  normal pytest collection and converted into this diagnostic artefact.
- Commit 11 (loss-hook injection) is **paused pending project-level
  review** of the findings recorded here.
- No acceptance threshold has been weakened. The original thresholds
  (`wrapper_vs_truth <= 3 * floor` and
  `correct * 1.5 <= wrong`) are recorded verbatim against the observed
  values below.

## Purpose

Capture the exact setup, observed MMD values, and interpretation of the
DCDI sampler-quality failure so the project can decide whether to revise
the wrapper design, the base-model selection, or the experimental SCM
configuration before any further DCDI wrapper commits are written.

## Probe

- Probe script: `inspection/probes/c_p11_dcdi_sampler_quality_diagnostic.py`
- Environment: project `.venv`, CPU only
- Re-running the probe on the same project commit and same environment
  must reproduce the recorded values verbatim. If they drift, this
  document must be updated together with the probe.

## Configuration (frozen)

### SCM

- `generate_linear_gaussian_scm(n_nodes=3, expected_edges=3, seed=0)`
- True adjacency (row-source / column-destination):

  ```
  [[0 1 0]
   [0 0 0]
   [1 1 0]]
  ```

- True weights (rounded to 4 decimals):

  ```
  [[0.0000 0.5041 0.0000]
   [0.0000 0.0000 0.0000]
   [1.7861 0.5504 0.0000]]
  ```

- Topological order: `(2, 0, 1)`
- True edge list with magnitudes: `2 -> 0` (1.7861), `2 -> 1` (0.5504),
  `0 -> 1` (0.5041)

### Data

- `X_train`: 5000 observational samples, `sample_observational` seed = 1
- `X_val`: 500 observational samples, `sample_observational` seed = 2
- Preprocessing: `CentredOnlyTransform` fitted on `X_train`; the same
  fitted preprocessor is reused for the interventional roundtrip during
  sampling.

### Model and DCDI training

- `make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8,
  nonlin="leaky-relu")`
- `DCDIConfig()` paper defaults: `h_threshold=1e-8`, `mu_init=1e-8`,
  `mu_mult_factor=2.0`, `gamma_init=0.0`, `omega_gamma=1e-4`,
  `omega_mu=0.9`, `lr=1e-3`, `train_batch_size=64`,
  `train_patience=5`, `stop_crit_win=100`, `reg_coeff=0.1`.
- `seed = 0`, `n_iter = 30000`.

### Intervention

- `do(X_2 = 2.0)` (target = source of the true topological order).

### Sampling

- Per-batch sample size: 1000 in raw SCM units (via
  `sample_raw_units_dcdi`).
- Ground-truth interventional batches: drawn through `intervene()` and
  `_ancestral_sample`.
- `n_floor = 5`, `n_wrapper = 5`.
- Seed bases:
  - `GT_FLOOR_SEED_BASE = 1000`
  - `GT_PAIRED_SEED_BASE = 1100`
  - `GT_WRONG_SEED_BASE  = 1200`
  - `WRAPPER_SEED_BASE        = 2000`
  - `WRAPPER_WRONG_SEED_BASE  = 2100`
  - `WRAPPER_TRUE_SEED_BASE   = 2200`  (diagnostic A)
  - `WRAPPER_AUGMENTED_SEED_BASE = 2300` (diagnostic B)

## Observed values

### Training outcome

- Training time: 146.3 s
- `n_iterations`: 30000
- `final_h`: 5.3119e-04 (above the configured `h_threshold = 1e-8`)
- `final_mu`: 2.6844e+00
- `final_gamma`: 4.4865e+00
- `converged`: False (patience never elapsed because `final_h` did not
  drop below `h_threshold` within `n_iter`)

### Learned continuous w_adj

```
[[0.000e+00 7.669e-01 5.000e-03]
 [1.600e-03 0.000e+00 3.000e-04]
 [2.300e-01 9.092e-01 0.000e+00]]
```

### Learned thresholded adjacency at 0.5

```
[[0 1 0]
 [0 0 0]
 [0 1 0]]
```

- Edges retained: `0 -> 1` (`w_adj = 0.7669`), `2 -> 1`
  (`w_adj = 0.9092`).
- Edge `2 -> 0` (`w_adj = 0.2300`, true weight magnitude 1.7861) does
  not cross the 0.5 threshold and is missing from the learned
  structure. This is the strongest true edge by absolute weight.
- `graph_status`: `valid_dag`.

### Monte Carlo floor

- Median pairwise MMD across 5 ground-truth interventional batches:
  `floor_mmd = -1.756334e-04`.
- The unbiased estimator is allowed to be negative when both samples
  come from the same distribution; this value indicates the typical
  scale of MMD noise at this batch size, on the order of `1e-4`.

### Wrapper-vs-truth (original Commit 10 MC-floor comparison)

- Wrapper batches: `seed_base = 2000`.
- Ground-truth batches paired: `seed_base = 1100`.
- `wrapper_vs_truth_mmd = +6.275322e-01`.
- Original acceptance: `wrapper_vs_truth <= 3 * floor_mmd = -5.27e-04`.
  Observed gap: about 6.3e-01 above zero versus a floor threshold at
  about -5.3e-04. The substantive gap is 3+ orders of magnitude
  irrespective of sign (see "Negative-floor caveat" below).

### Correct vs wrong structure

- Deleted edge: `2 -> 1` (the strongest true downstream edge that is
  also present in the learned adjacency; true `|weight| = 0.5504`).
- `correct_mmd = +6.283445e-01`
- `wrong_mmd   = +7.536961e-01`
- `wrong / correct ratio = 1.199`
- Original acceptance: ratio `>= 1.5`. Observed: ratio 1.199.

### Diagnostic A: MMD under the TRUE adjacency

- The same fitted DCDI model is used; sampling forces
  `model.adjacency = scm.adjacency` via `_structural_mask_context`.
- Wrapper batches: `seed_base = 2200`. Paired GT: `seed_base = 1200`.
- `true_struct_mmd = +5.259294e-02`.

### Diagnostic B: MMD under learned + strongest missing true edge

- Strongest true edge missing from `a_thresh`: `2 -> 0` (`|weight|`
  1.7861). Adding it to the learned adjacency yields the same DAG as
  the true adjacency in this fixture; `aug_status = valid_dag`.
- Augmented adjacency:

  ```
  [[0 1 0]
   [0 0 0]
   [1 1 0]]
  ```

- Wrapper batches: `seed_base = 2300`. Paired GT: `seed_base = 1200`.
- `augmented_mmd = +4.227701e-02`.

### Summary table

| Quantity                                    | Value           |
| ------------------------------------------- | --------------- |
| graph_status (learned at 0.5)               | valid_dag       |
| Monte Carlo floor MMD                       | -1.756334e-04   |
| wrapper-vs-truth MMD                        | +6.275322e-01   |
| correct-structure MMD                       | +6.283445e-01   |
| wrong-structure MMD                         | +7.536961e-01   |
| wrong / correct ratio                       | 1.199 (< 1.5)   |
| true-structure MMD (Diagnostic A)           | +5.259294e-02   |
| augmented-structure MMD (Diagnostic B)      | +4.227701e-02   |

## Interpretation

### Negative-floor caveat

The unbiased MMD estimator `mmd_rbf_unbiased` is allowed to take
negative values when both arguments come from the same distribution
(this is a known property of the unbiased U-statistic form). On this
fixture the floor sits at `-1.76e-04`, so `3 * floor` is `-5.27e-04`,
which is a value smaller than the floor itself. Using `3 * floor` as a
positive acceptance threshold is therefore not meaningful when the
floor is negative.

The substantive comparison is the order-of-magnitude gap between the
floor scale (`~1e-04` in absolute value) and the wrapper-vs-truth MMD
(`+6.3e-01`). That gap is over three orders of magnitude regardless of
sign, so the failure does not depend on the choice of acceptance
formula.

### Where the failure most likely sits

The two added diagnostics localise the failure to learned structure
quality rather than to sampler mechanics or DCDI conditional quality:

- Under the **TRUE adjacency** (Diagnostic A), with DCDI's learned
  conditionals reused unchanged, the wrapper-vs-truth MMD drops from
  `+6.28e-01` to `+5.26e-02`. That is a 12x reduction.
- Under the **learned + missing strongest true edge** (Diagnostic B,
  which in this fixture coincides with the true adjacency), the MMD is
  `+4.23e-02`. Both diagnostic comparisons sit roughly two orders of
  magnitude above the noise floor (`~1e-04`) but more than an order of
  magnitude below the original wrapper-vs-truth result.

This pattern is consistent with: DCDI's per-node conditional
distributions are usable, but the thresholded structure at 0.5 is
missing the strongest true edge (`2 -> 0`, true `|weight| = 1.7861`),
and that single structural omission dominates the sampler-vs-truth gap.
The sampler-mechanics tests (clamping, deterministic seeding,
structural masking, restoration on exception, raw-unit roundtrip) all
remain green in the normal pytest suite.

### Markov equivalence and equal-variance identifiability

A standard remark for linear-Gaussian SCMs is that the conditional
independence structure of observational data identifies the DAG only up
to its Markov equivalence class. This caveat is real for the **general**
linear-Gaussian setting and is the reason DCDI's official training adds
a post-second-stop `log_alpha` saturation step that crystallises the
chosen orientation.

However, the project SCM here uses the same noise scale on every
variable (`noise_scale = 1.0` for all three nodes). In the
**equal-error-variance linear-Gaussian** setting, the DAG is
identifiable from observational data (Peters and Buhlmann, 2014;
Loh and Buhlmann, 2014). The current failure is therefore not a
theoretical-impossibility result: methods specialised for equal-variance
linear-Gaussian discovery (e.g. score-based approaches under that
assumption) are expected to recover the true DAG on this kind of data.

DCDI-G is not specialised for the equal-variance linear-Gaussian case.
It optimises an augmented-Lagrangian objective over a general nonlinear
Gaussian-ANM with per-variable MLPs, and its inductive bias does not
encode the equal-variance assumption. The failure observed here is most
naturally read as a base-model fit/identification mismatch on this
particular data family, not as a sampler bug and not as a
fundamental impossibility.

### What this does not prove

- It does not prove DCDI-G is unfit for the thesis main study. The
  selection study in the broader plan compares DAGMA-linear and DCDI-G
  on the same evaluator; that comparison is the appropriate place to
  decide which base model is used.
- It does not prove that the wrapper's documented deviation
  (skipping post-second-stop `log_alpha` saturation) is or is not the
  root cause. A controlled experiment that turns the saturation step
  back on would be needed to attribute the structural-recovery gap.
- It does not weaken any acceptance threshold for the project. The
  original Commit 10 thresholds are recorded as-failed.

## What this artefact did NOT change

- No `src/`, `tests/test_dcdi_wrapper_*.py`, or
  `external/source_inspection/` source file was edited as part of this
  diagnostic.
- No dependency was installed, removed, or upgraded.
- No acceptance threshold was relaxed.
- No silent graph repair was introduced.
