# DCDI-G Wrapper Implementation Plan

## Status

Implementation plan only. No DCDI wrapper code, test code, or external
repository file is written by this commit. Implementation does not begin
until Doc 02 v1.3 and the supporting inspection/probe documents are
committed and this plan is reviewed.

### Execution status update

- Commits 1 through 9 are implemented and the corresponding pytest tests
  are green.
- **Commit 10 (sampler-quality validation) did NOT pass.** Both
  acceptance thresholds (`wrapper_vs_truth <= 3 * floor` and
  `correct * 1.5 <= wrong`) were missed by the actual fit. The
  observed values, learned vs true structure, and the additional
  diagnostic MMDs under the true adjacency and under the
  learned-plus-missing-strongest-edge adjacency are recorded in
  `docs/04f_dcdi_sampler_quality_diagnostic.md`. The pytest tests for
  Commit 10 have been removed from normal pytest collection and
  converted into the diagnostic probe
  `inspection/probes/c_p11_dcdi_sampler_quality_diagnostic.py`. No
  acceptance threshold has been weakened. The sampler-quality
  validation is **not** marked as passed.
- **Commit 11 (loss-hook injection) is paused pending project-level
  review** of the Commit 10 diagnostic findings.
- Commits 12, 13, and 14 remain blocked behind Commit 11.

This revision incorporates twelve review refinements: continuous-edge
preservation policy, tiered behavioural-equivalence tolerances,
sampler-quality validation that isolates sampler correctness from DCDI
fit error, explicit validation-data API, cleaner public method names,
loss-hook diagnostics, an explicit diagonal-zeroing test, a shared
threshold helper, sampler-quality test classification, an internal
module split, an updated commit sequence, and a final readiness
statement that names every previous blocker.

It also incorporates five v0.3 refinements: an equivalence-test
calibration artefact (`docs/04e_equivalence_calibration_results.md`)
cited by the equivalence test; justification of the wrong-structure
sampler-quality factor with high-impact edge selection; a preserved
continuous-edge-state test under induced sampler exceptions; commit 4
provisional-status wording; and a closing readiness summary stating
that implementation may proceed from commit 1 only, with commit 5 as
the gate.

And seven v0.4 corrections: device/dtype fix in the structural-mask
context manager; continuous-probability-range test wording corrected
to closed interval plus no-deliberate-saturation invariant; DCDI
finalisation deviation clarified to address potential MMD effects and
the evidence requirement; deterministic minibatch scheduling policy
added to the equivalence-test section; `config_snapshot` field added
to `WrapperDiagnostics`; commit 13 changed from optional to required;
and the closing readiness summary updated to v0.4.

---

## 1. Module and file structure

### New paths to create later

```
src/symbolic_priors_cd/wrappers/__init__.py
src/symbolic_priors_cd/wrappers/dcdi.py
src/symbolic_priors_cd/wrappers/status.py
src/symbolic_priors_cd/wrappers/preprocessing.py
src/symbolic_priors_cd/wrappers/_dcdi_training.py
src/symbolic_priors_cd/wrappers/_dcdi_sampling.py
src/symbolic_priors_cd/wrappers/_dcdi_utils.py
tests/test_dcdi_wrapper_interface.py
tests/test_dcdi_wrapper_training_equivalence.py
tests/test_dcdi_wrapper_thresholding.py
tests/test_dcdi_wrapper_sampler.py
tests/test_dcdi_wrapper_sampler_quality.py
tests/test_dcdi_wrapper_loss_hook.py
tests/test_dcdi_wrapper_preprocessing.py
```

### Module split rationale

Three internal modules instead of one umbrella file because the three
concerns are large enough to keep separate:

- `_dcdi_training.py`: augmented-Lagrangian training loop. Largest
  module by line count; line-by-line traceability against the inspected
  `dcdi/train.py:124-333` matters.
- `_dcdi_sampling.py`: structural-mask context manager, conditional
  sampling helpers, raw/model-frame transforms applied at sample time.
- `_dcdi_utils.py`: thin pieces that do not fit either of the above:
  model instantiation, parameter snapshotting helpers, diagnostics
  assembly. Underscore-prefixed to mark all three as internal.

A single utility file would mix three independent failure modes and
make commit 5 (training behavioural-equivalence) and commit 7 (sampling
mask context) hard to review independently.

### Public surface

`wrappers/dcdi.py` hosts the public `DCDIWrapper` class. `wrappers/status.py`
hosts the status `Literal`s and the `WrapperDiagnostics` `TypedDict`.
`wrappers/preprocessing.py` hosts `CentredOnlyTransform` and
`StandardisedTransform` plus the intervention-value transform.

`wrappers/__init__.py` re-exports:

```
DCDIWrapper, WrapperDiagnostics,
TrainingStatus, GraphStatus, SamplerStatus,
CentredOnlyTransform, StandardisedTransform
```

### Test layout

One test file per concern, matching the existing `tests/test_<topic>.py`
convention. `test_dcdi_wrapper_sampler_quality.py` is separate from
`test_dcdi_wrapper_sampler.py` because its tests are slower
(integration-scale).

### What is not added in this commit set

DAGMA wrapper, selection-study runner, experiment scripts under
`experiments/`, notebook code, and CLI entry points are all out of scope.

---

## 2. Wrapper responsibilities

The `DCDIWrapper` class will:

- Fit observational data only via `fit(X_train, X_val, ...)`. The wrapper
  does not accept interventional training data. The wrapper does not split
  internally and does not generate validation data; both `X_train` and
  `X_val` are caller-supplied (Section 9).
- Use low-level DCDI imports only:
  - `dcdi.models.learnables.LearnableModel_NonLinGaussANM`
  - `dcdi.dag_optim.{GumbelAdjacency, compute_dag_constraint}`
  - `dcdi.utils.penalty.compute_penalty`
- Never import `dcdi.train`. The wrapper writes its own training loop
  using the modules above.
- Expose native edge objects:
  - `native_edge_logits()` returns a CPU `torch.Tensor` clone of the
    preserved pre-threshold `log_alpha` (Section 3).
  - `native_edge_probabilities()` returns a CPU `torch.Tensor` clone of
    the preserved pre-threshold `get_w_adj()` (with diagonal exactly
    zero).
- Expose a thresholded boolean adjacency in row-source / column-
  destination convention via `thresholded_adjacency(threshold=0.5)`.
- Expose three independent status fields: `training_status`,
  `graph_status`, `sampler_status` per Doc 04 Section 7.
- Expose diagnostics via `get_diagnostics()` returning a
  `WrapperDiagnostics` `TypedDict`.
- Support future additive loss-hook capability via
  `set_loss_hook(hook, name="custom")`. One hook for now.
- Not perform preprocessing internally. The wrapper consumes already-
  transformed data and stores a reference to a fitted preprocessor for
  raw-unit roundtrips inside the sampler.

---

## 3. DCDI training-loop design

The training loop is in `wrappers/_dcdi_training.py` and replicates the
augmented-Lagrangian objective from
`external/source_inspection/dcdi/dcdi/train.py` lines 124-333, under
observational-only settings.

### Reused source-inspected objects

- `LearnableModel_NonLinGaussANM` (model architecture and parameters).
- `compute_dag_constraint` (acyclicity via matrix-exponential trace).
- `compute_penalty` (L1 penalty primitive).
- `GumbelAdjacency` used internally by the model.

### Intentionally avoided

- `dcdi.train.train`, `dcdi.train.retrain` (top-level entry points).
- `dcdi.plot`, `dcdi.utils.save` (plotting and persistence helpers).
- `monkey_patch_RMSprop`. The wrapper uses plain `torch.optim.RMSprop`.

### Continuous native-edge preservation

The wrapper preserves the learned continuous edge objects across the
entire fit. Concretely:

1. The training loop never permanently saturates `log_alpha`. The
   wrapper does NOT replicate the second-stop saturation step at
   `dcdi/train.py:320-333`. Termination occurs when the inner loop
   reaches `constraint_violation <= h_threshold` AND patience exhausts,
   without overwriting `log_alpha` with `+/- 100` values.
2. At the moment the training loop is about to exit, the wrapper stores:
   - `continuous_log_alpha_pre_threshold = model.gumbel_adjacency.log_alpha.detach().cpu().clone()`
   - `continuous_w_adj_pre_threshold = model.get_w_adj().detach().cpu().clone()`
3. All evaluator-facing thresholding (`thresholded_adjacency`), threshold
   robustness reporting (Doc 02 Section 7 item 5), and native-edge
   diagnostics use these two preserved tensors.
4. The saturation pattern used inside the C-P9 sampling context (Section 7)
   is applied to the live model parameters only, and is reverted before
   the sampler returns. The preserved continuous objects are never
   overwritten.

Documented deviation: the wrapper skips the post-second-stop saturation
and validation-NLL patience phase that DCDI's official loop runs at
`dcdi/train.py:320-333`. The deviation has the following consequences,
which must all be recorded in the wrapper module docstring:

- **Structural objects preserved:** the deviation does not change the
  preserved pre-threshold `continuous_log_alpha_pre_threshold` or
  `continuous_w_adj_pre_threshold`. These are the objects used for SID
  (once integrated), SHD, threshold robustness reporting, diagnostics,
  and future soft-prior work.
- **Potential effect on sampling quality:** skipping the official
  post-threshold finalisation may affect the quality of the learned
  conditional distributions used by the interventional sampler, and
  therefore may affect MMD. The distributions are not re-fitted after
  the structural mask is fixed.
- **Evidence that the deviation is acceptable:** the sampler-quality
  integration test (commit 10, Section 8) provides the primary evidence.
  If that test passes, the deviation is accepted for the selection study.
- **Fallback if evidence later shows material MMD impact:** a follow-up
  wrapper variant may preserve the continuous edge object at second stop
  and then run a separate finalised or post-threshold state specifically
  for sampling. This is not part of the current commit scope.

The equivalence test's iteration window (Section 4) is calibrated to
NOT reach the second-stop saturation point, so this deviation is
outside the scope of the equivalence test.

### Objective replicated in the wrapper

For each training iteration:

```
weights, biases, extra_params = model.get_parameters(mode="wbx")
log_lik = model.compute_log_likelihood(x_batch, weights, biases, extra_params)
nll = -log_lik.mean()
w_adj = model.get_w_adj()
h = compute_dag_constraint(w_adj) / constraint_norm
reg = reg_coeff * compute_penalty([w_adj], p=1) / (num_vars ** 2)
prior = loss_hook(w_adj) if loss_hook is not None else 0.0   # Section 5
aug = nll + reg + prior + gamma * h + 0.5 * mu * h**2
optimiser.zero_grad()
aug.backward()
optimiser.step()
```

`constraint_norm = compute_dag_constraint(ones(d,d) - eye(d))` is
initialised once.

### Lagrangian update schedule

Replicates `dcdi/train.py:269-296`:

1. Every `stop_crit_win` (default 100) step, evaluate validation NLL.
2. Every `2 * stop_crit_win` step, compute `delta_gamma` over the last
   three validation evaluations; set `delta_gamma = -inf` if monotone.
3. While `constraint_violation > h_threshold` or the discrete graph is
   cyclic: when `delta_gamma` plateaus, update gamma
   (`gamma += mu * h.item()`); if `constraint_violation` did not improve
   by factor `omega_mu`, `mu *= mu_mult_factor`.
4. Once `constraint_violation <= h_threshold` and discrete graph is
   acyclic, decrement `patience` per `stop_crit_win`. When patience
   reaches zero, capture the continuous edge objects (Section 3 above)
   and exit.

Doc 02 v1.3 default values are used.

### Seeds and deterministic settings

The wrapper sets `torch.manual_seed(seed)` and `np.random.seed(seed)`
inside `fit(...)` from the `seed` argument. CPU-only by default. The
wrapper does NOT call `torch.use_deterministic_algorithms(True)` because
that flag changes algorithm selection globally; probe C-P8 confirmed
bitwise CPU determinism at small scale without it.

---

## 4. Behavioural-equivalence requirement (tiered)

The wrapper's no-prior training loop must match a hand-replicated
reference loop derived from the inspected DCDI source. The reference
loop is implemented inside
`tests/test_dcdi_wrapper_training_equivalence.py`, with line citations
to `external/source_inspection/dcdi/dcdi/train.py`. `dcdi.train` is not
imported.

### Calibration artefact (frozen as a citable document)

The equivalence-test window (node count, `stop_crit_win`, iteration
count, seed, and observed gamma/mu update events) must not live only
in the test author's memory. As part of commit 5, a small calibration
probe runs against the wrapper's training loop and the reference loop,
and the result is recorded in a new document
`docs/04e_equivalence_calibration_results.md`. The probe is read-only
with respect to project source and external repositories; it follows
the same operating principles as the runtime probes in
`docs/04c_runtime_probe_plan.md` (lives under `inspection/probes/`,
no dependency install, CPU only).

The calibration document must record:

- the SCM configuration (node count, `expected_edges`, seed);
- training-data and validation-data shapes and seeds;
- `stop_crit_win`, `n_iter`, learning rate, batch size, and DCDIConfig
  values used;
- the iteration indices at which gamma-update and mu-update events
  fired;
- whether `h_threshold` was reached within `n_iter`;
- the final chosen equivalence-test window (node count, iteration count,
  seed, and the iteration indices at which assertions are made).

The behavioural-equivalence test file MUST cite this calibration
document by relative path in its module docstring or in a comment next
to the iteration constants. If the calibration document and the test
fall out of sync, the test fails CI; the test author updates the
calibration document and re-runs the probe.

### Validation setup

- SCM: 3 nodes, expected_edges = 3, project
  `generate_linear_gaussian_scm` at `seed=0`.
- Data: `X_train` of size 64 sampled at `seed=1`; `X_val` of size 64
  sampled at `seed=2`. Both observational.
- Model: `LearnableModel_NonLinGaussANM` with `num_layers=2`,
  `hid_dim=8`, `nonlin="leaky-relu"`, observational mode.
- Optimiser: `torch.optim.RMSprop`, `lr=1e-3`, batch size 32.
- Fixed seeds: `torch.manual_seed(0)`, `np.random.seed(0)` for both runs.

### Deterministic minibatch scheduling policy

The wrapper and the hand-replicated reference loop must consume the same
batch-index sequence at every iteration, or any divergence in the
trajectory is a test setup error rather than an acceptable model
deviation. The policy is:

- the batch-index sequence is generated once from the test seed
  (`np.random.seed(0)` plus `np.random.choice` or equivalent) and
  captured as a list of index arrays before either loop starts;
- both the wrapper loop and the reference loop index into `X_train`
  using this pre-generated sequence at each step;
- each loop must receive the same indices in the same order;
- different batch sequences are treated as test setup errors and must
  be fixed, not absorbed by tolerance.

The calibration artefact `docs/04e_equivalence_calibration_results.md`
must record the batch-index policy used (for example, random without
replacement using `np.random.RandomState(0)` with the given
`train_batch_size`), so the policy can be re-implemented exactly if
the test is revisited.

### Iteration count and h_threshold expectation

`h_threshold` convergence is NOT assumed on the tiny setting. The test
selects an iteration count that exercises at least one scheduled
gamma/mu update path on this setup, calibrated by a one-off probe run
before the test is committed. The probe will set:

- `stop_crit_win` to a small value such as 20 (smaller than the
  Doc 02 default of 100) so the gamma/mu update windows fire within a
  short run.
- iteration count to `n_iter = 6 * stop_crit_win = 120`, enough to reach
  the first `delta_gamma` evaluation (at iter `2 * stop_crit_win = 40`)
  and at least one update tick after that.

The exact iteration count, `stop_crit_win`, and node count are recorded
verbatim in the test docstring after the calibration probe.

If `h_threshold` is not reached within these iterations, the test still
compares trajectory and update schedule over the fixed iteration count.
A separate integration-style test (commit 13) exercises full convergence
and final-state semantics on a larger setting.

### Quantities compared per step

- Negative log-likelihood `nll`
- DAG constraint value `h`
- Sparsity penalty `reg`
- Augmented objective `aug`
- `log_alpha` (full tensor)
- `gamma` and `mu` after each Lagrangian-update event
- Final pre-threshold `get_w_adj()`

### Tiered tolerances

- **Early checks** (iterations 0, 1, 2, 5, 10): bitwise equality is the
  target. The test calls `torch.equal` on `log_alpha` and uses
  `==`-comparison on scalars. Bitwise failure here is treated as an
  implementation bug.
- **Mid-trajectory checks** (iterations 50, 100, 150 capped at the
  configured `n_iter`): `torch.allclose(atol=1e-4, rtol=1e-4)` on
  `log_alpha`; `abs(diff) <= 1e-4` on scalar quantities. Tighter
  tolerance accepted if the calibration probe shows it is stable.
- **Final-trajectory checks** at `n_iter`: elementwise
  `torch.allclose(atol=1e-3, rtol=1e-3)` on `log_alpha`, plus an L2-norm
  bound `||log_alpha_wrapper - log_alpha_reference||_2 / sqrt(d^2) <= 1e-3`.
  These tolerances absorb accumulated floating-point non-associativity
  across many RMSprop updates. The justification is recorded inline.
- **Lagrangian schedule equivalence**: every gamma-update event and
  every mu-update event MUST occur at the same iteration index in both
  loops. A schedule mismatch is an algorithmic divergence, not a
  numerical artefact, and fails the test regardless of trajectory
  closeness.

### Failure handling

A bitwise failure in the early window or a schedule mismatch is treated
as an implementation bug and blocks subsequent commits. A
late-trajectory deviation that exceeds the documented tolerance is
investigated rather than relaxed.

---

## 5. Loss-hook capability

### Where the penalty acts

The hook acts on `model.get_w_adj()` (`= sigmoid(log_alpha) * (1 - I)`)
during training. This matches the project's `P = sigmoid(Lambda)` naming
while structurally masking the diagonal.

### API

```
def set_loss_hook(
    self,
    hook: Callable[[torch.Tensor], torch.Tensor],
    name: str = "custom",
) -> None:
    """Register a single additive penalty acting on get_w_adj()."""
```

`hook` receives a `(num_vars, num_vars)` float tensor with `requires_grad=True`
and returns a scalar tensor. The wrapper adds `hook(w_adj)` to the
augmented Lagrangian before `.backward()`. `name` is stored in
`WrapperDiagnostics.loss_hook_name` for run records and ablation
comparisons.

Only one hook is supported. Adding multiple hooks would require an
associative combinator; deferred until Phase 6 explicitly needs it.

### Phase 6 compatibility

The hook signature is forward-compatible with the soft-prior thesis
penalty:
`sum_{(i,j) in F} c_ij * P_ij - sum_{(i,j) in O} c_ij * P_ij`.
The wrapper does not know about forbidden-edge sets or ordering
constraints; the prior is fully encoded inside the user-supplied
callable.

### Gradient-flow and shrinkage tests

See Section 11.

---

## 6. Thresholding and graph validation

### Public methods

```
def thresholded_adjacency(self, threshold: float = 0.5) -> np.ndarray:
    """Bool adjacency in row-source / column-destination convention."""
    return self._predict_adjacency_at(threshold)
```

### Shared private helper

```
def _predict_adjacency_at(self, threshold: float) -> np.ndarray:
    """Single source of truth for thresholding logic."""
    p = self._continuous_w_adj_pre_threshold.numpy()
    return (p >= threshold).astype(bool)
```

The threshold-monotonicity test, threshold-robustness reporting
(Doc 02 Section 7 item 5), and `thresholded_adjacency` all call this
helper. No duplicate thresholding logic anywhere in the wrapper.

### Validation steps performed inside `graph_status` machinery

- Shape `(num_vars, num_vars)` matches the model configuration.
- Dtype is strict bool.
- No diagonal entries True.
- No bidirected edges (no `(i, j)` and `(j, i)` both True for `i != j`).
- Acyclic (reuse `metrics/_graph_validation.py` patterns where possible
  and add the acyclicity check here).

### graph_status mapping

- `valid_dag`, `cyclic`, `bidirected`, `self_loop`, `invalid_shape` per
  Doc 04 Section 7. `self_loop` should never fire because `get_w_adj()`
  multiplies by `(1 - I)`; if it does fire, it is a wrapper bug.

### Effect on metric availability

- SID refuses non-`valid_dag` graphs; SID is recorded as unavailable
  with the reason.
- SHD is still computed but flagged as structurally invalid in the run
  record when `graph_status != "valid_dag"`.
- MMD requires a valid DAG. `sampler_status = unavailable_invalid_graph`
  in any other case.

---

## 7. Interventional sampler

### Save-mutate-sample-restore context manager (in `_dcdi_sampling.py`)

```
@contextmanager
def _structural_mask_context(model, a_thresh: np.ndarray):
    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()
    try:
        log_alpha = model.gumbel_adjacency.log_alpha
        device = log_alpha.device
        dtype = log_alpha.dtype
        mask_tensor = torch.as_tensor(a_thresh, dtype=dtype, device=device)
        saturated = mask_tensor * 100.0 + (1.0 - mask_tensor) * -100.0
        with torch.no_grad():
            model.adjacency.copy_(mask_tensor)
            model.gumbel_adjacency.log_alpha.copy_(saturated)
        yield
    finally:
        with torch.no_grad():
            model.adjacency.copy_(saved_adj)
            model.gumbel_adjacency.log_alpha.copy_(saved_log_alpha)
```

Note: `mask_tensor` and `saturated` are created on the same device and
with the same dtype as `model.gumbel_adjacency.log_alpha`. Do not
hardcode `dtype=torch.float32` or assume CPU placement inside the
context manager, even if CPU is the wrapper default.

This is the only place in the wrapper allowed to mutate `model.adjacency`
or `model.gumbel_adjacency.log_alpha`. The preserved continuous objects
(`continuous_log_alpha_pre_threshold` etc.) are never touched.

### Sampler sequence

For an `Intervention(target, value)` and `n_samples`:

1. Compute `a_thresh = self._predict_adjacency_at(0.5)`.
2. If `graph_status != "valid_dag"`, set
   `sampler_status = unavailable_invalid_graph` and return `None`.
3. Convert `value` from raw SCM units to model frame using the stored
   preprocessor (Section 9).
4. Enter `_structural_mask_context(self._model, a_thresh)`.
5. Inside the context, in topological order of `a_thresh`, for each node
   `j`:
   - If `j == target`, set `X[:, j] = v_model` (clamped).
   - Else call `forward_given_params(X, weights, biases)` with
     `mask=None, regime=None` and build the conditional Normal from
     `density_params[j]` and `transform_extra_params(extra_params)[j]`.
     Sample `X[:, j]`.
6. Exit the context manager.
7. Transform samples back to raw SCM units via the stored preprocessor.
8. Return `X_raw` of shape `(n_samples, num_vars)`.

### Restoration tests

Two tests:

1. Normal sampling: `model.adjacency` and `log_alpha` equal the pre-
   sampling clones after `sample_interventional` returns.
2. Induced exception: monkey-patch `forward_given_params` to raise mid-
   sample; restoration still happens in the `finally` block.

### MMD sampling RNG seed

`sample_interventional(..., sample_seed: int)` calls
`torch.manual_seed(sample_seed)` once at the top, before the context is
entered. The seed and derivation rule are logged in `WrapperDiagnostics`.

---

## 8. Sampler-quality validation

The clamping invariant alone is too weak. The sampler-quality test must
isolate sampler correctness from DCDI training fit error.

Retrospective note: C-P11 showed that this assumption did not hold for the tested linear-Gaussian diagnostic; sampler-quality validation was therefore converted into a diagnostic artefact rather than marked as passed.

### Why hand-construction is infeasible

The cleanest sampler-correctness test would inject a hand-constructed
DCDI model state whose conditional Normals match a known linear-Gaussian
SCM exactly, sample from it, and compare against ground truth. With
`LearnableModel_NonLinGaussANM`, this is INFEASIBLE in general: the
conditional means are predicted by per-variable MLPs with leaky-relu
activations and `num_layers=2`, `hid_dim=8`. Setting MLP weights to
implement the exact linear function `X @ W[:, j]` is not closed-form
because of the leaky-relu kinks; the closest exact representation
requires either disabling nonlinearities or a careful adversarial
construction that does not generalise to the production parameter
choice.

Recording this explicitly: the wrapper will not hand-construct a DCDI
state to match a known SCM. Two alternative tests are implemented
instead.

### Primary test: minimised-fit-error Monte Carlo floor (test file: `test_dcdi_wrapper_sampler_quality.py`)

Setup designed to minimise DCDI fit error so the residual signal is
sampler quality:

- SCM: 3-node ER2, project `generate_linear_gaussian_scm` at `seed=0`.
- Training data: 5000 observational samples. This is large for a 3-node
  problem and drives DCDI fit error to small values.
- Validation data: 500 observational samples (held out from training).
- Intervention: `do(X_1 = +2.0)` in raw SCM units.
- Sample size: 1000 per batch.
- `n_floor = 5` independent batches of ground-truth interventional
  samples drawn from the project SCM. Compute pairwise
  `mmd_rbf_unbiased` over the `binomial(5, 2) = 10` pairs and take the
  median. This is the Monte Carlo floor.
- `n_wrapper = 5` independent batches of wrapper-generated samples,
  each paired against a fresh ground-truth batch via
  `mmd_rbf_unbiased`. Take the median.
- Acceptance: wrapper-vs-truth MMD median <= `3 * floor`. The factor 3
  is justified by Monte Carlo variance plus residual DCDI approximation
  error on a 5000-sample fit; tighter (2x) is too brittle, looser (5x)
  loses diagnostic power.

### Fail-safe test: wrong-structure comparison (same test file)

Built into the same test module so it always runs alongside the primary
test:

- Build a wrong-structure sampler that uses the wrapper's
  `_structural_mask_context` with a deliberately wrong `a_thresh`.
- The deleted edge must be a **high-impact true edge on the
  intervention's downstream path**: specifically, among edges
  `(parent, child)` in the true DAG where `child` is reachable from the
  intervention target, pick the edge with the largest absolute true
  weight. This guarantees the wrong-structure sampler diverges from
  ground truth in a way the intervention's downstream support actually
  observes; deleting a peripheral edge would make the comparison
  uninformative.
- If no edge in the sampled SCM satisfies the criterion (for example
  the intervention target is a leaf and has no downstream path), the
  test resamples from a fixed fallback SCM seed listed in the test
  docstring. If even the fallback SCM has no usable downstream edge,
  the test uses a known hand-selected 3-node chain SCM
  (`X0 -> X1 -> X2`, true weights `[1.5, 1.5]`, intervention on `X_1`)
  with the deleted edge `X1 -> X2`. The hand-selected SCM is the
  final fallback and guarantees the test is always exercisable.
- Compute `mmd_rbf_unbiased` between (wrong-structure samples,
  ground-truth samples) and between (wrapper-true-structure samples,
  ground-truth samples). Use a common ground-truth batch for both
  comparisons within a paired draw so Monte Carlo noise cancels.

**Acceptance with justification:**

- Acceptance: `MMD(wrapper, truth) * 1.5 <= MMD(wrong_structure, truth)`,
  recorded in the test.
- Justification for the factor 1.5: a pragmatic margin that the
  correct-structure sampler is meaningfully closer to ground truth,
  not infinitesimally better. With `n=1000` per batch on a 3-node
  problem, Monte Carlo standard error on `mmd_rbf_unbiased` is
  typically of order `1e-3`. A 50 percent gap is much larger than that
  standard error and below the gap a high-impact edge deletion
  typically produces (which is usually a factor of several). The
  1.5 multiplier sits in the safe middle and discourages the wrong
  edge being chosen.
- If the 1.5 factor proves unstable during the implementation of
  commit 10 (more than 1 spurious failure in 20 reruns on the project
  hardware), the calibration outcome is recorded inline in the test
  docstring AND in a short note appended to
  `docs/04e_equivalence_calibration_results.md`. The factor may be
  adjusted only with explicit rationale; the adjusted value must be
  justified the same way (Monte Carlo SE budget vs effect size).

### Which test is binding

The primary Monte Carlo floor test is binding. The wrong-structure test
runs alongside as a fail-safe in case DCDI's fit error on a particular
seed inflates the floor comparison spuriously. If the primary test
fails but the wrong-structure test passes, the failure is investigated
before the wrapper is released, not silently downgraded.

### Test classification

`tests/test_dcdi_wrapper_sampler_quality.py` is marked as a slower
integration-style test. It is kept in its own file so the basic
sampler unit tests in `test_dcdi_wrapper_sampler.py` remain fast and
can be run on every commit.

---

## 9. Preprocessing policy and validation-data API

### Project-owned preprocessing

`wrappers/preprocessing.py` exposes:

```
class CentredOnlyTransform: ...
class StandardisedTransform: ...
```

Each class has `fit(X_train)`, `transform(X)`, `inverse_transform(X)`,
and `transform_intervention_value(value, target)`. Standardised stores
per-variable std; centred-only treats std as 1.0 conceptually.

### Validation-data API (frozen)

`DCDIWrapper.fit` signature (frozen by this plan):

```
def fit(
    self,
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    preprocessor: Union[CentredOnlyTransform, StandardisedTransform],
    seed: int,
    config: Optional[DCDIConfig] = None,
) -> None:
    ...
```

Rules:

- `X_train` and `X_val` must both already be in the model frame (the
  preprocessor has already transformed them). The wrapper does not call
  `preprocessor.fit` or `preprocessor.transform`.
- The wrapper does NOT split internally. If a caller does not have
  separate validation data, the caller (typically the selection-study
  runner) must produce it explicitly via its own held-out seed.
- The wrapper does NOT generate validation data.
- Tests provide tiny `X_train` and `X_val` arrays explicitly.

### DCDIConfig

```
@dataclass(frozen=True)
class DCDIConfig:
    h_threshold: float = 1e-8
    mu_init: float = 1e-8
    mu_mult_factor: float = 2.0
    gamma_init: float = 0.0
    omega_gamma: float = 1e-4
    omega_mu: float = 0.9
    lr: float = 1e-3
    train_batch_size: int = 64
    train_patience: int = 5
    stop_crit_win: int = 100
    reg_coeff: float = 0.1
    num_layers: int = 2
    hid_dim: int = 16
    nonlin: str = "leaky-relu"
```

Defaults match Doc 02 v1.3 Section 3.3 DCDI-G starting point. `config=None`
means use the dataclass defaults verbatim.

### Stored preprocessor

The wrapper holds a reference to the supplied `preprocessor` and uses
it inside `sample_interventional` to translate intervention values to
the model frame and to translate generated samples back to raw SCM
units. The wrapper never re-fits the preprocessor.

### No test-set leakage

Because preprocessing is owned by the caller, leakage prevention is
also the caller's responsibility. The wrapper's preprocessing tests
verify that the preprocessor classes themselves do not leak (their
`fit` runs on training data only and `inverse_transform` matches), and
that the wrapper's sampler honours the supplied preprocessor.

---

## 10. Diagnostics and logging

`get_diagnostics()` returns a `WrapperDiagnostics` `TypedDict` (frozen
shape by this plan):

```
class WrapperDiagnostics(TypedDict):
    training_status: TrainingStatus
    graph_status: GraphStatus
    sampler_status: SamplerStatus
    seed: int
    n_iterations: int
    config_snapshot: dict[str, object]          # resolved DCDIConfig values for this run
    loss_history: list[float]
    loss_decomposition_final: dict[str, float]  # nll, reg, prior, gamma, mu, h
    convergence_info: dict[str, object]         # first_stop, final_iter, converged
    continuous_log_alpha_pre_threshold: np.ndarray
    continuous_w_adj_pre_threshold: np.ndarray
    thresholded_adjacency: np.ndarray
    graph_invalid_reason: Optional[str]
    sampler_unavailable_reason: Optional[str]
    mmd_sampling_metadata: dict[str, object]    # sample_seed, transform mode, scaler stats
    loss_hook_name: Optional[str]
    numerical_tolerances: dict[str, float]
```

`config_snapshot` records the resolved `DCDIConfig` values used for the
run. If `config=None` was passed to `fit`, `config_snapshot` records the
dataclass defaults that were applied. This allows any run to be
reproduced from its diagnostics without needing the original call site.

Every key is present after a fit. Optional fields are explicitly `None`
when not applicable.

---

## 11. Tests required before wrapper completion

Grouped by purpose. Files are listed in Section 1.

### Interface/native-edge tests (`test_dcdi_wrapper_interface.py`)

- `test_dcdi_train_not_imported`: import the wrapper; assert
  `dcdi.train` not in `sys.modules`.
- `test_native_edge_shapes_and_dtypes`: shapes `(d, d)` and dtypes match
  the wrapper contract.
- `test_native_edge_probabilities_zero_diagonal_after_fit`: after a
  real fit, `native_edge_probabilities()` has exactly zero diagonal.
  This is the diagonal-zeroing test required by review item 7.
- `test_thresholded_adjacency_default_05`: `thresholded_adjacency()`
  equals `(native_edge_probabilities() >= 0.5)`.
- `test_thresholded_adjacency_uses_helper`: monkey-patch
  `_predict_adjacency_at` to assert it is the single source of
  thresholding logic.
- `test_mock_orientation`: inject a known `continuous_w_adj_pre_threshold`;
  `thresholded_adjacency` produces the expected boolean matrix in
  row-source / column-destination convention.
- `test_threshold_monotonicity`: predicting at thresholds
  `{0.4, 0.5, 0.6}` yields weakly decreasing edge counts.
- `test_diagnostics_completeness`: all `WrapperDiagnostics` keys present.

### Training/equivalence tests (`test_dcdi_wrapper_training_equivalence.py`)

- `test_deterministic_tiny_run`: two same-seed fits produce equal
  `continuous_log_alpha_pre_threshold`.
- `test_behavioural_equivalence_no_prior_early`: bitwise equality at
  iterations 0, 1, 2, 5, 10.
- `test_behavioural_equivalence_no_prior_mid`: `atol=1e-4, rtol=1e-4`
  at iterations 50, 100, 150 (capped at configured `n_iter`).
- `test_behavioural_equivalence_no_prior_final`: `atol=1e-3, rtol=1e-3`
  elementwise on `log_alpha` at `n_iter`, plus L2-norm bound.
- `test_lagrangian_schedule_equivalence`: gamma-update and mu-update
  iteration indices match between wrapper and reference loops exactly.

### Continuous-edge preservation tests (in `test_dcdi_wrapper_interface.py`)

- `test_continuous_log_alpha_not_saturated_after_fit`: no entry of
  `continuous_log_alpha_pre_threshold` equals `+100` or `-100`.
- `test_continuous_w_adj_range_and_diagonal`: all off-diagonal entries
  of `continuous_w_adj_pre_threshold` are in `[0, 1]`, the diagonal is
  exactly 0, and `continuous_log_alpha_pre_threshold` has not been
  deliberately saturated to `+/-100`. The important invariant is the
  absence of deliberate saturation, not strict open-interval
  probabilities.
- `test_continuous_edge_unchanged_by_sampling`: after several
  `sample_interventional` calls, the stored continuous edge tensors
  equal their post-fit values bitwise.
- `test_preserved_continuous_edges_unchanged_after_sampling_exception`:
  after fitting, clone the preserved
  `continuous_log_alpha_pre_threshold` and
  `continuous_w_adj_pre_threshold`. Monkey-patch a function inside the
  sampler path to raise mid-sample. Catch the exception and assert
  that BOTH preserved tensors equal their clones bitwise. This pairs
  with `test_restoration_after_induced_exception` (which asserts that
  the LIVE `model.adjacency` and `log_alpha` are restored) to cover
  both the live and the preserved state under exception.

### Graph-status and thresholding tests (`test_dcdi_wrapper_thresholding.py`)

- `test_graph_status_valid_dag`
- `test_graph_status_cyclic`
- `test_graph_status_bidirected`
- `test_graph_status_self_loop` (deliberately broken setup; should
  never fire under normal use)
- `test_invalid_graph_no_silent_repair`
- `test_sampler_status_invalid_graph`

### Sampler tests (`test_dcdi_wrapper_sampler.py`)

- `test_sampler_clamping`: target column exactly the requested raw
  value after inverse transform.
- `test_cp9_structural_masking`: extension of the C-P9 probe inside the
  sampler context, varying an excluded parent produces zero change in
  the target's predicted samples.
- `test_restoration_after_normal_sampling`: model state restored.
- `test_restoration_after_induced_exception`: restoration still happens
  in the `finally` block.
- `test_raw_unit_intervention_roundtrip_centred_only`
- `test_raw_unit_intervention_roundtrip_standardised`

### Sampler-quality tests (`test_dcdi_wrapper_sampler_quality.py`, integration)

- `test_sampler_quality_mc_floor`: primary test per Section 8.
- `test_sampler_wrong_structure_comparison`: fail-safe per Section 8.

### Preprocessing tests (`test_dcdi_wrapper_preprocessing.py`)

- `test_centred_only_roundtrip`
- `test_standardised_roundtrip`
- `test_intervention_value_transform`
- `test_no_double_transform_in_wrapper`: confirm the wrapper does not
  re-fit or re-transform.

### Loss-hook tests (`test_dcdi_wrapper_loss_hook.py`)

- `test_loss_hook_gradient_flow`: hook on a single entry changes the
  gradient by the expected amount.
- `test_loss_hook_behavioural_shrinkage`: a strong-L1 hook on a randomly
  chosen candidate edge (selected after a no-hook baseline) shrinks
  that entry in `native_edge_probabilities()` compared to the baseline.
- `test_loss_hook_name_in_diagnostics`: registering a hook with
  `name="forbidden_l1"` propagates that name into `WrapperDiagnostics.loss_hook_name`.

---

## 12. Atomic implementation order

14 commits. Commit 5 is the behavioural-equivalence gate; commit 4 is
where continuous-edge preservation is implemented and tested.

| # | Title | Files touched | Tests added | Acceptance | Risks |
|---|---|---|---|---|---|
| 1 | wrappers scaffolding | `wrappers/__init__.py`, `wrappers/status.py` | none | package imports cleanly; literals defined | trivial |
| 2 | preprocessing module | `wrappers/preprocessing.py` | `test_dcdi_wrapper_preprocessing.py` (4 tests) | round-trip identities hold within `1e-12` | low |
| 3 | DCDI low-level helpers | `wrappers/_dcdi_utils.py::make_dcdi_model`, snapshot helpers | `test_dcdi_train_not_imported` | model instantiates; `dcdi.train` absent from `sys.modules` | low; re-verifies C-P1 in CI |
| 4 | training loop + continuous-edge preservation | `wrappers/_dcdi_training.py` | `test_deterministic_tiny_run`, `test_continuous_log_alpha_not_saturated_after_fit`, `test_continuous_w_adj_range_and_diagonal`, `test_continuous_edge_unchanged_by_sampling` | two same-seed runs match; continuous edge preserved; no saturation introduced | MODERATE; preservation policy must be encoded correctly. **Provisional status:** commit 4 only verifies deterministic self-consistency and continuous-edge preservation. It does NOT prove the training loop matches DCDI. The training loop is not considered scientifically validated until the behavioural-equivalence gate in commit 5 passes. |
| 5 | behavioural-equivalence (gate) + calibration artefact | `tests/test_dcdi_wrapper_training_equivalence.py` (reference loop embedded with line citations); `inspection/probes/c_p10_equivalence_calibration.py`; `docs/04e_equivalence_calibration_results.md` (new doc) | `test_behavioural_equivalence_no_prior_early`, `_mid`, `_final`, `test_lagrangian_schedule_equivalence` | calibration probe runs and writes `docs/04e_equivalence_calibration_results.md`; test cites the calibration doc; early bitwise equality; mid `1e-4`; final `1e-3`; schedule indices match | HIGH; biggest single-commit risk. The calibration probe (probe ID C-P10) is sub-step of this commit; the calibration document and the test must be created together. |
| 6 | thresholding helper + graph validation | `wrappers/dcdi.py::_predict_adjacency_at`, `thresholded_adjacency`, graph_status machinery | `test_dcdi_wrapper_thresholding.py` (6 tests) | helper is single source of truth; all graph_status branches reachable | low |
| 7 | structural-mask context manager | `wrappers/_dcdi_sampling.py::_structural_mask_context` | `test_restoration_after_normal_sampling`, `test_restoration_after_induced_exception`, `test_cp9_structural_masking` | restoration normally and on exception; C-P9 invariance reproduces | moderate; corrupts model state if buggy |
| 8 | sampler core (no preprocessing yet) | `wrappers/dcdi.py::sample_interventional`, `wrappers/_dcdi_sampling.py::ancestral_sample_with_clamp` | `test_sampler_clamping` | target column clamped in model frame; shape/dtype correct | low |
| 9 | sampler raw-unit roundtrip | wire `preprocessing.py` into sampler | `test_raw_unit_intervention_roundtrip_centred_only`, `_standardised` | clamped value equals raw request after inverse transform | low |
| 10 | sampler-quality (integration) | `inspection/probes/c_p11_dcdi_sampler_quality_diagnostic.py`, `docs/04f_dcdi_sampler_quality_diagnostic.md` (was: `tests/test_dcdi_wrapper_sampler_quality.py`) | converted to diagnostic probe; no pytest collection | **DIAGNOSTIC FAILED / PAUSED.** Both acceptance thresholds missed. Observed values and full interpretation recorded in `docs/04f_dcdi_sampler_quality_diagnostic.md`. Original thresholds were not weakened. | resolved as a base-model / wrapper-design open question |
| 11 | loss-hook injection | `_dcdi_training.py::run_dcdi_training_loop` accepts `loss_hook`; `DCDIWrapper.set_loss_hook` | `test_loss_hook_gradient_flow`, `test_loss_hook_behavioural_shrinkage`, `test_loss_hook_name_in_diagnostics` | gradient changes match prediction; shrinkage observed; name propagates | **PAUSED** pending review of Commit 10 diagnostic |
| 12 | diagnostics and logging | `DCDIWrapper.get_diagnostics`, `WrapperDiagnostics` | `test_diagnostics_completeness`, `test_native_edge_probabilities_zero_diagonal_after_fit` | all keys present; diagonal exactly zero after fit | low |
| 13 | full-convergence integration | `tests/test_dcdi_wrapper_full_convergence.py` (required; marks as slow/integration) | one integration test that runs the wrapper to `h_threshold` on a small but non-trivial setting and verifies `convergence_info.first_stop` is set and the preserved continuous edge objects are non-saturated | training reaches second stop on the configured setting; `graph_status` is `valid_dag` or `cyclic` (documenting whichever occurs); continuous edge invariants pass | moderate; this test has a longer runtime than the unit tests; mark with pytest slow marker and exclude from the fast test suite |
| 14 | docstrings, public API stabilisation | `wrappers/dcdi.py` docstrings, `wrappers/__init__.py` re-exports | none | docstrings cite Doc 02 v1.3, Doc 04, Doc 04b-d | trivial |

Commit 5 remains the major gate. Commits 4 and 12 together cover the
continuous-edge preservation contract: commit 4 implements and tests
that the training loop never saturates; commit 12 makes those
quantities visible in `WrapperDiagnostics`.

---

## 13. Risks and unresolved questions

### Training-loop replication risk

- The augmented-Lagrangian state machine in `dcdi/train.py:269-296` is
  intricate. The reference loop is inlined in the test file with
  line citations to keep the replication auditable.
- The wrapper does not saturate `log_alpha` at second stop. This is a
  documented deviation. The equivalence test's iteration window is
  calibrated to NOT reach second-stop saturation, so the deviation is
  out of scope of the equivalence test.

### Dependency / environment risk

- `dcdi/train.py` pulls `cdt` and `cdt.utils.R`. The wrapper avoids this
  path; `test_dcdi_train_not_imported` re-verifies in CI.

### Performance risk

- DCDI training is CPU-only. The selection-study cell (10 nodes,
  1000 samples) should finish in reasonable time, but timing on the
  project hardware is not verified. The test suite uses small settings
  (3-5 nodes) to keep CI fast; commit 13's integration test is the
  only slow test in the wrapper suite.

### Sampler-quality risk

- The Monte Carlo floor test may need calibration of `n_floor`,
  `n_wrapper`, and the multiplier 3. The fail-safe wrong-structure
  test always runs alongside and provides a more robust signal.

### Loss-hook risk

- The behavioural-shrinkage test must pick a target edge dynamically
  from a no-hook baseline run so the chosen edge is non-zero in the
  baseline.

### Decisions resolved by this plan (no longer blockers)

1. Public class name: **`DCDIWrapper`**.
2. Public method names: **`fit(X_train, X_val, *, preprocessor, seed,
   config=None)`**, **`thresholded_adjacency(threshold=0.5)`**,
   **`native_edge_logits()`**, **`native_edge_probabilities()`**,
   **`sample_interventional(intervention, n_samples, *, sample_seed)`**,
   **`set_loss_hook(hook, name="custom")`**, **`get_diagnostics()`**.
3. `WrapperDiagnostics` field shape: frozen in Section 10.
4. Seed/config API: explicit `seed: int` keyword argument on `fit`;
   `config: Optional[DCDIConfig] = None` keyword argument carrying the
   tactical hyperparameters with Doc 02 defaults.
5. Validation-data API: caller-supplied `X_train` and `X_val` explicitly
   (no internal split, no internal generation).
6. Continuous native edge preservation: never permanently saturate
   `log_alpha`; store `continuous_log_alpha_pre_threshold` and
   `continuous_w_adj_pre_threshold` at the end of training; all
   evaluator-facing thresholding reads from these two tensors.

### Remaining uncertainty (informational, does not block coding)

- Whether the equivalence-test calibration probe produces a clean
  iteration window (`stop_crit_win=20`, `n_iter=120` is the proposed
  starting point but may need adjustment if no gamma/mu update fires
  on this seed).
- Whether the primary sampler-quality test (`3 * floor`) is tight
  enough to be useful without spurious failures. The plan tolerates
  either binding outcome (primary alone OR primary plus fail-safe
  wrong-structure test together).

These two items are calibration tasks executed during commits 5 and 10
respectively. They do not require pre-approval.

---

## 14. Verification (after implementation)

```
.venv/Scripts/python -m pytest tests/ -v
```

Existing 129 tests must still pass; new wrapper tests must pass; total
around 165 to 180 tests with 1 skipped (the SID pre-registered
scaffold).

The behavioural-equivalence test (commit 5) is the load-bearing
verification step. The sampler-quality test (commit 10) provides the
second-strongest validation signal. Commits 4 and 12 together
guarantee continuous-edge preservation, which is a Doc 02 v1.3
requirement.

---

## 15. Out of scope

DAGMA wrapper, selection-study runner, soft-prior implementation,
verified SID integration, performance optimisation of the training
loop, documentation amendments to Doc 02. All of these are tracked
separately.

---

## 16. Critical paths (existing files referenced)

- `external/source_inspection/dcdi/dcdi/models/learnables.py`
- `external/source_inspection/dcdi/dcdi/models/base_model.py`
- `external/source_inspection/dcdi/dcdi/dag_optim.py`
- `external/source_inspection/dcdi/dcdi/utils/penalty.py`
- `external/source_inspection/dcdi/dcdi/train.py` (read-only reference)
- `src/symbolic_priors_cd/data/_sampling.py` (project ancestral-sampling
  pattern reused for documentation)
- `src/symbolic_priors_cd/metrics/_graph_validation.py` (reuse where
  applicable inside graph_status machinery)
- `src/symbolic_priors_cd/metrics/interventional.py::mmd_rbf_unbiased`
  (used by sampler-quality tests)

---

## Closing readiness summary

The plan is **approved for implementation after these v0.4 refinements
are applied**.

Implementation should begin with **commit 1 only**, and proceed strictly
in the order documented in Section 12. Each commit must pass its
acceptance criteria before the next commit is started. Commits beyond
commit 1 are not pre-approved; each commit is reviewed in turn.

**Commit 5 remains the main gate.** Commit 5 carries both the
behavioural-equivalence test against the hand-replicated reference loop
AND the calibration artefact `docs/04e_equivalence_calibration_results.md`
required by refinement 1. If commit 5 fails, downstream commits (6
through 14) MUST stop, and the plan must be revisited before
implementation resumes. The continuous-edge preservation introduced in
commit 4 is provisional until commit 5 passes.

The five blockers previously flagged are resolved by this plan:

1. **Public class and method names**: frozen in Section 13.
   `DCDIWrapper`, `thresholded_adjacency`, `native_edge_logits`,
   `native_edge_probabilities`, `sample_interventional`,
   `set_loss_hook`, `get_diagnostics`.
2. **`WrapperDiagnostics` field shape**: frozen in Section 10.
3. **Seed/config API**: frozen as `fit(X_train, X_val, *, preprocessor,
   seed, config=None)` with `DCDIConfig` dataclass.
4. **Validation-data API**: frozen as caller-supplied `X_train` and
   `X_val`; wrapper does not split, does not generate validation data.
5. **Continuous native edge preservation policy**: frozen in Section 3.

Two calibration items are folded into the relevant commits:

- the equivalence-test iteration window is calibrated in commit 5 by
  probe C-P10 and recorded in
  `docs/04e_equivalence_calibration_results.md`;
- the wrong-structure sampler-quality multiplier (1.5) is justified in
  Section 8 with a Monte Carlo SE budget, and may be re-calibrated in
  commit 10 only with explicit rationale recorded in the test
  docstring and appended to `docs/04e_equivalence_calibration_results.md`.

No further plan-level decisions are pending. Implementation may begin
with commit 1 after human review once these v0.4 corrections are applied.

Deferred preprocessing hardening for Commit 9 or Commit 12:
- add read-only mean_ and std_ properties to CentredOnlyTransform and StandardisedTransform;
- ensure returned arrays are copies, not mutable references;
- add feature-count validation for transform/inverse_transform;
- add target-index validation for transform_intervention_value;
- optionally add training-data standardisation self-check: transformed training data has mean 0 and std 1 with ddof=0.