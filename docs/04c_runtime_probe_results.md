# Runtime Probe Results for DAGMA-linear and DCDI-G

## Purpose

This document records the outcomes of running the safe runtime probes
specified in `docs/04c_runtime_probe_plan.md`. The probes were executed
inside the existing project virtual environment, on CPU, against the
inspected commit hashes (DAGMA `0886168`, DCDI `594d328`). All probe
scripts live under the untracked `inspection/probes/` directory.

No wrapper code was written. No project source, tests, or external
repository copy was modified. No dependency was installed, removed, or
upgraded.

Each probe entry below uses the same format: probe ID, status, what
was run, captured key output, interpretation, and decision implication.
A summary section appears at the end.

---

## DAGMA probe results

### D-P1. Import feasibility from local source clone

Status: PASSED

What was run: `inspection/probes/d_p1_dagma_import.py`. Added
`external/source_inspection/dagma/src` to `sys.path` and imported
`DagmaLinear` and `dagma.utils` from the inspected source.

Captured key output:

```
DAGMA import OK <class 'dagma.linear.DagmaLinear'>
utils.set_random_seed exists: True
```

Interpretation: the inspected DAGMA source loads cleanly inside the
project virtual environment without any missing transitive dependency.

Decision implication: subsequent DAGMA probes use the inspected source
rather than the installed `dagma==1.1.1` package, which keeps the
findings consistent with the inspection report.

### D-P2. Run DagmaLinear with explicit Doc 02 hyperparameters on tiny synthetic data

Status: PASSED

What was run: `inspection/probes/d_p2_dagma_fit_doc02.py`. Generated a
5-node ER DAG with 5 expected edges, sampled 200 observations of
Gaussian-linear data, and called `DagmaLinear(loss_type="l2").fit` with
`lambda1=0.05`, `w_threshold=0.3`, `T=4`,
`s=[1.0, 0.9, 0.8, 0.7]`, `mu_init=1.0`, `mu_factor=0.1`,
`warm_iter=2000`, `max_iter=4000`, `lr=3e-4`.

Captured key output:

```
X shape: (200, 5)
True nonzero edges: 5
W_est shape: (5, 5)
W_est dtype: float64
W_est nnz: 5
W_est min nonzero abs: 0.804342480982...
W_est max abs: 1.818326926711...
h_final stored on model: 1.0549411262e-06
```

Interpretation: the Doc 02 supplementary hyperparameter combination is
accepted without error and produces a 5x5 `float64` matrix with five
surviving edges. `h_final` is below `1e-5`, indicating the path-following
acyclicity constraint was satisfied at convergence.

Decision implication: the wrapper can pass explicit Doc 02 values
rather than relying on library defaults. The `h_final` and
`score_final` attributes are stored on the model and are accessible to
the wrapper as diagnostics.

### D-P3. Pre-threshold W preservation via w_threshold=0

Status: PASSED

What was run: `inspection/probes/d_p3_dagma_preserve_W.py`. Two
back-to-back fits on the same tiny SCM, identical seed and
hyperparameters, only `w_threshold` differs. Compared nonzero counts
under the refined check
`small_nonzero = (abs_W > 1e-12) & (abs_W < 0.3)` and the cross-check
`nnz at 0.0 >= nnz at 0.3`.

Captured key output:

```
nnz at threshold 0.0: 25
nnz at threshold 0.3: 5
count of small nonzero entries (strictly between 1e-12 and 0.3)
  at threshold 0.0: 20
0.0 retains >= 0.3 nonzeros? True
min nonzero abs entry at threshold 0.0: 3.316656946570799e-07
max abs |W0 - W03|: 0.16464282894765692
```

Interpretation: `w_threshold=0.0` retains all 25 off-diagonal entries
on this 5x5 problem (the full upper plus lower triangle excluding the
diagonal), with twenty of them strictly between `1e-12` and `0.3`.
These are precisely the entries that the `w_threshold=0.3` run would
have zeroed. The minimum nonzero absolute value at threshold `0.0` is
about `3.3e-07`, confirming that no entry was filtered out by another
path.

Decision implication: `w_threshold=0.0` is sufficient for the wrapper
to preserve the continuous `W` before project-level thresholding. The
subclass-and-override fallback is not needed.

### D-P4. Input data mutation check

Status: PASSED

What was run: `inspection/probes/d_p4_dagma_mutation.py`. A copy of
`X` was taken before `fit`, then the original `X` array was passed to
`fit` and compared to the copy afterwards.

Captured key output:

```
array values unchanged? False
X.mean(axis=0) after fit:  [-1.02e-16 -7.55e-17  8.44e-17  8.88e-18  6.11e-18]
X.mean(axis=0) before fit: [-0.0859 -0.0317  0.0678  0.0019  0.0032]
max abs(post-fit mean): 1.021405182655144e-16
max abs(X - X_before): 0.08585580643292623
```

Interpretation: DAGMA mutates `X` in place during the L2 mean-centering
step at `linear.py:307`. The post-fit per-variable means are all at the
order of `1e-16` (machine zero), and the maximum absolute difference
between the post-fit `X` and the pre-fit copy is about `0.086`, which
equals the maximum absolute pre-fit mean. The mutation is consistent
with the inspection finding.

Decision implication: the wrapper must always pass `X.copy()` to
`DagmaLinear.fit` to avoid silently corrupting upstream data. This
applies whether the wrapper uses the inspected source or the installed
`dagma` package.

### D-P5. Residual statistics for the noise and intercept policy

Status: PASSED (evidence only; no policy frozen)

What was run: `inspection/probes/d_p5_dagma_residuals.py`. Sampled 500
observations from a 5-node ER SCM, fit DAGMA with Doc 02 hyperparameters
and `w_threshold=0.3`, then computed residuals in the centred frame
against the thresholded `W_est`:
`R = (X - X.mean) - (X - X.mean) @ W_est`.

Captured key output:

```
residuals computed against thresholded W (w_threshold=0.3)
per-variable raw means of X: [ 0.0176 -0.0027 -0.0255 -0.0148  0.0004]
per-variable residual variance: [1.1125 1.0068 1.0237 1.0772 0.9272]
per-variable residual std:     [1.0547 1.0034 1.0118 1.0379 0.9629]
global mean residual var: 1.0294604263113498
global mean residual std: 1.0141345658868226
```

Interpretation: per-variable residual variances cluster tightly around
`1.0`, which matches the true unit-variance Gaussian noise used by the
SCM simulator. Per-variable raw means are within a few hundredths of
zero. The residual stats reported here are computed against the
thresholded `W_est`, because thresholding is what the wrapper-side
ancestral sampler would use as the structural model.

Decision implication: this is evidence only. Both candidate policies
remain plausible: (a) unit-variance Gaussian noise with zero
intercept, supported by the residual variances being close to 1 and
the means being close to 0, and (b) empirical residual-variance and
empirical-mean intercepts, which are richer but require the wrapper to
expose and persist these statistics. The Doc 02 amendment must decide
between them after broader experiments. This probe explicitly does NOT
freeze the policy.

---

## DCDI probe results

### C-P1. Targeted imports without dcdi.train

Status: PASSED

What was run: `inspection/probes/c_p1_dcdi_imports.py`. Added
`external/source_inspection/dcdi` to `sys.path` and imported
`LearnableModel_NonLinGaussANM`, `GumbelAdjacency`,
`compute_dag_constraint`, and `compute_penalty` directly from their
submodules.

Captured key output:

```
DCDI low-level imports OK
LearnableModel_NonLinGaussANM: <class 'dcdi.models.learnables.LearnableModel_NonLinGaussANM'>
GumbelAdjacency: <class 'dcdi.dag_optim.GumbelAdjacency'>
compute_dag_constraint: <function compute_dag_constraint at 0x...>
compute_penalty: <function compute_penalty at 0x...>
```

Interpretation: the wrapper-relevant DCDI modules import cleanly in the
project environment without going through `dcdi.train`.

Decision implication: the targeted-imports strategy from the inspection
report works. The wrapper does not need `cdt` to load these modules.

### C-P2. Verify cdt is not imported by the targeted import set

Status: PASSED

What was run: `inspection/probes/c_p2_dcdi_no_cdt.py`. `sys.modules`
diff before and after the four targeted imports.

Captured key output:

```
newly imported count: 1310
cdt-related newly imported: []
cdt.utils.R newly imported: []
newly imported dcdi modules: ['dcdi', 'dcdi.dag_optim', 'dcdi.models',
  'dcdi.models.base_model', 'dcdi.models.learnables', 'dcdi.utils',
  'dcdi.utils.gumbel', 'dcdi.utils.penalty']
```

Interpretation: zero `cdt`-related modules are pulled in by the
targeted imports. The 1310 newly imported modules are torch, numpy,
and their transitive dependencies that were not already loaded.

Decision implication: closes the C-10 question in
`docs/04b_source_inspection.md`. The DCDI wrapper, written against the
targeted import set, does not require `cdt` and does not require R.
The `cdt` and R chain is confined to `dcdi.train`, which the wrapper
must not import.

### C-P3. Instantiate LearnableModel_NonLinGaussANM in observational mode on CPU

Status: PASSED

What was run: `inspection/probes/c_p3_dcdi_instantiate.py`. Constructed
a 3-node, 2-hidden-layer, 8-hidden-dim model with `intervention=False`,
`intervention_type="perfect"`, `intervention_knowledge="known"`,
`num_regimes=1`.

Captured key output:

```
No intervention
instantiated: LearnableModel_NonLinGaussANM
num_vars: 3
num_layers: 2
hid_dim: 8
intervention: False
intervention_type: perfect
intervention_knowledge: known
```

Interpretation: model instantiates without errors on CPU FP32. The
`No intervention` print is from `BaseModel.__init__` and confirms the
observational-mode branch was selected. A deprecation warning about
`torch.set_default_tensor_type` was emitted by torch but does not
affect functionality.

Decision implication: the model is usable in the project environment
under observational settings. The wrapper should plan to use
`torch.set_default_dtype` and `torch.set_default_device` in place of
`torch.set_default_tensor_type` when the deprecation is enforced, but
that is a future refinement and not a blocker.

### C-P4. Access log_alpha and get_w_adj

Status: PASSED

What was run: `inspection/probes/c_p4_dcdi_native_edge.py`. Read
`model.gumbel_adjacency.log_alpha` and called `model.get_w_adj()`.

Captured key output:

```
log_alpha shape: (3, 3)
log_alpha dtype: torch.float32
log_alpha requires_grad: True
log_alpha[0, 1] init value: 5.0
log_alpha unique values: [5.0]
w_adj shape: (3, 3)
w_adj dtype: torch.float32
w_adj diagonal: [0.0, 0.0, 0.0]
w_adj[0, 1] off-diag value: 0.9933071732521057
w_adj unique off-diag values: [0.9933071732521057]
expected sigmoid(5) approx: 0.9933071732521057
```

Interpretation: `log_alpha` is a `(3,3) float32` parameter with
`requires_grad=True`, initialised to the constant `5.0` at every
entry. `get_w_adj()` returns `sigmoid(log_alpha) * (1 - I)`, so the
diagonal is exactly zero and every off-diagonal entry is
`sigmoid(5) approximately 0.9933`. All values match the inspection
findings in
`docs/04b_source_inspection.md` items C-4 and C-5.

Decision implication: the wrapper can read both `log_alpha`
(the native parameter `Lambda`) and `get_w_adj()` (the native
continuous edge object `P` with diagonal masked) exactly as the
inspection report described. The soft-prior loss hook can act on
`get_w_adj()` with full gradient flow.

### C-P5. forward_given_params on a tiny batch in eval mode

Status: PASSED

What was run: `inspection/probes/c_p5_dcdi_forward.py`. Inspected the
signature with `inspect.signature`, then attempted the call with no
mask and no regime.

Captured key output:

```
forward_given_params signature: (x, weights, biases, mask=None, regime=None)
Attempt 1 succeeded: x, weights, biases (mask=None, regime=None)
call pattern that worked: x, weights, biases (mask=None, regime=None)
density_params type: tuple
density_params length: 3
  density_params[0] shape: (4, 1)
  density_params[1] shape: (4, 1)
  density_params[2] shape: (4, 1)
```

Interpretation: the minimal call pattern `(x, weights, biases)` works
in observational mode without supplying `mask` or `regime`. The
function returns a tuple of three tensors, one per variable, each of
shape `(batch_size, num_params)`. For `LearnableModel_NonLinGaussANM`
the predicted parameter is the mean of the conditional Normal
(`num_params=1`); the standard deviation lives in `extra_params`.

Decision implication: closes the C-6 question in
`docs/04b_source_inspection.md`. `forward_given_params` can be used in
eval mode with the minimal signature and is the foundation for
wrapper-side ancestral sampling.

### C-P6. Conditional Normal construction and sampling

Status: PASSED

What was run: `inspection/probes/c_p6_dcdi_conditional.py`. Printed the
shape contract of `density_params` and `transform_extra_params` before
unbinding, then constructed and sampled from the per-variable Normal.

Captured key output:

```
density_params type: tuple
density_params length: 3
  density_params[0] shape: (4, 1)
extra_params_t type: list
extra_params_t length: 3
  extra_params_t[0] shape: (1,) value(s): [1.0]
after unbind dim 1: 1 tensors, shapes=[(4,)]
after extend with extra_params: 2 tensors, shapes=[(4,), ()]
dist type: Normal
dist.loc shape: (4,)
dist.scale shape: (4,)
sample shape: (4,)
first few values: [-1.339, -0.423, 0.053, 1.362]
```

Interpretation: unbinding `density_params[i]` along dim 1 produces one
tensor of shape `(batch_size,)`. Unbinding `extra_params_t[i]` along
dim 0 produces one scalar (a `()` tensor). `get_distribution` builds a
`torch.distributions.normal.Normal(loc=mean, scale=std)`. With initial
log-std `0.0` for every variable, `transform_extra_params` returns
`exp(0) = 1.0`, so the initial standard deviation is exactly `1.0`. A
sample can be drawn.

Decision implication: confirms the conditional Normal can be
constructed and sampled outside the original log-likelihood path. The
shape contract is now explicit and usable by the wrapper's sampler.

### C-P7. Minimal ancestral sampling sketch with one variable clamped

Status: PASSED

What was run: `inspection/probes/c_p7_dcdi_ancestral.py`. Forced a
3-node chain `0 -> 1 -> 2` by setting `model.adjacency` and saturating
`log_alpha`, then sampled `X_0` from its conditional, clamped
`X_1 = 0.5`, and sampled `X_2` given clamped parents.

Captured key output:

```
model.adjacency: [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
model.get_w_adj(): [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
applying do(X_1=0.5)
samples under do(X_1 = 0.5):
[[-1.25,  0.5, -0.46],
 [-0.31,  0.5, -1.64],
 [ 0.46,  0.5,  0.37],
 [-0.00,  0.5,  1.66],
 [-0.61,  0.5, -0.95]]
clamping invariant holds? True
X_2 mean: -0.205
X_2 std:  1.274
ancestral-sampling sketch SUCCEEDED
```

Interpretation: with the structural mask saturated to the forced DAG,
the wrapper-side ancestral-sampling sketch produces samples whose
target column (`X_1`) is exactly `0.5` in every row, and the downstream
column (`X_2`) shows variation consistent with the clamped parent.
There were no shape mismatches, no API call failures, and no
conceptual blocker.

Decision implication: closes the C-6 follow-up. The DCDI wrapper can
implement post-hoc sampling under arbitrary `do(X_j = v)` using
`forward_given_params` plus `get_distribution`, traversed in
topological order with target clamping. The eventual
`sampler_status` for the DCDI wrapper can be `available` once the
wrapper code packages this pattern, conditioned on the thresholded
DAG being acyclic.

### C-P8. Deterministic repeatability on CPU for a tiny controlled run

Status: PASSED

What was run: `inspection/probes/c_p8_dcdi_determinism.py`. Two
back-to-back five-step training runs on the same 32-sample tiny
problem, with `torch.manual_seed(0)`, `np.random.seed(0)`, default
CPU float tensors, RMSprop optimiser at `lr=1e-3`, and the same
augmented Lagrangian objective used by `dcdi.train`.

Captured key output:

```
identical? True
result1[0, 1]: 4.968373775482178
result2[0, 1]: 4.968373775482178
```

Interpretation: under fixed seeds and CPU float32, two identical
training-step sequences produce bit-identical `log_alpha` tensors at
the end of training. The off-diagonal value decreased from the
initial `5.0` to about `4.968`, consistent with a small number of
optimisation steps under the L1 penalty.

Decision implication: the DCDI wrapper can claim bitwise CPU
determinism for fixed-seed runs of small models in the project
environment, per wrapper contract Section 13. Whether this holds at
the full selection-study scale and on different hardware remains
REQUIRES EXECUTION, but the small-case behaviour is encouraging.

---

## Summary

### Pass / fail counts

- Passed: D-P1, D-P2, D-P3, D-P4, D-P5, C-P1, C-P2, C-P3, C-P4, C-P5,
  C-P6, C-P7, C-P8. Thirteen probes, thirteen passes.
- Failed: zero.
- Inconclusive: zero.
- Skipped: zero.

### Key questions closed

- DAGMA `w_threshold=0.0` is sufficient for preserving the continuous
  `W`. The wrapper does not need a subclass-and-override workaround.
- DAGMA mutates the input data in place during L2 mean-centering. The
  wrapper must always pass `X.copy()` to `fit`.
- Targeted DCDI imports avoid `cdt` entirely. The wrapper-relevant
  imports load cleanly without R or any `cdt` submodule.
- DCDI `forward_given_params` works in eval mode with the minimal
  signature `(x, weights, biases)`, mask and regime defaulting to
  `None`.
- DCDI wrapper-side ancestral sampling under `do(X_j = v)` is
  feasible. A five-line sketch reproduced the clamping semantics on a
  small forced DAG.
- DCDI sampler status can eventually be `available` once the wrapper
  code packages the ancestral-sampling pattern and guards it with a
  graph-validity check.
- DCDI CPU bitwise determinism holds on a tiny controlled run with
  seeds set. Production-scale determinism remains to be verified during
  the selection study.

### What remains unresolved

- The DAGMA noise and intercept policy for post-hoc interventional
  sampling. D-P5 produced evidence, not a policy. Doc 02 amendment
  needed.
- The DCDI sampler-related performance at the full selection-study
  scale, including how sample quality changes after a full training
  run. C-P7 used a hand-set saturated mask, not a learned one.
- DCDI bitwise determinism at production scale and on different
  hardware. C-P8 only proved a 5-step CPU case.
- The behaviour of the deprecated `torch.set_default_tensor_type` call
  used by `dcdi.main`. The wrapper should avoid calling it directly
  and instead use `torch.set_default_dtype` plus
  `torch.set_default_device`.

### Doc 02 amendments now supported by evidence

The following amendments to `docs/02_base_model_selection.md` are now
supported by evidence collected in this probe run. They are not made
here; they would be made in a separate amendment commit.

- Wrapper-level standardisation policy. Evidence: DAGMA always
  mean-centres, DCDI does not unless `normalize=True`. Recommended
  amendment: both wrappers consume raw SCM units; DAGMA mean-centering
  is left to DAGMA itself; DCDI runs with `normalize=False`.
- Defensive `X.copy()` requirement for the DAGMA wrapper. Evidence:
  D-P4 confirmed in-place mutation.
- Explicit DAGMA hyperparameter override at the call site. Evidence:
  the library default `T=5, lambda1=0.03, s=[1,.9,.8,.7,.6]` differs
  from Doc 02. The wrapper must always pass `T=4, lambda1=0.05,
  s=[1.0, 0.9, 0.8, 0.7]` explicitly.
- Pre-threshold continuous `W` saving policy. Evidence: D-P3 showed
  `w_threshold=0.0` preserves all entries; recommend the wrapper always
  fit with `w_threshold=0.0` internally, store the continuous
  `W_est`, and apply the `0.3` threshold at the wrapper output
  boundary.
- DCDI wrapper avoids importing `dcdi.train`. Evidence: C-P1 and C-P2
  showed that the targeted imports alone are sufficient and that
  `cdt`/R is never pulled in.
- DCDI sampler is feasible. Evidence: C-P5 through C-P7 demonstrated
  the full ancestral-sampling chain. `sampler_status = available`
  becomes a credible state for the DCDI wrapper once code is written.
- DCDI determinism statement. Evidence: C-P8 showed bitwise CPU
  determinism on a tiny case. The wrapper contract Section 13 statement
  is supported by evidence at small scale and should be re-checked at
  selection-study scale.

### What is NOT changed by this probe run

- No project source file was modified.
- No project test file was modified.
- `docs/01_research_question_and_commitments.md` is not modified.
- `docs/02_base_model_selection.md` is not modified.
- `docs/03_decision_log.md` is not modified.
- `docs/04_wrapper_api_contract.md` is not modified.
- `docs/04a_orientation_audit.md` is not modified.
- `docs/04b_source_inspection.md` is not modified.
- `docs/phase_1_readout.md` is not modified.
- The cloned external repositories under `external/source_inspection/`
  were not modified.
- No dependency was installed, removed, or upgraded.
- Probe scripts live under `inspection/probes/`, which is outside the
  tracked project tree and is intended to remain untracked.
