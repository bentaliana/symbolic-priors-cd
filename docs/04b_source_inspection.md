# Source Inspection for DAGMA-linear and DCDI-G

## Purpose

This document reports findings from a read-only source inspection of the
DAGMA and DCDI repositories cloned locally for inspection. The aim is to
answer the questions listed in `docs/04_wrapper_api_contract.md` Section 14
against the actual code, so that wrapper design can proceed on verified
facts rather than memory of the papers.

No wrapper code is written here. No external repository file is modified.
No dependency is installed or changed. No existing project document
(01, 02, 03, 04, 04a, phase 1 readout) is modified.

The orientation audit in `docs/04a_orientation_audit.md` established that
the project convention is row-source / column-destination, i.e.
`adjacency[i, j] = True` means a directed edge from node `i` to node `j`.
This document records whether each external repository uses the same
convention or a different one.

---

## Inspection metadata

### DAGMA

- Repository path inspected: `external/source_inspection/dagma`
- Remote URL: `https://github.com/kevinsbello/dagma.git`
- Commit hash inspected: `088616885d71b56c0573cd4902c1fcbac02e649f`
- License: Apache License, Version 2.0 (`LICENSE`, top of file)
- Top-level structure: `src/dagma/{linear.py, nonlinear.py, utils.py,
  locally_connected.py, __init__.py}`, `examples/`, `docs/`, `pyproject.toml`

### DCDI

- Repository path inspected: `external/source_inspection/dcdi`
- Remote URL: `https://github.com/slachapelle/dcdi.git`
- Commit hash inspected: `594d328eae7795785e0d1a1138945e28a4fec037`
- License: MIT License (`LICENSE.md`)
- Top-level structure: `dcdi/{main.py, train.py, data.py, dag_optim.py,
  prox.py, torchkit.py, plot.py, models/, utils/}`, plus baselines
  (`cam/`, `gies/`, `igsp/`, `jci/`) and `data/`, `main.py` at repo root

---

## How findings are classified

Each finding below is marked with one of:

- CONFIRMED: directly read from the source code, file and line references given
- INFERRED: a reasonable conclusion drawn from the source, but not literally stated
- UNRESOLVED: not answerable from source inspection alone
- REQUIRES EXECUTION: only answerable by actually running the code in the
  project environment, e.g. dependency import success or runtime determinism

---

## DAGMA-linear findings

### D-1. Where W is stored after fitting

CONFIRMED. `DagmaLinear.fit` writes the running and final estimate to the
instance attribute `self.W_est`.

- `external/source_inspection/dagma/src/dagma/linear.py` line 325 initialises
  `self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)` at the start
  of `fit`.
- Line 348 stores the inner-loop result back: `self.W_est = W_temp`.
- Lines 352 to 354 finalise: compute `self.h_final`, `self.score_final`,
  and apply the threshold `self.W_est[np.abs(self.W_est) < w_threshold] = 0`.
- `fit` returns `self.W_est` (line 355).

The pre-threshold matrix is overwritten in place by the threshold step at
line 354. The wrapper must save the continuous `W` before threshold if it
wants offline threshold robustness checks. This is straightforward because
the wrapper controls the fit call site.

### D-2. Where the loss is computed

CONFIRMED. The objective is `Q(W) + h(W)` with a path-following weighting
on `Q`.

- `_score(W)` at `linear.py` lines 37 to 60 computes the OLS or logistic
  score and its gradient.
- `_h(W, s)` at lines 62 to 81 computes the log-det acyclicity constraint
  and its gradient.
- `_func(W, mu, s)` at lines 83 to 104 returns the combined objective
  `obj = mu * (score + lambda1 * |W|_1) + h`.

Crucially, the actual training-time gradient is built inline in `minimize`
at lines 205 to 210, not by calling `_func` for differentiation:

```
G_score = -mu * cov @ (I - W)                       # for L2
Gobj = G_score + mu * lambda1 * sign(W) + 2 * W * M.T + mask_inc * sign(W)
```

This matters for the loss-hook capability: the additive L1 term is
hand-coded inside `minimize`. A wrapper that wants to inject an extra
penalty must either replicate this gradient or subclass `DagmaLinear` and
override `minimize`.

### D-3. Whether a custom additive loss term can be inserted

INFERRED. Yes, but with non-trivial surgery on `minimize`.

The natural insertion strategy is to subclass `DagmaLinear` and override
`minimize`, adding `grad_prior` to `Gobj` before the Adam step at line 210.
The wrapper would need to:

1. Subclass `DagmaLinear` and override `minimize`.
2. Replicate lines 205 to 210 with an extra `+ grad_prior(W)` term.
3. Keep `_score`, `_h`, `_func`, `_adam_update`, and `fit` unchanged.

The penalty term itself acts on `W` directly, which is plain `np.ndarray`,
so gradients are computed by the wrapper or supplied as a closed-form
analytic gradient (since L1 has a simple subgradient). For the planned
soft-prior method, the project only needs additive L1-style penalties, so
a closed-form gradient is straightforward.

There is no in-process callback API in DAGMA. There is also no public hook
for modifying the loss. The wrapper must own the gradient update.

DAGMA also already supports `exclude_edges` and `include_edges` parameters
(`fit` signature lines 245 to 246), which add a hard-zero mask and a
negative-L1 push on selected edges via `mask_inc` and `mask_exc` (lines
182 to 187, 215). These are not the soft-prior mechanism we need, but they
show that the codebase already has the pattern of multiplying or adding
edge-indexed masks inside the gradient step.

### D-4. What adjacency orientation the source uses

CONFIRMED. DAGMA uses the same convention as this project:
row-source / column-destination.

- `linear.py` line 52: `dif = self.Id - W`. The L2 score is
  `0.5 * trace(dif.T @ cov @ dif)`. This corresponds to the OLS objective
  for `X = X @ W + E`, where each column `j` of `W` holds the regression
  coefficients that predict `X_j` from the other variables. Therefore
  `W[i, j]` is the coefficient of `X_i` in the equation for `X_j`, i.e.
  row is parent, column is child.
- `utils.py` line 171 in `simulate_linear_sem`:
  `X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])`.
  Parents of node `j` are read from `G.neighbors(j, mode=ig.IN)` (line 170),
  and the corresponding weights are `W[parents, j]`. This confirms the same
  convention for the ground-truth simulator.
- `utils.py` line 297 in `count_accuracy`: `reverse` is computed by
  intersecting predicted edges with `B_true.T`, which only makes sense
  under row-source / column-destination.

No orientation transformation is needed at the DAGMA wrapper boundary.

### D-5. Whether the source standardises inputs

CONFIRMED. DAGMA mean-centres for the L2 loss, but does not scale by
standard deviation.

- `linear.py` line 306 to 307: under `loss_type == 'l2'`, the code applies
  `self.X -= X.mean(axis=0, keepdims=True)`. This mutates `self.X` in place
  before the covariance is computed at line 324.
- There is no variance scaling step. The covariance `self.cov = X.T @ X / n`
  reflects the centred data only.

Consequence: passing raw data into DAGMA already produces a centred
internal representation. The wrapper should expect that interventional
samples generated post-hoc using `W` and a synthetic noise policy will be
in the centred frame unless the wrapper adds back the per-variable means.

### D-6. How seeds are handled

CONFIRMED. DAGMA uses the global NumPy and Python random state.

- `utils.py` lines 8 to 10: `set_random_seed(seed)` calls
  `random.seed(seed)` and `np.random.seed(seed)`.
- `linear.py` line 360 in `test()` calls `utils.set_random_seed(1)` to
  seed the example run.

INFERRED. Inside `fit` itself, the only sources of stochasticity are:

- Adam updates (deterministic given fixed gradients).
- W initialised to zeros at line 325 (deterministic).
- The `np.linalg.inv` call inside `minimize` (deterministic at fixed input).

DAGMA does not appear to seed any random initialisation of `W` and does
not draw any internal random tensors during the fit. Therefore, given a
fixed `X` and fixed `lambda1`, `s`, `T`, `mu_init`, `mu_factor`, `lr`,
`beta_1`, `beta_2`, and `warm_iter`/`max_iter`, the output `W_est` is
deterministic on a fixed platform.

Note that `set_random_seed` mutates the global NumPy state. The wrapper
should avoid relying on this and should either pass an explicit RNG into
upstream data generation or restore NumPy state after calling DAGMA.

### D-7. Whether threshold 0.3 applies to abs(W) or W directly

CONFIRMED. The threshold is applied to absolute value.

- `linear.py` line 354: `self.W_est[np.abs(self.W_est) < w_threshold] = 0`.
- The default `w_threshold` in `fit` (line 234) is `0.3`.

This matches the project convention: DAGMA edges are present when
`abs(W_ij) >= 0.3`.

### D-8. Whether residual noise variance or intercepts are estimated

CONFIRMED. Neither residual noise variance nor intercepts are estimated.

- The L2 loss in `_score` only uses `self.cov` and `W`, no intercept
  parameter (lines 52 to 55).
- Mean-centering at line 307 absorbs the global mean shift, but there is
  no per-variable additive bias kept on the model.
- There is no `sigma`, `variance`, or `intercept` attribute set by `fit`.

INFERRED. Post-hoc interventional sampling using only the fitted `W` is
under-specified: it requires an explicit noise model and intercept
policy. The wrapper must freeze:

- The noise scale per variable (e.g., unit variance, or estimated from
  residuals X - X @ W).
- The intercept policy (zero, or the empirical mean of each variable on
  the raw training data).

This decision is not made by source inspection. It is recorded as a
DAGMA-side amendment item for Doc 02.

### D-9. Whether defaults match the selection-study protocol

CONFIRMED partial mismatch. The library defaults in `fit` are close to but
not identical to the Doc 02 supplementary defaults.

Doc 02 supplementary defaults for DAGMA-linear:

- T = 4
- mu^(0) = 1
- alpha (mu decay factor) = 0.1
- beta_1 (L1 coefficient) = 0.05
- s sequence = [1.0, 0.9, 0.8, 0.7]

Library defaults at `linear.py` lines 234 to 244:

- `w_threshold=0.3` matches Doc 02.
- `T=5` does not match (Doc 02 says 4).
- `mu_init=1.0` matches.
- `mu_factor=0.1` matches.
- `lambda1=0.03` does not match (Doc 02 says 0.05).
- `s=[1.0, 0.9, 0.8, 0.7, 0.6]` does not match (Doc 02 says 4 values, not 5).
- `lr=0.0003`, `warm_iter=3e4`, `max_iter=6e4` are not specified in Doc 02.

The wrapper must pass explicit values matching Doc 02 rather than relying
on library defaults. Note that the discrepancy between Doc 02's `T=4` with
a length-4 `s` and the library's `T=5` with a length-5 `s` is consistent:
both versions match `T` to `len(s)`.

### D-10. Whether DAGMA supports post-hoc interventional sampling

CONFIRMED no. The library has no sampling function. Sampling exists only
in `utils.simulate_linear_sem` for the ground-truth simulator (lines 99 to
172), which uses noise scales supplied by the caller. It does not use the
fitted `W_est`.

For wrapper-side reconstruction, the strategy is:

1. Threshold `W_est` and verify it is a valid DAG.
2. Decide a noise and intercept policy (UNRESOLVED).
3. Ancestral sample using the valid thresholded DAG and the chosen noise
   policy, clamping the intervention target.

The reconstruction code is mechanically simple. The wrapper does not need
to invoke DAGMA at all for this step.

---

## DCDI-G findings

### C-1. Code path for observational-only training

CONFIRMED. Observational-only training is selected when the `--intervention`
command-line flag is absent.

- `main.py` line 86: `parser.add_argument('--intervention', action="store_true", ...)`.
  Default is `False`.
- `main.py` lines 87 to 91: `if not opt.intervention: opt.intervention_type = "perfect"; opt.intervention_knowledge = "known"`.
  The model is constructed in the perfect-known branch but with
  `intervention=False`.
- `main.py` lines 110 to 118: when `opt.model == "DCDI-G"`, the model is
  `LearnableModel_NonLinGaussANM(... intervention=opt.intervention, ...)`.

So DCDI-G in observational-only mode is exactly
`LearnableModel_NonLinGaussANM(..., intervention=False)`.

### C-2. Whether this path matches DCD-no-interv

INFERRED. Yes. The README in the repository root and the paper supplement
use "DCD-no-interv" to mean DCDI run on data that ignores intervention
labels. The corresponding source code path is exactly the `intervention=False`
branch.

In `data.py` line 134: when `intervention=False` but `dcd=True`, the loader
points to the interventional dataset file `data_interv{i}.npy`. This is
the DCD baseline. The `dcd` flag is a separate flag (`main.py` line 88).

For our use, the wrapper sets `intervention=False` and uses purely
observational data, which is the standard non-baseline observational
configuration. This is sufficient for the selection study.

### C-3. Which loss components are active for observational-only

CONFIRMED. Active terms for observational-only training (intervention=False,
intervention_type=perfect, intervention_knowledge=known) are:

1. Negative log-likelihood `loss = -mean(log_likelihood)`. With
   `intervention=False`, `compute_loss` (`train.py` lines 38 to 56) takes
   the else branch and uses the full log-likelihood sum without masking.
2. Sparsity penalty `reg = opt.reg_coeff * compute_penalty([w_adj], p=1) / (d^2)`
   on `w_adj = get_w_adj() = sigmoid(log_alpha) * (1 - I)` (`train.py`
   lines 138 to 144, `base_model.py` line 191).
3. Intervention-sparsity penalty `reg_interv = 0` (`train.py` lines 146 to
   151). This is non-zero only when
   `opt.coeff_interv_sparsity > 0 and opt.intervention_knowledge == "unknown"`.
   With known and `intervention=False`, this term is zero.
4. Augmented Lagrangian DAG-constraint terms `gamma * h + 0.5 * mu * h^2`
   where `h = compute_dag_constraint(w_adj) / constraint_normalization`
   (`train.py` lines 138 to 157). `gamma` and `mu` are updated by the
   outer Lagrangian loop.

So the operative training objective is:
`L = NLL + reg + gamma * h + 0.5 * mu * h^2`.

### C-4. Where Lambda is stored

CONFIRMED. `Lambda` lives on `model.gumbel_adjacency.log_alpha`.

- `dag_optim.py` line 85: `self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))`.
- `dag_optim.py` line 98: `reset_parameters` initialises `log_alpha` to a
  constant `5` via `torch.nn.init.constant_(self.log_alpha, 5)`.
  `sigmoid(5)` is approximately `0.9933`, so all edges start near 1.
  This matches the Doc 02 description "adj matrix entries starting at or
  near 1.0".

`log_alpha` is a leaf `nn.Parameter`, so it carries gradients during
training.

### C-5. Where P = sigmoid(Lambda) is materialised

CONFIRMED.

- `dag_optim.py` lines 93 to 95: `GumbelAdjacency.get_proba()` returns
  `torch.sigmoid(self.log_alpha)`. This is `P` in the project naming.
- `base_model.py` line 191: `get_w_adj` returns `self.gumbel_adjacency.get_proba() * self.adjacency`.
  Since `self.adjacency` is initialised to `ones - eye` (`base_model.py`
  line 72), `get_w_adj` is `P` with the diagonal zeroed.

The training loop at `train.py` line 138 reads `w_adj = model.get_w_adj()`
and passes it to the L1 penalty and the DAG constraint. The post-training
thresholding at `train.py` line 210 also uses `model.get_w_adj() > 0.5`.

### C-6. Whether trained conditionals can sample arbitrary do(X_j = v)

CONFIRMED no built-in sampler. INFERRED that wrapper-side reconstruction
is feasible.

The trained model exposes:

- `compute_log_likelihood(x, weights, biases, extra_params, ...)`
  (`learnables.py` lines 46 to 74): given data `x`, returns per-example
  log-likelihoods.
- `forward_given_params(x, weights, biases, mask, regime)` (`base_model.py`
  lines 118 to 187): given data `x` and the parameters, returns the
  conditional density parameters `density_params` per node.
- `get_distribution(dp)`
  (`learnables.py` lines 102 to 103 for the Gaussian-ANM case): returns
  `torch.distributions.normal.Normal(dp[0], dp[1])`, a per-variable
  conditional Gaussian whose mean and std are MLP-predicted from parents.

There is no public method that takes a do(X_j = v) specification and
returns samples.

INFERRED. To sample under do(X_j = v), the wrapper can:

1. Threshold `w_adj > 0.5` to get the boolean adjacency.
2. Verify the thresholded adjacency is a valid DAG (acyclic, no self-loops).
3. Run ancestral sampling using the thresholded DAG as parent structure,
   feeding the parent values into `forward_given_params` (or a wrapper-
   replicated version) to get the conditional Normal parameters, and
   drawing each non-target variable from that Normal.
4. At the target node, set `X_j = v` and skip the conditional draw.

This is mechanically possible because the conditional Gaussian for each
node is fully exposed once weights and biases are available. The cost is
that the wrapper must re-implement the MLP forward pass conditioned on
parent values for the post-hoc sampling path. Inside the training graph
this is `forward_given_params`, but that function is heavily tied to the
DCDI training setup. The wrapper would likely call it in eval mode with
an artificially constructed batch where the target column is clamped.

UNRESOLVED. Whether `forward_given_params` can be invoked in eval mode
with a frozen mask M (all ones up to the diagonal) and a frozen
intervention regime is determinable only by experiment. This is recorded
as REQUIRES EXECUTION before sampling is committed to as supported.

### C-7. Adjacency orientation in DCDI

CONFIRMED. DCDI uses the same convention as this project: row-source /
column-destination.

The key code path is the first-layer einsum in `base_model.py` line 138:

```
x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
```

Index meanings:

- `t` is the target (child) node.
- `j` is the parent node.
- `b` is the batch dimension.
- `i` is the MLP output dimension.
- `l` is a broadcasting axis of size 1.

The mask `M[batch, parent, target]` controls whether parent `j` feeds the
MLP for target `t`. `adj[layer, parent, target]` is the gating structural
mask. Since `M = gumbel_adjacency(bs)` and `gumbel_adjacency.log_alpha` has
shape `(num_vars, num_vars)`, the first index of `log_alpha` is the parent
and the second is the target. That is row-source / column-destination.

The same convention is used in `train.py` line 207 when printing the
ground-truth adjacency next to `w_adj`, and in the SHD computation at
`train.py` line 446 which uses `dcdi/utils/metrics.py`.

No orientation transformation is needed at the DCDI wrapper boundary.

### C-8. Whether DCDI standardises inputs

CONFIRMED. DCDI standardisation is optional and off by default.

- `main.py` line 62: `parser.add_argument('--normalize-data', action="store_true", ...)`.
  Default is `False`.
- `data.py` lines 110 to 116: when `normalize` is True, the dataset is
  transformed as `(self.dataset - self.mean) / self.std`. The wrapper can
  set `normalize=False` to keep raw units. The training pass at
  `main.py` lines 101 to 107 also lets test data inherit `mean` and `std`
  from train data, preventing test-set leakage.

For fairness in this project, the wrapper should pass `normalize=False`
and apply any project-level standardisation in a documented and reversible
way upstream.

### C-9. Convergence criterion

CONFIRMED. The training loop uses a multi-stage augmented Lagrangian gate
plus patience-based stopping.

In `train.py`:

1. At each iteration, compute `h = compute_dag_constraint(w_adj) / constraint_normalization`
   (lines 138 to 139).
2. If `constraint_violation <= opt.h_threshold` and the discrete
   thresholded adjacency is acyclic (`train.py` lines 210 to 213, 269), set
   `first_stop`.
3. If the constraint is not yet satisfied, possibly update `gamma` and
   `mu` (lines 274 to 285).
4. Once the constraint is satisfied, decrease validation patience
   (lines 299 to 318). When patience reaches zero, threshold the model
   permanently by copying `log_alpha = +/- 100` (lines 320 to 333) and
   set `second_stop`.
5. After thresholding, run a final patience phase on validation NLL until
   `patience_thresh` reaches zero (lines 335 to 350), then save and
   return.

Default values from `main.py` argparse:

- `h_threshold = 1e-8`
- `mu_init = 1e-8`
- `mu_mult_factor = 2`
- `gamma_init = 0`
- `omega_gamma = 1e-4`
- `omega_mu = 0.9`
- `train_patience = 5`
- `train_patience_post = 5`
- `optimizer = "rmsprop"`
- `lr = 1e-3`
- `train_batch_size = 64`

These align with the Doc 02 description for DCDI-G:

- gamma_0 = 0: MATCH.
- mu_0 = 1e-8: MATCH.
- penalty update factor n = 2: MATCH (`mu_mult_factor`).
- decrease threshold = 0.9: MATCH (`omega_mu`).
- h-threshold 1e-8: MATCH.
- RMSprop with batch size 64: MATCH.
- lr 1e-3: MATCH.

### C-10. Whether DCDI works in the project environment

REQUIRES EXECUTION. The DCDI source depends on:

- `torch`: already in our project.
- `numpy`: already in our project.
- `scipy`: already in our project (via DAGMA).
- `networkx` (`utils/metrics.py` line 1): not currently a project dependency.
- `cdt` (Causal Discovery Toolbox), imported in `train.py` line 20 as
  `import cdt`, and line 26 as `from cdt.utils.R import RPackages, launch_R_script`.
  The latter import touches an R-integration submodule.
- `cdt.metrics.SID` is called in `train.py` line 445 for final reporting.

The `cdt` package is heavyweight and may require R for some operations.
The `from cdt.utils.R import ...` at import time is the risk: even if we
never call `launch_R_script`, the import may fail in environments without
the required `cdt` configuration.

INFERRED mitigation: the wrapper should not import `dcdi.train` directly.
Instead, import only the model and the low-level helpers:

- `from dcdi.models.learnables import LearnableModel_NonLinGaussANM`
- `from dcdi.dag_optim import compute_dag_constraint`
- `from dcdi.utils.penalty import compute_penalty`
- `from dcdi.dag_optim import GumbelAdjacency`

These do not require `cdt`. The wrapper then writes its own training loop,
which is independently desirable for loss-hook integration and fairness.

Whether the model files can be imported in the project environment without
side effects is REQUIRES EXECUTION.

### C-11. Whether a serialisable penalty specification is possible

INFERRED yes, if the wrapper ever needs to run DCDI in a separate
environment.

The soft-prior penalty is additive L1 acting on the native edge
representation. The penalty specification can be encoded as JSON:

- penalty type (`"L1"`)
- coefficient
- target operand (`"sigmoid_log_alpha"` for `P` or `"log_alpha"` directly)
- edge subset, as a list of `[i, j]` pairs in the row-source /
  column-destination convention
- confidence weights, as a parallel list of floats

The subprocess instantiates the model, reads this JSON, builds the
penalty term inside its own training loop, and proceeds. No Python
callable crosses the process boundary. This is feasible because the
penalty is fully described by static data.

Whether DCDI actually needs a separate environment is REQUIRES EXECUTION.

---

## Both repositories: cross-cutting findings

### B-1. License terms

CONFIRMED.

- DAGMA: Apache License 2.0. Permits commercial and modified use with
  attribution, includes a patent grant.
- DCDI: MIT License (per `LICENSE.md`). Permits commercial and modified
  use with attribution and the original license text.

Both are compatible with the thesis project. Modifications used at wrapper
level should retain the upstream license notices in any redistribution.

### B-2. Deterministic settings supported

CONFIRMED partial. INFERRED rest.

DAGMA:

- `utils.set_random_seed(seed)` sets `random.seed` and `np.random.seed`.
- Internal `fit` is deterministic given inputs because `W` starts at zero
  and Adam plus Newton-style updates are deterministic at fixed seeds.

DCDI:

- `main.py` lines 48 to 49: `torch.manual_seed(opt.random_seed)` and
  `np.random.seed(opt.random_seed)`.
- The Gumbel softmax samples (`dag_optim.py` line 86) draw from a
  `torch.distributions.uniform.Uniform(0, 1)` distribution. With seeds
  set, this is reproducible on a fixed device.
- INFERRED bit-exactness is platform-sensitive on GPU and may not hold
  bitwise even with seeds set, due to nondeterministic CUDA kernels.

The wrapper should use CPU-only training where possible to maximise
determinism, and set both seeds explicitly. Exact bitwise reproducibility
across hardware is REQUIRES EXECUTION to verify.

### B-3. Default config alignment with Doc 02

CONFIRMED.

DAGMA:

- Doc 02 says `T=4`, `lambda1=0.05`, `s=[1.0,0.9,0.8,0.7]`,
  `mu_init=1`, `mu_factor=0.1`, threshold `0.3` on `abs(W)`.
- Library defaults are `T=5`, `lambda1=0.03`, `s=[1.0,0.9,0.8,0.7,0.6]`,
  `mu_init=1`, `mu_factor=0.1`, threshold `0.3`.
- Mismatches: `T`, `lambda1`, length of `s`. The wrapper must pass
  explicit Doc 02 values.

DCDI:

- Doc 02 says `gamma_0=0`, `mu_0=1e-8`, xavier init, `log_alpha` near 1.0
  (i.e. sigmoid output near 1, which means `log_alpha` near 5),
  `lr=1e-3`, `mu_mult_factor=2`, `omega_mu=0.9`, `h-threshold=1e-8`,
  RMSprop with batch size 64.
- Library defaults match all of these. Confirmed by direct argparse
  reading.

### B-4. Risks for loss-hook integration

INFERRED.

DAGMA:

- The training-time gradient is hand-coded in `minimize`. Adding a
  penalty requires either subclassing and overriding `minimize` or
  monkey-patching it. Either way, the wrapper takes on responsibility
  for the full inner-loop gradient expression. This is a moderate
  maintenance burden but does not require modifying upstream code.
- DAGMA is numpy-only. Gradient flow of a penalty term needs the
  wrapper to supply an analytic gradient. For L1-style penalties on `W`
  this is trivial: `lambda_prior * sign(W) * confidence_mask`.

DCDI:

- DCDI is PyTorch-based, so autograd flows naturally through any added
  loss term. The wrapper can write its own training loop using the
  imported model and helpers, and add `reg_prior` to the existing
  augmented Lagrangian. This is the cleaner of the two integrations.
- Risk: the wrapper must avoid `dcdi.train`'s top-level imports that
  pull in `cdt` and its R dependency. Mitigated by importing the model
  and helpers directly.

### B-5. Risks for MMD sampling

INFERRED.

- Neither library has a built-in arbitrary-do sampler.
- DAGMA wrapper must own a noise and intercept policy and an ancestral
  sampler. The policy choice is UNRESOLVED.
- DCDI wrapper must own an ancestral sampler that re-uses the learned
  conditional Gaussians via `forward_given_params` or a wrapper-side
  replica. Whether `forward_given_params` can be cleanly invoked in eval
  mode with a frozen mask is REQUIRES EXECUTION.
- Both sampler paths can succeed only on a valid thresholded DAG. If the
  thresholded graph is cyclic, `sampler_status` must be
  `unavailable_invalid_graph`.

### B-6. Risks for wrapper fairness

INFERRED.

- DAGMA mean-centres the data internally; DCDI optionally normalises but
  does not by default. To compare fairly, the wrapper layer should pass
  raw data to both and decide one project-wide standardisation policy
  upstream. Doc 02 has not frozen this policy; recommended that both
  wrappers consume raw data with no internal standardisation.
- DAGMA optimises deterministically from `W = 0`; DCDI is stochastic via
  minibatches and Gumbel sampling. Seed discipline at the wrapper layer
  must control both. The selection study should record seeds in both
  cases.
- DAGMA returns a thresholded `W_est` numpy matrix. DCDI returns
  `model.adjacency` (the in-training mask after final thresholding).
  Both must be converted to a `bool` adjacency in the project convention
  inside the wrapper.
- DAGMA accepts `exclude_edges` and `include_edges` as hard constraints;
  DCDI does not. For fairness, the wrapper should not use these features
  in the main selection study or the soft-prior experiments. They would
  give DAGMA capabilities DCDI does not have. The hard-constraint baseline is a future main-study implementation decision andmust not be equated with post-threshold masking. A post-threshold mask is not thesame as a training-time hard constraint, because the model would still have been allowed to allocate explanatory weight to forbidden edges during optimisation. For the main study, any hard-constraint baseline should be designed explicitly and recorded before implementation.

### B-7. Mismatches between the wrapper contract and source reality

CONFIRMED.

- Wrapper contract requires "live training-time access to the native
  edge object so that a future prior penalty can act on it with gradient
  flow." DCDI satisfies this naturally via `model.gumbel_adjacency.log_alpha`.
  DAGMA satisfies it only if the wrapper subclasses `DagmaLinear` and
  overrides `minimize` to add the penalty gradient by hand. There is no
  PyTorch-style autograd hook in DAGMA.
- Wrapper contract requires the wrapper to expose a sampler-status field.
  DAGMA: sampler is `unavailable_unresolved_noise_policy` until the
  noise/intercept policy is frozen in Doc 02. DCDI: sampler may be
  `unavailable_no_api` or it may become `available` after the wrapper
  implements its ancestral-sampling routine and the inspection step
  REQUIRES EXECUTION confirms `forward_given_params` works in eval mode.
- Wrapper contract requires no silent graph repair. Both libraries
  produce thresholded outputs that may be cyclic (DAGMA can have
  near-zero spurious entries that survive threshold; DCDI thresholds
  `w_adj > 0.5` at end of training but the discrete mask is taken from
  `model.adjacency` which is multiplied by the thresholded `w_adj` and
  may not be acyclic if convergence stalls). The wrapper must verify
  acyclicity itself rather than assuming it.

---

## Resolved conclusions

These items are resolved by source inspection and can be acted on without
further verification.

- Both DAGMA and DCDI use the project's row-source / column-destination
  adjacency convention. No transformation is needed at the wrapper
  boundary.
- DAGMA threshold 0.3 applies to `abs(W)`. DCDI threshold 0.5 applies to
  `P = sigmoid(log_alpha)`. Both match the project defaults.
- DAGMA stores `W_est` as a numpy attribute and has no intercept or noise
  parameters.
- DCDI stores `Lambda` as `model.gumbel_adjacency.log_alpha`. The
  continuous edge object is `model.get_w_adj()`. Both are accessible as
  PyTorch tensors during training and after fit.
- DCDI default hyperparameters match Doc 02 for DCDI-G in observational
  mode. DAGMA library defaults differ from Doc 02 and must be overridden
  at the call site.
- DAGMA license is Apache 2.0; DCDI license is MIT. Both are compatible
  with thesis use.
- DCDI standardisation is off by default. DAGMA always mean-centres for
  L2 loss but does not scale.

## Unresolved decisions

These items must be resolved before wrapper implementation completes.

- DAGMA noise and intercept policy for post-hoc interventional sampling.
  Options include unit Gaussian noise, residual-fitted noise, and
  empirical-mean intercepts. This must be frozen in a Doc 02 amendment
  before MMD-based selection-study comparisons run.
- DCDI sampler availability under arbitrary `do(X_j = v)`. The mechanical
  reconstruction is feasible from the learned conditional Gaussians, but
  cleanly invoking `forward_given_params` in eval mode is REQUIRES
  EXECUTION before sampler_status can be declared `available`.
- DCDI environment compatibility. The `cdt` and R import chain in
  `dcdi.train` may not load in the project environment. Mitigation is to
  import only model and helper modules directly. Whether this works is
  REQUIRES EXECUTION.
- Whether DCDI bitwise determinism can be achieved on the project's
  hardware. REQUIRES EXECUTION.

## Wrapper feasibility

INFERRED from the findings above:

- DAGMA wrapper implementation appears feasible. The pieces required by
  the wrapper contract are present in the source. The main risks are
  (a) subclassing `DagmaLinear.minimize` for the soft-prior loss hook,
  which is invasive but bounded, and (b) the noise/intercept policy for
  post-hoc sampling, which must be frozen externally in a Doc 02
  amendment.
- DCDI wrapper implementation appears feasible. The pieces required by
  the wrapper contract are present in the source. The main risks are
  (a) the `cdt` and R import chain in `dcdi.train`, which is avoidable
  by importing model and helper modules directly, and (b) verifying that
  `forward_given_params` can be invoked in eval mode for the sampler
  path. Both risks reduce to REQUIRES EXECUTION items, not to source
  blockers.

Neither library has a built-in sampler under arbitrary `do(X_j = v)`. In
both cases the wrapper must own the ancestral-sampling reconstruction.
This is consistent with what the wrapper contract anticipated.

## What must be amended in Doc 02 before the selection study

These items follow from this inspection and should be added to Doc 02 via
the documented amendment process, not by editing this document or Doc 02
in this commit.

- DAGMA hyperparameters that diverge from library defaults must be
  enumerated in Doc 02 with explicit values (T, lambda1, s sequence).
- DAGMA noise and intercept policy for post-hoc interventional sampling
  must be frozen, including which noise scale and which intercept policy.
- DCDI standardisation policy at the wrapper boundary must be declared
  (recommended: `normalize=False` and pass raw data, with any project-
  level standardisation handled upstream).
- MMD-unavailable tie policy: what to do if DCDI sampler is unavailable
  while DAGMA and DCDI SID is within the 10 percent margin.
- Threshold robustness reporting: both wrappers must save the continuous
  edge object alongside the thresholded boolean adjacency so threshold
  sweeps can be reproduced offline.
- Calibration versus evaluation seed split: when DCDI calibration phase
  uses 2 seeds per config, the same seeds must be used for both
  candidates or explicitly justified otherwise.
- DCDI environment contingency: a written fallback in case `cdt` cannot
  be imported in the project environment.

---

## What is NOT changed by this inspection

- No file in `external/source_inspection/dagma` is modified.
- No file in `external/source_inspection/dcdi` is modified.
- No project source file is modified.
- No project test file is modified.
- `docs/01_research_question_and_commitments.md` is not modified.
- `docs/02_base_model_selection.md` is not modified.
- `docs/03_decision_log.md` is not modified.
- `docs/04_wrapper_api_contract.md` is not modified.
- `docs/04a_orientation_audit.md` is not modified.
- `docs/phase_1_readout.md` is not modified.
- No external dependency is installed or removed.
- No wrapper code or wrapper skeleton is created.
