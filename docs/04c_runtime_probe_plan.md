# Runtime Probe Plan for DAGMA-linear and DCDI-G

## Purpose

This document specifies a small set of disposable runtime probes that
answer the REQUIRES EXECUTION items left open by
`docs/04b_source_inspection.md`. The probes are not wrapper
implementations and do not depend on any wrapper code. Each probe is
small enough to run in seconds and is designed to leave no persistent
trace in the project tree or environment.

This plan does not run any probe. It only specifies what would be run,
why, what success looks like, and what failure means.

## Operating principles

- Probe scripts, if and when run, live under an untracked top-level
  directory `inspection/probes/`. They do not live under `src/`.
- Probes use `sys.path.insert(0, "external/source_inspection/dagma/src")`
  and `sys.path.insert(0, "external/source_inspection/dcdi")` to import
  from the inspected commit hashes (DAGMA `0886168`, DCDI `594d328`),
  not from any installed package version. This keeps probe results
  consistent with the inspection report.
- No probe writes a file under `src/`, `tests/`, `docs/`, or
  `external/source_inspection/`.
- No probe modifies the cloned external repositories.
- No probe installs, removes, or upgrades a Python dependency.
- Every probe must be runnable inside the existing project virtual
  environment (`.venv`) without elevation.
- Every probe runs on CPU. No probe assumes GPU.
- Probes that mutate Python global state inside the running process
  (for example `torch.set_default_tensor_type` or `np.random.seed`) are
  acceptable because the state vanishes when the probe process exits.

## How to read each probe entry

Each probe entry below states:

- Purpose
- Script sketch (not the final code, but enough that a reviewer can
  see what would run)
- Expected success condition
- Failure interpretation
- Environment changes
- Dependencies required
- Files produced
- Decision the probe outcome informs

At the end of this document, every probe is also classified as one of:

- safe to run now
- requires approval because it may install dependencies or alter the
  environment
- should not be run yet

---

## DAGMA probes

### D-P1. Import feasibility from local source clone

Purpose: confirm the inspected DAGMA source loads cleanly inside the
project virtual environment, so subsequent DAGMA probes have a known
import path.

Script sketch:

```
import sys
sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils
print("DAGMA import OK", DagmaLinear)
```

Expected success: prints the `DagmaLinear` class object without raising.

Failure interpretation: an `ImportError` would mean a transitive
dependency is missing in the project environment. The likely missing
items are `igraph` or `tqdm`, which are already pyproject-listed, so
failure here is unlikely. If failure occurs, the wrapper plan must
fall back to the installed `dagma` package version (already a project
dependency) and the inspection commit hash needs re-checking against
the installed version.

Environment changes: only `sys.path` inside the probe process. No
persistent change.

Dependencies required: none beyond what is already installed.

Files produced: none.

Decision informed: confirms that subsequent DAGMA probes can use the
inspected source rather than the installed package.

### D-P2. Run DagmaLinear with explicit Doc 02 hyperparameters on tiny synthetic data

Purpose: confirm that a fit call with the Doc 02 supplementary
hyperparameters completes end-to-end without raising on a tiny problem.

Script sketch:

```
import sys, numpy as np
sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")

model = DagmaLinear(loss_type="l2")
W_est = model.fit(
    X,
    lambda1=0.05,
    w_threshold=0.3,
    T=4,
    mu_init=1.0,
    mu_factor=0.1,
    s=[1.0, 0.9, 0.8, 0.7],
    warm_iter=2000,
    max_iter=4000,
    lr=3e-4,
)
print("W_est shape", W_est.shape, "nnz", int((W_est != 0).sum()))
```

Expected success: returns a `(5, 5)` numpy array without raising. Edge
counts may differ from the ground truth; this probe does not check
recovery quality.

Failure interpretation: an exception suggests the supplied combination
of hyperparameters is not accepted by the library. The wrapper would
then need to investigate which value is rejected.

Environment changes: only `sys.path` and global numpy random state
(set by `set_random_seed(0)`).

Dependencies required: none beyond what is already installed.

Files produced: none.

Decision informed: confirms the wrapper can pass the explicit Doc 02
values rather than relying on library defaults.

### D-P3. Pre-threshold W preservation via w_threshold=0

Purpose: verify that calling `fit` with `w_threshold=0` returns a
continuous `W` that retains entries the `w_threshold=0.3` run would
have zeroed, confirming the simplest preservation strategy for the
wrapper.

The probe must NOT use the naive check `abs(W) < 0.3`, because zeros
also satisfy that. Use a strict-positive lower bound and compare nonzero
counts between the two runs on the same tiny setup.

Script sketch:

```
import sys, numpy as np
sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

def fit_with(thresh):
    utils.set_random_seed(0)
    B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
    W_true = utils.simulate_parameter(B)
    X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")
    model = DagmaLinear(loss_type="l2")
    return model.fit(
        X.copy(),
        lambda1=0.05, w_threshold=thresh, T=4,
        s=[1.0, 0.9, 0.8, 0.7],
        warm_iter=2000, max_iter=4000, lr=3e-4,
    )

W0   = fit_with(0.0)
W03  = fit_with(0.3)
abs_W0 = np.abs(W0)
small_nonzero = (abs_W0 > 1e-12) & (abs_W0 < 0.3)

print("nnz at threshold 0.0:", int((abs_W0 > 1e-12).sum()))
print("nnz at threshold 0.3:", int((np.abs(W03) > 1e-12).sum()))
print("count of small nonzero entries kept by 0.0:", int(small_nonzero.sum()))
print("0.0 retains >= 0.3 nonzeros?",
      int((abs_W0 > 1e-12).sum()) >= int((np.abs(W03) > 1e-12).sum()))
```

Expected success: `nnz at threshold 0.0` is greater than or equal to
`nnz at threshold 0.3`, proving nothing was zeroed by `w_threshold=0`.

Note: the absence of small nonzero entries is not by itself a failure.
The L1 penalty and DAGMA path-following may legitimately drive small
entries to numerical zero before the threshold step. What matters is
that `w_threshold=0` does not remove anything that survives the
optimisation.

Failure interpretation: if `nnz at threshold 0.0` is strictly less
than `nnz at threshold 0.3` on identical setups, then `w_threshold=0`
is filtering small entries through some other path. The wrapper must
then subclass `DagmaLinear` to capture `self.W_est` before the
threshold step at `linear.py:354`.

Environment changes: only `sys.path` and global numpy random state.

Dependencies required: none.

Files produced: none.

Decision informed: confirms the simplest preservation strategy. The
fallback is a subclass-and-override approach.

### D-P4. Input data mutation check

Purpose: confirm that DAGMA mutates the input array in place during
mean-centering at `linear.py:307`, so the wrapper knows it must defensively
copy `X` before passing it in.

Script sketch:

```
import sys, numpy as np
sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")
X_before = X.copy()
mean_before = X.mean(axis=0).copy()

model = DagmaLinear(loss_type="l2")
_ = model.fit(
    X, lambda1=0.05, w_threshold=0.3, T=2,
    s=[1.0, 0.9],
    warm_iter=200, max_iter=200, lr=3e-4,
)

print("array values unchanged?", np.allclose(X, X_before))
print("mean of X after fit", X.mean(axis=0))
print("mean of X before fit", mean_before)
```

The label "array values unchanged?" is preferred over "array identity
preserved?" because `np.allclose` checks value equality (not Python
object identity).

Expected success: `array values unchanged? False` and the post-fit
mean is approximately zero, confirming in-place centering.

Failure interpretation: if `True`, DAGMA may have been refactored away
from in-place mutation. The wrapper note becomes redundant but harmless.

Environment changes: only `sys.path` and global numpy random state.

Dependencies required: none.

Files produced: none.

Decision informed: confirms the wrapper must pass `X.copy()` to
`fit` to protect upstream data from in-place modification.

### D-P5. Residual statistics for the noise and intercept policy

Purpose: gather empirical residual variances and per-variable means on
a small synthetic case, to inform the eventual Doc 02 amendment that
freezes the DAGMA noise and intercept policy for post-hoc interventional
sampling.

Script sketch:

```
import sys, numpy as np
sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=500, sem_type="gauss")

means = X.mean(axis=0)
X_centred = X - means

model = DagmaLinear(loss_type="l2")
W_est = model.fit(
    X.copy(), lambda1=0.05, w_threshold=0.3, T=4,
    s=[1.0, 0.9, 0.8, 0.7],
    warm_iter=2000, max_iter=4000, lr=3e-4,
)

R = X_centred - X_centred @ W_est
print("per-variable means of raw X", means)
print("per-variable residual variance", R.var(axis=0))
print("per-variable residual std    ", R.std(axis=0))
print("residuals computed against thresholded W (w_threshold=0.3)")
```

The residuals reported by this probe are computed against the
thresholded `W_est`, not against the continuous pre-threshold matrix.
This matches what the wrapper would use for post-hoc interventional
sampling, because sampling requires a valid DAG structure and the
thresholded matrix is the structure the sampler must use.

This probe records evidence only. It does NOT freeze the DAGMA noise
and intercept policy. That decision belongs in a future Doc 02
amendment after broader experiments.

Expected success: prints the per-variable means and residual variances
without raising.

Failure interpretation: any exception means D-P2 or earlier prerequisites
did not produce a usable `W_est`.

Environment changes: only `sys.path` and global numpy random state.

Dependencies required: none.

Files produced: none.

Decision informed: provides numerical inputs to the eventual Doc 02
amendment that decides between unit-variance noise, residual-fitted
noise, and zero versus empirical-mean intercepts.

---

## DCDI probes

### C-P1. Targeted imports without dcdi.train

Purpose: confirm the wrapper-relevant DCDI modules can be imported in
the project environment, and that doing so does not transitively
require `cdt`.

Script sketch:

```
import sys
sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM
from dcdi.dag_optim import GumbelAdjacency, compute_dag_constraint
from dcdi.utils.penalty import compute_penalty
print("DCDI low-level imports OK")
```

Expected success: prints the OK message.

Failure interpretation: an `ImportError` for any module other than
`cdt` indicates a project-environment dependency gap, which would need
a documented fix. If the failure mentions `cdt`, this probe has
revealed that the targeted-imports mitigation does not work and the
wrapper must do further refactoring.

Environment changes: only `sys.path` inside the probe process.

Dependencies required: none expected. If the import surfaces a missing
dependency, that is itself the answer the probe was looking for.

Files produced: none.

Decision informed: confirms the wrapper strategy of importing model
and helper modules directly while avoiding `dcdi.train`.

### C-P2. Verify cdt is not imported by the targeted import set

Purpose: prove explicitly that the targeted imports do not pull in
`cdt`, by inspecting `sys.modules` before and after.

Script sketch:

```
import sys
sys.path.insert(0, "external/source_inspection/dcdi")
before = set(sys.modules)
from dcdi.models.learnables import LearnableModel_NonLinGaussANM  # noqa
from dcdi.dag_optim import GumbelAdjacency, compute_dag_constraint  # noqa
from dcdi.utils.penalty import compute_penalty  # noqa
after = set(sys.modules)
newly_imported = after - before
cdt_modules = sorted(m for m in newly_imported if "cdt" in m.lower())
print("newly imported count", len(newly_imported))
print("cdt-related newly imported:", cdt_modules)
```

Expected success: `cdt-related newly imported: []`.

Failure interpretation: a non-empty cdt list means the targeted
imports still trigger `cdt`, contradicting the C-10 finding in
`docs/04b_source_inspection.md`. The wrapper would then need a
different import path or `cdt` installation would have to be
considered.

Environment changes: only `sys.path` inside the probe process.

Dependencies required: none.

Files produced: none.

Decision informed: closes the C-10 question in
`docs/04b_source_inspection.md` and confirms the wrapper does not
require `cdt` for training.

### C-P3. Instantiate LearnableModel_NonLinGaussANM on a tiny artificial setting

Purpose: verify that the DCDI-G learnable model can be constructed in
observational mode on CPU.

Script sketch:

```
import sys, torch
sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM

torch.set_default_tensor_type(torch.FloatTensor)  # CPU FP32
torch.manual_seed(0)

model = LearnableModel_NonLinGaussANM(
    num_vars=3,
    num_layers=2,
    hid_dim=8,
    nonlin="leaky-relu",
    intervention=False,
    intervention_type="perfect",
    intervention_knowledge="known",
    num_regimes=1,
)
print("instantiated", type(model).__name__,
      "num_vars", model.num_vars,
      "num_layers", model.num_layers,
      "hid_dim", model.hid_dim)
```

Expected success: prints the type name and the configured sizes.

Failure interpretation: an exception during instantiation indicates a
constructor-time issue not visible from source inspection, for example
a torch tensor-default-type interaction.

Environment changes: torch default tensor type is set; the probe
process exits afterwards, so the change does not persist.

Dependencies required: none beyond torch.

Files produced: none.

Decision informed: confirms model instantiation in the project
environment.

### C-P4. Access log_alpha and get_w_adj

Purpose: confirm the native edge object access pattern matches the
inspection findings.

Script sketch:

```
# Continuing from C-P3 in the same process:
log_alpha = model.gumbel_adjacency.log_alpha
print("log_alpha shape", tuple(log_alpha.shape),
      "requires_grad", log_alpha.requires_grad,
      "init value [0,1]", float(log_alpha[0, 1].item()))

w_adj = model.get_w_adj()
print("w_adj shape", tuple(w_adj.shape),
      "diag", w_adj.diag().tolist(),
      "off-diag [0,1]", float(w_adj[0, 1].item()))
```

Expected success:

- `log_alpha shape (3, 3) requires_grad True init value [0,1] 5.0`
- `w_adj shape (3, 3) diag [0.0, 0.0, 0.0] off-diag [0,1] approximately 0.9933`

Failure interpretation: any deviation from the values above means
either the inspection finding was wrong or the model was constructed
in an unexpected mode.

Environment changes: none beyond C-P3.

Dependencies required: none.

Files produced: none.

Decision informed: confirms the wrapper can read both `log_alpha`
(the native parameter `Lambda`) and `get_w_adj()` (the native
continuous edge object `P` with diagonal masked).

### C-P5. forward_given_params on a tiny batch in eval mode

Purpose: verify that the model can compute conditional density
parameters on a tiny batch without going through training. Before
calling, inspect the actual signature so the call pattern is correct.

The probe must NOT assume the three-argument call `forward_given_params(x,
weights, biases)` is correct. Inspect the signature first. A `TypeError`
from a guess-call is not sufficient evidence that DCDI sampling is
unavailable; it might just mean the call pattern is wrong.

Script sketch:

```
# Continuing from C-P4 in the same process:
import inspect, torch

print("forward_given_params signature:",
      inspect.signature(model.forward_given_params))

model.eval()
bs = 4
x = torch.randn(bs, 3)
weights, biases, extra_params = model.get_parameters(mode="wbx")

# First attempt: minimal call (mask and regime default to None).
try:
    with torch.no_grad():
        density_params = model.forward_given_params(x, weights, biases)
    call_pattern = "x, weights, biases (mask=None, regime=None)"
except TypeError as e:
    print("minimal call failed with TypeError:", e)
    # Second attempt: provide explicit observational mask and regime.
    # Suggested starting values for observational mode:
    mask = torch.ones(bs, model.num_vars, model.num_vars)
    regime = torch.zeros(bs, dtype=torch.long)
    with torch.no_grad():
        density_params = model.forward_given_params(
            x, weights, biases, mask=mask, regime=regime
        )
    call_pattern = "x, weights, biases, mask=ones(bs,d,d), regime=zeros(bs)"

print("call pattern that worked:", call_pattern)
print("density_params type:", type(density_params).__name__)
print("density_params length:", len(density_params))
for i, dp in enumerate(density_params):
    print(f"  var {i} shape:", tuple(dp.shape))
```

Expected success: at least one of the two call patterns returns a
tuple of length `num_vars` (3) with each entry of shape `(bs, num_params)`
where `num_params=1` for `LearnableModel_NonLinGaussANM` (predicted mean
only; std comes from `extra_params`).

Failure interpretation: a `TypeError` on the first call alone does NOT
mean DCDI sampling is unavailable. Only if both patterns fail with
unrecoverable errors is the API genuinely unsupported. If the observed
signature requires additional arguments not covered by the two attempts,
record the signature and adapt accordingly in C-P7.

Environment changes: none beyond C-P3.

Dependencies required: none.

Files produced: none.

Decision informed: confirms the foundation for wrapper-side ancestral
sampling. If C-P5 fails, sampler_status for DCDI must be
`unavailable_no_api`.

### C-P6. Conditional Normal construction and sampling

Purpose: verify that the per-variable conditional Normal can be
constructed from the density parameters plus extra params, and a sample
can be drawn. Before constructing the distribution, print the shapes so
the unbind-and-extend pattern can be confirmed against the observed
contract rather than assumed.

Script sketch:

```
# Continuing from C-P5 in the same process:
import torch

with torch.no_grad():
    extra_params_t = model.transform_extra_params(model.extra_params)

print("density_params type:", type(density_params).__name__)
print("density_params length:", len(density_params))
for i, dp in enumerate(density_params):
    print(f"  density_params[{i}] shape:", tuple(dp.shape))
print("extra_params_t type:", type(extra_params_t).__name__)
print("extra_params_t length:", len(extra_params_t))
for i, ep in enumerate(extra_params_t):
    print(f"  extra_params_t[{i}] shape:", tuple(ep.shape))

# Construct the Normal only after confirming the shape contract.
i = 0
dp_i = list(torch.unbind(density_params[i], 1))
dp_i.extend(list(torch.unbind(extra_params_t[i], 0)))
print(f"unbound dp_i for var {i}: {len(dp_i)} tensors, "
      f"shapes={[tuple(t.shape) for t in dp_i]}")

dist = model.get_distribution(dp_i)
print("dist type:", type(dist).__name__)

with torch.no_grad():
    sample = dist.sample()
print("sample shape:", tuple(sample.shape))
print("first few values:", sample[:4].tolist())
```

Expected success: `dist type` is `Normal`, `sample shape` is `(bs,)` or
broadcast-compatible, and the printed shapes match the inspection
findings (`density_params[i]` has shape `(bs, 1)` for the mean,
`extra_params_t[i]` has shape `(1,)` for the log-std before transform).

Failure interpretation: an exception indicates that the distribution
construction pattern in `learnables.py:67-72` cannot be replicated
outside the original log-likelihood path, and the wrapper would need a
different approach to sample per-variable conditionals.

Environment changes: none beyond C-P3.

Dependencies required: none.

Files produced: none.

Decision informed: confirms that the per-node conditional Normal is
sample-able outside training, which is the core sampling primitive for
the wrapper's MMD path.

### C-P7. Minimal ancestral sampling sketch with one variable clamped

Purpose: verify a single-step ancestral sample under
`do(X_target = value)` works using only model APIs and a manually fixed
DAG structure.

Script sketch:

```
import sys, torch
sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM

torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)

model = LearnableModel_NonLinGaussANM(
    num_vars=3, num_layers=2, hid_dim=8,
    nonlin="leaky-relu", intervention=False,
    intervention_type="perfect", intervention_knowledge="known",
    num_regimes=1,
)
model.eval()

# Force a known DAG structure for the probe:
# Adjacency is the project convention: row=parent, column=child.
# Encode 0 -> 1, 1 -> 2 explicitly.
import numpy as np
adj = torch.zeros(3, 3)
adj[0, 1] = 1.0
adj[1, 2] = 1.0
with torch.no_grad():
    model.adjacency.copy_(adj)
    model.gumbel_adjacency.log_alpha.copy_((adj * 100.0) + (1.0 - adj) * -100.0)

bs = 5
x = torch.zeros(bs, 3)
weights, biases, extra_params = model.get_parameters(mode="wbx")
ext_t = model.transform_extra_params(model.extra_params)

with torch.no_grad():
    # Sample X_0 (root): conditional given all-zero parents
    density_params = model.forward_given_params(x, weights, biases)
    dp0 = list(torch.unbind(density_params[0], 1)) + list(torch.unbind(ext_t[0], 0))
    x[:, 0] = model.get_distribution(dp0).sample()

    # Clamp X_1 = 0.5 (do-intervention on node 1)
    x[:, 1] = 0.5

    # Sample X_2 from its conditional given (X_0, X_1=0.5)
    density_params = model.forward_given_params(x, weights, biases)
    dp2 = list(torch.unbind(density_params[2], 1)) + list(torch.unbind(ext_t[2], 0))
    x[:, 2] = model.get_distribution(dp2).sample()

print("sample under do(X_1=0.5):")
print(x.tolist())
assert torch.allclose(x[:, 1], torch.full((bs,), 0.5))
print("clamping invariant holds")
```

Note: if C-P5 or C-P6 revealed a different call pattern for
`forward_given_params` (for example explicit `mask` and `regime`
arguments), the C-P7 sketch must be adjusted to use the same pattern.

Expected success: the printed array shows constant `0.5` in column 1
on every row, while columns 0 and 2 contain sampled values that are
reasonable real numbers. The final assert does not raise.

Failure interpretation: distinguish carefully between

- Incorrect probe assembly (the script wires shapes or arguments
  wrongly): the failure is fixable in the probe; the API is fine.
- Missing API support (`forward_given_params` truly cannot run in
  eval mode for arbitrary inputs): sampler_status must be
  `unavailable_no_api`.
- Tensor shape mismatch (an unexpected dimension after unbind): the
  shape contract from C-P6 needs refinement before a wrapper sampler
  can be written, but the API is still usable.
- Conceptual sampler infeasibility (the model genuinely does not
  expose per-variable conditionals at all): also `unavailable_no_api`.

Each failure mode should be explicitly recorded so the report can
distinguish probe-side mistakes from real API limits.

Environment changes: torch default tensor type is set inside the
probe process.

Dependencies required: none beyond torch.

Files produced: none.

Decision informed: closes the C-6 question in
`docs/04b_source_inspection.md`. If C-P7 succeeds, the DCDI wrapper
sampler status can become `available` once the corresponding wrapper
code is written. If C-P7 fails, sampler_status remains
`unavailable_no_api`.

### C-P8. Deterministic repeatability on CPU for a tiny controlled run

Purpose: verify that with seeds set, two short fits of a tiny model on
CPU produce bit-identical native edge parameters.

Script sketch:

```
import sys, numpy as np, torch
sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM
from dcdi.dag_optim import compute_dag_constraint
from dcdi.utils.penalty import compute_penalty

def tiny_train():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_default_tensor_type(torch.FloatTensor)

    m = LearnableModel_NonLinGaussANM(
        num_vars=3, num_layers=2, hid_dim=8,
        nonlin="leaky-relu", intervention=False,
        intervention_type="perfect", intervention_knowledge="known",
        num_regimes=1,
    )
    x = torch.randn(32, 3)
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-3)
    full_adj = torch.ones(3, 3) - torch.eye(3)
    constraint_norm = float(compute_dag_constraint(full_adj).item())

    for _ in range(5):
        weights, biases, extra_params = m.get_parameters(mode="wbx")
        log_lik = m.compute_log_likelihood(x, weights, biases, extra_params)
        nll = -log_lik.mean()
        w_adj = m.get_w_adj()
        h = compute_dag_constraint(w_adj) / constraint_norm
        reg = 0.1 * compute_penalty([w_adj], p=1) / (3 ** 2)
        aug = nll + reg + h
        opt.zero_grad()
        aug.backward()
        opt.step()

    return m.gumbel_adjacency.log_alpha.detach().clone()

result1 = tiny_train()
result2 = tiny_train()
identical = bool(torch.equal(result1, result2))
print("identical?", identical)
if not identical:
    diff = (result1 - result2).abs()
    print("max abs diff:", float(diff.max().item()))
    print("close within 1e-6?", bool(torch.allclose(result1, result2, atol=1e-6)))
    print("close within 1e-4?", bool(torch.allclose(result1, result2, atol=1e-4)))
```

Expected success: `identical? True`.

Failure interpretation: a `False` result does not by itself disqualify
the DCDI wrapper. If outputs are close within a documented numerical
tolerance (for example `1e-6` or `1e-4`), the wrapper documentation may
claim near-bitwise determinism with that tolerance. Only if the
difference is large should the wrapper drop the determinism claim and
state explicitly that DCDI runs are reproducible only in distribution.
This matches the wrapper contract Section 13 wording.

Environment changes: torch default tensor type and torch and numpy
random states are set inside the probe process.

Dependencies required: none beyond torch.

Files produced: none.

Decision informed: confirms or refines the determinism statement that
the DCDI wrapper documentation must make.

---

## Risk classification

### Safe to run now

- D-P1, D-P2, D-P3, D-P4, D-P5
- C-P1, C-P2, C-P3, C-P4, C-P5, C-P6, C-P7, C-P8

All probes above use only the existing project virtual environment, do
not install or remove dependencies, do not write files outside the
proposed `inspection/probes/` directory (and even there, only if the
probe is later turned into a script file), and do not touch the
project source, tests, docs, or external repository copies. Each probe
runs in seconds.

### Requires approval because it may install dependencies or alter the environment

- None at this stage.

If C-P1 or C-P2 unexpectedly fails with a missing-dependency error
mentioning `cdt` or `networkx`, addressing that failure would move into
this category. The plan does not pre-approve such an installation. The
correct response is to record the failure and surface a Doc 02
amendment proposal that decides whether to install the missing
dependency, vendor a minimal substitute, or refactor the wrapper.

### Should not be run yet

- A full DCDI training run to convergence on the selection-study cell.
- A full DAGMA fit on 10-node 1000-sample data with the production
  hyperparameters.
- Any probe that imports `dcdi.train`. This module triggers the `cdt`
  and R import chain documented in `docs/04b_source_inspection.md`.
- Any probe that creates files under `src/`, `tests/`, `docs/`, or
  `external/source_inspection/`.

These items are deferred to wrapper implementation and selection-study
execution, both of which are explicitly out of scope here.

---

## Probe outputs the wrapper documentation will need

When the probes are eventually run (in a separate, approved session),
the following outputs become inputs to the wrapper documentation and
to a future Doc 02 amendment:

- D-P3 outcome: whether `w_threshold=0` is the wrapper's preservation
  strategy, or whether a subclass-and-override is required.
- D-P4 outcome: whether the wrapper must always pass `X.copy()` to
  `fit`. The inspection already strongly suggests yes; this probe
  confirms it.
- D-P5 outcome: numerical residual statistics that inform the eventual
  DAGMA noise and intercept policy frozen in Doc 02.
- C-P1 and C-P2 outcomes: confirm or refute the targeted-imports-only
  strategy that avoids `cdt`.
- C-P5 to C-P7 outcomes: determine the DCDI sampler status. If C-P7
  succeeds, the wrapper can implement post-hoc sampling under
  arbitrary `do(X_j = v)` and report `sampler_status = available`. If
  C-P7 fails, the wrapper must report `sampler_status = unavailable_no_api`
  and the selection study must consider the MMD-unavailable tie policy.
- C-P8 outcome: determines whether the DCDI wrapper can claim bitwise
  determinism on CPU or only documented numerical tolerance.

---

## What this plan does not do

- This plan does not run any probe.
- This plan does not implement DAGMAWrapper or DCDIWrapper.
- This plan does not create wrapper skeletons.
- This plan does not modify project source code or tests.
- This plan does not modify `docs/02_base_model_selection.md` or
  `docs/03_decision_log.md`.
- This plan does not modify the cloned external repositories.
- This plan does not install or change dependencies.
- This plan does not pre-approve any environment change. Approval
  decisions are deferred to a separate review cycle.
