# DAGMA-linear Wrapper Implementation Plan

## Status

Planning artefact only. No DAGMA wrapper code, test code, probe code, or
external repository file is created by this document. Implementation
does not begin until the plan is reviewed and Commit 1 is approved.

Version 1.1.

### Change log

- **v1.0 -> v1.1**:
  - removed the precautionary `np.random.seed(seed)` call from the
    DAGMA fit path; the wrapper fit path does not call
    `np.random.seed`, `torch.manual_seed`, or
    `dagma.utils.set_random_seed`. The `seed` argument is recorded
    for traceability only. Sampling continues to use a local
    `np.random.default_rng(sample_seed)`.
  - fixed the public `sample_interventional` signature in Section 5
    to include the `noise_policy` keyword.
  - replaced the misnamed `iterations_used` diagnostic field with
    `iterations_configured_upper_bound`, which is not presented as
    actual iterations used.
  - added an explicit requirement to Commit 5 that all existing DCDI
    wrapper tests remain green after the shared graph-status and
    topological-order helpers are moved into a sibling utility
    module.
  - softened wording about self-loops to reflect that they are not
    expected under the inspected DAGMA path but are reported and not
    repaired if they occur.
  - removed an incorrect "0.5 stub line" wording from the DAGMA
    sampler-quality diagnostic. DAGMA does not use a 0.5 threshold.
    The probe records the continuous `W` and threshold-grid
    adjacencies at 0.2, 0.3, and 0.4.
  - verified that every reference to "Doc 02 v1.3 Section 7 item 5"
    points to the threshold-robustness report (correct) and every
    reference to "Doc 02 v1.3 Section 7 item 6" points to the SID
    gate (correct); no replacement was needed.

This plan follows the same plan-then-implement discipline used in
`docs/05_dcdi_wrapper_implementation_plan.md`. It does not transplant
DCDI-specific machinery into DAGMA. Where the DCDI plan was driven by
the need to reimplement DCDI's training loop and to enforce a
structural mask through `model.adjacency` plus saturated `log_alpha`,
the DAGMA plan is driven by very different architectural facts: DAGMA
exposes a single supported entry point (`DagmaLinear.fit`) that the
wrapper calls directly, DAGMA produces a NumPy continuous `W` rather
than a probability-style edge object, and DAGMA has no learned
generative conditionals so sampling requires an explicit noise policy.

---

## 1. Authoritative inputs and current project state

### Inputs read for this plan

The plan is constructed from, and is subordinate to, the following
documents in priority order:

1. `docs/01_research_question_and_commitments.md` (frozen contract).
2. `docs/02_base_model_selection.md` v1.3 (frozen selection-study
   protocol including DAGMA hyperparameters, residual-fitted noise
   policy, and project-level threshold 0.3 applied externally).
3. `docs/03_decision_log.md` (evaluator conventions and wrapper-phase
   conventions).
4. `docs/04_wrapper_api_contract.md` (status taxonomy, no-silent-repair
   policy, sampler-status taxonomy).
5. `docs/04a_orientation_audit.md` (row-source / column-destination
   convention).
6. `docs/04b_source_inspection.md` (DAGMA findings D-1 through D-10).
7. `docs/04c_runtime_probe_results.md` (probes D-P1 through D-P5).
8. `docs/04f_dcdi_sampler_quality_diagnostic.md` and
   `docs/04g_equal_variance_identifiability_check.md` (lessons learned
   on sampler-quality validation as a binding pytest gate).
9. `docs/05_dcdi_wrapper_implementation_plan.md` (DCDI plan; treated
   as a sibling architecture, not a template).

### Project state at the start of this plan

- Phase 1 evaluator foundation is complete.
- Normal pytest collection is green: 190 passed, 1 skipped (verified
  SID scaffold).
- DCDI wrapper mechanics are implemented and tested through raw-unit
  interventional sampling (Commits 1 to 9 of Doc 05).
- DCDI Commit 10 sampler-quality validation did NOT pass and was
  converted into diagnostic artefact C-P11 in
  `docs/04f_dcdi_sampler_quality_diagnostic.md`.
- DCDI Commit 11 (loss-hook injection) is paused pending project-level
  review.
- No DAGMA wrapper, selection-study runner, soft-prior layer, or
  verified SID implementation exists yet.
- Verified SID integration remains a parallel blocker for scientific
  completeness of the selection study.

This plan does not depend on DCDI Commit 11 being unpaused. It also
does not depend on verified SID being available, beyond stating
honestly that selection-study conclusions cannot be final until SID is
verified.

---

## 2. Purpose and scope

### Purpose

Specify the DAGMA-linear wrapper at the level of detail required for a
reviewable, commit-by-commit implementation. After this plan is
approved, implementation may proceed Commit 1 at a time, in the order
documented in Section 14, with each commit gated on its acceptance
criterion before the next is started.

### Scope of this plan

In scope:

- DAGMA-linear fitting path through the inspected source at
  `external/source_inspection/dagma/src/dagma/linear.py`.
- Continuous `W` preservation policy at the wrapper boundary.
- Wrapper-level thresholding at `abs(W_continuous) >= 0.3`.
- Graph-status classification and no-silent-repair policy.
- Residual-fitted per-node noise estimation for interventional
  sampling.
- Wrapper-side ancestral sampler with model-frame and raw-unit
  variants.
- Source-faithfulness check against a direct `DagmaLinear.fit` call.
- Sampler-quality diagnostic probe (inspection artefact, not pytest
  gate).
- Wrapper diagnostics schema and logging.

Out of scope of this plan:

- Soft-prior loss-hook implementation for DAGMA. The likely strategy
  is documented in Section 18 but no code is written.
- Selection-study runner.
- Hard-constraint baseline implementation. DAGMA's
  `exclude_edges` / `include_edges` hooks are reserved for a separate,
  explicitly documented future commit (Section 18).
- Verified SID integration.
- Performance optimisation of the DAGMA fit.

---

## 3. Strategy after the DCDI sampler-quality diagnostic

The DCDI Commit 10 failure and the C-P12 follow-up have three
consequences for the DAGMA plan:

1. Learned sampler-quality is NOT treated as a normal pytest
   acceptance gate. It is a base-model property, not a wrapper
   property, and converting a base-model failure into a CI failure
   produces the wrong signal. The DAGMA wrapper plan reproduces the
   C-P11 fixture as an inspection probe and report from the start.
   Sampler MECHANICS (clamping, structural masking, restoration,
   determinism, raw-unit roundtrip) remain ordinary pytest gates,
   because those failure modes are wrapper failures.
2. Failure localisation must be designed in advance. The DAGMA
   sampler-quality probe must split structure-quality, coefficient
   quality, and noise-policy contributions into independently
   readable diagnostics rather than collapsing them.
3. No acceptance threshold is weakened. Wherever the DCDI plan
   recorded original thresholds against the failed C-P11 outcome,
   the DAGMA plan records identical Monte Carlo floor comparison
   logic and identical wrong-structure margin logic, with the
   negative-floor caveat documented up front.

This plan does NOT propose using the equal-variance Gaussian-BIC
exhaustive enumeration from C-P12 as a selection-study baseline or
as a model-selection method. C-P12 is a sanity check only.

---

## 4. Module and file structure

### New paths to create

```
src/symbolic_priors_cd/wrappers/dagma.py
src/symbolic_priors_cd/wrappers/_dagma_fit.py
src/symbolic_priors_cd/wrappers/_dagma_sampling.py
src/symbolic_priors_cd/wrappers/_dagma_utils.py
tests/test_dagma_wrapper_interface.py
tests/test_dagma_wrapper_source_faithfulness.py
tests/test_dagma_wrapper_thresholding.py
tests/test_dagma_wrapper_sampler.py
tests/test_dagma_wrapper_residual_noise.py
inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py
docs/04h_dagma_sampler_quality_diagnostic.md
```

### Existing modules touched

- `src/symbolic_priors_cd/wrappers/__init__.py`: extend re-exports to
  include the new DAGMA public surface (Section 5).
- `src/symbolic_priors_cd/wrappers/status.py`: see Section 15 for the
  diagnostics schema option. The recommended option introduces a
  `model_specific_diagnostics` field; the less disruptive option
  introduces a sibling `DagmaWrapperDiagnostics` TypedDict.

### Why three internal modules

- `_dagma_fit.py`: hosts the DAGMA fit call path and continuous `W`
  preservation logic. Single source of truth for hyperparameter
  defaults, `X.copy()` discipline, and source-faithfulness comparison
  helpers.
- `_dagma_sampling.py`: residual sigma estimation, model-frame
  ancestral sampler, raw-unit roundtrip, structural validity guards.
  Mirrors the role of `_dcdi_sampling.py` but does not import any DCDI
  module.
- `_dagma_utils.py`: thin pieces (DAGMA import shim, configuration
  resolution, diagnostics assembly).

A single umbrella file would mix three independent failure modes and
would make the source-faithfulness gate (Commit 4) hard to review
against the fit-path internals.

### What is not added

- No DAGMA loss-hook code.
- No DAGMA hard-constraint code (`exclude_edges` /
  `include_edges` remain unused at this stage).
- No selection-study runner.
- No notebook or CLI entry point.

---

## 5. Public wrapper surface

### Class and method names

```
class DAGMAWrapper:
    def fit(
        self,
        X_train: np.ndarray,
        *,
        preprocessor: Union[CentredOnlyTransform, StandardisedTransform],
        seed: int,
        config: Optional[DAGMAConfig] = None,
    ) -> None: ...

    def native_edge_continuous(self) -> np.ndarray: ...
    def thresholded_adjacency(self, threshold: float = 0.3) -> np.ndarray: ...

    def sample_interventional(
        self,
        intervention: Intervention,
        n_samples: int,
        *,
        sample_seed: int,
        noise_policy: Literal["residual_fitted", "unit_variance"] = "residual_fitted",
    ) -> Optional[np.ndarray]: ...

    def get_diagnostics(self) -> DagmaWrapperDiagnostics: ...
```

Notes:

- The wrapper does not accept `X_val`. DAGMA does not use a validation
  set and the wrapper does not invent one. Caller-side validation
  splits remain the caller's responsibility.
- `native_edge_continuous()` returns the preserved pre-threshold
  continuous `W` as `np.ndarray` (DAGMA is NumPy-native; there is no
  PyTorch tensor to clone).
- `thresholded_adjacency(threshold)` applies `abs(W_continuous) >= threshold`.
  Default is `0.3`, matching Doc 02 v1.3.
- `sample_interventional` returns `None` when `sampler_status` is not
  `available`. The reason is recorded in diagnostics.
- The wrapper does NOT expose `exclude_edges` / `include_edges`
  parameters. They are not threaded through `DAGMAConfig`.

### DAGMAConfig

```
@dataclass(frozen=True)
class DAGMAConfig:
    # Doc 02 v1.3 frozen values
    T: int = 4
    lambda1: float = 0.05
    s: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
    mu_init: float = 1.0
    mu_factor: float = 0.1
    w_threshold_internal: float = 0.0   # passed to DagmaLinear.fit
    # Library defaults, recorded explicitly so they appear in config_snapshot
    lr: float = 3e-4
    warm_iter: int = 30000
    max_iter: int = 60000
    beta_1: float = 0.99
    beta_2: float = 0.999
    loss_type: str = "l2"
    # Wrapper project-level threshold applied externally
    project_threshold: float = 0.3
    # Convergence-status threshold for h_final diagnostic
    h_diagnostic_threshold: float = 1e-5
```

Notes:

- `T`, `lambda1`, `s`, `mu_init`, `mu_factor`, and
  `w_threshold_internal=0.0` are frozen by Doc 02 v1.3 and MUST be
  passed at the call site rather than left to library defaults
  (which diverge from Doc 02; see D-9 in `docs/04b_source_inspection.md`).
- `lr`, `warm_iter`, `max_iter`, `beta_1`, `beta_2`, `loss_type` are
  library defaults at `external/source_inspection/dagma/src/dagma/linear.py:234-244`,
  recorded explicitly in the config and in `config_snapshot` so any
  run can be reproduced from its diagnostics.
- `project_threshold` is the externally applied threshold from
  Doc 02 v1.3 Section 9 (`abs(W_continuous) >= 0.3`). It is
  parameterised here so that threshold-robustness reporting can pass
  alternative values without retraining; for selection-study runs the
  default 0.3 is the canonical value.
- `h_diagnostic_threshold` is the wrapper-side diagnostic threshold
  for the `training_status` field (Section 10). Probe D-P2 observed
  `h_final = 1.05e-6` on the small fitness run; `1e-5` is a
  conservative diagnostic threshold and is NOT graph repair.

The dataclass is frozen so `config_snapshot` can serialise it
verbatim.

---

## 6. DAGMA source and API boundary

### Source path

The wrapper calls into the inspected DAGMA source at
`external/source_inspection/dagma/src/dagma/linear.py`, frozen at
commit `088616885d71b56c0573cd4902c1fcbac02e649f`. The wrapper does
not depend on the installed `dagma` package on PyPI. The wrapper
import shim adds the inspected source path to `sys.path` exactly
once, behind a guarded helper in `_dagma_utils.py`, mirroring the
DCDI pattern used at `wrappers/_dcdi_utils.py`.

### Imports used by the wrapper

- `from dagma.linear import DagmaLinear`
- Nothing else from the DAGMA package. `dagma.utils` is NOT imported.
  In particular, `dagma.utils.set_random_seed` is NOT called by the
  wrapper because (a) DAGMA `fit` is deterministic given fixed input
  and hyperparameters (Section 7 of `docs/04b_source_inspection.md`
  D-6, confirmed by D-P2 in
  `docs/04c_runtime_probe_results.md`), and (b)
  `set_random_seed` mutates global NumPy state.

### Imports the wrapper MUST NOT use

- `dagma.utils.set_random_seed` (mutates global state, not needed).
- `DagmaLinear`'s `exclude_edges` and `include_edges` arguments
  (hard-constraint surface; reserved for a separately documented
  future commit, Section 18).
- `dagma.nonlinear` (not in the selection-study scope).

### Hyperparameter override discipline

Library defaults for `T`, `lambda1`, and `s` diverge from Doc 02 v1.3
(D-9). The wrapper passes Doc 02 values explicitly at every fit call.
A unit test (Section 13, Section 16) asserts that the resolved values
in `config_snapshot` match the Doc 02 values when `config=None`.

---

## 7. DAGMA-linear fitting path

### Fit sequence

For a single `fit` call:

1. Validate that `X_train` is 2D `float` and has at least one row and
   at least 2 variables.
2. Apply `X_local = np.asarray(X_train, dtype=float).copy()`. This
   defensive copy is REQUIRED because DAGMA mutates its input array
   in place during L2 mean-centering at
   `linear.py:307`. Confirmed by probe D-P4. Forgetting the copy
   silently corrupts upstream training data.
3. Resolve `cfg = config if config is not None else DAGMAConfig()`.
   Record the resolved values in `config_snapshot` (Section 15).
4. Record the `seed` argument verbatim in diagnostics
   (`config_snapshot.seed`) for traceability. The wrapper fit path
   does NOT call `np.random.seed(seed)`, does NOT call
   `torch.manual_seed`, and does NOT call
   `dagma.utils.set_random_seed`. DAGMA `fit` is deterministic for
   fixed `X` and fixed hyperparameters (D-6 in
   `docs/04b_source_inspection.md`, confirmed by D-P2 in
   `docs/04c_runtime_probe_results.md`); the fit path requires no
   seeding to be reproducible. All wrapper-side stochasticity lives
   in the sampler, which uses a local
   `np.random.default_rng(sample_seed)` per call (Section 12). The
   wrapper therefore does not mutate any global RNG state.
5. Instantiate `model = DagmaLinear(loss_type=cfg.loss_type)`.
6. Call `model.fit(...)` with explicit Doc 02 values:

   ```
   W_returned = model.fit(
       X=X_local,
       lambda1=cfg.lambda1,
       w_threshold=cfg.w_threshold_internal,   # 0.0
       T=cfg.T,
       mu_init=cfg.mu_init,
       mu_factor=cfg.mu_factor,
       s=list(cfg.s),
       warm_iter=cfg.warm_iter,
       max_iter=cfg.max_iter,
       lr=cfg.lr,
       beta_1=cfg.beta_1,
       beta_2=cfg.beta_2,
       exclude_edges=None,
       include_edges=None,
   )
   ```

   The wrapper does not rely on library defaults for any
   selection-study-relevant argument.
7. Capture `h_final = float(model.h_final)` and
   `score_final = float(model.score_final)` from the fitted model
   attributes (D-1, D-2).
8. Store `self._continuous_w_pre_threshold = W_returned.copy()` as
   the canonical preserved continuous edge object. `W_returned` is
   already the matrix returned by `fit`; with
   `w_threshold_internal=0.0` no entries have been zeroed (probe
   D-P3). The defensive `.copy()` insulates the wrapper from any
   later in-place mutation of `model.W_est`.
9. Convert to thresholded boolean adjacency via the shared helper
   (Section 8). Store the resolved `graph_status` and any reason.
10. Estimate residual per-node sigmas in the model frame (Section 11),
    or set `sampler_status = unavailable_unresolved_noise_policy`
    with a reason if the estimate is degenerate.

The wrapper does not retain the live `DagmaLinear` instance after the
fit because DAGMA exposes no API requiring it. All downstream
behaviour reads from `_continuous_w_pre_threshold`, the residual
sigma vector, and the preprocessor reference.

### Independence from the installed DAGMA package

The wrapper uses the inspected source unconditionally. There is no
fallback path to the PyPI `dagma==1.1.1` package, because the
inspection report and the runtime probe were both run against the
inspected source. A test (Section 16) asserts that the imported
`DagmaLinear` module path is the inspected source path.

---

## 8. Continuous W preservation, thresholding, and orientation

### Preservation policy

- The wrapper always calls `DagmaLinear.fit` with
  `w_threshold=0.0`. Probe D-P3 confirmed that this preserves every
  off-diagonal entry of the continuous `W`.
- The post-fit copy `self._continuous_w_pre_threshold` is the
  canonical native edge object. It is the basis for:
  - `native_edge_continuous()` (public accessor),
  - `thresholded_adjacency(threshold)` (any threshold value),
  - residual sigma estimation (Section 11),
  - threshold-robustness reporting required by Doc 02 v1.3 Section 7
    item 5,
  - diagnostics under Section 15.
- The wrapper never mutates `self._continuous_w_pre_threshold` after
  fit. A test (Section 16) verifies this is unchanged across multiple
  sampling calls.

### Project thresholding

Doc 02 v1.3 freezes the DAGMA selection-study thresholding policy.
The wrapper:

- calls DAGMA with `w_threshold=0.0`,
- preserves `W_continuous`,
- applies project-level thresholding externally as
  `abs(W_continuous) >= 0.3`.

This is NOT optional or tentative. The wrapper test suite includes
explicit verification that the project-level threshold is applied to
`abs(W_continuous)` and not to `W_continuous` directly (D-7 in
`docs/04b_source_inspection.md`).

### Threshold helper

```
def _threshold_continuous_w(
    continuous_w: np.ndarray, threshold: float
) -> np.ndarray:
    """Return abs(continuous_w) >= threshold as a strict bool matrix."""
    return (np.abs(continuous_w) >= threshold).astype(bool)
```

The helper is the single source of truth for DAGMA thresholding. The
public `thresholded_adjacency(threshold)`, the diagnostics
threshold-grid counts (Section 15), and the threshold-robustness
report at Doc 02 v1.3 Section 7 item 5 all call it. A test
(Section 16) monkey-patches the helper to confirm it is the single
source.

### Orientation

DAGMA uses the same convention as this project (D-4 in
`docs/04b_source_inspection.md`): row-source / column-destination.
No orientation transformation is performed at the wrapper boundary.
A mock-orientation test (Section 16) builds a known continuous `W`
matrix with one off-diagonal entry above threshold and verifies that
`thresholded_adjacency` puts the `True` at the expected `(i, j)`
position.

### Float and dtype discipline

- `_continuous_w_pre_threshold` is stored as `float64`. DAGMA returns
  `float` matching its internal dtype (default `np.float64` at
  `linear.py`); the wrapper does not down-cast.
- `thresholded_adjacency` returns strict `np.bool_` of shape
  `(num_vars, num_vars)`.

---

## 9. Graph-status validation and no silent repair

### Reuse of the project graph-status machinery

The classification helper `classify_graph_status` in
`src/symbolic_priors_cd/wrappers/dcdi.py` is generic (it inspects a
bool adjacency matrix and never touches DCDI internals). The DAGMA
wrapper reuses it via either:

- importing `classify_graph_status` directly from
  `symbolic_priors_cd.wrappers.dcdi`, or
- moving the helper to a shared sibling module
  `wrappers/_graph_status.py` to remove the cross-import from DAGMA
  into a DCDI-named module.

The recommended option is to move the helper to a shared module in
Commit 5 of this plan; this is a small mechanical refactor and avoids
DAGMA importing from `wrappers.dcdi`. The DCDI wrapper's existing
tests continue to pass because the helper itself does not change.

### Priority order

The shared helper preserves the priority order already in use:
`invalid_shape` then `self_loop` then `bidirected` then `cyclic`
then `valid_dag`. A self-loop is not expected under the inspected
DAGMA path, but if thresholding produces one, the wrapper reports
`self_loop` and does not repair it. The classification is the only
action the wrapper takes; the offending entry is exposed verbatim
via `thresholded_adjacency` so it is visible to the run record.

### No silent repair

The DAGMA wrapper MUST NOT:

- remove edges to break cycles,
- break two-cycles by choosing the larger of two opposing entries,
- zero the diagonal post-threshold as a hidden repair,
- symmetrise the output,
- replace an invalid graph with a repaired graph,
- use `exclude_edges` to remove invalid edges as a post-hoc step.

If `graph_status` is not `valid_dag`:

- the invalid adjacency is preserved and exposed via
  `thresholded_adjacency`,
- `sampler_status` becomes `unavailable_invalid_graph` with the
  classification reason,
- `sample_interventional` returns `None` and the reason is logged in
  diagnostics,
- the diagnostics record makes the invalid count visible so
  selection-study aggregation can count failures.

### Test coverage

- bidirected, cyclic, and invalid-shape thresholded outputs all map
  to the correct `graph_status` and `sampler_status` strings;
- the invalid graph adjacency is preserved bitwise (no silent
  repair);
- `sample_interventional` returns `None` on each invalid case.

---

## 10. Training-status and convergence diagnostics

### What DAGMA exposes

After `fit`, DAGMA exposes `self.h_final` and `self.score_final` on
the model instance (`linear.py:352-354`). DAGMA does not expose a
convergence flag analogous to DCDI's `first_stop` and
`second_stop`; the path-following method either reaches its planned
`T` central-path stages or it does not.

### Provisional wrapper-level training_status mapping

- `training_status = "converged"` if `h_final <= cfg.h_diagnostic_threshold`
  (default `1e-5`, supported by probe D-P2 which observed
  `h_final = 1.05e-6` on a small synthetic problem).
- `training_status = "max_iter"` otherwise.
- `training_status = "wrapper_error"` is reserved for explicit
  wrapper-side exceptions.
- `training_status = "diverged"` is not currently used by DAGMA
  because the path-following method does not produce a divergent
  state in the source path; if a non-finite `h_final` or non-finite
  `W` ever appears, the wrapper records `training_status =
  "diverged"` and re-raises through the wrapper boundary so the
  failure is visible.

`h_diagnostic_threshold` is a diagnostic mapping only. It does not
trigger graph repair, does not change the saved continuous `W`, and
does not change `graph_status` (which is determined independently
from the thresholded boolean adjacency).

### Diagnostic fields recorded

- `h_final`: float, recorded verbatim.
- `score_final`: float, recorded verbatim.
- `iterations_configured_upper_bound`: integer, derived from the
  configured budget `cfg.T * cfg.max_iter + cfg.warm_iter`. This
  is NOT an observed iteration count and MUST NOT be presented as
  one. DAGMA does not expose an iteration counter for the inner
  Adam loop, so the wrapper records the configured upper bound from
  `DAGMAConfig` as a reproducibility surrogate only. If a future
  DAGMA version exposes an actual iteration count, the wrapper
  should add that field separately rather than relabelling this
  one.

---

## 11. Residual sigma estimation

Doc 02 v1.3 Section 4.2 freezes the DAGMA MMD-sampling noise policy
as residual-fitted per-node noise in the model frame, with
unit-variance noise as a sensitivity check.

### Primary policy: residual-fitted per-node noise

Inputs to estimation:

- `X_model_frame`: the training data in model frame. The wrapper
  receives `X_train` already transformed by the caller-supplied
  preprocessor. The wrapper does not refit the preprocessor.
- `W_continuous`: the preserved pre-threshold continuous `W`.
- `A_thresh`: `_threshold_continuous_w(W_continuous, project_threshold)`.

Estimation steps:

1. Build the sampling weight matrix on surviving edges:
   `W_sample = W_continuous * A_thresh.astype(W_continuous.dtype)`.
2. Compute residuals in the model frame:
   `R = X_model_frame - X_model_frame @ W_sample`.
3. Per-node sigma:
   `sigma_j = float(R[:, j].std(ddof=0))` for each node `j`.
4. Validate per-node sigmas (Section 11.2).

The residuals are computed against the thresholded sampling weight
matrix because the wrapper samples from the thresholded structure;
the residual must therefore capture variance the thresholded model
does not predict (see decision-log refinement on this point in
`docs/03_decision_log.md` and `docs/04c_runtime_probe_results.md`
D-P5).

### Sigma validation policy (no silent floor)

The wrapper does NOT introduce a variance floor (no `1e-6`
substitute, no clamp to `eps`). If any `sigma_j` is non-finite or
`<= 0`:

- the full per-node sigma vector and the failure reason are stored
  in diagnostics under the key
  `sigma_validation_failure_reason`,
- `sampler_status = unavailable_unresolved_noise_policy` is set, with
  the reason "Residual-fitted sigma estimate non-finite or
  non-positive at column j",
- `sample_interventional` returns `None`,
- no MMD sample is produced for that run.

This is the only resolution path that avoids silently altering the
generative model. A future amendment may propose a floor as an
explicit protocol change, but no such floor is part of the current
wrapper.

### Sensitivity policy: unit-variance noise

Doc 02 v1.3 Section 4.2 step 8 frozen sensitivity check. The
wrapper exposes this as an alternative noise mode at sample time
through an explicit method or argument:

```
def sample_interventional(
    self,
    intervention: Intervention,
    n_samples: int,
    *,
    sample_seed: int,
    noise_policy: Literal["residual_fitted", "unit_variance"] = "residual_fitted",
) -> Optional[np.ndarray]: ...
```

The default is `"residual_fitted"`, matching the Doc 02 v1.3 primary
policy. The selection-study runner is responsible for calling the
sampler under both policies and reporting both MMD values in the
selection-study record (Doc 02 v1.3 Section 3.4). The wrapper does
not maintain a separate stored sigma for unit-variance; it simply
overrides `sigma_j = 1.0` for every `j` at sample time. The diagnostics
record `noise_policy_default` and the diagnostics produced per call
record `noise_policy_used`.

### Tests required

- residual sigma matches a hand-computed value for a known
  3-node fixture;
- sigma vector logged in diagnostics has length `num_vars` and dtype
  `float`;
- a degenerate fixture (variance zero, NaN, or negative) triggers
  `sampler_status = unavailable_unresolved_noise_policy` and
  `sample_interventional` returns `None`;
- unit-variance mode is exercised and produces a different sigma
  field in the per-call diagnostics record.

---

## 12. Interventional sampler design

### Architecture

DAGMA has no learned conditional density (D-10, D-8). The wrapper-side
sampler is a plain linear-Gaussian ancestral sampler conditioned on
the thresholded adjacency and the chosen per-node sigma vector.

### Sampler sequence

For an `Intervention(target, value)` and `n_samples`:

1. Compute
   `A_thresh = _threshold_continuous_w(self._continuous_w_pre_threshold, self._cfg.project_threshold)`.
2. Classify `graph_status` via the shared helper (Section 9). If not
   `valid_dag`, set `sampler_status = unavailable_invalid_graph` and
   return `None`.
3. If `sampler_status` was previously set to
   `unavailable_unresolved_noise_policy` at fit time (Section 11),
   return `None` and propagate the reason.
4. Resolve the per-node sigma vector for this call from
   `noise_policy`:
   - `"residual_fitted"`: use `self._sigma_residual_fitted`.
   - `"unit_variance"`: use `np.ones(num_vars, dtype=float)`.
5. Build `W_sample = self._continuous_w_pre_threshold * A_thresh`.
6. Transform the raw intervention value into the model frame:
   `v_model = self._preprocessor.transform_intervention_value(intervention.value, intervention.target)`.
7. Construct a sampler RNG:
   `rng = np.random.default_rng(sample_seed)`. The wrapper does NOT
   rely on global NumPy state and does NOT call
   `np.random.seed(sample_seed)` for this purpose, so two
   `sample_interventional` calls with different `sample_seed` values
   in the same Python process do not interfere.
8. Compute the topological order of `A_thresh`. The wrapper reuses
   the project's `_topological_order` helper from
   `src/symbolic_priors_cd/wrappers/_dcdi_sampling.py` via the
   shared sibling module recommended in Section 9, or via a duplicate
   wrapper-local helper. The recommended option is to move
   `_topological_order` to a shared utility module alongside
   `classify_graph_status`.
9. Allocate `X = np.zeros((n_samples, num_vars), dtype=float)`.
10. For each node `j` in topological order:
    - if `j == intervention.target`: clamp `X[:, j] = v_model`.
    - otherwise: compute
      `mean_j = X @ W_sample[:, j]` (parents row-indexed) and draw
      `X[:, j] = mean_j + rng.normal(loc=0.0, scale=sigma_j, size=n_samples)`.
11. Inverse-transform back to raw SCM units via
    `self._preprocessor.inverse_transform(X)`.
12. Return the raw-unit `X` of shape `(n_samples, num_vars)`.

### Tests for sampler mechanics (mandatory pytest gates)

- `test_sampler_clamping_raw_value_exact`: target column equals the
  requested raw value after inverse transform (within float64
  tolerance).
- `test_sampler_shape_and_dtype`: shape `(n_samples, num_vars)`,
  dtype `float64`, no NaN.
- `test_sampler_deterministic_with_sample_seed`: two calls with the
  same `sample_seed` produce identical output element-wise.
- `test_sampler_different_seeds_differ`: two calls with different
  `sample_seed` values produce different output.
- `test_sampler_refuses_invalid_graph`: invalid `A_thresh` causes
  `None` return and `sampler_status = unavailable_invalid_graph`.
- `test_sampler_refuses_degenerate_sigma`: degenerate sigma causes
  `None` return and `sampler_status = unavailable_unresolved_noise_policy`.
- `test_sampler_raw_unit_roundtrip_centred_only` and
  `test_sampler_raw_unit_roundtrip_standardised`: clamped value
  equals raw request after the full transform-inverse-transform
  cycle.
- `test_sampler_preprocessor_not_refit`: the preprocessor's stored
  statistics are unchanged after several sampling calls.
- `test_sampler_unit_variance_policy`: sigma override mode is honoured
  at call time and exposed in per-call diagnostics.

These are sampler MECHANICS tests. Sampler QUALITY (whether the
generated distribution is close to the true interventional
distribution) is handled in Section 14 as a diagnostic probe.

---

## 13. Source-faithfulness check

### Why source-faithfulness rather than DCDI-style behavioural equivalence

The DCDI wrapper reimplemented DCDI's training loop because the
official entry point pulls in optional R dependencies. That
reimplementation creates a genuine risk of silent divergence from
the inspected source, which the DCDI plan addresses with a
tiered behavioural-equivalence test against a hand-replicated
reference loop.

DAGMA is different. The wrapper calls `DagmaLinear.fit` directly,
with explicit hyperparameter values and a defensive `X.copy()`. The
wrapper does not reimplement DAGMA's path-following Lagrangian
algorithm. The appropriate gate is therefore source-faithfulness:
the wrapper's output must match a direct
`DagmaLinear.fit(X.copy(), ...)` call with the same inputs and the
same hyperparameters.

Copying the DCDI tiered training-loop equivalence structure would be
ceremony, not a scientific check.

### Source-faithfulness gate

For a 5-node Gaussian SCM at seed 0, sample 200 observational
observations.

Step A (direct call): run

```
model_direct = DagmaLinear(loss_type="l2")
W_direct = model_direct.fit(
    X=X_train.copy(),
    lambda1=cfg.lambda1,
    w_threshold=0.0,
    T=cfg.T,
    mu_init=cfg.mu_init,
    mu_factor=cfg.mu_factor,
    s=list(cfg.s),
    warm_iter=cfg.warm_iter,
    max_iter=cfg.max_iter,
    lr=cfg.lr,
    beta_1=cfg.beta_1,
    beta_2=cfg.beta_2,
)
```

Step B (wrapper call): construct a `DAGMAWrapper`, pass
`CentredOnlyTransform` fitted on the same `X_train`, transform
`X_train` to model frame, call `wrapper.fit(X_train_model_frame,
preprocessor=..., seed=..., config=DAGMAConfig())`, and read
`W_wrapper = wrapper.native_edge_continuous()`.

Because DAGMA mean-centres internally at `linear.py:307`, passing
already-centred data is consistent with passing raw data: centring a
zero-mean array is a no-op (Doc 02 v1.3 Section 4.4 final paragraph).
The source-faithfulness gate uses identical numerical inputs in both
paths so the only freedom is implementation faithfulness.

### Tolerance policy

Default acceptance:

```
assert np.allclose(W_wrapper, W_direct, atol=1e-12, rtol=1e-12)
```

Optional stricter check: `np.array_equal(W_wrapper, W_direct)` for
bitwise equality. The stricter check is enabled if it is stable on
the project hardware; otherwise the `1e-12` allclose is the binding
gate. Loose numerical tolerances are not acceptable for a direct-call
equivalence check.

### Tests

- `test_source_faithfulness_no_preprocessor`: pass identical raw
  `X.copy()` through both paths (wrapper with a no-op preprocessor
  or with a centred-only preprocessor on already-centred data) and
  assert `allclose(atol=1e-12, rtol=1e-12)`.
- `test_source_faithfulness_explicit_hyperparameters`: when
  `config=None` is passed to the wrapper, `config_snapshot` records
  exactly the Doc 02 values (`T=4`, `lambda1=0.05`,
  `s=(1.0, 0.9, 0.8, 0.7)`, `mu_init=1.0`, `mu_factor=0.1`,
  `w_threshold_internal=0.0`).
- `test_wrapper_does_not_mutate_X`: pre-fit and post-fit copies of
  the caller's `X_train` are bitwise equal (the wrapper-internal
  `X.copy()` does its job and the caller's array is never mutated).

These tests live in `tests/test_dagma_wrapper_source_faithfulness.py`.

---

## 14. Sampler-quality diagnostic

### Form

Sampler-quality is delivered as an INSPECTION PROBE and a markdown
REPORT from the start, not as a pytest gate. Files:

- `inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py`
- `docs/04h_dagma_sampler_quality_diagnostic.md`

### Why a probe rather than a pytest gate

C-P11 demonstrated that learned sampler-quality can fail for
base-model reasons that have nothing to do with wrapper correctness.
Encoding such a failure as a pytest gate causes CI to fail for a
science finding, not a code defect. Sampler-quality is therefore
treated as a diagnostic from the start. Sampler MECHANICS (Section 12)
remain pytest gates.

### Fixture (mirrors C-P11)

- `generate_linear_gaussian_scm(n_nodes=3, expected_edges=3, seed=0)`.
- Training data: 5000 observational samples,
  `sample_observational` seed = 1.
- Validation data: 500 observational samples,
  `sample_observational` seed = 2. Not directly used in MMD but
  recorded for comparability with C-P11.
- Preprocessing: `CentredOnlyTransform` fitted on training data.
- Intervention: `do(X_2 = 2.0)` in raw SCM units.
- Per-batch sample size: 1000.
- `n_floor = 5`, `n_wrapper = 5`.
- Seed bases:
  - `GT_FLOOR_SEED_BASE = 1000`
  - `GT_PAIRED_SEED_BASE = 1100`
  - `GT_WRONG_SEED_BASE  = 1200`
  - `WRAPPER_SEED_BASE              = 2000`
  - `WRAPPER_WRONG_SEED_BASE        = 2100`
  - `WRAPPER_TRUE_SEED_BASE         = 2200`   (Diagnostic A)
  - `WRAPPER_LEARNED_AUG_SEED_BASE  = 2300`   (Diagnostic B1)
  - `WRAPPER_ORACLE_AUG_SEED_BASE   = 2400`   (Diagnostic B2)

The four wrapper seed bases at 2000, 2100, 2200, 2300, and 2400 give
each diagnostic an independent seed lane.

### Hyperparameters

DAGMAConfig() defaults, that is the Doc 02 v1.3 frozen values plus
the library defaults for `lr`, `warm_iter`, `max_iter`, `beta_1`,
`beta_2`, `loss_type`.

### Quantities recorded

- DAGMA training outcome: `h_final`, `score_final`, derived
  `training_status`.
- Learned continuous `W` (full pre-threshold matrix).
- Learned thresholded boolean adjacencies at thresholds 0.2, 0.3,
  and 0.4, computed by applying `abs(W_continuous) >= t` at each
  `t`. The project-level threshold 0.3 is the canonical
  selection-study threshold; 0.2 and 0.4 are the neighbouring
  thresholds for threshold-robustness reporting (Doc 02 v1.3
  Section 7 item 5).
- Threshold-grid edge counts at `{0.2, 0.3, 0.4}` derived from the
  recorded thresholded adjacencies.
- Residual sigma vector under the residual-fitted noise policy.
- Sigma vector under the unit-variance sensitivity policy (constant
  vector of ones).
- `graph_status` for the learned adjacency.
- Monte Carlo floor MMD: median pairwise MMD across 5 ground-truth
  interventional batches.
- Wrapper-vs-truth median MMD under residual-fitted noise.
- Wrapper-vs-truth median MMD under unit-variance noise.
- Correct-structure vs wrong-structure MMD (Diagnostic on structure
  sensitivity).
- Diagnostic A: MMD when the wrapper is forced to sample under the
  TRUE adjacency.
- Diagnostic B1: MMD when the wrapper is forced to sample under the
  learned adjacency plus the strongest missing true edge, with
  DAGMA's own learned `W_continuous` value for that edge.
- Diagnostic B2: MMD when the wrapper is forced to sample under the
  learned adjacency plus the strongest missing true edge, with the
  true SCM weight for that edge.

### Diagnostic A: structural correctness probe

Build a forced wrapper state in which `_continuous_w_pre_threshold`
is replaced by a tensor whose 0.3-threshold equals the true SCM
adjacency, while keeping DAGMA's learned `W_continuous` values on
surviving edges (entries in `A_true` already present in the learned
adjacency keep their learned value; entries in `A_true` missing from
the learned adjacency are set to the wrapper-substituted value
specified below).

Diagnostic A uses DAGMA's own learned `W_continuous` value for
missing-but-true edges. Concretely: if the true edge `(i, j)` is
missing from the learned thresholded adjacency, the Diagnostic A
sampling weight matrix uses
`W_sample[i, j] = self._continuous_w_pre_threshold[i, j]` even though
`|W_sample[i, j]| < 0.3`. The residual sigma vector is recomputed in
the model frame against this Diagnostic A `W_sample`. The aim is to
test whether the wrapper-vs-truth gap collapses when the structural
mask is corrected but DAGMA's own coefficient learning is otherwise
trusted.

If DAGMA's continuous `W` for a missing true edge is exactly zero
(numerically), Diagnostic A's signal collapses into Diagnostic B2.
The probe records the absolute value of every missing true edge's
learned continuous `W` and flags this collapse explicitly when it
happens.

### Diagnostic B1: learned-weight augmentation

Identify the strongest true edge by absolute true weight that is
absent from the learned thresholded adjacency. Build a sampling
weight matrix that augments the learned thresholded adjacency with
that edge, using DAGMA's own learned continuous `W` value for that
edge (which is below threshold). Recompute residual sigmas in the
model frame for this augmented weight matrix. Sample under the
augmented adjacency and compute wrapper-vs-truth MMD.

This tests whether the missing edge was present in DAGMA's
continuous solution as a sub-threshold signal that thresholding
suppressed.

### Diagnostic B2: oracle-weight augmentation

Same as B1 except the missing true edge is added at its true SCM
weight magnitude (with the true sign). Recompute residual sigmas in
the model frame for this oracle-augmented weight matrix. Sample
under this augmented adjacency and compute wrapper-vs-truth MMD.

This is an oracle localisation check, not a baseline. It tests
whether restoring the correct structural and coefficient information
substantially reduces wrapper-vs-truth MMD. The report makes this
clear: B2 uses ground-truth information that is unavailable in the
real selection study.

Why both B1 and B2 are needed: B1 isolates the contribution of
thresholding by reusing DAGMA's own coefficient. B2 isolates the
joint contribution of structure plus correct coefficient. If only B1
or only B2 were run, the report could not distinguish "DAGMA learned
the edge weakly and thresholding hid it" from "DAGMA never recovered
the right magnitude."

### Negative-MMD floor caveat

The probe records the literal comparison `wrapper_vs_truth <= 3 * floor`
for comparability with C-P11. The report explicitly notes that:

- the unbiased MMD estimator can be negative when both samples come
  from the same distribution;
- if the floor is negative, `3 * floor` is not a meaningful positive
  acceptance criterion;
- the substantive comparison is the order-of-magnitude gap between
  the absolute scale of the floor and the wrapper-vs-truth MMD;
- the localisation interpretation comes from Diagnostic A, B1, B2,
  and the wrong-structure comparison.

The probe and the report do NOT pretend `wrapper_vs_truth <= 3 * floor`
is mathematically meaningful when the floor is negative.

### Wrong-structure comparison

For the 3-node fixture used here, the intervention target `X_2` is
the source of the true topological order, so every true edge sits
downstream of the intervention. The wrong-structure comparison
deletes the strongest true edge that is present in the learned
thresholded adjacency and is on the downstream path of the
intervention. The probe records the `wrong / correct` MMD ratio and
the literal comparison `correct * 1.5 <= wrong` for comparability
with C-P11.

If on this fixture DAGMA's learned adjacency contains no downstream
edge to delete, the probe records the situation explicitly and falls
back to the hand-selected 3-node chain `X0 -> X1 -> X2` with true
weights `[1.5, 1.5]` and intervention on `X_1`, mirroring the
fallback design in the DCDI plan. The fallback fixture is documented
in the probe script.

### Reporting in `docs/04h_dagma_sampler_quality_diagnostic.md`

The report must record:

- the exact frozen setup,
- DAGMA training outcome (`h_final`, `score_final`, training_status),
- the learned continuous `W`, the learned thresholded adjacency, and
  the threshold-grid edge counts,
- the residual sigma vector and the unit-variance sigma vector,
- every MMD listed above with explicit values and seed bases,
- the literal comparisons against the C-P11-style thresholds,
- the negative-floor caveat,
- the localisation interpretation derived from A, B1, B2,
- a statement that no acceptance threshold has been weakened and no
  silent graph repair has been introduced,
- the explicit reading that DAGMA's performance on this fixture is a
  base-model property and is NOT used to revise the wrapper
  acceptance criteria.

The report does not propose DAGMA-versus-DCDI conclusions. The
selection study is the appropriate place for that comparison.

---

## 15. Diagnostics and logging

### Schema option (recommended)

Recommendation: extend the diagnostics module so the public schema
captures fields common to all wrappers and uses a
`model_specific_diagnostics: dict[str, object]` field for fields
that only one model populates. Concretely:

```
class WrapperDiagnostics(TypedDict):
    training_status: TrainingStatus
    graph_status: GraphStatus
    sampler_status: SamplerStatus
    seed: int
    n_iterations: int
    config_snapshot: dict[str, object]
    loss_history: list[float]              # empty list for DAGMA
    loss_decomposition_final: dict[str, float]
    convergence_info: dict[str, object]
    thresholded_adjacency: np.ndarray
    graph_invalid_reason: Optional[str]
    sampler_unavailable_reason: Optional[str]
    mmd_sampling_metadata: dict[str, object]
    loss_hook_name: Optional[str]
    numerical_tolerances: dict[str, float]
    model_specific_diagnostics: dict[str, object]
```

The two existing DCDI-only fields
`continuous_log_alpha_pre_threshold` and
`continuous_w_adj_pre_threshold` move from the top-level TypedDict
to the DCDI side of `model_specific_diagnostics`. This is a
low-disruption refactor at this time because no DCDI commit currently
returns a populated `WrapperDiagnostics` (Commit 12 of Doc 05 is
paused). The refactor lands in DAGMA Commit 1 of this plan together
with the DAGMA scaffolding.

Trade-off note: when DCDI Commit 12 eventually lands, the DCDI
diagnostics assembly will need to populate the DCDI-specific
sub-dictionary rather than top-level fields. This is a small change
in one location of `wrappers/dcdi.py` and does not touch any test
that currently runs in the green suite.

### Less disruptive fallback

If the project review explicitly prefers not to refactor the existing
`WrapperDiagnostics` shape, the fallback is to introduce a sibling
TypedDict named `DagmaWrapperDiagnostics` that shares the common
fields and adds DAGMA-specific fields directly. The two TypedDicts
share keys for `training_status`, `graph_status`, `sampler_status`,
`seed`, `n_iterations`, `config_snapshot`, `convergence_info`,
`thresholded_adjacency`, `graph_invalid_reason`,
`sampler_unavailable_reason`, `mmd_sampling_metadata`, and
`numerical_tolerances`. The DAGMA-specific fields are then top-level
on the DAGMA TypedDict only. This avoids touching `status.py` at
all, at the cost of two parallel schemas the selection-study runner
must handle.

The plan recommends the refactor option. The fallback is documented
so the choice is explicit.

### DAGMA-specific diagnostic fields

Independently of the schema choice, the DAGMA-specific fields are:

- `continuous_w_pre_threshold`: `np.ndarray` of shape
  `(num_vars, num_vars)`, the canonical pre-threshold continuous `W`.
- `h_final`: `float`.
- `score_final`: `float`.
- `sigma_vector_residual_fitted`: `np.ndarray` of shape
  `(num_vars,)`, or `None` if estimation was degenerate.
- `sigma_validation_failure_reason`: `Optional[str]`.
- `near_threshold_entry_count`: `int`, number of entries
  satisfying `1e-12 < abs(W_continuous) < project_threshold`. This
  captures the count of sub-threshold signal entries that would
  contribute if the threshold dropped, and informs threshold
  robustness reporting (Doc 02 v1.3 Section 7 item 5).
- `threshold_grid_edge_counts`: dict mapping
  `{0.2, 0.3, 0.4}` to integer edge counts after applying
  `abs(W_continuous) >= t` at each `t`.
- `dagma_source_path`: string path to the resolved `DagmaLinear`
  module file, recorded so an audit can confirm the inspected
  source was used and not an installed package.

### Per-call sampler diagnostics

For each `sample_interventional` call, a per-call dict is appended to
`mmd_sampling_metadata.calls` (or its DAGMA-side equivalent):

- `sample_seed`,
- `noise_policy_used` ("residual_fitted" or "unit_variance"),
- `sigma_vector_used`,
- `target`, `value`, `n_samples`,
- `graph_status_at_call`,
- `sampler_status_at_call`,
- returned-shape and dtype, or `None` if the call returned `None`.

### Reproducibility

`config_snapshot` records every resolved hyperparameter so any run can
be reproduced from its diagnostics alone, without the original call
site.

---

## 16. Required tests

Grouped by test file. Each test is mandatory before the corresponding
commit is considered complete. All tests are part of the normal
pytest collection.

### `tests/test_dagma_wrapper_interface.py`

- `test_dagma_source_path_uses_inspected_clone`
- `test_dagma_config_resolves_doc02_defaults`
- `test_native_edge_continuous_shape_and_dtype`
- `test_thresholded_adjacency_default_0_3`
- `test_thresholded_adjacency_uses_helper`
- `test_mock_orientation_dagma`
- `test_threshold_monotonicity`
- `test_diagnostics_completeness_dagma`
- `test_dagma_source_path_recorded_in_diagnostics`

### `tests/test_dagma_wrapper_source_faithfulness.py`

- `test_source_faithfulness_no_preprocessor`
- `test_source_faithfulness_centred_only_no_op`
- `test_source_faithfulness_explicit_hyperparameters_in_config_snapshot`
- `test_wrapper_does_not_mutate_X`
- `test_wrapper_does_not_call_dagma_set_random_seed`
- `test_wrapper_does_not_use_exclude_or_include_edges`

### `tests/test_dagma_wrapper_thresholding.py`

- `test_threshold_helper_single_source`
- `test_graph_status_valid_dag_dagma`
- `test_graph_status_cyclic_dagma`
- `test_graph_status_bidirected_dagma`
- `test_invalid_graph_no_silent_repair_dagma`
- `test_sampler_status_invalid_graph_dagma`

### `tests/test_dagma_wrapper_residual_noise.py`

- `test_residual_sigma_against_hand_computed_3_node_fixture`
- `test_residual_sigma_recorded_in_diagnostics`
- `test_unit_variance_policy_overrides_at_sample_time`
- `test_degenerate_sigma_marks_sampler_unresolved_noise`
- `test_no_silent_variance_floor`

### `tests/test_dagma_wrapper_sampler.py`

- `test_sampler_clamping_raw_value_exact`
- `test_sampler_shape_and_dtype_dagma`
- `test_sampler_deterministic_with_sample_seed_dagma`
- `test_sampler_different_seeds_differ_dagma`
- `test_sampler_refuses_invalid_graph_dagma`
- `test_sampler_refuses_degenerate_sigma_dagma`
- `test_sampler_raw_unit_roundtrip_centred_only_dagma`
- `test_sampler_raw_unit_roundtrip_standardised_dagma`
- `test_sampler_preprocessor_not_refit_dagma`
- `test_sampler_unit_variance_policy_dagma`

### Sampler-quality

No pytest test for learned sampler-quality. The inspection probe
`inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py` and the
report `docs/04h_dagma_sampler_quality_diagnostic.md` are the
artefacts.

### Tests deliberately NOT added

- No tiered behavioural-equivalence training-trajectory tests. DAGMA
  is not reimplemented; source-faithfulness replaces those tests.
- No loss-hook gradient tests. The hook is deferred (Section 18).

---

## 17. Atomic commit sequence

Commits are conservative and reviewable. Each commit must pass its
acceptance criterion before the next is started.

| # | Title | Files touched | Tests added | Acceptance | Risks |
|---|---|---|---|---|---|
| 1 | scaffolding, config, import boundary, diagnostics schema | `wrappers/dagma.py` (skeleton), `wrappers/_dagma_utils.py`, `wrappers/__init__.py` re-exports, `wrappers/status.py` schema refactor option | `test_dagma_source_path_uses_inspected_clone`, `test_dagma_config_resolves_doc02_defaults` | imports cleanly; DAGMA source path resolves to the inspected clone; existing DCDI tests remain green; `DAGMAConfig` defaults match Doc 02 v1.3 verbatim | LOW; refactor of `WrapperDiagnostics` is the main moving piece |
| 2 | fit path with `X.copy()` and explicit hyperparameters | `wrappers/_dagma_fit.py`, `wrappers/dagma.py::fit` | `test_wrapper_does_not_mutate_X`, `test_wrapper_does_not_use_exclude_or_include_edges`, `test_wrapper_does_not_call_dagma_set_random_seed` | a fit call on a tiny SCM completes; `X_train` is bitwise unchanged after fit; no hard-constraint argument is set | LOW |
| 3 | continuous `W` preservation | `wrappers/_dagma_fit.py`, `wrappers/dagma.py::native_edge_continuous` | `test_native_edge_continuous_shape_and_dtype`, `test_continuous_w_unchanged_by_sampling_calls` | `native_edge_continuous()` returns float64 `(d, d)` matching the post-fit `W_est`; preserved tensor unchanged across multiple sampling-style calls (sampling itself wired in Commit 7) | LOW |
| 4 | source-faithfulness gate | `tests/test_dagma_wrapper_source_faithfulness.py` | `test_source_faithfulness_no_preprocessor`, `test_source_faithfulness_centred_only_no_op`, `test_source_faithfulness_explicit_hyperparameters_in_config_snapshot` | `np.allclose(W_wrapper, W_direct, atol=1e-12, rtol=1e-12)`; optional `np.array_equal` passes if stable on project hardware | MODERATE; tightness of the tolerance is the main item |
| 5 | thresholding and graph-status machinery | `wrappers/dagma.py::thresholded_adjacency`, shared graph-status helper move (move `classify_graph_status`, `_is_acyclic_adjacency`, and `_topological_order` from `wrappers/dcdi.py` and `wrappers/_dcdi_sampling.py` into a sibling utility module such as `wrappers/_graph_status.py`; DCDI continues to import them from the new location) | `tests/test_dagma_wrapper_thresholding.py` (six tests) | `_threshold_continuous_w` is the single source of thresholding; all `graph_status` branches reachable; invalid graph is not repaired; **all existing DCDI wrapper tests remain green after the helper move** (binding requirement; if any DCDI test regresses, Commit 5 is not complete) | LOW to MODERATE; the helper move must not change DCDI behaviour |
| 6 | residual sigma estimation | `wrappers/_dagma_sampling.py::estimate_residual_sigmas`, integration into `fit` | `tests/test_dagma_wrapper_residual_noise.py` (five tests) | hand-computed sigmas match within `1e-12`; degenerate sigma triggers `unavailable_unresolved_noise_policy`; no silent variance floor | LOW to MODERATE |
| 7 | model-frame sampler | `wrappers/_dagma_sampling.py::sample_model_frame_dagma`, `wrappers/dagma.py::sample_interventional` (no preprocessor wiring yet) | `test_sampler_clamping_raw_value_exact` (against model-frame value), `test_sampler_shape_and_dtype_dagma`, `test_sampler_deterministic_with_sample_seed_dagma`, `test_sampler_different_seeds_differ_dagma`, `test_sampler_refuses_invalid_graph_dagma`, `test_sampler_refuses_degenerate_sigma_dagma` | clamping invariant; deterministic seed contract; refusal on invalid graph and degenerate sigma | MODERATE |
| 8 | raw-unit preprocessor roundtrip | wire `preprocessing.py` into the DAGMA sampler | `test_sampler_raw_unit_roundtrip_centred_only_dagma`, `test_sampler_raw_unit_roundtrip_standardised_dagma`, `test_sampler_preprocessor_not_refit_dagma`, `test_sampler_unit_variance_policy_dagma` | clamped target equals raw request after inverse transform; preprocessor statistics unchanged after several calls; unit-variance policy override honoured | LOW |
| 9 | sampler-quality diagnostic probe and report | `inspection/probes/c_p13_dagma_sampler_quality_diagnostic.py`, `docs/04h_dagma_sampler_quality_diagnostic.md` | none added to pytest collection | probe runs on the project hardware and writes the report verbatim with all values recorded; no acceptance threshold weakened; no silent graph repair introduced | resolved as a base-model/wrapper-design open question, not a CI failure |
| 10 | diagnostics and logging | `wrappers/dagma.py::get_diagnostics`, `wrappers/_dagma_utils.py::assemble_diagnostics` | `test_diagnostics_completeness_dagma`, `test_dagma_source_path_recorded_in_diagnostics` | all common and DAGMA-specific diagnostic keys present; per-call sampler metadata recorded; source path recorded | LOW |
| 11 | final readout and public API stabilisation | docstrings on `DAGMAWrapper`, `wrappers/__init__.py` re-exports, optional `docs/phase_2c_dagma_readout.md` | none | docstrings cite Doc 02 v1.3, Doc 04, Doc 04b, Doc 04c, Doc 06 (this plan), Doc 04h (report) | trivial |

Loss-hook is NOT a numbered commit in this plan. See Section 18.

Hard-constraint baseline through DAGMA's `exclude_edges` /
`include_edges` is NOT a numbered commit in this plan. See
Section 18.

---

## 18. Risks, open questions, and readiness criteria

### Risks and unresolved questions

1. Source-faithfulness tolerance. The plan defaults to
   `allclose(atol=1e-12, rtol=1e-12)` and opportunistically attempts
   `array_equal`. If `array_equal` is unstable on the project
   hardware (for example because BLAS implementation differences
   produce tiny non-deterministic differences), the binding gate is
   the allclose tolerance. The tightness is justified by DAGMA being
   deterministic given fixed inputs (D-6, D-P2), and any drift past
   `1e-12` is a wrapper bug, not a numerical artefact.
2. Residual sigma stability. If DAGMA's learned thresholded
   adjacency misses important true edges, residual sigmas estimated
   in the model frame may absorb that variance and be inflated.
   This is a base-model behaviour, not a wrapper bug. The
   sampler-quality diagnostic A and B1 / B2 distinguish residual
   inflation due to missing structure from residual inflation due to
   bad coefficient learning.
3. Threshold sensitivity. Doc 02 v1.3 specifies the threshold triple
   `{0.2, 0.3, 0.4}` for DAGMA threshold robustness reporting. The
   wrapper records edge counts at all three thresholds in diagnostics,
   but selection-study aggregation of robustness behaviour is the
   selection-study runner's responsibility, not the wrapper's.
4. DAGMA convergence interpretation. `h_final <= 1e-5` is the
   wrapper diagnostic threshold (Section 10). This is a
   training-status diagnostic only; `graph_status` is derived
   independently from the thresholded boolean adjacency. The two
   axes are independent per the wrapper contract.
5. Non-finite DAGMA outputs. The wrapper guards against non-finite
   entries in `W_continuous` and `h_final` and raises through the
   wrapper boundary if they appear. This is recorded as
   `training_status = "diverged"` in diagnostics before re-raise.
6. RNG hygiene. The wrapper uses `np.random.default_rng(sample_seed)`
   for sampler RNG and does not call `np.random.seed` for sampling
   purposes. If a future change requires interacting with DAGMA
   utilities that mutate global NumPy state, the wrapper must
   save and restore that state around the call; until such a need
   appears, the wrapper avoids global-state mutation entirely.
7. Verified SID. The DAGMA wrapper does not depend on verified SID,
   but selection-study conclusions cannot be declared final until
   verified SID is integrated and the SID scaffold test in
   `tests/test_interventional_metrics.py` is unskipped. This is a
   parallel project-level blocker, mirrored from Doc 02 v1.3
   Section 7 item 6.

### Loss-hook (deferred)

The thesis main-study soft prior acts on DAGMA's continuous `W`
matrix (Doc 01 Section 8). DAGMA's training-time gradient is
hand-coded inside `minimize` at `linear.py:205-210`, not assembled
through a generic loss function. Adding a soft-prior gradient term
therefore requires one of two strategies, both more invasive than
DCDI's autograd-based hook:

- subclass `DagmaLinear`, override `minimize`, and add
  `grad_prior(W)` to the inline `Gobj` assembly before the Adam step
  (the natural strategy per D-3); or
- monkey-patch `DagmaLinear.minimize` at wrapper-call time using a
  module-private subclass.

Both strategies require the wrapper to own the soft-prior gradient
expression, because DAGMA does not propagate it through autograd
automatically. For L1-style penalties this is straightforward
(`sign(W) * confidence_mask` with the standard `sign(0) = 0`
convention). The wrapper will not implement this until after:

- DAGMA wrapper mechanics (Commits 1 through 8) are merged green;
- the DAGMA sampler-quality diagnostic (Commit 9) is reviewed at the
  project level;
- the DCDI Commit 11 review either resumes or formally cancels DCDI
  loss-hook work, so DAGMA's loss-hook design can be checked against
  the project decision.

Until then, `DAGMAWrapper` does not expose `set_loss_hook` and the
public surface in Section 5 stands.

### Hard-constraint baseline through DAGMA

The base DAGMA wrapper MUST NOT use `exclude_edges` or
`include_edges` for any of: fitting, thresholding, post-hoc repair,
or improving selection-study performance. They are reserved for a
later, separately documented hard-constraint baseline implementation
recorded in Doc 03 before any main-study run uses it (per Doc 02
v1.3 Section 4.3 hard-constraint warning).

### Readiness criteria

#### Before Commit 1 begins

- The diagnostics schema decision (Section 15) is recorded
  explicitly: refactor of `WrapperDiagnostics`, or sibling
  `DagmaWrapperDiagnostics`.

#### Before sampler-quality probe runs (Commit 9)

- Commits 1 through 8 are green in the normal pytest collection.
- Residual sigma estimation is tested against a hand-computed
  3-node fixture.
- The probe file path and report file path match this plan.

#### Before the DAGMA wrapper is used in the selection study

- All Commits 1 through 11 are merged and the normal pytest
  collection remains green.
- The sampler-quality diagnostic report `docs/04h_dagma_sampler_quality_diagnostic.md`
  is written and reviewed at the project level.
- Verified SID integration is complete and the SID scaffold test is
  unskipped, OR the selection-study report explicitly defers SID
  per Doc 02 v1.3 Section 7 item 6.


---

## End of plan
