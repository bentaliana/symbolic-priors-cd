# Wrapper API Contract for DAGMA and DCDI

## Status

This document defines the scientific capabilities and invariants that any
DAGMA or DCDI wrapper in this project must satisfy before wrapper
implementation begins. It does not freezeconcrete Python method names or class structure. Final names and class structure may change after source inspection of the underlying DAGMA and DCDI codebases, provided the required capabilities and tests in this document remain satisfied.

The orientation audit recorded in `docs/04a_orientation_audit.md` found that
the current project code uses the row-source / column-destination convention:
`adjacency[i, j] = True` means a directed edge from node `i` to node `j`.
All wrappers must produce adjacency outputs in this convention before
passing them to the evaluator, once the audit is reviewed and accepted.

---

## 1. Purpose

The wrappers are the boundary between external causal discovery codebases
(DAGMA-linear, DCDI-G) and the project evaluator. They are responsible for
isolating model-specific quirks at one well-defined interface so the rest of
the project does not need to know how each model is implemented.

The wrappers must satisfy three downstream needs:

- The evaluator metrics need a thresholded boolean adjacency in the project
  orientation convention.
- The selection-study MMD checks need model-generated interventional samples
  where feasible.
- The later soft-prior experiments need loss-level access to the native
  continuous edge representation during training, with gradient flow.

A wrapper that only returns a final boolean adjacency is not sufficient.
Such a wrapper would block the soft-prior experiments and would not allow
threshold robustness checks without retraining.

---

## 2. Common wrapper capabilities

The following capabilities must exist in each wrapper. The exact Python
method names are deferred until source inspection, but each capability must
be addressable from outside the wrapper:

- Fit observational data.
- Expose the native continuous edge object after fitting.
- Expose a thresholded boolean adjacency after fitting.
- Expose training status.
- Expose graph validity status.
- Expose sampler availability status.
- Expose diagnostics.
- Support future additive loss-penalty injection during training.
- Support model-generated interventional sampling where feasible.

These capabilities are required regardless of whether DAGMA or DCDI is the
underlying model.

---

## 3. Native continuous edge object

The wrapper must expose the model-native continuous edge representation.

For DAGMA, this is `W`, the weighted adjacency matrix used by the optimisation
procedure.

For DCDI, this is `P = sigmoid(Lambda)`, the continuous edge-existence matrix
derived from the relaxed parameter `Lambda`.

The native edge object must be available in three distinct ways:

- During training, so that a future prior penalty can act on it with gradient
  flow.
- After fitting, so that it can be logged and re-thresholded offline without
  retraining.
- After fitting, so that it can support model-generated interventional
  sampling where feasible.

The contract distinguishes live training-time access from merely saving a
final post-fit matrix. Saving only the final matrix is necessary but not
sufficient. The wrapper must also allow a penalty term to read and
differentiate against the same object at training time.

---

## 4. Loss-hook capability

The wrapper must support adding an additive differentiable penalty to the
training objective. The penalty must act on the native continuous edge
representation defined above.

This contract does not freeze a specific callable API. The implementation
may later be:

- An in-process callback that receives the native edge object and returns a
  scalar penalty.
- A subclass or adapter that overrides the loss computation.
- A documented patch to the model loss within the wrapper boundary.
- A serialisable penalty specification, used only if a subprocess or a
  separate environment is required to run the model.

If a subprocess or a separate environment is used later, the penalty
specification must be JSON-serialisable or otherwise explicitly serialisable.
It must not rely on passing an in-memory Python callable across process
boundaries.

The planned thesis method only requires additive L1-style penalties on the
native edge representation. Non-additive loss composition is outside the
current contract. If a future research direction needs non-additive
composition, that is a separate amendment.

---

## 5. Thresholding

Thresholding belongs in the wrappers, not in the metric primitives. The
metric primitives accept already-thresholded boolean adjacency matrices.
The wrappers are responsible for converting native continuous edge objects
into boolean adjacency matrices in the project orientation convention.

Default thresholds at this stage of the project, taken from the selection
study protocol:

- DAGMA: apply a threshold of 0.3 to `abs(W_ij)`. An edge is present if
  `abs(W_ij) >= 0.3`.
- DCDI: apply a threshold of 0.5 to `P_ij`. An edge is present if
  `P_ij >= 0.5`.

The asymmetry is deliberate. DAGMA produces signed weights, so the natural
threshold uses absolute value. DCDI produces edge probabilities in `[0, 1]`,
so the natural threshold uses the value directly.

These defaults may be revised only by an explicitly approved amendment to
the selection study protocol.

The continuous edge object must always be saved together with any thresholded
adjacency, so that threshold robustness checks can be performed offline
without retraining.

---

## 6. No silent graph repair

The wrapper may validate the thresholded adjacency it produces but must not
silently modify it.

Specifically:

- The wrapper must not remove edges in order to break cycles.
- The wrapper must not break two-cycles by choosing the larger of the two
  edges.
- The wrapper must not symmetrise the output graph.
- The wrapper must not alter the output graph in any other way unless a
  later protocol explicitly authorises it.

If the thresholded adjacency is not a valid DAG, the wrapper must report
the output as invalid via the status taxonomy below, with an explicit reason.
The invalid output is still returned for inspection. It is not hidden, and
it is not replaced with a repaired version.

---

## 7. Status taxonomy

The wrapper exposes three independent status fields rather than one
overloaded fit status. Each axis answers a separate question.

`training_status` describes how the optimisation finished:

- `converged`
- `max_iter`
- `diverged`
- `wrapper_error`

DAGMA-specific note. The DAGMA wrapper does not implement observed
inner-loop early stopping; each stage of the central-path schedule runs
to its configured `warm_iter` or `max_iter` budget, and the top-level
`n_iterations` field on a DAGMA run record therefore remains `null`.
Within this taxonomy, `training_status = converged` for DAGMA means
`h_final <= h_diagnostic_threshold` after the configured optimisation
budget, and `training_status = max_iter` for DAGMA means `h_final` is
finite but above `h_diagnostic_threshold` after the configured
optimisation budget. The DAGMA converged-versus-`max_iter` distinction
is therefore an `h_final` predicate, not an early-stop iteration-count
predicate.

`graph_status` describes the structural validity of the thresholded boolean
adjacency:

- `valid_dag`
- `cyclic`
- `bidirected`
- `self_loop`
- `invalid_shape`

`sampler_status` describes whether the wrapper can generate interventional
samples:

- `available`
- `unavailable_invalid_graph`
- `unavailable_no_api`
- `unavailable_unresolved_noise_policy`

These three axes are independent. For example, training can converge while
the thresholded graph is cyclic, in which case `training_status = converged`
and `graph_status = cyclic`. Likewise, the graph can be a valid DAG while
the sampler is unavailable because the wrapper has not yet frozen a noise
policy, in which case `graph_status = valid_dag` and
`sampler_status = unavailable_unresolved_noise_policy`.

Invalid graph counts must be reported in the run record rather than hidden.
The selection study and any later analysis must be able to count failures
explicitly.

---

## 8. Metric behaviour for invalid graphs

The intended downstream metric behaviour when a wrapper reports a
non-`valid_dag` graph is:

- SID must refuse the input or report a NaN/status value, because SID
  requires a valid DAG. SID values must not be silently computed on a
  cyclic or bidirected graph.
- SHD may still be computed over a boolean matrix, but the result must be
  flagged as structurally invalid if the graph is not a valid DAG. The
  numerical SHD is preserved for inspection, but the run record must mark
  it as not directly comparable to SHD computed on valid DAGs.
- MMD sampling is unavailable if no valid graph or no usable sampler exists.
  In that case the run contributes no MMD value rather than a placeholder.
- Headline aggregations across seeds and conditions must report exclusion
  counts and failure counts so that aggregate numbers cannot hide silent
  failures.

This is a contract requirement on the run-record schema and the
aggregation pipeline. It is not yet implemented and is not required to be
implemented as part of the wrapper API itself, but the wrapper must produce
the inputs that allow this behaviour downstream.

---

## 9. Interventional sampling for MMD

Model-generated interventional sampling is treated as a protocol risk, not
as an automatic capability.

For DAGMA-linear:

- Post-hoc interventional sampling is likely to be a wrapper-side
  reconstruction strategy, because DAGMA-linear is a structure learner and
  not a generative model.
- A reconstruction strategy may use the fitted `W`, a thresholded valid
  graph, and an explicitly frozen noise and intercept policy.
- The noise and intercept policy is not frozen in this wrapper contract.
- The policy must be decided after source inspection of DAGMA, then frozen
  in a later amendment to the selection study protocol.

For DCDI-G:

- Post-hoc interventional sampling must be confirmed from the source code.
- It is not an assumption that the trained DCDI model can sample arbitrary
  `do(X_j = v)` interventions for unseen values of `v`.
- If no usable API exists for sampling under arbitrary single-node hard
  interventions, `sampler_status` must be `unavailable_no_api`.

Selection-study policy on MMD availability:

- If mean SID differs by more than 10 percent between candidates, MMD
  unavailability is a reported limitation but is not automatically
  selection-blocking, because SID alone is decisive.
- If mean SID is within the 10 percent tie margin, the current selection
  protocol requires MMD as a tiebreaker.
- If MMD is unavailable in that tie case, the protocol must be amended,
  the candidate must be disqualified with explicit rationale, or the
  selection must be declared inconclusive in that dimension.

---

## 10. Standardisation and intervention units

Intervention values are expressed in raw SCM units. For example,
`do(X_j = 2)` means raw value 2 in the original SCM units, even when the
wrapper standardises its training data internally.

If the wrapper standardises training data, it must transform any incoming
intervention into model-input space internally before generating samples,
and it must transform generated samples back to raw space before they are
compared against ground-truth interventional samples via MMD.

Failure to apply this round-trip would silently corrupt MMD comparisons,
because the ground-truth interventional samples are in raw SCM units.

This requirement should later be mirrored in an amendment to the selection
study protocol, because it affects how selection-study MMD numbers are
interpreted.

---

## 11. Fairness commitments

The wrappers will be compared against each other in the selection study,
so the comparison must not be polluted by accidental wrapper differences.

The following items must be shared across DAGMA and DCDI wrappers:

- Project orientation convention after the audit fixes it.
- Status taxonomy as defined in this document.
- Logging schema for training, graph, and sampler diagnostics.
- Seed discipline.
- Thresholding responsibility (always inside the wrapper).
- Invalid-output policy (no silent repair).
- Intervention-unit semantics (raw SCM units).
- Evaluator-facing output shape and dtype for boolean adjacency.

The following items can legitimately differ across wrappers, because they
reflect actual differences between the underlying models:

- Threshold value (0.3 for DAGMA, 0.5 for DCDI under current defaults).
- Hyperparameter grids frozen in the selection study protocol.
- Native continuous edge representation (`W` for DAGMA, `P = sigmoid(Lambda)`
  for DCDI).
- Internal optimiser.
- Model-specific source adaptation (any minimal patches needed to make the
  external code run under the project environment).
- Noise estimation strategy, once it has been frozen fairly for both models.

Asymmetries that are not justified by an actual model difference must be
removed before the selection study runs.

---

## 12. Diagnostics

Each wrapper must expose diagnostics that include at least the following:

- Final loss decomposition where available, so that later soft-prior work
  can prove the prior penalty actually contributed to training.
- Loss history, or a downsampled loss history, so that convergence can be
  inspected without retraining.
- Convergence information, including which stopping criterion fired.
- Edge counts at a small grid of thresholds around the project default,
  so that threshold sensitivity can be inspected offline.
- Graph validity reason when `graph_status` is not `valid_dag`.
- Sampler unavailable reason when `sampler_status` is not `available`.
- Wrapper warnings, including any non-fatal source patches applied.

Loss decomposition is particularly important for the later soft-prior
experiments. A wrapper that only logs total loss does not allow the prior
contribution to be measured at training time.

---

## 13. Determinism and reproducibility

Each wrapper must use the maximum deterministic settings that the
underlying framework supports.

The wrapper must produce identical thresholded boolean adjacency from the
same seed and configuration whenever feasible.

The native continuous edge object must be close within a documented
numerical tolerance if exact equality is not feasible. The tolerance must
be recorded in the wrapper diagnostics or documentation.

If bitwise determinism is not feasible, the reason must be documented
explicitly inside the wrapper documentation. Reasons may include
nondeterministic CUDA kernels, library-internal nondeterminism, or
unavoidable floating-point reductions. A wrapper that simply does not
seed correctly is not acceptable.

---

## 14. Source inspection checklist

Before wrapper implementation begins, the following questions must be
answered against the actual DAGMA and DCDI source code. These answers
inform the final wrapper design.

For DAGMA:

- Where is `W` stored after fitting.
- Where is the loss computed.
- Whether a custom additive loss term can be inserted at the loss
  computation site.
- What adjacency orientation the source uses.
- Whether the source standardises inputs.
- How seeds are handled.
- Whether the threshold 0.3 should apply to `abs(W)` or to `W` directly.
- Whether residual noise variance or intercepts are estimated by the
  fitting procedure.
- Whether the supplementary defaults match the selection study protocol.

For DCDI:

- What code path is used for observational-only training.
- Whether that code path matches the intended DCD-no-interv behaviour.
- Which loss components are active in that code path.
- Where `Lambda` is stored.
- Where `P = sigmoid(Lambda)` is materialised.
- Whether the trained conditionals can sample arbitrary `do(X_j = v)`
  interventions.
- What adjacency orientation the source uses.
- Whether the source standardises inputs.
- What convergence criterion is implemented.
- Whether the model works in the project environment or needs a separate
  environment.
- Whether a serialisable penalty specification would be possible if the
  model has to run in a separate environment.

For both:

- License terms.
- Supported deterministic settings.
- Default config alignment with the selection study protocol.

---

## 15. Tests required before a wrapper can be considered complete

The following tests must pass before a wrapper is treated as ready for the
selection study. The names below describe the tests, not Python identifiers.

- Shape and dtype test for the boolean adjacency output and the native
  continuous edge object.
- Mock-based orientation test using a fake native edge matrix. The test
  must construct a hand-chosen native edge matrix, pass it through the
  wrapper's threshold-and-orient stage, and verify the resulting boolean
  adjacency in the project row-source / column-destination convention.
  This must not depend on whether a stochastic fit recovers a one-edge
  graph.
- Threshold monotonicity test, verifying that increasing the threshold
  weakly decreases the number of present edges.
- Deterministic repeatability test using identical seed and configuration.
- Fit smoke test on a small SCM, verifying that the wrapper runs to
  completion without raising.
- Mechanical loss-hook gradient test, verifying that the additive penalty
  term flows gradient into the native edge representation.
- Behavioural shrinkage test on targeted edges. With a strong penalty
  applied to a small set of native edge entries, those entries must
  shrink relative to a no-penalty baseline.
- Optional stress test with a strong penalty applied to a true edge,
  verifying that the wrapper does not silently ignore the penalty.
- Sampler clamping test verifying that interventional samples have the
  intervened column exactly clamped to the requested value, when sampling
  is available.
- Standardisation roundtrip test verifying that interventions are applied
  in raw units and that generated samples are returned in raw units.
- Invalid graph behaviour test for a cyclic thresholded output.
- Invalid graph behaviour test for a bidirected thresholded output.
- Status taxonomy tests covering `training_status`, `graph_status`, and
  `sampler_status` independently.
- Diagnostics completeness test verifying that the diagnostics object
  contains all the fields listed in the diagnostics requirement.

The orientation test relies on a mock native edge object specifically so
that orientation correctness is not entangled with model fit quality.

---

## 16. Unresolved decisions

The following decisions are deliberately deferred and must be settled before
the relevant wrapper code is finalised:

- Formal decision-log entry for the project orientation convention, pending
  review of `docs/04a_orientation_audit.md`.
- DAGMA noise and intercept policy for post-hoc interventional sampling,
  pending source inspection.
- DCDI post-hoc sampling availability under arbitrary `do(X_j = v)`,
  pending source inspection.
- Concrete Python API names for the required capabilities, pending source
  inspection.
- MMD-unavailable tie policy in the selection study, pending an amendment
  to the selection study protocol.

---

## 17. What belongs elsewhere

The decision log should later record the orientation convention as a
project-level decision after `docs/04a_orientation_audit.md` is reviewed.

A future amendment to the selection study protocol should later cover:

- Raw-unit intervention semantics across wrappers.
- Invalid graph policy for SID, SHD, and MMD reporting.
- Threshold convention asymmetry between DAGMA and DCDI.
- Threshold robustness reporting using the saved native edge objects.
- Calibration versus evaluation seed split.
- Paper-aligned reproduction cells used to validate the wrapper before the
  selection study runs.
- DCDI environment contingency in case the project environment cannot host
  DCDI directly.
- DAGMA noise and intercept policy for post-hoc interventional sampling.
- MMD-unavailable tie policy.
- ER2 wording clarification, to avoid confusion between expected edges and
  expected degree.
- Explicit SID gate before the bake-off is declared complete.

The main study execution document should later cover:

- Random-prior sampling distribution.
- H3 graceful degradation statistic.
- H1 association test specification.
- H4 instability measure specification.
- Multiple-comparison families.
- Main-study thresholding choices for the chosen base model.

These items are out of scope for the current wrapper contract. They are
listed here so that nothing important is forgotten when the relevant
documents are next revised.

---

## 18. Examiner questions

The contract is designed to survive direct questioning from an examiner.
The following are hard questions an examiner might ask, written in plain
ASCII without arrow symbols:

- How do you guarantee that DAGMA and DCDI wrappers are compared fairly
  rather than reflecting wrapper differences.
- How do you confirm that the soft-prior penalty actually contributed to
  training rather than being silently ignored.
- How do you handle a run where DAGMA returns a cyclic thresholded graph.
- How do you decide between the candidates when SID is within the 10
  percent tie margin and DCDI cannot generate interventional samples.
- How do you ensure that intervention values like do X_j equals 2 are
  applied in raw SCM units even if the wrapper standardises training data.
- How do you know the orientation of the predicted adjacency matches the
  orientation expected by the evaluator metrics.
- How do you keep the threshold robustness story honest if you only saved
  the post-threshold boolean output.
- How do you defend choosing 0.3 for DAGMA and 0.5 for DCDI rather than a
  single shared threshold.
- How do you know the invalid-graph counts you report match what actually
  happened during the selection study.
- How do you avoid silently passing an in-memory Python callable across a
  subprocess boundary if DCDI ends up running in a separate environment.
