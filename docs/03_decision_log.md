23/04/2026 — Reproducible environment and packaging setup established

1. Package and Environment

Decision:

- The project is structured as an installable local Python package under `src/symbolic_priors_cd/`.
- A project-local virtual environment (`.venv`) is used to isolate dependencies from the global Python installation.
- Direct dependencies are declared in `pyproject.toml`.
- The exact working environment is frozen in `requirements-lock.txt`.
- Local structured outputs remain canonical; external experiment tracking tools are optional mirrors only.

Reason:
To ensure environment isolation, reproducibility, and a clean project-level software contract before thesis implementation begins.

---

23/04/2026 — SCM and intervention design decisions established

1. SCM representation

Decision:

- The synthetic causal model is represented as a frozen dataclass rather than as a raw tuple or a mutable stateful class.

Reason:
A raw tuple is too fragile and obscures the meaning of each component, while a mutable stateful class increases the risk of hidden state and accidental mutation of the ground-truth SCM. A frozen dataclass provides named structure, readability, and immutability, which makes the evaluator easier to inspect, test, and defend.

2. Intervention semantics

Decision:

- Interventions are non-mutating and return a new SCM or intervened sampling object rather than modifying the original SCM in place.

Reason:
The original SCM is the ground-truth reference object for the experiment and should remain stable across all evaluations. Non-mutating interventions reduce the risk of contaminating later runs and make the intervention semantics easier to reason about and debug.

3. Randomness handling

Decision:

- Randomness is external to the SCM.
- Sampling functions take a numpy random generator or an explicit seed.
- No RNG state is stored inside the SCM object.

Reason:
Keeping randomness external avoids hidden mutable state inside the SCM and makes reproducibility more transparent. It also makes testing easier, because stochastic behaviour is controlled explicitly by the calling code rather than implicitly by the object.

4. Graph density parameterisation

Decision:

- Graph generation uses the parameter `expected_edges`, not `expected_degree`.

Reason:
`expected_degree` is easy to misread and could lead to implementing the wrong graph density while believing the code is faithful to the intended benchmark. `expected_edges` is more explicit, less ambiguous, and safer for later maintenance and reproduction.

5. ER2 convention

Decision:

- The ER2 convention is defined as `expected_edges = 2 * n_nodes`.
- For the 10-node selection-study cell, this means `expected_edges = 20`.

Reason:
This makes the benchmark definition concrete and avoids interpretive drift. It ensures that the graph density used in implementation matches the intended sparse synthetic setting rather than an accidental alternative reading.

6. DAG generation procedure

Decision:

- DAG generation follows an acyclic ordered-edge procedure.
- A random topological ordering is sampled first.
- Each admissible forward edge is then included independently with probability `p = expected_edges / (n_nodes * (n_nodes - 1) / 2)`.

Reason:
This construction guarantees acyclicity by design while preserving the intended Erdős–Rényi-style sparsity logic. It is simpler and safer than generating a cyclic graph first and trying to repair it afterwards.

7. Edge-weight sampling

Decision:

- Linear Gaussian SCM edge weights are sampled from `Uniform(-2, -0.5)` union `Uniform(0.5, 2)`.

Reason:
Sampling weights away from zero avoids nominally present edges that behave as if they were absent. This produces cleaner synthetic benchmarks and makes structure recovery and intervention behaviour more meaningful.

8. Noise model

Decision:

- Noise terms are sampled as independent Gaussian noise with variance 1 for each node.

Reason:
This gives a simple and standard baseline noise model that is easy to analyse, reproduce, and defend. It keeps the initial synthetic regime controlled rather than introducing unnecessary complexity at the evaluator stage.

Overall rationale:
These decisions make the synthetic SCM layer explicit, reproducible, and safe against hidden state or accidental mutation, while keeping the initial benchmark aligned with the intended thesis setting.

---

23/04/2026 — Metric and evaluator testing decisions established

1. SHD convention

Decision:

- SHD is implemented with a configurable reversal convention.
- The default is set so that a reversed edge counts as 2 edits.

Reason:
Different SHD conventions exist, so configurability avoids hard-coding one interpretation silently. The default is set to the more literature-aligned convention for comparability, while still allowing explicit alternative analysis later if needed.

2. MMD definition

Decision:

- MMD is implemented as unbiased RBF MMD with median-heuristic bandwidth.

Reason:
The unbiased estimator is more appropriate for sanity checks because it is designed to sit near zero when both samples come from the same distribution. The RBF kernel with median-heuristic bandwidth is a practical and widely used default that is easy to justify before more detailed sensitivity analysis.

3. MMD reporting behaviour

Decision:

- The raw unbiased MMD value is computed internally and may be slightly negative because of finite-sample estimation noise.
- Clipping to zero is allowed only at the reporting layer, not in the core computation.

Reason:
Keeping the raw value preserves mathematical honesty and avoids hiding diagnostic information inside the implementation. Clipping is only a presentation convenience, not part of the metric definition itself.

4. SID implementation status

Decision:

- SID is given a stable interface now.
- The implementation is deliberately deferred until a candidate implementation is explicitly verified against a small hand-checkable example.

Reason:
SID is important but sufficiently non-trivial that a rushed or poorly validated implementation would create scientific risk. Defining the interface now preserves architectural clarity, while deferring the implementation avoids committing to an unverified method.

5. SID placeholder behaviour

Decision:

- SID should raise `NotImplementedError` until verification of the implementation is complete.

Reason:
Returning a placeholder number would be misleading and dangerous. Raising `NotImplementedError` makes the incompleteness explicit and prevents accidental reliance on an invalid metric.

6. Testing style

Decision:

- Tests use plain deterministic pytest tests rather than property-based testing.

Reason:
For the current evaluator invariants, deterministic tests are sufficient, easier to read, and easier to defend. Property-based testing would add complexity without providing enough additional value at this stage of the project.

7. Minimum evaluator test set

Decision:

- Initial evaluator tests must verify at least:
  - generated graphs are acyclic;
  - observational sample shapes are correct;
  - interventions change the intended variable behaviour;
  - `SHD(G, G) = 0`;
  - MMD between two samples from the same distribution is near zero.

Reason:
These tests check the minimum scientific invariants needed before any model comparison can be trusted. They ensure that the evaluator is functioning as a measurement instrument rather than merely that the code runs without crashing.

Overall rationale:
These decisions make the evaluator scientifically trustworthy before model comparison begins, reduce ambiguity in metric conventions, and ensure that later reproduction, debugging, and thesis defence rely on explicit and defensible implementation choices.

---

23/04/2026 — Intervention interface decisions established

1. Intervene return type

Decision:

- The intervention API returns a sampler object rather than a post-do SCM or a free function output only.
- The intended pattern is that `intervene(scm, intervention)` returns a reusable interventional sampler tied to the original immutable SCM and the specified intervention.

Reason:
This keeps the SCM itself clean and immutable while still giving a reusable handle for repeated interventional sampling during evaluation. It avoids forcing intervention-specific structure into the base SCM representation and is more reusable than a free-function-only design.

2. Intervention shape

Decision:

- Interventions are single-node only at this stage.
- The interface should still use an intervention object rather than raw `(target, value)` pairs.

Reason:
The current frozen study design only requires single-node hard interventions at `-2` and `+2`. Supporting multi-node interventions now would add unnecessary complexity without scientific benefit, while using an intervention object preserves a clean path for later extension if the research scope changes.

3. Provenance storage

Decision:

- The SCM carries a minimal `GenerationSpec`.
- The provenance record is generation-level only.

Reason:
A small self-describing provenance object is justified in a reproducibility-heavy project because it makes each generated SCM easier to inspect and trace. However, the scope must remain narrow: it should include only generation-level information such as node count, expected edges, edge probability, mechanism family, weight range, and generator seed, and should exclude run metadata, results, or experiment-tracking outputs.

4. Noise storage

Decision:

- Noise is stored as a single scalar `noise_scale` rather than as a per-node array.

Reason:
The current experimental commitment is homoscedastic unit-variance noise, so a scalar representation matches the actual scientific design more honestly and avoids implying unsupported heteroscedastic capability. If heteroscedastic noise is introduced later, it should be added through an explicit refactor and recorded as a deliberate design change rather than pre-built into the interface now.

Overall rationale:
These interface decisions were refined through multiple critique rounds and were chosen to match the frozen scientific commitments of Documents 01 and 02 rather than defaulting to maximum engineering flexibility. The aim is to keep the evaluator explicit, reproducible, and aligned with the actual thesis protocol.

---

04/05/2026 — Evaluator data-layer validation and sampling refinements established

1. Constructor invariants

Decision:

- `LinearGaussianSCM` now validates that every edge in `adjacency` respects the supplied `topological_order`.
- `LinearGaussianSCM` now validates that the attached `GenerationSpec` is coherent with the SCM itself, specifically:
  - `spec.n_nodes == n_nodes`
  - `spec.noise_scale == noise_scale`

Reason:
A permutation-only check on `topological_order` is not sufficient if the adjacency matrix can still violate that order. Likewise, provenance-bearing SCMs should not allow silent mismatch between the object and its recorded generation metadata. These checks strengthen the evaluator as a scientific object rather than merely as executable code.

2. Shared sampling-kernel boundary

Decision:

- The shared ancestral-sampling kernel and RNG coercion helper were moved into an internal module, `src/symbolic_priors_cd/data/_sampling.py`.
- Both observational and interventional sampling now import the shared internal helper from that module.

Reason:
This removes the previous private-import smell where one module imported a private helper from another module’s implementation file. The new boundary is cleaner and preserves the intended design property that observational and interventional sampling share one kernel by construction.

3. Evaluator test refinement

Decision:

- A non-unit-weight structural-equation test was added to the observational sampling tests.
- The previous exact-equality leaf-intervention test was retained but explicitly reclassified as an implementation-regression test for the shared-kernel design rather than as a general causal invariant.

Reason:
The new non-unit-weight test closes a real coverage gap by detecting coefficient-indexing or transposition mistakes that unit-weight chain tests may not expose. The retained exact-equality test is still useful, but its purpose is to protect the current shared-kernel implementation contract, not to claim a universal property of interventions.

4. Sampling documentation and hygiene

Decision:

- The public docstrings of `sample_observational` and `InterventionalSampler.sample` now explicitly distinguish the semantics of passing an integer seed versus a live `numpy.random.Generator`.
- Unused imports and unused logger declarations were removed from the affected files.
- Imprecise causal wording in the intervention tests was tightened.

Reason:
The distinction between integer seeds and stateful generators affects reproducibility and must be explicit at the public interface level. Minor hygiene and wording fixes reduce ambiguity and make the evaluator easier to inspect, maintain, and defend.

5. Validation outcome

Decision:

- After these refinements, the evaluator data-layer test suite passed in full.

Reason:
The data layer is now sufficiently hardened to serve as the foundation for the next implementation stage, namely the metrics layer and the ground-truth compatibility checks required before model comparison.

---

06/05/2026 — Metrics-layer implementation decisions established

1. Structural metric implementation

Decision:

- `SHD` is implemented in `src/symbolic_priors_cd/metrics/structural.py`.
- The public signature is `shd(predicted, true, reversal_cost=2)`.
- Inputs are validated as:
  - strict boolean adjacency matrices;
  - square 2D shape;
  - no diagonal self-loops;
  - matching shape between `predicted` and `true`.
- Full DAG-validity checks such as acyclicity and bidirected-edge detection are not performed inside the primitive metric.

Reason:
The structural metric should remain a small, explicit primitive over already-prepared adjacency matrices. Strict local validation prevents obvious misuse, while leaving full graph-construction responsibility to upstream code keeps the metric simple, readable, and reusable.

2. Shared graph-validation boundary

Decision:

- Adjacency-matrix validation was extracted into `src/symbolic_priors_cd/metrics/_graph_validation.py`.
- Both `structural.py` and `interventional.py` import the shared internal helper from this module.

Reason:
This removes duplication and avoids the private-import smell of importing underscore-prefixed helpers from another public module’s implementation file. It also keeps graph-level validation logic consistent across SHD and SID.

3. Internal metric argument-order convention

Decision:

- Metric functions currently use the internal argument-order convention `(predicted, true)`.
- This convention is used by both `shd(predicted, true, ...)` and `sid_score(predicted_dag, true_dag)`.

Reason:
Keeping argument order consistent across the currently implemented metrics reduces the risk of silent misuse in the evaluation pipeline. If external libraries later use a different convention, argument swapping should happen at the integration boundary rather than through ad hoc inconsistency inside the project API.

4. Interventional MMD implementation

Decision:

- `MMD` is implemented in `src/symbolic_priors_cd/metrics/interventional.py` as `mmd_rbf_unbiased(x, y, bandwidth=None)`.
- The kernel convention is:
  - `k(a, b) = exp(-||a - b||^2 / bandwidth)`.
- The median heuristic is computed from pairwise squared distances on the concatenated samples, using only the upper triangle and excluding self-distances.
- The unbiased estimator excludes diagonal self-pairs in the within-sample terms and requires at least 2 samples per group.
- The primitive returns the raw unbiased MMD value and does not clip negative results.
- Floating-point artefacts in squared-distance computation are clamped to zero before kernel evaluation.

Reason:
These choices make the implemented kernel, bandwidth heuristic, and estimator internally consistent and mathematically honest. Negative unbiased values are a valid finite-sample phenomenon and should not be hidden inside the core metric. The small numerical safeguard on squared distances improves robustness without changing the intended metric definition.

5. Interventional MMD sweep behaviour

Decision:

- `mmd_sensitivity_sweep(x, y, bandwidth_multipliers=(0.5, 1.0, 2.0))` computes the median bandwidth once and scales it by the supplied multipliers.
- The multiplier tuple must be non-empty and strictly positive.

Reason:
The sensitivity sweep is intended to test robustness to bandwidth scaling around a single baseline heuristic. Recomputing the heuristic separately for each sweep point would undermine the meaning of the sweep.

6. Degenerate-bandwidth handling

Decision:

- If the median heuristic yields a non-positive bandwidth, both `mmd_rbf_unbiased` and `mmd_sensitivity_sweep` raise `ValueError`.

Reason:
Degenerate samples should fail loudly and explicitly rather than silently propagating invalid bandwidth values through the evaluator.

7. SID placement and deferred implementation contract

Decision:

- `sid_score(predicted_dag, true_dag)` is implemented as a public stub in `src/symbolic_priors_cd/metrics/interventional.py`.
- `sid_score` validates graph inputs and then raises:
  - `NotImplementedError("SID implementation is deferred pending explicit verification.")`

Reason:
Within this project, SID functions as an interventional adequacy metric and belongs with the interventional evaluation layer even though its inputs are adjacency matrices. The explicit deferred-stub contract preserves the public interface while honestly recording that verified computation is not yet available.

8. SID deferred test scaffolding

Decision:

- A skipped SID test case was added to reserve a concrete small-graph verification case for later implementation.
- The currently written numerical expectation is provisional and must be verified against the adopted reference implementation before the test is unskipped.

Reason:
This preserves a clear place for later SID regression testing without pretending that the numerical result is already confirmed.

9. Metrics-layer validation outcome

Decision:

- After SHD, MMD, shared validation extraction, and the SID stub were implemented, the structural and interventional metrics test suites passed in full, with the SID deferred test intentionally skipped.

Reason:
The evaluator metrics layer is now sufficiently stable to support the next implementation stage, namely the ground-truth compatibility and sanity-check layer required before model-wrapper evaluation begins.

---

06/05/2026 — Ground-truth compatibility gate decisions established

1. Sanity-check layer scope

Decision:

- A dedicated sanity-check layer is implemented in `src/symbolic_priors_cd/metrics/sanity_checks.py`.
- This layer does not define new primitive metrics.
- Instead, it orchestrates the existing evaluator components into a pre-study compatibility gate.

Reason:
The evaluator already contains the required primitives in the data and metrics layers. The missing piece was a thin orchestration layer that can certify whether those primitives behave correctly together before any model comparison begins.

2. Compatibility report schema

Decision:

- `run_ground_truth_compatibility_checks(...)` returns a structured typed report rather than a boolean.
- The report is represented by a `CompatibilityReport` `TypedDict` with keys:
  - `sid_self_zero_status`
  - `sid_self_zero_value`
  - `mmd_same_intervention`
  - `mmd_same_observational`
  - `do_clamping_max_deviation`

Reason:
A boolean pass/fail result would hide which specific compatibility check failed and by how much. A typed structured report is more useful for debugging, logging, and later thesis write-up, while remaining simple enough for the current project stage.

3. SID self-zero status semantics

Decision:

- `check_sid_self_zero(true_dag)` calls `sid_score(true_dag, true_dag)`.
- If `sid_score` raises `NotImplementedError`, the check returns `None`.
- `run_ground_truth_compatibility_checks(...)` maps SID outcomes as follows:
  - `None` → `"deferred"`
  - `0` → `"passed"`
  - any non-zero integer → `"failed"`

Reason:
This makes the current deferred SID state explicit without pretending the metric is implemented. It also freezes the future semantics once SID becomes available, so the compatibility gate does not need to change shape later.

4. Deferred versus failed SID gate behaviour

Decision:

- In `assert_ground_truth_compatibility(...)`, a SID status of `"failed"` is always a hard error.
- A SID status of `"deferred"` fails only when `require_sid=True`.
- The default is `require_sid=False`.

Reason:
The evaluator should be honest about the current deferred SID state, but the project should not be blocked from completing the evaluator foundation while verified SID integration is still pending. At the same time, once SID is implemented, a failing self-zero check must always stop the gate regardless of the deferred-policy setting.

5. MMD gate semantics

Decision:

- `mmd_tolerance=0.01` is the default tolerance in `assert_ground_truth_compatibility(...)`.
- This tolerance is applied to both:
  - `mmd_same_intervention`
  - `mmd_same_observational`
- MMD checks use `abs(value) < mmd_tolerance`.

Reason:
The unbiased MMD estimator can return slightly negative values due to finite-sample noise, so the gate must compare absolute magnitude rather than raw sign. Applying a single tolerance to both same-distribution checks keeps the gate interface simple and conservative at this stage.

6. Clamping gate semantics

Decision:

- `clamp_tolerance=1e-12` is the default tolerance in `assert_ground_truth_compatibility(...)`.
- Clamping is checked via:
  - `do_clamping_max_deviation < clamp_tolerance`

Reason:
The do-operator should clamp exactly. Any deviation is expected to be floating-point-level noise rather than statistical variation, so a very tight tolerance is appropriate.

7. Sample-size defaults inside sanity checks

Decision:

- The MMD sanity checks default to `n_samples=1000`.
- The clamping check defaults to `n_samples=100`.
- `run_ground_truth_compatibility_checks(...)` forwards the caller-provided `n_samples` to the MMD checks only; the clamping check keeps its own default of 100.

Reason:
The MMD checks estimate a distributional quantity and therefore benefit from a larger sample size to reduce variance. The clamping check is deterministic and only needs enough samples to verify that the target column is always fixed to the intervention value, so a smaller sample count is sufficient and cheaper.

8. Gate failure reporting

Decision:

- `assert_ground_truth_compatibility(...)` aggregates all detected failures before raising.
- On failure, it raises `AssertionError` containing both:
  - a list of failure reasons
  - the full structured compatibility report

Reason:
The gate should function as a debugging instrument, not just a stop signal. Aggregating multiple failure reasons makes it easier to diagnose whether the problem lies in SID status handling, MMD behaviour, clamping behaviour, or several issues at once.

9. Validation outcome

Decision:

- After implementing the sanity-check layer, the full project test suite passed with the SID pre-registered scaffold still intentionally skipped.

Reason:
This confirms that the evaluator foundation now behaves coherently across the data layer, metrics primitives, and gate layer. The project can therefore move from evaluator construction into model-wrapper design, with verified SID integration remaining an explicit near-term dependency.

---


11/05/2026 - Wrapper contract and adjacency-orientation convention established

1. Wrapper API contract

Decision:

- A wrapper API contract has been added in `docs/04_wrapper_api_contract.md`.
- The contract defines the scientific capabilities and invariants required of future DAGMA and DCDI wrappers before implementation.
- The contract does not freeze concrete Python method names or class structure.
- Wrapper implementation must satisfy the contract before any wrapper is treated as ready for selection-study use.

Reason:

The wrapper layer is not only an evaluator adapter. It must also preserve access to native continuous edge objects, support future soft-prior loss injection, allow threshold-robustness checks, and support model-generated interventional sampling where feasible. Freezing these requirements before implementation reduces the risk of building wrappers that produce evaluator-compatible boolean adjacency matrices but fail later methodological requirements.

Consequence:

DAGMA and DCDI wrapper implementation must follow `docs/04_wrapper_api_contract.md`. Any deviation from the contract must be recorded explicitly befoAgreed on all points. ChatGPT's refinements are improvements, not arguments for the sake of arguing. Going through them:

**Step 0 admin update before probes**: ChatGPT is right and I had this backwards. The decision log is meant as a contemporaneous record, not a retrospective summary. Recording "04b inspection complete with these findings" now is a separable fact from "04c probes complete with these findings", and the two entries should reflect when they actually happened. Bundling them after probes would compress two distinct moments into one. Concede.

**Plan-before-execute for probes**: Also right. Even small probes touch the environment (failed imports, transitive dependency chains, GPU availability). The plan-then-implement discipline isn't bureaucracy here — it's the same discipline that caught the orientation question and the DAGMA hyperparameter divergence. Probe plans should be reviewed; the cost of one extra round-trip is small relative to the cost of a probe that silently changes the environment.

**Precision on DAGMA sampling policy**: This is the strongest refinement and worth lingering on. My phrasing — "estimate per-node residual variance from `X - X @ W` after fit" — was loose in exactly the way ChatGPT identifies. The subtlety is that `W` after fit means *something*, but it doesn't disambiguate between:

- residuals using the continuous `W` (what the model actually fitted)
- residuals using the thresholded `W` (what we'll actually sample from)

These give different variance estimates because below-threshold spurious entries either contribute to the prediction (continuous case) or contribute to the residual (thresholded case). Since we sample from the thresholded structure, the noise budget the sampler uses should account for whatever the thresholded structure *doesn't* predict — which means residuals computed against the thresholded `W`. That's the consistent choice. The amendment should specify this explicitly, plus the frame discipline (sample in centred space, add training means back before MMD compares against raw ground-truth, transform intervention values from raw SCM units into centred units inside the wrapper).

**Hard-constraint baseline cannot be post-hoc mask**: ChatGPT catches a real scientific oversimplification in the inspection report. The line "the hard-constraint baseline can still be implemented by both wrappers via a forced post-threshold mask if needed" is wrong, and accepting it now would store a Phase 6 problem. A training-time hard constraint prevents the model from placing credit on forbidden edges *and* lets the model redistribute that credit to other edges; a post-hoc mask just zeroes out forbidden edges after the model has been allowed to place credit anywhere. They are not the same procedure and they will not give the same results.

For DAGMA, hard constraints have a natural training-time implementation: the existing `mask_exc` machinery is already applied inside the gradient step, so adding forbidden edges as training-time constraints is exactly what `exclude_edges` does. For DCDI, hard constraints require clamping the relevant `log_alpha` entries to large-negative values during training (so `sigmoid(log_alpha) ≈ 0` throughout), or masking them out of the loss. Either is invasive but it is the *only* honest way to implement hard constraints in DCDI. This should be flagged in the Doc 02 amendment (or held over for Doc 03) as a Phase 6 implementation requirement, not papered over with post-hoc masking.

**One small thing I'd add to the probe plan**: when ChatGPT's prompt asks for "preserving pre-threshold W before project-level thresholding", the cleanest mechanism is probably to pass `w_threshold=0` (or whatever DAGMA accepts as "no thresholding") to DAGMA's `fit()`, so `W_est` survives unmodified, and then the wrapper applies the 0.3 threshold itself. The inspection notes that `self.W_est[np.abs(self.W_est) < w_threshold] = 0` happens at line 354 — if `w_threshold = 0`, this is a no-op. Worth confirming via probe that this is supported and doesn't cascade into other parts of `fit()`.

**On the overall workflow**: yes, the cycle is `admin update → probe plan → probe run → Doc 02 amendment → wrapper plan → wrapper implementation`. That is several round trips but the discipline is producing scientifically defensible work. The cost is paying for itself in caught issues — the orientation audit, the DAGMA hyperparameter divergence, the hard-constraint subtlety. Don't compress the loop just because it feels slow.

Execute ChatGPT's path. Step 0 (admin update) is small enough to do manually; Step 1 (probe plan prompt) is ready to send as-is. When the probe plan comes back I'll review it before run approval.re implementation proceeds.

2. Adjacency-orientation convention

Decision:

- The project canonical adjacency-orientation convention is row-source / column-destination.
- That is, `adjacency[i, j] = True` means a directed edge from node `i` to node `j`.
- Equivalently, column `j` lists the parents of node `j`.
- Weights follow the same convention: `weights[i, j]` is the coefficient of `X_i` in the structural equation for `X_j`.

Reason:

The orientation audit in `docs/04a_orientation_audit.md` found that the SCM generator, shared sampling kernel, SHD implementation, graph-validation boundary, and adjacency-related tests consistently follow this convention. Recording the convention in the decision log prevents silent transposition errors when adapting DAGMA and DCDI outputs to the evaluator.

Consequence:

All DAGMA and DCDI wrappers must translate external model outputs into the row-source / column-destination convention before passing adjacency matrices to evaluator metrics or downstream sampling logic. Orientation must be tested using a mock native-edge object before wrapper completion.

11/05/2026 - DAGMA and DCDI source inspection completed

1. Source inspection report

Decision:

- A read-only source inspection report has been added in `docs/04b_source_inspection.md`.
- The report inspected DAGMA at commit `088616885d71b56c0573cd4902c1fcbac02e649f`.
- The report inspected DCDI at commit `594d328eae7795785e0d1a1138945e28a4fec037`.
- Both DAGMA and DCDI use the project row-source / column-destination adjacency convention, so no orientation transformation is required at the wrapper boundary.
- DAGMA thresholding applies to `abs(W)`.
- DCDI thresholding applies to `P = sigmoid(log_alpha)`.
- DAGMA library defaults differ from the selection-study protocol and must be overridden explicitly.
- DCDI default hyperparameters largely match the selection-study protocol.
- Neither library provides a built-in sampler for arbitrary single-node hard interventions, so wrapper-side ancestral sampling is required for MMD.
- DAGMA soft-prior integration requires a bounded but invasive subclass or override of the hand-coded optimisation step.
- DCDI soft-prior integration is autograd-friendly, but the wrapper should avoid importing `dcdi.train` directly because of the `cdt` and R import chain.

Reason:

The source inspection converts several wrapper-design assumptions into code-level facts. Recording these findings prevents wrapper implementation from relying on undocumented library defaults, paper memory, or unstated source behaviour.

Consequence:

Before wrapper implementation begins, runtime probes must resolve the execution-dependent items left open by the source inspection. These include DCDI import feasibility, DCDI post-hoc sampling feasibility, DCDI determinism, DAGMA pre-threshold `W` preservation, and DAGMA mean-centring behaviour. Doc 02 must be amended before the selection study runs.


11/05/2026 - Phase 2 wrapper pre-implementation verification completed

1. Wrapper contract, source inspection, and runtime probes

Decision:

- The wrapper API contract is recorded in `docs/04_wrapper_api_contract.md`.
- The project adjacency-orientation audit is recorded in `docs/04a_orientation_audit.md`.
- The DAGMA and DCDI source inspection is recorded in `docs/04b_source_inspection.md`.
- The runtime probe plan and results are recorded in `docs/04c_runtime_probe_plan.md` and `docs/04c_runtime_probe_results.md`.
- DAGMA was inspected at commit `088616885d71b56c0573cd4902c1fcbac02e649f`.
- DCDI was inspected at commit `594d328eae7795785e0d1a1138945e28a4fec037`.
- Both DAGMA and DCDI use the project row-source / column-destination adjacency convention.
- DAGMA thresholding applies to `abs(W)`.
- DCDI thresholding applies to `P = sigmoid(log_alpha)`.
- DAGMA library defaults differ from the selection-study protocol and must be overridden explicitly.
- DAGMA preserves continuous `W` when called with `w_threshold=0.0`.
- DAGMA mutates input data during mean-centring, so wrappers must pass a defensive copy of `X`.
- DCDI low-level imports avoid `dcdi.train`, `cdt`, and the R integration chain.
- DCDI exposes `log_alpha` and `get_w_adj()` as expected.
- DCDI `forward_given_params` works in eval mode with the minimal signature `(x, weights, biases)`.
- DCDI wrapper-side ancestral sampling under a single-node hard intervention is mechanically feasible.
- DCDI CPU determinism held exactly on a small controlled runtime probe.

Reason:

The Phase 2 pre-implementation verification cycle resolved wrapper feasibility risks before implementation. The project did not assume external library conventions from papers or memory. Instead, it established a wrapper contract, audited the internal orientation convention, inspected the external source code at fixed commits, and executed minimal runtime probes to resolve execution-dependent uncertainties.

Consequence:

- DAGMA wrappers must call `DagmaLinear.fit` with `w_threshold=0.0`, save the continuous `W`, and apply the project threshold externally.
- DAGMA wrappers must pass `X.copy()` into DAGMA to avoid mutating upstream data.
- DCDI wrappers must avoid importing `dcdi.train` and instead use the probed low-level model and helper imports.
- DCDI wrappers may implement post-hoc interventional sampling using `forward_given_params` and `get_distribution`, guarded by graph-validity checks.
- Doc 02 must be amended before wrapper implementation to freeze preprocessing, DAGMA hyperparameter overrides, DAGMA MMD sampling policy, DCDI import policy, threshold-robustness reporting, calibration/evaluation seed split, MMD-unavailable tie policy, ER2 wording, and the SID gate.
- The runtime probes do not prove full-scale DCDI determinism or final sampler quality after a real trained model; those remain wrapper-validation and selection-study validation requirements.


11/05/2026 - Phase 2 pre-implementation verification cycle completed

Decision:

- The wrapper contract is recorded in `docs/04_wrapper_api_contract.md`.
- The orientation audit is recorded in `docs/04a_orientation_audit.md`.
- The source inspection is recorded in `docs/04b_source_inspection.md`.
- Runtime probe planning and results are recorded in
  `docs/04c_runtime_probe_plan.md` and
  `docs/04c_runtime_probe_results.md`.
- The DCDI structural-mask probe is recorded in
  `docs/04d_dcdi_mask_probe_results.md`.
- Doc 02 has been amended to v1.3 to incorporate wrapper-facing
  selection-study protocol details supported by the audits and probes.

Reason:

The project resolved wrapper-design risks before implementation rather
than relying on paper memory or API assumptions. The verification cycle
established the project adjacency convention, checked DAGMA and DCDI source
behaviour at fixed commits, verified runtime feasibility for DAGMA continuous
W preservation and DCDI post-hoc sampling, and identified the correct DCDI
structural-masking mechanism.

Consequence:

- Wrapper implementation may now proceed from the frozen protocol in
  `docs/02_base_model_selection.md`.
- DAGMA wrappers must preserve continuous `W` using `w_threshold=0.0`,
  apply project thresholding externally, and pass `X.copy()` into DAGMA.
- DCDI wrappers must avoid `dcdi.train`, use the low-level imports verified
  by probes, and enforce sampling-time structural masks by temporarily
  setting `model.adjacency` plus saturated `log_alpha`, then restoring the
  original state.
- Wrapper implementation must still pass the contract tests in
  `docs/04_wrapper_api_contract.md`.
- The selection study cannot be declared scientifically complete until SID
  is verified and integrated.


---

12/05/2026 — DCDI wrapper training-equivalence and graph-status boundary established

1. DCDI training-loop behavioural-equivalence gate

Decision:

- The DCDI wrapper uses the inspected DCDI low-level components rather than importing `dcdi.train`.
- The wrapper-side observational training loop is treated as scientifically validated for the covered objective and update schedule after passing the behavioural-equivalence gate.
- The behavioural-equivalence tests compare the wrapper loop against a hand-replicated reference loop derived from the inspected DCDI source.
- The comparison covers:
  - early `log_alpha` trajectory checkpoints with bitwise equality;
  - mid-trajectory checkpoints with documented tolerance;
  - final `log_alpha` and `get_w_adj`;
  - loss-history checkpoints;
  - training metadata;
  - exact gamma-update and mu-update iteration indices.
- The calibration artefact for this equivalence test is recorded in `docs/04e_equivalence_calibration_results.md`.
- The calibrated schedule used `stop_crit_win = 20`, `n_iter = 400`, and recorded:
  - `gamma_update_iters = [280, 400]`
  - `mu_update_iters = [400]`

Reason:
The wrapper deliberately avoids `dcdi.train` because that entry point imports optional `cdt` and R-related dependencies that are not needed for the thesis wrapper. However, avoiding `dcdi.train` creates a scientific risk: the wrapper-side loop could silently diverge from the inspected DCDI optimisation logic. The behavioural-equivalence gate reduces that risk by verifying the wrapper loop against an independently written reference loop over the objective, minibatch schedule, validation timing, gamma/mu schedule, trajectory, and metadata.

2. Continuous native-edge preservation

Decision:

- The DCDI training loop preserves the continuous native edge objects at training exit:
  - `continuous_log_alpha_pre_threshold`
  - `continuous_w_adj_pre_threshold`
- The wrapper does not permanently saturate `log_alpha` to `+/-100` during training.
- Any future saturation used for sampling-time structural masking must be temporary and restored afterwards.
- Thresholding and downstream graph-status logic must use the preserved continuous edge probabilities rather than relying on a permanently discretised DCDI state.

Reason:
The thesis requires access to the native continuous edge representation for three downstream purposes: threshold-robustness reporting, interventional sampling, and eventual soft-prior loss-hook integration. Permanently saturating `log_alpha` would destroy the continuous information needed for those purposes and would make threshold sensitivity impossible to evaluate honestly.

3. DCDI thresholding and graph-status boundary

Decision:

- DCDI thresholding is implemented at the wrapper boundary, not inside metric primitives.
- The wrapper converts preserved continuous edge probabilities into boolean adjacency matrices using the project convention:
  - row-source / column-destination
  - `adjacency[i, j] = True` means edge `i -> j`
- The default DCDI threshold remains `0.5`, applied to the preserved continuous edge-probability matrix.
- Graph validity is classified using a priority-ordered status boundary:
  - `invalid_shape`
  - `self_loop`
  - `bidirected`
  - `cyclic`
  - `valid_dag`
- Invalid graphs are classified, not repaired.
- The sampler-status boundary maps `valid_dag` to `available` and all non-valid graph statuses to `unavailable_invalid_graph`.

Reason:
Thresholding is a model-wrapper responsibility because it converts model-native continuous outputs into evaluator-facing boolean adjacency matrices. Keeping this logic outside metric primitives protects the metric layer from model-specific assumptions. Explicit graph-status classification also enforces the project policy of no silent graph repair: cyclic, bidirected, malformed, or self-loop-containing outputs are scientifically informative failure modes and must be reported rather than hidden by post-hoc edge deletion.

4. Validation outcome and remaining scope

Decision:

- After the DCDI behavioural-equivalence gate and graph-status boundary were implemented, the full test suite passed with the intentional SID scaffold still skipped.
- At this milestone, the DCDI wrapper has validated training-loop infrastructure and threshold/graph-status utilities, but the DCDI wrapper is not yet complete.
- Remaining DCDI wrapper work includes:
  - structural-mask context manager;
  - interventional sampler;
  - raw-unit intervention roundtrip;
  - sampler-quality validation;
  - loss-hook integration;
  - diagnostics assembly;
  - full-convergence integration test;
  - final public API stabilisation.
- Verified SID integration remains deferred and is still required before base-model selection results can be treated as scientifically complete.

Reason:
This records the distinction between a validated training/thresholding foundation and a complete model-selection wrapper. The project can proceed to the next DCDI wrapper commits, but it must not overclaim: sampler behaviour, loss-hook behaviour, diagnostics, full-convergence behaviour, and verified SID are still separate validation obligations.

Overall rationale:
This milestone closes the highest-risk part of the DCDI wrapper foundation: whether the project-owned wrapper training loop faithfully matches the inspected DCDI optimisation behaviour while preserving the continuous native edge representation required by the thesis. It also establishes the wrapper-side thresholding and graph-status boundary needed for downstream SID, SHD, MMD, and invalid-output reporting. This keeps the DCDI wrapper aligned with the thesis protocol while avoiding silent dependency, thresholding, or graph-repair assumptions.


---

13/05/2026 — DCDI sampler-quality diagnostic recorded and DCDI loss-hook work paused

Decision:

- DCDI sampler mechanics remain accepted through Commit 9: structural masking, restoration, clamping, deterministic sampling, and raw-unit intervention roundtrip are validated by the normal test suite.
- Commit 10 sampler-quality validation did not pass and has been converted into diagnostic probe C-P11 rather than retained as failing pytest tests.
- The diagnostic is recorded in `docs/04f_dcdi_sampler_quality_diagnostic.md`.
- DCDI Commit 11, loss-hook injection, is paused pending base-model selection review.
- The next implementation priority is DAGMA wrapper work, with SID verification continuing in parallel.

Reason:

The diagnostic showed that DCDI-G, under observational-only linear-Gaussian training, learned a valid but structurally incomplete DAG. The learned thresholded graph missed the strongest true edge, `2 -> 0`, with true weight magnitude 1.7861. Wrapper-vs-truth MMD was `+6.275e-01`, far above the Monte Carlo floor scale. When the same fitted DCDI conditionals were sampled under the true adjacency, MMD dropped to `+5.259e-02`; when the missing strongest edge was added back, MMD dropped to `+4.228e-02`.

This localises the dominant failure to learned-structure quality rather than to sampler mechanics alone. The result strengthens the thesis motivation that observational differentiable causal discovery can fail under unseen-intervention evaluation, but it weakens DCDI-G as a candidate base model until the full base-model selection protocol is run. No acceptance threshold was weakened and no silent graph repair was introduced.

C-P12 confirmed that the C-P11 fixture is recoverable under an equal-variance-aware exhaustive Gaussian-BIC score. The true DAG ranks 1/25 with a large BIC margin, while the DCDI-learned DAG ranks 19/25. This sharpens the C-P11 interpretation: DCDI-G’s failure is not data non-identifiability, but a base-model inductive-bias / optimisation / model-mismatch issue. DCDI Commit 11 remains paused.

13/05/2026 — DAGMA wrapper plan created and approved for implementation

Decision:

- `docs/06_dagma_wrapper_implementation_plan.md` v1.1 is accepted as the DAGMA wrapper implementation plan, following the same plan-then-implement discipline used for the DCDI wrapper plan.
- DAGMA implementation may proceed from Commit 1 only, following the atomic sequence in Doc 06.
- DAGMA uses source-faithfulness against a direct `DagmaLinear.fit` call rather than DCDI-style behavioural equivalence, because the DAGMA wrapper calls the official fit path directly and does not reimplement the optimisation loop.
- DAGMA learned sampler-quality is treated as an inspection probe and report from the start, not as a normal pytest gate. Wrapper mechanics remain covered by pytest.
- The diagnostics schema will use common wrapper fields plus `model_specific_diagnostics` for model-native fields.
- DAGMA fit will not call `dagma.utils.set_random_seed`, `np.random.seed`, or `torch.manual_seed`; the fit is deterministic for fixed input and hyperparameters, and sampler randomness is handled through local `np.random.default_rng(sample_seed)`.
- DAGMA `sample_interventional` exposes `noise_policy`, with `"residual_fitted"` as the primary policy and `"unit_variance"` as the Doc 02 sensitivity path.
- `h_final <= 1e-5` is used as the provisional DAGMA wrapper diagnostic threshold for `training_status`. This is diagnostic only and never triggers graph repair.
- Residual sigma estimation uses no silent variance floor. If any sigma is non-finite or non-positive, sampler availability is marked as unresolved and no MMD sample is produced.
- DAGMA loss-hook implementation remains deferred. DAGMA hard-constraint use through `exclude_edges` / `include_edges` is also deferred to a separately documented future baseline implementation.
- DCDI Commit 11 remains paused.
- Verified SID remains a parallel blocker before selection-study conclusions can be considered scientifically complete.

Reason:

The DAGMA wrapper plan preserves the frozen Doc 02 policy: `w_threshold=0.0` inside DAGMA, continuous `W` preservation, external `abs(W) >= 0.3` thresholding, residual-fitted DAGMA noise as primary, unit-variance sensitivity, no silent graph repair, and raw-unit intervention semantics. The plan also incorporates the C-P11 lesson that learned sampler-quality is a base-model diagnostic rather than a wrapper-mechanics invariant.

Consequence:

DAGMA wrapper implementation may begin from Commit 1 after this decision-log entry is committed. Any later deviation from Doc 06 must be recorded explicitly before implementation continues.


## 13/05/2026 — DAGMA fit path and continuous-W boundary established

### Decision

DAGMA wrapper Commits 2 and 3 are accepted as the implementation of the prior-free DAGMA fit path and native continuous-edge boundary.

Commit 2 establishes the DAGMA fit path:

- `DAGMAWrapper.fit` accepts observational training data already in model frame.
- The caller's input array is protected by passing a defensive copy into `DagmaLinear.fit`, because DAGMA mean-centres its input in place.
- All DAGMA fit hyperparameters are passed explicitly from `DAGMAConfig`, including `w_threshold_internal = 0.0`.
- `exclude_edges` and `include_edges` are passed as `None`; the prior-free DAGMA wrapper does not use DAGMA's hard-constraint API.
- The fit path does not call `np.random.seed`, `torch.manual_seed`, or `dagma.utils.set_random_seed`.
- `h_final` and `score_final` are captured from the fitted DAGMA model.
- Failed fits propagate the original exception and leave `_fitted = False`.

Commit 3 establishes the canonical continuous-edge boundary:

- The post-fit continuous DAGMA matrix is copied into `_continuous_w_pre_threshold`.
- `native_edge_continuous()` returns a defensive copy of `_continuous_w_pre_threshold`.
- Signed weights are preserved.
- Sub-threshold nonzero weights are preserved.
- No thresholding, transposition, sign modification, or graph repair occurs at this boundary.

### Reason

The DAGMA wrapper must preserve the native continuous weighted adjacency before any project-level thresholding occurs. This is load-bearing for later threshold robustness, source-faithfulness checks, graph-status validation, residual-noise estimation, and interventional sampling.

The fit path is intentionally narrow: it calls the inspected DAGMA implementation directly, with explicit configuration values and without global RNG mutation or hard-constraint mechanisms. This keeps the wrapper faithful to the prior-free selection-study setup and avoids contaminating later evaluation.

### Consequence

DAGMA now has a functional prior-free fit path and a canonical continuous-edge accessor. The next implementation step is the DAGMA source-faithfulness gate, which must verify that the wrapper's fitted continuous `W` matches a direct `DagmaLinear.fit` call under the same input and hyperparameters.

No selection-study conclusion can be drawn from this milestone. Thresholding, graph-status validation, residual-noise estimation, interventional sampling, sampler-quality diagnostics, and verified SID integration remain outstanding.

## 13/05/2026 — DAGMA thresholding and graph-status boundary established

### Decision

DAGMA wrapper Commit 5 is accepted as the implementation of wrapper-side thresholding and graph-status classification.

- DAGMA thresholding is external to the DAGMA library and uses `abs(W_continuous) >= threshold`.
- The default DAGMA project threshold is `0.3`.
- `thresholded_adjacency()` returns a boolean adjacency in row-source / column-destination convention.
- Thresholding does not mutate the preserved continuous `W`.
- Invalid graph patterns are reported, not repaired.
- Self-loops, bidirected pairs, and directed cycles remain present in the returned adjacency and are classified through `graph_status`.
- DAGMA stores graph-status and sampler-status state after fit.
- The shared graph-status helpers now live in `wrappers/_graph_status.py`.
- DCDI imports and re-exports the shared graph-status helpers, while keeping its DCDI-specific thresholding helper unchanged.

### Reason

The wrapper must convert DAGMA's continuous signed `W` into evaluator-compatible boolean adjacencies without hiding invalid structures. This boundary is load-bearing for later residual-noise estimation, interventional sampling, sampler-quality diagnostics, and model-selection reporting.

The shared helper prevents DAGMA and DCDI from drifting in their interpretation of graph validity.

### Consequence

DAGMA now exposes thresholded adjacency and graph-status state, but it still does not estimate residual noise or draw interventional samples. The next implementation step is residual-noise estimation for valid thresholded DAGs. If the thresholded graph is invalid, the sampler remains unavailable.

## 14/05/2026 — DAGMA residual-noise and sampler boundary established

### Decision

DAGMA wrapper Commits 6, 7, and 8 are accepted as the implementation of the DAGMA sampler mechanics boundary.

Commit 6 establishes residual-fitted noise estimation:

- The wrapper stores a model-frame copy of the training data for residual estimation.
- The sampling weight matrix is computed as `W_sample = W_continuous * A_thresh`.
- Residuals are computed as `R = X_model_frame - X_model_frame @ W_sample`.
- The residual-fitted sigma vector is computed as `std(R, axis=0, ddof=0)`.
- Residual sigma estimation is performed only when the thresholded graph is a valid DAG.
- Invalid graphs leave residual sampling quantities unavailable.
- Non-finite or non-positive residual sigmas make the primary residual-fitted sampler unavailable.
- No variance floor or sigma clamping is applied.

Commit 7 establishes model-frame ancestral sampling:

- A shared deterministic topological-order helper is used by both DAGMA and DCDI.
- The model-frame sampler traverses nodes in topological order.
- The intervention target is clamped exactly in model-frame units.
- Non-target nodes are sampled using the row-source / column-destination convention: parents of node `j` are `A_thresh[:, j]`, with weights `W_sample[parents, j]`.
- The sampler uses `np.random.default_rng(sample_seed)` and does not mutate global NumPy RNG state.
- Invalid graphs, invalid sigma vectors, invalid targets, and invalid sample counts are rejected.

Commit 8 establishes raw-unit public sampling:

- `DAGMAWrapper.sample_interventional()` now accepts raw-unit intervention values.
- The intervention value is transformed to model frame using the fitted preprocessor.
- Model-frame samples are generated by the DAGMA linear-Gaussian sampler.
- Returned samples are inverse-transformed back to raw SCM units.
- The target column is clamped to the requested raw intervention value under both centred-only and standardised preprocessing.
- The preprocessor is not refitted or mutated during sampling.
- The primary `residual_fitted` policy uses the stored residual sigma vector.
- The `unit_variance` sensitivity policy uses the same sampling weight matrix but replaces the sigma vector with ones.

### Reason

Interventional MMD comparisons require generated samples in the same raw SCM units as the ground-truth interventional samples. The sampler therefore cannot stop at model-frame samples. It must correctly transform intervention values into model frame and inverse-transform generated samples back to raw units.

The residual-fitted policy is the primary DAGMA sampling policy. The unit-variance policy is retained as a sensitivity check because it isolates the effect of the learned graph and weights from the residual-noise estimate.

### Consequence

DAGMA now has a complete mechanical sampling path:

`fit -> continuous W -> thresholded adjacency -> graph status -> residual sigma -> model-frame sampler -> raw-unit sample_interventional`.

The next step is the DAGMA sampler-quality diagnostic. This diagnostic must test whether the learned DAGMA structure and sampler produce credible interventional distributions on the frozen diagnostic fixture. Passing these mechanics tests does not by itself prove that DAGMA learns the correct structure or achieves good interventional MMD.

One naming debt is recorded: `_w_sample_residual_fitted` is policy-independent and is also used by the `unit_variance` sensitivity path. A later tidy-up may rename it to a policy-neutral name such as `_w_sample`, but no behavioural change is implied.


## 14/05/2026 — DAGMA sampler-quality diagnostic recorded

### Decision

C-P13 is accepted as the DAGMA sampler-quality diagnostic artefact on the frozen C-P11 fixture.

The diagnostic records that:

- DAGMA recovered the true adjacency exactly on this fixture: `SHD = 0`.
- The learned thresholded graph was a valid DAG: `graph_status = "valid_dag"`.
- The DAGMA sampler was available under the primary residual-fitted policy: `sampler_status = "available"`.
- The residual sigma vector was close to unit variance, consistent with the fixture's unit-variance Gaussian SCM.
- The literal primary MMD inequality remained non-informative because the Monte Carlo floor MMD was negative.
- DAGMA passed the wrong-structure fail-safe: deleting the dominant downstream true edge `2 -> 0` produced much worse MMD, with `wrong / correct = 91.334`.
- Compared with C-P11, DAGMA's much lower wrapper-vs-truth MMD on this same fixture is primarily evidence of better structure recovery on this fixture, not evidence that DAGMA's sampler mechanics are intrinsically superior to DCDI's.
- Diagnostic B1/B2 were not applicable because DAGMA did not miss any true edges on this fixture.

### Reason

C-P13 was created to test the DAGMA wrapper and sampler under the same diagnostic microscope that exposed DCDI-G's C-P11 failure. Replicating the C-P11 fixture and comparison protocol controls the major nuisance variables: SCM seed, training data, intervention, preprocessing mode, MMD estimator, batch size, seed layout, and aggregation rule.

The result reduces the risk that the DAGMA wrapper/sampler path is mechanically or empirically untrustworthy. It also provides a controlled same-fixture contrast with DCDI-G, whose C-P11 failure was localised to learned-structure quality.

However, C-P13 is fixture-specific evidence only. It does not select DAGMA as the base model, does not replace the Doc 02 selection study, and does not remove the verified-SID blocker.

### Consequence

DAGMA wrapper work may proceed to diagnostics assembly and phase readout.

The next wrapper-side step is to implement structured DAGMA diagnostics through `get_diagnostics()`, so future selection-study runs can log the fitted configuration, continuous `W`, thresholded adjacency, graph status, sampler status, residual sigma vector, sampling weights, DAGMA source path, and model-specific fit diagnostics.

A `phase_2c_dagma_readout.md` should be drafted after diagnostics assembly and public API stabilisation, summarising DAGMA wrapper mechanics, C-P13 results, remaining limitations, and the fact that base-model selection still depends on the full multi-seed study and verified SID.

DCDI loss-hook work remains paused. Nothing in C-P13 changes that decision.


---

15/05/2026 -- gadjid==0.1.0 adopted as the project SID backend dependency

Decision:

- `gadjid==0.1.0` is added as a pinned runtime dependency in `pyproject.toml`.
- The resolved entry `gadjid==0.1.0` is recorded in `requirements-lock.txt`.
- The empirical basis for this adoption is `docs/04i_gadjid_sid_backend_spike.md`,
  which records: successful install from a prebuilt abi3 wheel on Python 3.12 / win_amd64
  with no Rust toolchain; numpy-only runtime dependency; the API signature
  `gadjid.sid(g_true, g_guess, edge_direction)` returning `(normalised_distance, mistake_count)`;
  upstream R-SID cross-validation of `parent_aid` on 100-node DAG inputs; and locally
  verified `gadjid.sid == gadjid.parent_aid` identity on 20 random DAG pairs (0/20 mismatches).
- The project-facing API is unchanged: `sid_score(predicted_dag, true_dag) -> int`.
- The internal backend call will be
  `gadjid.sid(true_dag.astype(np.int8), predicted_dag.astype(np.int8), edge_direction="from row to column")[1]`.
  Argument order is flipped exactly once at the wrapper boundary; `edge_direction` is pinned
  and is never exposed in the public API.
- The raw integer mistake count is returned. The normalised SID score (`tuple[0]`) is discarded.
- Project-side prevalidation (shape, dtype bool, no self-loops, acyclicity) remains required
  before any call to `gadjid`. The wrapper must never call `gadjid` on unvalidated inputs.
- This is not a CDT or R runtime dependency. No R toolchain is required.
- `gadjid` is MPL-2.0; the project is MIT. Runtime use of an MPL-2.0 library from an MIT
  project is unproblematic. The project will not modify or vendor `gadjid` source files.
- The version is pinned exactly (`gadjid==0.1.0`). Upgrading requires rerunning the full SID
  regression test set, including the `sid == parent_aid` agreement test.
- The selection study remains blocked until `sid_score` is implemented, verified against the
  full regression test set (docs/07 Section 10), and the skipped SID scaffold is unskipped or
  deliberately replaced.

Reason:

The spike established that `gadjid` is a clean, stable, numpy-only backend whose `parent_aid`
function is directly cross-validated against R SID v1.1, and whose `sid` function was locally
confirmed to agree with `parent_aid` on DAG inputs. This makes `gadjid` safer and more
maintainable than an internal re-implementation of the Peters and Buhlmann SID algorithm,
which would require separate verification against an external reference. Adding the dependency
in a dedicated commit, with a pinned version, keeps the adoption auditable and reversible.

---

15/05/2026 -- SID verification closed

Decision:

- `gadjid==0.1.0` is the adopted SID backend.
- `sid_score(predicted_dag, true_dag) -> int` is implemented.
- Project-facing argument order is predicted first, true second. The internal call
  flips once at the boundary to `gadjid.sid(true_int8, predicted_int8, edge_direction="from row to column")[1]`.
- The raw integer mistake count is returned; the normalised SID score is discarded.
- Project-side validation (bool dtype, square 2D, no self-loops, acyclicity) rejects
  all invalid or non-DAG inputs before any backend call.
- The deferred SID scaffold was replaced by an active backend-reference test asserting
  `sid_score(empty 3x3, chain 0->1->2) == 3`.
- The `None -> "deferred"` path was removed from `_derive_sid_status` and
  `check_sid_self_zero`. Both functions now return `int` directly.
- `require_sid` on `assert_ground_truth_compatibility` is retained in the signature as
  a backward-compatible no-op. It no longer has any gate effect. Removing it is a small
  compatibility follow-up and is not a current blocker.
- The regression test set covers: backend importability and pinned version, identity on
  fixed and generated DAGs, raw-count extraction, argument-order asymmetry,
  backend-call mapping (monkeypatched), edge-direction sensitivity witness,
  `sid == parent_aid` agreement on fixed pairs, dtype contract (int8/int64/uint8/float64
  rejected), invalid-graph rejection (cyclic, bidirected, self-loop, non-square, shape
  mismatch), and no numeric fallback on invalid input.
- Full suite result at closure: 384 passed, 0 skipped, 2 warnings. The 2 warnings are
  the pre-existing `RuntimeWarning: invalid value encountered in matmul/subtract` from
  `test_dagma_wrapper_residuals.py::test_non_finite_sigma_sets_unavailable_unresolved_noise_policy`,
  unchanged and unrelated to SID.

Consequence:

- The selection study is now unblocked from the SID side.
- The selection study has not yet been run and must still follow the documented
  base-model selection protocol.
- This does not begin prior-loss implementation or DCDI Commit 11.



18/05/2026 -- Selection-study results schema frozen

Decision:

- docs/08a_experiment_tracking_and_results_schema.md is added as the
  frozen local result-storage and tracking contract for the base-
  model selection study, at schema_version = 1.
- The canonical run identifier is derived deterministically from
  (model, condition, seed_population, seed_replicate_index,
  configuration_hash). configuration_hash is the SHA-256 digest of
  the canonical JSON serialisation of config_resolved with sorted
  keys; the algorithm name is recorded per run via
  configuration_hash_algorithm.
- Local files under results/model_selection/... are the authoritative
  experiment record. W&B and any equivalent external tracking tool
  are optional mirrors only. Every reportable thesis value traces to
  a local file.
- The per-intervention list in Section 6.10 is the source of truth
  for MMD. Run-level MMD aggregates (mmd_primary,
  mmd_sensitivity_unit_variance, mmd_bandwidth_sweep,
  mmd_available_count, mmd_missing_count, mmd_bandwidth_used_value)
  are derived convenience fields that must be consistent with the
  per-intervention list.
- The schema introduces one taxonomy value (mmd_status =
  unavailable_other) that extends the docs/04 sampler_status
  taxonomy to cover MMD-specific failures. Future alignment is a
  docs/04 amendment concern, not a schema_version concern.
- Two open follow-ups are surfaced in Section 16 and must be
  resolved before the selection-study runner is written:
    - Section 16.1: docs/02 vs docs/03 seed-discipline conflict on
      whether the DAGMA fit calls torch.manual_seed, np.random.seed,
      and dagma.utils.set_random_seed.
    - Section 16.4: docs/02 v1.4 editorial amendment to remove
      deferred-SID phrasing from docs/02 Section 3.4 and Section 7
      item 6.

Reason:

The selection-study runner cannot be designed defensibly without a
frozen contract for what every fit records to disk. Without that
contract, the MMD-unavailable / reliability-limited rule in
docs/02_base_model_selection.md becomes unenforceable from saved
records, threshold-robustness reporting would require retraining,
and post-hoc rationalisation is structurally unprevented. The
contract is intentionally narrow: it freezes the selection-study
schema only; main-study extensions are introduced through future
schema_version bumps.

Consequence:

The next implementation artefact is docs/08_base_model_selection_plan.md,
which will follow the commit-structured planning pattern of docs/05
and docs/06 with a numbered commit sequence, per-commit acceptance
criteria, and explicit gate commits (schema-conformance gate and
end-to-end smoke-check gate). The Section 16.1 seed-discipline
conflict is expected to be resolved as part of the runner's
configuration and seed-derivation policy commit, not as a standalone
prerequisite. The Section 16.4 docs/02 v1.4 editorial amendment is a
small follow-up that should land before docs/08 is drafted, so that
docs/08 cites a consistent docs/02.

Prior-loss implementation remains deferred. DCDI Commit 11 remains
paused. DAGMA is not selected.


---


18/05/2026 -- Section 16.1 seed-discipline conflict resolved by Option A; docs/02 amended to v1.5

Decision:

- The seed-discipline conflict between docs/02_base_model_selection.md
  Section 3.5 v1.4 (which mandated uniform seed-setter calls across
  all candidates) and docs/03_decision_log.md 13/05/2026 (which
  recorded that the DAGMA fit does not call torch.manual_seed,
  np.random.seed, or dagma.utils.set_random_seed) is closed.
- The conflict was surfaced in
  docs/08a_experiment_tracking_and_results_schema.md Section 16.1
  and named two eligible resolution options:
  - Option A: amend docs/02 Section 3.5 to relax the uniform
    seed-setter requirement for candidates whose fit path is
    verified-deterministic by construction. The DAGMA wrapper stays
    unchanged. For DAGMA runs, the seed_torch, seed_numpy, and
    seed_dagma fields in the run record are null.
  - Option B: amend the DAGMA wrapper source and docs/03 to make
    the DAGMA fit path call torch.manual_seed, np.random.seed, and
    dagma.utils.set_random_seed at fit time, so the corresponding
    fields in the run record are always non-null.
- Option A is chosen.

Reason:

1. DAGMA fit is verified-deterministic. docs/04b_source_inspection.md
   D-6 confirms no internal random initialisation; W starts at zero;
   Adam is deterministic given fixed gradients. The runtime probe at
   docs/04c_runtime_probe_results.md D-P2 confirms this empirically.
   The seed setters are no-ops for the DAGMA fit path.
2. The DAGMA wrapper hermetic design was deliberate.
   docs/06_dagma_wrapper_implementation_plan.md Section 6 explicitly
   forbids importing dagma.utils on the grounds that
   dagma.utils.set_random_seed mutates global NumPy state.
3. Recording a seed value for a setter that was never called would
   be misleading. Null is the honest record.
4. Global RNG mutation has side effects beyond the wrapper, and the
   project has been moving away from this pattern.
5. docs/02 Section 3.5 was written before the source inspection and
   runtime probes. docs/03 13/05/2026 represents the more informed
   position. The proper resolution is to update the protocol, not to
   undo the evidence-based wrapper decision.

Consequence:

- docs/02_base_model_selection.md is amended to v1.5. The change is
  scoped to Section 3.5 and the change log: the uniform seed-setter
  requirement is relaxed to apply only to candidates whose fit path
  depends on global RNG state. DCDI continues to record non-null
  seed_torch and seed_numpy. DAGMA records null for seed_torch,
  seed_numpy, and seed_dagma. No frozen tactical constant in
  Section 9 is changed. The lexicographic decision rule, the
  disqualification conditions, the tie-breaker logic, the timeline,
  the budget, the model shortlist, the thresholds, the MMD policy,
  the metric definitions, and the calibration/evaluation seed split
  are unchanged.
- The DAGMA wrapper at src/symbolic_priors_cd/wrappers/dagma.py is
  unchanged by this decision.
- The DCDI wrapper is unchanged by this decision.
- The selection-study runner's configuration schema and per-purpose
  seed-derivation rule are implemented at
  experiments/selection_study/config.py per the commit-structured
  planning pattern of docs/08_base_model_selection_plan.md Commit 2.
- This amendment is reproducibility and seed-discipline only. No new
  scientific commitment is introduced.
- DAGMA is not selected by this commit. The base-model selection
  study has not been run.
- Prior-loss implementation is not started by this commit.
- DCDI Commit 11 is not resumed by this commit.

---

19/05/2026 - DCDI public wrapper assembly resumed as limited exception to docs/05 pause; Commit 11 loss-hook remains paused; C-P11 review remains pending

1. Blocker discovered during Commit 5 investigation

Decision:

- The Commit 5 schema-conformance gate investigation reached a hard
  stop: the public `DCDIWrapper` class required by
  `experiments/selection_study/pipeline.resolve_wrapper(
  "symbolic_priors_cd.wrappers.dcdi:DCDIWrapper")` did not exist in
  `src/symbolic_priors_cd/wrappers/`.
- The DCDI infrastructure that does exist (`_dcdi_training`,
  `_dcdi_sampling`, `_dcdi_utils`, the in-module thresholding helper
  `_predict_adjacency_at`, and the `_graph_status` classifiers)
  collectively contains every mechanism needed for a Commit-5 toy
  end-to-end fit, but no public class assembled them behind a
  unified surface returning a `WrapperDiagnostics` record.

Reason:
The selection-study runner in Commit 5 cannot proceed without a
public DCDI wrapper class. The remaining private modules already
expose the underlying mechanism. Building the public class is
therefore a strictly mechanical assembly, not a research step.

2. Scope of the limited exception

Decision:

- A public `DCDIWrapper` class is added to
  `src/symbolic_priors_cd/wrappers/dcdi.py`. The class consumes the
  existing private helpers verbatim. The wrapper does not
  reimplement training, sampling, thresholding, or status
  classification.
- The class exposes five public methods: `fit`,
  `native_edge_continuous`, `thresholded_adjacency`,
  `sample_interventional`, and `get_diagnostics`. The shape mirrors
  the existing `DAGMAWrapper` pattern at
  `src/symbolic_priors_cd/wrappers/dagma.py`.
- The `fit` signature is
  `fit(X_train, *, X_val, preprocessor, seed, n_iter, config=None)`.
  `X_val` and `n_iter` are present (and absent from the DAGMA
  pattern) because the consumed private training function
  `_dcdi_training.run_dcdi_training_loop` requires them and because
  `docs/05_dcdi_wrapper_implementation_plan.md` Section 9 freezes
  the validation-data API as caller-supplied with no internal split
  and no internal generation.
- `DCDIConfig` is re-exported from `_dcdi_training` via a
  module-level `__getattr__` so importing
  `symbolic_priors_cd.wrappers` does not eagerly load the pinned
  DCDI source. The lazy import preserves the convention recorded in
  the wrappers package docstring.
- The wrapper is registered in
  `src/symbolic_priors_cd/wrappers/__init__.py` alongside
  `DAGMAWrapper` and `DCDIConfig`.

Reason:
The wrapper is mechanical assembly only. Every component it consumes
already passed its own commit-level acceptance test (DCDI Commits 1
through 9 are green per `docs/phase_2b_dcdi_readout.md`). Adding a
public class to expose those components does not change any
algorithm, any selection criterion, any threshold, any seed
discipline rule, or any evaluation rule. The exception is narrowly
scoped to unblock `docs/08_base_model_selection_plan.md` Commit 5.

3. What remains paused

Decision:

- DCDI Commit 11 (loss-hook injection): paused. The wrapper does NOT
  expose a `set_loss_hook` method. The `WrapperDiagnostics.loss_hook_name`
  field is always `None`. The private training function's
  `loss_hook` parameter remains accessible only through the private
  module path; the public wrapper does not wire it up.
- C-P11 review (`docs/04f_dcdi_sampler_quality_diagnostic.md`):
  unchanged. The diagnostic remains as it stands. No threshold has
  been weakened. No silent graph repair has been introduced. The
  sampler-quality concern is a base-model / wrapper-design open
  question to be resolved at project level.
- DCDI Commits 12, 13, and 14 (diagnostics polish, full-convergence
  integration, public-API docstring stabilisation): not started in
  this commit. The wrapper exposes diagnostics, but the full-
  convergence integration test and the public-API stabilisation
  pass remain deferred to their original commits.
- Prior-loss implementation: not started.

4. Sampler-status discipline (CRITICAL)

Decision:

- `sampler_status` on the wrapper reports MECHANICAL availability
  only. Concretely: `valid_dag` and a callable sampling API map to
  `"available"`; non-`valid_dag` maps to
  `"unavailable_invalid_graph"`. The wrapper does NOT degrade
  `sampler_status` based on the QUALITY of the learned structure.
- C-P11 was a structure-quality failure (the learned graph missed
  strong true edges while the thresholded graph was a valid DAG and
  the sampling API was callable). Marking the sampler unavailable
  on structure-quality grounds would hide the C-P11 concern behind
  a status flag and pre-empt the project-level review. The wrapper
  refuses to do this.
- Structure-quality concerns continue to surface in downstream
  metrics (SHD, SID, MMD) and in the
  `docs/08a_experiment_tracking_and_results_schema.md` Section 6.10
  per-intervention MMD records, not in `sampler_status`.

Reason:
The status taxonomy in `docs/04_wrapper_api_contract.md` Section 7
defines `sampler_status` as a mechanical-availability axis, not a
quality axis. Conflating the two would make MMD-unavailable rates
incomparable across runs and would silently weaken the C-P11
diagnostic by encoding a quality judgement into a mechanical-status
field. The selection-study report logic in
`docs/02_base_model_selection.md` Section 6 Case 6 reads
`sampler_status` as a mechanical signal and would mis-attribute
structure failures to sampler failures if the wrapper conflated
them.

5. What does NOT change

Decision:

- No selection criterion changes. The lexicographic decision rule
  (`docs/02_base_model_selection.md` Section 2), the
  disqualification conditions (Section 5), the tie-breaker logic
  (Section 6), the timeline (Section 8), the budget (Section 8),
  the synthetic cell (Section 3.1), the intervention values
  (Section 4.2), the SID tie margin, the catastrophic SHD-
  degradation threshold, the threshold robustness triples, the
  DCDI threshold of `0.5`, the DAGMA threshold of `0.3`, and the
  calibration/evaluation seed split all remain as frozen.
- No evaluation rule changes. SID, SHD, MMD, and the
  MMD-unavailable rule (Section 6 Case 6) are unchanged.
- No metric primitive changes. `metrics/structural.py`,
  `metrics/interventional.py`, and `metrics/sanity_checks.py` are
  unchanged.
- No private DCDI module changes.
  `src/symbolic_priors_cd/wrappers/_dcdi_training.py`,
  `_dcdi_sampling.py`, and `_dcdi_utils.py` are consumed as-is.
- No DAGMA wrapper changes.
  `src/symbolic_priors_cd/wrappers/dagma.py` and the DAGMA private
  helpers are unchanged.
- `docs/02_base_model_selection.md`,
  `docs/04_wrapper_api_contract.md`,
  `docs/05_dcdi_wrapper_implementation_plan.md`, and
  `docs/08a_experiment_tracking_and_results_schema.md` are
  unchanged.
- The Commit-5 schema-conformance gate has NOT been opened by this
  decision. This entry only unblocks the wrapper-resolution step
  that Commit 5 will later perform.

Consequence:

- The selection-study runner's `pipeline.resolve_wrapper(...)` step
  can now dereference `symbolic_priors_cd.wrappers.dcdi:DCDIWrapper`
  to a class object. The Commit-5 toy end-to-end fit can proceed
  for both DAGMA and DCDI candidates.
- The C-P11 sampler-quality concern remains the binding open
  question for the DCDI candidate. The wrapper does not pre-empt
  this concern. Any selection-study conclusion that uses MMD for
  DCDI must still account for `docs/04f` per
  `docs/02_base_model_selection.md` Section 6 Case 6.
- DAGMA is not selected by this commit. The base-model selection
  study has not been run.
- DCDI Commit 11 is not resumed by this commit.
- Prior-loss implementation is not started by this commit.
- This decision is wrapper-assembly only. No new scientific
  commitment is introduced.

---

## 19/05/2026 — Configuration extended with SCM-generation parameters for offline threshold robustness

### Decision

Four SCM-generation fields are added as first-class members of the
selection-study runner's `Configuration` dataclass in
`experiments/selection_study/config.py`:

- `n_nodes: int = 3`
- `expected_edges: int = 3`
- `noise_scale: float = 1.0`
- `weight_magnitude_range: tuple[float, float] = (0.5, 2.0)`

These fields enter `to_canonical_dict`, participate in
`configuration_hash`, and are required by `load_config` when
reading a configuration file from disk. The pipeline's
`run_single_fit` now reads them from the resolved configuration
when constructing the SCM, in place of the previous module-level
`SCHEMA_GATE_N_NODES` and `SCHEMA_GATE_EXPECTED_EDGES` constants
(which are removed from `pipeline.py`). The remaining schema-gate
constants (`SCHEMA_GATE_N_TRAIN`, `SCHEMA_GATE_N_VAL_DCDI`,
`SCHEMA_GATE_DCDI_N_ITER`, `SCHEMA_GATE_DCDI_CONFIG_KWARGS`,
`SCHEMA_GATE_MMD_N_SAMPLES`) remain in place and are separate
Phase A/B concerns.

### Reason

Commit 7 (offline threshold-robustness re-computation) exposed
that the existing run record was insufficient to reconstruct the
true graph from saved fields alone. `run.json` persisted
`graph_seed` but not the remaining inputs to
`generate_linear_gaussian_scm`, so offline metric recomputation
on a saved run could not, by construction, produce the same
true adjacency that the run was scored against. The minimum
project-level fix considered two options:

- SCM-A: extend `Configuration` with the four SCM-generation
  fields. The fields enter `configuration_hash`, so any change
  to the SCM regime forces a new run identity; the existing
  schema-gate cell is preserved by the four default values; the
  Phase A/B runners will need these fields anyway under
  `docs/02_base_model_selection.md` Section 9. The selection-study
  protocol's SCM regime is part of the experimental configuration,
  not an off-protocol environment detail, so it belongs in
  `configuration_hash`.
- SCM-B: write a sibling artefact such as `scm_generation_spec.json`
  alongside `run.json`. Smaller code surface, but the SCM regime
  would not enter `configuration_hash` and consumers would carry
  an extra path next to every run.

SCM-A is chosen because the SCM regime is part of the
experimental configuration and must participate in
`configuration_hash`. SCM-B would have decoupled the SCM regime
from the run identity, which is the wrong direction for a study
that selects a base model under a fixed protocol cell.

### What does NOT change

- `docs/01_research_question_and_commitments.md`,
  `docs/02_base_model_selection.md`,
  `docs/04_wrapper_api_contract.md`,
  `docs/05_dcdi_wrapper_implementation_plan.md`,
  `docs/06_dagma_wrapper_implementation_plan.md`,
  `docs/08_base_model_selection_plan.md`, and
  `docs/08a_experiment_tracking_and_results_schema.md` are
  unchanged.
- No selection criterion changes; no evaluation rule changes; no
  metric primitive changes; no wrapper changes.
- The `run.json` schema_version remains `1`. The new fields
  appear inside `config_resolved`, which is already a passthrough
  of the resolved configuration; no new top-level mandatory key
  is introduced.
- The schema-gate cell is preserved by the four default values
  (`n_nodes=3`, `expected_edges=3`, `noise_scale=1.0`,
  `weight_magnitude_range=(0.5, 2.0)`). The existing
  schema-gate fits behave identically except that the SCM
  parameters now travel through `Configuration` and
  `config_resolved`.

### Consequence

- Existing schema-gate runs computed before this entry would no
  longer round-trip byte-for-byte because the canonical JSON,
  and hence `configuration_hash`, now also covers the four SCM
  fields. No persistent run records are being migrated by this
  entry; the change is forward-looking for the Commit-7 and
  Phase A/B work that has not yet been executed.
- The Phase A/B runner configurations must explicitly set
  `n_nodes=10`, `expected_edges=20`, and the noise / weight
  parameters required by `docs/02_base_model_selection.md`
  Section 9 (ER2: `expected_edges = 2 * n_nodes`). The defaults
  exist only to preserve the schema-gate cell; they are not
  authoritative for the real selection study.
- Commit 7's offline threshold-robustness re-computation now
  has a fully reconstructable true-graph path from saved fields
  alone, without touching `pipeline.py` or any wrapper module
  at recomputation time.
- The `docs/02_base_model_selection.md` constants for the real
  selection study (`n_nodes=10`, `expected_edges=20`) are
  unchanged by this entry; they will be supplied at Phase A/B
  configuration construction time.

---

## 20/05/2026 — Threshold robustness records structured SHD/SID unavailability

### Decision

The per-threshold records emitted by
`experiments/selection_study/threshold_robustness.py` carry two
new explicit fields, `shd_unavailable_reason` and
`sid_unavailable_reason`, alongside the existing `shd` and `sid`
fields. Each metric is now reported under one of two mutually
exclusive forms:

- a concrete integer value with the corresponding
  `*_unavailable_reason` set to `None`, or
- the value `None` with a non-empty string in the corresponding
  `*_unavailable_reason`.

Specifically:

- SHD is unavailable when the predicted adjacency at a threshold
  classifies as `invalid_shape` (square-shape pre-condition fails)
  or `self_loop` (predicted-diagonal pre-condition fails). For
  `valid_dag`, `cyclic`, and `bidirected` predictions, SHD is
  computed unconditionally and the reason is `None`.
- SID is unavailable for every non-`valid_dag` status because the
  SID primitive requires a valid DAG. For `valid_dag` predictions
  SID is computed and the reason is `None`.

An internal invariant check raises `RuntimeError` if either
mutual-exclusion pair is ever violated.

### Reason

The Commit-7 cleanup pass surfaced a boundary case: the project
SHD primitive in `src/symbolic_priors_cd/metrics/structural.py`
correctly rejects any predicted adjacency with non-zero diagonal
entries. A neighbouring threshold inside a robustness sweep can
legitimately produce a `self_loop` adjacency, and the structured-
metric-unavailability pattern is the appropriate way to record
this without weakening the source metric, inventing a fallback
inside the runner, or silently emitting `shd=None`.

### What does NOT change

- `src/` is unchanged. The SHD primitive's strict adjacency
  validation in
  `src/symbolic_priors_cd/metrics/_graph_validation.py` is
  preserved verbatim.
- `experiments/selection_study/config.py`,
  `experiments/selection_study/pipeline.py`,
  `experiments/selection_study/loader.py`,
  `experiments/selection_study/preflight.py`,
  `experiments/selection_study/identity.py`,
  `experiments/selection_study/sampling.py`, and
  `experiments/selection_study/run.py` are unchanged by this
  entry.
- The primary `run.json` schema is unchanged. The top-level
  `sid: int` invariant on `run.json` still holds; the
  schema-conformance gate still refuses to write a partial
  primary record when the primary threshold yields a non-`valid_dag`
  graph. Nullable `sid` exists only inside threshold-robustness
  records.
- The four-field SCM persistence introduced on 19/05/2026 is
  unaffected; `configuration_hash` still covers `n_nodes`,
  `expected_edges`, `noise_scale`, and `weight_magnitude_range`.
- No selection criterion or evaluation rule is changed by this
  entry.

### Consequence

- Threshold-robustness consumers (the eventual selection-study
  report) can disaggregate metric availability at each threshold
  by reading the explicit reason fields. Self-loop thresholds no
  longer raise; they are reported as `shd=None` with an explicit
  reason that names the failing pre-condition.
- The primary-threshold reconciliation against `run.json["shd"]`
  and `run.json["sid"]` remains exact integer equality; the
  primary threshold of any successful run is always `valid_dag`,
  so the new `*_unavailable_reason` fields are `None` at the
  primary index.
- No `run.json` migration is required. The new fields live only
  in the sibling `threshold_robustness.json` artefact and the
  in-memory record returned by `recompute_at_thresholds`.

---

## 20/05/2026 -- Validation NLL trajectory exposed on DCDI TrainingResult

### Decision

The DCDI augmented-Lagrangian training loop already computes a
per-stop-check validation-NLL trajectory while training. The
trajectory was previously discarded at function exit. It is now
returned on `TrainingResult` as
`validation_nll_history: list[float]` and surfaced through
`DCDIWrapper.get_diagnostics()` under
`convergence_info["validation_nll_history"]`, with the cadence
recorded alongside as
`convergence_info["validation_nll_stop_crit_win"]`.

### Reason

The DCDI training-budget pilot needs the validation-NLL
trajectory as one of the signals the user reads when choosing
`num_train_iter`. The trajectory was already collected internally
at the existing stop-check cadence (one pre-training baseline plus
one value every `stop_crit_win` iterations), so surfacing it
requires no recomputation and no change to optimisation
behaviour. This is diagnostic instrumentation for the DCDI
budget decision, not a behavioural change.

### What does NOT change

- Training behaviour is unchanged: losses, optimiser, seeds,
  stopping logic, thresholds, patience logic, graph thresholding,
  sampler behaviour, and model outputs are all preserved
  bit-for-bit. The change is a pure pass-through of an
  already-computed list to `TrainingResult` and onward to the
  wrapper diagnostics dict.
- No `run.json` schema bump. The trajectory lives inside the
  wrapper diagnostics dict, not as a top-level run-record field;
  it is consumed by the C-P15 pilot probe and any later notebook
  analysis but is not promoted to a mandatory `run.json` field.
- No selection criterion, evaluation rule, or metric primitive
  changes.
- DCDI Commit 11 (loss-hook injection) remains paused. This
  entry does not reopen Commit 11; it is purely a diagnostic
  surface.

### Consequence

- The C-P15 DCDI training-budget pilot probe writes a compact
  semicolon-delimited `validation_nll_trajectory_summary` field
  in its CSV output, computed from the new diagnostic. The
  summary covers count, finite/non-finite split, first / last /
  finite-min / finite-argmin, a short numeric tail, and the
  cadence.
- The DCDI augmented-Lagrangian state (`final_gamma`,
  `final_mu`, `gamma_update_iters`, `mu_update_iters`) was
  already exposed on `model_specific_diagnostics`; the pilot
  probe now records six additional columns derived from those
  fields (`final_gamma`, `final_mu`, `gamma_update_count`,
  `mu_update_count`, `last_gamma_update_iteration`,
  `last_mu_update_iteration`).
- Regression tests: the full pytest suite remains green
  (666 passed, 2 pre-existing warnings unrelated to this
  change). The DCDI training-equivalence regression test does
  not consult the new field and is unaffected; a new
  type-stability test asserts the field exists and is a list of
  floats.
- The pilot remains diagnostic-only. The user chooses
  `num_train_iter` after reading the pilot output; this decision
  log entry does not freeze a value.

---

## 20/05/2026 -- DCDI training-budget ceiling frozen from C-P15 pilot

### Decision

`dcdi_num_train_iter = 300000` is adopted as the hard maximum
iteration ceiling for DCDI in the selection study, with the
existing patience-based early stopping enabled
(`stop_crit_win = 100`, `train_patience = 5`,
`h_threshold = 1e-8`). The ceiling is a training budget rather
than a tunable hyperparameter; it MUST NOT be varied by held-out
evaluation records and MUST NOT appear as one of the five Phase B
configurations.

### Evidence

C-P15 full-pilot rows for reproduction-pool seeds 101, 102, 103
in
`inspection/probes/output/c_p15_dcdi_training_budget_pilot.csv`:

- All three seeds reached the first patience gate well below
  the `num_train_iter_cap = 300000` ceiling.
- `final_iteration` values: 118900 (seed 101), 75300 (seed 102),
  86700 (seed 103). Worst observed: 118900 on seed 101.
- All three seeds finished with `graph_status = valid_dag`,
  `sampler_status = available`, `training_status = converged`,
  `final_h <= 1e-8` (8.83e-09, 9.19e-09, 7.89e-09 respectively).
- `final_mu` shows substantial seed-to-seed variance
  (approximately 2.25e+07 on seeds 102 and 103, approximately
  3.69e+11 on seed 101), reflecting heterogeneity in how much
  acyclicity pressure different ER2 graph realisations require
  before `h` falls below the threshold.

C-P11
(`docs/04f_dcdi_sampler_quality_diagnostic.md`) was run at
`n_iter = 30000` on a 3-node fixture and is scoped as
under-budget diagnostic evidence. It is NOT binding evidence
against real-budget DCDI behaviour at the 10-node ER2 cell.

The pilot does NOT count as Phase A reproduction evidence. Phase
A reproduction uses paper-aligned reproduction seeds and a
separate acceptance protocol per `docs/02` Section 3.3.

### What does NOT change

- No selection criterion, no evaluation rule, no metric
  primitive.
- No `src/` change. The wrapper already exposes the diagnostic
  fields used by the pilot (see the 20/05/2026 "Validation NLL
  trajectory exposed on DCDI TrainingResult" entry).
- No `run.json` schema bump.
- DCDI Commit 11 (loss-hook injection) remains paused.
- `docs/02_base_model_selection.md` is unchanged. The frame in
  which this ceiling lives in the selection-study protocol is
  the responsibility of the separate user-adjudicated `docs/02`
  v1.6 amendment.

### Consequence

The DCDI training-budget question raised by `docs/08c` Section 2
is resolved by adoption of the ceiling above. The following
related items remain for the separate `docs/02` v1.6 amendment
and any associated `docs/03` entries:

- DAGMA paper-vs-library budget choice (`warm_iter`,
  `max_iter`).
- Phase B sparsity policy and grid endpoints.
- C-P11 reapplication policy before held-out evaluation (rerun
  at the new budget, or explicit scope-statement).
- `mmd_n_samples = 1000` elevation to a top-level
  `Configuration` field.
- `n_train` and DCDI validation-split `n_val_dcdi`
  `Configuration` fields.
- Visual / reporting artefact requirements for notebook
  inspection of the diagnostic trajectories.

The full readout of the pilot evidence and the per-seed table
is in `docs/08d_dcdi_training_budget_pilot.md`.

---

## 20/05/2026 -- docs/02 v1.6 real-run constants and Phase B sparsity policy frozen

### Decision

`docs/02_base_model_selection.md` is amended to **v1.6**. The
amendment encodes the remaining real-run protocol decisions
audited in `docs/08c_real_run_constants_and_training_budget_audit.md`
and corroborated by the C-P15 pilot
(`docs/08d_dcdi_training_budget_pilot.md`).

Summary of `docs/02` v1.6 amendments:

- Section 3.3 DAGMA-linear starting point extended with the
  paper-aligned optimisation values from DAGMA paper
  Section C.1.1: `warm_iter = 20000`, `max_iter = 70000`,
  Adam `lr = 3e-4`, `(beta_1, beta_2) = (0.99, 0.999)`. These
  override library defaults at the call site. The DAGMA paper's
  relative-loss convergence rule is documented; the project
  wrapper does not implement or expose a separate observed
  early-stopping iteration count, so the DAGMA run record's
  top-level `n_iterations` field remains `None` and the
  configured optimisation upper bound is recorded under
  `model_specific_diagnostics`.
- Section 3.3 DAGMA Phase B sparsity is now a five-value sweep
  on `lambda1 in {0.01, 0.025, 0.05, 0.1, 0.25}`, anchored on
  the paper value `0.05`. The previous pinned-only treatment is
  superseded.
- Section 3.3 DCDI-G starting point extended with:
  `dcdi_num_train_iter = 300000` hard ceiling with patience-based
  early stopping (pilot-derived in `docs/08d`),
  `stop_crit_win = 100`, `train_patience = 5`, MLP architecture
  (`hidden_units = 16`, `hidden_layers = 2`, leaky-ReLU, Xavier),
  and an 80/20 DCDI validation split (800 fit, 200 validation)
  drawn from the `n_train = 1000` observational batch.
- Section 3.3 DCDI Phase B sparsity is now a five-value local
  sweep on `reg_coeff in {0.01, 0.03, 0.1, 0.3, 1.0}` anchored
  on the upstream default `0.1`. The previous "5 values
  spanning `10^-7` to `10^2`" treatment is superseded.
- Phase B paragraph in Section 3.3 reads as exactly 5
  configurations per model in each model's native
  parameterisation, frozen before execution, with no post-hoc
  grid expansion allowed after seeing calibration or held-out
  results.
- Section 4.2 MMD-sample wording tightened: `mmd_n_samples = 1000`
  is now a top-level `Configuration` field that enters
  `configuration_hash`. It must not remain a schema-gate
  constant. Estimator, bandwidth policy, sensitivity sweep, and
  negative-MMD handling are unchanged.
- Section 7 has a new "C-P11 real-budget reapplication policy"
  subsection. The original C-P11 is scoped as under-budget
  diagnostic evidence; a C-P11-style sampler-quality diagnostic
  must be rerun at the real DCDI budget on a 10-node ER2
  fixture before any held-out evaluation result is interpreted
  as evidence about DCDI's interventional adequacy.
- Section 9 tactical-constants block is extended with the
  corresponding bullets: `n_train = 1000`, DCDI 800/200 split,
  `mmd_n_samples = 1000`, DAGMA `warm_iter`, `max_iter`, Adam
  values, DAGMA `lambda1` Phase B grid, DCDI training-budget
  ceiling, DCDI patience values, DCDI optimiser settings, DCDI
  MLP architecture, DCDI `reg_coeff` Phase B grid, and the
  C-P11 reapplication requirement.

### Reason

This entry closes the remaining Category A real-run protocol
decisions identified by `docs/08c` Sections 4 (DAGMA budget
choice), 5 (Phase B sparsity policy), 3 (C-P11 reapplication),
6 (MMD sample-size policy), and 8 (Configuration / hash
policy), before the runner's `Configuration` extension and the
Phase A runner work in Commit 8 of
`docs/08_base_model_selection_plan.md`. `docs/02` now precedes
the Configuration implementation: any field added to
`Configuration` must be traceable to a `docs/02` v1.6 bullet.

### What does NOT change

- No source code changed in this commit. No `src/`,
  `experiments/selection_study/`, or `tests/` edits are made by
  this `docs/02` amendment; the Configuration extension is the
  separate next step.
- No selection criterion change, no evaluation rule change, no
  metric primitive change. The lexicographic decision rule,
  disqualification conditions, tie-breaker logic, intervention
  values, threshold values, threshold robustness triples,
  calibration/evaluation seed split, timeline, and budget are
  preserved.
- DCDI Commit 11 (loss-hook injection) remains paused. This
  amendment is wrapper-API consumer policy only; it does not
  reopen loss-hook work.
- Visual / reporting artefact requirements for notebook
  inspection are NOT newly amended in `docs/02` v1.6. The
  existing local-file-authoritative reporting plan recorded in
  `docs/02` Section 3.4, Section 3.5, and the `docs/08a`
  schema continues to govern those artefacts. Any future
  notebook-side visualisation surface is a separate document.

### Consequence

- The runner's `Configuration` extension may now proceed: every
  new field has a corresponding `docs/02` v1.6 bullet, and the
  v1.6 amendment is the source of truth for the values that
  must enter `configuration_hash`.
- Phase A and Phase B may proceed in parallel with the C-P11
  real-budget rerun. Held-out evaluation interpretation for
  DCDI MUST NOT be completed until the C-P11 rerun result is
  available or `docs/03` records an explicit scoping decision.
- `docs/02_base_model_selection.md` is now at v1.6. Subsequent
  amendments use the same change-log discipline.

---

## 20/05/2026 -- Configuration extended with docs/02 v1.6 real-run constants

### Decision

`experiments/selection_study/config.py` is extended so the
`Configuration` dataclass carries the real-run constants frozen
in `docs/02` v1.6 (excluding Phase B sparsity grids). The new
fields are:

- shared: `n_train: int`, `mmd_n_samples: int`;
- DCDI-only (`None` for DAGMA configurations): `n_val_dcdi`,
  `dcdi_num_train_iter`, `dcdi_stop_crit_win`,
  `dcdi_train_patience`, `dcdi_train_batch_size`, `dcdi_lr`,
  `dcdi_h_threshold`, `dcdi_hidden_units`, `dcdi_hidden_layers`;
- DAGMA-only (`None` for DCDI configurations): `dagma_warm_iter`,
  `dagma_max_iter`, `dagma_lr`, `dagma_beta_1`, `dagma_beta_2`.

Each new field appears in `to_canonical_dict`, is required by
`load_config`, and participates in `configuration_hash`. The
dataclass validation enforces: positive numeric values, no
bool-as-int, `n_val_dcdi < n_train` for DCDI, and model-conditional
presence (DCDI configurations require every DCDI-only field
non-None and every DAGMA-only field None; DAGMA configurations
require the symmetric condition).

DAGMA `lambda1` and DCDI `reg_coeff` are intentionally NOT
top-level `Configuration` fields. They are Phase B sparsity
sweep values and remain inside `PhaseBConfiguration.hyperparameters`
per `docs/02` v1.6 Section 3.3.

### Reason

`docs/02` v1.6 froze the real-run protocol values that Commit 8
Phase A will need. Adding them to `Configuration` first lets the
selection-study runner persist a complete `config_resolved`
record under `configuration_hash`, with the schema-conformance
machinery and threshold-robustness reconstruction already in
place. The Configuration extension is a prerequisite for Commit 8;
this commit does not yet wire the new fields into pipeline
consumer behaviour.

### What does NOT change

- No source code outside `experiments/selection_study/config.py`
  and the tests that construct `Configuration` fixtures changed.
  No `src/`, `experiments/selection_study/pipeline.py`, or wrapper
  edit was needed.
- No selection criterion, evaluation rule, wrapper algorithm,
  or metric primitive changed.
- No notebook, configuration file, results directory, or
  dependency manifest changed.
- `docs/02` is not edited by this commit.
- The schema-gate-honest defaults are: `n_train = 64`,
  `mmd_n_samples = 64` for shared fields (matching the
  schema-gate constants the toy pipeline actually consumes);
  the DCDI-only / DAGMA-only fields default to `None` so they
  cannot silently leak across models. Existing toy-fit fixtures
  are updated to set the model-appropriate fields with values
  that match what the current schema-gate pipeline actually
  uses, so `config_resolved` remains honest for toy runs.
- Schema-gate toy defaults are NOT authoritative for Phase A/B.
  Phase A/B configurations MUST explicitly set every new field
  to the `docs/02` v1.6 value rather than rely on schema-gate
  defaults.

### Consequence

- `Configuration` is now ready for Commit 8 to consume; the
  pipeline will read `n_train`, `mmd_n_samples`, and the
  model-conditional fields from `resolved_config` in place of
  the schema-gate constants.
- Phase B sparsity sweep values continue to live inside
  `PhaseBConfiguration.hyperparameters`; the Phase B grid
  freezing in `docs/02` v1.6 Section 3.3 must be encoded in the
  Phase A/B configuration JSON files that the runner will later
  consume.
- The full pytest suite remains green (720 passed, 2 pre-existing
  RuntimeWarnings unrelated to this change). The new
  Configuration tests cover canonical-dict membership,
  hash-participation, missing-field rejection by `load_config`,
  bool-as-int rejection, zero/negative rejection,
  `n_val_dcdi < n_train`, model-conditional presence, and a
  round-trip preservation check for both DAGMA and DCDI.

---

## 20/05/2026 -- Pipeline consumes Configuration real-run fields

### Decision

`experiments/selection_study/pipeline.py:run_single_fit` is
migrated to read real-run constants from the resolved
Configuration instead of from module-level schema-gate constants.
Specifically:

- `config.n_train` is used for the observational sample count.
- `config.mmd_n_samples` is passed through to
  `compute_per_intervention_records` as the per-intervention
  MMD sample count.
- DAGMA fits build `DAGMAConfig` from
  `config.dagma_warm_iter`, `config.dagma_max_iter`,
  `config.dagma_lr`, `config.dagma_beta_1`, and
  `config.dagma_beta_2`. The remaining `DAGMAConfig` fields
  (`T`, `lambda1`, `s`, `mu_init`, `mu_factor`,
  `w_threshold_internal`, `project_threshold`,
  `h_diagnostic_threshold`) are not top-level Configuration
  fields and continue to use their `DAGMAConfig` defaults.
- DCDI fits build `DCDIConfig` from
  `config.dcdi_h_threshold`, `config.dcdi_lr`,
  `config.dcdi_train_batch_size`, `config.dcdi_train_patience`,
  `config.dcdi_stop_crit_win`, `config.dcdi_hidden_layers`
  (`DCDIConfig.num_layers`), and `config.dcdi_hidden_units`
  (`DCDIConfig.hid_dim`). The DCDI fit also receives
  `n_iter = config.dcdi_num_train_iter` and
  `X_val` constructed via the new split policy.

DCDI validation data is drawn from the same `config.n_train`
observational batch via a deterministic permutation seeded by
`validation_data_seed`. The pipeline no longer issues a separate
`sample_observational` call to draw an independent DCDI
validation set. DAGMA continues to use the full batch for
fitting and does not consume `validation_data_seed`.

The module-level constants `SCHEMA_GATE_N_TRAIN`,
`SCHEMA_GATE_N_VAL_DCDI`, `SCHEMA_GATE_DCDI_N_ITER`, and
`SCHEMA_GATE_DCDI_CONFIG_KWARGS` are removed from
`pipeline.py`. Their toy values now live on the Configuration
fixtures used by schema-gate tests, so `config_resolved` always
matches what the pipeline actually consumed.

DAGMA schema-gate fixtures are updated to use the docs/02 v1.6
paper-aligned optimisation values (`warm_iter = 20000`,
`max_iter = 70000`, `lr = 3e-4`, `beta_1 = 0.99`,
`beta_2 = 0.999`). The DAGMA toy run still produces a
schema-conforming run.json under those values; no silent
revert to the library defaults occurred.

### Reason

`docs/02` v1.6 froze the real-run constants and the Configuration
extension committed earlier today exposed them as top-level
fields. This commit closes the loop so the pipeline reads the
authoritative values from the Configuration rather than from
hidden module constants. The DCDI validation-split policy
aligns with `docs/02` v1.6 Section 3.3 ("DCDI validation split:
from the `n_train = 1000` observational sample, DCDI uses an
80/20 split into 800 fit samples and 200 validation samples;
no additional validation data is drawn").

### What does NOT change

- No source change outside `experiments/selection_study/pipeline.py`,
  the four test files that construct Configuration fixtures, and
  `docs/03_decision_log.md`. No wrapper, metric, notebook,
  configuration file, results-directory, or dependency-manifest
  edit.
- No selection criterion, evaluation rule, wrapper algorithm,
  or metric primitive changed.
- `docs/02` is not edited by this commit.
- Phase A reproduction-pass orchestration is NOT implemented in
  this commit; the schema-gate toy fits remain the only
  pipeline-driven runs at this point. Phase A precondition
  enforcement and the Phase A runner remain separate follow-ups.
- The C-P11 real-budget reapplication remains required before
  held-out interpretation of DCDI per `docs/02` v1.6
  Section 7; this commit does not address it.

### Consequence

- The schema-gate pipeline keeps working under toy
  Configuration values; existing pipeline tests still pass.
- Real Phase A/B Configuration JSON files will need to specify
  the docs/02 v1.6 values for every new field; the pipeline
  will then consume them directly from `resolved_config`.
- New tests pin the consumer behaviour: `n_train` flows into
  observational sampling, `mmd_n_samples` flows into the MMD
  call, DAGMA-only fields flow into `DAGMAConfig`, DCDI-only
  fields flow into `DCDIConfig` and `DCDIWrapper.fit`, the
  DCDI validation split is a single-batch split, and
  `config_resolved` in `run.json` matches the Configuration
  values used.
- Full pytest suite remains green: 727 passed, 2 pre-existing
  RuntimeWarnings unrelated to this work (count up from 720,
  +7 new pipeline tests).

---

## 20/05/2026 -- Phase A real-study protocol guard added; Phase A config files blocked on seed integers

### Decision

A new policy-only helper
`experiments.selection_study.real_study.assert_real_study_constants`
is added. The function affirmatively requires Phase A
configurations to carry the exact docs/02 v1.6 real-study
constants for both shared fields and the appropriate
model-specific fields, plus a non-empty `"reproduction"`
seed population. It is deliberately NOT called from
`Configuration.__post_init__`; toy and schema-gate
Configurations remain constructible. The runner must invoke
the guard explicitly before any Phase A activity.

The Phase A required values enforced by the guard:

- shared: `n_nodes = 10`, `expected_edges = 20`,
  `noise_scale = 1.0`,
  `weight_magnitude_range = (0.5, 2.0)`,
  `n_train = 1000`, `mmd_n_samples = 1000`;
- DAGMA: `model = "dagma"`,
  `threshold_robustness_triple = (0.2, 0.3, 0.4)`,
  `dagma_warm_iter = 20000`, `dagma_max_iter = 70000`,
  `dagma_lr = 3e-4`, `dagma_beta_1 = 0.99`,
  `dagma_beta_2 = 0.999`, every DCDI-only field `None`;
- DCDI: `model = "dcdi"`,
  `threshold_robustness_triple = (0.4, 0.5, 0.6)`,
  `n_val_dcdi = 200`, `dcdi_num_train_iter = 300000`,
  `dcdi_stop_crit_win = 100`, `dcdi_train_patience = 5`,
  `dcdi_train_batch_size = 64`, `dcdi_lr = 1e-3`,
  `dcdi_h_threshold = 1e-8`, `dcdi_hidden_units = 16`,
  `dcdi_hidden_layers = 2`, every DAGMA-only field `None`.

Two anchor-default regression tests pin the wrapper-side Phase A
sparsity anchors so a future wrapper-default change cannot
silently move them:

- `tests/test_dagma_wrapper_interface.py::test_dagma_config_default_lambda1_matches_phase_a_anchor`
  asserts `DAGMAConfig().lambda1 == 0.05`.
- `tests/test_dcdi_wrapper_assembly.py::test_dcdi_config_default_reg_coeff_matches_phase_a_anchor`
  asserts `DCDIConfig().reg_coeff == 0.1`.

These tests protect the current design where Phase A `lambda1`
and `reg_coeff` come from wrapper config defaults rather than
top-level `Configuration` fields.

### Phase A config files NOT created in this commit

The prompt requested two Phase A configuration JSON files at
`experiments/selection_study/configs/phase_a/dagma_reproduction.json`
and
`experiments/selection_study/configs/phase_a/dcdi_reproduction.json`.
Those files are NOT created here. The blocking issue is the
Phase A reproduction-pool seed integers, which are not pinned
by any of `docs/02`, `docs/03`, `docs/08`, or `docs/08d`:

- `docs/02` Section 3.3 names the `"reproduction"` seed
  population for the Phase A reproduction pass but does not
  pin the integer set or the count.
- `docs/08` Commit 8 acceptance criteria record runs under
  `seed_population = "reproduction"` but do not pin integer
  values.
- `docs/08d` C-P15 used seeds `(101, 102, 103)` from the
  reproduction pool but explicitly scopes those as
  pilot-only, not Phase A reproduction-pass evidence.

Seed integers participate in `configuration_hash` (they are
serialised inside `canonical_json`'s `seed_populations` field),
so the choice of integers is part of the Phase A
configuration's identity and must be deliberate. The prompt
forbids inventing them.

The Phase A config files will be added in a separate commit
after the user adjudicates the Phase A reproduction-pool seed
integers and the reproduction-pass seed count.

### What does NOT change

- No `experiments/selection_study/pipeline.py` edit. The
  pipeline already consumes Configuration values per the
  20/05/2026 "Pipeline consumes Configuration real-run fields"
  entry; this commit only adds a policy gate.
- No wrapper, metric, notebook, configuration-file, results-
  directory, or dependency-manifest edit.
- `docs/02`, `docs/08c`, `docs/08d` are not edited.
- `Configuration.__post_init__` is unchanged. Toy and
  schema-gate Configurations continue to construct without
  invoking the guard.
- No Phase A runner, no Phase B implementation.
- No selection criterion, evaluation rule, wrapper algorithm,
  or metric primitive changed.

### Consequence

- The runner (Commit 8c and later) can call
  `assert_real_study_constants(config, stage="phase_a")` to
  reject toy values before any Phase A fit is invoked. The
  guard's error messages name the offending field and the
  expected value.
- The Phase A anchor `lambda1 = 0.05` (DAGMA) and
  `reg_coeff = 0.1` (DCDI) are now regression-pinned.
- The 8a.1 implicit-default inspection
  (the 20/05/2026 inspection report) found no Phase A runtime
  default mismatch outside the already-overridden DAGMA
  `warm_iter` / `max_iter` defaults; no additional Phase A
  remediation is required by this commit.
- Full pytest suite remains green: 758 passed, 2 pre-existing
  RuntimeWarnings unrelated to this work (count up from 727,
  +29 guard tests, +1 DAGMA anchor test, +1 DCDI anchor test).

### Open follow-up

Phase A reproduction-pool seed integers must be pinned by user
adjudication before
`experiments/selection_study/configs/phase_a/*.json` can be
written. Once pinned, the config files plus their loading and
preflight tests can be added under the same authorisation.

---

## 20/05/2026 -- docs/02 v1.7 seed-pool integers frozen

### Decision

`docs/02` is amended to v1.7. The three selection-study seed
populations and their integer members are frozen as a single
block:

- `reproduction = (101, 102, 103)`
- `calibration = (201, 202)`
- `held_out_evaluation = (301, 302, 303, 304, 305)`

The three pools are disjoint by construction. Phase A uses the
three reproduction-pool seeds; Phase B uses the two
calibration-pool seeds per configuration; held-out evaluation
uses the five held-out-evaluation seeds. The amendment touches
Section 3.3 (new "Seed-pool convention" subsection plus explicit
pool references in the Phase A, Phase B, and held-out paragraphs)
and Section 9 (new tactical-constants bullet listing the three
pool integer sets).

### Reason

Seed integers participate in `configuration_hash` (they are
serialised inside `canonical_json`'s `seed_populations` field),
so they had to be pre-specified before any Phase A, Phase B, or
held-out config file could be written. The previous 20/05/2026
"Phase A real-study protocol guard added; Phase A config files
blocked on seed integers" entry recorded that the config files
were blocked on exactly this ambiguity; this amendment closes
the gap.

### C-P15 distinction

The C-P15 DCDI training-budget pilot reused the integers
`(101, 102, 103)` from the reproduction pool, but the C-P15
CSV remains pilot-only diagnostic evidence per the 20/05/2026
"DCDI training-budget ceiling frozen from C-P15 pilot" entry and
`docs/08d_dcdi_training_budget_pilot.md`. Phase A reruns these
seed identifiers through the full selection-study pipeline and
produces separate local artefacts (`run.json`, `config_resolved`,
SHD, SID, MMD, threshold-robustness records); convergence
properties on the same seeds are expected to reproduce because
the fit path is deterministic, but the formal evidence source is
the Phase A run, not the C-P15 pilot CSV.

### What does NOT change

- No source code, no test, no `experiments/selection_study/`
  edit by this commit.
- No selection criterion, evaluation rule, wrapper algorithm,
  metric primitive, model training budget, Phase B sparsity
  grid, threshold value, or visual / reporting artefact policy.
- `docs/08c`, `docs/08d`, and the other planning artefacts are
  not modified.

### Consequence

- The Phase A `experiments/selection_study/configs/phase_a/*.json`
  config files unblocked in the prior 8b entry can now be
  written in a follow-up commit under the same authorisation.
- Phase B and held-out-evaluation configurations may now be
  written when their respective Phase B / Phase C commits open.
- `Configuration` instances constructed from these JSON files
  carry the frozen pool integers inside `configuration_hash`,
  so a Phase A re-run on the same seeds reproduces the same
  identity tuple.

---

## 20/05/2026 -- Phase A reproduction config files created

### Decision

The two Phase A reproduction configuration files blocked by the
earlier 8b entry are now written:

- `experiments/selection_study/configs/phase_a/dagma_reproduction.json`
- `experiments/selection_study/configs/phase_a/dcdi_reproduction.json`

Both files use `seed_populations = {"reproduction": [101, 102, 103]}`
per the docs/02 v1.7 seed-pool freeze. The Phase A config files
carry the docs/02 v1.6 / v1.7 real-study constants for the
10-node ER2 cell:

- shared: `n_nodes = 10`, `expected_edges = 20`,
  `noise_scale = 1.0`,
  `weight_magnitude_range = [0.5, 2.0]`,
  `n_train = 1000`, `mmd_n_samples = 1000`;
- DAGMA: `condition = "centred_only"`,
  `threshold_robustness_triple = [0.2, 0.3, 0.4]`,
  `dagma_warm_iter = 20000`, `dagma_max_iter = 70000`,
  `dagma_lr = 3e-4`, `dagma_beta_1 = 0.99`,
  `dagma_beta_2 = 0.999`, every DCDI-only field `null`,
  all global RNG seeds `null` (DAGMA fit is
  deterministic by construction per docs/02 v1.5);
- DCDI: `condition = "centred_only"`,
  `threshold_robustness_triple = [0.4, 0.5, 0.6]`,
  `n_val_dcdi = 200`, `dcdi_num_train_iter = 300000`,
  `dcdi_stop_crit_win = 100`, `dcdi_train_patience = 5`,
  `dcdi_train_batch_size = 64`, `dcdi_lr = 1e-3`,
  `dcdi_h_threshold = 1e-8`, `dcdi_hidden_units = 16`,
  `dcdi_hidden_layers = 2`, every DAGMA-only field `null`,
  `seed_torch = seed_numpy = 42` (DCDI fit requires
  matched non-null global RNG seeds per docs/02 v1.5; the
  scalar `42` matches the project's existing fixture
  convention and enters `configuration_hash` so any change
  produces a new run identity).

Both configs use `phase_b_configurations = []`. Configuration
permits an empty tuple here; Phase A is the reproduction pass,
not a Phase B sparsity sweep, so the empty list is the
truthful representation. The Phase A `lambda1` (DAGMA = 0.05)
and `reg_coeff` (DCDI = 0.1) anchors continue to come from
`DAGMAConfig` / `DCDIConfig` defaults, pinned by the two
anchor-default regression tests added in the previous 8b
entry.

Three Phase A-only choices are recorded explicitly so they do
not get reinterpreted as broader protocol decisions:

- **`condition = "centred_only"` is a Phase A reproduction-pass
  default**, not a decision to omit the `"standardised"`
  condition from Phase B or held-out evaluation. The
  selection-study Section 4.4 Criterion-3 sweep over both
  conditions is unchanged and will be reflected in the Phase
  B / held-out config files when those are written.
- **The `do(X_0 = +/- 2)` intervention set is Phase A-only
  reproduction / smoke coverage**, not Criterion-1 held-out
  intervention evidence. Phase A verifies that the pipeline
  runs end to end and that one paper-aligned result is
  approximately reproducible; it does not commit to the full
  intervention grid. The eligible-node intervention policy for
  Phase B and held-out evaluation (which target nodes, how many
  per cell) remains to be frozen before those config files are
  created.
- **`seed_torch = seed_numpy = 42` on the DCDI Phase A config
  is an explicit, hash-participating Phase A value**, not
  accidental fixture inheritance. Changing it produces a new
  `configuration_hash` and therefore a new run identity, which
  is the right discipline for any future amendment.

DAGMA's `seed_torch`, `seed_numpy`, and `seed_dagma` are all
`null` in the DAGMA Phase A config. This is not an omission:
per the `docs/02` v1.5 seed-discipline decision, DAGMA is
verified deterministic-by-construction and does NOT call
`torch.manual_seed`, `np.random.seed`, or
`dagma.utils.set_random_seed`, so the null seed fields
correctly reflect that no global RNG setter was called.

### What does NOT change

- No source code, no test outside `tests/test_real_study.py`,
  no wrapper, no metric, no notebook, no configuration outside
  the two new JSON files, no results-directory, and no
  dependency-manifest edit. `pipeline.py` is unchanged from the
  prior Commit 8a state.
- No selection criterion, evaluation rule, wrapper algorithm,
  metric primitive, or `Configuration` schema change.
- `docs/02_base_model_selection.md`, `docs/08c`, and `docs/08d`
  are not edited.
- The C-P15 DCDI training-budget pilot CSV remains pilot-only
  diagnostic evidence and does NOT count as Phase A evidence,
  per the prior `docs/03` entries and `docs/08d`.

### Consequence

- `experiments.selection_study.config.load_config` accepts both
  files and produces `Configuration` instances that satisfy
  `assert_real_study_constants(config, stage="phase_a")`.
- `experiments.selection_study.preflight.enumerate_manifest`
  followed by `validate_manifest` succeeds on both files
  without importing any wrapper, `dagma`, `dcdi`, or `wandb`
  module, and without creating any run directory on disk.
- Both DAGMA and DCDI manifests enumerate exactly three
  reproduction entries.
- The Phase A runner (`experiments/selection_study/phase_a.py`)
  remains a stub. Its implementation is the next step (8c) and
  will consume these two config files end-to-end.
- Full pytest suite remains green: 774 passed, 2 pre-existing
  RuntimeWarnings unrelated to this work (count up from 758,
  +16 new Phase A config-file tests in
  `tests/test_real_study.py`).

---

21/05/2026 — Phase A reproduction-pass runner implemented

### What changes

- `experiments/selection_study/phase_a.py` now implements
  `run_phase_a(config_path, *, output_root=None) -> PhaseASummary`.
  The runner loads the Phase A configuration via
  `load_config`, validates it with
  `assert_real_study_constants(config, stage="phase_a")`,
  enumerates the preflight manifest with `enumerate_manifest`,
  validates the manifest with `validate_manifest`, iterates over
  the `reproduction`-population entries through `run_single_fit`,
  invokes `recompute_at_thresholds` against each completed run
  directory, and writes a Phase A summary JSON.
- `experiments/selection_study/run.py` gains a `--phase` flag
  (currently only `phase_a` is accepted) and a `--output-root`
  flag. `--phase phase_a --config PATH` dispatches to
  `run_phase_a`; `--config` alone without `--phase` continues to
  raise `NotImplementedError` to preserve the pre-existing CLI
  surface.
- `tests/test_phase_a_runner.py` is a new test module. The
  pipeline and threshold-robustness functions are patched on the
  `phase_a` module so the runner's orchestration is exercised
  without invoking DAGMA or DCDI code. Coverage includes happy
  paths for both DAGMA and DCDI configs, summary-path layout,
  end-to-end summary-field presence, schema-gate failure handling,
  non-schema-gate exception propagation, real-study guard
  rejection, missing-config-file rejection, and CLI dispatch via
  `experiments.selection_study.run`.
- `tests/test_selection_runner_scaffolding.py` no longer asserts
  that `phase_a.run_phase_a` raises `NotImplementedError`; it is
  now exercised under its own test module and the attribute is
  only touched for import-stability.

### Why this matters

- Closes the gap left by Commit 8a (Configuration consumption) and
  Commit 8b (Phase A config files + real-study guard): the runner
  can now ingest those configs end to end.
- Phase A is a reproduction-pass. It demonstrates that the
  schema-conformance pipeline runs cleanly on the paper-aligned
  reference cell under the protocol guard, with sibling
  threshold-robustness records produced offline. It does not, by
  itself, constitute base-model selection evidence; Phase B and
  held-out evaluation remain unimplemented.
- The summary JSON is the canonical Phase A artefact. It lives at
  `<output_root>/phase_a_summary/<configuration_hash>/phase_a_summary.json`
  and carries schema_version, config_path, model, condition,
  configuration_hash, seed_population, seed_values, run_ids,
  completed_run_count, failed_run_count, counts by graph_status /
  sampler_status / training_status, SHD / SID / MMD aggregates,
  threshold-robustness availability count, per-entry records, a
  `phase_a_status` of `passed` / `completed_with_warnings` /
  `failed_mechanical_gate`, and a `note` field that flags the
  reproduction-only scope.

### What does NOT change

- No new dependency. No edit to `docs/02`. No edit to any
  wrapper, metric primitive, evaluator, or sampling layer. No
  Phase B, no held-out evaluation, no C-P11 rerun, no
  prior-loss / loss-hook work.
- `Configuration` and `load_config` are unchanged.
- `assert_real_study_constants`, `enumerate_manifest`,
  `validate_manifest`, `run_single_fit`, and
  `recompute_at_thresholds` are consumed verbatim; their
  behaviour and contracts are not amended.
- The on-disk Phase A configuration files are unchanged.
- DCDI Commit 11 (loss-hook injection) and the DCDI Commit 10
  pause both remain in force; this commit does not resume them.

### Per-entry failure policy

- The pipeline's declared schema-gate stop conditions
  (`SchemaGateError` and its subclasses
  `InvalidGraphForSchemaGateError` and `DcdiSeedMismatchError`)
  are recorded as failed per-entry records rather than crashing
  the runner. The summary's `phase_a_status` flips from `passed`
  to `completed_with_warnings` whenever any reproduction entry
  fails this way.
- Any other exception propagates unhandled. No broad exception
  swallowing is performed; programmer errors and unanticipated
  runtime errors surface to the caller.
- A failure of `load_config`, `assert_real_study_constants`, or
  `validate_manifest` is a pre-iteration mechanical-gate failure.
  The current runner propagates the original exception and does
  not write a partial summary, on the reasoning that preserving
  the original exception is safer than producing a half-built
  artefact that hides the cause. `failed_mechanical_gate` is
  therefore declared as a reserved Phase A summary status for
  future explicit summary-recording of such pre-iteration
  failures; it is not produced by the current runner.

### `phase_a_status` derivation

The status is computed from per-entry records as follows:

- `"passed"` requires all of:
  - `failed_run_count == 0`;
  - every completed record has `graph_status == "valid_dag"`;
  - every completed record has `sampler_status == "available"`;
  - every completed record has `threshold_robustness_available is True`.
- `"completed_with_warnings"` is set when any of:
  - at least one `SchemaGateError`-derived per-entry failure was
    recorded;
  - at least one completed record has
    `graph_status != "valid_dag"`;
  - at least one completed record has
    `sampler_status != "available"`;
  - at least one completed record has
    `threshold_robustness_available is False`.
- `"failed_mechanical_gate"` is reserved (see above) and is not
  produced by current behaviour.

This makes `"passed"` a strict end-to-end health signal: every
reproduction entry not only completed schema-conformance but also
produced a valid DAG, a usable sampler path, and a sibling
threshold-robustness artefact.

### Consequence

- `python -m experiments.selection_study.run --phase reproduction_pass
  --config experiments/selection_study/configs/reproduction/dagma_reproduction.json`
  (and the DCDI counterpart) now drives the reproduction pass
  end to end. No filesystem path or wrapper module is hardcoded
  beyond the run-storage default `results/model_selection/`,
  which can be overridden with `--output-root PATH`. Note: the
  CLI value `--phase reproduction_pass` and the
  `configs/reproduction/` directory reflect the rename recorded
  in the follow-up entry "Implementation stage names made
  semantic before reproduction execution". The implementation
  module path and CLI value listed here are the post-rename
  values; the body of the present entry still describes the
  implementation introduced at this commit.
- The reproduction-pass runner does not commit generated run
  outputs; the caller is responsible for keeping those out of
  version control.
- Calibration and held-out evaluation remain stubs and are
  deferred to a later commit; their plans are unchanged.

---

21/05/2026 - Implementation stage names made semantic before reproduction execution

### What changes

- The selection-study protocol documents continue to refer to the
  two implemented stages as "Phase A: reproduction pass" and
  "Phase B: calibration". The implementation code now uses the
  semantic names `reproduction_pass` and `calibration` so module
  paths, the CLI flag value, test names, output directories, and
  summary field names are self-describing on first read.
- File and directory renames (via `git mv` to preserve history):
  - `experiments/selection_study/phase_a.py` ->
    `experiments/selection_study/reproduction_pass.py`
  - `experiments/selection_study/phase_b.py` ->
    `experiments/selection_study/calibration.py`
  - `experiments/selection_study/configs/phase_a/` ->
    `experiments/selection_study/configs/reproduction/`
  - `tests/test_phase_a_runner.py` ->
    `tests/test_reproduction_pass_runner.py`
- Public identifier renames:
  - `run_phase_a` -> `run_reproduction_pass`
  - `PhaseASummary` -> `ReproductionPassSummary`
  - `PhaseARunRecord` -> `ReproductionPassRunRecord`
  - `run_phase_b` -> `run_calibration`
  - `calibration_ranking` keeps its name.
- Summary / output identifier renames:
  - `phase_a_status` -> `reproduction_pass_status`
  - `phase_a_summary/` -> `reproduction_pass_summary/`
  - `phase_a_summary.json` -> `reproduction_pass_summary.json`
- CLI change:
  - The `--phase` flag is preserved.
  - The implemented value changes from `--phase phase_a` to
    `--phase reproduction_pass`. The old `--phase phase_a` is
    not accepted as an alias; argparse rejects it.

### Why this matters

- No real reproduction-pass artefacts had been generated before
  this rename. The prior Commit 8c implementation existed, was
  unit-tested with mocked pipeline fakes, but no real DAGMA/DCDI
  reproduction artefacts were committed or generated. Renaming
  now therefore does not invalidate any on-disk evidence.
- "Phase A" / "Phase B" are useful protocol shorthand; they are
  poor implementation names because a future reader has to map
  the letter back to a stage purpose. Renaming the
  implementation makes diffs, test failures, and CLI invocations
  self-describing without changing the protocol vocabulary.

### What does NOT change

- No selection-study constant, threshold value, seed pool,
  intervention value, model training budget, metric primitive,
  metric call, wrapper call, `run_single_fit` behaviour, or
  output-record semantics is changed.
- The Configuration dataclass field `phase_b_configurations` and
  the dataclass `PhaseBConfiguration` (both in
  `experiments/selection_study/config.py`) keep their names.
  They are configuration-schema identifiers, not runner
  identifiers, and `config.py` is in the do-not-edit set for
  this rename.
- The real-study guard's stage label remains the literal string
  `"phase_a"`. `experiments/selection_study/real_study.py` is in
  the do-not-edit set for this rename; `run_reproduction_pass`
  continues to call
  `assert_real_study_constants(config, stage="phase_a")` via the
  module-level constant `_REAL_STUDY_STAGE_LABEL`.
- The on-disk JSON configuration values
  (`dagma_reproduction.json`, `dcdi_reproduction.json`) are
  unchanged after the directory rename. Only their containing
  directory moved.
- No source code under `src/symbolic_priors_cd/`, no metric
  primitive, no wrapper, no notebook, and no generated result
  was edited.

### Behaviour-preservation verification

- `reproduction_pass.py` is behaviour-equivalent to the previous
  `phase_a.py` except for module name, summary directory name,
  summary file name, public class and function names, and the
  summary field rename `phase_a_status` ->
  `reproduction_pass_status`. The run-record schema, manifest
  validation, real-study guard call, threshold-robustness call,
  per-entry failure policy, and `reproduction_pass_status`
  derivation rule are unchanged.
- `calibration.py` retains the same stub behaviour previously
  held by `phase_b.py`: every callable raises
  `NotImplementedError` with an explicit module-qualified
  message; `calibration_ranking` keeps its name.
- The CLI dispatcher in `run.py` is unchanged except for the
  `--phase` choice value, the lazily-imported runner symbol, and
  the log line that names the dispatched stage.

### Tests

- `tests/test_reproduction_pass_runner.py` is the renamed test
  module. All test names, imports, monkeypatch targets, summary
  field assertions, and CLI assertions are updated to the new
  names. A new test
  `test_cli_rejects_legacy_phase_a_value` pins that the old
  `--phase phase_a` value is rejected by argparse, so the rename
  is not silently aliased back.
- `tests/test_selection_runner_scaffolding.py` imports
  `reproduction_pass` and `calibration` in place of `phase_a`
  and `phase_b`. The remaining-stub list now points at
  `calibration.run_calibration` and
  `calibration.calibration_ranking`.
- `tests/test_real_study.py` updates the on-disk config path
  variable, helper kwargs builders, and reproduction-related
  test names to the semantic vocabulary. The
  `assert_real_study_constants(..., stage="phase_a")` calls and
  the Configuration field references (`phase_b_configurations`,
  `PhaseBConfiguration`) are preserved verbatim, as they belong
  to do-not-edit modules.

### Search verification

A repository-wide search for `phase_a`, `PhaseA`, `phase_b`,
`PhaseB`, `phase_a_status`, `phase_a_summary`, `run_phase_a`,
`run_phase_b`, `PhaseASummary`, `PhaseARunRecord`,
`configs/phase_a`, `test_phase_a_runner`, and `--phase phase_a`
shows remaining hits only in three categories that are
deliberately retained:

1. `stage="phase_a"` calls (protocol stage label consumed by the
   real-study guard).
2. `PhaseBConfiguration` / `phase_b_configurations` references
   (Configuration dataclass / field names defined in
   `experiments/selection_study/config.py`).
3. Methodological prose in docstrings, comments, and earlier
   `docs/03` entries that refer to the protocol stages by their
   Phase A / Phase B names. These are historical or
   methodological references, not current implementation
   references.

Any remaining occurrence falls into one of those categories.

---

21/05/2026 - Semantic stage-name refactor completed across guard and configuration schema

### What changes

- The real-study guard's accepted stage label is renamed from
  `"phase_a"` to `"reproduction_pass"`. The constant
  `_VALID_STAGES` in
  `experiments/selection_study/real_study.py` now reads
  `("reproduction_pass",)` and the `assert_real_study_constants`
  docstring is updated accordingly.
- The reproduction-pass runner's module-level constant
  `_REAL_STUDY_STAGE_LABEL` in
  `experiments/selection_study/reproduction_pass.py` is set to
  `"reproduction_pass"`, removing the prior asymmetry between
  implementation module name and guard stage label.
- The Configuration calibration-grid schema is renamed:
  - dataclass `PhaseBConfiguration` -> `CalibrationConfiguration`;
  - field `Configuration.phase_b_configurations` ->
    `Configuration.calibration_configurations`;
  - the canonical-JSON key `"phase_b_configurations"` ->
    `"calibration_configurations"`;
  - the parsed-JSON local variable and `_REQUIRED_FIELDS` entry
    follow the same rename.
- The reproduction config files
  (`configs/reproduction/dagma_reproduction.json`,
  `configs/reproduction/dcdi_reproduction.json`) now carry the
  key `"calibration_configurations": []` in place of
  `"phase_b_configurations": []`. The list value is unchanged.
- `docs/08c_real_run_constants_and_training_budget_audit.md`
  Section table cell referencing
  `PhaseBConfiguration.hyperparameters` is updated to
  `CalibrationConfiguration.hyperparameters`. No other current
  docs reference required updating; `docs/02`, `docs/08`, and
  `docs/08a` did not contain stale schema-name references.

### Why this matters

- No real reproduction-pass artefacts existed before this
  refactor, so the `configuration_hash` rotation caused by
  renaming the canonical-JSON key
  `phase_b_configurations` -> `calibration_configurations` does
  not invalidate any on-disk evidence. Doing the rename now,
  rather than after real artefacts exist, is the right
  ordering.
- The prior semantic-rename commit deferred these last two
  identifier groups because `real_study.py` and `config.py`
  were do-not-edit at that point. With this commit those
  groups become consistent with the rest of the implementation:
  `reproduction_pass` is the single semantic name for the
  reproduction-pass stage everywhere in code (module name,
  guard stage label, CLI value, summary field), and
  `calibration` / `CalibrationConfiguration` is the single
  semantic name for the calibration grid (module name, schema
  class, schema field, canonical JSON key).

### Test updates

- `tests/test_real_study.py`: every
  `assert_real_study_constants(config, stage="phase_a")` call
  is updated to `stage="reproduction_pass"`. The
  unknown-stage rejection test that previously probed with
  `stage="phase_b"` now probes with `stage="unknown_stage"`.
  Imports of `PhaseBConfiguration` and uses of
  `phase_b_configurations` are renamed.
- `tests/test_config_schema.py`,
  `tests/test_preflight.py`, `tests/test_pipeline.py`,
  `tests/test_threshold_robustness.py`: schema-identifier
  imports and field usages renamed to
  `CalibrationConfiguration` and
  `calibration_configurations`. The private fixture constants
  `_PHASE_B`, `_PHASE_B_CFG` are renamed to `_CALIBRATION_CFG`.
- `tests/test_reproduction_pass_runner.py`: the prior
  `test_cli_rejects_legacy_phase_a_value` is replaced by
  `test_cli_rejects_unsupported_phase_value`, which probes
  rejection of a generic unsupported phase value
  (`"not_a_stage"`) rather than the stale `"phase_a"`
  literal. The neighbouring `test_cli_rejects_unknown_phase`
  uses `"unknown_stage"` in place of the previous
  `"calibration"`.

### What does NOT change

- No scientific constants changed. `_SHARED_REQUIRED_VALUES`,
  `_DAGMA_REQUIRED_VALUES`, `_DCDI_REQUIRED_VALUES`, every
  threshold, every seed pool, every intervention value, every
  model training budget, and every metric / wrapper / pipeline
  call is byte-for-byte unchanged.
- The reproduction-pass runner's control flow is unchanged
  except that `_REAL_STUDY_STAGE_LABEL` now equals
  `"reproduction_pass"` rather than `"phase_a"`. The guard
  enforces the same value-equality rules on the same fields;
  only the accepted stage label string changed.
- The calibration runner remains a stub with the same
  `NotImplementedError` behaviour previously held by the
  `phase_b` module.
- Reproduction config JSON files contain identical values
  (every numeric and boolean field is unchanged); only the
  key `phase_b_configurations` was renamed to
  `calibration_configurations`, with an empty list as before.

### Behaviour-preservation verification

- Full pytest suite: 800 passed, 2 pre-existing
  `RuntimeWarning`s from
  `tests/test_dagma_wrapper_residuals.py::test_non_finite_sigma_sets_unavailable_unresolved_noise_policy`,
  unrelated to this refactor.
- `configuration_hash` will differ for any Configuration
  serialised under the prior schema because the canonical
  key `phase_b_configurations` has been renamed to
  `calibration_configurations`. This rotation is expected and
  is the sole cause of any hash change. No real artefacts
  existed under the prior hash, so no resume / re-use logic
  is invalidated.
- The repository-wide search for `phase_a`, `phase_b`,
  `PhaseA`, `PhaseB`, `run_phase_a`, `run_phase_b`,
  `PhaseASummary`, `PhaseARunRecord`, `PhaseBConfiguration`,
  `phase_b_configurations`, `phase_a_status`,
  `phase_a_summary`, `configs/phase_a`,
  `test_phase_a_runner`, and `--phase phase_a` returns zero
  hits in `experiments/selection_study/`, `tests/`, and the
  reproduction config JSONs after this commit. Remaining hits
  in `docs/` are confined to: (a) the present and prior
  `docs/03` historical entries describing earlier commits,
  and (b) protocol / methodological prose in selection-study
  protocol docs. Both categories are explicitly allowed by
  the project's docs policy.

---

21/05/2026 - Path B reproduction-pass clarification (docs-only) before any reproduction-pass execution

### Decision

Path B is adopted as the active reproduction-pass interpretation for the selection study. Path B is recorded in `docs/02` v1.8 (new Section 12). Under Path B:

- the current `configs/reproduction/dagma_reproduction.json` and `configs/reproduction/dcdi_reproduction.json` carry the 10-node ER2 thesis selection cell and serve as thesis-cell compatibility / runner-sanity configs, not as strict paper-reproduction configs;
- the reproduction pass currently verifies end-to-end runner correctness, schema and artefact generation, compatibility with the 10-node ER2 thesis selection cell, and the project graph / sampler / training / threshold-robustness / metric availability taxonomy;
- the reproduction pass does NOT strictly reproduce any DAGMA or DCDI paper result, and the selection-study report must not claim that it does;
- `docs/02` Section 5 disqualification item 2 ("within 20% paper reproduction") applies only when a direct or explicitly frozen closely aligned paper target exists for the candidate before results are observed; where no such target exists, paper-reproduction comparison is reported as **not directly evaluable**, not as **passed**, and disqualification item 2 does not fire;
- a future strict paper-DGP reproduction sub-study remains possible but is deferred. Opening it would require a separate `docs/02` amendment, a contemporaneous `docs/03` entry, a separate per-model reproduction configuration, a real-study guard amendment, and possibly a Configuration-schema extension, per `docs/02` Section 12.6.

### What this entry records

- **Local verification finding (no source/config edits).** Before any reproduction-pass artefact was generated, the verification report dated 21/05/2026 inspected the DAGMA and DCDI papers, the current real-study guard in `experiments/selection_study/real_study.py`, the runner in `experiments/selection_study/reproduction_pass.py`, the two reproduction config JSON files, and the test surface in `tests/`. The verification found that the configs are 10-node ER2 thesis-cell configs, not strict paper-reproduction configs. No `results/` directory exists on disk; `results/model_selection/` does not exist; no Phase A / reproduction-pass run has occurred. NotebookLM was used as a cross-check; local verification against `papers/DAGMA.pdf`, `papers/DCDI.pdf`, and the repository remains the operational source of truth.
- **DAGMA paper-target finding.** The DAGMA paper (`papers/DAGMA.pdf`) contains no empirical recovery benchmark at d = 10. The closest small linear-Gaussian recovery evidence is Table 1 on page 21 (within Appendix C.1.1 "Small to Moderate Number of Nodes"), which reports SHD and runtime averaged across ER4 / SF4 graphs and Gaussian / Exponential / Gumbel noise at d in {20, 30, 50, 80, 100}; the DAGMA row at d = 20 is `SHD = 6.78 +/- 1.64` and `Runtime = 6.54 +/- 0.42 seconds`. The DAGMA paper's Figure 1 (page 6) is a 2-node toy illustration of the log-determinant acyclicity characterization and is not a recovery benchmark. The DAGMA paper's Figure 2 (page 7) plots the numerical decay of `h_expm` / `h_poly` on a single cycle graph as a function of `d`, mentioning that at `d = 13` `h_expm` is approximately `10^-9`; this is the "Argument (i)" numerical-cycle-detection illustration of gradient vanishing in alternative acyclicity functions, not a recovery benchmark. No DAGMA appendix table or figure (Tables 1, 2; Figures 3, 4, 5, 6, 8, 9, 10, 11, 12) extends down to `d = 10`.
- **DCDI paper-target finding.** The DCDI paper (`papers/DCDI.pdf`) Table 7 in Appendix C.4.1 "Perfect interventions" (page 38) reports DCD-no-interv at 10 nodes with `e = 1` (`SHD = 8.9 +/- 2.8`, `SID = 19.5 +/- 10.9`) and `e = 4` (`SHD = 26.7 +/- 5.9`, `SID = 69.0 +/- 11.4`); there is no `e = 2` row at 10 nodes in Table 7 or in any subsequent C.4 / C.5 / C.7 table. Appendix C.7 ("Comprehensive results of the main experiments") redisplays the DCDI-G / DCDI-DSF rows from the main experiments against IGSP / GIES / CAM baselines (Table 22 page 42) and does not reintroduce a separate DCD-no-interv row. The DCDI synthetic data generator (Appendix B.1, page 26) differs from the project data generator on weight magnitude range (paper `Uniform([-1, -0.25] union [0.25, 1])`, project `Uniform([-2, -0.5] union [0.5, 2])`), on the noise multiplier (paper `0.4 * N_j`, project no multiplier), on the per-node noise variance distribution (paper `sigma_j^2 ~ U[1, 2]`, project fixed unit variance), and on standardisation (paper mean-subtracted and divided by std, project `condition = "centred_only"`). The project's thesis cell density `expected_edges = 20` (ER2) does not correspond to either DCDI Table 7 published density at d = 10.
- **No artefact dependency.** `results/model_selection/` does not exist on disk; no reproduction-pass `run.json`, `reproduction_pass_summary.json`, or `threshold_robustness.json` has been generated. The Path B decision therefore does not invalidate any on-disk evidence, and the deferred future paper-DGP sub-study does not rotate any pre-existing `configuration_hash`.

### What does NOT change under v1.8

- No source file under `experiments/`, no module under `src/`, no test under `tests/`, no configuration JSON under `experiments/selection_study/configs/`, and no result under `results/` is modified by Path B. The 800-test pytest baseline is untouched.
- The real-study guard in `experiments/selection_study/real_study.py` is unchanged: it continues to require `n_nodes = 10`, `expected_edges = 20`, `noise_scale = 1.0`, `weight_magnitude_range = (0.5, 2.0)`, `n_train = 1000`, `mmd_n_samples = 1000` shared across DAGMA and DCDI reproduction configs, plus the per-model tactical constants.
- The configuration_hash of the current DAGMA and DCDI reproduction configs is unchanged because no field, no canonical-JSON key, and no canonical-JSON value is modified by Path B.
- The implementation identifier `reproduction_pass` is unchanged. Protocol-letter labels "Phase A" / "Phase B" continue to be acceptable in protocol prose; implementation prose continues to use the semantic names.
- The C-P11 real-budget reapplication policy on a 10-node ER2 fixture is unchanged.
- The C-P15 DCDI training-budget pilot continues to be pilot-only diagnostic evidence and continues NOT to count as reproduction-pass evidence, per `docs/08d` (whose paper-aligned-cell wording is now superseded by `docs/02` v1.8 Path B per the supersession note added to `docs/08d`).

### Retention reason

The current configs are retained, not amended, under Path B because they correctly enforce thesis-cell compatibility for Phase B calibration and held-out evaluation. Replacing them with paper-aligned configs is a separate decision that requires the design-package listed in `docs/02` Section 12.6 (config addition, guard amendment, possibly schema extension); doing it as part of Path B would conflate "verify the runner on the thesis cell" with "perform paper reproduction", which is exactly the conflation Path B exists to remove.

### Follow-up code patch flagged but not applied

The `reproduction_pass.py` summary `note` string (`_NOTE_REPRODUCTION_ONLY` at [experiments/selection_study/reproduction_pass.py:63-69](experiments/selection_study/reproduction_pass.py#L63-L69)) currently reads in part "the runner completed end to end on the paper-aligned reference cell". Under Path B this wording is now technically inaccurate because the runner completes end to end on the thesis-cell compatibility configs, not on a paper-aligned reference cell. The wording is left in place in this docs-only commit. A follow-up code patch is recommended that rewrites the summary note to read along the lines of "the runner completed end to end on the thesis-cell compatibility / runner-sanity configs (see `docs/02` Section 12)". The patch is out of scope for this commit per the docs-only constraint.

---

21/05/2026 - Reproduction-pass summary `note` field removed (Path B follow-up)

### Decision

The `note` prose field on the `ReproductionPassSummary` dataclass and on the on-disk reproduction-pass summary JSON has been removed. The `_NOTE_REPRODUCTION_ONLY` constant in `experiments/selection_study/reproduction_pass.py` is deleted. The runner no longer attaches a prose summary note to the reproduction-pass artefact. The supersedes-the-prior-flagged-follow-up entry immediately above resolves the docs/02 v1.8 Path B follow-up.

### Why

Local inspection found that the `note` field was non-load-bearing:

- the reproduction-pass summary artefact is not part of the published `docs/08a_experiment_tracking_and_results_schema.md` schema (08a defines the per-run `run.json` schema, not the reproduction-pass summary);
- no loader, no report module, no notebook, and no test outside `tests/test_reproduction_pass_runner.py` reads `summary.note` or the JSON `"note"` key;
- the only test references are three localised exercises of field-existence (prefix assertion, required-fields-set membership, fake-summary constructor kwarg), not content-binding assertions.

The previous prose ("the runner completed end to end on the paper-aligned reference cell ...") was both stale under Path B and a duplication of the tracked authoritative explanation in `docs/02` Section 12. Removing the field rather than replacing it with another prose note avoids re-introducing tracked-protocol prose into runtime artefacts.

### Scope

- **Source edit:** `experiments/selection_study/reproduction_pass.py` — deleted `_NOTE_REPRODUCTION_ONLY`; dropped the `note: str` dataclass field from `ReproductionPassSummary`; dropped the `"note": summary.note` key from `_summary_to_dict`; dropped the `note=...` kwarg from the `_assemble_summary` construction site; removed the corresponding Attributes-section line from the dataclass docstring; rewrote a single sentence of the module docstring that previously said "paper-aligned reference cell" to remove the stale Path-B-incorrect wording.
- **Test edits:** `tests/test_reproduction_pass_runner.py` — dropped the `summary.note.startswith(...)` assertion from `test_run_reproduction_pass_dagma_completes_with_passed_status`; dropped `"note"` from the required-fields set in `test_run_reproduction_pass_writes_summary_with_expected_top_level_fields`; dropped the `note="stub"` kwarg from the fake `ReproductionPassSummary` constructed inside `test_cli_reproduction_pass_invokes_runner`.
- **No changes** to `docs/02`, `docs/08*`, `src/`, `experiments/selection_study/configs/`, `results/`, `papers/`, `pyproject.toml`, or `requirements-lock.txt`.

### Behaviour preservation

- No selection-study constant, threshold, seed-pool integer, intervention value, training budget, metric primitive, metric call, wrapper call, real-study guard rule, configuration_hash field, or run-record schema field changed.
- The reproduction-pass summary artefact retains every load-bearing field: `schema_version`, `config_path`, `model`, `condition`, `configuration_hash`, `seed_population`, `seed_values`, `run_ids`, `completed_run_count`, `failed_run_count`, `graph_status_counts`, `sampler_status_counts`, `training_status_counts`, `shd_values`, `sid_values`, `mmd_primary_values`, `threshold_robustness_available_count`, `records`, `reproduction_pass_status`, `output_root`, `summary_path`. (The dataclass now defines 21 fields; the previous 22-field shape was the same set plus `note`.)
- Status derivation rule (`reproduction_pass_status ∈ {"passed", "completed_with_warnings", "failed_mechanical_gate"}`) is unchanged.
- The full test suite (800 tests) is green; targeted run `pytest tests/test_reproduction_pass_runner.py -v` reports 18 passed in 0.60 s.

### Why this aligns with Path B

The runtime artefact no longer carries protocol-level interpretive prose. Path B places the authoritative reading of the reproduction-pass scope inside `docs/02` Section 12 (the tracked source of truth). The summary artefact now records only structural and metric facts; the interpretation that those facts are thesis-cell compatibility evidence rather than strict paper reproduction lives where it belongs, in `docs/02` Section 12 and this `docs/03` log.

---

21/05/2026 - Reproduction-pass summary directory leaf unified with per-run 12-char prefix convention

### Decision

The reproduction-pass summary directory leaf now uses the first 12 characters of the `configuration_hash` rather than the full 64-character digest. The change is implemented in `experiments/selection_study/reproduction_pass.py` by importing `_HASH_PREFIX_LENGTH` from `experiments/selection_study/identity.py` and slicing `manifest.configuration_hash[:_HASH_PREFIX_LENGTH]` when constructing `summary_dir`. The summary directory leaf is therefore symmetric with the per-run directory leaf already documented in `docs/08a` Section 3 and produced by `identity.derive_run_directory`.

### Previous state

- Per-run directories used `<configuration_hash_prefix>` (the first 12 characters), via `identity.derive_run_directory`. Frozen in `docs/08a` Section 3 and Section 4.
- The manifest sidecar JSON file under `<manifest_dir>/manifest_<prefix>.json` used the first 12 characters, via `preflight.save_manifest`.
- The reproduction-pass summary directory leaf used the **full** 64-character `configuration_hash`, via `reproduction_pass._assemble_summary`. The full-hash choice was documented only in a source comment inside `reproduction_pass.py`; it was never frozen in `docs/08a` or in `docs/03`. The asymmetry was identified during the read-only path-convention inspection earlier on 21/05/2026.

### New state

- The reproduction-pass summary directory leaf uses the first 12 characters of `configuration_hash`, matching the per-run directory leaf and the manifest sidecar file name. The leaf-length constant `_HASH_PREFIX_LENGTH = 12` defined in `experiments/selection_study/identity.py` is the single source of truth; `reproduction_pass.py` imports it rather than hard-coding `12` again.
- The `configuration_hash` field value inside `reproduction_pass_summary.json`, inside every per-run `run.json`, and inside every sibling `threshold_robustness.json` remains the **full** 64-character lowercase hex SHA-256 digest. The directory-leaf shortening does not propagate into any content field; the digest itself stays full-length everywhere it is recorded as a value.
- The directory name `reproduction_pass_summary/` and the filename `reproduction_pass_summary.json` are unchanged.

### What does NOT change

- No selection criterion, evaluation rule, metric primitive, wrapper, runner control flow, real-study guard, schema field, seed pool integer, threshold triple, training budget, intervention value, or `configuration_hash` derivation algorithm changed.
- `experiments/selection_study/identity.py`, `pipeline.py`, `preflight.py`, `config.py`, `run.py`, `threshold_robustness.py`, the reproduction config JSONs, the wrappers under `src/`, the metric primitives, `papers/`, `pyproject.toml`, `requirements-lock.txt`, `docs/02`, `docs/08*` were not modified.
- The per-run directory leaf, the `run_id` string format, the manifest sidecar file name, and the `configuration_hash_prefix` helper are unchanged. They already used 12 characters and continue to.

### Migration

No artefacts exist under the old convention. `results/model_selection/` does not exist on disk; no reproduction-pass `reproduction_pass_summary.json`, no per-run `run.json`, and no sibling `threshold_robustness.json` has been generated. The directory-leaf change is therefore a forward-only convention update with no migration step.

### Why `docs/08a` was not amended

`docs/08a` Section 3 specifies the 12-character prefix convention for per-run directories. The reproduction-pass summary directory was not previously frozen in `docs/08a`. This entry treats the present change as bringing the summary directory into line with the existing Section 3 convention rather than as introducing a new convention. If a future commit adds a Section 3.x subsection describing the summary directory layout explicitly, that would be a separate documentation amendment; it is out of scope here.

### Test coverage

- `tests/test_reproduction_pass_runner.py::test_run_reproduction_pass_writes_summary_at_canonical_path` updated to expect the 12-character prefix on the summary directory leaf. The literal `12` is used in the test directly (rather than importing `_HASH_PREFIX_LENGTH`) so the test is independent of the source-side constant; the constant itself is pinned by separate regressions in `tests/test_run_identity.py` and `tests/test_config_schema.py`.
- New regression `tests/test_reproduction_pass_runner.py::test_run_reproduction_pass_summary_field_carries_full_hash` pins that `summary.configuration_hash` and the JSON `configuration_hash` key both retain the full 64-character lowercase hex digest, so the directory-leaf shortening cannot silently propagate into the content field in a future refactor.

---

22/05/2026 - Reproduction-pass stage mechanically closed under Path B

### Decision

The reproduction_pass stage is **mechanically closed** under docs/02 v1.8 Path B. Both DAGMA-linear and DCDI-G completed `--phase reproduction_pass` end-to-end on the 10-node ER2 selection cell and produced the full artefact set required by the project schema. Both candidates report `reproduction_pass_status = "passed"`. No model-selection decision is made or implied by this closure. The factual readout lives in `docs/08e_reproduction_pass_readout.md`; this `docs/03` entry records the closure.

### Configuration hashes

- DAGMA reproduction config: `15328a8f730f3bfc864ccf45f1aea38fbec2bc81dac8ff76485497ee2d676537` (directory leaf `15328a8f730f`).
- DCDI reproduction config: `826de9ce39d70f2ca2416523bf1526470b0f07734001ac05dbd2de00fb55ae0a` (directory leaf `826de9ce39d7`).

### Artefact-convention compliance

- Artefacts were generated under docs/02 v1.8 Path B.
- Per-run directories and the reproduction-pass summary directory both use the 12-character `configuration_hash` prefix as the directory leaf (the unified convention from the 21/05/2026 path-unification entry above). The full 64-character digest is retained as a content field in summary, per-run `run.json`, and sibling `threshold_robustness.json`.
- `ReproductionPassSummary` carries no `note` prose field; only 21 factual fields per the 21/05/2026 note-removal entry.
- No `phase_a` / `phase_b` implementation path appears in any artefact.

### Headline counts

For each candidate: 3 reproduction-pool entries; `completed_run_count = 3`, `failed_run_count = 0`; `graph_status_counts = {"valid_dag": 3}`; `sampler_status_counts = {"available": 3}`; `training_status_counts = {"converged": 3}`; `threshold_robustness_available_count = 3`.

### Section 5 disqualification interpretation

- Items 1, 3, 4: not triggered for either candidate.
- Item 2: not directly evaluable under Path B for the current configs; reported as "not directly evaluable", not "passed".
- Item 5: out of scope at this stage.

### Diagnostic warnings carried forward (not selection signals)

- **DCDI has substantially higher SHD/SID/MMD than DAGMA at anchor `reg_coeff = 0.1`**: DAGMA SHD `[2, 0, 0]` / SID `[6, 0, 0]` / mmd_primary `[0.0041, 0.0045, 0.0030]`; DCDI SHD `[26, 25, 31]` / SID `[65, 41, 65]` / mmd_primary `[0.0770, 0.0111, 0.1068]`. DCDI/DAGMA MMD ratios `[19.0x, 2.5x, 35.6x]` across seeds 0/1/2. The comparison is at the reproduction anchor only (DAGMA `lambda1 = 0.05`, DCDI `reg_coeff = 0.1`) and is not informative about either candidate's behaviour at calibration-selected sparsity values.
- **DCDI seed 2 is a coherent outlier**: sparse graph (thresholded-adjacency edge count `10` against the ER2 cell's `expected_edges = 20`), highest MMD primary, upper-end validation-NLL last-3-value mean (~2.27), and largest per-intervention median-heuristic bandwidths (~514-540). The four signals are mutually consistent on the same seed. Recorded as a diagnostic pattern only.
- **DCDI MMD gap does not by itself resolve the C-P11 question.** The C-P11 real-budget reapplication probe on a fresh 10-node ER2 fixture remains required before any held-out interpretation of DCDI sampler quality, per docs/02 Section 7.
- **DCDI validation-NLL trajectory is non-monotonic** (lower values earlier, later increases as the augmented-Lagrangian acyclicity pressure dominates), consistent with the C-P15 pilot shape documented in `docs/08d_dcdi_training_budget_pilot.md`.

### What this closure does not establish

- No base-model selection.
- No statement about paper reproduction for either candidate.
- No claim that DCDI is rejected; Phase B calibration may produce a selected `reg_coeff` that improves on the anchor.
- No H1-H4 hypothesis evidence; the DCDI seed-2 outlier pattern is consistent with the **type** of cross-seed instability H4 will later test, not evidence for H4.

### Next step

Pre-Commit-9 adjudication of the three open items recorded in `docs/08e` Section 8:

1. Eligible-node intervention-set policy for Phase B calibration and held-out evaluation.
2. DCDI fit-RNG convention beyond the reproduction_pass `seed_torch = seed_numpy = 42` value.
3. The selected-configuration artefact path produced by Phase B and consumed by held-out evaluation.

These three decisions must be frozen before Commit 9 (calibration runner) begins.

### What does NOT change under this closure

- No source / test / configuration JSON edit. The runtime artefacts under `results/model_selection/` are not modified by this closure.
- No selection criterion, evaluation rule, metric primitive, wrapper API, real-study guard, schema field, seed-pool integer, threshold triple, training budget, or intervention value changed.

---

22/05/2026 — DAGMA training_status semantics verified

### Context

After mechanical closure of the reproduction_pass (entry above), a focused read-only inspection was carried out on the DAGMA `training_status` field. The question was whether `training_status = "converged"` in the reproduction-pass DAGMA artefacts denotes a real algorithmic convergence check on the acyclicity penalty `h_final`, or merely the convention that the configured optimisation budget completed without error. The latter would matter, because the DAGMA wrapper does not implement observed inner-loop early stopping and every reproduction-pass run reached the full configured budget of `(T - 1) * warm_iter + max_iter = 130000` iterations.

### Inspection

The inspection was read-only. It traced the wrapper's status-emission branch in `src/symbolic_priors_cd/wrappers/dagma.py` and confirmed that `training_status` is set by an `if/elif/else` chain on `h_final`: `not np.isfinite(h_final)` -> `"diverged"`; `h_final <= cfg.h_diagnostic_threshold` -> `"converged"`; otherwise -> `"max_iter"`. The default `h_diagnostic_threshold` is `1e-5`. No code path emits `"converged"` purely on budget exhaustion; budget exhaustion with finite but above-threshold `h_final` is routed to `"max_iter"` instead. The pipeline writes the wrapper's `training_status` verbatim into `run.json` with no translation. The three existing DAGMA reproduction-pass runs recorded `h_final` of approximately `2.91e-6`, `2.84e-6`, `2.59e-6`, all well below `1e-5`, so the emission of `"converged"` on each run is independently justified by the source predicate. The three dedicated wrapper-diagnostics tests (`test_training_status_converged`, `test_training_status_max_iter`, `test_training_status_diverged`) pin the same three-way distinction at the test level.

### Decision

The DAGMA `training_status` semantics in the current code are correct and consistent with the wrapper API contract: `"converged"` is an `h_final`-predicate label, not a budget-completion label. No source, configuration, test, or result-artefact change is required. The mechanical closure of the reproduction_pass remains semantically reliable. The `training_status_counts = {"converged": 3}` value carried in the DAGMA reproduction-pass summary is supported by the per-run `h_final` evidence and by the source-level predicate.

### Documentation change

A short DAGMA-specific clarification was added to `docs/04_wrapper_api_contract.md` Section 7 immediately after the `training_status` bullet list. It states (a) that the DAGMA wrapper does not implement observed inner-loop early stopping and therefore leaves the top-level `n_iterations` field as `null`, and (b) that for DAGMA the `converged` / `max_iter` distinction is an `h_final` predicate, not an early-stop iteration-count predicate. This is a documentation clarification only; no new policy is introduced and the status taxonomy is not rewritten.

### What does NOT change under this entry

- No source, test, configuration JSON, or result artefact is modified.
- The reproduction-pass close-out recorded in the entry above remains in force.
- No selection criterion, evaluation rule, schema field, training budget, threshold value, or seed-pool value changed.
- `docs/02`, `docs/08`, `docs/08a`, `docs/08b`, `docs/08c`, `docs/08d`, and `CLAUDE.md` are not edited. `docs/08e_reproduction_pass_readout.md` is added; this `docs/03` entry records the closure.

---

22/05/2026 — Eligible-nodes intervention-set policy frozen for calibration and held-out evaluation (Adjudication (a) closed)

### Context

Phase B calibration and held-out evaluation require an explicitly frozen eligible-nodes intervention-set policy before Commit 9 (calibration runner) and Commit 10 (held-out evaluation runner) begin. Phase A reproduction-pass used minimal `do(X_0 = ±2)` smoke coverage; that policy is documented in docs/02 Section 3.3 and remains in force for Phase A. Calibration and held-out evaluation need a full policy that resolves the six per-axis questions: intervention magnitude, node coverage, intervention signs, inclusion of root-of-DAG nodes, topological-depth stratification, and the constancy of the policy across replicate seeds and across the two candidate base models within each stage.

The evidence base for the decision is `docs/08f_eligible_nodes_intervention_policy.md`. That document records the six per-axis findings, the primary precedent read directly (Brouillard et al. 2020 DCDI Appendix B.1 — the dominant precedent at `d = 10`), supplementary precedent for sample-based distributional evaluation (Chevalley et al. 2025, Communications Biology CausalBench), field-heterogeneity context from the Q1 Elicit report May 2026, and honest limitations regarding the literature's lack of convergence on intervention-selection policies for held-out evaluation of learned SCMs.

### Decision (six per-axis positions)

1. **Intervention magnitude:** unchanged from docs/02 Section 4.2: `|X_j| = 2`, deterministic point intervention, uniform across nodes.
2. **Node coverage:** all 10 nodes of the 10-node ER2 selection cell are intervention targets.
3. **Intervention signs:** both positive and negative for every targeted node, yielding 20 intervention conditions per seed (`{do(X_j = +2), do(X_j = -2) for j ∈ [0, 9]}`).
4. **Roots inclusion:** root-of-DAG nodes are included without exclusion.
5. **Topological stratification:** not applied; all-nodes coverage at `d = 10` obviates per-stratum selection.
6. **Cross-seed and cross-model uniformity:** the same node-index policy is applied across all replicate seeds within a stage (calibration-pool seeds `{201, 202}`; held-out-pool seeds `{301, 302, 303, 304, 305}`) and across both candidate base models (DAGMA, DCDI) within each stage. The same policy is used at calibration and at held-out evaluation. The node-index policy is index-stable across seeds; the causal roles of those indices (root, intermediate, leaf) vary per SCM realisation, as expected under random ER2 graph generation.

### Implementation arithmetic

- Calibration: 20 intervention conditions × 2 calibration seeds = 40 intervention cells per candidate configuration. The frozen calibration grid evaluates 5 candidate configurations per model, producing 200 intervention cells per model across the grid.
- Held-out: 20 intervention conditions × 5 held-out seeds = 100 intervention cells per selected configuration per model.

### Evidence base

- Brouillard et al. 2020 (DCDI, NeurIPS 2020), Appendix B.1 and B.5 — read directly as the primary precedent for all-nodes coverage at `d = 10`.
- Q1 Elicit report (May 2026) — supplied the field-heterogeneity context and the methodological framing that the literature does not converge on a single intervention-selection policy for held-out evaluation of learned SCMs.
- Chevalley et al. 2025 (Communications Biology, CausalBench) — supplementary precedent for sample-based distributional evaluation.
- Full per-axis reasoning, citation tier markers, and verification audit trail: `docs/08f_eligible_nodes_intervention_policy.md`.

The literature does not converge on a single policy for this question; this document records the project's reasoned choice from the documented defensible range. That framing is preserved verbatim from `docs/08f` §5.

### Documentation change

`docs/02_base_model_selection.md` updated to v1.9. A new Section 4.3 titled "Eligible-nodes intervention-set policy for calibration and held-out evaluation" is inserted immediately after Section 4.2. As a structural consequence, the previous Section 4.3 (Criterion 2: Prior-injection ergonomics) and Section 4.4 (Criterion 3: Standardisation robustness) are renumbered to Section 4.4 and Section 4.5 respectively. The substance of those two subsections is unchanged. The six internal cross-references that previously pointed to "Section 4.4" (the preprocessing equations under Project-owned preprocessing) are updated to "Section 4.5" as numeric-pointer updates only, with no substance change. The Section 4.2 pointer language "for eligible nodes in the selection study graph" is updated to reference Section 3.3 (Phase A coverage) and the new Section 4.3 (calibration and held-out coverage) as the eligible-nodes sources per stage.

### What does NOT change

- Section 4.2 intervention magnitude convention (`|X_j| = 2`).
- Section 3.3 Phase A reproduction-pass intervention coverage (minimal `do(X_0 = ±2)` smoke).
- Section 2 lexicographic decision rule.
- Section 7 C-P11 real-budget reapplication policy.
- Section 8 timeline and compute budget.
- Section 9 tactical-constants block, including `mmd_n_samples = 1000`.
- Seed pools `reproduction = (101, 102, 103)`, `calibration = (201, 202)`, `held_out_evaluation = (301, 302, 303, 304, 305)`.
- DAGMA and DCDI threshold triples and sparsity grids.
- DAGMA and DCDI training budgets (`warm_iter`, `max_iter`, `dcdi_num_train_iter = 300000`).
- Metric primitives (SHD, SID, MMD), wrapper APIs, the wrapper status taxonomy, and configuration-hash semantics.
- The substance of the previous Section 4.3 (Criterion 2) and Section 4.4 (Criterion 3); only their numeric labels and inbound cross-reference numbers changed.

### Forward note

The policy will be implemented by Commit 9 (calibration runner) and Commit 10 (held-out evaluation runner). The configuration hashes of those runners will incorporate the intervention-set policy via canonical JSON serialisation of the calibration and held-out configuration objects.

### Adjudications (b) and (c) remain open

Two pre-Commit-9 adjudications recorded in `docs/08e` Section 8 remain open and must be frozen before Commit 9 begins:

- Adjudication (b): DCDI fit-RNG convention beyond the reproduction_pass `seed_torch = seed_numpy = 42` value.
- Adjudication (c): selected-configuration artefact path produced by Phase B and consumed by held-out evaluation.

---

22/05/2026 — DCDI fit-RNG seed convention frozen for calibration and held-out evaluation (Adjudication (b) closed)

### Context

Adjudication (a) is already closed (eligible-nodes intervention-set policy, docs/02 v1.9). Adjudication (b) concerns optimiser/fit RNG, not SCM-generation RNG and not intervention-target selection RNG. The reproduction_pass already used `seed_torch = seed_numpy = 42` for DCDI fits (the value records the optimiser-RNG inputs to `LearnableModel_NonLinGaussANM` and the per-fit NumPy/Torch state initialisation inside the wrapper). The open question was whether the same single integer remains across Phase B configurations and held-out evaluation, or whether the calibration / held-out configs carry per-stage fit-RNG values.

The Q1 Elicit report did not address this question because Q1 concerned intervention-selection policy. No separate literature synthesis is required for adjudication (b); the decision is methodological and engineering-driven, and is recorded in this entry rather than in a new `docs/08*` document.

### Decision

- For every DCDI fit at Phase B calibration and at held-out evaluation, `seed_torch = seed_numpy = 42`.
- This matches the reproduction_pass convention exactly.
- Each `(SCM seed, DCDI configuration)` pair therefore produces a deterministic DCDI fit under the project's fixed fit-RNG convention.

### Rationale

- The selection study compares DAGMA and DCDI under their selected configurations on the fixed 10-node ER2 thesis cell. It does not aim to estimate DCDI optimiser variance.
- Only two calibration seeds (`{201, 202}`) are available per DCDI configuration; varying DCDI optimiser seeds during calibration would inject unestimated optimiser noise into the within-model hyperparameter ranking and contaminate the choice of `reg_coeff`.
- Fixed DCDI fit RNG makes DCDI deterministic conditional on the SCM/data seed, placing it on cleaner footing relative to DAGMA, whose optimisation is effectively deterministic given the input data and which does not use DCDI-style Gumbel/Bernoulli sampling.
- The primary held-out variation should therefore reflect SCM / data-realisation variation under the fixed fit-RNG convention, rather than a mixture of SCM variation and DCDI optimiser variation.

### Supplementary sensitivity diagnostic

A pre-declared post-selection diagnostic is added so that the project does not silently treat "DCDI is deterministic given the SCM seed" as evidence about total DCDI optimisation variance.

- Commit 10 held-out evaluation runs five additional DCDI fits on the calibration-selected DCDI configuration only.
- Held-out SCM seed: `301`.
- Additional DCDI fit RNGs: `{43, 44, 45, 46, 47}` (matched `seed_torch = seed_numpy = k` for each `k`).
- Report SHD, SID, primary MMD, per-intervention MMD, `graph_status`, `sampler_status`, `training_status`, `final_h`, `final_gamma` / `final_mu` where available, and runtime.
- The diagnostic is separate from calibration selection, from held-out base-model selection, and from the docs/02 Section 2 lexicographic decision rule.
- It is a local sensitivity estimate at one selected DCDI configuration and one held-out SCM seed; it is not a global variance bound and does not exhaustively characterise DCDI optimiser variance across configurations or held-out seeds.

### Documentation change

`docs/02_base_model_selection.md` updated to v1.10. A new Section 4.6 titled "DCDI fit-RNG seed convention for calibration and held-out evaluation" is added at the tail of Section 4, immediately after the existing Section 4.5 (Criterion 3: Standardisation robustness). No existing Section 4 subsection is renumbered. No internal cross-reference is updated by this amendment.

### What does NOT change

- Section 2 lexicographic selection rule.
- Reproduction_pass / Phase A policy (Section 3.3), including the Phase A `seed_torch = seed_numpy = 42` value already in force.
- Section 4.2 intervention magnitude convention (`|X_j| = 2`).
- Section 4.3 eligible-nodes intervention-set policy.
- Seed pools `reproduction = (101, 102, 103)`, `calibration = (201, 202)`, `held_out_evaluation = (301, 302, 303, 304, 305)`.
- DAGMA and DCDI threshold triples and sparsity grids.
- DAGMA and DCDI training budgets, including `dcdi_num_train_iter = 300000`.
- Metric primitives (SHD, SID, MMD), wrapper APIs, the wrapper status taxonomy, and configuration-hash semantics.
- The C-P11 real-budget reapplication requirement (Section 7).

### Forward note

- Commit 9 (calibration runner) must implement the fixed DCDI fit-RNG convention `seed_torch = seed_numpy = 42` across every DCDI fit in the calibration grid.
- Commit 10 (held-out evaluation runner) must implement and report the supplementary fit-RNG sensitivity diagnostic on the calibration-selected DCDI configuration at held-out SCM seed `301` over fit RNGs `{43, 44, 45, 46, 47}`, in a clearly separated artefact path.
- The reproduction_pass `seed_torch = seed_numpy = 42` convention recorded on 20/05/2026 is unchanged.

### Adjudication (c) remains open

Adjudication (c) (selected-configuration artefact path and schema produced by Phase B calibration and consumed by held-out evaluation) is now the only remaining pre-Commit-9 adjudication. It must be frozen before Commit 9 begins.