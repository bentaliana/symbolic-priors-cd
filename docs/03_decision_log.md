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