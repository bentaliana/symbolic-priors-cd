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