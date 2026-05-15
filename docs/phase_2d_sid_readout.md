# Phase 2d readout: SID implementation and verification subphase handoff

## 1. Status and purpose

- Date: 2026-05-15.
- Purpose: orient a future contributor to the completed SID implementation
  and verification subphase, summarise what was done and why, and state
  what the project can and cannot conclude from the result.

SID implementation and verification are complete. The full regression suite
is green with 0 skipped tests. The selection study is unblocked from the
SID side but has not yet been run. Prior-loss implementation remains
deferred.

This readout records the SID subphase only. It does not close the project;
base-model selection and prior-loss implementation are downstream.

Authoritative documents take priority over this readout if any conflict
arises.

## 2. Why SID mattered

SID (Structural Intervention Distance, Peters and Buhlmann) is the primary
interventional-structure criterion for base-model selection in this project.
It measures how many interventional distributions are incorrectly identified
under a predicted graph relative to the true graph. SHD and MMD are not
substitutes:

- SHD counts edge-edit differences and does not measure interventional
  correctness.
- MMD measures distributional distance on sampled data. Wrapper-vs-truth
  MMD depends on the wrapper's sampler and on structural recovery; it is a
  diagnostic, not a definition.

SID argument order, edge orientation, and whether the raw mistake count
or the normalised score is returned are all load-bearing for the selection
study. Getting any of these wrong would invalidate the comparison result.

## 3. Backend decision

`gadjid==0.1.0` was adopted as the runtime SID backend. The decision is
recorded in `docs/03_decision_log.md` (15/05/2026 entry) with the empirical
basis in `docs/04i_gadjid_sid_backend_spike.md`.

Summary of reasons:

- Installs cleanly on Python 3.12 / win_amd64 from a prebuilt abi3 wheel
  with no Rust toolchain invoked at install time.
- Runtime dependency is `numpy` only. No CDT, no R, no rpy2.
- API: `gadjid.sid(g_true, g_guess, edge_direction)` returns
  `(normalised_distance, mistake_count)`. The raw `int` mistake count is
  available as `tuple[1]`.
- `edge_direction="from row to column"` matches the project's row-source /
  column-destination adjacency convention.
- `gadjid.parent_aid` is directly cross-validated against R SID v1.1 in the
  upstream test suite on 100-node DAG inputs. The spike empirically verified
  `gadjid.sid == gadjid.parent_aid` on 20 random DAG pairs with 0/20
  mismatches, providing a transitive R-SID validation chain.
- License: MPL-2.0. Runtime use from an MIT project is unproblematic. The
  project does not vendor or modify `gadjid` source.

An internal Peters and Buhlmann SID re-implementation is now a fallback only.
It would only be revived if `gadjid` ceased to install on a target platform
or if a future review explicitly rejected the runtime dependency.

## 4. Project API and semantics

Public signature (unchanged from the original stub):

    sid_score(predicted_dag: np.ndarray, true_dag: np.ndarray) -> int

- Both inputs must be square boolean DAG adjacency matrices with no
  self-loops and no directed cycles. Row-source / column-destination
  convention: `adjacency[i, j] = True` means edge `i -> j`.
- Project-facing argument order is predicted first, true second. This
  matches the existing `shd(predicted, true, ...)` convention.
- The internal backend call flips argument order exactly once at the
  wrapper boundary:
  `gadjid.sid(true_dag.astype(np.int8), predicted_dag.astype(np.int8),
  edge_direction="from row to column")[1]`
- `edge_direction` is pinned and is never exposed in the public API.
- The return value is the raw integer mistake count. The normalised
  distance (`tuple[0]`) is discarded.
- Invalid inputs (non-bool dtype, non-square shape, self-loops, cycles,
  shape mismatch) raise `TypeError` or `ValueError` from project-side
  validation before any backend call. No silent graph repair and no SHD
  fallback.

## 5. Implementation summary

Changes made across the SID subphase commits:

- `gadjid==0.1.0` added as a pinned runtime dependency in `pyproject.toml`;
  resolved entry recorded in `requirements-lock.txt`.
- `_is_acyclic_adjacency` helper added to
  `src/symbolic_priors_cd/metrics/_graph_validation.py`. This mirrors
  the wrapper-side helper in `wrappers/_graph_status.py` using the same
  matrix-power algorithm. Both carry cross-reference comments. Metrics
  do not import from wrappers; the duplication is intentional.
- `sid_score` implemented in
  `src/symbolic_priors_cd/metrics/interventional.py`. Validation runs in
  this order: `_validate_adjacency` (bool dtype, square 2D, no self-loops),
  shape equality check, acyclicity check on both inputs, then `bool -> int8`
  cast, then the `gadjid.sid` call with `RuntimeError` and narrowly scoped
  `PanicException` backstop.
- `check_sid_self_zero` in `src/symbolic_priors_cd/metrics/sanity_checks.py`
  now calls `sid_score(true_dag, true_dag)` directly and returns its `int`
  result. The `try/except NotImplementedError` block is removed.
- `_derive_sid_status` in `sanity_checks.py` now accepts `int` only and
  returns `Literal["passed", "failed"]`. The `None -> "deferred"` mapping
  is removed.
- `CompatibilityReport.sid_self_zero_value` is now typed `int`, not
  `int | None`. `sid_self_zero_status` is typed `Literal["passed", "failed"]`.
- `require_sid` is retained as a keyword argument on
  `assert_ground_truth_compatibility` for backward compatibility, but the
  deferred-SID gate branch it controlled has been removed. The parameter is
  now a no-op and is documented as such.

## 6. Tests added and what they verify

All tests live in `tests/test_interventional_metrics.py` except the
sanity-gate tests, which are in `tests/test_sanity_checks.py`.

**Backend and version availability.**
`test_sid_backend_gadjid_importable`, `test_sid_backend_gadjid_parent_aid_callable`,
`test_sid_backend_gadjid_pinned_version`.
Guards against a missing or silently-upgraded dependency. Uses
`importlib.metadata.version` rather than a non-existent `gadjid.__version__`.

**Identity on fixed DAGs.**
`test_sid_score_identity_fixed_dags`.
Five structures (empty n=3, chain, fork, collider, chain n=5). A regression
that catches any argument-flip, convention mismatch, or off-by-one that
would produce a non-zero identity score.

**Identity on generated DAGs.**
`test_sid_score_identity_random_dags`.
Twenty cases from the project SCM generator spanning `n in {3, 5, 8}` and
sparse/ER2/dense densities. Confirms the identity property holds on
project-generated adjacency matrices, not just trivial hand-crafted ones.

**Raw-count extraction.**
`test_sid_score_returns_int_mistake_count`.
Asserts the return is a Python `int`, equals `backend[1]`, and does not
equal `backend[0]` (the normalised float). Guards against accidentally
returning the normalised score.

**Argument-order asymmetry.**
`test_sid_score_argument_order_asymmetric`.
With `predicted=empty 3x3` and `true=chain 0->1->2`, asserts
`sid_score(predicted, true) == 3` and `sid_score(true, predicted) == 0`.
This is the primary numerical guard on the argument-order flip.

**Backend-call monkeypatch mapping.**
`test_sid_score_wrapper_calls_gadjid_with_flipped_args_and_pinned_edge_direction`.
Replaces `gadjid.sid` with a recording fake at the module boundary. Asserts:
called exactly once, first arg is `true.astype(int8)`, second arg is
`predicted.astype(int8)`, both have `dtype == int8`,
`edge_direction == "from row to column"`, and the return is `int(fake[1])`.
This test is independent of any specific DAG pair and is the primary
safeguard against future argument-order or convention regressions.

**Edge-direction pinning / sensitivity witness.**
`test_sid_backend_edge_direction_sensitivity_witness`.
Calls `gadjid.sid` directly with the same int8 pair under both
`"from row to column"` and `"from column to row"` and asserts the results
differ. Fixture: `true=fork 0->{1,2}`, `pred={1->2}`. Confirms the two
conventions are semantically distinct and that pinning matters.

**sid == parent_aid agreement.**
`test_gadjid_sid_matches_parent_aid_on_fixed_dags`.
Six fixed int8 DAG pairs. Asserts `gadjid.sid(...)[1] == gadjid.parent_aid(...)[1]`
and that normalised distances agree within `1e-12`. Catches any future
`gadjid` release in which `sid` and `parent_aid` diverge on DAG inputs,
which would break the transitive R-SID validation chain.

**Dtype contract.**
`test_sid_score_rejects_int8_input_from_caller`, `..._int64_input`,
`..._uint8_input`, `..._float64_input`.
All four assert `TypeError` with `"bool"`. The project API accepts bool
inputs only; the `int8` cast is internal.

**Invalid graph rejection.**
`test_sid_score_rejects_cyclic_predicted`, `test_sid_score_rejects_cyclic_true`,
`test_sid_score_rejects_bidirected_pair`, plus the previously existing tests
for self-loops, non-square, and shape mismatch.
Note: a bidirected pair `A[i,j]=A[j,i]=True` is a directed 2-cycle and is
caught by the acyclicity check; the raised error says `"cycle"`, not
`"bidirected"`. This is correct and documented in the test.

**No numeric fallback on invalid input.**
`test_sid_score_raises_on_invalid_input_not_a_number`.
Asserts `ValueError` is raised on a cyclic input. The docstring explicitly
frames this as "raises rather than returning a number" to distinguish it
from any future SHD-fallback regression.

**Active backend-reference fixture.**
`test_sid_score_empty_predicted_vs_true_chain_returns_backend_reference_count`.
Replaces the previously skipped scaffold. Asserts
`sid_score(empty 3x3, chain 0->1->2) == 3`. The expected value 3 is the
backend-confirmed raw count, established by the argument-order asymmetry
test and consistent with the spike observations.

**Sanity-gate tests.**
In `tests/test_sanity_checks.py`: `test_check_sid_self_zero_returns_zero_for_valid_dag`,
`test_run_checks_sid_status_is_passed`, `test_run_checks_sid_value_is_zero`.
Verify that the evaluator compatibility gate now sees SID as active and
passing on a valid ground-truth SCM.

## 7. Test results

Full pytest suite at subphase closure: **384 passed, 0 skipped, 2 warnings.**

The previous skipped count was 1 (the deferred SID scaffold in
`tests/test_interventional_metrics.py`). That test was replaced by the
active backend-reference fixture in the final implementation commit, so the
skipped count dropped from 1 to 0.

The 2 warnings are the pre-existing `RuntimeWarning: invalid value
encountered in matmul` and `RuntimeWarning: invalid value encountered in
subtract` from
`tests/test_dagma_wrapper_residuals.py::test_non_finite_sigma_sets_unavailable_unresolved_noise_policy`.
These are generated by deliberately feeding a non-finite sigma matrix to
the DAGMA residual estimator. They are expected, documented, and unrelated
to SID.

## 8. What this verifies

- The project can compute SID through the adopted `gadjid` backend.
- The project-facing argument order `(predicted, true)` is correctly bridged
  to the backend order `(true, predicted)` at the wrapper boundary, verified
  by both numerical assertion and monkeypatched call inspection.
- The row-source / column-destination edge orientation is pinned via
  `edge_direction="from row to column"` and confirmed by a concrete witness
  fixture where the two conventions produce different outputs.
- The raw integer mistake count is returned; the normalised score is
  discarded.
- Invalid graphs (cyclic, bidirected, self-loop, non-square, shape mismatch,
  non-bool dtype) are rejected by project-side validation before any backend
  call.
- The evaluator compatibility gate treats SID as active: `check_sid_self_zero`
  returns 0 on a valid ground-truth DAG, `sid_self_zero_status` is `"passed"`,
  and the deferred-SID escape path no longer exists in the normal code flow.

## 9. What this does not verify

- This does not run the base-model selection study. No selection-study
  runner has been written, and no model-comparison results have been produced.
- This does not choose DAGMA or DCDI as the base model.
- This does not validate prior-loss implementation. No prior-loss term
  exists in code on either the DAGMA or DCDI side.
- This does not prove anything about symbolic priors, prior corruption, or
  the main-study research question.
- This does not replace later analysis of selection-study results under
  multi-seed conditions, 10-node ER2 graphs, and the full criterion ordering
  in `docs/02_base_model_selection.md`.

## 10. Next project state

- Verified SID is no longer blocking selection-study planning or execution.
- The immediate next substantive subphase is base-model selection planning
  and execution. The selection-study runner must be designed, implemented,
  and run under the protocol in `docs/02_base_model_selection.md`.
- The selection study must use the criterion ordering frozen in `docs/02`.
  No post hoc threshold or criterion adjustment.
- Prior-loss work remains deferred until after the base-model selection
  study has produced a defensible decision.
- DCDI Commit 11 (loss-hook injection) remains paused pending the
  project-level review of C-P11 / C-P12 and the selection-study outcome.
- `require_sid` on `assert_ground_truth_compatibility` remains in the
  signature as a no-op. Removing it is a small compatibility follow-up that
  can be addressed at any convenient point; it is not a current blocker.
