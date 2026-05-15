# SID verification and integration plan (Doc 07)

Version 1.0. Planning artefact only. No source code, test code,
dependency manifest, or external repository file is created or
modified by this document. Implementation does not begin until the
plan is reviewed and the first SID commit is approved.

This document supersedes the earlier internal-SID-first draft, which
was withdrawn after `docs/04i_gadjid_sid_backend_spike.md` demonstrated
that a clean external backend is available on the project environment.

## 1. Status and purpose

- SID is still **not implemented** in this project. The function
  `symbolic_priors_cd.metrics.interventional.sid_score(predicted_dag,
  true_dag)` exists as a typed stub that validates its inputs and
  raises `NotImplementedError` with the message "SID implementation
  is deferred pending explicit verification."
- A pre-registered scaffold test
  (`tests/test_interventional_metrics.py::
  test_sid_preregistered_hand_computed`) is **skipped** with the
  reason "SID not yet implemented - expected value is provisional
  scaffolding only" and an `expected_sid: int | None = None` line
  that must be replaced before the test is unskipped.
- This plan replaces the earlier internal-SID-first direction.
  Internal SID is now a fallback only.
- `docs/04i_gadjid_sid_backend_spike.md` is the empirical basis for
  adopting `gadjid` as the preferred backend. It records: clean
  Python 3.12 install from a prebuilt abi3 wheel, `numpy`-only
  runtime dependency, the API signature
  `gadjid.sid(g_true, g_guess, edge_direction)`, the return type
  `(normalised_distance, mistake_count)`, the
  `edge_direction="from row to column"` value matching the project
  convention, upstream R-SID cross-validation of `parent_aid` on
  DAGs, and a locally-verified
  `gadjid.sid == gadjid.parent_aid` identity on DAG inputs.
- This plan does **not** implement SID, does **not** add `gadjid` to
  project dependencies, and does **not** unskip any test. Those
  actions belong to the implementation sequence in Section 13.
- No base-model selection conclusion can be reported as
  scientifically complete until SID is implemented, verified, and
  the skipped SID scaffold has been unskipped or deliberately
  replaced. The block is reaffirmed in
  `docs/phase_2c_dagma_readout.md` Section 12 and the

## 2. Backend decision

The project **commits to `gadjid==0.1.0` as the preferred runtime
SID backend**, conditional on adding the dependency deliberately in a
later commit (Commit A in Section 13). This decision is grounded in
the empirical findings of `docs/04i_gadjid_sid_backend_spike.md`,
summarised here:

- `pip install --only-binary=:all: gadjid` succeeded on Python 3.12 /
  `win_amd64` from the prebuilt
  `gadjid-0.1.0-cp38-abi3-win_amd64.whl`. No Rust toolchain was
  invoked. abi3 implies CPython 3.8+ binary compatibility; the
  project requirement `>=3.12,<3.13` is well inside this range.
- Runtime dependency surface is `numpy` only. No R, no `cdt`, no
  `rpy2`. No CPython implementation tied to a single minor version.
- API: `gadjid.sid(g_true, g_guess, edge_direction)` returns
  `(normalised_distance, mistake_count)` where `mistake_count` is a
  Python `int`. The project will extract `tuple[1]` and discard the
  normalised score.
- `edge_direction="from row to column"` matches the project's
  row-source / column-destination convention recorded in
  `docs/03_decision_log.md` and `docs/04a_orientation_audit.md`.
- Upstream `gadjid_python/tests/test_parent_AID_against_R_SID.py`
  asserts `parent_aid(Gtrue, Gguess, edge_direction="from row to
  column")[1] == int(rsid)` on 100-node DAG fixtures sourced from the
  R `SID` package v1.1. The gadjid documentation states that
  `parent_aid` equals SID on DAG inputs. The spike empirically
  verified `gadjid.sid(A, B, ...) == gadjid.parent_aid(A, B, ...)`
  on 20 random DAG pairs with `n in [2, 9]` and varied densities,
  with 0 / 20 mismatches.
- License: `gadjid` is MPL-2.0. The project is MIT. Runtime use of
  MPL-2.0 code from MIT code is unproblematic; the file-level
  copyleft obligation only applies if the project modifies or
  distributes `gadjid` source files. The project will not vendor
  `gadjid` source.

The project explicitly does **not** adopt `cdt` or any R-based SID
package as a runtime dependency. Both options were ruled out during
the earlier environment inspection (`R` and `Rscript` are not on the
project `PATH`; `cdt` brings in `rpy2` and an R toolchain whose
absence in the project environment is already recorded in
`docs/03`).

An internal Peters and Buhlmann SID implementation remains a
**fallback** only. It is not the preferred route. It would only be
revived if `gadjid` ceases to install on a target platform or if a
later review explicitly rejects the runtime dependency.

## 3. SID definition and output convention

- The project uses **DAG-only Peters and Buhlmann SID** as exposed
  by `gadjid.sid` for DAG inputs.
- The project reports the **raw integer mistake count**, i.e.
  `gadjid.sid(...)[1]`. The integer is in `[0, n * (n - 1)]` for an
  `n`-node DAG pair.
- The project does **not** report the normalised SID score from the
  `sid_score` primitive. Normalisation, if ever needed, is the
  reporting layer's responsibility, not the metric primitive's.
- The project does **not** compute CPDAG lower / upper SID. The
  wrapper layer emits thresholded DAGs whenever
  `graph_status == "valid_dag"` (see
  `docs/04_wrapper_api_contract.md` and `docs/06`); when
  `graph_status` is not `valid_dag`, the upstream status gate
  already blocks downstream SID computation. If a future amendment
  introduces CPDAG inputs, a separate plan must justify the variant
  change.
- SID is asymmetric. `SID(G_true, G_pred) != SID(G_pred, G_true)` in
  general. Argument order matters and is tested explicitly
  (Sections 4 and 10).

## 4. Project-facing API and argument order

### 4.1 Public signature (unchanged)

The project-facing signature is preserved:

```python
sid_score(predicted_dag: np.ndarray, true_dag: np.ndarray) -> int
```

This matches the convention recorded in `docs/03_decision_log.md`
(metrics-layer entry pinning `(predicted, true)` ordering for both
`shd` and `sid_score`) and aligns SID with SHD's already-merged
`shd(predicted, true, reversal_cost=2)` order. SID is therefore
**not** an exception to the project metric convention.

### 4.2 Backend mapping (internal)

`gadjid` expects the true graph first and the guessed graph second.
The wrapper flips arguments exactly once at the boundary:

```python
gadjid.sid(
    true_dag.astype(np.int8),
    predicted_dag.astype(np.int8),
    edge_direction="from row to column",
)[1]
```

The return value of `sid_score` is `int(tuple[1])`. The normalised
score `tuple[0]` is discarded.

### 4.3 Asymmetry test requirement

The mapping above must be covered by a regression test that picks an
asymmetric DAG pair and asserts:

- `sid_score(P, T)` matches
  `int(gadjid.sid(T_int8, P_int8, edge_direction="from row to column")[1])`;
- `sid_score(P, T) != sid_score(T, P)` for the chosen pair.

An asymmetric witness is empirically available from
`docs/04i_gadjid_sid_backend_spike.md` Section 5.3: under the
project-facing call, `sid_score(empty_3x3, chain_0_1_2) = 3` and
`sid_score(chain_0_1_2, empty_3x3) = 0`. The implementation test
must reproduce these values once the backend is wired in.

## 5. Edge orientation convention

- The project convention is row-source / column-destination:
  `adjacency[i, j] = True` means edge `i -> j` (recorded in
  `docs/03` and `docs/04a`).
- `gadjid.sid` must **always** be called with
  `edge_direction="from row to column"`. This is hard-coded inside
  the wrapper.
- The public `sid_score` API must **not** expose `edge_direction`.
  Allowing the caller to choose would be a needless source of error
  and would conflict with the project convention.

### 5.1 Empirical context from docs/04i

The spike's specific small example
(`A = 0->1->2`, `B = 0->2`, both read row-to-column) returned the
same `(normalised, mistake_count)` tuple under both
`"from row to column"` and `"from column to row"`. This happens
because, for that particular pair, the column-to-row reading
relabels every node `i -> (n-1-i)` and the count of incorrectly
identified ordered pairs is invariant under that relabelling. It is
not a defect in `gadjid`. The two `edge_direction` values are
semantically distinct on general adjacency layouts.

Because the spike's chosen pair did not expose sensitivity, the
project plan treats this as a **regression-test requirement** rather
than a one-off observation. Two complementary tests are mandated:

1. **Wrapper-call assertion (deterministic, preferred).**
   Monkeypatch `gadjid.sid` at the metrics module's import boundary
   inside the test, capture the arguments of the call, and assert
   that `edge_direction == "from row to column"` was passed exactly
   once. This test is independent of any specific DAG pair and
   catches any future regression that would flip or omit the
   convention argument.

2. **Asymmetric-fixture witness (empirical, secondary).** Find or
   freeze a small DAG pair `(A, B)` such that
   `gadjid.sid(A, B, "from row to column")` differs from
   `gadjid.sid(A, B, "from column to row")` (a pair that breaks the
   `i -> (n-1-i)` relabelling symmetry). Record the fixture in the
   test file with a derivation note. The wrapper passing the wrong
   `edge_direction` then produces a different `mistake_count`.
   If no such small fixture is found at implementation time, the
   wrapper-call assertion is sufficient on its own; the empirical
   witness is recommended but not strictly required.

## 6. Input validation policy

The project API accepts **bool adjacency matrices** (same contract
as `shd`). Validation runs **before** any cast to `int8` and before
any call to `gadjid`:

1. 2D shape: `ndim == 2`.
2. Square: `shape[0] == shape[1]`.
3. Equal shapes: `predicted_dag.shape == true_dag.shape`.
4. dtype: `dtype == bool` (strict).
5. No self-loops: `np.any(np.diag(matrix)) is False`.
6. Predicted graph acyclic.
7. True graph acyclic.

Validation steps 1 to 5 are already enforced by
`symbolic_priors_cd.metrics._graph_validation._validate_adjacency`
(square 2D bool, no self-loops) and the existing shape-mismatch
check in the SID stub. Reuse that helper.

Acyclicity (steps 6 and 7) is new. The plan **pins** the metric-side
helper to the following design:

- **Location:** `src/symbolic_priors_cd/metrics/_graph_validation.py`
  (the existing module that already houses
  `_validate_adjacency`).
- **Name:** `_is_acyclic_adjacency` (matching the wrappers-side
  function so future consolidation is a single rename).
- **Purpose:** validate that bool adjacency inputs are DAGs before
  calling `gadjid`. The helper is metric-internal; it is not
  exported and should remain `_`-prefixed.
- **Algorithm:** mirror the existing
  `wrappers._graph_status._is_acyclic_adjacency` behaviour unless
  there is a compelling reason not to. That reference implementation
  builds successive matrix powers up to size `d`, casts the bool
  adjacency to `int64`, and returns `False` as soon as
  `np.trace(prod) != 0` at any power; otherwise returns `True` after
  `d` iterations.
- **Dependency direction:** **`metrics` must not import from
  `wrappers`**. The whole point of duplicating the implementation
  (rather than importing the wrappers-side helper) is to keep
  metrics independent of the wrapper layer.
- **Cross-reference:** both helpers must carry a short comment
  pointing at the sibling. Suggested wording for the metric-side
  helper:
  `# Mirrors wrappers._graph_status._is_acyclic_adjacency; metrics
  is deliberately kept independent of the wrappers package. Future
  consolidation must update both call sites together.`
  The wrappers-side helper gets the reciprocal comment (added in
  Commit B alongside the metric-side helper, so the cross-reference
  is consistent at the same point in time).
- **Tests:** the metric-side helper is covered indirectly by the
  cyclic-input rejection tests in Section 10.8. No dedicated
  helper-unit test is required.

This design is non-negotiable for the SID implementation commit. If
a later subphase consolidates the two helpers into a shared
sub-package, that consolidation is a separate commit and must update
both existing call sites simultaneously.

### 6.1 Rejection contract (no silent repair, no SHD fallback)

- Cyclic predicted: raise `ValueError` with message containing
  `"cycle"`.
- Cyclic true: same.
- Bidirected pair: counted as a 2-cycle, rejected by the acyclicity
  check above.
- Self-loop on either input: rejected by `_validate_adjacency`.
- Shape mismatch or non-square: rejected by `_validate_adjacency`.
- Non-bool dtype: rejected with `TypeError` (existing behaviour of
  `_validate_adjacency`).
- The wrapper must **never** delete cycles, symmetrise bidirected
  edges, or otherwise coerce inputs into DAGs.
- The wrapper must **never** fall back to SHD on invalid input.
  Failed SID is a hard error per `docs/03`.

After all validation passes, the wrapper casts `bool -> int8` for
both arguments and then calls `gadjid.sid`. The dtype cast happens
**only** after validation succeeds.

## 7. Backend error policy

`docs/04i` recorded that `gadjid` rejects every probed invalid input
but with inconsistent error classes:

- `RuntimeError` for cyclicity, non-square, shape mismatch, 1D
  input, and non-`int8` dtype;
- `PanicException` (Rust panic exposed via PyO3) for self-loops,
  bidirected pairs in DAG mode, and out-of-range int8 values like
  `-1`;
- `TypeError` only for bad `edge_direction` strings.

Policy:

1. **Prevalidation is the primary defence.** Sections 5 and 6
   already mandate this. After the project-side validation chain
   (shape, dtype, no self-loops, no cycles, equal shapes) passes,
   `gadjid` should not raise on a well-formed DAG. The wrapper
   contract is that `gadjid` is only ever called on inputs that
   the project has already certified as valid bool DAGs of equal
   shape.
2. **Catch `RuntimeError` and re-raise as project `ValueError`.**
   Wrap the `gadjid.sid` call in
   `try ... except RuntimeError as exc:` and re-raise as
   `ValueError` with a stable project-level message, for example
   `"gadjid backend rejected SID input"`. Chain the original via
   `raise ... from exc`. The project test suite asserts the error
   class only; it never asserts the Rust-side wording.
3. **No direct `pyo3` dependency.** The project must **not** add
   `pyo3` to its dependencies and must **not** import any name from
   a `pyo3` module. The `PanicException` class is internal to
   `gadjid`'s runtime; the project does not import it by name.
4. **Optional `PanicException` backstop, narrowly scoped.** A
   `PanicException` raised by the Rust side is a subclass of
   `BaseException`, not of `Exception`, so a bare `except Exception:`
   would not catch it. If the wrapper implements a backstop catch,
   it must:
   - use a narrow `except BaseException as exc:` branch placed
     **after** the `RuntimeError` branch;
   - immediately filter by class name, for example
     `if type(exc).__name__ != "PanicException": raise`;
   - re-raise the filtered case as a project `ValueError` with the
     stable message above, chaining the original via
     `raise ... from exc`.
   This pattern catches the Rust panic class by name without
   importing it.
5. **Do not catch unrelated `BaseException` types.** Specifically,
   the narrow branch above must **not** swallow `KeyboardInterrupt`
   or `SystemExit`. The class-name filter (`PanicException`) is
   what selects the panic case; everything else is re-raised
   unchanged.
6. **Backstop is optional, prevalidation is mandatory.** The
   `PanicException` backstop exists only against future gadjid
   releases that might change which conditions panic versus error.
   If the project review prefers to omit the backstop entirely and
   rely solely on prevalidation, that is acceptable; in that case
   any panic that escapes prevalidation is treated as a project
   bug, not a metric-of-the-day failure.
7. **No suppression of unrelated errors.** The wrapper must not
   wrap arbitrary `Exception` types from `gadjid`. The catch list
   is limited to `RuntimeError` and, optionally, the narrow
   `PanicException`-by-name branch above. Tests must not depend on
   Rust-side panic strings or `RuntimeError` strings.

## 8. Dependency and license policy

- Add `gadjid==0.1.0` deliberately, in a single dedicated commit
  (Commit A in Section 13). The same commit updates the project
  lock file (`requirements-lock.txt`) and adds a `docs/03` entry.
- Do **not** float the version. Pin exactly.
- Do **not** upgrade `gadjid` without rerunning the full SID
  regression test set (Section 10), including the
  `sid == parent_aid` agreement test.
- `gadjid` is MPL-2.0. The project is MIT. Runtime use is
  acceptable. File-level copyleft only applies to modified or
  distributed MPL-2.0 source. The project will **not** modify or
  vendor `gadjid` source.
- Note explicitly that this is **not** a CDT or R runtime
  dependency. The project environment does not require an R
  toolchain.

## 9. Required implementation changes

The implementation commits (Section 13) make the following changes,
in order:

1. **Dependency**: add `gadjid==0.1.0` to `pyproject.toml` and the
   relevant lock file; add a `docs/03_decision_log.md` entry that
   records the adoption, the spike citation (`docs/04i`), and the
   pinned version.
2. **`sid_score` implementation** in
   `src/symbolic_priors_cd/metrics/interventional.py`:
   - keep the public signature `sid_score(predicted_dag, true_dag) -> int`;
   - extend the docstring to document raw mistake count, asymmetry,
     argument-order convention, and the gadjid backend (without
     leaking gadjid into the function signature);
   - run the full Section 6 validation, then cast to `int8`, then
     call `gadjid.sid(true_dag_int8, predicted_dag_int8,
     edge_direction="from row to column")`, then return
     `int(tuple[1])`;
   - remove the `NotImplementedError` raise and the
     "implementation is deferred" note;
   - add the defence-in-depth catch for `RuntimeError` and
     (carefully) `PanicException`.
3. **Sanity-check / compatibility gate** in
   `src/symbolic_priors_cd/metrics/sanity_checks.py`:
   - `check_sid_self_zero(true_dag)` currently catches
     `NotImplementedError` and returns `None`, which downstream
     maps to `sid_self_zero_status="deferred"`. Once `sid_score`
     is implemented, the `try / except NotImplementedError` branch
     becomes dead code. Remove it. `check_sid_self_zero` now
     returns `int(sid_score(true_dag, true_dag))` directly, and
     `_derive_sid_status` maps `0 -> "passed"`, non-zero -> `"failed"`.
   - The `assert_ground_truth_compatibility` policy is unchanged:
     `"failed"` is always a hard error; `"deferred"` remains the
     escape hatch only if a future regression ever returns `None`.
4. **Existing skipped scaffold** in
   `tests/test_interventional_metrics.py`:
   - The two stub-contract tests
     (`test_sid_valid_inputs_raise_not_implemented`,
     `test_sid_not_implemented_message_content`) become stale and
     must be **removed**; their purpose was to enforce the
     deferred-stub contract.
   - The skipped scaffold `test_sid_preregistered_hand_computed`
     must be **unskipped or deliberately replaced**. The case it
     constructs is `true = chain 0->1->2`, `predicted = empty 3x3`;
     under the project-facing call `sid_score(predicted, true)`,
     `gadjid` yields `3` on this fixture (recorded in
     `docs/04i` Section 5.5). Set `expected_sid = 3` and remove the
     `"provisional scaffolding only"` warning. The unskipping
     happens in the same commit that lands the implementation
     (Commit D in Section 13).
5. **`docs/03_decision_log.md`** entries:
   - one entry on `gadjid` adoption, pinned version, license note,
     and the wrapper conventions (Section 4 and Section 5);
   - one entry, on SID verification completion, when the regression
     test set and the scaffold replacement are all green.

The wrappers and the existing SHD / MMD code are **not** changed.

## 10. Required tests

All tests live in `tests/test_interventional_metrics.py` alongside
the existing MMD tests, unless noted.

### 10.1 Backend dependency check

- `test_sid_backend_gadjid_importable`: import `gadjid`, assert the
  module has a callable `sid` attribute. This catches a missing or
  silently-removed dependency.
- `test_sid_backend_gadjid_pinned_version`: read the installed
  version via `importlib.metadata.version("gadjid")` and assert it
  equals the pinned version string (matching `pyproject.toml`).
  This catches accidental upgrades that bypass the regression
  gate.

### 10.2 Identity

- `test_sid_score_identity_fixed`: for several fixed DAGs
  (empty `n=3`, chain `0->1->2`, fork `0->{1,2}`, collider
  `{0,1}->2`, chain `n=5`), assert `sid_score(G, G) == 0`. The
  identity property does not depend on `gadjid`'s internal
  derivation, but it is the most fundamental sanity check.
- `test_sid_score_identity_random_dags`: generate at least 20
  random DAGs with `n in {3, 5, 8}` and a mix of sparse / ER2 /
  dense densities; assert `sid_score(G, G) == 0` on every one. Use
  the existing random-DAG generator in
  `symbolic_priors_cd.data.scm_generator`.

### 10.3 Raw count

- `test_sid_score_returns_int_mistake_count`: assert that
  `sid_score(...)` returns a Python `int`, not a float, on at least
  one asymmetric pair. This catches an accidental return of the
  normalised score.

### 10.4 Argument order / asymmetry

- `test_sid_score_argument_order_asymmetric`: with
  `predicted = empty 3x3` and `true = chain 0->1->2`, assert
  `sid_score(predicted, true) == 3` and
  `sid_score(true, predicted) == 0` (values verified empirically
  in `docs/04i`).
- `test_sid_score_argument_order_general`: for an arbitrary
  asymmetric DAG pair `(A, B)`, assert that `sid_score(A, B)` and
  `sid_score(B, A)` are not necessarily equal (specifically that
  they differ on the chosen pair).

### 10.5 Backend call mapping

- `test_sid_score_wrapper_calls_gadjid_with_flipped_args_and_pinned_edge_direction`:
  monkeypatch `gadjid.sid` (from the project's imported reference
  inside `metrics.interventional`) with a recording fake, call
  `sid_score(predicted, true)`, and assert:
  - the fake was called exactly once;
  - the first positional argument is `true` cast to `int8`;
  - the second positional argument is `predicted` cast to `int8`;
  - `edge_direction == "from row to column"`;
  - the wrapper returned `int(fake_return[1])`.
  This test is the primary safeguard against accidental
  argument-order flips or `edge_direction` regressions; it does not
  depend on any specific DAG pair.

### 10.6 sid == parent_aid agreement

- `test_sid_score_matches_parent_aid_on_fixed_dags`: on a small
  fixed DAG battery (at least 5 pairs), assert
  `gadjid.sid(true, pred, "from row to column")[1] ==
  gadjid.parent_aid(true, pred, "from row to column")[1]`. This
  test is a regression backstop for the transitive R-SID chain
  (`docs/04i` Section 7). If a future `gadjid` release diverges
  `sid` from `parent_aid` on DAG inputs, this test catches it
  before the selection-study runner consumes a misleading number.

### 10.7 Dtype policy

- `test_sid_score_rejects_int8_input_from_caller`: caller passes
  `dtype=int8`; the project API still requires `bool` and must
  raise `TypeError`. The wrapper's internal `int8` cast does not
  weaken the public dtype contract.
- `test_sid_score_rejects_int64_input`,
  `test_sid_score_rejects_uint8_input`,
  `test_sid_score_rejects_float64_input`: same expectation.

### 10.8 Invalid-graph rejection

- `test_sid_score_rejects_cyclic_predicted`: cyclic predicted DAG
  raises `ValueError` containing `"cycle"`.
- `test_sid_score_rejects_cyclic_true`: same.
- `test_sid_score_rejects_self_loop_predicted`: rejected by
  `_validate_adjacency` with `"self-loops"`.
- `test_sid_score_rejects_self_loop_true`: same.
- `test_sid_score_rejects_non_square_predicted`: `"square"`.
- `test_sid_score_rejects_shape_mismatch`: `"same shape"`.
- `test_sid_score_rejects_non_bool_dtype`: `"bool"`.
- `test_sid_score_rejects_bidirected_pair`: an off-diagonal pair
  `[i, j] = [j, i] = True` is a 2-cycle and must be rejected by
  the acyclicity check, **not** silently symmetrised.

### 10.9 No SHD fallback

- `test_sid_score_does_not_fall_back_to_shd_on_invalid_input`: on a
  cyclic predicted DAG, assert that `sid_score` raises
  `ValueError` rather than returning the SHD value. Use a pair
  where the SHD value would differ from the raised error semantics.

### 10.10 Scaffold regression

- The existing `test_sid_preregistered_hand_computed` is unskipped
  with `expected_sid = 3` and the `"provisional scaffolding only"`
  warning removed. This becomes a regression test for the
  `empty 3x3 vs chain 0->1->2` case at the project-facing call
  site.

### 10.11 Tests deliberately **not** added

- No hand-derived numeric SID values are introduced as canonical
  oracle values beyond row 3 of the spike. Once `gadjid` is the
  backend authority, an internal hand derivation that disagrees
  with `gadjid` would be a defect in the hand derivation, not in
  `gadjid`. Hand examples may still appear in test comments as
  explanatory sanity checks; they do not gate CI.
- No internal-SID re-implementation tests (the internal fallback
  is reserved for a separate plan if it is ever needed).

## 11. What not to test as primary oracle

- **Hand-derived nontrivial SID values are not the primary oracle.**
  After `gadjid` is adopted, `gadjid` is the backend authority on
  DAG inputs, transitively underwritten by R `SID` v1.1 through
  the `parent_aid` upstream test plus the local
  `sid == parent_aid` agreement test.
- **Do not include unverified hand-computed numeric SID values as
  canonical oracle values.** Specifically, do not invent expected
  SID values for chain-versus-reversed-chain, fork-versus-chain,
  collider, missing-middle-edge, or extra-edge variants. If a
  future review wants these as supplementary sanity checks, derive
  them by calling `gadjid.sid` once and recording the value, with a
  comment stating the derivation route.
- **Avoid overfitting tests to backend observations that merely
  restate the implementation.** Tests should target wrapper
  conventions (argument flip, edge_direction pin, dtype cast),
  validation (Section 6), raw-count extraction, and the backend
  invariants (identity, `sid == parent_aid`). They should not
  enumerate every internal gadjid output as if it were a project
  contract.

## 12. Computational complexity and tractability

- Selection-study graphs are small: 10 nodes, ER2 density,
  5 seeds, two candidate models, two preprocessing conditions.
- `gadjid` is implemented in Rust with documented better-than-SID
  complexity. On the spike, `gadjid.sid` returned in microseconds
  on `n <= 9` DAGs.
- For the selection-study cell described in `docs/02`, SID is
  called a few hundred times at most. Total SID cost is in the
  seconds range and is not a bottleneck.
- The runtime bottleneck remains model fitting (DAGMA: less than a
  second per fit at small `n`; DCDI: tens of seconds at small
  `n`). SID is negligible by comparison.
- Tests must remain lightweight. The random-DAG identity test
  (Section 10.2) caps at `n=8` and 20 iterations to keep the suite
  under one second of added cost.

## 13. Atomic implementation sequence

Each commit must pass its acceptance criterion before the next is
started. Tests are added only in the commit where they make sense;
the suite must stay green at every commit boundary.

**Green-at-every-boundary rule.** The full pytest suite (including
the existing skipped SID scaffold, which remains skipped until
Commit D) must pass at the boundary of every commit listed below.
No commit may leave any test in a "temporarily expected to fail"
state. No commit may rely on a follow-up to make the suite green.
The previous Commit B/C ambiguity is resolved by folding the two
stub-contract tests' removal **and** the minimal green-keeping
replacement tests into Commit B itself; Commit C then adds the
full regression suite.

| # | Title | Files touched | Acceptance |
|---|---|---|---|
| A | Dependency adoption | `pyproject.toml` (add `gadjid==0.1.0` to runtime dependencies), `requirements-lock.txt` (record the resolved entry), `docs/03_decision_log.md` (one new entry citing `docs/04i` and pinning `gadjid==0.1.0`) | `pip install -e .` resolves cleanly; `import gadjid` succeeds in the project venv; `gadjid.sid` and `gadjid.parent_aid` are callable; `importlib.metadata.version("gadjid") == "0.1.0"`; full pytest suite green; no SID source/test change in this commit (the existing skipped SID scaffold remains skipped) |
| B | `sid_score` implementation **plus** minimal test green-keeping | `src/symbolic_priors_cd/metrics/_graph_validation.py` (add metric-side `_is_acyclic_adjacency` per Section 6); `src/symbolic_priors_cd/wrappers/_graph_status.py` (add cross-reference comment only, no behavioural change); `src/symbolic_priors_cd/metrics/interventional.py` (replace the `NotImplementedError` stub with the validation chain, `bool -> int8` cast, `gadjid.sid(true, predicted, edge_direction="from row to column")` call, raw-count extraction, and the Section 7 error handling); `tests/test_interventional_metrics.py` (**delete** the two now-stale stub-contract tests `test_sid_valid_inputs_raise_not_implemented` and `test_sid_not_implemented_message_content`; **add** minimal replacement tests sufficient to keep coverage of the function on this commit: at least `test_sid_score_identity_fixed` (Section 10.2 fixed-DAG subset) and one rejection test for cyclic input) | full pytest suite green at this commit boundary; the existing skipped SID scaffold remains skipped (Commit D will unskip it); no test is left in `xfail` or "expected to fail" state |
| C | Full SID regression test set | `tests/test_interventional_metrics.py` (add the remaining tests from Section 10: 10.1 dependency import + version pin, 10.2 random-DAG identity, 10.3 raw count, 10.4 argument-order asymmetry, 10.5 wrapper-call mapping monkeypatch, 10.6 `sid == parent_aid` agreement, 10.7 dtype policy, 10.8 invalid-graph rejection minus the cyclic-predicted case already added in Commit B, 10.9 no SHD fallback) | full pytest suite green; every new test from Section 10.1-10.9 is asserted green; no `xfail` |
| D | Scaffold unskip and compatibility-gate update | `tests/test_interventional_metrics.py` (unskip `test_sid_preregistered_hand_computed`, set `expected_sid = 3`, remove the `"provisional scaffolding only"` warning); `src/symbolic_priors_cd/metrics/sanity_checks.py` (remove the `NotImplementedError` catch in `check_sid_self_zero`; `_derive_sid_status` simplifies to `0 -> "passed"`, non-zero -> `"failed"`; the `"deferred"` status remains in the Literal type but is no longer produced on a valid DAG) | full pytest suite green; the unskipped scaffold passes; `check_sid_self_zero(true_dag)` returns `0` on a valid DAG with no `NotImplementedError` path |
| E | Decision-log and readout closure | `docs/03_decision_log.md` (one new entry: SID verification complete, gadjid backend, raw-count contract, argument-order convention preserved, no SHD fallback, no silent repair); optionally a short note in a project readout if one is being prepared | doc entry merged; selection study unblocked from the SID gate |

Commits A, B, C, D, and E are each **independent reviewable units**
and must stay separate. None of them may be merged into another.
The plan does **not** prescribe a commit for selection-study
integration; that belongs to a separate subphase plan.

## 14. Acceptance criteria before base-model selection

The selection study described in `docs/02_base_model_selection.md`
remains blocked until **every** item below is true:

- `gadjid==0.1.0` is pinned in `pyproject.toml` and the lock file
  (Commit A);
- `sid_score(predicted_dag, true_dag) -> int` is implemented per
  Section 4 (Commit B);
- argument-order asymmetry test passes (Section 10.4);
- `edge_direction` is hard-pinned and the wrapper-call assertion
  test passes (Section 10.5);
- the raw-count test passes (Section 10.3);
- invalid-graph rejection tests all pass (Section 10.8);
- the `sid == parent_aid` regression test passes (Section 10.6);
- the existing skipped SID scaffold is unskipped or deliberately
  replaced (Commit D);
- the full pytest suite is green;
- `docs/03_decision_log.md` is updated to record adoption and the
  verification result (Commits A and E);
- no invalid graph is silently repaired anywhere in the code path;
- there is no fallback to SHD on invalid SID input.

Until every item is satisfied, no base-model selection conclusion
is scientifically complete.

## 15. Risks and mitigations

1. **Backend version drift.** A future `gadjid` release could
   change `sid` semantics or split it from `parent_aid` on DAG
   inputs. **Mitigation:** pin `gadjid==0.1.0`; the
   `sid == parent_aid` regression test (Section 10.6) catches a
   divergence before it affects the selection study.
2. **Convention mismatch.** A future refactor could accidentally
   pass the wrong `edge_direction` or flip the argument order.
   **Mitigation:** the wrapper-call assertion test (Section 10.5)
   is independent of any specific DAG pair and catches both kinds
   of regression.
3. **Invalid inputs.** A caller passes a cyclic DAG or a non-bool
   matrix. **Mitigation:** project-side pre-validation in Section 6
   raises `ValueError` / `TypeError` with stable messages before
   `gadjid` is ever called.
4. **Rust-side exception instability.** A future `gadjid` release
   could change which conditions raise `RuntimeError` versus
   `PanicException`. **Mitigation:** defence-in-depth in Section 7
   re-raises stable project exceptions; tests assert error class,
   not message.
5. **License misunderstanding.** A reviewer worries that MPL-2.0
   contaminates the project. **Mitigation:** Section 8 records
   that MPL-2.0 file-level copyleft applies only to modified or
   distributed source; runtime dependency use from an MIT project
   is unproblematic. The project will not vendor `gadjid` source.
6. **Overclaiming.** A reader interprets "SID verified" to mean
   "selection study complete" or "DAGMA selected". **Mitigation:**
   Section 1 and Section 14 state explicitly that verified SID
   validates the metric implementation, not any model-selection
   result. Selection is a separate subphase with its own document.
7. **Environment drift.** A reviewer worries that the
   `cp38-abi3-win_amd64` wheel will stop covering future CPython
   versions. **Mitigation:** abi3 wheels cover CPython 3.8+ by
   design; the project's `>=3.12,<3.13` constraint is well inside
   support. If `gadjid` ever drops abi3, that is itself a
   regression caught by Commit A's reinstall.

## 16. First implementation prompt

When the project is ready to begin SID implementation, the first
prompt should be scoped to **Commit A only**:

> "Implement Section 13 Commit A of `docs/07_sid_verification_plan.md`
> only.
>
> Changes:
> - add `gadjid==0.1.0` to the runtime-dependencies block of
>   `pyproject.toml`;
> - record the resolved entry in `requirements-lock.txt` (or the
>   project's equivalent lock file);
> - add a single `docs/03_decision_log.md` entry recording: the
>   pinned version `gadjid==0.1.0`, the MPL-2.0 license note, the
>   citation to `docs/04i_gadjid_sid_backend_spike.md` as the
>   empirical basis, the commitment to the project-facing API
>   `sid_score(predicted, true) -> int`, and the explicit statement
>   that the project will not modify or vendor `gadjid` source.
>
> Do not implement `sid_score`. Do not add the metric-side
> `_is_acyclic_adjacency` helper. Do not modify any test file. Do
> not unskip the existing skipped SID scaffold.
>
> Verification (must all pass before the commit is considered
> complete):
> 1. `pip install -e .` resolves in the project venv without errors.
> 2. `python -c "import gadjid; print(type(gadjid).__name__)"` runs
>    without error.
> 3. `python -c "import gadjid; assert callable(gadjid.sid)"` passes.
> 4. `python -c "import gadjid; assert callable(gadjid.parent_aid)"`
>    passes (the project regression suite later asserts
>    `sid == parent_aid` on DAG inputs; the function must be
>    importable).
> 5. `python -c "from importlib.metadata import version; assert
>    version('gadjid') == '0.1.0'"` passes. Use
>    `importlib.metadata.version` because `docs/04i` Section 4
>    recorded that `gadjid.__version__` is not exposed at the
>    Python level; this verification must not depend on a runtime
>    `__version__` attribute.
> 6. The full pytest suite is green at this commit boundary.
>
> Report: the install outcome (success/failure), the resolved
> `gadjid` version string from `importlib.metadata.version`, the
> output of each of the four `python -c` checks, the pytest result
> line, and the new `docs/03` decision-log entry text verbatim."

Commits B, C, D, and E each get their own prompts and follow the
atomic sequence in Section 13.

---

## End of plan
