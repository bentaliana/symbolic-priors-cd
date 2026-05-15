# 04i: gadjid SID-backend feasibility spike

## Status

Spike artefact, read-only with respect to project source, project tests,
project documentation, `pyproject.toml`, and `requirements-lock.txt`.
This document records the findings of a one-off feasibility spike on
the `gadjid` Python package as a candidate runtime backend for the
project's SID metric. No project file was modified by the spike. No
runtime dependency was added.

This document does not commit the project to using `gadjid`. The
SID-verification plan, when next drafted, will reference this document
for the source-of-truth observations and decide whether to depend on
`gadjid` or to pursue an alternative path.

Date: 2026-05-15.

## 1. Purpose

Determine whether the `gadjid` package
([PyPI](https://pypi.org/project/gadjid/),
[GitHub](https://github.com/CausalDisco/gadjid)) is a viable backend
for the project SID metric before any SID implementation work begins.

The project-facing API is fixed by `docs/03_decision_log.md`:
`sid_score(predicted_dag, true_dag) -> int`, returning the raw
Peters and Buhlmann mistake count, with predicted first and true
second. `gadjid` exposes
`sid(g_true, g_guess, edge_direction) -> (normalised, mistake_count)`,
so any project wrapper would have to flip the argument order at the
boundary and return only `tuple[1]`.

This spike answers, with evidence, whether that wrapping is safe.

## 2. Method

The spike was run in an **isolated temporary virtual environment**
outside the project tree. No project file, no project test, no
project doc, and no project dependency manifest was modified at any
point.

- Spike directory: `C:\Users\benja\AppData\Local\Temp\gadjid_spike\`
- Spike venv:
  `C:\Users\benja\AppData\Local\Temp\gadjid_spike\.venv\`
- Python: `3.12.0`, CPython, `win32` (`win_amd64`).
  Matches the project requirement `requires-python = ">=3.12,<3.13"`
  in `pyproject.toml`.
- Install command: `pip install --only-binary=:all: gadjid numpy`.
  The `--only-binary=:all:` flag is defensive; it guarantees that no
  Rust source build is attempted if a wheel is unavailable.
- Spike scripts (run from the venv, all ephemeral):
  - `spike_api.py` (API inspection)
  - `spike_conventions.py` (identity, asymmetry, edge_direction,
    project-wrapper mapping)
  - `spike_invalid.py` (cyclic, non-square, dtype, value-range,
    bad-string probes)
  - `spike_sid_vs_parent_aid.py` (transitive R-SID cross-check via
    `parent_aid`)
- Post-spike verification: `git status --porcelain` returns empty on
  the project tree; `gadjid` is **not** installed in the project venv
  (`importlib.util.find_spec('gadjid')` returned `None`).

All observed values below are recorded as **observations**, not as
hand-derived oracle values. The spike rules disallowed inventing
oracle values; oracle derivation is the next planning step's
responsibility.

## 3. Installation

### Observations

| Item | Result |
|---|---|
| `pip install` exit code | success |
| Installed version | `gadjid 0.1.0` |
| Latest version on PyPI | `0.1.0` (released 2024-07-11) |
| Wheel filename | `gadjid-0.1.0-cp38-abi3-win_amd64.whl` |
| Wheel vs source build | wheel only (no Rust toolchain invoked) |
| Wheel ABI tag | `abi3` (Python stable ABI, CPython 3.8+ compatible) |
| Required runtime deps | `numpy` only |
| `Requires-Python` (PyPI Trove) | `>=3.8` |
| License | `MPL-2.0` |
| Install warnings | none |

The wheel uses the Python stable ABI, so the same binary serves
CPython 3.8, 3.9, ..., 3.12, 3.13. The project's
`>=3.12,<3.13` constraint is well inside this range.

### Cross-platform availability (per PyPI Trove classifiers)

Wheels are published for:

- Windows x86-64 and ARM64
- macOS 10.12+ x86-64 and 11.0+ ARM64
- Linux manylinux/musllinux x86-64 and aarch64

Implementation language: Rust. Runtime exposure via PyO3.

## 4. API surface

Exact module surface from `dir(gadjid)`:

```
['ancestor_aid', 'gadjid', 'oset_aid', 'parent_aid', 'shd', 'sid']
```

`gadjid.__version__` is **not** exposed at the Python level; the
version is recorded only in the dist-info `METADATA` file.

### 4.1 `gadjid.sid` signature and docstring

`inspect.signature(gadjid.sid)` returned:

```
(g_true, g_guess, edge_direction)
```

`inspect.getdoc(gadjid.sid)`, verbatim:

> "Structural Identification Distance between two DAG adjacency
> matrices (sparse or dense)"

`type(gadjid.sid)` is `builtin_function_or_method` (Rust PyO3 binding).

### 4.2 Module-level docstring, relevant passages (verbatim)

> "Adjacency matrices are accepted as either int8 numpy ndarrays or
> int8 scipy sparse matrices in CSR or CSC format."

> "If `edge_direction='from row to column'`, then a `1` in row `r`
> and column `c` codes a directed edge `r -> c`; if
> `edge_direction='from column to row'`, then a `1` in row `r` and
> column `c` codes a directed edge `c -> r`; for either setting of
> `edge_direction`, a `2` in row `r` and column `c` codes an
> undirected edge `r - c` (an additional `2` in row `c` and row `r`
> is ignored; one of the two entries is sufficient to code an
> undirected edge)."

> "An adjacency matrix for a DAG may only contain 0s and 1s.
> An adjacency matrix for a CPDAG may only contain 0s, 1s and 2s.
> DAG and CPDAG inputs are validated for acyclicity."

### 4.3 Return type discovered by a sample call

`gadjid.sid(np.zeros((3, 3), np.int8), np.zeros((3, 3), np.int8),
edge_direction="from row to column")` returned:

```
type: tuple
repr: (0.0, 0)
  [0] type=float value=0.0
  [1] type=int   value=0
```

Index `[0]` is `normalised_distance` (Python `float`), index `[1]` is
`mistake_count` (Python `int`). The project would discard index `[0]`
and return index `[1]`.

### 4.4 Implication for the project wrapper

The project-facing call `sid_score(predicted, true)` maps to the
backend call as:

```python
gadjid.sid(
    true.astype(np.int8),
    predicted.astype(np.int8),
    edge_direction="from row to column",
)[1]
```

Argument order is flipped exactly once, at the wrapper boundary.
`edge_direction` is pinned at the row-source / column-destination
project convention recorded in
`docs/03_decision_log.md` and `docs/04a_orientation_audit.md`. The
project never exposes `edge_direction` in its public API.

## 5. Convention observations

All inputs were `numpy.int8` matrices in the project's row-source /
column-destination convention. Outputs are **recorded** as
observations and are not asserted against any hand-derived oracle.

### 5.1 Identity `sid(G, G)`

| G | n | `sid(G, G)` |
|---|---|---|
| empty | 3 | `(0.0, 0)` |
| chain `0->1->2` | 3 | `(0.0, 0)` |
| fork `0->{1,2}` | 3 | `(0.0, 0)` |
| collider `{0,1}->2` | 3 | `(0.0, 0)` |
| chain `0->1->2->3->4` | 5 | `(0.0, 0)` |

Identity property `sid(G, G) = (0.0, 0)` is reproduced on every
tested DAG (5 of 5).

### 5.2 Normalisation relation

The documented relation `norm == mistake_count / (n * (n - 1))`
was checked numerically on a battery of cases. Match was
**exact** (within `1e-12`) on every case observed:

| Pair | n | `sid(A, B)` | Relation holds |
|---|---|---|---|
| empty vs `0->1->2` | 3 | `(0.0, 0)` | yes |
| `0->1` vs empty | 3 | `(0.166666..., 1)` | yes |
| `0->1->2` vs reversed chain `2->1->0` | 3 | `(1.0, 6)` | yes |
| fork `0->{1,2}` vs chain `0->1->2` | 3 | `(0.166666..., 1)` | yes |

`tuple[1]` is therefore recoverable as `int(round(norm * n * (n - 1)))`,
or by direct indexing as `tuple[1]`. The project wrapper will use the
direct index.

### 5.3 Argument-order asymmetry

| A | B | `sid(A, B)` | `sid(B, A)` | Asymmetric? |
|---|---|---|---|---|
| empty | `0->1->2` | `(0.0, 0)` | `(0.5, 3)` | yes |
| `0->1` | empty | `(0.166666..., 1)` | `(0.0, 0)` | yes |
| `0->1->2` | reversed `2->1->0` | `(1.0, 6)` | `(1.0, 6)` | no (this specific pair) |

Asymmetry is demonstrated on the first two pairs. The third pair is
symmetric because it saturates the count of incorrect ordered pairs
(`n * (n - 1) = 6` in both directions, since every directed
relationship is reversed). This does not contradict the documented
asymmetric nature of SID; it is a saturated case.

### 5.4 `edge_direction` sensitivity

The chosen example (`A = 0->1->2`, `B = 0->2`, both read in
row-to-column convention) returned the same `(0.3333..., 2)` under
both `edge_direction="from row to column"` and
`edge_direction="from column to row"`. The example was **not**
sensitive to the convention flip. The most natural explanation is
that reading the same int8 matrix under the column-to-row convention
relabels every node `i -> (n-1-i)` for this particular pair, which
leaves the count of incorrectly identified ordered pairs unchanged.

This is **not** a defect in `gadjid`. The two conventions are
semantically distinct on general adjacency layouts. The practical
implication for the project is simpler:

- **The wrapper must always pin `edge_direction="from row to column"`**,
  matching the project convention. Allowing the caller to choose
  would be a needless source of error and would conflict with
  `docs/03` and `docs/04a`. The project never exposes
  `edge_direction` in its public API.

### 5.5 Project-wrapper mapping

A provisional, spike-only wrapper was written:

```python
def project_sid_score(predicted, true) -> int:
    norm, mistakes = gadjid.sid(
        true.astype(np.int8),
        predicted.astype(np.int8),
        edge_direction="from row to column",
    )
    return int(mistakes)
```

Observed values:

| Call | Output |
|---|---|
| `project_sid_score(empty, empty)` | `0` |
| `project_sid_score(chain, chain)` | `0` |
| `project_sid_score(fork, fork)` | `0` |
| `project_sid_score(empty, chain)` | `3` |
| `gadjid.sid(chain, empty, ...)[1]` (cross-check) | `3` |
| `project_sid_score(chain, empty)` | `0` |

The wrapper preserves the project-facing `(predicted, true)` argument
order, agrees with the backend on the asymmetric pair, and exhibits
the documented asymmetry under the project-facing call. **This
wrapper shape is recommended for the eventual implementation.**

## 6. Invalid-input behaviour

Every probed invalid input was rejected. `gadjid` never silently
computed on a malformed input. Error classes vary across cases, which
is the most important caveat for the project wrapper.

| Input | Error class | Message excerpt |
|---|---|---|
| Cyclic 3-cycle as `g_true` | `RuntimeError` | `"Graph is not acyclic"` |
| Cyclic 3-cycle as `g_guess` | `RuntimeError` | `"Graph is not acyclic"` |
| Self-loop on diagonal | `PanicException` | `"found unexpected self-looping edge '1' at position (0, 0)"` |
| Bidirected pair `[0,1]=1` and `[1,0]=1` in DAG mode | `PanicException` | `"Graph not simple: found both edge 1->0 and edge 0->1"` |
| Non-square `(3, 4)` | `RuntimeError` | `"Matrix must be square"` |
| Shape mismatch `(3, 3)` vs `(4, 4)` | `RuntimeError` | `"The two input graphs are not the same size"` |
| 1D input | `RuntimeError` (`TypeError` inside) | PyArray cannot be converted |
| `dtype=bool` | `RuntimeError` (`TypeError` inside) | PyArray cannot be converted |
| `dtype=int64` | same | same |
| `dtype=uint8` | same | same |
| `dtype=float64` | same | same |
| Int8 value `2` in DAG mode | `RuntimeError` | `"Guess graph is not a DAG. Use 'parent_aid' if you want to pass a CPDAG"` |
| Int8 value `-1` | `PanicException` | `"Found value '-1' in adjacency matrix at position (0, 1), expected to see only 0's, 1's or 2's for PDAG"` |
| Bad `edge_direction` string | `TypeError` | `"edge_direction string argument must be either 'from row to column' or 'from column to row'"` |

### 6.1 Implications for the project wrapper

1. **`dtype=bool` is rejected.** The project's
   `metrics._graph_validation._validate_adjacency` enforces
   `dtype == bool`. The wrapper must cast `bool -> int8` immediately
   before calling `gadjid.sid`. The cast is `.astype(np.int8)`.
2. **Error types are inconsistent at the Rust boundary.** Cyclicity
   and shape errors come back as `RuntimeError`; structural errors
   (self-loop, bidirected pair, out-of-range value) come back as
   `PanicException`; bad `edge_direction` raises a clean `TypeError`.
   The project test suite must not depend on Rust-side error
   wording. The wrapper must either:
   - validate every defect upstream (so gadjid never sees bad
     input), and / or
   - catch any `RuntimeError` or `PanicException` from gadjid and
     re-raise as a project-stable `ValueError` with a fixed message.
3. **Defence in depth is cheap.** The project already has
   `_validate_adjacency` (square 2D bool, no self-loops) and
   `wrappers._graph_status._is_acyclic_adjacency` (matrix-power
   acyclicity check). Calling both upstream of the cast guarantees
   that only valid DAGs reach gadjid. This converts every probed
   defect into a clean `ValueError` raised by project code, without
   any reliance on Rust panic strings.

## 7. Upstream cross-validation evidence

The PyPI page mentions the R `SID` package in the runtime comparison
tables ("Results obtained with gadjid v0.1.0 using the Python
interface and the SID R package v1.1 from CRAN"). Those are
**runtime** comparisons, not numerical-equivalence claims.

The GitHub repository's Python README does **not** contain a verbatim
statement that `gadjid.sid` itself has been numerically cross-checked
against the R `SID` package.

However, the repository **does** contain a dedicated R-SID
cross-validation test for `parent_aid`. Found via the GitHub tree
listing
(`repos/CausalDisco/gadjid/git/trees/main?recursive=1`):

- File: `gadjid_python/tests/test_parent_AID_against_R_SID.py`.
- Imports: `from gadjid import parent_aid`.
- Test data: a CSV fixture named `SID-100-node-DAGs.csv` containing
  100-node DAG pairs with precomputed R-SID values.
- Assertion (verbatim): `assert sid[1] == int(rsid)`, where
  `sid = parent_aid(Gtrue, Gguess, edge_direction="from row to
  column")`.

The gadjid module docstring states (verbatim, paraphrased here
because of bold-math characters in the original):

> "You may also calculate the SID between DAGs via
> `parent_aid(DAGtrue, DAGguess, edge_direction)`, but we recommend
> `ancestor_aid` and `oset_aid` and for CPDAG inputs the
> `parent_aid` does not coincide with the SID."

So on DAG inputs, `parent_aid` is documented to equal SID, and
`parent_aid` is what the upstream tests cross-validate against R
`SID` v1.1.

### 7.1 Transitive check (run locally in the spike venv)

To bring the `parent_aid` cross-validation evidence to bear on
`gadjid.sid` itself, the spike ran `spike_sid_vs_parent_aid.py`:

- 20 random DAG pairs.
- `n in [2, 9]`, edge densities `p in [0, 0.6]`.
- For each pair, both `gadjid.sid(A, B, ...)` and
  `gadjid.parent_aid(A, B, ...)` were called and compared as tuples.

**Result: 0 / 20 mismatches.** Every pair returned identical
`(normalised, mistake_count)` tuples.

Therefore:

- Upstream R-SID cross-validation is **direct** for `parent_aid` on
  100-node DAG inputs.
- Upstream R-SID cross-validation is **transitive** for `gadjid.sid`
  on DAG inputs via the documented identity
  `parent_aid == sid` on DAGs, empirically reproduced on a 20-pair
  random battery in the spike.

For the project's own scientific audit this should be **double-locked**
by a project regression test that, on a small fixed DAG battery,
asserts `gadjid.sid(...) == gadjid.parent_aid(...)`. Such a test
catches any future gadjid release in which `sid` and `parent_aid`
might diverge on DAG inputs before the selection-study run consumes
a misleading number.

A stronger direct cross-check (running R `SID` v1.1 from CRAN and
comparing numerically) would require an environment with R installed.
Neither `R` nor `Rscript` is on the project `PATH` (checked during
the earlier deleted SID planning pass via `shutil.which`). Direct R
cross-checking is therefore out of scope for the current spike. It
remains optional and can be performed later in a separate
verification venv if desired.

## 8. Recommendation

**`gadjid` is viable as the runtime SID backend for this project**,
conditional on the wrapper rules below. The dependency is small (one
Rust-backed wheel, only `numpy` at runtime, stable Python ABI), the
license (MPL-2.0) is compatible with the project's MIT license for
end-use, the API is documented, and the upstream tests provide
direct R-SID cross-validation for `parent_aid` plus a transitive
path for `sid` on DAG inputs that the spike verified locally.

### 8.1 Mandatory wrapper conditions

1. **Project-facing API stays as `sid_score(predicted, true) -> int`.**
   Argument order is `(predicted, true)`, recorded in `docs/03`. The
   flip to gadjid's `(g_true, g_guess)` order happens **once** at
   the wrapper boundary.
2. **Return raw `mistake_count`**, not the normalised score. Discard
   `tuple[0]` inside `sid_score`. Reporting the normalised value, if
   ever needed, belongs in a separate aggregation layer, not inside
   the metric primitive.
3. **Pin `edge_direction="from row to column"` always.** Never
   expose `edge_direction` in the public project API.
4. **Validate upstream.** Call the existing
   `metrics._graph_validation._validate_adjacency` (square 2D bool,
   no self-loops, shape match) plus an explicit acyclicity check
   (reuse `wrappers._graph_status._is_acyclic_adjacency` or a
   metric-side twin). Reject every defect with a clean project
   `ValueError` or `TypeError` before any `gadjid` call.
5. **Cast `bool -> int8` at the boundary.** gadjid accepts only
   `int8`; the project metric API uses `bool`. The wrapper does the
   cast immediately before the gadjid call. Do not change the
   project metric dtype contract.
6. **Defence in depth on the Rust boundary.** Catch any
   `RuntimeError` or `PanicException` that nevertheless escapes from
   gadjid and re-raise as a project `ValueError` with a fixed,
   project-stable message. The test suite must not depend on
   Rust-side wording.
7. **Pin the gadjid version.** Add `gadjid==0.1.0` to runtime
   dependencies when the decision to depend is taken. Allow patch
   upgrades only after a project regression test passes.
8. **Lock `sid == parent_aid` agreement.** Add a project regression
   test: on a small fixed DAG battery,
   `assert int(gadjid.sid(true, pred, ...)[1]) ==
   int(gadjid.parent_aid(true, pred, ...)[1])`. This catches any
   future release in which the two functions diverge on DAG inputs.
9. **Document the R-SID transitive chain.** The SID-verification
   plan (next planning document) records that R-SID cross-validation
   is transitive (R-SID <-> `parent_aid` upstream, plus locally
   verified `parent_aid == sid` on DAG inputs). A direct R cross-check
   is not required, but the option is preserved.

### 8.2 What this spike does **not** do

- It does not implement `sid_score`.
- It does not modify any project source, test, doc, or dependency
  manifest. `git status --porcelain` on the project tree is empty;
  `gadjid` is not in the project venv.
- It does not hand-derive any SID oracle value. Backend outputs are
  recorded as observations only.
- It does not run any external R-SID cross-check. R is not on the
  project `PATH`.

## 9. Summary checklist

| Item | Result |
|---|---|
| gadjid installed cleanly on Python 3.12 / win_amd64 | yes, no warnings |
| Installed version | `gadjid 0.1.0` |
| Wheel filename | `gadjid-0.1.0-cp38-abi3-win_amd64.whl` |
| Wheel vs source build | wheel only (prebuilt abi3) |
| Python / platform | CPython 3.12.0, `win32` / `win_amd64` |
| `gadjid.sid` signature | `sid(g_true, g_guess, edge_direction)` |
| Return type | `tuple[float, int]` = `(normalised, mistake_count)` |
| Accepted `edge_direction` values | `"from row to column"`, `"from column to row"` |
| Accepted dtype | `int8` only (numpy ndarray or scipy CSR/CSC); `bool`/`int64`/`uint8`/`float64` rejected |
| Identity `sid(G, G)` | returns `(0.0, 0)` on every tested DAG (5/5) |
| Documented normalisation relation | holds exactly on every checked case |
| Argument-order asymmetry | demonstrated on 2 / 3 candidates (third is saturated) |
| `edge_direction` sensitivity on the chosen example | not exposed (example was naturally symmetric); the wrapper will pin `"from row to column"` |
| Project-wrapper mapping recoverable | yes; `gadjid.sid(true, predicted, edge_direction="from row to column")[1]` |
| Raw `mistake_count` available | yes, `tuple[1]`, Python `int` |
| Invalid-input behaviour | every probed defect raises (no silent compute); error classes vary (`RuntimeError`, `PanicException`, `TypeError`) |
| Upstream R-SID cross-validation for `parent_aid` (DAG inputs) | confirmed via `gadjid_python/tests/test_parent_AID_against_R_SID.py`, 100-node DAGs, `assert sid[1] == int(rsid)` |
| Upstream R-SID cross-validation for `gadjid.sid` directly | not documented in upstream README; transitively underwritten via `sid == parent_aid` on DAGs, verified locally (0/20 mismatches) |
| Recommendation | **gadjid viable as runtime SID backend**, with the wrapper conditions in Section 8.1 |
| Project files changed by this spike | none; `git status --porcelain` empty on the project tree; `gadjid` not in project venv |

## 10. References

- PyPI:
  [https://pypi.org/project/gadjid/](https://pypi.org/project/gadjid/)
  (version `0.1.0`, released 2024-07-11; `Requires-Python: >=3.8`;
  license MPL-2.0).
- GitHub:
  [https://github.com/CausalDisco/gadjid](https://github.com/CausalDisco/gadjid)
  (Rust source, Python wrapper at `gadjid_python/`, R wrapper at
  `gadjid_r/`).
- Paper: Henckel L., Wurtzen T., Weichwald S. (2024). "Adjustment
  Identification Distance: A gadjid for Causal Structure Learning",
  Proceedings of the Fortieth Conference on Uncertainty in
  Artificial Intelligence (UAI). DOI:
  [10.48550/arXiv.2402.08616](https://doi.org/10.48550/arXiv.2402.08616).
- Upstream R-SID cross-validation test:
  `gadjid_python/tests/test_parent_AID_against_R_SID.py` in the
  `CausalDisco/gadjid` repository (main branch).
- Project documents referenced:
  - `docs/03_decision_log.md` (metric argument-order convention,
    `(predicted, true)`; SID exclusion bucket policy).
  - `docs/04_wrapper_api_contract.md` (no-silent-repair policy).
  - `docs/04a_orientation_audit.md` (row-source / column-destination
    adjacency convention).
  - `docs/04h_dagma_sampler_quality_diagnostic.md` (negative-floor
    caveat carried over from C-P11 for downstream MMD reporting; not
    directly relevant to SID but cited for consistency of probe-doc
    style).
  - `docs/phase_2c_dagma_readout.md` (Section 12 identifies verified
    SID as the next critical path).

## 11. End of spike

This spike is closed. The next document in the chain is the
SID-verification plan (Doc 07, to be redrafted with the new goals
the project will commit to), which will cite this document for
backend feasibility and wrapper conditions.
