# Orientation Audit

## Purpose

This document is a read-only audit of the project's adjacency-orientation
convention. The aim is to confirm that all current code and tests in the
data and metrics layers agree on a single convention, so that the wrapper
API contract in `docs/04_wrapper_api_contract.md` can reference a verified
project-wide convention.

This audit does not modify any code, does not modify the decision log, and
does not modify the selection study protocol. It only inspects the existing
code and tests and reports the findings. A proposed decision-log sentence
is given at the end of this document for later insertion, but it is not
inserted here.

---

## Inspected files

The audit inspected the following files. Paths are relative to the repository
root.

Source files:

- `src/symbolic_priors_cd/data/scm_generator.py`
- `src/symbolic_priors_cd/data/_sampling.py`
- `src/symbolic_priors_cd/metrics/structural.py`
- `src/symbolic_priors_cd/metrics/_graph_validation.py`

Test files:

- `tests/test_scm_generator.py`
- `tests/test_interventions.py`
- `tests/test_structural_metrics.py`
- `tests/test_sanity_checks.py`

---

## Convention used by all inspected components

Across every inspected file, the convention is:

- `adjacency[i, j] = True` means a directed edge from node `i` (row, source,
  parent) to node `j` (column, destination, child).
- Equivalently, column `j` of the adjacency matrix lists the parents of
  node `j`.
- Linear-Gaussian weights follow the same convention: `weights[i, j]` is
  the coefficient of `X_i` in the structural equation for `X_j`.

There is no place in the inspected code that interprets the adjacency in
the opposite direction.

---

## File-by-file findings

### `src/symbolic_priors_cd/data/scm_generator.py`

- The class docstring on `LinearGaussianSCM` (lines 38 to 48) establishes
  the canonical representation as a pair of matrices `adjacency` and
  `weights` plus a topological ordering tuple.
- `generate_er_dag` docstring at line 142 states explicitly that
  `adjacency[i, j] = True` means edge `i -> j`.
- Line 169, inside `generate_er_dag`, sets edges via
  `adjacency[perm[src_pos], perm[dst_pos]] = True`. The first index is the
  source (row), the second is the destination (column), and `src_pos` is
  strictly less than `dst_pos` in topological-position space. This means
  the only entries set to `True` are forward edges in the topological
  ordering, which is consistent with row-source / column-destination.
- `LinearGaussianSCM.__post_init__`, lines 87 to 94, builds a rank
  dictionary and asserts `rank[u] < rank[v]` for every `(u, v)` in
  `np.where(adj_bool)`. Here `u` is the row index of the True entry and
  `v` is the column index. The code therefore enforces that the row index
  is the earlier-ranked node, which is the source.
- The validator at line 79, `np.any(self.weights[~adj_bool] != 0.0)`,
  ties the weight matrix to the same orientation: weights can only be
  nonzero where the adjacency matrix marks an edge in the same orientation.

### `src/symbolic_priors_cd/data/_sampling.py`

- Line 70, inside `_ancestral_sample`, computes
  `parents = np.where(scm.adjacency[:, j])[0]`. This reads column `j` of
  the adjacency matrix to find the parent nodes of node `j`. This is the
  active runtime semantics of the sampling kernel and is consistent with
  row-source / column-destination.
- Line 71 then computes
  `parent_contrib = X[:, parents] @ scm.weights[parents, j]`. The
  structural equation `X_j = sum_i W[i, j] * X_i + noise` therefore uses
  row-source / column-destination for the weight matrix as well.
- Because both observational and interventional sampling delegate to this
  shared kernel, the same convention applies in both sampling paths by
  construction.

### `src/symbolic_priors_cd/metrics/structural.py`

- The `shd` docstring at line 38 states explicitly that
  `predicted[i, j] = True` means a directed edge `i -> j`.
- Lines 79 to 83 implement the reversal logic with
  `diff = predicted != true` and `reversal_mask = diff & diff.T`. Because
  the reversal mask is symmetric, the divide-by-two in `n_reversals` is
  correct only under a fixed orientation convention. The docstring fixes
  that convention, so the SHD computation is consistent with row-source /
  column-destination by construction.

### `src/symbolic_priors_cd/metrics/_graph_validation.py`

- The shared adjacency validator at lines 12 to 29 checks dtype, square
  2D shape, and the absence of self-loops. It does not assume any
  orientation. It is therefore compatible with any single project-wide
  convention, and it does not contradict the row-source /
  column-destination convention used elsewhere.

### `tests/test_scm_generator.py`

- Lines 65 to 75, in `test_generate_er_dag_topological_order_consistent_with_edges`,
  construct `rank = {node: pos for pos, node in enumerate(topo)}` and then
  iterate `for u, v in zip(rows.tolist(), cols.tolist())`. The test
  asserts `rank[u] < rank[v]`, which only matches the project orientation
  if `u` is the source (row) and `v` is the destination (column).
- The helper `_make_chain_scm` at lines 28 to 57 builds a chain
  `X0 -> X1 -> ... -> X(n-1)` by setting `adjacency[i, i+1] = True` and
  `weights[i, i+1] = weight`. The same row-source / column-destination
  convention is used here.

### `tests/test_interventions.py`

- The helper `_chain_scm` at lines 28 to 52 encodes the chain
  `X0 -> X1 -> X2` by setting `adjacency[0, 1] = True`,
  `adjacency[1, 2] = True`, `weights[0, 1] = weight`, and
  `weights[1, 2] = weight`. This is consistent with row-source /
  column-destination.
- `test_known_chain_do_operator_semantics` and
  `test_do_root_node_cuts_downstream_parents` then verify causal semantics
  consistent with that chain encoding. For example, `do(X0 = 3)` produces
  a downstream shift in `X1` and `X2`, which matches the orientation
  `X0 -> X1 -> X2`.

### `tests/test_structural_metrics.py`

- The helper `_chain_adj` at lines 22 to 27 sets
  `for i in range(n - 1): A[i, i+1] = True`. This is again row-source /
  column-destination.
- All hand-crafted boolean matrices in this file follow the same
  convention.

### `tests/test_sanity_checks.py`

- This test file only constructs SCMs through `generate_linear_gaussian_scm`
  and constructs `Intervention` objects. It does not directly create
  adjacency matrices, so it inherits the orientation convention from
  the generator and is consistent with the rest of the audit.

---

## Cross-file agreement

All inspected components agree on the same convention:

- `adjacency[i, j] = True` means a directed edge from node `i` to node `j`.
- Row index is the source (parent).
- Column index is the destination (child).
- Weights follow the same convention: `weights[i, j]` is the coefficient
  of `X_i` in the structural equation for `X_j`.

No file in the inspected set interprets the matrix in the opposite
direction.

---

## Risks or ambiguities

No orientation-convention inconsistencies were found in the audited files. The convention is documented in two source docstrings
(`scm_generator.generate_er_dag` and `structural.shd`), enforced by the
`__post_init__` topological-order check on `LinearGaussianSCM`, embodied
by the `_ancestral_sample` parents-from-column lookup, and used by every
hand-crafted adjacency matrix in the test suite.

The shared graph-validation helper does not assume any orientation, so it
neither confirms nor contradicts the convention. That is the appropriate
behaviour for a generic structural validator, since orientation is a
semantic concern rather than a structural one.

---

## Proposed decision-log sentence (not yet inserted)

The following sentence is proposed for later insertion into
`docs/03_decision_log.md`. It is not inserted by this commit. It is
recorded here so that the decision-log entry can be reviewed and added in
a separate documented step.

> The project canonical adjacency-orientation convention is row-source and
> column-destination. That is, `adjacency[i, j] = True` means a directed
> edge from node `i` to node `j`. All current code and tests in the data
> and metrics layers already follow this convention. Wrappers must conform
> to this convention before passing any adjacency output to the evaluator.

---

## What is not changed by this audit

The following are not changed by this audit, in line with the documentation
and audit only scope of this commit:

- `docs/01_research_question_and_commitments.md` is not modified.
- `docs/02_base_model_selection.md` is not modified.
- `docs/03_decision_log.md` is not modified. The proposed sentence above is
  recorded here as a proposal only.
- `docs/phase_1_readout.md` is not modified.
- `CLAUDE.md` is not modified.
- No source code is changed.
- No test code is changed.
- No external dependency is installed.
