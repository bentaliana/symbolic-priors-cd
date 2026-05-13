# Equal-Variance Identifiability Sanity Check (C-P12)

## Status

Sanity check only. **This is not a new baseline for the main study** and
not a proposed selection-study method. The probe exists to determine
whether the C-P11 sampler-quality failure on the 3-node ER2
linear-Gaussian fixture can be attributed to a fundamental
non-identifiability of the data, or whether the data is recoverable
under an equal-error-variance-aware exhaustive score.

## Purpose

In C-P11 the DCDI-G wrapper failed sampler-quality validation on this
fixture, learning a thresholded DAG that omits the strongest true edge.
A standard caveat for linear-Gaussian SCMs is that observational data
identifies the DAG only up to Markov equivalence in general, but under
the equal-error-variance assumption (Peters and Buhlmann 2014; Loh and
Buhlmann 2014) the DAG IS identifiable. The project SCM uses a
homogeneous noise scale, so equal-variance applies. This probe checks
whether an exhaustive enumeration with a shared-variance score actually
picks the true DAG out of the data DCDI saw.

## Probe

- Probe script: `inspection/probes/c_p12_equal_variance_identifiability_check.py`
- Environment: project `.venv`, CPU only
- The probe is read-only with respect to project source and external
  repositories. No dependency is installed. No pytest test is added.

## Configuration (frozen, matches C-P11 fixture)

- SCM: `generate_linear_gaussian_scm(n_nodes=3, expected_edges=3, seed=0)`
- Homogeneous `noise_scale = 1.0` for every node.
- True adjacency (row-source / column-destination):

  ```
  [[0 1 0]
   [0 0 0]
   [1 1 0]]
  ```

- True weights (rounded to 4 decimals):

  ```
  [[0.0000 0.5041 0.0000]
   [0.0000 0.0000 0.0000]
   [1.7861 0.5504 0.0000]]
  ```

- Topological order: `(2, 0, 1)`.
- Training data: `sample_observational(scm, n_samples=5000, rng=1)` in
  raw SCM units (no preprocessing).
- Reference DCDI-learned adjacency from C-P11 (threshold = 0.5):

  ```
  [[0 1 0]
   [0 0 0]
   [0 1 0]]
  ```

## Scoring formula

For each candidate DAG and each node `j`, the probe fits ordinary
least squares of `X[:, j]` on `X[:, parents(j)]` with an intercept.
RSS values are pooled, and a single shared residual variance is
estimated. The model is scored by Gaussian BIC under that shared
variance:

```
total_sse  = sum over j of OLS RSS of X[:, j] given X[:, parents(j)]
sigma2_hat = total_sse / (n * d)
log_lik    = -n*d/2 * (log(2*pi*sigma2_hat) + 1)
k_params   = n_edges + d + 1          (edge slopes + per-node intercepts + 1 shared variance)
BIC        = -2 * log_lik + k_params * log(n)
```

`n = 5000`, `d = 3`. Lower BIC is better. The shared-variance term in
`k_params` reflects that all nodes share one residual variance MLE.

## DAG enumeration

The probe iterates over all 2^6 = 64 directed graphs on 3 nodes and
filters for acyclicity using the project's `_is_acyclic_adjacency`
helper. The number of DAGs reported by the probe is **25**, matching
the known count for labelled DAGs on 3 nodes (OEIS A003024).

## Observed results

### Rank 1 (top-scoring DAG)

```
[[0 1 0]
 [0 0 0]
 [1 1 0]]
```

- BIC = 42502.1849
- `n_edges` = 3
- `sigma2_hat` = 0.991662 (very close to the data-generating
  `noise_scale^2 = 1.0`)
- `total_sse` = 14874.9328
- SHD to true = 0

The rank-1 DAG is **exactly the true SCM adjacency**.

### True DAG rank

- Rank: **1 of 25**
- BIC delta from top: +0.0000 (it is the top)

### DCDI-learned DAG rank (from C-P11)

```
[[0 1 0]
 [0 0 0]
 [0 1 0]]
```

- Rank: **19 of 25**
- BIC = 53136.4910
- BIC delta from top: **+10634.3061**
- `n_edges` = 2 (missing the strongest true edge `2 -> 0`)
- `sigma2_hat` = 2.016067 (about twice the true noise variance; the
  missing strong edge dumps its variance into the residual)
- SHD to true = 1

### Top score margin

- Rank-2 BIC minus rank-1 BIC: **+232.2224**
- A BIC gap of order 10^2 is far above conventional "very strong
  evidence" thresholds (a delta of 10 is already considered very
  strong). The top is cleanly separated, not a tie.

### Top 10 DAGs

| rank | BIC | delta_from_top | n_edges | SHD_to_true | adjacency |
| --- | --- | --- | --- | --- | --- |
| 1 | 42502.1849 | +0.0000 | 3 | 0 | `[[0,1,0], [0,0,0], [1,1,0]]` |
| 2 | 42734.4072 | +232.2224 | 3 | 2 | `[[0,0,0], [1,0,0], [1,1,0]]` |
| 3 | 42926.9637 | +424.7788 | 2 | 1 | `[[0,1,0], [0,0,0], [1,0,0]]` |
| 4 | 43672.3339 | +1170.1490 | 2 | 1 | `[[0,0,0], [0,0,0], [1,1,0]]` |
| 5 | 44994.3290 | +2492.1441 | 2 | 3 | `[[0,0,0], [1,0,0], [0,1,0]]` |
| 6 | 48819.5486 | +6317.3637 | 3 | 4 | `[[0,0,0], [1,0,1], [1,0,0]]` |
| 7 | 49448.4503 | +6946.2654 | 2 | 3 | `[[0,0,0], [0,0,1], [1,0,0]]` |
| 8 | 49947.5924 | +7445.4075 | 3 | 6 | `[[0,0,1], [1,0,1], [0,0,0]]` |
| 9 | 50000.0727 | +7497.8879 | 2 | 5 | `[[0,0,1], [1,0,0], [0,0,0]]` |
| 10 | 50360.4671 | +7858.2822 | 2 | 5 | `[[0,0,0], [1,0,1], [0,0,0]]` |

DCDI's learned DAG `[[0,1,0], [0,0,0], [0,1,0]]` appears at rank 19,
not in the top 10.

## Interpretation

The C-P11 fixture **is recoverable** under an equal-error-variance
exhaustive Gaussian-BIC score, with the true DAG selected uniquely and
with a large margin (rank-1-to-rank-2 BIC gap of ~232 on 5000 samples;
rank-19 is over 10^4 BIC units worse than rank 1). The strongest true
edge `2 -> 0` (true weight magnitude 1.79) accounts for a large share
of the variance in `X_0`, and the omission of this edge in DCDI's
learned structure inflates the shared residual variance estimate from
~1.0 to ~2.0.

Consequently, DCDI-G's structural-recovery failure on this fixture
is most naturally read as an **inductive-bias / optimisation /
model-mismatch** issue, not as data impossibility:

- DCDI-G learns per-node nonlinear conditional means via leaky-relu
  MLPs and an augmented-Lagrangian DAG penalty. Its training does
  not encode the equal-variance assumption.
- Under the wrapper's "no second-stop saturation" policy the structure
  never crystallises during training, and the thresholded adjacency at
  0.5 misses the strongest true edge despite that edge being the most
  informative one in the data.
- C-P11's "true adjacency MMD" and "augmented adjacency MMD"
  diagnostics already showed that DCDI's per-node conditionals are
  usable; the structural mistake is what dominates the MMD failure.

The probe DOES NOT show that an equal-variance method is the best base
model for the thesis, only that the data is recoverable in principle
when the right inductive bias is applied. The selection study in the
broader plan is the appropriate place to decide between DAGMA-linear,
DCDI-G, and any other candidate base model on equal footing.

## Caveat

This probe is a sanity check, not a baseline. The exhaustive
enumeration scales as the number of labelled DAGs on n nodes
(super-exponential), so the method here is feasible at 3 nodes but
unsuitable beyond 5-6 nodes without combinatorial search heuristics
or per-node decomposable scoring with permutation enumeration.
Equal-variance Gaussian BIC is also a specific score with strong
assumptions and is not proposed for the thesis main study.

No project source under `src/` was edited as part of this probe.
No pytest test was added. The normal test suite remains green.
