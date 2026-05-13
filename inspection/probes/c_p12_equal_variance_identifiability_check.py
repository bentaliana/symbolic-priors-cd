"""C-P12: equal-error-variance identifiability sanity check.

Reuses the C-P11 fixture: a 3-node ER2 linear-Gaussian SCM at seed 0 with
homogeneous noise scale 1.0, and 5000 observational samples drawn with
sample_observational seed 1. Enumerates all DAGs on 3 nodes, scores each
one by an equal-error-variance BIC, and reports the rank of the true DAG
and the rank of the DAG that DCDI learned in C-P11.

The scoring assumes one shared residual variance across all nodes (the
equal-error-variance assumption known to identify a linear-Gaussian DAG
from observational data alone). For each candidate DAG and each node, an
ordinary-least-squares regression of that node on its candidate parents
(with intercept) is run on the same training data. The per-node residual
sums of squares are pooled into a single shared-variance MLE, which gives
a Gaussian log-likelihood. A BIC penalty is added for free parameters
(edge coefficients, per-node intercepts, and the single shared variance).

This probe is read-only with respect to project source and external
repositories. CPU only. No dependency is installed. This is a sanity
check, not a new baseline for the main study.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from symbolic_priors_cd.data.scm_generator import (  # noqa: E402
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.metrics.structural import shd  # noqa: E402
from symbolic_priors_cd.wrappers.dcdi import _is_acyclic_adjacency  # noqa: E402


N_NODES = 3
EXPECTED_EDGES = 3
SCM_SEED = 0
TRAIN_SIZE = 5000
TRAIN_SEED = 1

# The DCDI-learned thresholded adjacency at 0.5 from C-P11.
DCDI_LEARNED_ADJ = np.array(
    [
        [False, True, False],
        [False, False, False],
        [False, True, False],
    ]
)


def _enumerate_dags(n: int):
    """Yield every directed acyclic adjacency on n labelled nodes."""
    positions = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_off = len(positions)
    for mask in range(2 ** n_off):
        adj = np.zeros((n, n), dtype=bool)
        for k, (i, j) in enumerate(positions):
            if (mask >> k) & 1:
                adj[i, j] = True
        if _is_acyclic_adjacency(adj):
            yield adj


def _ols_rss(y: np.ndarray, X_parents: np.ndarray) -> float:
    """Return residual sum of squares after OLS regression of y on parents.

    A column of ones is prepended for the intercept. When the candidate
    parent set is empty, the model collapses to a constant equal to the
    sample mean, and the RSS is the empirical sum of squared deviations.
    """
    if X_parents.size == 0 or X_parents.shape[1] == 0:
        residuals = y - y.mean()
        return float((residuals ** 2).sum())
    X_design = np.column_stack([np.ones(len(y)), X_parents])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    residuals = y - X_design @ beta
    return float((residuals ** 2).sum())


def _score_dag(
    adj: np.ndarray, X: np.ndarray
) -> tuple[float, float, float, float, int]:
    """Score one candidate DAG under the equal-variance Gaussian assumption.

    Returns (bic, neg_log_lik, sigma2_hat, total_sse, n_edges). Lower BIC
    is better. The number of free parameters used in the BIC penalty is
    (edge coefficients) + (per-node intercepts) + (one shared variance).
    """
    n, d = X.shape
    total_sse = 0.0
    n_edges = 0
    for j in range(d):
        parents = np.where(adj[:, j])[0]
        n_edges += int(parents.size)
        total_sse += _ols_rss(X[:, j], X[:, parents])
    sigma2_hat = total_sse / (n * d)
    # Log-likelihood at the shared-variance MLE:
    #   log L = -n*d/2 * (log(2*pi*sigma2_hat) + 1)
    log_lik = -0.5 * n * d * (np.log(2.0 * np.pi * sigma2_hat) + 1.0)
    k_params = n_edges + d + 1
    bic = -2.0 * log_lik + k_params * np.log(n)
    return float(bic), float(-log_lik), float(sigma2_hat), float(total_sse), n_edges


def _adj_to_str(adj: np.ndarray) -> str:
    """Compact one-line string for a 3x3 bool adjacency."""
    return "[" + ", ".join(
        "[" + ",".join(str(int(x)) for x in row) + "]" for row in adj
    ) + "]"


def main():
    print("=" * 72)
    print("C-P12: equal-error-variance identifiability sanity check")
    print("=" * 72)

    scm = generate_linear_gaussian_scm(
        n_nodes=N_NODES, expected_edges=EXPECTED_EDGES, seed=SCM_SEED,
    )
    X = sample_observational(scm, n_samples=TRAIN_SIZE, rng=TRAIN_SEED)

    print()
    print(f"SCM: n_nodes={N_NODES}, expected_edges={EXPECTED_EDGES}, seed={SCM_SEED}")
    print(f"  noise_scale (homogeneous): {scm.noise_scale}")
    print(f"  true adjacency:\n{scm.adjacency.astype(int)}")
    print(f"  true weights:\n{np.round(scm.weights, 4)}")
    print(f"  topological order: {scm.topological_order}")
    print(f"Data: n_train={TRAIN_SIZE}, sample_observational seed={TRAIN_SEED}")
    print()
    print("Scoring formula (lower is better):")
    print("  total_sse = sum over j of OLS RSS of X[:,j] regressed on X[:,parents(j)]")
    print("  sigma2_hat = total_sse / (n * d)")
    print("  log_lik = -n*d/2 * (log(2*pi*sigma2_hat) + 1)")
    print("  k_params = n_edges + d + 1   (edges + intercepts + shared variance)")
    print("  BIC = -2 * log_lik + k_params * log(n)")

    # Score all DAGs
    candidates = []
    for adj in _enumerate_dags(N_NODES):
        bic, nll, sig2, sse, n_edges = _score_dag(adj, X)
        candidates.append({
            "adj": adj,
            "bic": bic,
            "nll": nll,
            "sigma2_hat": sig2,
            "total_sse": sse,
            "n_edges": n_edges,
            "shd_to_true": int(shd(predicted=adj, true=scm.adjacency)),
        })

    candidates.sort(key=lambda c: c["bic"])
    n_candidates = len(candidates)
    print()
    print(f"Total DAGs enumerated: {n_candidates}")

    # Find true DAG rank
    true_rank = None
    for i, c in enumerate(candidates):
        if np.array_equal(c["adj"], scm.adjacency):
            true_rank = i + 1
            true_entry = c
            break
    assert true_rank is not None, "True adjacency must appear in the enumeration."

    # Find DCDI learned DAG rank
    dcdi_rank = None
    for i, c in enumerate(candidates):
        if np.array_equal(c["adj"], DCDI_LEARNED_ADJ):
            dcdi_rank = i + 1
            dcdi_entry = c
            break
    assert dcdi_rank is not None, (
        "The C-P11 DCDI learned adjacency must appear in the enumeration."
    )

    top = candidates[0]
    print()
    print("Top-ranked DAG (rank 1):")
    print(f"  adjacency:\n{top['adj'].astype(int)}")
    print(f"  BIC = {top['bic']:.4f}")
    print(f"  n_edges = {top['n_edges']}")
    print(f"  sigma2_hat = {top['sigma2_hat']:.6f}")
    print(f"  total_sse = {top['total_sse']:.4f}")
    print(f"  SHD to true = {top['shd_to_true']}")

    print()
    print(f"True DAG rank: {true_rank} of {n_candidates}")
    print(f"  BIC = {true_entry['bic']:.4f}  (delta from top: "
          f"{true_entry['bic'] - top['bic']:+.4f})")
    print(f"  n_edges = {true_entry['n_edges']}")
    print(f"  sigma2_hat = {true_entry['sigma2_hat']:.6f}")
    print(f"  SHD to true = {true_entry['shd_to_true']}")

    print()
    print(f"DCDI-learned DAG rank: {dcdi_rank} of {n_candidates}")
    print(f"  adjacency: {_adj_to_str(DCDI_LEARNED_ADJ)}")
    print(f"  BIC = {dcdi_entry['bic']:.4f}  (delta from top: "
          f"{dcdi_entry['bic'] - top['bic']:+.4f})")
    print(f"  n_edges = {dcdi_entry['n_edges']}")
    print(f"  sigma2_hat = {dcdi_entry['sigma2_hat']:.6f}")
    print(f"  SHD to true = {dcdi_entry['shd_to_true']}")

    print()
    print("Top 10 DAGs (rank, BIC, delta_from_top, n_edges, SHD_to_true, adjacency):")
    for i, c in enumerate(candidates[:10]):
        delta = c["bic"] - top["bic"]
        print(f"  {i+1:2d}  BIC={c['bic']:.4f}  d={delta:+.4f}  "
              f"E={c['n_edges']}  SHD={c['shd_to_true']}  {_adj_to_str(c['adj'])}")

    # Margin between rank 1 and rank 2 to gauge separation strength.
    if n_candidates >= 2:
        margin = candidates[1]["bic"] - candidates[0]["bic"]
        print()
        print(f"Top score margin (rank 2 BIC - rank 1 BIC): {margin:+.4f}")
        print("  Positive and well above 0 means the top DAG is clearly preferred.")


if __name__ == "__main__":
    main()
