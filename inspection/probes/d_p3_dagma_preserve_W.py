"""D-P3: Pre-threshold W preservation via w_threshold=0 vs w_threshold=0.3."""
import sys

import numpy as np

sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils


def fit_with(thresh: float) -> np.ndarray:
    utils.set_random_seed(0)
    B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
    W_true = utils.simulate_parameter(B)
    X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")
    model = DagmaLinear(loss_type="l2")
    return model.fit(
        X.copy(),
        lambda1=0.05,
        w_threshold=thresh,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        warm_iter=2000,
        max_iter=4000,
        lr=3e-4,
    )


print("=== fitting w_threshold=0.0 ===")
W0 = fit_with(0.0)
print("=== fitting w_threshold=0.3 ===")
W03 = fit_with(0.3)

abs_W0 = np.abs(W0)
abs_W03 = np.abs(W03)
small_nonzero = (abs_W0 > 1e-12) & (abs_W0 < 0.3)

nnz_0 = int((abs_W0 > 1e-12).sum())
nnz_03 = int((abs_W03 > 1e-12).sum())

print()
print("nnz at threshold 0.0:", nnz_0)
print("nnz at threshold 0.3:", nnz_03)
print("count of small nonzero entries (strictly between 1e-12 and 0.3) at threshold 0.0:",
      int(small_nonzero.sum()))
print("0.0 retains >= 0.3 nonzeros?", nnz_0 >= nnz_03)
print("min nonzero abs entry at threshold 0.0:",
      float(abs_W0[abs_W0 > 1e-12].min()) if nnz_0 > 0 else "no nonzeros")
print("max abs |W0 - W03|:", float(np.abs(W0 - W03).max()))
