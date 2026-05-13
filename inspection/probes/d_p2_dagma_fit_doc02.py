"""D-P2: Run DagmaLinear.fit with explicit Doc 02 hyperparameters on tiny data."""
import sys

import numpy as np

sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")

print("X shape:", X.shape)
print("True nonzero edges:", int(B.sum()))

model = DagmaLinear(loss_type="l2")
W_est = model.fit(
    X.copy(),
    lambda1=0.05,
    w_threshold=0.3,
    T=4,
    mu_init=1.0,
    mu_factor=0.1,
    s=[1.0, 0.9, 0.8, 0.7],
    warm_iter=2000,
    max_iter=4000,
    lr=3e-4,
)

print("W_est shape:", W_est.shape)
print("W_est dtype:", W_est.dtype)
print("W_est nnz:", int((W_est != 0).sum()))
print("W_est min nonzero abs:",
      float(np.abs(W_est[W_est != 0]).min()) if (W_est != 0).any() else "no nonzeros")
print("W_est max abs:", float(np.abs(W_est).max()))
print("h_final stored on model:", model.h_final)
