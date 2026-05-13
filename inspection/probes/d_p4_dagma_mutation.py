"""D-P4: Input data mutation check."""
import sys

import numpy as np

sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=200, sem_type="gauss")
X_before = X.copy()
mean_before = X.mean(axis=0).copy()

model = DagmaLinear(loss_type="l2")
_ = model.fit(
    X,
    lambda1=0.05,
    w_threshold=0.3,
    T=2,
    mu_init=1.0,
    mu_factor=0.1,
    s=[1.0, 0.9],
    warm_iter=200,
    max_iter=200,
    lr=3e-4,
)

print("array values unchanged?", bool(np.allclose(X, X_before)))
print("X.mean(axis=0) after fit:", X.mean(axis=0))
print("X.mean(axis=0) before fit:", mean_before)
print("max abs(post-fit mean):", float(np.abs(X.mean(axis=0)).max()))
print("max abs(X - X_before):", float(np.abs(X - X_before).max()))
