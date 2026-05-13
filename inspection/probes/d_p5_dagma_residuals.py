"""D-P5: Residual statistics for the noise/intercept policy (evidence only)."""
import sys

import numpy as np

sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

utils.set_random_seed(0)
B = utils.simulate_dag(d=5, s0=5, graph_type="ER")
W_true = utils.simulate_parameter(B)
X = utils.simulate_linear_sem(W_true, n=500, sem_type="gauss")

means = X.mean(axis=0).copy()
X_centred = X - means

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

# Residuals computed in the centred frame, against the thresholded W
# used for downstream sampling.
R = X_centred - X_centred @ W_est

print("residuals computed against thresholded W (w_threshold=0.3)")
print("per-variable raw means of X:", means)
print("per-variable residual variance:", R.var(axis=0))
print("per-variable residual std:", R.std(axis=0))
print("global mean residual var:", float(R.var(axis=0).mean()))
print("global mean residual std:", float(R.std(axis=0).mean()))
print("note: true SCM has unit-variance Gaussian noise; this gives a "
      "rough check on what residual-fitted noise would look like")
