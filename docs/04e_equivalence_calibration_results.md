# Equivalence Calibration Results (C-P10)

## Purpose

Capture and freeze the iteration window, SCM/data seeds, and observed
schedule events used by the wrapper behavioural-equivalence tests in
`tests/test_dcdi_wrapper_training_equivalence.py`. The probe script is
`inspection/probes/c_p10_equivalence_calibration.py`. The probe is
read-only with respect to project source and external repositories and
runs on CPU without installing dependencies.

If the probe is re-run on the same project commit and same environment,
the observed events must reproduce verbatim. If they drift, this
document and the test must be updated together.

## Configuration (frozen)

### SCM

- Family: ER (Erdos-Renyi)
- Mechanism: linear Gaussian
- Node count: 3
- Expected edges: 3
- Generation seed: 0
- Edges actually present in the generated DAG: 3

### Data

- X_train shape: (64, 3); sample_observational seed = 1
- X_val   shape: (64, 3); sample_observational seed = 2

### Model

- LearnableModel_NonLinGaussANM
- num_layers = 2
- hid_dim = 8
- nonlin = "leaky-relu"
- Observational mode: intervention=False, intervention_type="perfect",
  intervention_knowledge="known", num_regimes=1

### DCDIConfig values used

- h_threshold = 1e-8
- mu_init = 1e-8
- mu_mult_factor = 2.0
- gamma_init = 0.0
- omega_gamma = 1e-4
- omega_mu = 0.9
- lr = 1e-3
- train_batch_size = 32
- train_patience = 5
- stop_crit_win = 20
- reg_coeff = 0.1
- num_layers = 2 (unused at this layer; the model is constructed separately)
- hid_dim = 16 (unused at this layer; the model uses hid_dim=8 directly)
- nonlin = "leaky-relu" (unused at this layer)

### Iteration window

- n_iter = 400

The plan's starting suggestion of n_iter = 6 * stop_crit_win = 120 does
not fire any gamma/mu update on this seed and data, because the
validation NLL is still monotonically decreasing across the first three
windows (delta_gamma stays negative and larger in magnitude than
omega_gamma). The window is enlarged to n_iter = 400 so the test
actually exercises the Lagrangian schedule branch. The trajectory at
iterations 50, 100, and 150 is still verified within the mid-tolerance
band, and the final-trajectory check fires at iteration 400.

### Determinism

- torch.manual_seed(0) before model construction
- np.random.seed(0) before model construction
- torch.manual_seed(0) inside the training loop
- np.random.seed(0) inside the training loop
- Batch index sequence pre-generated from seed 0 using
  np.random.default_rng(0).choice(64, size=32, replace=False)

## Observed events

- gamma_update_iters = [280, 400]
- mu_update_iters    = [400]
- converged          = False
- first_stop         = None
- final_h            = 9.825867e-01
- final_gamma        = 1.964884e-08
- final_mu           = 2.000000e-08
- n_iterations       = 400

h_threshold is NOT reached within the calibration window; the test still
compares the trajectory and the update schedule over the fixed iteration
count. Convergence semantics on a larger setting are exercised by the
full-convergence integration test in a separate commit.

## Final equivalence-test window

- num_vars = 3
- expected_edges = 3
- scm_seed = 0
- train_seed = 1
- val_seed = 2
- train_size = 64
- val_size = 64
- batch_size = 32
- stop_crit_win = 20
- n_iter = 400
- run_seed = 0
- Iteration indices for early-iter bitwise checks: 0, 1, 2, 5, 10
- Iteration indices for mid-trajectory checks: 50, 100, 150
- Iteration index for final-trajectory check: 400 (= n_iter)
