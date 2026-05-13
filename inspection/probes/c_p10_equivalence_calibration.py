"""C-P10: equivalence calibration probe.

Runs the wrapper training loop on the calibrated equivalence-test setup
and prints the observed schedule events. The output of this script is
recorded verbatim in docs/04e_equivalence_calibration_results.md.

The probe is read-only with respect to project source and external
repositories. CPU only. No dependency is installed.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from symbolic_priors_cd.data.scm_generator import (  # noqa: E402
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.wrappers._dcdi_training import (  # noqa: E402
    DCDIConfig,
    run_dcdi_training_loop,
)
from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model  # noqa: E402


def main() -> None:
    # SCM
    num_vars = 3
    expected_edges = 3
    scm_seed = 0

    # Data
    train_size = 64
    val_size = 64
    train_seed = 1
    val_seed = 2

    # Schedule window. The plan's starting suggestion of n_iter = 6 * stop_crit_win
    # = 120 does not fire any gamma/mu updates on this seed and data; the
    # window is enlarged so the test actually exercises the Lagrangian schedule.
    stop_crit_win = 20
    n_iter = 400
    batch_size = 32
    run_seed = 0

    scm = generate_linear_gaussian_scm(
        n_nodes=num_vars, expected_edges=expected_edges, seed=scm_seed,
    )
    X_train = sample_observational(scm, n_samples=train_size, rng=train_seed)
    X_val = sample_observational(scm, n_samples=val_size, rng=val_seed)

    config = DCDIConfig(
        stop_crit_win=stop_crit_win,
        train_batch_size=batch_size,
        lr=1e-3,
    )

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    model = make_dcdi_model(
        num_vars=num_vars, num_layers=2, hid_dim=8, nonlin="leaky-relu",
    )

    result = run_dcdi_training_loop(
        model, X_train, X_val,
        config=config, seed=run_seed, n_iter=n_iter,
    )

    print("=" * 60)
    print("C-P10 Equivalence Calibration")
    print("=" * 60)
    print(f"SCM: {num_vars} nodes, expected_edges={expected_edges}, seed={scm_seed}")
    print(f"X_train: shape {X_train.shape}, seed={train_seed}")
    print(f"X_val:   shape {X_val.shape}, seed={val_seed}")
    print(f"stop_crit_win = {stop_crit_win}")
    print(f"n_iter         = {n_iter}")
    print(f"batch_size     = {batch_size}")
    print(f"lr             = {config.lr}")
    print(f"run seed       = {run_seed}")
    print(f"Edges in true DAG: {int(scm.adjacency.sum())}")
    print()
    print(f"gamma_update_iters: {result.gamma_update_iters}")
    print(f"mu_update_iters   : {result.mu_update_iters}")
    print(f"converged          : {result.converged}")
    print(f"first_stop         : {result.first_stop}")
    print(f"final_h            : {result.final_h:.6e}")
    print(f"final_gamma        : {result.final_gamma:.6e}")
    print(f"final_mu           : {result.final_mu:.6e}")
    print(f"n_iterations       : {result.n_iterations}")


if __name__ == "__main__":
    main()
