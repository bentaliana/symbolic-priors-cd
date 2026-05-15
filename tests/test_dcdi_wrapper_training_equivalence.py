"""Tests for DCDI training-loop determinism and behavioural equivalence.

The behavioural-equivalence tests compare run_dcdi_training_loop against
a hand-replicated reference loop derived from the inspected DCDI source
at external/source_inspection/dcdi/dcdi/train.py 

Calibration constants (SCM and data seeds, iteration window, expected
schedule events) are fixed through inspecting inspection/probes/c_p10_equivalence_calibration.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.wrappers._dcdi_training import (
    DCDIConfig,
    run_dcdi_training_loop,
)
from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model

# These imports resolve from the pinned DCDI source because _dcdi_utils
# set up sys.path at import time.
from dcdi.dag_optim import compute_dag_constraint  # noqa: E402
from dcdi.utils.penalty import compute_penalty  # noqa: E402


# ---------------------------------------------------------------------------
# Calibration constants 
# ---------------------------------------------------------------------------

_NUM_VARS = 3
_EXPECTED_EDGES = 3
_SCM_SEED = 0
_TRAIN_SIZE = 64
_VAL_SIZE = 64
_TRAIN_SEED = 1
_VAL_SEED = 2
_STOP_CRIT_WIN = 20
_N_ITER = 400
_BATCH_SIZE = 32
_LR = 1e-3
_RUN_SEED = 0

_EARLY_ITERS = (0, 1, 2, 5, 10)
_MID_ITERS = (50, 100, 150)


# ---------------------------------------------------------------------------
# Existing tests (determinism and input validation)
# ---------------------------------------------------------------------------


def test_deterministic_tiny_run():
    """Two same-seed runs produce equal continuous_log_alpha_pre_threshold.

    Verifies that the wrapper's training loop is deterministic on CPU when
    both torch and numpy seeds are set consistently. The check uses bitwise
    tensor equality on the preserved continuous edge objects.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((32, 3)).astype(np.float64)
    X_val = rng.standard_normal((16, 3)).astype(np.float64)
    config = DCDIConfig(stop_crit_win=10, train_batch_size=8)

    torch.manual_seed(0)
    np.random.seed(0)
    model1 = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    result1 = run_dcdi_training_loop(
        model1, X_train, X_val, config=config, seed=0, n_iter=20,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    model2 = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    result2 = run_dcdi_training_loop(
        model2, X_train, X_val, config=config, seed=0, n_iter=20,
    )

    assert torch.equal(
        result1.continuous_log_alpha_pre_threshold,
        result2.continuous_log_alpha_pre_threshold,
    ), "Same-seed runs produced different continuous_log_alpha_pre_threshold."

    assert torch.equal(
        result1.continuous_w_adj_pre_threshold,
        result2.continuous_w_adj_pre_threshold,
    ), "Same-seed runs produced different continuous_w_adj_pre_threshold."


def test_run_dcdi_training_loop_validates_input_shapes():
    """run_dcdi_training_loop raises ValueError for malformed X_train or X_val."""
    rng = np.random.default_rng(0)
    config = DCDIConfig(stop_crit_win=10, train_batch_size=8)

    torch.manual_seed(0)
    np.random.seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)

    good_X_train = rng.standard_normal((20, 3)).astype(np.float64)
    good_X_val = rng.standard_normal((10, 3)).astype(np.float64)

    bad_X_train_cols = rng.standard_normal((20, 5)).astype(np.float64)
    with pytest.raises(ValueError, match="X_train.shape"):
        run_dcdi_training_loop(
            model, bad_X_train_cols, good_X_val, config=config, seed=0, n_iter=5,
        )

    bad_X_val_cols = rng.standard_normal((10, 5)).astype(np.float64)
    with pytest.raises(ValueError, match="X_val.shape"):
        run_dcdi_training_loop(
            model, good_X_train, bad_X_val_cols, config=config, seed=0, n_iter=5,
        )

    bad_X_train_1d = rng.standard_normal(20).astype(np.float64)
    with pytest.raises(ValueError, match="X_train must be 2D"):
        run_dcdi_training_loop(
            model, bad_X_train_1d, good_X_val, config=config, seed=0, n_iter=5,
        )

    bad_X_val_1d = rng.standard_normal(10).astype(np.float64)
    with pytest.raises(ValueError, match="X_val must be 2D"):
        run_dcdi_training_loop(
            model, good_X_train, bad_X_val_1d, config=config, seed=0, n_iter=5,
        )

    too_small = rng.standard_normal((1, 3)).astype(np.float64)
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_dcdi_training_loop(
            model, too_small, good_X_val, config=config, seed=0, n_iter=5,
        )


# ---------------------------------------------------------------------------
# Shared equivalence-test setup
# ---------------------------------------------------------------------------


def _make_equivalence_data():
    """Generate the calibrated training and validation observations."""
    scm = generate_linear_gaussian_scm(
        n_nodes=_NUM_VARS, expected_edges=_EXPECTED_EDGES, seed=_SCM_SEED,
    )
    X_train = sample_observational(scm, n_samples=_TRAIN_SIZE, rng=_TRAIN_SEED)
    X_val = sample_observational(scm, n_samples=_VAL_SIZE, rng=_VAL_SEED)
    return X_train, X_val


def _make_batch_indices(seed: int = _RUN_SEED) -> list[np.ndarray]:
    """Pre-generate the deterministic minibatch index sequence.

    Both the wrapper loop and the reference loop must index into X_train
    using exactly this sequence. Different sequences are a test setup
    error, not an acceptable model deviation.
    """
    rng = np.random.default_rng(seed)
    return [
        rng.choice(_TRAIN_SIZE, size=_BATCH_SIZE, replace=False)
        for _ in range(_N_ITER)
    ]


def _make_equivalence_config() -> DCDIConfig:
    return DCDIConfig(
        stop_crit_win=_STOP_CRIT_WIN,
        train_batch_size=_BATCH_SIZE,
        lr=_LR,
    )


# ---------------------------------------------------------------------------
# Hand-replicated reference loop
# ---------------------------------------------------------------------------


def _reference_is_acyclic(adj: np.ndarray) -> bool:
    """Acyclicity test via successive matrix powers."""
    d = adj.shape[0]
    a = adj.astype(np.int64)
    prod = np.eye(d, dtype=np.int64)
    for _ in range(d):
        prod = prod @ a
        if np.trace(prod) != 0:
            return False
    return True


def _reference_eval_val(model, x_val_t: torch.Tensor) -> float:
    """Validation NLL in eval mode; restores training mode afterwards."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            w_, b_, e_ = model.get_parameters(mode="wbx")
            val_log_lik = model.compute_log_likelihood(x_val_t, w_, b_, e_)
            return float((-val_log_lik.mean()).item())
    finally:
        if was_training:
            model.train()


def reference_training_loop(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    config: DCDIConfig,
    seed: int,
    n_iter: int,
    batch_indices,
    log_alpha_snapshots_at=(),
) -> dict:
    """Hand-replicated DCDI training loop, independent of the wrapper.

    The augmented-Lagrangian forward and backward steps mirror
    dcdi/train.py:124-162. The periodic stop check mirrors
    dcdi/train.py:178-200. The delta_gamma plateau test and the gamma/mu
    update gate mirror dcdi/train.py:189-199 and 269-296. Conventions
    shared with the wrapper:
    - post-step h and w_adj for the stop check
    - initial validation evaluation before the loop
    - validation NLL (not val augmented Lagrangian) for delta_gamma
    - model.eval() during validation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    num_vars = model.num_vars

    ones_minus_eye = torch.ones((num_vars, num_vars)) - torch.eye(num_vars)
    constraint_norm = compute_dag_constraint(ones_minus_eye).detach().item()

    optimiser = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    model.train()

    gamma = config.gamma_init
    mu = config.mu_init
    patience = config.train_patience

    val_history: list[float] = []
    loss_history: list[float] = []
    constraint_violation_list: list[float] = []
    gamma_update_iters: list[int] = []
    mu_update_iters: list[int] = []
    log_alpha_snapshots: dict[int, torch.Tensor] = {}
    snapshot_set = set(int(i) for i in log_alpha_snapshots_at)
    first_stop: int | None = None
    converged = False
    final_h = 0.0
    iterations_completed = 0

    val_history.append(_reference_eval_val(model, x_val_t))

    for it in range(n_iter):
        iterations_completed = it + 1

        if it in snapshot_set:
            log_alpha_snapshots[it] = (
                model.gumbel_adjacency.log_alpha.detach().cpu().clone()
            )

        x_batch = x_train_t[batch_indices[it]]

        # Augmented Lagrangian forward.
        w_, b_, e_ = model.get_parameters(mode="wbx")
        log_lik = model.compute_log_likelihood(x_batch, w_, b_, e_)
        nll = -log_lik.mean()
        w_adj = model.get_w_adj()
        h = compute_dag_constraint(w_adj) / constraint_norm
        reg = config.reg_coeff * compute_penalty([w_adj], p=1) / (num_vars ** 2)
        prior_zero = w_adj.new_zeros(())
        aug = nll + reg + prior_zero + gamma * h + 0.5 * mu * h ** 2

        # Backward and optimiser step.
        optimiser.zero_grad()
        aug.backward()
        optimiser.step()
        loss_history.append(float(aug.item()))

        if (it + 1) % config.stop_crit_win == 0:
            # Post-step h and w_adj for the stop check.
            with torch.no_grad():
                current_w = model.get_w_adj()
                h_value = float(
                    (compute_dag_constraint(current_w) / constraint_norm).item()
                )
                adj_bool = current_w.cpu().numpy() > 0.5
            graph_acyclic = _reference_is_acyclic(adj_bool)
            final_h = h_value

            val_history.append(_reference_eval_val(model, x_val_t))

            if h_value <= config.h_threshold and graph_acyclic:
                if first_stop is None:
                    first_stop = it + 1
                patience -= 1
                if patience <= 0:
                    converged = True
                    break
            else:
                # delta_gamma over last three validation NLLs.
                if (
                    len(val_history) >= 3
                    and (it + 1) % (2 * config.stop_crit_win) == 0
                ):
                    t0, t_half, t1 = val_history[-3], val_history[-2], val_history[-1]
                    if min(t0, t1) < t_half < max(t0, t1):
                        delta_gamma = (t1 - t0) / config.stop_crit_win
                    else:
                        delta_gamma = float("-inf")
                else:
                    delta_gamma = float("-inf")

                # Gamma update on plateau or worsening; mu update only when
                # the constraint did not improve by factor omega_mu.
                if abs(delta_gamma) < config.omega_gamma or delta_gamma > 0:
                    gamma = gamma + mu * h_value
                    gamma_update_iters.append(it + 1)

                    constraint_violation_list.append(h_value)
                    if len(constraint_violation_list) >= 2:
                        prev_violation = constraint_violation_list[-2]
                        if h_value > prev_violation * config.omega_mu:
                            mu = mu * config.mu_mult_factor
                            mu_update_iters.append(it + 1)

    # Match the wrapper's post-loop final_h recompute: ensures final_h
    # reflects the post-step model state even when n_iter is not a
    # multiple of stop_crit_win. No-op for loops that ended on a stop
    # check.
    with torch.no_grad():
        final_h = float(
            (compute_dag_constraint(model.get_w_adj()) / constraint_norm).item()
        )

    return {
        "log_alpha": model.gumbel_adjacency.log_alpha.detach().cpu().clone(),
        "w_adj": model.get_w_adj().detach().cpu().clone(),
        "loss_history": loss_history,
        "val_history": val_history,
        "final_gamma": float(gamma),
        "final_mu": float(mu),
        "final_h": final_h,
        "first_stop": first_stop,
        "converged": converged,
        "n_iterations": iterations_completed,
        "gamma_update_iters": gamma_update_iters,
        "mu_update_iters": mu_update_iters,
        "log_alpha_snapshots": log_alpha_snapshots,
    }


# ---------------------------------------------------------------------------
# Pair-run helper
# ---------------------------------------------------------------------------


def _run_wrapper_and_reference(snapshots_at=()):
    """Run both loops with shared data, model init, and batch sequence."""
    X_train, X_val = _make_equivalence_data()
    batch_indices = _make_batch_indices()
    config = _make_equivalence_config()

    torch.manual_seed(_RUN_SEED)
    np.random.seed(_RUN_SEED)
    model_w = make_dcdi_model(
        num_vars=_NUM_VARS, num_layers=2, hid_dim=8, nonlin="leaky-relu",
    )
    result_w = run_dcdi_training_loop(
        model_w, X_train, X_val,
        config=config, seed=_RUN_SEED, n_iter=_N_ITER,
        batch_indices=batch_indices,
        log_alpha_snapshots_at=snapshots_at,
    )

    torch.manual_seed(_RUN_SEED)
    np.random.seed(_RUN_SEED)
    model_r = make_dcdi_model(
        num_vars=_NUM_VARS, num_layers=2, hid_dim=8, nonlin="leaky-relu",
    )
    result_r = reference_training_loop(
        model_r, X_train, X_val,
        config=config, seed=_RUN_SEED, n_iter=_N_ITER,
        batch_indices=batch_indices,
        log_alpha_snapshots_at=snapshots_at,
    )

    return result_w, result_r


# ---------------------------------------------------------------------------
# Behavioural-equivalence tests
# ---------------------------------------------------------------------------


def test_behavioural_equivalence_no_prior_early():
    """log_alpha matches bitwise at iterations 0, 1, 2, 5, 10.

    Bitwise failure here indicates an implementation bug: the wrapper and
    the hand-replicated reference loop computed differently at the very
    first training steps. The early-iter check guards against algorithmic
    drift (objective composition, optimiser ordering, RNG consumption).
    """
    result_w, result_r = _run_wrapper_and_reference(snapshots_at=_EARLY_ITERS)

    for it in _EARLY_ITERS:
        assert it in result_w.log_alpha_snapshots
        assert it in result_r["log_alpha_snapshots"]
        wrapper_la = result_w.log_alpha_snapshots[it]
        reference_la = result_r["log_alpha_snapshots"][it]
        assert torch.equal(wrapper_la, reference_la), (
            f"Wrapper and reference log_alpha diverged at iteration {it} "
            f"(early-iter bitwise check)."
        )


def test_behavioural_equivalence_no_prior_mid():
    """log_alpha matches within atol=rtol=1e-4 at iterations 50, 100, 150."""
    result_w, result_r = _run_wrapper_and_reference(snapshots_at=_MID_ITERS)

    for it in _MID_ITERS:
        wrapper_la = result_w.log_alpha_snapshots[it]
        reference_la = result_r["log_alpha_snapshots"][it]
        max_diff = (wrapper_la - reference_la).abs().max().item()
        assert torch.allclose(
            wrapper_la, reference_la, atol=1e-4, rtol=1e-4,
        ), (
            f"Wrapper and reference log_alpha diverged beyond mid tolerance "
            f"at iteration {it}; max abs diff = {max_diff:.3e}"
        )


def test_behavioural_equivalence_no_prior_final():
    """Final log_alpha and w_adj agree within atol=rtol=1e-3 plus an L2 bound.

    Tolerances absorb accumulated floating-point non-associativity across
    the full RMSprop trajectory; tighter bounds are not assumed because
    the deviation between two independent implementations of the same
    arithmetic can grow with iteration count.
    """
    result_w, result_r = _run_wrapper_and_reference()

    log_alpha_w = result_w.continuous_log_alpha_pre_threshold
    log_alpha_r = result_r["log_alpha"]

    max_diff = (log_alpha_w - log_alpha_r).abs().max().item()
    assert torch.allclose(
        log_alpha_w, log_alpha_r, atol=1e-3, rtol=1e-3,
    ), (
        f"Wrapper and reference final log_alpha exceeded element-wise "
        f"tolerance; max abs diff = {max_diff:.3e}"
    )

    d = log_alpha_w.shape[0]
    l2 = torch.norm(log_alpha_w - log_alpha_r).item()
    normalised_l2 = l2 / float(np.sqrt(d * d))
    assert normalised_l2 <= 1e-3, (
        f"Normalised L2 norm of log_alpha difference {normalised_l2:.3e} "
        f"exceeds bound 1e-3."
    )

    assert torch.allclose(
        result_w.continuous_w_adj_pre_threshold,
        result_r["w_adj"],
        atol=1e-3, rtol=1e-3,
    ), "Wrapper and reference final w_adj diverged beyond tolerance."


def test_lagrangian_schedule_equivalence():
    """Gamma- and mu-update iteration indices match exactly between loops.

    A schedule mismatch is an algorithmic divergence, not a numerical
    artefact, and must fail regardless of trajectory closeness. This test verifies both loops reproduce the same events.
    """
    result_w, result_r = _run_wrapper_and_reference()

    assert result_w.gamma_update_iters == result_r["gamma_update_iters"], (
        f"Wrapper gamma_update_iters = {result_w.gamma_update_iters} "
        f"differs from reference = {result_r['gamma_update_iters']}"
    )

    assert result_w.mu_update_iters == result_r["mu_update_iters"], (
        f"Wrapper mu_update_iters = {result_w.mu_update_iters} "
        f"differs from reference = {result_r['mu_update_iters']}"
    )

    # Sanity: the calibration probe recorded specific events; any correct
    # implementation of the training loop must reproduce them.
    assert result_w.gamma_update_iters == [280, 400], (
        f"Observed gamma_update_iters = {result_w.gamma_update_iters} "
        "differs from the calibration record [280, 400]; "
        "re-run inspection/probes/c_p10_equivalence_calibration.py "
        "if the observed events have drifted."
    )
    assert result_w.mu_update_iters == [400], (
        f"Observed mu_update_iters = {result_w.mu_update_iters} "
        "differs from the calibration record [400]; "
        "re-run the C-P10 probe and update the calibration document."
    )


def test_behavioural_equivalence_metadata_and_loss_history():
    """Wrapper and reference produce the same training metadata and loss history.

    Strict equality on deterministic discrete fields (iteration counts,
    convergence flags, loss-history length, mu). Floating tolerances on
    final_gamma, final_h, and loss-history checkpoints are chosen to
    match the tolerances already used for the log_alpha trajectory:
    bitwise on early iters, atol 1e-4 mid, atol 1e-3 final.
    """
    result_w, result_r = _run_wrapper_and_reference()

    # Discrete metadata: strict equality.
    assert result_w.n_iterations == result_r["n_iterations"]
    assert result_w.converged == result_r["converged"]
    assert result_w.first_stop == result_r["first_stop"]
    assert len(result_w.loss_history) == len(result_r["loss_history"])

    # mu is a chain of identical multiplications by mu_mult_factor; both
    # loops apply the same number of mu updates at identical iterations,
    # so final_mu is bitwise equal.
    assert result_w.final_mu == result_r["final_mu"], (
        f"final_mu diff: wrapper={result_w.final_mu!r}, "
        f"reference={result_r['final_mu']!r}"
    )

    # final_h and final_gamma read post-step h values that carry
    # accumulated floating-point error from the trajectory.
    h_diff = abs(result_w.final_h - result_r["final_h"])
    assert h_diff <= 1e-4, (
        f"final_h diff {h_diff:.3e} exceeds tolerance 1e-4; "
        f"wrapper={result_w.final_h}, reference={result_r['final_h']}"
    )
    gamma_diff = abs(result_w.final_gamma - result_r["final_gamma"])
    assert gamma_diff <= 1e-11, (
        f"final_gamma diff {gamma_diff:.3e} exceeds tolerance 1e-11; "
        f"wrapper={result_w.final_gamma}, reference={result_r['final_gamma']}"
    )

    # loss_history checkpoints, with tolerances mirroring the log_alpha
    # trajectory bands.
    n_losses = len(result_w.loss_history)
    final_idx = n_losses - 1

    for it in _EARLY_ITERS:
        if it >= n_losses:
            continue
        w_loss = result_w.loss_history[it]
        r_loss = result_r["loss_history"][it]
        assert w_loss == r_loss, (
            f"loss_history[{it}] bitwise mismatch: "
            f"wrapper={w_loss!r}, reference={r_loss!r}"
        )

    for it in _MID_ITERS:
        if it >= n_losses:
            continue
        w_loss = result_w.loss_history[it]
        r_loss = result_r["loss_history"][it]
        diff = abs(w_loss - r_loss)
        assert diff <= 1e-4, (
            f"loss_history[{it}] diff {diff:.3e} exceeds 1e-4; "
            f"wrapper={w_loss}, reference={r_loss}"
        )

    w_final_loss = result_w.loss_history[final_idx]
    r_final_loss = result_r["loss_history"][final_idx]
    final_diff = abs(w_final_loss - r_final_loss)
    assert final_diff <= 1e-3, (
        f"loss_history[{final_idx}] diff {final_diff:.3e} exceeds 1e-3; "
        f"wrapper={w_final_loss}, reference={r_final_loss}"
    )
