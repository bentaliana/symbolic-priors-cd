"""DCDI augmented-Lagrangian training loop for the wrapper layer.

Implements observational-mode training. Captures the continuous native
edge objects (log_alpha and get_w_adj()) as detached CPU clones at
training exit. log_alpha is never permanently saturated, so the
preserved tensors reflect the true learned continuous state and remain
usable for threshold robustness reporting and downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch

# Importing _dcdi_utils first sets up and verifies the DCDI sys.path entry.
from symbolic_priors_cd.wrappers._dcdi_utils import (
    LearnableModel_NonLinGaussANM,
    snapshot_log_alpha,
    snapshot_w_adj,
)

# DCDI low-level imports below depend on the sys.path setup above.
from dcdi.dag_optim import compute_dag_constraint  # noqa: E402
from dcdi.utils.penalty import compute_penalty  # noqa: E402


@dataclass(frozen=True)
class DCDIConfig:
    """Tactical hyperparameters for the DCDI augmented-Lagrangian loop.

    Defaults correspond to the DCDI-G observational training configuration.
    """

    h_threshold: float = 1e-8
    mu_init: float = 1e-8
    mu_mult_factor: float = 2.0
    gamma_init: float = 0.0
    omega_gamma: float = 1e-4
    omega_mu: float = 0.9
    lr: float = 1e-3
    train_batch_size: int = 64
    train_patience: int = 5
    stop_crit_win: int = 100
    reg_coeff: float = 0.1
    num_layers: int = 2
    hid_dim: int = 16
    nonlin: str = "leaky-relu"


@dataclass
class TrainingResult:
    """Outputs of run_dcdi_training_loop.

    The continuous edge tensors are detached CPU clones captured at exit;
    they are fully independent of the live model parameter buffers.

    ``gamma_update_iters`` and ``mu_update_iters`` record the 1-indexed
    iteration at which each Lagrangian coefficient was updated.
    ``log_alpha_snapshots`` is populated only at iteration indices listed
    in ``log_alpha_snapshots_at`` and stays empty otherwise.
    """

    continuous_log_alpha_pre_threshold: torch.Tensor
    continuous_w_adj_pre_threshold: torch.Tensor
    n_iterations: int
    loss_history: list[float]
    converged: bool
    first_stop: Optional[int]
    final_gamma: float
    final_mu: float
    final_h: float
    gamma_update_iters: list[int]
    mu_update_iters: list[int]
    log_alpha_snapshots: dict[int, torch.Tensor]
    # Per-evaluation validation NLL trajectory captured at the same
    # cadence the training loop already evaluates it: one pre-training
    # baseline followed by one value every stop_crit_win iterations.
    # The list is read-only diagnostic output; the loop does not
    # consume it after collection.
    validation_nll_history: list[float]


def _is_acyclic_bool(adj: np.ndarray) -> bool:
    """Return True if a boolean adjacency matrix is acyclic.

    Builds successive matrix powers up to size d; if any has a non-zero
    trace, a directed cycle exists.
    """
    d = adj.shape[0]
    a = adj.astype(np.int64)
    prod = np.eye(d, dtype=np.int64)
    for _ in range(d):
        prod = prod @ a
        if np.trace(prod) != 0:
            return False
    return True


def _compute_augmented_lagrangian(
    model: LearnableModel_NonLinGaussANM,
    x_batch: torch.Tensor,
    num_vars: int,
    constraint_norm: float,
    config: DCDIConfig,
    gamma: float,
    mu: float,
    loss_hook: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Compute one training step's augmented Lagrangian objective."""
    weights, biases, extra_params = model.get_parameters(mode="wbx")
    log_lik = model.compute_log_likelihood(x_batch, weights, biases, extra_params)
    nll = -log_lik.mean()
    w_adj = model.get_w_adj()
    h = compute_dag_constraint(w_adj) / constraint_norm
    reg = config.reg_coeff * compute_penalty([w_adj], p=1) / (num_vars ** 2)
    # The no-hook prior tensor takes its device and dtype from w_adj so the
    # augmented objective stays on a single device under future variants.
    prior = loss_hook(w_adj) if loss_hook is not None else w_adj.new_zeros(())
    return nll + reg + prior + gamma * h + 0.5 * mu * h ** 2


def _evaluate_val_nll(
    model: LearnableModel_NonLinGaussANM,
    x_val_t: torch.Tensor,
) -> float:
    """Compute the mean validation NLL using a no-grad forward pass.

    Switches the model to eval mode for the duration of the call and
    restores its previous mode afterwards. Harmless for the current
    Gaussian-ANM model (no dropout or batchnorm) and defensive for
    future variants.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            val_log_lik = model.compute_log_likelihood(
                x_val_t, weights, biases, extra_params
            )
            return float((-val_log_lik.mean()).item())
    finally:
        if was_training:
            model.train()


def _validate_inputs(
    model: LearnableModel_NonLinGaussANM,
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> None:
    """Raise ValueError for malformed training or validation arrays."""
    num_vars = model.num_vars
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got {X_train.ndim}D.")
    if X_val.ndim != 2:
        raise ValueError(f"X_val must be 2D, got {X_val.ndim}D.")
    if X_train.shape[1] != num_vars:
        raise ValueError(
            f"X_train.shape[1] must equal model.num_vars={num_vars}, "
            f"got {X_train.shape[1]}."
        )
    if X_val.shape[1] != num_vars:
        raise ValueError(
            f"X_val.shape[1] must equal model.num_vars={num_vars}, "
            f"got {X_val.shape[1]}."
        )
    if X_train.shape[0] < 2:
        raise ValueError(
            f"X_train must have at least 2 rows, got {X_train.shape[0]}."
        )


def run_dcdi_training_loop(
    model: LearnableModel_NonLinGaussANM,
    X_train: np.ndarray,
    X_val: np.ndarray,
    *,
    config: DCDIConfig,
    seed: int,
    n_iter: int,
    loss_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    batch_indices: Optional[Sequence[np.ndarray]] = None,
    log_alpha_snapshots_at: Optional[Iterable[int]] = None,
) -> TrainingResult:
    """Run the DCDI observational training loop.

    Optimises the augmented Lagrangian objective. Captures the continuous
    native edge objects as detached CPU clones at exit without permanently
    saturating log_alpha.

    Parameters
    ----------
    model : LearnableModel_NonLinGaussANM
        Model in observational mode; CPU expected.
    X_train : np.ndarray
        Training data already in the model frame, shape (n_train, num_vars).
    X_val : np.ndarray
        Validation data already in the model frame, shape (n_val, num_vars).
    config : DCDIConfig
        Tactical hyperparameters.
    seed : int
        Random seed used to reset torch and numpy state and to generate
        the minibatch index sequence when ``batch_indices`` is None.
    n_iter : int
        Maximum number of training iterations.
    loss_hook : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Optional additive penalty acting on get_w_adj(); receives the
        continuous (num_vars, num_vars) tensor and returns a scalar tensor.
    batch_indices : Optional[Sequence[np.ndarray]]
        Pre-generated minibatch index sequence with at least ``n_iter``
        entries. If None, the sequence is generated internally from
        ``seed`` using a private numpy Generator.
    log_alpha_snapshots_at : Optional[Iterable[int]]
        Iteration indices at which to capture a detached CPU clone of
        ``model.gumbel_adjacency.log_alpha`` at the start of the iteration
        (before that iteration's training step). Returned in
        ``TrainingResult.log_alpha_snapshots``.

    Returns
    -------
    TrainingResult
        Includes preserved continuous edge tensors and training metadata.
    """
    _validate_inputs(model, X_train, X_val)

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_vars = model.num_vars
    n_train = X_train.shape[0]

    batch_size = min(config.train_batch_size, n_train)
    if batch_indices is None:
        rng = np.random.default_rng(seed)
        batch_indices = [
            rng.choice(n_train, size=batch_size, replace=False)
            for _ in range(n_iter)
        ]
    elif len(batch_indices) < n_iter:
        raise ValueError(
            f"batch_indices length {len(batch_indices)} is smaller than "
            f"n_iter={n_iter}."
        )

    snapshot_set = (
        set(int(i) for i in log_alpha_snapshots_at)
        if log_alpha_snapshots_at is not None
        else set()
    )
    log_alpha_snapshots: dict[int, torch.Tensor] = {}
    gamma_update_iters: list[int] = []
    mu_update_iters: list[int] = []

    x_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(X_val, dtype=torch.float32)

    ones_minus_eye = torch.ones((num_vars, num_vars)) - torch.eye(num_vars)
    constraint_norm = compute_dag_constraint(ones_minus_eye).detach().item()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    model.train()

    gamma = config.gamma_init
    mu = config.mu_init
    patience = config.train_patience

    val_history: list[float] = []
    loss_history: list[float] = []
    constraint_violation_list: list[float] = []
    first_stop: Optional[int] = None
    converged = False
    final_h = 0.0
    iterations_completed = 0

    # Initial validation evaluation at the pre-training model state, so the
    # plateau test below has enough history to fire on the same schedule the
    # inspected DCDI loop uses (3 evaluations spanning 2 * stop_crit_win steps).
    val_history.append(_evaluate_val_nll(model, x_val_t))

    for it in range(n_iter):
        iterations_completed = it + 1

        if it in snapshot_set:
            log_alpha_snapshots[it] = (
                model.gumbel_adjacency.log_alpha.detach().cpu().clone()
            )

        x_batch = x_train_t[batch_indices[it]]

        aug = _compute_augmented_lagrangian(
            model, x_batch, num_vars, constraint_norm, config, gamma, mu, loss_hook,
        )
        optimizer.zero_grad()
        aug.backward()
        optimizer.step()
        loss_history.append(float(aug.item()))

        if (it + 1) % config.stop_crit_win == 0:
            # Recompute h and w_adj on the POST-step model state so the stop
            # check (h_value, graph_acyclic, val NLL) is internally consistent.
            with torch.no_grad():
                current_w_adj = model.get_w_adj()
                current_h_tensor = compute_dag_constraint(current_w_adj) / constraint_norm
                h_value = float(current_h_tensor.item())
                adj_bool = current_w_adj.cpu().numpy() > 0.5
            graph_acyclic = _is_acyclic_bool(adj_bool)
            final_h = h_value

            val_history.append(_evaluate_val_nll(model, x_val_t))

            if h_value <= config.h_threshold and graph_acyclic:
                if first_stop is None:
                    first_stop = it + 1
                patience -= 1
                if patience <= 0:
                    converged = True
                    break
            else:
                # Compute delta_gamma over the last three validation NLLs
                # (t0, t_half, t1). A monotone window (t_half strictly between
                # t0 and t1) yields the per-window rate of change; a window
                # that went up-and-down yields -inf and suppresses any update.
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

                # Plateau (small magnitude) or worsening (positive) triggers a
                # gamma update; clear improvement leaves coefficients alone.
                if abs(delta_gamma) < config.omega_gamma or delta_gamma > 0:
                    gamma = gamma + mu * h_value
                    gamma_update_iters.append(it + 1)

                    constraint_violation_list.append(h_value)
                    if len(constraint_violation_list) >= 2:
                        prev_violation = constraint_violation_list[-2]
                        if h_value > prev_violation * config.omega_mu:
                            mu = mu * config.mu_mult_factor
                            mu_update_iters.append(it + 1)

    # Recompute h on the final post-step model state so final_h is correct
    # even when n_iter is not a multiple of stop_crit_win (the in-loop
    # value above only reflects the last stop-check boundary). For loops
    # that ended on a stop check this is a no-op.
    with torch.no_grad():
        final_h = float(
            (compute_dag_constraint(model.get_w_adj()) / constraint_norm).item()
        )

    return TrainingResult(
        continuous_log_alpha_pre_threshold=snapshot_log_alpha(model),
        continuous_w_adj_pre_threshold=snapshot_w_adj(model),
        n_iterations=iterations_completed,
        loss_history=loss_history,
        converged=converged,
        first_stop=first_stop,
        final_gamma=float(gamma),
        final_mu=float(mu),
        final_h=final_h,
        gamma_update_iters=gamma_update_iters,
        mu_update_iters=mu_update_iters,
        log_alpha_snapshots=log_alpha_snapshots,
        validation_nll_history=[float(v) for v in val_history],
    )
