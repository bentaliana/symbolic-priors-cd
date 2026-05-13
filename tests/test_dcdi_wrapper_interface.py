"""Tests for the DCDI wrapper public interface.

Covers package importability, status Literal values, the WrapperDiagnostics
key contract, import isolation for the low-level DCDI helper module, and
continuous-edge preservation invariants after a tiny training run.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from symbolic_priors_cd.wrappers import (
    GraphStatus,
    SamplerStatus,
    TrainingStatus,
    WrapperDiagnostics,
)
from symbolic_priors_cd.wrappers._dcdi_training import (
    DCDIConfig,
    run_dcdi_training_loop,
)
from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model
from symbolic_priors_cd.wrappers.status import WrapperDiagnostics as _WDiag


def _tiny_fit(seed: int = 0, n_iter: int = 20):
    """Run a tiny DCDI training fit; return (model, TrainingResult).

    Uses 32 training samples and 16 validation samples on a 3-node model.
    Same-seed calls produce identical models and identical training output.
    """
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((32, 3)).astype(np.float64)
    X_val = rng.standard_normal((16, 3)).astype(np.float64)
    config = DCDIConfig(stop_crit_win=10, train_batch_size=8)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    result = run_dcdi_training_loop(
        model, X_train, X_val, config=config, seed=seed, n_iter=n_iter,
    )
    return model, result


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_wrappers_package_importable():
    """The wrappers package and its public names import without error."""
    import symbolic_priors_cd.wrappers as wrappers  # noqa: F401

    assert hasattr(wrappers, "TrainingStatus")
    assert hasattr(wrappers, "GraphStatus")
    assert hasattr(wrappers, "SamplerStatus")
    assert hasattr(wrappers, "WrapperDiagnostics")


# ---------------------------------------------------------------------------
# TrainingStatus
# ---------------------------------------------------------------------------


def test_training_status_values():
    """TrainingStatus Literal contains exactly the four documented values."""
    import typing

    args = typing.get_args(TrainingStatus)
    assert set(args) == {"converged", "max_iter", "diverged", "wrapper_error"}


# ---------------------------------------------------------------------------
# GraphStatus
# ---------------------------------------------------------------------------


def test_graph_status_values():
    """GraphStatus Literal contains exactly the five documented values."""
    import typing

    args = typing.get_args(GraphStatus)
    assert set(args) == {"valid_dag", "cyclic", "bidirected", "self_loop", "invalid_shape"}


# ---------------------------------------------------------------------------
# SamplerStatus
# ---------------------------------------------------------------------------


def test_sampler_status_values():
    """SamplerStatus Literal contains exactly the four documented values."""
    import typing

    args = typing.get_args(SamplerStatus)
    assert set(args) == {
        "available",
        "unavailable_invalid_graph",
        "unavailable_no_api",
        "unavailable_unresolved_noise_policy",
    }


# ---------------------------------------------------------------------------
# WrapperDiagnostics
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "training_status",
    "graph_status",
    "sampler_status",
    "seed",
    "n_iterations",
    "config_snapshot",
    "loss_history",
    "loss_decomposition_final",
    "convergence_info",
    "thresholded_adjacency",
    "graph_invalid_reason",
    "sampler_unavailable_reason",
    "mmd_sampling_metadata",
    "loss_hook_name",
    "numerical_tolerances",
    "model_specific_diagnostics",
}


def test_wrapper_diagnostics_has_expected_keys():
    """WrapperDiagnostics defines all expected keys."""
    annotations = _WDiag.__annotations__
    assert set(annotations.keys()) == _EXPECTED_KEYS


def test_wrapper_diagnostics_key_count():
    """WrapperDiagnostics has exactly the expected number of keys."""
    assert len(_WDiag.__annotations__) == len(_EXPECTED_KEYS)


# ---------------------------------------------------------------------------
# DCDI import isolation
# ---------------------------------------------------------------------------


def test_dcdi_train_not_imported():
    """Importing _dcdi_utils must not pull in dcdi.train as a side effect.

    Uses a subprocess so the check is not polluted by modules already loaded
    in the current pytest session.
    """
    code = (
        "import sys; "
        "import symbolic_priors_cd.wrappers._dcdi_utils; "
        "assert 'dcdi.train' not in sys.modules, "
        "'dcdi.train was imported as a side effect of _dcdi_utils'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"dcdi.train was imported as a side effect of importing _dcdi_utils.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_dcdi_imported_from_pinned_source():
    """_dcdi_utils must load DCDI from external/source_inspection/dcdi, not site-packages.

    Uses a subprocess so the check is not polluted by modules already loaded
    in the current pytest session.
    """
    code = (
        "import sys; "
        "from pathlib import Path; "
        "import symbolic_priors_cd.wrappers._dcdi_utils as u; "
        "f = Path(sys.modules['dcdi.models.learnables'].__file__).resolve(); "
        "ok = f.is_relative_to(u._DCDI_SRC); "
        "sys.exit(0 if ok else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "dcdi.models.learnables was not imported from the pinned source.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Continuous-edge preservation after a tiny training run
# ---------------------------------------------------------------------------


def test_continuous_log_alpha_not_saturated_after_fit():
    """No entry of continuous_log_alpha_pre_threshold equals the saturation values.

    The wrapper deliberately does not replicate DCDI's second-stop log_alpha
    overwrite to +/-100; the preserved continuous edge object must therefore
    contain no saturation sentinel values.
    """
    _, result = _tiny_fit()
    log_alpha = result.continuous_log_alpha_pre_threshold.numpy()
    assert not np.any(log_alpha == 100.0), (
        f"Found +100 saturation sentinel in log_alpha: max={log_alpha.max()}"
    )
    assert not np.any(log_alpha == -100.0), (
        f"Found -100 saturation sentinel in log_alpha: min={log_alpha.min()}"
    )


def test_continuous_w_adj_range_and_diagonal():
    """w_adj has off-diagonal entries in [0, 1], a zero diagonal, and no saturation."""
    _, result = _tiny_fit()
    w_adj = result.continuous_w_adj_pre_threshold.numpy()
    log_alpha = result.continuous_log_alpha_pre_threshold.numpy()

    diagonal = np.diag(w_adj)
    assert np.all(diagonal == 0.0), f"Diagonal must be exactly zero, got {diagonal}"

    off_diag_mask = ~np.eye(w_adj.shape[0], dtype=bool)
    off_diag = w_adj[off_diag_mask]
    assert np.all(off_diag >= 0.0), f"Off-diagonal has negative entry: min={off_diag.min()}"
    assert np.all(off_diag <= 1.0), f"Off-diagonal exceeds 1.0: max={off_diag.max()}"

    assert not np.any(np.abs(log_alpha) == 100.0), (
        "log_alpha appears deliberately saturated to +/-100."
    )


def test_continuous_edge_preservation_snapshots_are_independent():
    """Mutating preserved tensors must not mutate the model or its outputs.

    The preserved tensors are detached CPU clones, so writes to them must
    not propagate to model.gumbel_adjacency.log_alpha or to future
    model.get_w_adj() calls. Both tensors must also have requires_grad=False.
    """
    model, result = _tiny_fit()

    log_alpha_baseline = model.gumbel_adjacency.log_alpha.detach().clone()
    w_adj_baseline = model.get_w_adj().detach().clone()

    result.continuous_log_alpha_pre_threshold[0, 0] = 1e6
    result.continuous_w_adj_pre_threshold[0, 1] = 1e6

    assert torch.equal(
        model.gumbel_adjacency.log_alpha.detach(), log_alpha_baseline
    ), "Mutating the preserved log_alpha changed the live model parameter."
    assert torch.equal(
        model.get_w_adj().detach(), w_adj_baseline
    ), "Mutating the preserved w_adj changed model.get_w_adj()."

    assert not result.continuous_log_alpha_pre_threshold.requires_grad
    assert not result.continuous_w_adj_pre_threshold.requires_grad
