"""Tests for DCDI training-loop determinism and behavioural equivalence.

Currently covers determinism only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from symbolic_priors_cd.wrappers._dcdi_training import (
    DCDIConfig,
    run_dcdi_training_loop,
)
from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model


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

    # Wrong column count in X_train
    bad_X_train_cols = rng.standard_normal((20, 5)).astype(np.float64)
    with pytest.raises(ValueError, match="X_train.shape"):
        run_dcdi_training_loop(
            model, bad_X_train_cols, good_X_val, config=config, seed=0, n_iter=5,
        )

    # Wrong column count in X_val
    bad_X_val_cols = rng.standard_normal((10, 5)).astype(np.float64)
    with pytest.raises(ValueError, match="X_val.shape"):
        run_dcdi_training_loop(
            model, good_X_train, bad_X_val_cols, config=config, seed=0, n_iter=5,
        )

    # Non-2D X_train
    bad_X_train_1d = rng.standard_normal(20).astype(np.float64)
    with pytest.raises(ValueError, match="X_train must be 2D"):
        run_dcdi_training_loop(
            model, bad_X_train_1d, good_X_val, config=config, seed=0, n_iter=5,
        )

    # Non-2D X_val
    bad_X_val_1d = rng.standard_normal(10).astype(np.float64)
    with pytest.raises(ValueError, match="X_val must be 2D"):
        run_dcdi_training_loop(
            model, good_X_train, bad_X_val_1d, config=config, seed=0, n_iter=5,
        )

    # Too few rows in X_train
    too_small = rng.standard_normal((1, 3)).astype(np.float64)
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_dcdi_training_loop(
            model, too_small, good_X_val, config=config, seed=0, n_iter=5,
        )
