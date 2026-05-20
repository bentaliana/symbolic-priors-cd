"""Tests for the DCDI structural-mask context manager.

Covers in-context mask enforcement, restoration after normal exit and
after an induced exception, the structural-masking invariant that
excluded parents do not influence a target's density parameters while
included parents do, and preservation of detached continuous-edge
snapshots across an in-context exception.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from symbolic_priors_cd.wrappers._dcdi_sampling import (
    _structural_mask_context,
    sample_model_frame_dcdi,
    sample_raw_units_dcdi,
)
from symbolic_priors_cd.wrappers._dcdi_utils import (
    make_dcdi_model,
    snapshot_log_alpha,
    snapshot_w_adj,
)
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _build_model_and_mask():
    """Construct a 3-variable DCDI model and a mask with one excluded edge.

    The chosen mask has edges 0->1 and 0->2, no other edges. For target 2,
    parent 0 is included (mask[0, 2] = True) and parent 1 is excluded
    (mask[1, 2] = False).
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()
    mask = np.array([
        [False, True,  True],
        [False, False, False],
        [False, False, False],
    ])
    return model, mask


# ---------------------------------------------------------------------------
# In-context mask + restoration after normal exit
# ---------------------------------------------------------------------------


def test_structural_mask_context_sets_expected_mask_inside_context():
    """Inside the context, adjacency equals the mask and log_alpha is saturated.

    After normal exit, the model's adjacency and log_alpha are restored
    bitwise to their pre-context values.
    """
    model, mask = _build_model_and_mask()

    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()

    with _structural_mask_context(model, mask):
        adj_inside = model.adjacency.detach().clone()
        log_alpha_inside = model.gumbel_adjacency.log_alpha.detach().clone()

    expected_adj = torch.as_tensor(
        mask, dtype=saved_adj.dtype, device=saved_adj.device,
    )
    assert torch.equal(adj_inside, expected_adj), (
        f"Inside context adjacency was {adj_inside.tolist()}, "
        f"expected {expected_adj.tolist()}"
    )

    mask_tensor = torch.as_tensor(
        mask, dtype=log_alpha_inside.dtype, device=log_alpha_inside.device,
    )
    expected_log_alpha = torch.where(
        mask_tensor.bool(),
        torch.full_like(log_alpha_inside, 100.0),
        torch.full_like(log_alpha_inside, -100.0),
    )
    assert torch.equal(log_alpha_inside, expected_log_alpha), (
        "log_alpha inside the context is not saturated to +/-100 on the mask."
    )

    assert torch.equal(model.adjacency, saved_adj), (
        "model.adjacency was not restored after normal exit."
    )
    assert torch.equal(model.gumbel_adjacency.log_alpha, saved_log_alpha), (
        "model.gumbel_adjacency.log_alpha was not restored after normal exit."
    )


# ---------------------------------------------------------------------------
# Restoration after exception
# ---------------------------------------------------------------------------


def test_restoration_after_induced_exception():
    """An exception inside the context still triggers full restoration.

    The finally block of the context manager must run regardless of how
    the with-block exits. After catching the induced exception, the
    model's adjacency and log_alpha must equal their pre-context values
    bitwise.
    """
    model, mask = _build_model_and_mask()

    saved_adj = model.adjacency.detach().clone()
    saved_log_alpha = model.gumbel_adjacency.log_alpha.detach().clone()

    class _InducedError(RuntimeError):
        pass

    with pytest.raises(_InducedError):
        with _structural_mask_context(model, mask):
            raise _InducedError("induced for restoration test")

    assert torch.equal(model.adjacency, saved_adj), (
        "model.adjacency was not restored after an in-context exception."
    )
    assert torch.equal(model.gumbel_adjacency.log_alpha, saved_log_alpha), (
        "model.gumbel_adjacency.log_alpha was not restored after an "
        "in-context exception."
    )


# ---------------------------------------------------------------------------
# Structural-masking invariant: adjacency + log_alpha
# ---------------------------------------------------------------------------


def test_structural_masking_excluded_parents_do_not_influence_target():
    """Excluded parents do not influence a target's density parameters.

    With parent 1 excluded for target 2, varying column 1 of the input
    batch by a large amount leaves the target's density parameters
    unchanged within numerical tolerance. Varying column 0 (the included
    parent for target 2) by the same amount produces a visible change.
    The RNG seed is reset before each forward pass so the Gumbel draw
    is identical across paired calls.
    """
    model, mask = _build_model_and_mask()

    bs = 4
    torch.manual_seed(123)
    x_a = torch.randn(bs, 3)

    x_excluded = x_a.clone()
    x_excluded[:, 1] = x_a[:, 1] + 5.0

    x_included = x_a.clone()
    x_included[:, 0] = x_a[:, 0] + 5.0

    with _structural_mask_context(model, mask):
        weights, biases, _extra = model.get_parameters(mode="wbx")

        torch.manual_seed(42)
        with torch.no_grad():
            dp_a = model.forward_given_params(x_a, weights, biases)

        torch.manual_seed(42)
        with torch.no_grad():
            dp_excluded = model.forward_given_params(x_excluded, weights, biases)

        torch.manual_seed(42)
        with torch.no_grad():
            dp_included = model.forward_given_params(x_included, weights, biases)

    excluded_diff = float((dp_a[2] - dp_excluded[2]).abs().max().item())
    included_diff = float((dp_a[2] - dp_included[2]).abs().max().item())

    assert excluded_diff < 1e-6, (
        f"Excluded parent (column 1) still influenced target 2: "
        f"max |delta| = {excluded_diff:.6e}; structural masking failed."
    )
    assert included_diff > 1e-3, (
        f"Included parent (column 0) did not influence target 2: "
        f"max |delta| = {included_diff:.6e}; the included-parent sensitivity "
        "check is unexpectedly weak on this seed."
    )


# ---------------------------------------------------------------------------
# Preserved continuous-edge snapshots are independent of the live state
# ---------------------------------------------------------------------------


def test_preserved_continuous_edges_unchanged_after_context_exception():
    """Detached continuous-edge snapshots stay bitwise stable across the context.

    snapshot_log_alpha and snapshot_w_adj return detached CPU clones
    held by the caller. Neither the in-context mutation of the live
    model state nor an induced exception inside the with-block should
    affect those snapshots. This is distinct from restoring the live
    model state, which is covered by test_restoration_after_induced_exception.
    """
    model, mask = _build_model_and_mask()

    preserved_log_alpha = snapshot_log_alpha(model)
    preserved_w_adj = snapshot_w_adj(model)

    log_alpha_baseline = preserved_log_alpha.clone()
    w_adj_baseline = preserved_w_adj.clone()

    class _InducedError(RuntimeError):
        pass

    with pytest.raises(_InducedError):
        with _structural_mask_context(model, mask):
            raise _InducedError("induced for snapshot-stability test")

    assert torch.equal(preserved_log_alpha, log_alpha_baseline), (
        "Preserved log_alpha snapshot changed after an in-context exception."
    )
    assert torch.equal(preserved_w_adj, w_adj_baseline), (
        "Preserved w_adj snapshot changed after an in-context exception."
    )


# ---------------------------------------------------------------------------
# Sampler core: model-frame sampling
# ---------------------------------------------------------------------------


def _chain_dag_3() -> np.ndarray:
    """Return the 3-node chain DAG adjacency 0->1->2."""
    return np.array([
        [False, True,  False],
        [False, False, True],
        [False, False, False],
    ])


def test_sampler_clamping_model_frame():
    """Target column contains exactly the requested intervention value for all rows.

    Uses a 3-node chain 0->1->2 with n_samples=30, intervening on node 1.
    The value 3.0 is exactly representable in float32, so the comparison
    is exact.
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    target = 1
    intervention_value = 3.0
    n_samples = 30

    samples = sample_model_frame_dcdi(
        model, _chain_dag_3(), target, intervention_value, n_samples,
        sample_seed=0,
    )

    assert np.all(samples[:, target] == np.float32(intervention_value)), (
        f"Target column {target} not uniformly clamped to {intervention_value}. "
        f"Got min={samples[:, target].min()}, max={samples[:, target].max()}"
    )


def test_sampler_shape():
    """Returned array has shape (n_samples, num_vars)."""
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    n_samples = 17
    samples = sample_model_frame_dcdi(
        model, _chain_dag_3(), 0, 1.0, n_samples, sample_seed=0,
    )

    assert samples.shape == (n_samples, model.num_vars), (
        f"Expected shape ({n_samples}, {model.num_vars}), got {samples.shape}"
    )


def test_sampler_deterministic_with_sample_seed():
    """Same sample_seed produces identical output; different seeds produce different output.

    Non-target columns are expected to vary between seeds because they are
    drawn from learned conditionals. If the outputs happen to be equal for
    a different seed (extremely unlikely), the test would spuriously pass;
    but this is practically impossible for a real model and 20 samples.
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    target = 1
    a = _chain_dag_3()

    s1 = sample_model_frame_dcdi(model, a, target, 1.0, 20, sample_seed=7)
    s2 = sample_model_frame_dcdi(model, a, target, 1.0, 20, sample_seed=7)
    s3 = sample_model_frame_dcdi(model, a, target, 1.0, 20, sample_seed=99)

    np.testing.assert_array_equal(s1, s2, err_msg="Same seed produced different samples.")

    non_target = [j for j in range(model.num_vars) if j != target]
    for col in non_target:
        assert not np.array_equal(s1[:, col], s3[:, col]), (
            f"Different seeds produced identical non-target column {col}."
        )


def test_sampler_uses_structural_mask_context():
    """Changing the clamped value of an excluded-parent node does not alter
    the samples of downstream nodes that have no edge from that node.

    Mask: only edge 0->2. Node 1 is not a parent of node 2.
    Intervening on node 1 with two very different values (999 vs -999)
    must leave x[:, 2] bitwise identical when the same sample_seed is used,
    because node 2 only depends on node 0 in this mask and node 0 is
    sampled identically under both interventions (same seed, same topology).
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    # Only edge 0->2. Node 1 has no parents and no children.
    a_thresh = np.array([
        [False, False, True],
        [False, False, False],
        [False, False, False],
    ])
    target = 1
    n_samples = 50

    samples_high = sample_model_frame_dcdi(
        model, a_thresh, target, 999.0, n_samples, sample_seed=42,
    )
    samples_low = sample_model_frame_dcdi(
        model, a_thresh, target, -999.0, n_samples, sample_seed=42,
    )

    np.testing.assert_array_equal(
        samples_high[:, 2],
        samples_low[:, 2],
        err_msg=(
            "Node 2 samples differ between two interventions on node 1, "
            "which is not a parent of node 2 in the mask. "
            "Structural masking may not be enforced through the sampler."
        ),
    )
    assert np.all(samples_high[:, target] == np.float32(999.0))
    assert np.all(samples_low[:, target] == np.float32(-999.0))


def test_sampler_refuses_invalid_graph():
    """Calling the sampler with a non-DAG adjacency raises ValueError.

    The error message must mention graph status so the caller can diagnose
    the failure without inspecting source.
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    cyclic = np.array([
        [False, True,  False],
        [False, False, True],
        [True,  False, False],
    ])

    with pytest.raises(ValueError, match="graph status"):
        sample_model_frame_dcdi(model, cyclic, 0, 1.0, 10, sample_seed=0)


def test_sampler_restores_training_mode():
    """sample_model_frame_dcdi restores model.training to True when called in train mode."""
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.train()

    assert model.training, "Pre-condition: model must be in train mode before the call."

    sample_model_frame_dcdi(model, _chain_dag_3(), 0, 0.0, 5, sample_seed=0)

    assert model.training, (
        "model.training was not restored to True after sample_model_frame_dcdi."
    )


# ---------------------------------------------------------------------------
# Raw-unit intervention roundtrip
# ---------------------------------------------------------------------------


def _make_model_and_preprocessors():
    """Shared setup for raw-unit roundtrip tests.

    Returns a 3-variable model and two fitted preprocessors (centred-only and
    standardised) trained on data with non-trivial mean and variance so the
    roundtrip is a real test.
    """
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 3)) * np.array([2.0, 3.0, 1.5]) + np.array([1.0, -2.0, 4.0])

    centred = CentredOnlyTransform().fit(X_train)
    standardised = StandardisedTransform().fit(X_train)

    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    return model, centred, standardised


def test_raw_unit_intervention_roundtrip_centred_only():
    """Target column in raw-unit samples equals the raw do-value for CentredOnlyTransform.

    The raw do-value 2.0 is transformed to model frame, clamped during
    sampling, then transformed back. The result must equal 2.0 within
    float32 precision of the intermediate sample tensor.
    """
    model, centred, _ = _make_model_and_preprocessors()
    target = 1
    raw_value = 2.0

    samples = sample_raw_units_dcdi(
        model, _chain_dag_3(), target, raw_value, 40, sample_seed=0,
        preprocessor=centred,
    )

    np.testing.assert_allclose(
        samples[:, target], raw_value, rtol=1e-5,
        err_msg=f"Target column {target} not equal to raw do-value {raw_value} after roundtrip.",
    )


def test_raw_unit_intervention_roundtrip_standardised():
    """Target column in raw-unit samples equals the raw do-value for StandardisedTransform.

    The raw do-value 2.0 is transformed to model frame (mean-subtracted,
    std-divided), clamped during sampling, then transformed back. The result
    must equal 2.0 within float32 precision.
    """
    model, _, standardised = _make_model_and_preprocessors()
    target = 1
    raw_value = 2.0

    samples = sample_raw_units_dcdi(
        model, _chain_dag_3(), target, raw_value, 40, sample_seed=0,
        preprocessor=standardised,
    )

    np.testing.assert_allclose(
        samples[:, target], raw_value, rtol=1e-5,
        err_msg=f"Target column {target} not equal to raw do-value {raw_value} after roundtrip.",
    )


def test_raw_unit_sampler_does_not_refit_preprocessor():
    """Calling sample_raw_units_dcdi must not alter the preprocessor's stored statistics.

    Checks both CentredOnlyTransform (mean only) and StandardisedTransform
    (mean and std) to confirm neither is modified by the sampling call.
    """
    model, centred, standardised = _make_model_and_preprocessors()

    centred_mean_before = centred._mean.copy()
    std_mean_before = standardised._mean.copy()
    std_std_before = standardised._std.copy()

    sample_raw_units_dcdi(
        model, _chain_dag_3(), 1, 2.0, 20, sample_seed=0,
        preprocessor=centred,
    )
    sample_raw_units_dcdi(
        model, _chain_dag_3(), 1, 2.0, 20, sample_seed=0,
        preprocessor=standardised,
    )

    np.testing.assert_array_equal(
        centred._mean, centred_mean_before,
        err_msg="CentredOnlyTransform mean was modified by sample_raw_units_dcdi.",
    )
    np.testing.assert_array_equal(
        standardised._mean, std_mean_before,
        err_msg="StandardisedTransform mean was modified by sample_raw_units_dcdi.",
    )
    np.testing.assert_array_equal(
        standardised._std, std_std_before,
        err_msg="StandardisedTransform std was modified by sample_raw_units_dcdi.",
    )


def test_raw_unit_sampler_deterministic_with_sample_seed():
    """Same sample_seed produces bitwise-identical raw-unit output."""
    model, centred, _ = _make_model_and_preprocessors()

    s1 = sample_raw_units_dcdi(
        model, _chain_dag_3(), 1, 2.0, 25, sample_seed=7,
        preprocessor=centred,
    )
    s2 = sample_raw_units_dcdi(
        model, _chain_dag_3(), 1, 2.0, 25, sample_seed=7,
        preprocessor=centred,
    )

    np.testing.assert_array_equal(
        s1, s2, err_msg="Same sample_seed produced different raw-unit samples.",
    )


def test_raw_unit_sampler_rejects_invalid_preprocessor_type():
    """Passing an object that is not a recognised preprocessor raises TypeError.

    The error message must mention "preprocessor" so the caller can identify
    which argument is wrong.
    """
    torch.manual_seed(0)
    model = make_dcdi_model(num_vars=3, num_layers=2, hid_dim=8)
    model.eval()

    with pytest.raises(TypeError, match="preprocessor"):
        sample_raw_units_dcdi(
            model, _chain_dag_3(), 0, 1.0, 5, sample_seed=0,
            preprocessor=object(),
        )
