"""Scientific-invariant tests for the do-calculus intervention layer.

Tests verify correct do-operator semantics: target clamping, downstream
propagation, upstream isolation, and non-mutation of the original SCM.
"""

import numpy as np
import pytest

from symbolic_priors_cd.data import (
    GenerationSpec,
    Intervention,
    InterventionalSampler,
    LinearGaussianSCM,
    generate_linear_gaussian_scm,
    intervene,
    sample_observational,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _chain_scm(weight: float = 1.0, noise_scale: float = 1.0) -> LinearGaussianSCM:
    """3-node chain X0 -> X1 -> X2 with uniform edge weights."""
    n = 3
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 2] = True
    weights = np.zeros((n, n), dtype=np.float64)
    weights[0, 1] = weight
    weights[1, 2] = weight
    spec = GenerationSpec(
        graph_family="ER",
        mechanism_family="linear_gaussian",
        n_nodes=n,
        expected_edges=2,
        edge_probability=2.0 / 3,
        weight_magnitude_range=(0.5, 2.0),
        noise_scale=noise_scale,
        generation_seed=0,
    )
    return LinearGaussianSCM(
        n_nodes=n,
        adjacency=adjacency,
        weights=weights,
        noise_scale=noise_scale,
        topological_order=(0, 1, 2),
        spec=spec,
    )


@pytest.fixture()
def chain() -> LinearGaussianSCM:
    return _chain_scm(weight=1.0, noise_scale=1.0)


# ---------------------------------------------------------------------------
# intervene factory
# ---------------------------------------------------------------------------


def test_intervene_returns_sampler(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=1, value=2.0))
    assert isinstance(sampler, InterventionalSampler)


def test_intervene_references_original_scm(chain: LinearGaussianSCM) -> None:
    """The sampler must reference the same SCM object, not a copy."""
    sampler = intervene(chain, Intervention(target=0, value=0.0))
    assert sampler.scm is chain


def test_intervene_does_not_mutate_scm(chain: LinearGaussianSCM) -> None:
    """Calling intervene must not change the original SCM in any way."""
    adjacency_before = chain.adjacency.copy()
    weights_before = chain.weights.copy()
    topo_before = chain.topological_order
    intervene(chain, Intervention(target=1, value=5.0))
    np.testing.assert_array_equal(chain.adjacency, adjacency_before)
    np.testing.assert_array_equal(chain.weights, weights_before)
    assert chain.topological_order == topo_before


def test_intervene_validation_negative_target(chain: LinearGaussianSCM) -> None:
    with pytest.raises(ValueError, match="intervention.target"):
        intervene(chain, Intervention(target=-1, value=0.0))


def test_intervene_validation_out_of_range_target(chain: LinearGaussianSCM) -> None:
    with pytest.raises(ValueError, match="intervention.target"):
        intervene(chain, Intervention(target=3, value=0.0))


# ---------------------------------------------------------------------------
# Intervention target clamping
# ---------------------------------------------------------------------------


def test_intervention_target_column_is_constant(chain: LinearGaussianSCM) -> None:
    """Every sample in the target column must equal the intervention value."""
    sampler = intervene(chain, Intervention(target=1, value=2.0))
    X = sampler.sample(n_samples=500, rng=0)
    assert np.allclose(X[:, 1], 2.0), "target column is not clamped to intervention value"


def test_intervention_target_clamped_at_zero(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=0, value=0.0))
    X = sampler.sample(n_samples=500, rng=0)
    assert np.allclose(X[:, 0], 0.0)


def test_intervention_target_negative_value(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=2, value=-2.0))
    X = sampler.sample(n_samples=500, rng=0)
    assert np.allclose(X[:, 2], -2.0)


# ---------------------------------------------------------------------------
# Do-calculus semantics: known 3-node chain example
# ---------------------------------------------------------------------------


def test_known_chain_do_operator_semantics() -> None:
    """Verify exact distributional consequences of do(X1 = 2) on a 3-node chain.

    Chain: X0 -> X1 -> X2, weights = 1.0, noise_scale = 1.0.

    Under do(X1 = 2):
      - X0 ~ N(0, 1)   (not a descendant of X1; observational marginal preserved)
      - X1 = 2          (clamped exactly)
      - X2 = 2 + N(0,1) ~ N(2, 1)   (direct child receives clamped value)
    """
    scm = _chain_scm(weight=1.0, noise_scale=1.0)
    sampler = intervene(scm, Intervention(target=1, value=2.0))
    X = sampler.sample(n_samples=50_000, rng=7)

    # Target is exactly clamped.
    assert np.allclose(X[:, 1], 2.0)

    # X0 is not a descendant of X1, so its observational marginal is preserved.
    assert abs(X[:, 0].mean()) < 0.05
    assert abs(X[:, 0].std() - 1.0) < 0.05

    # Descendant X2 = 2 + noise ~ N(2, 1).
    assert abs(X[:, 2].mean() - 2.0) < 0.05
    assert abs(X[:, 2].std() - 1.0) < 0.05


def test_do_root_node_cuts_downstream_parents() -> None:
    """do(X0 = v) must make X1 and X2 independent of the original X0 marginal.

    Chain: X0 -> X1 -> X2, weights = 1.0.
    Under do(X0 = 3):
      - X0 = 3 exactly
      - X1 = 3 + N(0,1) ~ N(3, 1)
      - X2 = X1 + N(0,1) ~ N(3, sqrt(2))
    """
    scm = _chain_scm(weight=1.0, noise_scale=1.0)
    sampler = intervene(scm, Intervention(target=0, value=3.0))
    X = sampler.sample(n_samples=50_000, rng=8)

    assert np.allclose(X[:, 0], 3.0)
    assert abs(X[:, 1].mean() - 3.0) < 0.05
    assert abs(X[:, 1].std() - 1.0) < 0.05
    assert abs(X[:, 2].mean() - 3.0) < 0.05
    assert abs(X[:, 2].std() - np.sqrt(2.0)) < 0.05


# ---------------------------------------------------------------------------
# Consistency between observational and interventional samplers
# ---------------------------------------------------------------------------


def test_leaf_do_byte_identical_upstream_columns_regression(
    chain: LinearGaussianSCM,
) -> None:
    """Regression test for shared-kernel RNG ordering under a leaf intervention.

    It checks that with the same integer seed, clamping the leaf X2 does not change
    the sampled values of upstream columns X0 and X1 because the shared kernel
    reaches them through the same RNG path. The test exists to catch refactors
    that would change RNG consumption order while leaving distribution-level
    behaviour unchanged.
    """
    rng_seed = 42
    X_obs = sample_observational(chain, n_samples=1000, rng=rng_seed)
    sampler = intervene(chain, Intervention(target=2, value=0.0))
    X_do = sampler.sample(n_samples=1000, rng=rng_seed)

    # X0 and X1 must be byte-identical (same rng path up to the clamped leaf).
    np.testing.assert_array_equal(X_do[:, 0], X_obs[:, 0])
    np.testing.assert_array_equal(X_do[:, 1], X_obs[:, 1])


def test_sample_returns_correct_shape(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=1, value=0.0))
    X = sampler.sample(n_samples=200, rng=0)
    assert X.shape == (200, 3)
    assert X.dtype == np.float64


def test_sample_reproducible_with_int_seed(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=1, value=1.0))
    X1 = sampler.sample(100, rng=99)
    X2 = sampler.sample(100, rng=99)
    np.testing.assert_array_equal(X1, X2)


def test_sample_rejects_non_positive_n_samples(chain: LinearGaussianSCM) -> None:
    sampler = intervene(chain, Intervention(target=0, value=0.0))
    with pytest.raises(ValueError, match="n_samples must be positive"):
        sampler.sample(0, rng=0)


# ---------------------------------------------------------------------------
# Larger random SCM smoke test
# ---------------------------------------------------------------------------


def test_intervention_on_random_scm_target_clamped() -> None:
    """For any randomly generated SCM, the intervention target must be clamped."""
    for seed in range(10):
        scm = generate_linear_gaussian_scm(10, 20, seed=seed)
        target = seed % scm.n_nodes
        sampler = intervene(scm, Intervention(target=target, value=float(seed)))
        X = sampler.sample(n_samples=200, rng=seed)
        assert np.allclose(X[:, target], float(seed)), (
            f"seed={seed}: target column {target} not clamped to {float(seed)}"
        )
