"""Tests for CentredOnlyTransform and StandardisedTransform.

Covers roundtrip identities, intervention-value transforms, training-statistic
isolation, zero-variance error raising, shape preservation, and package
importability.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_package_import():
    """CentredOnlyTransform and StandardisedTransform are accessible from the wrappers package."""
    from symbolic_priors_cd.wrappers import CentredOnlyTransform, StandardisedTransform

    assert CentredOnlyTransform is not None
    assert StandardisedTransform is not None


# ---------------------------------------------------------------------------
# Centred-only roundtrip
# ---------------------------------------------------------------------------


def test_centred_only_roundtrip():
    """inverse_transform(transform(X)) recovers X within 1e-12 for CentredOnlyTransform."""
    from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50, 4))
    X_test = rng.standard_normal((20, 4))

    t = CentredOnlyTransform().fit(X_train)
    recovered = t.inverse_transform(t.transform(X_test))
    np.testing.assert_allclose(recovered, X_test, atol=1e-12)


# ---------------------------------------------------------------------------
# Standardised roundtrip
# ---------------------------------------------------------------------------


def test_standardised_roundtrip():
    """inverse_transform(transform(X)) recovers X within 1e-12 for StandardisedTransform."""
    from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform

    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((50, 4)) * 3.0 + 1.5
    X_test = rng.standard_normal((20, 4)) * 3.0 + 1.5

    t = StandardisedTransform().fit(X_train)
    recovered = t.inverse_transform(t.transform(X_test))
    np.testing.assert_allclose(recovered, X_test, atol=1e-12)


# ---------------------------------------------------------------------------
# Intervention-value transforms
# ---------------------------------------------------------------------------


def test_centred_only_intervention_value_transform():
    """transform_intervention_value subtracts the training mean for CentredOnlyTransform."""
    from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform

    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((100, 3)) + np.array([1.0, 2.0, 3.0])
    t = CentredOnlyTransform().fit(X_train)

    raw_value = 5.0
    target = 1
    model_value = t.transform_intervention_value(raw_value, target)

    # The returned value should equal what transform would give for that scalar.
    expected = raw_value - t._mean[target]
    assert abs(model_value - expected) < 1e-12

    # Applying inverse gives back raw_value.
    recovered = model_value + t._mean[target]
    assert abs(recovered - raw_value) < 1e-12


def test_standardised_intervention_value_transform():
    """transform_intervention_value applies mean subtraction and std division for StandardisedTransform."""
    from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform

    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((100, 3)) * 2.0 + 5.0
    t = StandardisedTransform().fit(X_train)

    raw_value = 8.0
    target = 2
    model_value = t.transform_intervention_value(raw_value, target)

    expected = (raw_value - t._mean[target]) / t._std[target]
    assert abs(model_value - expected) < 1e-12

    # Roundtrip: multiplying back by std and adding mean recovers raw_value.
    recovered = model_value * t._std[target] + t._mean[target]
    assert abs(recovered - raw_value) < 1e-12


# ---------------------------------------------------------------------------
# Training-statistic isolation
# ---------------------------------------------------------------------------


def test_transform_uses_stored_training_statistics():
    """transform uses the statistics from fit, not statistics of the array passed to transform."""
    from symbolic_priors_cd.wrappers.preprocessing import (
        CentredOnlyTransform,
        StandardisedTransform,
    )

    rng = np.random.default_rng(4)
    X_train = rng.standard_normal((200, 3))
    # X_shifted has a different mean (~100) and different std (~10).
    X_shifted = rng.standard_normal((200, 3)) * 10.0 + 100.0

    # CentredOnly: training mean is ~0. Subtracting it from X_shifted leaves
    # values near 100. If transform had used X_shifted's own mean, values
    # would be near zero.
    ct = CentredOnlyTransform().fit(X_train)
    transformed_ct = ct.transform(X_shifted)
    assert transformed_ct.mean() > 50.0, (
        "transform appears to have used the input mean rather than the training mean"
    )

    # Standardised: training std is ~1. Dividing X_shifted by it leaves
    # values with std ~10. If transform had used X_shifted's own std (~10),
    # the result would have std ~1.
    st = StandardisedTransform().fit(X_train)
    transformed_st = st.transform(X_shifted)
    assert transformed_st.std() > 5.0, (
        "transform appears to have used the input std rather than the training std"
    )


# ---------------------------------------------------------------------------
# Zero-variance guard
# ---------------------------------------------------------------------------


def test_standardised_zero_variance_raises():
    """StandardisedTransform.fit raises ValueError when a column has zero variance."""
    from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform

    rng = np.random.default_rng(5)
    X = rng.standard_normal((50, 3))
    X[:, 1] = 0.0  # constant column, std = 0

    with pytest.raises(ValueError, match="zero or near-zero"):
        StandardisedTransform().fit(X)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


def test_shapes_preserved():
    """transform and inverse_transform return arrays with the same shape as the input."""
    from symbolic_priors_cd.wrappers.preprocessing import (
        CentredOnlyTransform,
        StandardisedTransform,
    )

    rng = np.random.default_rng(6)
    X_train = rng.standard_normal((40, 5))
    X_test = rng.standard_normal((15, 5))

    ct = CentredOnlyTransform().fit(X_train)
    assert ct.transform(X_test).shape == X_test.shape
    assert ct.inverse_transform(ct.transform(X_test)).shape == X_test.shape

    st = StandardisedTransform().fit(X_train)
    assert st.transform(X_test).shape == X_test.shape
    assert st.inverse_transform(st.transform(X_test)).shape == X_test.shape
