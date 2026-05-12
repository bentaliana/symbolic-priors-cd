"""Preprocessing transforms for causal discovery wrapper inputs.

Both transforms fit exclusively on training data. The stored statistics are
used for all subsequent transform and inverse_transform calls. Callers are
responsible for preventing test-set leakage by fitting only on training data.

Intervention values are expressed in raw SCM units. transform_intervention_value
converts a single raw-unit value to model-frame space so it can be clamped
correctly during interventional sample generation.
"""

from __future__ import annotations

import numpy as np


class CentredOnlyTransform:
    """Per-variable mean subtraction without scale normalisation.

    Fitting stores the per-variable training mean. All subsequent transforms
    use those stored statistics. The conceptual standard deviation is 1.0,
    so inverse_transform recovers raw units by adding the mean back without
    any scaling step.

    Calling transform or inverse_transform before fit raises AttributeError.
    """

    def fit(self, X_train: np.ndarray) -> CentredOnlyTransform:
        """Compute and store the per-variable training mean.

        Parameters
        ----------
        X_train : np.ndarray
            Observed data of shape (n_samples, n_variables).

        Returns
        -------
        CentredOnlyTransform
            Self, to support method chaining.
        """
        X_train = np.asarray(X_train, dtype=float)
        self._mean: np.ndarray = X_train.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract the training mean from each variable.

        Uses the mean computed during fit, not statistics of X.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_variables).

        Returns
        -------
        np.ndarray
            Mean-centred array of the same shape.
        """
        return np.asarray(X, dtype=float) - self._mean

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Add the training mean back to recover raw-unit values.

        Parameters
        ----------
        X : np.ndarray
            Mean-centred data of shape (n_samples, n_variables).

        Returns
        -------
        np.ndarray
            Raw-unit array of the same shape.
        """
        return np.asarray(X, dtype=float) + self._mean

    def transform_intervention_value(self, value: float, target: int) -> float:
        """Convert a raw-unit intervention value to model-frame space.

        Parameters
        ----------
        value : float
            Intervention value in raw SCM units.
        target : int
            Column index of the intervened variable.

        Returns
        -------
        float
            Value in model-frame space (training mean subtracted).
        """
        return float(value) - float(self._mean[target])


class StandardisedTransform:
    """Per-variable mean subtraction and standard deviation normalisation.

    Fitting stores the per-variable training mean and training standard
    deviation (ddof=0). All subsequent transforms use those stored statistics.

    A zero or near-zero standard deviation in any variable raises ValueError
    during fit. No epsilon floor is silently added to avoid division by zero;
    callers must remove constant variables or switch to CentredOnlyTransform.

    Calling transform or inverse_transform before fit raises AttributeError.
    """

    _STD_FLOOR: float = 1e-8

    def fit(self, X_train: np.ndarray) -> StandardisedTransform:
        """Compute and store the per-variable training mean and standard deviation.

        Parameters
        ----------
        X_train : np.ndarray
            Observed data of shape (n_samples, n_variables).

        Returns
        -------
        StandardisedTransform
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If any variable has a standard deviation below the floor threshold,
            indicating a constant or near-constant column.
        """
        X_train = np.asarray(X_train, dtype=float)
        self._mean: np.ndarray = X_train.mean(axis=0)
        self._std: np.ndarray = X_train.std(axis=0, ddof=0)
        near_zero = self._std < self._STD_FLOOR
        if np.any(near_zero):
            bad_cols = np.where(near_zero)[0].tolist()
            raise ValueError(
                f"StandardisedTransform: standard deviation is zero or near-zero "
                f"in columns {bad_cols} (min std = {self._std.min():.3e}, "
                f"floor = {self._STD_FLOOR:.3e}). "
                "Division would produce numerically unstable results. "
                "Remove constant variables or use CentredOnlyTransform."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract the training mean and divide by the training standard deviation.

        Uses the mean and std computed during fit, not statistics of X.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_variables).

        Returns
        -------
        np.ndarray
            Standardised array of the same shape.
        """
        return (np.asarray(X, dtype=float) - self._mean) / self._std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Multiply by the training standard deviation and add the training mean.

        Parameters
        ----------
        X : np.ndarray
            Standardised data of shape (n_samples, n_variables).

        Returns
        -------
        np.ndarray
            Raw-unit array of the same shape.
        """
        return np.asarray(X, dtype=float) * self._std + self._mean

    def transform_intervention_value(self, value: float, target: int) -> float:
        """Convert a raw-unit intervention value to model-frame space.

        Parameters
        ----------
        value : float
            Intervention value in raw SCM units.
        target : int
            Column index of the intervened variable.

        Returns
        -------
        float
            Value in model-frame space (mean-subtracted, std-divided).
        """
        return (float(value) - float(self._mean[target])) / float(self._std[target])
