"""Real-study protocol guards for the selection-study runner.

Provides ``assert_real_study_constants``: a stage-keyed validator
that requires a ``Configuration`` to carry the exact real-study
constant values frozen in the selection-study protocol. The guard
is deliberately not invoked from ``Configuration.__post_init__``
so toy and schema-gate Configurations remain constructible; the
runner must call this guard explicitly before any phase that
demands real-study values.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from experiments.selection_study.config import Configuration


# Stage labels accepted by ``assert_real_study_constants``.
_VALID_STAGES: tuple[str, ...] = ("reproduction_pass",)

_REQUIRED_REPRODUCTION_POPULATION = "reproduction"

# Shared real-study constants. Each reproduction-pass
# Configuration must carry exactly these values regardless of
# model.
_SHARED_REQUIRED_VALUES: tuple[tuple[str, object], ...] = (
    ("n_nodes", 10),
    ("expected_edges", 20),
    ("noise_scale", 1.0),
    ("weight_magnitude_range", (0.5, 2.0)),
    ("n_train", 1000),
    ("mmd_n_samples", 1000),
)

# DAGMA-required values for the reproduction pass.
_DAGMA_REQUIRED_VALUES: tuple[tuple[str, object], ...] = (
    ("threshold_robustness_triple", (0.2, 0.3, 0.4)),
    ("dagma_warm_iter", 20000),
    ("dagma_max_iter", 70000),
    ("dagma_lr", 3e-4),
    ("dagma_beta_1", 0.99),
    ("dagma_beta_2", 0.999),
)

_DAGMA_REQUIRED_NONE_FIELDS: tuple[str, ...] = (
    "n_val_dcdi",
    "dcdi_num_train_iter",
    "dcdi_stop_crit_win",
    "dcdi_train_patience",
    "dcdi_train_batch_size",
    "dcdi_lr",
    "dcdi_h_threshold",
    "dcdi_hidden_units",
    "dcdi_hidden_layers",
)

# DCDI-required values for the reproduction pass.
_DCDI_REQUIRED_VALUES: tuple[tuple[str, object], ...] = (
    ("threshold_robustness_triple", (0.4, 0.5, 0.6)),
    ("n_val_dcdi", 200),
    ("dcdi_num_train_iter", 300000),
    ("dcdi_stop_crit_win", 100),
    ("dcdi_train_patience", 5),
    ("dcdi_train_batch_size", 64),
    ("dcdi_lr", 1e-3),
    ("dcdi_h_threshold", 1e-8),
    ("dcdi_hidden_units", 16),
    ("dcdi_hidden_layers", 2),
)

_DCDI_REQUIRED_NONE_FIELDS: tuple[str, ...] = (
    "dagma_warm_iter",
    "dagma_max_iter",
    "dagma_lr",
    "dagma_beta_1",
    "dagma_beta_2",
)


def _values_equal(actual: Any, expected: Any) -> bool:
    """Compare a Configuration field against an expected value.

    Floats are compared with ``math.isclose(abs_tol=1e-12)``;
    sequences (lists, tuples) are compared element-wise with the
    same rule; everything else uses ``==``. Bool is not treated
    as an int.
    """
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            return False
        return math.isclose(float(actual), float(expected), abs_tol=1e-12)
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)):
            return False
        if len(actual) != len(expected):
            return False
        return all(_values_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _enforce_field(
    config: Configuration,
    *,
    field_name: str,
    expected: Any,
) -> None:
    """Raise ValueError if ``config.<field_name>`` differs from ``expected``."""
    actual = getattr(config, field_name)
    if not _values_equal(actual, expected):
        raise ValueError(
            "reproduction-pass real-study protocol violation: field "
            f"{field_name!r} must equal {expected!r}; got {actual!r}"
        )


def _enforce_none(
    config: Configuration,
    *,
    field_names: Iterable[str],
    reason: str,
) -> None:
    """Raise ValueError if any listed field is not None on ``config``."""
    offenders = [
        name for name in field_names if getattr(config, name) is not None
    ]
    if offenders:
        raise ValueError(
            "reproduction-pass real-study protocol violation: "
            f"{reason}; offending field(s): {', '.join(offenders)}"
        )


def _enforce_reproduction_population(config: Configuration) -> None:
    """Require the configuration to carry a 'reproduction' seed population.

    The reproduction-pool seed integers themselves are not pinned
    by the selection-study protocol documents; this guard only
    verifies that the population is present and non-empty.
    """
    names = [name for name, _ in config.seed_populations]
    if _REQUIRED_REPRODUCTION_POPULATION not in names:
        raise ValueError(
            "reproduction-pass real-study protocol violation: "
            "seed_populations must contain the 'reproduction' "
            f"population; got populations={names!r}"
        )
    for name, seeds in config.seed_populations:
        if name == _REQUIRED_REPRODUCTION_POPULATION:
            if len(seeds) < 1:
                raise ValueError(
                    "reproduction-pass real-study protocol violation: "
                    "the 'reproduction' seed population is empty"
                )
            break


def assert_real_study_constants(
    config: Configuration, *, stage: str
) -> None:
    """Verify a Configuration carries real-study constants for ``stage``.

    Parameters
    ----------
    config : Configuration
        Configuration to validate against the real-study protocol.
    stage : str
        Stage name. The only accepted value is ``"reproduction_pass"``.
        Future stages may be added separately.

    Raises
    ------
    ValueError
        On unknown stage, on field value mismatch, on a required
        field being None when it must be non-None (or vice versa),
        or on missing seed populations. Each error names the
        offending field and the expected value.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(
            "unknown stage for assert_real_study_constants: "
            f"{stage!r}; accepted stages: {_VALID_STAGES!r}"
        )

    for field_name, expected_value in _SHARED_REQUIRED_VALUES:
        _enforce_field(
            config, field_name=field_name, expected=expected_value
        )

    if config.model == "dagma":
        for field_name, expected_value in _DAGMA_REQUIRED_VALUES:
            _enforce_field(
                config,
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            field_names=_DAGMA_REQUIRED_NONE_FIELDS,
            reason=(
                "DAGMA reproduction-pass configurations must leave "
                "every DCDI-only field None"
            ),
        )
    elif config.model == "dcdi":
        for field_name, expected_value in _DCDI_REQUIRED_VALUES:
            _enforce_field(
                config,
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            field_names=_DCDI_REQUIRED_NONE_FIELDS,
            reason=(
                "DCDI reproduction-pass configurations must leave "
                "every DAGMA-only field None"
            ),
        )
    else:
        raise ValueError(
            "reproduction-pass real-study protocol violation: "
            f"model must be 'dagma' or 'dcdi'; got {config.model!r}"
        )

    _enforce_reproduction_population(config)


__all__ = [
    "assert_real_study_constants",
]
