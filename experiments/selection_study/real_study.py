"""Real-study protocol guards for the selection-study runner.

Provides ``assert_real_study_constants``: a stage-keyed validator
that requires a ``Configuration`` to carry the exact real-study
constant values frozen in the selection-study protocol. The guard
is deliberately not invoked from ``Configuration.__post_init__``
so toy and schema-gate Configurations remain constructible; the
runner must call this guard explicitly before any phase that
demands real-study values.

Two stages are currently supported. ``stage="reproduction_pass"``
requires a non-empty ``"reproduction"`` seed population and the
DAGMA / DCDI defaults established for the reproduction-pass cell.
``stage="calibration"`` additionally requires the calibration seed
population to be exactly ``(201, 202)``, the DCDI fit-RNG to be
fixed at ``seed_torch = seed_numpy = 42`` for DCDI models, no
non-calibration seed population to be present, and the sparsity
grid to match the model's frozen calibration grid (DAGMA: lambda1
in (0.01, 0.025, 0.05, 0.1, 0.25); DCDI: reg_coeff in
(0.01, 0.03, 0.1, 0.3, 1.0)).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from experiments.selection_study.config import Configuration


# Stage labels accepted by ``assert_real_study_constants``.
_VALID_STAGES: tuple[str, ...] = ("reproduction_pass", "calibration")

_REQUIRED_REPRODUCTION_POPULATION = "reproduction"
_REQUIRED_CALIBRATION_POPULATION = "calibration"
_CALIBRATION_SEED_TUPLE: tuple[int, ...] = (201, 202)
_DCDI_CALIBRATION_FIT_RNG_VALUE = 42

# Shared real-study constants. Each reproduction-pass or calibration
# Configuration must carry exactly these values regardless of model.
_SHARED_REQUIRED_VALUES: tuple[tuple[str, object], ...] = (
    ("n_nodes", 10),
    ("expected_edges", 20),
    ("noise_scale", 1.0),
    ("weight_magnitude_range", (0.5, 2.0)),
    ("n_train", 1000),
    ("mmd_n_samples", 1000),
)

# DAGMA-required values for any DAGMA real-study Configuration.
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

# DCDI-required values for any DCDI real-study Configuration.
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

# Frozen calibration sparsity grids per model. The guard enforces
# exact set equality (order-independent comparison) over the
# hyperparameter values in calibration_configurations.
_DAGMA_CALIBRATION_GRID: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.1, 0.25,
)
_DAGMA_CALIBRATION_HYPERPARAMETER = "lambda1"
_DCDI_CALIBRATION_GRID: tuple[float, ...] = (
    0.01, 0.03, 0.1, 0.3, 1.0,
)
_DCDI_CALIBRATION_HYPERPARAMETER = "reg_coeff"


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
    stage: str,
    field_name: str,
    expected: Any,
) -> None:
    """Raise ValueError if ``config.<field_name>`` differs from ``expected``."""
    actual = getattr(config, field_name)
    if not _values_equal(actual, expected):
        raise ValueError(
            f"{stage} real-study protocol violation: field "
            f"{field_name!r} must equal {expected!r}; got {actual!r}"
        )


def _enforce_none(
    config: Configuration,
    *,
    stage: str,
    field_names: Iterable[str],
    reason: str,
) -> None:
    """Raise ValueError if any listed field is not None on ``config``."""
    offenders = [
        name for name in field_names if getattr(config, name) is not None
    ]
    if offenders:
        raise ValueError(
            f"{stage} real-study protocol violation: "
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
            "reproduction_pass real-study protocol violation: "
            "seed_populations must contain the 'reproduction' "
            f"population; got populations={names!r}"
        )
    for name, seeds in config.seed_populations:
        if name == _REQUIRED_REPRODUCTION_POPULATION:
            if len(seeds) < 1:
                raise ValueError(
                    "reproduction_pass real-study protocol violation: "
                    "the 'reproduction' seed population is empty"
                )
            break


def _enforce_calibration_population(config: Configuration) -> None:
    """Require the calibration seed population to be exactly (201, 202).

    The calibration phase uses a frozen seed pool of two integers.
    Any deviation (missing population, wrong integer values,
    duplicates, additional held-out values) is rejected. Other
    seed populations that the schema permits, such as
    ``"held_out_evaluation"``, must NOT appear on a calibration
    Configuration because the held-out runner owns those seeds.
    """
    names_present = [name for name, _ in config.seed_populations]

    if _REQUIRED_CALIBRATION_POPULATION not in names_present:
        raise ValueError(
            "calibration real-study protocol violation: "
            "seed_populations must contain the 'calibration' "
            f"population; got populations={names_present!r}"
        )

    extraneous = [
        name
        for name in names_present
        if name != _REQUIRED_CALIBRATION_POPULATION
    ]
    if extraneous:
        raise ValueError(
            "calibration real-study protocol violation: "
            "seed_populations must contain only the 'calibration' "
            "population on a calibration parent Configuration; "
            f"extraneous population(s): {extraneous!r}"
        )

    for name, seeds in config.seed_populations:
        if name != _REQUIRED_CALIBRATION_POPULATION:
            continue
        if tuple(seeds) != _CALIBRATION_SEED_TUPLE:
            raise ValueError(
                "calibration real-study protocol violation: the "
                "'calibration' seed population must equal "
                f"{list(_CALIBRATION_SEED_TUPLE)!r}; got "
                f"{list(seeds)!r}"
            )


def _enforce_dcdi_calibration_fit_rng(
    config: Configuration,
) -> None:
    """Require seed_torch == seed_numpy == 42 for DCDI calibration configs.

    DCDI's fit path depends on global PyTorch and NumPy RNG state
    during fit. The frozen fit-RNG convention pins both setters to
    the same integer (42) for every DCDI fit during calibration.
    DAGMA configurations are not subject to this check because the
    DAGMA wrapper does not call any global RNG setter.
    """
    if config.seed_torch != _DCDI_CALIBRATION_FIT_RNG_VALUE:
        raise ValueError(
            "calibration real-study protocol violation: DCDI "
            "calibration configurations must carry "
            f"seed_torch={_DCDI_CALIBRATION_FIT_RNG_VALUE}; got "
            f"{config.seed_torch!r}"
        )
    if config.seed_numpy != _DCDI_CALIBRATION_FIT_RNG_VALUE:
        raise ValueError(
            "calibration real-study protocol violation: DCDI "
            "calibration configurations must carry "
            f"seed_numpy={_DCDI_CALIBRATION_FIT_RNG_VALUE}; got "
            f"{config.seed_numpy!r}"
        )


def _enforce_calibration_grid(config: Configuration) -> None:
    """Require the parent calibration grid to match the frozen 5-point set.

    A parent calibration Configuration must carry exactly five
    ``CalibrationConfiguration`` entries whose hyperparameter values
    cover the model's frozen sparsity grid. Each entry must carry
    exactly one hyperparameter override whose name is ``"lambda1"``
    for DAGMA or ``"reg_coeff"`` for DCDI. Order is not checked
    because the schema sorts hyperparameter entries lexically on
    load; the values are compared as an unordered set with a
    floating-point absolute tolerance.
    """
    if config.model == "dagma":
        hyperparameter_name = _DAGMA_CALIBRATION_HYPERPARAMETER
        expected_grid = _DAGMA_CALIBRATION_GRID
    else:
        hyperparameter_name = _DCDI_CALIBRATION_HYPERPARAMETER
        expected_grid = _DCDI_CALIBRATION_GRID

    entries = config.calibration_configurations
    if len(entries) != len(expected_grid):
        raise ValueError(
            "calibration real-study protocol violation: "
            "calibration_configurations must hold exactly "
            f"{len(expected_grid)} grid points for model "
            f"{config.model!r}; got {len(entries)}"
        )

    seen_values: list[float] = []
    for entry in entries:
        hyperparameters = dict(entry.hyperparameters)
        if list(hyperparameters.keys()) != [hyperparameter_name]:
            raise ValueError(
                "calibration real-study protocol violation: each "
                "calibration grid point must carry exactly the "
                f"hyperparameter {hyperparameter_name!r} for model "
                f"{config.model!r}; got hyperparameters="
                f"{list(hyperparameters.keys())!r} on entry "
                f"name={entry.name!r}"
            )
        seen_values.append(float(hyperparameters[hyperparameter_name]))

    if len(seen_values) != len(set(round(v, 12) for v in seen_values)):
        raise ValueError(
            "calibration real-study protocol violation: duplicate "
            f"hyperparameter value(s) in {hyperparameter_name!r} grid: "
            f"{sorted(seen_values)!r}"
        )

    sorted_observed = sorted(seen_values)
    sorted_expected = sorted(expected_grid)
    if len(sorted_observed) != len(sorted_expected):
        raise ValueError(
            "calibration real-study protocol violation: "
            f"{hyperparameter_name!r} grid size mismatch; got "
            f"{sorted_observed!r}; expected {sorted_expected!r}"
        )
    for observed, expected in zip(sorted_observed, sorted_expected):
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise ValueError(
                "calibration real-study protocol violation: "
                f"{hyperparameter_name!r} grid mismatch; got sorted "
                f"values {sorted_observed!r}; expected sorted values "
                f"{sorted_expected!r}"
            )


def _assert_reproduction_pass(config: Configuration) -> None:
    """Validate a Configuration for the reproduction_pass stage."""
    for field_name, expected_value in _SHARED_REQUIRED_VALUES:
        _enforce_field(
            config,
            stage="reproduction_pass",
            field_name=field_name,
            expected=expected_value,
        )

    if config.model == "dagma":
        for field_name, expected_value in _DAGMA_REQUIRED_VALUES:
            _enforce_field(
                config,
                stage="reproduction_pass",
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            stage="reproduction_pass",
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
                stage="reproduction_pass",
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            stage="reproduction_pass",
            field_names=_DCDI_REQUIRED_NONE_FIELDS,
            reason=(
                "DCDI reproduction-pass configurations must leave "
                "every DAGMA-only field None"
            ),
        )
    else:
        raise ValueError(
            "reproduction_pass real-study protocol violation: "
            f"model must be 'dagma' or 'dcdi'; got {config.model!r}"
        )

    _enforce_reproduction_population(config)


def _assert_calibration(config: Configuration) -> None:
    """Validate a Configuration for the calibration stage."""
    for field_name, expected_value in _SHARED_REQUIRED_VALUES:
        _enforce_field(
            config,
            stage="calibration",
            field_name=field_name,
            expected=expected_value,
        )

    if config.model == "dagma":
        for field_name, expected_value in _DAGMA_REQUIRED_VALUES:
            _enforce_field(
                config,
                stage="calibration",
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            stage="calibration",
            field_names=_DAGMA_REQUIRED_NONE_FIELDS,
            reason=(
                "DAGMA calibration configurations must leave every "
                "DCDI-only field None"
            ),
        )
    elif config.model == "dcdi":
        for field_name, expected_value in _DCDI_REQUIRED_VALUES:
            _enforce_field(
                config,
                stage="calibration",
                field_name=field_name,
                expected=expected_value,
            )
        _enforce_none(
            config,
            stage="calibration",
            field_names=_DCDI_REQUIRED_NONE_FIELDS,
            reason=(
                "DCDI calibration configurations must leave every "
                "DAGMA-only field None"
            ),
        )
        _enforce_dcdi_calibration_fit_rng(config)
    else:
        raise ValueError(
            "calibration real-study protocol violation: "
            f"model must be 'dagma' or 'dcdi'; got {config.model!r}"
        )

    _enforce_calibration_population(config)
    _enforce_calibration_grid(config)


def assert_real_study_constants(
    config: Configuration, *, stage: str
) -> None:
    """Verify a Configuration carries real-study constants for ``stage``.

    Parameters
    ----------
    config : Configuration
        Configuration to validate against the real-study protocol.
    stage : str
        Stage name. Accepted values are ``"reproduction_pass"`` and
        ``"calibration"``. Each stage has its own validation rules;
        the reproduction-pass rules are unchanged by the addition
        of the calibration stage.

    Raises
    ------
    ValueError
        On unknown stage, on field value mismatch, on a required
        field being None when it must be non-None (or vice versa),
        on a missing or mis-sized seed population, on a wrong
        calibration grid for the calibration stage, or on a DCDI
        calibration Configuration whose fit-RNG fields are not
        pinned to the frozen value 42. Each error names the
        offending field and the expected value.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(
            "unknown stage for assert_real_study_constants: "
            f"{stage!r}; accepted stages: {_VALID_STAGES!r}"
        )

    if stage == "reproduction_pass":
        _assert_reproduction_pass(config)
    else:
        _assert_calibration(config)


__all__ = [
    "assert_real_study_constants",
]
