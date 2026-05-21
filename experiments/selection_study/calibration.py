"""Calibration runner.

This module will drive the equal-budget local calibration runs,
persist them under the ``calibration`` seed population, and apply
the frozen lexicographic calibration ranking rule to select the
configuration to be evaluated against held-out seeds. The current
state contains placeholders only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def run_calibration(config: Any) -> NoReturn:
    """Run the calibration runs.

    Parameters
    ----------
    config : Any
        The resolved runner configuration. The concrete type is not
        fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. The calibration runner is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.calibration.run_calibration is "
        "not implemented yet."
    )


def calibration_ranking(records: Any) -> NoReturn:
    """Apply the calibration ranking rule to a set of records.

    Parameters
    ----------
    records : Any
        Calibration run records. The concrete type is not fixed in
        the current state.

    Raises
    ------
    NotImplementedError
        Always. The calibration ranking is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.calibration.calibration_ranking "
        "is not implemented yet."
    )
