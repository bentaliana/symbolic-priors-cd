"""Held-out evaluation runner.

This module will run the configuration selected by the calibration
phase on the held-out evaluation seeds, enforce the calibration vs
held-out non-overlap rule at run time, and persist the resulting
records under the ``held_out_evaluation`` seed population. The
current state contains a placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def run_held_out_evaluation(config: Any) -> NoReturn:
    """Run the held-out evaluation runs.

    Parameters
    ----------
    config : Any
        The resolved runner configuration. The concrete type is not
        fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. The held-out evaluation runner is not implemented in
        the current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.held_out.run_held_out_evaluation "
        "is not implemented yet."
    )
