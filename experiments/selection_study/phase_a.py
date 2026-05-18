"""Phase A reproduction-pass runner.

This module will drive each model under paper-grounded defaults on
its paper-aligned reference cell and persist the resulting records
under the ``reproduction`` seed population. The current state
contains a placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def run_phase_a(config: Any) -> NoReturn:
    """Run the Phase A reproduction pass.

    Parameters
    ----------
    config : Any
        The resolved runner configuration. The concrete type is not
        fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. The Phase A runner is not implemented in the current
        state.
    """
    raise NotImplementedError(
        "experiments.selection_study.phase_a.run_phase_a is not "
        "implemented yet."
    )
