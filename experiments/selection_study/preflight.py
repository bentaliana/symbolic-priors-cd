"""Preflight manifest validation for the selection-study runner.

This module will enumerate every planned run, validate the manifest,
and save the validated manifest before any fit can be invoked. The
current state contains a placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def run_preflight(config: Any) -> NoReturn:
    """Run the preflight manifest validation.

    Parameters
    ----------
    config : Any
        The resolved runner configuration. The concrete type is not
        fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. Preflight is not implemented in the current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.preflight.run_preflight is not "
        "implemented yet."
    )
