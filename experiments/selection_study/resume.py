"""Halt-and-resume semantics for the selection-study runner.

This module will read the existing ``results/model_selection/`` tree,
identify completed runs, classify partial directories, and skip
completed runs on a resumed pass. The current state contains a
placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def resume_run(config: Any) -> NoReturn:
    """Resume a halted selection-study run.

    Parameters
    ----------
    config : Any
        The resolved runner configuration. The concrete type is not
        fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. Halt-and-resume is not implemented in the current
        state.
    """
    raise NotImplementedError(
        "experiments.selection_study.resume.resume_run is not "
        "implemented yet."
    )
