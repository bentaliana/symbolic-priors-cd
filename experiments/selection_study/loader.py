"""Loader interface for selection-study run records.

This module will expose ``load_run`` and ``load_runs``, the typed
interface notebooks and the report generator consume to read saved
run records and their referenced binary artefacts. The current state
contains placeholders only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def load_run(run_id: str) -> NoReturn:
    """Load a single run record by its canonical identifier.

    Parameters
    ----------
    run_id : str
        Canonical run identifier.

    Raises
    ------
    NotImplementedError
        Always. Single-run loading is not implemented in the current
        state.
    """
    raise NotImplementedError(
        "experiments.selection_study.loader.load_run is not "
        "implemented yet."
    )


def load_runs(filter_spec: Any) -> NoReturn:
    """Load a collection of run records that match a filter.

    Parameters
    ----------
    filter_spec : Any
        Filter over identity fields. The concrete type is not fixed
        in the current state.

    Raises
    ------
    NotImplementedError
        Always. Multi-run loading is not implemented in the current
        state.
    """
    raise NotImplementedError(
        "experiments.selection_study.loader.load_runs is not "
        "implemented yet."
    )
