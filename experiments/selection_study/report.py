"""Selection-study report generator.

This module will read saved run records through the loader and
produce the one-page selection-study report (per-criterion summary,
threshold-robustness table, MMD missingness disclosure, declared
base-model decision under the lexicographic rule). The current state
contains a placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def generate_report(record_set: Any) -> NoReturn:
    """Produce the selection-study report from a set of run records.

    Parameters
    ----------
    record_set : Any
        Set of run records, typically obtained via the loader. The
        concrete type is not fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. Report generation is not implemented in the current
        state.
    """
    raise NotImplementedError(
        "experiments.selection_study.report.generate_report is not "
        "implemented yet."
    )
