"""Per-intervention MMD sampling pipeline for the selection-study runner.

This module will compute per-intervention records, populate the
within-record consistency between ``sampler_status_for_intervention``,
``mmd_status``, and ``mmd_value``, and produce the derived run-level
MMD aggregates. The current state contains a placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def compute_per_intervention_records(manifest_entry: Any) -> NoReturn:
    """Compute the per-intervention records for one fitted wrapper.

    Parameters
    ----------
    manifest_entry : Any
        The manifest entry whose fit produced the wrapper under
        evaluation. The concrete type is not fixed in the current
        state.

    Raises
    ------
    NotImplementedError
        Always. Per-intervention sampling is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.sampling."
        "compute_per_intervention_records is not implemented yet."
    )
