"""Threshold-robustness offline re-computation.

This module will reload saved continuous edge objects and recompute
structural metrics at the documented threshold triples without
retraining. The current state contains a placeholder only.
"""

from __future__ import annotations

from typing import NoReturn


def recompute_at_thresholds(run_id: str) -> NoReturn:
    """Recompute structural metrics at the per-model threshold triple.

    Parameters
    ----------
    run_id : str
        Canonical identifier of the run whose continuous edge object
        is re-thresholded.

    Raises
    ------
    NotImplementedError
        Always. Offline threshold-robustness re-computation is not
        implemented in the current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.threshold_robustness."
        "recompute_at_thresholds is not implemented yet."
    )
