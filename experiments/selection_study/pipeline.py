"""Single-fit pipeline for the selection-study runner.

This module will drive a single wrapper fit per preflight-validated
manifest entry, emit the canonical run record, and write the binary
artefacts referenced from that record. The current state contains a
placeholder only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def run_single_fit(manifest_entry: Any) -> NoReturn:
    """Run a single wrapper fit for one manifest entry.

    Parameters
    ----------
    manifest_entry : Any
        One preflight-validated manifest entry. The concrete type is
        not fixed in the current state.

    Raises
    ------
    NotImplementedError
        Always. The single-fit pipeline is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.pipeline.run_single_fit is not "
        "implemented yet."
    )
