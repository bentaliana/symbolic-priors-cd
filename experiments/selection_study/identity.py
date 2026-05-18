"""Run identity and directory creation for the selection-study runner.

This module will derive ``run_id`` strings and run-directory paths
from the identity tuple ``(model, condition, seed_population,
seed_replicate_index, configuration_hash)``, and will enforce the
no-overwrite rule. The current state contains placeholders only.
"""

from __future__ import annotations

from typing import Any, NoReturn


def derive_run_id(identity: Any) -> NoReturn:
    """Derive the canonical ``run_id`` string from an identity tuple.

    Parameters
    ----------
    identity : Any
        The identity tuple or record from which the ``run_id`` is
        produced. The concrete type is not fixed in the current
        state.

    Raises
    ------
    NotImplementedError
        Always. ``run_id`` derivation is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.identity.derive_run_id is not "
        "implemented yet."
    )


def derive_run_directory(identity: Any) -> NoReturn:
    """Derive the run-directory path from an identity tuple.

    Parameters
    ----------
    identity : Any
        The identity tuple or record from which the run-directory
        path is produced. The concrete type is not fixed in the
        current state.

    Raises
    ------
    NotImplementedError
        Always. Directory-path derivation is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.identity.derive_run_directory "
        "is not implemented yet."
    )
