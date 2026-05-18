"""Configuration handling for the selection-study runner.

This module will host the runner's configuration object, the
canonical JSON serialisation, the SHA-256 configuration hash, and
the per-purpose seed-derivation rule. The current state contains a
placeholder only.
"""

from __future__ import annotations

from typing import NoReturn


def load_config(path: str) -> NoReturn:
    """Load and resolve the selection-study configuration.

    Parameters
    ----------
    path : str
        Filesystem path to a configuration file.

    Raises
    ------
    NotImplementedError
        Always. Configuration loading is not implemented in the
        current state.
    """
    raise NotImplementedError(
        "experiments.selection_study.config.load_config is not "
        "implemented yet."
    )
