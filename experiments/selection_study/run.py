"""Command-line entry point for the selection-study runner.

The CLI accepts four flags: ``--help``, ``--config``, ``--dry-run``,
and ``--resume``. Only the ``--help`` path is functional in the
current state. Every non-help execution path raises
``NotImplementedError`` with a message naming the unimplemented path.
No model fit is reachable from any code path in this module.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence


_LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the runner CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the recognised flags ``--config``,
        ``--dry-run``, and ``--resume``. The ``--help`` flag is added
        automatically by ``argparse``.
    """
    parser = argparse.ArgumentParser(
        prog="experiments.selection_study.run",
        description=(
            "Base-model selection-study runner. "
            "Drives Phase A reproduction, Phase B calibration, and "
            "held-out evaluation under the selection-study protocol."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the runner configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight only; no fits are invoked.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a halted run from the existing "
            "results/model_selection/ tree."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selection-study runner CLI.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector. When ``None``, ``argparse`` reads
        ``sys.argv[1:]``.

    Raises
    ------
    NotImplementedError
        For every execution path other than ``--help``. The message
        names the unimplemented path.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        raise NotImplementedError(
            "experiments.selection_study.run --dry-run is not "
            "implemented yet."
        )
    if args.resume:
        raise NotImplementedError(
            "experiments.selection_study.run --resume is not "
            "implemented yet."
        )
    if args.config is not None:
        raise NotImplementedError(
            "experiments.selection_study.run --config is not "
            "implemented yet; configuration loading is not wired "
            "into the runner."
        )
    raise NotImplementedError(
        "experiments.selection_study.run normal execution is not "
        "implemented yet."
    )


if __name__ == "__main__":
    main()
