"""Command-line entry point for the selection-study runner.

The CLI accepts the following flags: ``--help``, ``--config``,
``--dry-run``, ``--resume``, ``--phase``, and ``--output-root``.
The ``--dry-run`` path is functional: it runs preflight manifest
enumeration and validation and exits. The ``--phase reproduction_pass``
path is functional: it loads the configuration, validates it
against the real-study protocol guard, runs the reproduction pass,
and writes a reproduction-pass summary JSON. All other non-help
execution paths raise ``NotImplementedError`` with a message naming
the unimplemented path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path


_LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the runner CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the recognised flags ``--config``,
        ``--dry-run``, ``--resume``, ``--phase``, and
        ``--output-root``. The ``--help`` flag is added automatically
        by ``argparse``.
    """
    parser = argparse.ArgumentParser(
        prog="experiments.selection_study.run",
        description=(
            "Base-model selection-study runner. "
            "Drives the reproduction pass, calibration, and held-out "
            "evaluation under the selection-study protocol."
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
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=("reproduction_pass",),
        metavar="STAGE",
        help=(
            "Selection-study phase to drive. Only 'reproduction_pass' "
            "is implemented in the current state."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Run-storage base directory. Defaults to "
            "'results/model_selection/' when omitted."
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    _base_dir: Path | None = None,
    _manifest_dir: Path | None = None,
) -> None:
    """Run the selection-study runner CLI.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector. When ``None``, ``argparse`` reads
        ``sys.argv[1:]``.
    _base_dir : pathlib.Path or None, optional
        Testing hook: override the run-storage base directory passed to
        ``preflight.run_preflight``. When ``None``, the preflight default
        is used.
    _manifest_dir : pathlib.Path or None, optional
        Testing hook: override the manifest-storage directory passed to
        ``preflight.run_preflight``. When ``None``, the preflight default
        is used.

    Raises
    ------
    ValueError
        If ``--dry-run`` or ``--phase reproduction_pass`` is passed
        without ``--config``.
    NotImplementedError
        For still-unimplemented execution paths (``--resume`` and the
        bare ``--config`` path with no ``--phase``). The message
        names the unimplemented path. The ``--phase reproduction_pass``
        and ``--dry-run`` paths are implemented and do not raise
        this.
    SystemExit
        With status 1 if preflight validation fails.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        if args.config is None:
            raise ValueError(
                "--dry-run requires --config PATH; no configuration file "
                "was supplied."
            )
        from experiments.selection_study.preflight import (
            ManifestValidationError,
            run_preflight,
        )

        preflight_kwargs: dict[str, Path] = {}
        if _base_dir is not None:
            preflight_kwargs["base_dir"] = _base_dir
        if _manifest_dir is not None:
            preflight_kwargs["manifest_dir"] = _manifest_dir

        try:
            manifest_path = run_preflight(
                Path(args.config), **preflight_kwargs
            )
            _LOGGER.info(
                "preflight passed; manifest saved to %s", manifest_path
            )
        except ManifestValidationError as exc:
            _LOGGER.error("preflight validation failed: %s", exc)
            sys.exit(1)
        return

    if args.resume:
        raise NotImplementedError(
            "experiments.selection_study.run --resume is not "
            "implemented yet."
        )
    if args.phase == "reproduction_pass":
        if args.config is None:
            raise ValueError(
                "--phase reproduction_pass requires --config PATH; "
                "no configuration file was supplied."
            )
        from experiments.selection_study.reproduction_pass import (
            run_reproduction_pass,
        )

        output_root: Path | None = (
            Path(args.output_root) if args.output_root is not None else None
        )
        summary = run_reproduction_pass(
            Path(args.config), output_root=output_root
        )
        _LOGGER.info(
            "reproduction_pass completed with status %s; summary at %s",
            summary.reproduction_pass_status,
            summary.summary_path,
        )
        return
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
