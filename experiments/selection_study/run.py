"""Command-line entry point for the selection-study runner.

The CLI accepts the following flags: ``--help``, ``--config``,
``--dry-run``, ``--resume``, ``--phase``, and ``--output-root``.
The ``--dry-run`` path is functional: it runs preflight manifest
enumeration and validation and exits. The ``--phase reproduction_pass``
path is functional: it loads the configuration, validates it
against the real-study protocol guard, runs the reproduction pass,
and writes a reproduction-pass summary JSON.

The ``--phase calibration`` path is enumeration-only in the current
state. It loads the four parent calibration Configurations from the
directory supplied via ``--config``, validates each against the
calibration-stage real-study guard, expands each parent into one
executable Configuration per sparsity grid point, and prints the
workload arithmetic (20 executable candidates; 40 fit jobs after
calibration-seed expansion). No model fit is invoked from this code
path. Real calibration execution, ranking, and the selected-
configurations artefact writer are introduced later.

All other non-help execution paths raise ``NotImplementedError`` with
a message naming the unimplemented path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path


_LOGGER = logging.getLogger(__name__)


_CALIBRATION_PARENT_FILENAMES: tuple[str, ...] = (
    "dagma_calibration_centred_only.json",
    "dagma_calibration_standardised.json",
    "dcdi_calibration_centred_only.json",
    "dcdi_calibration_standardised.json",
)


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
        help=(
            "Path to the runner configuration. For "
            "--phase reproduction_pass this is a single JSON file. "
            "For --phase calibration this is a directory containing "
            "the four parent calibration configs "
            "(dagma_calibration_centred_only.json, "
            "dagma_calibration_standardised.json, "
            "dcdi_calibration_centred_only.json, "
            "dcdi_calibration_standardised.json)."
        ),
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
        choices=("reproduction_pass", "calibration"),
        metavar="STAGE",
        help=(
            "Selection-study phase to drive. "
            "'reproduction_pass' runs the reproduction-pass stage "
            "end-to-end. 'calibration' currently runs enumeration "
            "and the calibration-stage real-study guard only, "
            "without invoking any model fit."
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


def _resolve_calibration_parent_paths(config_dir: Path) -> tuple[Path, ...]:
    """Return the four expected parent calibration config paths.

    The runner expects all four files to be present in the supplied
    directory; a missing file is an explicit error rather than a
    silent partial-workload condition.
    """
    if not config_dir.exists():
        raise FileNotFoundError(
            "calibration --config directory does not exist: "
            f"{config_dir}"
        )
    if not config_dir.is_dir():
        raise NotADirectoryError(
            "calibration --config must be a directory containing "
            "the four parent calibration JSON files; got a path "
            f"that is not a directory: {config_dir}"
        )

    paths: list[Path] = []
    missing: list[str] = []
    for filename in _CALIBRATION_PARENT_FILENAMES:
        candidate = config_dir / filename
        if not candidate.is_file():
            missing.append(filename)
        else:
            paths.append(candidate)
    if missing:
        raise FileNotFoundError(
            "calibration --config directory is missing required "
            "parent config file(s): " + ", ".join(missing)
            + f"; directory={config_dir}"
        )
    return tuple(paths)


def _run_phase_calibration_enumeration(config_dir_arg: str) -> None:
    """Drive the enumeration-only calibration code path.

    Loads the four parent calibration configs, validates each via
    the calibration-stage real-study guard inside
    ``enumerate_calibration_workload``, expands each parent into per-
    grid-point executable Configurations, and logs the workload
    arithmetic. No model fit is invoked.
    """
    from experiments.selection_study.calibration import (
        enumerate_calibration_workload,
    )
    from experiments.selection_study.config import load_config

    config_dir = Path(config_dir_arg)
    parent_paths = _resolve_calibration_parent_paths(config_dir)
    parents = tuple(load_config(path) for path in parent_paths)
    workload = enumerate_calibration_workload(parents)

    _LOGGER.info(
        "calibration enumeration: %d executable candidates, "
        "%d fit jobs, calibration seeds %s",
        len(workload.candidates),
        len(workload.fit_jobs),
        list(workload.calibration_seeds),
    )
    for candidate in workload.candidates:
        _LOGGER.info(
            "calibration candidate: model=%s condition=%s "
            "grid_point=%s configuration_hash_prefix=%s",
            candidate.model,
            candidate.condition,
            candidate.grid_point_name,
            candidate.configuration_hash_prefix,
        )


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
        If ``--dry-run``, ``--phase reproduction_pass``, or
        ``--phase calibration`` is passed without ``--config``.
    NotImplementedError
        For still-unimplemented execution paths (``--resume`` and the
        bare ``--config`` path with no ``--phase``). The message
        names the unimplemented path.
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
    if args.phase == "calibration":
        if args.config is None:
            raise ValueError(
                "--phase calibration requires --config PATH pointing "
                "to a directory containing the four parent "
                "calibration configs; no configuration was supplied."
            )
        _run_phase_calibration_enumeration(args.config)
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
