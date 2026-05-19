"""Scaffolding tests for the selection-study runner.

These tests verify that the runner package imports cleanly, that the
``--help`` flag exits with status 0 and prints a usage string, that
no model fit is reachable from any code path under
``experiments/selection_study/``, that every stub function raises
``NotImplementedError`` when called, and that importing the runner
does not mutate global NumPy or PyTorch RNG state.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PACKAGE_DIR = PROJECT_ROOT / "experiments" / "selection_study"


def test_runner_package_imports_cleanly() -> None:
    """The runner package must import without raising."""
    import experiments.selection_study  # noqa: F401


def test_runner_run_module_imports_cleanly() -> None:
    """The CLI entry-point module must import without raising."""
    import experiments.selection_study.run  # noqa: F401


def test_cli_help_exits_zero_and_prints_usage() -> None:
    """``--help`` exits with status 0 and prints a usage string."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.selection_study.run",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help returned non-zero status {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    assert "usage:" in result.stdout.lower(), (
        f"--help stdout did not contain a usage string; "
        f"stdout={result.stdout!r}"
    )


def test_cli_help_does_not_import_dagma_or_dcdi_or_wrappers() -> None:
    """No wrapper, DAGMA, or DCDI module appears under ``sys.modules``
    after ``--help`` is invoked via the CLI entry point in a
    subprocess.

    ``argparse`` prints the usage text to ``stdout`` and calls
    ``sys.exit(0)``. The probe redirects ``stdout`` to ``os.devnull``
    around the ``--help`` invocation so the forbidden-modules report
    is the only content that reaches the parent process.
    """
    probe = (
        "import os\n"
        "import sys\n"
        "import experiments.selection_study.run as r\n"
        "_devnull = open(os.devnull, 'w')\n"
        "_saved_stdout = sys.stdout\n"
        "_saved_stderr = sys.stderr\n"
        "sys.stdout = _devnull\n"
        "sys.stderr = _devnull\n"
        "try:\n"
        "    try:\n"
        "        r.main(['--help'])\n"
        "    except SystemExit:\n"
        "        pass\n"
        "finally:\n"
        "    sys.stdout = _saved_stdout\n"
        "    sys.stderr = _saved_stderr\n"
        "    _devnull.close()\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if 'dagma' in m.lower()\n"
        "                or 'dcdi' in m.lower()\n"
        "                or m.startswith('symbolic_priors_cd.wrappers'))\n"
        "sys.stdout.write('FORBIDDEN_MODULES=' + ';'.join(loaded) + '\\n')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"probe failed with status {result.returncode}; "
        f"stderr={result.stderr!r}"
    )
    marker = "FORBIDDEN_MODULES="
    matches = [
        line[len(marker):]
        for line in result.stdout.splitlines()
        if line.startswith(marker)
    ]
    assert len(matches) == 1, (
        "probe did not emit exactly one FORBIDDEN_MODULES line; "
        f"stdout={result.stdout!r}"
    )
    loaded = matches[0].strip()
    assert loaded == "", (
        "--help path loaded forbidden modules: "
        f"{loaded!r}"
    )


def test_runner_source_contains_no_dagma_or_dcdi_or_wrapper_imports() -> None:
    """Static check that no source file under the runner package
    imports DAGMA, DCDI, or the project wrappers as Python modules.

    The patterns match only import statements at the start of a line.
    Bare string occurrences of ``"dagma"`` or ``"dcdi"`` are
    legitimate (for example, ``Literal["dagma", "dcdi"]`` annotations
    or string-typed wrapper-API references); only an executable
    ``import`` or ``from`` statement is forbidden.
    """
    forbidden_patterns = [
        re.compile(r"^\s*(import|from)\s+dagma\b", re.MULTILINE),
        re.compile(r"^\s*(import|from)\s+dcdi\b", re.MULTILINE),
        re.compile(
            r"^\s*from\s+symbolic_priors_cd\.wrappers\b",
            re.MULTILINE,
        ),
    ]
    runner_files = sorted(RUNNER_PACKAGE_DIR.glob("*.py"))
    assert runner_files, "no runner source files were discovered"
    for source_file in runner_files:
        text = source_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not pattern.search(text), (
                f"forbidden pattern {pattern.pattern!r} appears in "
                f"{source_file.name}"
            )


def test_resume_raises_not_implemented_error() -> None:
    """``--resume`` is a placeholder and must raise
    ``NotImplementedError`` rather than invoking a fit.
    """
    from experiments.selection_study.run import main

    with pytest.raises(NotImplementedError) as excinfo:
        main(["--resume"])
    assert "--resume" in str(excinfo.value)


def test_normal_execution_raises_not_implemented_error() -> None:
    """A no-flag invocation raises ``NotImplementedError`` because
    normal execution is a placeholder.
    """
    from experiments.selection_study.run import main

    with pytest.raises(NotImplementedError) as excinfo:
        main([])
    assert "normal execution" in str(excinfo.value)


def test_config_path_raises_not_implemented_error_without_reading_file(
    tmp_path: Path,
) -> None:
    """``--config`` is parsed but not validated or read at this stage.

    The CLI is invoked with a ``--config`` path that does not exist on
    disk. The expected outcome is ``NotImplementedError`` (not
    ``FileNotFoundError`` or any other I/O error), which proves the
    runner does not attempt to read the configuration file. The
    raised message must name the ``--config`` surface so callers can
    tell that the unimplemented path is configuration handling, not
    the generic normal-execution fallthrough.
    """
    from experiments.selection_study.run import main

    missing_config_path = tmp_path / "does_not_exist.yaml"
    assert not missing_config_path.exists(), (
        "the missing-config fixture must not exist on disk"
    )

    with pytest.raises(NotImplementedError) as excinfo:
        main(["--config", str(missing_config_path)])
    message = str(excinfo.value)
    assert "--config" in message or "config" in message, (
        "NotImplementedError message did not mention --config or "
        f"config: {message!r}"
    )


def test_every_stub_module_callable_raises_not_implemented_error() -> None:
    """Each placeholder function in the runner package raises
    ``NotImplementedError`` when called.
    """
    from experiments.selection_study import (
        config,
        held_out,
        identity,
        loader,
        phase_a,
        phase_b,
        pipeline,
        preflight,
        report,
        resume,
        sampling,
        threshold_robustness,
    )

    stub_callables: list[tuple[object, tuple[object, ...]]] = [
        (threshold_robustness.recompute_at_thresholds, ("run-id",)),
        (phase_a.run_phase_a, (None,)),
        (phase_b.run_phase_b, (None,)),
        (phase_b.calibration_ranking, (None,)),
        (held_out.run_held_out_evaluation, (None,)),
        (resume.resume_run, (None,)),
        (report.generate_report, (None,)),
        (loader.load_runs, (None,)),
    ]
    for stub, args in stub_callables:
        with pytest.raises(NotImplementedError):
            stub(*args)  # type: ignore[operator]
    # pipeline.run_single_fit, loader.load_run, and
    # sampling.compute_per_intervention_records are no longer stubs;
    # each is exercised under its own test module.
    _ = pipeline.run_single_fit
    _ = loader.load_run
    _ = sampling.compute_per_intervention_records


def _numpy_states_equal(
    before: tuple[object, ...], after: tuple[object, ...]
) -> bool:
    """Return True when two ``numpy`` legacy-RNG state tuples are equal."""
    if len(before) != len(after):
        return False
    for left, right in zip(before, after):
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not np.array_equal(left, right):
                return False
        else:
            if left != right:
                return False
    return True


def test_fresh_import_does_not_mutate_global_rng_state() -> None:
    """Importing the runner package from a clean ``sys.modules`` state
    does not change global ``numpy`` or ``torch`` RNG state.
    """
    np_state_before = np.random.get_state()
    torch_state_before = torch.random.get_rng_state().clone()

    for module_name in list(sys.modules):
        if module_name == "experiments" or module_name.startswith(
            "experiments."
        ):
            del sys.modules[module_name]

    import experiments.selection_study  # noqa: F401
    import experiments.selection_study.run  # noqa: F401
    from experiments.selection_study import (  # noqa: F401
        config,
        held_out,
        identity,
        loader,
        phase_a,
        phase_b,
        pipeline,
        preflight,
        report,
        resume,
        sampling,
        threshold_robustness,
    )

    np_state_after = np.random.get_state()
    torch_state_after = torch.random.get_rng_state()

    assert _numpy_states_equal(np_state_before, np_state_after), (
        "importing the runner mutated the global numpy RNG state"
    )
    assert torch.equal(torch_state_before, torch_state_after), (
        "importing the runner mutated the global torch RNG state"
    )
