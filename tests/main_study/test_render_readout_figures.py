"""Tests for the readout renderer (figures plus labelling summary).

All tests use small synthetic CSVs under ``tmp_path``. Real persisted
records and upstream readout tables are never touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.main_study import render_readout_figures as rrf
from experiments.main_study.render_readout_figures import (
    BASELINE_CONDITION_LABELS,
    BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
    BASELINE_LABEL_MATCHED_L1,
    BASELINE_LABEL_PRIOR_FREE,
    BASELINE_LABEL_SOFT_CLEAN_CONF1,
    presentation_table,
    FIG_BASELINE,
    FIG_DEGRADATION_MMD,
    FIG_DEGRADATION_SID,
    FIG_EDGE_COUNT_DIAG,
    FIG_REFERENCE_FORBIDDEN,
    FIG_SID_MMD_SCATTER,
    FIG_SOFT_MMD_HEATMAP,
    FIG_SOFT_SID_HEATMAP,
    READOUT_SUMMARY_FILENAME,
    ensure_output_dirs,
    main as cli_main,
    main_evaluation_readout_dir,
    plot_baseline_comparison,
    plot_degradation_curve,
    plot_edge_count_engagement_diagnostic,
    plot_reference_forbidden_edge_suppression,
    plot_sid_mmd_scatter,
    plot_soft_frobenius_heatmap,
    read_csv_table,
    read_statistics_summary,
    render_all_readout_outputs,
    write_readout_summary,
)


_RUN_HASH12 = "abcdef012345"
_SEEDS = (501, 502, 503, 504, 505, 506, 507)
_CORRUPTIONS = (0.0, 0.2, 0.4, 0.6, 0.8)
_CONFIDENCES = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


_FLAT_COLUMNS: tuple[str, ...] = (
    "run_id", "configuration_hash_full", "configuration_hash_prefix",
    "record_path",
    "method_family", "seed_value", "seed_population",
    "confidence", "corruption_fraction", "corruption_index",
    "lambda_prior", "matched_l1_lambda1", "dagma_lambda1",
    "parent_heldout_run_hash_full",
    "fit_status", "metric_status", "graph_status", "sampler_status",
    "sid", "shd", "mmd", "edge_count_from_thresholded_adjacency",
    "continuous_w_path", "thresholded_adjacency_path",
    "true_adjacency_path",
    "n_targeted_forbidden_edges",
    "mean_abs_w_targeted_forbidden_edges",
    "fraction_targeted_forbidden_above_threshold",
    "mean_abs_w_non_targeted_edges",
)


def _flat_row(
    *,
    method_family: str,
    seed: int,
    confidence: float | None = None,
    corruption_fraction: float | None = None,
    sid: float = 5.0,
    shd: float = 3.0,
    mmd: float = 0.02,
    edge_count: int = 18,
    lambda_prior: float | None = None,
    matched_l1_lambda1: float | None = None,
    n_targeted: int | None = None,
    mean_abs_targeted: float | None = None,
    frac_above_threshold: float | None = None,
    mean_abs_non_targeted: float | None = None,
) -> dict[str, Any]:
    return {
        "run_id": (
            f"{method_family}__main_evaluation__seed{seed}__cfg"
            f"{method_family[:8]:>8}"
        ),
        "configuration_hash_full": "a" * 64,
        "configuration_hash_prefix": "abcdef012345",
        "record_path": "x.json",
        "method_family": method_family,
        "seed_value": seed,
        "seed_population": "main_evaluation",
        "confidence": confidence,
        "corruption_fraction": corruption_fraction,
        "corruption_index": (
            None if corruption_fraction is None
            else _CORRUPTIONS.index(corruption_fraction)
        ),
        "lambda_prior": lambda_prior,
        "matched_l1_lambda1": matched_l1_lambda1,
        "dagma_lambda1": (
            matched_l1_lambda1 if matched_l1_lambda1 is not None
            else 0.05
        ),
        "parent_heldout_run_hash_full": "a" * 64,
        "fit_status": "success",
        "metric_status": "computed",
        "graph_status": "valid_dag",
        "sampler_status": "available",
        "sid": sid,
        "shd": shd,
        "mmd": mmd,
        "edge_count_from_thresholded_adjacency": edge_count,
        "continuous_w_path": "x.npz",
        "thresholded_adjacency_path": "y.npz",
        "true_adjacency_path": "z.npz",
        "n_targeted_forbidden_edges": n_targeted,
        "mean_abs_w_targeted_forbidden_edges": mean_abs_targeted,
        "fraction_targeted_forbidden_above_threshold": frac_above_threshold,
        "mean_abs_w_non_targeted_edges": mean_abs_non_targeted,
    }


def _build_synthetic_flat() -> pd.DataFrame:
    rows = []
    for seed in _SEEDS:
        # prior_free
        rows.append(_flat_row(
            method_family="prior_free", seed=seed,
            sid=8.0 + (seed - 501) * 0.5,
            shd=5.0, mmd=0.04, edge_count=20,
        ))
        # matched_l1
        rows.append(_flat_row(
            method_family="matched_l1", seed=seed,
            sid=7.0 + (seed - 501) * 0.5,
            shd=4.5, mmd=0.035, edge_count=18,
            matched_l1_lambda1=0.0625,
        ))
        # hard_exclusion across corruption
        for cf in _CORRUPTIONS:
            rows.append(_flat_row(
                method_family="hard_exclusion",
                seed=seed,
                corruption_fraction=cf,
                sid=5.0 + cf * 10,
                shd=3.0 + cf * 5,
                mmd=0.03 + cf * 0.04,
                edge_count=int(15 + cf * 6),
                n_targeted=10,
                mean_abs_targeted=0.01 + cf * 0.05,
                frac_above_threshold=0.1 + cf * 0.05,
                mean_abs_non_targeted=0.2,
            ))
        # soft_frobenius 5x5
        for cf in _CORRUPTIONS:
            for cn in _CONFIDENCES:
                rows.append(_flat_row(
                    method_family="soft_frobenius",
                    seed=seed,
                    confidence=cn,
                    corruption_fraction=cf,
                    sid=4.0 + cf * 8 + (1 - cn) * 2,
                    shd=2.0 + cf * 4 + (1 - cn),
                    mmd=0.02 + cf * 0.05 + (1 - cn) * 0.01,
                    edge_count=int(12 + cf * 4),
                    lambda_prior=2e-4,
                    n_targeted=10,
                    mean_abs_targeted=0.01,
                    frac_above_threshold=0.05,
                    mean_abs_non_targeted=0.2,
                ))
    return pd.DataFrame(rows, columns=list(_FLAT_COLUMNS))


def _build_synthetic_reference_df() -> pd.DataFrame:
    rows = []
    for seed in _SEEDS:
        for label in BASELINE_CONDITION_LABELS:
            family = {
                BASELINE_LABEL_PRIOR_FREE: "prior_free",
                BASELINE_LABEL_MATCHED_L1: "matched_l1",
                BASELINE_LABEL_SOFT_CLEAN_CONF1: "soft_frobenius",
                BASELINE_LABEL_HARD_EXCLUSION_CLEAN: "hard_exclusion",
            }[label]
            cf = (
                None if label in (
                    BASELINE_LABEL_PRIOR_FREE,
                    BASELINE_LABEL_MATCHED_L1,
                ) else 0.0
            )
            cn = (
                1.0 if label == BASELINE_LABEL_SOFT_CLEAN_CONF1 else None
            )
            rows.append({
                "seed_value": seed,
                "condition_label": label,
                "method_family": family,
                "corruption_fraction": cf,
                "confidence": cn,
                "n_reference_forbidden_edges": 10,
                "mean_abs_w_reference_forbidden_edges": (
                    0.05 if label != BASELINE_LABEL_SOFT_CLEAN_CONF1
                    else 0.01
                ),
                "fraction_reference_forbidden_above_threshold": (
                    0.15 if label != BASELINE_LABEL_SOFT_CLEAN_CONF1
                    else 0.03
                ),
                "mean_abs_w_reference_non_targeted_edges": 0.2,
                "edge_count_from_thresholded_adjacency": 18,
                "sid": 6.0,
                "shd": 4.0,
                "mmd": 0.03,
            })
    return pd.DataFrame(rows)


def _build_synthetic_correlations() -> pd.DataFrame:
    rows = []
    for x_metric, y_metric in (
        ("sid", "mmd"),
        ("shd", "mmd"),
        ("edge_count_from_thresholded_adjacency", "mmd"),
        ("sid", "shd"),
    ):
        rows.append({
            "group_label": "all",
            "method_family": "",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "n": 224,
            "pearson": 0.42,
            "spearman": 0.38,
            "kendall_tau_b": 0.27,
        })
    for fam in (
        "prior_free", "matched_l1", "soft_frobenius", "hard_exclusion",
    ):
        for x_metric, y_metric in (("sid", "mmd"),):
            rows.append({
                "group_label": f"method_family:{fam}",
                "method_family": fam,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n": 7 if fam in ("prior_free", "matched_l1") else 35,
                "pearson": 0.3,
                "spearman": 0.3,
                "kendall_tau_b": 0.2,
            })
    return pd.DataFrame(rows)


def _build_synthetic_statistics_summary() -> dict[str, Any]:
    return {
        "main_evaluation_run_hash12": _RUN_HASH12,
        "input_flat_csv":
            f"results/main_study/main_evaluation/{_RUN_HASH12}/"
            f"readout/main_evaluation_flat_records.csv",
        "output_files": [],
        "n_flat_rows": 224,
        "n_baseline_rows": 16,
        "n_paired_comparison_rows": 20,
        "n_correlation_rows": 20,
        "n_degradation_rows": 24,
        "n_forbidden_engagement_rows": 32,
        "n_reference_forbidden_rows": 28,
        "n_per_intervention_mmd_rows": 4480,
        "n_per_intervention_mmd_summary_rows": 640,
        "no_plots_created": True,
        "no_notebook_created": True,
        "no_hypothesis_verdicts": True,
    }


def _seed_synthetic_inputs(tmp_path: Path) -> Path:
    readout_dir = main_evaluation_readout_dir(tmp_path, _RUN_HASH12)
    readout_dir.mkdir(parents=True, exist_ok=True)
    flat = _build_synthetic_flat()
    flat.to_csv(
        readout_dir / "main_evaluation_flat_records.csv", index=False
    )
    _build_synthetic_reference_df().to_csv(
        readout_dir / "reference_forbidden_edge_comparison.csv",
        index=False,
    )
    _build_synthetic_correlations().to_csv(
        readout_dir / "metric_correlations.csv", index=False
    )
    (readout_dir / "statistics_summary.json").write_text(
        json.dumps(_build_synthetic_statistics_summary()),
        encoding="utf-8",
    )
    return readout_dir


# ===========================================================================
# I/O helpers
# ===========================================================================


def test_read_csv_table_loads_synthetic(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = read_csv_table(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_csv_table_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv_table(tmp_path / "nope.csv")


def test_read_statistics_summary_loads_json(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert read_statistics_summary(p) == {"k": 1}


def test_ensure_output_dirs_creates_figures_and_gif_frames(tmp_path):
    dirs = ensure_output_dirs(tmp_path / "readout")
    assert dirs["figures_dir"].is_dir()
    assert dirs["gif_frames_dir"].is_dir()
    assert dirs["readout_dir"] == tmp_path / "readout"


# ===========================================================================
# Plot functions (small synthetic data; each writes a PNG)
# ===========================================================================


def test_plot_baseline_comparison_writes_png(tmp_path):
    flat = _build_synthetic_flat()
    out = tmp_path / "bc.png"
    p = plot_baseline_comparison(flat, out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_reference_forbidden_edge_suppression_writes_png(tmp_path):
    ref = _build_synthetic_reference_df()
    out = tmp_path / "ref.png"
    p = plot_reference_forbidden_edge_suppression(ref, out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_degradation_curve_writes_png(tmp_path):
    flat = _build_synthetic_flat()
    out = tmp_path / "deg.png"
    p = plot_degradation_curve(flat, metric="sid", output_path=out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_soft_frobenius_heatmap_writes_png(tmp_path):
    flat = _build_synthetic_flat()
    out = tmp_path / "hm.png"
    p = plot_soft_frobenius_heatmap(flat, metric="sid", output_path=out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_sid_mmd_scatter_writes_png(tmp_path):
    flat = _build_synthetic_flat()
    corr = _build_synthetic_correlations()
    out = tmp_path / "sc.png"
    p = plot_sid_mmd_scatter(flat, corr, out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_edge_count_engagement_diagnostic_writes_png(tmp_path):
    flat = _build_synthetic_flat()
    ref = _build_synthetic_reference_df()
    out = tmp_path / "edge.png"
    p = plot_edge_count_engagement_diagnostic(flat, ref, out)
    assert p.exists() and p.stat().st_size > 0


# ===========================================================================
# Readout summary
# ===========================================================================


def _make_dummy_figure_paths(readout_dir: Path) -> dict[str, Path]:
    figures_dir = readout_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in (
        "fig01_baseline_comparison_sid_shd_mmd",
        "fig02_reference_forbidden_edge_suppression",
    ):
        p = figures_dir / f"{name}.png"
        p.write_bytes(b"")  # placeholder
        paths[name] = p
    return paths


def test_presentation_table_replaces_nan_with_em_dash():
    df = pd.DataFrame({
        "a": [1.0, float("nan"), 3.0],
        "b": [None, "hello", 2],
    })
    out = presentation_table(df)
    # Original input is not mutated.
    assert df.iloc[1, 0] != df.iloc[1, 0]  # NaN != NaN
    # NaN/None become em dashes.
    em = "—"
    assert out.iloc[1, 0] == em
    assert out.iloc[0, 1] == em
    # Numeric formatting preserves the rest as strings.
    assert out.iloc[0, 0] == "1"
    assert out.iloc[2, 0] == "3"
    assert out.iloc[1, 1] == "hello"
    assert out.iloc[2, 1] == "2"


def test_presentation_table_handles_empty():
    df = pd.DataFrame()
    out = presentation_table(df)
    assert out.empty


def test_readout_summary_renders_missing_as_em_dash(tmp_path):
    """The markdown readout summary must not contain raw NaN tokens."""
    from experiments.main_study.render_readout_figures import (
        FLAT_RECORDS_CSV,
        REFERENCE_FORBIDDEN_EDGE_COMPARISON_CSV,
        METRIC_CORRELATIONS_CSV,
        STATISTICS_SUMMARY_JSON,
    )
    readout_dir = _seed_synthetic_inputs(tmp_path)
    figure_paths = _make_dummy_figure_paths(readout_dir)
    out_path = readout_dir / READOUT_SUMMARY_FILENAME
    flat = _build_synthetic_flat()
    ref = _build_synthetic_reference_df()
    corr = _build_synthetic_correlations()
    stats = _build_synthetic_statistics_summary()
    write_readout_summary(
        main_evaluation_run_hash12=_RUN_HASH12,
        readout_dir=readout_dir,
        flat=flat,
        reference_df=ref,
        correlations=corr,
        statistics_summary=stats,
        figure_paths=figure_paths,
        extra_outputs={},
        output_path=out_path,
    )
    text = out_path.read_text(encoding="utf-8")
    # The table cells must not display the raw NaN token.
    assert " nan " not in text.lower()
    assert "NaN" not in text


def test_write_readout_summary_labelling_only(tmp_path):
    readout_dir = _seed_synthetic_inputs(tmp_path)
    figure_paths = _make_dummy_figure_paths(readout_dir)
    out_path = readout_dir / READOUT_SUMMARY_FILENAME
    flat = _build_synthetic_flat()
    ref = _build_synthetic_reference_df()
    corr = _build_synthetic_correlations()
    stats = _build_synthetic_statistics_summary()
    write_readout_summary(
        main_evaluation_run_hash12=_RUN_HASH12,
        readout_dir=readout_dir,
        flat=flat,
        reference_df=ref,
        correlations=corr,
        statistics_summary=stats,
        figure_paths=figure_paths,
        extra_outputs={},
        output_path=out_path,
    )
    text = out_path.read_text(encoding="utf-8")
    # Required structural sections.
    assert "Run identity" in text
    assert _RUN_HASH12 in text
    assert "matched_l1_lambda1" in text
    assert "lambda_prior" in text
    assert "evaluation seeds" in text
    # Forbidden interpretive phrases.
    forbidden = [
        "this suggests", "however", "winner", "best method",
        "proven", "refuted",
        "h1 is supported", "h2 is supported", "h3 is supported",
        "therefore", "this indicates",
    ]
    lower = text.lower()
    for token in forbidden:
        assert token not in lower, (
            f"forbidden interpretive phrase {token!r} appears in "
            "readout_summary.md"
        )


# ===========================================================================
# Orchestrator
# ===========================================================================


def test_render_all_readout_outputs_writes_all_figures(tmp_path):
    _seed_synthetic_inputs(tmp_path)
    manifest = render_all_readout_outputs(
        tmp_path, _RUN_HASH12, make_gif=False
    )
    expected = {
        "fig01_baseline_comparison_sid_shd_mmd",
        "fig02_reference_forbidden_edge_suppression",
        "fig03_degradation_curves_sid",
        "fig04_degradation_curves_mmd",
        "fig05_soft_frobenius_sid_heatmap",
        "fig06_soft_frobenius_mmd_heatmap",
        "fig07_sid_vs_mmd_correlation",
        "fig08_edge_count_and_engagement_diagnostic",
        "readout_summary",
    }
    assert expected.issubset(set(manifest.keys()))
    for k, p in manifest.items():
        assert p.exists(), f"missing artefact: {k}"


def test_cli_returns_zero_on_synthetic(tmp_path):
    _seed_synthetic_inputs(tmp_path)
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 0


def test_cli_returns_one_on_missing_inputs(tmp_path):
    # No CSVs prepared.
    rc = cli_main([
        "--output-root", str(tmp_path),
        "--main-evaluation-run-hash12", _RUN_HASH12,
    ])
    assert rc == 1


# ===========================================================================
# Notebook (if generated) and static scope checks
# ===========================================================================


_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "this suggests", "this indicates", "therefore", "however",
    "winner", "best method", "proven", "refuted",
    "h1 is supported", "h2 is supported", "h3 is supported",
    "is better than", "is worse than",
)


def test_notebook_if_exists_has_no_interpretive_phrases():
    nb_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks" / "main_evaluation_readout.ipynb"
    )
    if not nb_path.exists():
        pytest.skip("notebook not present in this environment")
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    lowered = json.dumps(payload).lower()
    for token in _FORBIDDEN_PHRASES:
        assert token not in lowered, (
            f"forbidden interpretive phrase {token!r} in notebook"
        )


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


_RENDER_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "json",
    "math",
    "pathlib",
    "sys",
    "typing",
    "numpy",
    "pandas",
    "matplotlib",
    "PIL",
    "imageio",
})


_RENDER_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "seaborn",
    "plotly",
    "scipy",
    "statsmodels",
    "sklearn",
    "symbolic_priors_cd.metrics",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "experiments.selection_study",
    "experiments.main_study.backends",
    "experiments.main_study.executor",
    "experiments.main_study.runner",
    "experiments.main_study.run_main_evaluation",
    "experiments.main_study.calibrate_matched_l1",
    "dagma",
    "dcdi",
    "tests",
)


def test_render_module_imports_are_allowlisted():
    src = Path(rrf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _RENDER_ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _RENDER_ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"render_readout_figures.py import {mod!r} not in the "
            f"allowlist {sorted(_RENDER_ALLOWED_PREFIXES)}."
        )


def test_render_module_does_not_import_forbidden_packages():
    src = Path(rrf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _RENDER_FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"render_readout_figures.py must not import {mod!r}; "
                f"forbidden prefix {forbidden!r}."
            )


def test_render_source_has_no_verdict_or_ranking_phrases():
    src = Path(rrf.__file__).read_text(encoding="utf-8").lower()
    for token in _FORBIDDEN_PHRASES:
        assert token not in src, (
            f"render_readout_figures.py source must not contain "
            f"phrase {token!r}."
        )


def test_tests_write_only_under_tmp_path(tmp_path):
    """Sentinel: every test in this file uses tmp_path for its writes."""
    src = Path(__file__).read_text(encoding="utf-8")
    # Every plotting test name starts with test_plot_; every such test
    # must mention tmp_path as a fixture parameter.
    for line in src.splitlines():
        ls = line.strip()
        if ls.startswith("def test_plot_") and "(" in ls:
            assert "tmp_path" in ls, (
                f"test {ls!r} must accept tmp_path"
            )
