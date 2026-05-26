"""Tests for the base-model selection visual readout renderer.

All tests use tiny synthetic CSV / JSON inputs under ``tmp_path``.
Real selection-study artefacts are never touched.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.selection_study import (
    render_base_model_selection_readout as rbm,
)
from experiments.selection_study.render_base_model_selection_readout import (
    BASE_MODEL_DECISION_LABEL,
    CALIBRATION_RUN_HASH_PREFIX,
    FIG02_NAME,
    FIG02B_NAME,
    FIG03_NAME,
    FIG05_NAME,
    FIG06_NAME,
    FIG07_NAME,
    FIG_DIR_NAME,
    FIG_STATUS_NAME,
    HELDOUT_RUN_HASH_PREFIX,
    MANIFEST_JSON_NAME,
    NOTEBOOK_NAME,
    REMOVED_FIGURE_NAMES,
    SELECTED_CONFIG_TABLE_CSV,
    SELECTION_SUMMARY_TABLE_CSV,
    SUMMARY_MD_NAME,
    audit_available_inputs,
    build_selected_configurations_table,
    build_selection_summary_table,
    create_base_model_selection_notebook,
    ensure_output_dir,
    figures_output_dir,
    load_selection_inputs,
    main as cli_main,
    plot_dagma_ceiling_and_headroom,
    plot_dcdi_fit_rng_sensitivity,
    plot_heldout_metric_means,
    plot_heldout_sid_per_seed,
    plot_paired_model_differences,
    plot_runtime_log_scale,
    plot_status_reliability,
    read_csv_if_exists,
    read_json_if_exists,
    render_base_model_selection_readout,
    selection_readout_dir,
    write_figure_manifest,
    write_summary_markdown,
)


# ---------------------------------------------------------------------------
# Synthetic-input builders
# ---------------------------------------------------------------------------


def _synthetic_selected_configurations() -> dict[str, Any]:
    return {
        "artefact_type": "calibration_selected_configurations",
        "calibration_run_hash_prefix": CALIBRATION_RUN_HASH_PREFIX,
        "calibration_seeds": [201, 202],
        "candidate_ranking": {
            "centred_only": {
                "dagma": [{
                    "aggregate_metrics": {
                        "mean_sid": 0.0,
                        "mean_mmd_primary": 0.006,
                        "mean_shd": 0.0,
                    },
                    "configuration_hash_prefix": "06ee98d13852",
                    "hyperparameters": {"lambda1": 0.25},
                }],
                "dcdi": [{
                    "aggregate_metrics": {
                        "mean_sid": 60.0,
                        "mean_mmd_primary": 0.089,
                        "mean_shd": 30.5,
                    },
                    "configuration_hash_prefix": "dd39d6325e7d",
                    "hyperparameters": {},
                }],
            },
            "standardised": {
                "dagma": [{
                    "aggregate_metrics": {
                        "mean_sid": 46.0,
                        "mean_mmd_primary": 0.097,
                        "mean_shd": 18.0,
                    },
                    "configuration_hash_prefix": "7b345b1b2e85",
                    "hyperparameters": {"lambda1": 0.1},
                }],
                "dcdi": [{
                    "aggregate_metrics": {
                        "mean_sid": 46.0,
                        "mean_mmd_primary": 0.103,
                        "mean_shd": 25.0,
                    },
                    "configuration_hash_prefix": "16f92df3d6af",
                    "hyperparameters": {},
                }],
            },
        },
    }


def _synthetic_main_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"condition": "centred_only", "model": "dagma",
         "mean_sid": 4.2, "mean_mmd_primary": 0.006,
         "mean_shd": 1.0, "mean_runtime_seconds": 1.05},
        {"condition": "centred_only", "model": "dcdi",
         "mean_sid": 63.4, "mean_mmd_primary": 0.116,
         "mean_shd": 30.4, "mean_runtime_seconds": 976.65},
        {"condition": "standardised", "model": "dagma",
         "mean_sid": 66.6, "mean_mmd_primary": 0.121,
         "mean_shd": 25.8, "mean_runtime_seconds": 1.47},
        {"condition": "standardised", "model": "dcdi",
         "mean_sid": 68.8, "mean_mmd_primary": 0.142,
         "mean_shd": 29.6, "mean_runtime_seconds": 902.69},
    ])


def _synthetic_per_seed_main() -> pd.DataFrame:
    # centred_only / DAGMA has zero-SID seeds; standardised / DAGMA
    # is positive across seeds. These match the real data pattern.
    rows = []
    for sid in (13.0, 0.0, 8.0, 0.0, 0.0):
        rows.append({
            "condition": "centred_only", "model": "dagma",
            "seed_value": 301, "fit_rng": "",
            "sid": sid, "mmd_primary": 0.01, "shd": 1.0,
            "runtime_seconds": 1.0,
            "training_status": "converged",
            "graph_status": "valid_dag",
            "sampler_status": "available",
        })
    for sid in (78.0, 51.0, 71.0, 71.0, 46.0):
        rows.append({
            "condition": "centred_only", "model": "dcdi",
            "seed_value": 301, "fit_rng": 42,
            "sid": sid, "mmd_primary": 0.12, "shd": 30.0,
            "runtime_seconds": 900.0,
            "training_status": "converged",
            "graph_status": "valid_dag",
            "sampler_status": "available",
        })
    for sid in (64.0, 66.0, 69.0, 64.0, 71.0):
        rows.append({
            "condition": "standardised", "model": "dagma",
            "seed_value": 301, "fit_rng": "",
            "sid": sid, "mmd_primary": 0.10, "shd": 25.0,
            "runtime_seconds": 1.4,
            "training_status": "converged",
            "graph_status": "valid_dag",
            "sampler_status": "available",
        })
    for sid in (76.0, 80.0, 84.0, 59.0, 45.0):
        rows.append({
            "condition": "standardised", "model": "dcdi",
            "seed_value": 301, "fit_rng": 42,
            "sid": sid, "mmd_primary": 0.14, "shd": 29.0,
            "runtime_seconds": 900.0,
            "training_status": "converged",
            "graph_status": "valid_dag",
            "sampler_status": "available",
        })
    return pd.DataFrame(rows)


def _synthetic_sensitivity_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"condition": "centred_only", "model": "dcdi",
         "scm_seed": 301, "fit_rng": 43, "sid": 64.0,
         "mmd_primary": 0.18, "shd": 31.0,
         "runtime_seconds": 1000.0,
         "training_status": "converged",
         "graph_status": "valid_dag",
         "sampler_status": "available",
         "main_evaluation_sid_at_seed_301": 78.0,
         "main_evaluation_mmd_primary_at_seed_301": 0.127},
        {"condition": "centred_only", "model": "dcdi",
         "scm_seed": 301, "fit_rng": 44, "sid": 68.0,
         "mmd_primary": 0.13, "shd": 30.0,
         "runtime_seconds": 1013.0,
         "training_status": "converged",
         "graph_status": "valid_dag",
         "sampler_status": "available",
         "main_evaluation_sid_at_seed_301": 78.0,
         "main_evaluation_mmd_primary_at_seed_301": 0.127},
    ])


def _seed_full_inputs(tmp_path: Path) -> None:
    """Persist all synthetic inputs at their canonical locations."""
    cal_dir = (
        tmp_path / "results" / "model_selection"
        / "calibration" / CALIBRATION_RUN_HASH_PREFIX
    )
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "selected_configurations.json").write_text(
        json.dumps(_synthetic_selected_configurations()),
        encoding="utf-8",
    )
    rd = (
        tmp_path / "results" / "model_selection" / "held_out"
        / HELDOUT_RUN_HASH_PREFIX / "readout"
    )
    rd.mkdir(parents=True, exist_ok=True)
    _synthetic_main_summary().to_csv(rd / "main_summary.csv", index=False)
    _synthetic_per_seed_main().to_csv(
        rd / "per_seed_main.csv", index=False,
    )
    _synthetic_sensitivity_summary().to_csv(
        rd / "sensitivity_summary.csv", index=False,
    )
    pd.DataFrame([
        {"kind": "main", "status_field": "graph_status",
         "status_value": "valid_dag", "count": 20},
    ]).to_csv(rd / "status_summary.csv", index=False)


# ===========================================================================
# 1. audit_available_inputs reports schemas
# ===========================================================================


def test_audit_reports_existing_files(tmp_path):
    _seed_full_inputs(tmp_path)
    audit = audit_available_inputs(tmp_path)
    assert audit["selected_configurations_json"]["exists"] is True
    assert audit["main_summary_csv"]["exists"] is True
    # CSV entries report columns + n_rows.
    assert "columns" in audit["main_summary_csv"]
    assert "mean_sid" in audit["main_summary_csv"]["columns"]
    assert audit["main_summary_csv"]["n_rows"] == 4
    # JSON entries report top_level_keys.
    assert "top_level_keys" in audit["selected_configurations_json"]


def test_audit_missing_files(tmp_path):
    audit = audit_available_inputs(tmp_path)
    # No file exists yet.
    for name, entry in audit.items():
        assert entry["exists"] is False
        assert "columns" not in entry
        assert "top_level_keys" not in entry


# ===========================================================================
# 2. missing inputs lead to skipped figures with explicit reasons
# ===========================================================================


def test_missing_inputs_skip_figures_with_reasons(tmp_path):
    """No inputs present -> every data-dependent figure is skipped
    with an explicit reason."""
    result = render_base_model_selection_readout(tmp_path)
    skipped = result["skipped_figures"]
    for name in (FIG02_NAME, FIG02B_NAME, FIG03_NAME, FIG05_NAME,
                 FIG06_NAME, FIG07_NAME, FIG_STATUS_NAME):
        assert name in skipped
        assert isinstance(skipped[name], str) and skipped[name]


# ===========================================================================
# 3/4. loaders return the synthetic data
# ===========================================================================


def test_load_selection_inputs_loads_everything(tmp_path):
    _seed_full_inputs(tmp_path)
    inputs = load_selection_inputs(tmp_path)
    assert inputs.selected_configurations is not None
    assert isinstance(inputs.main_summary, pd.DataFrame)
    assert isinstance(inputs.per_seed_main, pd.DataFrame)
    assert isinstance(inputs.sensitivity_summary, pd.DataFrame)
    assert isinstance(inputs.status_summary, pd.DataFrame)


def test_read_csv_if_exists_returns_none_for_missing(tmp_path):
    assert read_csv_if_exists(tmp_path / "absent.csv") is None


def test_read_json_if_exists_returns_none_for_missing(tmp_path):
    assert read_json_if_exists(tmp_path / "absent.json") is None


# ===========================================================================
# 5. each plotting function writes a PNG when given minimal valid data
# ===========================================================================


def test_plot_heldout_metric_means_writes_png(tmp_path):
    out = tmp_path / "fig02.png"
    p = plot_heldout_metric_means(
        _synthetic_main_summary(),
        _synthetic_per_seed_main(),
        out,
    )
    assert p.exists() and p.stat().st_size > 0


def test_plot_paired_model_differences_writes_png(tmp_path):
    out = tmp_path / "fig02b.png"
    p = plot_paired_model_differences(
        _synthetic_per_seed_main(), out,
    )
    assert p.exists() and p.stat().st_size > 0


def test_plot_heldout_sid_per_seed_writes_png(tmp_path):
    out = tmp_path / "fig03.png"
    p = plot_heldout_sid_per_seed(_synthetic_per_seed_main(), out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_runtime_log_scale_writes_png(tmp_path):
    out = tmp_path / "fig05.png"
    p = plot_runtime_log_scale(_synthetic_main_summary(), out)
    assert p.exists() and p.stat().st_size > 0


def test_plot_dcdi_fit_rng_sensitivity_writes_png(tmp_path):
    out = tmp_path / "fig06.png"
    p = plot_dcdi_fit_rng_sensitivity(
        _synthetic_sensitivity_summary(), out,
    )
    assert p.exists() and p.stat().st_size > 0


def test_plot_dagma_ceiling_and_headroom_writes_png(tmp_path):
    out = tmp_path / "fig07.png"
    p = plot_dagma_ceiling_and_headroom(
        _synthetic_per_seed_main(), out,
    )
    assert p.exists() and p.stat().st_size > 0


def test_plot_status_reliability_writes_png(tmp_path):
    out = tmp_path / "fig_status.png"
    status_df = pd.DataFrame([
        {"kind": "main", "status_field": "graph_status",
         "status_value": "valid_dag", "count": 20},
        {"kind": "sensitivity", "status_field": "graph_status",
         "status_value": "valid_dag", "count": 5},
    ])
    p = plot_status_reliability(status_df, out)
    assert p.exists() and p.stat().st_size > 0


def test_build_selected_configurations_table_columns():
    df = build_selected_configurations_table(
        _synthetic_selected_configurations()
    )
    assert {"condition", "model", "configuration_hash_prefix",
            "hyperparameters", "calibration_mean_sid"}.issubset(
        set(df.columns)
    )
    # 2 conditions x 2 models = 4 rows.
    assert len(df) == 4


def test_build_selection_summary_table_distinguishes_roles():
    df = build_selection_summary_table(_synthetic_main_summary())
    assert {"condition", "metric", "role", "dagma_value",
            "dcdi_value", "lower_model"}.issubset(set(df.columns))
    # Every role label must appear among the rows.
    roles = set(df["role"].astype(str))
    assert {"primary", "tie-breaker", "diagnostic",
            "feasibility"}.issubset(roles)


# ===========================================================================
# 6. DCDI fit-RNG sensitivity figure is skipped when data missing
# ===========================================================================


def test_dcdi_sensitivity_figure_skipped_when_no_data(tmp_path):
    # Seed every input except the sensitivity CSV.
    cal_dir = (
        tmp_path / "results" / "model_selection"
        / "calibration" / CALIBRATION_RUN_HASH_PREFIX
    )
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "selected_configurations.json").write_text(
        json.dumps(_synthetic_selected_configurations()),
        encoding="utf-8",
    )
    rd = (
        tmp_path / "results" / "model_selection" / "held_out"
        / HELDOUT_RUN_HASH_PREFIX / "readout"
    )
    rd.mkdir(parents=True, exist_ok=True)
    _synthetic_main_summary().to_csv(rd / "main_summary.csv", index=False)
    _synthetic_per_seed_main().to_csv(
        rd / "per_seed_main.csv", index=False,
    )
    result = render_base_model_selection_readout(tmp_path)
    assert FIG06_NAME in result["skipped_figures"]
    assert "fit-RNG" in result["skipped_figures"][FIG06_NAME]


# ===========================================================================
# 7. selection summary table distinguishes criterion roles
# ===========================================================================


def test_selection_summary_table_role_tokens_in_source():
    """The selection summary builder labels every role explicitly."""
    src = Path(rbm.__file__).read_text(encoding="utf-8")
    for token in ("primary", "tie-breaker", "diagnostic", "feasibility"):
        assert token in src, (
            f"selection summary must label role {token!r}"
        )


# ===========================================================================
# 8. DAGMA ceiling/headroom uses ceiling label only when supported
# ===========================================================================


def test_dagma_ceiling_label_only_when_zero_sid_exists(tmp_path):
    """If centred_only/DAGMA has no zero-SID seed, do not use
    'ceiling evidence' label."""
    df = _synthetic_per_seed_main().copy()
    # Force every centred_only/DAGMA SID to be positive.
    mask = (
        (df["condition"] == "centred_only")
        & (df["model"] == "dagma")
    )
    df.loc[mask, "sid"] = 5.0
    out = tmp_path / "fig07.png"
    p = plot_dagma_ceiling_and_headroom(df, out)
    assert p.exists()


def test_dagma_ceiling_label_present_when_supported(tmp_path):
    df = _synthetic_per_seed_main()
    # The fixture already has zero-SID seeds in centred_only/DAGMA.
    out = tmp_path / "fig07.png"
    p = plot_dagma_ceiling_and_headroom(df, out)
    assert p.exists()


# ===========================================================================
# 9. figure manifest lists generated and skipped figures
# ===========================================================================


def test_manifest_lists_generated_and_skipped(tmp_path):
    _seed_full_inputs(tmp_path)
    result = render_base_model_selection_readout(tmp_path)
    manifest = json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )
    assert "generated_figures" in manifest
    assert "skipped_figures" in manifest
    assert manifest["selected_base_model"] == BASE_MODEL_DECISION_LABEL
    assert manifest["no_new_fits"] is True
    assert manifest["no_metric_recomputation"] is True


# ===========================================================================
# 10. summary markdown is labelling-only
# ===========================================================================


_FORBIDDEN_SUMMARY_PHRASES: tuple[str, ...] = (
    "best method",
    "universally superior",
    "proves",
    "refutes",
    "p-hacking",
    "rescue",
)


def test_summary_markdown_contains_run_hashes_no_overclaims(tmp_path):
    _seed_full_inputs(tmp_path)
    result = render_base_model_selection_readout(tmp_path)
    text = Path(result["summary_markdown_path"]).read_text(
        encoding="utf-8"
    )
    assert CALIBRATION_RUN_HASH_PREFIX in text
    assert HELDOUT_RUN_HASH_PREFIX in text
    assert BASE_MODEL_DECISION_LABEL in text
    lower = text.lower()
    for token in _FORBIDDEN_SUMMARY_PHRASES:
        assert token not in lower, (
            f"forbidden overclaim {token!r} in summary markdown"
        )


# ===========================================================================
# 11. notebook has required sections and no forbidden phrases
# ===========================================================================


_REQUIRED_NOTEBOOK_SECTIONS: tuple[str, ...] = (
    "Setup and run identity",
    "Data availability audit",
    "Calibration handoff",
    "Held-out evaluation status",
    "Held-out metric means",
    "Per-seed SID evidence",
    "Frozen selection rule",
    "Runtime and feasibility",
    "DCDI fit-RNG sensitivity addendum",
    "DAGMA ceiling/headroom",
    "Selection summary",
    "Output manifest",
)


_FORBIDDEN_NOTEBOOK_PHRASES: tuple[str, ...] = (
    "best method",
    "universally superior",
    "proves",
    "refutes",
    "p-hacking",
    "rescue",
    "this suggests",
    "this indicates",
    "therefore",
    "however",
)


def test_notebook_has_required_sections_and_no_forbidden_phrases(
    tmp_path,
):
    _seed_full_inputs(tmp_path)
    result = render_base_model_selection_readout(tmp_path)
    nb_path = Path(result["notebook_path"])
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source_text = json.dumps(payload).lower()
    for section in _REQUIRED_NOTEBOOK_SECTIONS:
        assert section.lower() in source_text, (
            f"notebook missing section {section!r}"
        )
    for phrase in _FORBIDDEN_NOTEBOOK_PHRASES:
        assert phrase not in source_text, (
            f"forbidden phrase {phrase!r} in notebook"
        )


# ===========================================================================
# 12. render_base_model_selection_readout writes all required outputs
# ===========================================================================


def test_render_writes_all_required_outputs(tmp_path):
    _seed_full_inputs(tmp_path)
    result = render_base_model_selection_readout(tmp_path)
    rd = selection_readout_dir(tmp_path)
    assert (rd / SUMMARY_MD_NAME).exists()
    assert (rd / MANIFEST_JSON_NAME).exists()
    assert (rd / FIG_DIR_NAME).is_dir()
    # New figure set after the patch (no fig01/04/08).
    for fig in (FIG02_NAME, FIG02B_NAME, FIG03_NAME,
                FIG05_NAME, FIG06_NAME, FIG07_NAME,
                FIG_STATUS_NAME):
        assert (rd / FIG_DIR_NAME / fig).exists(), f"missing {fig}"
    # Side tables backing the notebook DataFrame sections.
    assert (rd / SELECTED_CONFIG_TABLE_CSV).exists()
    assert (rd / SELECTION_SUMMARY_TABLE_CSV).exists()
    assert (tmp_path / "notebooks" / NOTEBOOK_NAME).exists()


def test_removed_figures_not_present(tmp_path):
    """fig01 / fig04 / fig08 are no longer generated."""
    _seed_full_inputs(tmp_path)
    render_base_model_selection_readout(tmp_path)
    fig_dir = figures_output_dir(tmp_path)
    for name in REMOVED_FIGURE_NAMES:
        assert not (fig_dir / name).exists(), (
            f"removed figure {name!r} should not be regenerated"
        )


def test_manifest_records_removed_figures(tmp_path):
    _seed_full_inputs(tmp_path)
    result = render_base_model_selection_readout(tmp_path)
    manifest = json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )
    assert "removed_figures" in manifest
    assert set(manifest["removed_figures"]) == set(REMOVED_FIGURE_NAMES)


def test_cli_returns_zero_on_synthetic(tmp_path):
    _seed_full_inputs(tmp_path)
    rc = cli_main(["--output-root", str(tmp_path)])
    assert rc == 0


# ===========================================================================
# 13. static import check
# ===========================================================================


_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    "__future__",
    "argparse",
    "csv",
    "dataclasses",
    "json",
    "math",
    "pathlib",
    "sys",
    "typing",
    "numpy",
    "pandas",
    "matplotlib",
})


_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "seaborn",
    "plotly",
    "scipy",
    "statsmodels",
    "sklearn",
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.data",
    "symbolic_priors_cd.metrics",
    "experiments.main_study",
    "experiments.selection_study.held_out",
    "experiments.selection_study.calibration",
    "experiments.selection_study.run",
    "dagma",
    "dcdi",
    "gadjid",
    "tests",
)


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_module_imports_are_allowlisted():
    src = Path(rbm.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        ok = (
            mod in _ALLOWED_PREFIXES
            or any(
                mod.startswith(allowed + ".")
                for allowed in _ALLOWED_PREFIXES
            )
        )
        assert ok, (
            f"render_base_model_selection_readout.py import {mod!r} "
            f"not in allowlist {sorted(_ALLOWED_PREFIXES)}."
        )


def test_module_does_not_import_forbidden_packages():
    src = Path(rbm.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod in _module_imports(tree):
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"forbidden import: {mod!r}"
            )


# ===========================================================================
# 14. source hygiene check
# ===========================================================================


_MILESTONE_REGEX: re.Pattern[str] = re.compile(
    r"\bM[-_]?(?:[0-9]|10|11)[a-c]?\b"
)


_HYGIENE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bClaude\b"),
    re.compile(r"\bChatGPT\b"),
    re.compile(r"\bprompt(?:ed|ing|s)?\b"),
    re.compile(r"\bconversation\b"),
    re.compile(r"\buser\s+asked\b"),
    re.compile(r"\bsuggested\s+by\b"),
    re.compile(r"\bp[-_]?hacking\b"),
    re.compile(r"\brescue\b"),
)


def test_source_has_no_assistant_or_milestone_artefacts():
    src = Path(rbm.__file__).read_text(encoding="utf-8")
    match = _MILESTONE_REGEX.search(src)
    assert match is None, (
        f"render_base_model_selection_readout.py contains milestone "
        f"label {match.group(0)!r}"
    )
    for pattern in _HYGIENE_PATTERNS:
        m = pattern.search(src)
        assert m is None, (
            "render_base_model_selection_readout.py contains "
            f"hygiene-blocked token {m.group(0)!r}"
        )


# ===========================================================================
# 15. tests write only under tmp_path
# ===========================================================================


def test_tests_write_only_under_tmp_path(tmp_path):
    _seed_full_inputs(tmp_path)
    render_base_model_selection_readout(tmp_path)
    assert (tmp_path / "results").is_dir()
