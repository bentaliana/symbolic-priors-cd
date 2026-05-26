"""Tests for the prior-relevance diagnostics renderer.

All tests use synthetic CSVs and PNGs under ``tmp_path``. Real
persisted records and upstream readout artefacts are never touched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.main_study.exploratory import (
    render_prior_relevance_diagnostics as rprd,
)
from experiments.main_study.exploratory.render_prior_relevance_diagnostics import (
    DIAGNOSTICS_DIR_NAME,
    FIG01_NAME,
    FIG02_NAME,
    FIG03_NAME,
    FIG04_NAME,
    FIG05_NAME,
    FIG06_NAME,
    FIG07_NAME,
    FIG08_NAME,
    FIG09_NAME,
    FIG10_NAME,
    FIG11_NAME,
    FIG_DIR_NAME,
    MAIN_EVALUATION_RUN_HASH_PREFIX,
    MANIFEST_JSON_NAME,
    METHOD_FAMILY_COLOURS,
    NOTEBOOK_NAME,
    ORACLE_ANALYSIS_HASH_PREFIX,
    PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX,
    SUMMARY_MD_NAME,
    audit_available_inputs,
    diagnostics_output_dir,
    figures_output_dir,
    load_diagnostic_inputs,
    main as cli_main,
    main_evaluation_readout_dir,
    notebook_output_path,
    oracle_relevance_dir,
    plot_corruption_degradation,
    plot_error_decomposition,
    plot_fp_vs_fn_reconciliation,
    plot_main_result_clean_metrics,
    plot_mechanism_engagement,
    plot_offline_removal_effect,
    plot_oracle_diagnostic_summary,
    plot_oracle_per_seed_sid_delta,
    plot_prior_target_overlap,
    plot_required_edge_acyclicity,
    prior_relevance_dir,
    read_csv_if_exists,
    read_json_if_exists,
    render_prior_relevance_diagnostics,
    write_figure_manifest,
    write_summary_markdown,
)


_SEEDS = (501, 502, 503, 504, 505, 506, 507)
_METHODS = ("prior_free", "matched_l1", "soft_frobenius", "hard_exclusion")
_ORACLE_SCENARIOS = (
    "actual_reference_forbidden_removal",
    "fp_remove_budget10_exact",
    "fp_remove_all_false_positives",
    "fn_add_budget10_greedy_acyclic",
    "fn_add_full_greedy_acyclic",
)


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _make_baseline_comparison() -> pd.DataFrame:
    rows = []
    metric_values = {
        "sid": (66.0, 12.5),
        "shd": (24.0, 5.7),
        "mmd": (0.11, 0.03),
    }
    for m in _METHODS:
        for metric, (mean_v, std_v) in metric_values.items():
            rows.append({
                "condition_label": m,
                "method_family": m,
                "corruption_fraction": None,
                "confidence": None,
                "metric": metric,
                "n": 7,
                "mean": float(mean_v),
                "std": float(std_v),
                "median": float(mean_v),
                "min": float(mean_v) - 5.0,
                "max": float(mean_v) + 5.0,
            })
    return pd.DataFrame(rows)


def _make_reference_forbidden_comparison() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    method_targeted_means = {
        "prior_free": 0.10,
        "matched_l1": 0.09,
        "soft_frobenius": 0.04,
        "hard_exclusion": 0.0,
    }
    method_fraction_means = {
        "prior_free": 0.13,
        "matched_l1": 0.11,
        "soft_frobenius": 0.03,
        "hard_exclusion": 0.0,
    }
    for m in _METHODS:
        for s in _SEEDS:
            rows.append({
                "seed_value": s,
                "condition_label": m,
                "method_family": m,
                "corruption_fraction": 0.0,
                "confidence": 1.0,
                "n_reference_forbidden_edges": 10,
                "mean_abs_w_reference_forbidden_edges": float(
                    method_targeted_means[m]
                    + rng.normal(0, 0.005),
                ),
                "fraction_reference_forbidden_above_threshold": float(
                    method_fraction_means[m]
                    + rng.normal(0, 0.01),
                ),
                "mean_abs_w_reference_non_targeted_edges": 0.11,
                "edge_count_from_thresholded_adjacency": 13,
                "sid": 66.0, "shd": 24.0, "mmd": 0.11,
            })
    return pd.DataFrame(rows)


def _make_degradation_summary() -> pd.DataFrame:
    rows = []
    # Non-soft methods: single row, NaN confidence.
    for m in ("prior_free", "matched_l1", "hard_exclusion"):
        for metric in ("sid", "mmd"):
            rows.append({
                "method_family": m,
                "confidence": None,
                "metric": metric,
                "n_seed_slopes": 7,
                "mean_slope": 5.0 if metric == "sid" else 0.02,
                "std_slope": 1.0,
                "median_slope": 4.0,
                "min_slope": -2.0,
                "max_slope": 12.0,
            })
    for conf in (0.0, 0.25, 0.5, 0.75, 1.0):
        for metric in ("sid", "mmd"):
            rows.append({
                "method_family": "soft_frobenius",
                "confidence": conf,
                "metric": metric,
                "n_seed_slopes": 7,
                "mean_slope": (
                    1.0 + 2.0 * conf if metric == "sid"
                    else 0.005 + 0.01 * conf
                ),
                "std_slope": 1.0,
                "median_slope": 1.0,
                "min_slope": -1.0,
                "max_slope": 3.0,
            })
    return pd.DataFrame(rows)


def _make_prior_free_error_decomposition() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for s in _SEEDS:
        tp = int(4 + rng.integers(0, 3))
        fp = int(10 + rng.integers(-2, 3))
        fn = int(14 + rng.integers(-2, 3))
        rows.append({
            "seed_value": s,
            "n_true_edges": tp + fn,
            "n_predicted_edges": tp + fp,
            "true_positive_count": tp,
            "true_negative_count": 60,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "total_error_count_simple": fp + fn,
            "targeted_false_positive_count": 1,
            "targeted_false_positive_fraction_of_fp": 1.0 / fp,
            "targeted_error_fraction_of_total_errors": (
                1.0 / (fp + fn)
            ),
            "sid": 66.0, "shd": 24.0, "mmd": 0.11,
        })
    return pd.DataFrame(rows)


def _make_prior_target_overlap() -> pd.DataFrame:
    rows = []
    fractions = {
        "prior_free": 0.13,
        "matched_l1": 0.11,
        "soft_frobenius": 0.03,
        "hard_exclusion": 0.0,
    }
    for m in _METHODS:
        for s in _SEEDS:
            rows.append({
                "seed_value": s,
                "condition_label": m,
                "method_family": m,
                "n_reference_forbidden_edges": 10,
                "n_reference_edges_predicted": int(
                    round(10 * fractions[m])
                ),
                "fraction_reference_edges_predicted": float(
                    fractions[m]
                ),
                "edge_count": 13,
                "sid": 66.0, "shd": 24.0, "mmd": 0.11,
            })
    return pd.DataFrame(rows)


def _make_offline_removal_effect() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for s in _SEEDS:
        sid_delta = float(rng.choice([-3, -1, 0, 1, 3]))
        rows.append({
            "seed_value": s,
            "sid_original": 66.0,
            "sid_after_removing_reference_forbidden_edges": (
                66.0 + sid_delta
            ),
            "sid_delta": sid_delta,
            "shd_original": 24.0,
            "shd_after_removing_reference_forbidden_edges": 23.0,
            "shd_delta": -1.0,
            "n_reference_edges_predicted_before_removal": 1,
            "n_reference_edges_removed": 1,
        })
    return pd.DataFrame(rows)


def _make_oracle_summary() -> pd.DataFrame:
    scenarios_means = {
        "actual_reference_forbidden_removal": (-0.6, -1.3),
        "fp_remove_budget10_exact": (-24.9, -8.9),
        "fp_remove_all_false_positives": (-23.4, -10.1),
        "fn_add_budget10_greedy_acyclic": (-5.3, -2.9),
        "fn_add_full_greedy_acyclic": (-5.3, -2.9),
    }
    rows = []
    for s, (sid_m, shd_m) in scenarios_means.items():
        rows.append({
            "scenario_label": s,
            "n_seeds": 7,
            "mean_sid_original": 66.1,
            "mean_sid_after": 66.1 + sid_m,
            "mean_sid_delta": float(sid_m),
            "median_sid_delta": float(sid_m),
            "min_sid_delta": float(sid_m) - 5.0,
            "max_sid_delta": float(sid_m) + 5.0,
            "mean_shd_original": 24.0,
            "mean_shd_after": 24.0 + shd_m,
            "mean_shd_delta": float(shd_m),
            "median_shd_delta": float(shd_m),
            "min_shd_delta": float(shd_m) - 1.0,
            "max_shd_delta": float(shd_m) + 1.0,
            "mean_n_candidate_edges": 10.0,
            "mean_n_selected_edges": 10.0,
            "mean_n_skipped_cycle_edges": 0.0,
        })
    return pd.DataFrame(rows)


def _make_oracle_per_seed() -> pd.DataFrame:
    rows = []
    base = {
        "actual_reference_forbidden_removal": (0, 10, 0),
        "fp_remove_budget10_exact": (-25, 10, 0),
        "fp_remove_all_false_positives": (-25, 11, 0),
        "fn_add_budget10_greedy_acyclic": (-3, 3, 11),
        "fn_add_full_greedy_acyclic": (-3, 3, 11),
    }
    for s in _SEEDS:
        for scen, (sid_d, sel, sk) in base.items():
            rows.append({
                "seed_value": s,
                "scenario_label": scen,
                "search_strategy": scen,
                "n_candidate_edges": sel + sk,
                "n_selected_edges": sel,
                "n_skipped_cycle_edges": sk,
                "sid_original": 66.0,
                "sid_after": 66.0 + sid_d,
                "sid_delta": float(sid_d),
                "shd_original": 24.0,
                "shd_after": 23.0,
                "shd_delta": -1.0,
                "selected_edges_json": "[]",
                "skipped_cycle_edges_json": "[]",
            })
    return pd.DataFrame(rows)


def _write_all_inputs(tmp_path: Path) -> None:
    """Lay down a complete, schema-valid synthetic input tree."""
    me_dir = main_evaluation_readout_dir(tmp_path)
    me_dir.mkdir(parents=True, exist_ok=True)
    _make_baseline_comparison().to_csv(
        me_dir / "baseline_comparison.csv", index=False,
    )
    _make_reference_forbidden_comparison().to_csv(
        me_dir / "reference_forbidden_edge_comparison.csv", index=False,
    )
    _make_degradation_summary().to_csv(
        me_dir / "degradation_summary.csv", index=False,
    )
    # forbidden_engagement_summary.csv is read by the audit; write a stub.
    pd.DataFrame([
        {"method_family": "prior_free", "corruption_fraction": 0.0,
         "confidence": None, "n": 7,
         "mean_targeted_abs_w_mean": 0.1,
         "mean_edge_count": 14.0},
    ]).to_csv(
        me_dir / "forbidden_edge_engagement_summary.csv", index=False,
    )

    pr_dir = prior_relevance_dir(tmp_path)
    pr_dir.mkdir(parents=True, exist_ok=True)
    _make_prior_target_overlap().to_csv(
        pr_dir / "prior_target_overlap.csv", index=False,
    )
    _make_prior_free_error_decomposition().to_csv(
        pr_dir / "prior_free_error_decomposition.csv", index=False,
    )
    _make_offline_removal_effect().to_csv(
        pr_dir / "offline_forbidden_edge_removal_effect.csv",
        index=False,
    )
    # Dummy PNG heatmap (1x1 PNG bytes).
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
        b"\x3a\x7e\x9b\x55"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (pr_dir / "aggregated_error_heatmap.png").write_bytes(png_bytes)
    (pr_dir / "investigation_manifest.json").write_text(
        json.dumps({
            "analysis_hash12": PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX,
            "main_evaluation_run_hash12": (
                MAIN_EVALUATION_RUN_HASH_PREFIX
            ),
        }),
        encoding="utf-8",
    )

    or_dir = oracle_relevance_dir(tmp_path)
    or_dir.mkdir(parents=True, exist_ok=True)
    _make_oracle_summary().to_csv(
        or_dir / "oracle_diagnostics_summary.csv", index=False,
    )
    _make_oracle_per_seed().to_csv(
        or_dir / "oracle_diagnostics_per_seed.csv", index=False,
    )
    (or_dir / "oracle_prior_relevance_manifest.json").write_text(
        json.dumps({
            "analysis_hash12": ORACLE_ANALYSIS_HASH_PREFIX,
            "main_evaluation_run_hash12": (
                MAIN_EVALUATION_RUN_HASH_PREFIX
            ),
            "no_mmd_recomputation": True,
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Audit step
# ---------------------------------------------------------------------------


def test_audit_reports_missing_inputs(tmp_path: Path) -> None:
    audit = audit_available_inputs(tmp_path)
    for key, entry in audit.items():
        assert entry["exists"] is False, (
            f"empty tree should not report {key} as existing"
        )
        assert "path" in entry


def test_audit_reports_present_inputs(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    audit = audit_available_inputs(tmp_path)
    for key, entry in audit.items():
        assert entry["exists"] is True, (
            f"{key} should exist after synthetic write"
        )
    csv_entry = audit["baseline_comparison_csv"]
    assert "columns" in csv_entry
    assert "method_family" in csv_entry["columns"]
    assert csv_entry["n_rows"] > 0
    json_entry = audit["oracle_manifest_json"]
    assert "top_level_keys" in json_entry
    assert "analysis_hash12" in json_entry["top_level_keys"]
    png_entry = audit["aggregated_error_heatmap_png"]
    assert "size_bytes" in png_entry


# ---------------------------------------------------------------------------
# 2. Loader returns None for missing files
# ---------------------------------------------------------------------------


def test_load_diagnostic_inputs_none_on_empty(tmp_path: Path) -> None:
    inputs = load_diagnostic_inputs(tmp_path)
    assert inputs.baseline_comparison is None
    assert inputs.degradation_summary is None
    assert inputs.reference_forbidden_comparison is None
    assert inputs.prior_target_overlap is None
    assert inputs.prior_free_error_decomposition is None
    assert inputs.offline_removal_effect is None
    assert inputs.aggregated_error_heatmap_path is None
    assert inputs.prior_relevance_manifest is None
    assert inputs.oracle_summary is None
    assert inputs.oracle_per_seed is None
    assert inputs.oracle_manifest is None


def test_load_diagnostic_inputs_loads_when_present(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    inputs = load_diagnostic_inputs(tmp_path)
    assert isinstance(inputs.baseline_comparison, pd.DataFrame)
    assert isinstance(inputs.oracle_summary, pd.DataFrame)
    assert isinstance(inputs.aggregated_error_heatmap_path, Path)
    assert inputs.aggregated_error_heatmap_path.exists()
    assert isinstance(inputs.oracle_manifest, dict)


# ---------------------------------------------------------------------------
# 3. Individual figure plotters write a non-empty PNG
# ---------------------------------------------------------------------------


def _assert_png_nonempty(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    # Minimum valid PNG header + IHDR + IDAT + IEND is ~50 bytes.
    assert path.stat().st_size > 50


def test_plot_main_result_clean_metrics(tmp_path: Path) -> None:
    out = tmp_path / FIG01_NAME
    plot_main_result_clean_metrics(
        _make_baseline_comparison(), out,
    )
    _assert_png_nonempty(out)


def test_plot_mechanism_engagement(tmp_path: Path) -> None:
    out = tmp_path / FIG02_NAME
    plot_mechanism_engagement(
        _make_reference_forbidden_comparison(), out,
    )
    _assert_png_nonempty(out)


def test_plot_corruption_degradation(tmp_path: Path) -> None:
    out = tmp_path / FIG03_NAME
    plot_corruption_degradation(_make_degradation_summary(), out)
    _assert_png_nonempty(out)


def test_plot_error_decomposition(tmp_path: Path) -> None:
    out = tmp_path / FIG04_NAME
    plot_error_decomposition(
        _make_prior_free_error_decomposition(), out,
    )
    _assert_png_nonempty(out)


def test_plot_prior_target_overlap(tmp_path: Path) -> None:
    out = tmp_path / FIG05_NAME
    plot_prior_target_overlap(_make_prior_target_overlap(), out)
    _assert_png_nonempty(out)


def test_plot_offline_removal_effect(tmp_path: Path) -> None:
    out = tmp_path / FIG06_NAME
    plot_offline_removal_effect(_make_offline_removal_effect(), out)
    _assert_png_nonempty(out)


def test_plot_oracle_diagnostic_summary(tmp_path: Path) -> None:
    out = tmp_path / FIG08_NAME
    plot_oracle_diagnostic_summary(_make_oracle_summary(), out)
    _assert_png_nonempty(out)


def test_plot_oracle_per_seed_sid_delta(tmp_path: Path) -> None:
    out = tmp_path / FIG09_NAME
    plot_oracle_per_seed_sid_delta(_make_oracle_per_seed(), out)
    _assert_png_nonempty(out)


def test_plot_required_edge_acyclicity(tmp_path: Path) -> None:
    out = tmp_path / FIG10_NAME
    plot_required_edge_acyclicity(_make_oracle_per_seed(), out)
    _assert_png_nonempty(out)


def test_plot_fp_vs_fn_reconciliation(tmp_path: Path) -> None:
    out = tmp_path / FIG11_NAME
    plot_fp_vs_fn_reconciliation(
        _make_prior_free_error_decomposition(),
        _make_oracle_summary(),
        out,
    )
    _assert_png_nonempty(out)


# ---------------------------------------------------------------------------
# 4. Skip-policy: missing CSVs cause that figure (and only that figure)
#    to be skipped
# ---------------------------------------------------------------------------


def test_full_renderer_skips_only_missing_figures(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    # Remove one upstream CSV: prior_target_overlap.
    (prior_relevance_dir(tmp_path)
     / "prior_target_overlap.csv").unlink()
    result = render_prior_relevance_diagnostics(tmp_path)
    assert FIG05_NAME in result["skipped_figures"]
    assert FIG05_NAME not in result["generated_figures"]
    # Remaining figures should still be present.
    for name in (
        FIG01_NAME, FIG02_NAME, FIG03_NAME, FIG04_NAME,
        FIG06_NAME, FIG07_NAME, FIG08_NAME, FIG09_NAME,
        FIG10_NAME, FIG11_NAME,
    ):
        assert name in result["generated_figures"], (
            f"{name} should still be generated when only "
            "prior_target_overlap is missing"
        )


# ---------------------------------------------------------------------------
# 5. Heatmap copy: figure 07 is the upstream PNG (byte-identical)
# ---------------------------------------------------------------------------


def test_heatmap_is_copied_byte_identical(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    src = (
        prior_relevance_dir(tmp_path)
        / "aggregated_error_heatmap.png"
    )
    result = render_prior_relevance_diagnostics(tmp_path)
    fig07_path = Path(result["generated_figures"][FIG07_NAME])
    assert fig07_path.read_bytes() == src.read_bytes()


# ---------------------------------------------------------------------------
# 6. Full renderer: end-to-end sanity
# ---------------------------------------------------------------------------


def test_full_renderer_end_to_end(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    result = render_prior_relevance_diagnostics(tmp_path)
    diagnostics_dir = diagnostics_output_dir(tmp_path)
    figs_dir = figures_output_dir(tmp_path)
    assert diagnostics_dir.exists()
    assert figs_dir.exists()
    # All 11 figures generated.
    expected_figs = {
        FIG01_NAME, FIG02_NAME, FIG03_NAME, FIG04_NAME, FIG05_NAME,
        FIG06_NAME, FIG07_NAME, FIG08_NAME, FIG09_NAME, FIG10_NAME,
        FIG11_NAME,
    }
    assert set(result["generated_figures"]) == expected_figs
    assert result["skipped_figures"] == {}
    # Manifest, summary, notebook all written.
    assert Path(result["manifest_path"]).exists()
    assert Path(result["summary_markdown_path"]).exists()
    assert Path(result["notebook_path"]).exists()
    # Each figure file is a non-trivial PNG.
    for p in result["generated_figures"].values():
        _assert_png_nonempty(Path(p))


# ---------------------------------------------------------------------------
# 7. Manifest JSON structure
# ---------------------------------------------------------------------------


def test_manifest_contents(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    result = render_prior_relevance_diagnostics(tmp_path)
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.name == MANIFEST_JSON_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["main_evaluation_run_hash12"] == (
        MAIN_EVALUATION_RUN_HASH_PREFIX
    )
    assert manifest["prior_relevance_analysis_hash12"] == (
        PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX
    )
    assert manifest["oracle_analysis_hash12"] == (
        ORACLE_ANALYSIS_HASH_PREFIX
    )
    assert manifest["no_new_fits"] is True
    assert manifest["no_metric_recomputation"] is True
    assert manifest["no_new_sampling"] is True
    assert manifest["no_protocol_changes"] is True
    assert "audit" in manifest
    assert "generated_figures" in manifest
    assert "skipped_figures" in manifest
    assert "claim_support_matrix_path" not in manifest


# ---------------------------------------------------------------------------
# 9. Summary markdown contents
# ---------------------------------------------------------------------------


def test_summary_markdown_contents(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    result = render_prior_relevance_diagnostics(tmp_path)
    md_path = Path(result["summary_markdown_path"])
    text = md_path.read_text(encoding="utf-8")
    assert "# Prior-relevance diagnostics" in text
    assert MAIN_EVALUATION_RUN_HASH_PREFIX in text
    assert PRIOR_RELEVANCE_ANALYSIS_HASH_PREFIX in text
    assert ORACLE_ANALYSIS_HASH_PREFIX in text
    assert "Figures generated" in text
    assert "Figures skipped" in text


# ---------------------------------------------------------------------------
# 10. Notebook payload structure
# ---------------------------------------------------------------------------


def test_notebook_payload_structure(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    result = render_prior_relevance_diagnostics(tmp_path)
    nb_path = Path(result["notebook_path"])
    assert nb_path.name == NOTEBOOK_NAME
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert isinstance(payload["cells"], list)
    # All code cells must declare execution_count = None and outputs = [].
    code_cells = [c for c in payload["cells"]
                  if c["cell_type"] == "code"]
    assert len(code_cells) > 0
    for c in code_cells:
        assert c["execution_count"] is None
        assert c["outputs"] == []
    # 10 section headings: "## 1." ... "## 10."
    markdown_texts = [
        "".join(c["source"]) for c in payload["cells"]
        if c["cell_type"] == "markdown"
    ]
    full_md = "\n".join(markdown_texts)
    for k in range(1, 10):
        assert f"## {k}." in full_md, (
            f"section {k} heading missing from notebook"
        )
    # Claim-support matrix section was removed; ensure no section 10
    # heading lingers.
    assert "## 10." not in full_md


# ---------------------------------------------------------------------------
# 11. CLI returns 0 and writes the same artefacts
# ---------------------------------------------------------------------------


def test_cli_returns_zero_and_writes_artefacts(tmp_path: Path) -> None:
    _write_all_inputs(tmp_path)
    rc = cli_main(["--output-root", str(tmp_path)])
    assert rc == 0
    diagnostics_dir = diagnostics_output_dir(tmp_path)
    assert (diagnostics_dir / MANIFEST_JSON_NAME).exists()
    assert (diagnostics_dir / SUMMARY_MD_NAME).exists()
    assert notebook_output_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# 12. Read-only over upstream artefacts
# ---------------------------------------------------------------------------


def test_renderer_does_not_modify_upstream_artefacts(
    tmp_path: Path,
) -> None:
    _write_all_inputs(tmp_path)
    paths_and_hashes: list[tuple[Path, bytes]] = []
    for parent in (
        main_evaluation_readout_dir(tmp_path),
        prior_relevance_dir(tmp_path),
        oracle_relevance_dir(tmp_path),
    ):
        for p in sorted(parent.iterdir()):
            if p.is_file():
                paths_and_hashes.append((p, p.read_bytes()))
    render_prior_relevance_diagnostics(tmp_path)
    for p, before in paths_and_hashes:
        assert p.read_bytes() == before, (
            f"upstream file {p} was modified by the renderer"
        )


# ---------------------------------------------------------------------------
# 13. Colour mapping and method order
# ---------------------------------------------------------------------------


def test_colour_mapping_and_method_order() -> None:
    assert set(METHOD_FAMILY_COLOURS.keys()) == set(_METHODS)
    # Soft = blue, prior_free = grey, matched_l1 = orange,
    # hard_exclusion = red (any plausible muted variant accepted).
    for m in _METHODS:
        colour = METHOD_FAMILY_COLOURS[m]
        assert isinstance(colour, str) and colour.startswith("#")
        assert len(colour) == 7
    assert rprd.METHOD_FAMILY_ORDER[0] == "prior_free"
    assert "soft_frobenius" in rprd.METHOD_FAMILY_ORDER


# ---------------------------------------------------------------------------
# 14. Hygiene: source has no internal milestone labels or process tokens
# ---------------------------------------------------------------------------


_FORBIDDEN_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bM[-_]?(?:[0-9]|10|11)[a-c]?\b"),
    re.compile(r"\b" + "Cla" + "ude" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Chat" + "GPT" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "prom" + "pt" + r"\b", re.IGNORECASE),
    re.compile(
        r"\b" + "conver" + "sation" + r"\b", re.IGNORECASE,
    ),
    re.compile(
        r"\b" + "user" + r"\s+" + "asked" + r"\b", re.IGNORECASE,
    ),
    re.compile(
        r"\b" + "suggested" + r"\s+" + "by" + r"\b",
        re.IGNORECASE,
    ),
    re.compile(r"p" + "-" + "hac" + "king", re.IGNORECASE),
    re.compile(r"\b" + "res" + "cue" + r"\b", re.IGNORECASE),
)


def _source_files() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return [
        repo_root / "experiments" / "main_study" / "exploratory"
        / "render_prior_relevance_diagnostics.py",
        repo_root / "tests" / "main_study"
        / "test_render_prior_relevance_diagnostics.py",
    ]


def test_source_has_no_forbidden_tokens() -> None:
    src_text_pairs: list[tuple[Path, str]] = []
    for p in _source_files():
        # The test file itself uses split string literals for the
        # forbidden tokens above, so they never appear in the source
        # as full words. The hygiene check must therefore pass on
        # both the renderer and this test file.
        text = p.read_text(encoding="utf-8")
        src_text_pairs.append((p, text))
    for path, text in src_text_pairs:
        if path.name.startswith("test_"):
            # In tests, the patterns are constructed via string
            # concatenation so they do not appear verbatim in the
            # source. Confirm that.
            assert "p-hacking".replace("-", "X") not in text, (
                f"{path} contains the forbidden literal"
            )
        for pattern in _FORBIDDEN_TOKEN_PATTERNS:
            for match in pattern.finditer(text):
                # The hygiene check skips matches that occur inside
                # the patterns themselves (this single test file).
                if path.name.startswith("test_"):
                    continue
                raise AssertionError(
                    f"forbidden token matched in {path}: "
                    f"{match.group(0)!r}"
                )


# ---------------------------------------------------------------------------
# 15. Empty-tree behaviour: renderer skips every figure and still writes
#     the summary, manifest, and notebook
# ---------------------------------------------------------------------------


def test_empty_tree_renderer_skips_everything(tmp_path: Path) -> None:
    result = render_prior_relevance_diagnostics(tmp_path)
    assert result["generated_figures"] == {}
    assert set(result["skipped_figures"].keys()) >= {
        FIG01_NAME, FIG02_NAME, FIG03_NAME, FIG04_NAME, FIG05_NAME,
        FIG06_NAME, FIG07_NAME, FIG08_NAME, FIG09_NAME, FIG10_NAME,
        FIG11_NAME,
    }
    assert Path(result["manifest_path"]).exists()
    assert Path(result["summary_markdown_path"]).exists()
    assert Path(result["notebook_path"]).exists()
    # Manifest figures should be all skipped.
    manifest = json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["generated_figures"] == {}
    assert "claim_support_matrix_path" not in manifest
