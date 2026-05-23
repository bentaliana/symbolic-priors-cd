"""Tests for the calibration readout module.

These tests exercise the public entry point
``generate_calibration_readout`` against synthetic calibration trees
constructed under ``tmp_path``. They never touch the live calibration
artefacts or records, never invoke any model fit, and never modify
inputs after the readout is generated.

Coverage spans:

- happy-path readout from a valid 40-record synthetic tree;
- record-count validation (exactly 40 records required);
- rejection of held-out evaluation seed values in records;
- rejection of missing records directory and missing artefact;
- non-rejection of degenerate selections, with visible reporting in
  the markdown, the CSV summary, and the returned report dict;
- presence of every required CSV/PNG/MD output file;
- expected row counts in CSV summaries;
- non-empty PNG outputs;
- markdown section completeness (Scope / Incident / Standard-deviation
  / Reproducibility);
- forbidden final-winner language not appearing in any text output;
- JSON-serialisability of the returned report dict;
- input file immutability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from experiments.selection_study.calibration_readout import (
    EXPECTED_RECORD_COUNT,
    MARKDOWN_FILENAME,
    MMD_PNG_FILENAME,
    RANKING_CSV_FILENAME,
    READOUT_DIRECTORY_NAME,
    RECORDS_DIRECTORY_NAME,
    SELECTED_CSV_FILENAME,
    SHD_PNG_FILENAME,
    SID_PNG_FILENAME,
    STATUS_CSV_FILENAME,
    _metric_yerr,
    _nonnegative_lower_error,
    generate_calibration_readout,
)
from experiments.selection_study.calibration_ranking import (
    rank_calibration_records,
)
from experiments.selection_study.selection_artefact import (
    CALIBRATION_SEEDS,
    CONDITIONS,
    FIT_RNG_POLICY_REF,
    INTERVENTION_POLICY_REF,
    MODELS,
    SELECTED_CONFIGURATIONS_FILENAME,
    SELECTION_RULE_ID,
    SELECTION_RULE_REF,
    write_selected_configurations,
)


# ---------------------------------------------------------------------------
# Synthetic-tree builders
# ---------------------------------------------------------------------------


_DAGMA_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25)
_DCDI_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)


def _grid_for(model: str) -> tuple[float, ...]:
    return _DAGMA_GRID if model == "dagma" else _DCDI_GRID


def _hyperparameter_name_for(model: str) -> str:
    return "lambda1" if model == "dagma" else "reg_coeff"


def _candidate_hash_for(
    model: str, condition: str, hyper_value: float
) -> str:
    seed_str = f"calibration-readout-test|{model}|{condition}|{hyper_value!r}"
    return hashlib.sha256(seed_str.encode("utf-8")).hexdigest()


def _synthetic_threshold_metrics(model: str) -> list[dict[str, Any]]:
    if model == "dagma":
        thresholds = (0.2, 0.3, 0.4)
    else:
        thresholds = (0.4, 0.5, 0.6)
    return [
        {
            "threshold": float(value),
            "shd": 0,
            "sid": 0,
            "mmd_primary": None,
        }
        for value in thresholds
    ]


def _synthetic_mmd_by_intervention() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in range(10):
        for value in (-2.0, 2.0):
            rows.append(
                {
                    "intervention_target": target,
                    "intervention_value": value,
                    "mmd_primary": 0.001,
                }
            )
    return rows


def _synthetic_bandwidth_summaries() -> dict[str, float]:
    summaries: dict[str, float] = {}
    for target in range(10):
        for label, sign in (("neg2", "neg2"), ("pos2", "pos2")):
            summaries[f"do_X{target}_{sign}"] = 50.0 + target
    return summaries


def _make_record(
    *,
    model: str,
    condition: str,
    hyper_value: float,
    seed_value: int,
    sid: Any = None,
    shd: Any = None,
    mmd_primary: Any = None,
    training_status: str = "converged",
    graph_status: str = "valid_dag",
    sampler_status: str = "available",
) -> dict[str, Any]:
    """Build a synthetic per-fit record dict compatible with the ranker schema."""
    config_hash_full = _candidate_hash_for(model, condition, hyper_value)
    if sid is None:
        sid = 0
    if shd is None:
        shd = 0
    if mmd_primary is None:
        mmd_primary = 0.001 + 0.0001 * hyper_value
    return {
        "model": model,
        "condition": condition,
        "configuration_hash_full": config_hash_full,
        "configuration_hash_prefix": config_hash_full[:12],
        "hyperparameters": {_hyperparameter_name_for(model): hyper_value},
        "seed_value": seed_value,
        "shd": shd,
        "sid": sid,
        "mmd_primary": mmd_primary,
        "graph_status": graph_status,
        "sampler_status": sampler_status,
        "training_status": training_status,
        "runtime_seconds": 0.5,
        "n_iterations": None,
        "threshold_metrics": _synthetic_threshold_metrics(model),
        "mmd_by_intervention": _synthetic_mmd_by_intervention(),
        "bandwidth_summaries": _synthetic_bandwidth_summaries(),
        "run_id": (
            f"{model}__{condition}__calibration__"
            f"seed{seed_value - 201}__cfg{config_hash_full}"
        ),
    }


def _make_all_records(
    *,
    overrides: dict[tuple[str, str, float, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the 40 synthetic per-fit records, applying optional overrides.

    Overrides map (model, condition, hyper_value, seed_value) -> dict
    of fields to set on that specific record.
    """
    overrides = overrides or {}
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for model in MODELS:
            for hyper_value in _grid_for(model):
                for seed_value in CALIBRATION_SEEDS:
                    base = _make_record(
                        model=model,
                        condition=condition,
                        hyper_value=hyper_value,
                        seed_value=seed_value,
                    )
                    extra = overrides.get(
                        (model, condition, hyper_value, seed_value)
                    )
                    if extra:
                        base.update(extra)
                    records.append(base)
    return records


def _build_artefact_from_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a valid selected_configurations artefact from records."""
    rank_output = rank_calibration_records(records)
    full_hash = hashlib.sha256(
        b"synthetic-calibration-readout-test-tree"
    ).hexdigest()
    return {
        "schema_version": 1,
        "artefact_type": "calibration_selected_configurations",
        "decision_scope": "within_model_configuration_selection",
        "base_model_decision_made": False,
        "selected_configuration_semantics": "rank_1_within_model_and_condition",
        "calibration_run_hash_prefix": full_hash[:12],
        "calibration_run_hash_full": full_hash,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule_ref": SELECTION_RULE_REF,
        "seed_population": "calibration",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "intervention_policy_ref": INTERVENTION_POLICY_REF,
        "fit_rng_policy_ref": FIT_RNG_POLICY_REF,
        "selections": rank_output["selections"],
        "candidate_ranking": rank_output["candidate_ranking"],
        "generated_at_utc": "2026-05-22T20:00:00Z",
    }


def _record_filename(record: dict[str, Any]) -> str:
    return (
        f"{record['model']}_{record['condition']}_"
        f"{record['configuration_hash_prefix']}_seed{record['seed_value']}.json"
    )


def _write_records_to_disk(
    records: list[dict[str, Any]], records_dir: Path
) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = records_dir / _record_filename(record)
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_synthetic_calibration_tree(
    root: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    artefact_override: dict[str, Any] | None = None,
) -> Path:
    """Write a synthetic calibration run under ``root`` and return the run dir."""
    if records is None:
        records = _make_all_records()
    artefact = _build_artefact_from_records(records)
    if artefact_override:
        artefact = deepcopy(artefact)
        artefact.update(artefact_override)

    hash12 = artefact["calibration_run_hash_prefix"]
    run_dir = root / hash12
    run_dir.mkdir(parents=True)
    _write_records_to_disk(records, run_dir / RECORDS_DIRECTORY_NAME)
    write_selected_configurations(
        artefact, run_dir / SELECTED_CONFIGURATIONS_FILENAME
    )
    return run_dir


def _snapshot_directory(directory: Path) -> dict[str, bytes]:
    """Return a name -> bytes snapshot of every regular file under ``directory``."""
    snapshot: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(directory))] = path.read_bytes()
    return snapshot


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Forbidden-language guard
# ---------------------------------------------------------------------------


_FORBIDDEN_LANGUAGE_PHRASES: tuple[str, ...] = (
    "winner",
    "model_winner",
    "base_model_winner",
    "recommended_model",
    "final_decision",
    "DAGMA wins",
    "DCDI wins",
)


def _assert_no_forbidden_language(text: str) -> None:
    lower = text.lower()
    for phrase in _FORBIDDEN_LANGUAGE_PHRASES:
        assert phrase.lower() not in lower, (
            f"forbidden final-winner phrase {phrase!r} appeared in "
            "the readout output text"
        )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_happy_path_loads_and_writes_all_files(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)

    report = generate_calibration_readout(run_dir)

    readout_dir = run_dir / READOUT_DIRECTORY_NAME
    assert readout_dir.is_dir()

    for filename in (
        MARKDOWN_FILENAME,
        SELECTED_CSV_FILENAME,
        RANKING_CSV_FILENAME,
        STATUS_CSV_FILENAME,
        SID_PNG_FILENAME,
        MMD_PNG_FILENAME,
        SHD_PNG_FILENAME,
    ):
        assert (readout_dir / filename).is_file(), filename

    assert report["n_records_loaded"] == EXPECTED_RECORD_COUNT
    assert report["selected_configurations_validates"] is True
    assert report["any_selected_degenerate"] is False
    assert set(report["selected_degeneracy_flags"].keys()) == set(CONDITIONS)
    for condition in CONDITIONS:
        assert (
            set(report["selected_degeneracy_flags"][condition].keys())
            == set(MODELS)
        )


def test_returned_report_is_json_safe(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)

    report = generate_calibration_readout(run_dir)

    serialised = json.dumps(report, sort_keys=True, ensure_ascii=True)
    round_trip = json.loads(serialised)
    assert round_trip["n_records_loaded"] == EXPECTED_RECORD_COUNT


def test_selected_csv_has_four_rows(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    rows = _read_csv_rows(
        run_dir / READOUT_DIRECTORY_NAME / SELECTED_CSV_FILENAME
    )
    assert len(rows) == 4

    seen = {(row["condition"], row["model"]) for row in rows}
    expected = {
        (condition, model)
        for condition in CONDITIONS
        for model in MODELS
    }
    assert seen == expected


def test_ranking_csv_has_twenty_rows(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    rows = _read_csv_rows(
        run_dir / READOUT_DIRECTORY_NAME / RANKING_CSV_FILENAME
    )
    assert len(rows) == 20

    by_cell: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (row["condition"], row["model"])
        by_cell.setdefault(key, []).append(int(row["rank"]))
    assert len(by_cell) == 4
    for cell, ranks in by_cell.items():
        assert sorted(ranks) == [1, 2, 3, 4, 5], cell

    selected_rows = [row for row in rows if row["selected"] == "true"]
    assert len(selected_rows) == 4
    for row in selected_rows:
        assert int(row["rank"]) == 1


def test_status_csv_counts_match_record_set(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    rows = _read_csv_rows(
        run_dir / READOUT_DIRECTORY_NAME / STATUS_CSV_FILENAME
    )
    total = sum(int(row["count"]) for row in rows)
    assert total == EXPECTED_RECORD_COUNT

    expected_keys = {
        (model, condition)
        for model in MODELS
        for condition in CONDITIONS
    }
    seen_keys = {(row["model"], row["condition"]) for row in rows}
    assert expected_keys.issubset(seen_keys)


def test_plot_files_are_non_empty(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    for filename in (
        SID_PNG_FILENAME,
        MMD_PNG_FILENAME,
        SHD_PNG_FILENAME,
    ):
        path = run_dir / READOUT_DIRECTORY_NAME / filename
        assert path.is_file()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Markdown content tests
# ---------------------------------------------------------------------------


def test_markdown_contains_required_sections(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    for header in (
        "# Calibration readout",
        "## Status",
        "## Scope",
        "## Selected configurations",
        "## Candidate ranking summary",
        "## Status/failure summary",
        "## Calibration observations",
        "## Incident note",
        "## Standard-deviation note",
        "## Generated files",
        "## Reproducibility note",
    ):
        assert header in md, header


def test_markdown_references_incident_report_and_seed_201(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "docs/08g_file_exists_error_incident.md" in md
    assert "seed 201" in md
    assert "dagma" in md
    assert "centred_only" in md


def test_markdown_standard_deviation_note_mentions_n_equals_two(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "n=2" in md
    assert "ddof=1" in md
    assert "5 seeds" in md


def test_markdown_reproducibility_note_names_generator_and_inputs(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert (
        "experiments/selection_study/calibration_readout.py" in md
    )
    assert "selected_configurations.json" in md
    assert "records/*.json" in md


def test_no_forbidden_phrases_in_markdown_or_csv(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    generate_calibration_readout(run_dir)

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    selected_csv = (
        run_dir / READOUT_DIRECTORY_NAME / SELECTED_CSV_FILENAME
    ).read_text(encoding="utf-8")
    ranking_csv = (
        run_dir / READOUT_DIRECTORY_NAME / RANKING_CSV_FILENAME
    ).read_text(encoding="utf-8")
    status_csv = (
        run_dir / READOUT_DIRECTORY_NAME / STATUS_CSV_FILENAME
    ).read_text(encoding="utf-8")

    for text in (md, selected_csv, ranking_csv, status_csv):
        _assert_no_forbidden_language(text)


def test_no_forbidden_phrases_in_report_dict(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    report = generate_calibration_readout(run_dir)

    # The serialisation excludes absolute filesystem paths because
    # pytest temp directories embed the test function name, which
    # would otherwise contaminate the substring check.
    report_for_check = {
        key: value
        for key, value in report.items()
        if key not in {
            "calibration_run_dir",
            "output_dir",
            "generated_files",
        }
    }
    serialised = json.dumps(
        report_for_check, sort_keys=True, ensure_ascii=True
    )
    _assert_no_forbidden_language(serialised)


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_missing_records_directory_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    records_dir = run_dir / RECORDS_DIRECTORY_NAME
    for record_path in records_dir.iterdir():
        record_path.unlink()
    records_dir.rmdir()

    with pytest.raises(FileNotFoundError, match="records"):
        generate_calibration_readout(run_dir)


def test_missing_selected_configurations_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    (run_dir / SELECTED_CONFIGURATIONS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="selected_configurations"):
        generate_calibration_readout(run_dir)


def test_missing_run_directory_is_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        generate_calibration_readout(bogus)


def test_wrong_record_count_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    records_dir = run_dir / RECORDS_DIRECTORY_NAME
    record_paths = sorted(records_dir.glob("*.json"))
    record_paths[0].unlink()

    with pytest.raises(ValueError, match="exactly 40"):
        generate_calibration_readout(run_dir)


def test_held_out_seed_in_records_is_rejected(tmp_path: Path) -> None:
    records = _make_all_records()
    contaminated = records[0]
    contaminated["seed_value"] = 301
    contaminated["run_id"] = contaminated["run_id"].replace(
        "seed0", "seed100"
    )

    run_dir = tmp_path / "with-held-out"
    run_dir.mkdir()
    _write_records_to_disk(records, run_dir / RECORDS_DIRECTORY_NAME)

    # Write an artefact that still passes structural validation; the
    # held-out-seed leak lives only in the records dir.
    clean_records = _make_all_records()
    artefact = _build_artefact_from_records(clean_records)
    write_selected_configurations(
        artefact, run_dir / SELECTED_CONFIGURATIONS_FILENAME
    )

    with pytest.raises(ValueError, match="held-out"):
        generate_calibration_readout(run_dir)


def test_unknown_model_in_record_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    records_dir = run_dir / RECORDS_DIRECTORY_NAME
    record_path = sorted(records_dir.glob("*.json"))[0]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["model"] = "not-a-real-model"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown model"):
        generate_calibration_readout(run_dir)


# ---------------------------------------------------------------------------
# Degeneracy reporting
# ---------------------------------------------------------------------------


def test_degenerate_selection_is_reported_not_rejected(
    tmp_path: Path,
) -> None:
    # Make every centred_only/dagma candidate have a non-finite SID on
    # seed 201 so the entire cell is in the non-finite-SID layer; the
    # rank-1 candidate then carries degeneracy_flag=True.
    overrides: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for hyper_value in _grid_for("dagma"):
        overrides[("dagma", "centred_only", hyper_value, 201)] = {
            "sid": float("nan"),
            "graph_status": "ground_truth_compat_failed",
            "training_status": "converged",
        }
    records = _make_all_records(overrides=overrides)

    run_dir = _write_synthetic_calibration_tree(
        tmp_path, records=records
    )
    report = generate_calibration_readout(run_dir)

    assert report["any_selected_degenerate"] is True
    assert (
        report["selected_degeneracy_flags"]["centred_only"]["dagma"]
        is True
    )

    md = (run_dir / READOUT_DIRECTORY_NAME / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "degeneracy_flag=true" in md

    selected_rows = _read_csv_rows(
        run_dir / READOUT_DIRECTORY_NAME / SELECTED_CSV_FILENAME
    )
    flagged = [
        row
        for row in selected_rows
        if row["condition"] == "centred_only"
        and row["model"] == "dagma"
    ]
    assert len(flagged) == 1
    assert flagged[0]["degeneracy_flag"] == "true"


# ---------------------------------------------------------------------------
# Input immutability and idempotence
# ---------------------------------------------------------------------------


def test_input_artefact_and_records_are_not_modified(
    tmp_path: Path,
) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)

    inputs_before: dict[str, bytes] = {
        SELECTED_CONFIGURATIONS_FILENAME: (
            run_dir / SELECTED_CONFIGURATIONS_FILENAME
        ).read_bytes(),
    }
    records_dir = run_dir / RECORDS_DIRECTORY_NAME
    for path in sorted(records_dir.glob("*.json")):
        inputs_before[
            f"{RECORDS_DIRECTORY_NAME}/{path.name}"
        ] = path.read_bytes()

    generate_calibration_readout(run_dir)

    assert (
        run_dir / SELECTED_CONFIGURATIONS_FILENAME
    ).read_bytes() == inputs_before[SELECTED_CONFIGURATIONS_FILENAME]
    for path in sorted(records_dir.glob("*.json")):
        key = f"{RECORDS_DIRECTORY_NAME}/{path.name}"
        assert path.read_bytes() == inputs_before[key]


def test_explicit_output_dir_is_respected(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)
    output_dir = tmp_path / "custom-output"

    report = generate_calibration_readout(run_dir, output_dir=output_dir)

    assert output_dir.is_dir()
    assert (output_dir / MARKDOWN_FILENAME).is_file()
    assert report["output_dir"] == str(output_dir.resolve())
    # The default readout directory under run_dir should NOT exist
    # when an explicit output_dir was supplied.
    assert not (run_dir / READOUT_DIRECTORY_NAME).exists()


# ---------------------------------------------------------------------------
# Error-bar clipping for nonnegative metrics
# ---------------------------------------------------------------------------


def test_nonnegative_lower_error_clips_at_mean() -> None:
    # When std exceeds mean, the lower magnitude is reduced to mean so
    # the whisker stops at zero rather than crossing it.
    assert _nonnegative_lower_error(2.0, 5.0) == 2.0


def test_nonnegative_lower_error_preserves_smaller_std() -> None:
    # When std is smaller than mean, the lower magnitude equals std
    # (the regular symmetric whisker still fits above zero).
    assert _nonnegative_lower_error(5.0, 2.0) == 2.0


def test_nonnegative_lower_error_handles_zero_mean() -> None:
    # A zero mean for a nonnegative metric collapses the lower
    # whisker entirely; the upper whisker is unaffected.
    assert _nonnegative_lower_error(0.0, 3.0) == 0.0


def test_nonnegative_lower_error_handles_non_finite_mean() -> None:
    assert _nonnegative_lower_error(float("nan"), 1.0) == 0.0
    assert _nonnegative_lower_error(float("inf"), 1.0) == 0.0


def test_metric_yerr_returns_asymmetric_for_sid_and_shd() -> None:
    means = [0.0, 2.0, 10.0, 0.5]
    stds = [3.0, 5.0, 1.0, 2.0]
    for metric_field in ("sid", "shd"):
        yerr = _metric_yerr(metric_field, means, stds)
        assert isinstance(yerr, list)
        assert len(yerr) == 2, metric_field
        lower, upper = yerr
        assert len(lower) == len(means)
        assert len(upper) == len(means)
        # No lower magnitude can take a data point below zero.
        for mean, lo in zip(means, lower):
            assert lo >= 0.0
            assert mean - lo >= -1e-12, (
                f"{metric_field}: mean={mean!r} - lower={lo!r} "
                "would render below zero"
            )
        # Upper magnitudes are unchanged from the stored std.
        assert upper == [float(s) for s in stds]
        # Specific clip behaviour for the small-mean / large-std rows.
        assert lower[0] == 0.0  # mean 0.0 -> lower clipped to 0.0
        assert lower[1] == 2.0  # mean 2.0 < std 5.0 -> clipped to 2.0
        assert lower[2] == 1.0  # mean 10 > std 1 -> std preserved
        assert lower[3] == 0.5  # mean 0.5 < std 2.0 -> clipped to 0.5


def test_metric_yerr_returns_symmetric_for_mmd() -> None:
    means = [0.001, 0.05, 0.1]
    stds = [0.002, 0.04, 0.06]
    yerr = _metric_yerr("mmd_primary", means, stds)
    # MMD must continue using a flat list of magnitudes (the original
    # symmetric error-bar behaviour) so the rendering of MMD plots is
    # not affected by the SID/SHD clip refinement.
    assert isinstance(yerr, list)
    assert all(isinstance(value, float) for value in yerr)
    assert yerr == [0.002, 0.04, 0.06]


def test_metric_yerr_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        _metric_yerr("sid", [1.0, 2.0], [0.5])


def test_csv_std_values_are_unchanged_when_clipping_applies(
    tmp_path: Path,
) -> None:
    # Use a synthetic record set where every centred_only/dagma
    # candidate has mean SID == 0 and the seeds disagree on SHD so
    # std_shd > mean_shd. Despite the plot clipping, the stored
    # std values in the CSV must equal the ranker's std_shd output
    # exactly (no propagation back into reported metrics).
    overrides: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for hyper_value in _grid_for("dagma"):
        overrides[("dagma", "centred_only", hyper_value, 201)] = {
            "shd": 0,
            "sid": 0,
            "mmd_primary": 0.001,
        }
        overrides[("dagma", "centred_only", hyper_value, 202)] = {
            "shd": 4,
            "sid": 0,
            "mmd_primary": 0.001,
        }
    records = _make_all_records(overrides=overrides)

    run_dir = _write_synthetic_calibration_tree(
        tmp_path, records=records
    )
    generate_calibration_readout(run_dir)

    ranking_rows = _read_csv_rows(
        run_dir / READOUT_DIRECTORY_NAME / RANKING_CSV_FILENAME
    )
    dagma_centred = [
        row
        for row in ranking_rows
        if row["condition"] == "centred_only"
        and row["model"] == "dagma"
    ]
    assert dagma_centred, "expected centred_only/dagma rows"
    for row in dagma_centred:
        mean_shd = float(row["mean_shd"])
        std_shd = float(row["std_shd"])
        # The std value is the ranker's reported sample std; it must
        # not be reduced by the plot clip even though mean_shd < std_shd.
        assert std_shd > mean_shd
        # Sample std with ddof=1 of the values {0, 4} is sqrt(8) ~ 2.828.
        assert std_shd == pytest.approx(math.sqrt(8.0))

    # The clip applies to the plot yerr only.
    yerr = _metric_yerr(
        "shd",
        [float(row["mean_shd"]) for row in dagma_centred],
        [float(row["std_shd"]) for row in dagma_centred],
    )
    lower, upper = yerr
    assert all(lo == 0.0 for lo in lower), lower
    assert upper == [float(row["std_shd"]) for row in dagma_centred]


def test_readout_is_idempotent_on_rerun(tmp_path: Path) -> None:
    run_dir = _write_synthetic_calibration_tree(tmp_path)

    generate_calibration_readout(run_dir)
    first_snapshot = _snapshot_directory(run_dir / READOUT_DIRECTORY_NAME)

    generate_calibration_readout(run_dir)
    second_snapshot = _snapshot_directory(run_dir / READOUT_DIRECTORY_NAME)

    # Markdown and CSV outputs are deterministic; PNG outputs depend
    # on matplotlib internals and are allowed to differ. We assert
    # determinism only for the text artefacts.
    for filename in (
        MARKDOWN_FILENAME,
        SELECTED_CSV_FILENAME,
        RANKING_CSV_FILENAME,
        STATUS_CSV_FILENAME,
    ):
        assert first_snapshot[filename] == second_snapshot[filename]
