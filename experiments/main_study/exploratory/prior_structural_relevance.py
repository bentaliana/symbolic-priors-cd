"""Offline diagnostic over existing main-evaluation artefacts.

Investigates the structural relationship between the clean
forbidden-edge priors used in the frozen main evaluation and the
empirical structural errors made by the prior-free baseline. Inputs
are existing per-run records and their persisted thresholded /
continuous / true adjacency artefacts. No model is trained, no
metric is recomputed beyond offline SID / SHD recomputation on
edited adjacency matrices, and no new interventional sampling is
performed. MMD is read from the existing records only.

Output directory:
    ``<output_root>/results/main_study/exploratory/
    prior_structural_relevance/<analysis_hash12>/``
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from experiments.main_study.records import (
    MainStudyRunRecord,
    record_from_json,
)
from experiments.main_study.run_io import resolve_relative_path
from symbolic_priors_cd.metrics import shd, sid_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


ANALYSIS_PROTOCOL_VERSION: str = "prior_structural_relevance_v1"

EVALUATION_SEED_VALUES: tuple[int, ...] = (
    501, 502, 503, 504, 505, 506, 507,
)

BASELINE_LABEL_PRIOR_FREE: str = "prior_free"
BASELINE_LABEL_MATCHED_L1: str = "matched_l1"
BASELINE_LABEL_SOFT_CLEAN_CONF1: str = "soft_frobenius_clean_conf1"
BASELINE_LABEL_HARD_EXCLUSION_CLEAN: str = "hard_exclusion_clean"

BASELINE_CONDITION_LABELS: tuple[str, ...] = (
    BASELINE_LABEL_PRIOR_FREE,
    BASELINE_LABEL_MATCHED_L1,
    BASELINE_LABEL_SOFT_CLEAN_CONF1,
    BASELINE_LABEL_HARD_EXCLUSION_CLEAN,
)

THRESHOLDED_KEY: str = "thresholded_adjacency"
CONTINUOUS_W_KEY: str = "continuous_w"
TRUE_ADJACENCY_KEY: str = "true_adjacency"

PROJECT_THRESHOLD: float = 0.3
SHD_REVERSAL_COST: int = 2

# Output filenames.
PRIOR_TARGET_OVERLAP_CSV: str = "prior_target_overlap.csv"
PRIOR_FREE_ERROR_DECOMPOSITION_CSV: str = (
    "prior_free_error_decomposition.csv"
)
OFFLINE_REMOVAL_EFFECT_CSV: str = (
    "offline_forbidden_edge_removal_effect.csv"
)
TOPOLOGICAL_RELEVANCE_CSV: str = (
    "prior_edge_topological_relevance.csv"
)
READOUT_MARKDOWN: str = "investigation_readout.md"
MANIFEST_JSON: str = "investigation_manifest.json"
AGGREGATED_ERROR_HEATMAP_PNG: str = "aggregated_error_heatmap.png"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def records_dir_for_run(
    output_root: Path, main_evaluation_run_hash12: str
) -> Path:
    """Return the canonical per-run records directory."""
    return (
        output_root
        / "results"
        / "main_study"
        / main_evaluation_run_hash12
        / "records"
    )


def analysis_output_dir(
    output_root: Path, analysis_hash12: str
) -> Path:
    """Return the exploratory output directory for ``analysis_hash12``."""
    return (
        output_root
        / "results"
        / "main_study"
        / "exploratory"
        / "prior_structural_relevance"
        / analysis_hash12
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_main_records(
    output_root: Path, main_evaluation_run_hash12: str
) -> tuple[MainStudyRunRecord, ...]:
    """Load every ``*.json`` record under the canonical records directory.

    Sorted lexicographically by run_id. Rejects empty directories and
    duplicate run_ids.
    """
    rd = records_dir_for_run(output_root, main_evaluation_run_hash12)
    if not rd.exists() or not rd.is_dir():
        raise FileNotFoundError(
            f"records directory {rd!r} does not exist or is not a "
            "directory."
        )
    paths = sorted(rd.glob("*.json"))
    if not paths:
        raise ValueError(
            f"records directory {rd!r} contains no *.json records."
        )
    out: list[MainStudyRunRecord] = []
    seen: set[str] = set()
    for path in paths:
        record = record_from_json(path.read_text(encoding="utf-8"))
        if record.run_id in seen:
            raise ValueError(
                f"duplicate run_id {record.run_id!r} in {rd!r}."
            )
        seen.add(record.run_id)
        out.append(record)
    out.sort(key=lambda r: r.run_id)
    return tuple(out)


def _is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0.0)


def _matches_clean_soft_conf1(record: MainStudyRunRecord) -> bool:
    cfg = record.config
    if cfg.method_family != "soft_frobenius":
        return False
    if cfg.corrupted_prior_spec is None:
        return False
    if not _is_close(
        float(cfg.corrupted_prior_spec.corruption_fraction), 0.0
    ):
        return False
    return cfg.confidence is not None and _is_close(
        float(cfg.confidence), 1.0
    )


def _matches_hard_exclusion_clean(record: MainStudyRunRecord) -> bool:
    cfg = record.config
    if cfg.method_family != "hard_exclusion":
        return False
    if cfg.corrupted_prior_spec is None:
        return False
    return _is_close(
        float(cfg.corrupted_prior_spec.corruption_fraction), 0.0
    )


def _matches_baseline_label(
    record: MainStudyRunRecord, label: str
) -> bool:
    if label == BASELINE_LABEL_PRIOR_FREE:
        return record.config.method_family == "prior_free"
    if label == BASELINE_LABEL_MATCHED_L1:
        return record.config.method_family == "matched_l1"
    if label == BASELINE_LABEL_SOFT_CLEAN_CONF1:
        return _matches_clean_soft_conf1(record)
    if label == BASELINE_LABEL_HARD_EXCLUSION_CLEAN:
        return _matches_hard_exclusion_clean(record)
    raise ValueError(f"unknown baseline label: {label!r}")


def find_clean_soft_reference_records(
    records: Iterable[MainStudyRunRecord],
) -> dict[int, MainStudyRunRecord]:
    """Return ``seed -> clean-soft soft_frobenius record`` mapping.

    Exactly one soft_frobenius record at
    ``(corruption_fraction=0.0, confidence=1.0)`` must exist per
    evaluation seed; otherwise raises ``ValueError``.
    """
    found: dict[int, list[MainStudyRunRecord]] = {}
    for rec in records:
        if not _matches_clean_soft_conf1(rec):
            continue
        found.setdefault(int(rec.config.seed_value), []).append(rec)
    out: dict[int, MainStudyRunRecord] = {}
    for seed in EVALUATION_SEED_VALUES:
        members = found.get(int(seed), [])
        if len(members) != 1:
            raise ValueError(
                "expected exactly one clean-soft "
                "(corruption=0.0, confidence=1.0) reference record "
                f"per seed; seed {seed!r} has {len(members)}."
            )
        out[int(seed)] = members[0]
    return out


def find_baseline_condition_records(
    records: Iterable[MainStudyRunRecord],
) -> dict[tuple[int, str], MainStudyRunRecord]:
    """Return ``(seed, baseline_label) -> record`` for every baseline.

    Raises if any seed-condition pair is missing or duplicated.
    """
    records_t = tuple(records)
    out: dict[tuple[int, str], MainStudyRunRecord] = {}
    for seed in EVALUATION_SEED_VALUES:
        for label in BASELINE_CONDITION_LABELS:
            matches = [
                r for r in records_t
                if int(r.config.seed_value) == int(seed)
                and _matches_baseline_label(r, label)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "expected exactly one record for "
                    f"seed={seed!r}, condition={label!r}; "
                    f"got {len(matches)}."
                )
            out[(int(seed), label)] = matches[0]
    return out


def _validate_square_bool_array(
    arr: np.ndarray, *, name: str
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy ndarray; got "
            f"{type(arr).__name__}."
        )
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a 2D square array; got shape {arr.shape}."
        )
    if arr.dtype.kind not in "bi":
        raise ValueError(
            f"{name} must be a bool or integer array; got dtype "
            f"{arr.dtype}."
        )
    return np.asarray(arr, dtype=bool)


def _validate_square_numeric_array(
    arr: np.ndarray, *, name: str
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy ndarray; got "
            f"{type(arr).__name__}."
        )
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a 2D square array; got shape {arr.shape}."
        )
    if arr.dtype.kind not in "fiu":
        raise ValueError(
            f"{name} must be a numeric array; got dtype {arr.dtype}."
        )
    return np.asarray(arr, dtype=float)


def _load_npz_array(
    relative_path: Optional[str],
    *,
    base_dir: Path,
    key: str,
) -> Optional[np.ndarray]:
    if relative_path is None:
        return None
    full = resolve_relative_path(relative_path, base_dir=base_dir)
    if not full.exists():
        raise FileNotFoundError(
            f"artefact {relative_path!r} not found under base_dir "
            f"{base_dir!r}."
        )
    with np.load(full) as data:
        if key not in data.files:
            raise ValueError(
                f"npz at {relative_path!r} is missing key {key!r}; "
                f"available: {list(data.files)}."
            )
        return np.asarray(data[key]).copy()


def load_thresholded_adjacency(
    record: MainStudyRunRecord, base_dir: Path
) -> np.ndarray:
    """Load and validate the persisted thresholded adjacency."""
    arr = _load_npz_array(
        record.thresholded_adjacency_path,
        base_dir=base_dir, key=THRESHOLDED_KEY,
    )
    if arr is None:
        raise ValueError(
            f"record {record.run_id!r} has no thresholded_adjacency_path."
        )
    return _validate_square_bool_array(arr, name="thresholded_adjacency")


def load_true_adjacency(
    record: MainStudyRunRecord, base_dir: Path
) -> np.ndarray:
    """Load and validate the persisted true adjacency."""
    arr = _load_npz_array(
        record.true_adjacency_path,
        base_dir=base_dir, key=TRUE_ADJACENCY_KEY,
    )
    if arr is None:
        raise ValueError(
            f"record {record.run_id!r} has no true_adjacency_path."
        )
    return _validate_square_bool_array(arr, name="true_adjacency")


def load_continuous_w(
    record: MainStudyRunRecord, base_dir: Path
) -> np.ndarray:
    """Load and validate the persisted continuous weight matrix."""
    arr = _load_npz_array(
        record.continuous_w_path,
        base_dir=base_dir, key=CONTINUOUS_W_KEY,
    )
    if arr is None:
        raise ValueError(
            f"record {record.run_id!r} has no continuous_w_path."
        )
    return _validate_square_numeric_array(arr, name="continuous_w")


# ---------------------------------------------------------------------------
# Edge-set utilities
# ---------------------------------------------------------------------------


def _off_diagonal_mask(n: int) -> np.ndarray:
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m


def edge_count(adjacency: np.ndarray) -> int:
    """Count off-diagonal True entries only."""
    arr = _validate_square_bool_array(adjacency, name="adjacency")
    a = arr.copy()
    np.fill_diagonal(a, False)
    return int(a.sum())


def classify_edges(
    predicted: np.ndarray, true: np.ndarray
) -> dict[str, set[tuple[int, int]]]:
    """Off-diagonal TP / TN / FP / FN edge sets.

    Returns a dict with keys ``true_positive_edges``,
    ``true_negative_edges``, ``false_positive_edges``,
    ``false_negative_edges``. Each value is a set of ``(i, j)``
    pairs; row index ``i`` is the source, column index ``j`` is the
    destination (project's row-source / column-destination
    convention).
    """
    pred = _validate_square_bool_array(predicted, name="predicted")
    truth = _validate_square_bool_array(true, name="true")
    if pred.shape != truth.shape:
        raise ValueError(
            f"predicted shape {pred.shape} does not match true "
            f"shape {truth.shape}."
        )
    n = int(pred.shape[0])
    off = _off_diagonal_mask(n)
    tp_mask = pred & truth & off
    tn_mask = (~pred) & (~truth) & off
    fp_mask = pred & (~truth) & off
    fn_mask = (~pred) & truth & off
    def _to_set(mask: np.ndarray) -> set[tuple[int, int]]:
        ii, jj = np.where(mask)
        return {(int(i), int(j)) for i, j in zip(ii, jj)}
    return {
        "true_positive_edges": _to_set(tp_mask),
        "true_negative_edges": _to_set(tn_mask),
        "false_positive_edges": _to_set(fp_mask),
        "false_negative_edges": _to_set(fn_mask),
    }


def remove_reference_forbidden_edges(
    predicted: np.ndarray,
    forbidden_edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Return a copy of ``predicted`` with ``forbidden_edges`` zeroed.

    The input array is not mutated.
    """
    pred = _validate_square_bool_array(predicted, name="predicted")
    edited = pred.copy()
    n = int(edited.shape[0])
    for (i, j) in forbidden_edges:
        if not (0 <= int(i) < n) or not (0 <= int(j) < n):
            raise ValueError(
                f"forbidden edge {(i, j)!r} out of range for shape "
                f"{edited.shape}."
            )
        edited[int(i), int(j)] = False
    return edited


def _forbidden_edges_for_seed(
    reference_records: dict[int, MainStudyRunRecord], seed: int,
) -> tuple[tuple[int, int], ...]:
    rec = reference_records[int(seed)]
    cps = rec.config.corrupted_prior_spec
    if cps is None or not cps.forbidden_edges:
        raise ValueError(
            f"clean-soft reference for seed {seed!r} has no "
            "forbidden_edges."
        )
    return tuple((int(i), int(j)) for (i, j) in cps.forbidden_edges)


# ---------------------------------------------------------------------------
# Topology helpers (over the true DAG)
# ---------------------------------------------------------------------------


def _descendants(true_adj: np.ndarray, node: int) -> set[int]:
    """Return all nodes reachable from ``node`` in the true DAG."""
    n = int(true_adj.shape[0])
    visited: set[int] = set()
    stack: list[int] = [int(node)]
    while stack:
        cur = stack.pop()
        for nxt in range(n):
            if nxt == cur:
                continue
            if true_adj[cur, nxt] and nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    visited.discard(int(node))
    return visited


def _ancestors(true_adj: np.ndarray, node: int) -> set[int]:
    """Return all nodes that can reach ``node`` in the true DAG."""
    n = int(true_adj.shape[0])
    visited: set[int] = set()
    stack: list[int] = [int(node)]
    while stack:
        cur = stack.pop()
        for prv in range(n):
            if prv == cur:
                continue
            if true_adj[prv, cur] and prv not in visited:
                visited.add(prv)
                stack.append(prv)
    visited.discard(int(node))
    return visited


def _out_degree(adj: np.ndarray, node: int) -> int:
    arr = np.asarray(adj, dtype=bool)
    return int(arr[int(node), :].sum() - bool(arr[int(node), int(node)]))


def _in_degree(adj: np.ndarray, node: int) -> int:
    arr = np.asarray(adj, dtype=bool)
    return int(arr[:, int(node)].sum() - bool(arr[int(node), int(node)]))


# ---------------------------------------------------------------------------
# CSV row computation
# ---------------------------------------------------------------------------


def compute_prior_target_overlap(
    *,
    baseline_records: dict[tuple[int, str], MainStudyRunRecord],
    reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """Per-seed-per-condition overlap with the clean-soft prior edge set."""
    rows: list[dict[str, Any]] = []
    for seed in EVALUATION_SEED_VALUES:
        forbidden = _forbidden_edges_for_seed(reference_records, seed)
        for label in BASELINE_CONDITION_LABELS:
            rec = baseline_records[(int(seed), label)]
            pred = load_thresholded_adjacency(rec, base_dir)
            n_pred_in_ref = sum(
                1 for (i, j) in forbidden if bool(pred[int(i), int(j)])
            )
            n_ref = len(forbidden)
            fraction = (
                float(n_pred_in_ref) / float(n_ref) if n_ref > 0 else None
            )
            rows.append({
                "seed_value": int(seed),
                "condition_label": label,
                "method_family": rec.config.method_family,
                "n_reference_forbidden_edges": int(n_ref),
                "n_reference_edges_predicted": int(n_pred_in_ref),
                "fraction_reference_edges_predicted": (
                    None if fraction is None else float(fraction)
                ),
                "edge_count": int(edge_count(pred)),
                "sid": (None if rec.sid is None else float(rec.sid)),
                "shd": (None if rec.shd is None else float(rec.shd)),
                "mmd": (None if rec.mmd is None else float(rec.mmd)),
            })
    return tuple(rows)


def compute_prior_free_error_decomposition(
    *,
    baseline_records: dict[tuple[int, str], MainStudyRunRecord],
    reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """Per-seed TP/TN/FP/FN counts plus targeted-FP coverage of the prior."""
    rows: list[dict[str, Any]] = []
    for seed in EVALUATION_SEED_VALUES:
        rec = baseline_records[(int(seed), BASELINE_LABEL_PRIOR_FREE)]
        pred = load_thresholded_adjacency(rec, base_dir)
        truth = load_true_adjacency(rec, base_dir)
        classes = classify_edges(pred, truth)
        n_tp = len(classes["true_positive_edges"])
        n_tn = len(classes["true_negative_edges"])
        n_fp = len(classes["false_positive_edges"])
        n_fn = len(classes["false_negative_edges"])
        n_true = int(np.asarray(truth, dtype=bool).sum() - np.trace(truth))
        n_pred = edge_count(pred)
        total_error = n_fp + n_fn
        targeted = set(_forbidden_edges_for_seed(reference_records, seed))
        targeted_fp_set = classes["false_positive_edges"] & targeted
        targeted_fp_n = len(targeted_fp_set)
        targeted_error_n = len(
            (classes["false_positive_edges"] | classes["false_negative_edges"])
            & targeted
        )
        rows.append({
            "seed_value": int(seed),
            "n_true_edges": int(n_true),
            "n_predicted_edges": int(n_pred),
            "true_positive_count": int(n_tp),
            "true_negative_count": int(n_tn),
            "false_positive_count": int(n_fp),
            "false_negative_count": int(n_fn),
            "total_error_count_simple": int(total_error),
            "targeted_false_positive_count": int(targeted_fp_n),
            "targeted_false_positive_fraction_of_fp": (
                None if n_fp == 0
                else float(targeted_fp_n) / float(n_fp)
            ),
            "targeted_error_fraction_of_total_errors": (
                None if total_error == 0
                else float(targeted_error_n) / float(total_error)
            ),
            "sid": (None if rec.sid is None else float(rec.sid)),
            "shd": (None if rec.shd is None else float(rec.shd)),
            "mmd": (None if rec.mmd is None else float(rec.mmd)),
        })
    return tuple(rows)


def compute_offline_removal_effect(
    *,
    baseline_records: dict[tuple[int, str], MainStudyRunRecord],
    reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
    sid_fn: Callable[[np.ndarray, np.ndarray], int] = sid_score,
    shd_fn: Callable[..., int] = shd,
) -> tuple[dict[str, Any], ...]:
    """Offline SID/SHD effect of zeroing the reference forbidden edges.

    For each evaluation seed, starts from the prior-free thresholded
    adjacency, removes the seed-specific reference forbidden edges,
    and recomputes SID and SHD using the project's public metric
    functions. MMD is intentionally not recomputed; that field is
    not in the output schema.
    """
    rows: list[dict[str, Any]] = []
    for seed in EVALUATION_SEED_VALUES:
        rec = baseline_records[(int(seed), BASELINE_LABEL_PRIOR_FREE)]
        pred = load_thresholded_adjacency(rec, base_dir)
        truth = load_true_adjacency(rec, base_dir)
        forbidden = _forbidden_edges_for_seed(reference_records, seed)
        n_ref_in_pred = sum(
            1 for (i, j) in forbidden if bool(pred[int(i), int(j)])
        )
        edited = remove_reference_forbidden_edges(pred, forbidden)
        n_removed = int(
            np.asarray(pred, dtype=bool).sum()
            - np.asarray(edited, dtype=bool).sum()
        )
        sid_orig = int(sid_fn(pred, truth))
        sid_after = int(sid_fn(edited, truth))
        shd_orig = int(shd_fn(pred, truth, reversal_cost=SHD_REVERSAL_COST))
        shd_after = int(shd_fn(edited, truth, reversal_cost=SHD_REVERSAL_COST))
        rows.append({
            "seed_value": int(seed),
            "sid_original": int(sid_orig),
            "sid_after_removing_reference_forbidden_edges": int(sid_after),
            "sid_delta": int(sid_after - sid_orig),
            "shd_original": int(shd_orig),
            "shd_after_removing_reference_forbidden_edges": int(shd_after),
            "shd_delta": int(shd_after - shd_orig),
            "n_reference_edges_predicted_before_removal": int(n_ref_in_pred),
            "n_reference_edges_removed": int(n_removed),
        })
    return tuple(rows)


def compute_minimal_topological_relevance(
    *,
    baseline_records: dict[tuple[int, str], MainStudyRunRecord],
    reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """Per-edge minimal topological descriptors over the true DAG."""
    rows: list[dict[str, Any]] = []
    for seed in EVALUATION_SEED_VALUES:
        rec_pf = baseline_records[
            (int(seed), BASELINE_LABEL_PRIOR_FREE)
        ]
        pred = load_thresholded_adjacency(rec_pf, base_dir)
        truth = load_true_adjacency(rec_pf, base_dir)
        forbidden = _forbidden_edges_for_seed(reference_records, seed)
        for (i, j) in forbidden:
            rows.append({
                "seed_value": int(seed),
                "source_node": int(i),
                "target_node": int(j),
                "predicted_by_prior_free": bool(pred[int(i), int(j)]),
                "target_descendant_count": int(
                    len(_descendants(truth, int(j)))
                ),
                "source_ancestor_count": int(
                    len(_ancestors(truth, int(i)))
                ),
                "target_out_degree": _out_degree(truth, int(j)),
                "target_in_degree": _in_degree(truth, int(j)),
                "source_out_degree": _out_degree(truth, int(i)),
                "source_in_degree": _in_degree(truth, int(i)),
            })
    return tuple(rows)


# ---------------------------------------------------------------------------
# Hash and manifest
# ---------------------------------------------------------------------------


def compute_analysis_hash12(
    *,
    main_evaluation_run_hash12: str,
    input_run_ids: Iterable[str],
    input_configuration_hashes: Iterable[str],
    analysis_protocol_version: str = ANALYSIS_PROTOCOL_VERSION,
) -> tuple[str, dict[str, Any]]:
    """Deterministic 12-char hex hash and the exact payload that produced it."""
    payload = {
        "main_evaluation_run_hash12": str(main_evaluation_run_hash12),
        "analysis_protocol_version": str(analysis_protocol_version),
        "input_run_ids_sorted": sorted(str(x) for x in input_run_ids),
        "input_configuration_hashes_sorted": sorted(
            str(x) for x in input_configuration_hashes
        ),
    }
    serialised = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return digest[:12], payload


# ---------------------------------------------------------------------------
# CSV / JSON / Markdown writers
# ---------------------------------------------------------------------------


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return repr(float(value))
    return str(value)


def write_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    """Write ``rows`` with fixed column order. None becomes empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    field_set = set(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(fieldnames))
        for r in rows:
            extras = set(r.keys()) - field_set
            if extras:
                raise ValueError(
                    "write_csv: row contains unexpected keys "
                    f"{sorted(extras)} (allowed: {sorted(field_set)})."
                )
            writer.writerow([
                _csv_cell(r.get(col, None)) for col in fieldnames
            ])


def write_manifest_json(
    manifest: dict[str, Any], path: Path
) -> None:
    """Write the analysis manifest as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def _agg_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [
        float(v) for v in values
        if v is not None and math.isfinite(float(v))
    ]
    if not finite:
        return None
    return float(sum(finite)) / float(len(finite))


def write_analysis_readout(
    *,
    output_path: Path,
    main_evaluation_run_hash12: str,
    analysis_hash12: str,
    output_dir_relative: str,
    overlap_rows: tuple[dict[str, Any], ...],
    decomposition_rows: tuple[dict[str, Any], ...],
    removal_rows: tuple[dict[str, Any], ...],
    topology_rows: tuple[dict[str, Any], ...],
    heatmap_status: str,
) -> Path:
    """Write the cautious labelling-only investigation readout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate descriptors used in the readout tables.
    pf_overlap = [
        r for r in overlap_rows
        if r["condition_label"] == BASELINE_LABEL_PRIOR_FREE
    ]
    soft_overlap = [
        r for r in overlap_rows
        if r["condition_label"] == BASELINE_LABEL_SOFT_CLEAN_CONF1
    ]
    hard_overlap = [
        r for r in overlap_rows
        if r["condition_label"] == BASELINE_LABEL_HARD_EXCLUSION_CLEAN
    ]
    matched_overlap = [
        r for r in overlap_rows
        if r["condition_label"] == BASELINE_LABEL_MATCHED_L1
    ]

    overlap_table_lines = ["| condition | mean fraction of reference edges predicted | n seeds |",
                          "| --- | --- | --- |"]
    for label, subset in (
        (BASELINE_LABEL_PRIOR_FREE, pf_overlap),
        (BASELINE_LABEL_MATCHED_L1, matched_overlap),
        (BASELINE_LABEL_SOFT_CLEAN_CONF1, soft_overlap),
        (BASELINE_LABEL_HARD_EXCLUSION_CLEAN, hard_overlap),
    ):
        mean_frac = _agg_mean(
            r["fraction_reference_edges_predicted"] for r in subset
        )
        overlap_table_lines.append(
            f"| `{label}` | "
            f"{('-' if mean_frac is None else f'{mean_frac:.4g}')} "
            f"| {len(subset)} |"
        )

    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4g}"

    decomp_table_lines = [
        "| seed | n_true_edges | n_predicted | TP | FP | FN | "
        "total_error | targeted_FP | targeted_FP / FP | SID | SHD | MMD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in decomposition_rows:
        frac = r["targeted_false_positive_fraction_of_fp"]
        sid_v = r["sid"]
        shd_v = r["shd"]
        mmd_v = r["mmd"]
        decomp_table_lines.append(
            f"| {r['seed_value']} | {r['n_true_edges']} | "
            f"{r['n_predicted_edges']} | {r['true_positive_count']} | "
            f"{r['false_positive_count']} | {r['false_negative_count']} | "
            f"{r['total_error_count_simple']} | "
            f"{r['targeted_false_positive_count']} | "
            f"{_fmt(frac)} | {_fmt(sid_v)} | {_fmt(shd_v)} | {_fmt(mmd_v)} |"
        )

    removal_table_lines = [
        "| seed | SID_orig | SID_after | dSID | SHD_orig | SHD_after | "
        "dSHD | n_ref_edges_predicted_before | n_removed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in removal_rows:
        removal_table_lines.append(
            f"| {r['seed_value']} | {r['sid_original']} | "
            f"{r['sid_after_removing_reference_forbidden_edges']} | "
            f"{r['sid_delta']} | {r['shd_original']} | "
            f"{r['shd_after_removing_reference_forbidden_edges']} | "
            f"{r['shd_delta']} | "
            f"{r['n_reference_edges_predicted_before_removal']} | "
            f"{r['n_reference_edges_removed']} |"
        )

    mean_dsid = _agg_mean(r["sid_delta"] for r in removal_rows)
    mean_dshd = _agg_mean(r["shd_delta"] for r in removal_rows)

    mean_target_descendants = _agg_mean(
        r["target_descendant_count"] for r in topology_rows
    )
    mean_target_in = _agg_mean(
        r["target_in_degree"] for r in topology_rows
    )
    mean_target_out = _agg_mean(
        r["target_out_degree"] for r in topology_rows
    )

    lines: list[str] = []
    lines.append("# Prior structural relevance: exploratory analysis")
    lines.append("")
    lines.append("## Run identity")
    lines.append("")
    lines.append(
        f"- `main_evaluation_run_hash12`: `{main_evaluation_run_hash12}`"
    )
    lines.append(f"- `analysis_hash12`: `{analysis_hash12}`")
    lines.append(
        f"- analysis protocol version: `{ANALYSIS_PROTOCOL_VERSION}`"
    )
    lines.append(f"- output directory: `{output_dir_relative}`")
    lines.append("")
    lines.append(
        "This analysis is exploratory. Existing saved artefacts only "
        "were used. No new model fitting, no MMD recomputation, and "
        "no new interventional sampling were performed. This analysis "
        "does not replace the frozen primary result."
    )
    lines.append("")
    lines.append("## Evidence files used")
    lines.append("")
    lines.append(
        f"- 28 records loaded from "
        f"`results/main_study/{main_evaluation_run_hash12}/records/`: "
        "the 4 baseline conditions x 7 evaluation seeds."
    )
    lines.append(
        "- For each record, the persisted "
        "`thresholded_adjacency.npz`, `continuous_w.npz`, and "
        "`true_adjacency.npz` artefacts were read."
    )
    lines.append("")
    lines.append("## Prior-target overlap summary")
    lines.append("")
    lines.append(
        "Per-condition mean fraction of the seed-specific clean-soft "
        "reference forbidden-edge set that the condition predicts as "
        "edges. Lower values mean the condition suppresses the "
        "reference forbidden edges more strongly."
    )
    lines.append("")
    lines.extend(overlap_table_lines)
    lines.append("")
    lines.append("## Prior-free error decomposition summary")
    lines.append("")
    lines.append(
        "Off-diagonal TP / FP / FN counts for the prior-free baseline "
        "per seed, with targeted-false-positive counts and the "
        "primary relevance quantity "
        "`targeted_false_positive_fraction_of_fp`. SID, SHD, and MMD "
        "are read from the saved records."
    )
    lines.append("")
    lines.extend(decomp_table_lines)
    lines.append("")
    lines.append("## Offline SID/SHD removal summary")
    lines.append("")
    lines.append(
        "For each seed, the prior-free thresholded adjacency was "
        "edited offline by zeroing the seed-specific reference "
        "forbidden-edge positions, and SID and SHD were recomputed "
        "with the project's public metric functions. MMD is not "
        "recomputed; the column is intentionally omitted."
    )
    lines.append("")
    lines.extend(removal_table_lines)
    lines.append("")
    if mean_dsid is not None:
        lines.append(
            f"- Mean dSID across seeds: {mean_dsid:.4g} "
            "(after - original)."
        )
    if mean_dshd is not None:
        lines.append(
            f"- Mean dSHD across seeds: {mean_dshd:.4g} "
            "(after - original)."
        )
    lines.append("")
    lines.append("## Minimal topological relevance summary")
    lines.append("")
    lines.append(
        "Per reference forbidden edge `(source, target)`, descriptive "
        "topological properties over the true DAG: target descendant "
        "count, source ancestor count, and target/source in- and "
        "out-degrees. Path-length analysis, centrality measures, and "
        "intervention-effect computations are intentionally out of "
        "scope."
    )
    if mean_target_descendants is not None:
        lines.append(
            "- Mean target descendant count across all reference "
            f"edges: {mean_target_descendants:.4g}."
        )
    if mean_target_in is not None:
        lines.append(
            "- Mean target in-degree across all reference edges: "
            f"{mean_target_in:.4g}."
        )
    if mean_target_out is not None:
        lines.append(
            "- Mean target out-degree across all reference edges: "
            f"{mean_target_out:.4g}."
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- This analysis is offline and exploratory; it cannot "
        "substitute for a pre-registered statistical test."
    )
    lines.append(
        "- Offline SID/SHD recomputation on edited adjacency matrices "
        "is a structural counterfactual; it does not estimate the "
        "downstream interventional-distribution effect."
    )
    lines.append(
        "- MMD counterfactuals are explicitly out of scope; the saved "
        "MMD values are read as-is."
    )
    lines.append(
        "- Coverage bands and topological summaries are heuristic "
        "diagnostic aids, not statistical thresholds."
    )
    lines.append("")
    lines.append("## Implication for possible lambda_prior sensitivity")
    lines.append("")
    lines.append(
        "If the offline removal effect on SID and SHD is small in "
        "magnitude across seeds, then perfect targeted suppression "
        "of the reference forbidden edges would have produced only a "
        "small direct improvement on these structural metrics. A "
        "future sensitivity study at varied `lambda_prior` could "
        "examine indirect optimisation effects; such a study is out "
        "of scope here."
    )
    lines.append("")
    lines.append(f"- aggregated error heatmap: {heatmap_status}.")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Optional aggregated error heatmap
# ---------------------------------------------------------------------------


def make_aggregated_error_heatmap(
    *,
    baseline_records: dict[tuple[int, str], MainStudyRunRecord],
    reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
    output_path: Path,
) -> Optional[Path]:
    """Optional 10x10 heatmap of prior-free FP+FN frequency across seeds.

    Returns the output path on success, ``None`` on any failure
    (e.g. matplotlib unavailable). Reference forbidden-edge positions
    are marked with white outlined dots.
    """
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None
    first_rec = baseline_records[
        (int(EVALUATION_SEED_VALUES[0]), BASELINE_LABEL_PRIOR_FREE)
    ]
    pred0 = load_thresholded_adjacency(first_rec, base_dir)
    n = int(pred0.shape[0])
    counts = np.zeros((n, n), dtype=float)
    reference_positions: set[tuple[int, int]] = set()
    for seed in EVALUATION_SEED_VALUES:
        rec = baseline_records[(int(seed), BASELINE_LABEL_PRIOR_FREE)]
        pred = load_thresholded_adjacency(rec, base_dir)
        truth = load_true_adjacency(rec, base_dir)
        classes = classify_edges(pred, truth)
        for (i, j) in classes["false_positive_edges"]:
            counts[int(i), int(j)] += 1.0
        for (i, j) in classes["false_negative_edges"]:
            counts[int(i), int(j)] += 1.0
        for edge in _forbidden_edges_for_seed(reference_records, seed):
            reference_positions.add(tuple(edge))
    try:
        fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
        im = ax.imshow(counts, aspect="equal", cmap="magma", origin="lower")
        ax.set_xlabel("destination node (column)")
        ax.set_ylabel("source node (row)")
        ax.set_title(
            "Aggregate prior-free FP + FN frequency across 7 seeds"
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("number of seeds with FP or FN at (i, j)")
        for (i, j) in sorted(reference_positions):
            ax.plot(
                int(j), int(i),
                marker="o", markersize=6,
                markerfacecolor="none",
                markeredgecolor="white", markeredgewidth=1.2,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Output column orders
# ---------------------------------------------------------------------------


_OVERLAP_COLUMNS: tuple[str, ...] = (
    "seed_value", "condition_label", "method_family",
    "n_reference_forbidden_edges", "n_reference_edges_predicted",
    "fraction_reference_edges_predicted",
    "edge_count", "sid", "shd", "mmd",
)

_DECOMPOSITION_COLUMNS: tuple[str, ...] = (
    "seed_value", "n_true_edges", "n_predicted_edges",
    "true_positive_count", "true_negative_count",
    "false_positive_count", "false_negative_count",
    "total_error_count_simple",
    "targeted_false_positive_count",
    "targeted_false_positive_fraction_of_fp",
    "targeted_error_fraction_of_total_errors",
    "sid", "shd", "mmd",
)

_REMOVAL_COLUMNS: tuple[str, ...] = (
    "seed_value",
    "sid_original",
    "sid_after_removing_reference_forbidden_edges",
    "sid_delta",
    "shd_original",
    "shd_after_removing_reference_forbidden_edges",
    "shd_delta",
    "n_reference_edges_predicted_before_removal",
    "n_reference_edges_removed",
)

_TOPOLOGY_COLUMNS: tuple[str, ...] = (
    "seed_value", "source_node", "target_node",
    "predicted_by_prior_free",
    "target_descendant_count", "source_ancestor_count",
    "target_out_degree", "target_in_degree",
    "source_out_degree", "source_in_degree",
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_prior_structural_relevance_analysis(
    *,
    output_root: Path,
    main_evaluation_run_hash12: str,
) -> dict[str, Any]:
    """End-to-end exploratory analysis. Returns the manifest dict."""
    if not isinstance(output_root, Path):
        raise TypeError(
            f"output_root must be a pathlib.Path; got "
            f"{type(output_root).__name__}."
        )
    if (
        not isinstance(main_evaluation_run_hash12, str)
        or len(main_evaluation_run_hash12) != 12
    ):
        raise ValueError(
            "main_evaluation_run_hash12 must be a 12-character "
            f"string; got {main_evaluation_run_hash12!r}."
        )
    records = load_main_records(output_root, main_evaluation_run_hash12)
    reference_records = find_clean_soft_reference_records(records)
    baseline_records = find_baseline_condition_records(records)

    # Build the 28-record input identity (4 conditions x 7 seeds).
    input_records: list[MainStudyRunRecord] = []
    for seed in EVALUATION_SEED_VALUES:
        for label in BASELINE_CONDITION_LABELS:
            input_records.append(baseline_records[(int(seed), label)])
    input_run_ids = [r.run_id for r in input_records]
    input_configuration_hashes = [
        r.configuration_hash_full for r in input_records
    ]
    analysis_hash12, hash_payload = compute_analysis_hash12(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        input_run_ids=input_run_ids,
        input_configuration_hashes=input_configuration_hashes,
    )

    out_dir = analysis_output_dir(output_root, analysis_hash12)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows = compute_prior_target_overlap(
        baseline_records=baseline_records,
        reference_records=reference_records,
        base_dir=output_root,
    )
    decomposition_rows = compute_prior_free_error_decomposition(
        baseline_records=baseline_records,
        reference_records=reference_records,
        base_dir=output_root,
    )
    removal_rows = compute_offline_removal_effect(
        baseline_records=baseline_records,
        reference_records=reference_records,
        base_dir=output_root,
    )
    topology_rows = compute_minimal_topological_relevance(
        baseline_records=baseline_records,
        reference_records=reference_records,
        base_dir=output_root,
    )

    write_csv(
        overlap_rows,
        out_dir / PRIOR_TARGET_OVERLAP_CSV,
        _OVERLAP_COLUMNS,
    )
    write_csv(
        decomposition_rows,
        out_dir / PRIOR_FREE_ERROR_DECOMPOSITION_CSV,
        _DECOMPOSITION_COLUMNS,
    )
    write_csv(
        removal_rows,
        out_dir / OFFLINE_REMOVAL_EFFECT_CSV,
        _REMOVAL_COLUMNS,
    )
    write_csv(
        topology_rows,
        out_dir / TOPOLOGICAL_RELEVANCE_CSV,
        _TOPOLOGY_COLUMNS,
    )

    heatmap_path = make_aggregated_error_heatmap(
        baseline_records=baseline_records,
        reference_records=reference_records,
        base_dir=output_root,
        output_path=out_dir / AGGREGATED_ERROR_HEATMAP_PNG,
    )
    heatmap_status = (
        f"generated at `{heatmap_path.name}`"
        if heatmap_path is not None else "skipped"
    )

    output_dir_relative = str(
        out_dir.relative_to(output_root)
    ).replace("\\", "/")

    write_analysis_readout(
        output_path=out_dir / READOUT_MARKDOWN,
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        analysis_hash12=analysis_hash12,
        output_dir_relative=output_dir_relative,
        overlap_rows=overlap_rows,
        decomposition_rows=decomposition_rows,
        removal_rows=removal_rows,
        topology_rows=topology_rows,
        heatmap_status=heatmap_status,
    )

    output_files: list[str] = [
        PRIOR_TARGET_OVERLAP_CSV,
        PRIOR_FREE_ERROR_DECOMPOSITION_CSV,
        OFFLINE_REMOVAL_EFFECT_CSV,
        TOPOLOGICAL_RELEVANCE_CSV,
        READOUT_MARKDOWN,
        MANIFEST_JSON,
    ]
    if heatmap_path is not None:
        output_files.append(AGGREGATED_ERROR_HEATMAP_PNG)

    manifest: dict[str, Any] = {
        "main_evaluation_run_hash12": main_evaluation_run_hash12,
        "analysis_hash12": analysis_hash12,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "hash_payload": hash_payload,
        "output_files": sorted(output_files),
        "n_records_loaded": int(len(records)),
        "n_input_records_hashed": int(len(input_records)),
        "n_seeds": int(len(EVALUATION_SEED_VALUES)),
        "no_new_fits": True,
        "no_mmd_recomputation": True,
        "no_new_sampling": True,
        "no_protocol_changes": True,
    }
    write_manifest_json(manifest, out_dir / MANIFEST_JSON)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prior_structural_relevance",
        description=(
            "Exploratory offline diagnostic over existing "
            "main-evaluation artefacts. Read-only over records and "
            "artefacts; no model fitting, no MMD recomputation, no "
            "new interventional sampling."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help=(
            "Root directory under which results/main_study/... is "
            "located."
        ),
    )
    parser.add_argument(
        "--main-evaluation-run-hash12", type=str, required=True,
        help="12-character main-evaluation run hash.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_prior_structural_relevance_analysis(
            output_root=args.output_root,
            main_evaluation_run_hash12=args.main_evaluation_run_hash12,
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            "prior_structural_relevance: error: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return 1
    sys.stdout.write(
        f"analysis_hash12: {manifest['analysis_hash12']}\n"
    )
    for f in manifest["output_files"]:
        sys.stdout.write(f"- {f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ANALYSIS_PROTOCOL_VERSION",
    "BASELINE_CONDITION_LABELS",
    "BASELINE_LABEL_HARD_EXCLUSION_CLEAN",
    "BASELINE_LABEL_MATCHED_L1",
    "BASELINE_LABEL_PRIOR_FREE",
    "BASELINE_LABEL_SOFT_CLEAN_CONF1",
    "EVALUATION_SEED_VALUES",
    "analysis_output_dir",
    "classify_edges",
    "compute_analysis_hash12",
    "compute_minimal_topological_relevance",
    "compute_offline_removal_effect",
    "compute_prior_free_error_decomposition",
    "compute_prior_target_overlap",
    "edge_count",
    "find_baseline_condition_records",
    "find_clean_soft_reference_records",
    "load_continuous_w",
    "load_main_records",
    "load_thresholded_adjacency",
    "load_true_adjacency",
    "main",
    "make_aggregated_error_heatmap",
    "records_dir_for_run",
    "remove_reference_forbidden_edges",
    "run_prior_structural_relevance_analysis",
    "write_analysis_readout",
    "write_csv",
    "write_manifest_json",
]
