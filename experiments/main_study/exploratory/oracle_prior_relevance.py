"""Offline alternative-prior relevance diagnostic.

Estimates how much SID/SHD improvement was directly available if
priors had targeted more relevant structural errors than the
original randomly-sampled forbidden-edge prior. Uses existing saved
graphs only: no model is trained, no metric is recomputed from raw
data beyond SID/SHD on edited adjacency matrices, no MMD
counterfactual is attempted, and no new interventional sample is
generated.

Diagnostic scenarios per seed:

- ``actual_reference_forbidden_removal``: reproduces the prior
  removal experiment by removing the seed-specific clean
  forbidden-edge set.
- ``fp_remove_budget10_exact``: exact exhaustive subset search over
  the prior-free false-positive set, removing up to ``budget_k``
  edges. The selection rule is SID-primary; this is not a guaranteed
  SID ceiling.
- ``fp_remove_all_false_positives``: full structural false-positive
  correction. This is a structural full-correction diagnostic, not
  guaranteed SID-optimal.
- ``fn_add_budget10_greedy_acyclic``: greedy SID-primary addition of
  up to ``budget_k`` prior-free false negatives, guarded by an
  acyclicity check. This is a greedy diagnostic approximation, not a
  global optimum.
- ``fn_add_full_greedy_acyclic``: same greedy procedure without the
  ``budget_k`` cap; continues until no beneficial valid addition
  remains.

This is the final scheduled exploratory diagnostic before thesis
writing.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from experiments.main_study.exploratory import (
    prior_structural_relevance as psr,
)
from experiments.main_study.records import (
    MainStudyRunRecord,
    record_from_json,
)
from experiments.main_study.run_io import resolve_relative_path
from symbolic_priors_cd.metrics import shd, sid_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


ANALYSIS_PROTOCOL_VERSION: str = "oracle_prior_relevance_v1"

ORACLE_BUDGET_K: int = 10

EVALUATION_SEED_VALUES: tuple[int, ...] = psr.EVALUATION_SEED_VALUES

SHD_REVERSAL_COST: int = 2

# Scenario labels (exact strings used in the per-seed CSV).
SCENARIO_ACTUAL_REFERENCE: str = "actual_reference_forbidden_removal"
SCENARIO_FP_BUDGET_EXACT: str = "fp_remove_budget10_exact"
SCENARIO_FP_REMOVE_ALL: str = "fp_remove_all_false_positives"
SCENARIO_FN_BUDGET_GREEDY: str = "fn_add_budget10_greedy_acyclic"
SCENARIO_FN_FULL_GREEDY: str = "fn_add_full_greedy_acyclic"

SCENARIO_LABELS: tuple[str, ...] = (
    SCENARIO_ACTUAL_REFERENCE,
    SCENARIO_FP_BUDGET_EXACT,
    SCENARIO_FP_REMOVE_ALL,
    SCENARIO_FN_BUDGET_GREEDY,
    SCENARIO_FN_FULL_GREEDY,
)

# Search-strategy strings emitted in the per-seed CSV.
STRATEGY_REMOVE_REFERENCE_FORBIDDEN: str = "remove_reference_forbidden"
STRATEGY_EXACT_SUBSET_SID_PRIMARY: str = "exact_subset_sid_primary"
STRATEGY_REMOVE_ALL_FALSE_POSITIVES: str = "remove_all_false_positives"
STRATEGY_GREEDY_BUDGET10: str = (
    "greedy_acyclic_sid_primary_budget10"
)
STRATEGY_GREEDY_FULL: str = (
    "greedy_acyclic_sid_primary_full_candidate"
)

# Output filenames.
ORACLE_PER_SEED_CSV: str = "oracle_diagnostics_per_seed.csv"
ORACLE_SUMMARY_CSV: str = "oracle_diagnostics_summary.csv"
ORACLE_READOUT_MD: str = "oracle_prior_relevance_readout.md"
ORACLE_MANIFEST_JSON: str = "oracle_prior_relevance_manifest.json"
ORACLE_SUMMARY_PLOT_PNG: str = "oracle_summary_plot.png"


# ---------------------------------------------------------------------------
# Reused helpers (thin re-exports / wrappers).
# ---------------------------------------------------------------------------


def classify_edges(
    predicted: np.ndarray, true: np.ndarray
) -> dict[str, set[tuple[int, int]]]:
    """Off-diagonal TP/TN/FP/FN sets (delegates to the existing helper)."""
    return psr.classify_edges(predicted, true)


def edge_count(adjacency: np.ndarray) -> int:
    """Count off-diagonal True entries (delegates to the existing helper)."""
    return psr.edge_count(adjacency)


def load_thresholded_adjacency(
    record: MainStudyRunRecord, base_dir: Path
) -> np.ndarray:
    """Load and validate the persisted thresholded adjacency."""
    return psr.load_thresholded_adjacency(record, base_dir)


def load_true_adjacency(
    record: MainStudyRunRecord, base_dir: Path
) -> np.ndarray:
    """Load and validate the persisted true adjacency."""
    return psr.load_true_adjacency(record, base_dir)


def write_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    """Deterministic CSV writer (delegates to the existing helper)."""
    psr.write_csv(rows, path, fieldnames)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_prior_free_records(
    output_root: Path, main_evaluation_run_hash12: str
) -> tuple[MainStudyRunRecord, ...]:
    """Return the seven prior_free records in seed-ascending order.

    Rejects missing, duplicate, or non-success records.
    """
    all_records = psr.load_main_records(
        output_root, main_evaluation_run_hash12
    )
    matches: dict[int, list[MainStudyRunRecord]] = {}
    for rec in all_records:
        if rec.config.method_family != "prior_free":
            continue
        matches.setdefault(int(rec.config.seed_value), []).append(rec)
    out: list[MainStudyRunRecord] = []
    for seed in EVALUATION_SEED_VALUES:
        members = matches.get(int(seed), [])
        if len(members) != 1:
            raise ValueError(
                "expected exactly one prior_free record per evaluation "
                f"seed; seed {seed!r} has {len(members)}."
            )
        rec = members[0]
        if rec.fit_status != "success":
            raise ValueError(
                f"prior_free record at seed {seed!r} has "
                f"fit_status={rec.fit_status!r}; expected 'success'."
            )
        out.append(rec)
    return tuple(out)


def load_clean_soft_reference_records(
    output_root: Path, main_evaluation_run_hash12: str
) -> dict[int, MainStudyRunRecord]:
    """Return ``seed -> clean-soft soft_frobenius record`` mapping."""
    all_records = psr.load_main_records(
        output_root, main_evaluation_run_hash12
    )
    return psr.find_clean_soft_reference_records(all_records)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def is_dag(adjacency: np.ndarray) -> bool:
    """Acyclicity check via Kahn's topological sort.

    Self-loops on the diagonal are treated as cycles. Returns
    ``True`` if and only if the graph (over the off-diagonal +
    diagonal entries) is acyclic.
    """
    if not isinstance(adjacency, np.ndarray):
        raise TypeError(
            f"adjacency must be a numpy ndarray; got "
            f"{type(adjacency).__name__}."
        )
    if (
        adjacency.ndim != 2
        or adjacency.shape[0] != adjacency.shape[1]
    ):
        raise ValueError(
            f"adjacency must be a 2D square array; got shape "
            f"{adjacency.shape}."
        )
    a = np.asarray(adjacency, dtype=bool).copy()
    n = int(a.shape[0])
    if np.any(np.diag(a)):
        return False
    in_degree = a.sum(axis=0).astype(int)
    queue: list[int] = [i for i in range(n) if int(in_degree[i]) == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in range(n):
            if a[node, nxt]:
                in_degree[nxt] -= 1
                if int(in_degree[nxt]) == 0:
                    queue.append(int(nxt))
    return visited == n


def compute_sid_shd(
    predicted: np.ndarray, true: np.ndarray
) -> dict[str, int]:
    """Recompute SID and SHD using the project's public metric API.

    Returns a dict with integer ``sid`` and ``shd``. MMD is never
    computed here.
    """
    sid_value = int(sid_score(predicted, true))
    shd_value = int(shd(predicted, true, reversal_cost=SHD_REVERSAL_COST))
    return {"sid": sid_value, "shd": shd_value}


def remove_edges(
    adjacency: np.ndarray, edges: Iterable[tuple[int, int]]
) -> np.ndarray:
    """Return a copy of ``adjacency`` with the listed entries set to False."""
    pred = np.asarray(adjacency, dtype=bool).copy()
    n = int(pred.shape[0])
    for (i, j) in edges:
        if not (0 <= int(i) < n) or not (0 <= int(j) < n):
            raise ValueError(
                f"edge {(i, j)!r} is out of range for shape "
                f"{pred.shape}."
            )
        pred[int(i), int(j)] = False
    return pred


def add_edges_with_acyclicity_guard(
    adjacency: np.ndarray,
    edges: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Try adding each edge in order; skip if it would create a cycle.

    Returns ``(edited_adjacency, added_edges, skipped_cycle_edges)``.
    Self-loops are always rejected as cycles.
    """
    pred = np.asarray(adjacency, dtype=bool).copy()
    added: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if i == j:
            skipped.append((i, j))
            continue
        if pred[i, j]:
            # Already present; no-op but record as added for clarity.
            continue
        trial = pred.copy()
        trial[i, j] = True
        if is_dag(trial):
            pred = trial
            added.append((i, j))
        else:
            skipped.append((i, j))
    return pred, added, skipped


# ---------------------------------------------------------------------------
# False-positive diagnostics
# ---------------------------------------------------------------------------


def _candidate_false_positive_edges(
    predicted: np.ndarray, true: np.ndarray
) -> tuple[tuple[int, int], ...]:
    """Return the FP edge set sorted lexicographically."""
    classes = classify_edges(predicted, true)
    return tuple(sorted(classes["false_positive_edges"]))


def _candidate_false_negative_edges(
    predicted: np.ndarray, true: np.ndarray
) -> tuple[tuple[int, int], ...]:
    """Return the FN edge set sorted lexicographically."""
    classes = classify_edges(predicted, true)
    return tuple(sorted(classes["false_negative_edges"]))


def exact_budget_false_positive_removal(
    *,
    predicted: np.ndarray,
    true: np.ndarray,
    candidate_edges: Sequence[tuple[int, int]],
    budget_k: int,
) -> dict[str, Any]:
    """Exhaustive subset search over FP candidates up to ``budget_k``.

    The empty subset is always included, so the selected result is
    never worse than the original under the selection rule. The
    selection rule is SID-primary:

    - lowest ``sid_after``;
    - then lowest ``shd_after``;
    - then fewest selected edges;
    - then lexicographically smallest sorted-tuple representation.

    This is a structural full-correction diagnostic; it is not a
    guaranteed SID ceiling for k-budget priors because the
    relationship between forbidden-edge prior strength and the
    optimisation landscape is not modelled.
    """
    if int(budget_k) < 0:
        raise ValueError(
            f"budget_k must be non-negative; got {budget_k!r}."
        )
    candidates = tuple(
        (int(i), int(j)) for (i, j) in candidate_edges
    )
    n_cand = len(candidates)
    max_r = min(int(budget_k), n_cand)
    base = compute_sid_shd(predicted, true)
    best_key: tuple[Any, ...] = (
        int(base["sid"]),
        int(base["shd"]),
        0,
        tuple(),
    )
    best_subset: tuple[tuple[int, int], ...] = tuple()
    best_after_sid: int = int(base["sid"])
    best_after_shd: int = int(base["shd"])
    for r in range(0, max_r + 1):
        if r == 0:
            # The empty subset is already the initial best_key.
            continue
        for subset in itertools.combinations(candidates, r):
            edited = remove_edges(predicted, subset)
            metrics = compute_sid_shd(edited, true)
            key = (
                int(metrics["sid"]),
                int(metrics["shd"]),
                len(subset),
                tuple(sorted(subset)),
            )
            if key < best_key:
                best_key = key
                best_subset = tuple(sorted(subset))
                best_after_sid = int(metrics["sid"])
                best_after_shd = int(metrics["shd"])
    return {
        "selected_edges": list(best_subset),
        "sid_original": int(base["sid"]),
        "sid_after": int(best_after_sid),
        "sid_delta": int(best_after_sid - int(base["sid"])),
        "shd_original": int(base["shd"]),
        "shd_after": int(best_after_shd),
        "shd_delta": int(best_after_shd - int(base["shd"])),
        "n_candidate_edges": int(n_cand),
        "n_selected_edges": int(len(best_subset)),
        "search_strategy": STRATEGY_EXACT_SUBSET_SID_PRIMARY,
    }


def full_false_positive_removal(
    *,
    predicted: np.ndarray,
    true: np.ndarray,
    candidate_edges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Remove all FP candidates and recompute SID/SHD.

    This is a structural full-correction diagnostic. It is not a
    guaranteed SID ceiling: the relationship between the chosen
    forbidden-edge prior class and DAGMA's optimisation is not
    modelled.
    """
    candidates = tuple(
        (int(i), int(j)) for (i, j) in candidate_edges
    )
    base = compute_sid_shd(predicted, true)
    edited = remove_edges(predicted, candidates)
    metrics = compute_sid_shd(edited, true)
    return {
        "selected_edges": [
            (int(i), int(j)) for (i, j) in sorted(candidates)
        ],
        "sid_original": int(base["sid"]),
        "sid_after": int(metrics["sid"]),
        "sid_delta": int(int(metrics["sid"]) - int(base["sid"])),
        "shd_original": int(base["shd"]),
        "shd_after": int(metrics["shd"]),
        "shd_delta": int(int(metrics["shd"]) - int(base["shd"])),
        "n_candidate_edges": int(len(candidates)),
        "n_selected_edges": int(len(candidates)),
        "search_strategy": STRATEGY_REMOVE_ALL_FALSE_POSITIVES,
    }


# ---------------------------------------------------------------------------
# False-negative greedy diagnostics
# ---------------------------------------------------------------------------


def evaluate_single_edge_addition(
    *,
    current: np.ndarray,
    true: np.ndarray,
    edge: tuple[int, int],
) -> dict[str, Any]:
    """Evaluate adding one edge with the acyclicity guard.

    On cycle creation returns ``{"valid": False,
    "invalid_reason": "cycle", ...}`` with metric values set to
    ``None``. Otherwise returns the recomputed SID/SHD and the
    deltas relative to ``current``.
    """
    i, j = int(edge[0]), int(edge[1])
    n = int(np.asarray(current).shape[0])
    if not (0 <= i < n) or not (0 <= j < n):
        raise ValueError(
            f"edge {(i, j)!r} is out of range for shape "
            f"{np.asarray(current).shape}."
        )
    if i == j:
        return {
            "edge": (i, j),
            "valid": False,
            "invalid_reason": "cycle",
            "sid_after": None,
            "shd_after": None,
            "sid_delta": None,
            "shd_delta": None,
        }
    trial = np.asarray(current, dtype=bool).copy()
    if trial[i, j]:
        # Already present; cycle check trivially passes; deltas are 0.
        base = compute_sid_shd(current, true)
        return {
            "edge": (i, j),
            "valid": True,
            "invalid_reason": None,
            "sid_after": int(base["sid"]),
            "shd_after": int(base["shd"]),
            "sid_delta": 0,
            "shd_delta": 0,
        }
    trial[i, j] = True
    if not is_dag(trial):
        return {
            "edge": (i, j),
            "valid": False,
            "invalid_reason": "cycle",
            "sid_after": None,
            "shd_after": None,
            "sid_delta": None,
            "shd_delta": None,
        }
    base = compute_sid_shd(current, true)
    metrics = compute_sid_shd(trial, true)
    return {
        "edge": (i, j),
        "valid": True,
        "invalid_reason": None,
        "sid_after": int(metrics["sid"]),
        "shd_after": int(metrics["shd"]),
        "sid_delta": int(int(metrics["sid"]) - int(base["sid"])),
        "shd_delta": int(int(metrics["shd"]) - int(base["shd"])),
    }


def greedy_acyclic_false_negative_addition(
    *,
    predicted: np.ndarray,
    true: np.ndarray,
    candidate_edges: Sequence[tuple[int, int]],
    budget_k: Optional[int],
) -> dict[str, Any]:
    """Greedy SID-primary FN addition with acyclicity guard.

    At each step, every remaining candidate is evaluated from the
    current graph. Cycle-inducing candidates are filtered out and
    recorded in the cumulative ``skipped_cycle_edges`` list. Among
    valid candidates, the one with the lowest ``sid_after`` is
    selected (ties broken by lower ``shd_after`` then by
    lexicographic edge order). The selected candidate is accepted
    only if it strictly improves SID, or if SID is unchanged and SHD
    strictly improves. The procedure stops when no beneficial valid
    candidate remains, or when ``budget_k`` accepted additions have
    been made (if ``budget_k`` is not ``None``).
    """
    if budget_k is not None and int(budget_k) < 0:
        raise ValueError(
            f"budget_k must be non-negative or None; got {budget_k!r}."
        )
    base = compute_sid_shd(predicted, true)
    current = np.asarray(predicted, dtype=bool).copy()
    remaining: list[tuple[int, int]] = [
        (int(i), int(j)) for (i, j) in candidate_edges
    ]
    selected: list[tuple[int, int]] = []
    skipped_cycle: list[tuple[int, int]] = []
    cur_sid = int(base["sid"])
    cur_shd = int(base["shd"])
    while remaining and (
        budget_k is None or len(selected) < int(budget_k)
    ):
        evaluated: list[tuple[tuple[int, int], dict[str, Any]]] = []
        next_remaining: list[tuple[int, int]] = []
        for edge in remaining:
            res = evaluate_single_edge_addition(
                current=current, true=true, edge=edge,
            )
            if res["valid"]:
                evaluated.append((edge, res))
                next_remaining.append(edge)
            else:
                if edge not in skipped_cycle:
                    skipped_cycle.append(edge)
        if not evaluated:
            break
        evaluated.sort(key=lambda t: (
            int(t[1]["sid_after"]),
            int(t[1]["shd_after"]),
            (int(t[0][0]), int(t[0][1])),
        ))
        best_edge, best_res = evaluated[0]
        sid_after = int(best_res["sid_after"])
        shd_after = int(best_res["shd_after"])
        improves_sid = sid_after < cur_sid
        sid_same_shd_better = (
            sid_after == cur_sid and shd_after < cur_shd
        )
        if not (improves_sid or sid_same_shd_better):
            break
        i, j = int(best_edge[0]), int(best_edge[1])
        current[i, j] = True
        selected.append((i, j))
        cur_sid = sid_after
        cur_shd = shd_after
        # Refresh the candidate list: keep only edges that were valid
        # in the current iteration and have not been accepted.
        remaining = [e for e in next_remaining if e != best_edge]
    strategy = (
        STRATEGY_GREEDY_BUDGET10
        if budget_k is not None and int(budget_k) == ORACLE_BUDGET_K
        else STRATEGY_GREEDY_FULL
    )
    return {
        "selected_edges": list(selected),
        "skipped_cycle_edges": list(skipped_cycle),
        "sid_original": int(base["sid"]),
        "sid_after": int(cur_sid),
        "sid_delta": int(cur_sid - int(base["sid"])),
        "shd_original": int(base["shd"]),
        "shd_after": int(cur_shd),
        "shd_delta": int(cur_shd - int(base["shd"])),
        "n_candidate_edges": int(len(candidate_edges)),
        "n_selected_edges": int(len(selected)),
        "n_skipped_cycle_edges": int(len(skipped_cycle)),
        "search_strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Actual reference forbidden removal (scenario A)
# ---------------------------------------------------------------------------


def actual_reference_forbidden_removal(
    *,
    predicted: np.ndarray,
    true: np.ndarray,
    reference_forbidden_edges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Remove the clean reference forbidden-edge set and recompute SID/SHD.

    Reproduces the offline removal diagnostic from the prior
    structural relevance analysis using the same metric API.
    """
    base = compute_sid_shd(predicted, true)
    edited = remove_edges(predicted, reference_forbidden_edges)
    metrics = compute_sid_shd(edited, true)
    return {
        "selected_edges": [
            (int(i), int(j))
            for (i, j) in sorted(
                (int(a), int(b)) for (a, b) in reference_forbidden_edges
            )
        ],
        "sid_original": int(base["sid"]),
        "sid_after": int(metrics["sid"]),
        "sid_delta": int(int(metrics["sid"]) - int(base["sid"])),
        "shd_original": int(base["shd"]),
        "shd_after": int(metrics["shd"]),
        "shd_delta": int(int(metrics["shd"]) - int(base["shd"])),
        "n_candidate_edges": int(len(reference_forbidden_edges)),
        "n_selected_edges": int(len(reference_forbidden_edges)),
        "search_strategy": STRATEGY_REMOVE_REFERENCE_FORBIDDEN,
    }


# ---------------------------------------------------------------------------
# Per-seed orchestration
# ---------------------------------------------------------------------------


def _row_for_scenario(
    *,
    seed_value: int,
    scenario_label: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Pack a diagnostic result into the per-seed CSV row format."""
    skipped = result.get("skipped_cycle_edges", [])
    return {
        "seed_value": int(seed_value),
        "scenario_label": str(scenario_label),
        "search_strategy": str(result.get("search_strategy", "")),
        "n_candidate_edges": int(result.get("n_candidate_edges", 0)),
        "n_selected_edges": int(result.get("n_selected_edges", 0)),
        "n_skipped_cycle_edges": int(
            result.get("n_skipped_cycle_edges", len(skipped))
        ),
        "sid_original": int(result["sid_original"]),
        "sid_after": int(result["sid_after"]),
        "sid_delta": int(result["sid_delta"]),
        "shd_original": int(result["shd_original"]),
        "shd_after": int(result["shd_after"]),
        "shd_delta": int(result["shd_delta"]),
        "selected_edges_json": json.dumps(
            [[int(i), int(j)] for (i, j) in result.get(
                "selected_edges", []
            )],
            separators=(",", ":"),
        ),
        "skipped_cycle_edges_json": json.dumps(
            [[int(i), int(j)] for (i, j) in skipped],
            separators=(",", ":"),
        ),
    }


def compute_oracle_diagnostics_for_seed(
    *,
    prior_free_record: MainStudyRunRecord,
    clean_soft_reference_record: MainStudyRunRecord,
    base_dir: Path,
    budget_k: int = ORACLE_BUDGET_K,
) -> tuple[dict[str, Any], ...]:
    """Compute the five diagnostic scenarios for one seed."""
    if (
        int(prior_free_record.config.seed_value)
        != int(clean_soft_reference_record.config.seed_value)
    ):
        raise ValueError(
            "seed mismatch between prior_free record "
            f"({prior_free_record.config.seed_value!r}) and clean-soft "
            f"reference ({clean_soft_reference_record.config.seed_value!r})."
        )
    seed_value = int(prior_free_record.config.seed_value)
    predicted = load_thresholded_adjacency(prior_free_record, base_dir)
    true_adj = load_true_adjacency(prior_free_record, base_dir)
    cps = clean_soft_reference_record.config.corrupted_prior_spec
    if cps is None or not cps.forbidden_edges:
        raise ValueError(
            f"clean-soft reference at seed {seed_value!r} has no "
            "forbidden_edges."
        )
    reference_forbidden = tuple(
        (int(i), int(j)) for (i, j) in cps.forbidden_edges
    )
    fp_candidates = _candidate_false_positive_edges(predicted, true_adj)
    fn_candidates = _candidate_false_negative_edges(predicted, true_adj)

    actual = actual_reference_forbidden_removal(
        predicted=predicted, true=true_adj,
        reference_forbidden_edges=reference_forbidden,
    )
    fp_budget = exact_budget_false_positive_removal(
        predicted=predicted, true=true_adj,
        candidate_edges=fp_candidates, budget_k=int(budget_k),
    )
    fp_all = full_false_positive_removal(
        predicted=predicted, true=true_adj,
        candidate_edges=fp_candidates,
    )
    fn_budget = greedy_acyclic_false_negative_addition(
        predicted=predicted, true=true_adj,
        candidate_edges=fn_candidates, budget_k=int(budget_k),
    )
    fn_full = greedy_acyclic_false_negative_addition(
        predicted=predicted, true=true_adj,
        candidate_edges=fn_candidates, budget_k=None,
    )
    return (
        _row_for_scenario(
            seed_value=seed_value,
            scenario_label=SCENARIO_ACTUAL_REFERENCE,
            result=actual,
        ),
        _row_for_scenario(
            seed_value=seed_value,
            scenario_label=SCENARIO_FP_BUDGET_EXACT,
            result=fp_budget,
        ),
        _row_for_scenario(
            seed_value=seed_value,
            scenario_label=SCENARIO_FP_REMOVE_ALL,
            result=fp_all,
        ),
        _row_for_scenario(
            seed_value=seed_value,
            scenario_label=SCENARIO_FN_BUDGET_GREEDY,
            result=fn_budget,
        ),
        _row_for_scenario(
            seed_value=seed_value,
            scenario_label=SCENARIO_FN_FULL_GREEDY,
            result=fn_full,
        ),
    )


def compute_all_oracle_diagnostics(
    *,
    prior_free_records: Sequence[MainStudyRunRecord],
    clean_soft_reference_records: dict[int, MainStudyRunRecord],
    base_dir: Path,
    budget_k: int = ORACLE_BUDGET_K,
) -> tuple[dict[str, Any], ...]:
    """Run the five-scenario diagnostic across all seven evaluation seeds."""
    rows: list[dict[str, Any]] = []
    for rec in prior_free_records:
        seed = int(rec.config.seed_value)
        ref = clean_soft_reference_records.get(int(seed))
        if ref is None:
            raise ValueError(
                f"no clean-soft reference record for seed {seed!r}."
            )
        rows.extend(
            compute_oracle_diagnostics_for_seed(
                prior_free_record=rec,
                clean_soft_reference_record=ref,
                base_dir=base_dir,
                budget_k=int(budget_k),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[int]) -> dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None, "median": None, "min": None, "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarise_oracle_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Group rows by scenario_label and compute descriptive statistics."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r["scenario_label"]), []).append(r)
    out: list[dict[str, Any]] = []
    for scenario in SCENARIO_LABELS:
        members = grouped.get(scenario, [])
        sid_orig = [int(m["sid_original"]) for m in members]
        sid_after = [int(m["sid_after"]) for m in members]
        sid_delta = [int(m["sid_delta"]) for m in members]
        shd_orig = [int(m["shd_original"]) for m in members]
        shd_after = [int(m["shd_after"]) for m in members]
        shd_delta = [int(m["shd_delta"]) for m in members]
        cand = [int(m["n_candidate_edges"]) for m in members]
        sel = [int(m["n_selected_edges"]) for m in members]
        skipped = [int(m["n_skipped_cycle_edges"]) for m in members]
        sid_orig_s = _stats(sid_orig)
        sid_after_s = _stats(sid_after)
        sid_delta_s = _stats(sid_delta)
        shd_orig_s = _stats(shd_orig)
        shd_after_s = _stats(shd_after)
        shd_delta_s = _stats(shd_delta)
        cand_s = _stats(cand)
        sel_s = _stats(sel)
        skipped_s = _stats(skipped)
        out.append({
            "scenario_label": scenario,
            "n_seeds": int(len(members)),
            "mean_sid_original": sid_orig_s["mean"],
            "mean_sid_after": sid_after_s["mean"],
            "mean_sid_delta": sid_delta_s["mean"],
            "median_sid_delta": sid_delta_s["median"],
            "min_sid_delta": sid_delta_s["min"],
            "max_sid_delta": sid_delta_s["max"],
            "mean_shd_original": shd_orig_s["mean"],
            "mean_shd_after": shd_after_s["mean"],
            "mean_shd_delta": shd_delta_s["mean"],
            "median_shd_delta": shd_delta_s["median"],
            "min_shd_delta": shd_delta_s["min"],
            "max_shd_delta": shd_delta_s["max"],
            "mean_n_candidate_edges": cand_s["mean"],
            "mean_n_selected_edges": sel_s["mean"],
            "mean_n_skipped_cycle_edges": skipped_s["mean"],
        })
    return tuple(out)


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------


def compute_analysis_hash12(
    *,
    main_evaluation_run_hash12: str,
    prior_relevance_analysis_hash12: str,
    budget_k: int,
    sorted_input_run_ids: Iterable[str],
    sorted_input_configuration_hashes: Iterable[str],
    analysis_protocol_version: str = ANALYSIS_PROTOCOL_VERSION,
) -> tuple[str, dict[str, Any]]:
    """Deterministic 12-char hex hash and the exact payload used."""
    payload = {
        "main_evaluation_run_hash12": str(main_evaluation_run_hash12),
        "analysis_protocol_version": str(analysis_protocol_version),
        "input_run_ids_sorted": sorted(
            str(x) for x in sorted_input_run_ids
        ),
        "input_configuration_hashes_sorted": sorted(
            str(x) for x in sorted_input_configuration_hashes
        ),
        "prior_relevance_analysis_hash12": str(
            prior_relevance_analysis_hash12
        ),
        "budget_k": int(budget_k),
    }
    serialised = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return digest[:12], payload


def write_manifest_json(
    manifest: dict[str, Any], path: Path
) -> None:
    """Write the analysis manifest as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


_PER_SEED_COLUMNS: tuple[str, ...] = (
    "seed_value", "scenario_label", "search_strategy",
    "n_candidate_edges", "n_selected_edges", "n_skipped_cycle_edges",
    "sid_original", "sid_after", "sid_delta",
    "shd_original", "shd_after", "shd_delta",
    "selected_edges_json", "skipped_cycle_edges_json",
)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario_label", "n_seeds",
    "mean_sid_original", "mean_sid_after",
    "mean_sid_delta", "median_sid_delta",
    "min_sid_delta", "max_sid_delta",
    "mean_shd_original", "mean_shd_after",
    "mean_shd_delta", "median_shd_delta",
    "min_shd_delta", "max_shd_delta",
    "mean_n_candidate_edges", "mean_n_selected_edges",
    "mean_n_skipped_cycle_edges",
)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.4g}"
    return str(value)


def write_oracle_readout(
    *,
    rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    output_path: Path,
    main_evaluation_run_hash12: str,
    analysis_hash12: str,
    prior_relevance_analysis_hash12: str,
    budget_k: int,
    output_dir_relative: str,
    plot_status: str,
) -> Path:
    """Write the cautious labelling-only oracle-diagnostic readout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(
        "# Oracle prior relevance: exploratory diagnostic"
    )
    lines.append("")
    lines.append("## Run identity")
    lines.append("")
    lines.append(
        f"- `main_evaluation_run_hash12`: `{main_evaluation_run_hash12}`"
    )
    lines.append(f"- `analysis_hash12`: `{analysis_hash12}`")
    lines.append(
        "- prior structural relevance analysis: "
        f"`{prior_relevance_analysis_hash12}`"
    )
    lines.append(
        f"- analysis protocol version: `{ANALYSIS_PROTOCOL_VERSION}`"
    )
    lines.append(f"- output directory: `{output_dir_relative}`")
    lines.append("")
    lines.append(
        "Existing saved artefacts only were used. No new model "
        "fitting, no MMD recomputation, and no new interventional "
        "sampling were performed. This is an offline structural "
        "diagnostic. This is the final scheduled exploratory "
        "diagnostic before thesis writing."
    )
    lines.append("")
    lines.append("## Evidence files used")
    lines.append("")
    lines.append(
        "- 7 prior_free records loaded from "
        f"`results/main_study/{main_evaluation_run_hash12}/records/`."
    )
    lines.append(
        "- 7 clean-soft reference records "
        "(soft_frobenius, corruption=0.0, confidence=1.0) loaded "
        "from the same directory."
    )
    lines.append(
        "- For each prior_free record, the persisted "
        "`thresholded_adjacency.npz` and `true_adjacency.npz` "
        "artefacts were read; the continuous-W artefact is not used "
        "by this diagnostic."
    )
    lines.append("")
    lines.append("## Diagnostic scenarios")
    lines.append("")
    lines.append(
        "Five per-seed scenarios are computed for each of the seven "
        "evaluation seeds:"
    )
    lines.append(
        f"- `{SCENARIO_ACTUAL_REFERENCE}`: remove the seed-specific "
        "clean reference forbidden-edge set."
    )
    lines.append(
        f"- `{SCENARIO_FP_BUDGET_EXACT}`: exact exhaustive subset "
        "search over prior-free false positives, up to "
        f"`budget_k = {int(budget_k)}` removed edges. SID-primary "
        "selection with deterministic tie-breaks."
    )
    lines.append(
        f"- `{SCENARIO_FP_REMOVE_ALL}`: remove every prior-free false "
        "positive. Structural full-correction diagnostic; not "
        "claimed as a guaranteed SID ceiling."
    )
    lines.append(
        f"- `{SCENARIO_FN_BUDGET_GREEDY}`: greedy SID-primary "
        f"addition of up to `budget_k = {int(budget_k)}` prior-free "
        "false negatives, guarded by acyclicity. Greedy "
        "approximation, not a global optimum."
    )
    lines.append(
        f"- `{SCENARIO_FN_FULL_GREEDY}`: same greedy procedure "
        "without the `budget_k` cap. Continues until no beneficial "
        "valid addition remains."
    )
    lines.append("")
    lines.append("## Budget convention")
    lines.append("")
    lines.append(
        f"`budget_k = {int(budget_k)}` matches the original "
        "forbidden-edge prior budget (10 edges per seed in the main "
        "evaluation). The comparison is "
        "\"what could a 10-edge prior budget have achieved across "
        "different prior classes?\". `budget_k = 10` is not claimed "
        "to be optimal for any required-edge prior."
    )
    lines.append("")

    # Summary table.
    lines.append(
        "## Aggregate summary (mean / median / min / max) by scenario"
    )
    lines.append("")
    lines.append(
        "| scenario | n | mean dSID | median dSID | min dSID | max dSID | "
        "mean dSHD | median dSHD | min dSHD | max dSHD | "
        "mean n_cand | mean n_selected | mean n_skipped_cycle |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |"
    )
    for s in summary_rows:
        lines.append(
            "| `" + str(s["scenario_label"]) + "` | "
            + f"{int(s['n_seeds'])} | "
            + f"{_fmt(s['mean_sid_delta'])} | "
            + f"{_fmt(s['median_sid_delta'])} | "
            + f"{_fmt(s['min_sid_delta'])} | "
            + f"{_fmt(s['max_sid_delta'])} | "
            + f"{_fmt(s['mean_shd_delta'])} | "
            + f"{_fmt(s['median_shd_delta'])} | "
            + f"{_fmt(s['min_shd_delta'])} | "
            + f"{_fmt(s['max_shd_delta'])} | "
            + f"{_fmt(s['mean_n_candidate_edges'])} | "
            + f"{_fmt(s['mean_n_selected_edges'])} | "
            + f"{_fmt(s['mean_n_skipped_cycle_edges'])} |"
        )
    lines.append("")

    def _rows_for(scenario: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["scenario_label"] == scenario]

    def _per_seed_table(rs: list[dict[str, Any]]) -> list[str]:
        out_lines = [
            "| seed | SID_orig | SID_after | dSID | SHD_orig | "
            "SHD_after | dSHD | n_cand | n_selected | n_skipped_cycle |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rs:
            out_lines.append(
                f"| {int(r['seed_value'])} | "
                f"{int(r['sid_original'])} | "
                f"{int(r['sid_after'])} | "
                f"{int(r['sid_delta'])} | "
                f"{int(r['shd_original'])} | "
                f"{int(r['shd_after'])} | "
                f"{int(r['shd_delta'])} | "
                f"{int(r['n_candidate_edges'])} | "
                f"{int(r['n_selected_edges'])} | "
                f"{int(r['n_skipped_cycle_edges'])} |"
            )
        return out_lines

    lines.append("## Actual reference-forbidden removal")
    lines.append("")
    lines.append(
        "Removes the seed-specific clean reference forbidden-edge "
        "set from the prior-free thresholded adjacency and "
        "recomputes SID and SHD. Reproduces the prior structural "
        "relevance offline-removal diagnostic; deltas should match "
        "that earlier output within numerical precision."
    )
    lines.append("")
    lines.extend(_per_seed_table(_rows_for(SCENARIO_ACTUAL_REFERENCE)))
    lines.append("")

    lines.append("## Exact budget-matched false-positive diagnostic")
    lines.append("")
    lines.append(
        f"Exhaustive subset search over prior-free false positives "
        f"up to `budget_k = {int(budget_k)}`. Selection rule is "
        "SID-primary with deterministic tie-breaks. The empty "
        "subset is included; the selected result cannot be worse "
        "than the original under the selection rule."
    )
    lines.append("")
    lines.extend(_per_seed_table(_rows_for(SCENARIO_FP_BUDGET_EXACT)))
    lines.append("")

    lines.append("## Full false-positive removal diagnostic")
    lines.append("")
    lines.append(
        "Removes every prior-free false positive and recomputes "
        "SID and SHD. This is a structural full-correction "
        "diagnostic; it is not a guaranteed SID ceiling."
    )
    lines.append("")
    lines.extend(_per_seed_table(_rows_for(SCENARIO_FP_REMOVE_ALL)))
    lines.append("")

    lines.append("## Greedy acyclicity-guarded false-negative diagnostic")
    lines.append("")
    lines.append(
        f"Budget-matched variant with `budget_k = {int(budget_k)}`:"
    )
    lines.append("")
    lines.extend(_per_seed_table(_rows_for(SCENARIO_FN_BUDGET_GREEDY)))
    lines.append("")
    lines.append("Full-candidate variant (`budget_k = None`):")
    lines.append("")
    lines.extend(_per_seed_table(_rows_for(SCENARIO_FN_FULL_GREEDY)))
    lines.append("")
    lines.append(
        "These are greedy diagnostic approximations. Subset "
        "interactions and acyclicity constraints mean the result is "
        "not a guaranteed global optimum."
    )
    lines.append("")

    lines.append("## Acyclicity guard summary")
    lines.append("")
    lines.append(
        "Per-seed cycle-skip counts encountered during the greedy "
        "false-negative additions:"
    )
    lines.append("")
    fn_budget_rows = _rows_for(SCENARIO_FN_BUDGET_GREEDY)
    fn_full_rows = _rows_for(SCENARIO_FN_FULL_GREEDY)
    lines.append(
        "| seed | budget-k skipped | full-candidate skipped |"
    )
    lines.append("| --- | --- | --- |")
    for r_budget, r_full in zip(fn_budget_rows, fn_full_rows):
        lines.append(
            f"| {int(r_budget['seed_value'])} | "
            f"{int(r_budget['n_skipped_cycle_edges'])} | "
            f"{int(r_full['n_skipped_cycle_edges'])} |"
        )
    lines.append("")

    lines.append("## Comparison to original prior-target removal")
    lines.append("")
    lines.append(
        "The `actual_reference_forbidden_removal` rows reproduce the "
        "prior structural relevance offline removal output. Any "
        "byte-for-byte agreement on SID and SHD deltas is a "
        "consistency check, not new evidence."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The exact false-positive budget diagnostic is exhaustive "
        "over FP subsets up to "
        f"`budget_k = {int(budget_k)}`; it does not model the "
        "optimisation-side relationship between forbidden-edge "
        "prior strength and DAGMA's learned graph."
    )
    lines.append(
        "- Removing all false positives is a full structural "
        "correction; it is not a guaranteed SID ceiling for any "
        "prior class."
    )
    lines.append(
        "- The false-negative diagnostics are greedy "
        "acyclicity-guarded approximations. They are not global "
        "optima."
    )
    lines.append(
        "- MMD counterfactuals are explicitly out of scope; saved "
        "MMD values are not modified."
    )
    lines.append("")

    lines.append("## Implication for thesis discussion")
    lines.append("")
    lines.append(
        "The five scenarios characterise the maximum direct "
        "structural-metric improvement available under different "
        "offline prior-class proxies at the same 10-edge budget "
        "used in the frozen main evaluation. They are descriptive "
        "diagnostics intended to support cautious thesis "
        "discussion; they do not constitute a new headline "
        "comparison and do not replace the frozen primary result."
    )
    lines.append("")
    lines.append("## Stop condition")
    lines.append("")
    lines.append(
        "This is the final scheduled exploratory diagnostic before "
        "thesis writing. Any idea emerging from this analysis "
        "(required-edge prior implementation, lambda_prior tuning, "
        "new main study) is recorded as future work rather than "
        "implemented within the current project timeline."
    )
    lines.append("")
    lines.append(f"- oracle summary plot: {plot_status}.")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Optional summary plot
# ---------------------------------------------------------------------------


def make_oracle_summary_plot(
    *,
    summary_rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    """Optional bar chart of mean SID/SHD delta by scenario.

    Returns the output path on success, ``None`` on any failure.
    The plot uses cautious labelling only; no ranking language.
    """
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        scenarios = [s["scenario_label"] for s in summary_rows]
        mean_dsid = [
            float(s["mean_sid_delta"])
            if s["mean_sid_delta"] is not None else float("nan")
            for s in summary_rows
        ]
        mean_dshd = [
            float(s["mean_shd_delta"])
            if s["mean_shd_delta"] is not None else float("nan")
            for s in summary_rows
        ]
        fig, axes = plt.subplots(
            1, 2, figsize=(11.0, 4.4), constrained_layout=True
        )
        x_positions = np.arange(len(scenarios))
        colour = "#3F7DBE"
        axes[0].bar(
            x_positions, mean_dsid,
            color=colour, alpha=0.7,
            edgecolor=colour, linewidth=0.8,
        )
        axes[0].set_xticks(x_positions)
        axes[0].set_xticklabels(
            scenarios, rotation=22, ha="right", fontsize=8,
        )
        axes[0].axhline(0.0, color="#999999", linewidth=0.6)
        axes[0].set_ylabel("mean dSID across 7 seeds")
        axes[0].set_title("Mean dSID by scenario (descriptive)")
        axes[0].grid(axis="y", alpha=0.4)
        for spine in ("top", "right"):
            axes[0].spines[spine].set_visible(False)
        axes[1].bar(
            x_positions, mean_dshd,
            color="#E07B39", alpha=0.7,
            edgecolor="#E07B39", linewidth=0.8,
        )
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels(
            scenarios, rotation=22, ha="right", fontsize=8,
        )
        axes[1].axhline(0.0, color="#999999", linewidth=0.6)
        axes[1].set_ylabel("mean dSHD across 7 seeds")
        axes[1].set_title("Mean dSHD by scenario (descriptive)")
        axes[1].grid(axis="y", alpha=0.4)
        for spine in ("top", "right"):
            axes[1].spines[spine].set_visible(False)
        fig.suptitle(
            "Offline structural-counterfactual diagnostic "
            "(no ranking; n=7 seeds)",
            fontsize=11,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analysis_output_dir(
    output_root: Path, analysis_hash12: str
) -> Path:
    """Return the oracle-diagnostic output directory for the hash."""
    return (
        output_root
        / "results"
        / "main_study"
        / "exploratory"
        / "oracle_prior_relevance"
        / analysis_hash12
    )


def run_oracle_prior_relevance_analysis(
    *,
    output_root: Path,
    main_evaluation_run_hash12: str,
    prior_relevance_analysis_hash12: str = "6f660aaeef3d",
    budget_k: int = ORACLE_BUDGET_K,
) -> dict[str, Any]:
    """End-to-end oracle-diagnostic analysis. Returns the manifest dict."""
    if not isinstance(output_root, Path):
        raise TypeError(
            "output_root must be a pathlib.Path; got "
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
    if (
        not isinstance(prior_relevance_analysis_hash12, str)
        or len(prior_relevance_analysis_hash12) != 12
    ):
        raise ValueError(
            "prior_relevance_analysis_hash12 must be a 12-character "
            f"string; got {prior_relevance_analysis_hash12!r}."
        )

    prior_free_records = load_prior_free_records(
        output_root, main_evaluation_run_hash12,
    )
    clean_soft_records = load_clean_soft_reference_records(
        output_root, main_evaluation_run_hash12,
    )

    input_run_ids = [r.run_id for r in prior_free_records]
    input_configuration_hashes = [
        r.configuration_hash_full for r in prior_free_records
    ]
    analysis_hash12, hash_payload = compute_analysis_hash12(
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        prior_relevance_analysis_hash12=prior_relevance_analysis_hash12,
        budget_k=int(budget_k),
        sorted_input_run_ids=input_run_ids,
        sorted_input_configuration_hashes=input_configuration_hashes,
    )
    out_dir = analysis_output_dir(output_root, analysis_hash12)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_all_oracle_diagnostics(
        prior_free_records=prior_free_records,
        clean_soft_reference_records=clean_soft_records,
        base_dir=output_root,
        budget_k=int(budget_k),
    )
    summary_rows = summarise_oracle_diagnostics(rows)

    write_csv(rows, out_dir / ORACLE_PER_SEED_CSV, _PER_SEED_COLUMNS)
    write_csv(
        summary_rows, out_dir / ORACLE_SUMMARY_CSV, _SUMMARY_COLUMNS,
    )

    plot_path = make_oracle_summary_plot(
        summary_rows=summary_rows,
        output_path=out_dir / ORACLE_SUMMARY_PLOT_PNG,
    )
    plot_status = (
        f"generated at `{plot_path.name}`"
        if plot_path is not None else "skipped"
    )

    output_dir_relative = str(
        out_dir.relative_to(output_root)
    ).replace("\\", "/")

    write_oracle_readout(
        rows=rows,
        summary_rows=summary_rows,
        output_path=out_dir / ORACLE_READOUT_MD,
        main_evaluation_run_hash12=main_evaluation_run_hash12,
        analysis_hash12=analysis_hash12,
        prior_relevance_analysis_hash12=(
            prior_relevance_analysis_hash12
        ),
        budget_k=int(budget_k),
        output_dir_relative=output_dir_relative,
        plot_status=plot_status,
    )

    output_files: list[str] = [
        ORACLE_PER_SEED_CSV,
        ORACLE_SUMMARY_CSV,
        ORACLE_READOUT_MD,
        ORACLE_MANIFEST_JSON,
    ]
    if plot_path is not None:
        output_files.append(ORACLE_SUMMARY_PLOT_PNG)

    manifest: dict[str, Any] = {
        "main_evaluation_run_hash12": main_evaluation_run_hash12,
        "analysis_hash12": analysis_hash12,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "prior_relevance_analysis_hash12":
            prior_relevance_analysis_hash12,
        "budget_k": int(budget_k),
        "hash_payload": hash_payload,
        "output_files": sorted(output_files),
        "n_prior_free_records": int(len(prior_free_records)),
        "no_new_fits": True,
        "no_mmd_recomputation": True,
        "no_new_sampling": True,
        "no_protocol_changes": True,
        "final_scheduled_exploratory_diagnostic": True,
    }
    write_manifest_json(manifest, out_dir / ORACLE_MANIFEST_JSON)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle_prior_relevance",
        description=(
            "Offline alternative-prior relevance diagnostic. "
            "Read-only over existing records and artefacts; no model "
            "fitting, no MMD recomputation, no new sampling."
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
    parser.add_argument(
        "--prior-relevance-analysis-hash12",
        type=str, default="6f660aaeef3d",
        help=(
            "12-character prior structural relevance analysis hash. "
            "Used only as a provenance reference; this analysis "
            "does not read that directory."
        ),
    )
    parser.add_argument(
        "--budget-k", type=int, default=ORACLE_BUDGET_K,
        help=(
            "Edge-budget for the budget-matched diagnostics "
            "(default 10, matching the original prior budget)."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_oracle_prior_relevance_analysis(
            output_root=args.output_root,
            main_evaluation_run_hash12=args.main_evaluation_run_hash12,
            prior_relevance_analysis_hash12=(
                args.prior_relevance_analysis_hash12
            ),
            budget_k=int(args.budget_k),
        )
    except SystemExit:
        raise
    except BaseException as exc:
        sys.stderr.write(
            "oracle_prior_relevance: error: "
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
    "EVALUATION_SEED_VALUES",
    "ORACLE_BUDGET_K",
    "SCENARIO_ACTUAL_REFERENCE",
    "SCENARIO_FN_BUDGET_GREEDY",
    "SCENARIO_FN_FULL_GREEDY",
    "SCENARIO_FP_BUDGET_EXACT",
    "SCENARIO_FP_REMOVE_ALL",
    "SCENARIO_LABELS",
    "actual_reference_forbidden_removal",
    "add_edges_with_acyclicity_guard",
    "analysis_output_dir",
    "classify_edges",
    "compute_all_oracle_diagnostics",
    "compute_analysis_hash12",
    "compute_oracle_diagnostics_for_seed",
    "compute_sid_shd",
    "edge_count",
    "evaluate_single_edge_addition",
    "exact_budget_false_positive_removal",
    "full_false_positive_removal",
    "greedy_acyclic_false_negative_addition",
    "is_dag",
    "load_clean_soft_reference_records",
    "load_prior_free_records",
    "load_thresholded_adjacency",
    "load_true_adjacency",
    "main",
    "make_oracle_summary_plot",
    "remove_edges",
    "run_oracle_prior_relevance_analysis",
    "summarise_oracle_diagnostics",
    "write_csv",
    "write_manifest_json",
    "write_oracle_readout",
]
