"""DCDI training-budget pilot probe.

Measurement-only diagnostic on the real 10-node ER2 linear-Gaussian
selection-study SCM cell. Records DCDI stopping behaviour across
pilot-pool seeds so the project can choose a real-study
``num_train_iter`` ceiling under user review.

Read-only with respect to project source and external repositories.
CPU only. No dependency is installed. Does not run any Phase A/B
logic, does not consume calibration or held-out evaluation seeds,
and does not amend any planning document.

Two modes:

- ``--mode smoke`` exercises the pipeline quickly with a small
  iteration cap and a small observational sample. Use this to
  verify the code path before running the full pilot.
- ``--mode full`` runs at the user-specified iteration cap on the
  real selection-study cell. Do not run full mode unless
  authorised.

Both modes emit the same CSV columns in the same order.

Output:

- ``inspection/probes/output/c_p15_dcdi_training_budget_pilot.csv``
  (one row per (mode, seed) pair). The file is created if missing;
  rows are appended otherwise. Existing rows are not rewritten.
"""

import argparse
import csv
import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from symbolic_priors_cd.data.scm_generator import (  # noqa: E402
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.wrappers.preprocessing import (  # noqa: E402
    CentredOnlyTransform,
)


# Real 10-node ER2 pilot SCM cell.
SCM_N_NODES = 10
SCM_EXPECTED_EDGES = 20
SCM_NOISE_SCALE = 1.0
SCM_WEIGHT_RANGE = (0.5, 2.0)

# Patience-based DCDI training configuration. The project wrapper's
# augmented-Lagrangian loop consumes the first patience gate
# (acyclicity-and-validation patience countdown); it does not
# exercise a second-stage permanent-thresholding gate. The
# train_patience_post knob is therefore not part of this probe; the
# readout records its absence under "Field exposure".
DCDI_STOP_CRIT_WIN = 100
DCDI_TRAIN_PATIENCE = 5
DCDI_TRAIN_BATCH_SIZE = 64
DCDI_H_THRESHOLD = 1e-8
DCDI_LR = 1e-3

# 80/20 train/validation split of the observational sample.
TRAIN_FRACTION = 0.8

# Pilot seeds. These are pilot-only seeds drawn from the
# reproduction seed pool; calibration and held-out evaluation pools
# are not consulted.
PILOT_SEEDS_SMOKE = (101, 102)
PILOT_SEEDS_FULL = (101, 102, 103)

# Smoke-mode sizing: small enough to run in a minute or two on CPU
# while still exercising the 10-node code path.
SMOKE_N_TOTAL = 200
SMOKE_NUM_TRAIN_ITER_CAP = 200

# Full-mode sizing: matches the selection-study observational
# sample size and uses a high ceiling so the patience gate decides
# the actual stop iteration.
FULL_N_TOTAL = 1000
FULL_NUM_TRAIN_ITER_CAP_DEFAULT = 300_000

OUTPUT_DIR = _PROJECT_ROOT / "inspection" / "probes" / "output"
OUTPUT_FILE = OUTPUT_DIR / "c_p15_dcdi_training_budget_pilot.csv"

# Frozen output schema. Smoke and full modes emit the same columns
# in the same order.
CSV_COLUMNS = (
    "mode",
    "seed",
    "graph_seed",
    "validation_data_seed",
    "train_data_seed",
    "n_total",
    "n_fit_samples",
    "n_val_samples",
    "num_train_iter_cap",
    "final_iteration",
    "first_stop_iteration",
    "second_stop_iteration",
    "final_h",
    "final_gamma",
    "final_mu",
    "gamma_update_count",
    "mu_update_count",
    "last_gamma_update_iteration",
    "last_mu_update_iteration",
    "graph_status",
    "sampler_status",
    "training_status",
    "runtime_seconds",
    "validation_nll_trajectory_summary",
)


def _build_scm(graph_seed: int):
    """Build the 10-node ER2 linear-Gaussian SCM at the given seed."""
    return generate_linear_gaussian_scm(
        n_nodes=SCM_N_NODES,
        expected_edges=SCM_EXPECTED_EDGES,
        seed=int(graph_seed),
        noise_scale=SCM_NOISE_SCALE,
        weight_magnitude_range=SCM_WEIGHT_RANGE,
    )


def _split_train_val(
    x_raw: np.ndarray, rng_seed: int, train_fraction: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministically split observational data into train / val.

    Returns (X_train_raw, X_val_raw, validation_data_seed). The
    validation_data_seed is the seed passed to the permutation RNG;
    it is recorded in the output row so the split is reproducible.
    """
    rng = np.random.default_rng(rng_seed)
    n_total = x_raw.shape[0]
    perm = rng.permutation(n_total)
    n_train = int(round(train_fraction * n_total))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return x_raw[train_idx], x_raw[val_idx], int(rng_seed)


def _build_dcdi_config():
    """Build a DCDIConfig with patience and stop-window values fixed.

    Imports DCDIConfig lazily so that the probe does not pull in the
    DCDI source on module import; matches the lazy import used in
    the public DCDIWrapper.
    """
    from symbolic_priors_cd.wrappers._dcdi_training import DCDIConfig

    return DCDIConfig(
        h_threshold=DCDI_H_THRESHOLD,
        lr=DCDI_LR,
        train_batch_size=DCDI_TRAIN_BATCH_SIZE,
        train_patience=DCDI_TRAIN_PATIENCE,
        stop_crit_win=DCDI_STOP_CRIT_WIN,
    )


# Fixed large-prime offsets used to decorrelate the graph,
# train-data, and validation-split seeds derived from a single
# pilot integer. They have no scientific meaning.
_TRAIN_DATA_SEED_OFFSET = 7919
_VALIDATION_DATA_SEED_OFFSET = 4111

# Number of trailing validation-NLL values included in the row
# summary. Kept small so the CSV cell remains human-readable.
_VAL_NLL_TAIL_LENGTH = 5


def _summarise_validation_nll(convergence_info) -> str:
    """Return a compact semicolon-delimited summary of validation NLLs.

    Reads ``validation_nll_history`` and ``validation_nll_stop_crit_win``
    from the wrapper's convergence-info diagnostics. Non-finite
    values (NaN or +/-inf) are counted separately as
    ``nonfinite_count``; the ``min`` and ``argmin`` fields are
    computed over finite values only and report ``not_exposed``
    when no finite value exists. The summary returns
    ``not_exposed`` when the history is absent or empty so the
    schema stays stable across wrapper versions.
    """
    import math

    if not isinstance(convergence_info, dict):
        return "not_exposed"
    history = convergence_info.get("validation_nll_history")
    cadence = convergence_info.get("validation_nll_stop_crit_win")
    if not isinstance(history, list) or not history:
        return "not_exposed"
    numeric_history = [
        float(v) for v in history if isinstance(v, (int, float))
        and not isinstance(v, bool)
    ]
    if len(numeric_history) != len(history):
        return "not_exposed"
    count = len(numeric_history)
    finite_indices = [
        i for i, v in enumerate(numeric_history) if math.isfinite(v)
    ]
    nonfinite_count = count - len(finite_indices)
    first = numeric_history[0]
    last = numeric_history[-1]
    if finite_indices:
        argmin_index = min(finite_indices, key=lambda i: numeric_history[i])
        minimum = numeric_history[argmin_index]
        min_str = f"{minimum:.6g}"
        argmin_str = str(argmin_index)
    else:
        min_str = "not_exposed"
        argmin_str = "not_exposed"
    tail = numeric_history[-_VAL_NLL_TAIL_LENGTH:]
    tail_str = "[" + ",".join(f"{v:.6g}" for v in tail) + "]"
    parts = [
        f"count={count}",
        f"nonfinite_count={nonfinite_count}",
        f"first={first:.6g}",
        f"last={last:.6g}",
        f"min={min_str}",
        f"argmin={argmin_str}",
        f"tail={tail_str}",
    ]
    if isinstance(cadence, int) and cadence > 0:
        parts.append(f"stop_crit_win={int(cadence)}")
    return ";".join(parts)


def _fit_one(
    *,
    seed: int,
    n_total: int,
    num_train_iter_cap: int,
) -> dict:
    """Fit DCDI on one pilot seed and return the measurement row."""
    from symbolic_priors_cd.wrappers.dcdi import DCDIWrapper

    graph_seed = int(seed)
    train_seed = int(seed) + _TRAIN_DATA_SEED_OFFSET
    validation_data_seed = int(seed) + _VALIDATION_DATA_SEED_OFFSET
    fit_seed = int(seed)

    scm = _build_scm(graph_seed)
    x_raw = sample_observational(scm, n_samples=int(n_total), rng=train_seed)
    x_train_raw, x_val_raw, val_seed_recorded = _split_train_val(
        x_raw, rng_seed=validation_data_seed, train_fraction=TRAIN_FRACTION
    )

    preprocessor = CentredOnlyTransform()
    preprocessor.fit(x_train_raw)
    x_train_model = preprocessor.transform(x_train_raw)
    x_val_model = preprocessor.transform(x_val_raw)

    config = _build_dcdi_config()
    wrapper = DCDIWrapper()

    t_start = time.perf_counter()
    wrapper.fit(
        x_train_model,
        X_val=x_val_model,
        preprocessor=preprocessor,
        seed=fit_seed,
        n_iter=int(num_train_iter_cap),
        config=config,
    )
    runtime_seconds = float(time.perf_counter() - t_start)

    diagnostics = wrapper.get_diagnostics()
    model_specific = diagnostics["model_specific_diagnostics"]
    convergence_info = diagnostics.get("convergence_info")
    training_status = str(diagnostics["training_status"])
    graph_status = str(diagnostics["graph_status"])
    sampler_status = str(diagnostics["sampler_status"])
    n_iterations = diagnostics.get("n_iterations")
    final_h = (
        model_specific.get("final_h")
        if isinstance(model_specific, dict)
        else None
    )
    first_stop = (
        model_specific.get("first_stop")
        if isinstance(model_specific, dict)
        else None
    )
    final_gamma = (
        model_specific.get("final_gamma")
        if isinstance(model_specific, dict)
        else None
    )
    final_mu = (
        model_specific.get("final_mu")
        if isinstance(model_specific, dict)
        else None
    )
    gamma_update_iters = (
        model_specific.get("gamma_update_iters")
        if isinstance(model_specific, dict)
        else None
    )
    mu_update_iters = (
        model_specific.get("mu_update_iters")
        if isinstance(model_specific, dict)
        else None
    )
    if not isinstance(gamma_update_iters, list):
        gamma_update_iters = []
    if not isinstance(mu_update_iters, list):
        mu_update_iters = []
    gamma_update_count = len(gamma_update_iters)
    mu_update_count = len(mu_update_iters)
    last_gamma_update_iteration = (
        int(gamma_update_iters[-1]) if gamma_update_iters else "not_exposed"
    )
    last_mu_update_iteration = (
        int(mu_update_iters[-1]) if mu_update_iters else "not_exposed"
    )
    validation_nll_summary = _summarise_validation_nll(convergence_info)

    return {
        "seed": int(fit_seed),
        "graph_seed": int(graph_seed),
        "validation_data_seed": int(val_seed_recorded),
        "train_data_seed": int(train_seed),
        "n_total": int(n_total),
        "n_fit_samples": int(x_train_raw.shape[0]),
        "n_val_samples": int(x_val_raw.shape[0]),
        "num_train_iter_cap": int(num_train_iter_cap),
        "final_iteration": (
            "not_exposed" if n_iterations is None else int(n_iterations)
        ),
        "first_stop_iteration": (
            "not_exposed" if first_stop is None else int(first_stop)
        ),
        "second_stop_iteration": "not_exposed",
        "final_h": (
            "not_exposed" if final_h is None else float(final_h)
        ),
        "final_gamma": (
            "not_exposed" if final_gamma is None else float(final_gamma)
        ),
        "final_mu": (
            "not_exposed" if final_mu is None else float(final_mu)
        ),
        "gamma_update_count": int(gamma_update_count),
        "mu_update_count": int(mu_update_count),
        "last_gamma_update_iteration": last_gamma_update_iteration,
        "last_mu_update_iteration": last_mu_update_iteration,
        "graph_status": graph_status,
        "sampler_status": sampler_status,
        "training_status": training_status,
        "runtime_seconds": float(runtime_seconds),
        "validation_nll_trajectory_summary": validation_nll_summary,
    }


def _row_with_mode(row: dict, mode: str) -> dict:
    """Prepend the mode field and return a fresh dict ordered for CSV."""
    full = {"mode": mode}
    full.update(row)
    return full


def _append_rows(rows: list[dict]) -> Path:
    """Append rows to the CSV file, creating it with header if absent.

    Refuses to append to a file whose header does not exactly match
    the current CSV_COLUMNS schema; raises ValueError instead.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.is_file():
        with open(OUTPUT_FILE, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
        if tuple(existing_header) != CSV_COLUMNS:
            raise ValueError(
                "existing CSV header does not match the current schema; "
                f"file={OUTPUT_FILE}, "
                f"expected={CSV_COLUMNS}, "
                f"found={tuple(existing_header)}. Delete or move the "
                "file before re-running the probe."
            )
    file_exists = OUTPUT_FILE.is_file()
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            ordered = {column: row.get(column, "") for column in CSV_COLUMNS}
            writer.writerow(ordered)
    return OUTPUT_FILE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DCDI training-budget pilot probe. Measurement only; does "
            "not freeze num_train_iter."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
        help=(
            "smoke: fast smoke check with a small iteration cap and a "
            "small observational sample. full: real selection-study "
            "cell with a high cap; runs longer."
        ),
    )
    parser.add_argument(
        "--full-num-train-iter-cap",
        type=int,
        default=FULL_NUM_TRAIN_ITER_CAP_DEFAULT,
        help=(
            "Iteration cap used in full mode. The patience gate "
            "decides the actual stop iteration when it fires below "
            "this cap. Default %(default)s."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated pilot seeds (override the "
            "default list). Must be drawn from the reproduction pool; "
            "calibration and held-out seeds must not appear."
        ),
    )
    return parser.parse_args()


def _resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds is None:
        return PILOT_SEEDS_SMOKE if args.mode == "smoke" else PILOT_SEEDS_FULL
    raw_tokens = [token.strip() for token in args.seeds.split(",")]
    parsed: list[int] = []
    for token in raw_tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--seeds entry {token!r} is not a valid integer"
            ) from exc
        if value <= 0:
            raise ValueError(
                f"--seeds entry {value} must be a positive integer"
            )
        parsed.append(value)
    if not parsed:
        raise ValueError(
            "--seeds was provided but resolved to an empty list"
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError(
            f"--seeds contains duplicate values: {parsed}"
        )
    return tuple(parsed)


def main() -> None:
    args = _parse_args()
    seeds = _resolve_seeds(args)

    if args.mode == "smoke":
        n_total = SMOKE_N_TOTAL
        num_train_iter_cap = SMOKE_NUM_TRAIN_ITER_CAP
    else:
        n_total = FULL_N_TOTAL
        num_train_iter_cap = int(args.full_num_train_iter_cap)

    sys.stdout.write(
        f"DCDI training-budget pilot: mode={args.mode} "
        f"seeds={seeds} n_total={n_total} "
        f"num_train_iter_cap={num_train_iter_cap}\n"
    )
    sys.stdout.flush()

    rows: list[dict] = []
    for seed in seeds:
        sys.stdout.write(f"  fitting seed={seed} ...\n")
        sys.stdout.flush()
        measurement = _fit_one(
            seed=int(seed),
            n_total=int(n_total),
            num_train_iter_cap=int(num_train_iter_cap),
        )
        rows.append(_row_with_mode(measurement, args.mode))
        sys.stdout.write(
            f"    final_iteration={measurement['final_iteration']} "
            f"first_stop={measurement['first_stop_iteration']} "
            f"final_h={measurement['final_h']} "
            f"graph_status={measurement['graph_status']} "
            f"sampler_status={measurement['sampler_status']} "
            f"training_status={measurement['training_status']} "
            f"runtime={measurement['runtime_seconds']:.2f}s\n"
        )
        sys.stdout.flush()

    output_path = _append_rows(rows)
    sys.stdout.write(f"wrote {len(rows)} row(s) to {output_path}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
