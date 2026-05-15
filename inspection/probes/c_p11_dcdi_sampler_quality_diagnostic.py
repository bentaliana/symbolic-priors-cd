"""C-P11: DCDI sampler-quality diagnostic probe.

Runs the DCDI sampler-quality setup and records every datum needed to
interpret the result. Four MMD comparisons are computed:

- Wrapper-vs-truth MMD (paired against ground-truth batches).
- Correct vs wrong structure MMD ratio, where the wrong-structure variant
  deletes the strongest true downstream edge from the learned thresholded
  adjacency.
- MMD under the TRUE SCM adjacency, using DCDI's learned conditionals.
- MMD under the learned thresholded adjacency augmented with the strongest
  missing true edge (only when the augmented adjacency is acyclic).

The last two help separate "structure was wrong" from "conditionals were
wrong".

This probe is read-only with respect to project source and external
repositories. CPU only. No dependency is installed.
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from symbolic_priors_cd.data.interventions import Intervention, intervene  # noqa: E402
from symbolic_priors_cd.data.scm_generator import (  # noqa: E402
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.metrics.interventional import mmd_rbf_unbiased  # noqa: E402
from symbolic_priors_cd.wrappers._dcdi_sampling import sample_raw_units_dcdi  # noqa: E402
from symbolic_priors_cd.wrappers._dcdi_training import (  # noqa: E402
    DCDIConfig,
    run_dcdi_training_loop,
)
from symbolic_priors_cd.wrappers._dcdi_utils import make_dcdi_model  # noqa: E402
from symbolic_priors_cd.wrappers.dcdi import (  # noqa: E402
    _is_acyclic_adjacency,
    _predict_adjacency_at,
    classify_graph_status,
)
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform  # noqa: E402


# Frozen settings for this probe (fixture-specific; do not edit without
# rerunning the probe and updating any downstream records).
NUM_VARS = 3
EXPECTED_EDGES = 3
SCM_SEED = 0
TRAIN_SIZE = 5000
VAL_SIZE = 500
TRAIN_SEED = 1
VAL_SEED = 2
DCDI_SEED = 0
N_ITER = 30000
INTERVENTION_VALUE = 2.0
N_FLOOR = 5
N_WRAPPER = 5
BATCH_SIZE = 1000

GT_FLOOR_SEED_BASE = 1000
GT_PAIRED_SEED_BASE = 1100
GT_WRONG_SEED_BASE = 1200
WRAPPER_SEED_BASE = 2000
WRAPPER_WRONG_SEED_BASE = 2100
WRAPPER_TRUE_SEED_BASE = 2200
WRAPPER_AUGMENTED_SEED_BASE = 2300


def _draw_gt(scm, target, value, n_batches, batch_size, seed_base):
    sampler = intervene(scm, Intervention(target=target, value=value))
    return [
        sampler.sample(n_samples=batch_size, rng=seed_base + k)
        for k in range(n_batches)
    ]


def _draw_wrapper(model, a, target, raw_value, preprocessor, n_batches, batch_size, seed_base):
    return [
        sample_raw_units_dcdi(
            model, a, target, raw_value, batch_size,
            sample_seed=seed_base + k, preprocessor=preprocessor,
        )
        for k in range(n_batches)
    ]


def _median_pairwise(batches):
    mmds = []
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            mmds.append(mmd_rbf_unbiased(batches[i], batches[j]))
    return float(np.median(mmds))


def _median_paired(a_list, b_list):
    return float(np.median([mmd_rbf_unbiased(a, b) for a, b in zip(a_list, b_list)]))


def _descendants_of(adj, source):
    n = adj.shape[0]
    visited = {source}
    queue = [source]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if adj[u, v] and v not in visited:
                visited.add(v)
                queue.append(v)
    return visited - {source}


def main():
    print("=" * 72)
    print("C-P11: DCDI Sampler-Quality Diagnostic")
    print("=" * 72)

    # SCM
    scm = generate_linear_gaussian_scm(
        n_nodes=NUM_VARS, expected_edges=EXPECTED_EDGES, seed=SCM_SEED,
    )
    target = int(scm.topological_order[0])
    print()
    print(f"SCM: n_nodes={NUM_VARS} expected_edges={EXPECTED_EDGES} seed={SCM_SEED}")
    print(f"True adjacency:\n{scm.adjacency.astype(int)}")
    print(f"True weights:\n{np.round(scm.weights, 4)}")
    print(f"Topological order: {scm.topological_order}")
    print(f"Intervention: do(X_{target} = {INTERVENTION_VALUE})")

    # Data + preprocessing
    X_train_raw = sample_observational(scm, n_samples=TRAIN_SIZE, rng=TRAIN_SEED)
    X_val_raw = sample_observational(scm, n_samples=VAL_SIZE, rng=VAL_SEED)
    preprocessor = CentredOnlyTransform().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    print()
    print(f"Data: n_train={TRAIN_SIZE} (seed={TRAIN_SEED}) "
          f"n_val={VAL_SIZE} (seed={VAL_SEED})")
    print("Preprocessor: CentredOnlyTransform fitted on training data")

    # DCDI training
    torch.manual_seed(DCDI_SEED)
    np.random.seed(DCDI_SEED)
    model = make_dcdi_model(num_vars=NUM_VARS, num_layers=2, hid_dim=8)
    config = DCDIConfig()
    print()
    print("DCDI: make_dcdi_model(num_layers=2, hid_dim=8)")
    print(f"      DCDIConfig() paper defaults, seed={DCDI_SEED}, n_iter={N_ITER}")

    t0 = time.time()
    result = run_dcdi_training_loop(
        model, X_train, X_val,
        config=config, seed=DCDI_SEED, n_iter=N_ITER,
    )
    train_time = time.time() - t0
    print()
    print(f"Training time : {train_time:.1f}s")
    print(f"n_iterations  : {result.n_iterations}")
    print(f"final_h       : {result.final_h:.4e}")
    print(f"final_mu      : {result.final_mu:.4e}")
    print(f"final_gamma   : {result.final_gamma:.4e}")
    print(f"converged     : {result.converged}")

    # Thresholded adjacency
    a_thresh = _predict_adjacency_at(
        result.continuous_w_adj_pre_threshold, threshold=0.5,
    )
    graph_status, _ = classify_graph_status(a_thresh)
    w_adj_np = result.continuous_w_adj_pre_threshold.numpy()
    print()
    print(f"Learned continuous w_adj:\n{np.round(w_adj_np, 4)}")
    print(f"Thresholded adjacency at 0.5:\n{a_thresh.astype(int)}")
    print(f"graph_status: {graph_status}")
    if graph_status != "valid_dag":
        print()
        print("graph_status is not valid_dag; downstream MMD comparisons skipped.")
        return

    # Floor MMD
    gt_floor = _draw_gt(
        scm, target, INTERVENTION_VALUE,
        N_FLOOR, BATCH_SIZE, GT_FLOOR_SEED_BASE,
    )
    floor_mmd = _median_pairwise(gt_floor)
    print()
    print("Monte Carlo floor")
    print(f"  ground-truth pairwise (n_floor={N_FLOOR}, "
          f"seed_base={GT_FLOOR_SEED_BASE}, batch={BATCH_SIZE})")
    print(f"  floor_mmd = {floor_mmd:.6e}")
    print("  Note: the unbiased MMD estimator can be negative when both "
          "samples come from the same distribution.")

    # Wrapper-vs-truth (original MC-floor comparison)
    wrapper_paired = _draw_wrapper(
        model, a_thresh, target, INTERVENTION_VALUE, preprocessor,
        N_WRAPPER, BATCH_SIZE, WRAPPER_SEED_BASE,
    )
    gt_paired = _draw_gt(
        scm, target, INTERVENTION_VALUE,
        N_WRAPPER, BATCH_SIZE, GT_PAIRED_SEED_BASE,
    )
    wrapper_vs_truth = _median_paired(wrapper_paired, gt_paired)
    print()
    print("Wrapper-vs-truth (paired MC-floor comparison)")
    print(f"  wrapper seed_base={WRAPPER_SEED_BASE}, "
          f"gt seed_base={GT_PAIRED_SEED_BASE}")
    print(f"  wrapper_vs_truth_mmd = {wrapper_vs_truth:.6e}")

    # Correct vs wrong structure (using GT_WRONG batches as the shared reference)
    gt_wrong_paired = _draw_gt(
        scm, target, INTERVENTION_VALUE,
        N_WRAPPER, BATCH_SIZE, GT_WRONG_SEED_BASE,
    )
    correct_batches = _draw_wrapper(
        model, a_thresh, target, INTERVENTION_VALUE, preprocessor,
        N_WRAPPER, BATCH_SIZE, WRAPPER_SEED_BASE,
    )

    descendants = _descendants_of(scm.adjacency, target)
    candidates = []
    for i in range(NUM_VARS):
        for j in range(NUM_VARS):
            if (
                scm.adjacency[i, j]
                and j in descendants
                and a_thresh[i, j]
            ):
                candidates.append((i, j, float(abs(scm.weights[i, j]))))
    candidates.sort(key=lambda c: c[2], reverse=True)
    chosen = candidates[0] if candidates else None

    correct_mmd = None
    wrong_mmd = None
    if chosen is None:
        print()
        print("Wrong-structure comparison: no candidate true downstream edge "
              "present in learned a_thresh; skipped.")
    else:
        wrong_a_thresh = a_thresh.copy()
        wrong_a_thresh[chosen[0], chosen[1]] = False
        wrong_batches = _draw_wrapper(
            model, wrong_a_thresh, target, INTERVENTION_VALUE, preprocessor,
            N_WRAPPER, BATCH_SIZE, WRAPPER_WRONG_SEED_BASE,
        )
        correct_mmd = _median_paired(correct_batches, gt_wrong_paired)
        wrong_mmd = _median_paired(wrong_batches, gt_wrong_paired)
        ratio = wrong_mmd / correct_mmd if correct_mmd > 0 else float("inf")
        print()
        print("Correct vs wrong structure (paired against gt_wrong)")
        print(f"  deleted edge: {chosen[0]} -> {chosen[1]} "
              f"(true |weight| = {chosen[2]:.4f})")
        print(f"  correct_mmd = {correct_mmd:.6e}")
        print(f"  wrong_mmd   = {wrong_mmd:.6e}")
        print(f"  ratio wrong / correct = {ratio:.3f}")
        print("  Sampler-quality acceptance threshold: ratio >= 1.5")

    # Diagnostic A: MMD under the TRUE SCM adjacency
    true_adj = scm.adjacency.copy()
    true_status, _ = classify_graph_status(true_adj)
    print()
    print("Diagnostic A: MMD using DCDI conditionals under the TRUE adjacency")
    print(f"  true adjacency status: {true_status}")
    if true_status == "valid_dag":
        true_batches = _draw_wrapper(
            model, true_adj, target, INTERVENTION_VALUE, preprocessor,
            N_WRAPPER, BATCH_SIZE, WRAPPER_TRUE_SEED_BASE,
        )
        true_mmd = _median_paired(true_batches, gt_wrong_paired)
        print(f"  wrapper seed_base={WRAPPER_TRUE_SEED_BASE}, "
              f"paired GT seed_base={GT_WRONG_SEED_BASE}")
        print(f"  true_struct_mmd = {true_mmd:.6e}")
    else:
        true_mmd = None
        print("  skipped: project SCM adjacency is not classified as valid_dag.")

    # Diagnostic B: MMD under learned + strongest missing true edge (if acyclic)
    learned_set = {
        (i, j) for i in range(NUM_VARS) for j in range(NUM_VARS) if a_thresh[i, j]
    }
    true_set = {
        (i, j) for i in range(NUM_VARS) for j in range(NUM_VARS) if scm.adjacency[i, j]
    }
    missing_true_edges = true_set - learned_set
    print()
    print("Diagnostic B: MMD under learned + strongest missing true edge")
    if not missing_true_edges:
        print("  no missing true edges; skipped.")
        aug_mmd = None
        aug_a_thresh = None
        aug_status = None
        strongest = None
    else:
        weighted = [
            (i, j, float(abs(scm.weights[i, j]))) for (i, j) in missing_true_edges
        ]
        weighted.sort(key=lambda x: x[2], reverse=True)
        strongest = weighted[0]
        aug_a_thresh = a_thresh.copy()
        aug_a_thresh[strongest[0], strongest[1]] = True
        if not _is_acyclic_adjacency(aug_a_thresh):
            aug_mmd = None
            aug_status = "cyclic"
            print(f"  strongest missing edge {strongest[0]} -> {strongest[1]} "
                  f"(|weight| = {strongest[2]:.4f}) would create a cycle; skipped.")
        else:
            aug_status, _ = classify_graph_status(aug_a_thresh)
            if aug_status != "valid_dag":
                aug_mmd = None
                print(f"  augmented adjacency status is {aug_status}; skipped.")
            else:
                aug_batches = _draw_wrapper(
                    model, aug_a_thresh, target, INTERVENTION_VALUE, preprocessor,
                    N_WRAPPER, BATCH_SIZE, WRAPPER_AUGMENTED_SEED_BASE,
                )
                aug_mmd = _median_paired(aug_batches, gt_wrong_paired)
                print(f"  added edge: {strongest[0]} -> {strongest[1]} "
                      f"(|weight| = {strongest[2]:.4f})")
                print(f"  augmented adjacency:\n{aug_a_thresh.astype(int)}")
                print(f"  aug_status: {aug_status}")
                print(f"  wrapper seed_base={WRAPPER_AUGMENTED_SEED_BASE}, "
                      f"paired GT seed_base={GT_WRONG_SEED_BASE}")
                print(f"  augmented_mmd = {aug_mmd:.6e}")

    # Summary
    print()
    print("=" * 72)
    print("Summary table")
    print("=" * 72)
    print(f"  graph_status (learned at 0.5)      : {graph_status}")
    print(f"  Monte Carlo floor MMD              : {floor_mmd:+.6e}")
    print(f"  wrapper-vs-truth MMD               : {wrapper_vs_truth:+.6e}")
    if correct_mmd is not None:
        print(f"  correct-structure MMD              : {correct_mmd:+.6e}")
        print(f"  wrong-structure MMD                : {wrong_mmd:+.6e}")
        print(f"  wrong / correct ratio              : "
              f"{(wrong_mmd / correct_mmd):.3f}")
    if true_mmd is not None:
        print(f"  true-structure MMD (Diagnostic A)  : {true_mmd:+.6e}")
    if aug_mmd is not None:
        print(f"  augmented-structure MMD (Diag B)   : {aug_mmd:+.6e}")
    print()
    print("End of probe output.")


if __name__ == "__main__":
    main()
