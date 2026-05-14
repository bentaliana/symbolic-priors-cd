"""C-P13: DAGMA sampler-quality diagnostic probe.

Reuses the C-P11 fixture for a controlled DAGMA-vs-DCDI same-fixture
comparison on the interventional-adequacy probe. Records every datum
needed to interpret the result and to localise any failure to learned
structure, coefficient quality, residual noise, invalid graph status,
or sampler mechanics.

Protocol replication:
- Aggregation rules mirror C-P11 exactly:
    floor:     median over all C(N_floor, 2) pairwise GT MMDs
    paired:    median over N_wrapper paired (wrapper_k, gt_k) MMDs
- Seed-base layout matches C-P11. Two new seed bases are added at 2300
  and 2400 for B1 (learned-weight augmentation) and B2 (oracle-weight
  augmentation), reflecting that DAGMA exposes an explicit weight per
  edge whereas DCDI used a single augmented diagnostic.
- The MMD function is mmd_rbf_unbiased with its default median-heuristic
  bandwidth, called exactly as in C-P11.


This probe is read-only with respect to project source and external
repositories. CPU only. No dependency is installed.
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from symbolic_priors_cd.data.interventions import Intervention, intervene  # noqa: E402
from symbolic_priors_cd.data.scm_generator import (  # noqa: E402
    generate_linear_gaussian_scm,
    sample_observational,
)
from symbolic_priors_cd.metrics.interventional import mmd_rbf_unbiased  # noqa: E402
from symbolic_priors_cd.metrics.structural import shd  # noqa: E402
from symbolic_priors_cd.wrappers._dagma_sampling import (  # noqa: E402
    estimate_residual_sigmas,
    sample_linear_gaussian_model_frame,
)
from symbolic_priors_cd.wrappers._graph_status import (  # noqa: E402
    classify_graph_status,
)
from symbolic_priors_cd.wrappers.dagma import (  # noqa: E402
    DAGMAConfig,
    DAGMAWrapper,
    _threshold_continuous_w,
)
from symbolic_priors_cd.wrappers.preprocessing import CentredOnlyTransform  # noqa: E402


# Frozen settings (mirror C-P11 exactly where applicable).
NUM_VARS = 3
EXPECTED_EDGES = 3
SCM_SEED = 0
TRAIN_SIZE = 5000
VAL_SIZE = 500
TRAIN_SEED = 1
VAL_SEED = 2
DAGMA_SEED = 0
INTERVENTION_VALUE = 2.0
N_FLOOR = 5
N_WRAPPER = 5
BATCH_SIZE = 1000
PROJECT_THRESHOLD = 0.3

# Seed bases: 1000s for ground truth, 2000s for wrapper paths.
GT_FLOOR_SEED_BASE = 1000
GT_PAIRED_SEED_BASE = 1100
GT_WRONG_SEED_BASE = 1200
WRAPPER_SEED_BASE = 2000
WRAPPER_WRONG_SEED_BASE = 2100
WRAPPER_TRUE_SEED_BASE = 2200
WRAPPER_LEARNED_AUG_SEED_BASE = 2300   # B1: learned continuous-W value
WRAPPER_ORACLE_AUG_SEED_BASE = 2400    # B2: oracle true SCM weight

# Primary acceptance thresholds (carried over from C-P11 unchanged).
PRIMARY_MMD_RATIO = 3.0     # wrapper_vs_truth <= 3 * floor
WRONG_STRUCTURE_RATIO = 1.5  # correct * 1.5 <= wrong


def _draw_gt(scm, target, value, n_batches, batch_size, seed_base):
    """Draw n_batches independent ground-truth interventional batches."""
    sampler = intervene(scm, Intervention(target=target, value=value))
    return [
        sampler.sample(n_samples=batch_size, rng=seed_base + k)
        for k in range(n_batches)
    ]


def _draw_wrapper_learned(wrapper, target, raw_value, n_batches, batch_size,
                          seed_base, noise_policy="residual_fitted"):
    """Draw n_batches wrapper batches via the public sample_interventional path."""
    interv = Intervention(target=target, value=raw_value)
    batches = []
    for k in range(n_batches):
        out = wrapper.sample_interventional(
            interv, n_samples=batch_size,
            sample_seed=seed_base + k, noise_policy=noise_policy,
        )
        if out is None:
            return None
        batches.append(out)
    return batches


def _draw_dagma_custom(X_model_frame, continuous_w_eff, a_eff, preprocessor,
                       target, raw_value, n_batches, batch_size, seed_base):
    """Draw raw-unit DAGMA samples under a custom (continuous_W, A) pair.

    Recomputes W_sample and residual sigma in model frame using the same
    estimator the wrapper uses, then drives the wrapper-side ancestral
    sampler directly. Returns None when the residual sigma is degenerate
    (non-finite or non-positive), mirroring the wrapper's no-floor policy.
    """
    w_sample, sigma = estimate_residual_sigmas(
        X_model_frame, continuous_w_eff, a_eff,
    )
    sigma_ok = bool(np.all(np.isfinite(sigma)) and np.all(sigma > 0))
    if not sigma_ok:
        return None, w_sample, sigma
    value_model = preprocessor.transform_intervention_value(raw_value, target)
    batches = []
    for k in range(n_batches):
        mf = sample_linear_gaussian_model_frame(
            a_eff, w_sample, sigma,
            target=target, value_model=value_model,
            n_samples=batch_size, sample_seed=seed_base + k,
        )
        batches.append(preprocessor.inverse_transform(mf))
    return batches, w_sample, sigma


def _median_pairwise(batches):
    """Median over all C(N, 2) pairwise MMDs across the batches list."""
    mmds = []
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            mmds.append(mmd_rbf_unbiased(batches[i], batches[j]))
    return float(np.median(mmds))


def _median_paired(a_list, b_list):
    """Median over zipped paired MMDs of (a_k, b_k)."""
    return float(np.median([
        mmd_rbf_unbiased(a, b) for a, b in zip(a_list, b_list)
    ]))


def _descendants_of(adj, source):
    """BFS descendants of the source node in the given boolean adjacency."""
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
    print("C-P13: DAGMA Sampler-Quality Diagnostic")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # SCM and ground truth
    # ----------------------------------------------------------------------
    scm = generate_linear_gaussian_scm(
        n_nodes=NUM_VARS, expected_edges=EXPECTED_EDGES, seed=SCM_SEED,
    )
    target = int(scm.topological_order[0])
    print()
    print(f"SCM: n_nodes={NUM_VARS} expected_edges={EXPECTED_EDGES} "
          f"seed={SCM_SEED}")
    print(f"True adjacency:\n{scm.adjacency.astype(int)}")
    print(f"True weights:\n{np.round(scm.weights, 4)}")
    print(f"Topological order: {scm.topological_order}")
    print(f"Intervention: do(X_{target} = {INTERVENTION_VALUE})")

    # ----------------------------------------------------------------------
    # Data and preprocessing
    # ----------------------------------------------------------------------
    X_train_raw = sample_observational(
        scm, n_samples=TRAIN_SIZE, rng=TRAIN_SEED,
    )
    _ = sample_observational(scm, n_samples=VAL_SIZE, rng=VAL_SEED)
    preprocessor = CentredOnlyTransform().fit(X_train_raw)
    X_train_mf = preprocessor.transform(X_train_raw)
    print()
    print(f"Data: n_train={TRAIN_SIZE} (seed={TRAIN_SEED}) "
          f"n_val={VAL_SIZE} (seed={VAL_SEED}, fitted-on rule mirrored)")
    print("Preprocessor: CentredOnlyTransform fitted on training data")

    # ----------------------------------------------------------------------
    # DAGMA fit through the wrapper
    # ----------------------------------------------------------------------
    print()
    print("DAGMA: DAGMAWrapper(DAGMAConfig() defaults), "
          "w_threshold_internal=0.0, project threshold 0.3 external")
    cfg = DAGMAConfig()
    wrapper = DAGMAWrapper()
    t0 = time.time()
    wrapper.fit(X_train_mf, preprocessor=preprocessor, seed=DAGMA_SEED, config=cfg)
    fit_time = time.time() - t0
    print(f"Fit time      : {fit_time:.1f}s")
    print(f"h_final       : {wrapper._fit_result.h_final:.4e}")
    print(f"score_final   : {wrapper._fit_result.score_final:.4e}")

    # ----------------------------------------------------------------------
    # Learned-structure diagnostics
    # ----------------------------------------------------------------------
    continuous_w = wrapper.native_edge_continuous()
    a_thresh = wrapper.thresholded_adjacency(PROJECT_THRESHOLD)
    graph_status = wrapper._graph_status
    sampler_status = wrapper._sampler_status
    sampler_reason = wrapper._sampler_unavailable_reason
    shd_value = int(shd(a_thresh, scm.adjacency))
    print()
    print(f"Learned continuous W:\n{np.round(continuous_w, 4)}")
    print(f"Thresholded adjacency at {PROJECT_THRESHOLD}:\n{a_thresh.astype(int)}")
    print(f"graph_status         : {graph_status}")
    print(f"sampler_status       : {sampler_status}")
    if sampler_reason:
        print(f"sampler_unavailable  : {sampler_reason}")
    print(f"SHD to true adjacency: {shd_value}")
    if wrapper._sigma_vector_residual_fitted is not None:
        print(f"Residual sigma vector: "
              f"{np.round(wrapper._sigma_vector_residual_fitted, 4)}")
    if wrapper._w_sample_residual_fitted is not None:
        print(f"Learned W_sample:\n{np.round(wrapper._w_sample_residual_fitted, 4)}")

    # Continuous-W values at every true-edge position (transparency for Diag A).
    print()
    print("Continuous-W values at TRUE-edge positions:")
    for i in range(NUM_VARS):
        for j in range(NUM_VARS):
            if scm.adjacency[i, j]:
                in_learned = bool(a_thresh[i, j])
                print(f"  ({i}->{j}) continuous_W = {continuous_w[i, j]:+.4f} "
                      f"true_w = {scm.weights[i, j]:+.4f} "
                      f"in_learned_adj = {in_learned}")

    if graph_status != "valid_dag":
        print()
        print("graph_status is not valid_dag; learned-structure MMD comparisons "
              "are blocked under the no-repair contract. Diagnostic A is still "
              "evaluated against the true adjacency.")

    # ----------------------------------------------------------------------
    # Monte Carlo floor MMD (pairwise GT)
    # ----------------------------------------------------------------------
    gt_floor = _draw_gt(
        scm, target, INTERVENTION_VALUE, N_FLOOR, BATCH_SIZE, GT_FLOOR_SEED_BASE,
    )
    floor_mmd = _median_pairwise(gt_floor)
    print()
    print("Monte Carlo floor")
    print(f"  ground-truth pairwise (n_floor={N_FLOOR}, "
          f"seed_base={GT_FLOOR_SEED_BASE}, batch={BATCH_SIZE})")
    print(f"  floor_mmd = {floor_mmd:+.6e}")
    print("  Note: the unbiased MMD estimator can be negative when both "
          "samples come from the same distribution.")

    # Pre-draw the paired GT batches that are reused across multiple comparisons.
    gt_paired = _draw_gt(
        scm, target, INTERVENTION_VALUE, N_WRAPPER, BATCH_SIZE,
        GT_PAIRED_SEED_BASE,
    )
    gt_wrong_paired = _draw_gt(
        scm, target, INTERVENTION_VALUE, N_WRAPPER, BATCH_SIZE,
        GT_WRONG_SEED_BASE,
    )

    # ----------------------------------------------------------------------
    # Wrapper-vs-truth (residual_fitted)
    # ----------------------------------------------------------------------
    wrapper_vs_truth = None
    wrapper_paired = None
    if graph_status == "valid_dag" and sampler_status == "available":
        wrapper_paired = _draw_wrapper_learned(
            wrapper, target, INTERVENTION_VALUE,
            N_WRAPPER, BATCH_SIZE, WRAPPER_SEED_BASE,
            noise_policy="residual_fitted",
        )
        if wrapper_paired is not None:
            wrapper_vs_truth = _median_paired(wrapper_paired, gt_paired)
    print()
    print("Wrapper-vs-truth (residual_fitted, learned structure)")
    if wrapper_vs_truth is None:
        print("  unavailable (graph_status or sampler_status blocks sampling)")
    else:
        print(f"  wrapper seed_base={WRAPPER_SEED_BASE}, "
              f"gt seed_base={GT_PAIRED_SEED_BASE}")
        print(f"  wrapper_vs_truth_mmd = {wrapper_vs_truth:+.6e}")

    # Primary acceptance comparison.
    primary_pass = None
    if wrapper_vs_truth is not None:
        primary_pass = bool(wrapper_vs_truth <= PRIMARY_MMD_RATIO * floor_mmd)
        print(f"  primary threshold: wrapper_vs_truth <= "
              f"{PRIMARY_MMD_RATIO} * floor_mmd "
              f"= {PRIMARY_MMD_RATIO * floor_mmd:+.6e} -> "
              f"{'PASS' if primary_pass else 'FAIL'}")
        if floor_mmd <= 0:
            print("  caveat: floor_mmd is non-positive; the literal threshold "
                  "comparison is recorded for protocol comparability but is "
                  "not a meaningful positive acceptance criterion. Compare "
                  "absolute scales below.")
        print(f"  |floor_mmd|              = {abs(floor_mmd):.6e}")
        print(f"  |wrapper_vs_truth_mmd|   = {abs(wrapper_vs_truth):.6e}")

    # ----------------------------------------------------------------------
    # Correct vs wrong structure
    # ----------------------------------------------------------------------
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
    wrong_correct_ratio = None
    failsafe_pass = None
    if graph_status != "valid_dag" or sampler_status != "available":
        print()
        print("Wrong-structure comparison: skipped because the wrapper sampler "
              "is unavailable on the learned graph.")
    elif chosen is None:
        print()
        print("Wrong-structure comparison: no true downstream edge present in "
              "the learned thresholded adjacency to delete. Reporting "
              "not applicable; no replacement is invented.")
    else:
        wrong_a_thresh = a_thresh.copy()
        wrong_a_thresh[chosen[0], chosen[1]] = False
        wrong_batches, _, _ = _draw_dagma_custom(
            wrapper._X_train_model_frame, continuous_w, wrong_a_thresh,
            preprocessor, target, INTERVENTION_VALUE,
            N_WRAPPER, BATCH_SIZE, WRAPPER_WRONG_SEED_BASE,
        )
        # The "correct" batches reuse the seed lane used for wrapper-vs-truth,
        # but paired against the gt_wrong_paired batches as in C-P11.
        correct_mmd = _median_paired(wrapper_paired, gt_wrong_paired)
        if wrong_batches is None:
            print()
            print(f"Wrong-structure comparison: residual sigma is degenerate "
                  f"after deleting {chosen[0]} -> {chosen[1]}; skipped.")
        else:
            wrong_mmd = _median_paired(wrong_batches, gt_wrong_paired)
            wrong_correct_ratio = (
                wrong_mmd / correct_mmd if correct_mmd > 0 else float("inf")
            )
            failsafe_pass = bool(
                correct_mmd * WRONG_STRUCTURE_RATIO <= wrong_mmd
            )
            print()
            print("Correct vs wrong structure (paired against gt_wrong)")
            print(f"  deleted edge: {chosen[0]} -> {chosen[1]} "
                  f"(true |weight| = {chosen[2]:.4f})")
            print(f"  correct_mmd = {correct_mmd:+.6e}")
            print(f"  wrong_mmd   = {wrong_mmd:+.6e}")
            print(f"  ratio wrong / correct = {wrong_correct_ratio:.3f}")
            print(f"  fail-safe: correct * {WRONG_STRUCTURE_RATIO} <= wrong -> "
                  f"{'PASS' if failsafe_pass else 'FAIL'}")

    # ----------------------------------------------------------------------
    # Diagnostic A: TRUE adjacency, learned continuous-W coefficients
    # ----------------------------------------------------------------------
    true_adj = scm.adjacency.copy()
    true_status, _ = classify_graph_status(true_adj)
    diag_a_mmd = None
    diag_a_w_sample = None
    diag_a_sigma = None
    print()
    print("Diagnostic A: MMD under TRUE adjacency, DAGMA learned continuous-W")
    print(f"  true adjacency status: {true_status}")
    if true_status != "valid_dag":
        print("  skipped: project SCM adjacency is not classified as valid_dag.")
    else:
        true_batches, diag_a_w_sample, diag_a_sigma = _draw_dagma_custom(
            wrapper._X_train_model_frame, continuous_w, true_adj,
            preprocessor, target, INTERVENTION_VALUE,
            N_WRAPPER, BATCH_SIZE, WRAPPER_TRUE_SEED_BASE,
        )
        print(f"  Diagnostic-A W_sample:\n  {np.round(diag_a_w_sample, 4)}")
        print(f"  Diagnostic-A sigma:    {np.round(diag_a_sigma, 4)}")
        if true_batches is None:
            print("  skipped: residual sigma degenerate under TRUE adjacency.")
        else:
            diag_a_mmd = _median_paired(true_batches, gt_wrong_paired)
            print(f"  wrapper seed_base={WRAPPER_TRUE_SEED_BASE}, "
                  f"paired GT seed_base={GT_WRONG_SEED_BASE}")
            print(f"  true_struct_mmd = {diag_a_mmd:+.6e}")
        # Near-zero learned-coefficient flag for any true edge.
        near_zero_edges = []
        for i in range(NUM_VARS):
            for j in range(NUM_VARS):
                if scm.adjacency[i, j] and abs(continuous_w[i, j]) < PROJECT_THRESHOLD:
                    near_zero_edges.append(
                        (i, j, float(continuous_w[i, j]),
                         float(scm.weights[i, j]))
                    )
        if near_zero_edges:
            print("  near-threshold true edges (|continuous_W| < "
                  f"{PROJECT_THRESHOLD}):")
            for i, j, w_learn, w_true in near_zero_edges:
                print(f"    {i} -> {j}: continuous_W = {w_learn:+.4f} "
                      f"true_w = {w_true:+.4f}")
        else:
            print("  all true edges have |continuous_W| >= "
                  f"{PROJECT_THRESHOLD}.")

    # ----------------------------------------------------------------------
    # Diagnostic B: strongest missing true edge
    # ----------------------------------------------------------------------
    learned_set = {
        (i, j) for i in range(NUM_VARS) for j in range(NUM_VARS) if a_thresh[i, j]
    }
    true_set = {
        (i, j) for i in range(NUM_VARS) for j in range(NUM_VARS)
        if scm.adjacency[i, j]
    }
    missing_true_edges = true_set - learned_set

    diag_b1_mmd = None
    diag_b2_mmd = None
    strongest = None
    aug_a_thresh = None
    aug_status = None

    print()
    print("Diagnostic B: learned adjacency augmented with strongest missing "
          "true edge")
    if not missing_true_edges:
        print("  no missing true edges; B1 and B2 not applicable.")
    else:
        weighted = [
            (i, j, float(abs(scm.weights[i, j])))
            for (i, j) in missing_true_edges
        ]
        weighted.sort(key=lambda x: x[2], reverse=True)
        strongest = weighted[0]
        aug_a_thresh = a_thresh.copy()
        aug_a_thresh[strongest[0], strongest[1]] = True

        aug_status, aug_reason = classify_graph_status(aug_a_thresh)
        print(f"  added edge: {strongest[0]} -> {strongest[1]} "
              f"(|true_w| = {strongest[2]:.4f})")
        print(f"  augmented adjacency:\n  {aug_a_thresh.astype(int)}")
        print(f"  aug_status: {aug_status}")

        if aug_status != "valid_dag":
            print(f"  B1/B2 skipped: augmented adjacency status is "
                  f"'{aug_status}'. Reason: {aug_reason}")
        else:
            # B1: learned continuous-W value for the added edge.
            b1_batches, b1_w_sample, b1_sigma = _draw_dagma_custom(
                wrapper._X_train_model_frame, continuous_w, aug_a_thresh,
                preprocessor, target, INTERVENTION_VALUE,
                N_WRAPPER, BATCH_SIZE, WRAPPER_LEARNED_AUG_SEED_BASE,
            )
            print("  B1 (learned continuous-W for added edge)")
            print(f"    learned continuous_W at added edge = "
                  f"{continuous_w[strongest[0], strongest[1]]:+.4f}")
            print(f"    B1 W_sample:\n    {np.round(b1_w_sample, 4)}")
            print(f"    B1 sigma:    {np.round(b1_sigma, 4)}")
            if b1_batches is None:
                print("    skipped: residual sigma degenerate under B1.")
            else:
                diag_b1_mmd = _median_paired(b1_batches, gt_wrong_paired)
                print(f"    wrapper seed_base="
                      f"{WRAPPER_LEARNED_AUG_SEED_BASE}, paired GT "
                      f"seed_base={GT_WRONG_SEED_BASE}")
                print(f"    B1_mmd = {diag_b1_mmd:+.6e}")
            if (
                abs(continuous_w[strongest[0], strongest[1]]) < 1e-12
            ):
                print("    note: learned continuous_W at the added edge is "
                      "near zero, so B1 behaves close to the unaugmented "
                      "learned-structure result (the added edge contributes "
                      "near-zero coefficient); B2 still uses the oracle "
                      "weight and is the comparator for coefficient effects.")

            # B2: oracle true SCM weight for the added edge.
            cont_w_b2 = continuous_w.copy()
            cont_w_b2[strongest[0], strongest[1]] = float(
                scm.weights[strongest[0], strongest[1]]
            )
            b2_batches, b2_w_sample, b2_sigma = _draw_dagma_custom(
                wrapper._X_train_model_frame, cont_w_b2, aug_a_thresh,
                preprocessor, target, INTERVENTION_VALUE,
                N_WRAPPER, BATCH_SIZE, WRAPPER_ORACLE_AUG_SEED_BASE,
            )
            print("  B2 (oracle true SCM weight for added edge)")
            print(f"    oracle weight at added edge = "
                  f"{scm.weights[strongest[0], strongest[1]]:+.4f}")
            print(f"    B2 W_sample:\n    {np.round(b2_w_sample, 4)}")
            print(f"    B2 sigma:    {np.round(b2_sigma, 4)}")
            if b2_batches is None:
                print("    skipped: residual sigma degenerate under B2.")
            else:
                diag_b2_mmd = _median_paired(b2_batches, gt_wrong_paired)
                print(f"    wrapper seed_base="
                      f"{WRAPPER_ORACLE_AUG_SEED_BASE}, paired GT "
                      f"seed_base={GT_WRONG_SEED_BASE}")
                print(f"    B2_mmd = {diag_b2_mmd:+.6e}")

    # ----------------------------------------------------------------------
    # Unit-variance sensitivity (separate from primary result)
    # ----------------------------------------------------------------------
    unit_var_mmd = None
    print()
    print("Unit-variance sensitivity (separate from primary result)")
    if graph_status != "valid_dag":
        print("  skipped: graph_status blocks any wrapper sampling.")
    else:
        unit_var_paired = _draw_wrapper_learned(
            wrapper, target, INTERVENTION_VALUE,
            N_WRAPPER, BATCH_SIZE, WRAPPER_SEED_BASE,
            noise_policy="unit_variance",
        )
        if unit_var_paired is None:
            print("  skipped: unit_variance sampler returned None.")
        else:
            unit_var_mmd = _median_paired(unit_var_paired, gt_paired)
            print(f"  unit_variance_mmd = {unit_var_mmd:+.6e}")
            if wrapper_vs_truth is not None:
                print(f"  delta vs residual_fitted = "
                      f"{unit_var_mmd - wrapper_vs_truth:+.6e}")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  graph_status (learned at {PROJECT_THRESHOLD})      : "
          f"{graph_status}")
    print(f"  sampler_status                       : {sampler_status}")
    print(f"  SHD to true graph                    : {shd_value}")
    print(f"  floor MMD median                     : {floor_mmd:+.6e}")
    if wrapper_vs_truth is not None:
        print(f"  wrapper-vs-truth MMD median          : "
              f"{wrapper_vs_truth:+.6e}")
        print(f"  primary threshold (<= {PRIMARY_MMD_RATIO}x floor)        : "
              f"{'PASS' if primary_pass else 'FAIL'}")
    else:
        print("  wrapper-vs-truth MMD median          : N/A")
    if wrong_mmd is not None:
        print(f"  correct vs wrong ratio (wrong/correct): "
              f"{wrong_correct_ratio:.3f}")
        print(f"  fail-safe (correct * {WRONG_STRUCTURE_RATIO} <= wrong)  : "
              f"{'PASS' if failsafe_pass else 'FAIL'}")
    else:
        print("  fail-safe                            : N/A")
    if diag_a_mmd is not None:
        print(f"  Diagnostic A MMD (true adj)          : {diag_a_mmd:+.6e}")
    if diag_b1_mmd is not None:
        print(f"  Diagnostic B1 MMD (learned aug)      : {diag_b1_mmd:+.6e}")
    if diag_b2_mmd is not None:
        print(f"  Diagnostic B2 MMD (oracle aug)       : {diag_b2_mmd:+.6e}")
    if unit_var_mmd is not None:
        print(f"  unit_variance MMD (sensitivity)      : {unit_var_mmd:+.6e}")
    print()
    print("End of probe output.")


if __name__ == "__main__":
    main()
