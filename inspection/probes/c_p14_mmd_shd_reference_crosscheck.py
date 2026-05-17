"""C-P14: MMD and SHD independent cross-check probe.

Verifies the project mmd_rbf_unbiased and shd implementations against
independent references. This probe is read-only with respect to project
source, tests, and dependency manifests.
"""

import inspect
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gadjid  # noqa: E402
from symbolic_priors_cd.metrics.interventional import mmd_rbf_unbiased  # noqa: E402
from symbolic_priors_cd.metrics.structural import shd  # noqa: E402


# ---------------------------------------------------------------------------
# Small adjacency helpers
# ---------------------------------------------------------------------------


def _dag_bool(edges, n):
    """Return an (n, n) bool adjacency from a list of (i, j) edge tuples."""
    A = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        A[i, j] = True
    return A


def _dag_int8(edges, n):
    return _dag_bool(edges, n).astype(np.int8)


def _empty_bool(n):
    return np.zeros((n, n), dtype=bool)


def _empty_int8(n):
    return np.zeros((n, n), dtype=np.int8)


# ---------------------------------------------------------------------------
# Loop-based MMD reference
#
# Replicates the project formula exactly:
#   kernel: k(a, b) = exp(-||a - b||^2 / bandwidth)
#   within-sample: sum over i != j, denominator m*(m-1) and n*(n-1)
#   cross-sample:  2 * sum over all (i, j), denominator m*n
#   result: xx_term - xy_term + yy_term  (never clipped)
# ---------------------------------------------------------------------------


def _ref_kernel(a, b, bw):
    sq_dist = sum((float(a[d]) - float(b[d])) ** 2 for d in range(len(a)))
    return math.exp(-sq_dist / bw)


def _ref_mmd(x, y, bw):
    """Loop-based reference MMD matching the project formula exactly."""
    m, n = len(x), len(y)
    xx = sum(
        _ref_kernel(x[i], x[j], bw)
        for i in range(m) for j in range(m) if i != j
    )
    yy = sum(
        _ref_kernel(y[i], y[j], bw)
        for i in range(n) for j in range(n) if i != j
    )
    xy = sum(
        _ref_kernel(x[i], y[j], bw)
        for i in range(m) for j in range(n)
    )
    return xx / (m * (m - 1)) - 2.0 * xy / (m * n) + yy / (n * (n - 1))


def _ref_median_bw(x, y):
    """Reference median bandwidth: squared distances, concatenated, upper triangle."""
    z = [np.asarray(row, dtype=float) for row in list(x) + list(y)]
    N = len(z)
    sq_dists = []
    for i in range(N):
        for j in range(i + 1, N):
            diff = z[i] - z[j]
            sq_dists.append(float(np.dot(diff, diff)))
    sq_dists.sort()
    N2 = len(sq_dists)
    if N2 == 0:
        return 0.0
    if N2 % 2 == 0:
        return (sq_dists[N2 // 2 - 1] + sq_dists[N2 // 2]) / 2.0
    return float(sq_dists[N2 // 2])


# ---------------------------------------------------------------------------
# gadjid.shd wrapper
# ---------------------------------------------------------------------------


def _gadjid_shd(pred_bool, true_bool):
    """Call gadjid.shd with int8 inputs.

    gadjid.shd takes only (g_true, g_guess) with no edge_direction argument.
    SHD is symmetric so no convention flip is required.
    """
    return gadjid.shd(
        true_bool.astype(np.int8),
        pred_bool.astype(np.int8),
    )


def _gadjid_shd_count(result):
    """Extract integer count from gadjid.shd return (tuple or scalar)."""
    if isinstance(result, tuple):
        return int(result[1])
    return int(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    overall_ok = True

    print("=" * 72)
    print("C-P14: MMD and SHD reference cross-check")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. Inspected project MMD formula
    # -----------------------------------------------------------------------
    print("\n--- 1. Inspected project mmd_rbf_unbiased formula ---")
    print("  Kernel: k(a, b) = exp(-||a - b||^2 / bandwidth)")
    print("  Within-sample XX: (Kxx.sum() - m) / (m * (m - 1))  [diagonal excluded]")
    print("  Within-sample YY: (Kyy.sum() - n) / (n * (n - 1))  [diagonal excluded]")
    print("  Cross-sample XY:  2.0 * Kxy.sum() / (m * n)  [all pairs included]")
    print("  MMD^2 = xx_term - xy_term + yy_term")
    print("  Median heuristic: np.median over squared pairwise distances on")
    print("    concatenated [x; y], upper triangle only (i < j), no self-distances.")
    print("  Return value is raw (never clipped; negative values valid).")

    # -----------------------------------------------------------------------
    # 2. MMD loop-reference comparisons
    # -----------------------------------------------------------------------
    print("\n--- 2. MMD loop-reference comparisons ---")
    tol = 1e-10

    # Case A: fixed small arrays, explicit bandwidth
    x_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y_a = np.array([[2.0, 0.0], [2.0, 1.0]])
    bw_a = 1.0
    proj_a = mmd_rbf_unbiased(x_a, y_a, bandwidth=bw_a)
    ref_a  = _ref_mmd(x_a.tolist(), y_a.tolist(), bw_a)
    diff_a = abs(proj_a - ref_a)
    flag_a = "OK" if diff_a < tol else "MISMATCH"
    print(f"  [A] fixed arrays, bw=1.0:")
    print(f"      project = {proj_a:.12f}")
    print(f"      ref     = {ref_a:.12f}")
    print(f"      diff    = {diff_a:.3e}  [{flag_a}]")
    if flag_a != "OK":
        overall_ok = False

    # Case B: known negative unbiased-estimator example (cross-kernel dominates)
    x_b = np.array([[0.0], [2.0]])
    y_b = np.array([[1.0], [3.0]])
    bw_b = 0.5
    proj_b = mmd_rbf_unbiased(x_b, y_b, bandwidth=bw_b)
    ref_b  = _ref_mmd(x_b.tolist(), y_b.tolist(), bw_b)
    diff_b = abs(proj_b - ref_b)
    flag_b = "OK" if diff_b < tol else "MISMATCH"
    negative_b = "OK" if proj_b < 0 else "WARN: expected negative"
    print(f"  [B] negative-estimator example, bw=0.5:")
    print(f"      project = {proj_b:.12f}  [{negative_b}]")
    print(f"      ref     = {ref_b:.12f}")
    print(f"      diff    = {diff_b:.3e}  [{flag_b}]")
    if flag_b != "OK" or proj_b >= 0:
        overall_ok = False

    # Case C: median-heuristic bandwidth (x=[[0],[1]], y=[[2],[3]])
    x_c = np.array([[0.0], [1.0]])
    y_c = np.array([[2.0], [3.0]])
    ref_bw_c = _ref_median_bw(x_c.tolist(), y_c.tolist())
    proj_c   = mmd_rbf_unbiased(x_c, y_c)
    ref_c    = _ref_mmd(x_c.tolist(), y_c.tolist(), ref_bw_c)
    diff_c   = abs(proj_c - ref_c)
    flag_c   = "OK" if diff_c < tol else "MISMATCH"
    print(f"  [C] median-heuristic, bw={ref_bw_c:.4g}:")
    print(f"      project = {proj_c:.12f}")
    print(f"      ref     = {ref_c:.12f}")
    print(f"      diff    = {diff_c:.3e}  [{flag_c}]")
    if flag_c != "OK":
        overall_ok = False

    # Case D: same-distribution seeded samples near zero
    rng   = np.random.default_rng(0)
    x_d   = rng.standard_normal((200, 3))
    y_d   = rng.standard_normal((200, 3))
    proj_d = mmd_rbf_unbiased(x_d, y_d)
    near_zero_tol = 0.05
    flag_d = "OK" if abs(proj_d) < near_zero_tol else "WARN"
    print(f"  [D] same-dist (n=200, seed=0):")
    print(f"      project = {proj_d:.6f}  (expect |.| < {near_zero_tol})  [{flag_d}]")
    if flag_d != "OK":
        print(f"      NOTE: value outside expected range; large-sample variance.")

    # Case E: degenerate all-identical samples (median bandwidth = 0)
    x_e = np.ones((5, 2))
    try:
        mmd_rbf_unbiased(x_e, x_e)
        print("  [E] degenerate (all-identical): expected ValueError, got none  [MISMATCH]")
        overall_ok = False
    except ValueError as exc:
        msg = str(exc)
        flag_e = "OK" if "non-positive bandwidth" in msg else "OK-ish (different message)"
        print(f"  [E] degenerate (all-identical): raised ValueError  [{flag_e}]")
        print(f"      message: '{msg[:70]}'")

    # -----------------------------------------------------------------------
    # 3. SHD hand cases
    # -----------------------------------------------------------------------
    print("\n--- 3. SHD hand cases ---")
    shd_cases = [
        # (label, predicted, true, reversal_cost, expected)
        ("identity empty 3x3",
         _empty_bool(3), _empty_bool(3), 2, 0),
        ("identity chain 0->1->2",
         _dag_bool([(0, 1), (1, 2)], 3), _dag_bool([(0, 1), (1, 2)], 3), 2, 0),
        ("empty predicted, single true edge",
         _empty_bool(2), _dag_bool([(0, 1)], 2), 2, 1),
        ("single predicted edge, empty true",
         _dag_bool([(0, 1)], 2), _empty_bool(2), 2, 1),
        ("two predicted edges, empty true",
         _dag_bool([(0, 1), (0, 2)], 3), _empty_bool(3), 2, 2),
        ("reversed edge, reversal_cost=2",
         _dag_bool([(1, 0)], 2), _dag_bool([(0, 1)], 2), 2, 2),
        ("reversed edge, reversal_cost=1",
         _dag_bool([(1, 0)], 2), _dag_bool([(0, 1)], 2), 1, 1),
    ]
    shd_all_ok = True
    for label, pred, true_g, rc, expected in shd_cases:
        result = shd(pred, true_g, reversal_cost=rc)
        flag = "OK" if result == expected else "MISMATCH"
        print(f"  shd(predicted, true, reversal_cost={rc}) [{label}]:"
              f" got={result} expected={expected} [{flag}]")
        if flag != "OK":
            shd_all_ok = False
            overall_ok = False
    if shd_all_ok:
        print("  All SHD hand cases matched expected values.")

    # -----------------------------------------------------------------------
    # 4. gadjid.shd comparison
    # -----------------------------------------------------------------------
    print("\n--- 4. gadjid.shd comparison ---")
    try:
        sig = inspect.signature(gadjid.shd)
        print(f"  gadjid.shd signature: {sig}")
    except (TypeError, ValueError):
        print("  gadjid.shd signature: (not inspectable via Python)")

    gadjid_dag_cases = [
        ("identity empty 2x2",
         _empty_bool(2), _empty_bool(2), 0),
        ("identity chain 0->1->2",
         _dag_bool([(0, 1), (1, 2)], 3), _dag_bool([(0, 1), (1, 2)], 3), 0),
        ("empty predicted, single true edge",
         _empty_bool(2), _dag_bool([(0, 1)], 2), 1),
        ("two predicted edges, empty true",
         _dag_bool([(0, 1), (0, 2)], 3), _empty_bool(3), 2),
    ]
    gadjid_dag_ok = True
    for label, pred, true_g, expected in gadjid_dag_cases:
        result = _gadjid_shd(pred, true_g)
        count  = _gadjid_shd_count(result)
        flag   = "OK" if count == expected else "CHECK"
        print(f"  gadjid.shd [{label}]: got={count} expected={expected} [{flag}]")
        if flag != "OK":
            gadjid_dag_ok = False

    # Reversed-edge comparison: gadjid vs project reversal_cost=1 and =2
    pred_rev = _dag_bool([(1, 0)], 2)
    true_rev = _dag_bool([(0, 1)], 2)
    gadjid_rev_raw   = _gadjid_shd(pred_rev, true_rev)
    gadjid_rev_count = _gadjid_shd_count(gadjid_rev_raw)
    proj_rev_1       = shd(pred_rev, true_rev, reversal_cost=1)
    proj_rev_2       = shd(pred_rev, true_rev, reversal_cost=2)
    print(f"  Reversed-edge 2-node (predicted=1->0, true=0->1):")
    print(f"    gadjid.shd count             = {gadjid_rev_count}  (raw={gadjid_rev_raw!r})")
    print(f"    project shd reversal_cost=1  = {proj_rev_1}")
    print(f"    project shd reversal_cost=2  = {proj_rev_2}")
    if gadjid_rev_count == proj_rev_2:
        print(f"    gadjid.shd matches project reversal_cost=2 (stricter convention)")
    elif gadjid_rev_count == proj_rev_1:
        print(f"    gadjid.shd matches project reversal_cost=1 (cheaper convention)")
    else:
        print(f"    gadjid.shd does not match either project reversal convention")

    # gadjid.shd invalid-input probe: cyclic 3-node graph as g_guess
    print(f"\n  gadjid.shd invalid-input probe (3-node cycle as g_guess):")
    cycle_int8 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
    try:
        result = gadjid.shd(_empty_int8(3), cycle_int8)
        print(f"    RETURNED (did not raise): {result!r}")
    except BaseException as exc:
        print(f"    RAISED {type(exc).__name__}: {str(exc)[:80]!r}")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print("\n--- 5. Summary ---")
    print(f"  MMD loop-reference (cases A/B/C): all diffs < {tol}  "
          f"{'[OK]' if overall_ok else '[see above]'}")
    print(f"  MMD negative-value case (B): confirmed unbiased estimator < 0")
    print(f"  MMD degenerate case (E): raises ValueError on zero-bandwidth inputs")
    print(f"  SHD hand cases: {'all matched' if shd_all_ok else 'mismatches found'}")
    print(f"  gadjid.shd DAG cases: {'matched project' if gadjid_dag_ok else 'check above'}")
    print(f"  Overall: {'PASS' if overall_ok and shd_all_ok and gadjid_dag_ok else 'ISSUES FOUND'}")
    print()


if __name__ == "__main__":
    main()
