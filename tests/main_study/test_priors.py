"""Tests for the main-study prior generation and corruption utilities.

All tests operate on adjacency matrices and seeds only. No DAGMA
fit, no metric computation, and no result-record I/O is exercised
by these tests. Adjacency fixtures are either hand-built or
generated from the project SCM utility via a known seed; in both
cases only the boolean adjacency is consumed downstream.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.main_study import priors as priors_mod
from experiments.main_study.priors import (
    CORRUPTION_GRID,
    CORRUPTION_SEED_BASE,
    CorruptedPriorSpec,
    EDGE_LABEL_TRUE_NEGATIVE_RETAINED,
    EDGE_LABEL_TRUE_POSITIVE_CORRUPTED_REPLACEMENT,
    PRIOR_K,
    PRIOR_SEED_BASE,
    PriorSpec,
    build_confidence_mask,
    corrupt_prior,
    corruption_index_for_fraction,
    edge_key_to_tuple,
    edge_tuple_to_key,
    generate_prior_for_scm_seed,
    sample_clean_forbidden_edges,
    true_negative_edges,
    true_positive_edges,
    validate_adjacency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hand_built_small_adj() -> np.ndarray:
    """3x3 adjacency with edges 0->1 and 1->2 only."""
    a = np.zeros((3, 3), dtype=bool)
    a[0, 1] = True
    a[1, 2] = True
    return a


def _hand_built_eight_node_adj() -> np.ndarray:
    """8-node DAG with 10 true-positive edges and 46 true-negative edges.

    Topological order is the natural integer order 0..7. All edges
    point from a lower index to a higher index, so the graph is
    trivially acyclic. The selected edges provide enough true
    positives (10) to support the full corruption sweep at
    ``prior_k = 10``, and enough true negatives (46) to support
    sampling 10 clean prior edges.
    """
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5),
        (5, 7),
    ]
    a = np.zeros((8, 8), dtype=bool)
    for i, j in edges:
        a[i, j] = True
    assert int(a.sum()) == len(edges)
    return a


@pytest.fixture(scope="module")
def small_adj() -> np.ndarray:
    return _hand_built_small_adj()


@pytest.fixture(scope="module")
def eight_node_adj() -> np.ndarray:
    return _hand_built_eight_node_adj()


@pytest.fixture(scope="module")
def calibration_seed_prior() -> tuple[PriorSpec, np.ndarray]:
    """Clean prior on the project's deterministic SCM at calibration seed 401."""
    from symbolic_priors_cd.data.scm_generator import (
        generate_linear_gaussian_scm,
    )

    scm = generate_linear_gaussian_scm(
        n_nodes=10, expected_edges=20, seed=401
    )
    true_adj = np.asarray(scm.adjacency, dtype=bool).copy()
    spec = sample_clean_forbidden_edges(
        true_adjacency=true_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=PRIOR_SEED_BASE + 401,
        scm_seed=401,
    )
    return spec, true_adj


# ---------------------------------------------------------------------------
# T-1: true-negative purity
# ---------------------------------------------------------------------------


def test_clean_prior_returns_exactly_prior_k_edges(eight_node_adj):
    spec = sample_clean_forbidden_edges(
        true_adjacency=eight_node_adj,
        prior_k=10,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    assert len(spec.forbidden_edges) == 10


def test_clean_prior_contains_only_true_negative_edges(eight_node_adj):
    spec = sample_clean_forbidden_edges(
        true_adjacency=eight_node_adj,
        prior_k=10,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    for i, j in spec.forbidden_edges:
        assert not bool(eight_node_adj[i, j]), (
            f"edge ({i}, {j}) is present in the true adjacency."
        )


def test_clean_prior_has_no_self_loops_no_duplicates_in_range(eight_node_adj):
    spec = sample_clean_forbidden_edges(
        true_adjacency=eight_node_adj,
        prior_k=10,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    n = eight_node_adj.shape[0]
    seen = set()
    for i, j in spec.forbidden_edges:
        assert i != j, f"self-loop edge ({i}, {j})"
        assert 0 <= i < n and 0 <= j < n, f"out-of-range edge ({i}, {j})"
        assert (i, j) not in seen, f"duplicate edge ({i}, {j})"
        seen.add((i, j))


# ---------------------------------------------------------------------------
# T-2: true-positive and true-negative enumeration
# ---------------------------------------------------------------------------


def test_true_positive_edges_lexicographic_on_small_graph(small_adj):
    assert true_positive_edges(small_adj) == ((0, 1), (1, 2))


def test_true_negative_edges_lexicographic_on_small_graph(small_adj):
    # Off-diagonal absent edges in row-major order on the 3x3 with
    # only (0,1) and (1,2) present.
    expected = ((0, 2), (1, 0), (2, 0), (2, 1))
    assert true_negative_edges(small_adj) == expected


# ---------------------------------------------------------------------------
# T-3: deterministic clean prior sampling
# ---------------------------------------------------------------------------


def test_same_seed_yields_identical_clean_prior(eight_node_adj):
    a = sample_clean_forbidden_edges(
        eight_node_adj, prior_k=10, prior_selection_seed=42, scm_seed=99,
    )
    b = sample_clean_forbidden_edges(
        eight_node_adj, prior_k=10, prior_selection_seed=42, scm_seed=99,
    )
    assert a == b


def test_different_seed_yields_different_clean_prior(eight_node_adj):
    a = sample_clean_forbidden_edges(
        eight_node_adj, prior_k=10, prior_selection_seed=0, scm_seed=99,
    )
    b = sample_clean_forbidden_edges(
        eight_node_adj, prior_k=10, prior_selection_seed=12345, scm_seed=99,
    )
    assert a.forbidden_edges != b.forbidden_edges


# ---------------------------------------------------------------------------
# T-4: corruption index lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_index",
    [
        (0.0, 0),
        (0.2, 1),
        (0.4, 2),
        (0.6, 3),
        (0.8, 4),
    ],
)
def test_corruption_index_for_each_grid_value(value, expected_index):
    assert corruption_index_for_fraction(value) == expected_index


def test_corruption_index_accepts_near_equal_floats():
    # Tolerance: abs_tol=1e-12.
    near = 0.2 + 1e-13
    assert corruption_index_for_fraction(near) == 1


@pytest.mark.parametrize(
    "off_grid_value",
    [0.1, -0.2, 1.2, float("nan"), float("inf"), float("-inf")],
)
def test_corruption_index_rejects_off_grid_values(off_grid_value):
    with pytest.raises(ValueError):
        corruption_index_for_fraction(off_grid_value)


# ---------------------------------------------------------------------------
# T-5: corruption count and purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", list(CORRUPTION_GRID))
def test_corruption_count_and_purity(eight_node_adj, fraction):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, fraction)
    n_corrupt_expected = int(round(fraction * PRIOR_K))
    assert cp.n_corrupted == n_corrupt_expected
    assert cp.n_correct == PRIOR_K - n_corrupt_expected
    assert len(cp.forbidden_edges) == PRIOR_K

    # Retained edges must be true negatives.
    tn_set = set(true_negative_edges(eight_node_adj))
    retained = set(cp.forbidden_edges) - set(cp.added_true_positive_edges)
    assert retained.issubset(tn_set)

    # Added replacement edges must be true positives.
    tp_set = set(true_positive_edges(eight_node_adj))
    assert set(cp.added_true_positive_edges).issubset(tp_set)


# ---------------------------------------------------------------------------
# T-6: corruption 0.0 audit seed
# ---------------------------------------------------------------------------


def test_zero_corruption_returns_clean_prior_exactly(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.0)
    assert cp.forbidden_edges == prior.forbidden_edges
    assert cp.removed_clean_edges == tuple()
    assert cp.added_true_positive_edges == tuple()
    assert cp.n_corrupted == 0
    assert cp.n_correct == PRIOR_K


def test_zero_corruption_populates_corruption_seed(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.0)
    # corruption_index=0, scm_seed=99 -> seed = CORRUPTION_SEED_BASE + 99 + 0.
    expected_seed = CORRUPTION_SEED_BASE + 99 + 0
    assert cp.corruption_seed == expected_seed


def test_zero_corruption_labels_all_clean_edges_as_true_negative_retained(
    eight_node_adj,
):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.0)
    assert set(cp.edge_labels.keys()) == {
        edge_tuple_to_key(e) for e in prior.forbidden_edges
    }
    assert all(
        label == EDGE_LABEL_TRUE_NEGATIVE_RETAINED
        for label in cp.edge_labels.values()
    )


# ---------------------------------------------------------------------------
# T-7: deterministic corruption
# ---------------------------------------------------------------------------


def test_same_inputs_yield_identical_corrupted_spec(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    a = corrupt_prior(prior, eight_node_adj, 0.4)
    b = corrupt_prior(prior, eight_node_adj, 0.4)
    assert a == b


def test_different_fraction_yields_different_corruption_seed(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    a = corrupt_prior(prior, eight_node_adj, 0.2)
    b = corrupt_prior(prior, eight_node_adj, 0.4)
    assert a.corruption_seed != b.corruption_seed


def test_different_fraction_changes_added_or_removed_edges(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    a = corrupt_prior(prior, eight_node_adj, 0.2)
    b = corrupt_prior(prior, eight_node_adj, 0.4)
    # Either the added or the removed set must differ. With different
    # fractions producing different sizes this is guaranteed.
    assert (
        a.added_true_positive_edges != b.added_true_positive_edges
        or a.removed_clean_edges != b.removed_clean_edges
    )


# ---------------------------------------------------------------------------
# T-8: no duplicate or invalid corrupted edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", list(CORRUPTION_GRID))
def test_corrupted_edges_are_unique_and_valid(eight_node_adj, fraction):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, fraction)
    n = eight_node_adj.shape[0]
    seen = set()
    for i, j in cp.forbidden_edges:
        assert i != j, "self-loop edge present"
        assert 0 <= i < n and 0 <= j < n, "out-of-range edge"
        assert (i, j) not in seen, "duplicate edge"
        seen.add((i, j))


# ---------------------------------------------------------------------------
# T-9: confidence mask
# ---------------------------------------------------------------------------


def test_confidence_mask_shape_and_dtype(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    mask = build_confidence_mask(cp, 0.75)
    assert mask.shape == (8, 8)
    assert mask.dtype == float


def test_confidence_mask_has_value_at_forbidden_positions(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    confidence = 0.625
    mask = build_confidence_mask(cp, confidence)
    for i, j in cp.forbidden_edges:
        assert mask[i, j] == confidence


def test_confidence_mask_zero_outside_forbidden_positions(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    mask = build_confidence_mask(cp, 0.75)
    n = eight_node_adj.shape[0]
    forbidden_set = set(cp.forbidden_edges)
    for i in range(n):
        for j in range(n):
            if (i, j) in forbidden_set:
                continue
            assert mask[i, j] == 0.0


def test_confidence_mask_diagonal_is_exactly_zero(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    mask = build_confidence_mask(cp, 1.0)
    assert np.all(np.diag(mask) == 0.0)


def test_confidence_zero_yields_all_zero_mask(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    mask = build_confidence_mask(cp, 0.0)
    assert np.all(mask == 0.0)


# ---------------------------------------------------------------------------
# T-10: confidence validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [-0.1, -1.0, 1.0001, 2.0, float("nan"), float("inf"), float("-inf")],
)
def test_confidence_value_rejects_invalid(eight_node_adj, bad):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.0)
    with pytest.raises(ValueError):
        build_confidence_mask(cp, bad)


# ---------------------------------------------------------------------------
# T-11: edge-label integrity
# ---------------------------------------------------------------------------


def test_edge_labels_cover_every_final_edge(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    expected_keys = {edge_tuple_to_key(e) for e in cp.forbidden_edges}
    assert set(cp.edge_labels.keys()) == expected_keys


def test_edge_labels_distinguish_retained_from_added(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    added_set = set(cp.added_true_positive_edges)
    retained_set = set(cp.forbidden_edges) - added_set
    for edge in retained_set:
        assert (
            cp.edge_labels[edge_tuple_to_key(edge)]
            == EDGE_LABEL_TRUE_NEGATIVE_RETAINED
        )
    for edge in added_set:
        assert (
            cp.edge_labels[edge_tuple_to_key(edge)]
            == EDGE_LABEL_TRUE_POSITIVE_CORRUPTED_REPLACEMENT
        )


def test_edge_labels_do_not_contain_removed_edges(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    cp = corrupt_prior(prior, eight_node_adj, 0.4)
    for edge in cp.removed_clean_edges:
        assert edge_tuple_to_key(edge) not in cp.edge_labels


# ---------------------------------------------------------------------------
# T-12: edge key helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edge", [(0, 1), (1, 2), (7, 3), (10, 0), (0, 99)]
)
def test_edge_key_round_trip(edge):
    key = edge_tuple_to_key(edge)
    assert edge_key_to_tuple(key) == edge


@pytest.mark.parametrize(
    "bad_edge",
    [
        (1,),
        (1, 2, 3),
        (-1, 2),
        (1, -1),
        (1, 1),
        ("1", 2),
        (1, "2"),
        (True, 1),
        (1, True),
    ],
)
def test_edge_tuple_to_key_rejects_malformed(bad_edge):
    with pytest.raises(ValueError):
        edge_tuple_to_key(bad_edge)


@pytest.mark.parametrize(
    "bad_key",
    [
        "1",
        "1,2,3",
        "abc",
        "1,",
        ",2",
        "1,a",
        "-1,2",
        "1,-1",
        "1,1",
        "+1,2",
    ],
)
def test_edge_key_to_tuple_rejects_malformed(bad_key):
    with pytest.raises(ValueError):
        edge_key_to_tuple(bad_key)


def test_edge_key_to_tuple_rejects_non_string():
    with pytest.raises(ValueError):
        edge_key_to_tuple(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T-13: validation failures
# ---------------------------------------------------------------------------


def test_validate_adjacency_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        validate_adjacency(np.zeros(5, dtype=bool))


def test_validate_adjacency_rejects_3d():
    with pytest.raises(ValueError, match="2D"):
        validate_adjacency(np.zeros((2, 2, 2), dtype=bool))


def test_validate_adjacency_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        validate_adjacency(np.zeros((3, 4), dtype=bool))


def test_validate_adjacency_rejects_diagonal_self_loop():
    bad = np.zeros((3, 3), dtype=bool)
    bad[1, 1] = True
    with pytest.raises(ValueError, match="diagonal"):
        validate_adjacency(bad)


def test_validate_adjacency_rejects_nan_entry():
    bad = np.zeros((3, 3), dtype=float)
    bad[0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_adjacency(bad)


def test_validate_adjacency_rejects_infinite_entry():
    bad = np.zeros((3, 3), dtype=float)
    bad[0, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_adjacency(bad)


def test_validate_adjacency_rejects_object_dtype():
    bad = np.array([[None, None], [None, None]], dtype=object)
    with pytest.raises(ValueError, match="object"):
        validate_adjacency(bad)


def test_sample_clean_forbidden_edges_rejects_non_positive_prior_k(
    eight_node_adj,
):
    with pytest.raises(ValueError, match="positive"):
        sample_clean_forbidden_edges(
            eight_node_adj, prior_k=0, prior_selection_seed=0
        )
    with pytest.raises(ValueError, match="positive"):
        sample_clean_forbidden_edges(
            eight_node_adj, prior_k=-5, prior_selection_seed=0
        )


def test_sample_clean_forbidden_edges_rejects_too_large_prior_k(small_adj):
    # The 3x3 fixture has 4 true-negative edges; asking for 5 must fail.
    with pytest.raises(ValueError, match="Not enough"):
        sample_clean_forbidden_edges(
            small_adj, prior_k=5, prior_selection_seed=0
        )


def test_corrupt_prior_rejects_malformed_prior_edges(eight_node_adj):
    bad_prior = PriorSpec(
        n_nodes=8,
        scm_seed=99,
        prior_selection_seed=12345,
        forbidden_edges=((0, 0),),  # self-loop, malformed
    )
    with pytest.raises(ValueError):
        corrupt_prior(bad_prior, eight_node_adj, 0.0)


def test_corrupt_prior_rejects_non_true_negative_edges(eight_node_adj):
    # (0, 1) is a true positive in eight_node_adj.
    bad_prior = PriorSpec(
        n_nodes=8,
        scm_seed=99,
        prior_selection_seed=12345,
        forbidden_edges=((0, 1),),
    )
    with pytest.raises(ValueError, match="true-negative"):
        corrupt_prior(bad_prior, eight_node_adj, 0.0)


def test_corrupt_prior_rejects_none_scm_seed(eight_node_adj):
    prior_no_seed = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=None,
    )
    with pytest.raises(ValueError, match="scm_seed"):
        corrupt_prior(prior_no_seed, eight_node_adj, 0.0)


def test_corrupt_prior_rejects_shape_mismatch(eight_node_adj):
    prior = sample_clean_forbidden_edges(
        eight_node_adj,
        prior_k=PRIOR_K,
        prior_selection_seed=12345,
        scm_seed=99,
    )
    smaller = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="match"):
        corrupt_prior(prior, smaller, 0.0)


# ---------------------------------------------------------------------------
# T-14: leakage / import guard
# ---------------------------------------------------------------------------


_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "symbolic_priors_cd.wrappers",
    "symbolic_priors_cd.metrics",
    "experiments.selection_study",
    "experiments.main_study.calibration_lambda_prior",
    "dagma",
    "dcdi",
)


_ALLOWED_MODULE_LEVEL_PREFIXES: frozenset[str] = frozenset({
    "numpy",
    "math",
    "dataclasses",
    "typing",
    "__future__",
    "symbolic_priors_cd.data",
})


def _module_imports(tree: ast.Module) -> list[tuple[str, bool]]:
    """Return (module_name, is_module_level) pairs for every import in tree."""
    top_level_ids = {id(node) for node in tree.body}
    imports: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, id(node) in top_level_ids))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.module, id(node) in top_level_ids))
    return imports


def test_priors_module_does_not_import_forbidden_packages():
    src = Path(priors_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod, _is_top in _module_imports(tree):
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            assert not mod.startswith(forbidden), (
                f"priors.py must not import {mod!r}; "
                f"forbidden prefix {forbidden!r}."
            )


def test_priors_module_level_imports_are_allowlisted():
    src = Path(priors_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for mod, is_top in _module_imports(tree):
        if not is_top:
            continue
        first = mod.split(".")[0]
        ok = (
            first in {"numpy", "math", "dataclasses", "typing", "__future__"}
            or mod.startswith("symbolic_priors_cd.data")
        )
        assert ok, (
            f"Module-level import of {mod!r} is not in the allowlist "
            f"{_ALLOWED_MODULE_LEVEL_PREFIXES}."
        )


_FORBIDDEN_PARAMETER_NAMES: frozenset[str] = frozenset({
    "X_train",
    "x_train",
    "X",
    "W",
    "w_learned",
    "learned_w",
    "W_learned",
})


def test_public_functions_do_not_accept_x_train_or_learned_w():
    """The adjacency-only API must not expose X_train or learned W."""
    public_functions = [
        priors_mod.validate_adjacency,
        priors_mod.true_negative_edges,
        priors_mod.true_positive_edges,
        priors_mod.edge_tuple_to_key,
        priors_mod.edge_key_to_tuple,
        priors_mod.sample_clean_forbidden_edges,
        priors_mod.generate_prior_for_scm_seed,
        priors_mod.corruption_index_for_fraction,
        priors_mod.corrupt_prior,
        priors_mod.build_confidence_mask,
    ]
    for fn in public_functions:
        params = set(inspect.signature(fn).parameters)
        bad = params & _FORBIDDEN_PARAMETER_NAMES
        assert not bad, (
            f"{fn.__name__} accepts forbidden parameter(s) {bad}."
        )


# ---------------------------------------------------------------------------
# Coverage: convenience wrapper exercises the project SCM utility only
# ---------------------------------------------------------------------------


def test_generate_prior_for_scm_seed_uses_derived_selection_seed(
    calibration_seed_prior,
):
    """generate_prior_for_scm_seed must derive prior_selection_seed
    deterministically as ``PRIOR_SEED_BASE + scm_seed``."""
    spec, _adj = calibration_seed_prior
    direct = generate_prior_for_scm_seed(
        scm_seed=401, n_nodes=10, expected_edges=20, prior_k=PRIOR_K,
        prior_seed_base=PRIOR_SEED_BASE,
    )
    assert direct.prior_selection_seed == PRIOR_SEED_BASE + 401
    assert direct.scm_seed == 401
    assert direct == spec


def test_corruption_seed_derivation_on_calibration_seed(
    calibration_seed_prior,
):
    """corruption_seed = base + scm_seed + corruption_index."""
    spec, true_adj = calibration_seed_prior
    for fraction in CORRUPTION_GRID:
        cp = corrupt_prior(spec, true_adj, fraction)
        idx = corruption_index_for_fraction(fraction)
        assert cp.corruption_seed == CORRUPTION_SEED_BASE + 401 + idx
        # Round-trip the seed-derivation equation.
        assert cp.corruption_index == idx
        assert math.isclose(cp.corruption_fraction, fraction, abs_tol=1e-12)
