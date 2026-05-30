# Neuro-Symbolic Causal Discovery: Integrating Statistical and Symbolic AI

[![Final Year Project](https://img.shields.io/badge/FINAL_YEAR_PROJECT-Department_of_AI_%7C_2026-1f6feb?labelColor=555555)](https://github.com/bentaliana/symbolic-priors-cd)
[![tests](https://github.com/bentaliana/symbolic-priors-cd/actions/workflows/tests.yml/badge.svg)](https://github.com/bentaliana/symbolic-priors-cd/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-1f6feb)](https://www.python.org/downloads/release/python-3120/)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

B.Sc. I.T. (Hons.) in Artificial Intelligence thesis project that adds **confidence-weighted soft structural priors** to **differentiable causal discovery**. The implementation is a soft forbidden-edge prior wrapper around **DAGMA**-linear, compared against three baselines on a 10-node ER2 linear-Gaussian regime under observational-only training: prior-free DAGMA, matched-L1 DAGMA (sparsity-matched generic regulariser), and hard-exclusion DAGMA (DAGMA's native `exclude_edges` projection). The corruption axis sweeps a 5x5 grid of `(confidence, corruption_fraction)`, metrics are SID (primary), MMD (secondary, distributional), SHD, and thresholded edge count.

Author: **Ben Taliana** (`ben.taliana.23@um.edu.mt`)<br>
Supervisor: **Prof. Alexiei Dingli**.

---

## The soft-prior penalty term

The soft prior is added to DAGMA's hand-coded gradient as a single extra term per Adam iteration; the rest of DAGMA's path-following optimisation is unchanged. The penalty form is

$$L_{\text{prior}}(W) = \lambda_{\text{prior}} \sum_{(i,j) \in F} c_{ij} W_{ij}^2$$

and its gradient, added to DAGMA's assembled `Gobj` immediately before the Adam update, is

$$\nabla_W L_{\text{prior}}(W) = 2\lambda_{\text{prior}} (C \odot W)$$

where $W$ is the continuous weighted adjacency matrix learned by DAGMA, $F$ is the seed-specific forbidden-edge set, $C$ is the per-entry confidence mask ($C_{ij} = c$ on $(i,j) \in F$, zero elsewhere, zero on the diagonal), $c \in [0, 1]$ is the global confidence value for the run, and $\odot$ is the elementwise (Hadamard) product.

Three properties govern the interpretation of the results:

- The gradient is **not** scaled by DAGMA's central-path coefficient `mu`. It acts at constant strength across all warm-up and final stages.
- The penalty is **soft**: it adds a continuous gradient proportional to `W_ij`, it does not project anything to zero. The data score can equilibrate against it at non-zero `W`.
- At `confidence = 0` the mask is the zero matrix and the soft-prior fit reduces exactly to prior-free DAGMA. This is enforced as a smoke gate in the test suite and verified empirically in the saved records.

Frozen protocol values used by the main study (set by held-out adjudication and matched-L1 calibration):

| constant                   |      value | source                                                          |
| -------------------------- | ---------: | --------------------------------------------------------------- |
| DAGMA `lambda1` (backbone) |     `0.10` | held-out selection at `lambda1 = 0.10`                          |
| DAGMA `warm_iter`          |    `20000` | DAGMA paper anchor (Section C.1.1)                              |
| DAGMA `max_iter`           |    `70000` | same                                                            |
| project threshold          |     `0.30` | applied as `abs(W) >= 0.30`                                     |
| `lambda_prior`             |     `2e-4` | smallest non-degenerate value from the lambda_prior calibration |
| `matched_l1_lambda1`       |     `0.10` | match-by-sparsity calibration `71bfe6629b9d`                    |
| evaluation seeds           | `501..507` | main-study pool                                                 |
| calibration seeds          | `401, 402` | main-study pool                                                 |

---

## Requirements and environment

- **Python:** `3.12` (pinned by `pyproject.toml`: `>=3.12,<3.13`).
- **Operating systems tested:** Windows 11 (PowerShell) and Ubuntu (CI).
- **Dependency pinning:** top-level pins in `pyproject.toml`; full transitive lockfile in `requirements-lock.txt` (UTF-8, LF).
- **Disk:** saved records and artefacts occupy roughly 1.2 GB.

### `pip install` alone will NOT reproduce results

The wrapper code does not use the pip-installed DAGMA package. At import time, `src/symbolic_priors_cd/wrappers/_dagma_utils.py` and `_dcdi_utils.py` insert a project-local source tree at the head of `sys.path` so that DAGMA and DCDI resolve to the pinned source clones under `external/source_inspection/`. A bare `pip install -e .` without the source clones will raise `ImportError` from the first import that touches the wrappers.

You must clone the upstream sources to the exact paths below before running anything that imports the wrappers (which includes the entire test suite and every experiment entry point).

### Awkward dependency call-outs

1. **`gadjid==0.1.0`** (the SID backend) installs from a prebuilt abi3 wheel on Windows + Python 3.12 with `pip install --only-binary=:all: gadjid==0.1.0`. On other platforms abi3 should still find a matching wheel; if `pip` falls back to a source build, a Rust toolchain (`rustup`) is required. Runtime dependency: `numpy` only. MPL-2.0 licence; the project does not vendor or modify `gadjid` source.

2. **DCDI source** is loaded as source from `external/source_inspection/dcdi/` via a `sys.path` insertion in `_dcdi_utils.py`. DCDI is **not** installed as a pip package. Only `dcdi.models.learnables`, `dcdi.dag_optim`, and `dcdi.utils.penalty` are imported, so DCDI's own `train`-time dependencies (`cdt`, R, GPyTorch) are not pulled in. A shallow clone of DCDI is sufficient.

### Required external clones

```
external/
└── source_inspection/
    ├── dagma/           # https://github.com/kevinsbello/dagma
    └── dcdi/            # https://github.com/slachapelle/dcdi
```

Both must exist at exactly those paths.

---

## Installation, copy-pasteable

Commands below are PowerShell. For bash / zsh, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` and `Remove-Item -Recurse -Force` with `rm -rf`.

```powershell
# 1. Clean previous venv if any
Remove-Item -Recurse -Force .venv

# 2. Create a Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Clone the pinned upstream sources (required)
git clone --depth 1 https://github.com/kevinsbello/dagma external/source_inspection/dagma
git clone --depth 1 https://github.com/slachapelle/dcdi  external/source_inspection/dcdi

# 4. Install the locked environment, then the project itself
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --no-deps

# 5. Verify the imports resolve to the pinned sources
.\.venv\Scripts\python.exe -c "import symbolic_priors_cd, dagma, gadjid, torch, numpy, scipy, networkx, pandas, matplotlib; print('OK')"
```

The `--no-deps` flag on step 4's second pip command is intentional: step 4's first command has already pinned every transitive version from the lockfile, and `--no-deps` prevents pip from re-resolving.

---

## Quickstart smoke test

Two one-liners that exercise the full machinery without running the heavy grid.

```bash
# (a) Metric primitives only
python -m pytest tests/test_interventional_metrics.py tests/test_structural_metrics.py -q

# (b) One soft-prior fit on seed 501 and print SID / SHD against the true graph
python - <<'EOF'
import numpy as np
from symbolic_priors_cd.data.scm_generator import (
    generate_linear_gaussian_scm, sample_observational,
)
from symbolic_priors_cd.wrappers._soft_prior_dagma import SoftPriorDagmaLinear
from symbolic_priors_cd.wrappers.preprocessing import StandardisedTransform
from symbolic_priors_cd.metrics import sid_score, shd

scm = generate_linear_gaussian_scm(n_nodes=10, expected_edges=20, seed=501)
X   = sample_observational(scm, n_samples=1000, rng=np.random.default_rng(501))
pre = StandardisedTransform(); pre.fit(X)
Xm  = pre.transform(X)
mask = np.zeros((10, 10)); mask[0, 1] = 1.0
m   = SoftPriorDagmaLinear(loss_type="l2", lambda_prior=2e-4, confidence_mask=mask)
W   = m.fit(Xm.copy(), lambda1=0.10, warm_iter=2000, max_iter=4000,
            w_threshold=0.0, T=4, mu_init=1.0, mu_factor=0.1,
            s=[1.0, 0.9, 0.8, 0.7], lr=3e-4)
A   = (np.abs(W) >= 0.3).astype(bool)
print("SID =", sid_score(A, scm.adjacency), "SHD =", shd(A, scm.adjacency))
EOF
```

Either output proves the install is complete and the full pipeline will run end-to-end.

---

## How to reproduce the results

Every command below writes to `results/...` and is content-addressable: each stage computes a 12-character hex hash from its scientific-identity inputs and uses that hash as its output directory name. Active hashes:

- **Base-model held-out adjudication:** `88da382e8672`
- **Base-model calibration:** `4a67117a10b1`
- **Matched-L1 calibration:** `71bfe6629b9d`
- **Main evaluation:** `166c792c43bc`
- **Prior structural relevance diagnostic:** `6f660aaeef3d`
- **Oracle prior relevance diagnostic:** `079fda7ac4f4`

### Stage 1 - Base-model selection (DAGMA vs DCDI)

The CLI exposes reproduction-pass and calibration-enumeration entry points only. Calibration fits and held-out fits are driven through the public Python API (`run_calibration`, `run_held_out_evaluation`); the runner scripts used during this project are preserved under `archive/`. The DCDI clone is required for this stage.

```bash
# Reproduction pass (Phase A):
python -m experiments.selection_study.run \
    --phase reproduction_pass \
    --config experiments/selection_study/configs/reproduction

# Calibration enumeration and real-study guard (no fits at this entry point):
python -m experiments.selection_study.run \
    --phase calibration \
    --config experiments/selection_study/configs/calibration

# Render base-model selection figures from saved artefacts:
python -m experiments.selection_study.render_base_model_selection_readout \
    --output-root .
```

Outputs land under:

```
results/model_selection/calibration/4a67117a10b1/
results/model_selection/held_out/88da382e8672/
```

### Stage 2 - Matched-L1 calibration

```bash
python -m experiments.main_study.calibrate_matched_l1 \
    --output-root . \
    --parent-hash 88da382e8672650e44f44e675011dda1a81868c9075acb86faef6c6caf23fd17
```

Approximate runtime: ~5 min. Selected `matched_l1_lambda1 = 0.10`. Output: `results/main_study/calibration/matched_l1/71bfe6629b9d/`.

### Stage 3 - Main evaluation (224-fit grid)

```bash
python -m experiments.main_study.run_main_evaluation \
    --output-root . \
    --parent-heldout-run-hash-full 88da382e8672650e44f44e675011dda1a81868c9075acb86faef6c6caf23fd17
```

Approximate runtime: ~10 min on a modern laptop CPU (no GPU required). Grid composition: 7 prior_free + 7 matched_l1 + 175 soft_frobenius (5x5x7) + 35 hard_exclusion (5x7) = 224 fits. Outputs:

```
results/main_study/166c792c43bc/                      records + per-run artefacts
results/main_study/main_evaluation/166c792c43bc/      summary + readout directory
```

### Stage 4 - Readout tables and figures

```bash
python -m experiments.main_study.readout \
    --output-root . --main-evaluation-run-hash12 166c792c43bc

python -m experiments.main_study.render_readout_figures \
    --output-root . --main-evaluation-run-hash12 166c792c43bc
```

Outputs land in `results/main_study/main_evaluation/166c792c43bc/readout/` and its `figures/` subdirectory (`fig01..fig08`).

### Stage 5 - Exploratory structural diagnostics

```bash
# Prior structural relevance (saved-records only, ~5 min):
python -m experiments.main_study.exploratory.prior_structural_relevance \
    --output-root . --main-evaluation-run-hash12 166c792c43bc

# Oracle prior relevance (saved-records only, ~5 min):
python -m experiments.main_study.exploratory.oracle_prior_relevance \
    --output-root . \
    --main-evaluation-run-hash12 166c792c43bc \
    --prior-relevance-analysis-hash12 6f660aaeef3d

# Presentation-only diagnostic figures (11 figures):
python -m experiments.main_study.exploratory.render_prior_relevance_diagnostics \
    --output-root .
```

### Stage 6 - Notebooks

The three notebooks under `notebooks/` are labelling-only. They read saved records and produce no new fits and no new metric computations.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/base_model_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/main_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/prior_relevance_diagnostics.ipynb
```

Notebook -> output mapping:

| notebook                            | reads from                                     | regenerates                                                                        |
| ----------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| `base_model_selection.ipynb`        | `88da382e8672`, `4a67117a10b1`                 | 7 base-model selection figures and 2 side tables                                   |
| `main_evaluation.ipynb`             | `166c792c43bc`                                 | `fig01..fig08`, baseline / paired / degradation / correlation tables               |
| `prior_relevance_diagnostics.ipynb` | `166c792c43bc`, `6f660aaeef3d`, `079fda7ac4f4` | `fig01..fig11`, prior-target overlap, error decomposition, oracle scenario summary |

Executing these three notebooks regenerates every thesis figure and table from saved artefacts without re-running Stages 1-5.

---

## Repository structure (main results, library, runnable studies)

```
symbolic-priors-cd/
├── README.md
├── pyproject.toml                      # exact dependency pins
├── requirements-lock.txt               # frozen transitive environment (UTF-8 / LF)
├── .github/workflows/tests.yml         # CI: clones upstream sources, installs, runs pytest
├── src/symbolic_priors_cd/             # library code
│   ├── data/                           # SCM generator, interventions, sampling
│   ├── metrics/                        # SID (gadjid backend), MMD, SHD, validation
│   └── wrappers/
│       ├── _soft_prior_dagma.py        # soft-prior penalty injection (the contribution)
│       ├── _dagma_fit.py, _dagma_utils.py, _dagma_sampling.py
│       ├── _dcdi_utils.py, _dcdi_training.py, _dcdi_sampling.py
│       ├── _graph_status.py            # valid_dag / cyclic / bidirected / self_loop classifier
│       ├── dagma.py, dcdi.py           # public wrappers (DAGMAWrapper, DCDIWrapper)
│       └── preprocessing.py            # StandardisedTransform, CentredOnlyTransform
├── experiments/
│   ├── selection_study/                # base-model selection (DAGMA vs DCDI)
│   │   ├── run.py                      # CLI: --phase {reproduction_pass, calibration}
│   │   ├── pipeline.py, calibration.py, held_out.py, reproduction_pass.py
│   │   ├── render_base_model_selection_readout.py
│   │   └── configs/                    # reproduction + calibration grids
│   └── main_study/                     # main evaluation + diagnostics
│       ├── schema.py                   # MainStudyConfig, configuration_hash, protocol DAGMA factory
│       ├── priors.py                   # PriorSpec / CorruptedPriorSpec, F generation, C
│       ├── workloads.py                # plan enumeration; matched_l1 lambda1 replacement
│       ├── runner.py, executor.py      # orchestration + single-run executor
│       ├── backends.py                 # DataBundleLoader, DAGMABackend, SoftPriorBackend, RealMetricBackend
│       ├── records.py, run_io.py       # MainStudyRunRecord, atomic persistence
│       ├── calibration_lambda_prior.py # lambda_prior calibration
│       ├── calibrate_matched_l1.py     # matched-L1 calibration entry point
│       ├── run_main_evaluation.py      # main-evaluation entry point
│       ├── readout.py                  # statistics + tables
│       ├── render_readout_figures.py   # figure renderer
│       └── exploratory/
│           ├── prior_structural_relevance.py        # prior-target relevance entry point
│           ├── oracle_prior_relevance.py            # oracle FP/FN entry point
│           └── render_prior_relevance_diagnostics.py # presentation readout
├── notebooks/
│   ├── base_model_selection.ipynb
│   ├── main_evaluation.ipynb
│   └── prior_relevance_diagnostics.ipynb
└── results/                            # all persisted records and readouts (see next tree)
```

Where the active results live:

```
results/
├── model_selection/
│   ├── calibration/4a67117a10b1/
│   └── held_out/88da382e8672/
│       ├── heldout_evaluation.json
│       ├── records/                                                  # 25 per-run JSONs
│       └── readout/
│           ├── main_summary.csv, per_seed_main.csv, sensitivity_summary.csv, status_summary.csv
│           ├── heldout_readout.md, selected_configurations_table.csv, selection_summary_table.csv
│           └── base_model_selection_figures/                         # 7 PNGs
└── main_study/
    ├── 71bfe6629b9d/                                                 # matched-L1 calibration records
    ├── 166c792c43bc/                                                 # main-evaluation records + artefacts (224 fits)
    ├── calibration/matched_l1/71bfe6629b9d/                          # matched-L1 calibration summary
    ├── main_evaluation/166c792c43bc/
    │   ├── main_evaluation_execution_summary.json|.md
    │   ├── main_evaluation_workload_status.csv
    │   └── readout/
    │       ├── main_evaluation_flat_records.csv (224 rows)
    │       ├── baseline_comparison.csv, cell_summary.csv, paired_seed_comparisons.csv
    │       ├── degradation_summary.csv, metric_correlations.csv
    │       ├── forbidden_edge_engagement{,_summary}.csv, reference_forbidden_edge_comparison.csv
    │       ├── per_intervention_mmd_{long,summary}.csv (4480 + 640 rows)
    │       ├── status_summary.csv, validation_summary.json, statistics_summary.json
    │       ├── readout_summary.md
    │       └── figures/                                              # fig01..fig08 PNGs
    └── exploratory/
        ├── prior_structural_relevance/6f660aaeef3d/                  # prior-target relevance
        ├── oracle_prior_relevance/079fda7ac4f4/                      # oracle FP/FN
        └── prior_relevance_diagnostics/                              # presentation readout
            └── figures/                                              # fig01..fig11 PNGs
```

---

## Repository structure (secondary: probes and the rest of the tests)

Not required for reproducing thesis figures. Used for source-inspection audits, early-validation replays, and the regression test suite.

```
symbolic-priors-cd/
├── external/source_inspection/
│   ├── dagma/                          # cloned source (https://github.com/kevinsbello/dagma)
│   └── dcdi/                           # cloned source (https://github.com/slachapelle/dcdi)
├── inspection/probes/                  # numbered source-inspection and runtime probes
│   ├── c_p1_dcdi_imports.py
│   ├── c_p2_dcdi_no_cdt.py
│   ├── c_p3_dcdi_instantiate.py
│   ├── c_p4_dcdi_native_edge.py
│   ├── c_p5_dcdi_forward.py
│   ├── c_p6_dcdi_conditional.py
│   ├── c_p7_dcdi_ancestral.py
│   ├── c_p8_dcdi_determinism.py
│   ├── c_p9_dcdi_mask.py
│   ├── c_p10_equivalence_calibration.py
│   ├── c_p11_dcdi_sampler_quality_diagnostic.py
│   ├── c_p12_equal_variance_identifiability_check.py
│   ├── c_p13_dagma_sampler_quality_diagnostic.py
│   ├── c_p14_mmd_shd_reference_crosscheck.py
│   └── c_p15_dcdi_training_budget_pilot.py
└── tests/
    ├── main_study/                     # 1,171 tests for the main-study modules
    ├── selection_study/                # 30 tests for the selection-study readout
    ├── test_dagma_wrapper_*.py         # 9 wrapper-level test files
    ├── test_dcdi_wrapper_*.py          # 7 DCDI wrapper test files
    ├── test_calibration_*.py           # 7 selection-study calibration-stack test files
    ├── test_held_out_*.py              # 5 held-out adjudication test files
    ├── test_interventional_metrics.py  # 49 tests: MMD and SID
    ├── test_structural_metrics.py      # 26 tests: SHD
    ├── test_sanity_checks.py           # ground-truth compatibility gate
    ├── test_scm_generator.py           # data layer
    ├── test_interventions.py, test_sampling.py
    ├── test_loader.py, test_pipeline.py
    ├── test_real_study.py, test_preflight.py
    ├── test_reproduction_pass_runner.py, test_selection_runner_scaffolding.py, test_selection_artefact.py
    ├── test_config_schema.py
    ├── test_run_identity.py
    └── test_threshold_robustness.py    # 28 tests: offline threshold recomputation
```

---

## Configuration and the hashing scheme

Every run is identified by a `MainStudyConfig` (main study) or `RealStudyConfig` (selection study) dataclass. The dataclass is frozen, validated at construction, and serialised through a canonical JSON form. A SHA-256 over that canonical form gives a **configuration hash**; the first 12 hex characters become the run's `configuration_hash_prefix` and form part of the directory name.

Consequences:

- **Re-running an identical config resolves to the same hash.** The atomic writer in `run_io.py` uses `mode="raise"` by default and refuses to overwrite.
- **Changing any scientific-identity input changes the hash.** Scientific identity includes the protocol DAGMA backbone, the matched-L1 calibrated value, `lambda_prior`, the seed pool, and the grid definition.
- **`code_version` is captured for provenance but not hashed**, so two runs at the same scientific identity but different commit SHAs produce the same hash and are not duplicated.

What every successful main-study fit stores (per `experiments/main_study/executor.py`):

| artefact                                      | content                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------------ |
| record JSON                                   | full config, run hashes, run_id, statuses, SID/SHD/MMD, runtime, code_version, paths |
| `continuous_w.npz`                            | DAGMA pre-threshold continuous adjacency `W`                                         |
| `thresholded_adjacency.npz`                   | bool, `abs(W) >= 0.3`                                                                |
| `true_adjacency.npz`                          | ground-truth SCM adjacency                                                           |
| `interventions_mmd.json`                      | per-intervention MMD + bandwidth sweep                                               |
| `confidence_mask.npz` (soft_frobenius only)   | the actual `C` used in the fit                                                       |
| `prior_edge_set_clean.json` (soft / hard)     | reconstructed clean `F`                                                              |
| `prior_edge_set_corrupted.json` (soft / hard) | actually-applied `F_corrupted`                                                       |
| `per_edge_labels.json` (soft / hard)          | per-edge `true_negative_retained` or `true_positive_corrupted_replacement`           |

Records are loaded by `record_from_json` in `experiments/main_study/records.py`; the loader validates the full schema and rejects unknown or missing fields.

---

## Tests

```bash
# Full suite (about 6 minutes on a modern laptop):
python -m pytest -q

# Focused subsets:
python -m pytest tests/main_study/test_soft_prior_dagma.py -q
python -m pytest tests/test_interventional_metrics.py -q
python -m pytest tests/main_study/test_calibrate_matched_l1.py -q
```

**Total: 2,430 tests collected** (verified live with `pytest --collect-only -q`). One SID scaffold is skipped intentionally; everything else is active. The count is parametrised: e.g. `test_records.py` reports 202 because each `@pytest.mark.parametrize` case is a separate test ID.

What the suite covers:

- **Soft-prior mechanism** (`tests/main_study/test_soft_prior_dagma.py`): exact gradient formula `G_prior = 2 * lambda_prior * (C o W)`; no `mu` argument; zero-mask and zero-lambda equivalence with prior-free DAGMA; shrinkage direction without clamping; validation gates on `lambda_prior` and the mask.
- **Hard exclusion** (`tests/main_study/test_hard_exclusion.py`): exact-zero post-projection on the continuous `W`; strict tuple-of-tuples validation (upstream DAGMA's own validator is broken and this wrapper backstops it).
- **Metric primitives** (`tests/test_interventional_metrics.py`, `tests/test_structural_metrics.py`): MMD on hand-checkable fixtures including the negative-value case; SID identity, argument order, edge-direction pin via `gadjid==0.1.0`, dtype rejection.
- **Schema and hashing** (`tests/main_study/test_schema.py`, `tests/test_config_schema.py`, `tests/test_run_identity.py`): frozen-dataclass invariants; deterministic configuration hashing; method-family rules; matched-L1 `lambda1` replacement; hard-exclusion `exclude_edges` equality with `forbidden_edges`.
- **Runners and persistence** (`tests/main_study/test_runner.py`, `test_executor.py`, `test_run_io.py`, `test_records.py`): atomic writes, resumability modes, status taxonomy, artefact path validation.
- **Calibration stacks** (`tests/main_study/test_calibrate_matched_l1.py`, `tests/main_study/test_lambda_prior_calibration.py`, selection-study calibration tests): protocol-version pinning, seed-population gates, candidate-grid pinning, selection-rule pinning.
- **Readout and figures** (`tests/main_study/test_readout.py`, `test_render_readout_figures.py`): output CSV schemas, expected row counts, correlation coverage, validation summary.

Continuous integration runs the full suite from a fresh clone on Ubuntu Python 3.12 against the locked dependencies and both upstream source clones. The `tests` badge at the top of this README reflects the latest run.

---

## Determinism and seeds

Every stochastic component is explicitly seeded. Seed pools (disjoint by construction):

| pool                           | seeds           | use                       |
| ------------------------------ | --------------- | ------------------------- |
| reproduction (selection study) | `101, 102, 103` | reproduction pass         |
| calibration (selection study)  | `201, 202`      | base-model calibration    |
| held-out (selection study)     | `301..305`      | held-out adjudication     |
| main-study calibration         | `401, 402`      | lambda_prior + matched-L1 |
| main-study evaluation          | `501..507`      | headline grid             |

Derived seeds inside a run:

- **Prior selection** (per SCM seed): `prior_selection_seed = 9000 + scm_seed`.
- **Corruption** (per SCM seed + corruption-grid index): `corruption_seed = 9100 + scm_seed + corruption_index`.
- **Per-intervention ground-truth sampling**: `gt_seed = scm_seed + 10000 + intervention_index`.
- **Per-intervention model sampling**: `model_seed = scm_seed + 20000 + intervention_index`.

DAGMA itself does not call any global RNG; it is deterministic for fixed input and fixed configuration (verified in `tests/test_dagma_wrapper_*.py`). The wrappers do not seed any global state either; randomness is parameterised by explicit RNGs created via `np.random.default_rng(seed)`.

Reproducibility caveats:

- Saved numerics are reproducible on the tested environment (Windows 11 / Python 3.12.0 and Ubuntu CI / Python 3.12). BLAS implementation and thread count can change the last few ULPs on summation; SID, SHD, and edge_count are integer and unaffected.
- The unbiased RBF MMD estimator can take small negative finite values; these are preserved as-is (the readout never clips). This is mathematical, not numerical, and is pinned by `test_mmd_unbiased_can_be_negative`.
- Across platforms, exact bit-reproducibility is not claimed. Approximate reproducibility (means and per-seed values matching to 4-5 decimal places on the protocol metrics) is robust across the tested environments.

---

## Provenance

An earlier set of results was produced under an off-protocol DAGMA backbone (`lambda1 = 0.05`, `warm_iter = 30000`, `max_iter = 60000`). The corrected backbone is `lambda1 = 0.10`, `warm_iter = 20000`, `max_iter = 70000`. The superseded artefacts remain on disk under `_superseded_lambda1_0p05_*/` directories with `README.md` provenance markers; they are not referenced by any active code, notebook, or result table. Do not cite them as active results.

---

## Licence

Code: MIT. The `gadjid` runtime dependency is MPL-2.0; the project uses it as a runtime library and does not modify or distribute its source.
