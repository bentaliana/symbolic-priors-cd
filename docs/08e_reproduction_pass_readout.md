# 08e_reproduction_pass_readout

## Status

Closed mechanically under docs/02 v1.8 Path B on 22/05/2026. Both DAGMA-linear and DCDI-G completed the reproduction_pass under the frozen reproduction-pass configs without infrastructure / wrapper / schema failure. No base-model selection is made or implied by this readout.

---

## 1. Scope

The reproduction_pass under docs/02 v1.8 Path B is a **thesis-cell compatibility / runner-sanity pass** on the 10-node ER2 selection cell. It is **not** strict paper reproduction and **not** base-model selection evidence. docs/02 Section 12 distinguishes the 10-node ER2 thesis selection cell from strict paper-reproduction cells; the latter remain a deferred optional sub-study under docs/02 Section 12.6.

### 1.1 Protocol and artefact conventions in force

- Artefacts were generated under docs/02 v1.8 Path B.
- docs/02 Section 5 item 2 (within-20% paper-reproduction disqualification) is **not directly evaluable** where no direct or explicitly frozen closely aligned paper target exists. For both candidates on the 10-node ER2 cell, no such target is frozen; item 2 does not fire and is reported as "not directly evaluable", not "passed".
- Per-run directories and the reproduction-pass summary directory both use the unified **12-character `configuration_hash` prefix** as the directory leaf.
- The `ReproductionPassSummary` dataclass and the on-disk `reproduction_pass_summary.json` contain **no prose `note` field**; they record 21 factual fields only.
- The **full 64-character lowercase hex `configuration_hash`** is retained as a content field in the summary, every per-run `run.json`, and every sibling `threshold_robustness.json`.

### 1.2 Configuration hashes (full 64-character digests)

- **DAGMA:** `15328a8f730f3bfc864ccf45f1aea38fbec2bc81dac8ff76485497ee2d676537`
- **DCDI:**  `826de9ce39d70f2ca2416523bf1526470b0f07734001ac05dbd2de00fb55ae0a`

Per-run directory leaves: `15328a8f730f` (DAGMA), `826de9ce39d7` (DCDI). Summary directory leaves match.

---

## 2. DAGMA reproduction_pass summary

- `reproduction_pass_status`: **passed**.
- `completed_run_count`: 3; `failed_run_count`: 0.
- `training_status_counts`: `{"converged": 3}`.
- `graph_status_counts`: `{"valid_dag": 3}`.
- `sampler_status_counts`: `{"available": 3}`.
- `threshold_robustness_available_count`: 3.
- SHD per seed: `[2, 0, 0]`.
- SID per seed: `[6, 0, 0]`.
- `mmd_primary` per seed: `[0.0041, 0.0045, 0.0030]` (more precisely 0.004051, 0.004451, 0.002996).
- Per-run runtime: ~0.87-1.15 s.

### Diagnostic note (DAGMA)

Seed 0 shows the largest threshold sensitivity across the DAGMA `[0.2, 0.3, 0.4]` triple: SHD `1 -> 2 -> 2` and SID `0 -> 6 -> 10`. Seeds 1 and 2 are stable across the triple (SHD = 0, SID = 0 at every threshold). This is recorded per docs/02 Section 7 "threshold robustness" reporting; it is not a selection criterion at reproduction_pass.

---

## 3. DCDI reproduction_pass summary

- `reproduction_pass_status`: **passed**.
- `completed_run_count`: 3; `failed_run_count`: 0.
- `training_status_counts`: `{"converged": 3}`.
- `graph_status_counts`: `{"valid_dag": 3}`.
- `sampler_status_counts`: `{"available": 3}`.
- `threshold_robustness_available_count`: 3.
- SHD per seed: `[26, 25, 31]`.
- SID per seed: `[65, 41, 65]`.
- `mmd_primary` per seed: `[0.0770, 0.0111, 0.1068]`.
- Per-run runtime: ~750-886 s.

DCDI training-state diagnostics per seed:

| seed | n_iterations | final_h | final_gamma | final_mu | first_stop |
| --- | --- | --- | --- | --- | --- |
| 0 | 86800 | 6.60e-09 | 3.17e+05 | 3.60e+08 | 86400 |
| 1 | 81000 | 8.25e-09 | 1.76e+04 | 1.13e+07 | 80600 |
| 2 | 80900 | 5.18e-09 | 2.15e+06 | 7.21e+08 | 80500 |

All three runs converged below `h_threshold = 1e-08` well before the `dcdi_num_train_iter = 300000` cap.

### Diagnostic notes (DCDI)

- DCDI shows substantially higher SHD / SID / MMD than DAGMA at the reproduction anchor. See Section 4.
- Seed 2 is a coherent outlier: thresholded-adjacency edge count `10` against a ground-truth ER2 cell with `expected_edges = 20` (sparse graph); MMD primary `0.1068` (the highest of the three seeds); validation-NLL last-3-value mean ~2.27 (the highest of the three); per-intervention median-heuristic bandwidths ~514-540 (substantially larger than seeds 0 and 1 and than any DAGMA run). The four signals are consistent with one another and arise on the same seed; recorded as a coherent diagnostic pattern, not as a selection signal at reproduction_pass.

---

## 4. DCDI-vs-DAGMA comparison at the reproduction anchor

| metric | DAGMA per seed | DCDI per seed |
| --- | --- | --- |
| SHD | `[2, 0, 0]` | `[26, 25, 31]` |
| SID | `[6, 0, 0]` | `[65, 41, 65]` |
| `mmd_primary` | `[0.0041, 0.0045, 0.0030]` | `[0.0770, 0.0111, 0.1068]` |
| DCDI/DAGMA MMD ratio | --- | `[19.0x, 2.5x, 35.6x]` |

This comparison is at the **reproduction anchor only**, namely DAGMA `lambda1 = 0.05` and DCDI `reg_coeff = 0.1`. It is **not informative** about either candidate's behaviour at other sparsity values; behaviour across each candidate's frozen Phase B sparsity grid (`docs/02` Section 3.3 / Section 9) is the subject of Phase B calibration, not reproduction_pass.

### 4.1 DCDI validation-NLL behaviour

- The top-level `validation_nll` field in `run.json` is `null` for all three DCDI runs.
- The per-iteration trajectory is preserved under `wrapper_diagnostics.convergence_info.validation_nll_history` (length 810-869 entries per run, recorded at the `validation_nll_stop_crit_win = 100` cadence).
- Last-3-value mean per seed: seed 0 ~1.98; seed 1 ~1.57; seed 2 ~2.27. All values finite.
- The trajectory shape is non-monotonic: lower values appear earlier, followed by later increases as the acyclicity-penalty pressure dominates the augmented-Lagrangian schedule. This shape is consistent with the C-P15 pilot finding documented in `docs/08d_dcdi_training_budget_pilot.md`.
- Recorded as diagnostic context, not a blocker.

---

## 5. Section 5 disqualification interpretation

- **Item 1** (install/run failure within 5 working days): **not triggered**. Both candidates installed and produced complete artefacts.
- **Item 2** (within-20% paper-reproduction comparison): **not directly evaluable** under Path B for the current configs; **not "passed"**. The 10-node ER2 thesis selection cell has no direct DAGMA-paper row at d = 10 and no DCDI Table 7 row at d = 10, e = 2; per docs/02 Section 12.4 the comparison is recorded as "not directly evaluable" and item 2 does not fire.
- **Item 3** (NaN / divergence / non-converged training in more than 50% of seeds): **not triggered**. 0/3 failed for both candidates; both candidates have `training_status` in the clean variant for every seed; DCDI converged below `h_threshold` in all three runs.
- **Item 4** (intervention outputs unusable without ad hoc undocumented modifications): **not triggered**. Every intervention on every run reports `mmd_status = "available"` with finite `mmd_value`; no ad hoc repair applied.
- **Item 5** (prior-injection smoke test): **out of scope** for this stage; not evaluated by reproduction_pass.

---

## 6. No base-model selection

This readout makes **no base-model selection**. Section 4 quantifies the DCDI-vs-DAGMA gap at the reproduction anchor only; it does not adjudicate selection. Base-model selection is produced by Phase B calibration plus held-out evaluation under the lexicographic decision rule in docs/02 Section 2, not by reproduction_pass.

DCDI's poor reproduction-anchor performance is **not yet evidence to reject DCDI** because Phase B calibration's pre-registered `reg_coeff` grid `{0.01, 0.03, 0.1, 0.3, 1.0}` may produce a selected DCDI configuration that improves on the anchor `reg_coeff = 0.1` evaluated here.

---

## 7. C-P11 reapplication

The DCDI MMD gap recorded in Section 4 is **relevant context** for the deferred C-P11 real-budget reapplication probe (DCDI sampler-quality diagnostic at `dcdi_num_train_iter = 300000` on a fresh 10-node ER2 fixture, per docs/02 Section 7 "C-P11 real-budget reapplication policy"). It does **not replace** that probe. The C-P11 rerun **remains required** before any held-out interpretation of DCDI sampler quality is accepted into the selection-study report.

---

## 8. Next required decisions before Commit 9 (calibration runner)

Three protocol decisions remain open before Phase B calibration begins:

- **Eligible-node intervention-set policy.** The reproduction_pass used `do(X_0 = +/-2)` as Phase A-only smoke / sanity coverage (recorded 20/05/2026). Phase B and held-out evaluation need the full eligible-node intervention-set policy (target-node selection rule, intervention values, intervention count per cell) frozen.
- **DCDI fit-RNG convention beyond reproduction_pass.** The reproduction_pass used `seed_torch = seed_numpy = 42` (recorded 20/05/2026 docs/03 entry). Whether this single integer remains across Phase B configurations and held-out evaluation, or whether the calibration / held-out configs carry per-stage fit-RNG values, needs an explicit decision.
- **Selected-configuration artefact path.** Phase B produces a "selected configuration per model per condition" record consumed by held-out evaluation (per docs/08 Commit 9 / Commit 10). The artefact's on-disk path, filename, and schema are not yet specified.

---

## What this reproduction_pass does not establish

- **No base-model selection** is made or implied. Selection is produced by Phase B calibration plus held-out evaluation under docs/02 Section 2, not by this stage.
- **No statement about paper reproduction** is made for either candidate. docs/02 Section 5 item 2 is reported as "not directly evaluable" under Path B for the current configs, not "passed".
- **The DCDI-vs-DAGMA comparison in Section 4 is at the reproduction anchor only** (DAGMA `lambda1 = 0.05`, DCDI `reg_coeff = 0.1`). It does **not** establish either candidate's behaviour at the sparsity values that Phase B calibration will explore for that candidate.
- **DCDI's MMD on this cell does not by itself resolve the C-P11 reapplication question.** The C-P11 rerun on a fresh 10-node ER2 fixture remains required per docs/02 Section 7.
- **The DCDI seed-2 outlier pattern** (sparse graph + high MMD + upper-end validation-NLL + large bandwidth) is consistent with the **type** of cross-seed instability the project's H4 hypothesis will later test. The reproduction_pass is **not** the venue for H1-H4 evidence and **no H4 claim** is made or supported here.
