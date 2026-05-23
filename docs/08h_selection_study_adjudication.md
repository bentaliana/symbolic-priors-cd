# Selection-study adjudication: base-model choice before prior experiments

## 1. Purpose and scope

This report adjudicates the base-model selection study after the
Commit 10 held-out evaluation completed against the post-repair
calibration handoff. Its sole purpose is to apply the frozen
base-model selection protocol to the completed calibration and
held-out evidence and to record the resulting base-model decision
before any prior-loss experiments are started.

The report does not modify the frozen protocol, does not introduce
prior-loss experiments, does not change the held-out artefact, the
calibration artefact, or any per-fit record, and does not start any
new fit. It is documentation only.

## 2. Evidence sources

Two cryptographically-identified artefacts form the entire evidence
base for this adjudication:

- Calibration handoff:
  `calibration_run_hash_prefix = 4a67117a10b1`
  (40 calibration records under
  `results/model_selection/calibration/4a67117a10b1/`).
- Held-out evaluation:
  `heldout_run_hash_prefix = 88da382e8672`
  (25 held-out records under
  `results/model_selection/held_out/88da382e8672/`).

The held-out artefact validates against the held-out evaluation
schema and carries 25 records: 20 main records (4 selected
configurations x 5 held-out SCM seeds in `[301, 302, 303, 304, 305]`)
and 5 DCDI fit-RNG sensitivity records (`dcdi` / `centred_only` /
SCM seed 301 / `fit_rng` in `[43, 44, 45, 46, 47]`). Every record
reports `training_status = "converged"`, `graph_status = "valid_dag"`,
and `sampler_status = "available"`.

## 3. Calibration-selected configurations

Calibration ranked five candidates per `(model, condition)` cell
under the within-model lexicographic rule and emitted one rank-1
configuration per cell. The four selected configurations are:

| condition    | model | selected hash   | hyperparameters     |
| ------------ | ----- | --------------- | ------------------- |
| centred_only | dagma | `06ee98d13852`  | `lambda1 = 0.25`    |
| centred_only | dcdi  | `dd39d6325e7d`  | `reg_coeff = 0.1`   |
| standardised | dagma | `7b345b1b2e85`  | `lambda1 = 0.1`     |
| standardised | dcdi  | `16f92df3d6af`  | `reg_coeff = 0.3`   |

These four configurations are the only ones evaluated at the
held-out stage. Calibration did not, and must not, encode any
base-model decision; it only selected the rank-1 within-model
within-condition hyperparameter point.

## 4. Held-out main evidence

### 4.1 Cell-mean summary

The held-out evaluation produced the following per-cell aggregates
over the five held-out SCM seeds:

| condition    | model | mean SID | mean MMD primary    | mean SHD | mean runtime (s) |
| ------------ | ----- | -------- | ------------------- | -------- | ---------------- |
| centred_only | dagma | 4.2      | 0.005814472573953375 | 1.0      | about 1.05       |
| centred_only | dcdi  | 63.4     | 0.11588378616596792  | 30.4     | about 976.65     |
| standardised | dagma | 66.6     | 0.12135898121296962  | 25.8     | about 1.47       |
| standardised | dcdi  | 68.8     | 0.14197222340635784  | 29.6     | about 902.69     |

### 4.2 Per-seed variability

Cell means are reported above for compactness. The per-seed values
are listed here so the adjudication does not rest on means alone:

- `centred_only` / `dagma`: per-seed SID = `[0, 0, 0, 8, 13]`;
  median 0, mean 4.2. Three seeds achieved perfect structural
  recovery; two seeds had residual errors. Mean MMD is about
  `0.006`.
- `centred_only` / `dcdi`: per-seed SID = `[48, 51, 69, 71, 78]`;
  consistently high SID across all five seeds.
- `standardised` / `dagma`: per-seed SID = `[64, 64, 65, 69, 71]`;
  tightly clustered but with substantial structural error.
- `standardised` / `dcdi`: per-seed SID = `[45, 59, 76, 80, 84]`;
  wider spread, with consistently substantial error.

These per-seed lists are read directly from
`results/model_selection/held_out/88da382e8672/readout/per_seed_main.csv`
and from the held-out evaluation artefact's `per_seed_records`
arrays.

## 5. Application of the frozen selection logic

The frozen base-model selection rule is lexicographic:

1. SID is the primary criterion (lower is better).
2. MMD is the secondary criterion, applied as a tiebreaker only
   when SID differs by 10 percent or less between the candidates.
3. SHD is reported as a structural diagnostic.
4. Runtime and failure behaviour support feasibility but do not
   replace the metric ordering.

Applying the rule cell by cell to the held-out main evidence:

- In `centred_only`: DAGMA's mean SID (`4.2`) is far below DCDI's
  (`63.4`); the SID-margin tiebreaker is not invoked. Mean MMD
  primary is `~0.006` for DAGMA versus `~0.116` for DCDI. Mean SHD
  is `1.0` for DAGMA versus `30.4` for DCDI. Every metric in the
  cell favours DAGMA in this controlled d=10 ER2 linear-Gaussian
  selection study.
- In `standardised`: DAGMA's mean SID (`66.6`) is below DCDI's
  (`68.8`); the gap is well inside the 10 percent SID margin
  triggering the MMD tiebreaker. Mean MMD primary is `~0.121` for
  DAGMA versus `~0.142` for DCDI, again favouring DAGMA. Mean SHD
  is `25.8` for DAGMA versus `29.6` for DCDI.

DAGMA is therefore favoured over DCDI in both conditions by the
held-out SID, MMD, and SHD evidence in this controlled d=10 ER2
linear-Gaussian selection study. The `centred_only` / `dagma` cell
is the strongest evaluated cell on all three metrics.

The runtime gap (DAGMA: about 1.05 s and 1.47 s mean per fit; DCDI:
about 977 s and 903 s mean per fit, roughly three orders of
magnitude) does not enter the Section 2 lexicographic decision; it
is recorded as feasibility evidence only.

## 6. DCDI fit-RNG sensitivity addendum

The DCDI `centred_only` SCM seed-301 sensitivity probe at fit_rng
in `[43, 44, 45, 46, 47]` is a diagnostic. By construction it does
not enter the Section 2 lexicographic rule and does not change the
DAGMA-vs-DCDI ranking. It exists to answer one local question:
was the fixed-RNG DCDI seed-301 result an isolated unlucky fit?

| reference                  | SID  | MMD primary           | SHD  |
| -------------------------- | ---- | --------------------- | ---- |
| main fit_rng=42 at seed 301 | 78   | 0.12734844766889647   | 28   |
| sensitivity mean (43..47)  | 70.6 | 0.1468576398162918    | 30.0 |
| sensitivity SID range      | [61, 83] |                   |      |

The fixed-RNG `fit_rng=42` SID value (`78`) lies inside the
sensitivity SID range `[61, 83]`. The DCDI held-out result on
`centred_only` is therefore not explained away by a single unlucky
fixed-RNG run. This is the only inference the sensitivity addendum
licenses.

## 7. Base-model adjudication

Applying the frozen Section 2 lexicographic rule to the held-out
main evidence:

- **Selected base model for the next phase: DAGMA.**
- **Strongest evaluated condition: `centred_only`.**
- DCDI is not carried forward into the prior-loss main study
  because the held-out SID, MMD, and SHD evidence favours DAGMA in
  both conditions in this controlled d=10 ER2 linear-Gaussian
  selection study, including under the SID-margin tiebreaker in
  the `standardised` condition.
- The substantial runtime gap further supports excluding DCDI from
  the prior-sweep phase on feasibility grounds. Runtime is
  supporting evidence; it does not replace the metric-based
  selection logic, and the selection would stand on the metric
  evidence alone.

This adjudication does not claim DAGMA is universally superior to
DCDI. It states only that, in this controlled d=10 ER2
linear-Gaussian selection study, DAGMA is the appropriate base
model for the next phase under the frozen lexicographic rule
applied to the available held-out evidence.

## 8. Ceiling-effect and main-study implication

The `centred_only` / `dagma` cell is close to the structural
ceiling for the chosen graph regime: mean SID `4.2`, median SID
`0`, mean SHD `1.0`. Three out of five held-out seeds achieve
perfect structural recovery. The room left for structural
improvement on this cell is therefore limited by construction.

The `standardised` / `dagma` cell carries substantially more
headroom: mean SID `66.6`, mean SHD `25.8`, mean MMD primary about
`0.121`. DAGMA's structural recovery on this condition is far from
the ceiling in this controlled d=10 ER2 linear-Gaussian setting.

### 8.1 Implications for the prior-loss main-study contribution

The structural ceiling on `centred_only` / `dagma` constrains
where a prior-loss contribution can be measured. A separate
main-study scoping note should consider:

- **(a)** targeting the `standardised` condition, where DAGMA shows
  substantial structural error. This preserves the d=10 ER2
  methodological lineage of the selection study while providing
  empirical headroom on the primary structural criterion.
- **(b)** framing the contribution around sampler quality and
  unseen-intervention generalisation rather than only structural
  recovery. On `centred_only`, structural recovery is close to
  complete on the favoured cell, but MMD is nonzero and provides a
  distributional axis to measure improvement on. On `standardised`,
  MMD is much higher and provides corresponding headroom on the
  distributional axis as well.

This adjudication report does not decide the main-study stress
regime. It restricts itself to the base-model adjudication and
defers the prior-study design (condition, intervention set,
prior-corruption grid, ablations) to a separate scoping note that
will read this adjudication as input.

## 9. Operational and provenance lessons

- Local artefacts are authoritative for this study. The calibration
  handoff and the held-out artefact under `results/` are the source
  of truth for adjudication; no external tracking system was
  consulted.
- The curated calibration and held-out artefacts (the JSON
  envelopes plus per-fit records) are small, deterministic, and
  suitable for versioning alongside source code. Their readouts
  (`readout/` directories) summarise the same evidence for human
  inspection.
- Raw per-fit model outputs remain regeneratable from the
  Configuration and the on-disk run.json files; they should not be
  treated as primary thesis artefacts.
- The pre-Commit-10 FileExistsError calibration incident
  (`docs/08g_file_exists_error_incident.md`) informed the held-out
  orchestrator's failure-handling design: `FileExistsError` from
  the per-fit pipeline is reclassified as infrastructure failure
  and aborts the run, while ordinary fit exceptions are captured as
  degenerate records that the aggregator surfaces honestly.
- **Cryptographic provenance from calibration to held-out is
  preserved.** The held-out artefact embeds
  `parent_calibration_run_hash_full =
  4a67117a10b1e52ee247a11b28e43b03020ae3504d9008d7fa699b6d8e516598`,
  linking the held-out evaluation to the post-repair calibration
  artefact. The prior-loss main-study artefact should preserve the
  same chain by embedding the held-out artefact hash as a parent
  reference.

## 10. Conclusion

- Base-model selection is complete on the held-out evidence
  collected for this study.
- DAGMA is the selected base model for the next phase.
- Prior-loss work should proceed with a deliberately scoped
  DAGMA-only design that addresses the ceiling effect noted in
  Section 8.
- DCDI, additional model baselines, larger graph regimes, and
  additional prior types are deferred unless time remains after
  the scoped DAGMA-only prior-loss main study completes.

This adjudication applies only to the controlled d=10 ER2
linear-Gaussian selection study evaluated here. It is the formal
base-model decision required by the frozen protocol and is
documented before any prior-loss experiment begins.
