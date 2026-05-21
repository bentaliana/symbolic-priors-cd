# 08b: Selection-study constants and fairness audit

## Status

Read-only audit produced before Commit 7. This document is not a
re-specification of the selection-study protocol. `docs/02_base_model_selection.md`
is the currently frozen operational protocol that the running code and
configurations follow; this audit treats `docs/02` as the operational
contract, not as an epistemic source of truth. Where paper evidence,
source inspection, or runtime probes support, contradict, or refine a
choice frozen in `docs/02`, the audit records the underlying evidence
and identifies any future `docs/02` amendment that the evidence would
justify. The audit itself amends nothing.

Evidence types used in this audit:

- **external-paper** support: a published reference reachable inside
  this repository asserts the claim. The DAGMA paper (Bello, Aragam,
  Ravikumar, NeurIPS 2022) and the DCDI paper (Brouillard,
  Lachapelle, Lacoste-Julien, Lacoste, Drouin, NeurIPS 2020) are
  both reachable at `papers/DAGMA.pdf` and `papers/DCDI.pdf` and are
  cited by section / appendix number below.
- **source-inspection** support: `docs/04b_source_inspection.md` or
  `docs/04c_runtime_probe_results.md` verifies the claim against the
  pinned external source.
- **runtime-probe** support: an in-repo inspection probe artefact
  records the claim.
- **project-document** support: a frozen project document records
  the claim.
- **unsupported assumption**: none of the above kinds of evidence is
  found.

Stronger evidence categories take precedence. A claim with external-
paper support is treated as paper-grounded even if `docs/02`
phrases it less precisely.

---

## 1. Executive verdict

**Proceed to Commit 7 with prompt refinements.**

The Commit-7 specification is internally consistent and implementable
against the current saved-record format and metric primitives. The
acceptance gate (bitwise-exact recomputed adjacency, exact integer
SHD/SID/edge-count match against a direct reference computation) is
testable today without changing any other module. No conflict between
`docs/02`, `docs/08a`, `docs/08`, and the current code prevents
Commit 7 from being implemented as planned.

The wider audit, with the two papers now in repository, identifies
three follow-up issues that do not block Commit 7 but should be
resolved before the real selection-study runner is exercised:

1. The threshold-robustness step of +/- 0.1 around each model's
   primary threshold is project-decided. The two papers do not
   prescribe a sensitivity-sweep step. The sweep is acceptable as
   defensive local sensitivity analysis and should be acknowledged
   as such, not described as calibration.
2. `SCHEMA_GATE_MMD_N_SAMPLES = 64` in
   `experiments/selection_study/sampling.py` is a gate-only default.
   `docs/02` Section 4.2 frozen value is 1000 MMD samples per
   intervention. The selection-study runner (Phase A and Phase B,
   not yet implemented) must override the gate default. If it does
   not, MMD numbers will be too noisy by roughly a factor of four in
   standard error.
3. DCDI `n_iter` for the real study is not frozen in `docs/02`. The
   DCDI paper records that DCDI-G on a 10-node sparse graph reaches
   its stopping criterion around iteration 62 000 (Section B.3,
   `papers/DCDI.pdf` page 30, learning-dynamics figure). The
   selection-study runner must choose a value compatible with this
   convergence behaviour and record it explicitly.

None of these blocks Commit 7.

The DAGMA threshold of 0.3 and the DAGMA hyperparameter overrides
that earlier audit drafts described as "not locally substantiated"
are now confirmed by paper evidence in Section 3 and Section 6.5
below. Likewise for the DCDI 0.5 threshold and the DCDI training
defaults in Section 6.6.

---

## 2. Purpose of threshold robustness

### Decision-driving versus defensive

Threshold robustness is **defensive local sensitivity analysis**,
not threshold calibration. The decision rule frozen in `docs/02`
Sections 2 and 6 is lexicographic over SID, MMD, prior-injection
ergonomics, and standardisation robustness; threshold robustness is
not one of the criteria. The selection decision is computed at the
**primary** thresholds (DAGMA 0.3, DCDI 0.5); the sweep documents
whether the selection-relevant ordering is stable, not what the
selection is.

`docs/02` Section 7 item 5 is explicit: "Report whether the
selection-relevant ordering of the two candidates is stable across
the three threshold values." That is a stability check, not a
calibration step.

### Threshold calibration versus local sensitivity analysis

This is local sensitivity analysis around the primary value, not
threshold calibration. Calibration would require an explicit
held-out target on which the threshold is selected and would have
documented decision criteria for moving the primary value. The
protocol does neither: the primary thresholds are frozen in
`docs/02` Section 9 and may only be revised by an explicit
amendment to `docs/02` recorded in `docs/03` before the held-out
evaluation runs.

### Post-hoc threshold tuning

The threshold-robustness sweep must NOT be used to choose a better
threshold after seeing results. The protocol does not authorise that
move; the primary thresholds are pre-registered. If a future
sensitivity result revealed that the primary thresholds were
unfavourable to one candidate, the right response is an explicit
amendment to `docs/02` Section 9 plus a `docs/03` entry recorded
before held-out evaluation, not a silent post-hoc change. Commit 7's
implementation should ensure the report makes this constraint
visible (for example by recording the primary threshold per record
and labelling the +/- 0.1 thresholds as sensitivity values).

### Why three thresholds is acceptable for the formal report

Three thresholds (primary plus one step in each direction) is the
minimum that lets the report state "ordering of candidates is
stable across primary and one step in each direction". That is a
defensible stability claim for a formal selection report. Five
thresholds would provide a finer curve-shape diagnostic and would
let a reader see whether the metric curve is monotone, flat, or has
local structure near the primary. Three points cannot show
curvature.

The audit recommends retaining three for the formal selection-study
report. A five-point sweep is acceptable as **optional exploratory
analysis** in a notebook or supplementary section, but must not be
elevated into the formal record without an explicit `docs/02`
amendment recorded before held-out evaluation. The reason is
pre-registration discipline: the sweep is the safety check for the
primary decision; expanding the sweep after seeing results would
create a free parameter that could be tuned to favour one
candidate. Keeping the formal sweep at three points and explicitly
labelling any five-point analysis as exploratory preserves the
pre-registered decision.

### Support for fair base-model selection

Threshold robustness supports fairness in three ways:

- It exposes whether a candidate's reported SHD / SID / edge-count
  at the primary threshold is a knife-edge artefact rather than a
  stable property.
- It surfaces invalid-graph behaviour at neighbouring thresholds,
  which can reveal that a candidate is operating very close to its
  cyclic / bidirected boundary.
- It provides a within-candidate stability number that can be
  reported alongside the primary metric, so a reader can judge
  whether a small SHD or SID difference between candidates is
  bigger or smaller than each candidate's own within-candidate
  threshold sensitivity.

It does not, on its own, compensate for differences in primary
threshold scales between models; see Section 3 below.

---

## 3. Threshold choices

### DAGMA

- **Primary threshold:** 0.3, applied to `abs(W_continuous)`.
- **Threshold-robustness triple:** `{0.2, 0.3, 0.4}`.
- **Sources:**
  - external-paper: `papers/DAGMA.pdf` Section C.1.1 (page 21,
    "Small to Moderate Number of Nodes") states verbatim:
    *"Finally, as in previous work including the baseline methods,
    a final thresholding step is performed as it was shown to help
    reduce the number of false discoveries. For all cases, we use a
    threshold of 0.3."* The threshold is applied to the absolute
    value of the weighted-adjacency entries per the procedure
    described in the same appendix.
  - source-inspection: `docs/04b` D-7 confirms the inspected DAGMA
    library default in `dagma/linear.py:234` is also
    `w_threshold=0.3` and the threshold is applied to `abs(W_est)`.
    The project wrapper passes `w_threshold=0.0` to preserve
    `W_continuous`, then applies the project threshold externally.
- **Status of the primary value:** **paper-grounded** (DAGMA paper
  Section C.1.1) plus source-inspection-confirmed. This resolves the
  earlier audit's "not locally substantiated" caveat.
- **Status of the triple:** project-decided. The +/- 0.1 step is not
  prescribed by the DAGMA paper, which only documents the single
  primary value 0.3 used "for all cases". The triple is a project-
  added defensive sensitivity probe and is acceptable as such.

### DCDI

- **Primary threshold:** 0.5, applied to `model.get_w_adj()` (which
  equals `sigmoid(log_alpha) * (1 - I)`).
- **Threshold-robustness triple:** `{0.4, 0.5, 0.6}`.
- **Sources:**
  - external-paper (semantics): `papers/DCDI.pdf` Section 3.2
    (page 6) frames the adjacency matrix as a random Bernoulli
    matrix: *"we treat the adjacency matrix M^G as random, where
    the entries M^G_{ij} are independent Bernoulli variables with
    success probability sigma(alpha_{ij}) ... and alpha_{ij} is a
    scalar parameter."* Footnote 2 records that *"sigma(Lambda)
    tends to become deterministic as we optimize"*. The
    Bernoulli interpretation makes 0.5 a principled midpoint of
    the success-probability axis (0.5 is the Bayes-optimal
    threshold for converting a calibrated success probability into
    a deterministic edge decision; this is a probability-theoretic
    consequence of the Bernoulli formulation, not a calibration
    decision).
  - external-paper (threshold value): `papers/DCDI.pdf` Section
    B.3 (page 30, "Learning dynamics") states: *"Around iteration
    62000 ... Edges with a sigma(alpha_{ij}) higher than 0.5 are
    set to 1 and others set to 0."* This is the DCDI paper's
    explicit rule for converting the learned soft adjacency into a
    deterministic DAG at the end of training.
  - source-inspection: `docs/04b` records that the inspected DCDI
    source also uses `model.get_w_adj() > 0.5` for its own
    deterministic thresholding at `train.py:210`.
- **Status of the primary value:** **paper-explicit** at 0.5
  (DCDI paper Section B.3) plus **probability-semantics
  justified** (DCDI paper Section 3.2 Bernoulli formulation) plus
  source-inspection-confirmed.
- **Status of the triple:** project-decided. The DCDI paper does
  not prescribe a +/- 0.1 sweep around 0.5.

### Configurability and propagation

The triple is per-Configuration and propagates through
`config_resolved` into the saved run record. In
`experiments/selection_study/config.py:229` it is typed
`tuple[float, float, float]`; after JSON serialisation it appears as
a list of three floats inside
`config_resolved["threshold_robustness_triple"]`. There is no
separate top-level run-record field. This is correct under the
schema: the triple is configuration metadata, not a derived per-run
aggregate.

The triple is therefore per-Configuration fixed (per model, per
condition, per phase, as written) but the type system does NOT
prevent two Configurations within the same selection study from
disagreeing on the triple. Commit 7 should validate the triple it
reads from `config_resolved` against the per-model protocol
constants to catch drift; see Section 8.

### Why three thresholds, not five

The DAGMA and DCDI papers do not perform a within-paper threshold
sweep. DAGMA fixes 0.3 "for all cases" (Section C.1.1); DCDI fixes
0.5 (Section B.3). The choice of three sensitivity points is a
project decision. See Section 2 above for the recommendation to
keep three points in the formal selection report and treat a
five-point sweep as optional exploratory analysis.

### Symmetry around the primary

- DAGMA `{0.2, 0.3, 0.4}` is symmetric around 0.3 with an additive
  step of 0.1.
- DCDI `{0.4, 0.5, 0.6}` is symmetric around 0.5 with an additive
  step of 0.1.

Relative to the primary value the steps differ (DAGMA +/- 33%
relative; DCDI +/- 20% relative). The audit does not regard this as
a fairness blocker: the underlying mathematical objects are
different (signed weights vs Bernoulli probabilities) so a relative
step is not the natural unit. An additive step in the native object
is the right unit per candidate, and the two triples are each
within-candidate symmetric.

### Within-model only versus cross-model comparable

The DAGMA threshold operates on unbounded signed weights;
specifically the DAGMA paper Section C.1 records that edge weights
are drawn from `Unif([-2, -0.5] union [0.5, 2])`, so the natural
range of `abs(W)` is roughly `[0, 2]`. The DCDI threshold operates
on Bernoulli success probabilities in `[0, 1]`. A numerical
comparison such as "DAGMA's 0.3 is lower than DCDI's 0.5" is
therefore not meaningful. The two sweeps are native-object
thresholds and must not be numerically cross-compared.

For Commit 7 this means: the threshold-robustness record should not
cross-tabulate DAGMA and DCDI thresholds as if they were on the
same axis. Each candidate's report is independent. Any cross-
candidate stability statement must be a comparison of within-
candidate aggregates, not of threshold values.

### Sufficiency of the sweep

Three thresholds with a single additive step is sufficient to
detect the worst class of threshold-sensitivity bug (the selection-
relevant ordering flips when the primary is moved by 0.1). It is
not sufficient to detect non-monotone behaviour requiring at least
five points to see, or threshold-sensitivity that emerges only far
from the primary value, or per-seed threshold sensitivity that is
masked by mean aggregation across seeds. The audit recommends
documenting these limitations in the report header rather than
amending the protocol mid-flight.

---

## 4. Thresholding and interventional generalisation

### Does thresholding affect interventional generalisation?

Yes, in this study, through a load-bearing path: the boolean
thresholded adjacency is the parent set used for downstream
ancestral / model-frame interventional sampling, and that sampling
produces the MMD ground-truth comparison and the SID computation.

- DAGMA: `A_thresh = abs(W_continuous) >= 0.3`, then
  `W_sample = W_continuous * A_thresh.astype(float)`. The
  thresholded parent set defines which edges contribute to a
  child's conditional in the residual-fitted sampler.
- DCDI: `A_thresh = model.get_w_adj() >= 0.5`. The structural mask
  is enforced through `model.adjacency` plus saturated `log_alpha`;
  sub-threshold soft edges must not contribute to a child's
  conditional. The thresholded parent set is again load-bearing.
- SID is computed against the thresholded adjacency. SID requires a
  valid DAG, which is determined by the threshold.

Threshold choice therefore directly affects what the model is
"saying" about each variable's parents, which affects both SID and
the interventional MMD via the sampler.

### Project and paper evidence linking threshold choice to interventional metrics

- external-paper (DAGMA): the paper reports SHD, TPR, and FDR
  (`papers/DAGMA.pdf` Section C.1.1, Table 1 and Figure 8). The
  paper does not report SID or interventional MMD. The 0.3
  threshold is justified by false-discovery reduction (a
  structural-recovery argument), not by an interventional metric.
- external-paper (DCDI): the paper reports SHD and SID on 20-node
  graphs (Section 4, Figures 2-4; Tables 4-6 in Appendix C.4). The
  DCDI paper does not justify the 0.5 threshold by an
  interventional metric either; the threshold is the deterministic
  conversion from the Bernoulli formulation at the end of training.
- runtime-probe: `docs/04h` (C-P13) and `docs/04f` (C-P11) record
  that on a 3-node ER2 fixture at the primary thresholds, DAGMA
  recovered the true adjacency exactly while DCDI missed the
  strongest true edge. The MMD difference between the two
  candidates at the primary thresholds was roughly two orders of
  magnitude on that fixture. This is fixture-specific evidence
  that primary threshold choices interact with interventional MMD;
  it does not generalise to the 10-node ER2 selection-study cell.
- runtime-probe: `docs/04g` (C-P12) records that the C-P11 fixture
  is recoverable in principle under an equal-variance-aware
  exhaustive Gaussian-BIC score, sharpening the C-P11
  interpretation as inductive-bias / optimisation / model-mismatch
  rather than data-impossibility.

### Threshold value that optimises interventional generalisation

No universal optimum exists across SCMs, sample sizes, and
candidate models. Neither the DAGMA paper nor the DCDI paper
asserts a threshold value optimised for interventional metrics.
DAGMA's 0.3 is justified by false-discovery reduction; DCDI's 0.5
is the Bayes-optimal cut for a Bernoulli probability under the
trained model's own formulation. Both are reasonable starting
points; neither is asserted to optimise SID or MMD specifically.

### Avoiding post-hoc threshold tuning

The protocol avoids post-hoc threshold tuning by pre-registering
the primary thresholds in `docs/02` Section 9, pre-registering the
sweep triples in `docs/02` Section 7 item 5, restricting the
threshold-robustness sweep to a stability report rather than a
re-selection mechanism, and requiring threshold or selection-
criterion revisions to go through an explicit `docs/02` amendment
plus a `docs/03` entry recorded before held-out evaluation. Commit
7 should not weaken any of these constraints. The threshold-
robustness record should be a passive sensitivity report, not an
input to a selection algorithm.

---

## 5. Fairness of model comparison

This section audits each fairness item declared in `docs/02`,
`docs/04`, and `docs/08a` against the current code. The framing is
**best defensible settings under thesis constraints, equal
calibration budget, and no held-out leakage**, not "best settings"
in the unconstrained sense. The two papers use different tuning
conventions; the project uses Phase A reproduction plus equal-
budget Phase B calibration to avoid favouring one paper's
protocol.

### Different tuning conventions across the two papers

- DAGMA paper Section 5 (`papers/DAGMA.pdf` page 9, Remark 5 in
  Section C, page 20): *"Consistent with previous work in this
  area (e.g. NOTEARS and its follow-ups), we have not performed
  any hyperparameter optimization: this is to avoid presenting
  unintentionally biased results. As a concrete example, for each
  of the following SEM settings, we simply chose a reasonable
  value for the L1 penalty coefficient and used that same value
  for all graphs across many different numbers of nodes."*
- DCDI paper Section B.5 (`papers/DCDI.pdf` page 31): *"For DCDI, a
  grid search was performed over 10 values of the regularization
  coefficient (see Table 1) for known interventions ... The
  hyperparameter combination chosen was the one that induced the
  lowest negative log-likelihood on the held-out examples."*

DAGMA fixes a single reasonable hyperparameter; DCDI performs a
grid search on held-out NLL. Following either paper's convention
verbatim would favour that paper's candidate. The project's Phase
B equal-budget local calibration (5 configurations per model and 2
calibration seeds per configuration, with calibration and held-out
seeds disjoint) is the project's response to this asymmetry: each
candidate is given the same calibration budget and the same
calibration metric, applied to the project's selection-study cell.

### Items confirmed fair

- **Same SCM regime.** Both candidates fit on the same
  `LinearGaussianSCM` generator (`docs/02` Section 3.1; code path
  `symbolic_priors_cd.data.generate_linear_gaussian_scm`). The
  seed, graph density, noise scale, and weight magnitude range are
  identical across candidates.
- **Same train / evaluation seed discipline.** Per-run seeds are
  derived from a single SHA-256-based rule
  (`SEED_DERIVATION_RULE_NAME` in `config.py`) per the manifest
  enumeration. DAGMA records `seed_torch / seed_numpy / seed_dagma
  = null` honestly (its fit is verified-deterministic per
  `docs/04b` D-6 and `docs/04c` D-P2 plus the 13/05/2026 entry in
  `docs/03`); DCDI records them as the actual fit seed.
- **Same metrics.** SHD (project default `reversal_cost=2`) and
  SID (`gadjid==0.1.0`) primitives are model-agnostic and cross-
  checked in `docs/04j`. MMD primary uses the unbiased RBF
  estimator with the median heuristic and is consumed unchanged
  across both candidates.
- **Same intervention set.** `config.intervention_set` is one tuple
  per Configuration, shared across both candidates in the same
  study.
- **Same MMD bandwidth policy.** The same deterministic median
  heuristic and same sweep multipliers `{0.5x, 1.0x, 2.0x}` are
  used for both candidates. Negative unbiased MMD is preserved
  verbatim for both.
- **Same preprocessing conditions.** Centred-only and standardised
  conditions are applied identically to both candidates via the
  same `wrappers.preprocessing` classes.
- **No held-out leakage.** Calibration and held-out evaluation
  seed populations are validated as disjoint by preflight rule (a)
  and by the Configuration's own `seed_populations` validation.
- **Equal Phase B budget.** 5 configurations per model and 2
  calibration seeds per configuration is the same compute budget
  for both candidates (`docs/02` Section 3.3).
- **Model-specific paper-grounded thresholds.** DAGMA 0.3 is
  paper-grounded by DAGMA Section C.1.1; DCDI 0.5 is paper-explicit
  in DCDI Section B.3 and probability-semantics-justified by DCDI
  Section 3.2. The per-candidate threshold choice is fair as
  **native-object thresholds**. They must not be numerically cross-
  compared (Section 3 above).
- **No post-hoc threshold or hyperparameter changes.** Frozen by
  `docs/02` Section 9 with explicit amendment discipline.

### Items with caveats

- **Equal calibration budget for unequal model complexity.** DCDI's
  `LearnableModel_NonLinGaussANM` is a nonlinear MLP-based model
  applied to a linear-Gaussian SCM. DAGMA-linear is correctly
  specified for the SCM family. Equal calibration budget is
  therefore fair in compute but is not equally generous in terms
  of inductive bias. The selection-study cell is the thesis's
  intended 10-node ER2 linear-Gaussian regime (`docs/01` Section
  10.1), so this asymmetry is by design and by the thesis
  question; it would be unfair only if the selection decision
  were claimed to generalise outside that regime.
- **Observational-only DCDI is not DCDI's strongest setting.** The
  DCDI paper title and abstract make explicit that DCDI is
  designed to leverage interventional data: *"Differentiable
  Causal Discovery from Interventional Data ... This work
  constitutes a new step in this direction by proposing a
  theoretically-grounded method based on neural networks that can
  leverage interventional data."* (`papers/DCDI.pdf` page 1).
  Tables 4-6 in Appendix C.4.3 separately report a DCD-no-interv
  ablation (DCDI-G applied to purely observational data), which
  is exactly the regime this thesis uses. The DCDI paper itself
  treats DCD-no-interv as a baseline rather than as DCDI's main
  contribution. Fairness consequence: the thesis evaluates DCDI in
  a fair but **constrained** setting (observational-only is the
  thesis constraint per `docs/01` Section 7.1). The selection-
  study result should not be described as a full test of DCDI's
  strongest intended regime.

### Items requiring active monitoring

- **MMD sample count.** `docs/02` Section 4.2 frozen value is 1000
  per intervention. The current code default
  `SCHEMA_GATE_MMD_N_SAMPLES = 64` in
  `experiments/selection_study/sampling.py` is documented as
  schema-gate-only. The selection-study runner (Phase A and Phase
  B, not yet implemented) must override this with 1000. This is a
  runner-side responsibility; the audit recommends Commit 8 or the
  Phase A / B prompts treat this override as a precondition and
  test it explicitly.

### Are we trying each model in its best defensible setting under thesis constraints?

For DAGMA: yes. The DAGMA hyperparameter overrides recorded in
`docs/02` Section 3.3 are exactly the values stated in DAGMA Section
C.1.1 (`papers/DAGMA.pdf` page 21) for the linear SEM setting; see
Section 6.5 for the per-parameter mapping. Threshold 0.3 matches
DAGMA Section C.1.1 verbatim.

For DCDI: yes, under the thesis's observational-only constraint.
The DCDI hyperparameter defaults recorded in `docs/02` Section 3.3
match the DCDI default-hyperparameter table at `papers/DCDI.pdf`
Section B.5 / Table 2 (page 31-32). Threshold 0.5 matches DCDI
Section B.3 verbatim. The DCDI-G observational-only formulation is
exactly the DCD-no-interv ablation reported by the DCDI paper
itself.

### Is one model being advantaged?

The audit finds no scientifically-relevant asymmetry that
advantages one model over the other within the linear-Gaussian
10-node ER2 cell beyond the model-mismatch concern flagged above.
Same SCM, same seeds, same metrics, same sample sizes (modulo the
MMD-count caveat above), same intervention set, same preprocessing,
same calibration budget. Threshold scales differ but operate on
different native objects.

### Are any settings currently arbitrary?

- **DAGMA and DCDI threshold-robustness step (+/- 0.1).**
  Project-decided. Neither paper prescribes a within-paper
  sensitivity sweep; the project added a +/- 0.1 step as
  defensive sensitivity analysis. Acceptable for that purpose,
  flagged here for transparency.
- **MMD primary policy choice for DAGMA = residual_fitted.**
  Project-decided: justified in `docs/02` Section 4.2 by
  source-inspection of DAGMA's no-built-in-sampler property plus
  a frozen wrapper policy. Not arbitrary.
- **n_samples=1000 for the selection-study MMD.** Project-document
  support in `docs/02` Section 4.2 plus default-by-design
  rationale; no local power analysis is reachable.

---

## 6. Other protocol constants audit

Each row of the table is structured as:

| Constant | Value | Source | Status | Scope | In run.json / config_resolved | Rationale | Evidence type | Action |
|---|---|---|---|---|---|---|---|---|

Status legend:

- **protocol-critical**: changing it changes the selection decision
  or its scientific defensibility.
- **diagnostic-only**: changing it changes a diagnostic but not the
  primary decision.
- **toy-gate-only**: present in code as a schema-conformance-gate
  constant; must be overridden by the real-study runner.

Scope legend:

- **global** / **per-model** / **per-condition** / **per-phase**.

### 6.1 SCM regime

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| n_nodes (primary cell) | 10 | docs/02 Section 3.1 | protocol-critical | global | yes (via config_resolved) | thesis primary cell per docs/01 Section 10.1 | project-document | none |
| expected_edges (ER2) | 2 * n_nodes = 20 | docs/02 Section 3.1 | protocol-critical | global | yes | "ER2" convention; docs/01 Section 10.1 | project-document | none |
| noise_scale | 1.0 (per node) | docs/02 Section 3.1; docs/01 | protocol-critical | global | yes | unit-variance Gaussian; consistent with DAGMA paper Section C.1 noise model; identifiability check in C-P12 | project-document, external-paper, runtime-probe | none |
| observational sample size | 1000 | docs/02 Section 3.1 / Section 9 | protocol-critical | global | yes | thesis-aligned default; both papers use 1000 (DAGMA Section C.1) or 10000 (DCDI Section 4) as defaults; the project's 1000 is within the papers' range | project-document, external-paper | none |

### 6.2 Seed populations

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| Phase A reproduction seeds | not numerically frozen in docs/02 (reproduction cell is paper-aligned) | docs/02 Section 3.3 Phase A | diagnostic-only | per-model | will be in run record via seed_population="reproduction" | reproduction pass is a sanity check, not a primary measurement | project-document | confirm count when Commit 8 implements Phase A |
<!--
Supersession note (docs/02 v1.8 Path B, 21/05/2026): the "reproduction cell is paper-aligned" wording in the row above anticipates per-model paper-aligned reproduction cells for Phase A. Under docs/02 v1.8 Path B the current execution uses thesis-cell compatibility / runner-sanity reproduction configs (10-node ER2) instead of paper-aligned cells. Any statement in this audit that anticipates paper-aligned Phase A reproduction cells must be read as referring to a future deferred sub-study rather than the current Phase A execution. The reproduction-pool seed integers (101, 102, 103) are frozen by docs/02 v1.7 and are unaffected by Path B.
-->


| Phase B calibration seeds | 2 per configuration | docs/02 Section 3.3 Phase B | protocol-critical | per-configuration | yes | leakage-prevention requires disjointness from held-out | project-document | none |
| Phase B configurations | 5 per model | docs/02 Section 3.3 Phase B | protocol-critical | per-model | yes | equal-budget local calibration; DCDI paper itself uses a 10-value grid (Section B.5) so 5 is a project compute compromise | project-document, external-paper | none |
| Held-out evaluation seeds | 5 per model per condition | docs/02 Section 3.3 Phase B | protocol-critical | per-model per condition | yes | held-out measurement count | project-document | none |
| Calibration / held-out non-overlap | hard rule | docs/02 Section 3.3, preflight rule (a) | protocol-critical | global | enforced by preflight | leakage prevention | project-document, code | none |

### 6.3 Ranking rule

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| SID tie margin (Criterion 1) | 10% | docs/02 Section 2 Criterion 1b | protocol-critical | global | indirectly | SID-first lexicographic ordering | project-document | none |
| MMD tiebreaker | mean over available interventions | docs/02 Sections 2 and 4.2 | protocol-critical | global | per-record via interventions list | secondary to SID | project-document | none |
| Criterion 2 (prior-injection ergonomics) | qualitative; smoke test | docs/02 Section 4.3 | protocol-critical | global | not currently in run record | this criterion is post-selection-study, not a per-run field | project-document | none |
| Criterion 3 (standardisation robustness) | 50% mean-SHD-degradation threshold | docs/02 Section 4.4 | protocol-critical | global | implicitly via condition pairs | catastrophic-scale-artefact test | project-document | none |
| SHD reversal cost | 2 | docs/03 metrics-layer entry; docs/04j | diagnostic-only | global | yes (shd_reversal_cost field) | stricter convention, documented | project-document, source-inspection, runtime-probe | none |

### 6.4 MMD policy

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| MMD sample count (real study) | 1000 per intervention | docs/02 Section 4.2 | protocol-critical | global | yes (n_ground_truth_samples / n_model_samples per record) | matches observational sample size | project-document | runner must override schema-gate default |
| MMD sample count (schema gate) | 64 | sampling.py SCHEMA_GATE_MMD_N_SAMPLES | toy-gate-only | gate-only | yes | gate quick path; not selection number | code | block any selection runner that uses gate default |
| MMD bandwidth heuristic | median heuristic over concatenated upper triangle | docs/02 Section 4.2; docs/04j | protocol-critical | global | yes (mmd_bandwidth_used_value) | source-cross-checked in docs/04j | project-document, runtime-probe | none |
| MMD bandwidth sweep | {0.5x, 1.0x, 2.0x} of median | docs/02 Section 4.2; sampling.py | protocol-critical | global | yes (mmd_bandwidth_sweep) | sensitivity check | project-document | none |
| MMD clip policy | no clip; negative values preserved | docs/02 Section 4.2; docs/04j | protocol-critical | global | yes (mmd_clip_policy) | unbiased estimator can be negative | source-inspection, runtime-probe | none |

### 6.5 DAGMA hyperparameters

The values listed in `docs/02` Section 3.3 are exactly the values
DAGMA Section C.1.1 (`papers/DAGMA.pdf` page 21) records for the
linear SEM experimental setting. The DAGMA paper text is:

> "We use the following setting for DAGMA (Algorithm 1): Number of
> iterations T = 4, initial central path coefficient mu^(0) = 1,
> decay factor alpha = 0.1, L1 coefficient beta_1 = 0.05, log-det
> parameter s = {1, .9, .8, .7}."

Section 4.1 of the main text (page 8) additionally records the
example mu and s sequences `mu = {1, 0.1, 0.001, 0}` and
`s = {1, 0.9, 0.8, 0.7}` for `T = 4`.

| Constant | Project value | Paper value | Source | Status | Evidence |
|---|---|---|---|---|---|
| T | 4 | T = 4 | DAGMA C.1.1 | protocol-critical | external-paper |
| lambda1 | 0.05 | beta_1 = 0.05 | DAGMA C.1.1 | protocol-critical | external-paper |
| s sequence | (1.0, 0.9, 0.8, 0.7) | s = {1, .9, .8, .7} | DAGMA C.1.1 | protocol-critical | external-paper |
| mu_init | 1.0 | mu^(0) = 1 | DAGMA C.1.1 | protocol-critical | external-paper |
| mu_factor (decay alpha) | 0.1 | alpha = 0.1 | DAGMA C.1.1 | protocol-critical | external-paper |
| ADAM lr | 3e-4 | "Learning rate of 3 x 10^-4" | DAGMA C.1.1 | protocol-critical | external-paper |
| ADAM (beta1, beta2) | (0.99, 0.999) | "(beta1, beta2) = (0.99, 0.999)" | DAGMA C.1.1 | protocol-critical | external-paper |
| w_threshold (in-library) | 0.0 | not paper-fixed; project decision for preservation | docs/02 Section 9; docs/04c D-P1 | protocol-critical | project-document, runtime-probe |
| project threshold (DAGMA) | 0.3 | "we use a threshold of 0.3" | DAGMA C.1.1 | protocol-critical | external-paper |

All DAGMA tactical constants used by the project are externally
paper-substantiated. The single project-decided constant in this
group is `w_threshold = 0.0`, which is a wrapper-internal
convention to preserve `W_continuous` for offline threshold-
robustness; the DAGMA paper does not address this because the
paper's setup does not need offline re-thresholding.

### 6.6 DCDI hyperparameters

The DCDI paper records its default hyperparameters in Table 2
(`papers/DCDI.pdf` page 32) and in Section B.5 (page 31). The
project values in `docs/02` Section 3.3 match the paper's defaults
exactly; the optimizer choice and batch size are recorded in
Section B.5 page 31.

| Constant | Project value | Paper value | Source | Status | Evidence |
|---|---|---|---|---|---|
| h_threshold | 1e-8 | "Augmented Lagrangian constraint threshold: 10^-8" | DCDI Table 2 | protocol-critical | external-paper |
| mu_0 (mu_init) | 1e-8 | "mu_0: 10^-8" | DCDI Table 2 | protocol-critical | external-paper |
| gamma_0 (gamma_init) | 0.0 | "gamma_0: 0" | DCDI Table 2 | protocol-critical | external-paper |
| mu_mult_factor (eta) | 2 | "eta: 2" | DCDI Table 2 | protocol-critical | external-paper |
| omega_mu (delta) | 0.9 | "delta: 0.9" | DCDI Table 2 | protocol-critical | external-paper |
| lr | 1e-3 | "learning rate: 10^-3" | DCDI Table 2 | protocol-critical | external-paper |
| hid_dim | 16 | "# hidden units: 16" | DCDI Table 2 | protocol-critical | external-paper |
| num_layers | 2 | "# hidden layers: 2" | DCDI Table 2 | protocol-critical | external-paper |
| nonlin | leaky-relu | "neural network activation functions were leaky-ReLU" | DCDI Section B.5 | protocol-critical | external-paper |
| optimiser | RMSprop | "RMSprop was used as the optimizer" | DCDI Section B.5 | protocol-critical | external-paper |
| train_batch_size | 64 | "minibatches of size 64" | DCDI Section B.5 | protocol-critical | external-paper |
| stop_crit_win | 100 (default) | not isolated in paper; project default tracks the iteration-count check structure described informally in Section B.3 (gradient-step plateau test) | docs/02 Section 3.3 (default) | protocol-critical | project-document |
| n_iter (real study) | not frozen in docs/02 | DCDI paper Section B.3 (page 30) records DCDI-G on a sparse 10-node graph reaching its stopping criterion at iteration 62000; this is a single illustrative number, not a frozen experimental constant | none yet | protocol-critical | external-paper context only |
| project threshold (DCDI) | 0.5 | "Edges with a sigma(alpha_{ij}) higher than 0.5 are set to 1 and others set to 0" | DCDI Section B.3 | protocol-critical | external-paper |

All DCDI tactical constants in `docs/02` Section 3.3 that correspond
to Table 2 / Section B.5 of the DCDI paper are externally paper-
substantiated. `stop_crit_win = 100` is a sensible default value
not explicitly frozen by the paper but consistent with the paper's
informal description of the gradient-step plateau test. `n_iter` is
not frozen anywhere; the runner-side choice should be informed by
the paper's reported convergence point on a comparable cell.

### 6.7 Preprocessing

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| centred_only | per-variable mean subtraction; no scaling | docs/02 Section 4.4 | protocol-critical | per-condition | yes (condition field) | leakage-safe centring | project-document | none |
| standardised | per-variable mean and std normalisation | docs/02 Section 4.4 | protocol-critical | per-condition | yes | leakage-safe standardisation | project-document | none |

### 6.8 Sampling policies

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| DAGMA primary policy | residual_fitted | docs/02 Section 4.2 | protocol-critical | per-model | yes (sampler_policy_used) | per-node residual sigma from training data | project-document, source-inspection | none |
| DAGMA sensitivity policy | unit_variance | docs/02 Section 4.2 | diagnostic-only | per-model | yes (mmd_sensitivity_unit_variance) | sigma=1 control | project-document | none |
| DCDI native policy | dcdi_native via forward_given_params + get_distribution | docs/02 Section 4.2; docs/04c C-P5..C-P7 | protocol-critical | per-model | yes | verified path | project-document, runtime-probe | none |
| Structural mask enforcement (DCDI) | model.adjacency plus saturated log_alpha; save-mutate-restore | docs/02 Section 4.2; docs/04d | protocol-critical | per-model | yes via wrapper diagnostics | required because forward_given_params multiplies by model.adjacency | project-document, source-inspection, runtime-probe | none |

### 6.9 Invalid-graph handling

| Constant | Value | Source | Status | Scope | Run record | Rationale | Evidence | Action |
|---|---|---|---|---|---|---|---|---|
| Invalid graph in primary run record | pipeline raises InvalidGraphForSchemaGateError | pipeline.py | protocol-critical | global | no run.json is written | schema requires sid as plain int; no silent partial record | project-document, source-inspection | none |
| Invalid graph in threshold-robustness record | per-threshold graph_status with sid nullable | docs/08a Section 8; this audit recommendation | protocol-critical | per-threshold | yes (sibling artefact) | a multi-threshold report cannot stop on one bad threshold | project-document, this audit | implement per Section 8 |

### 6.10 Schema-gate toy constants that must not leak

| Constant | Value | Where defined | Must be overridden by | Risk if leaked |
|---|---|---|---|---|
| SCHEMA_GATE_N_NODES | 3 | pipeline.py | Phase A and Phase B runners (use 10 from docs/02 Section 3.1) | toy SCM hides selection-relevant signal |
| SCHEMA_GATE_EXPECTED_EDGES | 3 | pipeline.py | Phase A and Phase B runners | not ER2 |
| SCHEMA_GATE_N_TRAIN | 64 | pipeline.py | Phase A and Phase B runners (1000 per docs/02) | underpowered fits |
| SCHEMA_GATE_N_VAL_DCDI | 32 | pipeline.py | Phase A and Phase B runners | underpowered DCDI Lagrangian schedule |
| SCHEMA_GATE_DCDI_N_ITER | 30 | pipeline.py | Phase A and Phase B runners (real value informed by DCDI paper Section B.3 convergence point) | DCDI does not converge; selection-relevant signal lost |
| SCHEMA_GATE_DCDI_CONFIG_KWARGS | {stop_crit_win: 10, train_batch_size: 8} | pipeline.py | Phase A and Phase B runners (stop_crit_win=100, train_batch_size=64 per docs/02) | DCDI schedule fires too frequently and on too-small batches |
| SCHEMA_GATE_MMD_N_SAMPLES | 64 | sampling.py | Phase A and Phase B runners (1000 per docs/02 Section 4.2) | MMD too noisy; selection decisions amplified by noise |

All seven schema-gate constants must be overridden by the
selection-study runner. The audit recommends Phase A / Phase B
runner prompts treat this as a precondition with explicit tests.

---

## 7. Conflicts, stale assumptions, and amendments

### Conflicts

- **None blocking Commit 7.** The Commit 7 specification, the
  current saved-record schema, the metric primitives, and the
  graph-status helper are mutually consistent.

### Stale assumptions

- **Earlier audit drafts described the DAGMA 0.3 threshold and
  DAGMA hyperparameter overrides as "not locally substantiated".**
  That claim is now stale: `papers/DAGMA.pdf` Section C.1.1 (page
  21) is in repository and substantiates these values exactly. The
  current revision treats them as externally paper-grounded.
- **Earlier audit drafts described the DCDI 0.5 threshold as
  project-decided.** That claim is now stale: `papers/DCDI.pdf`
  Section B.3 (page 30) is paper-explicit on the 0.5 cutoff
  ("Edges with a sigma(alpha_{ij}) higher than 0.5 are set to 1
  and others set to 0"). The current revision treats this as
  externally paper-grounded.

### Executable-but-not-justified values that remain

- **+/- 0.1 threshold-robustness step.** Project-decided. Neither
  paper performs a within-paper sensitivity sweep around its
  primary threshold. The +/- 0.1 step is a project-added defensive
  sensitivity probe and should be acknowledged as such, not as
  paper-grounded calibration.
- **MMD sample count default of 64 in code (vs the protocol's
  1000).** Justified for the schema-conformance gate only. Risk if
  the selection-study runner forgets to override.
- **DCDI n_iter for the real study** is not frozen in `docs/02`.
  The schema-gate uses 30 (which does not converge by design). The
  DCDI paper reports DCDI-G reaching its stopping criterion at
  about iteration 62000 on a sparse 10-node graph (`papers/DCDI.pdf`
  Section B.3, page 30); the selection-study runner needs a
  concrete value that is consistent with this convergence
  behaviour and the project's compute budget.

### Recommended `docs/02` amendments (not blocking Commit 7)

1. Section 9 Notes: replace any residual "paper-grounded" wording
   for DAGMA 0.3 with a direct citation to DAGMA Section C.1.1
   (now in repository at `papers/DAGMA.pdf`). Same for the DAGMA
   hyperparameter overrides; same for DCDI 0.5 and the DCDI
   default-hyperparameter table.
2. Section 3.3: freeze a concrete DCDI `n_iter` for the
   selection-study cell, informed by DCDI Section B.3 page 30
   convergence behaviour and the project's compute budget.
3. Section 7 item 5: add a one-line acknowledgement that the
   +/- 0.1 threshold step is a project-decided defensive step,
   not a paper-grounded calibration.

### Recommended `docs/03` decision-log entries

If any of the three `docs/02` amendments above is made, a paired
`docs/03` entry recording the rationale and the pre-registration
discipline (the amendment must precede held-out evaluation runs).

### Required code changes before Commit 7

**None.** Commit 7 can be implemented against the current code
state. The follow-ups above are pre-selection-study concerns, not
Commit-7 concerns.

---

## 8. Recommended Commit 7 implementation policy

### Where to read thresholds from

Read the triple from
`record["config_resolved"]["threshold_robustness_triple"]` on the
loaded run record. Validate against per-model protocol constants
at read time:

- if `record["model"] == "dagma"`: expected triple `(0.2, 0.3,
  0.4)`.
- if `record["model"] == "dcdi"`: expected triple `(0.4, 0.5,
  0.6)`.

The validation should be a strict-equality check with a clear
error message naming both the read and expected triples. The
selection-study protocol does not permit per-run triple drift; a
mismatch should raise rather than silently use whatever the run
record carries.

### Whether to validate against protocol constants

Yes, as above. The triple in `config_resolved` is the run-time
materialisation; the per-model constants are the protocol's
identity. Validating against the constants catches configuration
drift between runs and across reruns.

### Native object to threshold

- DAGMA: threshold is applied to `abs(W_continuous)` (DAGMA
  Section C.1.1 verbatim). Commit 7 should load `W_continuous`
  from `continuous_edge_object.npz`, take its absolute value, and
  compare against each threshold in the triple.
- DCDI: threshold is applied to `w_adj` (= `sigmoid(log_alpha) *
  (1 - I)`; DCDI Section B.3 verbatim). Commit 7 should load
  `w_adj` from `continuous_edge_object.npz` (the npz also carries
  `log_alpha`, which is preserved for diagnostic uses but is not
  the thresholding object).

### Whether to write `threshold_robustness.json`

Yes. Write a sibling artefact in the same run directory as
`run.json`. `docs/08a` Section 7 forbids adding mandatory fields
to `run.json` without a `schema_version` bump, so the threshold-
robustness output must live outside `run.json`. `docs/08` Commit 7
explicitly authorises a sibling artefact in the run directory.
The file name `threshold_robustness.json` is descriptive and
aligns with the rest of the record.

### Fields the threshold-robustness record should contain

Minimum required for the acceptance gate:

- `run_id`: string, copied from `run.json` for cross-reference.
- `model`: string, copied from `run.json`.
- `condition`: string, copied from `run.json`.
- `continuous_edge_object_artefact`: string, relative path of the
  artefact actually thresholded.
- `primary_threshold`: float, copied from `docs/02` Section 9 per
  the candidate's model.
- `triple`: list of three floats, the actual triple used.
- `records`: list of three per-threshold records.

Each per-threshold record:

- `threshold`: float.
- `edge_count`: integer (sum of the boolean adjacency).
- `graph_status`: string from the existing taxonomy (`valid_dag`,
  `cyclic`, `bidirected`, `self_loop`, `invalid_shape`).
- `graph_status_reason`: string or null.
- `shd`: integer, computed unconditionally because boolean SHD is
  well-defined on a non-DAG.
- `sid`: integer or null. Null when `graph_status != "valid_dag"`
  because SID requires a valid DAG. Never silently zero.

The record should also carry a `shd_reversal_cost` field for
consumer convenience (mirroring `run.json`).

### How invalid graphs should be represented

Record `graph_status` and `graph_status_reason` explicitly,
compute `shd` as an integer always, and set `sid` to null when the
graph is not a valid DAG. Do not repair the graph. Do not silently
report `sid=0`. Do not invent a new status enum value; use the
existing four invalid-graph values plus `valid_dag`.

This differs from the primary-record behaviour, which is to stop
entirely on a non-DAG. The behavioural asymmetry is intentional: a
sweep across multiple thresholds cannot stop on one bad threshold
without losing the sensitivity signal. Stopping on a single bad
threshold would also coalesce two distinct events ("the primary fit
cannot be reported under the schema" versus "one neighbouring
threshold happens to be non-DAG") into one error path, which is
what makes the primary-record stop correct there but wrong here.

### SID nullable inside threshold-robustness records

Yes, but **only** inside threshold-robustness records, never in
the primary `run.json`. `docs/08a` Section 8 records the rule for
threshold robustness: "SID is not silently computed on a non-DAG,
SHD may still be computed but is flagged as structurally invalid
in the report." Nullable `sid` in the threshold-robustness record,
plus structured `graph_status`, honours this without introducing a
new enum. The primary `run.json` remains strictly typed with `sid:
int`; the primary-record stop condition in `pipeline.py` continues
to refuse to write a partial primary record when the primary
threshold yields a non-DAG.

### `run.json` immutability

`run.json` must remain immutable. The threshold-robustness record
must be a sibling artefact; `run.json` is at `schema_version = 1`
and modifying its fields without a version bump would silently
break loader contracts elsewhere. Commit 7 must not touch
`run.json`; tests should assert that `run.json` byte-content is
unchanged after `threshold_robustness.json` is written.

### Implementation readiness

Implementation may proceed. No source-code change to wrappers,
metrics, identity, preflight, config, or loader is required. The
Commit 7 module
`experiments/selection_study/threshold_robustness.py` is the only
file that needs to grow beyond its current `NotImplementedError`
stub.

---

## Appendix A: paper citations used in this audit

DAGMA paper, Bello, Aragam, Ravikumar, "DAGMA: Learning DAGs via
M-matrices and a Log-Determinant Acyclicity Characterization",
NeurIPS 2022. In repository at `papers/DAGMA.pdf`.

- Section C.1.1 (page 21, "Small to Moderate Number of Nodes"):
  experimental hyperparameters for the linear SEM setting; final
  thresholding step at 0.3 for all cases.
- Section 4.1 (page 8, "Practical Considerations"): example mu and
  s sequences for `T = 4`.
- Section 5 (page 9): explicit no-hyperparameter-optimisation
  policy.
- Section C (page 20, Remark 5): same no-hyperparameter-optimisation
  policy stated as part of the experimental protocol.
- Section C.1 (page 20): edge-weight distribution
  `Unif([-2, -0.5] union [0.5, 2])` for the linear SEM setting,
  which is the natural range that the project's 0.3 threshold
  operates within.

DCDI paper, Brouillard, Lachapelle, Lacoste-Julien, Lacoste,
Drouin, "Differentiable Causal Discovery from Interventional Data",
NeurIPS 2020. In repository at `papers/DCDI.pdf`.

- Abstract and Introduction (page 1): DCDI is designed to leverage
  interventional data; observational-only is a constrained ablation
  (DCD-no-interv).
- Section 3.1 (page 4): the adjacency matrix is a binary mask
  `M^G in {0, 1}^{d x d}` acting on neural-network inputs.
- Section 3.2 (page 6): the entries of `M^G` are independent
  Bernoulli variables with success probability
  `sigma(alpha_{ij})`. Footnote 2: `sigma(Lambda)` tends to
  become deterministic as the optimisation proceeds.
- Section B.3 (page 30): the explicit deterministic thresholding
  rule at the end of training, "Edges with a sigma(alpha_{ij})
  higher than 0.5 are set to 1 and others set to 0", and the
  illustrative convergence point at iteration 62000 on a sparse
  10-node graph.
- Section B.5 (page 31): the optimiser (RMSprop), the batch size
  (64), the initialisation (Xavier), the activation function
  (leaky-ReLU).
- Table 2 (page 32): the default DCDI hyperparameter table.
- Appendix C.4 (page 35-36): the DCD-no-interv ablation, which is
  exactly the regime this thesis uses for DCDI.
