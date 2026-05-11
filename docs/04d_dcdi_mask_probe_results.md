# DCDI Mask Probe Results

## Purpose

Resolve the DCDI mask-handling ambiguity raised during review of
`docs/02_base_model_selection.md` v1.2 Section 4.2. The question is
whether wrapper-side sampling can force DCDI's `forward_given_params`
to ignore non-surviving parents of a target node, so that the DCDI MMD
sampler honours the thresholded valid DAG and does not let soft edges
in `P = sigmoid(log_alpha)` influence sampled values.

This probe extends `docs/04c_runtime_probe_results.md` with a single
new test, **C-P9**. It does not modify wrappers, project source,
tests, external repositories, or `docs/02`. The probe script lives
under the untracked `inspection/probes/` directory.

## Probe metadata

- Probe script: `inspection/probes/c_p9_dcdi_mask.py`
- Target source: `external/source_inspection/dcdi`, commit
  `594d328eae7795785e0d1a1138945e28a4fec037`
- Python environment: project `.venv`
- Device: CPU
- Imports: `dcdi.models.learnables.LearnableModel_NonLinGaussANM`
  via the same low-level path verified by C-P1 and C-P2 in
  `docs/04c_runtime_probe_results.md`.

## What was run

The probe constructs a 3-node `LearnableModel_NonLinGaussANM` in
observational mode (`intervention=False`,
`intervention_type="perfect"`, `intervention_knowledge="known"`,
`num_regimes=1`) and then runs three steps.

1. Inspect the signature of `forward_given_params`.
2. Test the `mask=` keyword argument with the structural-mask shape
   `(num_vars, num_vars)` and with the documented intervention-mask
   shape `(bs, num_vars)`.
3. Enforce a structural parent mask via `model.adjacency` plus a
   saturated `model.gumbel_adjacency.log_alpha`, then test whether
   varying an excluded parent leaves the target's density parameters
   invariant and whether varying an included parent changes them.

The structural mask used in step 3 sets parent 0 included for target
2, parent 1 EXCLUDED for target 2, and parent 0 included for target 1
(peripheral). The forced adjacency is

```
[[0, 1, 1],
 [0, 0, 0],
 [0, 0, 0]]
```

in the project's row-source / column-destination convention. The
saturated `log_alpha` matches the structural mask
(`+100` on edges, `-100` elsewhere) so the Gumbel mask `M` is
deterministically aligned with `model.adjacency` for the duration of
this probe.

For step 3 the two probe batches differ only in the column of the
parent under test (column 1 for the excluded parent, column 0 for the
included parent). RNG is seeded identically before each
`forward_given_params` call so the Gumbel draw is the same across
paired calls.

## Captured output (verbatim)

```
forward_given_params signature: (x, weights, biases, mask=None, regime=None)

=== Step 1: mask= argument with structural-mask shape (3, 3) ===
  call accepted with structural-shape mask
  density_params shapes: [(4, 1), (4, 1), (4, 1)]

=== Step 2: mask= argument with intervention-mask shape (bs, num_vars) ===
  call accepted with intervention-shape mask
  density_params shapes: [(4, 1), (4, 1), (4, 1)]
  note: this is the INTERVENTION mask path, not structural masking

=== Step 3: enforce structural mask via model.adjacency + log_alpha ===
model.adjacency:
[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
model.get_w_adj():
[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

density_params[2] for x_a       : [-0.077677, -0.167538, -0.677595, -0.075625]
density_params[2] for x_excluded: [-0.077677, -0.167538, -0.677595, -0.075625]

max |delta| target 2 when EXCLUDED parent varied: 0.000000e+00
excluded-parent invariance holds? True
density_params[2] for x_included: [-0.391989, -0.381649, -0.322958, -0.392225]

max |delta| target 2 when INCLUDED parent varied: 3.546366e-01
included-parent sensitivity observed? True
```

## Interpretation

### Q1: Was the explicit `mask=` argument accepted?

The signature is
`forward_given_params(x, weights, biases, mask=None, regime=None)`.
The function accepts the `mask=` argument with EITHER a
structural-mask-shaped tensor `(num_vars, num_vars)` or an
intervention-mask-shaped tensor `(bs, num_vars)`. Neither call raised.

However, the source code reveals that the `mask` argument is the
**intervention mask**, not a structural parent mask. In
`base_model.py:138` (the layer-0 einsum), the structural part of the
forward pass is

```
x = einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
```

where `M = self.gumbel_adjacency(bs)` and `adj = self.adjacency`. The
`mask` argument enters only in branches that handle imperfect or
unknown interventions (`base_model.py:142-167`). In observational mode
(`intervention=False`, which is the selection-study path), the `mask`
argument is silently ignored at the first-layer einsum.

This means **passing a structural mask via `mask=` does nothing**. The
call is accepted but has no effect on which parents reach the target.

### Q2: Did excluded-parent invariance hold under structural masking?

Yes. With `model.adjacency` and `log_alpha` forced to the structural
mask, varying the excluded parent (column 1) by `+5.0` produced
`max |delta target 2| = 0.000000e+00`. The four sample values for
`density_params[2]` were bit-identical to the baseline.

Two factors made the result clean:

- `model.adjacency[1, 2] = 0` zeroes the parent-1-to-target-2
  contribution in the einsum (the `ljt` axis of `adj`).
- `log_alpha[1, 2] = -100` makes `gumbel_sigmoid` produce `M[:, 1, 2]
  = 0` deterministically, so `M * adj` is also zero at that position.

Either alone is sufficient because the einsum multiplies `M` by `adj`.
Setting both is belt-and-braces and recommended for the wrapper.

### Q3: Did included-parent sensitivity hold?

Yes. Varying the included parent (column 0) by `+5.0` produced
`max |delta target 2| = 3.546366e-01`, well above the `1e-3` threshold
set in the probe. The four sample values for `density_params[2]`
shifted visibly between the baseline and the included-parent batch.

The included-parent test could have been weak if random MLP weights
happened to align such that node 0's path to node 2 was effectively
zero. That did not occur here. The probe still reported that this
check is secondary and the excluded-parent invariance is the
load-bearing result.

## Recommendation

**Option A is feasible.** The wrapper enforces structural masking by:

1. Computing `A_thresh = model.get_w_adj() >= 0.5` (or equivalently
   `model.get_w_adj() > 0.5`, matching the in-source threshold at
   `dcdi/train.py:210` and the project's external threshold) AFTER
   training has finished and the model is in `eval()` mode.
2. Verifying `A_thresh` is a valid DAG; if not, setting
   `sampler_status = unavailable_invalid_graph`.
3. Before sampling, COPYING `A_thresh` into `model.adjacency` and
   saturating `model.gumbel_adjacency.log_alpha` to match
   (`+100` on edges, `-100` elsewhere). Both writes are needed: the
   first guarantees structural masking via `adj`; the second
   guarantees that the Gumbel mask `M` does not stochastically
   re-introduce excluded parents through the `M * adj` product, which
   matters because both `M` and `adj` are multiplied in the einsum.
4. Calling `forward_given_params(x, weights, biases)` with no `mask=`
   argument (the minimal call pattern verified by C-P5 and C-P7).
5. Restoring the original `model.adjacency` and `log_alpha` after
   sampling if the wrapper needs to leave the model object untouched
   for downstream use.

Option B (`log_alpha` saturation alone) would also work because the
einsum multiplies `M` by `adj`. But Option A as described above
hardens the masking on both sides and is what `dcdi/train.py:320-328`
does internally at the end of training. The probe verified the
combined approach; testing `log_alpha`-only was not run because it
would be strictly weaker.

Option C is not needed. Doc 02 v1.2's claim that DCDI MMD sampling
uses the thresholded valid DAG as the active sampling graph is
mechanically achievable in the source.

## Does Doc 02 need a correction?

The normative claim in Doc 02 v1.2 Section 4.2 ("The thresholded
valid DAG is the active sampling graph. ... soft edges in P that do
not survive the project threshold MUST NOT be used as parents during
sampling") is correct and supported by this probe.

However, Doc 02 v1.2 step 11 does not yet describe HOW the wrapper
must enforce this masking. A one-line clarification would help. The
suggested addition to Doc 02 Section 4.2 step 11 is:

> Before calling `forward_given_params`, set `model.adjacency` to a
> tensor copy of `A_thresh` and set `model.gumbel_adjacency.log_alpha`
> to a saturated tensor (`+100` on edges in `A_thresh`, `-100`
> elsewhere). Both writes are required because the structural part of
> DCDI's forward pass multiplies `M = gumbel_adjacency(bs)` by
> `model.adjacency` in the layer-0 einsum.

This is an implementation note that closes the gap between the
normative claim and the mechanism. Doc 02 is NOT amended in this
commit. The correction is recorded here as a proposal.

## Summary

- What was run: probe `c_p9_dcdi_mask.py` with three steps.
- Mask shape used: structural mask of shape `(3, 3)`; intervention
  mask of shape `(4, 3)`; saturated `log_alpha` of shape `(3, 3)`.
- Explicit `mask=` accepted: yes for both shapes, but the argument is
  the intervention mask and does NOT enforce structural masking in
  observational mode.
- Excluded-parent invariance held: yes,
  `max |delta target 2| = 0.000000e+00` when the excluded parent
  varied by `+5.0`.
- Included-parent sensitivity observed: yes,
  `max |delta target 2| = 3.546366e-01` when the included parent
  varied by `+5.0`.
- Recommended policy: **Option A** (set `model.adjacency = A_thresh`
  and saturate `log_alpha` before sampling). The wrapper restores the
  originals after sampling.
- Doc 02 needs a one-line correction in Section 4.2 step 11 to spell
  out the mechanism. Not applied in this commit.

## What this probe did NOT change

- No file under `src/`, `tests/`, or `external/source_inspection/`
  was modified.
- No project document was modified, including `docs/02_base_model_selection.md`.
- No dependency was installed, removed, or upgraded.
- The probe script lives under `inspection/probes/`, which is outside
  the tracked project tree.
