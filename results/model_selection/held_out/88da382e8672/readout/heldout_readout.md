# Held-out evaluation readout

heldout_run_hash_prefix: 88da382e8672
parent_calibration_run_hash_prefix: 4a67117a10b1
generated_at_utc (artefact): 2026-05-23T14:09:25Z

## Status

heldout_evaluation.json validates against the held-out evaluation schema.

Records loaded: 25 total (20 main + 5 sensitivity).

All 25 records converged, produced valid DAGs, and had available samplers.

The DCDI fit-RNG sensitivity addendum is a supplementary diagnostic, structurally separate from main evidence; it does not enter the main aggregates.

No prior-loss experiment is started by this readout. Final base-model adjudication is performed outside this generator.

## Scope

This file audits one held-out run identified by the heldout_run_hash above. It loads the held-out evaluation artefact and the 25 per-fit JSON records, writes four CSV summaries and five PNG figures, and emits this markdown report. No model fits are invoked, no input file is modified, and no automatic final-decision logic is applied.

## Main held-out summary

| condition | model | mean SID | mean MMD | mean SHD | mean runtime (s) |
| --- | --- | --- | --- | --- | --- |
| centred_only | dagma | 4.2 | 0.005814472573953375 | 1.0 | 1.0513981799944303 |
| centred_only | dcdi | 63.4 | 0.11588378616596792 | 30.4 | 976.6540220399969 |
| standardised | dagma | 66.6 | 0.12135898121296962 | 25.8 | 1.4749967600102536 |
| standardised | dcdi | 68.8 | 0.14197222340635784 | 29.6 | 902.6897642000113 |

## Per-seed observations

- centred_only / dagma: SID: [0, 0, 0, 8, 13]; MMD: 0.00047128080441082856, 0.002154124142866101, 0.0032764204885345843, 0.004721405279772059, 0.0184491321541833; SHD: [0, 0, 0, 2, 3].
- centred_only / dcdi: SID: [48, 51, 69, 71, 78]; MMD: 0.09673003539130162, 0.10631815594371, 0.12001970322751579, 0.12734844766889647, 0.12900258859841576; SHD: [23, 28, 31, 32, 38].
- standardised / dagma: SID: [64, 64, 65, 69, 71]; MMD: 0.0930076138867018, 0.09451157024966125, 0.09721597837705162, 0.10506762974135164, 0.21699211381008174; SHD: [23, 25, 26, 26, 29].
- standardised / dcdi: SID: [45, 59, 76, 80, 84]; MMD: 0.1263855457497403, 0.13299423652141637, 0.14520574250117319, 0.1485238300548259, 0.1567517622046334; SHD: [26, 27, 29, 33, 33].

## DCDI fit-RNG sensitivity addendum

Target cell: centred_only / dcdi at SCM seed 301.

Sensitivity per-fit values:

| fit_rng | SID | MMD | SHD | runtime (s) |
| --- | --- | --- | --- | --- |
| 43 | 64.0 | 0.1777443666547534 | 31.0 | 1001.5132004999905 |
| 44 | 68.0 | 0.1320883235689646 | 30.0 | 1013.6139970999793 |
| 45 | 61.0 | 0.11677970916006894 | 22.0 | 811.9882216000115 |
| 46 | 83.0 | 0.20505995084606235 | 35.0 | 1006.0180546999909 |
| 47 | 77.0 | 0.10261584885160957 | 32.0 | 840.6431904999772 |

DCDI fit-RNG sensitivity does not suggest the fixed-RNG DCDI result was an isolated outlier: fixed-RNG SID at fit_rng=42 is 78.0 and the fit_rng=43..47 SID range is [61.0, 83.0] (within the sensitivity range).

## Runtime summary

Per-cell mean runtime values appear in main_summary.csv and in the runtime figure (log y-axis).

## Methodological interpretation

- centred_only / dagma has substantially lower mean SID, MMD, and SHD than the other cells: mean SID 4.2 versus 63.4 for centred_only / dcdi.
- standardised appears substantially harder than centred_only for DAGMA: mean SID 66.6 under standardised versus 4.2 under centred_only.

## Generated files

- heldout_readout.md
- main_summary.csv
- per_seed_main.csv
- sensitivity_summary.csv
- status_summary.csv
- heldout_mean_sid.png
- heldout_mean_mmd.png
- heldout_mean_shd.png
- heldout_runtime.png
- heldout_sensitivity_addendum.png

## Reproducibility note

- generator: experiments/selection_study/held_out_readout.py
- inputs: heldout_evaluation.json and records/*.json under the held-out run directory above
- outputs: the CSV summaries, PNG figures, and this markdown report
