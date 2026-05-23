# Calibration readout

calibration_run_hash_prefix: 4a67117a10b1
generated_at_utc (artefact): 2026-05-22T20:01:17Z
records loaded: 40

## Status

All four selected configurations carry degeneracy_flag=false. The artefact validates structurally and the 40 per-fit records load without identity errors.

This readout uses the selected configurations strictly as the within-model, within-condition calibration selections they represent. No base-model choice is made by this readout; that step belongs to held-out evaluation.

## Scope

This file audits one calibration run identified by the calibration_run_hash above. It loads the 40 per-fit records from the records directory, the rank-1 configuration per (condition, model), and the full 5-candidate ranking per (condition, model). It writes a markdown summary, three CSV tables, and three PNG figures into the readout directory. No model fits are run, no input file is modified, and no final base-model selection is made.

## Selected configurations

| condition | model | selected hash | hyperparameters | mean SID | mean MMD | mean SHD | degeneracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centred_only | dagma | 06ee98d13852 | {"lambda1":0.25} | 0.0 | 0.0059540658211281626 | 0.0 | false |
| centred_only | dcdi | dd39d6325e7d | {"reg_coeff":0.1} | 60.0 | 0.08875084654991505 | 30.5 | false |
| standardised | dagma | 7b345b1b2e85 | {"lambda1":0.1} | 46.0 | 0.09699846001057562 | 18.0 | false |
| standardised | dcdi | 16f92df3d6af | {"reg_coeff":0.3} | 46.0 | 0.1028451515945459 | 25.0 | false |

## Candidate ranking summary

| condition | model | rank | hash | hyperparameters | mean SID | mean MMD | mean SHD | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centred_only | dagma | 1 | 06ee98d13852 | {"lambda1":0.25} | 0.0 | 0.0059540658211281626 | 0.0 | true |
| centred_only | dagma | 2 | 2d3c669afb1e | {"lambda1":0.1} | 0.0 | 0.006442551460912249 | 0.0 | false |
| centred_only | dagma | 3 | d8045d32dd87 | {"lambda1":0.05} | 3.0 | 0.008343560705224698 | 1.0 | false |
| centred_only | dagma | 4 | 4e2dcb231c27 | {"lambda1":0.025} | 3.5 | 0.003888878201468997 | 0.5 | false |
| centred_only | dagma | 5 | 501f38765f49 | {"lambda1":0.01} | 5.5 | 0.004297208161665568 | 2.0 | false |
| centred_only | dcdi | 1 | dd39d6325e7d | {"reg_coeff":0.1} | 60.0 | 0.08875084654991505 | 30.5 | true |
| centred_only | dcdi | 2 | 3aaff84947d2 | {"reg_coeff":1.0} | 58.0 | 0.11307361421379457 | 31.5 | false |
| centred_only | dcdi | 3 | ba78e78a188c | {"reg_coeff":0.03} | 67.5 | 0.11439396619440374 | 25.5 | false |
| centred_only | dcdi | 4 | 53fc88693ed5 | {"reg_coeff":0.3} | 73.5 | 0.11415362755828673 | 34.5 | false |
| centred_only | dcdi | 5 | 0a151792af82 | {"reg_coeff":0.01} | 75.5 | 0.16086494893648481 | 33.0 | false |
| standardised | dagma | 1 | 7b345b1b2e85 | {"lambda1":0.1} | 46.0 | 0.09699846001057562 | 18.0 | true |
| standardised | dagma | 2 | 2c17153a69e9 | {"lambda1":0.01} | 67.0 | 0.10572477355312308 | 24.0 | false |
| standardised | dagma | 3 | 6ae2c1d1ec9a | {"lambda1":0.025} | 68.5 | 0.10580835598945951 | 26.0 | false |
| standardised | dagma | 4 | 4fe8dc48eedd | {"lambda1":0.25} | 78.0 | 0.1657184002281885 | 29.0 | false |
| standardised | dagma | 5 | 545be5dbcbbb | {"lambda1":0.05} | 79.0 | 0.15705911963306782 | 24.5 | false |
| standardised | dcdi | 1 | 16f92df3d6af | {"reg_coeff":0.3} | 46.0 | 0.1028451515945459 | 25.0 | true |
| standardised | dcdi | 2 | c1978a4c02bf | {"reg_coeff":1.0} | 61.5 | 0.12737214733743882 | 25.5 | false |
| standardised | dcdi | 3 | 5904acec44be | {"reg_coeff":0.03} | 75.0 | 0.13324711130265554 | 26.0 | false |
| standardised | dcdi | 4 | d064fb91dba6 | {"reg_coeff":0.01} | 75.0 | 0.15328208569921856 | 32.0 | false |
| standardised | dcdi | 5 | 6d74ecd1e072 | {"reg_coeff":0.1} | 80.0 | 0.12339795230677081 | 38.0 | false |

## Status/failure summary

| model | condition | training_status | graph_status | sampler_status | count |
| --- | --- | --- | --- | --- | --- |
| dagma | centred_only | converged | valid_dag | available | 10 |
| dagma | standardised | converged | valid_dag | available | 10 |
| dcdi | centred_only | converged | valid_dag | available | 10 |
| dcdi | standardised | converged | valid_dag | available | 10 |

## Calibration observations

- DAGMA centred_only selected lambda1=0.25 with mean SID 0.0 and mean SHD 0.0.
- Standardised calibration appears harder than centred_only for DAGMA: mean SID is 46.0 under standardised versus 0.0 under centred_only.
- DCDI calibration SID is materially higher than DAGMA in centred_only: 60.0 versus 0.0.

## Incident note

A FileExistsError incident affected the dagma / centred_only / seed 201 fit. The stale per-run directory was a residue from an earlier interrupted attempt and produced one degenerate per-fit record. The incident was repaired before this readout was generated; this readout inspects the post-repair selected_configurations.json. The audit trail for the incident lives at docs/08g_file_exists_error_incident.md.

## Standard-deviation note

The std fields in the tables and the error bars in the figures are sample standard deviations computed from n=2 calibration seeds with ddof=1. They are range/variation indicators on a two-element sample, not strong uncertainty estimates. Held-out evaluation will use 5 seeds, which will provide more informative variability estimates.

## Generated files

- calibration_readout.md
- selected_configurations_summary.csv
- candidate_ranking_summary.csv
- status_summary.csv
- calibration_mean_sid.png
- calibration_mean_mmd.png
- calibration_mean_shd.png

## Reproducibility note

- generator: experiments/selection_study/calibration_readout.py
- inputs: selected_configurations.json and records/*.json under the calibration run directory above
- outputs: the CSV summaries and PNG figures listed above, plus this markdown report
