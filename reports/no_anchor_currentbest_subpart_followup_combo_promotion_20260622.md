# No-Anchor Current-Best Subpart Follow-up Combo Promotion - 2026-06-22

## Summary

This iteration continues the no-anchor VLINCS global-ID research loop. Ground truth is used only for official evaluation, not as anchors, training seeds, or identity labels.

Promoted result:

- Candidate: `subpart_combo_r01_r02_17seq_assignments`
- Edit family: extend the previous `35->60` MCAM04 Tc6 positive with the balanced rank02 same-family repair
- Moved tracklets: 17
- E2E IDF1 / HOTA / AssA: `0.657624 / 0.520692 / 0.535785`
- Previous best IDF1 / HOTA / AssA: `0.657475 / 0.520599 / 0.535769`
- Delta IDF1 / HOTA / AssA: `+0.000149 / +0.000093 / +0.000016`

The global-id pair model remains above the model-side target (`F1=0.775234`, `P=0.820504`, `R=0.734698`), but the end-to-end goal remains open because promoted E2E IDF1 is still below `0.70`.

## Ablation Results

| candidate | moved | IDF1 | HOTA | AssA | DetPr | DetRe | MCAM04 IDF1 | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `subpart_combo_r01_r02_17seq_assignments` | 17 | 0.657624 | 0.520692 | 0.535785 | 0.765616 | 0.576331 | 0.565454 | promoted: smaller tied best |
| `subpart_combo_r01_r02_r07_19seq_assignments` | 17 | 0.657624 | 0.520692 | 0.535785 | 0.765616 | 0.576331 | 0.565454 | tied best, extra r07 no aggregate gain |
| `subpart_combo_r01_r02_r05_19seq_assignments` | 17 | 0.657590 | 0.520652 | 0.535748 | 0.765494 | 0.576349 | 0.565454 | negative side-effect label |
| `subpart_combo_r01_r02_r05_r07_21seq_assignments` | 17 | 0.657590 | 0.520652 | 0.535748 | 0.765494 | 0.576349 | 0.565454 | negative side-effect label |
| `subpart_combo_r01_r07_16seq_assignments` | 17 | 0.657475 | 0.520599 | 0.535769 | 0.765435 | 0.576205 | 0.565121 | ablation |
| `subpart_combo_r01_r05_16seq_assignments` | 17 | 0.657441 | 0.520559 | 0.535732 | 0.765313 | 0.576222 | 0.565121 | negative side-effect label |
| `subpart_combo_r01_r05_r07_18seq_assignments` | 17 | 0.657441 | 0.520559 | 0.535732 | 0.765313 | 0.576222 | 0.565121 | negative side-effect label |

Interpretation:

- `r01+r02` and `r01+r02+r07` tie on aggregate metrics; `r01+r02` is promoted because it is the smaller tied best.
- `r05` variants are consistent negative side-effect labels and should be treated as hard negatives for the next side-effect referee.
- `r07` alone merely recovers the previous best; it is not an independent positive beyond the `r02` gain.

## Per-Video Metrics For Promoted Best

| video | IDF1 | HOTA | AssA | DetPr | DetRe |
|---|---:|---:|---:|---:|---:|
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc6` | 0.879281 | 0.815354 | 0.836352 | 0.911551 | 0.849217 |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc8` | 0.828377 | 0.749534 | 0.785530 | 0.910124 | 0.760104 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | 0.692726 | 0.583770 | 0.624754 | 0.815714 | 0.601966 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | 0.628528 | 0.510282 | 0.550552 | 0.764663 | 0.533540 |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | 0.565454 | 0.449375 | 0.492028 | 0.688457 | 0.479740 |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc6` | 0.711374 | 0.600973 | 0.639418 | 0.823168 | 0.626314 |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc8` | 0.793298 | 0.699897 | 0.729415 | 0.867834 | 0.730553 |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | 0.610296 | 0.518506 | 0.600591 | 0.839559 | 0.479388 |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | 0.704885 | 0.590536 | 0.620589 | 0.800943 | 0.629400 |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | 0.769911 | 0.662145 | 0.680594 | 0.818775 | 0.726550 |

## Artifacts

Local lightweight run copy:

```text
/Users/zcai/Codex/vlincs_reid_by_search/local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_followup_20260622
```

Remote full run, including submission zips:

```text
/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_followup_20260622
```

Key files:

```text
local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_followup_20260622/combo_results_summary.json
local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_followup_20260622/combo_results_summary.csv
local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_followup_20260622/subpart_combo_r01_r02_17seq_assignments_density_p005_area.json
local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_followup_20260622/balanced_combos/assignments/subpart_combo_r01_r02_17seq_assignments.csv
```

S3 publication target:

```text
s3://dit-scale-up/zcai/vlincs/
remote_runs_h100-test-3_20260622/no_anchor_currentbest_subpart_followup_20260622/
current_best_no_anchor/
core_snapshot_20260622/
```

## Next Direction

Use this iteration as a labelled side-effect bank:

- positive: `35->60` with balanced rank02 extension (`r01+r02`)
- neutral/tied: `r07` only when paired with `r02`
- negative: all `r05` variants

Next step is a no-anchor subpart side-effect referee/ranker that predicts whether a proposed subpart edit improves canonical p005 IDF1 before paying the full official score cost.
