# No-Anchor Learned Attach Ranker Rolls

Date: 2026-06-22

## Result

The learned tiny-attach ranker produced a chain of verified no-anchor micro-promotions:

| step | best edit added | IDF1 | HOTA | AssA | delta IDF1 |
|---|---|---:|---:|---:|---:|
| previous combo 4+5 base | rank04+rank05 from hand-critic pool | 0.656083 | 0.519553 | 0.535174 | -- |
| roll1 | 382+2821+7262->2329 | 0.656141 | 0.519617 | 0.535233 | +0.000058 |
| roll2 | 9202->15 | 0.656193 | 0.519675 | 0.535276 | +0.000052 |
| roll3 | 6960->10 | 0.656225 | 0.519723 | 0.535329 | +0.000032 |

Current no-anchor e2e best:
- IDF1/HOTA/AssA = 0.656225 / 0.519723 / 0.535329
- DetRe/DetPr = 0.574789 / 0.764545
- p005 area dropped rows = 7603

The global-id pair model remains above target:
- pair F1/P/R = 0.775234 / 0.820504 / 0.734698

The end-to-end target is still not met:
- 0.656225 < 0.70

## Ranker Diagnostics

| roll | base | candidate count | training rows | LOOCV corr | LOOCV MAE |
|---|---|---:|---:|---:|---:|
| roll1 | combo45base | 67 | 62 | 0.035787 | 0.000061451 |
| roll2 | roll1 rank01 base | 66 | 58 | 0.276849 | 0.000074747 |
| roll3 | roll2 rank02 base | 65 | 54 | 0.086520 | 0.000072857 |

Interpretation:
- The ranker has real but tiny scheduling signal.
- The signal is not stable enough for blind roll4.
- The useful output is a labeled micro-edit dataset with positives, hard negatives, and near-zero edits.

## Roll2 Ablation

Base: roll1 best assignment.

| rank | edit | density IDF1 | p005 IDF1 | verdict |
|---:|---|---:|---:|---|
| 1 | 2013->37 | 0.655987 | 0.656083 | negative, cancels roll1 gain |
| 2 | 9202->15 | 0.656099 | 0.656193 | promoted |
| 3 | 986->2329 | 0.656047 | 0.656141 | neutral/back to roll1 |
| 4 | 7208->10 | 0.656055 | 0.656149 | weak positive but below rank2 |

## Roll3 Ablation

Base: roll2 rank02 best assignment.

| rank | edit | density IDF1 | p005 IDF1 | verdict |
|---:|---|---:|---:|---|
| 1 | 1065+1083->26 | 0.656058 | 0.656157 | negative |
| 2 | 7176->48 | 0.656099 | 0.656193 | neutral/no-op after p005 |
| 3 | 3662->37 | 0.656090 | 0.656184 | near miss |
| 4 | 6960->10 | 0.656131 | 0.656225 | promoted |

## Final Per-Video Metrics

| video | IDF1 | HOTA | AssA | DetRe | DetPr |
|---|---:|---:|---:|---:|---:|
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.879281 | 0.815354 | 0.836352 | 0.849217 | 0.911551 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.828377 | 0.749534 | 0.785530 | 0.760104 | 0.910124 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.691923 | 0.582848 | 0.624020 | 0.600633 | 0.815936 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.628528 | 0.510443 | 0.550903 | 0.533540 | 0.764663 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.562439 | 0.448616 | 0.493734 | 0.476722 | 0.685737 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.711374 | 0.600973 | 0.639418 | 0.626314 | 0.823168 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.793298 | 0.699897 | 0.729415 | 0.730553 | 0.867834 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.610296 | 0.518506 | 0.600591 | 0.479388 | 0.839559 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704885 | 0.590536 | 0.620589 | 0.629400 | 0.800943 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.769911 | 0.662239 | 0.680787 | 0.726550 | 0.818775 |

The final roll mainly improves MCAM06 Tc6 and MCAM08 Tc6. MCAM04 Tc6 remains the main bottleneck at IDF1 0.562439.

## Artifacts

Local mirrors:
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_20260622/`
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_roll2_20260622/`
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_roll3_20260622/`

Remote runs:
- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_learned_ranker_20260622`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_learned_ranker_roll2_20260622`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_learned_ranker_roll3_20260622`

Final best metric file:
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_roll3_20260622/rank04_time_agglom_local_attach_source_assignments_density_p005_area.json`

Final best assignment:
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_roll3_20260622/assignments_learned/rank04_time_agglom_local_attach_source_assignments.csv`

## Next Direction

Do not continue blind ridge roll4 as-is. The marginal returns are too small and the top-ranked roll3 rows were mostly negative or neutral.

Next research direction:
1. Convert all verified attach edits into a side-effect dataset with per-video deltas.
2. Train or hand-build a conservative gate that predicts MCAM04/MCAM06/MCAM08 side effects, not just global delta.
3. Search for candidates specifically touching the low-video bottlenecks, especially MCAM04 Tc6, with the current tiny-attach positives as low-risk examples.
4. If this still produces only <0.0001 IDF1 gains, pivot back to higher-mass component or detector-tracklet evidence.

