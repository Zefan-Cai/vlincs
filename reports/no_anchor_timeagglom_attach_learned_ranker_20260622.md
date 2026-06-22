# No-Anchor Time-Agglom Attach Learned Ranker

Date: 2026-06-22

## Summary

This iteration applied the Deli AutoResearch loop to the current no-anchor DS1 best. The hypothesis was that previously verified tiny local edits can train a small no-GT scheduler ranker that selects better next attach edits around the combo 4+5 base.

Result: a micro-promotion, not a breakthrough.

- Previous no-anchor e2e best: IDF1/HOTA/AssA = 0.656083 / 0.519553 / 0.535174.
- New best: IDF1/HOTA/AssA = 0.656141 / 0.519617 / 0.535233.
- Delta IDF1 = +0.000058.
- Global-id model remains above target: pair F1/P/R = 0.775234 / 0.820504 / 0.734698.
- End-to-end remains below target: 0.656141 < 0.70.

## Data And Method

Candidate pools were regenerated on h100-test-3:

| pool | base assignment | raw candidates | selected |
|---|---:|---:|---:|
| oldbase relaxed | old labelled w0.016 base | 70 | 70 |
| rank08base relaxed | currentbest rank08 base | 69 | 69 |
| combo45base relaxed | combo 4+5 base | 67 | 67 |

Training labels came from prior canonical full-score deltas:

| label | delta IDF1 |
|---|---:|
| 4973->55 | +0.000037 |
| 3558->2329 | -0.000013 |
| 4580->21 | -0.000068 |
| 4859->29 | -0.000106 |
| 9435->8 | -0.000009 |
| 6234->17 | -0.000012 |
| 1249+1291->50 | +0.000120 |
| 3322->3 | +0.000015 |

The ranker is a tiny ridge model over no-GT temporal/namespace/quality features. It is only a budget scheduler. It does not train identity anchors.

Ranker diagnostics:
- matched training rows: 62
- LOOCV correlation: 0.035787
- LOOCV MAE: 0.000061451

Interpretation: the ranker is weak but has enough signal to find one local positive. It should not be trusted as a global ordering model.

## Top-4 Ablation

All rows were materialized from `combo_4_5_assignments.csv`, then evaluated with the canonical density + p005 area path.

| rank | edit | learned delta | moved tracklets | density IDF1 | p005 IDF1 |
|---:|---|---:|---:|---:|---:|
| 1 | 382+2821+7262->2329 | +0.000105 | 3 | 0.656047 | 0.656141 |
| 2 | 421+5326+6250->25 | +0.000093 | 3 | 0.655879 | 0.655980 |
| 3 | 8085+8105->39 | +0.000082 | 2 | 0.656039 | 0.656133 |
| 4 | 140+7471+8319->28 | +0.000077 | 3 | 0.655878 | 0.656049 |

Best candidate:
- `rank01_time_agglom_local_attach_source_assignments`
- moved seqs: 382, 2821, 7262
- source component: 19
- target component: 2329
- target dominance in time component: 0.923729

## Per-Video Metrics For New Best

| video | IDF1 | HOTA | AssA | DetRe | DetPr | delta IDF1 vs previous best |
|---|---:|---:|---:|---:|---:|---:|
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.879281 | 0.815354 | 0.836352 | 0.849217 | 0.911551 | +0.000000 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.828377 | 0.749534 | 0.785530 | 0.760104 | 0.910124 | -0.000173 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.691923 | 0.582848 | 0.624020 | 0.600633 | 0.815936 | +0.000000 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.628528 | 0.510443 | 0.550903 | 0.533540 | 0.764663 | +0.000000 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.562439 | 0.448616 | 0.493734 | 0.476722 | 0.685737 | +0.000208 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.711374 | 0.600973 | 0.639418 | 0.626314 | 0.823168 | +0.000000 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.793298 | 0.699897 | 0.729415 | 0.730553 | 0.867834 | +0.000000 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.609022 | 0.517275 | 0.599722 | 0.477036 | 0.841981 | +0.000000 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704885 | 0.590536 | 0.620589 | 0.629400 | 0.800943 | -0.000934 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.769732 | 0.662040 | 0.680704 | 0.726071 | 0.818979 | +0.000000 |

The global gain comes mostly from MCAM04 Tc6, partly offset by MCAM06 Tc8. This reinforces that local attach edits have cross-video side effects even when only a few tracklets move.

## Artifacts

Local mirror:
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_20260622/`

Remote run:
- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_learned_ranker_20260622`

Important files:
- `kit/rank_no_anchor_attach_candidates_by_verified_labels.py`
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_20260622/learned_ranker_combo45_candidates.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_20260622/manifest_learned_ranker_assignments.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_learned_ranker_20260622/rank01_time_agglom_local_attach_source_assignments_density_p005_area.json`

## Next Direction

Do not keep spending full-score budget on independent single attach rank lists. The signal is too small.

Next experiment:
1. Build a combo-aware attach ranker that scores interactions among locally positive edits.
2. Add per-video side-effect features, especially MCAM04 Tc6 gain versus MCAM06 Tc8 loss.
3. Generate a larger pool from the new rank01 base, but gate by predicted side-effect and temporal impossibility.
4. If the next two iterations stay below +0.0001 IDF1, pivot away from tiny attach and back to a higher-mass candidate family.

