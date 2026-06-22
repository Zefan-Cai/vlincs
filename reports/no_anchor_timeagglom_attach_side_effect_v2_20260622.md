# No-Anchor VLINCS: Side-Effect Gate V2 Near-Tie Refutation

Date: 2026-06-22

## Why this run

Side-effect gate V1 was refuted, but it produced useful new labels. I fed the corrected V1 single/combo results back into the label bank and trained a V2 gate. This follows the AutoResearch self-play loop: evaluated failures become future scheduler evidence.

No anchors were used. GT was used only through the canonical DS1 scorer and metric-derived side-effect labels.

## Label bank

V2 labels:

- roll1 labels: 4
- roll2 labels: 4
- roll3 labels: 4
- side-effect V1 singles: 4
- side-effect V1 combos: 4
- total after signature de-dup: 20

New label files:

- `local_runs/remote_h100_test_3_20260622/no_anchor_attach_side_effect_labels_20260622/side_effect_gate_v1_single_labels.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_attach_side_effect_labels_20260622/side_effect_gate_v1_combo_labels.json`

V2 ranking artifact:

- `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_side_effect_gate_20260622/side_effect_v2_ranked_candidates.json`

## Diagnostics

| Target | LOOCV corr | LOOCV MAE |
|---|---:|---:|
| global delta IDF1 | -0.256258 | 0.000062690 |
| MCAM03 Tc8 delta | -0.036669 | 0.000131225 |
| MCAM04 Tc6 delta | +0.260832 | 0.000067460 |
| MCAM06 Tc6 delta | -0.051049 | 0.000342888 |
| MCAM08 Tc6 delta | +0.028224 | 0.000073888 |

Compared with V1, MCAM04 became positive, but the global regressor remains anti-correlated. So V2 is only an MCAM04-oriented hypothesis generator, not a reliable production scheduler.

## Candidates scored

Remote run:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_side_effect_v2_20260622`

Canonical path:

`assignment CSV -> full submission zip -> density_simple sourcezip -> p005_area filter -> DS1 HOTA/IDF1 scorer`

The new p005 guard in `kit/run_no_anchor_density_area_pipeline.sh` passed for all four runs:

- `config_name=p005_area`
- `dropped_rows=7603`

| Rank | Edit | Pred global delta | Pred MCAM04 delta | Side-effect score | Density IDF1 | p005 IDF1 | HOTA | AssA | Delta vs best |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `6797->9` | +0.000026933 | +0.000015481 | 0.000339503 | 0.656131 | 0.656225 | 0.519723 | 0.535329 | 0.000000 |
| 2 | `1811+6864->46` | -0.000021096 | +0.000005024 | 0.000288954 | 0.656087 | 0.656182 | 0.519685 | 0.535306 | -0.000043 |
| 3 | `313->21` | +0.000035690 | +0.000037381 | 0.000253117 | 0.656064 | 0.656158 | 0.519659 | 0.535281 | -0.000067 |
| 4 | `6234->17` | +0.000015832 | -0.000052869 | 0.000226105 | 0.656119 | 0.656214 | 0.519710 | 0.535316 | -0.000011 |

## Conclusion

V2 did not beat the current best.

Current best remains:

- `IDF1/HOTA/AssA = 0.656225 / 0.519723 / 0.535329`

What the run taught:

- Adding V1 negatives/near-misses improved MCAM04 calibration but did not make global uplift reliable.
- `6797->9` is a true no-op under p005: it changes density IDF1 but the final p005 score is exactly unchanged.
- `6234->17` is a near-miss hard negative: strong critic, high candidate quality, but still -0.000011 IDF1.
- The local attach family appears saturated around the current best; remaining gains are too small for the 0.70 target.

Next direction:

1. Stop local attach-only side-effect ranker iterations.
2. Pivot to high-mass component evidence:
   - component rollback / split candidates;
   - MCAM04-focused structural repairs;
   - cannot-link or same-frame conflict driven split/rollback labels.
3. Keep V1/V2 labels as a scheduler dataset for rejecting harmful local attaches, not as the main improvement path.
