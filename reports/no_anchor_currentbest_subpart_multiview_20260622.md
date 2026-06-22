# No-Anchor Multiview Subpart Candidate Sweep

Date: 2026-06-22

## Standing

- Setting: no-anchor global-id research; GT is used only for canonical evaluation.
- Previous best: IDF1 / HOTA / AssA = `0.657624 / 0.520692 / 0.535785`.
- New best: IDF1 / HOTA / AssA = `0.657653 / 0.520723 / 0.535819`.
- Delta vs previous best: IDF1 `+0.000029`, HOTA `+0.000031`, AssA `+0.000034`.
- E2E target remains open: `0.657653 < 0.70`.

## Method

Starting point was `subpart_combo_r01_r02_17seq_assignments`, the previous no-anchor current best. I generated multiview subpart candidates from five feature views: base DINO, DINO-heavy, SigLIP-fused, weakmetric, and OSNet-010. The side-effect selector penalized known negative transfers such as `55->58`, `2329->40`, `47->2330`, broad high-move edits, and already exhausted `35->60` variants. Ten diverse candidates were full-scored with the canonical p005-area detection filter.

## Results

| Rank | Feature | Edit | Moved | Label | IDF1 | HOTA | AssA | MCAM04 IDF1 | Delta IDF1 |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | weakmetric | `10->22` | 2 | positive | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000029 |
| 2 | weakmetric | `10->9` | 2 | positive | 0.657639 | 0.520707 | 0.535804 | 0.565508 | +0.000015 |
| 3 | weakmetric | `32->9` | 6 | tie | 0.657624 | 0.520692 | 0.535785 | 0.565454 | +0.000000 |
| 4 | siglip_fused | `24->41` | 7 | tie | 0.657624 | 0.520692 | 0.535785 | 0.565454 | +0.000000 |
| 5 | weakmetric | `9->19` | 9 | tie | 0.657624 | 0.520692 | 0.535785 | 0.565454 | +0.000000 |
| 6 | osnet010 | `9->32` | 9 | tie | 0.657624 | 0.520692 | 0.535785 | 0.565454 | +0.000000 |
| 7 | weakmetric | `19->11` | 7 | negative | 0.657541 | 0.520590 | 0.535664 | 0.565673 | -0.000083 |
| 8 | siglip_fused | `31->25` | 3 | negative | 0.657540 | 0.520609 | 0.535721 | 0.565424 | -0.000084 |
| 9 | siglip_fused | `10->15` | 8 | negative | 0.656956 | 0.519885 | 0.534966 | 0.565454 | -0.000668 |
| 10 | weakmetric | `11->40` | 16 | negative | 0.656102 | 0.519236 | 0.534675 | 0.565177 | -0.001522 |

## Interpretation

- `weakmetric 10->22` is promoted as the new current best. It moves only 2 tracklets, but improves aggregate IDF1 and all three headline metrics slightly.
- `weakmetric 10->9` is a second small positive but below the promoted edit; it is useful as a near-positive label.
- `9->32`, `9->19`, `24->41`, and `32->9` tie the previous best and are safe-neutral labels under this current composition.
- `10->15` and especially `11->40` are strong negative side-effect labels. `11->40` moved 16 tracklets and dropped IDF1 to `0.656102`, so the referee should learn a penalty for broad edits even when the heuristic score looks plausible.
- The experiment supports the current AutoResearch loop: candidate retrieval can find micro-promotions, but the full-score gate is still the only reliable arbiter. The next useful step is a trained or calibrated side-effect referee using these positive, tie, and negative labels before composing larger edits.

## Artifacts

- Summary JSON: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_20260622/multiview_results_summary.json`
- Summary CSV: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_20260622/multiview_results_summary.csv`
- Selection manifest: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_20260622/multiview_side_effect_selected_candidates.json`
- Best assignment: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_20260622/weakmetric_assignments/rank29_subpart_s10_to22_2seq_assignments.csv`
- Best score JSON: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_20260622/rank29_subpart_s10_to22_2seq_assignments_density_p005_area.json`
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_multiview_20260622`

