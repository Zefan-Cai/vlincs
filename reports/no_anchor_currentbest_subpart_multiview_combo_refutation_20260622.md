# No-Anchor Multiview Subpart Combo Refutation

Date: 2026-06-22

## Standing

- Setting: no-anchor global-id research; GT is used only for canonical evaluation.
- Current best before this run: IDF1 / HOTA / AssA = `0.657653 / 0.520723 / 0.535819`.
- New best found: no. The standing best remains `0.657653 / 0.520723 / 0.535819`.
- E2E target remains open: `0.657653 < 0.70`.

## Method

This run tested whether the latest promoted edit, `weakmetric 10 -> 22`, can be amplified by composing it with the low-risk/tie multiview subpart candidates from the previous sweep. The candidate manifest originally had feature-local rank numbers that collide across feature views, so the combo manifest normalized `rank = selection_rank` and preserved `original_rank` for provenance.

The base assignment was `subpart_combo_r01_r02_17seq_assignments.csv`. Twelve combinations were generated and full-scored on h100-test-3 with the canonical p005-area detection filter.

## Results

| Ranks | Edits | Assignment | Moved | Label | IDF1 | HOTA | AssA | MCAM04 IDF1 | Delta IDF1 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `2+1` | `10->22+9->32` | `subpart_combo_r02_r01_11seq_assignments` | 11 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+5` | `10->22+24->41` | `subpart_combo_r02_r05_9seq_assignments` | 9 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+6` | `10->22+32->9` | `subpart_combo_r02_r06_8seq_assignments` | 8 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+7` | `10->22+9->19` | `subpart_combo_r02_r07_11seq_assignments` | 11 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+1+5` | `10->22+9->32+24->41` | `subpart_combo_r02_r01_r05_18seq_assignments` | 18 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+1+6` | `10->22+9->32+32->9` | `subpart_combo_r02_r01_r06_17seq_assignments` | 17 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+1+7` | `10->22+9->32+9->19` | `subpart_combo_r02_r01_r07_20seq_assignments` | 20 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+5+6` | `10->22+24->41+32->9` | `subpart_combo_r02_r05_r06_15seq_assignments` | 15 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+5+7` | `10->22+24->41+9->19` | `subpart_combo_r02_r05_r07_18seq_assignments` | 18 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+6+7` | `10->22+32->9+9->19` | `subpart_combo_r02_r06_r07_17seq_assignments` | 17 | tie | 0.657653 | 0.520723 | 0.535819 | 0.565566 | +0.000000 |
| `2+3` | `10->22+19->11` | `subpart_combo_r02_r03_9seq_assignments` | 9 | negative | 0.657570 | 0.520622 | 0.535698 | 0.565785 | -0.000083 |
| `2+1+3` | `10->22+9->32+19->11` | `subpart_combo_r02_r01_r03_18seq_assignments` | 18 | negative | 0.657570 | 0.520622 | 0.535698 | 0.565785 | -0.000083 |

## Interpretation

- No composed candidate improved over the standing current best.
- Ten combinations tied the current best exactly. These edits are useful as safe-neutral labels, but they do not add measurable global-ID value when layered on top of `10 -> 22`.
- The two combinations containing selection rank `3` (`19 -> 11`) dropped to `IDF1 0.657570`. This confirms `19 -> 11` as a robust negative side-effect label even though its MCAM04 local IDF1 is slightly higher.
- The next useful direction is not more blind composition of tie edits. It should train/calibrate a side-effect referee and search for new high-recall false-split repairs, with rank-3 style counterexamples included as hard negatives.

## Artifacts

- Summary JSON: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/combo_results_summary.json`
- Summary CSV: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/combo_results_summary.csv`
- Combo manifest: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/combos/subpart_combo_manifest.json`
- Selection-rank manifest: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/multiview_selection_rank_manifest.json`
- Batch log: `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/p005_combo_batch.log`
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_multiview_combo_20260622`
