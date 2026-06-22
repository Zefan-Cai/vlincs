# No-Anchor VLINCS: Side-Effect Gate V1 Refutation

Date: 2026-06-22

## Deli AutoResearch distillation

Source pages read:

- Deli_AutoResearch framework: https://victorchen96.github.io/auto_research/framework.html
- AutoResearch paper index / Self-Play survey: https://victorchen96.github.io/auto_research/paper.html
- Self-play story: https://victorchen96.github.io/blog_self_play_story.html

Operational takeaways for this VLINCS run:

1. Treat long-horizon research as a stateful autonomous loop, not a sequence of isolated best-score attempts. Keep explicit progress, findings, directions tried, artifacts, and next direction.
2. Use self-play style counterfactuals: a proposer generates edits, a referee scores real downstream metrics, and the scheduler learns from both positive and negative outcomes.
3. Honest declines matter. Deli's self-play writeup explicitly highlights score drops after external checks; in our setting, negative VLINCS edits must become training labels rather than discarded logs.
4. Submission failures should become pre-submit checks. This run caught one metric hygiene issue: a long remote shell argument caused the p005 config to be eaten for rank02-04, making assignment CSV paths appear as `config_name`. I fixed this by writing the p005 config to a remote file and rerunning with `@p005_area_config.txt`.
5. Pivot after stall. The rolling tiny attach ranker was still improving, but the gain slope is too small. Side-effect gate V1 was the intended structural test; because it failed, next work should change label quality or evidence mass instead of tuning the same ridge.

## Goal context

No anchors are allowed. GT is used only for evaluation and for offline full-score side-effect labels derived from prior scored candidates; it is not used as an anchor or training identity source.

Current best before this experiment:

| Metric | Value |
|---|---:|
| Global-id pair F1 / P / R | 0.775234 / 0.820504 / 0.734698 |
| E2E IDF1 / HOTA / AssA | 0.656225 / 0.519723 / 0.535329 |
| Best assignment | `no_anchor_timeagglom_attach_learned_ranker_roll3_20260622/assignments_learned/rank04_time_agglom_local_attach_source_assignments.csv` |

## What was built

New scripts:

- `kit/build_no_anchor_attach_side_effect_labels.py`
  - Input: candidate JSON plus base/full-score metric JSONs.
  - Output: side-effect labels with global `delta_idf1`, `delta_hota`, `delta_assa`, and per-video delta IDF1.
  - Safety metadata: `uses_anchors=false`, `uses_gt_for_training_or_anchors=false`, `uses_gt_for_evaluation_only=true`.

- `kit/rank_no_anchor_attach_candidates_by_side_effect_labels.py`
  - Trains ridge regressors on metric-derived side-effect labels.
  - Scores no-anchor candidate rows using predicted global delta, weak-video deltas, MCAM04 delta, and negative weak-video penalty.
  - Output ranker version: `tiny_attach_side_effect_ridge_v1`.

Local artifacts:

- Labels:
  - `local_runs/remote_h100_test_3_20260622/no_anchor_attach_side_effect_labels_20260622/roll1_labels.json`
  - `local_runs/remote_h100_test_3_20260622/no_anchor_attach_side_effect_labels_20260622/roll2_labels.json`
  - `local_runs/remote_h100_test_3_20260622/no_anchor_attach_side_effect_labels_20260622/roll3_labels.json`
- Experiment mirror:
  - `local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_side_effect_gate_20260622/`

Remote run:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_side_effect_gate_20260622`

## Side-effect label set

The first label set is tiny: 12 verified edits from roll1-roll3.

Key examples:

| Roll | Edit | Global delta IDF1 | Important local delta |
|---|---|---:|---|
| roll1 | `382+2821+7262->2329` | +0.000058 | MCAM04 +0.000208 |
| roll1 | `421+5326+6250->25` | -0.000103 | MCAM04 -0.000036 |
| roll2 | `9202->15` | +0.000052 | MCAM08 +0.000179 |
| roll3 | `6960->10` | +0.000032 | MCAM06 +0.001274 |
| roll3 | `1065+1083->26` | -0.000036 | negative global side effect |

LOOCV diagnostics already warned that the model was weak:

| Target | LOOCV corr | LOOCV MAE |
|---|---:|---:|
| global delta IDF1 | -0.481683 | 0.000102388 |
| MCAM03 Tc8 delta | -0.084298 | 0.000255462 |
| MCAM04 Tc6 delta | -0.168230 | 0.000060452 |
| MCAM06 Tc6 delta | -0.058447 | 0.000620137 |
| MCAM08 Tc6 delta | +0.551672 | 0.000045088 |

Interpretation: only MCAM08 had a positive calibration signal. Global and MCAM04/MCAM06 side effects were anti-correlated under leave-one-out, so the ranker should be treated as a hypothesis generator, not a trusted scheduler.

## Candidate generation

Base assignment:

- `no_anchor_timeagglom_attach_learned_ranker_roll3_20260622/assignments_learned/rank04_time_agglom_local_attach_source_assignments.csv`

Candidate pool:

- `finalbest_relaxed_candidates.json`
- Raw candidates: 64
- Side-effect selected singles: 12
- Combo candidates evaluated: 4

Top single candidates by side-effect ranker:

| Rank | Edit | Pred global delta | Side-effect score | Candidate rank | Critic score |
|---:|---|---:|---:|---:|---:|
| 1 | `2829->28` | 0.000098820 | 0.000936306 | 20 | 0.265646 |
| 2 | `9487->33` | 0.000115518 | 0.000722648 | 48 | 0.136648 |
| 3 | `3883->29` | 0.000084882 | 0.000685438 | 2 | 0.414697 |
| 4 | `2681->28` | 0.000042187 | 0.000626144 | 23 | 0.232200 |

Single-edit canonical p005 results:

| Rank | Edit | IDF1 | HOTA | AssA | Delta vs best |
|---:|---|---:|---:|---:|---:|
| 1 | `2829->28` | 0.656178 | 0.519651 | 0.535242 | -0.000047 |
| 2 | `9487->33` | 0.656211 | 0.519704 | 0.535308 | -0.000014 |
| 3 | `3883->29` | 0.656166 | 0.519647 | 0.535253 | -0.000059 |
| 4 | `2681->28` | 0.656214 | 0.519711 | 0.535315 | -0.000011 |

The best single edit was extremely close but still below the current best.

## Combo ablation

Because single edits were near-misses, I tested four small combos from the top side-effect candidates.

Canonical path:

`assignment CSV -> full submission zip -> density_simple sourcezip -> p005_area filter -> DS1 HOTA/IDF1 scorer`

Important hygiene note:

- First rank02-04 run had a bad p005 argument due long remote-shell expansion; the JSON showed `config_name` equal to an assignment path and `dropped_rows=0`.
- I wrote `p005_area_config.txt` on the remote and reran rank02-04 using `@p005_area_config.txt`.
- Final table below uses only corrected `config_name=p005_area` JSONs with `dropped_rows=7603`.

| Rank | Combo | Pred delta IDF1 | Side-effect score | Density IDF1 | p005 IDF1 | HOTA | AssA | Delta vs best |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `9487->33 + 2681->28` | 0.000157705 | 0.000953740 | 0.656086 | 0.656200 | 0.519692 | 0.535294 | -0.000025 |
| 2 | `9487->33 + 3883->29` | 0.000200400 | 0.000995667 | 0.656038 | 0.656151 | 0.519628 | 0.535232 | -0.000074 |
| 3 | `9487->33 + 3883->29 + 2681->28` | 0.000242587 | 0.001174463 | 0.656027 | 0.656141 | 0.519616 | 0.535218 | -0.000084 |
| 4 | `2829->28 + 9487->33 + 2681->28` | 0.000256525 | 0.001319302 | 0.656039 | 0.656153 | 0.519621 | 0.535207 | -0.000072 |

Result:

- No combo beats the current best `0.656225`.
- The ranker's predicted score is not monotonic with actual p005 IDF1.
- Best combo is close but still negative.

## Conclusion

Side-effect gate V1 is refuted as a production scheduler.

What worked:

- The infrastructure is useful: it turns prior positive/negative full-score edits into reusable labels.
- The canonical eval path is now cleaner, with an explicit p005 config file to prevent shell argument bugs.
- The candidates remain no-anchor and do not use GT identity labels as anchors.

What failed:

- 12 labels are not enough for a continuous uplift regressor.
- Global delta labels hide per-video interactions.
- MCAM04 and MCAM06, the bottleneck videos, were anti-correlated in LOOCV.
- Combining near-miss edits creates interference; the highest predicted combo was not the best actual combo.

Updated best:

- Unchanged: `IDF1/HOTA/AssA = 0.656225 / 0.519723 / 0.535329`

Next direction:

1. Do not run blind side-effect ridge v1 roll4.
2. Build a richer side-effect label bank:
   - include corrected p005 labels from this run as negatives/near-misses;
   - add per-video deltas for all videos, not only weak videos;
   - label undo/rollback edits on the current best to learn harmful merges directly.
3. Pivot candidate generation toward higher-mass evidence:
   - component-level rollback/split candidates;
   - MCAM04-targeted evidence because it remains a weak cell;
   - multi-tracklet component evidence rather than one-tracklet attaches.
4. Add a pre-submit metric check:
   - p005 JSON must have `config_name=p005_area`;
   - p005 JSON must have nonzero `dropped_rows`;
   - candidate promotion requires corrected p005 JSON, not density-only score.
