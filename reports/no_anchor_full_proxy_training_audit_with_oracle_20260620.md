# No-Anchor Full-Proxy Training Audit

- rows: `36`
- features: `29`
- include oracle: `True`
- min full IDF1: `0.55`
- feature mode: `compact`
- full IDF1 range: `0.602445` - `0.706202`

## Ridge LOOCV

{
  "alpha": 10.0,
  "corr": 0.9290210949245336,
  "mae": 0.0040831873135505565,
  "rmse": 0.007219429137424084
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `tracklet_pair_f1` | `0.964504` |
| `tracklet_pair_recall` | `0.944068` |
| `tracklet_pair_precision` | `0.847460` |
| `preview_mean_target_mean_sim` | `0.006945` |
| `full_side_effect_proxy` | `0.006621` |
| `preview_mean_target_best_sim` | `0.006619` |
| `preview_min_target_min_view_sim` | `0.006426` |
| `preview_mean_source_quality` | `0.005933` |
| `preview_mean_target_view_vote` | `0.005596` |
| `preview_mean_source_score` | `0.005114` |
| `accepted_reassignments` | `-0.004777` |
| `moved_tracklets` | `-0.004777` |
| `target_components_used` | `-0.004777` |
| `max_reassignments` | `-0.004777` |
| `candidate_search_prefix` | `0.004052` |
| `max_sources_per_target` | `-0.000215` |
| `target_margin` | `0.000000` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.706202` | `1.0` | `oracle_all_gt_majority` | `local_runs/remote_h100_test_3_20260620/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json` |
| `0.706008` | `0.996362` | `split_top_40_then_merge_top_40` | `local_runs/remote_h100_test_3_20260620/no_anchor_softcut_current_oracle_repair_decomposition_full_top1_20260620.json` |
| `0.705997` | `0.99604` | `split_top_40_then_merge_top_40` | `local_runs/remote_h100_test_3_20260619/no_anchor_oracle_repair_decomposition_full_top1_20260619.json` |
| `0.654009` | `0.77354` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_full_20260620.json` |
