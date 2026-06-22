# No-Anchor Full-Proxy Training Audit

- rows: `41`
- features: `35`
- include oracle: `True`
- min full IDF1: `0.0`
- feature mode: `compact`
- full IDF1 range: `0.085353` - `0.706202`

## Ridge LOOCV

{
  "alpha": 0.1,
  "corr": 0.9961015364320167,
  "mae": 0.006663740367464346,
  "rmse": 0.01659461624559356
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `delivery_tracklets_mean` | `0.995432` |
| `eval_tracklets` | `0.995296` |
| `delivery_tracklets_min` | `0.995296` |
| `output_tracklets` | `0.995227` |
| `tracklet_pair_recall` | `-0.645055` |
| `tracklet_pair_f1` | `-0.577253` |
| `tracklet_pair_precision` | `-0.428392` |
| `coverage_ratio` | `0.422342` |
| `candidate_search_prefix` | `-0.121583` |
| `preview_mean_source_quality` | `-0.080350` |
| `max_sources_per_target` | `0.058632` |
| `preview_mean_source_score` | `0.050284` |
| `accepted_reassignments` | `0.044749` |
| `moved_tracklets` | `0.044749` |
| `target_components_used` | `0.044749` |
| `max_reassignments` | `0.044749` |
| `full_side_effect_proxy` | `-0.043230` |
| `preview_min_target_min_view_sim` | `-0.034951` |
| `preview_mean_target_mean_sim` | `-0.030580` |
| `preview_mean_target_view_vote` | `-0.009986` |

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
