# No-Anchor Full-Proxy Training Audit

- rows: `32`
- features: `34`
- include oracle: `False`
- min full IDF1: `0.55`
- feature mode: `compact`
- full IDF1 range: `0.602445` - `0.654009`

## Ridge LOOCV

{
  "alpha": 1.0,
  "corr": 0.9690758142599307,
  "mae": 0.0017359237256611738,
  "rmse": 0.004556118723893414
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `tracklet_pair_f1` | `0.940311` |
| `tracklet_pair_recall` | `0.816396` |
| `delivery_tracklets_mean` | `-0.404175` |
| `tracklet_pair_precision` | `0.378136` |
| `eval_tracklets` | `-0.236495` |
| `delivery_tracklets_min` | `-0.236495` |
| `output_tracklets` | `0.166086` |
| `candidate_search_prefix` | `-0.152187` |
| `preview_mean_source_quality` | `-0.094346` |
| `max_sources_per_target` | `0.075353` |
| `preview_mean_source_score` | `0.073235` |
| `accepted_reassignments` | `0.049875` |
| `moved_tracklets` | `0.049875` |
| `target_components_used` | `0.049875` |
| `max_reassignments` | `0.049875` |
| `full_side_effect_proxy` | `-0.044893` |
| `preview_min_target_min_view_sim` | `-0.034513` |
| `preview_mean_target_mean_sim` | `-0.028021` |
| `accepted_edges` | `-0.012449` |
| `preview_mean_target_best_sim` | `0.011088` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.654009` | `0.77354` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_top1_full_20260620.json` |
| `0.653724` | `0.772399` | `conflict_subcluster_reassign` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_narrow_top1_full_20260620.json` |
| `0.653541` | `0.775234` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_full1_20260620.json` |
| `0.653541` | `0.775064` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
