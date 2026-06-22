# No-Anchor Full-Proxy Training Audit

- rows: `37`
- features: `34`
- include oracle: `False`
- min full IDF1: `0.0`
- feature mode: `compact`
- full IDF1 range: `0.085353` - `0.654009`

## Ridge LOOCV

{
  "alpha": 0.1,
  "corr": 0.997456703330382,
  "mae": 0.003380873738961346,
  "rmse": 0.013988874187266931
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `eval_tracklets` | `0.998394` |
| `delivery_tracklets_min` | `0.998394` |
| `output_tracklets` | `0.998343` |
| `delivery_tracklets_mean` | `0.997850` |
| `tracklet_pair_f1` | `-0.905949` |
| `tracklet_pair_recall` | `-0.898706` |
| `tracklet_pair_precision` | `-0.889737` |
| `coverage_ratio` | `0.421024` |
| `candidate_search_prefix` | `-0.145800` |
| `preview_mean_source_quality` | `-0.096091` |
| `max_sources_per_target` | `0.069965` |
| `preview_mean_source_score` | `0.059892` |
| `accepted_reassignments` | `0.053445` |
| `moved_tracklets` | `0.053445` |
| `target_components_used` | `0.053445` |
| `max_reassignments` | `0.053445` |
| `full_side_effect_proxy` | `-0.051662` |
| `preview_min_target_min_view_sim` | `-0.041775` |
| `preview_mean_target_mean_sim` | `-0.036568` |
| `preview_mean_target_view_vote` | `-0.011992` |

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
