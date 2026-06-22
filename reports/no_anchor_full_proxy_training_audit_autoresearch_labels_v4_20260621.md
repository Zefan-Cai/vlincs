# No-Anchor Full-Proxy Training Audit

- rows: `80`
- features: `45`
- include oracle: `False`
- min full IDF1: `0.55`
- feature mode: `compact`
- full IDF1 range: `0.602445` - `0.654009`

## Ridge LOOCV

{
  "alpha": 1000.0,
  "corr": -0.03240320436616162,
  "mae": 0.012658462366371238,
  "rmse": 0.06057446982873896
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `moved_tracklets` | `-0.500598` |
| `tracklet_pair_recall` | `0.100459` |
| `preview_mean_source_size` | `0.094972` |
| `candidate_search_prefix` | `-0.089639` |
| `tracklet_pair_f1` | `0.077234` |
| `preview_mean_source_quality` | `-0.060181` |
| `output_tracklets` | `0.054321` |
| `source_size` | `0.044248` |
| `preview_mean_source_score` | `0.043825` |
| `max_sources_per_target` | `0.042534` |
| `source_margin_mean` | `-0.036510` |
| `target_best_sim` | `-0.036510` |
| `source_quality` | `-0.036510` |
| `source_score` | `-0.036510` |
| `target_mean_sim` | `-0.036510` |
| `target_min_view_sim` | `-0.036510` |
| `target_quality` | `-0.036510` |
| `target_score` | `-0.036510` |
| `target_margin` | `-0.036510` |
| `target_size` | `-0.036411` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.654009` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.654009` | `0.77354` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `0.653963` | `0.772364` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_k3_mass_bridge_proxy_search_20260621/search.json` |
| `0.653827` | `0.772685` | `video_temporal_relink` | `local_runs/remote_mcam04_06_temporal_20260621/edge_temporal_target_mcam04_06_ultrawide_top_full.json` |
| `0.653823` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `` | `conflict_subcluster_reassign` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
