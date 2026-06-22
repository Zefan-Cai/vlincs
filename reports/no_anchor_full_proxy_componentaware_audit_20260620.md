# No-Anchor Full-Proxy Training Audit

- rows: `41`
- features: `96`
- include oracle: `False`
- min full IDF1: `0.55`
- feature mode: `all`
- full IDF1 range: `0.602445` - `0.654009`

## Ridge LOOCV

{
  "alpha": 10.0,
  "corr": -0.056881080024926846,
  "mae": 9719.505845422163,
  "rmse": 62235.19016019376
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `tracklet_pair_f1` | `0.939123` |
| `tracklet_pair_recall` | `0.814798` |
| `delivery_tracklets_max` | `-0.433426` |
| `delivery_tracklets_mean` | `-0.427662` |
| `tracklet_pair_precision` | `0.389550` |
| `delivery_tracklets_min` | `-0.256358` |
| `eval_tracklets` | `-0.256358` |
| `output_tracklets` | `0.178073` |
| `min_target_view_vote` | `0.145196` |
| `preview_min_source_score` | `0.116245` |
| `candidate_search_prefix` | `-0.115237` |
| `candidate_edges` | `-0.105150` |
| `source_expand_sim` | `0.102232` |
| `source_seed_sim` | `0.102232` |
| `source_max_total_groups` | `-0.100009` |
| `source_candidates` | `-0.098765` |
| `source_conflicted_components` | `-0.098765` |
| `source_conflict_edges` | `-0.097210` |
| `source_conflict_nodes` | `-0.096713` |
| `preview_min_target_margin` | `-0.088013` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.654009` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.654009` | `0.77354` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_tq075_full2_20260620.json` |
| `0.653823` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `` | `conflict_subcluster_reassign` | `local_runs/no_anchor_full_proxy_autoresearch_refresh_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_g8_stricttarget_prefix_full4_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.772654` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_pairfull_20260620.json` |
| `0.653823` | `0.774082` | `conflict_subcluster_reassign_candidate_search` | `local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_fullproxy_unique_pairfull_20260620.json` |
