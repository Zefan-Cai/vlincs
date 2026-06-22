# No-Anchor Full-Proxy Training Audit

- rows: `6`
- features: `136`
- include oracle: `False`
- min full IDF1: `0.0`
- feature mode: `all`
- full IDF1 range: `0.654642` - `0.655817`

## Ridge LOOCV

{
  "alpha": 100.0,
  "corr": 0.39378805863068744,
  "mae": 0.00026625717082503125,
  "rmse": 0.00034234128101401975
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `tracklet_pair_f1` | `0.987251` |
| `tracklet_pair_recall` | `0.927594` |
| `preview_max_source_target_video_jaccard` | `-0.885624` |
| `preview_mean_source_target_video_jaccard` | `-0.885624` |
| `preview_min_source_target_video_jaccard` | `-0.885624` |
| `tracklet_pair_precision` | `0.807492` |
| `preview_max_combined_opponent_risk_score` | `-0.745051` |
| `preview_max_visual_opponent_risk_score` | `-0.745051` |
| `preview_mean_combined_opponent_risk_score` | `-0.745051` |
| `preview_mean_visual_opponent_risk_score` | `-0.745051` |
| `preview_min_combined_opponent_risk_score` | `-0.745051` |
| `preview_min_visual_opponent_risk_score` | `-0.745051` |
| `preview_max_source_quality` | `-0.739795` |
| `preview_mean_source_quality` | `-0.739795` |
| `preview_min_source_quality` | `-0.739795` |
| `source_quality` | `-0.739795` |
| `preview_max_source_internal_sim` | `0.677542` |
| `preview_mean_source_internal_sim` | `0.677542` |
| `preview_min_source_internal_sim` | `0.677542` |
| `source_internal_sim` | `0.677542` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.655817` | `0.770741` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655479` | `0.769857` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655214` | `0.768921` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655169` | `0.768538` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.654968` | `0.768234` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.654642` | `0.767572` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
