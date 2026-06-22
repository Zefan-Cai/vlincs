# No-Anchor Full-Proxy Training Audit

- rows: `9`
- features: `136`
- include oracle: `False`
- min full IDF1: `0.0`
- feature mode: `all`
- full IDF1 range: `0.654343` - `0.655817`

## Ridge LOOCV

{
  "alpha": 10.0,
  "corr": 0.7147492676696399,
  "mae": 0.00020818137043548339,
  "rmse": 0.00028547348593806454
}

## Top Feature Correlations

| feature | corr |
| --- | ---: |
| `tracklet_pair_f1` | `0.919585` |
| `tracklet_pair_recall` | `0.841020` |
| `preview_max_source_quality` | `-0.793732` |
| `preview_mean_source_quality` | `-0.793732` |
| `preview_min_source_quality` | `-0.793732` |
| `source_quality` | `-0.793732` |
| `tracklet_pair_precision` | `0.735006` |
| `preview_max_target_mean_sim` | `0.680332` |
| `preview_mean_target_mean_sim` | `0.680332` |
| `preview_min_target_mean_sim` | `0.680332` |
| `target_mean_sim` | `0.680332` |
| `preview_max_view_margin_mean` | `0.645095` |
| `preview_mean_view_margin_mean` | `0.645095` |
| `preview_min_view_margin_mean` | `0.645095` |
| `full_side_effect_proxy` | `0.640594` |
| `target_score` | `0.620544` |
| `preview_max_target_best_sim` | `0.612787` |
| `preview_mean_target_best_sim` | `0.612787` |
| `preview_min_target_best_sim` | `0.612787` |
| `target_best_sim` | `0.612787` |

## Top Full Rows

| full IDF1 | pair F1 | mode | artifact |
| ---: | ---: | --- | --- |
| `0.655817` | `0.770741` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655479` | `0.769857` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655214` | `0.768921` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655169` | `0.768538` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.655140` | `0.768284` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_9_with_clamped_negatives.json` |
| `0.655037` | `0.768849` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_9_with_clamped_negatives.json` |
| `0.654968` | `0.768234` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.654642` | `0.767572` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_6.json` |
| `0.654343` | `0.767746` | `conflict_subcluster_reassign_candidate_search` | `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_9_with_clamped_negatives.json` |
