# No-Anchor Clamped Postfilter Proxy Refutation

- Setup: clamped postfilter-aware ridge proxy trained on ranks 1-6, side-effect risk weight `0.016`.
- Delivery policy: fixed source-zip `density_simple`.
- Decision: reject all three new clamped top candidates and add them as hard-negative labels.

| rank | move | pair F1 | predicted | raw IDF1 | density IDF1 | density HOTA | density AssA | decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `39->24 (8 trk)` | 0.768849 | 0.656817 | 0.652866 | 0.655037 | 0.518467 | 0.534288 | reject; below best 0.655817 |
| 2 | `14->24 (8 trk)` | 0.768284 | 0.656817 | 0.652973 | 0.655140 | 0.518440 | 0.534130 | reject; below best 0.655817 |
| 3 | `29->50 (8 trk)` | 0.767746 | 0.656817 | 0.652170 | 0.654343 | 0.517579 | 0.533299 | reject; below best 0.655817 |

## What this refutes

- Clamping prevents absurd extrapolated scores, but does not by itself distinguish local-positive edges from global harmful merges.
- The top clamped candidates all move 8 tracklets and have pair F1 around `0.768`, yet source-zip density delivery remains below the standing best.
- Next scheduler training should treat these as hard negatives and weight component/video side effects more heavily.

## Artifacts

- `local_runs/no_anchor_postfilter_proxy_clamped_r1_3_refutation_summary_20260621.json`
- `local_runs/no_anchor_postfilter_proxy_labels_20260621/postfilter_labels_r1_9_with_clamped_negatives.json`
- `local_runs/no_anchor_temporal_clean_bridge_queue_postfilter_proxy_scheduler_clamped_w0p016_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_postfilter_proxy_clamped_r1_3_fullscore_20260621/manifest_assignments.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_postfilter_proxy_clamped_r1_3_density_simple_sourcezip_20260621`
