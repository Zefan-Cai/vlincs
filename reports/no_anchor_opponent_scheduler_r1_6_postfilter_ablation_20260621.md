# Opponent Scheduler Ranks 1-6 Postfilter Ablation

Date: 2026-06-21

## Result

- Best remains rank01 after fixed `density_simple` source-zip postfilter: IDF1 `0.655817` / HOTA `0.519228` / AssA `0.534791`.
- Ranks 4-6 do not beat rank01, so blindly expanding this scheduler queue is low value.

| rank | raw IDF1 | density IDF1 | postfilter gain | HOTA | AssA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rank01` | `0.653709` | `0.655817` | `+0.002108` | `0.519228` | `0.534791` | standing_best |
| `rank02` | `0.653305` | `0.655479` | `+0.002174` | `0.518943` | `0.534693` | negative_for_postfilter_reranker |
| `rank03` | `0.652473` | `0.654642` | `+0.002169` | `0.517845` | `0.533555` | negative_for_postfilter_reranker |
| `rank04` | `0.652939` | `0.655169` | `+0.002230` | `0.518442` | `0.534042` | negative_for_postfilter_reranker |
| `rank05` | `0.652791` | `0.654968` | `+0.002177` | `0.518305` | `0.534038` | negative_for_postfilter_reranker |
| `rank06` | `0.653041` | `0.655214` | `+0.002173` | `0.518549` | `0.534233` | negative_for_postfilter_reranker |

## Interpretation

- The fixed delivery filter consistently adds about +0.002 IDF1, but it does not erase identity-side damage.
- Rank01 is the only row in ranks 1-6 that clears the current standing best after delivery filtering.
- Next research should train or update a postfilter-aware full-score proxy using density-filtered labels, not raw full-score labels alone.

## Artifacts

- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_density_simple_sourcezip_20260621/summary.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_r4_6_fullscore_20260621/summary_postfilter.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_r4_6_density_simple_sourcezip_20260621`
- `local_runs/no_anchor_opponent_scheduler_r1_6_postfilter_ablation_20260621.json`
