# Opponent Scheduler Density-Simple Postfilter Promotion

Date: 2026-06-21

## Result

- Fixed no-GT delivery policy: `density_simple` on candidate source zips.
- New standing best: IDF1 `0.655817` / HOTA `0.519228` / AssA `0.534791`.
- Previous best: IDF1 `0.655385` / HOTA `0.518806` / AssA `0.534548`.
- Delta vs previous best: IDF1 `+0.000432`.

## Ablation

| rank | raw IDF1 | density IDF1 | delta vs raw | delta vs prev best | HOTA | AssA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rank01` | `0.653709` | `0.655817` | `+0.002108` | `+0.000432` | `0.519228` | `0.534791` | promote |
| `rank02` | `0.653305` | `0.655479` | `+0.002174` | `+0.000094` | `0.518943` | `0.534693` | reject_lower_than_rank01 |
| `rank03` | `0.652473` | `0.654642` | `+0.002169` | `-0.000743` | `0.517845` | `0.533555` | reject_lower_than_rank01 |

## Interpretation

- The raw rank01 relink was below the standing best, but the fixed source-zip density filter made it positive.
- This changes the evaluator loop: scheduler candidates should be judged by raw full-score plus fixed delivery-policy postfilter, because the production artifact is the filtered submission zip.
- Countertarget hard-gate remains useful as a cost-control signal, but this run shows a reject_countertarget relink can still yield small e2e improvement after detection filtering. Treat it as provisional evidence, not a merge-quality endorsement.

## Artifacts

- `local_runs/no_anchor_opponent_scheduler_density_simple_postfilter_promotion_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_density_simple_sourcezip_20260621/summary.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_opponent_scheduler_labelled_w0p016_density_simple_sourcezip_20260621/rank01_density_simple_sourcezip.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_opponent_scheduler_labelled_w0p016_density_simple_sourcezip_20260621`
