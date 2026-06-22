# No-Anchor Density Simple Source-Zip Promotion

Date: 2026-06-21

## Result

| artifact | IDF1 | HOTA | AssA | DetRe | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| previous standing `density_oracle_lite` | `0.655378` | `0.518798` | `0.534546` | `0.573869` | superseded |
| source-zip `density_simple` | `0.655385` | `0.518806` | `0.534548` | `0.573941` | promote micro-best |

Delta vs previous best: IDF1 `+0.000007`, HOTA `+0.000008`, AssA `+0.000002`.

## Provenance Note

- This promotion uses the same source-zip filtering path as the earlier density ablation: `--source-zip full.zip --policies density_simple`.
- Rebuilding from `assignments.csv` with the same named policy produced only IDF1 `0.653427`, so the delivery path is part of the artifact provenance and must not be silently swapped.
- No anchors are used; GT is used only by the final evaluator. The policy itself is fixed and no-GT.

## Artifacts

- `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_density_simple_sourcezip_20260621/density_simple_sourcezip.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_density_simple_sourcezip_20260621/density_simple_sourcezip.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/full.zip`
- `local_runs/no_anchor_density_simple_sourcezip_promotion_20260621.json`
