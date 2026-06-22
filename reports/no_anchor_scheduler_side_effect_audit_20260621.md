# No-Anchor Scheduler Side-Effect Admission Audit

Date: 2026-06-21

This audit is the VLINCS adaptation of the Deli AutoResearch pivot rule: after repeated score drops, change the structural guard instead of retuning thresholds.  It uses no anchors and no GT for production selection; optional full-score values are shown only as post-hoc labels.

## Summary

- candidate rows: `7`
- recommendation counts: `{'defer_needs_counterfactual': 2, 'reject_or_quarantine': 5}`
- base assignment: `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv`

## Candidate Rows

| manifest | rank | moved | preview | proxy | known full | overlap | target videos | target max size | risk | recommendation | reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json` | 1 | 12 | 1 | 0.6581794732947428 | 0.652184 | 0 | 9 | 158 | 2.0 | `defer_needs_counterfactual` | `large_multivideo_target; historical_fullscore_below_best` |
| `no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json` | 2 | 12 | 1 | 0.6583016512693843 | None | 0 | 9 | 93 | 3.0 | `defer_needs_counterfactual` | `large_multivideo_target; not_scored_namespace_incompatible` |
| `no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json` | 3 | 8 | 1 | 0.6552536343495614 | None | 8 | 10 | 207 | 6.75 | `reject_or_quarantine` | `same_video_temporal_overlap; large_multivideo_target; large_target_component; not_scored_provenance_lookup_failed` |
| `no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json` | 4 | 268 | 3 | 0.6623610047923143 | 0.625017 | 360 | 10 | 172 | 8.5 | `reject_or_quarantine` | `moved_tracklets>20; multi_edge_preview; same_video_temporal_overlap; large_multivideo_target; historical_fullscore_below_best` |
| `no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json` | 3 | 276 | 4 | 0.6667896529501237 | 0.624978 | 600 | 10 | 216 | 9.25 | `reject_or_quarantine` | `moved_tracklets>20; multi_edge_preview; same_video_temporal_overlap; large_multivideo_target; large_target_component; historical_fullscore_below_best` |
| `no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json` | 1 | 28 | 3 | 0.6657882243786953 | 0.650714 | 47 | 10 | 216 | 9.25 | `reject_or_quarantine` | `moved_tracklets>20; multi_edge_preview; same_video_temporal_overlap; large_multivideo_target; large_target_component; historical_fullscore_below_best` |
| `no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json` | 2 | 256 | 2 | 0.6643894901570562 | 0.626798 | 218 | 8 | 216 | 9.25 | `reject_or_quarantine` | `moved_tracklets>20; multi_edge_preview; same_video_temporal_overlap; large_multivideo_target; large_target_component; historical_fullscore_below_best` |

## Decision

Use this as an admission/referee layer before spending canonical full-score budget.  Candidates with replay failure, namespace drift, multi-edge movement, temporal overlap, or historically negative full-score labels should be quarantined or reduced to smaller counterfactuals before remote scoring.
