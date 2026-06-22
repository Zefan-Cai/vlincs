# No-Anchor Current-Best Rank01 Local Continuity Refutation

Date: 2026-06-21

## Context

Standing no-anchor delivery best:

- model-side pair F1/P/R: `0.775234 / 0.820504 / 0.734698`
- e2e delivery IDF1/HOTA/AssA: `0.655817 / 0.519228 / 0.534791`

After the coarse softcut split failed, this branch tested whether local
continuity can produce safer, smaller edits:

1. `local_track_id` relink: use the tracklet key's local-track grouping.
2. video-temporal relink: link adjacent `(video, current_component)` nodes using
   time gap, endpoint geometry, and multiview visual similarity.

No anchors were used.  GT was used only for pair/full evaluation.

## Local-Track Relink

Remote probe:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_rank01_localtrack_relink_probe_20260621/localtrack_probe.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_rank01_localtrack_relink_forced_single_20260621/localtrack_forced_single.json`

Local artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_localtrack_relink_probe_20260621/localtrack_probe.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_localtrack_relink_forced_single_20260621/localtrack_forced_single.json`

Result:

| probe | touched groups | rewritten seqs | pair F1 | note |
|---|---:|---:|---:|---|
| grid top 50 | `0` | `0` | `0.770741` | all top rows no-op |
| forced permissive single config | `0` | `0` | `0.770741` | no usable groups even with max components disabled |

Decision: low coverage.  The current `tracklet_key` local id does not expose
useful cross-fragment continuity for the active assignment.

## Video-Temporal Relink

Remote probe:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_rank01_video_temporal_relink_probe_20260621/result.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_rank01_video_temporal_oneedge_fullscore_20260621`

Local artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_video_temporal_relink_probe_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_video_temporal_relink_probe_20260621/result.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_video_temporal_oneedge_fullscore_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_video_temporal_oneedge_fullscore_20260621/top_full_export.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_video_temporal_oneedge_fullscore_20260621/density_simple_sourcezip.json`

Probe coverage:

| field | value |
|---|---:|
| video nodes | `380` |
| candidate edges in top config | `1` |
| accepted-edge CSV rows | `18 / 729` |
| accepted edges per accepted row | `1` |
| video-local pair F1/P/R | `0.347938 / 0.820894 / 0.220752` |

The full-scored edge:

| field | value |
|---|---:|
| video | `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` |
| source node / base label | `351 / 31` |
| target node / base label | `377 / 80` |
| source / target tracklets | `5 / 1` |
| app sim | `0.690218` |
| score | `0.703277` |
| gap ms | `3166` |
| center distance norm | `1.495013` |

Full-score:

| output | IDF1 | HOTA | AssA | note |
|---|---:|---:|---:|---|
| standing best density-simple | `0.655817` | `0.519228` | `0.534791` | current best |
| one-edge raw full-score | `0.653709` | `0.517528` | `0.532975` | below best |
| one-edge density-simple | `0.655817` | `0.519228` | `0.534791` | exactly ties best |

Decision: neutral/low coverage.  A single temporal edge is not harmful after the
density-simple delivery policy, but it does not improve the standing best.

## Conclusion

This refutes local continuity as currently implemented:

- `local_track_id` continuity has no coverage.
- video-temporal adjacency finds only one useful edge in the target MCAM04/08
  slice.
- density-simple can neutralize that edge, but cannot turn it into a gain.

The next branch must expand candidate recall before adding more gates.  The
candidate generator should look beyond direct temporal adjacency:

1. crop-pair retrieval from neighboring frames around fragment endpoints;
2. component-level predecessor/successor search with wider temporal windows;
3. motion/size extrapolation to propose candidates before visual gating;
4. a no-GT side-effect predictor for unmatched-FP and per-video namespace drift.

