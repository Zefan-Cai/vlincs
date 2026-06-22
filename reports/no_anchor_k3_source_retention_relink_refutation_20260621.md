# No-Anchor K3 Source-Retention Relink Critic

Date: 2026-06-21

## Purpose

The previous conflict-outlier relink run found a real single-tracklet repair,
but target-margin ranking admitted false relinks at top-2/top-3.  This run adds
a no-GT source-retention critic to the relink proposer.

No anchors are used.  Candidate generation uses only:

- current k3 assignment;
- same-stream temporal cannot-link;
- fused tracklet features;
- source/target component centroid and nearest-neighbor support;
- source/target same-video temporal support.

GT is used only after materialization for full-score and eval-only diagnosis.

## Code

Updated:

- `kit/no_anchor_conflict_outlier_relink.py`

New candidate features:

- `source_nn_max`, `source_nn_topk_mean`
- `target_nn_max`, `target_nn_topk_mean`
- `neighbor_margin = target_nn_topk_mean - source_nn_topk_mean`
- `source_temporal_support`, `source_temporal_overlap`
- `target_temporal_support`, `target_temporal_overlap`
- rank modes: `centroid`, `source_retention`, `target_neighbor`

## Candidate Filter

The useful no-GT config was:

- `rank_mode=source_retention`
- `min_neighbor_margin=0.18`
- `max_source_neighbor_sim=0.75`
- `min_target_sim=0.72`
- `min_margin=0.02`
- `support_top_k=3`

It reduced the broader 44-candidate pool to two accepted relinks:

| rank | seq | move | camera | neighbor margin | source nn max | target nn mean |
|---:|---:|---|---|---:|---:|---:|
| 1 | `5828` | `10 -> 22` | MCAM04 | `0.318018` | `0.647238` | `0.853282` |
| 2 | `5777` | `35 -> 40` | MCAM04 | `0.189292` | `0.731339` | `0.857426` |

Eval-only audit:

| seq | GT | source top GT | target top GT | verdict |
|---:|---:|---:|---:|---|
| `5828` | `48` | `7` | `48` | true relink |
| `5777` | `20` | `36` | `20` | true relink |

The same audit explains why less strict configs are unsafe:

| seq | move | GT | source top GT | target top GT | verdict |
|---:|---|---:|---:|---:|---|
| `3727` | `29 -> 11` | `51` | `3` | `12` | false relink |
| `9343` | `49 -> 28` | `40` | `19` | `40` | true relink but ranked after false relink |

## Full-Score

| config | pair F1 | pair precision | pair recall | IDF1 | HOTA | AssA |
|---|---:|---:|---:|---:|---:|---:|
| top-1 | `0.769463` | `0.816638` | `0.727441` | `0.653258` | `0.517085` | `0.532738` |
| top-2 | `0.769474` | `0.816792` | `0.727338` | `0.653384` | `0.517144` | `0.532701` |

Applying the no-GT density filter to the top-2 assignment:

| policy | IDF1 | HOTA | AssA | dropped rows |
|---|---:|---:|---:|---:|
| `density_oracle_lite` | `0.653595` | `0.517138` | `0.532803` | `34434` |
| `density_simple` | `0.653600` | `0.517144` | `0.532802` | `33685` |

Standing best remains:

| standing run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 + density primary | `0.655378` | `0.518798` | `0.534546` |

## Group Probe

I also reran the strict source-retention gate with a wider per-source proposal
budget:

- `max_candidates_per_source=20`
- `top_k in {1,2,4,8,16}`
- same strict `min_neighbor_margin=0.18` and `max_source_neighbor_sim=0.75`

The candidate count stayed at two for every non-top-1 row.  In other words, the
gate is not merely clipping a longer ranked list; the current single-tracklet
conflict-outlier proposer has only two high-quality no-GT relinks under this
evidence model.

Artifact:

- `local_runs/no_anchor_k3_source_retention_group_probe_20260621/result.json`

## Interpretation

The source-retention critic works at the candidate-quality level:

- it removes the earlier false relinks `7232` and `6241`;
- it keeps two eval-only true relinks;
- it improves the relink+density result from `0.653473` to `0.653600`.

But the delivery impact is tiny.  Two true single-tracklet relinks do not move
enough mass to challenge the standing k3+density result.

Decision: keep the source-retention critic, but do not spend more full-score
budget on single-tracklet relinks alone.  The group probe makes this stronger:
relaxing `top_k` and per-source budget does not uncover more strict candidates,
so the next improvement must come from a grouped/component proposer, not another
single-node threshold sweep.

## Next

Use the critic as an admission gate inside a higher-mass proposer:

1. generate candidate groups around MCAM04/MCAM08 hard components rather than
   isolated conflict nodes;
2. require each member to pass source-retention checks;
3. rank groups by detection mass and hard-video coverage before full-score.

The next useful question is no longer "can we find a true relink?"  We can.  The
question is whether we can find a no-anchor group of true relinks large enough
to affect IDF1 without introducing false merges.
