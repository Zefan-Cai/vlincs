# No-Anchor Current-k3 Component-Graph Rescue Refutation

Date: 2026-06-21

## Why This Was Run

This was the follow-up structural pivot after the Deli AutoResearch
distillation and the old-base component-rescue refutation.  Instead of
reusing the 2026-06-20 assignment namespace, this run generated component
bridges directly from the current k3 no-anchor assignment:

`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`

The candidate generator is no-anchor.  It reads assignment metadata and
tracklet feature files only.  GT is used only by the canonical DS1 evaluator
after candidate assignments are produced.

## Implementation Note

The first remote run exposed a useful feature-schema bug: current feature NPZs
use `seqs/features`, not `track_ids/vectors`, and their dimensions differ.
`kit/compose_no_anchor_component_graph_candidates.py` now supports both
schemas and aligns `seqs` through the assignment `seq -> tracklet_key` map.
Each feature source is L2-normalized, missing source features are zero-filled,
and the per-tracklet vectors are concatenated before component scoring.

The successful candidate generation used:

- fused feature: `2322` dims
- DINOv2-base: `768` dims
- pose/color: `188` dims

## Candidate Generation

Remote output:

`/mnt/localssd/vlincs_reid_runs/no_anchor_current_k3_component_graph_candidates_20260621`

Local mirror:

`local_runs/remote_h100_test_3_20260621/no_anchor_current_k3_component_graph_candidates_20260621/`

Summary:

| item | value |
| --- | ---: |
| components | `83` |
| eligible components | `83` |
| tracklet embeddings loaded | `7487` |
| raw component-graph rows | `46` |
| rescue rows kept | `2` |

The rescue rule selected two high-similarity, low-vote candidates:

| rank | source | target | moved | graph score | rescue score | best sim | centroid | vote | overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `35` | `40` | `276` | `0.777551` | `0.691658` | `0.823822` | `0.795631` | `0.500` | `0.007040` |
| 2 | `40` | `35` | `74` | `0.766162` | `0.669518` | `0.823822` | `0.795631` | `0.500` | `0.007040` |

## Full-Score Result

Current promoted best:

`IDF1/HOTA/AssA = 0.655911 / 0.519311 / 0.534922`

Both current-k3 component-graph rescue candidates lose.

| rank | edit | moved | raw IDF1 | raw HOTA | raw AssA | p0.5 IDF1 | p0.5 HOTA | p0.5 AssA | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `35 -> 40` | `276` | `0.646435` | `0.512959` | `0.532450` | `0.648646` | `0.514943` | `0.534419` | reject |
| 2 | `40 -> 35` | `74` | `0.646435` | `0.512959` | `0.532450` | `0.648646` | `0.514943` | `0.534419` | reject |

Remote run:

`/mnt/localssd/vlincs_reid_runs/no_anchor_current_k3_component_graph_rescue_r1_2_fullscore_20260621`

Local mirror:

`local_runs/remote_h100_test_3_20260621/no_anchor_current_k3_component_graph_rescue_r1_2_fullscore_20260621/`

## Interpretation

This is a clean hard negative for the AutoResearch opponent.  The local visual
evidence is strong enough to pass retrieval:

- best tracklet-pair similarity: `0.823822`
- component centroid similarity: `0.795631`
- low same-video overlap: `0.007040`

But the full system score drops by `0.007265` raw IDF1 and remains
`0.007265` below current best after the frozen p0.5 delivery filter
comparison.  This means high visual similarity between large components is not
enough; the candidate family needs a side-effect critic that penalizes broad
false merges before full-score budget is spent.

The delivery filter still helps (`0.646435 -> 0.648646`), but it cannot repair
an identity-level false merge.

## Next Direction

Do not run more broad current-k3 component-graph rescue rows as production
candidates without a new critic.  The next proposer should either:

1. prefer small-fragment current-k3 attachments from the candidate tail, where
   moved tracklets are `4-6` instead of `74-276`; or
2. shift down to detector/tracklet regeneration for the known weak videos
   MCAM04 Tc6, MCAM06 Tc6, and MCAM03 Tc8.

This refutation should be added as hard-negative training/evaluation data for
the no-anchor side-effect critic.
