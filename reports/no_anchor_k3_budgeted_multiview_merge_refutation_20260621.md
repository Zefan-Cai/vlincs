# No-Anchor K3 Budgeted Multiview Merge Refutation

Date: 2026-06-21

## Verdict

Rejected as a production merge policy.

This branch was launched after the oracle repair diagnostic showed that the
current k3 assignment can exceed the end-to-end target if false-split structure
is repaired: the eval-only oracle rows reached IDF1 about `0.706`. The question
here was whether a no-GT multiview bridge proposer could emulate that repair by
using visual similarity, trajectory/forbidden constraints, body-part features,
color histograms, DINO, and DB embeddings.

It did not. The best full-scored rows all landed below the standing best.

## Setup

Run on h100-test-3:

- input assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- primary feature:
  `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`
- extra views:
  posecolor, colorhist, DINOv2, DB embedding
- candidate top-k: `128`
- edge top-k per component: `8`
- rank ks: `3,5,10`
- sim thresholds: `0.62,0.66,0.70`
- score modes: `hybrid, mean_min, min_sim`
- accepted edge budgets: `10,20,40`
- one edge per component: enabled
- no anchors, no GT for training or admission

## Results

| Row | accepted edges | max accepted edges | max component size | pair F1 | full IDF1 | HOTA | AssA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 10 | 300 | 0.769163 | 0.653210 | 0.517030 | 0.532678 |
| 2 | 9 | 20 | 300 | 0.769163 | 0.653210 | 0.517030 | 0.532678 |
| 3 | 9 | 40 | 300 | 0.769163 | 0.653210 | 0.517030 | 0.532678 |
| 4 | 9 | 10 | 500 | 0.769163 | 0.653210 | 0.517030 | 0.532678 |

Standing best remains:

| Run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 |

## Failure Case

The first accepted edge dominates the accepted mass:

| field | value |
|---|---:|
| source component | 9 |
| target component | 60 |
| source size / target size | 77 / 6 |
| source weight / target weight | 14162 / 1283 |
| source rank / target rank | 25 / 12 |
| score | 0.666768 |
| mean view similarity | 0.826601 |
| min view similarity | 0.570151 |
| rank vote | 0.40 |
| sim vote | 0.80 |
| bridge mass proxy | 4262.61 |

This is exactly the kind of edge the current proposer likes: high mass, visually
plausible mean similarity, and enough per-view support to pass a loose gate.
But it is not a strong identity edge. The minimum view similarity is weak, the
mutual rank evidence is weak, and the full-score result drops below baseline.

## Interpretation

This refutes wider budgeted multiview merging at the current admission boundary.
It does not refute the oracle gap itself.

The important distinction:

- GT oracle repair says the current component graph contains enough structure
  to reach `>0.70` IDF1.
- This no-GT proposer cannot reliably identify those repair edges yet.

So the next branch should not relax thresholds, increase the accepted edge
budget, or add another rank-by variant over the same edge family. The bottleneck
has moved to an edge verifier that can reject high-mass impostor bridges.

## Next Pivot

Turn the refuted edges into opponent data:

1. train a no-anchor cross-tracklet verifier from weak positives and hard
   cannot-link negatives;
2. include clothing/body consistency and sample-pair evidence;
3. specifically check whether it downranks the `9 -> 60` style bridge;
4. only then materialize merge edits into the forced global-ID output.

The next active experiment is a strict clothing/body continuation verifier on
the current k3 assignment, with full-score top rows.

## Artifacts

- `local_runs/no_anchor_k3_budgeted_multiview_merge_20260621_summary.json`
- `local_runs/no_anchor_k3_budgeted_multiview_merge_20260621/merge.json`
- `local_runs/no_anchor_k3_budgeted_multiview_merge_20260621/merge.csv`
- `local_runs/no_anchor_k3_budgeted_multiview_merge_20260621/top_assignments.csv`
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_budgeted_multiview_merge_20260621`
