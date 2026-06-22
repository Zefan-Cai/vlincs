# No-Anchor K3 Clothing Edge Referee Audit

Date: 2026-06-21

## Verdict

Promote the signal, not the current full-graph sweep.

After the budgeted multiview merge dropped to IDF1 `0.653210`, I audited the
nine accepted merge edges with a no-anchor clothing/body verifier. This verifier
uses weak positives from same-stream short-gap continuation and hard negatives
from same-stream overlap cannot-link pairs. It uses no anchors and no GT.

The targeted audit strongly rejected every multiview-accepted edge.

## Why This Audit Was Needed

The earlier full-graph clothing verifier script was too expensive for the
AutoResearch loop at the current k3 interface:

- wide run: stopped after about 13 minutes with no intermediate artifact;
- scoped run: stopped after about 8 minutes with no intermediate artifact;
- tiny run: stopped after about 8 minutes with no intermediate artifact.

The issue was not lack of signal. The issue was tool shape: full-graph
candidate generation, verifier scoring, merge sweep, and optional full-score
were bundled into one black-box step. I added:

- `kit/no_anchor_clothing_edge_audit.py`

This targeted referee trains the same verifier, then scores only specified
component edges. The nine-edge audit completed in about 20 seconds.

## Training Signal

| field | value |
|---|---:|
| weak positive source | same-stream short-gap visual clothing continuation |
| positive candidates considered | 62305 |
| positive train pairs | 2000 |
| hard negative source | same-stream overlap cannot-link |
| hard negative candidates | 174788 |
| negative train pairs | 2000 |
| model | HGB |
| train AUC | 0.999034 |
| train AP | 0.999007 |

The high train scores are not delivery evidence by themselves. They only say
the weak-label construction is separable for the sampled training rows.
Delivery evidence comes from how it scores the proposed merge edges.

## Edge Scores

| source | target | sizes | probability | top-mean probability | OSNet max | posecolor max | colorhist max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 60 | 77 / 6 | 0.000260 | 0.000218 | 0.848892 | 0.975193 | 0.981956 |
| 43 | 71 | 1 / 2 | 0.000292 | 0.000248 | 0.632482 | 0.938931 | 0.868028 |
| 67 | 68 | 1 / 1 | 0.000231 | 0.000231 | 0.460970 | 0.888645 | 0.867890 |
| 16 | 27 | 2 / 1 | 0.000250 | 0.000249 | 0.659631 | 0.836754 | 0.980244 |
| 45 | 70 | 1 / 1 | 0.000265 | 0.000265 | 0.543218 | 0.916168 | 0.920219 |
| 79 | 82 | 1 / 1 | 0.000297 | 0.000297 | 0.535309 | 0.971682 | 0.980821 |
| 18 | 56 | 1 / 1 | 0.000289 | 0.000289 | 0.343025 | 0.956712 | 0.944412 |
| 54 | 65 | 1 / 1 | 0.000234 | 0.000234 | 0.610108 | 0.984301 | 0.928069 |
| 75 | 80 | 1 / 1 | 0.000278 | 0.000278 | 0.426675 | 0.932517 | 0.991301 |

The key case is `9 -> 60`: the multiview proposer liked it because the mean
visual/body support was high and the mass proxy was large. The clothing/body
referee assigns probability `0.000260`, effectively a hard rejection, despite
high max color/body similarities. This is useful because it shows the verifier
is responding to the sample-pair evidence pattern, not simply to one large
color-similarity number.

## Interpretation

This creates a more precise next step:

- do not run another wider multiview merge;
- do not run the current full-graph clothing merge sweep as one opaque job;
- build a cached edge table first, then apply the clothing/body referee as an
  admission filter before any merge is materialized.

In Deli AutoResearch terms:

- proposer: multiview merge found visually plausible high-mass bridges;
- opponent/referee: targeted clothing verifier rejected all nine bridges;
- gate: no merge should be materialized from this edge family until it passes
  the learned referee.

## Next Experiment

Implement or adapt a two-stage cached verifier pipeline:

1. generate a bounded edge table from the current k3 assignment;
2. score edges with the no-anchor clothing/body verifier and write the edge
   table immediately;
3. merge only edges above a high calibrated threshold;
4. full-score only the top admitted policies.

The candidate production hypothesis is now:

`k3 assignment + density selector + cached clothing/body referee gate`

rather than direct multiview bridge merging.

## Artifacts

- `kit/no_anchor_clothing_edge_audit.py`
- `local_runs/no_anchor_k3_clothing_edge_audit_20260621_summary.json`
- `local_runs/no_anchor_k3_clothing_edge_audit_20260621/audit.json`
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_clothing_edge_audit_20260621`
