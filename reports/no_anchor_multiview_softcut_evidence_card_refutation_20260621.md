# No-Anchor Multiview Softcut Evidence Card Refutation

Date: 2026-06-21

## Why This Branch

After direct reassignment and tiny-local merges failed, this branch tested the PDF/Deli-style identity-component evidence card idea: split suspicious target components before trying to merge or relabel into them.

Selection remained no-anchor and no-GT:

- Candidate generation used the current best assignment plus visual/temporal/cannot-link evidence.
- The relaxed probe generated a bounded grid; local selection used no-GT evidence cards, not full IDF1.
- GT appears only in posthoc pair/full labels for refutation.

## Probe Summary

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_identity_card_multiview_softcut_aggressive_probe_20260621/result.json`
- Remote: `/mnt/localssd/vlincs_reid_runs/no_anchor_identity_card_multiview_softcut_aggressive_probe_20260621/result.json`

| probe | rows in JSON top | non-noop rows | finding |
|---|---:|---:|---|
| strict multiview softcut | 16 | 0 | all candidates failed conflict-reduction gate |
| relaxed multiview softcut | 100 | 60 | only aggressive splits moved components |

Views used: fused primary, DINOv2, pose/color, color histogram, FaceNet.

## Full-Score Results

| selected by | split components | split tracklets | pair F1 | IDF1 | HOTA | AssA | density_simple IDF1 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no-GT mass-heavy evidence card | 2 | 455 | 0.770085 | 0.653405 | 0.517077 | 0.532436 |  | reject |
| no-GT single-best component evidence | 1 | 237 | 0.770741 | 0.653781 | 0.517623 | 0.533092 | 0.653924 | reject |

Current best remains `IDF1=0.655817 / HOTA=0.519228 / AssA=0.534791`.

## Interpretation

The split evidence was visually real but operationally harmful:

- high visual margin did not imply a better delivered identity component;
- reducing a small number of cannot-link edges was not enough;
- mass-heavy split selection made things worse;
- even single-best split plus density post-filter stayed below best.

New critic rule: target split requires a component-purity opponent, not just visual margin. A future evidence card needs countertarget negatives, per-video namespace delta, and expected delivery-FP impact before materialization.

## Next Direction

Move from whole-component softcut to subpart-level admission:

1. Find the small split part itself, not the whole component split action.
2. Attach or quarantine only the low-confidence part if it has independent countertarget evidence.
3. Penalize any split that changes hundreds of tracklets unless the no-GT opponent predicts a delivery gain.

Status: pair/global-ID model remains above 70; e2e remains below 70.


## Subpart Attach Follow-Up

A smaller action was tested after the mass-heavy split failed: take the smallest part from the highest no-GT split evidence card and attach only those tracklets to the nearest compatible external component.

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_identity_card_subpart_attach_fullscore_20260621/result.json`
- `local_runs/no_anchor_subpart_attach_evidence_card_refutation_20260621.json`
- Remote: `/mnt/localssd/vlincs_reid_runs/no_anchor_identity_card_subpart_attach_fullscore_20260621/result.json`

| field | value |
|---|---:|
| moved seqs | `8888, 9282` |
| source component | 5 |
| target component | 38 |
| target top5 sim | 0.500794 |
| pair F1 | 0.770741 |
| full IDF1 / HOTA / AssA | 0.653718 / 0.517537 / 0.532993 |

Verdict: reject. The action is tiny and no-GT plausible, but it still hurts global delivery. This suggests current visual-centroid evidence is not calibrated enough for attaching fragments.
