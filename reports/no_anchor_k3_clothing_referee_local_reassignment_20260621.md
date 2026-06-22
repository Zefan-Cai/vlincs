# No-Anchor Clothing Referee Local Reassignment

Date: 2026-06-21

## Why this experiment

The previous self-play/refutation loop showed that two direct uses of the clothing referee were unsafe: whole-component merges were blocked by size/cannot-link constraints, and creating a new micro-component reduced full IDF1. This run tests a narrower action: keep existing global-ID components, but reassign only the visual-support sample tracklets into an existing component label.

No anchors are used. Ground truth is loaded only after an assignment is produced, for pair/full metrics.

## Setup

- Base assignment: `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- Candidate source: clothing verifier `surgery_inputs.json` from the preview edge audit.
- Candidate variants after moved-signature dedup: 11 unique assignments; 133 duplicates removed before full-score.
- Actions tried: `left_to_target`, `right_to_source`, `both_to_source`, `both_to_target`, using top-1/top-2/top-3/all support pairs, with cannot-link guard enabled.

## Results

| row | full IDF1 | HOTA | AssA | pair F1 | action | moved |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.653417 | 0.517306 | 0.533005 | 0.769801 | edge 3 `right_to_source` top1 | edge 3 right_to_source top1 conf=0.956621; moves 1929:26->10 |
| 2 | 0.653381 | 0.517255 | 0.532944 | 0.769695 | edge 3 `right_to_source` top2 | edge 3 right_to_source top2 conf=0.956621; moves 1929:26->10, 390:26->10 |
| 3 | 0.653210 | 0.517030 | 0.532678 | 0.769367 | edge 3 `left_to_target` top1 | none |
| 4 | 0.653229 | 0.517040 | 0.532678 | 0.769366 | edge 2 `left_to_target` top1 | edge 2 left_to_target top1 conf=0.863755; moves 3871:35->61 |
| 5 | 0.653229 | 0.517040 | 0.532678 | 0.769366 | edge 2 `both_to_target` top1 | edge 2 both_to_target top1 conf=0.863755; moves 3871:35->61, 3891:60->61 |
| 6 | 0.653327 | 0.517093 | 0.532676 | 0.769360 | edge 2 `left_to_target` top2 | edge 2 left_to_target top2 conf=0.863755; moves 3871:35->61, 3733:35->61 |
| 7 | 0.653327 | 0.517093 | 0.532676 | 0.769360 | edge 2 `both_to_target` top2 | edge 2 both_to_target top2 conf=0.863755; moves 3871:35->61, 3733:35->61, 3891:60->61 |
| 8 | 0.653433 | 0.517149 | 0.532674 | 0.769359 | edge 2 `left_to_target` top3 | edge 2 left_to_target top3 conf=0.863755; moves 3871:35->61, 3733:35->61, 3690:35->61 |

## Best full row

Best full IDF1 row: rank 8, IDF1 0.653433, HOTA 0.517149, AssA 0.532674.
Action: edge 2 left_to_target top3 conf=0.863755; moves 3871:35->61, 3733:35->61, 3690:35->61.
Delta vs raw k3: IDF1 +0.000223, HOTA +0.000119, AssA -0.000004.
Delta vs standing best: IDF1 -0.001945, HOTA -0.001649, AssA -0.001872.

## Interpretation

- The clothing referee contains a real pinpoint signal: moving one or a few support tracklets can improve full IDF1 over raw k3.
- The signal is not component-level. Larger support-pair moves often reduce pair F1 or only nudge IDF1 through one video, and cannot-link blocks several directions.
- Pair F1 is an imperfect proxy for end-to-end IDF1 here: the pair-best row is not the full-best row. The next proposer needs a full-score-aware reranker or a stronger cheap proxy.

## Next direction

Move from component-edge decisions to pinpoint fragment reassignment search: retrieve individual candidate tracklets/fragments with clothing/body evidence, enforce cannot-link and directionality, then rerank only unique moved-signatures with a full-score budget. The useful primitive is not merge, split, or new ID; it is a calibrated single-fragment relink operation.

## Artifacts

- Result JSON: `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_k3_clothing_referee_local_reassignment_20260621/result.json`
- Result CSV: `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_k3_clothing_referee_local_reassignment_20260621/result.csv`
- Pair-best assignments: `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_k3_clothing_referee_local_reassignment_20260621/top_assignments.csv`
- Summary: `/Users/zcai/Codex/vlincs_reid_by_search/local_runs/no_anchor_k3_clothing_referee_local_reassignment_20260621_summary.json`
