# No-Anchor Current-Best Residual Split/Merge Refutation

Date: 2026-06-22

## Result

No new best was found.

Current delivery best remains:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

## Residual Single Splits

Top residual single split assignments were exported from the current
`best_combo_split_assignments.csv` base.  Six candidates were full-scored:

- `rank001` component `96000032`, SigLIP `k=4`
- `rank002` component `96000032`, weakmetric `k=6`
- `rank003` component `96000032`, SigLIP `k=5`
- `rank004` component `96000032`, SigLIP `k=3`
- `rank008` component `96000009`, weakmetric `k=6`
- `rank009` component `96000009`, weakmetric `k=5`

All six tied direct full:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

The most promising lower-FP single, `rank009`, was also tested through the
delivery wrappers:

- `density_simple`: `0.663955 / 0.524825 / 0.536460`
- `density_simple + p005_area`: `0.664050 / 0.524893 / 0.536568`

This ties the current best and does not promote.

## Small Fragment Re-Merge

Component-edge diagnostics were rerun on the current best assignment, not the
older rank58 assignment.  With `candidate_min_component_size=1`, the no-GT
top-50 merge pool had only `6/50` eval-true edges.  The first eval-true edge
appeared at no-GT rank 4:

`96000043 -> 96000069`, moving 1 tracklet.

The no-GT ranks 1-4 were full-scored:

- `rank001` `960000205 -> 960000203`, 9 tracklets
- `rank002` `960000201 -> 960000482`, 13 tracklets
- `rank003` `960000201 -> 96002330`, 13 tracklets
- `rank004` `96000043 -> 96000069`, 1 tracklet, eval-true

All four tied direct full:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

## Interpretation

Two simple continuations are now refuted:

1. Keep adding residual split candidates because the pair proxy improves.
2. Recover recall by merging top visual-neighbor small fragments.

Both can change predicted ID counts or false-positive bookkeeping, but the
end-to-end HOTA/IDF1 scorer is insensitive at the current magnitude.

The next useful direction is a full-score-sensitive admission model:

- positives: the three-component promoted split;
- hard negatives: residual single splits and residual greedy12 splits that
  improve pair precision but tie full score;
- hard negatives: no-GT top small merges that look visually plausible but tie
  full score;
- features: component size, moved detection mass, view agreement, silhouette,
  per-video slice, predicted ID count delta, unmatched-FP delta, and whether
  the edit touches high-mass MCAM04/Tc6 failure components.

## Data-Use Boundary

- No anchors were used.
- Candidate generation used assignment CSVs and no-GT feature views.
- GT/eval cache was used only for diagnostics, ablation scoring, and final
  metric validation.
