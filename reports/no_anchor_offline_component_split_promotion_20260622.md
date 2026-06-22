# No-Anchor Offline Component Split Promotion

Date: 2026-06-22

## Result

New no-anchor current best:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

Previous current best:

`IDF1 / HOTA / AssA = 0.658025 / 0.521057 / 0.536049`

Delta:

`+0.006025 / +0.003836 / +0.000519`

The goal is still open because end-to-end IDF1 remains below `0.70`.

## Candidate

The promoted candidate is a no-anchor split/admission edit on top of rank58:

- split component `96000035` with SigLIP, `k=2`;
- split component `96000048` with weakmetric, `k=4`;
- split component `96000020` with weakmetric, `k=6`.

Assignment:

`local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_assignments.csv`

Canonical output artifacts:

- direct full: `local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_full_export.json`
- density simple: `local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_density_simple.json`
- density simple zip: `local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_density_simple.zip`
- p005 area: `local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_density_p005_area.json`
- p005 area zip: `local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_density_p005_area.zip`

## Why This Branch Worked

The current-best error audit showed that the remaining gap is a mixture of
false splits and impure false merges.  Broad component merges were unsafe:
three-view component edge diagnostics retrieved most false-split mass, but high
confidence merge edges had poor precision.  The strongest rule had only two
candidate edges and `0.5` eval-only precision.

The better move was therefore not another merge, but local splitting of visibly
multi-modal impure components.  Pair-level eval-only proxy improved from:

`F1 / precision / recall = 0.773980 / 0.823426 / 0.730136`

to:

`F1 / precision / recall = 0.781330 / 0.849326 / 0.723413`

This is a precision-heavy split: it raises precision substantially while
spending some recall.

## Full-Score Validation

The local scorer was first calibrated by reproducing the previous direct full
score exactly from parquet tracklets plus assignment CSV:

`IDF1 / HOTA / AssA = 0.655836 / 0.519304 / 0.534154`

The split candidate direct full score:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

After the existing no-GT `density_simple` wrapper:

`IDF1 / HOTA / AssA = 0.663955 / 0.524825 / 0.536460`

After the existing `p005_area` delivery filter:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

## Per-Video Effects

Direct full MCAM04/Tc6 improved from:

`IDF1 / HOTA / AssA = 0.564208 / 0.448206 / 0.489904`

to:

`IDF1 / HOTA / AssA = 0.577029 / 0.455721 / 0.490317`

This is the intended slice: MCAM04/Tc6 was the worst high-mass video in the
current-best audit.

## Ablation

An extended greedy split added smaller positive pair-proxy components:

- `96000032` weakmetric `k=6`
- `96000019` SigLIP `k=3`
- `96000009` weakmetric `k=6`
- `96000040` SigLIP `k=6`
- `96000041` weakmetric `k=6`
- `96000031` SigLIP `k=3`
- `96000024` weakmetric `k=6`

The pair proxy reached:

`F1 / precision / recall = 0.784701 / 0.860558 / 0.721135`

But direct full score stayed identical at 6 decimals to the 3-component split:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

So the promoted production candidate remains the simpler 3-component split.

## Data-Use Boundary

- No anchors were used.
- Candidate transforms are generated from assignment CSVs and no-GT features.
- GT/eval cache was used only for diagnostics, ablation scoring, and final
  metric validation.
- This is not yet a learned no-GT split admission policy; that is the next
  research step.

## Next Direction

Train or hand-code a no-GT split admission gate from the successful and failed
split ablations:

1. positive: `96000035` SigLIP `k=2`, `96000048` weakmetric `k=4`,
   `96000020` weakmetric `k=6`;
2. near-neutral: greedy small split extensions that improved pair proxy but not
   full score;
3. negative: pure false-split components such as `96000000`, `96000037`, and
   `96002329`, where splitting damages recall.

The next useful move is to generate more impure-component split candidates and
score them through the now-local parquet scorer, then learn a no-GT admission
rule that can generalize beyond these three components.
