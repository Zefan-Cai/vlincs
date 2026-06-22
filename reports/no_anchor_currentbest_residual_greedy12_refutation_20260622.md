# No-Anchor Current-Best Residual Greedy12 Refutation

Date: 2026-06-22

## Result

The residual split probe did not beat the current best.

Current promoted delivery best remains:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

The residual greedy12 direct full score was:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

This ties the promoted 3-component split direct full score at 6 decimals, so it
is not a production promotion.

## Candidate

Base assignment:

`local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_assignments.csv`

Residual greedy assignment:

`local_runs/offline_no_anchor_split_probe_20260622/currentbest_residual_greedy12_split_assignments.csv`

Full-score artifact:

`local_runs/offline_no_anchor_split_probe_20260622/currentbest_residual_greedy12_full_export.json`

Submission zip:

`local_runs/offline_no_anchor_split_probe_20260622/currentbest_residual_greedy12_full_export.zip`

## Pair Proxy

The residual greedy12 probe improved the eval-only pair proxy:

Base:

`F1 / precision / recall = 0.781330 / 0.849326 / 0.723413`

Greedy12:

`F1 / precision / recall = 0.785529 / 0.862643 / 0.721070`

Selected residual splits:

- `96000032` SigLIP `k=4`
- `96000009` weakmetric `k=6`
- `96000040` SigLIP `k=6`
- `96000019` SigLIP `k=6`
- `960000480` SigLIP `k=6`
- `960000482` SigLIP `k=5`
- `960000481` weakmetric `k=6`
- `96000041` weakmetric `k=5`
- `96000031` SigLIP `k=3`
- `96000024` weakmetric `k=6`
- `960000200` weakmetric `k=6`
- `960000203` DINO `k=6`

## Interpretation

This is a useful hard negative for the no-GT split admission model. The extra
splits increase pair precision and pair F1, but do not move end-to-end IDF1.
The admission gate should therefore include full-pipeline-sensitive features,
not just component purity or pair precision proxies.

The current production candidate remains the simpler 3-component split:

- `96000035` SigLIP `k=2`
- `96000048` weakmetric `k=4`
- `96000020` weakmetric `k=6`

## Data-Use Boundary

- No anchors were used.
- Candidate generation used assignment CSVs and no-GT feature views.
- GT/eval cache was used only for diagnostics, ablation scoring, and final
  metric validation.
