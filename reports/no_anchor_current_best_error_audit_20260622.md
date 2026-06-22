# No-Anchor Current-Best Error Audit

Date: 2026-06-22

## Current Best

- Setting: no anchors; GT is used only for evaluation/audit.
- Best canonical delivery score remains IDF1/HOTA/AssA `0.658025 / 0.521057 / 0.536049`.
- Current best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`
- Canonical scoring path:
  assignment CSV -> `kit/evaluate_db_assignments_full.py` -> `kit/no_anchor_pervideo_filter_selector.py --policies density_simple` -> `kit/evaluate_submission_detection_filter.py --config p005_area`

## Audit Command

```bash
/mnt/localssd/vlincs_reid_venv/bin/python kit/analyze_no_anchor_assignment_errors.py \
  --dbname gallery_ds1 --role match \
  --assignment-csv /mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv \
  --eval-cache /mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1_match.npz \
  --full --top-n 80 \
  --json /mnt/localssd/vlincs_reid_runs/no_anchor_current_best_error_audit_20260622/current_best_error_audit_top80_full.json
```

Local artifacts:

- `local_runs/remote_h100_test_3_20260622/no_anchor_current_best_error_audit_20260622/current_best_error_audit_top80_full.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_current_best_error_audit_20260622/run_current_best_error_audit.log`

## Metrics

Pair-level metrics:

- eval tracklets: `7289`
- GT pair mass: `33,533,982,861`
- predicted pair mass: `29,734,743,818`
- true pair mass: `24,484,351,755`
- F1 / precision / recall: `0.773980 / 0.823426 / 0.730136`

Direct full export metrics:

- IDF1 / HOTA / AssA: `0.655836 / 0.519304 / 0.534154`
- DetPr / DetRe: `0.757955 / 0.577967`

Note: direct full export is not the canonical delivery number. The current canonical delivery score remains `density_simple + p005_area = 0.658025 / 0.521057 / 0.536049`.

## Per-Video Pair Metrics

Worst pair-F1 videos:

| Video | Pair F1 | Precision | Recall | Eval Tracklets |
| --- | ---: | ---: | ---: | ---: |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | 0.760171 | 0.798548 | 0.725314 | 2941 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | 0.765587 | 0.805533 | 0.729416 | 545 |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | 0.771364 | 0.837235 | 0.715102 | 174 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | 0.794854 | 0.844468 | 0.750746 | 803 |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc8` | 0.803268 | 0.903257 | 0.723211 | 291 |

MCAM04/Tc6 is still the largest and worst high-mass slice, so it is the main pressure point for the next no-anchor repair pass.

## Dominant False Splits

Top false-split identities:

| GT ID | False-Split Mass | Pred Components | Dominant Pred | Dominant Fraction | Main Videos |
| ---: | ---: | ---: | ---: | ---: | --- |
| 9 | 1,047,637,352 | 27 | 96000000 | 0.701911 | MCAM04/Tc6, MCAM08/Tc6, MCAM03/Tc6 |
| 36 | 678,282,757 | 17 | 96000035 | 0.657854 | MCAM04/Tc6, MCAM03/Tc6, MCAM06/Tc8 |
| 11 | 672,659,005 | 25 | 96000037 | 0.786685 | MCAM08/Tc6, MCAM03, MCAM04/Tc6 |
| 52 | 598,994,097 | 24 | 96002329 | 0.720353 | MCAM04/Tc6, MCAM03/Tc6, MCAM08/Tc6 |
| 43 | 571,489,162 | 20 | 96000015 | 0.766295 | MCAM04/Tc6, MCAM03/Tc6, MCAM08/Tc6 |

Interpretation: the remaining gap is not just local edge noise. Several high-mass identities are split across many predicted components, often spanning MCAM04, MCAM03, and MCAM08.

## Dominant False Merges

Top false-merge predicted components:

| Predicted ID | False-Merge Mass | GT Count | Dominant GT | Dominant Fraction | Main Issue |
| ---: | ---: | ---: | ---: | ---: | --- |
| 96000035 | 666,422,541 | 6 | 36 | 0.642142 | Mixed GT36 with GT20, mostly MCAM04-heavy |
| 96000048 | 368,583,737 | 30 | 36 | 0.445855 | Very impure multi-GT component |
| 96000021 | 316,423,972 | 19 | 31 | 0.818143 | Mostly GT31 but includes GT9 and others |
| 96000026 | 306,768,505 | 22 | 37 | 0.810261 | Mostly GT37 with contaminating fragments |

Interpretation: aggressive merge proposals are unsafe unless paired with a split/admission guard. Teacher-consensus rank58 already showed zero safe accepted merge edges.

## Next Direction

The next no-anchor branch should target concentrated false-split and impure-component repairs rather than broad teacher-consensus merging:

1. Build component-level edge diagnostics around GT9/GT36/GT11-like cases using weakmetric, DINO, SigLIP, and DB evidence views.
2. Use teacher outputs as hard-negative/referee evidence, not as direct promotion sources.
3. Prefer local split/admission gates on impure components such as `96000035`, `96000048`, and `96000021`.
4. Keep canonical promotion gated by `density_simple + p005_area`, because direct full export and canonical delivery differ slightly.

## Runtime Note

The follow-up edge-diagnostic script was prepared at:

- `local_runs/remote_h100_test_3_20260622/no_anchor_current_best_edge_diagnostics_20260622/run_edge_diagnostics.sh`

The remote H100 nodes entered stopping state before this diagnostic could complete, so compute restoration is the next operational step before continuing the experiment loop.
