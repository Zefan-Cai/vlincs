# No-Anchor Rank58 Residual Subpart Refutation

Date: 2026-06-22

## Setup

This iteration used the current best no-anchor assignment as the new base:

`/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`

Current best remains:

| metric | value |
| --- | ---: |
| IDF1 | 0.658025 |
| HOTA | 0.521057 |
| AssA | 0.536049 |

No anchors or GT identity labels were used to generate candidates. Ground truth was used only by the offline evaluator.

## Candidate Generation

Generated new residual subpart repair pools on top of rank58:

| pool | selected | candidate count | feature view |
| --- | ---: | ---: | --- |
| `weakmetric_dino` | 61 | 61 | weakmetric primary + DINO 0.25 |
| `siglip_dino` | 72 | 72 | SigLIP-fused primary + DINO 0.25 |
| `dino_weak` | 120 | 3356 | DINO primary + weakmetric 0.35 |

The first referee pass had only one exact-stem label on the new base, so it was mostly heuristic. I selected a diverse queue of 12 candidates, then added 4 follow-up candidates after retraining the referee on the first 12 labels.

## Full-Score Results

All results use the same p005-area detection filter as the current best.

| order | candidate | IDF1 | HOTA | AssA | note |
| ---: | --- | ---: | ---: | ---: | --- |
| 6 | `weakmetric_dino_r55_29_to2330` | 0.657971 | 0.521239 | 0.536131 | best residual, still below rank58 |
| 2 | `weakmetric_dino_r59_24_to31` | 0.657938 | 0.521206 | 0.536091 | below best |
| 3 | `siglip_dino_r72_31_to24` | 0.657938 | 0.521206 | 0.536091 | below best |
| 4 | `weakmetric_dino_r50_9_to32` | 0.657938 | 0.521206 | 0.536091 | below best |
| 5 | `weakmetric_dino_r56_9_to2330` | 0.657938 | 0.521206 | 0.536091 | below best |
| 11 | `dino_weak_r27_48_to24` | 0.657938 | 0.521206 | 0.536091 | below best |
| 15 | `weakmetric_dino_r60_31_to24` | 0.657938 | 0.521206 | 0.536091 | metric from stdout; zip export hit disk quota |
| 14 | `weakmetric_dino_r58_31_to25` | 0.657852 | 0.521120 | 0.536023 | metric from stdout; zip export hit disk quota |
| 13 | `siglip_dino_r70_31_to25` | 0.657826 | 0.521094 | 0.536003 | below best |
| 1 | `siglip_dino_r65_10_to9` | 0.657728 | 0.520919 | 0.535761 | old positive direction became negative when enlarged |
| 16 | `weakmetric_dino_r57_44_to29` | 0.657720 | 0.520998 | 0.535937 | below best |
| 8 | `siglip_dino_r61_11_to19` | 0.657554 | 0.520754 | 0.535672 | hard negative |
| 7 | `siglip_dino_r64_10_to15` | 0.657399 | 0.520542 | 0.535401 | hard negative |
| 12 | `dino_weak_r117_55_to58` | 0.656701 | 0.519953 | 0.535144 | confirms 55->58 remains harmful |
| 10 | `dino_weak_r45_21_to19` | 0.656695 | 0.520023 | 0.535333 | large residual 21 move harmful |
| 9 | `siglip_dino_r57_11_to40` | 0.655516 | 0.518953 | 0.534473 | strong negative |

## Referee Diagnostic

After adding the rank58-base labels, the residual-subpart referee fit 13 exact labels:

- labeled count: 13
- label max: 0.657971
- LOOCV RMSE: 0.000883
- LOOCV rank correlation: 0.132

The low rank correlation means this referee is useful as a negative-memory filter, not as a positive scheduler. Its highest remaining suggestions were small `31->25/31->24` variants plus many `55->58` variants. The follow-up test showed the small variants still fail to beat rank58, and `55->58` is already repeatedly negative.

## Conclusion

The rank58 residual subpart branch is refuted for now. It generated many plausible local repairs, but all 16 full-scored candidates are below the current best. The plateau around `0.657938` suggests these moves are mostly rearranging the same residual identity errors after rank58 rather than creating new correct global-ID resolution.

Next useful direction: pivot away from residual subpart reassignments and target structure-level changes:

- component graph false-split/false-merge decomposition on the rank58 base;
- detection/admission policy that changes committed evidence rather than only component labels;
- candidate generation from cannot-link/component conflict islands, with the 16 residual subpart labels as hard negatives.

## Artifacts

- New candidate manifests: `local_runs/remote_h100_test_3_20260622/no_anchor_rank58_residual_subpart_20260622/`
- Full-score summary: `local_runs/remote_h100_test_3_20260622/no_anchor_rank58_residual_fullscore_20260622/p005_rank58_residual_summary_with_followup.json`
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_rank58_residual_fullscore_20260622`
- Parallel runner artifact: `local_runs/remote_h100_test_3_20260622/no_anchor_rank58_residual_subpart_20260622/run_parallel_rank58_residual.py`
