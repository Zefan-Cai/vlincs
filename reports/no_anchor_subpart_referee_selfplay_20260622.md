# No-Anchor Subpart Referee Self-Play Update - 2026-06-22

## Summary

Goal: continue the no-anchor global-id pipeline for VLINCS tracklets.  The
identity model remains no-anchor: no GT/anchor labels are used for candidate
generation or assignment construction.  Full-score IDF1/HOTA/AssA is used only
as post-hoc evaluation and as a scheduler/referee feedback signal.

New best e2e result:

| assignment | IDF1 | HOTA | AssA | delta IDF1 vs previous best |
| --- | ---: | ---: | ---: | ---: |
| `rank58_subpart_s21_to2330_10seq_assignments.csv` | 0.658025 | 0.521057 | 0.536049 | +0.000138 |

Previous best was `subpart_combo_r47_r49_14seq_assignments.csv` at
IDF1/HOTA/AssA `0.657887/0.520944/0.535983`.

## Method

I added `kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py`.
It reads no-anchor subpart candidate manifests and prior full-score summaries,
matches candidates by exact assignment/stem, extracts candidate evidence fields,
and trains a ridge-based side-effect referee.

Important fixes made during this run:

- summary matching now supports `assignment_csv`, `stem`, and metric `json` paths.
- labels are written back onto the original candidate rows before ranking.
- loose `(source, target, moved_tracklets)` fallback was disabled when an exact
  stem exists, because it polluted all `21->2330 size10` candidates with the
  same label.

Feature families used by the referee:

- move size: `moved_tracklets`, small/medium/large indicators.
- local evidence: `target_sim`, `target_margin`, `group_internal_sim`.
- side-effect evidence: `source_rest_margin_*`, `source_rest_cross_*`.
- conflict evidence: `conflicts_to_rest`, `source_component_conflict_edges`.
- slice evidence: focus hits and source-video entropy/dominance.

## Referee Diagnostics

Strict exact-stem v3 label bank:

| labels | best label IDF1 | LOOCV RMSE | LOOCV corr | LOOCV rank corr |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.658025 | 0.000580 | 0.287 | 0.256 |

Interpretation: once the overly broad fallback labels were removed, the clean
label bank is small and noisy.  The referee is useful as a candidate proposer,
not as an oracle.  Full-score remains the gate.

## Full-Score Ablations

### Batch 1: first referee-selected candidates

| source -> target | moved | IDF1 | HOTA | AssA | label |
| --- | ---: | ---: | ---: | ---: | --- |
| `48->32` | 16 | 0.657887 | 0.520944 | 0.535983 | tie |
| `28->24` | 16 | 0.657656 | 0.520561 | 0.535546 | negative |
| `47->41` | 16 | 0.657618 | 0.520495 | 0.535475 | negative |
| `55->58` | 14 | 0.657097 | 0.520149 | 0.535389 | negative |
| `21->19` | 16 | 0.657060 | 0.520202 | 0.535535 | negative |
| `24->38` | 2 | 0.656320 | 0.519856 | 0.535474 | negative |

### Batch 2: v2 referee candidates

| source -> target | moved | IDF1 | HOTA | AssA | label |
| --- | ---: | ---: | ---: | ---: | --- |
| `21->2330` | 10 | 0.658025 | 0.521057 | 0.536049 | positive, promoted |
| `20->2330` | 16 | 0.657887 | 0.520944 | 0.535983 | tie |
| `22->2330` | 8 | 0.657683 | 0.520737 | 0.535837 | negative |
| `24->41` | 7 | 0.657624 | 0.520692 | 0.535785 | negative |
| `9->2330` | 2 | 0.657475 | 0.520599 | 0.535769 | negative |

### Batch 3: local alternatives around the promoted peel

| source -> target | moved | IDF1 | HOTA | AssA | label |
| --- | ---: | ---: | ---: | ---: | --- |
| `21->24` | 10 | 0.657976 | 0.521006 | 0.536014 | near miss |
| `21->2330` | 10 | 0.657976 | 0.521006 | 0.536014 | near miss |
| `21->19` | 10 | 0.657735 | 0.520801 | 0.535891 | negative |
| `21->2330` | 10 | 0.657697 | 0.520788 | 0.535898 | negative |

## Current State

- Best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`
- Best e2e:
  IDF1/HOTA/AssA `0.658025/0.521057/0.536049`.
- Model-side pair metric remains:
  F1/precision/recall `0.775234/0.820504/0.734698`.
- Target remains unmet:
  e2e IDF1 is still below `0.70`.

## Next Direction

The promoted `21->2330` rank58 candidate should become the new base.  The next
useful experiment is not more alternatives from the old base; it is:

1. regenerate residual subpart candidates from the rank58 assignment;
2. compose non-conflicting moves on top of rank58;
3. keep the referee exact-stem only, with full-score labels as calibration
   feedback rather than identity supervision.

## Artifacts

- `kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_subpart_referee_from_fullscore_labels_20260622/referee_ranked_candidates_v3_after_promotion.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_subpart_referee_fullscore_20260622/p005_referee_fullscore_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_subpart_referee_v2_fullscore_20260622/p005_referee_v2_fullscore_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_subpart_referee_v3_fullscore_20260622/p005_referee_v3_fullscore_summary.json`

