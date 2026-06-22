# No-Anchor Current-Best Rank01 Softcut True-Split Refutation

Date: 2026-06-21

## Context

Standing no-anchor delivery best:

- model-side pair F1/P/R: `0.775234 / 0.820504 / 0.734698`
- e2e delivery IDF1/HOTA/AssA: `0.655817 / 0.519228 / 0.534791`

The active error decomposition suggested large false-merge components mixed with
small impostor fragments.  This experiment tested whether one structural split
of a high-conflict component could improve delivery without anchors.

No anchors were used.  GT was used only for evaluation and post-hoc scoring.

## Candidate

Input assignment:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/assignments/rank01_conflict_subcluster_reassign_candidate_search_augmented_candidates_assignments.csv`

Remote run:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_rank01_softcut_true_split_20260621`

Local artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_softcut_true_split_20260621/softcut.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_softcut_true_split_20260621/softcut.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_softcut_true_split_20260621/top_full_export.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_rank01_softcut_true_split_20260621/density_simple_sourcezip.json`

Softcut accepted exactly one split:

| field | value |
|---|---:|
| split components | `1` |
| split tracklets | `162` |
| split parts | `2` |
| component label | `6` |
| conflict edges before / after | `44 / 34` |
| conflict reduction | `0.227273` |
| within / cross similarity | `0.637032 / 0.519812` |
| visual margin | `0.117221` |
| part sizes | `48 / 114` |

## Result

| output | IDF1 | HOTA | AssA | note |
|---|---:|---:|---:|---|
| base current-best delivery | `0.655817` | `0.519228` | `0.534791` | standing best |
| true-split raw full-score | `0.649324` | `0.513477` | `0.529945` | exported from assignment |
| true-split density-simple | `0.651408` | `0.515161` | `0.531766` | source-zip policy |

Pair metrics also dropped:

| assignment | pair F1 | precision | recall |
|---|---:|---:|---:|
| base | `0.770741` | `0.817123` | `0.729341` |
| true split | `0.766079` | `0.815685` | `0.722160` |

Per-video raw full-score after split:

| video | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| MCAM00 Tc6 | `0.878679` | `0.814753` | `0.835909` |
| MCAM00 Tc8 | `0.827729` | `0.748625` | `0.784533` |
| MCAM03 Tc6 | `0.659610` | `0.553346` | `0.603765` |
| MCAM03 Tc8 | `0.620188` | `0.506093` | `0.551627` |
| MCAM04 Tc6 | `0.560301` | `0.446703` | `0.491070` |
| MCAM05 Tc6 | `0.710965` | `0.601047` | `0.639585` |
| MCAM05 Tc8 | `0.784856` | `0.696780` | `0.735920` |
| MCAM06 Tc6 | `0.606895` | `0.514826` | `0.596222` |
| MCAM06 Tc8 | `0.683117` | `0.575717` | `0.619398` |
| MCAM08 Tc6 | `0.766858` | `0.659467` | `0.678289` |

## Decision

Refuted.

The split reduced cannot-link conflicts, but it removed too much true identity
continuity.  The pair recall drop and raw/density full-score drop agree, so this
is not a delivery-policy artifact.  Soft visual/conflict cutting of large
components should remain an opponent diagnostic, not a primary production edit.

## Next Direction

The next candidate space should stop cutting whole source components.  The more
promising structural pivot is target-local continuity:

1. use the current assignment as a namespace scaffold;
2. identify small provisional fragments whose predecessor/successor tracklets
   strongly prefer an existing target component;
3. require the opponent to show no same-video temporal contradiction;
4. materialize only local relabels whose expected side-effect is smaller than
   the standing density-simple margin.

