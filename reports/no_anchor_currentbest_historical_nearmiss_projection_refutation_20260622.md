# No-Anchor Historical Near-Miss Projection Refutation

Date: 2026-06-22

## Result

No new best was found.

Current delivery best remains:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

Current direct full baseline remains:

`IDF1 / HOTA / AssA = 0.661843 / 0.523137 / 0.534683`

## Experiment

The old high-mass `21->60` peel branch contained a near miss before the current
component-split promotion.  To test compatibility with the new split-best base,
the old candidate deltas were replayed onto:

`local_runs/offline_no_anchor_split_probe_20260622/best_combo_split_assignments.csv`

The tested projections were:

- `rank01_s21_to60_size22`, 31 changed tracklets
- `rank06_s21_to60_size16`, 25 changed tracklets
- `rank48_s21_to60_size10`, 20 changed tracklets

## Full-Score Results

| Candidate | IDF1 | HOTA | AssA | Verdict |
| --- | ---: | ---: | ---: | --- |
| `rank01_s21_to60_size22` | 0.661628 | 0.522908 | 0.534521 | negative |
| `rank06_s21_to60_size16` | 0.661687 | 0.522985 | 0.534582 | negative |
| `rank48_s21_to60_size10` | 0.661316 | 0.522591 | 0.534283 | negative |

All three are below the current direct full baseline.

## Interpretation

The old near-miss branch does not compose with the current component-split
best.  The likely failure mode is interaction with the newly split components:
the old peel moves improve or nearly improve the earlier assignment, but they
undo useful precision structure after the current split promotion.

The next branch should regenerate candidates from the current split-best state
instead of replaying old near-miss edits.  The admission model should treat
old-base near misses as conditional labels, not universally reusable positives.

## Data-Use Boundary

- No anchors were used.
- Candidate deltas came from prior no-anchor candidates.
- GT/eval cache was used only for scoring and diagnostics.
