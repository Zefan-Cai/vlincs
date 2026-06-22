# No-Anchor Rank58 Teacher-Consensus Refutation, 2026-06-22

## Summary

This run tested whether multiple no-anchor assignments could act as weak
teachers for safe component merges on top of the current rank58 best.

Current best remains:

- IDF1 / HOTA / AssA: `0.658025 / 0.521057 / 0.536049`
- Delivery path: assignment CSV -> full export -> `density_simple` -> `p005_area`
- Best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`

## Tested Teacher Sets

| Teacher set | Teacher edges | Accepted edges | Rejection pattern | p005 IDF1 | p005 HOTA | p005 AssA | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| base_rank58 | n/a | n/a | baseline | 0.658025 | 0.521057 | 0.536049 | current best |
| structural_policy_best | 3403 | 0 | 3402 threshold, 1 teacher | 0.658025 | 0.521057 | 0.536049 | tie, no promotion |
| subpart_promoted_best | 3403 | 0 | 3403 threshold | 0.658025 | 0.521057 | 0.536049 | tie, no promotion |
| timeagglom_diverse_best | 3321 | 0 | 3320 threshold, 1 conflict | 0.658025 | 0.521057 | 0.536049 | tie, no promotion |

## Evidence Notes

The subpart-promoted teacher family mostly acted as a hard-negative referee:
its top visual edges had `teacher_same = 0` and high `teacher_diff`, so the
consensus logic correctly refused broad merges.

The time-agglomeration teacher family found one strong same-vote edge
(`35 -> 61`, `teacher_same = 7`, `teacher_same_frac = 1.0`), but it was rejected
by the cannot-link conflict check. The remaining edges did not pass the
threshold.

The structural-policy family similarly produced no safe merge. Its highest
candidate (`35 -> 61`) had only partial teacher agreement and was rejected by
the teacher condition.

## Conclusion

Teacher-consensus merge is refuted for the current rank58 neighborhood. The
teacher assignments are still useful as hard-negative/referee evidence, but not
as positive merge proposers. The next branch should audit the current-best
assignment errors directly and target the oracle gap through concentrated false
split or admission repair rather than another broad consensus-merge sweep.

## Artifacts

Remote:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_teacher_consensus_rank58_20260622/`

Local:

- `local_runs/remote_h100_test_3_20260622/no_anchor_teacher_consensus_rank58_20260622/`
- `local_runs/remote_h100_test_3_20260622/no_anchor_teacher_consensus_rank58_20260622/key_artifacts.tgz`
- `local_runs/remote_h100_test_3_20260622/no_anchor_teacher_consensus_rank58_20260622/p005_teacher_consensus_summary.json`

S3:

- `s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_teacher_consensus_rank58_20260622/`
- `s3://dit-scale-up/zcai/vlincs/research_snapshot_current/local_runs/remote_h100_test_3_20260622/no_anchor_teacher_consensus_rank58_20260622/`
