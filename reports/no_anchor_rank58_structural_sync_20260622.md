# No-Anchor Rank58 Structural Sync, 2026-06-22

## Summary

This sync records the h100-test-3 rank58-base structural run while the final
`conflict_subcluster_weakmetric_dino` stage is still running remotely.

Current best remains:

- IDF1 / HOTA / AssA: `0.658025 / 0.521057 / 0.536049`
- Delivery path: assignment CSV -> full export -> `density_simple` -> `p005_area`
- Best assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`

## Completed Structural Checks

| Candidate | Pair F1 | Pair precision | Pair recall | p005 IDF1 | p005 HOTA | p005 AssA | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base_rank58 | 0.773980 | 0.823426 | 0.730136 | 0.658025 | 0.521057 | 0.536049 | current best |
| cannotlink_split_best | 0.774129 | 0.823847 | 0.730069 | 0.658025 | 0.521057 | 0.536049 | tie, no promotion |
| cannotlink_nms_singleton_best | 0.773980 | 0.823426 | 0.730136 | 0.658025 | 0.521057 | 0.536049 | no-op tie |
| state_policy_best | 0.775522 | 0.827771 | 0.729478 | 0.658025 | 0.521057 | 0.536049 | pair-proxy gain, delivery tie |

The state-policy branch is useful evidence: its pair proxy improved, but the
delivery score did not move after `density_simple + p005_area`. That supports
the current hypothesis that more pair mass precision alone is not enough unless
the delivery/export filter changes a visible set of scored detections.

## Artifacts

Remote:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_rank58_structural_20260622_v2/`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_rank58_structural_20260622_v2_key_artifacts.tgz`

Local synced subset:

- `local_runs/remote_h100_test_3_20260622/no_anchor_rank58_structural_20260622_v2/`

S3:

- `s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_rank58_structural_20260622_v2/`
- `s3://dit-scale-up/zcai/vlincs/research_snapshot_current/local_runs/remote_h100_test_3_20260622/no_anchor_rank58_structural_20260622_v2/`
- `s3://dit-scale-up/zcai/vlincs/core_snapshot_20260622/`
- `s3://dit-scale-up/zcai/vlincs/LATEST_NO_ANCHOR_PROGRESS.txt`

## Still Running

`conflict_subcluster_weakmetric_dino` started after state-policy scoring. It is
not included as a completed result in this report yet, and it should not be
treated as a promotion until its `density_simple + p005_area` score lands.
