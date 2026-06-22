# VLINCS no-anchor publish sync

Date: 2026-06-22 13:44 PDT

## Current best

- End-to-end IDF1 / HOTA / AssA: 0.658025 / 0.521057 / 0.536049
- Model-side pair F1 / precision / recall: 0.775234 / 0.820504 / 0.734698
- Best assignment source: `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622/peel21/size10_assignments/rank58_subpart_s21_to2330_10seq_assignments.csv`
- Delivery path: assignment CSV -> `kit/evaluate_db_assignments_full.py` -> `kit/no_anchor_pervideo_filter_selector.py --policies density_simple` -> `kit/evaluate_submission_detection_filter.py --config p005_area`

## Latest remote state

- Node: h100-test-3
- Remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_rank58_structural_20260622_v2`
- Completed structural variants:
  - `cannotlink_split_best`: delivery tied current best at 0.658025 / 0.521057 / 0.536049
  - `cannotlink_nms_singleton_best`: delivery tied current best at 0.658025 / 0.521057 / 0.536049
  - `state_policy_best`: pair proxy improved to 0.775522 / 0.827771 / 0.729478, but delivery still tied current best
- Still running at sync time: `conflict_subcluster_weakmetric_dino`

## Published artifact intent

The S3 sync should preserve the current research state, reports, small JSON/CSV metrics, and key remote logs. Large generated zips and huge scratch CSV sweeps are intentionally excluded from the curated publish path.

Primary S3 prefix:

`s3://dit-scale-up/zcai/vlincs/`

Recommended entry files:

- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/no_anchor_publish_sync_20260622.md`
- `reports/no_anchor_rank58_structural_sync_20260622.md`
- `reports/publish_manifest_20260622.md`
- `autoresearch_state/no_anchor_global_id/state/progress.json`

Latest remote artifact bundle:

- Local: `local_runs/remote_h100_test_3_20260622/no_anchor_rank58_structural_20260622_v2/key_artifacts_current.tgz`
- Remote source: `/mnt/localssd/vlincs_reid_runs/no_anchor_rank58_structural_20260622_v2_key_artifacts_current.tgz`

## GitHub code target

Core code mirror target:

`https://github.com/Zefan-Cai/vlincs`

The GitHub mirror should include the no-anchor global-id scripts under `kit/`, progress state, and curated reports. It should not include bulky `local_runs/` generated artifacts unless they are already intentionally tracked or small enough for source review.
