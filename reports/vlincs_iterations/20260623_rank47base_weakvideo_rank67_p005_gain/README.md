# No-anchor rank47-base weakvideo rank67 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7 candidate retrieval + M10 scheduler + M8 graph resolution + M12 delivery calibration`
- Used in pipeline: `yes: promoted as current no-anchor best assignment and canonical p005 delivery zip`
- Status: `gain`
- No-anchor: `True`

## Summary

Weakvideo rank67 moves 4 MCAM04/MCAM08 tracklets from component 43 / gid 96000046 to component 38 / gid 96000041. Direct full improves to 0.662969 and canonical density_simple+p005_area improves from 0.665084/0.525797/0.537115 to 0.665145/0.525833/0.537128.

## Metrics

- Baseline: `0.665084`
- Candidate: `0.665145`
- Delta: `0.000061`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Generated rank47base balanced, multiview, and weakvideo no-anchor subpart candidates from current assignment and feature NPZs. The reviewer ranked 784 old+new candidates with 35 prior full-score labels, then direct-scored five diverse rank47base candidates. Only weakvideo rank67 was direct-positive; it then passed density_simple and valid p005_area delivery.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `Feature cache: local_runs/s3_feature_cache_20260622`
- `No anchors; GT used only for post-hoc scoring/evaluation labels`

## Commands

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_multiview_assignments/rank47_subpart_s4_to38_2seq_assignments.csv --feature-npz local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz ... --json rank47base_subpart_{balanced,multiview,weakvideo}_manifest.json
```

```bash
python kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py --manifest promoted/rank15/rank40/rank46/rank47base manifests --label-summary promoted/rank15/rank40/rank46 labels --baseline-idf1 0.662902 --output-json local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_ranked_with_oldlabels.json
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --assignments local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_weakvideo_assignments/rank67_subpart_s43_to38_4seq_assignments.csv --fallback singleton --json local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_fullscore/weakvideo_rank67_s43_to38_4seq_full_export.json --zip-out ...zip
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_fullscore/weakvideo_rank67_s43_to38_4seq_full_export.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_simple_withenv.json --zip-out ...zip
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_simple_withenv.zip --config @local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt --json local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_p005_area.json --zip-out ...zip
```

```bash
python kit/export_no_anchor_subpart_visual_case.py --before-assignments rank47_current_best.csv --after-assignments rank67_subpart_s43_to38_4seq_assignments.csv --manifest rank47base_subpart_weakvideo_manifest.json --rank 67 ...
```

## Code Paths

- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `reports/vlincs_iterations/20260623_rank47base_weakvideo_rank67_p005_gain/`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_balanced_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_multiview_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_weakvideo_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_ranked_with_oldlabels.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_weakvideo_assignments/rank67_subpart_s43_to38_4seq_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_fullscore/weakvideo_rank67_s43_to38_4seq_full_export.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_simple_withenv.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/weakvideo_rank67_s43_to38_4seq_density_p005_area.zip`

## Visual Cases

- Rank67 repairs a weakvideo MCAM04/MCAM08 residual island: Four weakvideo tracklets move from component 43 / gid 96000046 to component 38 / gid 96000041 on top of the rank47 base.
  - failure: After rank47, component 43 still held a cross-camera residual island whose weakvideo evidence was closer to component 38, leaving the delivery output split across two forced global IDs.
  - improvement: The weakvideo proposer isolates only the four compatible tracklets, keeps the larger source component intact, and the canonical density_simple+p005_area delivery score improves.
  - html: `cases/rank67_subpart_visual/case.html`
  - json: `cases/rank67_subpart_visual/case.json`
  - image: `cases/rank67_subpart_visual/rank67_bbox_evidence.png`
- Rank47 context: MCAM04 residual move: Two MCAM04 tracklets moved from component 4 / gid 96000004 to component 38 / gid 96000041, becoming the previous canonical best.
  - failure: Before rank47, component 4 retained a small island that shared the target global ID evidence with component 38.
  - improvement: Rank47 committed the two-tracklet island and established the rank47 base for this follow-up.
  - html: `cases/rank47_subpart_visual/case.html`
  - json: `cases/rank47_subpart_visual/case.json`
  - image: `cases/rank47_subpart_visual/rank47_bbox_evidence.png`
- Rank46 context: MCAM08 residual island: Two MCAM08 tracklets moved from component 27 / gid 96000029 to component 29 / gid 96000031.
  - failure: The graph still held a small high-similarity MCAM08 residual island after rank40.
  - improvement: The reviewer committed only that local residual island and improved both density_simple and p005_area.
  - html: `cases/rank46_subpart_visual/case.html`
  - json: `cases/rank46_subpart_visual/case.json`
  - image: `cases/rank46_subpart_visual/rank46_bbox_evidence.png`
- Rank40 context: tiny MCAM08 false-split island: Two MCAM08 tracklets moved from component 5 / gid 96000005 to component 35 / gid 96000038.
  - failure: A high-internal-similarity pair remained in the source component despite delivery preferring the target component.
  - improvement: The multiview subpart repair committed only two compatible MCAM08 tracklets.
  - html: `cases/rank40_subpart_visual/case.html`
  - json: `cases/rank40_subpart_visual/case.json`
  - image: `cases/rank40_subpart_visual/rank40_bbox_evidence.png`
  - image: `cases/rank40_subpart_visual/tracking_recall_failure_gt_m0038_montage.png`
- Rank15 context: early false-split island: Three edge/border tracklets moved from component 27 / gid 96000029 to component 10 / gid 96000011.
  - failure: Low/edge-quality MCAM04/MCAM08 fragments were left in a larger source component.
  - improvement: The subpart repair tested a small conflict-supported group instead of moving the full component.
  - html: `cases/rank15_subpart_visual/case.html`
  - json: `cases/rank15_subpart_visual/case.json`
  - image: `cases/rank15_subpart_visual/rank15_bbox_evidence.png`
  - image: `cases/rank15_subpart_visual/tracking_precision_failure_pred_m0044_montage.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| weakvideo rank67 43->38 | move 4 MCAM04/MCAM08 tracklets | direct 0.662969/0.524082/0.535252; density 0.665046/0.525758/0.537012; p005 0.665145/0.525833/0.537128 | promote |
| multiview rank47 90->85 | move 2 tracklets | direct tied 0.662902/0.524042/0.535239 | hard neutral label |
| balanced rank17 82->80 | move 6 tracklets | direct tied 0.662902/0.524042/0.535239 | hard neutral label |
| balanced rank28 30->85 | move 2 tracklets | direct tied 0.662902/0.524042/0.535239 | hard neutral label |
| multiview rank40 90->81 | move 2 tracklets | direct tied 0.662902/0.524042/0.535239 | hard neutral label |
| rank47base reviewer | 784 candidates, 35 matched labels | LOOCV RMSE 0.000139; rank corr 0.773; model predicted all rank47base candidates as weak/negative, but weakvideo rank67 was a small direct-positive surprise | add labels and continue with weakvideo/residual-island family |

## Upload

- Bitbucket: `committed to Bitbucket wisc in this package; final commit hash reported by Codex after push`
- S3: `blocked: `aws sts get-caller-identity` failed locally with "Unable to locate credentials"; artifacts remain under reports/vlincs_iterations/20260623_rank47base_weakvideo_rank67_p005_gain and local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_reviewer_delivery/`

## Next

Fold rank67 label into the reviewer, search compatible weakvideo residual islands around component 43/38, and test small combos only after direct or density signals.
