# No-anchor rank40-base rank46 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M8 graph resolution + M12 delivery calibration`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment and canonical delivery zip`
- Status: `gain`
- No-anchor: `True`

## Summary

Multiview rank46 moves 2 MCAM08 tracklets from component 27 / gid 96000029 to component 29 / gid 96000031. It improves direct full to 0.662835 and valid p005_area to the new canonical best 0.665029/0.525739/0.537061.

## Metrics

- Baseline: `0.664898`
- Candidate: `0.665029`
- Delta: `0.000131`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Regenerated no-anchor subpart candidates on the rank40 base and ranked 469 old+new candidates with 9 matched full-score labels. The promoted assignment is local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_multiview_assignments/rank46_subpart_s27_to29_2seq_assignments.csv. The same visual-case exporter produces bbox evidence for the residual MCAM08 island.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `No anchors; GT used only for evaluation/selection`

## Commands

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank15base_subpart_multiview_assignments/rank40_subpart_s5_to35_2seq_assignments.csv ...
```

```bash
python kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py --manifest promoted_combo... --manifest rank15base... --manifest rank40base... --label-summary local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_fullscore_label_summary.json --baseline-idf1 0.662694 ...
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_multiview_assignments/rank46_subpart_s27_to29_2seq_assignments.csv --fallback singleton --json local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_full_export.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_full_export.zip
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_full_export.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_simple.zip
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_simple.zip --config @local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt --json local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_p005_area.zip
```

## Code Paths

- `kit/export_no_anchor_subpart_visual_case.py`
- `kit/evaluate_submission_detection_filter.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_rank40base_rank46_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank40base_rank46_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_balanced_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_multiview_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_weakvideo_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_reviewer_ranked_with_oldlabels.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_multiview_assignments/rank46_subpart_s27_to29_2seq_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank40base_rank46_delivery/rank46_subpart_s27_to29_2seq_assignments_density_p005_area.zip`
- `reports/vlincs_iterations/20260623_rank40base_rank46_p005_gain/metrics/direct_full.json`
- `reports/vlincs_iterations/20260623_rank40base_rank46_p005_gain/metrics/density_simple.json`
- `reports/vlincs_iterations/20260623_rank40base_rank46_p005_gain/metrics/p005_area.json`

## Visual Cases

- Rank15 fixes a no-anchor false-split island: Three edge/border tracklets move from component 27 / gid 96000029 to component 10 / gid 96000011.
  - failure: The previous graph kept three low/edge-quality MCAM04/MCAM08 fragments inside a large source component, so the identity was split away from the visually closer target component.
  - improvement: The new subpart repair tests a small conflict-supported group instead of moving the whole component, then commits only the three compatible tracklets to the target component.
  - html: `cases/rank15_subpart_visual/case.html`
  - json: `cases/rank15_subpart_visual/case.json`
  - image: `cases/rank15_subpart_visual/rank15_bbox_evidence.png`
  - image: `cases/rank15_subpart_visual/tracking_precision_failure_pred_m0044_montage.png`
- Rank40 fixes a tiny MCAM08 false-split island: Two MCAM08 tracklets move from component 5 / gid 96000005 to component 35 / gid 96000038 on top of the rank15 base.
  - failure: The previous graph left a high-internal-similarity pair inside source component 5 even though delivery p005 preferred the target component after density filtering.
  - improvement: The multiview subpart repair commits only the two compatible MCAM08 tracklets, giving a tiny direct gain and a valid p005_area canonical gain.
  - html: `cases/rank40_subpart_visual/case.html`
  - json: `cases/rank40_subpart_visual/case.json`
  - image: `cases/rank40_subpart_visual/rank40_bbox_evidence.png`
  - image: `cases/rank40_subpart_visual/tracking_recall_failure_gt_m0038_montage.png`
- Rank46 peels another MCAM08 residual island: Two MCAM08 tracklets move from component 27 / gid 96000029 to component 29 / gid 96000031 on top of the rank40 base.
  - failure: After rank15 and rank40, component 27 still contained a small high-similarity MCAM08 residual island that was not committed to its closer target identity.
  - improvement: The multiview reviewer identifies the two-tracklet island and commits only that local repair; density_simple and p005_area both improve.
  - html: `cases/rank46_subpart_visual/case.html`
  - json: `cases/rank46_subpart_visual/case.json`
  - image: `cases/rank46_subpart_visual/rank46_bbox_evidence.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| multiview rank46 27->29 | move 2 MCAM08 tracklets | direct 0.662835/0.523970/0.535170; density 0.664929/0.525663/0.536945; p005 0.665029/0.525739/0.537061 | promote |
| multiview rank47 22->38 | move 2 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| multiview rank21 82->80 | move 6 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| multiview rank44 30->85 | move 2 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| ranker old+rank15+rank40 labels | 469 candidates, 9 matched labels | LOOCV RMSE 0.000011; highest useful candidate was not rank1 | keep full-score labels and continue |

## Upload

- Bitbucket: `pushed branch wisc; rank46 original commit 8e80475, enhanced report update pending`
- S3: `blocked locally: aws sts get-caller-identity failed with Unable to locate credentials`

## Next

Use the rank46base tie/positive labels to penalize stale subpart families; promote rank47 4->38 only if its canonical p005 gain survives packaging.
