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

- Rank46 residual MCAM08 island visual case: Two MCAM08 tracklets move from component 27 / gid 96000029 to component 29 / gid 96000031. The bbox evidence shows a remaining residual island after rank15 and rank40.
  - image: `cases/rank46_subpart_visual/rank46_bbox_evidence.png`
  - html: `cases/rank46_subpart_visual/case.html`
  - json: `cases/rank46_subpart_visual/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| multiview rank46 27->29 | move 2 MCAM08 tracklets | direct 0.662835/0.523970/0.535170; density 0.664929/0.525663/0.536945; p005 0.665029/0.525739/0.537061 | promote |
| multiview rank47 22->38 | move 2 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| multiview rank21 82->80 | move 6 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| multiview rank44 30->85 | move 2 tracklets | direct tied 0.662694/0.523807/0.534999 | hard neutral |
| ranker old+rank15+rank40 labels | 469 candidates, 9 matched labels | LOOCV RMSE 0.000011; highest useful candidate was not rank1 | keep full-score labels and continue |

## Upload

- Bitbucket: `target branch wisc; commit after artifact generation`
- S3: `blocked locally: aws sts get-caller-identity failed with Unable to locate credentials`

## Next

Continue peeling component-27 and MCAM08 residual islands, then search compatible multi-edge combos once single-edge margins saturate.
