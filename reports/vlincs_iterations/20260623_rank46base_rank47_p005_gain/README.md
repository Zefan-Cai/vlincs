# No-anchor rank46-base rank47 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M8 graph resolution + M10 scheduler/opponent + M12 delivery calibration`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment and canonical delivery zip`
- Status: `gain`
- No-anchor: `True`

## Summary

Multiview rank47 moves 2 MCAM04/Tc6-sensitive tracklets from component 4 / gid 96000004 to component 38 / gid 96000041. It improves valid p005_area to 0.665084/0.525797/0.537115, a new canonical no-anchor best.

## Metrics

- Baseline: `0.665029`
- Candidate: `0.665084`
- Delta: `0.000055`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Rank47 is a no-anchor residual subpart repair on top of rank46. The proposer generated candidates from assignment CSVs and feature NPZs only. The reviewer used post-hoc full-score labels only as scheduler calibration, not as anchors. The promoted edit moves two tracklets from source component 4 to target component 38, then passes direct full-score, density_simple, and p005_area delivery validation.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `No anchors; GT used only for evaluation/selection`

## Commands

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank40base_subpart_multiview_assignments/rank46_subpart_s27_to29_2seq_assignments.csv ...
```

```bash
python kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py --manifest promoted_combo... --manifest rank15base... --manifest rank40base... --manifest rank46base... --label-summary local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_fullscore_label_summary.json --baseline-idf1 0.662835 ...
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_multiview_assignments/rank47_subpart_s4_to38_2seq_assignments.csv --fallback singleton --json local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_fullscore/multiview_rank47_s4_to38_2seq_full_export.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_fullscore/multiview_rank47_s4_to38_2seq_full_export.zip
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_fullscore/multiview_rank47_s4_to38_2seq_full_export.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_simple.zip
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_simple.zip --config @local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt --json local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_p005_area.zip
```

## Code Paths

- `kit/export_no_anchor_subpart_visual_case.py`
- `kit/evaluate_submission_detection_filter.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_rank46base_rank47_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank46base_rank47_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_balanced_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_multiview_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_weakvideo_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_ranked_with_oldlabels.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_fullscore_label_summary.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_multiview_assignments/rank47_subpart_s4_to38_2seq_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank46base_subpart_reviewer_delivery/multiview_rank47_s4_to38_2seq_density_p005_area.zip`
- `reports/vlincs_iterations/20260623_rank46base_rank47_p005_gain/metrics/direct_full.json`
- `reports/vlincs_iterations/20260623_rank46base_rank47_p005_gain/metrics/density_simple.json`
- `reports/vlincs_iterations/20260623_rank46base_rank47_p005_gain/metrics/p005_area.json`

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
- Rank47 peels a source-4 residual island toward component 38: Two tracklets move from component 4 / gid 96000004 to component 38 / gid 96000041 on top of the rank46 base.
  - failure: After rank46, source component 4 still carried a small pair whose target similarity to component 38 was higher than the source-rest margin, but the earlier reviewer had treated related 22->38 and 30->85 moves as hard-neutral ties.
  - improvement: The updated opponent filters stale tied families, then tests the new 4->38 family; direct, density_simple, and p005_area all improve, so the local repair becomes the next canonical best.
  - html: `cases/rank47_subpart_visual/case.html`
  - json: `cases/rank47_subpart_visual/case.json`
  - image: `cases/rank47_subpart_visual/rank47_bbox_evidence.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| multiview rank47 4->38 | move 2 tracklets | direct 0.662902/0.524042/0.535239; delta +0.000067 IDF1 | promote after p005 |
| multiview rank37 27->79 | move 3 tracklets | direct 0.662875/0.524014/0.535223; delta +0.000040 IDF1 | direct positive but lower p005 than rank47 |
| multiview rank45 22->38 | move 2 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| multiview rank21 82->80 | move 6 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| multiview rank24 82->80 | move 5 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| multiview rank43 30->85 | move 2 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| multiview rank42 22->29 | move 5 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| multiview rank48 90->85 | move 2 tracklets | direct 0.662835/0.523970/0.535170; delta +0.000000 IDF1 | hard neutral/tie |
| density+p005 validation rank47 | apply density_simple then p005_area to 4->38 | 0.665084/0.525797/0.537115; config_name=p005_area; dropped_rows=45467 | canonical promotion |
| density+p005 validation rank37 | apply density_simple then p005_area to 27->79 | 0.665065/0.525777/0.537106; below rank47 by 0.000019 IDF1 | keep as positive label, not promote |

## Upload

- Bitbucket: `target branch wisc; commit after artifact generation`
- S3: `blocked locally: aws sts get-caller-identity failed with Unable to locate credentials`

## Next

Use rank46base labels to penalize stale tied families; launch a preemptible large-budget no-anchor structural/subpart sweep if local rank47base queues saturate.
