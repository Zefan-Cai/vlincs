# No-anchor rank67-base rank77+rank36 combo p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7 candidate retrieval + M10 scheduler + M8 graph resolution + M12 delivery calibration`
- Used in pipeline: `yes: promoted as current no-anchor best assignment and canonical p005 delivery zip`
- Status: `gain`
- No-anchor: `True`

## Summary

The rank67-base reviewer found two independent direct-positive residual repairs: weakvideo rank77 43->38 and multiview rank36 27->79. A new overlay composer combined their explicit moved tracklets without conflicts. Canonical p005 improves from 0.665145/0.525833/0.537128 to 0.665246/0.525919/0.537198.

## Metrics

- Baseline: `0.665145`
- Candidate: `0.665246`
- Delta: `0.000101`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Generated rank67-base balanced, multiview, and weakvideo candidates from the promoted rank67 assignment. Added rank47base labels to the scheduler, direct-scored seven diverse rank67-base candidates, then composed rank77 and rank36 with a reusable no-anchor assignment overlay script. The final combo moves five additional tracklets and passes density_simple+p005_area delivery validation. The visual-case exporter was hardened to prefer raw-frame/video-root rendering when frames are available and to record `unavailable_coordinate_fallback` when only bbox parquet evidence is present.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `Feature cache: local_runs/s3_feature_cache_20260622`
- `No anchors; GT used only for post-hoc scoring/evaluation labels`
- `Raw-frame status: local Box/CloudStorage search, old raw-frame cache search, and Pluto h100-test-2/3 /mnt/localssd inspection did not contain the rank77/rank36 source frames; current case images are explicitly marked coordinate-only fallback`

## Commands

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_weakvideo_assignments/rank67_subpart_s43_to38_4seq_assignments.csv --feature-npz local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz ... --json rank67base_subpart_{balanced,multiview,weakvideo}_manifest.json
```

```bash
python kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py --manifest promoted/rank15/rank40/rank46/rank47/rank67base manifests --label-summary promoted/rank15/rank40/rank46/rank47 labels --baseline-idf1 0.662969 --output-json local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_ranked_with_oldlabels.json
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments <candidate.csv> --fallback singleton --json rank67base_subpart_reviewer_fullscore/<candidate>_full_export.json --zip-out <candidate>_full_export.zip
```

```bash
python kit/compose_no_anchor_assignment_overlays.py --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank47base_subpart_weakvideo_assignments/rank67_subpart_s43_to38_4seq_assignments.csv --candidate-assignment-csv rank77_subpart_s43_to38_2seq_assignments.csv --candidate-assignment-csv rank36_subpart_s27_to79_3seq_assignments.csv --assignment-out combo_rank77_rank36_s43_to38_s27_to79_5seq_assignments.csv --json combo_rank77_rank36_s43_to38_s27_to79_5seq_projection.json
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip combo_rank77_rank36_s43_to38_s27_to79_5seq_full_export.zip --policies density_simple --json combo_density_simple.json --zip-out combo_density_simple.zip
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip combo_density_simple.zip --config @local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt --json combo_p005_area.json --zip-out combo_p005_area.zip
```

```bash
python kit/export_no_anchor_subpart_visual_case.py --before-assignments rank67_base.csv --after-assignments combo_rank77_rank36_s43_to38_s27_to79_5seq_assignments.csv --manifest rank67base_subpart_{weakvideo,multiview}_manifest.json --rank {77,36} ...
```

## Code Paths

- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py`
- `kit/compose_no_anchor_assignment_overlays.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `reports/vlincs_iterations/20260623_rank67base_combo_rank77_rank36_p005_gain/`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_balanced_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_multiview_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_weakvideo_manifest.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_ranked_with_oldlabels.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_fullscore_label_summary.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_combo_assignments/combo_rank77_rank36_s43_to38_s27_to79_5seq_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_combo_assignments/combo_rank77_rank36_s43_to38_s27_to79_5seq_projection.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_fullscore/combo_rank77_rank36_s43_to38_s27_to79_5seq_full_export.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_delivery/combo_rank77_rank36_s43_to38_s27_to79_5seq_density_simple.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_delivery/combo_rank77_rank36_s43_to38_s27_to79_5seq_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_delivery/combo_rank77_rank36_s43_to38_s27_to79_5seq_density_p005_area.zip`

## Visual Cases

- Rank77 extends component 43 -> 38: Two additional tracklets move from component 43 / gid 96000046 to component 38 / gid 96000041. Image is coordinate-only fallback because the raw source frames were not present locally or on the current H100 nodes.
  - failure: Rank67 fixed four tracklets but left a smaller component-43 residual island.
  - improvement: Rank77 adds only the compatible two-tracklet subgroup and becomes one half of the promoted combo.
  - html: `cases/rank77_subpart_visual/case.html`
  - json: `cases/rank77_subpart_visual/case.json`
  - image: `cases/rank77_subpart_visual/rank77_bbox_evidence.png`
- Rank36 repairs component 27 -> 79: Three MCAM04 tracklets move from component 27 / gid 96000029 to component 79 / gid 96002330. Image is coordinate-only fallback because the raw source frames were not present locally or on the current H100 nodes.
  - failure: The rank67 base still had a small MCAM04 island in component 27 that matched component 79.
  - improvement: Rank36 adds an independent positive repair; rank77+rank36 composes into the best p005 score.
  - html: `cases/rank36_subpart_visual/case.html`
  - json: `cases/rank36_subpart_visual/case.json`
  - image: `cases/rank36_subpart_visual/rank36_bbox_evidence.png`
- Rank67 context: previous weakvideo repair: Four tracklets moved from component 43 / gid 96000046 to component 38 / gid 96000041 and established the previous best.
  - failure: Component 43 held a cross-camera residual island after rank47.
  - improvement: Rank67 repaired the first four compatible tracklets; rank77 extends this same residual family.
  - html: `cases/rank67_subpart_visual/case.html`
  - json: `cases/rank67_subpart_visual/case.json`
  - image: `cases/rank67_subpart_visual/rank67_bbox_evidence.png`
- Rank47 context: MCAM04 residual move: Two tracklets moved from component 4 / gid 96000004 to component 38 / gid 96000041.
  - failure: A small component-4 island shared target evidence with component 38.
  - improvement: This earlier repair created the component-38 target that rank67/rank77 continue to consolidate.
  - html: `cases/rank47_subpart_visual/case.html`
  - json: `cases/rank47_subpart_visual/case.json`
  - image: `cases/rank47_subpart_visual/rank47_bbox_evidence.png`
- Rank40 context: MCAM08 false-split island: Two MCAM08 tracklets moved from component 5 / gid 96000005 to component 35 / gid 96000038.
  - failure: A high-internal-similarity residual pair was left in the wrong source component.
  - improvement: The same subpart-repair pattern is now used by the rank67base combo.
  - html: `cases/rank40_subpart_visual/case.html`
  - json: `cases/rank40_subpart_visual/case.json`
  - image: `cases/rank40_subpart_visual/rank40_bbox_evidence.png`
  - image: `cases/rank40_subpart_visual/tracking_recall_failure_gt_m0038_montage.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| multiview rank35 83->82 | move 4 tracklets | direct tied 0.662969/0.524082/0.535252 | neutral label |
| multiview rank45 29->80 | move 2 tracklets | direct tied 0.662969/0.524082/0.535252 | neutral label |
| balanced rank20 83->79 | move 6 tracklets | direct tied 0.662969/0.524082/0.535252 | neutral label |
| multiview rank43 89->57 | move 2 tracklets | direct tied 0.662969/0.524082/0.535252 | neutral label |
| balanced rank27 90->81 | move 2 tracklets | direct tied 0.662969/0.524082/0.535252 | neutral label |
| weakvideo rank77 43->38 | move 2 tracklets | direct 0.663037/0.524131/0.535276; density 0.665112/0.525806/0.537036; p005 0.665211/0.525881/0.537152 | positive member |
| multiview rank36 27->79 | move 3 tracklets | direct 0.663009/0.524126/0.535304; density 0.665081/0.525795/0.537057; p005 0.665180/0.525871/0.537174 | positive member |
| combo rank77+rank36 | overlay 5 explicit moved tracklets from two candidate assignments | direct 0.663077/0.524175/0.535328; density 0.665148/0.525843/0.537081; p005 0.665246/0.525919/0.537198 | promote canonical best |
| rank67base reviewer | 940 candidates, 52 matched labels after adding rank47base labels | LOOCV RMSE 0.000118; rank corr 0.714; top predicted positives were mostly neutral but rank77/rank36 composed positively | keep residual-island combo search |

## Upload

- Bitbucket: `pushed to Bitbucket wisc branch as `Promote rank77 rank36 combo p005 gain``
- S3: `blocked: `aws sts get-caller-identity` failed locally with "Unable to locate credentials"; artifacts remain under reports/vlincs_iterations/20260623_rank67base_combo_rank77_rank36_p005_gain and local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_reviewer_delivery/`

## Next

Promote combo as rank77+rank36 base, fold eight new labels into the reviewer, search additional non-conflicting residual-island overlays, and restore a durable raw-frame mirror so future case studies render real pixels rather than coordinate fallback panels.
