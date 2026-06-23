# No-anchor rank08 top28 subset 37->86 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7+M8+M12`
- Used in pipeline: `yes: promoted as current no-anchor best after valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

Side-effect-aware subset proposer splits the prior 69-tracklet component 37 bridge. Rank08 moves only the top 28 target-similarity tracklets into component 86 / gid 960000350 and improves valid p005_area from 0.667536/0.528210/0.538672 to 0.668146/0.528669/0.539001.

## Metrics

- Baseline: `0.667536`
- Candidate: `0.668146`
- Delta: `0.000610`
- Metric name: `canonical p005_area IDF1`

## Implementation

Added a no-anchor component-subset proposer that ranks source-component 37 tracklets by target-component 86 centroid similarity and by video/camera side-effect slices. The promoted rank08 variant attaches 28 high-similarity tracklets, avoiding the weaker MCAM03/Tc8/MCAM08 tail from the full 69-tracklet bridge. The exporter was also made schema-compatible with subset candidates so case evidence tables show moved_tracklets, source/target components, GIDs, and similarity fields.

## Environment

- `cwd=/Users/zcai/Codex/vlincs_reid_by_search`
- `DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Python native local run; no anchors; GT labels only used after materialization for full-score evaluation`
- `p005 config source=local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt`

## Commands

```bash
python kit/compose_no_anchor_component_subset_variants.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --source-component 37 --target-component 86 --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_dinov2base_s1_20260620.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_siglip2_person_reid_s1_20260620.npz --json local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.json --csv local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.csv --md local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.md
```

```bash
DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe --selection-ranks 1,2,3,4,5,6,7,8,9,13,14,17 --fallback singleton
```

```bash
DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_component_subset_attach_source_assignments.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_simple.zip
```

```bash
CONFIG=$(cat local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt); DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_simple.zip --config "$CONFIG" --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_p005_area.zip
```

```bash
python kit/export_no_anchor_subpart_visual_case.py --before-assignments local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --after-assignments local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/assignments/rank08_component_subset_attach_source_assignments.csv --manifest local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.json --rank 8 --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --output-dir reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/cases/rank08_top28_subset_visual
```

```bash
python /Users/zcai/.codex/skills/vlincs-open-world-reid/scripts/make_iteration_artifacts.py --spec-json reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/spec.json --output-dir reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain
```

## Code Paths

- `kit/compose_no_anchor_component_subset_variants.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_component_subset_variants.md`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/sample_full_results.jsonl`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/assignments/rank08_component_subset_attach_source_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_simple.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_subset_probe/rank08_density_p005_area.json`
- `reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/presentation.html`
- `reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/spec.json`

## Visual Cases

- Rank08 top28 subset: component 37 -> 86: 28 moved tracklets with three sampled frames each. The case shows predicted_global_id 96000040 -> 960000350 and component 37 -> 86 for the committed subset. Raw frame pixels are absent locally, so this is coordinate-only fallback evidence, not a path failure.
  - failure: The previous full 69-tracklet bridge included weaker tail tracklets and produced small side-effect losses.
  - improvement: The subset proposer commits only the strongest target-similarity island while keeping the delivery score positive.
  - image: `cases/rank08_top28_subset_visual/rank08_bbox_evidence.png`
  - html: `cases/rank08_top28_subset_visual/case.html`
  - json: `cases/rank08_top28_subset_visual/case.json`
- Real-frame context: cross-camera global-id success: Previously extracted real-frame montage showing what a true global-id evidence chain should look like when raw frames are available. Included to contrast with coordinate-only rank08 panels.
  - failure: Coordinate-only case panels cannot show clothing/person appearance.
  - improvement: The report carries real-frame context alongside coordinate evidence until the current rank08 videos are available.
  - image: `cases/context_real_frame_examples/case_success_cross_camera_m0048_real.png`
- Real-frame context: same tracklet sequence: Real-frame tracklet timeline from the earlier detailed report, used as a visual reference for bbox crops and repeated-frame support.
  - failure: Single-frame diagnostics do not show temporal consistency.
  - improvement: Timeline panels show how repeated frames support a global-id decision.
  - image: `cases/context_real_frame_examples/case_focus_m0012_tracklet_sequence_real.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| full 37->86 bridge context | Move all 69 component-37 tracklets to component 86; previous promoted bridge. | valid p005_area IDF1/HOTA/AssA=0.667536/0.528210/0.538672; direct=0.665296/0.526390/0.536737 | baseline for this subset search; useful but retained side-effect losses |
| direct rank08 top28_target_sim | top 28 source tracklets by target-centroid similarity; subset_size=28; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 26} | direct IDF1/HOTA/AssA=0.665950/0.526899/0.537108 | promoted to density_simple + p005_area |
| direct rank06 top36_target_sim | top 36 source tracklets by target-centroid similarity; subset_size=36; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 2, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 33} | direct IDF1/HOTA/AssA=0.665742/0.526743/0.536994 | not promoted; lower direct score than rank08 |
| direct rank05 top45_target_sim | top 45 source tracklets by target-centroid similarity; subset_size=45; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 2, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 6, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 36, 'vlincs_MS01_MC0001_MCAM08_2024-03-Tc6': 1} | direct IDF1/HOTA/AssA=0.665657/0.526681/0.536946 | not promoted; lower direct score than rank08 |
| direct rank02 top55_target_sim | top 55 source tracklets by target-centroid similarity; subset_size=55; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 2, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 12, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 39, 'vlincs_MS01_MC0001_MCAM08_2024-03-Tc6': 2} | direct IDF1/HOTA/AssA=0.665630/0.526700/0.536977 | not promoted; lower direct score than rank08 |
| direct rank03 sideeffect_positive_videos_mcam04tc6_mcam03tc8 | post-hoc reviewer side-effect label: keep videos that gained in full p005; subset_size=59; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 14, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 45} | direct IDF1/HOTA/AssA=0.665627/0.526740/0.537032 | not promoted; lower direct score than rank08 |
| direct rank04 exclude_loss_videos_mcam03tc6_mcam08tc6 | post-hoc reviewer side-effect label: exclude videos that dropped in full p005; subset_size=61; videos={'vlincs_MS01_MC0001_MCAM00_2024-03-Tc8': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 14, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 45, 'vlincs_MS01_MC0001_MCAM06_2024-03-Tc6': 1} | direct IDF1/HOTA/AssA=0.665592/0.526702/0.536999 | not promoted; lower direct score than rank08 |
| direct rank01 top64_target_sim | top 64 source tracklets by target-centroid similarity; subset_size=64; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 2, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 14, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 43, 'vlincs_MS01_MC0001_MCAM06_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM08_2024-03-Tc6': 4} | direct IDF1/HOTA/AssA=0.665471/0.526564/0.536876 | not promoted; lower direct score than rank08 |
| direct rank09 top20_target_sim | top 20 source tracklets by target-centroid similarity; subset_size=20; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 19} | direct IDF1/HOTA/AssA=0.665339/0.526280/0.536625 | not promoted; lower direct score than rank08 |
| direct rank07 camera_MCAM04 | only source tracklets from camera MCAM04; subset_size=45; videos={'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 45} | direct IDF1/HOTA/AssA=0.665139/0.526184/0.536581 | not promoted; lower direct score than rank08 |
| direct rank14 video_MCAM03_Tc8 | only source tracklets from video vlincs_MS01_MC0001_MCAM03_2024-03-Tc8; subset_size=14; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 14} | direct IDF1/HOTA/AssA=0.664893/0.525884/0.536358 | not promoted; lower direct score than rank08 |
| direct rank13 camera_MCAM03 | only source tracklets from camera MCAM03; subset_size=17; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 3, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 14} | direct IDF1/HOTA/AssA=0.664768/0.525763/0.536270 | not promoted; lower direct score than rank08 |
| direct rank17 camera_MCAM08 | only source tracklets from camera MCAM08; subset_size=5; videos={'vlincs_MS01_MC0001_MCAM08_2024-03-Tc6': 5} | direct IDF1/HOTA/AssA=0.664232/0.525253/0.535968 | not promoted; lower direct score than rank08 |
| rank08 density_simple | Apply delivery density_simple filter to rank08 top28 assignment zip. | IDF1/HOTA/AssA=0.668041/0.528586/0.538876; dropped_rows=35029; rows=1688041 | positive delivery interaction; run canonical p005_area |
| rank08 p005_area | Apply fixed p005_area thresholds after density_simple. | IDF1/HOTA/AssA=0.668146/0.528669/0.539001; config_name=p005_area; dropped_rows=45467; rows=1642574 | canonical no-anchor e2e best; promote |

## Upload

- Bitbucket: `pushed to Novateur/vlincs_reid_by_search wisc in commit 87fea21c4f44c1cfe708c9833df442b2c8a7f3ad`
- S3: `blocked: `aws sts get-caller-identity` returned `Unable to locate credentials`; local package remains at reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/; target prefix would be s3://dit-scale-up/zcai/vlincs/reports/vlincs_iterations/20260623_rank01_37to86_top28_p005_gain/`

## Next

Continue side-effect-aware subset search around top24/top32 and add scorer caching or a preemptible Pluto batch. The current result is still below the 0.70 end-to-end target, so the goal remains active.
