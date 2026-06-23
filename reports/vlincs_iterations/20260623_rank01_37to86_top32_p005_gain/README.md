# No-anchor rank06 top32 subset 37->86 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7+M8+M12`
- Used in pipeline: `yes: promoted as current no-anchor best after valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

A local top-k boundary search around the promoted top28 bridge found that top32 recovers four additional compatible component-37 tracklets. Rank06 top32 improves valid p005_area from 0.668146/0.528669/0.539001 to 0.668198/0.528747/0.539071.

## Metrics

- Baseline: `0.668146`
- Candidate: `0.668198`
- Delta: `0.000052`
- Metric name: `canonical p005_area IDF1`

## Implementation

Reused the no-anchor component-subset proposer with custom top-k candidates 22,24,26,28,30,32,34,36. Direct scoring showed top32 as the best neighbor. The top32 delivery path keeps the same fixed density_simple and p005_area gates as the previous promoted result, so the gain is a boundary refinement rather than a new delivery heuristic.

## Environment

- `cwd=/Users/zcai/Codex/vlincs_reid_by_search`
- `DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `No anchors; GT labels only used after materialization for full-score evaluation`
- `p005 config source=local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt`

## Commands

```bash
python kit/compose_no_anchor_component_subset_variants.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --source-component 37 --target-component 86 --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_dinov2base_s1_20260620.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_siglip2_person_reid_s1_20260620.npz --top-ks 22,24,26,28,30,32,34,36 --json local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.json --csv local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.csv --md local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.md
```

```bash
DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe --selection-ranks 5,6,7,9,10,11 --fallback singleton
```

```bash
DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_component_subset_attach_source_assignments.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_simple.zip
```

```bash
CONFIG=$(cat local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt); DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_simple.zip --config "$CONFIG" --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_p005_area.zip
```

```bash
python /Users/zcai/.codex/skills/vlincs-open-world-reid/scripts/make_iteration_artifacts.py --spec-json reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/spec.json --output-dir reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain
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

- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/rank01_37to86_top28_neighbor_variants.md`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/sample_full_results.jsonl`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/assignments/rank06_component_subset_attach_source_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_simple.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank01_37to86_top28_neighbor_probe/rank06_density_p005_area.json`
- `reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/presentation.html`
- `reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/spec.json`

## Visual Cases

- Rank06 top32 subset: component 37 -> 86: 32 moved tracklets with three sampled frames each. Shows predicted_global_id 96000040 -> 960000350 and component 37 -> 86. Raw frame pixels are absent locally, so this is coordinate-only fallback evidence.
  - failure: Top28 omitted four compatible high-similarity tracklets near the boundary.
  - improvement: Top32 includes those four tracklets and improves direct, density, and p005 delivery.
  - image: `cases/rank06_top32_subset_visual/rank06_bbox_evidence.png`
  - html: `cases/rank06_top32_subset_visual/case.html`
  - json: `cases/rank06_top32_subset_visual/case.json`
- Real-frame context: cross-camera global-id success: Previously extracted real-frame montage showing what a true global-id evidence chain should look like when raw frames are available.
  - failure: Coordinate-only panels cannot show appearance cues.
  - improvement: The package carries real-frame context alongside coordinate evidence until rank06 raw frames are available.
  - image: `cases/context_real_frame_examples/case_success_cross_camera_m0048_real.png`
- Real-frame context: same tracklet sequence: Real-frame tracklet timeline from the earlier detailed report, used as a visual reference for bbox crops and repeated-frame support.
  - failure: Single-frame diagnostics do not show temporal consistency.
  - improvement: Timeline panels show repeated-frame evidence for a global-id decision.
  - image: `cases/context_real_frame_examples/case_focus_m0012_tracklet_sequence_real.png`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| previous top28 promoted baseline | Move top28 target-similarity subset from component 37 to component 86. | direct=0.665950/0.526899/0.537108; density_simple=0.668041/0.528586/0.538876; p005_area=0.668146/0.528669/0.539001 | baseline for local top-k boundary search |
| direct rank06 top32_target_sim | top 32 source tracklets by target-centroid similarity; subset_size=32; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 2, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 29} | direct IDF1/HOTA/AssA=0.666005/0.526982/0.537182 | promoted to density_simple + p005_area |
| direct rank07 top30_target_sim | top 30 source tracklets by target-centroid similarity; subset_size=30; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 28} | direct IDF1/HOTA/AssA=0.665963/0.526925/0.537131 | not promoted; lower direct score than top32 |
| direct rank09 top26_target_sim | top 26 source tracklets by target-centroid similarity; subset_size=26; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 24} | direct IDF1/HOTA/AssA=0.665931/0.526865/0.537077 | not promoted; lower direct score than top32 |
| direct rank05 top34_target_sim | top 34 source tracklets by target-centroid similarity; subset_size=34; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 2, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 31} | direct IDF1/HOTA/AssA=0.665897/0.526878/0.537095 | not promoted; lower direct score than top32 |
| direct rank10 top24_target_sim | top 24 source tracklets by target-centroid similarity; subset_size=24; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM03_2024-03-Tc8': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 22} | direct IDF1/HOTA/AssA=0.665711/0.526638/0.536894 | not promoted; lower direct score than top32 |
| direct rank11 top22_target_sim | top 22 source tracklets by target-centroid similarity; subset_size=22; videos={'vlincs_MS01_MC0001_MCAM03_2024-03-Tc6': 1, 'vlincs_MS01_MC0001_MCAM04_2024-03-Tc6': 21} | direct IDF1/HOTA/AssA=0.665573/0.526502/0.536789 | not promoted; lower direct score than top32 |
| rank06 density_simple | Apply delivery density_simple filter to top32 assignment zip. | IDF1/HOTA/AssA=0.668093/0.528665/0.538947; dropped_rows=35029; rows=1688041 | positive delivery interaction; run canonical p005_area |
| rank06 p005_area | Apply fixed p005_area thresholds after density_simple. | IDF1/HOTA/AssA=0.668198/0.528747/0.539071; config_name=p005_area; dropped_rows=45467; rows=1642574 | canonical no-anchor e2e best; promote |

## Upload

- Bitbucket: `pending publish to Novateur/vlincs_reid_by_search wisc`
- S3: `blocked: `aws sts get-caller-identity` returned `Unable to locate credentials`; local package remains at reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/; target prefix would be s3://dit-scale-up/zcai/vlincs/reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/`

## Next

Continue beyond local top-k boundary search: test interaction with additional component repairs and run a preemptible Pluto batch for broader side-effect-aware subsets. Still below the 0.70 end-to-end target.
