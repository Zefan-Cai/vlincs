# No-anchor rank04 small-combo rank01 37->86 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7 candidate retrieval + M8 graph resolution + M12 delivery validation`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment/delivery candidate after valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

On top of the rank04+small-combo best, component-graph rank01 moves 69 component 37 tracklets into component 86 / gid 960000350. Valid density_simple+p005_area improves IDF1/HOTA/AssA from 0.666597/0.527169/0.537950 to 0.667536/0.528210/0.538672.

## Metrics

- Baseline: `0.666597`
- Candidate: `0.667536`
- Delta: `0.000939`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

The current-best assignment is re-indexed as a component graph. The proposer loads three no-anchor tracklet feature caches and emits 30 bridge candidates. The strict small-fragment filter has no candidates; the directional filter only recovers stale hard-negative 79->4, so the reviewer samples high-scoring non-stale ranks 1, 7, and 8. Rank01 37->86 is direct-positive, while 91->90 and 90->91 tie. Only rank01 is sent through density_simple and p005_area.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Feature caches: local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz; ds1_tracklet_dinov2base_s1_20260620.npz; ds1_tracklet_siglip2_person_reid_s1_20260620.npz`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `No anchors; GT used only after materialization for direct/full-score and delivery evaluation`
- `Raw video frames are still unavailable locally, so visual cases use coordinate-only bbox fallback panels`

## Commands

```bash
python kit/compose_no_anchor_component_graph_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_dinov2base_s1_20260620.npz --embeddings local_runs/s3_feature_cache_20260622/ds1_tracklet_siglip2_person_reid_s1_20260620.npz --min-score 0.50 --top-n 260 --json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.json --csv local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.csv --md local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.md
```

```bash
python kit/filter_no_anchor_component_graph_directional_rescue.py --candidates-json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.json --json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_directional_rescue.json --csv local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_directional_rescue.csv --md local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_directional_rescue.md
```

```bash
python kit/filter_no_anchor_component_graph_small_fragment_rule.py --candidates-json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.json --json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_small_fragment.json --csv local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_small_fragment.csv --md local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_small_fragment.md
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe --selection-ranks 1,7,8 --fallback singleton
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_component_graph_high_mass_bridge_source_assignments.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_simple.zip
```

```bash
CONFIG=$(cat local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt); env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_simple.zip --config "$CONFIG" --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_p005_area.zip
```

## Code Paths

- `kit/compose_no_anchor_component_graph_candidates.py`
- `kit/filter_no_anchor_component_graph_directional_rescue.py`
- `kit/filter_no_anchor_component_graph_small_fragment_rule.py`
- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_rank04_small_combo_rank01_37to86_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank04_small_combo_rank01_37to86_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_candidates.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_directional_rescue.json`
- `local_runs/offline_no_anchor_split_probe_20260623/rank04_plus_small_combo_component_graph_small_fragment.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/summary.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/rank01_density_p005_area.zip`
- `reports/vlincs_iterations/20260623_rank04_small_combo_rank01_37to86_p005_gain/metrics/rank01_density_p005_area.json`

## Visual Cases

- Context: rank04 component graph bridge 56->86: Earlier base repair that created component 86 / gid 960000350 as the strong target.
  - failure: Before rank04, component 56 was a residual forced identity.
  - improvement: Component 86 becomes a stable target for later residual attachments.
  - image: `cases/context_rank04_component_graph_directional/rank04_bbox_evidence.png`
  - html: `cases/context_rank04_component_graph_directional/case.html`
  - json: `cases/context_rank04_component_graph_directional/case.json`
- Context: rank04 + small-fragment combo: Previous best that added 68->27 and 57->46 for six moved tracklets.
  - failure: Small forced identities remained after rank04.
  - improvement: The previous iteration resolved two tiny residual islands.
  - image: `cases/context_rank04_plus_small_combo/rank01_bbox_evidence.png`
  - html: `cases/context_rank04_plus_small_combo/case.html`
  - json: `cases/context_rank04_plus_small_combo/case.json`
- Rank01 component bridge 37->86: Current promoted move: 69 component 37 tracklets attach to component 86 / gid 960000350.
  - failure: The previous best left component 37 as a separate forced global ID despite high component-graph affinity to component 86.
  - improvement: The new bridge improves the net canonical score while recording per-video side effects.
  - image: `cases/rank01_37to86_component_bridge/rank01_bbox_evidence.png`
  - html: `cases/rank01_37to86_component_bridge/case.html`
  - json: `cases/rank01_37to86_component_bridge/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| strict small-fragment filter | rerun on current best | 0 selected from 30 candidates | no materialized candidate |
| directional filter | rerun on current best | selected only 79->4, a known stale direct-negative family | skip as hard negative |
| rank01 37->86 | move 69 component 37 tracklets into component 86 | direct 0.665296/0.526390/0.536737; density 0.667439/0.528134/0.538552; p005 0.667536/0.528210/0.538672 | promote |
| rank07 91->90 | move 54 tracklets | direct 0.664403/0.525408/0.536070 | tie; do not run delivery |
| rank08 90->91 | move 40 tracklets | direct 0.664403/0.525408/0.536070 | tie; do not run delivery |

## Upload

- Bitbucket: `pushed to Novateur/vlincs_reid_by_search wisc at commit decbb2e`
- S3: `blocked: aws sts get-caller-identity returned Unable to locate credentials; local large artifacts remain under local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_small_combo_next_component_probe/`

## Next

Search side-effect-aware variants of 37->86: preserve the MCAM04/Tc6 and MCAM03/Tc8 gains while reducing MCAM03/Tc6 and MCAM08/Tc6 losses, likely by sub-splitting component 37 before attach.
