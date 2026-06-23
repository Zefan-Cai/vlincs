# No-anchor component-graph rank04 p005 gain

- Date: `2026-06-23`
- Pipeline module: `M7 candidate retrieval + M8 graph resolution + M12 delivery validation`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment/delivery candidate after valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

A no-anchor component-graph directional rescue moves 21 MCAM04/Tc6 tracklets from component 56 / gid 96000060 to component 86 / gid 960000350. Valid density_simple+p005_area improves IDF1/HOTA/AssA from 0.665246/0.525919/0.537198 to 0.666422/0.526871/0.537617.

## Metrics

- Baseline: `0.665246`
- Candidate: `0.666422`
- Delta: `0.001176`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

Generated 172 component-graph bridge candidates from the rank77+rank36 combo best using three feature views. The directional reviewer selected 8 non-GT larger-target bridges; rank04 is source component 56 to target component 86, with no-GT graph score 0.800816, target mean similarity 0.797995, best similarity 0.935692, min-view similarity 0.840768, overlap ratio 0.018367, source size 21, target size 107. The assignment exporter materialized only those 21 source tracklets as manifest_reassign; scoring then used the unchanged density_simple and p005_area delivery gates.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `No anchors; GT used only after materialization for full-score evaluation`

## Commands

```bash
python kit/compose_no_anchor_component_graph_candidates.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_combo_assignments/combo_rank77_rank36_s43_to38_s27_to79_5seq_assignments.csv --embeddings weakmetric_osnet --embeddings dinov2 --embeddings siglip2 --min-score 0.50 --top-n 220 --json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_candidates.json ...
```

```bash
python kit/filter_no_anchor_component_graph_small_fragment_rule.py --candidates-json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_candidates.json --json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_small_fragment.json ...
```

```bash
python kit/filter_no_anchor_component_graph_directional_rescue.py --candidates-json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_candidates.json --json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_directional_rescue.json ...
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_directional_rescue.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/rank67base_subpart_combo_assignments/combo_rank77_rank36_s43_to38_s27_to79_5seq_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue --selection-ranks 2,4,8 --fallback singleton
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_component_graph_high_mass_bridge_source_assignments.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_simple.zip
```

```bash
CONFIG=$(cat local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt); env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_simple.zip --config "$CONFIG" --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_p005_area.zip
```

```bash
python kit/export_no_anchor_subpart_visual_case.py --before-assignments <rank67 combo base> --after-assignments <rank04 assignments> --manifest reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/case_manifests/rank04_component_graph_directional_visual_manifest.json --rank 4 --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --output-dir reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/cases/rank04_component_graph_directional
```

## Code Paths

- `kit/compose_no_anchor_component_graph_candidates.py`
- `kit/filter_no_anchor_component_graph_directional_rescue.py`
- `kit/filter_no_anchor_component_graph_small_fragment_rule.py`
- `kit/filter_no_anchor_component_graph_rescue_rule.py`
- `kit/compose_no_anchor_small_fragment_combos.py`
- `kit/export_no_anchor_scheduler_manifest_assignments.py`
- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_candidates.json`
- `local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_directional_rescue.json`
- `local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_small_fragment.json`
- `local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_small_fragment_combos.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/assignments/rank04_component_graph_high_mass_bridge_source_assignments.csv`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_p005_area.zip`
- `reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/metrics/direct_full.json`
- `reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/metrics/density_simple.json`
- `reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/metrics/p005_area.json`

## Visual Cases

- Rank04 component graph bridge 56->86: Promoted case: 21 moved MCAM04/Tc6 tracklets, 8 sampled in the visual case, gid 96000060 -> 960000350.
  - failure: The previous combo kept a coherent residual island in component 56, so delivery forced those tracklets to gid 96000060 rather than the visually closer split-child target.
  - improvement: Directional component-graph rescue accepts the high-similarity larger-target bridge and improves direct, density_simple, and p005_area.
  - image: `cases/rank04_component_graph_directional/rank04_bbox_evidence.png`
  - html: `cases/rank04_component_graph_directional/case.html`
  - json: `cases/rank04_component_graph_directional/case.json`
- Rank02 small fragment 57->46: Positive ablation: 4 moved tracklets, gid 96000061 -> 96000050, but lower p005 gain than rank04.
  - failure: A tiny source component remained isolated as its own forced identity.
  - improvement: The small-fragment rule correctly moves only the low-mass source into a stronger target, but its global effect is smaller.
  - image: `cases/rank02_component_graph_small_fragment/rank02_bbox_evidence.png`
  - html: `cases/rank02_component_graph_small_fragment/case.html`
  - json: `cases/rank02_component_graph_small_fragment/case.json`
- Rank01 small-fragment combo: Positive ablation: 6 moved tracklets from two small fragments; improves p005 but remains below rank04.
  - failure: Two plausible tiny residual islands were both unresolved, but their combined delivery impact is small.
  - improvement: The combo confirms the family composes without hurting delivery, while the reviewer still promotes rank04 as the stronger move.
  - image: `cases/rank01_component_graph_small_combo/rank01_bbox_evidence.png`
  - html: `cases/rank01_component_graph_small_combo/case.html`
  - json: `cases/rank01_component_graph_small_combo/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| directional rank04 56->86 | move 21 MCAM04/Tc6 tracklets into target component 86 | direct 0.664229 / 0.525110 / 0.535737; density 0.666319 / 0.526792 / 0.537498; p005 0.666422 / 0.526871 / 0.537617 | promote |
| small fragment rank02 57->46 | move 4 small-fragment tracklets | p005 0.665420/0.526215/0.537529 | positive but below rank04 |
| small-fragment combo rank01 | compose 68->27 and 57->46 for 6 moved tracklets | p005 0.665422/0.526218/0.537532 | positive but below rank04 |
| directional rank02 30->14 | move 109 tracklets | direct 0.662644/0.523955/0.535174 | negative; do not run delivery |
| directional rank08 79->4 | move 26 tracklets | direct 0.662408/0.523506/0.534731 | negative; do not run delivery |
| small fragment rank01 68->27 | move 2 tracklets | direct 0.663080/0.524178/0.535331 | near-tie; tested in small-fragment combo |
| low-vote rescue rule | apply rescue filter to same 172 broad candidates | 0 selected | no materialized candidate |

## Upload

- Bitbucket: `prepared for publish to Novateur/vlincs_reid_by_search wisc; commit hash will be recorded after push`
- S3: `blocked: aws sts get-caller-identity returned Unable to locate credentials; local large artifacts remain at local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/rank04_density_p005_area.zip and related JSON/CSV paths`

## Next

Use this strong positive label to train/hand-tune a component-graph directional admission gate, then search combinations that include 56->86 while guarding against broad high-mass negative moves such as 30->14 and 79->4.
