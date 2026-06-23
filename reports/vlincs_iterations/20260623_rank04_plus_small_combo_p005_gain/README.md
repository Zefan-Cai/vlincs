# No-anchor rank04 plus small-fragment combo p005 gain

- Date: `2026-06-23`
- Pipeline module: `M8 graph resolution + M12 delivery validation`
- Used in pipeline: `yes: promoted as the current no-anchor best assignment/delivery candidate after valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

Starting from the rank04 best, adding the small-fragment combo 68->27 and 57->46 moves six residual tracklets and improves valid density_simple+p005_area IDF1/HOTA/AssA from 0.666422/0.526871/0.537617 to 0.666597/0.527169/0.537950.

## Metrics

- Baseline: `0.666422`
- Candidate: `0.666597`
- Delta: `0.000175`
- Metric name: `canonical density_simple+p005_area IDF1`

## Implementation

The proposer reused no-anchor component-graph small-fragment manifests generated before scoring. Rank02 alone moves four component 57 tracklets to component 46 / gid 96000050. The combo additionally moves two component 68 tracklets to component 27 / gid 96000029. The replay step uses rank04 as the base assignment, materializes only the accepted_preview source_seqs, refreshes component sizes/status, then runs direct full-score followed by density_simple and p005_area.

## Environment

- `Repo: /Users/zcai/Codex/vlincs_reid_by_search`
- `Dataset root: /Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `Tracklet parquets: kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `No anchors; GT used only after materialization for direct/full-score and delivery evaluation`
- `Raw video frames are still unavailable locally, so visual cases use coordinate-only bbox fallback panels`

## Commands

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_small_fragment.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/assignments/rank04_component_graph_high_mass_bridge_source_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment --selection-ranks 1,2 --fallback singleton
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --scheduler-json local_runs/offline_no_anchor_split_probe_20260623/combo_rank77_rank36_component_graph_small_fragment_combos.json --base-assignment-csv local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/assignments/rank04_component_graph_high_mass_bridge_source_assignments.csv --run-dir local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo --selection-ranks 1 --fallback singleton
```

```bash
env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.zip --policies density_simple --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_simple.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_simple.zip
```

```bash
CONFIG=$(cat local_runs/offline_no_anchor_split_probe_20260623/p005_area_config.txt); env DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_simple.zip --config "$CONFIG" --json local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_p005_area.json --zip-out local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_p005_area.zip
```

```bash
python kit/export_no_anchor_subpart_visual_case.py --before-assignments local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/directional_rescue/assignments/rank04_component_graph_high_mass_bridge_source_assignments.csv --after-assignments local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv --manifest reports/vlincs_iterations/20260623_component_graph_rank04_p005_gain/case_manifests/rank01_component_graph_small_combo_visual_manifest.json --rank 1 --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --output-dir reports/vlincs_iterations/20260623_rank04_plus_small_combo_p005_gain/cases/rank04_plus_small_combo
```

## Code Paths

- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `kit/export_no_anchor_scheduler_manifest_assignments.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/vlincs_iterations/20260623_rank04_plus_small_combo_p005_gain/README.md`
- `reports/vlincs_iterations/20260623_rank04_plus_small_combo_p005_gain/presentation.html`

## Artifacts

- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment/summary.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment/rank02_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment/rank02_density_p005_area.zip`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/summary.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_p005_area.json`
- `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/rank01_density_p005_area.zip`
- `reports/vlincs_iterations/20260623_rank04_plus_small_combo_p005_gain/metrics/rank04_plus_small_combo_density_p005_area.json`

## Visual Cases

- Context: rank04 component graph bridge 56->86: Previous promoted base: 21 MCAM04/Tc6 tracklets moved from gid 96000060 to gid 960000350.
  - failure: Before rank04, component 56 stayed as a forced residual island.
  - improvement: Rank04 becomes the base used by this iteration.
  - image: `cases/context_rank04_component_graph_directional/rank04_bbox_evidence.png`
  - html: `cases/context_rank04_component_graph_directional/case.html`
  - json: `cases/context_rank04_component_graph_directional/case.json`
- Rank04 + rank02 small fragment 57->46: Four residual component 57 tracklets move to component 46 / gid 96000050; p005 reaches 0.666595.
  - failure: Rank04 still left component 57 as a tiny forced identity.
  - improvement: Small-fragment attachment resolves the residual island and lifts MCAM04/Tc6.
  - image: `cases/rank04_plus_rank02_small_fragment/rank02_bbox_evidence.png`
  - html: `cases/rank04_plus_rank02_small_fragment/case.html`
  - json: `cases/rank04_plus_rank02_small_fragment/case.json`
- Rank04 + small-fragment combo: The best candidate composes 68->27 and 57->46 for six moved tracklets; p005 reaches 0.666597.
  - failure: Two tiny residual islands remained after the high-mass rank04 bridge.
  - improvement: The combo adds two non-overlapping no-anchor small-fragment moves and gives the best canonical score.
  - image: `cases/rank04_plus_small_combo/rank01_bbox_evidence.png`
  - html: `cases/rank04_plus_small_combo/case.html`
  - json: `cases/rank04_plus_small_combo/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| rank04 + small rank01 68->27 | add two component 68 tracklets to component 27 | direct 0.664231/0.525112/0.535740 | near-tie; not promoted alone |
| rank04 + small rank02 57->46 | add four component 57 tracklets to component 46 | direct 0.664401/0.525406/0.536068; density 0.666492/0.527088/0.537828; p005 0.666595/0.527166/0.537947 | canonical positive |
| rank04 + small-fragment combo | compose 68->27 and 57->46 for six moved tracklets | direct 0.664403/0.525408/0.536070; density 0.666494/0.527090/0.537830; p005 0.666597/0.527169/0.537950 | promote best |
| p005 validity check | reuse fixed p005_area config string, not @path literal | config_name=p005_area; dropped_rows=45467; rows=1642574 | valid canonical delivery |

## Upload

- Bitbucket: `published to Novateur/vlincs_reid_by_search branch wisc at commit b4694d9 (Promote rank04 small combo p005 gain)`
- S3: `blocked: aws sts get-caller-identity returned Unable to locate credentials; local large artifacts remain under local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/`

## Next

Use the new positive interaction label to search residual small-fragment combos on top of rank04_plus_small_combo, but downweight near-tie 68->27 unless it composes with a stronger MCAM04 move.
