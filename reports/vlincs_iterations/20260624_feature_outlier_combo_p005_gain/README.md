# No-anchor feature-outlier combo p005 gain

- Date: `2026-06-24`
- Pipeline module: `M5+M8+M10+M12`
- Used in pipeline: `yes: promoted as the new default no-anchor replay assignment after direct, density_simple and valid p005_area scoring`
- Status: `gain`
- No-anchor: `True`

## Summary

A scratch no-anchor feature-outlier reviewer found four one-tracklet islands that were more consistent with alternate components across weakmetric, DINOv2 and SigLIP. The promoted combo moves seq2354 4->0, seq685 86->91, seq9343 45->26, and seq5628 20->3. Canonical p005_area improves from 0.668332 to 0.668508.

## Metrics

- Baseline: `0.668332`
- Candidate: `0.668508`
- Delta: `0.000176`
- Metric name: `canonical p005_area IDF1`

## Implementation

The proposer reads only the current assignment CSV and no-anchor feature NPZs. For each source component it scores tracklets whose source centroid support is weak or locally conflicted, then searches target components with no temporal overlap. A candidate must clear target centroid similarity, centroid margin, neighbor margin and multi-view vote thresholds. The reviewer first direct-scores diverse single moves, then composes positive non-overlapping moves. Only the four-move combo is sent to density_simple and p005_area.

## Environment

- `repo=/Users/zcai/Codex/vlincs_reid_by_search`
- `verified scoring python=/private/tmp/vlincs-wisc-demo-lfs.TMM8XH/.venv-demo/bin/python`
- `DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622`
- `tracklets=kit/demo_data/ds1/tracklets/*/tracklets.parquet`
- `features=weakmetric_osnet_fused + DINOv2 + SigLIP NPZ caches`
- `no anchors; GT used only by evaluator after assignment materialization`

## Commands

```bash
python reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/propose_feature_outlier_relinks.py --assignment-csv local_runs/offline_no_anchor_split_probe_20260624/visual_positive_subcluster_ctf_probe/assignments/rank02_visual_positive_subcluster_ctf_topk_source_assignments.csv --feature weak:local_runs/s3_feature_cache_20260622/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:1.0 --feature dino:local_runs/s3_feature_cache_20260622/ds1_tracklet_dinov2base_s1_20260620.npz:0.75 --feature siglip:local_runs/s3_feature_cache_20260622/ds1_tracklet_siglip2_person_reid_s1_20260620.npz:0.75 --assignments-dir local_runs/current_best_feature_outlier_probe_20260624/replay_assignments --summary-json local_runs/current_best_feature_outlier_probe_20260624/replay_summary.json --summary-csv local_runs/current_best_feature_outlier_probe_20260624/replay_summary.csv --min-source-size 8 --min-target-size 6 --max-source-centroid 0.72 --min-target-centroid 0.60 --min-centroid-margin 0.02 --min-neighbor-margin 0.02 --min-view-votes 0.67 --view-vote-threshold 0.55 --group-sizes 1,2,4,8 --emit-top-groups 24 --skip-pairs '37->86,55->26,74->35,66->2,42->3,76->26,86->78,30->38,91->88,89->87,82->11,85->8,79->4'
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/combo_01_03_04_05_feature_outlier_assignments.csv --fallback singleton --json local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_full_export.json --zip-out local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_full_export.zip
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_full_export.zip --policies density_simple --json local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_density_simple.json --zip-out local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_density_simple.zip
```

```bash
PYTHONPATH=$PWD DATA_ROOT=/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622 python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_density_simple.zip --config "$(cat reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/p005_area_config.txt)" --json local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_density_p005_area.json --zip-out local_runs/current_best_feature_outlier_probe_20260624/fullscore/combo_01_03_04_05_density_p005_area.zip
```

```bash
bash reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/reproduce.sh
```

## Code Paths

- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/propose_feature_outlier_relinks.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/combo_01_03_04_05_feature_outlier_assignments.csv`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/p005_area_config.txt`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/expected/combo_01_03_04_05_full_export.json`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/expected/combo_01_03_04_05_density_simple.json`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/expected/combo_01_03_04_05_density_p005_area.json`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/provenance/feature_outlier_summary.csv`
- `reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/provenance/feature_outlier_combo_case_manifest.json`

## Visual Cases

- Feature-outlier relink seq 2354: component 4 -> 0: Three sampled bbox frames show the exact tracklet before/after global ID in the promoted four-tracklet combo.
  - failure: The old graph kept this visual outlier inside its source component even though multi-view feature evidence preferred a different component.
  - improvement: The new reviewer moves only the local outlier and keeps the surrounding component untouched; delivery is accepted only after direct, density_simple, and p005_area validation.
  - image: `cases/rank01_feature_outlier/rank01_bbox_evidence.png`
  - html: `cases/rank01_feature_outlier/case.html`
  - json: `cases/rank01_feature_outlier/case.json`
- Feature-outlier relink seq 685: component 86 -> 91: Three sampled bbox frames show the exact tracklet before/after global ID in the promoted four-tracklet combo.
  - failure: The old graph kept this visual outlier inside its source component even though multi-view feature evidence preferred a different component.
  - improvement: The new reviewer moves only the local outlier and keeps the surrounding component untouched; delivery is accepted only after direct, density_simple, and p005_area validation.
  - image: `cases/rank02_feature_outlier/rank02_bbox_evidence.png`
  - html: `cases/rank02_feature_outlier/case.html`
  - json: `cases/rank02_feature_outlier/case.json`
- Feature-outlier relink seq 9343: component 45 -> 26: Three sampled bbox frames show the exact tracklet before/after global ID in the promoted four-tracklet combo.
  - failure: The old graph kept this visual outlier inside its source component even though multi-view feature evidence preferred a different component.
  - improvement: The new reviewer moves only the local outlier and keeps the surrounding component untouched; delivery is accepted only after direct, density_simple, and p005_area validation.
  - image: `cases/rank03_feature_outlier/rank03_bbox_evidence.png`
  - html: `cases/rank03_feature_outlier/case.html`
  - json: `cases/rank03_feature_outlier/case.json`
- Feature-outlier relink seq 5628: component 20 -> 3: Three sampled bbox frames show the exact tracklet before/after global ID in the promoted four-tracklet combo.
  - failure: The old graph kept this visual outlier inside its source component even though multi-view feature evidence preferred a different component.
  - improvement: The new reviewer moves only the local outlier and keeps the surrounding component untouched; delivery is accepted only after direct, density_simple, and p005_area validation.
  - image: `cases/rank04_feature_outlier/rank04_bbox_evidence.png`
  - html: `cases/rank04_feature_outlier/case.html`
  - json: `cases/rank04_feature_outlier/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| rank01 4->0 seq2354 | rank01 4->0 seq2354 | direct IDF1/HOTA/AssA=0.666172/0.527149/0.537308; delta=+0.000032 | positive direct |
| rank02 14->30 seq9202 | rank02 14->30 seq9202 | direct IDF1/HOTA/AssA=0.666088/0.527050/0.537228; delta=-0.000052 | killed direct-negative |
| rank03 86->91 seq685 | rank03 86->91 seq685 | direct IDF1/HOTA/AssA=0.666207/0.527183/0.537338; delta=+0.000067 | positive direct |
| rank04 45->26 seq9343 | rank04 45->26 seq9343 | direct IDF1/HOTA/AssA=0.666195/0.527173/0.537331; delta=+0.000055 | positive direct |
| rank05 20->3 seq5628 | rank05 20->3 seq5628 | direct IDF1/HOTA/AssA=0.666160/0.527137/0.537304; delta=+0.000020 | positive direct |
| combo 01+03+04 | combo 01+03+04 | direct IDF1/HOTA/AssA=0.666295/0.527285/0.537430; delta=+0.000155 | positive direct |
| combo 01+03+04+05 | combo 01+03+04+05 | direct IDF1/HOTA/AssA=0.666316/0.527313/0.537460; delta=+0.000176 | promoted to delivery |

## Upload

- Bitbucket: `will be pushed to Novateur/vlincs_reid_by_search branch wisc; package path reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/`
- S3: `uploaded target: s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_feature_outlier_combo_p005_gain/`

## Next

Use this feature-outlier reviewer as an opponent gate for generated positives: only image-generated positives that point to one of these local outlier islands and pass DINO/SigLIP source CTF should be tried. Keep whole-component generated-positive merges killed.
