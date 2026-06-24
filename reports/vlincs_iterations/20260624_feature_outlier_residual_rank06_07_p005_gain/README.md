# Residual Feature-Outlier Combo Rank06+Rank07 P005 Gain

- Date: `2026-06-24`
- Pipeline module: `M8`
- Used in pipeline: `yes`
- Status: `gain`
- No-anchor: `True`

## Summary

A two-tracklet residual feature-outlier combo improves canonical p005_area IDF1 from 0.668508 to 0.668673 (+0.000165) without anchors.

## Metrics

- Baseline: `0.668508`
- Candidate: `0.668673`
- Delta: `0.000165`
- Metric name: `canonical p005_area IDF1`

## Implementation

Starting from the current best assignment, compare each residual candidate assignment against the same base, extract changed seq rows, reject conflicts, compose rank06 and rank07, then run direct full-score followed by density_simple and p005_area. Evidence uses only current assignment state and cached weakmetric/OSNet, DINOv2, and SigLIP tracklet features; GT appears only in the evaluator.

## Environment

- `repo=<fresh clone of Novateur/vlincs_reid_by_search on wisc>`
- `python=python3 or .venv-demo/bin/python`
- `DATA_ROOT=kit/demo_data/ds1/gt` by default
- `no anchors; GT used only by full-score/delivery evaluators`

`DATA_ROOT` contains the 10 dense DS1 GT parquet files. They are not
embeddings and not training labels for the identity resolver. They are the
evaluation answer key used by `reid_hota` to compare the generated submission
zip against ground truth and compute IDF1/HOTA/AssA. Without these files the
demo can still build a submission, but it cannot reproduce or verify the
reported 0.668673 score.

Fresh clones must materialize both tracklet parquet files and GT parquet files
through Git LFS:

```bash
git lfs pull --include="kit/demo_data/ds1/**"
```

## Commands

```bash
python kit/compose_no_anchor_assignment_overlays.py --base-assignment-csv reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/combo_01_03_04_05_feature_outlier_assignments.csv --candidate-assignment-csv local_runs/feature_outlier_after_combo_probe_20260624/assignments/rank06_after_combo_feature_outlier_assignments.csv --candidate-assignment-csv local_runs/feature_outlier_after_combo_probe_20260624/assignments/rank07_after_combo_feature_outlier_assignments.csv --assignment-out local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_after_combo_feature_outlier_assignments.csv --json local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_after_combo_feature_outlier_manifest.json
```

```bash
python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_after_combo_feature_outlier_assignments.csv --fallback singleton --json local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_full_export.json --zip-out local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_full_export.zip
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_full_export.zip --policies density_simple --json local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_density_simple.json --zip-out local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_density_simple.zip
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_density_simple.zip --config "$(cat reports/vlincs_iterations/20260624_feature_outlier_combo_p005_gain/repro/input/p005_area_config.txt)" --json local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_density_p005_area.json --zip-out local_runs/feature_outlier_after_combo_probe_20260624/combos/rank06_07_density_p005_area.zip
```

```bash
./demo.sh
```

## Code Paths

- `kit/compose_no_anchor_assignment_overlays.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/input/rank06_07_residual_feature_outlier_assignments.csv`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/expected/rank06_07_full_export.json`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/expected/rank06_07_density_simple.json`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/expected/rank06_07_density_p005_area.json`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/provenance/rank06_07_case_manifest.json`
- `reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/repro/provenance/after_combo_overlay_summary.csv`

## Visual Cases

- Residual relink seq1548: component 86 -> 91: Three sampled bbox frames show the residual MCAM03/Tc6 tracklet moved after the previous combo had already moved seq685 into component 91.
  - failure: The previous best left seq1548 in component 86 even though its multi-feature evidence preferred component 91, creating a residual false split beside an already accepted neighbor.
  - improvement: The new combo moves only seq1548 and pairs it with one independent MCAM04/Tc6 outlier; p005 validates the combined identity decision.
  - image: `cases/rank06_residual_feature_outlier/rank06_bbox_evidence.png`
  - html: `cases/rank06_residual_feature_outlier/case.html`
  - json: `cases/rank06_residual_feature_outlier/case.json`
- Residual relink seq4043: component 86 -> 37: Three sampled bbox frames show the MCAM04/Tc6 one-tracklet island moved from component 86 to component 37.
  - failure: The previous best kept seq4043 in component 86, but feature-outlier evidence and source/target margins favored component 37.
  - improvement: The reviewer keeps the repair local, avoids broad component merge, and promotes it only when composed with rank06 through direct, density_simple, and p005_area gates.
  - image: `cases/rank07_residual_feature_outlier/rank07_bbox_evidence.png`
  - html: `cases/rank07_residual_feature_outlier/case.html`
  - json: `cases/rank07_residual_feature_outlier/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| rank06 86->91 | move seqs [1548] after previous four-tracklet combo | direct IDF1 0.666383 (+0.000067) | compose |
| rank07 86->37 | move seqs [4043] after previous four-tracklet combo | direct IDF1 0.666411 (+0.000095) | compose |
| rank08 87->91 | move seqs [1123] after previous four-tracklet combo | direct IDF1 0.666324 (+0.000008) | kill tiny positive |
| rank09 3->29 | move seqs [1498] after previous four-tracklet combo | direct IDF1 0.666334 (+0.000018) | kill tiny positive |
| rank10 83->34 | move seqs [7659] after previous four-tracklet combo | direct IDF1 0.666318 (+0.000002) | kill tiny positive |
| rank11 24->85 | move seqs [5824] after previous four-tracklet combo | direct IDF1 0.666331 (+0.000015) | kill tiny positive |
| rank12 51->31 | move seqs [2344] after previous four-tracklet combo | direct IDF1 0.666305 (-0.000011) | kill negative |
| rank06+rank07 combo | compose seq1548 86->91 and seq4043 86->37; two independent one-tracklet residual islands | direct 0.666479; density_simple 0.668568; p005_area 0.668673 | promote canonical p005 gain |

## Upload

- Bitbucket: `https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/`
- S3: `s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_feature_outlier_residual_rank06_07_p005_gain/`

## Next

Use this residual reviewer as the gate for generated positives: a generated/synthetic positive must reinforce a local outlier island, pass source-specific DINO/SigLIP CTF, and clear direct plus p005 delivery before entering best.
