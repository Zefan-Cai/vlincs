# Fresh Subpart Repair seq8367 P005 Gain

- Date: `2026-06-24`
- Pipeline module: `M8`
- Used in pipeline: `yes`
- Status: `gain`
- No-anchor: `True`

## Summary

A fresh one-tracklet subpart repair moves seq8367 from component 10 to component 77 and improves canonical p005_area IDF1 from 0.668673 to 0.668767 without anchors.

## Metrics

- Baseline: `0.668673`
- Candidate: `0.668767`
- Delta: `0.000094`
- Metric name: `canonical p005_area IDF1`

## Implementation

Starting from the method-reproduced rank06+rank07 current-best assignment, run a weakmetric-primary subpart proposer with DINOv2 and SigLIP support views. The promoted action is weak-primary rank04: move seq8367 from component 10 / gid 96000011 to component 77 / gid 96002328. The run then rebuilds the submission and validates direct, density_simple, and p005_area delivery. GT is used only by the evaluator after the assignment is materialized.

## Environment

- `branch=wisc`
- `python with numpy pandas pyarrow scikit-learn reid-hota`
- `DATA_ROOT defaults to kit/demo_data/ds1/gt`
- `features from kit/demo_data/ds1/features`
- `tracklets from kit/demo_data/ds1/tracklets`

## Commands

```bash
./demo.sh
```

```bash
reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/reproduce.sh
```

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv <generated_rank06_07_assignment> --feature-npz kit/demo_data/ds1/features/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz --view dino:kit/demo_data/ds1/features/ds1_tracklet_dinov2base_s1_20260620.npz:0.80 --view siglip:kit/demo_data/ds1/features/ds1_tracklet_siglip2_person_reid_s1_20260620.npz:0.30 --top-n 12
```

```bash
python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments <rank04_subpart_s10_to77_assignment> --fallback singleton
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip <direct_zip> --policies density_simple
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip <density_zip> --config "$(cat reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/repro/input/p005_area_config.txt)"
```

## Code Paths

- `demo.sh`
- `kit/propose_no_anchor_feature_outlier_relinks.py`
- `kit/apply_no_anchor_ranked_repairs.py`
- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/metrics/seq8367_10_to77_full_export.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/metrics/seq8367_10_to77_density_simple.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/metrics/seq8367_10_to77_density_p005_area.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/repro/expected/seq8367_10_to77_full_export.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/repro/expected/seq8367_10_to77_density_simple.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/repro/expected/seq8367_10_to77_density_p005_area.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/cases/seq8367_fresh_subpart/case.json`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/cases/seq8367_fresh_subpart/case.html`
- `reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/cases/seq8367_fresh_subpart/rank04_bbox_evidence.png`

## Visual Cases

- seq8367 fresh subpart island: One MCAM08 tracklet moves from gid 96000011 to gid 96002328 after weakmetric+DINO+SigLIP evidence ranks target component 77 above the source rest.
  - failure: The previous current-best graph left seq8367 inside a large component 10 even though it had conflicts to the rest and a strong target margin to a tiny component 77.
  - improvement: The fresh subpart stage keeps the repair local: only seq8367 is moved, then the full delivery path confirms the global ID decision.
  - image: `cases/seq8367_fresh_subpart/rank04_bbox_evidence.png`
  - html: `cases/seq8367_fresh_subpart/case.html`
  - json: `cases/seq8367_fresh_subpart/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| weak rank04 10->77 | move seq8367 from component 10 / gid 96000011 to component 77 / gid 96002328 | direct 0.666572; density_simple 0.668661; p005_area 0.668767 | promote |
| siglip rank09 10->18 | move seq8367 from component 10 / gid 96000011 to component 18 / gid 96000019 | direct 0.666572; density_simple 0.668661; p005_area 0.668767 | tie; keep weak rank04 because reviewer score and proposer rank are stronger |
| residual rank06+rank07 baseline | previous current-best two-tracklet residual feature-outlier combo | direct 0.666479; density_simple 0.668568; p005_area 0.668673 | baseline for this gain |
| generated-positive budget reviewer | review 24 residual feature-outlier candidates before spending GPT-image budget | 0/24 approved for generation; two strict candidates were known direct negatives | refute image generation for that stale family |

## Upload

- Bitbucket: `https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/`
- S3: `s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_fresh_subpart_seq8367_p005_gain/`

## Next

Use the fresh subpart reviewer as the new local gate for generated positives: only spend image-generation budget when a candidate clears multi-view target margin and is not a stale direct-negative family. The e2e target is still 0.70, so continue searching for larger independent islands rather than stacking many tiny noisy moves.
