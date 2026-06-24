# Local-Gated Subpart Combo P005 Gain

- Date: `2026-06-24`
- Pipeline module: `M8`
- Used in pipeline: `yes`
- Status: `gain`
- No-anchor: `True`

## Summary

A no-anchor local-gated subpart rescan on top of the seq8367 best composes three tiny repairs: seq1199 89->87, seq4690 86->37, and seq5716 9->26. Canonical p005_area IDF1 improves from 0.668767 to 0.668974.

## Metrics

- Baseline: `0.668767`
- Candidate: `0.668974`
- Delta: `0.000207`
- Metric name: `canonical p005_area IDF1`

## Implementation

Starting from the fully reproduced seq8367 assignment, rerun weakmetric-primary and SigLIP-primary subpart proposers with DINO/SigLIP/weak support views. A new cross-manifest composer materializes exactly three local repairs from production evidence only. The assignment is then evaluated through direct full export, density_simple, and p005_area. GT is used only by the evaluator after materialization.

## Environment

- `branch wisc`
- `python with numpy pandas pyarrow scikit-learn reid-hota`
- `git lfs materialized kit/demo_data/ds1/**`
- `DATA_ROOT defaults to kit/demo_data/ds1/gt`

## Commands

```bash
./demo.sh
```

```bash
reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/reproduce.sh
```

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv <seq8367_assignment> --feature-npz <weak_or_siglip_feature> --view ...
```

```bash
python kit/compose_no_anchor_cross_manifest_repairs.py --base-assignment-csv <seq8367_assignment> --candidate <siglip_manifest>:1 --candidate <weak_manifest>:3 --candidate <weak_manifest>:1
```

```bash
python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments <combo_assignment> --fallback singleton
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip <direct_zip> --policies density_simple
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip <density_zip> --config "$(cat reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/repro/input/p005_area_config.txt)"
```

## Code Paths

- `kit/compose_no_anchor_cross_manifest_repairs.py`
- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/metrics/combo_top3_full_export.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/metrics/combo_top3_density_simple.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/metrics/combo_top3_density_p005_area.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/repro/provenance/direct_score_summary.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/repro/provenance/single_delivery_summary.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/repro/provenance/combo_delivery_summary.json`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/cases/siglip_rank01_seq1199/rank01_bbox_evidence.png`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/cases/weak_rank03_seq4690/rank03_bbox_evidence.png`
- `reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/cases/weak_rank01_seq5716/rank01_bbox_evidence.png`

## Visual Cases

- SigLIP local-gated island seq1199: A one-tracklet MCAM03/Tc6 island moves from component 89 / gid 960000481 to component 87 / gid 960000351. This is the strongest single p005 repair.
  - failure: The previous graph left seq1199 as a residual island in component 89 even though local SigLIP evidence preferred component 87.
  - improvement: Move only seq1199, then require direct+density+p005 validation before using it.
  - image: `cases/siglip_rank01_seq1199/rank01_bbox_evidence.png`
  - html: `cases/siglip_rank01_seq1199/case.html`
  - json: `cases/siglip_rank01_seq1199/case.json`
- Weakmetric local-gated island seq4690: A one-tracklet MCAM04/Tc6 island moves from component 86 / gid 960000350 to component 37 / gid 96000040.
  - failure: A broad 86->37 merge is risky, but this individual island has local support and direct/p005 validation.
  - improvement: Move only seq4690; avoid broad component merge.
  - image: `cases/weak_rank03_seq4690/rank03_bbox_evidence.png`
  - html: `cases/weak_rank03_seq4690/case.html`
  - json: `cases/weak_rank03_seq4690/case.json`
- Weakmetric local-gated island seq5716: A one-tracklet MCAM04/Tc6 island moves from component 9 / gid 96000010 to component 26 / gid 96000028.
  - failure: The previous graph kept seq5716 in a large source component despite local evidence for component 26.
  - improvement: Move only seq5716 and keep it only because it improves the top3 combo.
  - image: `cases/weak_rank01_seq5716/rank01_bbox_evidence.png`
  - html: `cases/weak_rank01_seq5716/case.html`
  - json: `cases/weak_rank01_seq5716/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| siglip_rank01_s89_to87_seq1199 | rank01_subpart_s89_to87_1seq_assignments.csv | direct 0.666656; density 0.668745; p005 0.668851 | positive single, compose |
| weak_rank03_s86_to37_seq4690 | rank03_subpart_s86_to37_1seq_assignments.csv | direct 0.666651; density 0.668741; p005 0.668846 | positive single, compose |
| weak_rank01_s9_to26_seq5716 | rank01_subpart_s9_to26_1seq_assignments.csv | direct 0.666616; density 0.668704; p005 0.668810 | positive single, compose |
| combo_top3_siglip1_weak3_weak1 | combo_top3_siglip1_weak3_weak1_assignments.csv | direct 0.666778; density 0.668868; p005 0.668974 | promote |
| combo_top2_siglip1_weak3 | combo_top2_siglip1_weak3_assignments.csv | direct 0.666735; density 0.668825; p005 0.668930 | positive but lower than top3 |
| siglip_rank02_s43_to36_seq4844 | rank02_subpart_s43_to36_1seq_assignments.csv | direct 0.666575; delta +0.000003 | direct-only negative or too small |
| weak_rank02_s16_to38_seq3355_3598 | rank02_subpart_s16_to38_2seq_assignments.csv | direct 0.666541; delta -0.000031 | direct-only negative or too small |
| dino_rank01_s24_to88_seq1913_1924 | rank01_subpart_s24_to88_2seq_assignments.csv | direct 0.666345; delta -0.000227 | direct-only negative or too small |

## Upload

- Bitbucket: `https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260624_local_gated_subpart_combo_p005_gain/`
- S3: `s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_local_gated_subpart_combo_p005_gain/`

## Next

Use the positive/negative labels from this local-gated scan as the next generated-positive scheduler: only spend GPT-image calls on candidates that clear DINO/SigLIP local evidence and are not known direct negatives.
