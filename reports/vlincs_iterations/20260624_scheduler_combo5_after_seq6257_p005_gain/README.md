# Scheduler Combo5 After Seq6257 P005 Gain

- Date: `2026-06-24`
- Pipeline module: `M7/M10 + M8 + M12`
- Used in pipeline: `yes: root ./demo.sh now defaults to this DS1 method reproduction`
- Status: `gain`
- No-anchor: `True`

## Summary

After the seq6257 best, a direct-score reviewer tested 10 more local repair candidates and promoted a five-candidate combo: SigLIP ranks 18, 6, 9, 2 plus weak rank12. Canonical p005_area IDF1 improves from 0.669019 to 0.669311.

## Metrics

- Baseline: `0.669019`
- Candidate: `0.669311`
- Delta: `0.000292`
- Metric name: `canonical p005_area IDF1`

## Implementation

The implementation does not train on anchors or GT identities. It rebuilds the prior no-anchor best from committed feature evidence, reruns weakmetric/SigLIP local-gated candidate generation, materializes five non-overlapping scheduler repairs after seq6257, exports a fresh submission zip, and validates direct, density_simple, and p005_area. GT is used only by the evaluator after materialization. The generated-positive lesson here is scheduler/admission: keep only local feature positives that survive full-score reviewer/opponent gates; image API positives remain future work and must use environment-variable credentials plus CTF.

## Environment

- `branch wisc`
- `python with numpy pandas pyarrow scikit-learn reid-hota and Pillow`
- `git lfs materialized kit/demo_data/ds1/**`
- `DATA_ROOT defaults to kit/demo_data/ds1/gt`
- `No plaintext API keys in commands, logs, scripts, repo, or S3; image-generation experiments must read OPENAI_API_KEY from env only`

## Commands

```bash
./demo.sh
```

```bash
reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/reproduce.sh
```

```bash
python kit/compose_no_anchor_cross_manifest_repairs.py --base-assignment-csv <seq6257_assignment> --candidate <siglip_manifest>:18 --candidate <siglip_manifest>:6 --candidate <weak_manifest>:12 --candidate <siglip_manifest>:9 --candidate <siglip_manifest>:2
```

```bash
python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments <combo5_assignment> --fallback singleton
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip <direct_zip> --policies density_simple
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip <density_zip> --config "$(cat reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/repro/input/p005_area_config.txt)"
```

## Code Paths

- `kit/compose_no_anchor_cross_manifest_repairs.py`
- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/reproduce.sh`
- `demo.sh`

## Artifacts

- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/metrics/combo5_full_export.json`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/metrics/combo5_density_simple.json`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/metrics/combo5_density_p005_area.json`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/repro/provenance/post_seq6257_direct_probe_summary.json`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/repro/provenance/post_seq6257_combo_probe_summary.json`
- `reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/repro/provenance/combo5_delivery_summary.json`

## Visual Cases

- Combo5 SigLIP rank18: seq2344/2390/2549: Three component-51 tracklets move to component 29; this was the strongest individual direct-positive candidate.
  - failure: The old graph kept a small component-51 island split from a component-29 identity despite local SigLIP evidence.
  - improvement: Move only the three island tracklets and keep the repair only inside a direct-positive and p005-positive combo.
  - image: `cases/siglip_rank18_seq2344_2390_2549/rank18_bbox_evidence.png`
  - html: `cases/siglip_rank18_seq2344_2390_2549/case.html`
  - json: `cases/siglip_rank18_seq2344_2390_2549/case.json`
- Combo5 SigLIP rank06: seq7200: seq7200 moves from component 18 to component 0.
  - failure: The previous component assignment left seq7200 in a local island with weaker target evidence.
  - improvement: Single-tracklet move, admitted only after direct and combo validation.
  - image: `cases/siglip_rank06_seq7200/rank06_bbox_evidence.png`
  - html: `cases/siglip_rank06_seq7200/case.html`
  - json: `cases/siglip_rank06_seq7200/case.json`
- Combo5 Weak rank12: seq582: seq582 moves from component 2 to component 43.
  - failure: A low raw-sim but high-margin weakmetric candidate was not trusted until full-score direct evidence showed a lift.
  - improvement: Promote through combo only; do not create a broad 2->43 merge.
  - image: `cases/weak_rank12_seq582/rank12_bbox_evidence.png`
  - html: `cases/weak_rank12_seq582/case.html`
  - json: `cases/weak_rank12_seq582/case.json`
- Combo5 SigLIP rank09: seq1073: seq1073 moves from component 88 to component 14.
  - failure: The old graph left a tiny false-split island with a small direct-positive signal.
  - improvement: Keep only as part of the p005-positive combo, not as a standalone claim.
  - image: `cases/siglip_rank09_seq1073/rank09_bbox_evidence.png`
  - html: `cases/siglip_rank09_seq1073/case.html`
  - json: `cases/siglip_rank09_seq1073/case.json`
- Combo5 SigLIP rank02: seq4844: seq4844 moves from component 43 to component 36.
  - failure: Standalone direct gain was only +0.000003, below the promotion threshold.
  - improvement: Retain only because the five-repair combo remains p005-positive.
  - image: `cases/siglip_rank02_seq4844/rank02_bbox_evidence.png`
  - html: `cases/siglip_rank02_seq4844/case.html`
  - json: `cases/siglip_rank02_seq4844/case.json`
- Generated-positive scheduler seq6257: A one-tracklet MCAM05/Tc6 island moves from component 30 / gid 96000032 to component 16 / gid 96000017 after local weak-positive scheduling and SigLIP-led reviewer evidence.
  - failure: The previous graph kept seq6257 in component 30 even though local visual evidence across SigLIP, weakmetric and DINO manifests pointed to component 16.
  - improvement: Move only seq6257, then require direct, density_simple and p005_area end-to-end validation before promotion.
  - image: `cases/siglip_rank10_seq6257/rank10_bbox_evidence.png`
  - html: `cases/siglip_rank10_seq6257/case.html`
  - json: `cases/siglip_rank10_seq6257/case.json`
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
| siglip_r02_s43_to36_seq4844 | rank 2, moved seqs 4844 | direct IDF1 0.666827, HOTA 0.527837, AssA 0.537932, delta +0.000003 | tiny direct-positive; combo-only |
| weak_r02_s16_to38_seq3355_3598 | rank 2, moved seqs 3355,3598 | direct IDF1 0.666793, HOTA 0.527796, AssA 0.537901, delta -0.000031 | kill: direct-negative |
| weak_r04_s18_to88_seq1874 | rank 4, moved seqs 1874 | direct IDF1 0.666824, HOTA 0.527835, AssA 0.537933, delta +0.000000 | kill: tie |
| siglip_r06_s18_to0_seq7200 | rank 6, moved seqs 7200 | direct IDF1 0.666917, HOTA 0.527950, AssA 0.538051, delta +0.000093 | compose candidate |
| siglip_r08_s83_to30_seq1011 | rank 8, moved seqs 1011 | direct IDF1 0.666824, HOTA 0.527835, AssA 0.537933, delta +0.000000 | kill: tie |
| siglip_r09_s88_to14_seq1073 | rank 9, moved seqs 1073 | direct IDF1 0.666847, HOTA 0.527857, AssA 0.537949, delta +0.000023 | tiny direct-positive; combo-only |
| weak_r10_s8_to81_seq1236 | rank 10, moved seqs 1236 | direct IDF1 0.666824, HOTA 0.527835, AssA 0.537933, delta +0.000000 | kill: tie |
| weak_r12_s2_to43_seq582 | rank 12, moved seqs 582 | direct IDF1 0.666864, HOTA 0.527921, AssA 0.538083, delta +0.000040 | compose candidate |
| siglip_r13_s7_to37_seq4266 | rank 13, moved seqs 4266 | direct IDF1 0.666822, HOTA 0.527832, AssA 0.537930, delta -0.000002 | kill: direct-negative |
| siglip_r18_s51_to29_seq2344_2390_2549 | rank 18, moved seqs 2344,2390,2549 | direct IDF1 0.666952, HOTA 0.527982, AssA 0.538064, delta +0.000128 | compose candidate |
| combo_sig18_sig6 | local_gated_siglip_primary:18;local_gated_siglip_primary:6 | direct IDF1 0.667045, HOTA 0.528097, AssA 0.538182, delta +0.000221 | positive combo, superseded by combo5 |
| combo_sig18_sig6_weak12 | local_gated_siglip_primary:18;local_gated_siglip_primary:6;local_gated_weak_primary:12 | direct IDF1 0.667085, HOTA 0.528183, AssA 0.538333, delta +0.000261 | positive combo, superseded by combo5 |
| combo_sig18_sig6_weak12_sig9 | local_gated_siglip_primary:18;local_gated_siglip_primary:6;local_gated_weak_primary:12;local_gated_siglip_primary:9 | direct IDF1 0.667108, HOTA 0.528206, AssA 0.538349, delta +0.000284 | positive combo, superseded by combo5 |
| combo_sig18_sig6_weak12_sig9_sig2 | local_gated_siglip_primary:18;local_gated_siglip_primary:6;local_gated_weak_primary:12;local_gated_siglip_primary:9;local_gated_siglip_primary:2 | direct IDF1 0.667111, HOTA 0.528208, AssA 0.538349, delta +0.000287 | promote through delivery |
| combo5_density_simple | run density_simple after combo5 direct export | IDF1 0.669205, HOTA 0.529898, AssA 0.540120, dropped 35029 | positive intermediate |
| combo5_p005_area | canonical p005_area delivery validation | IDF1 0.669311, HOTA 0.529982, AssA 0.540246, dropped 45467, config p005_area | promote canonical gain |

## Upload

- Bitbucket: `https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/`
- S3: `s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260624_scheduler_combo5_after_seq6257_p005_gain/`

## Next

Use combo5 positives as scheduler labels and keep the killed/tie direct candidates as hard negatives. Next useful step is a learned reviewer over local-gated candidates; image-generated positives should only enter after env-key API calls plus DINO/SigLIP/weakmetric CTF, never plaintext-key commands.
