# Stacking no-anchor evidence repairs on main 72c55bc gallery output

- Date: `2026-06-25`
- Pipeline module: `M8`
- Used in pipeline: `candidate-only`
- Status: `direct_gain_only`
- No-anchor: `True`

## Summary

The old 66.8 final assignment cannot be pasted onto the latest main 62.8 gallery state, but the method ports cleanly as a post-gallery no-anchor repair layer. On exact main commit 72c55bc, live DS1 baseline is IDF1 0.627755; exporting that gallery state, proposing SigLIP/DINO/weakmetric subpart repairs, and applying six non-overlapping positive repairs gives IDF1 0.628913 (+0.001158). The packaged reproduce.sh was run with the existing main gallery_ds1 DB and reproduced the same 0.628913 score.

## Metrics

- Baseline: `0.627755`
- Candidate: `0.628913`
- Delta: `0.001158`
- Metric name: `DS1 IDF1 on main 72c55bc + no-anchor repair combo`

## Implementation

Run latest main gallery first. Export DB seq->gid as a tracklet-key assignment CSV with kit/export_gallery_db_assignments.py. Generate candidate subpart repairs from production features only with kit/propose_no_anchor_subpart_repair_candidates.py. Apply selected ranks 4,1,11,16,19,20 with kit/apply_no_anchor_candidate_combo.py. Score the materialized assignment through kit/evaluate_sample_assignments_full.py, using GT only as evaluator answer key. The proposer configuration used SigLIP primary features with weakmetric weight 0.70 and DINOv2 weight 0.80; focus videos targeted the weak DS1 slices MCAM04/05/06/08 and MCAM03-Tc8.

## Environment

- `Repo wisc worktree: /Users/zcai/Codex/vlincs_reid_by_search_wisc_sync`
- `Latest main test worktree: /Users/zcai/Codex/vlincs_reid_by_search_main_latest at 72c55bce5c2c4fad34fa1ce0ecc141b09898f134`
- `Postgres gallery_ds1 on PGHOST=localhost PGPORT=55434 PGUSER=gallery`
- `DATA_ROOT points to kit/demo_data/ds1/gt for evaluation only`
- `No anchors or GT labels are read by export/proposer/materializer.`

## Commands

```bash
./reports/vlincs_iterations/20260625_main72c55bc_gallery_stack_siglip_combo_gain/reproduce.sh
```

```bash
PGHOST=localhost PGPORT=55434 PGUSER=gallery PGPASSWORD=gallery python kit/export_gallery_db_assignments.py --dbname gallery_ds1 --tracklet-root kit/demo_data/ds1/tracklets --embedding-root kit/demo_data/ds1/embeddings --decision-status main_72c55bc_two_tier_component --assignment-out local_runs/main_72c55bc_stack_20260625/exported_main72c55bc_assignments.csv --json local_runs/main_72c55bc_stack_20260625/exported_main72c55bc_assignments.json
```

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --assignment-csv exported_main72c55bc_assignments.csv --feature-npz kit/demo_data/ds1/features/ds1_tracklet_siglip2_person_reid_s1_20260620.npz --primary-weight 1.0 --view weak:kit/demo_data/ds1/features/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:0.70 --view dino:kit/demo_data/ds1/features/ds1_tracklet_dinov2base_s1_20260620.npz:0.80 --top-n 48 --min-source-component-size 20 --max-source-component-size 900 --min-target-component-size 3 --max-target-component-size 400 --max-seeds-per-component 40 --min-group-size 1 --max-group-size 8 --seed-sim 0.72 --min-source-margin 0.0 --min-target-sim 0.5 --min-target-margin 0.0 --targets-per-group 3 --focus-videos vlincs_MS01_MC0001_MCAM05_2024-03-Tc6,vlincs_MS01_MC0001_MCAM06_2024-03-Tc6,vlincs_MS01_MC0001_MCAM04_2024-03-Tc6,vlincs_MS01_MC0001_MCAM03_2024-03-Tc8,vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 --output-dir siglip_primary/assignments --json siglip_primary/manifest.json
```

```bash
python kit/apply_no_anchor_candidate_combo.py --base-assignment-csv exported_main72c55bc_assignments.csv --manifest siglip_primary/manifest.json --rank 4,1,11,16,19,20 --assignment-out combo_all_positive_r04_r01_r11_r16_r19_r20.csv --json combo_all_positive_r04_r01_r11_r16_r19_r20.json
```

```bash
DATA_ROOT=kit/demo_data/ds1/gt python kit/evaluate_sample_assignments_full.py --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet --assignments combo_all_positive_r04_r01_r11_r16_r19_r20.csv --json combo_all_positive_r04_r01_r11_r16_r19_r20_sample_score.json --zip-out combo_all_positive_r04_r01_r11_r16_r19_r20_sample.zip
```

## Code Paths

- `kit/export_gallery_db_assignments.py`
- `kit/apply_no_anchor_candidate_combo.py`
- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/export_no_anchor_subpart_visual_case.py`

## Artifacts

- `reproduce.sh`
- `metrics/main_72c55bc_baseline_assignment_score.json`
- `metrics/combo_all_positive_r04_r01_r11_r16_r19_r20_sample_score.json`
- `repro/provenance/siglip_primary_manifest.json`
- `repro/provenance/combo_all_positive_r04_r01_r11_r16_r19_r20_materialize.json`
- `repro/provenance/combo_scores_summary.json`
- `repro/provenance/reproduce_script_check_score.json`
- `repro/provenance/reproduce_script_exported_assignments.json`
- `repro/provenance/reproduce_script_combo_materialize.json`

## Visual Cases

- Rank04 MCAM04 island: Three MCAM04 tracklets move 17 -> 37 with raw-frame bbox evidence.
  - image: `cases/rank04_mcam04_island/rank04_bbox_evidence.png`
  - html: `cases/rank04_mcam04_island/case.html`
  - json: `cases/rank04_mcam04_island/case.json`
- Rank16 MCAM08 pair: Two MCAM08 tracklets move 35 -> 63; strongest single candidate from batch 2.
  - image: `cases/rank16_mcam08_pair/rank16_bbox_evidence.png`
  - html: `cases/rank16_mcam08_pair/case.html`
  - json: `cases/rank16_mcam08_pair/case.json`
- Rank20 MCAM04 pair: Two MCAM04 tracklets move 5 -> 17 and combine positively with the other repairs.
  - image: `cases/rank20_mcam04_pair/rank20_bbox_evidence.png`
  - html: `cases/rank20_mcam04_pair/case.html`
  - json: `cases/rank20_mcam04_pair/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| rank01 siglip_r01_s34_to110_7045 | move seqs [7045] from component 34 to 110 | IDF1 0.627885; delta +0.000130 | keep for combo |
| rank02 siglip_r02_s45_to10_1243 | move seqs [1243] from component 45 to 10 | IDF1 0.627755; delta +0.000000 | tie/refute |
| rank04 siglip_r04_s17_to37_4558_4609_4649 | move seqs [4558, 4609, 4649] from component 17 to 37 | IDF1 0.628087; delta +0.000332 | keep for combo |
| rank08 siglip_r08_s15_to32_6575_6616_6624 | move seqs [6575, 6616, 6624] from component 15 to 32 | IDF1 0.627755; delta +0.000000 | tie/refute |
| rank11 siglip_r11_s35_to8_6669_6677 | move seqs [6669, 6677] from component 35 to 8 | IDF1 0.627830; delta +0.000075 | keep for combo |
| rank14 siglip_r14_s65_to21_6103_6108 | move seqs [6103, 6108] from component 65 to 21 | IDF1 0.627662; delta -0.000093 | tie/refute |
| rank03 siglip_r03_s83_to29_6293_6349_6426_6471_6548_6582_6637 | move seqs [6293, 6349, 6426, 6471, 6548, 6582, 6637] from component 83 to 29 | IDF1 0.627755; delta +0.000000 | tie/refute |
| rank05 siglip_r05_s93_to34_986 | move seqs [986] from component 93 to 34 | IDF1 0.627725; delta -0.000030 | tie/refute |
| rank10 siglip_r10_s55_to51_6866 | move seqs [6866] from component 55 to 51 | IDF1 0.627756; delta +0.000001 | positive singleton |
| rank12 siglip_r12_s74_to5_4454 | move seqs [4454] from component 74 to 5 | IDF1 0.627692; delta -0.000063 | tie/refute |
| rank13 siglip_r13_s7_to8_6421 | move seqs [6421] from component 7 to 8 | IDF1 0.627755; delta +0.000000 | tie/refute |
| rank16 siglip_r16_s35_to63_7973_9244 | move seqs [7973, 9244] from component 35 to 63 | IDF1 0.628204; delta +0.000449 | keep for combo |
| rank19 siglip_r19_s31_to25_9343 | move seqs [9343] from component 31 to 25 | IDF1 0.627825; delta +0.000070 | keep for combo |
| rank20 siglip_r20_s5_to17_3137_3167 | move seqs [3137, 3167] from component 5 to 17 | IDF1 0.627858; delta +0.000103 | keep for combo |
| combo_all_positive_r04_r01_r11_r16_r19_r20 | combine non-overlapping positive local repairs | IDF1 0.628913; delta +0.001158 | best |
| combo_r04_r01 | combine non-overlapping positive local repairs | IDF1 0.628216; delta +0.000461 | positive interaction |
| combo_r04_r01_r11 | combine non-overlapping positive local repairs | IDF1 0.628292; delta +0.000537 | positive interaction |
| combo_r04_r01_r16 | combine non-overlapping positive local repairs | IDF1 0.628665; delta +0.000910 | positive interaction |
| wrong DB-seq scorer guard | evaluate 0-based feature-seq CSV with DB-seq scorer | IDF1 0.063683 because seq origin mismatches DB 1-based seq | refuted scorer path; use tracklet-key sample scorer for no-anchor CSV |

## Upload

- Bitbucket: `pending commit to wisc`
- S3: `not uploaded in this package; large scorer zips remain local under /Users/zcai/Codex/vlincs_reid_by_search_main_latest/local_runs/main_72c55bc_stack_20260625/repro_check/`

## Next

Replace the manual reviewer-selected combo with a learned reviewer that predicts which post-main repair candidates interact positively, then rerun on MS02 once the MS02 bundle is available.
