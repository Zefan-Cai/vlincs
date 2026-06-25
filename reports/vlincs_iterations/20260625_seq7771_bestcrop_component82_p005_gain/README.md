# Seq7771 Best-Crop Component-82 P005 Gain

- Date: `2026-06-25`
- Pipeline module: `M2/M3 + M7/M10 + M8 + M12`
- Used in pipeline: `candidate-only now; eligible to become next default after reproduce.sh verification and demo.sh integration`
- Status: `gain`
- No-anchor: `True`

## Summary

A seq7771 best-crop CTF pass makes component 82 a stable generated-positive target. The focused SigLIP scheduler proposes rank17, moving seq3403 from comp11/gid96000012 to comp82/gid960000202. Canonical p005_area IDF1 improves from 0.669466 to 0.669494.

## Metrics

- Baseline: `0.669466`
- Candidate: `0.669494`
- Delta: `0.000028`
- Metric name: `canonical p005_area IDF1`

## Implementation

No anchors or GT labels are used as identity evidence. The method starts from the previous no-anchor best assignment, rebuilds a MCAM08-focused SigLIP/weak/DINO subpart candidate manifest from tracklet features, materializes SigLIP rank17 as a one-tracklet repair, exports a fresh submission zip, runs direct full-score, then validates density_simple and p005_area. GT is used only by the evaluator after materialization.

## Environment

- `branch wisc`
- `Python with numpy pandas pyarrow reid-hota scikit-learn Pillow`
- `git lfs materialized kit/demo_data/ds1/**`
- `DATA_ROOT defaults to kit/demo_data/ds1/gt`
- `OPENAI_API_KEY not required for this promoted gain and never stored in commands/files`

## Commands

```bash
reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/reproduce.sh
```

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --focus-videos vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 ...
```

```bash
python kit/evaluate_sample_assignments_full.py --assignments <rank17 assignment> ...
```

```bash
python kit/no_anchor_pervideo_filter_selector.py --source-zip <direct zip> --policies density_simple ...
```

```bash
python kit/evaluate_submission_detection_filter.py --submission-zip <density zip> --config "$(cat repro/input/p005_area_config.txt)" ...
```

## Code Paths

- `kit/build_no_anchor_generated_positive_prompt_assets.py`
- `kit/generate_no_anchor_tracklet_local_positives.py`
- `kit/audit_no_anchor_generated_positive_ctf.py`
- `kit/propose_no_anchor_subpart_repair_candidates.py`
- `kit/evaluate_sample_assignments_full.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/reproduce.sh`

## Artifacts

- `reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/metrics/siglip_r17_s11_to82_seq3403_full_export.json`
- `reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/metrics/siglip_r17_s11_to82_seq3403_density_simple.json`
- `reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/metrics/siglip_r17_s11_to82_seq3403_density_p005_area.json`
- `reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/repro/provenance/direct_score_summary.json`
- `local_runs/generated_positive_bestcrop_scheduler_seq7771_20260625/`

## Visual Cases

- Seq7771 best-crop target-82 repair: seq3403: SigLIP rank17 moves seq3403 from comp11/gid96000012 to comp82/gid960000202 after component 82 is admitted as a DINO/SigLIP-stable generated-positive target.
  - image: `cases/siglip_rank17_seq3403/rank17_bbox_evidence.png`
  - html: `cases/siglip_rank17_seq3403/case.html`
  - json: `cases/siglip_rank17_seq3403/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| siglip_r17_s11_to82_seq3403 | direct full-score candidate from MCAM08 component-82 focused scheduler | IDF1 0.667294, HOTA 0.528414, AssA 0.538540, delta +0.000028 | promote to delivery |
| dino_r07_s82_to11_seq5382 | direct full-score candidate from MCAM08 component-82 focused scheduler | IDF1 0.667264, HOTA 0.528377, AssA 0.538494, delta -0.000002 | kill: direct tie/negative |
| dino_r08_s11_to82_seq4488 | direct full-score candidate from MCAM08 component-82 focused scheduler | IDF1 0.667264, HOTA 0.528374, AssA 0.538491, delta -0.000002 | kill: direct tie/negative |
| weak_r10_s82_to11_seq5382 | direct full-score candidate from MCAM08 component-82 focused scheduler | IDF1 0.667264, HOTA 0.528377, AssA 0.538494, delta -0.000002 | kill: direct tie/negative |
| siglip_r17_density_simple | run no-GT density_simple delivery on direct-positive zip | IDF1 0.669388, HOTA 0.530104, AssA 0.540312, dropped 35029 | continue to p005 because direct was positive |
| siglip_r17_p005_area | run canonical p005_area after density_simple | IDF1 0.669494, HOTA 0.530188, AssA 0.540439, dropped 45467, config p005_area | promote: canonical e2e gain over 0.669466 |
| seq7771_multicrop_vs_bestcrop_ctf | generated-positive admission ablation before scheduler | seq7771 DINOv2 source CTF improves 1/3 -> 3/3; SigLIP stays 3/3 | keep best-crop admission for component-82 target evidence |
| seq5486_bestcrop_ctf | same best-crop rule on seq5486 | DINOv2 remains 0/3; SigLIP 3/3 | reject for GPT-image and downstream scheduler |

## Upload

- Bitbucket: `https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260625_seq7771_bestcrop_component82_p005_gain/`
- S3: `pending: local AWS credentials still unavailable in this session; package records local artifact paths`

## Next

Integrate this gain into root demo.sh only after reproduce.sh verifies from a fresh method run; then search for additional component-82 target repairs and GPT-image only from safe OPENAI_API_KEY env.
