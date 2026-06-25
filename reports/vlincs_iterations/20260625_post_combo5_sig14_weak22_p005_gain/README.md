# Post-Combo5 SigLIP14 + Weak22 P005 Gain

- Date: `2026-06-25`
- Pipeline module: `M7/M10 + M8 + M12`
- Used in pipeline: `yes: root ./demo.sh defaults to this DS1 method reproduction after this package is promoted`
- Status: `gain`
- No-anchor: `True`

## Summary

After the scheduler combo5 best, the reviewer tested six compatible remaining local-gated candidates. SigLIP rank14 seq5486 and weakmetric rank22 seq7771 were both direct-positive; their two-tracklet combo improves canonical p005_area IDF1 from 0.669311 to 0.669466.

## Metrics

- Baseline: `0.669311`
- Candidate: `0.669466`
- Delta: `0.000155`
- Metric name: `canonical p005_area IDF1`

## Implementation

No anchors and no GT identities are used for proposer/reviewer evidence. The method rebuilds the prior combo5 assignment from committed no-anchor feature manifests, composes SigLIP rank14 and weakmetric rank22 with kit/compose_no_anchor_cross_manifest_repairs.py, exports a fresh submission zip, and scores direct, density_simple, and p005_area. GT is used only by the evaluator after materialization.

## Environment

- `branch wisc`
- `Python with numpy pandas pyarrow scikit-learn reid-hota and Pillow`
- `git lfs materialized kit/demo_data/ds1/**`
- `DATA_ROOT defaults to kit/demo_data/ds1/gt`
- `No plaintext API keys in commands, logs, scripts, repo, or S3; future GPT-image experiments must read OPENAI_API_KEY from env only`

## Commands

```bash
bash reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/reproduce.sh
```

```bash
./demo.sh
```

## Code Paths

- `kit/compose_no_anchor_cross_manifest_repairs.py`
- `kit/export_no_anchor_subpart_visual_case.py`
- `kit/no_anchor_pervideo_filter_selector.py`
- `kit/evaluate_submission_detection_filter.py`
- `reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/reproduce.sh`
- `demo.sh`

## Artifacts

- `reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/metrics/`
- `reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/repro/`
- `reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/cases/`
- `S3 upload attempt: aws s3 sync reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/ s3://dit-scale-up/zcai/vlincs/no_anchor_gains/20260625_post_combo5_sig14_weak22_p005_gain/ --exclude *.zip --exclude *.parquet --exclude *.npz -> Unable to locate credentials`

## Visual Cases

- SigLIP rank14 seq5486: A one-tracklet MCAM04/Tc6 island moves from component 14 / gid 96000015 to component 47 / gid 96000051.
  - failure: The old graph kept seq5486 inside a large source component even though the target component was the best local SigLIP explanation.
  - improvement: Move only seq5486 and promote only after direct+density+p005 validation.
  - image: `cases/siglip_rank14_seq5486/rank14_bbox_evidence.png`
  - html: `cases/siglip_rank14_seq5486/case.html`
  - json: `cases/siglip_rank14_seq5486/case.json`
- Weakmetric rank22 seq7771: A one-tracklet MCAM08/Tc6 island moves from component 6 / gid 96000007 to component 82 / gid 960000202.
  - failure: Standalone gain was small, so this candidate was not trusted as a standalone identity decision.
  - improvement: Use it as an interaction repair paired with SigLIP rank14; keep only because canonical p005 improves.
  - image: `cases/weak_rank22_seq7771/rank22_bbox_evidence.png`
  - html: `cases/weak_rank22_seq7771/case.html`
  - json: `cases/weak_rank22_seq7771/case.json`
- Context: combo5 SigLIP rank18 seq2344/2390/2549: Previous combo5 repaired a three-tracklet component-51 island into component 29.
  - failure: The earlier graph kept a small false-split island.
  - improvement: This gain continues the same local-island strategy after combo5.
  - image: `cases/context_siglip_rank18_seq2344_2390_2549/rank18_bbox_evidence.png`
  - html: `cases/context_siglip_rank18_seq2344_2390_2549/case.html`
  - json: `cases/context_siglip_rank18_seq2344_2390_2549/case.json`
- Context: combo5 SigLIP rank06 seq7200: Previous combo5 moved seq7200 from component 18 to component 0.
  - failure: A residual island needed a local move, not a broad component merge.
  - improvement: The current gain applies the same constrained repair rule to two new islands.
  - image: `cases/context_siglip_rank06_seq7200/rank06_bbox_evidence.png`
  - html: `cases/context_siglip_rank06_seq7200/case.html`
  - json: `cases/context_siglip_rank06_seq7200/case.json`
- Context: combo5 weak rank12 seq582: Previous combo5 moved seq582 from component 2 to component 43.
  - failure: Weakmetric evidence alone is not enough; it needed full-score validation.
  - improvement: Weak rank22 follows that same reviewer-gated path.
  - image: `cases/context_weak_rank12_seq582/rank12_bbox_evidence.png`
  - html: `cases/context_weak_rank12_seq582/case.html`
  - json: `cases/context_weak_rank12_seq582/case.json`

## Ablations

| name | change | result | decision |
|---|---|---|---|
| siglip_r12_s91_to83_seq633 | SIG rank 12 | direct IDF1 0.667111, HOTA 0.528208, AssA 0.538349, delta +0.000000 | tie; record as neutral evidence |
| siglip_r14_s14_to47_seq5486 | SIG rank 14 | direct IDF1 0.667210, HOTA 0.528331, AssA 0.538464, delta +0.000099 | direct-positive; test in combo |
| siglip_r15_s87_to91_seq991 | SIG rank 15 | direct IDF1 0.667030, HOTA 0.528140, AssA 0.538314, delta -0.000081 | killed as direct-negative |
| siglip_r19_s89_to91_seq4487_4566 | SIG rank 19 | direct IDF1 0.667111, HOTA 0.528208, AssA 0.538349, delta +0.000000 | tie; record as neutral evidence |
| weak_r20_s19_to81_seq1419_6295 | WEAK rank 20 | direct IDF1 0.666927, HOTA 0.528044, AssA 0.538247, delta -0.000184 | killed as direct-negative |
| weak_r22_s6_to82_seq7771 | WEAK rank 22 | direct IDF1 0.667167, HOTA 0.528257, AssA 0.538383, delta +0.000056 | tiny direct-positive; combo-only |
| combo_sig14_weak22_direct | compose seq5486 and seq7771 after combo5 | direct IDF1 0.667266, delta +0.000155 | send to delivery |
| combo_sig14_weak22_density_simple | fixed M12 density_simple | IDF1 0.669360, delta +0.000155 | positive intermediate |
| combo_sig14_weak22_p005_area | valid p005_area after density_simple | IDF1 0.669466, HOTA 0.530154, AssA 0.540397; config=p005_area dropped=45467 | promoted canonical e2e gain |

## Upload

- Bitbucket: `will be pushed to https://bitbucket.org/Novateur/vlincs_reid_by_search/src/wisc/reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/`
- S3: `upload blocked: local aws s3 sync failed with Unable to locate credentials; h100-test-2-codex and h100-test-3-codex SSH probes failed with stdio forwarding failed / Connection closed. Local package is complete under reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/.`

## Next

Use safe environment-variable OPENAI_API_KEY to run GPT-image positive generation only after DINO/SigLIP/weakmetric CTF gates; continue toward >0.70 with larger but reviewer-bounded candidate batches.
