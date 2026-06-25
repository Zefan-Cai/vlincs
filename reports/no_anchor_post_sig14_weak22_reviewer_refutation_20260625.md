# No-Anchor Post-Sig14+Weak22 Reviewer Refutation

Date: 2026-06-25

Status: no gain; direct-only reviewer batch killed.

## Hypothesis

After the current best `siglip_r14 + weak_r22` gain, some compatible remaining
local-gated SigLIP/weakmetric candidates may still provide independent local
island repairs.

## Baseline

Current direct baseline from the promoted assignment:

| Metric | Value |
|---|---:|
| IDF1 | 0.667266 |
| HOTA | 0.528380 |
| AssA | 0.538499 |

Current canonical p005 best remains:

| Metric | Value |
|---|---:|
| IDF1 | 0.669466 |
| HOTA | 0.530154 |
| AssA | 0.540397 |

## Method

- Base assignment:
  `local_runs/demo_post_combo5_sig14_weak22_verify_20260625/method_reproduction/post_combo5_sig14_weak22_assignments.csv`
- Candidate manifests:
  - `local_runs/demo_post_combo5_sig14_weak22_verify_20260625/previous_combo5/method_reproduction/local_gated_siglip_primary/manifest.json`
  - `local_runs/demo_post_combo5_sig14_weak22_verify_20260625/previous_combo5/method_reproduction/local_gated_weak_primary/manifest.json`
- Materializer:
  `kit/compose_no_anchor_cross_manifest_repairs.py`
- Scorer:
  `kit/evaluate_sample_assignments_full.py`

No anchors or GT identities were used to propose or materialize candidates. GT
was used only by the evaluator after each candidate assignment was materialized.

## Direct Results

| Candidate | Move | Direct IDF1 | Delta vs current direct | Decision |
|---|---|---:|---:|---|
| weak r02 | `16 -> 38`, seq3355/3598 | 0.667236 | -0.000030 | killed |
| weak r04 | `18 -> 88`, seq1874 | 0.667266 | 0.000000 | tie |
| weak r07 | `22 -> 13`, seq9565 | 0.667263 | -0.000003 | killed |
| siglip r05 | `1 -> 18`, seq8315 | 0.667266 | 0.000000 | tie |
| weak r08 | `18 -> 77`, seq8336/8559 | 0.667266 | 0.000000 | tie |
| siglip r08 | `83 -> 30`, seq1011 | 0.667266 | 0.000000 | tie |

## Decision

No candidate cleared the direct-score gate. The four ties and two negatives are
not promoted and were not sent to `density_simple + p005_area`.

This keeps the pipeline simple: after the current best, these remaining local
candidate families are mostly exhausted unless a stronger evidence source
appears, such as CTF-passed GPT-image positives.

## Next

Focus on generated-positive evidence:

1. run GPT-image only from a safe `OPENAI_API_KEY` environment variable;
2. require DINOv2 source CTF and SigLIP source CTF;
3. use surviving generated positives only as weak evidence for new M7/M8
   repair proposals;
4. run direct full-score before any p005 delivery.
