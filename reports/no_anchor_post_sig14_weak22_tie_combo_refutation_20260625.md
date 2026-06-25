# No-Anchor Post-Sig14/Weak22 Tie-Combo Refutation

Date: 2026-06-25

Status: refutation record, not an end-to-end gain.

## Hypothesis

After the current best (`0.669466` p005_area IDF1), several remaining
local-gated candidates were individually direct-score ties. A structured
combination might still matter if it performs a local component-18 reshuffle:

- add `seq8315` into component 18 via SigLIP rank 5;
- remove `seq1874` from component 18 to component 88 via weak rank 4;
- remove `seq8336,8559` from component 18 to component 77 via weak rank 8.

This is no-anchor M7/M10 reviewer evidence plus M8 local repair materialization.
GT is used only by the evaluator after materialization.

## Commands

```bash
export DATA_ROOT="$PWD/kit/demo_data/ds1/gt"
OUT=local_runs/post_sig14_weak22_tie_combo_probe_20260625_single_triple
BASE=local_runs/demo_post_combo5_sig14_weak22_verify_20260625/method_reproduction/post_combo5_sig14_weak22_assignments.csv
W=local_runs/demo_post_combo5_sig14_weak22_verify_20260625/previous_combo5/method_reproduction/local_gated_weak_primary/manifest.json
S=local_runs/demo_post_combo5_sig14_weak22_verify_20260625/previous_combo5/method_reproduction/local_gated_siglip_primary/manifest.json

python kit/compose_no_anchor_cross_manifest_repairs.py \
  --base-assignment-csv "$BASE" \
  --candidate "$S:5" \
  --candidate "$W:4" \
  --candidate "$W:8" \
  --assignment-out "$OUT/assignments/combo_sig05_weak04_weak08_assignments.csv" \
  --json "$OUT/manifests/combo_sig05_weak04_weak08_manifest.json" \
  --decision-status post_sig14_tie_combo_focus

python kit/evaluate_sample_assignments_full.py \
  --tracklet-parquet kit/demo_data/ds1/tracklets/*/tracklets.parquet \
  --assignments "$OUT/assignments/combo_sig05_weak04_weak08_assignments.csv" \
  --fallback singleton \
  --json "$OUT/direct/combo_sig05_weak04_weak08.json"
```

## Result

| Probe | Moved tracklets | Direct IDF1 | Direct HOTA | Direct AssA | Decision |
|---|---:|---:|---:|---:|---|
| current direct baseline | 0 | 0.667266 | 0.528380 | 0.538499 | reference |
| weak rank 4 only | 1 | 0.667266 | 0.528380 | 0.538499 | tie |
| siglip rank 5 + weak rank 4 + weak rank 8 | 4 | 0.667266 | 0.528380 | 0.538499 | tie, no delivery |

The focused triple moved `seq8315`, `seq1874`, `seq8336`, and `seq8559`.
It exactly tied the current direct baseline, so density_simple and p005_area
were skipped. Best remains:

`IDF1 / HOTA / AssA = 0.669466 / 0.530154 / 0.540397`.

## Notes

- A full 15-combo grid was started but killed after the first score because the
  evaluator takes about 88 seconds per direct run on this machine.
- An attempted `n_workers=8` speedup failed under macOS multiprocessing because
  heredoc scripts are seen as `<stdin>`; no result from that run was used.
- No anchors or GT labels were used to construct the assignment. GT was used
  only for the final direct metric.

## Decision

Kill this combo family for now. The remaining post-best tie candidates should
not consume delivery budget unless new generated-positive evidence changes the
reviewer score before materialization.
