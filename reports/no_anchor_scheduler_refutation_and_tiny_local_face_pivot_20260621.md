# No-Anchor Scheduler Refutation And Tiny-Local Face Pivot

Date: 2026-06-21

## Status

The no-anchor pair/global-ID model target is still satisfied, but end-to-end is not:

| metric | value | target |
|---|---:|---:|
| pair F1 / P / R | 0.775234 / 0.820504 / 0.734698 | >= 0.70 each |
| best end-to-end IDF1 / HOTA / AssA | 0.655817 / 0.519228 / 0.534791 | IDF1 >= 0.70 |

This iteration used the Deli AutoResearch self-play pattern: propose, full-score, write the failure down as a critic label, then pivot structure instead of lowering thresholds.

## Broad scheduler result

The global candidate pool is exhausted under the current no-GT critic:

- broad scheduler top6 all full-scored below best; best of that batch was `0.653183`.
- global scorefloor selected 2 candidates; full scores were `0.653062` and `0.654370`.
- after adding those hard negatives, the global pool had `3357` raw rows but `0` eligible rows above scorefloor.

Interpretation: the old pool is not short on rows. It is short on structurally new evidence.

## Tiny-local top1 refutation

Tiny-local proposer: one source island, source size 2-8, one reassignment, strict no-GT visual/temporal/cannot-link admission.

| field | value |
|---|---|
| family | `conflict_subcluster_reassign_candidate_search:component:21->0` |
| source seqs | `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553` |
| target top seqs | `2656, 2591, 5838` |
| target component size | 199 |
| target best / mean sim | 0.947318 / 0.940898 |
| target view vote | 1.000 |
| committee full proxy | 0.650515 |

Full-score result:

| variant | IDF1 | HOTA | AssA | DetRe | DetPr |
|---|---:|---:|---:|---:|---:|
| raw tiny-local top1 | 0.654461 | 0.518445 | 0.533845 | 0.576334 | 0.757090 |
| density_oracle_lite | 0.654603 | 0.518358 | 0.533889 | 0.573928 | 0.761668 | 34434 |
| density_simple | 0.654608 | 0.518364 | 0.533889 | 0.573998 | 0.761558 | 33685 |
| confidence_tail | 0.654504 | 0.518219 | 0.533745 | 0.573655 | 0.761881 | 36385 |

Verdict: hard negative. Even density/post-filter peaks at `0.654608`, below current best `0.655817`.

Artifacts:

- `local_runs/no_anchor_tiny_local_top1_refutation_20260621.json`
- `local_runs/no_anchor_tiny_local_top1_refutation_20260621.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_tiny_local_top1_density_fullscore_20260621/raw_full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_tiny_local_top1_density_fullscore_20260621/density_filter.json`
- Remote: `/mnt/localssd/vlincs_reid_runs/no_anchor_tiny_local_top1_density_fullscore_20260621/`

## Face evidence audit

FaceNet is useful as auxiliary evidence, not as a standalone accept/reject rule:

| measure | value |
|---|---:|
| failed tiny-local face pair count | 24 |
| failed tiny-local face max sim | 0.766124 |
| failed tiny-local face mean sim | 0.584382 |
| source face-valid / target face-valid | 8 / 3 |

Weak no-GT calibration:

| FaceNet threshold | weak-positive rate | cannot-link hard-negative rate |
|---:|---:|---:|
| 0.50 | 0.3053 | 0.0863 |
| 0.60 | 0.2346 | 0.0466 |
| 0.65 | 0.2042 | 0.0323 |
| 0.70 | 0.1706 | 0.0220 |
| 0.75 | 0.1373 | 0.0146 |
| 0.80 | 0.1028 | 0.0083 |

The failed candidate has max face sim `0.766`, which is above many positives but also near the cannot-link p99 tail (`0.783879`). So a pure face threshold would either reject too much true evidence or admit some false merges. Face should become a calibrated logLR/veto feature combined with temporal and component-purity evidence.

## Louvain history gate

Old Louvain full-score history does not hide a better end-to-end configuration. The best recovered old full row is below current best: `0.652398`.

Artifact:

- `local_runs/remote_h100_test_3_20260621/no_anchor_louvain_history_full_gate_20260621.json`

## Next research move

Do not relax scheduler thresholds. Generate a structurally different proposer:

1. Build an identity-component evidence card rather than a single reassignment: source island, target component, nearest countertargets, FaceNet logLR, DINO/color/pose votes, time-gap feasibility, same-frame cannot-link density, and expected delivery-side delta.
2. Use face only after calibration. Treat face max around `0.75-0.80` as weak support unless temporal/countertarget evidence agrees.
3. Search for small target-component splits before merges. The failure pattern is visually plausible source -> large target component, which suggests the target component is impure; direct relabeling is too blunt.
4. Full-score only candidates whose scheduler score remains above current best after side-effect risk, action-family hard negatives, and density/post-filter history.

Status: continue. The pair/global-ID model is above 70, but e2e IDF1 remains below 70.
