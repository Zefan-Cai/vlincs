# No-Anchor VLINCS: Targeted MCAM04 Subpart Repair Promotion

Date: 2026-06-22

## Why this run

The previous no-anchor best was the subpart combo:

`rank01 + rank02 + rank04`

with canonical p005-area:

`IDF1/HOTA/AssA = 0.656563 / 0.520067 / 0.535717`

The weak slice was still MCAM04 Tc6:

`IDF1/HOTA/AssA = 0.563103 / 0.448751 / 0.493408`

This run regenerates subpart-repair candidates from that current-best assignment
instead of the older roll3 base.  The proposer uses only assignment CSV plus
no-anchor feature NPZ files.  GT is used only by the final scorer.

## Pipeline Hygiene Fix

Two small robustness fixes were needed before the batch could score cleanly:

- `kit/run_no_anchor_density_area_pipeline.sh` now defaults
  `DATA_ROOT=/mnt/localssd/vlincs_reid_data`, so remote ad hoc scoring does not
  silently see zero DS1 GT videos.
- `kit/no_anchor_pervideo_filter_selector.py` gained explicit `--skip-score`,
  allowing density policy export without the intermediate diagnostic GT score.
  The final p005 scorer still produces the authoritative metrics.

## Candidate Queues

Input assignment:

`/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_combo_20260622/assignments/subpart_combo_r01_r02_r04_11seq_assignments.csv`

Feature views:

- fused match/person/color/face/OSNet feature
- DINOv2 base feature as an auxiliary view

Generated queues:

| Queue | Candidates | Purpose |
|---|---:|---|
| `balanced` | 8 | relaxed but bounded MCAM04-focused subpart repair |
| `weakvideo` | 20 | more aggressive weak-video recall probe |

## Canonical p005 Results

| Candidate | Edit | Moved | IDF1 | HOTA | AssA | DetPr | DetRe | MCAM04 IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced rank01 | `35 -> 60` | 14 | 0.657475 | 0.520599 | 0.535769 | 0.765435 | 0.576205 | 0.565121 |
| balanced rank02 | `35 -> 60` | 14 | 0.657425 | 0.520578 | 0.535769 | 0.765383 | 0.576157 | 0.565006 |
| weak rank02 | `35 -> 60` | 20 | 0.657238 | 0.520561 | 0.535906 | 0.765944 | 0.575554 | 0.564429 |
| weak rank01 | `35 -> 60` | 20 | 0.657029 | 0.520322 | 0.535727 | 0.765075 | 0.575724 | 0.564088 |
| balanced rank07 | `9 -> 2330` | 2 | 0.656563 | 0.520067 | 0.535717 | 0.764242 | 0.575480 | 0.563103 |
| balanced rank05 | `47 -> 2329` | 2 | 0.656530 | 0.520026 | 0.535680 | 0.764121 | 0.575498 | 0.563103 |
| balanced rank04 | `55 -> 58` | 14 | 0.656185 | 0.519615 | 0.535337 | 0.764671 | 0.574657 | 0.562074 |
| weak rank08 | `55 -> 58` | 20 | 0.655546 | 0.519008 | 0.534897 | 0.764339 | 0.573864 | 0.560553 |
| weak rank03 | `55 -> 58` | 20 | 0.655481 | 0.518953 | 0.534863 | 0.764273 | 0.573803 | 0.560407 |
| weak rank11 | `2329 -> 40` | 20 | 0.654579 | 0.518277 | 0.534489 | 0.761906 | 0.573756 | 0.563103 |

## Promoted Result

Promote balanced rank01 `35 -> 60`:

`IDF1/HOTA/AssA = 0.657475 / 0.520599 / 0.535769`

Delta from previous best:

- IDF1: `+0.000912`
- HOTA: `+0.000532`
- AssA: `+0.000052`

MCAM04 Tc6 changed:

- IDF1: `0.563103 -> 0.565121`
- DetPr: `0.685295 -> 0.688037`
- DetRe: `0.477892 -> 0.479465`
- AssA: `0.493408 -> 0.492193`

The gain comes from better MCAM04 detection/identity recall while losing a
little MCAM04 association quality.  This is still below the 0.70 e2e goal.

## Side-Effect Lessons

The key ablation is not merely that `35 -> 60` helps.  It also shows what the
next side-effect referee must learn:

- `35 -> 60` is positive, but the 14-tracklet group beats the 20-tracklet
  expansions.  More weak-video recall is not monotonic.
- `55 -> 58` has high focus hits and high target similarity, but is strongly
  negative.  Focus-video count and target_sim are insufficient.
- broad cross-video `2329 -> 40` is a large negative and hurts MCAM06.
- `9 -> 2330` is effectively neutral under p005.

## Artifacts

- Remote run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_next_20260622`
- Local no-zip mirror:
  `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_next_20260622`
- Best JSON:
  `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_next_20260622/rank01_subpart_s35_to60_14seq_assignments_density_p005_area.json`
- Candidate manifests:
  `balanced_manifest.json`, `weakvideo_manifest.json`

## Next

Use the new side-effect labels to compose compatible positives and train a
subpart-specific referee.  The next experiment should test whether `35 -> 60`
can be safely combined with earlier positive subpart repairs while excluding
`55 -> 58`-like high-similarity false positives.
