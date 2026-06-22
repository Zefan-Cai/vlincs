# No-Anchor VLINCS: Recall-Guarded Subpart Repair Promotion

Date: 2026-06-22

## Why this run

The previous MCAM04 admission run showed that pruning low-area tracklets raises
DetPr but loses too much DetRe.  This run keeps delivery recall intact: no rows
are dropped by the proposer.  It instead moves small conflicted visual subparts
from one predicted component to a safer target component.

No anchors or GT labels are used by the proposer.  Inputs are only the current
best assignment CSV plus no-anchor feature NPZ files.  GT appears only inside
the canonical DS1 scorer.

## New utility

- `kit/propose_no_anchor_subpart_repair_candidates.py`

Production-side inputs:

`assignment CSV + fused/DINO tracklet features -> subpart candidates -> assignment CSVs`

The candidate rule:

- source component must have same-stream temporal conflicts;
- moved group is visually tight and internally non-overlapping;
- target component must have no same-stream temporal overlap with the moved
  group;
- no tracklet is dropped;
- selected rows are materialized as assignment CSVs and then scored by the
  existing density + `p005_area` pipeline.

## Candidate queues

Strict queue:

| Rank | Edit | Moved | Focus hits | target_sim | target_margin |
|---:|---|---:|---:|---:|---:|
| 1 | `9 -> 21` | 2 | 0 | 0.699515 | 0.015791 |

Relaxed queue:

| Rank | Edit | Moved | Source videos | Focus hits | target_sim | target_margin |
|---:|---|---:|---|---:|---:|---:|
| 1 | `8 -> 40` | 6 | MCAM04 Tc6 x6 | 6 | 0.644933 | -0.024567 |
| 2 | `24 -> 38` | 2 | MCAM03 Tc6 x2 | 0 | 0.773924 | 0.102956 |
| 4 | `40 -> 2329` | 3 | MCAM03 Tc8 x1, MCAM04 Tc6 x1, MCAM08 Tc6 x1 | 2 | 0.655727 | -0.042199 |

## Canonical p005 results

Previous best:

| Assignment | IDF1 | HOTA | AssA | DetPr | DetRe |
|---|---:|---:|---:|---:|---:|
| roll3 rank04 current best | 0.656225 | 0.519723 | 0.535329 | 0.764545 | 0.574789 |

Subpart repair candidates:

| Candidate | Density IDF1 | p005 IDF1 | HOTA | AssA | DetPr | DetRe | Delta IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict rank01 `9 -> 21` | 0.656051 | 0.656146 | 0.519647 | 0.535272 | 0.764330 | 0.574789 | -0.000079 |
| relaxed rank01 `8 -> 40` | 0.656358 | 0.656453 | 0.519993 | 0.535718 | 0.764260 | 0.575300 | +0.000228 |
| relaxed rank02 `24 -> 38` | 0.656224 | 0.656320 | 0.519856 | 0.535474 | 0.764535 | 0.574941 | +0.000095 |
| relaxed rank04 `40 -> 2329` | 0.656146 | 0.656241 | 0.519663 | 0.535184 | 0.764538 | 0.574818 | +0.000016 |

## Per-video effect

The promoted edit is `8 -> 40`, moving six MCAM04 Tc6 tracklets:

`[3886, 3929, 3967, 4002, 4043, 4079]`

| Slice | Previous IDF1 | New IDF1 | Previous HOTA | New HOTA | Previous AssA | New AssA |
|---|---:|---:|---:|---:|---:|---:|
| MCAM04 Tc6 | 0.562439 | 0.563063 | 0.448616 | 0.448739 | 0.493734 | 0.493444 |
| MCAM06 Tc6 | 0.610296 | 0.610296 | 0.518506 | 0.518506 | 0.600591 | 0.600591 |
| MCAM03 Tc8 | 0.628528 | 0.628528 | 0.510443 | 0.510443 | 0.550903 | 0.550903 |
| MCAM08 Tc6 | 0.769911 | 0.769911 | 0.662239 | 0.662239 | 0.680787 | 0.680787 |

MCAM04 improves through DetRe (`0.476722 -> 0.477831`) while DetPr drops
slightly.  This is the opposite tradeoff from the rejected area-admission
branch, and is better aligned with the recall bottleneck.

## Decision

Promote relaxed rank01 as the new no-anchor e2e best:

`IDF1/HOTA/AssA = 0.656453 / 0.519993 / 0.535718`

This is still far below the 0.70 end-to-end goal, but it is a real structural
promotion after several stale local-attach/admission branches.

Next direction:

1. Continue recall-guarded subpart repair, but make the scorer explicitly
   aware that mild negative target_margin can still help when the moved subpart
   concentrates weak-video temporal conflicts.
2. Generate combination candidates from compatible positive subpart edits:
   relaxed rank01 plus other non-overlapping positive/near-positive edits.
3. Add a side-effect label bank for subpart repairs so future queues are ranked
   by p005 outcome, weak-video DetRe, and target-margin risk instead of one
   hand score.

## Artifacts

- Remote strict run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_repair_20260622`
- Remote relaxed run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_repair_relaxed_20260622`
- Local strict mirror:
  `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_repair_20260622`
- Local relaxed mirror:
  `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_repair_relaxed_20260622`
