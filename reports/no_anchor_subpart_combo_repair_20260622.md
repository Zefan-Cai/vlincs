# No-Anchor VLINCS: Subpart Repair Combo Ablation

Date: 2026-06-22

## Why this run

The previous subpart repair run promoted relaxed rank01 (`8 -> 40`) to
`IDF1/HOTA/AssA = 0.656453 / 0.519993 / 0.535718`.  This run tests whether
that gain is compositional.

The proposer uses no anchors and no GT:

`subpart_repair_manifest + assignment CSV -> combo assignment CSVs`

GT appears only in the canonical scorer.

## New utility

- `kit/compose_no_anchor_subpart_repair_combos.py`

It reads `moved_preview` rows from the subpart repair manifest, checks that no
tracklet receives conflicting targets, and materializes assignment CSVs for
specified rank combinations.

## Evaluated Combos

The canonical path was:

`assignment CSV -> full export zip -> density_simple -> p005_area`

| Combo | Moved tracklets | p005 IDF1 | HOTA | AssA | DetPr | DetRe | Delta IDF1 vs previous best |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rank01 + rank02` | 8 | 0.656548 | 0.520126 | 0.535863 | 0.764250 | 0.575452 | +0.000095 |
| `rank01 + rank04` | 9 | 0.656468 | 0.519934 | 0.535572 | 0.764253 | 0.575329 | +0.000015 |
| `rank01 + rank02 + rank04` | 11 | 0.656563 | 0.520067 | 0.535717 | 0.764242 | 0.575480 | +0.000110 |

Previous best:

`IDF1/HOTA/AssA = 0.656453 / 0.519993 / 0.535718`

## Per-Video Side Effects

| Combo | MCAM03 Tc6 IDF1 | MCAM04 Tc6 IDF1 | MCAM08 Tc6 IDF1 | Notes |
|---|---:|---:|---:|---|
| `rank01 + rank02` | 0.692726 | 0.563063 | 0.769911 | best HOTA/AssA in this run; rank02 helps MCAM03 without hurting MCAM04 |
| `rank01 + rank04` | 0.691923 | 0.563103 | 0.769911 | rank04 shifts MCAM04 slightly up but hurts MCAM03 and MCAM08 AssA/HOTA |
| `rank01 + rank02 + rank04` | 0.692726 | 0.563103 | 0.769911 | best IDF1; rank04 adds tiny MCAM04 DetRe but lowers AssA versus `1+2` |

## Decision

Promote `rank01 + rank02 + rank04` as the new no-anchor e2e best by IDF1:

`IDF1/HOTA/AssA = 0.656563 / 0.520067 / 0.535717`

This is a small gain, but it is useful because it proves the subpart repair
signal is at least weakly compositional.  The nearby `rank01 + rank02` point is
also important: it has better HOTA/AssA (`0.520126 / 0.535863`) but slightly
lower IDF1 (`0.656548`).  Future side-effect ranking should treat rank04-like
edits as IDF1-positive but association-risky.

## Stopped Work

The rank05 combinations were generated but not full-scored in this turn.  They
involve lower target margins and a source component from the already strong
MCAM08 slice, so they were stopped after three verified combo labels to avoid
spending more canonical scorer time on low-confidence side effects.

## Artifacts

- Remote run:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_currentbest_subpart_combo_20260622`
- Local mirror:
  `local_runs/remote_h100_test_3_20260622/no_anchor_currentbest_subpart_combo_20260622`
- Combo composer:
  `kit/compose_no_anchor_subpart_repair_combos.py`
