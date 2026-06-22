# No-Anchor K3 Clothing Referee Micro-Surgery Refutation

Date: 2026-06-21

## Verdict

Reject "new micro-component" surgery as currently implemented.

This branch upgraded `kit/no_anchor_clothing_edge_audit.py` to emit top
supporting tracklet seq pairs, then fed those pairs into
`kit/apply_no_anchor_visual_edge_surgery.py`. The goal was to avoid invalid
whole-component merge by extracting only the local tracklets supported by the
learned clothing/body referee.

The result still dropped.

## Setup

Input:

- base k3 assignment:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- surgery input:
  `local_runs/no_anchor_k3_clothing_edge_preview_audit_v2_20260621/surgery_inputs.json`

Positive referee edges:

| edge id | source | target | confidence | left seqs | right seqs |
|---:|---:|---:|---:|---|---|
| 2 | 35 | 61 | 0.863755 | 3871, 3733, 3690, 3286, 3847 | 3891, 3315 |
| 3 | 10 | 26 | 0.956621 | 1936, 377, 481, 8999 | 1929, 390, 388, 480, 8990 |

## Best Full-Scored Row

| accepted edge ids | accepted tracklets | pair F1 | full IDF1 | HOTA | AssA | unmatched FP |
|---|---:|---:|---:|---:|---:|---:|
| `[3]` | 6 | 0.768826 | 0.653051 | 0.516802 | 0.532424 | 116283 |

Base k3 at the same interface:

| Run | pair F1 | full IDF1 | HOTA | AssA |
|---|---:|---:|---:|---:|
| k3 raw full-score | 0.769367 | 0.653210 | 0.517030 | 0.532678 |

Standing best remains:

| Run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 |

## Interpretation

The learned referee is still useful, but the action is wrong.

The surgery extracted six tracklets from edge `10 -> 26` into a new component:

`1936, 377, 481, 8999, 1929, 390`

This reduced pair recall and increased unmatched false positives. That means
the local evidence probably identifies a continuation or sub-tracklet relation,
but creating a new isolated component loses the surrounding identity context.

The next action should be:

1. keep the referee and supporting seq provenance;
2. relink high-confidence seqs into an existing target component when physical
   constraints allow local reassignment;
3. avoid creating a new identity unless it repairs a known conflict or has
   enough temporal continuity support.

In short: use the referee evidence as a local reassignment/label-propagation
edge, not as a standalone new global ID.

## Artifacts

- `kit/no_anchor_clothing_edge_audit.py`
- `local_runs/no_anchor_k3_clothing_edge_preview_audit_v2_20260621/audit.json`
- `local_runs/no_anchor_k3_clothing_edge_preview_audit_v2_20260621/surgery_inputs.json`
- `local_runs/no_anchor_k3_clothing_referee_micro_surgery_20260621_summary.json`
- `local_runs/no_anchor_k3_clothing_referee_micro_surgery_20260621/result.json`
- `local_runs/no_anchor_k3_clothing_referee_micro_surgery_20260621/result.csv`
- `local_runs/no_anchor_k3_clothing_referee_micro_surgery_20260621/top_assignments.csv`
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_clothing_referee_micro_surgery_20260621`
