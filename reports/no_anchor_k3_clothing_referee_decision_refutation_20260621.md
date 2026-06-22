# No-Anchor K3 Clothing Referee Decision Refutation

Date: 2026-06-21

## Verdict

Reject whole-component merge materialization; promote micro-component surgery as
the next branch.

The targeted clothing/body referee produced two high-confidence positive edges
from the multiview preview set:

| source | target | confidence | source size | target size |
|---:|---:|---:|---:|---:|
| 35 | 61 | 0.863755 | 276 | 4 |
| 10 | 26 | 0.956621 | 246 | 252 |

The other preview edges were rejected at around `3e-4`, and the nine edges that
the budgeted multiview merge had accepted were also rejected at around `2e-4`.
So the clothing/body referee is informative.

However, directly materializing the two positive edges as whole-component merges
accepted zero edges after physical constraints.

## Full-Score Result

| threshold | max component size | accepted edges | rejected forbidden | rejected size | full IDF1 | HOTA | AssA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 300 | 0 | 1 | 1 | 0.653210 | 0.517030 | 0.532678 |
| 0.85 | 300 | 0 | 1 | 1 | 0.653210 | 0.517030 | 0.532678 |
| 0.95 | 300 | 0 | 0 | 1 | 0.653210 | 0.517030 | 0.532678 |
| 0.70 | 500 | 0 | 2 | 0 | 0.653210 | 0.517030 | 0.532678 |

Standing best remains:

| Run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 |

## Interpretation

This is not a failure of the learned referee. It is a failure of the
materialization action.

The referee says:

- `35 -> 61` and `10 -> 26` contain strong local same-person evidence.

The physical gate says:

- those edges cannot be merged at the whole-component level without violating
  same-video / overlap constraints or component-size constraints.

Therefore the next action should be surgical:

1. make `no_anchor_clothing_edge_audit.py` output the top supporting left/right
   tracklet seqs for each high-confidence edge;
2. feed those seq groups into `kit/apply_no_anchor_visual_edge_surgery.py`;
3. create micro-components from only the visually verified tracklets;
4. full-score the resulting assignment.

This matches the oracle-gap diagnosis: the repair target is not simply merging
large components. It is extracting the right sub-tracklets from conflicted
components.

## Deli Interpretation

- proposer: multiview previews found possible repairs;
- learned referee: accepted `35 -> 61` and `10 -> 26`, rejected the bad bridge
  family including `9 -> 60`;
- physical opponent: blocked whole-component merge;
- gate: no forced global-ID edit should be committed yet;
- next proposer: generate micro-component surgery using the referee's top
  sample-pair provenance.

## Artifacts

- `local_runs/no_anchor_k3_clothing_edge_preview_audit_20260621/audit.json`
- `local_runs/no_anchor_k3_clothing_edge_preview_audit_20260621/decisions.json`
- `local_runs/no_anchor_k3_clothing_referee_decision_fullscore_20260621_summary.json`
- `local_runs/no_anchor_k3_clothing_referee_decision_fullscore_20260621/result.json`
- `local_runs/no_anchor_k3_clothing_referee_decision_fullscore_20260621/result.csv`
- `local_runs/no_anchor_k3_clothing_referee_decision_fullscore_20260621/top_assignments.csv`
- remote run dir:
  `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_clothing_referee_decision_fullscore_20260621`
