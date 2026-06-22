# No-Anchor K3 State Drop-Forced Refutation

Date: 2026-06-21

## Purpose

Test whether a PDF-style commit/defer layer can improve forced submission by quarantining only high-conflict components after the new k3 softcut best.

## Policy

- Base assignment: `no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`.
- Policy: `drop_forced`, `conflict_rate_threshold=0.01`, `committed_min_size=16`, `pending_max_size=0`.
- No-GT selection basis: 4 components are `forced_conflict`; 252 of 7487 tracklets are quarantined.
- GT is used only by evaluator after the assignment is emitted.

## Result

| candidate | raw IDF1 | raw HOTA | raw AssA | primary density IDF1 | HOTA | AssA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `k3_drop_forced_thr001` | 0.645007 | 0.512337 | 0.533096 | 0.647252 | 0.514147 | 0.534981 | reject for forced output |

Standing best remains IDF1 `0.655378` / HOTA `0.518798` / AssA `0.534546`.

## Why It Failed

- Pair diagnostics improved because the policy removed difficult high-conflict components, but full DS1 delivery suffered: raw IDF1 fell to `0.645007`.
- Density filtering recovered only to `0.647252`, still far below the standing best.
- This validates the reporting/state idea but refutes direct deletion of forced-conflict components as the submission strategy.

## Artifacts

- `local_runs/no_anchor_k3_state_drop_forced_refutation_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_policy_probe/state_policy_probe_compact.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_drop_forced001_fullscore/state_policy.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_drop_forced001_fullscore/component_states.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_drop_forced001_fullscore/full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_drop_forced001_fullscore/density_filter.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_state_drop_forced001_fullscore_20260621/full.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_state_drop_forced001_fullscore_20260621/density_primary.zip`
