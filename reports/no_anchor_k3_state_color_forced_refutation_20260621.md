# No-Anchor K3 State Color-Forced Refutation

Date: 2026-06-21

## Purpose

Test a no-anchor state-layer alternative to `drop_forced`: keep all tracklets, but split high-conflict components using cannot-link graph coloring.

## Policy

- Base assignment: `no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`.
- Policy: `color_forced`, `conflict_rate_threshold=0.01`, `committed_min_size=16`, `pending_max_size=0`.
- No-GT selection basis: 4 `forced_conflict` components, 252 tracklets colored into 15 parts, all 7487 tracklets still delivered.
- GT is used only by evaluator after the assignment is emitted.

## Result

| candidate | raw IDF1 | raw HOTA | raw AssA | primary density IDF1 | HOTA | AssA | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `k3_color_forced_thr001` | 0.652914 | 0.516991 | 0.532992 | 0.655110 | 0.518786 | 0.534885 | reject |

Standing best remains IDF1 `0.655378` / HOTA `0.518798` / AssA `0.534546`.

## Interpretation

- `color_forced` is much less destructive than `drop_forced`, but it still does not beat the k3 softcut best.
- The pair diagnostic rose to `0.770306`, yet full-score primary density reached only `0.655110`; pair precision alone is not enough.
- Pure cannot-link coloring should not be used as the next production move unless paired with a visual identity verifier or a bridge/merge repair that restores false-split recall.

## Artifacts

- `local_runs/no_anchor_k3_state_color_forced_refutation_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_color_forced001_fullscore/state_policy.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_color_forced001_fullscore/component_states.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_color_forced001_fullscore/full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_state_color_forced001_fullscore/density_filter.json`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_state_color_forced001_fullscore_20260621/full.zip`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_state_color_forced001_fullscore_20260621/density_primary.zip`
