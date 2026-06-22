# Broad Scheduler Side-Effect Audit Summary

Date: 2026-06-21

## Summary

- scheduler manifests scanned: `31`
- candidate rows: `217`
- recommendation counts: `{'defer_needs_counterfactual': 4, 'reject_or_quarantine': 179, 'reject_replay_failure': 19, 'safe_to_fullscore': 15}`
- safe rows before dedup: `15`
- unique safe component signatures: `2`

## Safe Signatures

| source components | target components | moved | rows | best proxy | example |
| --- | --- | ---: | ---: | ---: | --- |
| `[21]` | `[19]` | 12 | 9 | 0.658302 | `no_anchor_fullscore_scheduler_mass_features_20260620.json` rank 2 |
| `[32]` | `[15]` | 12 | 6 | 0.658179 | `no_anchor_fullscore_scheduler_mass_features_20260620.json` rank 1 |

## Decision

Broad side-effect audit found 15 safe-to-fullscore rows, but after component-level deduplication they reduce to already tested 21->19 and 32->15 directions. Existing scheduler families are saturated; next research should split source components or bring in new evidence rather than score duplicate manifests.
