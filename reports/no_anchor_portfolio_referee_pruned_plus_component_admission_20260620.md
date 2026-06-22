# No-Anchor Hub-Bridge Portfolio Candidates

- source scheduler: `local_runs/no_anchor_union_referee_pruned_plus_component_admission_20260620.json`
- candidate rows: `5`
- portfolio rows: `4`

| rank | predicted full | moved | edits | targets | source ranks | pair-mass proxy |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `0.668290` | `320` | `5` | `15+24+6+68+7` | `[3, 5]` | `7448.000` |
| 2 | `0.666628` | `72` | `4` | `15+24+6+7` | `[1, 5]` | `7200.000` |
| 3 | `0.665889` | `300` | `3` | `24+68+7` | `[2, 5]` | `4176.000` |
| 4 | `0.663861` | `312` | `4` | `15+24+6+68` | `[4, 5]` | `5720.000` |

## Notes

- Portfolios are alternatives: each row is one assignment candidate.
- Production selection is no-anchor; source full-score labels are not used.
- Compatibility requires no source-seq overlap and no chained source/target component edits.
