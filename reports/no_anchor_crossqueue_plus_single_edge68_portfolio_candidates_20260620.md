# No-Anchor Hub-Bridge Portfolio Candidates

- source scheduler: `local_runs/no_anchor_union_crossqueue_plus_single_edge68_20260620.json`
- candidate rows: `4`
- portfolio rows: `3`

| rank | predicted full | moved | edits | targets | source ranks | pair-mass proxy |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `0.664682` | `304` | `7` | `15+19+55+6+68` | `[1, 4]` | `8540.000` |
| 2 | `0.663496` | `308` | `8` | `15+21+55+6+68` | `[2, 4]` | `11056.000` |
| 3 | `0.662130` | `320` | `9` | `15+21+55+6+68` | `[3, 4]` | `13120.000` |

## Notes

- Portfolios are alternatives: each row is one assignment candidate.
- Production selection is no-anchor; source full-score labels are not used.
- Compatibility requires no source-seq overlap and no chained source/target component edits.
