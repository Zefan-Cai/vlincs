# No-Anchor Hub-Bridge Portfolio Candidates

- source scheduler: `local_runs/no_anchor_union_crossqueue_singleedge68_localized_island_20260620.json`
- candidate rows: `5`
- portfolio rows: `10`

| rank | predicted full | moved | edits | targets | source ranks | pair-mass proxy |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `0.668890` | `312` | `8` | `15+19+55+6+68+7` | `[1, 4, 5]` | `10268.000` |
| 2 | `0.668110` | `64` | `7` | `15+19+55+6+7` | `[1, 5]` | `10020.000` |
| 3 | `0.668105` | `316` | `9` | `15+21+55+6+68+7` | `[2, 4, 5]` | `12784.000` |
| 4 | `0.667385` | `68` | `8` | `15+21+55+6+7` | `[2, 5]` | `12536.000` |
| 5 | `0.667354` | `328` | `10` | `15+21+55+6+68+7` | `[3, 4, 5]` | `14848.000` |
| 6 | `0.666814` | `80` | `9` | `15+21+55+6+7` | `[3, 5]` | `14600.000` |
| 7 | `0.664682` | `304` | `7` | `15+19+55+6+68` | `[1, 4]` | `8540.000` |
| 8 | `0.664389` | `256` | `2` | `68+7` | `[4, 5]` | `1976.000` |
| 9 | `0.663496` | `308` | `8` | `15+21+55+6+68` | `[2, 4]` | `11056.000` |
| 10 | `0.662130` | `320` | `9` | `15+21+55+6+68` | `[3, 4]` | `13120.000` |

## Notes

- Portfolios are alternatives: each row is one assignment candidate.
- Production selection is no-anchor; source full-score labels are not used.
- Compatibility requires no source-seq overlap and no chained source/target component edits.
