# No-Anchor Hub-Bridge Portfolio Candidates

- source scheduler: `local_runs/no_anchor_union_scheduler_existing_portfolios_20260620.json`
- candidate rows: `41`
- portfolio rows: `7`

| rank | predicted full | moved | edits | targets | source ranks | pair-mass proxy |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `0.663182` | `56` | `6` | `15+19+55+6` | `[2, 16]` | `8292.000` |
| 2 | `0.663182` | `56` | `6` | `15+19+55+6` | `[3, 14]` | `8292.000` |
| 3 | `0.662840` | `56` | `6` | `15+19+55+6` | `[6, 8]` | `8292.000` |
| 4 | `0.661996` | `60` | `7` | `15+21+55+6` | `[5, 14]` | `10808.000` |
| 5 | `0.661969` | `60` | `7` | `15+21+55+6` | `[2, 25]` | `10808.000` |
| 6 | `0.661826` | `60` | `7` | `15+21+55+6` | `[8, 9]` | `10808.000` |
| 7 | `0.660630` | `72` | `8` | `15+21+55+6` | `[13, 25]` | `12872.000` |

## Notes

- Portfolios are alternatives: each row is one assignment candidate.
- Production selection is no-anchor; source full-score labels are not used.
- Compatibility requires no source-seq overlap and no chained source/target component edits.
