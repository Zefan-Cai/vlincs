# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `3`
- audited rows: `3`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `158174490.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `304.0` | `7.0` | `0.6646824333637429` | `0.769356125` | `0.005791` | `3` | `4` | `hub_bridge_portfolio` |
| 2 | `2` | `308.0` | `8.0` | `0.6634962919044319` | `0.7701670625` | `0.005791` | `3` | `6` | `hub_bridge_portfolio` |
| 3 | `3` | `320.0` | `9.0` | `0.662129796368133` | `0.7701721041666667` | `0.005791` | `3` | `7` | `hub_bridge_portfolio` |

## Top Edge Evidence

### Candidate rank 1
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 2
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 3
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `19` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

