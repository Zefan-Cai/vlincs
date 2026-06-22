# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `7`
- audited rows: `7`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `8363530.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `56.0` | `6.0` | `0.6631824333637429` | `0.7690142499999999` | `0.000131` | `2` | `3` | `hub_bridge_portfolio` |
| 2 | `2` | `56.0` | `6.0` | `0.6631824333637429` | `0.7690142499999999` | `0.000131` | `2` | `3` | `hub_bridge_portfolio` |
| 3 | `3` | `56.0` | `6.0` | `0.6628399097509031` | `0.76901425` | `0.000131` | `2` | `3` | `hub_bridge_portfolio` |
| 4 | `4` | `60.0` | `7.0` | `0.661996291904432` | `0.770636125` | `0.000131` | `2` | `5` | `hub_bridge_portfolio` |
| 5 | `5` | `60.0` | `7.0` | `0.661969291904432` | `0.770636125` | `0.000131` | `2` | `5` | `hub_bridge_portfolio` |
| 6 | `6` | `60.0` | `7.0` | `0.6618262968719494` | `0.770636125` | `0.000131` | `2` | `5` | `hub_bridge_portfolio` |
| 7 | `7` | `72.0` | `8.0` | `0.660629796368133` | `0.7706462083333333` | `0.000131` | `2` | `6` | `hub_bridge_portfolio` |

## Top Edge Evidence

### Candidate rank 1
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 2
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 3
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 4
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 5
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`

