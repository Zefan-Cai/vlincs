# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `8`
- audited rows: `8`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `392937196.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `5` | `312.0` | `8.0` | `0.6688896529501238` | `0.7692052500000001` | `0.008114` | `4` | `5` | `hub_bridge_portfolio` |
| 2 | `6` | `316.0` | `9.0` | `0.6681054507044938` | `0.7697458749999999` | `0.008114` | `4` | `7` | `hub_bridge_portfolio` |
| 3 | `4` | `256.0` | `2.0` | `0.6643894901570562` | `0.76930075` | `0.007983` | `2` | `2` | `hub_bridge_portfolio` |
| 4 | `7` | `304.0` | `7.0` | `0.6646824333637429` | `0.769356125` | `0.005791` | `3` | `4` | `hub_bridge_portfolio` |
| 5 | `8` | `308.0` | `8.0` | `0.6634962919044319` | `0.7701670625` | `0.005791` | `3` | `6` | `hub_bridge_portfolio` |
| 6 | `1` | `64.0` | `7.0` | `0.6681096529501239` | `0.7689588749999999` | `0.002455` | `3` | `4` | `hub_bridge_portfolio` |
| 7 | `2` | `68.0` | `8.0` | `0.6673854507044938` | `0.7697698125` | `0.002455` | `3` | `6` | `hub_bridge_portfolio` |
| 8 | `3` | `80.0` | `9.0` | `0.6668138781595293` | `0.7697748541666667` | `0.002455` | `3` | `7` | `hub_bridge_portfolio` |

## Top Edge Evidence

### Candidate rank 5
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 6
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 4
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 7
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `21` -> `19`: mass `0.000`, best_gt ``, target_gt_count `None`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 8
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `0` -> `6`: mass `0.000`, best_gt ``, target_gt_count `14.0`
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `26` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `33` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `28` -> `55`: mass `0.000`, best_gt ``, target_gt_count `None`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

