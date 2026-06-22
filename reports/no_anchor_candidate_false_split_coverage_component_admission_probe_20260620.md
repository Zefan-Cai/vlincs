# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `4`
- audited rows: `4`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `313042635.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `3` | `320.0` | `5.0` | `0.6682896529501237` | `0.38460262500000003` | `0.010624` | `5` | `5` | `hub_bridge_portfolio` |
| 2 | `2` | `300.0` | `3.0` | `0.6658894901570561` | `0.384650375` | `0.010493` | `3` | `3` | `hub_bridge_portfolio` |
| 3 | `4` | `312.0` | `4.0` | `0.6638610047923142` | `0.3846780625` | `0.008300` | `4` | `4` | `hub_bridge_portfolio` |
| 4 | `1` | `72.0` | `4.0` | `0.6666282243786953` | `0.38447943749999997` | `0.004964` | `4` | `4` | `hub_bridge_portfolio` |

## Top Edge Evidence

### Candidate rank 3
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 2
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 4
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 1
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

