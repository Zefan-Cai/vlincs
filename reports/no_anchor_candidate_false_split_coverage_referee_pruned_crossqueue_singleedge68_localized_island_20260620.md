# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `4`
- audited rows: `4`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `221636223.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `3` | `276.0` | `4.0` | `0.6667896529501237` | `0.7692052500000001` | `0.008114` | `4` | `4` | `referee_pruned_portfolio` |
| 2 | `2` | `256.0` | `2.0` | `0.6643894901570562` | `0.76930075` | `0.007983` | `2` | `2` | `referee_pruned_portfolio` |
| 3 | `4` | `268.0` | `3.0` | `0.6623610047923143` | `0.769356125` | `0.005791` | `3` | `3` | `referee_pruned_portfolio` |
| 4 | `1` | `28.0` | `3.0` | `0.6657882243786953` | `0.7689588749999999` | `0.002455` | `3` | `3` | `referee_pruned_portfolio` |

## Top Edge Evidence

### Candidate rank 3
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 2
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 4
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 1
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

