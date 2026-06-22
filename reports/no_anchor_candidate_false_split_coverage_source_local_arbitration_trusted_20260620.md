# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `2`
- audited rows: `2`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `170609085.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `320.0` | `5.0` | `0.66474` | `None` | `0.010624` | `5` | `5` | `source_local_arbitrated_all` |
| 2 | `2` | `276.0` | `4.0` | `0.66324` | `None` | `0.008114` | `4` | `4` | `source_local_arbitrated_highconf` |

## Top Edge Evidence

### Candidate rank 1
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 2
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

