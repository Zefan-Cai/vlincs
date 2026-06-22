# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `7`
- audited rows: `7`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `51530040.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `6` | `248.0` | `1.0` | `None` | `0.769698` | `0.005659` | `1` | `1` | `edge_table_single_edge` |
| 2 | `1` | `156.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `1` | `edge_table_single_edge` |
| 3 | `2` | `195.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `0` | `edge_table_single_edge` |
| 4 | `3` | `206.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `0` | `edge_table_single_edge` |
| 5 | `4` | `214.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `1` | `edge_table_single_edge` |
| 6 | `5` | `233.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `0` | `edge_table_single_edge` |
| 7 | `7` | `276.0` | `1.0` | `None` | `0.769698` | `0.000000` | `0` | `0` | `edge_table_single_edge` |

## Top Edge Evidence

### Candidate rank 6
- `3` -> `68`: mass `51530040.000`, best_gt `4`, target_gt_count `11.0`

### Candidate rank 1
- `25` -> `30`: mass `0.000`, best_gt ``, target_gt_count `12.0`

### Candidate rank 2
- `4` -> `34`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 3
- `38` -> `78`: mass `0.000`, best_gt ``, target_gt_count `None`

### Candidate rank 4
- `14` -> `27`: mass `0.000`, best_gt ``, target_gt_count `8.0`

