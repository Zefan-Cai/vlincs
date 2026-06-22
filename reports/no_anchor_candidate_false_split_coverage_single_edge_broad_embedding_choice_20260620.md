# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `17`
- audited rows: `17`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `22482521.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `15` | `8.0` | `1.0` | `None` | `0.7689035` | `0.002323` | `1` | `1` | `broad_single_edge_from_scheduler` |
| 2 | `7` | `12.0` | `1.0` | `None` | `0.7724307499999999` | `0.000103` | `1` | `1` | `broad_single_edge_from_scheduler` |
| 3 | `6` | `8.0` | `1.0` | `None` | `0.7729314166666666` | `0.000029` | `1` | `1` | `broad_single_edge_from_scheduler` |
| 4 | `11` | `8.0` | `1.0` | `None` | `0.7708031666666667` | `0.000015` | `1` | `1` | `broad_single_edge_from_scheduler` |
| 5 | `1` | `8.0` | `1.0` | `None` | `0.7738164999999999` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 6 | `2` | `8.0` | `1.0` | `None` | `0.7738164999999999` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 7 | `4` | `12.0` | `1.0` | `None` | `0.7729314166666666` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 8 | `5` | `8.0` | `1.0` | `None` | `0.7729314166666666` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 9 | `3` | `8.0` | `1.0` | `None` | `0.772654` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 10 | `8` | `8.0` | `1.0` | `None` | `0.7708031666666667` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 11 | `9` | `8.0` | `1.0` | `None` | `0.7708031666666667` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 12 | `10` | `8.0` | `1.0` | `None` | `0.7708031666666667` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |
| 13 | `12` | `8.0` | `1.0` | `None` | `0.7706462083333334` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 14 | `13` | `12.0` | `1.0` | `None` | `0.7703633333333334` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 15 | `14` | `12.0` | `1.0` | `None` | `0.7687698000000001` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 16 | `16` | `8.0` | `1.0` | `None` | `0.7687698000000001` | `0.000000` | `0` | `0` | `broad_single_edge_from_scheduler` |
| 17 | `17` | `8.0` | `1.0` | `None` | `0.768665` | `0.000000` | `0` | `1` | `broad_single_edge_from_scheduler` |

## Top Edge Evidence

### Candidate rank 15
- `9` -> `7`: mass `21153911.000`, best_gt `52`, target_gt_count `6.0`

### Candidate rank 7
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`

### Candidate rank 6
- `4` -> `6`: mass `261415.000`, best_gt `43`, target_gt_count `14.0`

### Candidate rank 11
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`

### Candidate rank 1
- `40` -> `21`: mass `0.000`, best_gt ``, target_gt_count `6.0`

