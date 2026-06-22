# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `8`
- audited rows: `8`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `45970846.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `50.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |
| 2 | `2` | `44.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |
| 3 | `4` | `233.0` | `1.0` | `None` | `None` | `0.000015` | `1` | `1` | `component_graph_high_mass_bridge` |
| 4 | `5` | `223.0` | `1.0` | `None` | `None` | `0.000015` | `1` | `1` | `component_graph_high_mass_bridge` |
| 5 | `3` | `173.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 6 | `6` | `130.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 7 | `7` | `187.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 8 | `8` | `140.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |

## Top Edge Evidence

### Candidate rank 1
- `24` -> `31`: mass `22851603.000`, best_gt `43`, target_gt_count `25.0`

### Candidate rank 2
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 4
- `2` -> `13`: mass `133820.000`, best_gt `4`, target_gt_count `24.0`

### Candidate rank 5
- `13` -> `2`: mass `133820.000`, best_gt `4`, target_gt_count `10.0`

### Candidate rank 3
- `8` -> `44`: mass `0.000`, best_gt ``, target_gt_count `None`

