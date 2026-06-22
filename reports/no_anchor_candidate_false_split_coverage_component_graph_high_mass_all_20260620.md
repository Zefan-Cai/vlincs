# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `94`
- audited rows: `94`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `48547656.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `83` | `50.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |
| 2 | `84` | `44.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |
| 3 | `1` | `158.0` | `1.0` | `None` | `None` | `0.000103` | `1` | `1` | `component_graph_high_mass_bridge` |
| 4 | `2` | `117.0` | `1.0` | `None` | `None` | `0.000103` | `1` | `1` | `component_graph_high_mass_bridge` |
| 5 | `35` | `121.0` | `1.0` | `None` | `None` | `0.000020` | `1` | `1` | `component_graph_high_mass_bridge` |
| 6 | `40` | `50.0` | `1.0` | `None` | `None` | `0.000020` | `1` | `1` | `component_graph_high_mass_bridge` |
| 7 | `77` | `172.0` | `1.0` | `None` | `None` | `0.000019` | `1` | `1` | `component_graph_high_mass_bridge` |
| 8 | `78` | `77.0` | `1.0` | `None` | `None` | `0.000019` | `1` | `1` | `component_graph_high_mass_bridge` |
| 9 | `42` | `223.0` | `1.0` | `None` | `None` | `0.000015` | `1` | `1` | `component_graph_high_mass_bridge` |
| 10 | `43` | `233.0` | `1.0` | `None` | `None` | `0.000015` | `1` | `1` | `component_graph_high_mass_bridge` |
| 11 | `3` | `140.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 12 | `4` | `183.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 13 | `5` | `41.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 14 | `6` | `93.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 15 | `7` | `172.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 16 | `8` | `93.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 17 | `9` | `203.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 18 | `10` | `158.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 19 | `11` | `196.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 20 | `12` | `172.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 21 | `13` | `206.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 22 | `14` | `199.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 23 | `15` | `155.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 24 | `16` | `219.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 25 | `17` | `196.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 26 | `18` | `50.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 27 | `19` | `156.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 28 | `20` | `196.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `1` | `component_graph_high_mass_bridge` |
| 29 | `21` | `44.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |
| 30 | `22` | `77.0` | `1.0` | `None` | `None` | `0.000000` | `0` | `0` | `component_graph_high_mass_bridge` |

## Top Edge Evidence

### Candidate rank 83
- `24` -> `31`: mass `22851603.000`, best_gt `43`, target_gt_count `25.0`

### Candidate rank 84
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

### Candidate rank 1
- `15` -> `32`: mass `933375.000`, best_gt `7`, target_gt_count `19.0`

### Candidate rank 2
- `32` -> `15`: mass `933375.000`, best_gt `7`, target_gt_count `10.0`

### Candidate rank 35
- `39` -> `24`: mass `185592.000`, best_gt `7`, target_gt_count `11.0`

