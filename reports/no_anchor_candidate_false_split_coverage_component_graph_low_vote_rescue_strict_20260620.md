# No-Anchor Candidate False-Split Coverage Audit

This is an eval-only opponent report. It must not be used as a production selector.

- candidates: `2`
- audited rows: `2`
- missing true-pair mass denominator: `9105079781.000`
- summed positive bridge mass in top rows: `45703206.000`

| audit rank | candidate rank | moved | edits | predicted full | pair F1 | coverage | positive edges | impure targets | mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1` | `50.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |
| 2 | `2` | `44.0` | `1.0` | `None` | `None` | `0.002510` | `1` | `1` | `component_graph_high_mass_bridge` |

## Top Edge Evidence

### Candidate rank 1
- `24` -> `31`: mass `22851603.000`, best_gt `43`, target_gt_count `25.0`

### Candidate rank 2
- `31` -> `24`: mass `22851603.000`, best_gt `43`, target_gt_count `11.0`

