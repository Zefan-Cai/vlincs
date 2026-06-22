# No-Anchor Source-Local Target Arbitration

Production selector: uses no GT, no anchors, and no full-score feedback.

- edge pool: `30`
- kept edges: `14`
- selected portfolios: `2`

## Selected Portfolios

| rank | mode | sources | targets | moved | mean score | min margin |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `source_local_arbitrated_all` | `21+15+19+35+48+33+2+3+36+10+20+8+47+31` | `0+14+6+60+9+55+13+68+5+26+50+44+17+24` | 1012 | 0.731755 | 0.040661 |
| 2 | `source_local_arbitrated_top` | `21+15+19+35+48+33+2+3` | `0+14+6+60+9+55+13+68` | 580 | 0.786208 | 0.040661 |

## Kept Edge Preview

- `21` -> `0` score `0.914865`, margin `0.186114`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `3`
- `15` -> `14` score `0.830325`, margin `0.830325`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `15`
- `19` -> `6` score `0.819162`, margin `0.819162`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `4`
- `35` -> `60` score `0.782741`, margin `0.782741`, origin `no_anchor_single_edge_wide_scheduler_candidates_20260620.json` rank `20`
- `48` -> `9` score `0.751114`, margin `0.040661`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `10`
- `33` -> `55` score `0.738066`, margin `0.738066`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `8`
- `2` -> `13` score `0.732486`, margin `0.099381`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `11`
- `3` -> `68` score `0.720904`, margin `0.720904`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `2`
- `36` -> `5` score `0.697988`, margin `0.697988`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `9`
- `10` -> `26` score `0.695393`, margin `0.695393`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `18`
- `20` -> `50` score `0.680690`, margin `0.680690`, origin `no_anchor_single_edge_broad_scheduler_candidates_20260620.json` rank `14`
- `8` -> `44` score `0.632487`, margin `0.632487`, origin `no_anchor_fullscore_scheduler_referee_pruned_plus_component_rescue_dedup_probe_20260620.json` rank `5`
- `47` -> `17` score `0.627311`, margin `0.627311`, origin `no_anchor_fullscore_scheduler_referee_pruned_plus_component_rescue_dedup_probe_20260620.json` rank `6`
- `31` -> `24` score `0.621043`, margin `0.621043`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `5`

## Rejected Edge Preview

- `40` -> `21` score `0.835233`, reason `chained_component_edit`
- `32` -> `15` score `0.793787`, reason `chained_component_edit`
- `26` -> `21` score `0.773996`, reason `chained_component_edit`
- `4` -> `6` score `0.766997`, reason `target_capacity`
- `0` -> `6` score `0.766444`, reason `chained_component_edit`
- `9` -> `7` score `0.746590`, reason `chained_component_edit`
- `21` -> `19` score `0.728751`, reason `source_local_lower_score`
- `48` -> `46` score `0.710453`, reason `source_local_lower_score`
- `13` -> `49` score `0.707161`, reason `chained_component_edit`
- `28` -> `55` score `0.692274`, reason `target_capacity`
- `9` -> `6` score `0.659547`, reason `source_local_lower_score`
- `2` -> `13` score `0.633105`, reason `source_local_lower_score`
- `13` -> `2` score `0.632173`, reason `source_local_lower_score`
- `44` -> `8` score `0.630704`, reason `chained_component_edit`
- `17` -> `47` score `0.627113`, reason `chained_component_edit`
- `24` -> `31` score `0.621002`, reason `chained_component_edit`
