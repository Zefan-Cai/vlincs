# No-Anchor Source-Local Target Arbitration

Production selector: uses no GT, no anchors, and no full-score feedback.

- edge pool: `6`
- kept edges: `5`
- selected portfolios: `2`

## Selected Portfolios

| rank | mode | sources | targets | moved | mean score | min margin |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `source_local_arbitrated_all` | `32+4+9+3+31` | `15+6+7+68+24` | 320 | 0.729864 | 0.087042 |
| 2 | `source_local_arbitrated_highconf` | `32+4+9+3` | `15+6+7+68` | 276 | 0.757069 | 0.087042 |

## Kept Edge Preview

- `32` -> `15` score `0.793787`, margin `0.793787`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `1`
- `4` -> `6` score `0.766997`, margin `0.766997`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `1`
- `9` -> `7` score `0.746590`, margin `0.087042`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `1`
- `3` -> `68` score `0.720904`, margin `0.720904`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `2`
- `31` -> `24` score `0.621043`, margin `0.621043`, origin `no_anchor_union_referee_pruned_plus_target_impurity_proxy_20260620.json` rank `5`

## Rejected Edge Preview

- `9` -> `6` score `0.659547`, reason `source_local_lower_score`
