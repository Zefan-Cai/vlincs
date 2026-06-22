# No-Anchor Proxy-Reviewer Ensemble Scheduler

- raw candidates: `33`
- eligible: `21`
- rejected: `12`
- proxy reviewers: `4`
- current best full IDF1: `0.655240`
- min median full: `0.650000`
- min reviewers above current: `0`

## Selected

| rank | ensemble | median | min | max | std | above current | pair F1 | family |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.653247` | `0.653775` | `0.652423` | `0.653814` | `0.000592` | `0` | `0.769747` | `hub_bridge_portfolio:component:26+28+33+40->21+55` |
| 2 | `0.652663` | `0.653249` | `0.653238` | `0.654311` | `0.000462` | `0` | `0.768959` | `referee_pruned_portfolio:component:32+4+9->15+6+7` |
| 3 | `0.652124` | `0.653451` | `0.652644` | `0.653485` | `0.000355` | `0` | `0.768553` | `hub_bridge_portfolio:component:21+28+33->0+19+55` |
| 4 | `0.651646` | `0.654675` | `0.653918` | `0.654711` | `0.000335` | `0` | `0.768337` | `hub_bridge_portfolio:component:0+21+28+33+4->19+55+6` |
| 5 | `0.650680` | `0.653384` | `0.652161` | `0.653418` | `0.000535` | `0` | `0.770179` | `hub_bridge_portfolio:component:26+28+32+33+40->15+21+55` |
| 6 | `0.650290` | `0.653908` | `0.653210` | `0.653932` | `0.000306` | `0` | `0.770500` | `hub_bridge_portfolio:component:0+26+28+33+4+40->21+55+6` |
| 7 | `0.649574` | `0.653043` | `0.652267` | `0.653073` | `0.000341` | `0` | `0.769176` | `hub_bridge_portfolio:component:21+28+32+33->0+15+19+55` |
| 8 | `0.648783` | `0.653934` | `0.653282` | `0.653964` | `0.000288` | `0` | `0.769014` | `hub_bridge_portfolio:component:0+21+28+32+33+4->15+19+55+6` |

## Proxy Reviewers

- `local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json`
- `local_runs/no_anchor_full_proxy_compact_ridge_model_20260620.json`
- `local_runs/no_anchor_full_proxy_mass_features_ridge_model_20260620.json`
- `local_runs/no_anchor_full_proxy_autoresearch_refresh_model_20260620.json`
