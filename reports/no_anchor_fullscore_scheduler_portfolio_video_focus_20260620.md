# No-Anchor Video-Focus Scheduler

- input scheduler: `local_runs/no_anchor_fullscore_scheduler_portfolio_next_queue_20260620.json`
- selected rows: `8`
- target IDF1 floor: `0.700`
- uses anchors: `False`
- uses GT for training/anchors: `False`
- uses GT only for experiment-budget prioritization: `True`

## Bottleneck Weights

| video | focus weight |
| --- | ---: |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | `0.139306` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | `0.091622` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `0.071385` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | `0.009180` |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc6` | `0.000000` |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc8` | `0.000000` |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc6` | `0.000000` |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc8` | `0.000000` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `0.000000` |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `0.000000` |

## Selected Candidates

| rank | focus | predicted full | original rank | family | source videos | target videos |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `5.024` | `0.659998` | `19` | `hub_bridge_portfolio:component:21+28+32+33->0+15+19+55` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 28, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 20}` | `{"vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 8, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 6}` |
| 2 | `5.024` | `0.658602` | `18` | `hub_bridge_portfolio:component:21+28+33->0+19+55` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 28, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 8}` | `{"vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 8, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 3}` |
| 3 | `4.597` | `0.658977` | `20` | `hub_bridge_portfolio:component:26+28+32+33+40->15+21+55` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 24, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 20}` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 9, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 6}` |
| 4 | `3.501` | `0.662503` | `17` | `hub_bridge_portfolio:component:0+21+28+32+33+4->15+19+55+6` | `{"vlincs_MS01_MC0001_MCAM00_2024-03-Tc6": 13, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 20, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 23}` | `{"vlincs_MS01_MC0001_MCAM00_2024-03-Tc6": 2, "vlincs_MS01_MC0001_MCAM00_2024-03-Tc8": 1, "vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 2, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 5, "vlincs_MS01_MC0001_MCAM05_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 7}` |
| 5 | `3.501` | `0.661107` | `12` | `hub_bridge_portfolio:component:0+21+28+33+4->19+55+6` | `{"vlincs_MS01_MC0001_MCAM00_2024-03-Tc6": 13, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 20, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 11}` | `{"vlincs_MS01_MC0001_MCAM00_2024-03-Tc6": 2, "vlincs_MS01_MC0001_MCAM00_2024-03-Tc8": 1, "vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 2, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 5, "vlincs_MS01_MC0001_MCAM05_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 4}` |
| 6 | `3.492` | `0.660278` | `11` | `hub_bridge_portfolio:component:21+28+32+33->15+19+55` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 20, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 20}` | `{"vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 5, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 6}` |
| 7 | `3.492` | `0.660158` | `4` | `hub_bridge_portfolio:component:21+32->0+15+19` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 20, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 12}` | `{"vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 5, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 3}` |
| 8 | `3.492` | `0.658482` | `16` | `hub_bridge_portfolio:component:21+28+33->19+55` | `{"vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 20, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 8}` | `{"vlincs_MS01_MC0001_MCAM03_2024-03-Tc6": 1, "vlincs_MS01_MC0001_MCAM04_2024-03-Tc6": 5, "vlincs_MS01_MC0001_MCAM08_2024-03-Tc6": 3}` |
