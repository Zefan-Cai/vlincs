# No-Anchor Local Per-Video Source Selector

Production selector: no anchors, no GT, no full-score feedback.

- strategy: `conservative`
- base source: `base`
- reference source: `base`
- assignment rows: `7487`
- predicted IDs: `81`

## Policy

| video | source | base score | best score | gain | changed |
| --- | --- | ---: | ---: | ---: | ---: |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc6` | `trusted_arb` | 0.343557 | 0.360789 | 0.017232 | 0.040179 |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc8` | `trusted_arb` | 0.388279 | 0.413465 | 0.025186 | 0.027119 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | `trusted_arb` | 0.379267 | 0.385606 | 0.006339 | 0.014742 |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `base` | 0.399426 | 0.399426 | 0.000000 | 0.000000 |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | `trusted_arb` | 0.395541 | 0.402109 | 0.006568 | 0.006187 |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc6` | `trusted_arb` | 0.376827 | 0.385684 | 0.008857 | 0.009524 |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc8` | `trusted_arb` | 0.384583 | 0.395208 | 0.010625 | 0.015625 |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | `base` | 0.368083 | 0.368083 | 0.000000 | 0.000000 |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `trusted_arb` | 0.387962 | 0.397320 | 0.009358 | 0.013761 |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `trusted_arb` | 0.414573 | 0.422339 | 0.007766 | 0.008915 |

## Source Counts

- `base`: `725` rows
- `trusted_arb`: `6762` rows
