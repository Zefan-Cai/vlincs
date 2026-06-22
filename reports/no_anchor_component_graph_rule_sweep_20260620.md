# Component-Graph Rule Sweep

- candidates: `94`
- unordered pair directions: `47`
- positive pair directions: `5`
- positive pair bridge mass: `24273828`
- conclusion: Reject broad high-vote supplement: it pulls too many zero-mass edges. Keep current committed low-vote direction and treat clean high-vote singleton as redundant with existing portfolio edges.

| family | selected | positive | false | precision | bridge mass | top coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_committed_low_vote_directional` | `2` | `1` | `1` | `0.500` | `22851603` | `0.002510` |
| `broad_high_vote_supplement` | `11` | `2` | `9` | `0.182` | `1118967` | `0.000103` |
| `clean_high_vote_singleton` | `1` | `1` | `0` | `1.000` | `933375` | `0.000103` |

## current_committed_low_vote_directional

| rank | source | target | mass | best | mean | min-view | vote | overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `84` | `31` | `24` | `22851603` | `0.841` | `0.490` | `0.722` | `0.30` | `0.0169` |
| `30` | `49` | `33` | `0` | `0.889` | `0.484` | `0.696` | `1.00` | `0.0196` |

## broad_high_vote_supplement

| rank | source | target | mass | best | mean | min-view | vote | overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `41` | `17` | `0` | `0.943` | `0.681` | `0.893` | `1.00` | `0.0197` |
| `2` | `32` | `15` | `933375` | `0.924` | `0.628` | `0.893` | `1.00` | `0.0127` |
| `6` | `20` | `12` | `0` | `0.927` | `0.640` | `0.895` | `1.00` | `0.0187` |
| `8` | `19` | `6` | `0` | `0.908` | `0.653` | `0.866` | `1.00` | `0.0177` |
| `21` | `31` | `25` | `0` | `0.924` | `0.568` | `0.818` | `1.00` | `0.0163` |
| `18` | `24` | `38` | `0` | `0.935` | `0.565` | `0.784` | `1.00` | `0.0186` |
| `17` | `48` | `21` | `0` | `0.868` | `0.584` | `0.815` | `1.00` | `0.0168` |
| `22` | `9` | `48` | `0` | `0.871` | `0.557` | `0.827` | `1.00` | `0.0191` |
| `40` | `24` | `39` | `185592` | `0.876` | `0.527` | `0.772` | `1.00` | `0.0198` |
| `30` | `49` | `33` | `0` | `0.889` | `0.484` | `0.696` | `1.00` | `0.0196` |
| `54` | `41` | `32` | `0` | `0.843` | `0.498` | `0.775` | `1.00` | `0.0102` |

## clean_high_vote_singleton

| rank | source | target | mass | best | mean | min-view | vote | overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `32` | `15` | `933375` | `0.924` | `0.628` | `0.893` | `1.00` | `0.0127` |
