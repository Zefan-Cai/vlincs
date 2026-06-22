# No-Anchor Per-Video Consensus Source Selector

Production selector: no anchors, no GT, no full-score feedback.

- grid runs: `96`
- min vote fraction: `0.7`
- assignment rows: `7487`
- predicted IDs: `81`

## Consensus Policy

| video | chosen | top source | votes | fraction | reason |
| --- | --- | --- | ---: | ---: | --- |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc6` | `trusted_arb` | `trusted_arb` | `72/96` | `0.750` | `committed_consensus` |
| `vlincs_MS01_MC0001_MCAM00_2024-03-Tc8` | `trusted_arb` | `trusted_arb` | `96/96` | `1.000` | `committed_consensus` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `base` | `base` | `72/96` | `0.750` | `base_consensus` |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc6` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |
| `vlincs_MS01_MC0001_MCAM05_2024-03-Tc8` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | `base` | `base` | `72/96` | `0.750` | `base_consensus` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `base` | `trusted_arb` | `48/96` | `0.500` | `quarantine_unstable_vote` |

## Grid Settings

- rank `1`: strategy `conservative`, gain `0.002`, changed `0.08`, coverage `0.96`
- rank `2`: strategy `conservative`, gain `0.002`, changed `0.08`, coverage `0.98`
- rank `3`: strategy `conservative`, gain `0.002`, changed `0.08`, coverage `0.995`
- rank `4`: strategy `conservative`, gain `0.002`, changed `0.12`, coverage `0.96`
- rank `5`: strategy `conservative`, gain `0.002`, changed `0.12`, coverage `0.98`
- rank `6`: strategy `conservative`, gain `0.002`, changed `0.12`, coverage `0.995`
- rank `7`: strategy `conservative`, gain `0.002`, changed `0.16`, coverage `0.96`
- rank `8`: strategy `conservative`, gain `0.002`, changed `0.16`, coverage `0.98`
- rank `9`: strategy `conservative`, gain `0.002`, changed `0.16`, coverage `0.995`
- rank `10`: strategy `conservative`, gain `0.002`, changed `0.22`, coverage `0.96`
- rank `11`: strategy `conservative`, gain `0.002`, changed `0.22`, coverage `0.98`
- rank `12`: strategy `conservative`, gain `0.002`, changed `0.22`, coverage `0.995`
- rank `13`: strategy `conservative`, gain `0.006`, changed `0.08`, coverage `0.96`
- rank `14`: strategy `conservative`, gain `0.006`, changed `0.08`, coverage `0.98`
- rank `15`: strategy `conservative`, gain `0.006`, changed `0.08`, coverage `0.995`
- rank `16`: strategy `conservative`, gain `0.006`, changed `0.12`, coverage `0.96`
- rank `17`: strategy `conservative`, gain `0.006`, changed `0.12`, coverage `0.98`
- rank `18`: strategy `conservative`, gain `0.006`, changed `0.12`, coverage `0.995`
- rank `19`: strategy `conservative`, gain `0.006`, changed `0.16`, coverage `0.96`
- rank `20`: strategy `conservative`, gain `0.006`, changed `0.16`, coverage `0.98`
- rank `21`: strategy `conservative`, gain `0.006`, changed `0.16`, coverage `0.995`
- rank `22`: strategy `conservative`, gain `0.006`, changed `0.22`, coverage `0.96`
- rank `23`: strategy `conservative`, gain `0.006`, changed `0.22`, coverage `0.98`
- rank `24`: strategy `conservative`, gain `0.006`, changed `0.22`, coverage `0.995`
- rank `25`: strategy `conservative`, gain `0.012`, changed `0.08`, coverage `0.96`
- rank `26`: strategy `conservative`, gain `0.012`, changed `0.08`, coverage `0.98`
- rank `27`: strategy `conservative`, gain `0.012`, changed `0.08`, coverage `0.995`
- rank `28`: strategy `conservative`, gain `0.012`, changed `0.12`, coverage `0.96`
- rank `29`: strategy `conservative`, gain `0.012`, changed `0.12`, coverage `0.98`
- rank `30`: strategy `conservative`, gain `0.012`, changed `0.12`, coverage `0.995`
