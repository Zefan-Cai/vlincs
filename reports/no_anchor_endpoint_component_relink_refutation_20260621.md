# No-Anchor Endpoint Component Relink Refutation

Date: 2026-06-21

## Context

Standing no-anchor delivery best:

- model-side pair F1/P/R: `0.775234 / 0.820504 / 0.734698`
- e2e delivery IDF1/HOTA/AssA: `0.655817 / 0.519228 / 0.534791`

The previous local-continuity branch had too little coverage.  This branch added
an endpoint-extrapolation proposer:

- source unit: small current component;
- target unit: larger same-video current component;
- evidence: tracklet endpoint visual similarity, extrapolated bbox position,
  frame gap, bbox scale, and cannot-link checks;
- no anchors and no GT for selection;
- GT used only after prediction for pair/full diagnostics.

New tool:

- `kit/no_anchor_endpoint_component_relink_sweep.py`

## Candidate Recall

Remote probes:

- `/mnt/localssd/vlincs_reid_runs/no_anchor_endpoint_component_relink_smoke_mcam04_08_20260621`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_endpoint_component_relink_smoke_allvideos_20260621`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_endpoint_component_relink_top1_fullscore_20260621`

Local artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_smoke_mcam04_08_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_smoke_mcam04_08_20260621/result.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_smoke_allvideos_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_smoke_allvideos_20260621/result.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_top1_fullscore_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_top1_fullscore_20260621/top_full_export.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_component_relink_top1_fullscore_20260621/density_simple_sourcezip.json`

| probe | candidates | accepted relinks | pair F1 | precision | recall |
|---|---:|---:|---:|---:|---:|
| MCAM04/08 smoke | `1` | `1` | `0.770740` | `0.817122` | `0.729341` |
| all-video smoke | `5` | `1` | `0.770738` | `0.817116` | `0.729341` |

The proposer increased candidate recall compared with direct video-temporal
adjacency, but the no-GT ranking selected a precision-risk edge first.

## Full-Scored Top Edge

Top all-video endpoint edge:

| field | value |
|---|---:|
| source seq / label / size | `1604 / 52 / 1` |
| target seq / label / size | `1549 / 11 / 149` |
| video | `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` |
| direction | `target_before_source` |
| gap frames | `73` |
| visual | `0.849789` |
| endpoint score | `0.854767` |
| position score | `0.999998` |
| scale score | `0.980125` |

Full-score:

| output | IDF1 | HOTA | AssA | note |
|---|---:|---:|---:|---|
| standing best density-simple | `0.655817` | `0.519228` | `0.534791` | current best |
| endpoint top1 raw full-score | `0.653707` | `0.517526` | `0.532972` | below best |
| endpoint top1 density-simple | `0.655815` | `0.519226` | `0.534788` | below best by `0.000002` IDF1 |

## Decision

Refuted as a production edit.

Endpoint extrapolation did improve candidate recall from the previous
video-temporal branch, but the current ranking treats geometry as too strong:
the selected edge is visually and geometrically plausible yet slightly reduces
precision and does not improve delivery.

## Next Direction

Keep the endpoint proposer as a candidate source, but add an opponent/referee
before materialization:

1. score endpoint edges against counter-target crops or nearest target
   competitors;
2. require source-to-target improvement over source-to-current-component support;
3. penalize large target components when the source is a singleton with only one
   support edge;
4. train/update the no-anchor side-effect gate with this endpoint false positive.

