# No-Anchor Mass-Features Diverse Full-Score Refutation

Date: 2026-06-21

## Purpose

The referee-pruned crossqueue branch showed that large component moves can look good under pair/proxy metrics while damaging canonical e2e. To check whether this was only a "large move" issue, I tested the more conservative `mass_features_diverse` manifest.

This manifest selected only three small candidates:

| selection rank | proxy predicted IDF1 | moved tracklets | intended family |
| ---: | ---: | ---: | --- |
| 1 | 0.658179 | 12 | `component:32->15` / replayed as `64->15` in current base |
| 2 | 0.658302 | 12 | `component:21->19` |
| 3 | 0.655254 | 8 | `component:21->0` |

## Full-Score Result

Only rank1 was directly replayable against the current h100-test-3 canonical base assignment.

| rank | replay status | pair F1 | full IDF1 | HOTA | AssA | notes |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | scored | 0.771045 | 0.652184 | 0.515838 | 0.531285 | 12-tracklet move, below standing best |
| 2 | not scored | 0.767329 | n/a | n/a | n/a | `target_component=19` missing in current base namespace |
| 3 | not scored | n/a | n/a | n/a | n/a | no `accepted_preview`; provenance lookup failed |

Standing best remains:

- IDF1 `0.655240`
- HOTA `0.518652`
- AssA `0.534359`

## Interpretation

The replayable small move still failed. This is weaker evidence than the crossqueue branch because only one row scored, but it is useful:

- reducing moved-tracklet count from hundreds to 12 is not sufficient;
- pair F1 can rise to `0.771045` without improving full IDF1;
- old manifests are not all replay-compatible with the current base namespace;
- ready-candidate execution needs a namespace/provenance gate before remote full-score time is spent.

## Decision

Do not keep promoting existing component-repair manifests blindly. The next branch should build a side-effect-calibrated admission proxy that explicitly models:

- moved-tracklet mass;
- source/target namespace compatibility;
- whether `accepted_preview` can be replayed;
- per-video side effects on MCAM04/06 and MCAM08;
- false-merge risk from high-impurity target components.

No goal completion is claimed. Model-side pair metrics remain above target, but verified no-anchor e2e is still below `0.70`.
