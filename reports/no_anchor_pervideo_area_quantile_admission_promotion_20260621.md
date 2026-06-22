# Per-Video Area Quantile Admission

Date: 2026-06-21

## Result

After the global `area >= 800` detector admission rule improved the no-anchor e2e score, I tested a video-adaptive version: for each video, drop the lowest-q fraction of detections by bbox area. This changes only delivered detections, not identity assignment.

| policy | IDF1 | HOTA | AssA | dropped rows |
| --- | ---: | ---: | ---: | ---: |
| global area800 | 0.655855 | 0.519271 | 0.534837 | 1404 |
| per-video p0.1% area | 0.655859 | 0.519273 | 0.534840 | 1520 |
| per-video p0.2% area | 0.655872 | 0.519289 | 0.534871 | 3023 |
| per-video p0.3% area | 0.655887 | 0.519303 | 0.534897 | 4531 |
| per-video p0.5% area | 0.655911 | 0.519311 | 0.534922 | 7603 |
| per-video p0.75% area | 0.655891 | 0.519258 | 0.534888 | 11415 |
| per-video p1.0% area | 0.655825 | 0.519147 | 0.534791 | 15219 |

Promoted candidate: `density_simple_plus_pervideo_area_p005`.

Remote artifact:

- JSON: `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_pervideo_area_quantile_fine_20260621/pervideo_area_quantile_fine.json`
- ZIP: `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_pervideo_area_quantile_fine_20260621/best_pervideo_area_quantile_fine.zip`

## Per-Video Metrics For p0.5%

| video | IDF1 | HOTA | AssA | dropped rows |
| --- | ---: | ---: | ---: | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.879281 | 0.815354 | 0.836352 | 210 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.828550 | 0.749705 | 0.785602 | 275 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.690913 | 0.582450 | 0.624738 | 902 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.628528 | 0.510443 | 0.550903 | 549 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.562087 | 0.448184 | 0.493260 | 3266 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.711374 | 0.600973 | 0.639418 | 80 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.793298 | 0.699897 | 0.729415 | 95 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.609022 | 0.517275 | 0.599722 | 177 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.705819 | 0.591803 | 0.621927 | 188 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.769732 | 0.662040 | 0.680704 | 1861 |

## Interpretation

The curve improves from p0.1% through p0.5%, then falls at p0.75% and p1.0%. That suggests this rule is pruning detector false positives, but only in the tiny-box tail; beyond that point the rule starts losing useful detections.

This remains a small postfilter. It moves the best no-anchor e2e IDF1 from `0.655855` to `0.655911`, not toward 0.70 by itself.

The rule uses only bbox geometry at inference. The specific p0.5% quantile was chosen by this ablation, so it should be treated as a promoted production candidate and frozen before any future validation slice.

## Next Direction

Detector admission is useful but saturated. The next route should keep p0.5% per-video area admission as a postfilter and change the identity evidence itself:

- Focus on MCAM04 Tc6 and MCAM06 Tc6, where p0.5% did not solve the low IDF1 bottleneck.
- Generate new tracklet-level evidence rather than rearranging existing components.
- Prefer candidate recall expansion or detector-level tracklet regeneration over more confidence/area pruning.
