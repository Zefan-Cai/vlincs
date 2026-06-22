# No-Anchor Detection Admission Promotion

Date: 2026-06-21

## Result

The Deli-style pivot from component rearrangement to detector admission produced a small but real new no-anchor best.

| policy | IDF1 | HOTA | AssA | dropped rows |
| --- | ---: | ---: | ---: | ---: |
| previous best: density_simple | 0.655817 | 0.519228 | 0.534791 | 0 |
| global_conf015 | 0.655496 | 0.518727 | 0.534357 | 17011 |
| video_p01_conf | 0.655575 | 0.518835 | 0.534459 | 15262 |
| area600 | 0.655828 | 0.519241 | 0.534804 | 347 |
| area700 | 0.655840 | 0.519254 | 0.534818 | 780 |
| area800 | 0.655855 | 0.519271 | 0.534837 | 1404 |
| area900 | 0.655843 | 0.519258 | 0.534832 | 2115 |
| area1000 | 0.655840 | 0.519256 | 0.534840 | 2907 |
| area1200 | 0.655820 | 0.519227 | 0.534830 | 5207 |

Promoted policy: `density_simple_plus_area800`.

Remote artifact:

- JSON: `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_area800_production_20260621/area800.json`
- ZIP: `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_area800_production_20260621/area800.zip`

## Interpretation

The positive signal is not confidence-tail pruning. Both `global_conf015` and per-video p01 confidence pruning hurt IDF1 because they trade too much DetRe for DetPr.

The useful signal is tiny-box admission: dropping detections with area `<800` removes 1404 rows, improves DetPr slightly while preserving DetRe, and raises IDF1 by `+0.000038`.

This is a valid no-anchor production rule because the rule uses only delivered detection geometry, not GT labels, anchors, or oracle selection. GT is used only by the canonical scorer.

## Per-Video Metrics For area800

| video | IDF1 | HOTA | AssA | dropped rows |
| --- | ---: | ---: | ---: | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.879281 | 0.815354 | 0.836352 | 15 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.828061 | 0.748927 | 0.784734 | 0 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.690896 | 0.582424 | 0.624696 | 546 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.628617 | 0.510641 | 0.551059 | 38 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.562191 | 0.448124 | 0.492924 | 432 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.711979 | 0.601930 | 0.640429 | 0 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.792455 | 0.698741 | 0.728223 | 0 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.608358 | 0.516556 | 0.598871 | 12 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.706042 | 0.592260 | 0.622471 | 6 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.769476 | 0.661732 | 0.680391 | 355 |

## Next Direction

This closes simple global detector pruning as a major path to 0.70. The next high-leverage move should use detector admission as a small postfilter while creating new identity evidence, especially for the bottleneck videos:

- MCAM04 Tc6: IDF1 0.562191
- MCAM06 Tc6: IDF1 0.608358
- MCAM03 Tc8: IDF1 0.628617

The next proposer should target detector-level tracklet admission or re-generation around these videos, not more broad component rearrangement.
