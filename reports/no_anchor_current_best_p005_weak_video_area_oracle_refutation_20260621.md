# Weak-Video Area Oracle Refutation

Date: 2026-06-21

## Question

After the promoted no-anchor p0.5% per-video bbox-area admission rule, can we
gain more DS1 IDF1 by applying additional area pruning only on the weak videos?

This is an oracle diagnostic, not a production policy. It uses no anchors and
does not use GT labels as identity evidence. GT is used only to score the
candidate filters and select the eval-only upper bound.

Target videos:

- `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6`
- `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6`
- `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8`

## Result

The oracle selected `area_q=0` for all three weak videos. In other words, the
best rule is to keep the current p0.5% promoted submission unchanged.

Combined score:

| setting | IDF1 | HOTA | AssA | DetPr | DetRe | extra dropped rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current p0.5 submission | 0.655911 | 0.519311 | 0.534922 | 0.764814 | 0.574156 | 0 |
| weak-video area oracle | 0.655911 | 0.519311 | 0.534922 | 0.764814 | 0.574156 | 0 |

## Per-Video Area Sweep

| video | area q | area threshold | IDF1 | HOTA | AssA | dropped rows | unmatched FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MCAM03 Tc8 | 0 | 0.0 | 0.632479 | 0.512650 | 0.550240 | 0 | 7242 |
| MCAM03 Tc8 | 0.005 | 2296.0 | 0.632138 | 0.512090 | 0.549632 | 546 | 7057 |
| MCAM03 Tc8 | 0.010 | 2625.0 | 0.631829 | 0.511648 | 0.549262 | 1090 | 6952 |
| MCAM03 Tc8 | 0.020 | 3124.0 | 0.631377 | 0.510800 | 0.548320 | 2184 | 6672 |
| MCAM04 Tc6 | 0 | 0.0 | 0.564234 | 0.451124 | 0.496566 | 0 | 37372 |
| MCAM04 Tc6 | 0.005 | 1716.0 | 0.564122 | 0.451061 | 0.496684 | 3243 | 36521 |
| MCAM04 Tc6 | 0.010 | 1972.0 | 0.563760 | 0.450702 | 0.496485 | 6495 | 35826 |
| MCAM04 Tc6 | 0.020 | 2310.0 | 0.562992 | 0.450076 | 0.496405 | 13082 | 34362 |
| MCAM06 Tc6 | 0 | 0.0 | 0.609699 | 0.517698 | 0.599722 | 0 | 3094 |
| MCAM06 Tc6 | 0.005 | 2310.0 | 0.609574 | 0.517569 | 0.599690 | 173 | 3022 |
| MCAM06 Tc6 | 0.010 | 2964.0 | 0.609294 | 0.517245 | 0.599415 | 350 | 2943 |
| MCAM06 Tc6 | 0.020 | 3739.7 | 0.608263 | 0.515939 | 0.598108 | 701 | 2849 |

## Interpretation

This closes the extra area-admission branch. The current p0.5% detector
postfilter already captures the useful tiny-box false-positive pruning. Extra
weak-video area pruning improves DetPr and reduces unmatched FP, but it loses
DetRe and IDF1 on every target video.

The failure mode is therefore not "more detector rows should be dropped"; it is
missing or misresolved identity evidence in weak videos, especially MCAM04 Tc6.

## Next Direction

Stop spending full-score budget on additional confidence/area row filters. The
next structural pivot should be one of:

1. detector/tracklet regeneration for MCAM04 Tc6 and MCAM06 Tc6;
2. crop-level evidence extraction for weak-video tracklets, then retrain the
   no-anchor global-ID model or pair verifier;
3. a side-effect critic with per-video detector recall/false-positive features,
   not row-summary-only features.

## Follow-Up Data Recon

Remote DS1 inputs are available under
`/mnt/localssd/vlincs_reid_data/Box/VLINCS_Performer/`.

The weak-video rows show a coverage/fragmentation gap between GT tracklets and
the real BoTSORT pipeline:

| video | source | rows | tracklets | mean length | median length | short <=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MCAM03 Tc8 | GT tracklets | 134125 | 286 | 468.97 | 200.5 | 2 |
| MCAM03 Tc8 | BoTSORT | 86857 | 261 | 332.79 | 197.0 | 3 |
| MCAM04 Tc6 | GT tracklets | 776465 | 473 | 1641.58 | 309.0 | 3 |
| MCAM04 Tc6 | BoTSORT | 338341 | 689 | 491.06 | 131.0 | 13 |
| MCAM06 Tc6 | GT tracklets | 52310 | 73 | 716.58 | 364.0 | 0 |
| MCAM06 Tc6 | BoTSORT | 34532 | 68 | 507.82 | 345.5 | 1 |

MCAM04 is the clearest case: the real pipeline has less than half the GT-row
coverage while producing more fragmented tracklets. That matches the oracle
result: dropping more boxes cannot solve the weak-video bottleneck because it
mostly removes recall. The next experiment should compare global-ID performance
under GT-tracklet upper-bound vs BoTSORT real pipeline and then train/score a
tracklet regeneration or crop-evidence model for this gap.

Artifacts:

- JSON: `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_p005_weak_video_area_oracle_20260621/oracle.json`
- Log: `local_runs/remote_h100_test_3_20260621/no_anchor_current_best_p005_weak_video_area_oracle_20260621/run.log`
- Remote ZIP: `/mnt/localssd/vlincs_reid_runs/no_anchor_current_best_p005_weak_video_area_oracle_20260621/oracle.zip`
