# No-Anchor High-Mass Refutation From r47+r49 Best

Date: 2026-06-22

## Standing Best

The standing no-anchor DS1 best remains the side-effect blacklisted subpart combo
`32 -> 15 + 47 -> 2330`, moved 14 tracklets:

`IDF1 / HOTA / AssA = 0.657887 / 0.520944 / 0.535983`

The no-anchor global-id pair model target remains above 0.70, but the end-to-end
IDF1 target of 0.70 is still open.

## What Was Tested

Starting from the promoted `r47+r49` assignment, I generated higher-mass subpart
repair candidates with no production GT/anchors:

- base assignment: `/mnt/localssd/vlincs_reid_runs/no_anchor_sideeffect_blacklisted_subpart_search_20260622/combos/assignments/subpart_combo_r47_r49_14seq_assignments.csv`;
- proposer inputs: assignment CSV, DINO/weakmetric/SigLIP-fused tracklet
  features, component sizes, focus-video support, temporal conflicts, and the
  accumulated side-effect blacklist;
- high-mass proposer settings: minimum group size 10, maximum group size 34,
  seed similarity 0.30;
- selected candidates: 143 after blacklist and
  diversity filtering;
- canonical scorer: assignment CSV -> full submission zip -> density_simple ->
  `p005_area` detection/admission filter;
- GT use: evaluation only.

## High-Mass Top-6 Full Score

| edit | pool/rank | moved | focus | IDF1 | HOTA | AssA | delta | MCAM04 | MCAM03 Tc8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32->41 | dino_weak / 115 | 34 | 29 | 0.657887 | 0.520944 | 0.535983 | +0.000000 | 0.565570 | 0.629337 |
| 21->60 | dino_weak / 8 | 34 | 34 | 0.657196 | 0.520254 | 0.535554 | -0.000691 | 0.564480 | 0.629337 |
| 55->2330 | dino_weak / 97 | 34 | 34 | 0.656359 | 0.519460 | 0.534925 | -0.001528 | 0.562745 | 0.628997 |
| 26->24 | dino_weak / 75 | 34 | 34 | 0.656356 | 0.518802 | 0.533550 | -0.001531 | 0.562050 | 0.628015 |
| 28->31 | dino_weak / 107 | 34 | 33 | 0.656311 | 0.518755 | 0.533544 | -0.001576 | 0.562819 | 0.629337 |
| 49->19 | dino_weak / 117 | 34 | 34 | 0.656169 | 0.519565 | 0.535339 | -0.001718 | 0.564144 | 0.615713 |

Interpretation: direct 34-tracklet high-mass edits are too broad. `32 -> 41`
was score-neutral and does not improve the standing best. The other high-mass
edits dropped IDF1 by 0.000691 to 0.001718, typically through MCAM04 / MCAM03
side effects. These rows should become negative side-effect labels, not
promotions.

## 21 -> 60 Peel Ablation

The 34-tracklet `21 -> 60` edit was the strongest high-mass proposal but scored
only `0.657196`, so I tested whether smaller peels from the same source/target
family rescue the signal.

| size | rank | moved | focus | IDF1 | HOTA | AssA | delta | MCAM04 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 6 | 16 | 16 | 0.657877 | 0.520918 | 0.535962 | -0.000010 | 0.565798 |
| 22 | 1 | 22 | 20 | 0.657810 | 0.520837 | 0.535901 | -0.000077 | 0.565761 |
| 10 | 48 | 10 | 10 | 0.657498 | 0.520517 | 0.535657 | -0.000389 | 0.564684 |

The best peel is size 16, `rank06_subpart_s21_to60_16seq`, at IDF1 `0.657877`,
only `-0.000010` below the standing best. This means the source/target family is
not nonsense; the failure mode is over-broad selection and calibration at the
border. It is close enough to keep as a hard near-miss for a learned referee,
but not close enough to promote.

## Decision

No new best is promoted in iteration 91.

- current best stays `0.657887 / 0.520944 / 0.535983`;
- high-mass direct edits are recorded as negative side-effect evidence;
- `21 -> 60` size-16 peel is recorded as a near-tie hard negative;
- next direction should train or calibrate a subpart/high-mass side-effect
  referee from positives, ties, broad negatives, and near-miss peel labels before
  spending more full-score budget on large moved-mass candidates.

## Artifacts

- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/p005_fullscore_top6_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/p005_peel21_fullscore_summary.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/highmass_blacklisted_selection.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/peel21/size10_manifest.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/peel21/size16_manifest.json`
- `local_runs/remote_h100_test_3_20260622/no_anchor_highmass_from_r47r49_20260622/peel21/size22_manifest.json`
- remote run: `/mnt/localssd/vlincs_reid_runs/no_anchor_highmass_from_r47r49_20260622`
