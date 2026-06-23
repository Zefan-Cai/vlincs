# No-anchor component subset variants

- assignment: `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv`
- source -> target: `37 -> 86`
- rows: `19`

| rank | variant | subset | score | target sim | margin | reason |
|---:|---|---:|---:|---:|---:|---|
| 1 | `top64_target_sim` | 64 | 0.499848 | 0.466962 | -0.200800 | top 64 source tracklets by target-centroid similarity |
| 2 | `top55_target_sim` | 55 | 0.496681 | 0.480902 | -0.203178 | top 55 source tracklets by target-centroid similarity |
| 3 | `sideeffect_positive_videos_mcam04tc6_mcam03tc8` | 59 | 0.492660 | 0.467436 | -0.214767 | post-hoc reviewer side-effect label: keep videos that gained in full p005 |
| 4 | `exclude_loss_videos_mcam03tc6_mcam08tc6` | 61 | 0.492645 | 0.462659 | -0.212589 | post-hoc reviewer side-effect label: exclude videos that dropped in full p005 |
| 5 | `top45_target_sim` | 45 | 0.488877 | 0.493443 | -0.211660 | top 45 source tracklets by target-centroid similarity |
| 6 | `top36_target_sim` | 36 | 0.480043 | 0.506039 | -0.218372 | top 36 source tracklets by target-centroid similarity |
| 7 | `camera_MCAM04` | 45 | 0.479161 | 0.476979 | -0.224866 | only source tracklets from camera MCAM04 |
| 8 | `top28_target_sim` | 28 | 0.468189 | 0.517481 | -0.229192 | top 28 source tracklets by target-centroid similarity |
| 9 | `top20_target_sim` | 20 | 0.453667 | 0.526905 | -0.221041 | top 20 source tracklets by target-centroid similarity |
| 10 | `top16_target_sim` | 16 | 0.442192 | 0.531781 | -0.225855 | top 16 source tracklets by target-centroid similarity |
| 11 | `top12_target_sim` | 12 | 0.430147 | 0.537125 | -0.214562 | top 12 source tracklets by target-centroid similarity |
| 12 | `top8_target_sim` | 8 | 0.414330 | 0.541953 | -0.182542 | top 8 source tracklets by target-centroid similarity |
| 13 | `camera_MCAM03` | 17 | 0.413488 | 0.434695 | -0.158532 | only source tracklets from camera MCAM03 |
| 14 | `video_MCAM03_Tc8` | 14 | 0.398448 | 0.436760 | -0.182309 | only source tracklets from video vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 |
| 15 | `video_MCAM03_Tc6` | 3 | 0.336227 | 0.425058 | -0.047574 | only source tracklets from video vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 |
| 16 | `positive_attach_margin` | 1 | 0.336153 | 0.503127 | 0.005386 | move only rows whose target similarity exceeds source-centroid similarity |
| 17 | `camera_MCAM08` | 5 | 0.331372 | 0.378438 | -0.087117 | only source tracklets from camera MCAM08 |
| 18 | `camera_MCAM06` | 1 | 0.242987 | 0.382509 | -0.123050 | only source tracklets from camera MCAM06 |
| 19 | `camera_MCAM00` | 1 | 0.173863 | 0.261014 | -0.173604 | only source tracklets from camera MCAM00 |
