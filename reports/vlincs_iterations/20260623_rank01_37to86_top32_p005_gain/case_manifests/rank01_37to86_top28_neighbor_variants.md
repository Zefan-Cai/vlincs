# No-anchor component subset variants

- assignment: `local_runs/offline_no_anchor_split_probe_20260623/component_graph_fullscore/rank04_plus_small_fragment_combo/assignments/rank01_small_fragment_combo_combo_rank77_rank36_component_graph_small_fragment_assignments.csv`
- source -> target: `37 -> 86`
- rows: `18`

| rank | variant | subset | score | target sim | margin | reason |
|---:|---|---:|---:|---:|---:|---|
| 1 | `sideeffect_positive_videos_mcam04tc6_mcam03tc8` | 59 | 0.492660 | 0.467436 | -0.214767 | post-hoc reviewer side-effect label: keep videos that gained in full p005 |
| 2 | `exclude_loss_videos_mcam03tc6_mcam08tc6` | 61 | 0.492645 | 0.462659 | -0.212589 | post-hoc reviewer side-effect label: exclude videos that dropped in full p005 |
| 3 | `top36_target_sim` | 36 | 0.480043 | 0.506039 | -0.218372 | top 36 source tracklets by target-centroid similarity |
| 4 | `camera_MCAM04` | 45 | 0.479161 | 0.476979 | -0.224866 | only source tracklets from camera MCAM04 |
| 5 | `top34_target_sim` | 34 | 0.477323 | 0.509064 | -0.220843 | top 34 source tracklets by target-centroid similarity |
| 6 | `top32_target_sim` | 32 | 0.473116 | 0.512141 | -0.230924 | top 32 source tracklets by target-centroid similarity |
| 7 | `top30_target_sim` | 30 | 0.470675 | 0.514752 | -0.231497 | top 30 source tracklets by target-centroid similarity |
| 8 | `top28_target_sim` | 28 | 0.468189 | 0.517481 | -0.229192 | top 28 source tracklets by target-centroid similarity |
| 9 | `top26_target_sim` | 26 | 0.464501 | 0.519967 | -0.230154 | top 26 source tracklets by target-centroid similarity |
| 10 | `top24_target_sim` | 24 | 0.461320 | 0.522231 | -0.226212 | top 24 source tracklets by target-centroid similarity |
| 11 | `top22_target_sim` | 22 | 0.457719 | 0.524537 | -0.224888 | top 22 source tracklets by target-centroid similarity |
| 12 | `camera_MCAM03` | 17 | 0.413488 | 0.434695 | -0.158532 | only source tracklets from camera MCAM03 |
| 13 | `video_MCAM03_Tc8` | 14 | 0.398448 | 0.436760 | -0.182309 | only source tracklets from video vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 |
| 14 | `video_MCAM03_Tc6` | 3 | 0.336227 | 0.425058 | -0.047574 | only source tracklets from video vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 |
| 15 | `positive_attach_margin` | 1 | 0.336153 | 0.503127 | 0.005386 | move only rows whose target similarity exceeds source-centroid similarity |
| 16 | `camera_MCAM08` | 5 | 0.331372 | 0.378438 | -0.087117 | only source tracklets from camera MCAM08 |
| 17 | `camera_MCAM06` | 1 | 0.242987 | 0.382509 | -0.123050 | only source tracklets from camera MCAM06 |
| 18 | `camera_MCAM00` | 1 | 0.173863 | 0.261014 | -0.173604 | only source tracklets from camera MCAM00 |
