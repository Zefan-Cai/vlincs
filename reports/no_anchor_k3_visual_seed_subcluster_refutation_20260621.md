# No-Anchor K3 Visual-Seed Subcluster Refutation

Date: 2026-06-21

## Verdict

Rejected. Reusing the earlier no-GT visual edge decisions as k3 subcluster seeds made the canonical score worse:

| Run | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| Standing best before this run | 0.655378 | 0.518798 | 0.534546 |
| K3 visual-seed subcluster sanity run | 0.649982 | 0.512746 | 0.527745 |

This is a stronger refutation than the mass-bridge result: even high-confidence visual positives can hurt after the assignment namespace and component structure change.

## What Was Tested

Starting point:

- Assignment: `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- Feature: `/mnt/localssd/vlincs_reid_runs/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz`

Visual referee input:

- Decisions: `/mnt/localssd/vlincs_reid_runs/codex_visual_edge_decisions_split_t040_m16_20260619.json`
- Montage metadata: `/mnt/localssd/vlincs_reid_runs/vlm_edge_montages_split_t040_m16_20260619.json`
- `uses_gt_for_decision=false`

Single sanity configuration:

- confidence threshold: `0.94`
- seed sim threshold: `0.92`
- max fraction per source side: `0.03`
- max selected per side: `8`
- max groups: `4`
- pair grid metrics skipped with the new `--skip-pair-grid-metrics` flag to avoid repeated O(n^2) pair metric computation.

## Selected Groups

The run accepted 4 visual seed groups and moved 33 tracklets:

| Edge | Size | Source label | Target label | Confidence | Notes |
|---:|---:|---:|---:|---:|---|
| 16 | 7 | 28 | 59 | 0.96 | high-sim hoodie seed plus tiny fragment |
| 19 | 9 | 3 | 45 | 0.95 | patterned coat seed plus tiny fragment |
| 27 | 8 | 38 | 80 | 0.95 | blonde white/black outfit seed plus tiny fragment |
| 17 | 9 | 2 | 72 | 0.94 | pink sweater seed plus tiny fragment |

All four groups are visually plausible, and each large-side seed selection had very high local similarity (`mean_selected_sim` around 0.994-0.997). The failure is therefore not that the visual matcher is obviously poor; it is that visual-positive seed surgery is not enough to preserve the global delivery objective.

## Per-Video Outcome

| Video | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.872224 | 0.805083 | 0.827126 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.825779 | 0.745172 | 0.780515 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688387 | 0.580446 | 0.623006 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.617753 | 0.496481 | 0.533281 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.556556 | 0.442784 | 0.487185 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 | 0.601047 | 0.639585 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 | 0.697946 | 0.727424 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.597054 | 0.499064 | 0.573422 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.699006 | 0.584775 | 0.615582 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.763571 | 0.654708 | 0.673338 |

The main losses are on already weak videos such as MCAM04 Tc6, MCAM06 Tc6, and MCAM03 Tc8. Visual seed repair increases delivery fragility rather than repairing those slices.

## Tooling Note

`kit/no_anchor_visual_seed_subcluster_merge.py` now has `--skip-pair-grid-metrics`. This keeps default behavior unchanged, but allows single-config or scarce-budget full-score sanity checks without spending minutes on repeated pair metric computation over all tracklets.

## Interpretation

This closes direct reuse of the 2026-06-19 visual decisions as a k3 verifier. The next visual verifier must be case-specific:

- regenerate counter-target evidence under the current k3 assignment,
- compare a proposed source island against nearest non-target alternatives,
- include detector-quality and per-video side-effect checks,
- and only then export a surgical assignment.

## Artifacts

- Summary: `local_runs/no_anchor_k3_visual_seed_subcluster_single_fast_20260621_summary.json`
- Remote JSON: `local_runs/no_anchor_k3_visual_seed_subcluster_single_fast_20260621/search.json`
- Remote CSV: `local_runs/no_anchor_k3_visual_seed_subcluster_single_fast_20260621/search.csv`
- Candidate assignment: `local_runs/no_anchor_k3_visual_seed_subcluster_single_fast_20260621/top_assignments.csv`
