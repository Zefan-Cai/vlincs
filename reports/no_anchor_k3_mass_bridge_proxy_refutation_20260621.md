# No-Anchor K3 Mass-Bridge Proxy Refutation

Date: 2026-06-21

## Verdict

Rejected. The candidate improves pair diagnostics but fails canonical end-to-end scoring:

| Run | IDF1 | HOTA | AssA | Pair F1 | Pair P | Pair R |
|---|---:|---:|---:|---:|---:|---:|
| Standing best before this run | 0.655378 | 0.518798 | 0.534546 | 0.775234 | 0.820504 | 0.734698 |
| K3 mass-bridge candidate | 0.653963 | 0.517938 | 0.533529 | 0.772364 | 0.819418 | 0.730421 |
| K3 base pair diagnostic | n/a | n/a | n/a | 0.769367 | 0.816518 | 0.727364 |

The no-GT bridge scorer found a visually plausible local repair, but the full DS1 opponent rejected it. This is a Deli AutoResearch-style negative result: the score drop is useful evidence, not a failed run to hide.

## What Was Tested

Starting point:

- Assignment: `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
- Current protocol standing best from this family: IDF1 `0.655378`, HOTA `0.518798`, AssA `0.534546`

Candidate proposer:

- Script: `kit/no_anchor_assignment_conflict_reassign_sweep.py`
- Ranking: `--candidate-edge-rank-by mass_bridge`, `--rank-by mass_bridge_proxy`
- Views: fused primary, posecolor, colorhist, DINO
- Anchor policy: `uses_anchors=false`
- GT policy: `uses_gt_for_training_or_anchors=false`; GT only appears in canonical evaluation.

## Selected Candidate

The top candidate moved one coherent source island:

| Field | Value |
|---|---|
| Source component | `21` |
| Source tracklet seqs | `2232, 2270, 2308, 2374, 2415, 2452, 2488, 2553` |
| Target component | `0` |
| Target support seqs | `2656, 2591, 5838` |
| Accepted reassignments | `1` |
| Moved tracklets | `8` |
| Mass-bridge proxy | `0.823978` |
| Source internal sim | `0.911467` |
| Source cross mean sim | `0.623840` |
| Target best / mean sim | `0.947318 / 0.940898` |
| Target min-view sim / view vote | `0.751997 / 1.000000` |

This is exactly the kind of local evidence that looks compelling: high target similarity, all-view agreement, a compact source island, and zero forbidden target pairs.

## Full-Score Outcome

Canonical full score:

| Video | IDF1 | HOTA | AssA |
|---|---:|---:|---:|
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.878694 | 0.814760 | 0.835905 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.827822 | 0.748689 | 0.784523 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688387 | 0.580446 | 0.623006 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.626660 | 0.508952 | 0.549346 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.560896 | 0.447978 | 0.493149 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 | 0.601047 | 0.639585 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 | 0.697946 | 0.727424 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.606895 | 0.514826 | 0.596222 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704148 | 0.590733 | 0.621206 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.766899 | 0.659505 | 0.678304 |

The bottlenecks remain the same: MCAM04 Tc6, MCAM06 Tc6, and MCAM03 Tc8. A single compact visual bridge does not materially fix those delivery errors.

## Interpretation

This closes the current "small constrained bridge after k3" hypothesis. Even after:

- k3 softcut split,
- no-GT forced-conflict state metadata,
- multiview visual agreement,
- mass-aware edge pre-ranking,
- and canonical full-score verification,

the candidate lands below the standing best. The failure mode is now clear: local visual consistency is not enough to certify a global identity merge/reassign, because the end-to-end score is dominated by broader false-split/false-merge and per-video delivery side effects.

## Next Structural Pivot

Do not continue tuning `mass_bridge_proxy` thresholds in this small-bridge family. The next direction should add an explicit opponent before export:

1. Visual identity verifier: compare source island against target component and nearest counter-targets with crop-level evidence, not only aggregate component means.
2. Detector-quality admission: estimate whether the moved island is likely a detection/tracklet-quality artifact that hurts HOTA/IDF1 even when ReID similarity is high.
3. High-mass false-split repair: prioritize identities with concentrated missing mass rather than compact conflict islands.

## Artifacts

- Summary: `local_runs/no_anchor_k3_mass_bridge_proxy_search_20260621_summary.json`
- Remote search JSON: `local_runs/no_anchor_k3_mass_bridge_proxy_search_20260621/search.json`
- Remote search CSV: `local_runs/no_anchor_k3_mass_bridge_proxy_search_20260621/search.csv`
- Candidate assignment: `local_runs/no_anchor_k3_mass_bridge_proxy_search_20260621/top_assignments.csv`
