# No-Anchor Opponent Scheduler Full-Score Refutation

Date: 2026-06-21

## Decision

- Reject all three full-scored scheduler candidates.
- Promote `countertarget_verdict == accept` as a hard admission gate before scheduler/full-score spend.
- Keep standing best unchanged: IDF1 `0.655378` / HOTA `0.518798` / AssA `0.534546`.

## Full-Score Results

| rank | edit | moved | pair F1 | proxy IDF1 | opponent | risk | full IDF1 | HOTA | AssA | delta vs best |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `40->21` | `8` | `0.770741` | `0.874452` | `reject_countertarget` | `0.638` | `0.653709` | `0.517528` | `0.532975` | `-0.001669` |
| 2 | `24->38` | `2` | `0.769857` | `0.874804` | `reject_countertarget` | `0.886` | `0.653305` | `0.517168` | `0.532825` | `-0.002073` |
| 3 | `15->14` | `8` | `0.767572` | `0.859148` | `reject_countertarget` | `1.000` | `0.652473` | `0.516075` | `0.531692` | `-0.002905` |

## Worst Per-Video Deltas Vs Standing Density Best

### Rank 1 `40->21`
| video | candidate IDF1 | standing IDF1 | delta |
| --- | ---: | ---: | ---: |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `0.766899` | `0.769440` | `-0.002541` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc6` | `0.688387` | `0.690820` | `-0.002433` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `0.626660` | `0.628615` | `-0.001955` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `0.704148` | `0.705985` | `-0.001837` |

### Rank 2 `24->38`
| video | candidate IDF1 | standing IDF1 | delta |
| --- | ---: | ---: | ---: |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `0.766899` | `0.769440` | `-0.002541` |
| `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6` | `0.559050` | `0.561069` | `-0.002019` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `0.626660` | `0.628615` | `-0.001955` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `0.704148` | `0.705985` | `-0.001837` |

### Rank 3 `15->14`
| video | candidate IDF1 | standing IDF1 | delta |
| --- | ---: | ---: | ---: |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc6` | `0.599366` | `0.608378` | `-0.009012` |
| `vlincs_MS01_MC0001_MCAM03_2024-03-Tc8` | `0.620441` | `0.628615` | `-0.008174` |
| `vlincs_MS01_MC0001_MCAM06_2024-03-Tc8` | `0.702533` | `0.705985` | `-0.003452` |
| `vlincs_MS01_MC0001_MCAM08_2024-03-Tc6` | `0.766899` | `0.769440` | `-0.002541` |

## Hard-Gate Check

- input rows: `100`
- raw edges: `100`
- kept edges with `--require-countertarget-accept`: `0`
- emitted rows: `0`

## Interpretation

- The scheduler/proxy model was strongly over-optimistic on rows with countertarget rejection.
- The negative result is useful because it converts the opponent audit from report-only evidence into an executable admission gate.
- The next structural pivot should generate candidates under the hard gate first, then score only rows with accepted opponent evidence or use split/quarantine actions where countertarget relink evidence is not applicable.

## Artifacts

- `local_runs/no_anchor_temporal_clean_bridge_queue_opponent_scheduler_labelled_w0p016_20260621.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/manifest_assignments.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/rank01_conflict_subcluster_reassign_candidate_search_augmented_candidates_assignments_full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/rank02_conflict_subcluster_reassign_candidate_search_augmented_candidates_assignments_full.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/rank03_conflict_subcluster_reassign_candidate_search_augmented_candidates_assignments_full.json`
- `local_runs/no_anchor_temporal_clean_bridge_queue_opponent_hardgate_20260621.json`
- `reports/no_anchor_temporal_clean_bridge_queue_opponent_hardgate_20260621.md`
- `/mnt/localssd/vlincs_reid_runs/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621`
