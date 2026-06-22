# No-Anchor AutoResearch Update: Temporal And Target-Fragment Refutation

Date: 2026-06-21

## Source Distillation

The Deli AutoResearch announcement is useful here as an operating protocol, not as a new identity model.  The parts that map directly onto VLINCS are:

- Persist state to files and treat conversation context as disposable.
- Ready means execute: once a no-GT candidate passes admission, materialize and score it.
- Separate proposer and opponent: proxy pair metrics propose candidates, canonical e2e HOTA/IDF1 can veto them.
- Honest score drops are findings.  They should update the next search constraints rather than disappear from the log.
- After repeated stale iterations, pivot the structure of the search, not just thresholds.

For this no-anchor global-id loop, the immediate translation is:

- Pair-F1 is a proposer signal only.
- Full e2e score is the opponent/reviewer.
- A candidate that slightly improves pair-F1 but drops e2e IDF1 is a negative result.
- Repeated failures in small-fragment relinks should force a structural pivot away from same-family threshold tuning.

## Standing Best

Current standing no-anchor e2e best remains:

| source | IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: |
| no_anchor_softcut_split_k3_red010_fullscore_20260621 density-filter best | 0.655378 | 0.518798 | 0.534546 |

Current global-id pair model target is already above 70:

| metric | value |
| --- | ---: |
| pair F1 | 0.775234 |
| pair precision | 0.820504 |
| pair recall | 0.734698 |

The active gap is still e2e IDF1 > 0.70.

## Experiment 1: MCAM04/08 Local Temporal Continuity

Goal: test whether the weakest videos have missed local temporal links that can be recovered without anchors.

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_mcam04_08_temporal_policy_probe_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_mcam04_08_temporal_ultrawide_probe_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_mcam04_08_temporal_ultrawide_fullscore_20260621/top_policy_full.json`

Findings:

| setting | accepted edges | proxy pair F1 | e2e IDF1 | HOTA | AssA |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict temporal policy | 0 | 0.769367 | not scored | not scored | not scored |
| ultrawide temporal policy | 1 | 0.769367 | 0.653210 | 0.517030 | 0.532678 |

Per-video IDF1 for the ultrawide full-score candidate:

| video | IDF1 |
| --- | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.878694 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.827822 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688387 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.626660 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.559050 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.606895 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704148 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.766899 |

Decision: reject.  Strict temporal continuity has no mass.  Widening enough to find an edge still scores below the standing best, so this is not the next productive axis.

## Experiment 2: Target-Fragment Softcut Selection

Goal: localize tiny target fragments that should be attached back to large components using the current softcut edge table, while sorting by a no-GT `selection_score` instead of eval pair-F1.

Code change:

- `kit/no_anchor_edge_table_target_repair_sweep.py` now supports `--sort-key selection_score`, with `selection_score`, `selection_score_mean`, and `selection_weighted_evidence` recorded in sweep rows.

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_target_fragment_softcut_narrow_selection_probe_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_target_fragment_softcut_top_selection_fullscore_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_target_fragment_softcut_top_selection_fullscore_20260621/top_selection_full.json`
- Remote full submission zip: `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_target_fragment_softcut_top_selection_fullscore_20260621/top_selection_full.zip`

Top no-GT selection:

| selected edges | localized targets | selection score | proxy pair F1 | pair precision | pair recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 4.966963 | 0.769406 | 0.816527 | 0.727428 |

Canonical full-score:

| IDF1 | HOTA | AssA |
| ---: | ---: | ---: |
| 0.653220 | 0.517042 | 0.532691 |

Per-video IDF1 for the target-fragment full-score candidate:

| video | IDF1 |
| --- | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.878694 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.827822 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688393 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.626660 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.559050 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.606895 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704189 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.766931 |

Decision: reject.  The no-GT target-fragment rule finds plausible high-confidence small-component merges and slightly improves proxy pair-F1, but canonical e2e IDF1 drops relative to the standing best.

## Diagnosis

## Experiment 3: Conservative Source-Side Conflict Quarantine

Goal: test the structural pivot suggested after target-fragment failure: split only existing cannot-link conflict nodes that are visual/clothing outliers relative to their predicted component.

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_clothing_conflict_quarantine_current_probe_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_clothing_conflict_quarantine_current_probe_20260621/result.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_clothing_conflict_quarantine_1node_fullscore_20260621/result.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_k3_clothing_conflict_quarantine_1node_fullscore_20260621/top_full.json`
- Remote full submission zip: `/mnt/localssd/vlincs_reid_runs/no_anchor_k3_clothing_conflict_quarantine_1node_fullscore_20260621/top_full.zip`

Probe result:

| threshold | min component size | split components | split nodes | proxy pair F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.45 | 32 | 0 | 0 | 0.769367 |
| 0.55 | 32 | 1 | 1 | 0.769367 |
| 0.65 | 32 | 6 | 6 | 0.769130 |

Canonical full-score for the most conservative non-noop row:

| split nodes | IDF1 | HOTA | AssA |
| ---: | ---: | ---: | ---: |
| 1 | 0.653242 | 0.517070 | 0.532728 |

Per-video IDF1:

| video | IDF1 |
| --- | ---: |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc6 | 0.878694 |
| vlincs_MS01_MC0001_MCAM00_2024-03-Tc8 | 0.827822 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc6 | 0.688387 |
| vlincs_MS01_MC0001_MCAM03_2024-03-Tc8 | 0.626660 |
| vlincs_MS01_MC0001_MCAM04_2024-03-Tc6 | 0.559050 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc6 | 0.710965 |
| vlincs_MS01_MC0001_MCAM05_2024-03-Tc8 | 0.791599 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc6 | 0.606895 |
| vlincs_MS01_MC0001_MCAM06_2024-03-Tc8 | 0.704148 |
| vlincs_MS01_MC0001_MCAM08_2024-03-Tc6 | 0.767037 |

Decision: reject.  The 1-node quarantine is safe but too small.  More aggressive thresholds begin to hurt proxy recall, so this branch should not be expanded as threshold tuning.

## Diagnosis

The three probes converge to the same conclusion:

- Low-mass local continuity and target-fragment repair do not move the e2e objective.
- Conservative source-side quarantine is also too low-mass.
- Proxy pair-F1 can be locally positive while e2e IDF1 is negative.
- The weakest slices remain MCAM04 and MCAM06/03, but single-edge or tiny-fragment edits are too small and can disturb the delivered ID namespace.

AutoResearch-style takeaway: this is a structural stall, not a threshold problem.  The next direction should change the game being played, for example:

1. Build an opponent/critic that predicts e2e side effects before full-score, using density-filter deltas, component-size shifts, and per-video namespace churn.
2. Mine positives at component/subcomponent scale, not single-tracklet scale.
3. Treat conservative quarantine as a safety feature, not a primary score-improvement axis.

## Side-Effect Critic Refresh

The three new refutations were converted into full-score critic labels:

- `mcam04_08_temporal_ultrawide_1edge`
- `target_fragment_softcut_selection_5edges`
- `clothing_conflict_quarantine_1node`

Artifacts:

- `local_runs/no_anchor_autoresearch_new_refutation_labels_20260621.json`
- `local_runs/no_anchor_full_proxy_training_audit_autoresearch_labels_v4_20260621.json`
- `local_runs/no_anchor_full_proxy_selfplay_ridge_model_v4_20260621.json`
- `reports/no_anchor_full_proxy_training_audit_autoresearch_labels_v4_20260621.md`

The compact ridge side-effect model now has 80 deduplicated full-score rows and 45 compact features, but its LOOCV score is still weak:

| rows | features | LOOCV corr | MAE | RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 45 | -0.032403 | 0.012658 | 0.060574 |

Interpretation: the current row-summary feature table is not enough to predict full e2e side effects.  It is useful as a label bank and refutation memory, but not reliable enough to be a production ranker.  The next critic needs stronger opponent evidence: counter-target crop pairs, per-video namespace delta features, and detector-quality / unmatched-FP predictors.

## Status

No-anchor condition preserved:

- `uses_anchors=false`
- `uses_gt_for_training_or_anchors=false`
- GT used only for post-hoc evaluation/audit.

Goal status:

- Global-id model/pair metrics: above 70.
- End-to-end pipeline: still below 70; best unchanged at IDF1 0.655378.
