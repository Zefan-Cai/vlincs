# No-Anchor Self-Play Critic Refutation Of Old Queues

Date: 2026-06-21

## Summary

This run applied the Deli AutoResearch/self-play protocol to VLINCS no-anchor
global-ID research:

- proposer: old assignment-summary, adversary bridge, crossqueue portfolio, and
  learned-proxy single-edge queues;
- referee/opponent: post-hoc canonical DS1 full-score plus no-GT side-effect
  risk;
- gate: only full-score can promote a delivery candidate; pair/proxy wins are
  critic labels, not completion evidence.

Outcome: the old 20260620 candidate queues are now rejected as production paths.
They generate plausible pair/proxy scores but do not beat the standing
production best `IDF1/HOTA/AssA = 0.655378 / 0.518798 / 0.534546`.

## Deli Distillation Applied

The relevant Deli rules were:

- persist state and evidence in files, not chat;
- ready means execute: score prepared candidates instead of leaving them as
  claims;
- score drops are useful refutations;
- keep proposer, referee, opponent, and gate separated;
- when stale, change the candidate space or admission logic, not just thresholds.

This run implemented that directly by turning failed full-score runs into
`local_runs/no_anchor_selfplay_fullscore_labels_20260621.json` and adding:

- `--fullscore-label` to `kit/no_anchor_fullscore_scheduler.py`;
- `--side-effect-risk-weight` to penalize risky multi-edge / large-target /
  high-move candidates before spending full-score budget.

## Full-Score Critic Labels

All rows are no-anchor production candidates. GT is used only by the canonical
evaluator after an assignment has already been materialized.

| label | family | predicted/proxy context | full IDF1 | HOTA | AssA | unmatched FP | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `rank01_local_pervideo_balanced` | assignment summary selector | proxy top | `0.653177` | `0.517094` | `0.532626` | `108661` | reject |
| `rank02_adversary_25_to_31` | adversary bridge | moved 156 tracklets | `0.653177` | `0.517094` | `0.532626` | `108661` | reject as no-op for e2e |
| `rank01_crossqueue_singleedge68_localized_island` | crossqueue portfolio | predicted `0.668110`; moved 64 | `0.650203` | `0.513940` | `0.529860` | `112280` | reject |
| `rank01_hub_bridge_portfolio_selfplay_strict` | hub-bridge portfolio | predicted `0.662663`; moved 40 | `0.652320` | `0.516260` | `0.531958` | `112922` | reject |
| `rank01_sideeffect_21to19` | single-edge `21->19` | predicted `0.656447`; moved 12 | `0.652468` | `0.516281` | `0.532087` | `116221` | reject |

The most important pattern is that even low-risk old-queue single-edge repair
still underperforms once replayed on the canonical base.  The issue is not only
large multi-edge side effects; the old candidate space itself is stale.

## Scheduler Ablation

Artifacts:

- `local_runs/no_anchor_full_proxy_training_audit_selfplay_labels_v3_20260621.json`
- `local_runs/no_anchor_full_proxy_selfplay_ridge_model_v3_20260621.json`
- `local_runs/no_anchor_fullscore_scheduler_selfplay_sideeffect_w0_20260621.json`
- `local_runs/no_anchor_fullscore_scheduler_selfplay_sideeffect_w0p001_20260621.json`
- `local_runs/no_anchor_fullscore_scheduler_selfplay_sideeffect_w0p002_20260621.json`
- `local_runs/no_anchor_fullscore_scheduler_selfplay_sideeffect_w0p004_20260621.json`

Side-effect risk changed the scheduler ranking as intended:

| side-effect weight | rank1 family | rank1 risk | interpretation |
| ---: | --- | ---: | --- |
| `0` | hub-bridge portfolio `0+32+4 -> 15+6` | `4.0` | old proxy still prefers risky portfolio |
| `0.001` | hub-bridge portfolio `0+32+4 -> 15+6` | `4.0` | partially penalized, still too high |
| `0.002` | single-edge `21->19` | `0.0` | high-risk portfolio displaced |
| `0.004` | single-edge `21->19` | `0.0` | same top, stronger quarantine |

Then the top low-risk row `21->19` was full-scored and rejected at `0.652468`.
So the side-effect guard is useful as an admission tool, but it does not rescue
the old queue.

## Diagnosis

This is a clean self-play failure:

- pair/proxy scores remain around `0.767-0.771`;
- full-score drops because edits increase unmatched FP or harm low-score videos;
- hub/crossqueue portfolios overfit to visual support and underprice target
  namespace and temporal side effects;
- the old base queue was built around
  `no_anchor_softcut_then_softoverlap_best_assignments_20260619`, while the
  current standing best comes from the later
  `no_anchor_softcut_split_k3_red010_fullscore_20260621` plus density filtering.

The next branch should stop spending budget on old-queue replay.  It should use
the current k3/density-filter best as the state to explain.

## Next Direction

Move from "repair an old candidate queue" to "explain the current best failure":

1. Use current standing-best artifacts:
   - `/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_split_k3_red010_fullscore_20260621/assignments.csv`
   - `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_red010_fullscore/full.json`
   - `local_runs/remote_h100_test_3_20260621/no_anchor_softcut_split_k3_red010_fullscore/density_filter.json`
2. Run per-video/per-component error concentration on MCAM04 and MCAM06, because
   these are still the low-score videos.
3. Generate candidates only inside the current k3 namespace:
   detector-quality quarantine, density admission, and tiny-fragment reassignment
   with explicit same-video temporal contradiction checks.
4. Treat every old-queue family above as a negative critic label for future
   scheduler admission.

The global-ID pair model remains above target, but end-to-end delivery remains
below target.  Current verified status:

- pair F1/P/R: `0.775234 / 0.820504 / 0.734698`;
- best e2e IDF1/HOTA/AssA: `0.655378 / 0.518798 / 0.534546`;
- target e2e IDF1: `0.700000`.
