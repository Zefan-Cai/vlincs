# No-Anchor Temporal-Clean Bridge Queue Refutation

Date: 2026-06-21

## Verdict

Rejected as a production candidate family.

The Deli-style structural pivot was to stop rescoring old mass-bridge edges and
instead generate a fresh queue after skipping the first 200 mass-bridge edge
families. The queue still failed the visual+temporal counter-target opponent:

| Check | Value |
|---|---:|
| generated candidate rows | 100 |
| audited candidate edges | 100 |
| accepted edges | 0 |
| reject_countertarget | 100 |
| weighted margin > 0 | 11 |
| rank vote >= 0.75 | 2 |
| margin vote >= 0.50 | 0 |
| max weighted margin | 0.074511 |
| median weighted margin | -0.068839 |

This is a useful negative result. The old visual-bridge family is saturated
under counter-target checking: visually plausible source islands do not have
enough multi-view margin against alternative components.

## Best Full-Scored Candidate

The proposer still full-scored its rank-1 row during candidate search. It was
negative relative to the standing best:

| Run | IDF1 | HOTA | AssA | Decision |
|---|---:|---:|---:|---|
| Standing best, k3 softcut + density_oracle_lite | 0.655378 | 0.518798 | 0.534546 | keep |
| Temporal-clean queue rank1 raw full-score | 0.653339 | 0.517177 | 0.532767 | reject |

Rank1 moved 12 tracklets:

- source component: `19`
- target component: `6`
- source seqs: `138, 580, 983, 7086, 7187, 7409, 8465, 8517, 8609, 8703, 8755, 8814`
- target support seqs: `8860, 1141, 641`
- pair F1: `0.770451`, which is higher than base pair F1 `0.769367`

This repeats the core lesson: small pair-metric gains do not imply forced-output
global-ID delivery gains.

## Why No Edges Passed

The temporal opponent did not need to reject most rows; the counter-target
visual referee already rejected all 100.

Representative rejected rows:

| source | target | weighted margin | rank vote | margin vote | temporal overlap rejects |
|---:|---:|---:|---:|---:|---:|
| 19 | 6 | -0.035760 | 0.00 | 0.00 | 0 |
| 32 | 15 | 0.010745 | 0.25 | 0.25 | 0 |
| 6 | 19 | -0.052245 | 0.00 | 0.00 | 0 |
| 21 | 0 | 0.074511 | 0.75 | 0.25 | 0 |

The strongest remaining old-style bridge, `21 -> 0`, again has insufficient
margin vote. It has primary-like support, but the ensemble does not agree enough
to admit it.

## Deli-Style Interpretation

This round produced a falsification, not a promotion:

- proposer: generated a broader fresh queue;
- referee: visual counter-target margins rejected all edges;
- opponent: temporal gate stayed armed and will catch same-camera overlaps;
- gate: no materialization because `accepted_edges = 0`;
- evaluator: the only full-scored proposer row dropped to IDF1 `0.653339`.

The next structural pivot should not be another wider mass-bridge queue. Move to
one of:

1. train a counter-target verifier with temporal hard negatives collected from
   `13 -> 49`, `21 -> 0`, and this queue;
2. add detector-quality/density admission before merge proposal;
3. build a local-track continuity scorer that must explain predecessor and
   successor support better than competing components;
4. switch delivery semantics to committed/provisional/pending for reporting,
   while keeping forced output only for benchmark submission.

## Artifacts

- `reports/deli_autoresearch_selfplay_operationalization_20260621.md`
- `local_runs/no_anchor_temporal_clean_bridge_queue_20260621_summary.json`
- `local_runs/no_anchor_temporal_clean_bridge_queue_20260621/search.json`
- `local_runs/no_anchor_temporal_clean_bridge_queue_20260621/search.csv`
- `local_runs/no_anchor_temporal_clean_bridge_queue_20260621/audit_visual_temporal.json`
- `local_runs/no_anchor_temporal_clean_bridge_queue_20260621/audit_visual_temporal.csv`
- remote run dir: `/mnt/localssd/vlincs_reid_runs/no_anchor_temporal_clean_bridge_queue_20260621`
