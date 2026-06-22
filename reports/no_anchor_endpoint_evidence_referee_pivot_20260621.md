# No-Anchor Endpoint Evidence Referee Pivot

Date: 2026-06-21

## Deli AutoResearch distilled for this run

Source material:

- Deli AutoResearch framework: https://victorchen96.github.io/auto_research/framework.html
- Deli AutoResearch papers page: https://victorchen96.github.io/auto_research/paper.html
- Self-play story: https://victorchen96.github.io/blog_self_play_story.html

Actionable distillation for VLINCS global ID:

1. Persist state to files, not conversation. Every experiment branch must leave `result.json`, `result.csv`, a report, and an iteration log entry.
2. Ready means execute. Candidate generation, full scoring, and failure diagnosis do not wait for manual confirmation.
3. Honest score drops are progress. If pair metrics improve but end-to-end IDF1 drops, log it as a referee result, not as a hidden success.
4. After repeated stalls, pivot structure, not thresholds. The current pivot is from direct endpoint relabeling to endpoint-as-evidence for the scheduler/referee.
5. Separate proposer from referee. Endpoint geometry can propose candidate evidence; only full-score and side-effect critics can decide whether to materialize it.

This matches the self-play story's most useful lesson for us: the system should mark itself down when evidence demands it, then convert the failure into a stronger next constraint.

## Endpoint audit

Artifact:

- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_candidate_audit_20260621/audit.json`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_candidate_audit_20260621/audit.csv`
- Remote: `/mnt/localssd/vlincs_reid_runs/no_anchor_endpoint_candidate_audit_20260621/audit.json`

The audit is evaluation-only. It uses GT labels only after no-GT candidates are generated.

| source seq | target seq | video | support | local GT result | target component dominant GT | source GT fraction in target component | diagnosis |
|---:|---:|---|---:|---|---:|---:|---|
| 1604 | 1549 | MCAM03 Tc8 | 2 | 20 -> 20 true | 12 at 0.887 | 0.014 | local true endpoint, impure target component |
| 1619 | 1622 | MCAM03 Tc8 | 2 | 9 -> 9 true | 4 at 0.900 | 0.015 | local true endpoint, impure target component |
| 109 | 127 | MCAM00 Tc6 | 1 | 9 -> 31 false | 31 at 0.796 | 0.062 | local false endpoint |

Key finding: endpoint evidence is often locally right, but the target component is dominated by another identity. Whole-component merge is therefore the wrong materialization.

## Micro-surgery ablation

New code:

- `kit/audit_no_anchor_endpoint_candidates.py`
- `kit/no_anchor_endpoint_pair_micro_surgery.py`
- `kit/no_anchor_fullscore_scheduler.py` now recognizes `action_signature` families and endpoint direct-action side-effect risk.

Artifacts:

- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_pair_micro_surgery_dedup_fullscore_20260621/result.csv`
- `local_runs/remote_h100_test_3_20260621/no_anchor_endpoint_pair_micro_surgery_dedup_fullscore_20260621/result.json`
- Remote: `/mnt/localssd/vlincs_reid_runs/no_anchor_endpoint_pair_micro_surgery_dedup_fullscore_20260621/result.json`

Standing best remains:

| metric | value |
|---|---:|
| IDF1 | 0.655817 |
| HOTA | 0.519228 |
| AssA | 0.534791 |
| model pair F1 / P / R | 0.775234 / 0.820504 / 0.734698 |

Micro-surgery full-score rows, deduplicated by actual moved seq signature:

| rank | action | moved evidence | pair F1 | full IDF1 | HOTA | AssA | verdict |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | source + target endpoint -> new component | 1549+1604, 1619+1622 | 0.770956 | 0.653731 | 0.517546 | 0.532996 | reject |
| 2 | target endpoint singleton peel | 1549, 1622 | 0.770956 | 0.653731 | 0.517546 | 0.532996 | reject |
| 3 | source + target endpoint, loose support | 109+127, 1549+1604, 1619+1622, 9526+9580 | 0.770867 | 0.653701 | 0.517518 | 0.532977 | reject |
| 4 | target endpoint singleton peel, loose support | 127, 1549, 1622, 9526 | 0.770867 | 0.653701 | 0.517518 | 0.532977 | reject |

Interpretation:

- Pair F1 can rise because target endpoint peeling reduces local component impurity.
- End-to-end IDF1 drops because the action breaks frame-level identity continuity.
- Support_edges=1 brings in false endpoint pairs and makes the full score worse.
- Endpoint evidence should be demoted from assignment action to scheduler/referee feature.

## Next direction

Use endpoint evidence in the no-anchor full-score scheduler, not as a direct relabel operation:

1. Add action-signature de-duplication to all full-score queues.
2. Add endpoint evidence features to candidate rows: support_edges, visual, position score, scale score, target component size, source component size, and whether the action peels an endpoint from a large component.
3. Train or hand-code a side-effect critic that penalizes endpoint actions likely to break continuity, using this refutation as negative labels.
4. Require every future candidate report to include an evidence card: local pair support, countertarget evidence, expected side effect, full-score result, and provenance.

## Scheduler guard

Artifacts:

- `local_runs/no_anchor_endpoint_micro_surgery_scheduler_guard_json_20260621.json`
- `local_runs/no_anchor_endpoint_micro_surgery_scheduler_guard_json_20260621.csv`
- `reports/no_anchor_endpoint_micro_surgery_scheduler_guard_json_20260621.md`

The updated scheduler reads the JSON preview and rejects all 100 endpoint micro-surgery rows:

| input rows | eligible | selected | primary reasons |
|---:|---:|---:|---|
| 100 | 0 | 0 | predicted_not_above_current_best, already_full_scored, known_below_current_best |

For the top rejected endpoint action, the no-GT side-effect reasons are:

- `large_target_component`
- `endpoint_direct_action`
- `endpoint_large_target_component`

This is the reusable guardrail from the refutation: endpoint-like local evidence is still useful, but direct endpoint relabeling must be penalized unless a future candidate has stronger independent counter-evidence.

Status: no-anchor model target is satisfied on pair metrics, but end-to-end remains below 0.70. Continue research; do not mark the goal complete.
