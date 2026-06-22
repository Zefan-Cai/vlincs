# Deli AutoResearch Self-Play Operationalization For VLINCS

Date: 2026-06-21

## Read Sources

- Deli_AutoResearch framework page:
  https://victorchen96.github.io/auto_research/framework.html
- Deli AutoResearch papers page:
  https://victorchen96.github.io/auto_research/paper.html
- Self-play story:
  https://victorchen96.github.io/blog_self_play_story.html

## Distilled Mechanism

Deli AutoResearch is less a code package than a protocol for not getting stuck.
The transferable pieces are:

1. File-backed state, not chat-backed state.
2. Ready means execute: prepared candidates should be run, scored, debugged, and
   logged without asking for a go-ahead.
3. Honest downward scoring: a score drop is a useful refutation, not a failed
   iteration to hide.
4. Proposer / referee / opponent / gate separation.
5. Stale iterations force structural pivots, not more local threshold tuning.
6. A watchdog or fresh-session loop is needed because long-horizon agents stall
   more often than they crash.

The self-play story adds one extra point that matters here: the winning loop did
not only write a survey; it created an experiment that could falsify its own
claim, ran the experiment, accepted score drops, and then changed the bottleneck
from empirical validation to theory hardening.

## Translation To No-Anchor VLINCS

Current VLINCS objective:

- no-anchor global-id model already passes the pair target:
  F1/P/R = 0.775234 / 0.820504 / 0.734698.
- end-to-end delivery remains below target:
  best IDF1/HOTA/AssA = 0.655817 / 0.519228 / 0.534791.

The loop should therefore stop treating pair F1 improvement as enough evidence.
For each candidate identity edit:

1. Proposer generates a no-GT candidate edge or policy.
2. Referee checks visual, trajectory, density, and provenance support.
3. Opponent tries to find same-frame, same-video temporal overlap, namespace
   drift, target ambiguity, and detector-quality failure.
4. Gate materializes only candidates that pass referee and opponent.
5. Full-score is used only as the delivery evaluator, not as production evidence
   or anchor/training signal.

## Immediate Structural Rule

Because the active loop is already stale, the next research branch must change
the candidate space or the admission logic. Do not spend another iteration only
relaxing:

- visual similarity thresholds,
- source component size thresholds,
- rank-by variants over the same accepted edges.

Acceptable structural pivots:

- generate temporally clean candidates before visual admission;
- train a verifier using accepted false positives as temporal hard negatives;
- introduce detector-quality/density quarantine before any merge;
- change output semantics to committed/provisional/pending while preserving
  forced output only for benchmark submission;
- add a local-track continuity model that must explain predecessor/successor
  support, not just appearance similarity.

## Research Since Distillation

The self-play loop produced three immediate outcomes on h100-test-3.

| Branch | Result | Decision |
|---|---:|---|
| temporal-clean bridge queue | `accepted_edges = 0`; full rank1 IDF1 `0.653339` | reject |
| k3 global admission grid | top full rows IDF1 `0.653210` | reject |
| k3 budgeted multiview merge | accepted 9 edges; full IDF1 `0.653210` | reject |
| targeted clothing audit | bad edges rejected at about `2e-4`; preview edges `35->61`, `10->26` high | promote signal |
| whole-component decision merge | high edges blocked by forbidden/size; accepted_edges `0` | reject action |
| sample-pair micro surgery | accepted 6 tracklets; IDF1 `0.653051` | reject action |
| current-best delivery policy ablation | density_simple / oracle-lite / confidence-tail IDF1 `0.655817 / 0.655810 / 0.655800` | plateau |
| current-best softcut true split | split 1 component / 162 tracklets; density-simple IDF1 `0.651408` | reject |
| local continuity relink | local-track coverage `0`; temporal one-edge density-simple IDF1 `0.655817` | neutral / low coverage |

Those branches are useful refutations: the current bottleneck is not a larger
bridge queue, not global tracklet admission, not a wider multiview edge budget,
not delivery filtering alone, and not coarse soft-cut splitting of current-best
components.  Local continuity is also not yet enough: the available
`local_track_id` groups do not cross fragments, and temporal adjacency produced
only one neutral edge in the MCAM04/08 target slice.

The fourth branch was a targeted opponent audit:

- tool added:
  `kit/no_anchor_clothing_edge_audit.py`;
- audited the nine edges accepted by the budgeted multiview merge;
- all nine received clothing/body verifier probability around `2e-4`;
- the dominant bad bridge `9 -> 60` received probability `0.000260`.

This says the next production path is not direct multiview merging. It should be
a cached edge table plus a learned no-anchor clothing/body referee before any
edit is materialized. The first materialization should be micro-component
surgery, not whole-component merge, because the high-confidence referee edges
are still blocked by physical constraints at component scale. The first
micro-surgery attempt also dropped, so the next action should be local
reassignment or label propagation into an existing identity component, not
creation of a new isolated component.

## Working Hypothesis

The current failure mode is not missing visual similarity. It is that visually
plausible merges inside dense same-camera cells are often temporal impostors.
The self-play analog is:

- proposer: "these two tracklet islands look like the same identity";
- opponent: "show me that they could coexist in time, and that local trajectory
  continuity does not prefer another component";
- gate: "merge only if the opponent fails to produce a concrete contradiction."

That makes the next few iterations about contradiction mining and referee
calibration, not bigger positive pools. The immediate build target is:

1. generate a bounded k3 edge table and write it as an intermediate artifact;
2. score that table with the no-anchor clothing/body verifier;
3. export the top supporting left/right tracklet seqs for high-confidence
   edges;
4. use those seqs for local reassignment / label propagation into existing
   components;
5. full-score only the admitted reassignment policies.
