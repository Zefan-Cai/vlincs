# Deli AutoResearch -> VLINCS No-Anchor Protocol

Date: 2026-06-22

Sources:
- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html
- https://github.com/karpathy/autoresearch

## Distilled Rules

The useful part for VLINCS is not another model trick. It is an operating protocol for long-horizon research where every loop is forced to produce durable state, verifiable artifacts, and an honest score update.

1. Persist the loop in files:
   - `state/progress.json`: current best metrics, last result, next direction.
   - `state/findings.jsonl`: append-only verified findings.
   - `state/directions_tried.json`: closed, active, stale, or promoted directions.
   - `state/iteration_log.jsonl`: one entry per executable iteration.

2. Ready means execute:
   - If a candidate pool, export path, and canonical scorer are available, run the scorer.
   - Do not stop at proxy rows or visual plausibility.
   - For VLINCS, the judge is the canonical assignment CSV -> full submission zip -> density/admission -> HOTA/IDF1 path.

3. Separate roles:
   - Proposer: generates no-anchor candidate edits.
   - Opponent/referee: rejects visually plausible but temporally or namespace-impossible edits.
   - Executor: materializes selected edits into assignment CSVs.
   - Judge: scores full DS1 metrics and per-video slices.
   - Scheduler: uses prior verified results only to allocate experiment budget, never as anchors.

4. Treat drops as evidence:
   - Negative full-score rows are not wasted compute.
   - They become hard negatives for the next scheduler/ranker.
   - A proxy that predicts gains but causes DS1 side effects must be refuted, not tuned blindly.

5. Pivot structurally after stale local loops:
   - If several iterations produce only tiny deltas, change the candidate family or evidence source.
   - In this run, top-k15 time-agglom local attach produces real but tiny gains. The next structural step is to model edit interactions or generate higher-mass, low-risk candidates, not to keep hand-ranking single attaches.

## VLINCS-Specific Contract

Production evidence must satisfy:
- `uses_anchors=false`.
- `uses_gt_for_training_or_anchors=false`.
- GT may appear only in metric/evaluation JSON, never in production assignment evidence.
- Candidate labels derived from previous full-score deltas are allowed only for experiment scheduling and ranker calibration, not for identity anchors.

Current goal state:
- Global-id model target is passed: pair F1/P/R = 0.775234 / 0.820504 / 0.734698.
- End-to-end target is not passed: best IDF1 is still 0.656225, below 0.70.

## Update After Self-Play Thread

The latest Deli AutoResearch/self-play release strengthens one lesson that is
directly useful here: do not protect a monotone story.  A valid loop can lower
its own score, refute a tempting branch, and record that as progress.

For VLINCS this means:

1. Repeated local attach gains below 0.001 IDF1 are not enough; after two stale
   loops, the next direction must change the structural assumption.
2. Prior full-score deltas may train a scheduler/referee, but they remain
   experiment-budget evidence, never identity anchors.
3. A branch that improves DetPr while hurting DetRe must be logged as a
   delivery-side failure, not retuned until it appears positive.
4. The judge remains canonical DS1 IDF1/HOTA/AssA with per-video slices; proxy
   wins cannot update best state.

## Self-Play Transfer To VLINCS

The Self-play survey's most useful claim for this project is verifier-centric:
the quality of the verification signal sets the ceiling of self-improvement.
For VLINCS, the analogy is:

- Player / proposer: a no-anchor candidate generator that edits component
  assignments from visual, temporal, and trajectory evidence.
- Opponent / referee: cannot-link checks, temporal conflict checks, visual
  counterexamples, and side-effect predictors trained only from prior verified
  experiment outcomes.
- Verifier: canonical DS1 submission scoring plus per-video IDF1/HOTA/AssA and
  DetPr/DetRe slices. This is the only signal allowed to promote best state.
- Population: multiple edit families, not one local hill-climber: time attach,
  subpart repair, component split/merge, admission/quarantine, rollback, and
  counterfactual undo.
- KL anchor analogue: conservative edit priors that keep the assignment close
  to the current best unless evidence clears a higher bar. Weakening this prior
  may improve a targeted slice while damaging held-out videos, so every promoted
  edit must show global and per-video side effects.

Actionable protocol change:

1. Treat proxy rankers as noisy verifiers. They can allocate compute, but their
   error must be measured against full-score outcomes.
2. Run self-play-style proposer/opponent pairs: one generator proposes edits,
   another searches for counterexamples such as same-stream overlap, target
   impurity, or prior negative siblings.
3. Keep downward score movements visible. A failed branch becomes opponent
   training data instead of an embarrassment to hide.
4. After a micro-promotion, immediately try compatible combinations and undo
   counterfactuals. This tests whether the gain is compositional or a narrow
   local artifact.
