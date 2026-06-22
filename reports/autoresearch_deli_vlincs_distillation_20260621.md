# Deli AutoResearch Distillation For VLINCS

Date: 2026-06-21

## Sources Read

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`
- `https://github.com/karpathy/autoresearch`

## Distilled Operating Rules

The useful part is not a specific RL algorithm.  The reusable pattern is a
research-control protocol for long-horizon agents:

1. Persist state in files, not chat memory.
2. Separate proposer, reviewer, opponent, and scorer.
3. Treat a metric drop or no new finding as stale; after repeated stale rounds,
   change the structural evidence source rather than retuning thresholds.
4. Run the expensive experiment once the candidate is ready.  Do not stop at
   "ready to submit".
5. Allow the reviewer/opponent score to go down when evidence gets worse.
6. Keep production features no-anchor.  GT/full-score labels are only post-hoc
   reviewer labels or final evaluator output.

## Mapping To VLINCS No-Anchor Global ID

| AutoResearch role | VLINCS implementation |
| --- | --- |
| State files | `autoresearch_state/no_anchor_global_id/state/progress.json`, `directions_tried.json` |
| Proposer | no-anchor candidate generators over tracklet/component evidence |
| Reviewer | pair/global-ID proxy, scheduler, leakage audit |
| Opponent | counter-target verifier, side-effect critic, Deli opponent gate |
| Scorer | canonical DS1 full evaluator, GT evaluation only |
| Commit/defer | production assignment + optional delivery postfilter; internal reports keep provisional/quarantine statuses |

## Consequence For The Current Plateau

The active best is `IDF1/HOTA/AssA = 0.655911 / 0.519311 / 0.534922`.
Recent gains are detector-delivery postfilters, not identity evidence gains.
That family is now stale: it improves by `+0.000056` over area800 and is far
from the `0.70` end-to-end target.

The next AutoResearch-style move should therefore be a structural pivot:

- freeze the promoted per-video area p0.5 delivery policy for comparisons;
- stop rewarding pair-F1-only candidates unless full DS1 score improves;
- spend full-score budget on candidates that change identity evidence or
  generate new detector/tracklet evidence;
- record negative results as useful opponent data, not as failed turns.

## Experiment Chosen Next

I selected the component-graph rescue probe as a bounded structural experiment.
It rewrites multi-edge identity assignments rather than tuning delivery
thresholds.  This is intentionally marked as a probe because its exported
assignment CSVs are based on the 2026-06-20 softcut/softoverlap base, not the
current k3 p0.5 production artifact.

Protocol for this probe:

1. Full-score rank 1-3 on h100-test-3.
2. Apply the frozen per-video p0.5 area admission to each resulting zip.
3. Accept only if the postfiltered score exceeds `0.655911`.
4. If all lose, log the family as refuted and pivot to a current-k3 namespace
   proposer or detector-level tracklet regeneration for MCAM04/MCAM06.
