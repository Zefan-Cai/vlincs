# Deli AutoResearch Distillation For VLINCS No-Anchor Global ID

Sources read on 2026-06-20:
- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html

## What To Import

The useful part is not a new model architecture. It is an execution protocol for long-horizon research where the agent keeps working without asking for confirmation, persists state to files, accepts score drops as evidence, and separates proposal from evaluation.

For the VLINCS no-anchor global-ID loop, the mapping is:

| Deli protocol idea | VLINCS implementation rule |
| --- | --- |
| Zero interaction / ready means execute | If a no-anchor candidate passes local checks, export the assignment/zip and put it in the full-score queue. Do not stop at "should we submit?" |
| Persistent state files | Update `progress.json`, `directions_tried.json`, and `findings.jsonl` every iteration; do not rely on chat memory. |
| Score drop is honest evidence | Treat the sample-slice 0.710 assignment-summary proxy false positive as a negative finding, not as progress. |
| Forced pivot after stalls | With stale_count >= 8, change structural assumptions: source-selector and threshold-only families are saturated. |
| Reviewer personas | Keep distinct roles: proposer, no-GT referee, protocol auditor, result gate, and Deli opponent. |
| Real experiment > rhetorical score | A proxy-ranked queue is not completion evidence. Only canonical full DS1 scoring can move the verified e2e best. |

## Current Application

The assignment-summary proxy is useful as a scheduler, but the Deli-style opponent rejects it as proof:

- Unfiltered top prediction `0.710` was an out-of-distribution false positive from 2-video sample assignments.
- Full-DS1 coverage filter removed sample-slice candidates.
- Filtered top assignment is `no_anchor_local_pervideo_source_selector_balanced`, predicted full IDF1 `0.664464`.
- The top-20 full-score queue has ready zip paths, but remote Pluto/SSH is currently unavailable.
- Result gate still says `pass_joint=false`; verified e2e best remains `0.655240`.

## Next Research Constraint

Do not spend the next iteration on more source-selector threshold tuning. The next structural branch should be one of:

1. restore canonical remote scoring or local DS1 GT access;
2. build a no-GT false-split solver that changes the graph/component structure, not just per-video source choice;
3. create an adversarial referee that searches for counter-evidence before committing a large component bridge.

The next branch should explicitly report:

- candidate recall proxy,
- opponent/referee rejection reasons,
- changed tracklet count and component count,
- whether the assignment is committed/provisional/quarantined,
- canonical full-score status.
