# No-Anchor AutoResearch Single-Edge Ablation

Date: 2026-06-20

## Distillation

Deli AutoResearch is most useful here as an operating protocol, not as a model
architecture.  The public framework emphasizes file-backed state, zero
interaction, fresh-session iterations, stall detection, structural pivots, and
guardian/worker separation.  The self-play story adds the key research lesson:
an autonomous loop must be willing to mark its own score down when external
checks contradict the proxy.

Sources:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html

VLINCS translation:

- proposer: no-anchor candidate generators and accepted-preview composers;
- executor: manifest exporter plus sample/full submission materialization;
- opponent: frozen result gate plus eval-only false-split coverage audit;
- state: `autoresearch_state/no_anchor_global_id/state/`;
- success condition: pair F1 above 0.70 is insufficient unless full/e2e IDF1
  moves toward 0.70.

## What Changed

Added an eval-only opponent:

- `kit/audit_no_anchor_candidate_false_split_coverage.py`

It reads candidate `accepted_preview` edits and checks whether proposed
source->target component moves connect different predicted fragments of the
same GT identity in the completed error-analysis artifact.  It is explicitly
not a production selector.

Fixed materialization:

- `kit/export_no_anchor_scheduler_manifest_assignments.py`

The exporter can now replay component-level preview items such as
`source/target` and `source_rep/target_rep`, not only explicit `source_seqs`.
It also records duplicate-source skips as `skipped_empty_source_components`
instead of crashing when a later edit references a source component already
moved by an earlier edit.

Added atomic ablation generation:

- `kit/compose_no_anchor_single_edge_candidates.py`

This explodes composite `accepted_preview` rows into one-source-one-target
candidate rows so good edges are not hidden by bad portfolio ordering.

## Opponent Findings

Current standing metrics remain unchanged:

- best pair model: F1 `0.775234`, precision `0.820504`, recall `0.734698`;
- best verified e2e full IDF1: `0.655240`;
- target e2e full IDF1: `0.700000`.

False-split coverage audits:

| queue | rows | best coverage of missing true-pair mass | key finding |
| --- | ---: | ---: | --- |
| crossqueue portfolio | 7 | `0.000131` | top proxy rows barely touch the oracle gap |
| video-focus big relaxed | 19 | `0.003602` | some useful `9/37 -> 7` edges, but many zero-mass edges |
| edge-table focused portfolio | 100 | `0.005659` | only `3 -> 68` carries most audited mass, but composite replay skipped it |
| edge-table single-edge | 8 | `0.005659` | `3 -> 68` is audit rank 1 but scheduler rank 7 |

The important ablation result is that the no-GT scheduler cannot distinguish
two same-source alternatives:

- `3 -> 45`: scheduler rank 6, predicted full `0.648263`, audit coverage `0`;
- `3 -> 68`: scheduler rank 7, predicted full `0.648263`, audit coverage
  `0.005659`.

So the next model improvement is not another broad threshold sweep.  It is a
target-choice proxy for duplicate-source alternatives.

## Materialized Artifacts

Single-edge scheduler:

- JSON: `local_runs/no_anchor_fullscore_scheduler_single_edge_edge_table_focused_20260620.json`
- CSV: `local_runs/no_anchor_fullscore_scheduler_single_edge_edge_table_focused_20260620.csv`
- report: `reports/no_anchor_fullscore_scheduler_single_edge_edge_table_focused_20260620.md`

Single-edge local export:

- directory:
  `local_runs/no_anchor_single_edge_edge_table_focused_recovered_base_local_export_20260620`
- GT status: `gt_available=false` on this Mac;
- outputs: 8 assignment CSVs and 8 submission zips;
- each zip contains 10 video parquets.

Key local export rows:

| rank | edit | moved tracklets | predicted full proxy | zip |
| ---: | --- | ---: | ---: | --- |
| 1 | `25 -> 30` | 156 | `0.649695` | `rank01_edge_table_single_edge_no_anchor_single_edge_edge_table_focused_candidates_20260620_assignments.zip` |
| 6 | `3 -> 45` | 248 | `0.648263` | `rank06_edge_table_single_edge_no_anchor_single_edge_edge_table_focused_candidates_20260620_assignments.zip` |
| 7 | `3 -> 68` | 248 | `0.648263` | `rank07_edge_table_single_edge_no_anchor_single_edge_edge_table_focused_candidates_20260620_assignments.zip` |
| 8 | `35 -> 60` | 276 | `0.647827` | `rank08_edge_table_single_edge_no_anchor_single_edge_edge_table_focused_candidates_20260620_assignments.zip` |

## Next Direction

Canonical scoring priority when Pluto or DS1 GT is available:

1. score the 8 single-edge zips, especially rank 7 `3 -> 68`;
2. compare rank 6 `3 -> 45` against rank 7 `3 -> 68` as a duplicate-source
   target-choice ablation;
3. use the result to add no-GT target-choice features to the full proxy:
   target singleton risk, duplicate-source alternatives, edge rank, fused/db
   score disagreement, target impurity proxy, and source-size penalty.

This keeps the no-anchor rule intact: GT was used only as an opponent/audit
label after proposals existed, not for anchors or production assignment
evidence.

## Remote Status

Remote canonical scoring was attempted after materialization.

Commands attempted:

```bash
conda run -n adobe python -m colligo.pluto.sdk.cli job status h100-test-3 --project video-world-models
conda run -n adobe python -m colligo.pluto.sdk.cli job status h100-test-2 --project video-world-models
conda run -n adobe python -m colligo.pluto.sdk.cli job status test-video-0 --project video-world-models
```

Observed status:

- `h100-test-3`: Pluto API `Failed to connect to Pluto service`;
- `h100-test-2`: Pluto API `Failed to connect to Pluto service`;
- `test-video-0`: Pluto API `Failed to connect to Pluto service`.

SSH dry-run using existing generated configs also failed for all three with
`Connection timed out during banner exchange`.

Therefore this iteration produced a ready-to-score queue, but no new verified
canonical e2e IDF1.  The standing verified best remains `0.655240`.
