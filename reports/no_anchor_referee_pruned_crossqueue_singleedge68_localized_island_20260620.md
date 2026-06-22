# No-Anchor Referee-Pruned Portfolio Queue

Date: 2026-06-20

## Purpose

The previous crossqueue + `3 -> 68` + localized `9 -> 7` queue improved
eval-only false-split coverage, but several portfolio rows carried many weak
extra edges. This iteration adds an edge-level no-GT referee that prunes weak
portfolio edges while preserving three evidence families:

1. `large_source_singleton_seed`, e.g. `3 -> 68`
2. `localized_island`, e.g. `9 -> 7`
3. `stable_multiview_attach`, e.g. `4 -> 6` and `32 -> 15`

No anchors or GT are used for pruning or assignment generation. GT is used only
for the opponent audit.

## New Code

- `kit/filter_no_anchor_portfolio_edges_by_referee.py`

Self-test passes.

## Candidate Generation

Input:

- `local_runs/no_anchor_crossqueue_singleedge68_localized_island_portfolio_candidates_20260620.json`

Output:

- `local_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- `reports/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_20260620.md`

Pruning result:

- input rows: `10`
- raw edges: `77`
- kept edges: `32`
- emitted rows: `10`

## Scheduler Result

Scheduler artifacts:

- `local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- `reports/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.md`

Selected rows:

| rank | proxy full | pair F1 | P | R | moved | edges |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.665788` | `0.768959` | `0.814525` | `0.728222` | `28` | `4 -> 6`, `32 -> 15`, `9 -> 7` |
| 2 | `0.664389` | `0.769301` | `0.814920` | `0.728519` | `256` | `3 -> 68`, `9 -> 7` |
| 3 | `0.666790` | `0.769205` | `0.814938` | `0.728334` | `276` | `4 -> 6`, `32 -> 15`, `3 -> 68`, `9 -> 7` |
| 4 | `0.662361` | `0.769356` | `0.815368` | `0.728260` | `268` | `4 -> 6`, `32 -> 15`, `3 -> 68` |

This queue is lower risk than the unpruned 8-row queue: the best opponent row
keeps the same useful four positive bridge edges while removing weak extras.

## Eval-Only Opponent

Audit artifacts:

- `local_runs/no_anchor_candidate_false_split_coverage_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- `reports/no_anchor_candidate_false_split_coverage_referee_pruned_crossqueue_singleedge68_localized_island_20260620.md`

Result:

- top coverage: `0.008114`
- top row positive bridge mass: `73878741`
- summed positive bridge mass in top rows: `221636223`

Top audited row is scheduler rank 3:

| edge | positive bridge mass |
| --- | ---: |
| `3 -> 68` | `51530040` |
| `9 -> 7` | `21153911` |
| `32 -> 15` | `933375` |
| `4 -> 6` | `261415` |

Compared to the previous unpruned queue, top coverage is unchanged
(`0.008114`), but the row has only 4 kept edges instead of carrying weak
portfolio additions.

## Local Exports

Assignment materialization:

- `local_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_local_export_20260620/manifest.json`

Sample export:

- `local_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_sample_export_20260620/summary.json`
- `local_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_sample_export_20260620/sample_full_results.jsonl`
- 4 rank zip files under
  `local_runs/no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_sample_export_20260620/`

Local GT remains unavailable:

- `gt_available=false`
- `best_idf1=null`

## Remote Command

Run this when Pluto/SSH recovers:

```bash
SCHEDULER_JSON=local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json \
RUN_NAME=no_anchor_referee_pruned_crossqueue_singleedge68_localized_island_fullscore_20260620 \
kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3,4
```

## Status

Verified model-side pair metrics remain above 70. Verified end-to-end best is
still `IDF1=0.655240` until canonical DS1 full-score runs.
