# No-Anchor Crossqueue Portfolio + Single-Edge 3->68 Seed

## Why

The single-edge embedding-choice referee found one useful no-GT seed:

- `3 -> 68`
- moved tracklets: `248`
- eval-only positive bridge mass: `51530040`
- eval-only coverage: `0.005659`

The existing best unverified crossqueue portfolio was broader but missed this
edge. This experiment composes the seed with existing crossqueue selected rows
to create higher-mass candidate assignments.

## Inputs

- Crossqueue scheduler:
  `local_runs/no_anchor_fullscore_scheduler_crossqueue_portfolio_20260620.json`
- Single-edge referee candidates:
  `local_runs/no_anchor_single_edge_embedding_choice_candidates_20260620.json`
- Union scheduler:
  `local_runs/no_anchor_union_crossqueue_plus_single_edge68_20260620.json`

No anchors or GT labels are used for production candidate generation.

## Candidate Composition

Command output:

- candidates: `3`
- source rows in union: `4`
- all candidates include target component `68`

Candidate report:

- `reports/no_anchor_crossqueue_plus_single_edge68_portfolio_candidates_20260620.md`

Top candidate:

- predicted full IDF1 proxy: `0.664682`
- moved tracklets: `304`
- accepted edits: `7`
- targets: `15+19+55+6+68`
- source ranks: `[1, 4]`

## Scheduler Output

Guarded scheduler:

- `local_runs/no_anchor_fullscore_scheduler_crossqueue_plus_single_edge68_20260620.json`
- `reports/no_anchor_fullscore_scheduler_crossqueue_plus_single_edge68_20260620.md`

Selected rows:

| rank | predicted full | pair F1 | moved | target components |
| ---: | ---: | ---: | ---: | --- |
| 1 | `0.664682` | `0.769356` | `304` | `6,15,19,55,68` |
| 2 | `0.663496` | `0.770167` | `308` | `6,15,21,55,68` |
| 3 | `0.662130` | `0.770172` | `320` | `6,15,21,55,68` |

The scheduler risk score is lower than the predicted-full rank because these
are large edits, but they remain pair-gate passing.

## Eval-Only Opponent Audit

Audit file:

- `reports/no_anchor_candidate_false_split_coverage_crossqueue_plus_single_edge68_20260620.md`

All three rows have:

- coverage: `0.005791`
- positive bridge mass per row: includes `3 -> 68` mass `51530040`, plus
  smaller positive bridges `4 -> 6` and `32 -> 15`
- summed positive bridge mass across top rows: `158174490`

This is not a production selector. It is a case explanation showing the new
portfolio contains both the single-edge referee seed and previous crossqueue
positive edges.

## Export

Assignment-level local export:

- `local_runs/no_anchor_crossqueue_plus_single_edge68_local_export_20260620`

Submission-style sample export:

- `local_runs/no_anchor_crossqueue_plus_single_edge68_sample_export_20260620`

Generated zips:

- `rank01_hub_bridge_portfolio_no_anchor_crossqueue_plus_single_edge68_portfolio_candidates_20260620_assignments.zip`
- `rank02_hub_bridge_portfolio_no_anchor_crossqueue_plus_single_edge68_portfolio_candidates_20260620_assignments.zip`
- `rank03_hub_bridge_portfolio_no_anchor_crossqueue_plus_single_edge68_portfolio_candidates_20260620_assignments.zip`

Local GT is still unavailable:

- `gt_available=false`
- message: `no overlap between predictions and local DS1 ground truth`

So no canonical e2e IDF1 is claimed. The standing verified e2e best remains
`0.655240`.

## Next

When Pluto recovers, full-score:

```bash
SCHEDULER_JSON=local_runs/no_anchor_fullscore_scheduler_crossqueue_plus_single_edge68_20260620.json \
RUN_NAME=no_anchor_crossqueue_plus_single_edge68_fullscore_20260620 \
kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3
```

If remote scoring remains unavailable, the next local research direction is to
search for more no-GT seeds like `3 -> 68`, then compose only seeds that are
component-chain compatible with the current crossqueue top row.
