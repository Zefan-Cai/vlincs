# No-Anchor Crossqueue + Single-Edge 68 + Localized Island Queue

Date: 2026-06-20

## Summary

This iteration turns the broad single-edge referee failure into a narrow
no-GT proposer:

- keep the earlier crossqueue portfolio rows,
- keep the no-GT single-edge embedding-choice seed `3 -> 68`,
- add a new localized-island single-edge proposer that selects `9 -> 7`.

No anchors are used. No GT is used for candidate generation, reranking, or
assignment export. GT appears only in the posthoc opponent audit below.

## New Code

- `kit/rerank_no_anchor_single_edge_localized_island.py`

The script accepts only single-edge candidate fields and rewards:

- high `target_best_sim`,
- high `target_best_sim - target_mean_sim` localized peak,
- low `source_cross_mean_sim`,
- low `source_conflicts_to_rest`,
- high source internal coherence and source quality,
- small source island and sufficiently large target.

Self-test passes.

## Localized-Island Candidate

Input:

- `local_runs/no_anchor_single_edge_broad_scheduler_candidates_20260620.json`

Output:

- `local_runs/no_anchor_single_edge_localized_island_20260620.json`
- `reports/no_anchor_single_edge_localized_island_20260620.md`

Result:

| source | target | island score | target best | localized peak | source cross mean | conflicts | pair F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `9` | `7` | `0.809065` | `0.943198` | `0.224714` | `0.509388` | `3` | `0.768903` |

This is intentionally narrow: 19 broad single-edge candidates in, 1 candidate
out.

## Portfolio Queue

Union scheduler:

- `local_runs/no_anchor_union_crossqueue_singleedge68_localized_island_20260620.json`

Portfolio candidates:

- `local_runs/no_anchor_crossqueue_singleedge68_localized_island_portfolio_candidates_20260620.json`
- `reports/no_anchor_crossqueue_singleedge68_localized_island_portfolio_candidates_20260620.md`

Full-score scheduler:

- `local_runs/no_anchor_fullscore_scheduler_crossqueue_singleedge68_localized_island_20260620.json`
- `reports/no_anchor_fullscore_scheduler_crossqueue_singleedge68_localized_island_20260620.md`

Scheduler result:

- raw candidates: `10`
- eligible: `10`
- selected: `8`

Top selected rows:

| scheduler rank | predicted full proxy | pair F1 | P | R | moved | key edges |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.668110` | `0.768959` | `0.814525` | `0.728222` | `64` | crossqueue + `9 -> 7` |
| 2 | `0.667385` | `0.769770` | `0.815179` | `0.729153` | `68` | crossqueue + `9 -> 7` |
| 3 | `0.666814` | `0.769775` | `0.815161` | `0.729177` | `80` | crossqueue + `9 -> 7` |
| 4 | `0.664389` | `0.769301` | `0.814920` | `0.728519` | `256` | `3 -> 68`, `9 -> 7` |
| 5 | `0.668890` | `0.769205` | `0.814938` | `0.728334` | `312` | crossqueue + `3 -> 68` + `9 -> 7` |
| 6 | `0.668105` | `0.769746` | `0.815374` | `0.728955` | `316` | crossqueue + `3 -> 68` + `9 -> 7` |

The highest proxy is rank 5 (`0.668890`), but the scheduler places smaller
delivery-risk candidates first. This is useful: the next remote full-score
should run ranks `1..8`, not only proxy rank 5.

## Eval-Only Opponent

Audit:

- `local_runs/no_anchor_candidate_false_split_coverage_crossqueue_singleedge68_localized_island_20260620.json`
- `reports/no_anchor_candidate_false_split_coverage_crossqueue_singleedge68_localized_island_20260620.md`

Result:

- top coverage: `0.008114`
- previous crossqueue + `3 -> 68` coverage: `0.005791`
- broad single-edge-only top coverage: `0.002323`
- summed positive bridge mass in audited rows: `392937196`

The strongest row in the audit is scheduler rank 5:

| edge | positive bridge mass |
| --- | ---: |
| `3 -> 68` | `51530040` |
| `9 -> 7` | `21153911` |
| `32 -> 15` | `933375` |
| `4 -> 6` | `261415` |

This supports the localized-island hypothesis, but the audit is not a
production selector.

## Local Exports

Assignment materialization:

- `local_runs/no_anchor_crossqueue_singleedge68_localized_island_local_export_20260620/manifest.json`
- assignment CSVs under
  `local_runs/no_anchor_crossqueue_singleedge68_localized_island_local_export_20260620/assignments/`

Sample submission export:

- `local_runs/no_anchor_crossqueue_singleedge68_localized_island_sample_export_20260620/summary.json`
- `local_runs/no_anchor_crossqueue_singleedge68_localized_island_sample_export_20260620/sample_full_results.jsonl`
- 8 rank zip files under
  `local_runs/no_anchor_crossqueue_singleedge68_localized_island_sample_export_20260620/`

Local GT status:

- `gt_available=false`
- `gt_message=no overlap between predictions and local DS1 ground truth`

So no canonical e2e score is claimed from local export.

## Remote Full-Score Command

Run this when Pluto/SSH recovers:

```bash
SCHEDULER_JSON=local_runs/no_anchor_fullscore_scheduler_crossqueue_singleedge68_localized_island_20260620.json \
RUN_NAME=no_anchor_crossqueue_singleedge68_localized_island_fullscore_20260620 \
kit/run_no_anchor_scheduler_manifest_fullscore.sh --ranks 1,2,3,4,5,6,7,8
```

## Current Status

- Verified model-side pair metrics remain above 70:
  `F1=0.775234`, `P=0.820504`, `R=0.734698`.
- Verified e2e best remains `IDF1=0.655240`.
- This queue is the best current no-anchor challenger by proxy/opponent
  evidence, but it is not a verified e2e result until canonical DS1 full-score
  runs.
