# No-Anchor Broad Single-Edge Referee Probe

Date: 2026-06-20

## Purpose

This probe expands the Deli-style worker/referee loop beyond the focused 8-row
single-edge queue. It decomposes all 20260620 scheduler rows into single-edge
candidate moves, then applies the same no-GT embedding-choice referee used for
`3 -> 68`.

Production constraint: the candidate generation and rerank use no anchors and
no GT. The false-split coverage audit below is an eval-only opponent.

## Artifacts

- Candidate pool:
  `local_runs/no_anchor_single_edge_broad_scheduler_candidates_20260620.json`
- No-GT embedding-choice rerank:
  `local_runs/no_anchor_single_edge_broad_embedding_choice_20260620.json`
- Eval-only opponent audit:
  `local_runs/no_anchor_candidate_false_split_coverage_single_edge_broad_embedding_choice_20260620.json`
- Tables:
  `reports/no_anchor_single_edge_broad_scheduler_candidates_20260620.md`
  `reports/no_anchor_single_edge_broad_embedding_choice_20260620.md`
  `reports/no_anchor_candidate_false_split_coverage_single_edge_broad_embedding_choice_20260620.md`

## No-GT Rerank Result

- Parent scheduler rows scanned: `168`
- Single-edge rows emitted after filtering/dedup: `19`
- Referee-selected rows: `17`
- Suppressed duplicate-source rows: `2`
- Tracklet embeddings loaded: `4391`

Top no-GT rows by proxy/choice:

| rank | edge | proxy | choice | pair F1 | moved |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `40 -> 21` | `0.668713` | `0.676030` | `0.773816` | `8` |
| 2 | `26 -> 21` | `0.668604` | `0.619424` | `0.773816` | `8` |
| 3 | `21 -> 0` | `0.668173` | `0.657831` | `0.772654` | `8` |
| 4 | `19 -> 6` | `0.667660` | `0.617444` | `0.772931` | `12` |
| 5 | `0 -> 6` | `0.666980` | `0.630684` | `0.772931` | `8` |

## Eval-Only Opponent Result

The broad queue does not beat the focused `3 -> 68` seed.

- Top broad coverage: `0.002323`
- Focused `3 -> 68` coverage from the previous audit: `0.005659`
- Broad summed positive bridge mass: `22482521`

Top positive broad edge:

| audit rank | edge | positive bridge mass | coverage | no-GT rerank rank |
| ---: | --- | ---: | ---: | ---: |
| 1 | `9 -> 7` | `21153911` | `0.002323` | `15` |
| 2 | `32 -> 15` | `933375` | `0.000103` | `7` |
| 3 | `4 -> 6` | `261415` | `0.000029` | `6` |
| 4 | `2 -> 13` | `133820` | `0.000015` | `11` |

The no-GT top rows `40 -> 21`, `26 -> 21`, and `21 -> 0` have zero eval-only
positive bridge mass in the current error-analysis table.

## Referee Failure Hypothesis

The current no-GT embedding-choice referee favors high aggregate source-to-target
embedding support and large high-quality targets. That is useful for duplicate
source arbitration such as `3 -> 68` vs `3 -> 45`, but it misses a different
edge type:

- `9 -> 7` has lower aggregate embedding support (`0.514952`) and lower
  source-to-target top5 (`0.476108`).
- It has very high target best similarity (`0.943198`), low source cross-mean
  similarity (`0.509388`), and few source conflicts to rest (`3`).
- This looks like a clean source island matching a localized subregion of a
  large target, not a globally similar source-target pair.

Next scorer direction:

1. Keep the focused single-edge referee for duplicate-source arbitration.
2. Add a separate "localized island attach" candidate family that rewards high
   `target_best_sim`, low `source_cross_mean_sim`, and low
   `source_conflicts_to_rest`.
3. Gate this family tightly because the eval-only signal came from the opponent;
   it should generate candidates for remote full-score, not become a production
   selector by itself.

## Current Status

No canonical e2e score is claimed. Local DS1 GT remains unavailable for these
sample exports, and the Pluto full-score nodes are still unreachable. The
standing verified e2e best remains `0.655240`; model-side pair metrics remain
above 70.
