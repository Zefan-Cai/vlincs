# VLINCS No-Anchor AutoResearch Iteration 9

Date: 2026-06-20

## Deli AutoResearch Distillation

Source pages reviewed:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

Useful transfer into this VLINCS loop:

- State is the product: keep `progress.json`, `directions_tried.json`, `findings.jsonl`, reports, and local run artifacts as the authoritative context.
- If a direction stalls, pivot a structural constraint, not just a threshold. Here the pivot was from portfolio rank tweaking to a new component-graph high-mass false-split proposer.
- Separate worker and reviewer: candidate generators are no-GT workers; eval-only audits and proxy-reviewers are opponents/referees and must not be confused with production selectors.
- Median/reviewer honesty matters: a reviewer is allowed to reject every candidate, and scores may go down. This happened when historical fullscore proxies were OOD for the new component-graph family.
- Replay is part of validation: a candidate is not real until its `accepted_preview` can be materialized into assignment CSVs and submission zips without skipped source components.

## New No-GT Worker: Component Graph High-Mass Bridge

Added:

- `kit/compose_no_anchor_component_graph_candidates.py`
- `kit/filter_no_anchor_component_graph_rescue_rule.py`

Input:

- base assignment: `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv`
- DS1 embedding shards: `kit/demo_data/ds1/embeddings/*/embeddings.npz`

Generation output:

- candidates: `94`
- output JSON: `local_runs/no_anchor_component_graph_high_mass_candidates_20260620.json`
- output report: `reports/no_anchor_component_graph_high_mass_candidates_20260620.md`

The raw graph score was not enough. Eval-only audit over all 94 rows found 10 positive rows with summed positive bridge mass `48,547,656`, but the strongest true positives were ranked only 83 and 84 by the no-GT graph score:

- `24 -> 31`: positive bridge mass `22,851,603`, coverage `0.002510`
- `31 -> 24`: positive bridge mass `22,851,603`, coverage `0.002510`

This is an important opponent finding: current no-GT graph ranking sees the useful edge family, but buries it.

## Proxy Reviewer Result

Historical fullscore proxy-reviewer ensemble:

- input rows: `94`
- eligible rows: `0`
- artifact: `local_runs/no_anchor_component_graph_proxy_reviewer_scheduler_20260620.json`
- report: `reports/no_anchor_component_graph_proxy_reviewer_scheduler_20260620.md`

Interpretation: this is not a clean production rejection of the edge family. It is an OOD failure of the old proxy-reviewers on a new candidate family, so this family needs either direct full-score probing or a new proxy trained with component-graph outcomes.

## Rescue Rule Ablation

A low-vote, high-top-similarity rescue rule was derived from eval-only opponent analysis but uses only no-GT fields at selection time:

- `target_view_vote` in `[0.25, 0.35]`
- high `target_best_sim`
- sufficient `target_min_view_sim`
- bounded same-video overlap

Strict rescue:

- selected rows: `2`
- eval-only positives: `2/2`
- positive bridge mass: `45,703,206`
- artifact: `local_runs/no_anchor_component_graph_low_vote_rescue_strict_20260620.json`

Broad rescue:

- selected rows: `8`
- eval-only positives: `4/8`
- positive bridge mass: `45,970,846`
- artifact: `local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json`
- local export: `local_runs/no_anchor_component_graph_low_vote_rescue_broad_local_export_20260620`
- local GT status: `gt_available=false`

The broad rule adds small positives `2 <-> 13` on top of the big `24 <-> 31` pair. This is useful for full-score probing, but it is not yet a canonical e2e gain.

## Replay Bug Found And Fixed

When composing rescue edges with the current referee-pruned portfolio, materialization exposed a bug:

- predicted moved tracklets: `524`
- replayed moved tracklets: `276`
- `skipped_empty_source_components: [3]`

Cause: `compose_no_anchor_portfolio_candidates.py` checked duplicate source seqs, but some preview items do not carry `source_seqs`. Two rows with the same `source_component_label` could be combined, causing the first edit to empty the source and the second to be skipped.

Fix:

- `_compatible()` now rejects duplicate source components even when explicit seqs are absent.
- Added self-test coverage for duplicate source components without `source_seqs`.
- Validation: `python -m py_compile kit/compose_no_anchor_portfolio_candidates.py` and `python kit/compose_no_anchor_portfolio_candidates.py --self-test` passed.

## Next Probe Queue

Constructed a no-GT union of:

- current referee-pruned portfolio scheduler
- broad component-graph rescue rows

Then recomposed with duplicate-source guard enabled:

- union: `local_runs/no_anchor_union_referee_pruned_plus_component_rescue_20260620.json`
- dedup portfolios: `local_runs/no_anchor_portfolio_referee_pruned_plus_component_rescue_dedup_20260620.json`
- portfolio rows: `80`

Strict scheduler after dedup:

- selected rows: `0`
- reason: component-graph rescue rows lack pair-model fields, so pair metrics fall below strict scheduler thresholds.

Probe scheduler:

- selected rows: `12`
- output: `local_runs/no_anchor_fullscore_scheduler_referee_pruned_plus_component_rescue_dedup_probe_20260620.json`
- local export: `local_runs/no_anchor_referee_pruned_plus_component_rescue_dedup_probe_local_export_20260620`
- exported rows: `12`
- zip files per row: `10`
- local GT status: `gt_available=false`
- replay check: `skipped_empty_source_components=[]`

This probe queue is ready for remote/full-score submission when the scorer is available, but it should be treated as a probe, not a high-confidence production queue.

## Current Verified Score State

No new canonical end-to-end score was produced locally because the DS1 full GT is not available in this workspace.

Standing verified scores remain:

- global-id pair model F1/P/R: `0.775234 / 0.820504 / 0.734698`
- best e2e IDF1/HOTA/AssA: `0.655240 / 0.518652 / 0.534359`

Next direction:

- Either run the 12-row dedup probe queue on remote/full-score, or train an OOD-aware component-graph proxy once direct outcomes for this family exist.
