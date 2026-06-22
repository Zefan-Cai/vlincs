# Deli AutoResearch Distillation For VLINCS No-Anchor Global ID

## Source Material

- Framework: https://victorchen96.github.io/auto_research/framework.html
- Paper index: https://victorchen96.github.io/auto_research/paper.html
- Self-play story: https://victorchen96.github.io/blog_self_play_story.html

## What To Keep

Deli AutoResearch is useful here as an operating protocol, not as executable code. The important transferable pieces are:

1. Persistent state, not chat memory. Each iteration must update `state/progress.json`, append findings, and leave reusable artifacts.
2. Ready means execute. Once a candidate is materialized, produce the assignment artifact immediately instead of stopping at a plan.
3. Honest score movement. A negative or lower-scoring result is evidence and should close or pivot a direction, not be hidden.
4. Worker/referee separation. The candidate generator should not be the only judge. Use a separate opponent/audit/referee layer.
5. Structural pivots after stalls. If local parameter sweeps saturate, change the proposal structure.

## Mapping To VLINCS

The active no-anchor loop already has durable state under:

- `autoresearch_state/no_anchor_global_id/state/progress.json`
- `autoresearch_state/no_anchor_global_id/state/findings.jsonl`
- `autoresearch_state/no_anchor_global_id/state/directions_tried.json`

For VLINCS global ID, the Deli-style roles map cleanly:

- Proposer: generates no-anchor candidate repairs from tracklet evidence.
- Executor: materializes candidate rows into assignment CSVs and submission zips.
- Referee: ranks or arbitrates candidates using no-GT evidence only.
- Opponent/audit: uses GT only after the fact to explain failure modes, never to select production rows.

## New Research Step

The previous single-edge queue had a concrete failure:

- Scheduler rank 6: `3 -> 45`, selected first for source component `3`, but eval-only gap coverage was zero.
- Scheduler rank 7: `3 -> 68`, same source component, slightly lower scheduler score, but eval-only coverage was `0.005659`.

I added a source-local referee:

- Code: `kit/rerank_no_anchor_single_edge_by_embedding_choice.py`
- Inputs:
  - `local_runs/no_anchor_single_edge_edge_table_focused_candidates_20260620.json`
  - `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv`
  - `kit/demo_data/ds1/embeddings/*/embeddings.npz`
- Output:
  - `local_runs/no_anchor_single_edge_embedding_choice_candidates_20260620.json`
  - `reports/no_anchor_single_edge_embedding_choice_candidates_20260620.md`

The referee only arbitrates duplicate-source candidates. It computes target-choice score from:

- source-tracklet to target-component embedding support: top-5, p99, max cosine
- target quality: detection count and average detector confidence
- existing scheduler support: fused, DB, and primary visual similarities

It preserves global scheduler ordering after choosing the best target for each source.

## Result

The source-local referee selected 7 rows from 8 candidates and suppressed the duplicate-source loser:

- kept: `3 -> 68`, choice score `0.831753`
- suppressed: `3 -> 45`, choice score `0.762733`

The clean candidate report:

- `reports/no_anchor_single_edge_embedding_choice_candidates_20260620.md`

Eval-only audit, for explanation only:

- `reports/no_anchor_candidate_false_split_coverage_single_edge_embedding_choice_20260620.md`
- candidates: `7`
- top coverage: `0.005659`
- positive bridge mass: `51530040`
- positive edge: `3 -> 68`

Materialized assignment artifacts:

- `local_runs/no_anchor_single_edge_embedding_choice_local_export_20260620/summary.json`
- rank06 zip: `local_runs/no_anchor_single_edge_embedding_choice_local_export_20260620/rank06_edge_table_single_edge_source_assignments.zip`
- rank06 edit: target component `[68]`, moved tracklets `248`

## Interpretation

This is a small but real Deli-style improvement to the research loop:

- It does not claim a new canonical e2e score because local DS1 GT and remote full scoring are still unavailable.
- It does produce an executable no-anchor candidate that fixes a documented source-target arbitration failure.
- It turns a previous audit-only observation into a no-GT production heuristic: duplicate-source target choice should be decided by a separate embedding/quality referee before assignment materialization.

## Next Direction

Generalize the source-local referee beyond single-edge rows:

1. Apply duplicate-source target-choice arbitration to hub/portfolio candidates before composing portfolios.
2. Add duplicate-target/source-chain conflict checks so portfolios do not lose useful high-mass edges by ordering accident.
3. When Pluto or local GT returns, full-score rank06 and compare against the standing e2e best IDF1 `0.655240`.
