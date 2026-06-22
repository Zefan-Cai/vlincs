# No-Anchor Portfolio Embedding Choice Referee

## Purpose

This is the portfolio-level follow-up to the single-edge source-local referee.
The hypothesis was:

> If the same source component appears with multiple target components inside
> one portfolio, choose a single target by embedding/quality evidence before
> exporting the assignment.

## Implementation

- Code: `kit/rerank_no_anchor_portfolio_by_embedding_choice.py`
- Input queue: `local_runs/no_anchor_video_focus_portfolio_moderate_candidates_20260620.json`
- Base assignment: `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_softcut_then_softoverlap_base_assignments_20260620.csv`
- Embeddings: `kit/demo_data/ds1/embeddings/*/embeddings.npz`

The script scores each accepted-preview item with the same no-GT evidence as
the single-edge referee:

- source-to-target crop embedding support
- target detection count and confidence
- existing scheduler visual similarities

Inside each portfolio row, if one source component maps to multiple targets,
the script keeps the target with the higher average choice score and suppresses
the other preview items.

## Conflict Rate

On the moderate video-focus portfolio candidate file:

- input rows: `16`
- rows with same-source multi-target conflict before cleaning: `8`
- suppressed preview items after cleaning: `8`
- tracklet embeddings loaded: `3880`

Cleaned artifact:

- `local_runs/no_anchor_video_focus_portfolio_moderate_embedding_choice_20260620.json`
- `reports/no_anchor_video_focus_portfolio_moderate_embedding_choice_20260620.md`

Scheduler output:

- `local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_embedding_choice_20260620.json`
- `reports/no_anchor_fullscore_scheduler_video_focus_moderate_embedding_choice_20260620.md`
- local export: `local_runs/no_anchor_video_focus_moderate_embedding_choice_local_export_20260620`

## Eval-Only Comparison

This audit is not used for production selection.

Original moderate scheduler:

- `local_runs/no_anchor_candidate_false_split_coverage_video_focus_moderate_original_20260620.json`
- top coverage: `0.002338`
- summed positive bridge mass across top rows: `63863193`

Embedding-choice cleaned scheduler:

- `local_runs/no_anchor_candidate_false_split_coverage_video_focus_moderate_embedding_choice_20260620.json`
- top coverage: `0.002323`
- summed positive bridge mass across top rows: `21555371`

## Interpretation

The portfolio-level blanket rule is too aggressive.

Single-edge duplicate-source arbitration is useful because the whole candidate
is one source-to-one-target decision. In a portfolio, however, same-source
multi-target edits can be legitimate: a large impure component may contain
multiple identity fragments, and sending disjoint source-seq islands from that
component to different targets can recover more positive bridge mass.

So the production lesson is narrower:

- keep the single-edge source-local referee;
- do not globally enforce one target per source inside portfolios;
- only apply portfolio source-target arbitration when the losing target has a
  very low choice score or when cannot-link/chain checks show that the split is
  physically impossible.

This is a good Deli-style result: the structural pivot reduced one visible
conflict class, but the opponent audit shows it also removes useful repair
mass. The next version should be margin-gated, not absolute.
