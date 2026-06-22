# No-Anchor Deli Self-Play Bridge Ablation

Date: 2026-06-20

## Objective

Apply the Deli AutoResearch lesson "opponent reviewers may lower the score" to the current VLINCS no-anchor queue.

The top assignment-summary candidate already includes the committed component-graph bridge `31 -> 24`, so the next useful question is whether non-committed component-graph admissions should be promoted:

- provisional `44 -> 8`
- quarantine `13 -> 2`
- quarantine `17 -> 47`
- all three non-committed bridges together

## Inputs

- base assignment: `local_runs/no_anchor_local_pervideo_source_selector_balanced_20260620_assignments.csv`
- admission source: `local_runs/no_anchor_component_graph_admission_20260620.json`
- proxy reviewer: `kit/analyze_no_anchor_assignment_summary_proxy.py`

## Assignment-Summary Reviewer Result

The base balanced selector remains highest:

| candidate | changed tracklets | assignment-summary predicted full IDF1 | decision |
| --- | ---: | ---: | --- |
| balanced base | 0 | `0.664464` | keep as queue rank 1 |
| balanced + provisional `44 -> 8` | 130 | `0.663957` | low-priority probe |
| balanced + quarantine `17 -> 47` | 140 | `0.663641` | do not prioritize |
| balanced + quarantine `13 -> 2` | 223 | `0.661941` | reject for now |
| balanced + all non-committed bridges | 493 | `0.660916` | reject; batch promotion is harmful |

Interpretation: the admission/quarantine layer is doing useful work. The provisional edge is close enough to export as a probe, but the quarantine rows should not be batch-promoted without a stronger adversarial referee.

## Exported Probe Zips

Two zips were exported with local DS1 tracklet parquets and `gt_available=false`:

- `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank01_balanced_plus_provisional_44_to_8.zip`
- `local_runs/no_anchor_deli_selfplay_bridge_ablation_20260620/rank03_balanced_plus_quarantine_17_to_47.zip`

Queue manifest:

- `local_runs/no_anchor_deli_selfplay_bridge_ablation_queue_20260620.json`

These are ready for canonical remote full scoring when Pluto/SSH recovers, but they are not completion evidence.

## Gate/Opponent

Protocol audit passed:

- `local_runs/no_anchor_autoresearch_protocol_audit_selfplay_bridge_ablation_20260620.json`
- hard blockers: `0`

Result gate remains:

- `pass_joint=false`
- best newly gated full IDF1: `0.654009`
- standing best e2e IDF1: `0.655240`

Deli opponent verdict remains `pivot`:

- pair/global metric passes, but full DS1 IDF1 is below target;
- no newly gated artifact beats the standing e2e best;
- stale_count requires structural pivot, not selector/threshold sweeps.

## Next

This branch should not consume broad full-score budget. Spend at most one remote slot on provisional `44 -> 8` after the primary assignment-summary queue is scored. The next structural step should be a stronger adversarial component-bridge verifier or restoration of canonical full scoring.
