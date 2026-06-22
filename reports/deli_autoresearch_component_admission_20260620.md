# Deli AutoResearch Distillation: Component Admission Iteration

Date: 2026-06-20

## Distilled Protocol

Source pages:

- `https://victorchen96.github.io/auto_research/framework.html`
- `https://victorchen96.github.io/auto_research/paper.html`
- `https://victorchen96.github.io/blog_self_play_story.html`

The useful transfer is operational, not a new model architecture:

1. Persist state to files. Chat context is not the research state.
2. Ready means execute. Once a candidate can be materialized, export the artifact.
3. Treat score drops and reviewer rejection as information.
4. Separate proposer, admission referee, executor, and eval-only opponent.
5. After stalls, pivot structural constraints rather than retuning thresholds.

For VLINCS no-anchor global ID, that maps to:

- proposer: no-GT component/tracklet candidate generator;
- admission referee: no-GT status assignment, `committed_probe / provisional_probe / quarantine`;
- executor: assignment CSV and submission zip materializer;
- opponent: GT/eval-only explanation after the fact, never production selection.

## External Scoring Blockers

Local DS1 GT is not mounted:

- `DATA_ROOT=/mnt/datastore2_videolincs/data`
- local `DATA` path does not exist on this Mac;
- `load_ds1_gt_by_video()` returns 0 video keys.

Remote scoring is unavailable this turn:

- Pluto CLI `job status` for `h100-test-3`, `h100-test-2`, and `test-video-0` returned `Failed to connect to Pluto service`;
- the bare SSH names `h100-test-3`, `h100-test-2`, and `test-video-0` are not usable host aliases;
- old Pluto SSH configs still exist for `pluto-prod-zcai-h100-test-3-0` and `pluto-prod-zcai-h100-test-2-0`, but direct probes through those configs still timed out during SSH banner exchange at the Pluto gateway.

Therefore no canonical e2e score is claimed in this iteration.

## New Admission Layer

Added:

- `kit/admit_no_anchor_component_graph_candidates.py`

Input:

- `local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json`

Output:

- `local_runs/no_anchor_component_graph_admission_20260620.json`
- `local_runs/no_anchor_component_graph_admission_20260620.csv`
- `reports/no_anchor_component_graph_admission_20260620.md`

Admission result:

- raw rescue rows: `8`
- unordered component-pair groups: `4`
- committed probes: `1`
- provisional probes: `1`
- quarantined groups: `2`
- rejected direction duplicates: `4`

Committed probe:

- `31 -> 24`
- moved tracklets: `44`
- target best similarity: `0.841304`
- target min-view similarity: `0.721598`
- same-video overlap ratio: `0.016901`
- risk retained in provenance: `spiky_low_vote_evidence`

Provisional probe:

- `44 -> 8`
- moved tracklets: `130`
- target best similarity: `0.765438`
- target min-view similarity: `0.701892`
- same-video overlap ratio: `0.015240`

Quarantine:

- `13 -> 2`
- `17 -> 47`

The important behavior is that broad rescue rows no longer flow straight into
full-score budget. They first become explicit identity-resolution states.

## Materialized Admission-Gated Queue

Union:

- `local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- committed rows from `local_runs/no_anchor_component_graph_admission_20260620.json`

Artifacts:

- union: `local_runs/no_anchor_union_referee_pruned_plus_component_admission_20260620.json`
- portfolios: `local_runs/no_anchor_portfolio_referee_pruned_plus_component_admission_20260620.json`
- scheduler: `local_runs/no_anchor_fullscore_scheduler_referee_pruned_plus_component_admission_probe_20260620.json`
- local export: `local_runs/no_anchor_referee_pruned_plus_component_admission_probe_local_export_20260620`

Replay/export result:

- selected scheduler rows: `4`
- exported rows: `4`
- zip files per row: `10`
- `skipped_empty_source_components=[]`
- `gt_available=false`

Eval-only opponent audit, for explanation only:

- audited rows: `4`
- top false-split gap coverage: `0.010624`
- total positive bridge mass over audited rows: `313,042,635`
- best row positive bridge mass: `96,730,344`

Best-row useful edges:

- `3 -> 68`, mass `51,530,040`
- `9 -> 7`, mass `21,153,911`
- `31 -> 24`, mass `22,851,603`
- `32 -> 15`, mass `933,375`
- `4 -> 6`, mass `261,415`

This preserves the previous strongest eval-only coverage while adding a no-GT
admission/quarantine state that is easier to inspect and safer to schedule.

## Current Score State

Verified pair/global-ID model:

- F1: `0.775234`
- precision: `0.820504`
- recall: `0.734698`

Verified e2e:

- IDF1: `0.655240`
- HOTA: `0.518652`
- AssA: `0.534359`

The global-ID model threshold is satisfied, but the end-to-end threshold is not.
The next decisive step is still canonical full scoring once Pluto/GT access is
restored.
