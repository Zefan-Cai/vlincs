# No-Anchor Proxy-Reviewer AutoResearch Iteration

Date: 2026-06-20

## Distilled Deli AutoResearch Protocol

Sources:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html
- https://victorchen96.github.io/auto_research/skill/paper-writing.html

The transferable idea is not a model architecture. It is a research operating
protocol:

1. Persist state in files, not chat memory.
2. Treat "ready" as execute: once a candidate is materialized, produce the
   artifact and validation output.
3. Separate worker and referee. The proposer should not be the only evaluator.
4. Use honest score movement. A lower score is evidence and can force a pivot.
5. Use reviewer diversity and medians so one optimistic judge cannot dominate.
6. After repeated stalls, pivot structurally rather than tuning the same
   thresholds.

## VLINCS Translation

For no-anchor global ID:

- Worker/proposer: creates assignment edits from tracklet evidence.
- Executor: materializes scheduler rows into assignment CSVs and submission zips.
- Referee: ranks rows using no-GT metadata/proxy models.
- Opponent/audit: uses GT only after the fact to explain failures, never to
  select production rows.

Current standing state:

- global-id pair model: F1 `0.775234`, precision `0.820504`, recall `0.734698`;
- best verified e2e full IDF1: `0.655240`;
- target: full e2e IDF1 `0.70`.

## Local Full-Score Feasibility Check

The local DS1 full scorer is not usable on this Mac for current sample exports:

- `load_ds1_gt_by_video()` returns `0` GT video keys;
- the 10 demo DS1 tracklet parquet files contain only detection/tracklet fields:
  `video_key`, `local_track_id`, `tracklet_key`, `det_id`, `frame_idx`, `x1`,
  `y1`, `x2`, `y2`, `score`, `coco_cls`;
- those parquets have no `gt_id` or `tracklet_majority_gt_id`;
- the only GT-like local parquet found is `kit/demo_data/ms02/tracklets.parquet`,
  which is not the current DS1 full-score sample set.

Therefore local runs may export assignment/submission artifacts, but must not
claim a new canonical IDF1.

## New Referee Layer

Added:

- `kit/rerank_no_anchor_scheduler_by_proxy_ensemble.py`

This script treats historical full-score proxy models as independent reviewers:

- delivery-aware ridge model;
- compact full-score ridge model;
- mass-feature ridge model;
- refreshed AutoResearch proxy model.

For each no-anchor candidate row, it computes:

- reviewer scores;
- median predicted full IDF1;
- min/max/std across reviewers;
- number of reviewers above current best;
- an uncertainty-penalized scheduler score.

No GT labels, anchors, or identity labels are used at selection time.

## Strict Referee Result

Strict run:

- input candidates: `33`;
- eligible: `0`;
- selected: `0`;
- requirement: median full proxy >= current best `0.655240`, and at least 2
  reviewers above current best.

Interpretation:

The previous single-proxy scheduler was too optimistic for this candidate
family. Several rows that looked like `0.66+` under one inherited proxy collapse
to roughly `0.653-0.655` under independent proxy-reviewer median. This is a
Deli-style negative result: it lowers our confidence rather than preserving an
inflated score.

## Exploration Queue

Relaxed exploration run:

- input candidates: `33`;
- eligible: `21`;
- selected: `8`;
- median full proxy range among selected rows: `0.653043` to `0.654675`;
- reviewers above current best: `0` for every selected row.

Top selected rows:

| rank | median | min | max | std | moved family |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.653775` | `0.652423` | `0.653814` | `0.000592` | `26+28+33+40 -> 21+55` |
| 2 | `0.653249` | `0.653238` | `0.654311` | `0.000462` | `32+4+9 -> 15+6+7` |
| 3 | `0.653451` | `0.652644` | `0.653485` | `0.000355` | `21+28+33 -> 0+19+55` |
| 4 | `0.654675` | `0.653918` | `0.654711` | `0.000335` | `0+21+28+33+4 -> 19+55+6` |

Materialization:

- exported rank assignment CSVs: `8`;
- exported submission zips: `8`;
- local GT available: `false`;
- best local IDF1: `null`.

## Artifacts

- `kit/rerank_no_anchor_scheduler_by_proxy_ensemble.py`
- `local_runs/no_anchor_proxy_reviewer_ensemble_scheduler_20260620.json`
- `local_runs/no_anchor_proxy_reviewer_ensemble_exploration_scheduler_20260620.json`
- `reports/no_anchor_proxy_reviewer_ensemble_scheduler_20260620.md`
- `reports/no_anchor_proxy_reviewer_ensemble_exploration_scheduler_20260620.md`
- `local_runs/no_anchor_proxy_reviewer_ensemble_exploration_local_export_20260620/summary.json`
- `local_runs/no_anchor_proxy_reviewer_ensemble_exploration_local_export_20260620/*.zip`

## Next Direction

Do not spend the next thinking cycle on more rank tweaks inside the same
portfolio family. The proxy-reviewer ensemble says this family is likely below
current best. Keep the exported queue for remote validation when Pluto/SSH
recovers, but the research pivot should be structural:

1. learn a no-GT proposer for high-mass false-split identities rather than
   composing more low-margin portfolios;
2. use proxy-reviewer median as an admission gate before materialization;
3. keep eval-only audits only as refutation/explanation, not production
   selection.
