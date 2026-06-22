# No-Anchor Time-Agglom Attach Critic On Rank08 Base

Date: 2026-06-22

## Question

After rank08 moved the e2e best to `0.655948`, the next question was whether
the same top-k15 time-agglom candidate family could be made less random by a
no-GT side-effect critic.

This round used rank08 as the base assignment and generated relaxed single
attach candidates from the existing DS1 time-agglom assignments. No anchors
were used. GT was used only by the canonical scorer.

## Method

`kit/compose_no_anchor_time_agglom_attach_candidates.py` was extended with
no-GT temporal/namespace features:

- source average confidence and detection count;
- source start/end rank inside its current component;
- source terminal fraction;
- source and target temporal gap around the moved tracklet;
- source component video entropy;
- target same-video fraction;
- source same-video fraction.

The new `--rank-by critic` mode keeps the old scheduler behavior untouched by
default. The critic is deliberately simple and hand-written; this experiment
tests whether it is useful enough to spend full-score budget.

Relaxed generator settings:

- base: rank08 assignment from the prior run;
- max source count: 2;
- max source component size: 80;
- min target count: 16;
- min target component size: 32;
- min target dominance: 0.65;
- same-video and same-camera overlap: 0;
- compose sizes: 1 only for first pass.

## Candidate Ranking

The critic produced 21 raw single attach candidates. The top tested rows were:

| critic rank | moved seqs | source -> target | critic | scheduler | target gap | target same-video | result |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 9435 | 24 -> 8 | 0.432187 | 0.639077 | 470 | 0.335260 | negative |
| 3 | 6234 | 31 -> 17 | 0.379085 | 0.579829 | 383 | 0.028571 | negative |
| 4 | 1249,1291 | 9 -> 50 | 0.312741 | 0.604583 | 1009 | 0.051724 | promote |
| 5 | 3322 | 31 -> 3 | 0.275339 | 0.757658 | 150 | 0.346774 | weak promote |

The hand critic is therefore not a reliable ranker yet: it under-ranked the
best single candidate and over-ranked two losers. But it did expand the search
space enough to find a new positive.

## Metrics

Canonical path:

```bash
assignment CSV
  -> full submission zip
  -> density_simple sourcezip
  -> p005 area scorer
```

| candidate | moved tracklets | IDF1 | HOTA | AssA | DetPr | DetRe | delta vs previous best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous best rank08 | 1 | 0.655948 | 0.519356 | 0.534962 | 0.764796 | 0.574223 | 0.000000 |
| critic rank01 | 1 | 0.655939 | 0.519346 | 0.534954 | 0.764771 | 0.574223 | -0.000009 |
| critic rank03 | 1 | 0.655936 | 0.519343 | 0.534950 | 0.764765 | 0.574223 | -0.000012 |
| critic rank04 | 2 | 0.656068 | 0.519526 | 0.535146 | 0.764750 | 0.574433 | +0.000120 |
| critic rank05 | 1 | 0.655963 | 0.519383 | 0.534991 | 0.764739 | 0.574278 | +0.000015 |
| combo 4+5 | 3 | 0.656083 | 0.519553 | 0.535174 | 0.764694 | 0.574488 | +0.000135 |
| combo 4+1 | 3 | 0.656059 | 0.519517 | 0.535138 | 0.764726 | 0.574433 | +0.000111 |
| combo 4+3 | 3 | 0.656057 | 0.519513 | 0.535133 | 0.764719 | 0.574433 | +0.000109 |

New best:

- IDF1: `0.655948 -> 0.656083`
- HOTA: `0.519356 -> 0.519553`
- AssA: `0.534962 -> 0.535174`

Relative to the pre-rank08 base:

- IDF1: `0.655911 -> 0.656083`
- HOTA: `0.519311 -> 0.519553`
- AssA: `0.534922 -> 0.535174`

## Verdict

Promote combo 4+5 as the current no-anchor e2e best.

This is still far below the `0.70` e2e target, but it proves that the local
time-agglom attach family can accumulate small gains when evaluated through
the correct sourcezip path.

The hand-written critic should not be trusted as a final ranker. Its main
output is a better labeled dataset:

- positives: rank08, critic rank04, critic rank05, combo 4+5;
- hard negatives: prior rank09-11, critic rank01, critic rank03, combo 4+1,
  combo 4+3.

Next direction: train or calibrate a tiny attach ranker from these tested
local edits using only no-GT features, then generate a larger candidate pool
from the current combo 4+5 base. If the learned ranker cannot separate rank04
from rank01/rank03, pivot away from time-agglom attach and toward stronger
visual/namespace evidence.

## Artifacts

Remote:

`/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_critic_rank08base_20260622`

Local mirror:

`local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_critic_rank08base_20260622/`

Key files:

- `critic_attach_candidates_v2.json`
- `manifest_assignments_v2.json`
- `rank04_time_agglom_local_attach_source_assignments_density_p005_area.json`
- `rank05_time_agglom_local_attach_source_assignments_density_p005_area.json`
- `combo_4_5_assignments_density_p005_area.json`
- `combo_eval_inputs/combo_4_5_assignments.csv`
