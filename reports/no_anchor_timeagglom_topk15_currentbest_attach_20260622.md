# No-Anchor Time-Agglom Top-k15 Attach To Current Best

Date: 2026-06-22

## Deli AutoResearch Distillation

The useful part of the public Deli AutoResearch thread is an operating
protocol, not a particular model: keep durable state files, execute once a
candidate is ready, separate worker/referee roles, and downgrade honestly when
external checks contradict the story.

For this VLINCS loop, that translates to four concrete rules:

1. State lives in `autoresearch_state/no_anchor_global_id/state/`, not in chat.
2. A proxy candidate is not evidence until it is materialized and scored by the
   canonical DS1 gate.
3. The evaluator path is part of the experiment definition.
4. Near-miss failures become hard negatives for the next side-effect critic.

Sources:

- https://victorchen96.github.io/auto_research/framework.html
- https://victorchen96.github.io/auto_research/paper.html
- https://victorchen96.github.io/blog_self_play_story.html
- https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating

## Question

The BoTSORT sample result promoted `time_agglom top_k=15 theta=0.035` to
sample identity F1 `0.702092`, but direct DS1 replacement was poor
(`0.587838` IDF1). This experiment tests the narrower hypothesis:

Can the DS1 `time_agglom top_k=15` output act as a local high-precision
candidate generator attached to the current best no-anchor assignment?

No anchors were used. GT is used only for final benchmark scoring.

## Candidate Generator

Base assignment:

`/mnt/localssd/vlincs_reid_runs/no_anchor_opponent_scheduler_labelled_w0p016_fullscore_20260621/assignments/rank01_conflict_subcluster_reassign_candidate_search_augmented_candidates_assignments.csv`

Time-agglom candidate assignments:

- `theta=0.025`
- `theta=0.035`
- `theta=0.045`

Filter:

- top_k = 15
- min_dets = 10
- exclude_same = camera
- temporal_bonus = 0.005
- time_window_ms = 1000
- max same-video overlap = 0
- max same-camera overlap = 0
- target dominance >= 0.74
- source component <= 40 tracklets
- target component >= 32 tracklets

The generator found 4 raw single attaches, then composed them into 11 selected
ranked candidates. Only the 4 single attaches were fully evaluated here because
they are interpretable and have bounded side effects.

| rank | moved seq | source component | target component | scheduler score | target dominance | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 08 | 4973 | 58 | 55 | 0.863542 | 0.902778 | promote |
| 09 | 3558 | 2330 | 2329 | 0.852256 | 0.877698 | refute |
| 10 | 4580 | 61 | 21 | 0.836979 | 0.843750 | refute |
| 11 | 4859 | 61 | 29 | 0.769792 | 0.854167 | refute |

All four had weak-video source fraction `1.0` and same-video/same-camera
overlap `0`.

## Canonical Eval Path

The correct comparable path is:

```bash
assignment CSV
  -> full submission zip, no GT scoring
  -> density_simple source-zip filtering
  -> p005 area filter
  -> DS1 evaluate_submission_detection_filter.py
```

A direct assignment-CSV filter path is not equivalent to the current best,
because the promoted current best is a `density_simple sourcezip` artifact.
During this round an attempted runner missed `DATA_ROOT=/mnt/localssd/vlincs`;
that produced an empty HOTA input in `density_simple`. The run was discarded
and rerun with the correct environment.

To prevent this pitfall recurring, two helper scripts were added:

- `kit/export_no_anchor_assignment_zip.py`
- `kit/run_no_anchor_density_area_pipeline.sh`

## Metrics

| candidate | IDF1 | HOTA | AssA | DetPr | DetRe | rows | delta IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base densityzip p005 | 0.655911 | 0.519311 | 0.534922 | 0.764814 | 0.574156 | 1518065 | 0.000000 |
| rank08 single attach | 0.655948 | 0.519356 | 0.534962 | 0.764796 | 0.574223 | 1518065 | +0.000037 |
| rank09 single attach | 0.655898 | 0.519295 | 0.534904 | 0.764771 | 0.574160 | 1518065 | -0.000013 |
| rank10 single attach | 0.655843 | 0.519245 | 0.534875 | 0.764692 | 0.574121 | 1518065 | -0.000068 |
| rank11 single attach | 0.655805 | 0.519197 | 0.534802 | 0.764691 | 0.574063 | 1518065 | -0.000106 |

## Verdict

Promote rank08 as the new no-anchor e2e best:

- IDF1: `0.655911 -> 0.655948`
- HOTA: `0.519311 -> 0.519356`
- AssA: `0.534922 -> 0.534962`

This is a micro-gain, not a path to 0.70 by itself. The important research
signal is sharper:

- top-k15 time agglomeration is useful as a precision-biased local proposer;
- scheduler score alone cannot distinguish the one safe attach from three
  very similar losers;
- rank09-11 should become near-miss hard negatives for the next no-anchor
  side-effect critic.

Do not spend more full-score budget on unvetted combo candidates from this
family. The next structural direction should train or calibrate a tiny-attach
side-effect critic using rank08 as a positive and rank09-11 as hard negatives,
then pivot to detector/namespace features if it cannot separate them.

## Artifacts

Local mirror:

`local_runs/remote_h100_test_3_20260622/no_anchor_timeagglom_attach_currentbest_20260622/`

Remote run:

`/mnt/localssd/vlincs_reid_runs/no_anchor_timeagglom_attach_currentbest_20260622`

Key files:

- `timeagglom_attach_candidates.json`
- `timeagglom_attach_candidates.csv`
- `manifest_assignments.json`
- `rank08_time_agglom_local_attach_source_assignments_density_p005_area.json`
- `rank09_time_agglom_local_attach_source_assignments_density_p005_area.json`
- `rank10_time_agglom_local_attach_source_assignments_density_p005_area.json`
- `rank11_time_agglom_local_attach_source_assignments_density_p005_area.json`
