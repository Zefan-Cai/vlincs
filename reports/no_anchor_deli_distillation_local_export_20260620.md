# No-Anchor Global-ID: Deli Distillation + Local Export

Date: 2026-06-20

This note distills the public Deli AutoResearch thread into the current VLINCS
no-anchor global-ID loop and records the new local manifest-export path.

## Deli AutoResearch distilled for this task

Useful operating rules:

1. Persist state in files, not conversation context.
2. Treat metric drops and missing findings as evidence, not failure.
3. Separate proposer, executor, scorer, and judge.
4. When an experiment is ready, execute or materialize it; do not stop at
   "ready to submit".
5. If stale, change the structural constraint rather than tuning the same
   threshold harder.
6. Keep every claimed improvement tied to a reproducible artifact.

Mapping into VLINCS:

- proposer: no-anchor candidate generators such as
  `compose_no_anchor_video_focus_portfolio_candidates.py`;
- scheduler: `no_anchor_fullscore_scheduler.py` and video-focus reranking;
- executor: `export_no_anchor_scheduler_manifest_assignments.py`,
  `run_no_anchor_scheduler_manifest_fullscore.sh`, and now the local parquet
  runner below;
- judge: canonical DS1 full scorer when GT is mounted, plus result-gate parser
  and opponent guard;
- state: `autoresearch_state/no_anchor_global_id/state/*.json*`.

Important discipline:

- pair F1 above 0.70 is not a completion signal;
- predicted full IDF1 is only a budget allocator;
- e2e progress is recognized only after canonical full scoring;
- GT/per-video scores can guide research budget, but not training labels,
  anchors, or production assignment evidence.

## Local manifest export path

Added:

- `kit/run_no_anchor_scheduler_manifest_sample_fullscore.py`
- `--allow-no-gt-export` in `kit/evaluate_sample_assignments_full.py`

Purpose:

- materialize scheduler rows locally without PostgreSQL;
- merge tracklet-level assignment CSVs with DS1 detection-level tracklet
  parquets under `kit/demo_data/ds1/tracklets`;
- export submission zips even on machines where DS1 GT is not mounted;
- score with the canonical evaluator automatically when local DS1 GT is
  available.

Validation:

```bash
python -m py_compile \
  kit/evaluate_sample_assignments_full.py \
  kit/run_no_anchor_scheduler_manifest_sample_fullscore.py \
  kit/export_no_anchor_scheduler_manifest_assignments.py

python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py --self-test
```

Both passed.

## Recovered local base assignment

The remote canonical base
`/mnt/localssd/vlincs_reid_runs/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv`
is not present locally.  I reconstructed a close local base by majority vote
over the seven returned `90xxxxxx`-namespace assignment CSVs.

Artifacts:

- `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_90family_base_assignments_20260620.csv`
- `local_runs/remote_h100_test_3_20260620/no_anchor_recovered_90family_base_assignments_20260620.json`

Recovery stats:

- input CSVs: `7`
- recovered rows: `7487`
- missing seqs across the seven CSVs: `0`
- unstable seqs after majority vote: `40`
- no anchors or GT used

This is a local execution base, not a canonical scoring claim.  Canonical
remote scoring should still use the original remote base if the node recovers.

## Moderate video-focus top-8 local export

Command:

```bash
python kit/run_no_anchor_scheduler_manifest_sample_fullscore.py \
  --scheduler-json local_runs/no_anchor_fullscore_scheduler_video_focus_moderate_portfolio_20260620.json \
  --base-assignment-csv local_runs/remote_h100_test_3_20260620/no_anchor_recovered_90family_base_assignments_20260620.csv \
  --run-dir local_runs/no_anchor_video_focus_moderate_portfolio_recovered_base_local_export_20260620 \
  --selection-ranks 1,2,3,4,5,6,7,8 \
  --allow-no-gt-export
```

Output directory:

- `local_runs/no_anchor_video_focus_moderate_portfolio_recovered_base_local_export_20260620`

Summary:

| rank | moved tracklets | predicted full IDF1 | pair F1 | zip |
|---:|---:|---:|---:|---|
| 1 | 52 | 0.658768 | 0.768474 | `rank01_...assignments.zip` |
| 2 | 52 | 0.658286 | 0.770803 | `rank02_...assignments.zip` |
| 3 | 52 | 0.658781 | 0.766403 | `rank03_...assignments.zip` |
| 4 | 52 | 0.658257 | 0.768903 | `rank04_...assignments.zip` |
| 5 | 52 | 0.658737 | 0.766686 | `rank05_...assignments.zip` |
| 6 | 52 | 0.658897 | 0.766042 | `rank06_...assignments.zip` |
| 7 | 52 | 0.658485 | 0.767485 | `rank07_...assignments.zip` |
| 8 | 56 | 0.658162 | 0.769424 | `rank08_...assignments.zip` |

Local GT status:

- `gt_available=false`
- local `load_ds1_gt_by_video()` returned no videos because DS1 GT is not
  mounted on this Mac;
- each zip contains 10 video parquets and about 1.72M detection rows;
- run directory size is about `655M`.

Next executable step:

- if a Pluto node becomes reachable, run the canonical DB scorer with the
  original remote base;
- if only local filesystem is available, mount/sync DS1 GT under `DATA_ROOT`
  and rerun the same local command to get canonical full metrics from the
  already materialized assignment/zip path.

