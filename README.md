# VLINCS ReID / Global-ID Pipeline

This repository contains the current VLINCS tracklet-to-global-ID research
pipeline.  The input is a set of tracklets with bounding boxes; the output is a
submission / assignment in which every tracklet receives a `predicted_global_id`
plus component metadata.  The current best implementation is a no-anchor
pipeline: it does not use GT identity labels or fixed identity anchors for
training.  GT labels, when available, are used only for reporting and
full-score evaluation.

## WISC Branch Integration

This `wisc` branch imports the UWISC research snapshot from
`Novateur/vlincs_reid_uwisc` `main` at commit
`e47c026354988886022de336cb74c003be4e44ff` (2026-06-22).  The branch keeps the
target repository's all-branches Bitbucket pipeline and existing small demo/LFS
bundles so branch-level auto-evaluation can still run without a separate setup
change.  The no-anchor global-ID code, reports, and active research state come
from the UWISC snapshot.

## Current Best

Snapshot date: 2026-06-23.

Best reproducible WISC no-anchor DS1 delivery score:

| IDF1 | HOTA | AssA |
|---:|---:|---:|
| 0.668198 | 0.528747 | 0.539071 |

The root `./demo.sh` replays and verifies this promoted no-anchor top32
identity-decision artifact.  It runs direct export, `density_simple`, and the
fixed `p005_area` delivery gate; the final verification line should report
`IDF1/HOTA/AssA = 0.668198/0.528747/0.539071`.

Fresh checkouts must materialize the DS1 Git LFS demo data before running the
replay:

```bash
git checkout wisc
git lfs pull --include="kit/demo_data/ds1/**"
./demo.sh
```

Without Git LFS, the DS1 parquet files remain pointer text files and cannot be
scored.

Best model-side pair metric:

| Pair F1 | Precision | Recall |
|---:|---:|---:|
| 0.782244 | 0.850111 | 0.724411 |

The best promoted edit at this snapshot is the rank06 no-anchor top32
component-subset repair `37 -> 86`, followed by the fixed delivery calibration
path.  The e2e target remains above 0.70 IDF1, so the included reports and
state files are part of the active research trail rather than a final solved
result.

## Repository Layout

| Path | Purpose |
|---|---|
| `kit/` | Pipeline entrypoints, feature extraction, no-anchor resolvers, candidate generators, schedulers, full-score wrappers, and ablation tools. |
| `vlincs_gallery/` | Core gallery, weak-graph, tracklet, geometry, and scoring helpers. |
| `reports/` | Markdown reports for major experiments, ablations, promotions, and refutations. |
| `autoresearch_state/no_anchor_global_id/state/` | Lightweight state for the active no-anchor research loop. |
| `db/` | Optional local PostgreSQL / pgvector development setup. |
| `tests/` | Small unit tests for graph/consolidation helpers. |
| `gallery-ui/` | Lightweight UI code from the upstream worktree. |

Large experiment artifacts are intentionally not committed.  Full local/remote
run outputs, zip submissions, model weights, SQLite caches, and videos should
live on local scratch or object storage.  The target repository's small demo/LFS
bundles are kept only to preserve the existing branch auto-evaluation harness.

## Environment

On Pluto/H100 nodes the working environment used during this snapshot was:

```bash
export REPO_ROOT=/mnt/localssd/vlincs_reid_by_search
export RUN_ROOT=/mnt/localssd/vlincs_reid_runs
export DATA_ROOT=/mnt/localssd/vlincs_reid_data
export VENV=/mnt/localssd/vlincs_reid_venv
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/kit"
```

For a fresh local or remote environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kit/requirements-ds1.txt
export PYTHONPATH="$PWD:$PWD/kit"
```

The DS1 scripts expect VLINCS data under `DATA_ROOT`.  Do not commit raw data,
video frames, model checkpoints, or generated zip submissions into this repo.

## Auto-Evaluation Entrypoints

The root replay path is the current WISC no-anchor handoff check:

```bash
./demo.sh
```

This is intentionally different from the original online-gallery DS1 demo.  It
rebuilds the promoted global-ID submission from the committed assignment CSV and
DS1 demo tracklets, then verifies the canonical delivery score.

The branch-level auto-evaluation path is `kit/demo.sh`.  It wraps the Docker
demo command, builds the right app image, starts the DB, and runs the same
`kit/demo.py` path used by local demos.

```bash
cd kit
DEMO_HEADLESS=1 DEMO_SCORE_FILE=../ds1_score.txt ./demo.sh ds1 --no-cannot-link
```

For DS1, `kit/demo.sh ds1` enables the UWISC no-anchor weak/global-ID resolve
by default (`--weak-resolve --auto-weak-labels`, source `bbox-auto`).  Set
`WEAK_RESOLVE=0` to run only the base gallery resolve, or pass explicit
`demo.py` flags after the dataset name to override defaults.

## Main DS1 Pipeline

The fixed DS1 reproduction entrypoint is:

```bash
bash kit/run_pluto_ds1.sh h100-test-3
```

Useful overrides:

```bash
REPO_ROOT=/path/to/repo \
RUN_ROOT=/path/to/runs \
DATA_ROOT=/path/to/vlincs_reid_data \
VENV=/path/to/venv \
WEAK_RESOLVE=1 \
AUTO_WEAK_LABELS=1 \
bash kit/run_pluto_ds1.sh my-run-name
```

The script compiles the core modules, runs the DS1 demo pipeline, exports a
submission zip, and writes logs under `$RUN_ROOT/logs`.

## No-Anchor Global-ID Model

Core pair-model entrypoint:

```bash
python kit/no_anchor_global_id_model.py --help
```

The model trains from label-free evidence:

- weak positives from agreement between independent resolvers and very strong
  visual/temporal links;
- hard negatives from cannot-link constraints, same-time conflicts, impossible
  transitions, and low-similarity background pairs;
- GT labels only for reporting, diagnostics, and full-score evaluation.

For a more concrete explanation of the current verified method, the difference
between pair metrics and end-to-end IDF1, and the exact weak-positive /
hard-negative construction rules, see
[`docs/global_id_method_notes_20260621.md`](docs/global_id_method_notes_20260621.md).

Related feature builders:

```bash
python kit/extract_tracklet_foundation_features.py --help
python kit/extract_tracklet_osnet_features.py --help
python kit/make_no_anchor_fused_features.py --help
python kit/make_no_anchor_weak_metric_features.py --help
```

## Candidate Search And Referee Loop

The current best research loop follows this pattern:

1. Generate no-anchor subpart / merge / attach candidates from current
   assignments and feature views.
2. Rank candidates with side-effect heuristics or a full-score-calibrated
   referee.
3. Materialize selected assignment CSVs.
4. Full-score only a small batch.
5. Feed positive, tie, near-miss, and negative labels back into the referee.

Important entrypoints:

```bash
python kit/propose_no_anchor_subpart_repair_candidates.py --help
python kit/compose_no_anchor_subpart_repair_combos.py --help
python kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py --help
python kit/no_anchor_fullscore_scheduler.py --help
```

Full-score wrapper for generated assignment CSVs:

```bash
export PYTHON_BIN=python
export DATA_ROOT=/path/to/vlincs_reid_data
export NO_ANCHOR_FILTER_SKIP_SCORE=1

bash kit/run_no_anchor_density_area_pipeline.sh \
  /path/to/run_dir \
  @/path/to/p005_area_config.txt \
  /path/to/candidate_assignments.csv
```

This wrapper exports a submission zip, applies the `density_simple` delivery
filter, then evaluates with the configured `p005_area` detection filter.

## Reports To Start With

Read these first:

- `LATEST_NO_ANCHOR_PROGRESS.txt`
- `reports/no_anchor_highmass_from_r47r49_refutation_20260622.md`
- `reports/no_anchor_sideeffect_blacklisted_subpart_promotion_20260622.md`
- `reports/no_anchor_filter_sweep_currentbest_20260622.md`
- `reports/no_anchor_timeagglom_attach_side_effect_gate_20260622.md`

The state file for the active research loop is:

```bash
autoresearch_state/no_anchor_global_id/state/progress.json
```

## Tests

Run lightweight tests with:

```bash
python -m pytest tests
```

At minimum, syntax-check the core modules before launching a long remote run:

```bash
python -m py_compile vlincs_gallery/gallery.py vlincs_gallery/weak_graph.py \
  kit/online.py kit/no_anchor_global_id_model.py \
  kit/rank_no_anchor_subpart_candidates_by_fullscore_labels.py
```

## Publication Boundary

This repo is the lightweight code and report snapshot.  It replaces the older
agentic CCVID/MEVID package that previously lived in this Bitbucket repo.

Not included:

- raw VLINCS data and videos;
- generated `local_runs/` directories;
- submission zips and full-score zip exports;
- model checkpoints and large feature artifacts;
- SQLite metadata caches and logs.

Keep those artifacts in scratch storage or object storage and reference their
paths from reports when needed.
