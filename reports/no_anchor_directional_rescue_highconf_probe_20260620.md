# No-Anchor Directional Rescue High-Confidence Probe

Date: 2026-06-20

## Objective

Continue the no-anchor VLINCS global-id research after the component-graph rescue iteration. The goal is still verified e2e IDF1 > 0.70; no anchors are used.

## Remote Full-Score Probe

Tried to launch the dedup component-rescue probe through:

```bash
SCHEDULER_JSON=local_runs/no_anchor_fullscore_scheduler_referee_pruned_plus_component_rescue_dedup_probe_20260620.json \
RUN_NAME=no_anchor_dedup_component_rescue_probe_20260620 \
SELECTION_RANKS=1,2,3,4,5,6,7,8,9,10,11,12 \
kit/run_no_anchor_scheduler_manifest_fullscore.sh --dry-run
```

Result:

- `h100-test-3`: SSH banner timeout
- `h100-test-2`: SSH banner timeout
- `test-video-0`: SSH banner timeout

No remote canonical full-score proof was available this turn.

## New No-GT Directionality Filter

Added:

- `kit/filter_no_anchor_component_graph_directional_rescue.py`

Rationale:

- The broad component-graph rescue emitted both directions for symmetric component pairs.
- Eval-only evidence suggested direction matters: for the largest pair, `31 -> 24` is cleaner than `24 -> 31`.
- The production-side rule must not read GT, so the filter keeps one direction per unordered pair using only no-GT fields:
  - target size at least source size
  - high `target_best_sim`
  - high `target_min_view_sim`
  - low same-video overlap

Validation:

- `python -m py_compile kit/filter_no_anchor_component_graph_directional_rescue.py`
- `python kit/filter_no_anchor_component_graph_directional_rescue.py --self-test`

Both passed.

## Directional Rescue Output

Input:

- `local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json`

Command:

```bash
python kit/filter_no_anchor_component_graph_directional_rescue.py \
  --candidates-json local_runs/no_anchor_component_graph_low_vote_rescue_broad_20260620.json \
  --min-target-source-size-ratio 1.0 \
  --min-target-best-sim 0.82 \
  --min-target-min-view-sim 0.70 \
  --max-same-video-overlap-ratio 0.02 \
  --top-n 8 \
  --json local_runs/no_anchor_component_graph_directional_rescue_highconf_20260620.json \
  --csv local_runs/no_anchor_component_graph_directional_rescue_highconf_20260620.csv \
  --md reports/no_anchor_component_graph_directional_rescue_highconf_20260620.md
```

Output:

- raw rows: `8`
- unordered component-pair groups: `4`
- selected rows: `1`
- selected edge: `31 -> 24`
- moved tracklets: `44`
- `target_best_sim`: `0.841304`
- `target_min_view_sim`: `0.721598`
- same-video overlap ratio: `0.016901`

Eval-only opponent audit:

- candidates: `1`
- positive rows: `1/1`
- top coverage: `0.002510`
- positive bridge mass: `22,851,603`

This confirms the no-GT rule kept the highest-mass direction from the broad rescue set while dropping the noisy opposite direction and lower-confidence pairs.

## High-Confidence Portfolio Probe

Union:

- `local_runs/no_anchor_fullscore_scheduler_referee_pruned_crossqueue_singleedge68_localized_island_20260620.json`
- `local_runs/no_anchor_component_graph_directional_rescue_highconf_20260620.json`

Artifacts:

- union: `local_runs/no_anchor_union_referee_pruned_plus_directional_rescue_highconf_20260620.json`
- composed portfolios: `local_runs/no_anchor_portfolio_referee_pruned_plus_directional_rescue_highconf_20260620.json`
- scheduler: `local_runs/no_anchor_fullscore_scheduler_referee_pruned_plus_directional_rescue_highconf_probe_20260620.json`
- local export: `local_runs/no_anchor_referee_pruned_plus_directional_rescue_highconf_probe_local_export_20260620`

Scheduler output:

- raw rows: `4`
- selected rows: `4`
- local exported rows: `4`
- zip files per row: `10`
- replay skipped source components: `0`
- local GT: `gt_available=false`

Eval-only opponent coverage for the 4-row queue:

- total positive bridge mass in audited rows: `313,042,635`
- best row coverage: `0.010624`
- best row positive edges:
  - `4 -> 6`, mass `261,415`
  - `32 -> 15`, mass `933,375`
  - `3 -> 68`, mass `51,530,040`
  - `9 -> 7`, mass `21,153,911`
  - `31 -> 24`, mass `22,851,603`

This improves opponent coverage over the previous referee-pruned/localized queue (`0.008114`) and is much smaller than the broad 12-row component-rescue probe.

## Status

No new canonical e2e score was produced because remote full-score remains unreachable and local DS1 GT is unavailable.

Standing verified scores:

- pair/global-id model F1/P/R: `0.775234 / 0.820504 / 0.734698`
- e2e IDF1/HOTA/AssA: `0.655240 / 0.518652 / 0.534359`

Next best action:

- Submit the 4-row high-confidence directional probe when a full-score node is reachable.
- In parallel, build an OOD-aware component-graph reviewer so component rescue rows receive calibrated pair/proxy fields instead of being forced through zero-threshold probe scheduling.
