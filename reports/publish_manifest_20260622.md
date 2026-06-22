# VLINCS Publish Manifest

Date: 2026-06-22

## Current Standing

- Setting: no-anchor global-id research.
- Global-id pair model: F1 / precision / recall =
  `0.775234 / 0.820504 / 0.734698`.
- End-to-end best: IDF1 / HOTA / AssA =
  `0.657887 / 0.520944 / 0.535983`.
- Goal remains open: end-to-end IDF1 is still below `0.70`.

## Main Recent Promotions

Recall-guarded subpart repair promoted the current no-anchor e2e best.
The promoted edit moved six MCAM04 Tc6 tracklets from component `8` to
component `40`:

`[3886, 3929, 3967, 4002, 4043, 4079]`

Canonical p005-area scoring improved from
`0.656225 / 0.519723 / 0.535329` to
`0.656453 / 0.519993 / 0.535718`.

Then a subpart combo ablation promoted the IDF1 best again.  Combining
`rank01 + rank02 + rank04` moved 11 tracklets and reached:

`IDF1 / HOTA / AssA = 0.656563 / 0.520067 / 0.535717`

The latest targeted MCAM04 subpart rerun generated candidates from that combo
best and promoted balanced rank01 `35 -> 60`, moving 14 MCAM04 Tc6 tracklets:

`IDF1 / HOTA / AssA = 0.657475 / 0.520599 / 0.535769`

This improved MCAM04 Tc6 IDF1 from `0.563103` to `0.565121`.  The same
experiment produced useful hard negatives: `55 -> 58` variants and broad
`2329 -> 40` hurt the full score, so the next branch should learn a
subpart-specific side-effect referee before expanding more candidates.

Follow-up combo scoring then tested seven compatible extensions on top of the
`35 -> 60` promotion.  `subpart_combo_r01_r02_17seq_assignments` and
`subpart_combo_r01_r02_r07_19seq_assignments` tied at the best aggregate score;
the smaller `r01+r02` edit is promoted:

`IDF1 / HOTA / AssA = 0.657624 / 0.520692 / 0.535785`

It moved 17 MCAM04 Tc6 tracklets and improved MCAM04 Tc6 IDF1 to `0.565454`.
Variants containing `r05` fell to `0.657441`, making them negative side-effect
labels for the next no-anchor subpart referee.

The latest multiview side-effect sweep generated candidates from DINO,
DINO-heavy, SigLIP-fused, weakmetric, and OSNet-010 views, then full-scored 10
diverse no-anchor subpart edits.  `weakmetric 10 -> 22` moved only 2 tracklets
and promoted the current best:

`IDF1 / HOTA / AssA = 0.657653 / 0.520723 / 0.535819`

`weakmetric 10 -> 9` was a smaller positive at `0.657639`; several edits tied
the previous best; `10 -> 15` and broad `11 -> 40` were clear negative
side-effect labels.  This is still a micro-promotion, not the final 0.70 e2e
target.

## Publish Scope

S3 receives the full current progress snapshot, including:

- source scripts under `kit/`;
- reports under `reports/`;
- durable autoresearch state under `autoresearch_state/`;
- selected local run artifacts under `local_runs/`;
- repository metadata needed for continuity, excluding `.git`, caches, and
  local OS/editor junk.

GitHub receives the core reproducible code and state:

- `kit/`;
- `reports/`;
- `autoresearch_state/`;
- lightweight repo docs/config that already belong to source control.

Large `local_runs/` experiment artifacts are intentionally S3-only.


## Latest S3 Pointers

```text
s3://dit-scale-up/zcai/vlincs/LATEST_NO_ANCHOR_PROGRESS.txt
s3://dit-scale-up/zcai/vlincs/progress_snapshots/20260622_no_anchor_global_id_current/
s3://dit-scale-up/zcai/vlincs/current_best_no_anchor/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_currentbest_subpart_multiview_20260622/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_currentbest_subpart_followup_20260622/
s3://dit-scale-up/zcai/vlincs/core_snapshot_20260622/
```

## Latest Multiview Combo Refutation

A follow-up composition run tested 12 low-risk/tie multiview subpart combos on top of the latest `weakmetric 10 -> 22` promotion.

Result: no new best. Ten combinations tied the standing current best exactly:

`IDF1 / HOTA / AssA = 0.657653 / 0.520723 / 0.535819`

The two combinations containing selection rank `3`, corresponding to `19 -> 11`, dropped to:

`IDF1 / HOTA / AssA = 0.657570 / 0.520622 / 0.535698`

This converts `19 -> 11` from a suspicious near-negative into a confirmed hard-negative side-effect label. The next branch should train/calibrate a side-effect referee and search new high-recall false-split repair sources instead of blindly composing tie edits.

## Latest Detection Filter Refutation

A corrected h100-test-3 sweep tested six no-anchor detection/admission filters
around the current best assignment.  The first remote wrapper attempt omitted
`DATA_ROOT`, produced empty cost matrices, and is recorded as a runner bug; the
authoritative rerun used `DATA_ROOT=/mnt/localssd/vlincs_reid_data`.

Result: no new best.  Canonical `p005_area` remains the local best:

`IDF1 / HOTA / AssA = 0.657653 / 0.520723 / 0.535819`

Relaxing the area filter improved recall but lost enough precision to lower
IDF1.  Tightening the filter improved precision but lost recall.  The
MCAM04-only strict `q05_area` variant was harmful, dropping to:

`IDF1 / HOTA / AssA = 0.656550 / 0.519248 / 0.534720`

This closes the immediate detection/filter neighborhood and sends the next
research branch back to side-effect-aware no-anchor identity evidence.

## Latest Side-Effect Blacklisted Subpart Promotion

After closing the local filter neighborhood, the next no-anchor run generated
three subpart-repair pools from the current best assignment and filtered them
with the accumulated side-effect blacklist.  The production-side proposal stage
used only assignment CSVs, tracklet features, temporal-overlap conflicts,
component sizes, and focus-video bookkeeping; GT was used only for canonical
full-score evaluation.

Singles:

- `32 -> 15` moved 7 tracklets and promoted IDF1 to `0.657860`.
- `47 -> 2330` moved 7 tracklets and promoted IDF1 to `0.657680`.
- `24 -> 31` and `31 -> 24` tied the standing best.
- `11 -> 19` and `44 -> 29` were negatives.

The compatible combo `32 -> 15 + 47 -> 2330` moved 14 tracklets and is the new
current best:

`IDF1 / HOTA / AssA = 0.657887 / 0.520944 / 0.535983`

This is still well below the `0.70` e2e goal, but it adds two new positive
side-effect labels, two tie labels, and two negatives for the next no-anchor
referee/search iteration.

Additional S3 pointers:

```text
s3://dit-scale-up/zcai/vlincs/vlincs_reid_runs_h100-test-3_current/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_currentbest_subpart_multiview_combo_20260622/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_filter_sweep_currentbest_20260622_env/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_sideeffect_blacklisted_subpart_search_20260622/
s3://dit-scale-up/zcai/vlincs/research_snapshot_current/
```


## Latest High-Mass Refutation

Starting from the `32 -> 15 + 47 -> 2330` current best, a high-mass no-anchor
candidate sweep selected 143 candidate edits after side-effect blacklist
filtering.  Canonical p005 full scoring of the top six found no new best:
`32 -> 41` tied the standing score exactly, while the remaining broad 34-tracklet
moves were negative.

A targeted `21 -> 60` peel ablation tested sizes 10/16/22.  The best near miss
was size 16:

`IDF1 / HOTA / AssA = 0.657877 / 0.520918 / 0.535962`

This is only `0.000010` below the current best, so it is retained as a hard
near-miss side-effect label rather than promoted.  The current best remains:

`IDF1 / HOTA / AssA = 0.657887 / 0.520944 / 0.535983`

Additional S3 pointer:

```text
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_highmass_from_r47r49_20260622/
```
