# VLINCS Publish Manifest

Date: 2026-06-22

## Current Standing

- Setting: no-anchor global-id research.
- Global-id pair model: F1 / precision / recall =
  `0.775234 / 0.820504 / 0.734698`.
- End-to-end best: IDF1 / HOTA / AssA =
  `0.657624 / 0.520692 / 0.535785`.
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
s3://dit-scale-up/zcai/vlincs/current_best_no_anchor/
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_currentbest_subpart_followup_20260622/
s3://dit-scale-up/zcai/vlincs/core_snapshot_20260622/
```
