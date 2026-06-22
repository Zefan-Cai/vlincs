# VLINCS Publish Manifest

Date: 2026-06-22

## Current Standing

- Setting: no-anchor global-id research.
- Global-id pair model: F1 / precision / recall =
  `0.775234 / 0.820504 / 0.734698`.
- End-to-end best: IDF1 / HOTA / AssA =
  `0.658025 / 0.521057 / 0.536049`.
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

## Latest Subpart Referee Self-Play Promotion

A strict no-anchor subpart side-effect referee was trained from accumulated
full-score labels and used as a self-play scheduler.  The production proposal
stage used assignment CSVs, feature-view agreement, component membership,
temporal/co-camera conflict metadata, and side-effect label memory; GT remained
offline-only for full-score evaluation.

The promoted edit is:

`21 -> 2330`, `rank58`, `size10`, moving 10 tracklets on top of the
`32 -> 15 + 47 -> 2330` base.

It is the current best:

`IDF1 / HOTA / AssA = 0.658025 / 0.521057 / 0.536049`

Old-base follow-up tests did not beat rank58; the closest alternatives
`21 -> 24 rank63` and `21 -> 2330 rank86` reached `0.657976`.

## Latest Rank58 Residual Refutation

Starting from rank58 as the new base, the residual subpart sweep generated 253
no-anchor candidates from three feature-view pools:

- `weakmetric_dino`: 61 selected / 61 candidates;
- `siglip_dino`: 72 selected / 72 candidates;
- `dino_weak`: 120 selected / 3356 candidates.

Sixteen p005-area full-score tests all failed to beat the current best.  The
best residual candidate was `weakmetric_dino_r55_29_to2330`:

`IDF1 / HOTA / AssA = 0.657971 / 0.521239 / 0.536131`

Several small moves collapsed to the same `0.657938` plateau, while broader
residual edits such as `21 -> 19`, `55 -> 58`, and `11 -> 40` were harmful.
This closes the immediate residual-subpart branch and records those candidates
as hard negatives for the next structure-level pivot.

Fresh sync pointers for this state:

```text
s3://dit-scale-up/zcai/vlincs/LATEST_NO_ANCHOR_PROGRESS.txt
s3://dit-scale-up/zcai/vlincs/reports/publish_manifest_20260622.md
s3://dit-scale-up/zcai/vlincs/reports/no_anchor_rank58_residual_refutation_20260622.md
s3://dit-scale-up/zcai/vlincs/research_snapshot_current/
s3://dit-scale-up/zcai/vlincs/research_snapshot_current/local_runs/remote_h100_test_3_20260622/no_anchor_rank58_residual_fullscore_20260622/
```

## Latest Teacher-Consensus Refutation

The rank58 teacher-consensus experiment tested whether several no-anchor
assignment families could safely propose new component merges.  Three teacher
sets were evaluated: structural-policy assignments, subpart-promoted
assignments, and time-agglomeration assignments.

Result: no new best.  All canonical `density_simple + p005_area` delivery
scores tied the current best exactly:

`IDF1 / HOTA / AssA = 0.658025 / 0.521057 / 0.536049`

No merge edge was accepted by any teacher set:

- `subpart_promoted`: 3403 teacher edges, 0 accepted, all rejected by threshold.
- `timeagglom_diverse`: 3321 teacher edges, 0 accepted, 3320 threshold
  rejections and 1 cannot-link conflict rejection.
- `structural_policy`: 3403 teacher edges, 0 accepted, 3402 threshold
  rejections and 1 teacher-condition rejection.

This refutes teacher-consensus merge as the next promotion mechanism in the
current rank58 neighborhood.  The outputs remain useful as hard-negative and
referee evidence for the next error-audit branch.

Fresh sync pointers:

```text
s3://dit-scale-up/zcai/vlincs/reports/no_anchor_teacher_consensus_rank58_refutation_20260622.md
s3://dit-scale-up/zcai/vlincs/remote_runs_h100-test-3_20260622/no_anchor_teacher_consensus_rank58_20260622/
s3://dit-scale-up/zcai/vlincs/research_snapshot_current/local_runs/remote_h100_test_3_20260622/no_anchor_teacher_consensus_rank58_20260622/
```

## Latest Current-Best Error Audit

The current best rank58 assignment was audited with no anchors and GT used only
for evaluation.  Pair-level F1 / precision / recall is:

`0.773980 / 0.823426 / 0.730136`

The direct full export score is:

`IDF1 / HOTA / AssA = 0.655836 / 0.519304 / 0.534154`

This direct export is not the canonical delivery wrapper.  The canonical
`density_simple + p005_area` score remains:

`IDF1 / HOTA / AssA = 0.658025 / 0.521057 / 0.536049`

Dominant false splits are GT IDs `9`, `36`, `11`, `52`, and `43`.  Dominant
false merges are predicted components `96000035`, `96000048`, `96000021`, and
`96000026`.  MCAM04/Tc6 remains the worst high-mass slice.  The next no-anchor
branch should restore compute, run component edge diagnostics, then test local
split/admission guards rather than broad teacher-consensus merges.

Fresh sync pointers:

```text
s3://dit-scale-up/zcai/vlincs/LATEST_NO_ANCHOR_PROGRESS.txt
s3://dit-scale-up/zcai/vlincs/reports_current/no_anchor_current_best_error_audit_20260622.md
s3://dit-scale-up/zcai/vlincs/reports_current/publish_manifest_20260622.md
s3://dit-scale-up/zcai/vlincs/autoresearch_state_current/no_anchor_global_id/state/progress.json
s3://dit-scale-up/zcai/vlincs/local_runs_current/remote_h100_test_3_20260622/no_anchor_current_best_error_audit_20260622/current_best_error_audit_top80_full.json
s3://dit-scale-up/zcai/vlincs/local_runs_current/remote_h100_test_3_20260622/no_anchor_current_best_edge_diagnostics_20260622/run_edge_diagnostics.sh
```

Publish note: the 2026-06-22 light sync uploaded progress, reports, durable
state, core code, and current audit artifacts.  A full raw `local_runs/` sync
encountered S3 multipart `AccessDenied` on older large assignment archives, so
those old archive payloads remain S3-best-effort rather than a blocking
deliverable for this publish.  The current audit JSON, reports, state, and code
needed to continue the no-anchor research loop are included in the verified
publish set.
