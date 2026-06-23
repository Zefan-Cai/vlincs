# No-Anchor Current-Best Regenerated Subpart Promotion

Date: 2026-06-22

## Result

New no-anchor current best:

`IDF1 / HOTA / AssA = 0.664835 / 0.525495 / 0.536810`

Previous best:

`IDF1 / HOTA / AssA = 0.664050 / 0.524893 / 0.536568`

Delta:

`+0.000785 / +0.000602 / +0.000242`

The end-to-end goal remains open because IDF1 is still below `0.70`.

## Candidate

The promoted candidate is a no-anchor replay of two regenerated current-best
subpart moves:

- rank02: move 14 tracklets from `960000350` to `960000351`;
- rank09: move 5 tracklets from `960000350` to `960000480`.

Assignment:

`local_runs/offline_no_anchor_split_probe_20260622/currentbest_regenerated_subpart_balanced_assignments/combo_rank02_rank09_s86_to87_s86_to88_19seq.csv`

Canonical output:

- direct full:
  `local_runs/offline_no_anchor_split_probe_20260622/currentbest_regenerated_subpart_balanced_fullscore/combo_rank02_rank09_19seq_full_export.json`
- density simple:
  `local_runs/offline_no_anchor_split_probe_20260622/currentbest_regenerated_subpart_balanced_fullscore/combo_rank02_rank09_19seq_density_simple.json`
- p005 area:
  `local_runs/offline_no_anchor_split_probe_20260622/currentbest_regenerated_subpart_balanced_fullscore/combo_rank02_rank09_19seq_density_p005_area.json`
- p005 zip:
  `local_runs/offline_no_anchor_split_probe_20260622/currentbest_regenerated_subpart_balanced_fullscore/combo_rank02_rank09_19seq_density_p005_area.zip`

## Protocol

The proposer used only production-side evidence:

`assignment CSV + weakmetric/OSNet fused features + DINOv2 features`

GT was used only by the pair diagnostic cache and canonical DS1 scorer after
candidate materialization.

Before regenerating candidates, the current best assignment had to be refreshed:
the promoted component split had updated `predicted_global_id`, but its
`component_label` still grouped split children together.  A no-op projection
created a refreshed assignment with 92 component labels matching 92 predicted
IDs.  The replay utility added for this is:

`kit/project_no_anchor_manifest_moves.py`

## Ablation

| candidate | moved | pair F1 | pair P | pair R | direct IDF1 | p005 IDF1 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| previous best | 0 | 0.781330 | 0.849326 | 0.723413 | 0.661843 | 0.664050 | old best |
| rank02 `350->351` | 14 | 0.782323 | 0.850144 | 0.724524 | 0.662481 | 0.664678 | positive |
| rank09 `350->480` | 5 | 0.781253 | 0.849305 | 0.723297 | 0.661996 | not run | direct positive, pair proxy negative |
| combo rank02+rank09 | 19 | 0.782244 | 0.850111 | 0.724411 | 0.662638 | 0.664835 | promoted |
| rank03 `9->201` | 6 | not recorded | not recorded | not recorded | 0.661843 | not run | direct tie |
| rank11 `21->201` | 14 | not recorded | not recorded | not recorded | 0.661116 | not run | negative |
| old side-effect `55->58` | 14 | not recorded | not recorded | not recorded | 0.661483 | not run | negative |
| old side-effect `47->2329` | 2 | not recorded | not recorded | not recorded | 0.661784 | not run | negative |
| old side-effect `9->2330` | 2 | not recorded | not recorded | not recorded | 0.661843 | not run | direct tie |

The important admission lesson is that pair proxy is not monotonic with full
score: rank09 lowers pair F1 slightly but improves direct IDF1.  The next gate
should therefore use full-score side-effect labels and per-video deltas, not
only pair precision/recall.

## Per-Video Effects

The combo mainly improves MCAM08 Tc6 and slightly improves MCAM04 Tc6:

- MCAM08 Tc6 direct IDF1: `0.768938 -> 0.771731`
- MCAM04 Tc6 direct IDF1: `0.577029 -> 0.577461`

The final p005 delivery score improved with the same fixed no-GT delivery
filter:

`0.664050 -> 0.664835`

## Next Direction

1. Treat `55->58`, `21->201`, and the old projected near-miss moves as hard
   negatives for the regenerated subpart proposer.
2. Train or hand-build a full-score-sensitive admission gate over regenerated
   current-best candidates, with features for split-child provenance,
   target-child namespace, focus-video mass, moved detection mass, target
   similarity, target margin, and per-video side-effect risk.
3. Regenerate another current-best candidate queue after promoting
   rank02+rank09, then score only candidates that pass this gate.
