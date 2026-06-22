# No-Anchor GT-Box vs BoTSORT Tracklet Evidence Diagnostic

Date: 2026-06-21

## Why This Test

The Deli AutoResearch lesson for this loop is: after repeated stale iterations,
change the structural question instead of tuning the same policy.  The current
no-anchor DS1 loop has already shown:

- pair/global-ID model target is met:
  F1/P/R = `0.775234 / 0.820504 / 0.734698`;
- end-to-end DS1 delivery is still below target:
  IDF1/HOTA/AssA = `0.655911 / 0.519311 / 0.534922`;
- extra weak-video bbox area pruning after p0.5% is refuted.

So this diagnostic asks a sharper question:

> If we replace real BoTSORT tracklets with GT-box tracklets, does weak
> crop+bbox evidence become enough for no-anchor identity resolution?

If yes, the bottleneck is mostly detector/tracklet quality.  If no, the
bottleneck is missing identity evidence.

## Data Prepared

New tool:

- `kit/prepare_botsort_no_anchor_sample.py`

It reads:

- BoTSORT sample tracklets:
  `/mnt/localssd/vlincs_reid_data/Box/VLINCS_Performer/sample/tracklets/sample/botsort_baseline/tracklets`
- BoTSORT crop cache:
  `/mnt/localssd/vlincs_reid_runs/botsort_crop3_cache_v1.npz`

It writes eval-only labels by frame-level IoU against DS1 GT.  GT identity is
used only for evaluation columns:

- `tracklet_majority_gt_id`
- `tracklet_majority_gt_fraction`

Feature blocks contain only crop, bbox, and trajectory summaries.

BoTSORT prepare summary:

| field | value |
|---|---:|
| rows | `1,061,652` |
| tracklets | `2,406` |
| eval-labeled tracklets at IoU 0.50 | `2,049` |
| rejected tracklets | `357` |
| matched rows | `982,618` |
| eval purity mean / p10 | `0.871573 / 0.555003` |
| crop feature tracklets | `2,406` |
| missing crop features | `513` |

Important caveat: `no_anchor_sample_parquet_sweep.py` evaluates sample identity
and pair metrics, not canonical DS1 HOTA/IDF1.  This is a tracklet-evidence
diagnostic, not a delivery submission.

## Comparable Weak-Feature Runs

Both rows use the same weak feature family:

- `features_crop`
- `features_bbox` with weight `0.25`
- no anchors
- GT labels only for evaluation
- pair-model solver family `consensus_guard / consensus_attach`

| tracklet source | best solver | identity F1 | pair F1 | pair P | pair R | component purity mean / p10 / min | eval tracklets | all tracklets |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GT-box sample, old run | `consensus_guard@0.35` | `0.217860` | `0.032319` | `0.040175` | `0.027033` | `0.874406 / 0.277409 / 0.104397` | `1,887` | `1,887` |
| BoTSORT sample, new run | `consensus_guard@0.65` | `0.191162` | `0.032079` | `0.034875` | `0.029697` | `0.720666 / 0.179718 / 0.088122` | `1,893` | `2,406` |

## Strong Evidence Follow-up

After the weak-feature refutation, the structural pivot was to change the
identity evidence source while keeping the no-anchor setting fixed.  I patched
`kit/extract_sample_tracklet_osnet_features.py` so BoTSORT sample videos with
hash-suffixed keys, such as
`vlincs_MS01_MC0001_MCAM06_2024-03-Tc8__439cd5c7a4`, resolve to the original
raw-video key for frame extraction.  This patch affects video lookup only; GT
identity is still eval-only.

Feature extraction:

| field | value |
|---|---:|
| feature file | `features_botsort_osnet_color_s3_20260621.npz` |
| tracklets with OSNet/color features | `2,406 / 2,406` |
| crops seen | `7,214` |
| missing video / missing crop | `0 / 0` |
| OSNet dim / color dim | `512 / 82` |
| model | `osnet_x1_0_msmt17+color_hist` |

Comparable BoTSORT sample results:

| evidence | best solver | identity F1 | pair F1 | pair P | pair R | component purity mean / p10 / min | components | largest component |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| weak crop+bbox | `consensus_guard@0.65` | `0.191162` | `0.032079` | `0.034875` | `0.029697` | `0.720666 / 0.179718 / 0.088122` | `215` | `120` |
| OSNet + color | `consensus_guard@0.35` | `0.380024` | `0.253572` | `0.315609` | `0.211916` | `0.743022 / 0.356520 / 0.241489` | `172` | `114` |

This is the first useful positive signal from the sample-tracklet branch:

- identity F1 improves from `0.191162` to `0.380024`;
- pair F1 improves from `0.032079` to `0.253572`;
- pair precision improves from `0.034875` to `0.315609`;
- pair recall improves from `0.029697` to `0.211916`.

The result is not near the target yet, but it cleanly says the bottleneck is
identity evidence, not just graph policy.  Under the Deli AutoResearch protocol,
this is a `positive-but-incomplete` direction: keep it, then ask whether GT-box
OSNet/color and better metric learning can raise the upper-bound before putting
the evidence back into full DS1 delivery.

## Interpretation

GT-box tracklets do not rescue the weak crop+bbox no-anchor model.  Pair F1 is
essentially identical:

- GT-box weak-feature pair F1: `0.032319`
- BoTSORT weak-feature pair F1: `0.032079`

That is the important result.  Detection/tracklet quality matters for e2e, but
the current weak crop+bbox evidence is not sufficient even when the boxes are
upper-bound GT boxes.  This rules out a pure "better boxes + same weak identity
resolver" route to 0.70.

The BoTSORT run also exposes real-pipeline tracklet impurity:

- 357 tracklets have no accepted eval majority label;
- p10 eval purity is only `0.555`;
- 513 tracklets lack crop-cache features in the current cache.

Those are production problems, but they are not the only problem: the GT-box
upper-bound still has pair F1 around `0.032`.

## Next Structural Pivot

Close the weak-feature branch as a refutation of weak-feature upper-bound
rescue.  Keep the OSNet/color branch as the next evidence source.

Next direction:

1. run the same OSNet/color evidence on GT-box sample tracklets to estimate the
   best-case feature ceiling without BoTSORT fragmentation;
2. add NFC or supervised metric transforms trained only from no-anchor weak
   positives/negatives;
3. start with the weak videos already identified as e2e bottlenecks:
   `MCAM04 Tc6`, `MCAM06 Tc6`, `MCAM03 Tc8`;
4. compare GT-box vs BoTSORT under the stronger evidence before spending more
   cycles on graph policy or delivery postfilters;
5. only after sample pair F1 moves materially above the current OSNet/color
   `0.254` floor should the result be inserted back into the DS1 no-anchor
   pipeline.

## Artifacts

Local:

- `kit/prepare_botsort_no_anchor_sample.py`
- `kit/extract_sample_tracklet_osnet_features.py`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/prepare_summary.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_focused_crop_bbox_v1.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_focused_crop_bbox_v1.csv`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_focused_crop_bbox_v1_slices.csv`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s3_20260621.npz`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s3_v1.json`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s3_v1.csv`
- `local_runs/remote_h100_test_3_20260621/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s3_v1_slices.csv`

Remote:

- `/mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/botsort_eval.parquet`
- `/mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/features_botsort_crop_bbox.npz`
- `/mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_focused_crop_bbox_v1.json`
- `/mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/features_botsort_osnet_color_s3_20260621.npz`
- `/mnt/localssd/vlincs_reid_runs/botsort_no_anchor_sample_20260621/no_anchor_botsort_sample_osnet_color_s3_v1.json`
- `/mnt/localssd/vlincs_reid_runs/gtbox_no_anchor_sample_20260618/no_anchor_gtbox_sample_focused_crop_bbox_v1.csv`
