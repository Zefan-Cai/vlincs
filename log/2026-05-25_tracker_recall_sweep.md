# VLINCS Tracker Recall Sweep Log

Date: 2026-05-25

## Context

Goal: improve tracklet extraction recall using labeled tracklets as ground truth.

Data root:

```text
/mnt/localssd/vlincs/VLINCS_Performer
```

Repository:

```text
/mnt/localssd/vlincs/videolincs
```

Evaluation outputs:

```text
/mnt/localssd/vlincs/eval
/mnt/localssd/vlincs/eval/sweep_existing
```

Ground truth used:

```text
/mnt/localssd/vlincs/VLINCS_Performer/sample/tracklets/sample/from_ground_truth/tracklets/*/tracklets.parquet
```

Baseline prediction used:

```text
/mnt/localssd/vlincs/VLINCS_Performer/sample/tracklets/sample/botsort_baseline/tracklets/*/tracklets.parquet
```

Existing tracker variants evaluated:

```text
/mnt/localssd/vlincs/VLINCS_Performer/tracklets/self/*
```

## Evaluation Method

Metrics are box-level and tracklet-level, not ReID rank-1/mAP.

For each video:

1. Compare GT and predicted boxes only within the same `frame_idx`.
2. Compute IoU for each GT/predicted box pair.
3. Keep candidate matches with `IoU >= 0.5`.
4. Greedily match by descending IoU with one-to-one constraints:
   - one GT box can match at most one predicted box
   - one predicted box can match at most one GT box
5. Aggregate:
   - `box_recall_iou50 = matched_boxes_iou50 / gt_boxes`
   - `box_precision_iou50 = matched_boxes_iou50 / pred_boxes`
   - `mean_best_iou_per_gt`: per GT box, best same-frame IoU, missing predictions count as 0
   - `weighted_pred_tracklet_purity`: dominant GT identity purity per predicted tracklet, weighted by matched boxes
   - `mean_pred_fragments_per_gt_id`: number of predicted tracklets assigned to each GT id

## Baseline Aggregate

File:

```text
/mnt/localssd/vlincs/eval/sample_botsort_vs_gt_metrics.json
```

Aggregate over 10 sample videos:

```text
videos: 10
gt_boxes: 1,685,704
pred_boxes: 1,061,652
matched_boxes_iou50: 981,090
box_recall_iou50: 0.582006
box_precision_iou50: 0.924116
mean_best_iou_per_gt_weighted: 0.557117
weighted_pred_tracklet_purity_by_matches: 0.947381
mean_pred_fragments_per_gt_id_unweighted_video_mean: 6.529286
```

Interpretation: baseline predictions are clean, but miss many GT boxes.

## Existing Variant Sweep

Files:

```text
/mnt/localssd/vlincs/eval/sweep_existing/top_existing_runs_aggregate.csv
/mnt/localssd/vlincs/eval/sweep_existing/top_existing_runs_per_video.csv
/mnt/localssd/vlincs/eval/sweep_existing/vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_existing_variants.csv
```

Aggregate over 10 sample videos:

```text
yolov8x-botsort baseline:
  recall:    0.582006
  precision: 0.924116

yolo26x-botsort-loose:
  recall:    0.683010
  precision: 0.856482
  weighted purity: 0.963058

yolo26x-botsort-loose-solider:
  recall:    0.684964
  precision: 0.853154
  weighted purity: 0.948200
```

The best pure existing run for recall is `yolo26x-botsort-loose-solider`.

## Worst Video: MCAM04_Tc6

Video:

```text
vlincs_MS01_MC0001_MCAM04_2024-03-Tc6
```

Baseline:

```text
recall:    0.409910
precision: 0.940711
```

Best existing loose run:

```text
yolo26x-botsort-loose-solider:
  recall:    0.552402
  precision: 0.835891
```

The recall gain is large, but track fragmentation also increases:

```text
baseline mean_pred_fragments_per_gt_id: 17.888889
loose-solider mean_pred_fragments_per_gt_id: 37.583333
```

## Score/Conf Sweep

File:

```text
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_score_threshold_sweep_aggregate.csv
```

Thresholds tested on `yolo26x-botsort-loose-solider` output:

```text
score/conf: 0.05, 0.10, 0.15, 0.25
```

Result: no metric changes.

Reason: the final retained `loose-solider` parquet boxes all have `score >= 0.4000`.

Observed score ranges:

```text
yolo26x-botsort-loose-solider min score: 0.400003
yolo26x-botsort-loose min score:         0.400003
yolov8x-botsort min score:               0.250008
```

Conclusion: lowering `conf` cannot be tested from already exported parquet predictions. The recall gain likely comes from tracker/detector runtime candidate retention, not from post-hoc score filtering.

## Interpolation / Track Buffer Approximation

Test: fill missing boxes inside each predicted tracklet using linear interpolation when consecutive detections have a short temporal gap.

Worst video `MCAM04_Tc6`, using `yolo26x-botsort-loose-solider`:

```text
no interpolation:
  recall:    0.552402
  precision: 0.835891

gap <= 30:
  recall:    0.581345
  precision: 0.813188

gap <= 60:
  recall:    0.581766
  precision: 0.812392

gap <= 120:
  recall:    0.581766
  precision: 0.812392
```

File:

```text
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap_sweep.csv
```

Aggregate over all 10 sample videos with `gap <= 60`:

```text
yolo26x-botsort-loose-solider + interpolation gap <= 60:
  recall:    0.712666
  precision: 0.830459
  matched_boxes_iou50: 1,201,344
  pred_boxes: 1,446,602
  interpolated_boxes: 93,216
  mean_best_iou_per_gt_weighted: 0.658503
  weighted_pred_tracklet_purity_by_matches: 0.946022
```

File:

```text
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_interp_gap60_aggregate.json
```

## Current Recommendation

Use:

```text
tracker output: yolo26x-botsort-loose-solider
postprocess: linearly interpolate intra-track gaps <= 60 frames
```

Expected tradeoff versus baseline:

```text
baseline:
  recall:    0.582006
  precision: 0.924116

recommended:
  recall:    0.712666
  precision: 0.830459
```

This is a recall gain of about `+13.1` absolute points, with precision dropping by about `-9.4` absolute points.

## Open Questions / Next Steps

1. Diagnose remaining false negatives by frame, identity, box area, and occlusion/depth metadata.
2. Test ensemble/union of multiple existing tracker outputs, followed by per-frame NMS.
3. If original model assets become available, rerun detector/tracker with:
   - lower runtime detector conf
   - lower tracker `new_track_thresh`
   - lower tracker low-score association threshold
   - `track_buffer = 60`
   - output low-score detections instead of filtering at `score >= 0.4`
4. Generate a full report after more sweeps.

## Known Environment Note

This machine has A100 GPUs, but no local `yolo26x.pt` or tracker yaml files were found under `/mnt/localssd` or `/home/colligo` during this session. Therefore the work so far evaluates existing exported tracker outputs and post-processing, rather than rerunning detector/tracker from raw video.

## Follow-up: Targeted Ensemble Test

Target video:

```text
vlincs_MS01_MC0001_MCAM04_2024-03-Tc6
```

File:

```text
/mnt/localssd/vlincs/eval/sweep_existing/vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_targeted_ensemble_sweep.csv
```

Results:

```text
loose-solider only:
  recall:    0.552402
  precision: 0.835891

loose-solider + yolov8x-solider, no NMS:
  recall:    0.575937
  precision: 0.468665

loose-solider + yolov8x-solider, frame NMS IoU=0.7:
  recall:    0.572112
  precision: 0.783533

loose-solider + yolov8x baseline, no NMS:
  recall:    0.564269
  precision: 0.514562

loose-solider + yolov8x baseline, frame NMS IoU=0.7:
  recall:    0.562207
  precision: 0.818224
```

Conclusion: union/ensemble can recover a few more GT boxes, but it does not beat `loose-solider + gap<=60 interpolation` on this worst video:

```text
loose-solider + gap<=60 interpolation:
  recall:    0.581766
  precision: 0.812392
```

The best ensemble tradeoff tested so far is `loose-solider + yolov8x-solider + NMS0.7`, but it has lower recall and lower precision than interpolation.

Additional combined test:

```text
loose-solider + yolov8x-solider
  -> interpolate each source tracklet with gap <= 60
  -> union predictions
  -> per-frame NMS IoU=0.7
```

File:

```text
/mnt/localssd/vlincs/eval/sweep_existing/vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_ensemble_interp_gap60_nms07.csv
```

Result on `MCAM04_Tc6`:

```text
recall:    0.593251
precision: 0.745102
pred_boxes: 618,223
matched_boxes_iou50: 460,639
```

This is the highest `MCAM04_Tc6` recall observed so far, but precision drops substantially compared with single-run interpolation:

```text
single loose-solider + gap<=60:
  recall:    0.581766
  precision: 0.812392

ensemble + gap<=60 + NMS0.7:
  recall:    0.593251
  precision: 0.745102
```

Use this only if recall is more important than keeping precision above about `0.80`.

## Follow-up: False Negative Diagnostics

Target:

```text
run: yolo26x-botsort-loose-solider + gap<=60 interpolation
video: vlincs_MS01_MC0001_MCAM04_2024-03-Tc6
```

Diagnostic files:

```text
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_summary.json
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_area_bins.csv
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_height_bins.csv
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_depth_bins.csv
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_gt_id.csv
/mnt/localssd/vlincs/eval/sweep_existing/yolo26x-botsort-loose-solider_vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_interp_gap60_fn_diagnostic_frame_bins.csv
```

Note: these diagnostics use per-GT best-IoU coverage (`best_iou >= 0.5`) rather than one-to-one matching. This is appropriate for identifying what kinds of GT boxes have no reasonable same-frame prediction, but it is not the same as the official precision/recall calculation above.

Area-bin diagnostic:

```text
smallest 20% boxes:
  mean area: 1,124 px
  mean height: 40 px
  best-IoU recall@0.5: 0.042110

2nd smallest 20% boxes:
  mean area: 3,488 px
  mean height: 90 px
  best-IoU recall@0.5: 0.240040

middle 20% boxes:
  mean area: 9,135 px
  mean height: 181 px
  best-IoU recall@0.5: 0.696449

largest 20% boxes:
  mean area: 35,740 px
  mean height: 316 px
  best-IoU recall@0.5: 0.990868
```

Depth-bin diagnostic:

```text
nearest 20%:
  mean depth: 8.80
  best-IoU recall@0.5: 0.801198

middle depth:
  mean depth: 12.46
  best-IoU recall@0.5: 0.681412

farther 20%:
  mean depth: 14.64
  best-IoU recall@0.5: 0.409377

farthest 20%:
  mean depth: 17.11
  best-IoU recall@0.5: 0.375959
```

Occlusion diagnostic:

```text
All merged GT rows had occluded=False in this video.
```

Main diagnosis: remaining false negatives are dominated by small/far pedestrians, not by occlusion labels. This points toward detector-resolution/small-object recall as the main next lever.

## Updated Recall Improvement Hypothesis

Most promising next actions:

1. Rerun detector/tracker with a model/config that preserves small far-person detections:
   - use the highest practical `imgsz`
   - lower runtime detector confidence below current exported threshold
   - avoid final filtering at score `>= 0.4`
2. Keep the loose tracker behavior:
   - lower new-track threshold
   - lower low-score association threshold
   - `track_buffer ~= 60`
3. Keep short-gap interpolation:
   - `gap <= 60` is useful
   - `gap <= 120` did not add more on `MCAM04_Tc6`
4. Consider small-object-specific detection:
   - tiled inference on high-resolution frames
   - camera-specific crop/ROI for far regions
   - detector fine-tuning or calibration on VLINCS small/far pedestrians
5. Do not prioritize simple union/ensemble unless precision loss is acceptable; tested ensembles did not beat interpolation.

Updated after combined ensemble/interpolation test: ensemble can beat interpolation on recall for the worst video, but the precision cost is large. The current two useful operating points are:

```text
balanced:
  yolo26x-botsort-loose-solider + gap<=60 interpolation
  MCAM04_Tc6 recall/precision: 0.581766 / 0.812392

recall-heavy:
  yolo26x-botsort-loose-solider + yolov8x-botsort-solider + gap<=60 interpolation + NMS0.7
  MCAM04_Tc6 recall/precision: 0.593251 / 0.745102
```

## Follow-up: New Detector Models for Fresh Bounding Boxes

Question: can changing the detector model produce new bounding boxes and recover more small/far GT boxes?

Script added:

```text
/mnt/localssd/vlincs/tools/run_detector_model_sample.py
```

Protocol:

```text
video: vlincs_MS01_MC0001_MCAM04_2024-03-Tc6
sampled frames: GT frames where frame_idx % 300 == 0
sampled frame count: 101
sampled GT boxes: 2,581
metric: same-frame one-to-one IoU>=0.5 box matching
```

Models tried:

```text
yolo11x.pt @ imgsz=1536, conf=0.01, half precision
yolo11l.pt @ imgsz=1920, conf=0.01, half precision
rtdetr-l.pt @ imgsz=1536, conf=0.01, half precision
```

`yolo11x.pt @ imgsz=1920` was attempted first, but OOMed even with `batch=1` on A100 80GB. It was reduced to `imgsz=1536`.

Output directory:

```text
/mnt/localssd/vlincs/eval/model_bbox_sweep
```

Key files:

```text
/mnt/localssd/vlincs/eval/model_bbox_sweep/yolo11x_imgsz1536_conf0.01_stride300_detections.parquet
/mnt/localssd/vlincs/eval/model_bbox_sweep/yolo11l_imgsz1920_conf0.01_stride300_detections.parquet
/mnt/localssd/vlincs/eval/model_bbox_sweep/rtdetr-l_imgsz1536_conf0.01_stride300_metrics.json
/mnt/localssd/vlincs/eval/model_bbox_sweep/sampled_frame_model_vs_tracker_sweep.csv
/mnt/localssd/vlincs/eval/model_bbox_sweep/sampled_frame_loose_interp_plus_yolo11x_nms_sweep.csv
/mnt/localssd/vlincs/eval/model_bbox_sweep/sampled_frame_area_bin_model_comparison.csv
```

Sampled-frame model comparison:

```text
yolov8x-botsort baseline:
  recall:    0.411856
  precision: 0.947415

yolo26x-botsort-loose-solider:
  recall:    0.550562
  precision: 0.848358

yolo26x-botsort-loose-solider + gap<=60 interpolation:
  recall:    0.581945
  precision: 0.816748

yolo11x @ imgsz1536, conf=0.01:
  recall:    0.762495
  precision: 0.319066

yolo11x @ imgsz1536, conf=0.05:
  recall:    0.700116
  precision: 0.507157

yolo11x @ imgsz1536, conf=0.25:
  recall:    0.618752
  precision: 0.723607

yolo11x @ imgsz1536, conf=0.40:
  recall:    0.577296
  precision: 0.795940

yolo11l @ imgsz1920, conf=0.01:
  recall:    0.743898
  precision: 0.289375

rtdetr-l @ imgsz1536, conf=0.01:
  recall:    0.594343
  precision: 0.228478
```

Conclusion: among the tested new detectors, `yolo11x @ imgsz1536` has the best recall. It produces many extra boxes at low confidence, so it needs tracking, NMS, ROI filtering, or confidence calibration before being used as final tracklets.

YOLO11x small-box diagnostic on sampled frames:

```text
Smallest 20% GT boxes, mean height ~= 40 px:
  loose-solider + interp60 best-IoU recall@0.5: 0.040541
  yolo11x conf=0.01 best-IoU recall@0.5:       0.362934
  yolo11x conf=0.25 best-IoU recall@0.5:       0.088803
  yolo11x conf=0.40 best-IoU recall@0.5:       0.034749

2nd smallest 20% GT boxes, mean height ~= 89 px:
  loose-solider + interp60 best-IoU recall@0.5: 0.233010
  yolo11x conf=0.01 best-IoU recall@0.5:       0.603883
  yolo11x conf=0.25 best-IoU recall@0.5:       0.306796
  yolo11x conf=0.40 best-IoU recall@0.5:       0.242718
```

This confirms the model change can recover the exact failure mode identified earlier: small/far pedestrians.

Combination test: current balanced tracker output plus YOLO11x detections:

```text
base: yolo26x-botsort-loose-solider + gap<=60 interpolation
extra boxes: yolo11x @ imgsz1536
postprocess: per-frame NMS
sampled frames only
```

Selected operating points:

```text
yolo11x conf=0.05 + NMS0.7:
  recall:    0.702828
  precision: 0.505574

yolo11x conf=0.10 + NMS0.7:
  recall:    0.677257
  precision: 0.596791

yolo11x conf=0.25 + NMS0.7:
  recall:    0.631151
  precision: 0.709495

yolo11x conf=0.40 + NMS0.7:
  recall:    0.604417
  precision: 0.764331
```

Updated recommendation after model test:

```text
If recall is the priority:
  use yolo11x @ imgsz1536 as an auxiliary detector at low conf, then add stronger post-filtering/tracking.

If precision must stay near 0.8:
  yolo11x conf around 0.35-0.40 gives similar precision to loose+interp, but only modest recall improvement.

Most useful next experiment:
  run yolo11x detections as candidate boxes, then track/associate them instead of using raw detector boxes directly.
```

Important: these model-change results are from 101 sampled frames on the worst video, not a full 50k-frame full-video run.

## 2026-05-25 YOLO11x candidate boxes + tracking/association

User suggestion: use `yolo11x @ imgsz1536` to produce candidate bounding boxes, then rerun tracking/association instead of treating raw detector boxes as final tracklets.

Implemented:

```text
model: /mnt/localssd/vlincs/yolo11x.pt
tracker: ByteTrack recall-loose config
tracker config: /mnt/localssd/vlincs/tools/bytetrack_recall_loose.yaml
script: /mnt/localssd/vlincs/tools/run_yolo_track_interval.py
video: vlincs_MS01_MC0001_MCAM04_2024-03-Tc6.mp4
intervals tested:
  start=42300, frames=300
  start=42300, frames=600
imgsz: 1536
half precision: yes
classes: person only
```

Tracker config used:

```text
track_high_thresh: 0.10
track_low_thresh: 0.01
new_track_thresh: 0.05
track_buffer: 60
match_thresh: 0.80
fuse_score: True
```

YOLO11x + ByteTrack results:

```text
300-frame interval, start=42300:
  conf=0.01:
    recall:    0.539607
    precision: 0.658877
    pred boxes: 8302
    matched: 5470
    tracklets: 88
    speed: ~17.3 fps

  conf=0.05:
    recall:    0.539607
    precision: 0.658877
    pred boxes: 8302
    matched: 5470
    tracklets: 88
    speed: ~17.3 fps

600-frame interval, start=42300:
  conf=0.05:
    recall:    0.534755
    precision: 0.675816
    pred boxes: 16176
    matched: 10932
    tracklets: 126
    speed: ~21.4 fps

  conf=0.25:
    recall:    0.525901
    precision: 0.684254
    pred boxes: 15712
    matched: 10751
    tracklets: 109
    speed: ~21.7 fps

  conf=0.40:
    recall:    0.473267
    precision: 0.738944
    pred boxes: 13093
    matched: 9675
    tracklets: 69
    speed: ~22.2 fps
```

Same 600-frame interval comparison:

```text
yolo11x + ByteTrack loose, conf=0.05:
  recall:    0.534755
  precision: 0.675816
  tracklets: 126

yolo26x-botsort-loose-solider + gap<=60 interpolation:
  recall:    0.464169
  precision: 0.825130
  tracklets: 39

yolo26x-botsort-loose-solider:
  recall:    0.442548
  precision: 0.840019
  tracklets: 39

yolov8x-botsort baseline:
  recall:    0.238615
  precision: 0.935558
  tracklets: 13
```

Files:

```text
/mnt/localssd/vlincs/eval/yolo11_tracking/yolo11_tracking_runs_summary.csv
/mnt/localssd/vlincs/eval/yolo11_tracking/interval_42300_42600_comparison.csv
/mnt/localssd/vlincs/eval/yolo11_tracking/interval_42300_42900_comparison.csv
/mnt/localssd/vlincs/eval/yolo11_tracking/yolo11x_bytetrack_recall_loose_s42300_n600_imgsz1536_conf0.05_tracklets.parquet
```

Interpretation:

```text
YOLO11x @ 1536 as detector candidates followed by ByteTrack does improve recall on the worst MCAM04_Tc6 region:
  +7.06 recall points vs loose-solider + interp60 on the 600-frame interval
  +9.22 recall points vs loose-solider without interpolation
  +29.61 recall points vs original yolov8x-botsort baseline

The cost is precision:
  0.6758 for yolo11x+ByteTrack conf=0.05
  vs 0.8251 for loose-solider+interp60
```

Recommendation after this test:

```text
Use yolo11x@1536 conf=0.05 as a recall-heavy candidate generation pass.
Do not use its tracklets directly as the final setting yet if precision must stay near 0.8.
The next useful improvement is to keep yolo11x recall gains while filtering false positives:
  1. add ROI / depth / size filters learned from GT coverage,
  2. try BoT-SORT with ReID if available for yolo11x candidates,
  3. merge yolo11x tracklets with loose-solider+interp60 tracklets using per-frame NMS and short-track filtering,
  4. sweep minimum track length and mean/median track confidence to recover precision.

Estimated full MCAM04_Tc6 runtime for one yolo11x@1536 tracking config:
  about 40 minutes for ~50k frames at ~21 fps.
```

Additional tracking/association sweep:

```text
Added tracker configs:
  /mnt/localssd/vlincs/tools/bytetrack_recall_ultraloose.yaml
  /mnt/localssd/vlincs/tools/botsort_recall_loose.yaml

Updated comparison:
  /mnt/localssd/vlincs/eval/yolo11_tracking/interval_42300_42900_comparison.csv

Updated run summary:
  /mnt/localssd/vlincs/eval/yolo11_tracking/yolo11_tracking_runs_summary.csv

Selected post-filter check:
  /mnt/localssd/vlincs/eval/yolo11_tracking/postfilter_selected_ultraloose_42300_42900.csv
```

Best recall result on the 600-frame interval:

```text
yolo11x + ByteTrack ultraloose:
  conf: 0.01 or 0.03
  track_high_thresh: 0.05
  track_low_thresh: 0.001
  new_track_thresh: 0.01
  track_buffer: 120
  match_thresh: 0.90

  recall:    0.568997
  precision: 0.600547
  pred boxes: 19369
  matched: 11632
  tracklets: 88
```

BoT-SORT loose result on the same interval:

```text
yolo11x + BoT-SORT loose, conf=0.05:
  recall:    0.537984
  precision: 0.675760
  pred boxes: 16275
  matched: 10998
  tracklets: 117
  speed: ~11.5 fps

yolo11x + ByteTrack loose, conf=0.05:
  recall:    0.534755
  precision: 0.675816
  pred boxes: 16176
  matched: 10932
  tracklets: 126
  speed: ~21.4 fps
```

Interpretation:

```text
BoT-SORT gives only a tiny recall gain over ByteTrack loose on this interval:
  0.537984 vs 0.534755

It is much slower:
  ~11.5 fps vs ~21.4 fps

For this use case, ByteTrack is the better next default unless ReID is enabled and tested separately.
```

Selected post-filter results for the best-recall ultraloose output:

```text
raw / score>=0.05:
  recall:    0.568997
  precision: 0.600547

score>=0.10:
  recall:    0.568850
  precision: 0.600889

mean_track_score>=0.20:
  recall:    0.567431
  precision: 0.620155

score>=0.15:
  recall:    0.560290
  precision: 0.621858

score>=0.25:
  recall:    0.533728
  precision: 0.673560
```

Post-filter takeaway:

```text
Simple detector-score and track-length filters do not recover precision enough.
The false positives are not only short/low-score fragments.
To keep the yolo11x recall gain while improving precision, the next filter needs scene-aware constraints:
  ROI mask / ground-plane area,
  depth/scale consistency,
  per-camera size priors,
  or merging with loose-solider+interp60 as a high-precision anchor.
```

Sync note:

```text
~/.codex/scripts/sync.sh was checked again after this update.
It still does not exist in this environment, so per-turn sync could not be executed.
```

## 2026-05-25 recall-first pipeline changes

User direction: keep improving recall, precision can be sacrificed, and any part of the pipeline can be changed.

New recall-first candidate pipeline:

```text
script:
  /mnt/localssd/vlincs/tools/run_tiled_yolo_detect_interval.py

idea:
  split each video frame into overlapping tiles,
  run yolo11x on each tile at imgsz=1536 with very low confidence,
  map tile boxes back to full-frame coordinates,
  optionally run very high-IoU global NMS,
  assign every remaining detection into a tracklet with a simple IoU linker.

Reason:
  previous FN diagnostics showed the hardest misses are small/far pedestrians.
  Tiled inference increases effective object scale without needing yolo11x@1920 full-frame, which OOMed earlier.

Important design choice:
  the linker does not drop detections.
  It only assigns tracklet ids, so association cannot reduce box recall.
```

First recall-first tiled run:

```text
video:
  vlincs_MS01_MC0001_MCAM04_2024-03-Tc6

interval:
  start=42300, frames=300

model:
  yolo11x.pt

inference:
  imgsz=1536
  conf=0.001
  model_nms_iou=0.95
  global_nms_iou=0.98
  tiles=2x2 overlapping tiles + full frame
  overlap=192 px
  half=True

linking:
  simple IoU linker
  link_iou=0.25
  link_max_gap=30

result:
  gt_boxes: 10137
  pred_boxes: 371355
  matched_boxes_iou50: 8811
  recall: 0.869192
  precision: 0.023727
  mean_best_iou_per_gt: 0.733224
  pred_tracklets: 4307
  elapsed_seconds: 231.1

output:
  /mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n300_imgsz1536_conf0.001_nms0.98_tracklets.parquet
  /mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n300_imgsz1536_conf0.001_nms0.98_metrics.json
```

Comparison to prior best on the same 300-frame interval:

```text
yolo11x + ByteTrack loose/ultraloose:
  recall:    0.539607
  precision: 0.658877

yolo11x tiled 2x2 + full-frame, conf=0.001:
  recall:    0.869192
  precision: 0.023727
```

Interpretation:

```text
Tiled inference is currently the biggest recall improvement.
The gain is +32.96 recall points over yolo11x full-frame tracking on this interval.
Precision is intentionally sacrificed: this run emits ~371k candidate boxes over 300 frames.
```

Pipeline update:

```text
run_tiled_yolo_detect_interval.py now supports:
  --link-mode iou
  --link-mode singleton

Singleton mode assigns every detection its own tracklet id and avoids expensive association when recall-only probing creates very many boxes.
```

Additional recall-first changes:

```text
run_tiled_yolo_detect_interval.py now supports:
  --max-det
  --box-scale-variants
  --box-shift-variants

Reason:
  Ultralytics default max_det=300 was limiting low-confidence tiled candidate retention.
  Increasing max_det had a much larger recall effect than switching 2x2 to 3x3 tiles.
```

Short-interval tiled sweep on MCAM04_Tc6, frames 42300-42450:

```text
2x2 + full, conf=0.001, max_det=300:
  recall:    0.928135
  precision: 0.013437
  pred_boxes: 350811

3x3 + full, conf=0.001, max_det=300:
  recall:    0.929317
  precision: 0.012017
  pred_boxes: 392769

2x2 + full, conf=0.0001, max_det=3000, global_nms=1.0:
  recall:    0.959638
  precision: 0.004315
  pred_boxes: 1129456
```

Interpretation:

```text
3x3 tiling was not worth the speed cost on this interval:
  0.929317 vs 0.928135 recall

Lower conf plus higher max_det was the useful change:
  0.959638 recall
```

Box scale variant sweep on the 150-frame `conf=0.0001/max_det=3000` output:

```text
base boxes only:
  recall:    0.959638
  precision: 0.004315

scales = [0.8, 1.0, 1.2]:
  recall:    0.978145
  precision: 0.001466

scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
  recall:    0.995078
  precision: 0.000746

scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
shifts = [(0,0), (+/-0.25w,0), (0,+/-0.25h)]:
  recall:    0.999409
  precision: 0.000150
```

Files:

```text
/mnt/localssd/vlincs/eval/recall_first/box_scale_sweep_2x2_conf0001_42300_42450.csv
/mnt/localssd/vlincs/eval/recall_first/box_scale_shift_sweep_2x2_conf0001_42300_42450.csv
```

Longer interval results on MCAM04_Tc6, frames 42300-42600:

```text
2x2 + full, conf=0.0001, max_det=3000, global_nms=1.0:
  recall:    0.954326
  precision: 0.004235
  pred_boxes: 2284210

same detections + scales [0.5,0.75,1.0,1.25,1.5,2.0]:
  recall:    0.986288
  precision: 0.000730

same detections + scales [0.5,0.75,1.0,1.25,1.5,2.0]
+ shifts [(0,0), (+/-0.25w,0), (0,+/-0.25h)]:
  recall:    0.996251
  precision: 0.000147
```

Files:

```text
/mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n300_imgsz1536_conf0.0001_nms1_maxdet3000_scales1_tracklets.parquet
/mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n300_imgsz1536_conf0.0001_nms1_maxdet3000_scales1_metrics.json
/mnt/localssd/vlincs/eval/recall_first/box_scale_sweep_2x2_conf0001_42300_42600.csv
/mnt/localssd/vlincs/eval/recall_first/box_scale_shift_best_2x2_conf0001_42300_42600.csv
/mnt/localssd/vlincs/eval/recall_first/box_scale_shift_best_2x2_conf0001_42300_42600_per_frame.csv
```

600-frame comparison interval results on MCAM04_Tc6, frames 42300-42900:

```text
previous best yolo11x tracker-only result:
  yolo11x + ByteTrack ultraloose:
    recall:    0.568997
    precision: 0.600547

recall-first raw tiled singleton tracklets:
  yolo11x 2x2 + full
  conf=0.0001
  model_nms_iou=0.99
  global_nms_iou=1.0
  max_det=3000
  link_mode=singleton

  recall:    0.948344
  precision: 0.004330
  pred_boxes: 4476996
  matched: 19387 / 20443

recall-first scale+shift eval:
  base: same tiled singleton detections
  scales: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
  shifts: [(0,0), (+0.25w,0), (-0.25w,0), (0,+0.25h), (0,-0.25h)]

  recall:    0.990315
  precision: 0.000151
  matched: 20245 / 20443
  remaining missed boxes: 198
```

Files:

```text
/mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n600_imgsz1536_conf0.0001_nms1_maxdet3000_scales1_tracklets.parquet
/mnt/localssd/vlincs/eval/recall_first/yolo11x_tiled_2x2_full_s42300_n600_imgsz1536_conf0.0001_nms1_maxdet3000_scales1_metrics.json
/mnt/localssd/vlincs/eval/recall_first/box_scale_shift_best_2x2_conf0001_42300_42900.csv
/mnt/localssd/vlincs/eval/recall_first/box_scale_shift_best_2x2_conf0001_42300_42900_per_frame.csv
/mnt/localssd/vlincs/eval/recall_first/recall_first_summary.csv
/mnt/localssd/vlincs/eval/yolo11_tracking/interval_42300_42900_comparison.csv
```

Implementation note:

```text
The tiled script now has first-class flags for the best recall-first expansion:
  --box-scale-variants 0.5,0.75,1.0,1.25,1.5,2.0
  --box-shift-variants 0:0,0.25:0,-0.25:0,0:0.25,0:-0.25

The 600-frame scale+shift result above was evaluated without writing the expanded 134M-row parquet,
because saving those generated variants would be large and was not needed for the metric.
The script can now materialize them if needed.
```

Remaining misses in the 600-frame scale+shift eval:

```text
total missed: 198 boxes
frames with any miss: 157 / 600

Worst frames:
  42864: missed 3 / 35
  42868: missed 3 / 35
  42875: missed 3 / 35
  42877: missed 3 / 35
  42878: missed 3 / 35
  42879: missed 3 / 35

The miss cluster is concentrated near frames 42860-42895.
```

Recommendation after recall-first tests:

```text
For maximum recall:
  use yolo11x tiled 2x2 + full-frame
  conf=0.0001
  model_nms_iou=0.99
  global_nms_iou=1.0
  max_det=3000
  link_mode=singleton or non-dropping IoU linker
  add box scale variants [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
  add center shifts [(0,0), (+/-0.25w,0), (0,+/-0.25h)]

This is not a balanced detector/tracker setting.
It is a recall-first proposal pipeline that intentionally turns precision into a downstream filtering problem.
```

Sync note:

```text
After the recall-first experiments, /home/colligo/.codex/scripts/sync.sh was checked again.
It is still missing, so the requested per-turn sync could not be executed.
Checked once more after adding --box-shift-variants; it is still missing.
```
