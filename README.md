# VLINCS recall-first tracking experiments

This repository contains the code, logs, and lightweight evaluation artifacts needed to reproduce the VLINCS recall-improvement experiments.

Large assets are intentionally not committed:

- Raw VLINCS data
- model weights such as `yolo11x.pt`
- large parquet tracklet dumps
- large diagnostic CSVs

Those files were synchronized separately to:

```text
s3://dit-scale-up/zcai/vlincs/
```

## Contents

```text
tools/
  run_detector_model_sample.py
  run_yolo_track_interval.py
  run_tiled_yolo_detect_interval.py
  run_recall_first_sample_all.py
  *_tracker*.yaml

log/
  2026-05-25_tracker_recall_sweep.md

eval/
  Lightweight CSV/JSON summaries used in the report.
```

## Key results

Worst-region comparison on `vlincs_MS01_MC0001_MCAM04_2024-03-Tc6`, frames `42300-42900`:

```text
yolo11x + ByteTrack ultraloose:
  pred boxes: 19,369
  matched:    11,632 / 20,443 GT boxes
  recall:     0.568997
  precision:  0.600547

recall-first tiled singleton:
  pred boxes: 4,476,996
  matched:    19,387 / 20,443 GT boxes
  recall:     0.948344
  precision:  0.004330

recall-first scale+shift eval:
  pred boxes: 134,309,880
  matched:    20,245 / 20,443 GT boxes
  recall:     0.990315
  precision:  0.000151
```

All-frame per-video metrics for the earlier `yolo26x-botsort-loose-solider + gap<=60 interpolation` run are in:

```text
eval/sweep_existing/yolo26x-botsort-loose-solider_interp_gap60_per_video.csv
```

Aggregate over the 10 sample videos:

```text
GT boxes:        1,685,704
pred boxes:      1,446,602
matched boxes:   1,201,344
overall recall:  0.712666
overall precision: 0.830459
```

## Reproduction notes

Expected local layout:

```text
/mnt/localssd/vlincs/
  VLINCS_Performer/
  yolo11x.pt
  tools/
```

The recall-first interval run can be reproduced with:

```bash
python tools/run_tiled_yolo_detect_interval.py \
  --video /mnt/localssd/vlincs/VLINCS_Performer/sample/videos/vlincs_MS01_MC0001_MCAM04_2024-03-Tc6.mp4 \
  --gt-parquet /mnt/localssd/vlincs/VLINCS_Performer/sample/reference_annotations/vlincs_MS01_MC0001_MCAM04_2024-03-Tc6_v1.7.2.parquet \
  --model /mnt/localssd/vlincs/yolo11x.pt \
  --output-dir /mnt/localssd/vlincs/eval/recall_first \
  --start-frame 42300 \
  --num-frames 600 \
  --imgsz 1536 \
  --conf 0.0001 \
  --model-nms-iou 0.99 \
  --global-nms-iou 1.0 \
  --max-det 3000 \
  --rows 2 \
  --cols 2 \
  --overlap 192 \
  --include-full-frame \
  --batch 5 \
  --device 0 \
  --half \
  --link-mode singleton
```

For the maximum-recall proposal expansion, add:

```bash
--box-scale-variants 0.5,0.75,1.0,1.25,1.5,2.0 \
--box-shift-variants 0:0,0.25:0,-0.25:0,0:0.25,0:-0.25
```

The full experiment narrative is in:

```text
log/2026-05-25_tracker_recall_sweep.md
```
