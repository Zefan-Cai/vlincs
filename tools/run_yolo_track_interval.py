#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1 = a[:, 0, None]
    ay1 = a[:, 1, None]
    ax2 = a[:, 2, None]
    ay2 = a[:, 3, None]
    bx1 = b[:, 0]
    by1 = b[:, 1]
    bx2 = b[:, 2]
    by2 = b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    return inter / np.maximum(1e-9, area_a + area_b - inter)


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, iou_threshold: float) -> dict[str, float | int]:
    pred_by_frame = {int(k): v for k, v in pred.groupby("frame_idx", sort=False)}
    matched = 0
    best_sum = 0.0
    gt_total = 0
    for frame_idx, gt_frame in gt.groupby("frame_idx", sort=False):
        pred_frame = pred_by_frame.get(int(frame_idx), pred.iloc[0:0])
        gt_total += len(gt_frame)
        if pred_frame.empty:
            continue
        mat = iou_matrix(
            gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
            pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
        )
        best_sum += float(mat.max(axis=1).sum())
        coords = np.argwhere(mat >= iou_threshold)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for i, j in sorted(
            ((int(i), int(j)) for i, j in coords),
            key=lambda ij: mat[ij[0], ij[1]],
            reverse=True,
        ):
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            matched += 1
    pred_total = int(len(pred))
    return {
        "gt_boxes": int(gt_total),
        "pred_boxes": pred_total,
        "matched_boxes_iou50": int(matched),
        "box_recall_iou50": float(matched / gt_total) if gt_total else 0.0,
        "box_precision_iou50": float(matched / pred_total) if pred_total else 0.0,
        "mean_best_iou_per_gt": float(best_sum / gt_total) if gt_total else 0.0,
        "pred_tracklets": int(pred["tracklet_key"].nunique()) if "tracklet_key" in pred and not pred.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO tracking over one continuous video interval and evaluate boxes.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    end_frame = args.start_frame + args.num_frames

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    rows: list[dict[str, float | int | str]] = []
    start_time = time.time()
    frame_idx = args.start_frame
    processed = 0
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(
            source=frame,
            persist=True,
            stream=False,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            classes=[0],
            tracker=args.tracker,
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes), dtype=int)
            for det_idx, (box, score, cls, track_id) in enumerate(zip(xyxy, scores, classes, ids)):
                x1, y1, x2, y2 = [float(v) for v in box]
                rows.append(
                    {
                        "video_key": video_path.stem,
                        "frame_idx": int(frame_idx),
                        "local_track_id": int(track_id),
                        "tracklet_key": f"{video_path.stem}:{int(track_id)}",
                        "det_id": int(det_idx),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(score),
                        "coco_cls": int(cls),
                    }
                )
        processed += 1
        if processed % 100 == 0:
            elapsed = max(1e-6, time.time() - start_time)
            print(f"[track] processed={processed}/{args.num_frames} fps={processed / elapsed:.2f}", flush=True)
        frame_idx += 1
    cap.release()

    pred = pd.DataFrame(rows)
    if pred.empty:
        pred = pd.DataFrame(
            columns=["video_key", "frame_idx", "local_track_id", "tracklet_key", "det_id", "x1", "y1", "x2", "y2", "score", "coco_cls"]
        )
    gt = pd.read_parquet(args.gt_parquet)
    gt = gt.rename(columns={"frame": "frame_idx", "id": "gt_id"})
    gt = gt[(gt["frame_idx"] >= args.start_frame) & (gt["frame_idx"] < end_frame)]
    gt = gt[(gt["x2"] > gt["x1"]) & (gt["y2"] > gt["y1"])].reset_index(drop=True)
    metrics = evaluate(gt, pred, args.iou_threshold)
    metrics.update(
        {
            "video": str(video_path),
            "gt_parquet": args.gt_parquet,
            "model": args.model,
            "tracker": args.tracker,
            "start_frame": int(args.start_frame),
            "num_frames_requested": int(args.num_frames),
            "num_frames_processed": int(processed),
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "half": bool(args.half),
            "elapsed_seconds": float(time.time() - start_time),
        }
    )
    stem = f"{Path(args.model).stem}_{Path(args.tracker).stem}_s{args.start_frame}_n{processed}_imgsz{args.imgsz}_conf{args.conf:g}"
    pred.to_parquet(output_dir / f"{stem}_tracklets.parquet", index=False)
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
