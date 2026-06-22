#!/usr/bin/env python
"""Probe detector recall against DS1 reference boxes on a bounded frame window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    lx1, ly1, lx2, ly2 = [left[:, i][:, None] for i in range(4)]
    rx1, ry1, rx2, ry2 = [right[:, i][None, :] for i in range(4)]
    ix1 = np.maximum(lx1, rx1)
    iy1 = np.maximum(ly1, ry1)
    ix2 = np.minimum(lx2, rx2)
    iy2 = np.minimum(ly2, ry2)
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    la = np.maximum(lx2 - lx1, 0.0) * np.maximum(ly2 - ly1, 0.0)
    ra = np.maximum(rx2 - rx1, 0.0) * np.maximum(ry2 - ry1, 0.0)
    return (inter / np.maximum(la + ra - inter, 1e-9)).astype(np.float32)


def _read_frames(video_path: str, start: int, end: int) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
    frames: list[np.ndarray] = []
    indices: list[int] = []
    for frame_idx in range(int(start), int(end) + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        indices.append(frame_idx)
    cap.release()
    return frames, indices


def _score_detections(det: pd.DataFrame, gt: pd.DataFrame, iou_thr: float) -> dict[str, object]:
    gt_by_frame = {int(frame): rows for frame, rows in gt.groupby("frame", sort=False)}
    det_by_frame = {int(frame): rows for frame, rows in det.groupby("frame", sort=False)}
    gt_total = int(len(gt))
    det_total = int(len(det))
    gt_matched = 0
    det_matched = 0
    best_gt_ious: list[float] = []
    best_det_ious: list[float] = []
    for frame, gframe in gt_by_frame.items():
        dframe = det_by_frame.get(frame)
        if dframe is None or dframe.empty:
            best_gt_ious.extend([0.0] * len(gframe))
            continue
        gboxes = gframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        dboxes = dframe[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        ious = _iou_matrix(gboxes, dboxes)
        gbest = ious.max(axis=1) if ious.size else np.zeros(len(gframe), dtype=np.float32)
        dbest = ious.max(axis=0) if ious.size else np.zeros(len(dframe), dtype=np.float32)
        best_gt_ious.extend(float(x) for x in gbest.tolist())
        best_det_ious.extend(float(x) for x in dbest.tolist())
        gt_matched += int((gbest >= iou_thr).sum())
        det_matched += int((dbest >= iou_thr).sum())
    for frame, dframe in det_by_frame.items():
        if frame not in gt_by_frame:
            best_det_ious.extend([0.0] * len(dframe))
    return {
        "frames": int(gt["frame"].nunique()),
        "gt_rows": gt_total,
        "det_rows": det_total,
        "gt_matched_rows": int(gt_matched),
        "det_matched_rows": int(det_matched),
        "recall_at_iou": round(float(gt_matched / max(gt_total, 1)), 6),
        "precision_at_iou": round(float(det_matched / max(det_total, 1)), 6),
        "mean_best_gt_iou": round(float(np.mean(best_gt_ious)) if best_gt_ious else 0.0, 6),
        "mean_best_det_iou": round(float(np.mean(best_det_ious)) if best_det_ious else 0.0, 6),
    }


def _tile_frame(frame: np.ndarray, grid: int, overlap: float) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    if grid <= 1:
        return [frame], [(0, 0)]
    height, width = frame.shape[:2]
    tile_w = int(np.ceil(width / grid))
    tile_h = int(np.ceil(height / grid))
    pad_w = int(round(tile_w * overlap))
    pad_h = int(round(tile_h * overlap))
    tiles: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            x1 = max(0, gx * tile_w - pad_w)
            y1 = max(0, gy * tile_h - pad_h)
            x2 = min(width, (gx + 1) * tile_w + pad_w)
            y2 = min(height, (gy + 1) * tile_h + pad_h)
            tiles.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
    return tiles, offsets


def run_yolo(args: argparse.Namespace) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(args.model)
    rows = []
    frames, indices = _read_frames(args.video, args.frame_start, args.frame_end)
    t0 = time.time()
    tiled_frames: list[np.ndarray] = []
    tiled_meta: list[tuple[int, int, int]] = []
    for frame_idx, frame in zip(indices, frames, strict=True):
        tiles, offsets = _tile_frame(frame, args.tile_grid, args.tile_overlap)
        for tile_id, (tile, (xoff, yoff)) in enumerate(zip(tiles, offsets, strict=True)):
            tiled_frames.append(tile)
            tiled_meta.append((frame_idx, tile_id, xoff, yoff))
    for offset in range(0, len(tiled_frames), args.batch):
        batch = tiled_frames[offset : offset + args.batch]
        batch_meta = tiled_meta[offset : offset + args.batch]
        results = model.predict(
            batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            classes=[0],
            verbose=False,
            device=args.device,
            half=args.half,
        )
        for (frame_idx, tile_id, xoff, yoff), result in zip(batch_meta, results, strict=True):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy()
            for det_idx, (box, score, coco_cls) in enumerate(zip(xyxy, conf, cls, strict=True)):
                rows.append(
                    {
                        "frame": int(frame_idx),
                        "det_idx": int(tile_id * 1_000_000 + det_idx),
                        "tile_id": int(tile_id),
                        "x1": float(box[0] + xoff),
                        "y1": float(box[1] + yoff),
                        "x2": float(box[2] + xoff),
                        "y2": float(box[3] + yoff),
                        "score": float(score),
                        "coco_cls": int(coco_cls),
                    }
                )
    det = pd.DataFrame(rows, columns=["frame", "det_idx", "tile_id", "x1", "y1", "x2", "y2", "score", "coco_cls"])
    det.attrs["elapsed_sec"] = round(time.time() - t0, 3)
    det.attrs["frames_read"] = len(frames)
    return det


def load_existing(path: str, start: int, end: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "frame_idx" in df.columns and "frame" not in df.columns:
        df = df.rename(columns={"frame_idx": "frame"})
    if "confidence" in df.columns and "score" not in df.columns:
        df = df.rename(columns={"confidence": "score"})
    df = df[(df["frame"] >= start) & (df["frame"] <= end)].copy()
    if "coco_cls" in df.columns:
        df = df[df["coco_cls"].astype(int) == 0].copy()
    return df[["frame", "x1", "y1", "x2", "y2", "score"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--frame-end", type=int, required=True)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tile-grid", type=int, default=1, help="split each frame into grid x grid overlapping tiles before detection")
    ap.add_argument("--tile-overlap", type=float, default=0.1)
    ap.add_argument("--score-iou", type=float, default=0.5)
    ap.add_argument("--existing-det", default=None, help="optional parquet to score instead of running YOLO")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-parquet", default=None)
    args = ap.parse_args()

    gt = pd.read_parquet(args.reference, columns=["frame", "id", "x1", "y1", "x2", "y2"])
    gt = gt[(gt["frame"] >= args.frame_start) & (gt["frame"] <= args.frame_end)].copy()
    if args.existing_det:
        det = load_existing(args.existing_det, args.frame_start, args.frame_end)
        elapsed = 0.0
        frames_read = 0
    else:
        det = run_yolo(args)
        elapsed = float(det.attrs.get("elapsed_sec", 0.0))
        frames_read = int(det.attrs.get("frames_read", 0))
    metrics = _score_detections(det, gt, args.score_iou)
    result = {
        "video": args.video,
        "reference": args.reference,
        "existing_det": args.existing_det,
        "model": args.model,
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "imgsz": int(args.imgsz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames_read": frames_read,
        "elapsed_sec": elapsed,
        "tile_grid": int(args.tile_grid),
        "tile_overlap": float(args.tile_overlap),
        "score_iou": float(args.score_iou),
        "metrics": metrics,
    }
    if args.out_parquet:
        out = Path(args.out_parquet)
        out.parent.mkdir(parents=True, exist_ok=True)
        det.to_parquet(out, index=False)
        result["out_parquet"] = str(out)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
