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


def axis_starts(length: int, tile_length: int, count: int) -> list[int]:
    if count <= 1 or tile_length >= length:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, length - tile_length, count)})


def make_tiles(width: int, height: int, rows: int, cols: int, overlap: int, include_full_frame: bool) -> list[tuple[str, int, int, int, int]]:
    tile_w = min(width, int(np.ceil(width / cols)) + overlap)
    tile_h = min(height, int(np.ceil(height / rows)) + overlap)
    xs = axis_starts(width, tile_w, cols)
    ys = axis_starts(height, tile_h, rows)
    tiles: list[tuple[str, int, int, int, int]] = []
    if include_full_frame:
        tiles.append(("full", 0, 0, width, height))
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tiles.append((f"r{r}c{c}", x0, y0, min(width, x0 + tile_w), min(height, y0 + tile_h)))
    return tiles


def parse_scales(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_shifts(value: str) -> list[tuple[float, float]]:
    shifts: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dx, dy = item.split(":", 1)
        shifts.append((float(dx), float(dy)))
    return shifts


def box_variants(base: np.ndarray, scales: list[float], shifts: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    if len(base) == 0:
        return base
    cx = (base[:, 0] + base[:, 2]) / 2.0
    cy = (base[:, 1] + base[:, 3]) / 2.0
    box_w = base[:, 2] - base[:, 0]
    box_h = base[:, 3] - base[:, 1]
    parts: list[np.ndarray] = []
    for scale in scales:
        scaled_w = box_w * scale
        scaled_h = box_h * scale
        for shift_x, shift_y in shifts:
            shifted_cx = cx + shift_x * scaled_w
            shifted_cy = cy + shift_y * scaled_h
            out = np.empty_like(base)
            out[:, 0] = np.clip(shifted_cx - scaled_w / 2.0, 0, width)
            out[:, 1] = np.clip(shifted_cy - scaled_h / 2.0, 0, height)
            out[:, 2] = np.clip(shifted_cx + scaled_w / 2.0, 0, width)
            out[:, 3] = np.clip(shifted_cy + scaled_h / 2.0, 0, height)
            parts.append(out)
    return np.concatenate(parts, axis=0)


def evaluate_by_frame(
    gt: pd.DataFrame,
    pred_by_frame: dict[int, pd.DataFrame],
    iou_threshold: float,
    scales: list[float] | None = None,
    shifts: list[tuple[float, float]] | None = None,
    width: int = 2560,
    height: int = 1920,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    matched = 0
    best_sum = 0.0
    gt_total = 0
    pred_total = 0
    per_frame: list[dict[str, float | int]] = []
    for frame_idx, gt_frame in gt.groupby("frame_idx", sort=False):
        pred_frame = pred_by_frame.get(int(frame_idx))
        gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        gt_total += len(gt_boxes)
        if pred_frame is None or pred_frame.empty:
            per_frame.append({"frame_idx": int(frame_idx), "gt_boxes": len(gt_boxes), "pred_boxes": 0, "matched": 0, "recall": 0.0})
            continue
        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        if scales is not None and shifts is not None:
            pred_boxes = box_variants(pred_boxes, scales, shifts, width, height)
        pred_total += len(pred_boxes)
        mat = iou_matrix(gt_boxes, pred_boxes)
        best_sum += float(mat.max(axis=1).sum())
        coords = np.argwhere(mat >= iou_threshold)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        frame_matched = 0
        for i, j in sorted(((int(i), int(j)) for i, j in coords), key=lambda ij: mat[ij[0], ij[1]], reverse=True):
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            matched += 1
            frame_matched += 1
        per_frame.append(
            {
                "frame_idx": int(frame_idx),
                "gt_boxes": int(len(gt_boxes)),
                "pred_boxes": int(len(pred_boxes)),
                "matched": int(frame_matched),
                "recall": float(frame_matched / len(gt_boxes)) if len(gt_boxes) else 0.0,
            }
        )
    metrics = {
        "frames": int(gt["frame_idx"].nunique()),
        "gt_boxes": int(gt_total),
        "pred_boxes": int(pred_total),
        "matched_boxes_iou50": int(matched),
        "box_recall_iou50": float(matched / gt_total) if gt_total else 0.0,
        "box_precision_iou50": float(matched / pred_total) if pred_total else 0.0,
        "mean_best_iou_per_gt": float(best_sum / gt_total) if gt_total else 0.0,
    }
    return metrics, per_frame


def load_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def detect_video(args: argparse.Namespace, model: YOLO, video_path: Path, gt_path: Path) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    video_key = video_path.stem
    gt = pd.read_parquet(gt_path).rename(columns={"frame": "frame_idx", "id": "gt_id"})
    gt = gt[(gt["x2"] > gt["x1"]) & (gt["y2"] > gt["y1"])].copy()
    frame_ids = sorted(int(frame_id) for frame_id in gt["frame_idx"].unique() if int(frame_id) % args.frame_stride == 0)
    if args.max_frames_per_video > 0:
        frame_ids = frame_ids[: args.max_frames_per_video]
    gt = gt[gt["frame_idx"].isin(frame_ids)].reset_index(drop=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tiles = make_tiles(width, height, args.rows, args.cols, args.overlap, args.include_full_frame)

    rows: list[dict[str, float | int | str]] = []
    start = time.time()
    for idx, frame_idx in enumerate(frame_ids, start=1):
        frame = load_frame(cap, frame_idx)
        tile_images = [frame[y0:y1, x0:x1] for _name, x0, y0, x1, y1 in tiles]
        results = model.predict(
            source=tile_images,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.model_nms_iou,
            max_det=args.max_det,
            classes=[0],
            device=args.device,
            batch=args.batch,
            half=args.half,
            verbose=False,
        )
        for (tile_name, x0, y0, _x1, _y1), result in zip(tiles, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            for det_idx, (box, score, cls) in enumerate(zip(xyxy, scores, classes)):
                x1, y1, x2, y2 = [float(v) for v in box]
                rows.append(
                    {
                        "video": video_key,
                        "frame_idx": int(frame_idx),
                        "tile": tile_name,
                        "det_id": int(det_idx),
                        "x1": max(0.0, min(float(width), x1 + x0)),
                        "y1": max(0.0, min(float(height), y1 + y0)),
                        "x2": max(0.0, min(float(width), x2 + x0)),
                        "y2": max(0.0, min(float(height), y2 + y0)),
                        "score": float(score),
                        "coco_cls": int(cls),
                    }
                )
        if idx % 10 == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(f"[sample-all] {video_key} frames={idx}/{len(frame_ids)} fps={idx / elapsed:.2f}", flush=True)
    cap.release()

    pred = pd.DataFrame(rows)
    if pred.empty:
        pred = pd.DataFrame(columns=["video", "frame_idx", "tile", "det_id", "x1", "y1", "x2", "y2", "score", "coco_cls"])
    pred = pred[(pred["x2"] > pred["x1"]) & (pred["y2"] > pred["y1"])].reset_index(drop=True)
    return rows, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampled recall-first tiled YOLO evaluation for all sample VLINCS videos.")
    parser.add_argument("--sample-root", default="/mnt/localssd/vlincs/VLINCS_Performer/sample")
    parser.add_argument("--model", default="/mnt/localssd/vlincs/yolo11x.pt")
    parser.add_argument("--output-dir", default="/mnt/localssd/vlincs/eval/recall_first_all_videos")
    parser.add_argument("--frame-stride", type=int, default=600)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.0001)
    parser.add_argument("--model-nms-iou", type=float, default=0.99)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--include-full-frame", action="store_true")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--box-scale-variants", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--box-shift-variants", default="0:0,0.25:0,-0.25:0,0:0.25,0:-0.25")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    videos = sorted((sample_root / "videos").glob("*.mp4"))
    model = YOLO(args.model)
    scales = parse_scales(args.box_scale_variants)
    shifts = parse_shifts(args.box_shift_variants)

    all_summary: list[dict[str, float | int | str]] = []
    aggregate_pred_total = 0
    aggregate_gt_total = 0
    aggregate_matched_raw = 0
    aggregate_matched_expanded = 0
    for video_path in videos:
        video_key = video_path.stem
        gt_path = sample_root / "reference_annotations" / f"{video_key}_v1.7.2.parquet"
        rows, gt = detect_video(args, model, video_path, gt_path)
        pred = pd.DataFrame(rows)
        if pred.empty:
            pred = pd.DataFrame(columns=["video", "frame_idx", "tile", "det_id", "x1", "y1", "x2", "y2", "score", "coco_cls"])
        pred = pred[(pred["x2"] > pred["x1"]) & (pred["y2"] > pred["y1"])].reset_index(drop=True)
        pred_by_frame = {int(k): v for k, v in pred.groupby("frame_idx", sort=False)}
        raw_metrics, raw_per_frame = evaluate_by_frame(gt, pred_by_frame, args.iou_threshold)
        expanded_metrics, expanded_per_frame = evaluate_by_frame(gt, pred_by_frame, args.iou_threshold, scales, shifts)

        pred.to_parquet(output_dir / f"{video_key}_sampled_stride{args.frame_stride}_raw_detections.parquet", index=False)
        pd.DataFrame(raw_per_frame).to_csv(output_dir / f"{video_key}_sampled_stride{args.frame_stride}_raw_per_frame.csv", index=False)
        pd.DataFrame(expanded_per_frame).to_csv(output_dir / f"{video_key}_sampled_stride{args.frame_stride}_scale_shift_per_frame.csv", index=False)

        for method, metrics in [("raw_tiled", raw_metrics), ("scale_shift_eval", expanded_metrics)]:
            all_summary.append(
                {
                    "video": video_key,
                    "method": method,
                    "frame_stride": int(args.frame_stride),
                    "frames": metrics["frames"],
                    "gt_boxes": metrics["gt_boxes"],
                    "pred_boxes": metrics["pred_boxes"],
                    "matched_boxes_iou50": metrics["matched_boxes_iou50"],
                    "box_recall_iou50": metrics["box_recall_iou50"],
                    "box_precision_iou50": metrics["box_precision_iou50"],
                    "mean_best_iou_per_gt": metrics["mean_best_iou_per_gt"],
                }
            )
        aggregate_gt_total += int(raw_metrics["gt_boxes"])
        aggregate_pred_total += int(raw_metrics["pred_boxes"])
        aggregate_matched_raw += int(raw_metrics["matched_boxes_iou50"])
        aggregate_matched_expanded += int(expanded_metrics["matched_boxes_iou50"])
        print(
            f"[sample-all] done {video_key} raw_recall={raw_metrics['box_recall_iou50']:.6f} "
            f"expanded_recall={expanded_metrics['box_recall_iou50']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(all_summary)
    summary.to_csv(output_dir / f"recall_first_all_videos_stride{args.frame_stride}_per_video.csv", index=False)
    aggregate = {
        "frame_stride": int(args.frame_stride),
        "videos": int(len(videos)),
        "raw_gt_boxes": int(aggregate_gt_total),
        "raw_pred_boxes": int(aggregate_pred_total),
        "raw_matched_boxes_iou50": int(aggregate_matched_raw),
        "raw_box_recall_iou50": float(aggregate_matched_raw / aggregate_gt_total) if aggregate_gt_total else 0.0,
        "raw_box_precision_iou50": float(aggregate_matched_raw / aggregate_pred_total) if aggregate_pred_total else 0.0,
        "expanded_pred_boxes": int(aggregate_pred_total * len(scales) * len(shifts)),
        "expanded_matched_boxes_iou50": int(aggregate_matched_expanded),
        "expanded_box_recall_iou50": float(aggregate_matched_expanded / aggregate_gt_total) if aggregate_gt_total else 0.0,
        "expanded_box_precision_iou50": float(aggregate_matched_expanded / (aggregate_pred_total * len(scales) * len(shifts)))
        if aggregate_pred_total
        else 0.0,
        "box_scale_variants": scales,
        "box_shift_variants": shifts,
    }
    (output_dir / f"recall_first_all_videos_stride{args.frame_stride}_aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
