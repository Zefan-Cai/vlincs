#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sample_frames(video_path: Path, frame_ids: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
        frames.append(frame)
    cap.release()
    return frames


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


def greedy_match(gt: pd.DataFrame, pred: pd.DataFrame, iou_threshold: float) -> tuple[int, float]:
    if gt.empty:
        return 0, 0.0
    if pred.empty:
        return 0, 0.0
    mat = iou_matrix(
        gt[["x1", "y1", "x2", "y2"]].to_numpy(float),
        pred[["x1", "y1", "x2", "y2"]].to_numpy(float),
    )
    best_sum = float(mat.max(axis=1).sum())
    coords = np.argwhere(mat >= iou_threshold)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched = 0
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
    return matched, best_sum


def evaluate(gt: pd.DataFrame, detections: pd.DataFrame, iou_threshold: float) -> dict[str, float | int]:
    det_by_frame = {int(k): v for k, v in detections.groupby("frame_idx", sort=False)}
    matched = 0
    best_sum = 0.0
    gt_total = 0
    per_frame_rows: list[dict[str, float | int]] = []
    for frame_idx, gt_frame in gt.groupby("frame_idx", sort=False):
        pred_frame = det_by_frame.get(int(frame_idx), detections.iloc[0:0])
        frame_matched, frame_best_sum = greedy_match(gt_frame, pred_frame, iou_threshold)
        matched += frame_matched
        best_sum += frame_best_sum
        gt_total += len(gt_frame)
        per_frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "gt_boxes": int(len(gt_frame)),
                "pred_boxes": int(len(pred_frame)),
                "matched_boxes_iou50": int(frame_matched),
                "box_recall_iou50": float(frame_matched / len(gt_frame)) if len(gt_frame) else 0.0,
                "box_precision_iou50": float(frame_matched / len(pred_frame)) if len(pred_frame) else 0.0,
            }
        )
    pred_total = int(len(detections))
    return {
        "frames": int(gt["frame_idx"].nunique()),
        "gt_boxes": int(gt_total),
        "pred_boxes": pred_total,
        "matched_boxes_iou50": int(matched),
        "box_recall_iou50": float(matched / gt_total) if gt_total else 0.0,
        "box_precision_iou50": float(matched / pred_total) if pred_total else 0.0,
        "mean_best_iou_per_gt": float(best_sum / gt_total) if gt_total else 0.0,
        "per_frame": per_frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a detector model on sampled video frames and evaluate GT box recall.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-parquet", required=True)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-stride", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    gt = pd.read_parquet(args.gt_parquet)
    gt = gt.rename(columns={"frame": "frame_idx", "id": "gt_id"})
    gt = gt[(gt["x2"] > gt["x1"]) & (gt["y2"] > gt["y1"])].copy()
    all_frame_ids = sorted(int(frame_id) for frame_id in gt["frame_idx"].unique())
    frame_ids = [frame_id for frame_id in all_frame_ids if frame_id % args.frame_stride == 0]
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise ValueError("No frames selected")
    gt = gt[gt["frame_idx"].isin(frame_ids)].reset_index(drop=True)

    frames = load_sample_frames(video_path, frame_ids)
    model = YOLO(args.model)
    results = model.predict(
        source=frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        classes=[0],
        device=args.device,
        batch=args.batch,
        half=args.half,
        verbose=False,
    )

    rows: list[dict[str, float | int | str]] = []
    video_key = video_path.stem
    for frame_idx, result in zip(frame_ids, results):
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
                    "video_key": video_key,
                    "frame_idx": int(frame_idx),
                    "det_id": int(det_idx),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(score),
                    "coco_cls": int(cls),
                    "model": args.model,
                    "imgsz": int(args.imgsz),
                    "conf": float(args.conf),
                }
            )

    detections = pd.DataFrame(rows)
    if detections.empty:
        detections = pd.DataFrame(
            columns=["video_key", "frame_idx", "det_id", "x1", "y1", "x2", "y2", "score", "coco_cls", "model", "imgsz", "conf"]
        )
    metrics = evaluate(gt, detections, args.iou_threshold)
    metrics.update(
        {
            "video": str(video_path),
            "gt_parquet": str(args.gt_parquet),
            "model": args.model,
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
            "selected_frames": frame_ids,
        }
    )

    stem = f"{Path(args.model).stem}_imgsz{args.imgsz}_conf{args.conf:g}_stride{args.frame_stride}"
    detections.to_parquet(output_dir / f"{stem}_detections.parquet", index=False)
    pd.DataFrame(metrics["per_frame"]).to_csv(output_dir / f"{stem}_per_frame.csv", index=False)
    metrics_without_frames = {key: value for key, value in metrics.items() if key != "per_frame"}
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics_without_frames, indent=2) + "\n")
    print(json.dumps(metrics_without_frames, indent=2))


if __name__ == "__main__":
    main()
