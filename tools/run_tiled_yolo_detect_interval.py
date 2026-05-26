#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


@dataclass(frozen=True)
class Tile:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int


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


def axis_starts(length: int, tile_length: int, count: int) -> list[int]:
    if count <= 1 or tile_length >= length:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, length - tile_length, count)})


def make_tiles(width: int, height: int, rows: int, cols: int, overlap: int, include_full_frame: bool) -> list[Tile]:
    tile_w = min(width, int(np.ceil(width / cols)) + overlap)
    tile_h = min(height, int(np.ceil(height / rows)) + overlap)
    xs = axis_starts(width, tile_w, cols)
    ys = axis_starts(height, tile_h, rows)
    tiles: list[Tile] = []
    if include_full_frame:
        tiles.append(Tile("full", 0, 0, width, height))
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tiles.append(Tile(f"r{r}c{c}", x0, y0, min(width, x0 + tile_w), min(height, y0 + tile_h)))
    return tiles


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return keep


def global_nms(frame_df: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    if frame_df.empty:
        return frame_df
    boxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy(float)
    scores = frame_df["score"].to_numpy(float)
    keep = nms_indices(boxes, scores, iou_threshold)
    return frame_df.iloc[keep].reset_index(drop=True)


def parse_scales(value: str) -> list[float]:
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not scales:
        raise ValueError("--box-scale-variants must contain at least one scale")
    if any(scale <= 0 for scale in scales):
        raise ValueError("--box-scale-variants values must be positive")
    return scales


def parse_shifts(value: str) -> list[tuple[float, float]]:
    shifts: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--box-shift-variants entries must use dx:dy format, for example 0.25:0")
        dx, dy = item.split(":", 1)
        shifts.append((float(dx), float(dy)))
    if not shifts:
        raise ValueError("--box-shift-variants must contain at least one dx:dy pair")
    return shifts


def augment_box_variants(
    detections: pd.DataFrame,
    scales: list[float],
    shifts: list[tuple[float, float]],
    width: int,
    height: int,
) -> pd.DataFrame:
    if detections.empty or (scales == [1.0] and shifts == [(0.0, 0.0)]):
        detections = detections.copy()
        detections["box_scale"] = 1.0
        detections["box_shift_x"] = 0.0
        detections["box_shift_y"] = 0.0
        return detections
    parts: list[pd.DataFrame] = []
    base_boxes = detections[["x1", "y1", "x2", "y2"]].to_numpy(float)
    cx = (base_boxes[:, 0] + base_boxes[:, 2]) / 2.0
    cy = (base_boxes[:, 1] + base_boxes[:, 3]) / 2.0
    box_w = base_boxes[:, 2] - base_boxes[:, 0]
    box_h = base_boxes[:, 3] - base_boxes[:, 1]
    for scale in scales:
        scaled_w = box_w * scale
        scaled_h = box_h * scale
        for shift_x, shift_y in shifts:
            shifted_cx = cx + shift_x * scaled_w
            shifted_cy = cy + shift_y * scaled_h
            variant = detections.copy()
            variant["x1"] = np.clip(shifted_cx - scaled_w / 2.0, 0, width)
            variant["y1"] = np.clip(shifted_cy - scaled_h / 2.0, 0, height)
            variant["x2"] = np.clip(shifted_cx + scaled_w / 2.0, 0, width)
            variant["y2"] = np.clip(shifted_cy + scaled_h / 2.0, 0, height)
            variant["box_scale"] = float(scale)
            variant["box_shift_x"] = float(shift_x)
            variant["box_shift_y"] = float(shift_y)
            parts.append(variant)
    return pd.concat(parts, ignore_index=True)


def assign_singleton_tracklets(detections: pd.DataFrame, video_key: str) -> pd.DataFrame:
    detections = detections.copy()
    if detections.empty:
        detections["local_track_id"] = []
        detections["tracklet_key"] = []
        return detections
    detections["local_track_id"] = np.arange(1, len(detections) + 1, dtype=int)
    detections["tracklet_key"] = [f"{video_key}:single:{track_id}" for track_id in detections["local_track_id"]]
    return detections


def assign_iou_tracklets(detections: pd.DataFrame, video_key: str, iou_threshold: float, max_gap: int) -> pd.DataFrame:
    if detections.empty:
        detections = detections.copy()
        detections["local_track_id"] = []
        detections["tracklet_key"] = []
        return detections

    rows: list[pd.DataFrame] = []
    active: dict[int, dict[str, object]] = {}
    next_track_id = 1
    for frame_idx, frame_df in detections.groupby("frame_idx", sort=True):
        frame_df = frame_df.copy().reset_index(drop=True)
        frame_boxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy(float)
        candidate_ids = [tid for tid, state in active.items() if int(frame_idx) - int(state["frame_idx"]) <= max_gap]
        assigned_tracks = np.full(len(frame_df), -1, dtype=int)

        if candidate_ids and len(frame_df) > 0:
            active_boxes = np.array([active[tid]["box"] for tid in candidate_ids], dtype=float)
            mat = iou_matrix(active_boxes, frame_boxes)
            pairs = [
                (float(mat[i, j]), int(i), int(j))
                for i in range(mat.shape[0])
                for j in range(mat.shape[1])
                if mat[i, j] >= iou_threshold
            ]
            used_active: set[int] = set()
            used_det: set[int] = set()
            for _score, active_i, det_j in sorted(pairs, reverse=True):
                if active_i in used_active or det_j in used_det:
                    continue
                track_id = candidate_ids[active_i]
                assigned_tracks[det_j] = track_id
                used_active.add(active_i)
                used_det.add(det_j)

        for det_j, track_id in enumerate(assigned_tracks):
            if track_id < 0:
                track_id = next_track_id
                next_track_id += 1
                assigned_tracks[det_j] = track_id
            active[int(track_id)] = {
                "frame_idx": int(frame_idx),
                "box": frame_boxes[det_j].tolist(),
            }

        frame_df["local_track_id"] = assigned_tracks.astype(int)
        frame_df["tracklet_key"] = [f"{video_key}:tile:{track_id}" for track_id in assigned_tracks]
        rows.append(frame_df)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiled YOLO detection over one video interval, link boxes into tracklets, and evaluate recall.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--model-nms-iou", type=float, default=0.95)
    parser.add_argument("--global-nms-iou", type=float, default=0.98)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--include-full-frame", action="store_true")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--link-mode", choices=["iou", "singleton"], default="iou")
    parser.add_argument("--link-iou", type=float, default=0.25)
    parser.add_argument("--link-max-gap", type=int, default=30)
    parser.add_argument("--box-scale-variants", default="1.0")
    parser.add_argument("--box-shift-variants", default="0:0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    video_key = video_path.stem
    end_frame = args.start_frame + args.num_frames

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    box_scales = parse_scales(args.box_scale_variants)
    box_shifts = parse_shifts(args.box_shift_variants)

    all_rows: list[dict[str, float | int | str]] = []
    start_time = time.time()
    frame_idx = args.start_frame
    processed = 0
    tile_spec: list[dict[str, int | str]] | None = None
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        tiles = make_tiles(width, height, args.rows, args.cols, args.overlap, args.include_full_frame)
        if tile_spec is None:
            tile_spec = [tile.__dict__ for tile in tiles]

        tile_images = [frame[tile.y0 : tile.y1, tile.x0 : tile.x1] for tile in tiles]
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
            augment=args.augment,
            verbose=False,
        )
        frame_rows: list[dict[str, float | int | str]] = []
        for tile, result in zip(tiles, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            for det_idx, (box, score, cls) in enumerate(zip(xyxy, scores, classes)):
                x1, y1, x2, y2 = [float(v) for v in box]
                frame_rows.append(
                    {
                        "video_key": video_key,
                        "frame_idx": int(frame_idx),
                        "tile": tile.name,
                        "tile_x0": int(tile.x0),
                        "tile_y0": int(tile.y0),
                        "det_id": int(det_idx),
                        "x1": max(0.0, min(float(width), x1 + tile.x0)),
                        "y1": max(0.0, min(float(height), y1 + tile.y0)),
                        "x2": max(0.0, min(float(width), x2 + tile.x0)),
                        "y2": max(0.0, min(float(height), y2 + tile.y0)),
                        "score": float(score),
                        "coco_cls": int(cls),
                    }
                )
        if frame_rows:
            frame_df = pd.DataFrame(frame_rows)
            frame_df = frame_df[(frame_df["x2"] > frame_df["x1"]) & (frame_df["y2"] > frame_df["y1"])]
            if args.global_nms_iou < 1.0:
                frame_df = global_nms(frame_df, args.global_nms_iou)
            all_rows.extend(frame_df.to_dict("records"))

        processed += 1
        if processed % 50 == 0:
            elapsed = max(1e-6, time.time() - start_time)
            print(f"[tiled] processed={processed}/{args.num_frames} fps={processed / elapsed:.2f}", flush=True)
        frame_idx += 1
    cap.release()

    detections = pd.DataFrame(all_rows)
    if detections.empty:
        detections = pd.DataFrame(
            columns=["video_key", "frame_idx", "tile", "tile_x0", "tile_y0", "det_id", "x1", "y1", "x2", "y2", "score", "coco_cls"]
        )
    detections = augment_box_variants(detections, box_scales, box_shifts, frame_width, frame_height)
    if args.link_mode == "singleton":
        detections = assign_singleton_tracklets(detections, video_key)
    else:
        detections = assign_iou_tracklets(detections, video_key, args.link_iou, args.link_max_gap)

    gt = pd.read_parquet(args.gt_parquet)
    gt = gt.rename(columns={"frame": "frame_idx", "id": "gt_id"})
    gt = gt[(gt["frame_idx"] >= args.start_frame) & (gt["frame_idx"] < args.start_frame + processed)]
    gt = gt[(gt["x2"] > gt["x1"]) & (gt["y2"] > gt["y1"])].reset_index(drop=True)

    metrics = evaluate(gt, detections, args.iou_threshold)
    metrics.update(
        {
            "video": str(video_path),
            "gt_parquet": args.gt_parquet,
            "model": args.model,
            "start_frame": int(args.start_frame),
            "num_frames_requested": int(args.num_frames),
            "num_frames_processed": int(processed),
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "model_nms_iou": float(args.model_nms_iou),
            "global_nms_iou": float(args.global_nms_iou),
            "max_det": int(args.max_det),
            "rows": int(args.rows),
            "cols": int(args.cols),
            "overlap": int(args.overlap),
            "include_full_frame": bool(args.include_full_frame),
            "half": bool(args.half),
            "augment": bool(args.augment),
            "link_mode": args.link_mode,
            "link_iou": float(args.link_iou),
            "link_max_gap": int(args.link_max_gap),
            "box_scale_variants": box_scales,
            "box_shift_variants": box_shifts,
            "elapsed_seconds": float(time.time() - start_time),
            "tile_spec": tile_spec or [],
        }
    )

    full_tag = "full" if args.include_full_frame else "tiles"
    aug_tag = "_aug" if args.augment else ""
    scale_tag = "scales" + "-".join(f"{scale:g}" for scale in box_scales)
    shift_tag = "shifts" + "-".join(f"{dx:g}x{dy:g}" for dx, dy in box_shifts)
    stem = (
        f"{Path(args.model).stem}_tiled_{args.rows}x{args.cols}_{full_tag}"
        f"_s{args.start_frame}_n{processed}_imgsz{args.imgsz}_conf{args.conf:g}"
        f"_nms{args.global_nms_iou:g}_maxdet{args.max_det}_{scale_tag}_{shift_tag}{aug_tag}"
    )
    detections.to_parquet(output_dir / f"{stem}_tracklets.parquet", index=False)
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
