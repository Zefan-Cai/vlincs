#!/usr/bin/env python
"""Extract no-anchor pose-aligned body-part color features for tracklets.

This uses YOLO pose only as an evidence extractor.  It reads tracklet boxes,
samples source frames, crops each box, estimates keypoints inside the crop, and
builds head/torso/legs color summaries.  It never reads identity labels.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from kit.extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths
except ModuleNotFoundError:
    from extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths


def _l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1.0e-9)


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    xi1 = max(0, min(w - 1, int(np.floor(x1))))
    yi1 = max(0, min(h - 1, int(np.floor(y1))))
    xi2 = max(xi1 + 1, min(w, int(np.ceil(x2))))
    yi2 = max(yi1 + 1, min(h, int(np.ceil(y2))))
    return xi1, yi1, xi2, yi2


def _part_hist(rgb: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).reshape(-1)
    s_hist = cv2.calcHist([hsv], [1], None, [6], [0, 256]).reshape(-1)
    v_hist = cv2.calcHist([hsv], [2], None, [6], [0, 256]).reshape(-1)
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= float(hist.sum() + 1.0e-6)
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    stats = np.concatenate([flat.mean(axis=0), flat.std(axis=0), np.median(flat, axis=0)]).astype(np.float32)
    return np.concatenate([hist, stats]).astype(np.float32)


def _best_keypoints(result, width: int, height: int) -> np.ndarray | None:
    if result.keypoints is None or result.keypoints.data is None:
        return None
    data = result.keypoints.data.detach().cpu().numpy()
    if data.size == 0:
        return None
    boxes = result.boxes
    if boxes is not None and boxes.conf is not None and len(boxes.conf):
        idx = int(np.argmax(boxes.conf.detach().cpu().numpy()))
    else:
        idx = 0
    keypoints = np.asarray(data[idx], dtype=np.float32)
    if keypoints.shape[0] < 17 or keypoints.shape[1] < 3:
        return None
    visible = keypoints[:, 2] > 0.20
    if int(np.count_nonzero(visible)) < 5:
        return None
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, max(width - 1, 1))
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, max(height - 1, 1))
    return keypoints


def _mean_point(kpts: np.ndarray, indices: list[int], min_conf: float = 0.20) -> tuple[float, float] | None:
    pts = [kpts[idx, :2] for idx in indices if idx < len(kpts) and float(kpts[idx, 2]) >= min_conf]
    if not pts:
        return None
    arr = np.stack(pts).astype(np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _part_boxes(width: int, height: int, kpts: np.ndarray | None) -> tuple[list[tuple[int, int, int, int]], int]:
    if kpts is None:
        return [
            _clip_box(0, 0, width, 0.30 * height, width, height),
            _clip_box(0, 0.25 * height, width, 0.66 * height, width, height),
            _clip_box(0, 0.58 * height, width, height, width, height),
            _clip_box(0, 0, width, height, width, height),
        ], 0

    visible = kpts[:, 2] >= 0.20
    xs = kpts[visible, 0] if np.any(visible) else np.asarray([0.0, float(width)])
    shoulder = _mean_point(kpts, [5, 6]) or (float(width) / 2.0, 0.28 * float(height))
    hip = _mean_point(kpts, [11, 12]) or (float(width) / 2.0, 0.62 * float(height))
    knee = _mean_point(kpts, [13, 14]) or (float(width) / 2.0, 0.78 * float(height))
    ankle = _mean_point(kpts, [15, 16]) or (float(width) / 2.0, 0.96 * float(height))
    x_min = float(np.percentile(xs, 10))
    x_max = float(np.percentile(xs, 90))
    pad = 0.18 * max(x_max - x_min, 0.45 * width)
    body_x1 = x_min - pad
    body_x2 = x_max + pad
    head = _clip_box(body_x1, 0, body_x2, max(shoulder[1], 0.25 * height), width, height)
    torso = _clip_box(body_x1, max(0, shoulder[1] - 0.05 * height), body_x2, min(height, hip[1] + 0.08 * height), width, height)
    legs = _clip_box(body_x1, max(0, hip[1] - 0.04 * height), body_x2, min(height, max(knee[1], ankle[1])), width, height)
    full = _clip_box(0, 0, width, height, width, height)
    return [head, torso, legs, full], 1


def _feature(image: Image.Image, result) -> tuple[np.ndarray, int]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    kpts = _best_keypoints(result, width, height)
    boxes, pose_ok = _part_boxes(width, height, kpts)
    parts = []
    area_total = float(max(width * height, 1))
    for x1, y1, x2, y2 in boxes:
        crop = rgb[y1:y2, x1:x2]
        parts.append(_part_hist(crop))
        parts.append(np.asarray([(x2 - x1) * (y2 - y1) / area_total], dtype=np.float32))
    if kpts is None:
        pose_vec = np.zeros(17 * 3, dtype=np.float32)
    else:
        pose = kpts[:17].copy()
        pose[:, 0] /= max(float(width), 1.0)
        pose[:, 1] /= max(float(height), 1.0)
        pose[:, 2] = np.clip(pose[:, 2], 0.0, 1.0)
        pose_vec = pose.reshape(-1).astype(np.float32)
    out = np.concatenate([*parts, pose_vec, np.asarray([float(pose_ok)], dtype=np.float32)]).astype(np.float32)
    return _l2n(out), int(pose_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--video-root", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--crop-margin", type=float, default=0.08)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--max-tracklets", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--tracklet-order", default="seq", choices=["seq", "n_dets_desc"])
    ap.add_argument("--sequential-read-max-gap", type=int, default=300)
    args = ap.parse_args()

    con = _connect(args.dbname)
    rows = _sample_rows(
        con,
        int(args.samples),
        max_tracklets=int(args.max_tracklets),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        tracklet_order=str(args.tracklet_order),
    )
    video_paths = _video_paths(args.dataset, list(args.video_root))
    model = YOLO(args.model)

    caps = {}
    cap_frame: dict[str, int] = {}
    cap_cached: dict[str, np.ndarray] = {}
    pending_images: list[Image.Image] = []
    pending_seqs: list[int] = []
    seq_feats: dict[int, list[np.ndarray]] = defaultdict(list)
    pose_counts: defaultdict[int, int] = defaultdict(int)
    missing_video = 0
    missing_crop = 0
    seen_crops = 0

    def read_frame(path: str, frame_idx: int):
        cap = caps.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            caps[path] = cap
            cap_frame[path] = -1
        target = int(frame_idx)
        current = int(cap_frame.get(path, -1))
        if current == target and path in cap_cached:
            return True, cap_cached[path]
        if current >= 0 and target > current and target - current <= int(args.sequential_read_max_gap):
            ok = False
            frame = None
            for next_frame in range(current + 1, target + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap_frame[path] = next_frame - 1
                    return False, None
            cap_frame[path] = target
            cap_cached[path] = frame
            return True, frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap_frame[path] = target
            cap_cached[path] = frame
        return ok, frame

    def flush() -> None:
        if not pending_images:
            return
        results = model.predict(
            pending_images,
            imgsz=int(args.imgsz),
            device=str(args.device),
            verbose=False,
        )
        for seq, image, result in zip(pending_seqs, pending_images, results):
            feat, pose_ok = _feature(image, result)
            seq_feats[int(seq)].append(feat)
            pose_counts[int(seq)] += int(pose_ok)
        pending_images.clear()
        pending_seqs.clear()

    try:
        for seq, video, frame_idx, x1, y1, x2, y2 in rows:
            path = video_paths.get(str(video))
            if not path:
                missing_video += 1
                continue
            ok, frame = read_frame(path, int(frame_idx))
            if not ok or frame is None:
                missing_crop += 1
                continue
            image = _crop(frame, (x1, y1, x2, y2), float(args.crop_margin))
            if image is None:
                missing_crop += 1
                continue
            pending_images.append(image)
            pending_seqs.append(int(seq))
            seen_crops += 1
            if len(pending_images) >= int(args.batch_size):
                flush()
            if seen_crops and seen_crops % 1000 == 0:
                print(json.dumps({"processed_crops": seen_crops, "seqs_with_features": len(seq_feats)}), flush=True)
        flush()
    finally:
        for cap in caps.values():
            cap.release()

    seqs = np.asarray(sorted(seq_feats), dtype=np.int64)
    features = []
    sample_counts = []
    pose_success_counts = []
    for seq in seqs:
        arr = np.stack(seq_feats[int(seq)]).astype(np.float32)
        feat = arr.mean(axis=0)
        features.append(_l2n(feat).astype(np.float32))
        sample_counts.append(int(len(arr)))
        pose_success_counts.append(int(pose_counts[int(seq)]))
    features_np = np.stack(features).astype(np.float32) if len(features) else np.zeros((0, 0), np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seqs=seqs,
        features=features_np,
        sample_counts=np.asarray(sample_counts, dtype=np.int16),
        pose_success_counts=np.asarray(pose_success_counts, dtype=np.int16),
        model=np.asarray([args.model], dtype=object),
    )
    info = {
        "out": str(out),
        "model": str(args.model),
        "tracklets_with_features": int(len(seqs)),
        "feature_dim": int(features_np.shape[1]) if features_np.ndim == 2 and len(features_np) else 0,
        "sample_rows": int(len(rows)),
        "seen_crops": int(seen_crops),
        "pose_success_rows": int(sum(pose_success_counts)),
        "pose_success_tracklets": int(sum(1 for value in pose_success_counts if int(value) > 0)),
        "missing_video_rows": int(missing_video),
        "missing_crop_rows": int(missing_crop),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "uses_anchors": False,
        "uses_gt": False,
    }
    print(json.dumps(info, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
