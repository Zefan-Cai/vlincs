#!/usr/bin/env python
"""Extract OSNet person ReID features for gallery tracklets.

The extractor samples frames from each tracklet, crops the detection boxes, and
averages an OSNet-MSMT17 embedding.  It reads no identity labels or anchors.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from kit.extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths
except ModuleNotFoundError:
    from extract_tracklet_foundation_features import _connect, _crop, _sample_rows, _video_paths


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1.0e-9)


def _load_osnet(model_path: str, device: str):
    from torchreid.reid.utils.feature_extractor import FeatureExtractor

    return FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=model_path,
        device=device,
        verbose=False,
    )


def _features(model, images: list[Image.Image]) -> np.ndarray:
    with torch.inference_mode():
        feats = model([np.asarray(image.convert("RGB")) for image in images])
    arr = feats.detach().float().cpu().numpy().astype(np.float32)
    return _l2n(arr).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--video-root", action="append", default=[])
    ap.add_argument("--model-path", default="/mnt/localssd/vlincs/models/osnet_x1_0_msmt17.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--crop-margin", type=float, default=0.08)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tracklets", type=int, default=0)
    ap.add_argument("--save-sample-features", action="store_true")
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
    model = _load_osnet(str(args.model_path), str(args.device))

    import cv2  # type: ignore

    caps = {}
    cap_frame: dict[str, int] = {}
    cap_cached: dict[str, np.ndarray] = {}
    seq_feats: dict[int, list[np.ndarray]] = defaultdict(list)
    pending_images: list[Image.Image] = []
    pending_seqs: list[int] = []
    missing_video = 0
    missing_crop = 0
    seen_crops = 0

    def flush() -> None:
        if not pending_images:
            return
        feats = _features(model, pending_images)
        for seq, feat in zip(pending_seqs, feats):
            seq_feats[int(seq)].append(feat)
        pending_images.clear()
        pending_seqs.clear()

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
            pending_images.append(image.resize((128, 256), Image.BILINEAR))
            pending_seqs.append(int(seq))
            seen_crops += 1
            if len(pending_images) >= int(args.batch_size):
                flush()
            if seen_crops and seen_crops % 2000 == 0:
                print(json.dumps({"processed_crops": seen_crops, "seqs_with_features": len(seq_feats)}), flush=True)
        flush()
    finally:
        for cap in caps.values():
            cap.release()

    seqs = np.asarray(sorted(seq_feats), dtype=np.int64)
    features = []
    sample_counts = []
    sample_arrays = []
    for seq in seqs:
        arr = np.stack(seq_feats[int(seq)]).astype(np.float32)
        feat = arr.mean(axis=0)
        features.append((feat / (np.linalg.norm(feat) + 1.0e-9)).astype(np.float32))
        sample_counts.append(int(len(arr)))
        sample_arrays.append(arr)
    features_np = np.stack(features).astype(np.float32) if len(features) else np.zeros((0, 512), np.float32)
    save_kwargs = {}
    if bool(args.save_sample_features):
        max_count = max(sample_counts, default=0)
        sample_np = np.zeros((len(seqs), max_count, features_np.shape[1] if features_np.ndim == 2 else 512), dtype=np.float32)
        for pos, arr in enumerate(sample_arrays):
            sample_np[pos, : len(arr)] = arr
        save_kwargs["sample_features"] = sample_np
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seqs=seqs,
        features=features_np,
        sample_counts=np.asarray(sample_counts, dtype=np.int16),
        **save_kwargs,
        meta=json.dumps(
            {
                "model": "osnet_x1_0_msmt17",
                "model_path": str(args.model_path),
                "samples": int(args.samples),
                "save_sample_features": bool(args.save_sample_features),
                "crop_margin": float(args.crop_margin),
                "sample_counts_min": int(min(sample_counts, default=0)),
                "sample_counts_max": int(max(sample_counts, default=0)),
                "tracklets_with_features": int(len(seqs)),
                "requested_rows": int(len(rows)),
                "seen_crops": int(seen_crops),
                "missing_video": int(missing_video),
                "missing_crop": int(missing_crop),
                "uses_anchors": False,
                "uses_gt": False,
            },
            sort_keys=True,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "shape": list(features_np.shape),
                "tracklets_with_features": int(len(seqs)),
                "seen_crops": int(seen_crops),
                "missing_video": int(missing_video),
                "missing_crop": int(missing_crop),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
