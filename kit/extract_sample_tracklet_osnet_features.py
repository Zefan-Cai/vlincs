#!/usr/bin/env python
"""Extract OSNet/color crop features for sample-parquet tracklets.

The existing OSNet extractor reads current gallery DB tracklets.  This variant
reads a sample parquet produced by ``prepare_relink_no_anchor_sample.py`` so
alternative relinked tracklets can get fresh crop-level evidence rather than
inheriting the old pooled DB embedding.  It reads no identity labels/anchors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from kit.extract_tracklet_foundation_features import _color_hist_features, _crop, _video_paths
    from kit.extract_tracklet_osnet_features import _load_osnet
except ModuleNotFoundError:
    from extract_tracklet_foundation_features import _color_hist_features, _crop, _video_paths
    from extract_tracklet_osnet_features import _load_osnet


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return (x / (float(np.linalg.norm(x)) + 1.0e-9)).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _base_video_key(video_key: object) -> str:
    return str(video_key).split("__", 1)[0]


def _sample_tracklet_rows(df: pd.DataFrame, samples: int, *, num_shards: int, shard_index: int, max_tracklets: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    samples = max(int(samples), 1)
    num_shards = max(int(num_shards), 1)
    shard_index = min(max(int(shard_index), 0), num_shards - 1)
    ordered = (
        df.groupby("tracklet_key", sort=False)
        .agg(video_key=("video_key", "first"), start_frame=("frame_idx", "min"), end_frame=("frame_idx", "max"), n_dets=("frame_idx", "size"))
        .reset_index()
        .sort_values(["video_key", "start_frame", "end_frame", "tracklet_key"], kind="mergesort")
        .reset_index(drop=True)
    )
    ordered["index"] = np.arange(len(ordered), dtype=np.int64)
    selected = ordered[(ordered["index"] % num_shards) == shard_index].copy()
    if int(max_tracklets) > 0:
        selected = selected.head(int(max_tracklets)).copy()
    selected_keys = set(selected["tracklet_key"].astype(str).tolist())
    rows = []
    records = []
    for index, (key, group) in enumerate(df[df["tracklet_key"].astype(str).isin(selected_keys)].sort_values(["tracklet_key", "frame_idx"], kind="mergesort").groupby("tracklet_key", sort=False)):
        group = group.sort_values("frame_idx", kind="mergesort")
        if samples == 1:
            positions = np.asarray([len(group) // 2], dtype=int)
        elif samples == 2:
            positions = np.asarray([0, len(group) - 1], dtype=int)
        else:
            positions = np.linspace(0, len(group) - 1, min(samples, len(group))).round().astype(int)
        positions = sorted(set(int(pos) for pos in positions.tolist()))
        sample = group.iloc[positions].copy()
        rows.append(sample)
        first = group.iloc[0]
        records.append(
            {
                "index": int(index),
                "tracklet_key": str(key),
                "video": str(first["video_key"]),
                "video_lookup_key": _base_video_key(first["video_key"]),
                "camera": str(first.get("camera", "")),
                "start_frame": int(group["frame_idx"].min()),
                "end_frame": int(group["frame_idx"].max()),
                "n_dets": int(len(group)),
            }
        )
    if not rows:
        raise RuntimeError("no sampled rows selected")
    sampled = pd.concat(rows, ignore_index=True).sort_values(["video_key", "frame_idx", "tracklet_key"], kind="mergesort")
    return sampled, records


def _osnet_features(model, images: list[Image.Image]) -> np.ndarray:
    with torch.inference_mode():
        feats = model([np.asarray(image.convert("RGB")) for image in images])
    return _l2n(feats.detach().float().cpu().numpy().astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklet-parquet", required=True)
    parser.add_argument("--dataset", default="ds1")
    parser.add_argument("--video-root", action="append", default=[])
    parser.add_argument("--model-path", default="/mnt/localssd/vlincs/models/osnet_x1_0_msmt17.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--crop-margin", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tracklets", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sequential-read-max-gap", type=int, default=300)
    parser.add_argument(
        "--save-prototypes",
        action="store_true",
        help="also save per-sampled-crop prototype tensors for pair-feature ablations",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.tracklet_parquet)
    required = {"video_key", "frame_idx", "x1", "y1", "x2", "y2", "tracklet_key"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"sample parquet missing columns: {missing}")
    sampled, records = _sample_tracklet_rows(
        df,
        int(args.samples),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        max_tracklets=int(args.max_tracklets),
    )
    key_to_index = {str(record["tracklet_key"]): int(record["index"]) for record in records}
    video_paths = _video_paths(args.dataset, list(args.video_root))
    model = _load_osnet(str(args.model_path), str(args.device))

    import cv2  # type: ignore

    caps = {}
    cap_frame: dict[str, int] = {}
    cap_cached: dict[str, np.ndarray] = {}
    osnet_by_key: dict[str, list[np.ndarray]] = defaultdict(list)
    color_by_key: dict[str, list[np.ndarray]] = defaultdict(list)
    pending_images: list[Image.Image] = []
    pending_keys: list[str] = []
    missing_video = 0
    missing_crop = 0
    seen_crops = 0

    def flush() -> None:
        if not pending_images:
            return
        osnet = _osnet_features(model, pending_images)
        color = _color_hist_features(pending_images)
        for key, os_feat, color_feat in zip(pending_keys, osnet, color):
            osnet_by_key[str(key)].append(os_feat)
            color_by_key[str(key)].append(color_feat)
        pending_images.clear()
        pending_keys.clear()

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
        for row in sampled.itertuples(index=False):
            video = str(row.video_key)
            path = video_paths.get(video) or video_paths.get(_base_video_key(video))
            if not path:
                missing_video += 1
                continue
            ok, frame = read_frame(path, int(row.frame_idx))
            if not ok or frame is None:
                missing_crop += 1
                continue
            image = _crop(frame, (row.x1, row.y1, row.x2, row.y2), float(args.crop_margin))
            if image is None:
                missing_crop += 1
                continue
            pending_images.append(image.resize((128, 256), Image.BILINEAR))
            pending_keys.append(str(row.tracklet_key))
            seen_crops += 1
            if len(pending_images) >= int(args.batch_size):
                flush()
            if seen_crops and seen_crops % 2000 == 0:
                print(
                    json.dumps(
                        {
                            "processed_crops": int(seen_crops),
                            "tracklets_with_osnet": int(len(osnet_by_key)),
                            "missing_video": int(missing_video),
                            "missing_crop": int(missing_crop),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        flush()
    finally:
        for cap in caps.values():
            cap.release()

    n = len(records)
    osnet_features = np.zeros((n, 512), dtype=np.float32)
    osnet_std_features = np.zeros((n, 512), dtype=np.float32)
    color_dim = 82
    color_features = np.zeros((n, color_dim), dtype=np.float32)
    color_std_features = np.zeros((n, color_dim), dtype=np.float32)
    valid_osnet = np.zeros((n,), dtype=bool)
    valid_color = np.zeros((n,), dtype=bool)
    sample_counts = np.zeros((n,), dtype=np.int16)
    prototype_slots = max(int(args.samples), 1)
    save_prototypes = bool(args.save_prototypes)
    if save_prototypes:
        osnet_prototypes = np.zeros((n, prototype_slots, 512), dtype=np.float32)
        color_prototypes = np.zeros((n, prototype_slots, color_dim), dtype=np.float32)
        valid_osnet_prototypes = np.zeros((n, prototype_slots), dtype=bool)
        valid_color_prototypes = np.zeros((n, prototype_slots), dtype=bool)
    for key, index in key_to_index.items():
        os_feats = osnet_by_key.get(key, [])
        if os_feats:
            stacked = np.stack(os_feats).astype(np.float32)
            feat = np.mean(stacked, axis=0)
            osnet_features[index] = _l2n(feat)
            osnet_std_features[index] = _l2n(np.std(stacked, axis=0))
            valid_osnet[index] = True
            sample_counts[index] = len(os_feats)
            if save_prototypes:
                count = min(int(stacked.shape[0]), prototype_slots)
                osnet_prototypes[index, :count] = _l2n(stacked[:count])
                valid_osnet_prototypes[index, :count] = True
        color_feats = color_by_key.get(key, [])
        if color_feats:
            stacked = np.stack(color_feats).astype(np.float32)
            feat = np.mean(stacked, axis=0)
            color_features[index] = _l2n(feat)
            color_std_features[index] = _l2n(np.std(stacked, axis=0))
            valid_color[index] = True
            if save_prototypes:
                count = min(int(stacked.shape[0]), prototype_slots)
                color_prototypes[index, :count] = _l2n(stacked[:count])
                valid_color_prototypes[index, :count] = True

    metadata = {
        "model": "osnet_x1_0_msmt17+color_hist",
        "model_path": str(args.model_path),
        "tracklet_parquet": str(args.tracklet_parquet),
        "records": records,
        "samples": int(args.samples),
        "crop_margin": float(args.crop_margin),
        "sample_rows": int(len(sampled)),
        "seen_crops": int(seen_crops),
        "missing_video": int(missing_video),
        "missing_crop": int(missing_crop),
        "feature_blocks": {
            "features_osnet": list(osnet_features.shape),
            "features_osnet_std": list(osnet_std_features.shape),
            "features_color": list(color_features.shape),
            "features_color_std": list(color_std_features.shape),
        },
        "saved_prototypes": save_prototypes,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": False,
    }
    payload = {
        "metadata": json.dumps(metadata, sort_keys=True),
        "features_osnet": osnet_features,
        "features_osnet_std": osnet_std_features,
        "features_color": color_features,
        "features_color_std": color_std_features,
        "valid_osnet": valid_osnet,
        "valid_color": valid_color,
        "sample_counts": sample_counts,
    }
    if save_prototypes:
        metadata["feature_blocks"].update(
            {
                "features_osnet_prototypes": list(osnet_prototypes.shape),
                "features_color_prototypes": list(color_prototypes.shape),
            }
        )
        payload["metadata"] = json.dumps(metadata, sort_keys=True)
        payload.update(
            {
                "features_osnet_prototypes": osnet_prototypes,
                "features_color_prototypes": color_prototypes,
                "valid_osnet_prototypes": valid_osnet_prototypes,
                "valid_color_prototypes": valid_color_prototypes,
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    feature_shapes = {
        "features_osnet": list(osnet_features.shape),
        "features_osnet_std": list(osnet_std_features.shape),
        "features_color": list(color_features.shape),
        "features_color_std": list(color_std_features.shape),
    }
    if save_prototypes:
        feature_shapes.update(
            {
                "features_osnet_prototypes": list(osnet_prototypes.shape),
                "features_color_prototypes": list(color_prototypes.shape),
            }
        )
    print(
        json.dumps(
            {
                "out": str(out),
                **feature_shapes,
                "valid_osnet": int(valid_osnet.sum()),
                "valid_color": int(valid_color.sum()),
                "seen_crops": int(seen_crops),
                "missing_video": int(missing_video),
                "missing_crop": int(missing_crop),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
