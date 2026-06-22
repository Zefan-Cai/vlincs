#!/usr/bin/env python
"""Prepare from_ground_truth tracklet parquets for no-anchor sample sweeps.

The source parquet contains ``gt_id`` because it is generated from reference
boxes.  This helper copies that identity only into evaluation columns expected
by ``no_anchor_sample_parquet_sweep.py`` and builds feature NPZ blocks from
crop-cache and bbox/trajectory statistics without using identity labels as
features.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_CACHE_KEY_RE = re.compile(r"^(.*)::tracklet:(\d+)$")


def _l2n(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1.0e-6)).astype(np.float32)


def _cache_key_index(path: Path) -> tuple[dict[tuple[str, int], int], np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    keys = [str(key) for key in data["keys"].tolist()]
    features = data["features"].astype(np.float32)
    by_video_local: dict[tuple[str, int], int] = {}
    for idx, key in enumerate(keys):
        match = _CACHE_KEY_RE.match(key)
        if not match:
            continue
        by_video_local[(match.group(1), int(match.group(2)))] = int(idx)
    meta = {
        "crop_cache": str(path),
        "crop_cache_rows": int(len(keys)),
        "crop_feature_dim": int(features.shape[1]),
        "crop_samples": int(np.asarray(data.get("samples", [0])).reshape(-1)[0]) if "samples" in data.files else None,
        "crop_margin": float(np.asarray(data.get("margin", [0.0])).reshape(-1)[0]) if "margin" in data.files else None,
    }
    return by_video_local, features, meta


def _majority_gt(group: pd.DataFrame) -> tuple[str, float]:
    counts = group["gt_id"].astype(str).value_counts(dropna=True)
    if counts.empty:
        return "UNKNOWN", 0.0
    gid = str(counts.index[0])
    frac = float(counts.iloc[0] / max(len(group), 1))
    return gid, frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tracklets-root",
        default="/mnt/localssd/vlincs_reid_data/Box/VLINCS_Performer/sample/tracklets/sample/from_ground_truth/tracklets",
    )
    ap.add_argument("--crop-cache", default="/mnt/localssd/vlincs_reid_runs/gtbox_crop3_cache_v1.npz")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.tracklets_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "gtbox_eval.parquet"
    feature_out = out_dir / "features_gtbox_crop_bbox.npz"
    summary_out = out_dir / "prepare_summary.json"

    frames = []
    for path in sorted(root.rglob("tracklets.parquet")):
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no tracklets.parquet under {root}")
    df = pd.concat(frames, ignore_index=True)
    required = {"video_key", "local_track_id", "tracklet_key", "frame_idx", "x1", "y1", "x2", "y2", "score", "coco_cls", "gt_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")

    majority = (
        df.groupby("tracklet_key", sort=False)
        .apply(lambda group: pd.Series(_majority_gt(group), index=["tracklet_majority_gt_id", "tracklet_majority_gt_fraction"]))
        .reset_index()
    )
    df = df.merge(majority, on="tracklet_key", how="left")
    df.to_parquet(parquet_out, index=False)

    crop_index, crop_features, crop_meta = _cache_key_index(Path(args.crop_cache))
    ordered = (
        df.groupby("tracklet_key", sort=False)
        .agg(
            video_key=("video_key", "first"),
            local_track_id=("local_track_id", "first"),
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            n_rows=("frame_idx", "size"),
            mean_x1=("x1", "mean"),
            mean_y1=("y1", "mean"),
            mean_x2=("x2", "mean"),
            mean_y2=("y2", "mean"),
            med_x1=("x1", "median"),
            med_y1=("y1", "median"),
            med_x2=("x2", "median"),
            med_y2=("y2", "median"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["video_key", "start_frame", "end_frame", "tracklet_key"], kind="mergesort")
        .reset_index(drop=True)
    )
    records = []
    crop_rows = []
    bbox_rows = []
    traj_rows = []
    missing_crop = 0
    for idx, row in enumerate(ordered.itertuples(index=False)):
        key = str(row.tracklet_key)
        video = str(row.video_key)
        local_id = int(row.local_track_id)
        records.append({"index": int(idx), "tracklet_key": key, "video_key": video, "local_track_id": local_id})
        cache_pos = crop_index.get((video, local_id))
        if cache_pos is None:
            missing_crop += 1
            crop_rows.append(np.zeros((crop_features.shape[1],), dtype=np.float32))
        else:
            crop_rows.append(crop_features[int(cache_pos)])
        w = max(float(row.mean_x2) - float(row.mean_x1), 1.0)
        h = max(float(row.mean_y2) - float(row.mean_y1), 1.0)
        cx = (float(row.mean_x1) + float(row.mean_x2)) * 0.5
        cy = (float(row.mean_y1) + float(row.mean_y2)) * 0.5
        duration = max(int(row.end_frame) - int(row.start_frame) + 1, 1)
        bbox_rows.append(
            [
                np.log1p(float(row.n_rows)),
                np.log1p(float(duration)),
                cx,
                cy,
                w,
                h,
                np.log1p(w * h),
                h / max(w, 1.0),
                float(row.mean_score),
            ]
        )
        traj_rows.append(
            [
                float(row.start_frame),
                float(row.end_frame),
                float(duration),
                cx,
                cy,
                float(row.med_x1),
                float(row.med_y1),
                float(row.med_x2),
                float(row.med_y2),
            ]
        )

    features_crop = _l2n(np.vstack(crop_rows).astype(np.float32))
    features_bbox = _l2n(_standardize(np.asarray(bbox_rows, dtype=np.float32)))
    features_trajectory = _standardize(np.asarray(traj_rows, dtype=np.float32))
    metadata = {
        "schema_version": 1,
        "model": "gtbox_crop_cache_plus_bbox_stats",
        "tracklets_root": str(root),
        "records": records,
        "uses_gt_identity_for_features": False,
        **crop_meta,
        "missing_crop_features": int(missing_crop),
    }
    np.savez_compressed(
        feature_out,
        features_crop=features_crop,
        features_bbox=features_bbox,
        features_trajectory=features_trajectory.astype(np.float32),
        metadata=np.array(json.dumps(metadata, sort_keys=True), dtype=object),
    )
    summary = {
        "parquet": str(parquet_out),
        "feature_npz": str(feature_out),
        "rows": int(len(df)),
        "tracklets": int(len(ordered)),
        "unique_gt_ids_eval_only": int(df["gt_id"].astype(str).nunique()),
        "missing_crop_features": int(missing_crop),
        "uses_gt_identity_for_features": False,
        "uses_gt_identity_for_evaluation_columns": True,
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
