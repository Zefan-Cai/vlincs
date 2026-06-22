#!/usr/bin/env python
"""Prepare BoTSORT sample tracklets for no-anchor sample sweeps.

The BoTSORT sample root has no identity column.  This helper attaches
``tracklet_majority_gt_id`` only for evaluation by frame-level IoU matching
against DS1 reference boxes.  Identity labels are never written into feature
blocks and should not be used as anchors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from vlincs_gallery.eval.score import load_ds1_gt_by_video


def _base_video(video_key: object) -> str:
    return str(video_key).split("__", 1)[0]


def _l2n(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)).astype(np.float32)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1.0e-6)).astype(np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1.0e-9)


def _cache_key_index(path: Path) -> tuple[dict[str, int], np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    keys = [str(key) for key in data["keys"].tolist()]
    features = data["features"].astype(np.float32)
    meta = {
        "crop_cache": str(path),
        "crop_cache_rows": int(len(keys)),
        "crop_feature_dim": int(features.shape[1]),
        "crop_samples": int(np.asarray(data.get("samples", [0])).reshape(-1)[0]) if "samples" in data.files else None,
        "crop_margin": float(np.asarray(data.get("margin", [0.0])).reshape(-1)[0]) if "margin" in data.files else None,
    }
    return {key: idx for idx, key in enumerate(keys)}, features, meta


def _read_tracklets(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("tracklets.parquet")):
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no tracklets.parquet under {root}")
    df = pd.concat(frames, ignore_index=True)
    required = {"video_key", "local_track_id", "tracklet_key", "frame_idx", "x1", "y1", "x2", "y2", "score", "coco_cls"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")
    df["eval_video_key"] = df["video_key"].map(_base_video)
    return df


def _attach_eval_labels(df: pd.DataFrame, *, iou_thr: float, min_matches: int) -> tuple[pd.DataFrame, dict[str, object]]:
    gt_by_video = load_ds1_gt_by_video()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals = Counter(str(key) for key in df["tracklet_key"].tolist())
    matched_rows = 0
    total_rows = 0

    for video, pred in df.groupby("eval_video_key", sort=True):
        gt = gt_by_video.get(str(video))
        if gt is None:
            continue
        pred_sorted = pred.sort_values("frame_idx", kind="mergesort")
        gt_sorted = gt.sort_values("frame", kind="mergesort")
        p_frames = pred_sorted["frame_idx"].to_numpy(np.int64)
        p_keys = pred_sorted["tracklet_key"].astype(str).to_numpy()
        p_boxes = pred_sorted[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        g_frames = gt_sorted["frame"].to_numpy(np.int64)
        g_ids = gt_sorted["id"].astype(str).to_numpy()
        g_boxes = gt_sorted[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        total_rows += int(len(pred_sorted))

        pi = 0
        while pi < len(p_frames):
            frame = int(p_frames[pi])
            pj = int(np.searchsorted(p_frames, frame, side="right"))
            gi = int(np.searchsorted(g_frames, frame, side="left"))
            if gi >= len(g_frames) or int(g_frames[gi]) != frame:
                pi = pj
                continue
            gj = int(np.searchsorted(g_frames, frame, side="right"))
            ious = _iou_matrix(p_boxes[pi:pj], g_boxes[gi:gj])
            best = ious.argmax(axis=1) if ious.size else np.zeros((pj - pi,), dtype=np.int64)
            best_iou = ious[np.arange(pj - pi), best] if ious.size else np.zeros((pj - pi,), dtype=np.float32)
            gt_ids = g_ids[gi:gj]
            for key, gt_pos, score in zip(p_keys[pi:pj], best, best_iou):
                if float(score) >= float(iou_thr):
                    counts[str(key)][str(gt_ids[int(gt_pos)])] += 1
                    matched_rows += 1
            pi = pj

    rows = []
    rejected = 0
    purities = []
    for key, total in totals.items():
        key_counts = counts.get(str(key), Counter())
        if not key_counts:
            rows.append((key, "UNKNOWN", 0.0, 0))
            rejected += 1
            continue
        gid, n = key_counts.most_common(1)[0]
        purity = float(n) / max(float(total), 1.0)
        if int(n) < int(min_matches):
            rows.append((key, "UNKNOWN", 0.0, int(n)))
            rejected += 1
        else:
            rows.append((key, str(gid), purity, int(n)))
            purities.append(purity)
    labels = pd.DataFrame(
        rows,
        columns=["tracklet_key", "tracklet_majority_gt_id", "tracklet_majority_gt_fraction", "tracklet_gt_match_rows"],
    )
    out = df.merge(labels, on="tracklet_key", how="left")
    return out, {
        "eval_iou_thr": float(iou_thr),
        "eval_min_matches": int(min_matches),
        "eval_total_rows": int(total_rows),
        "eval_matched_rows": int(matched_rows),
        "eval_labeled_tracklets": int(len(purities)),
        "eval_rejected_tracklets": int(rejected),
        "eval_purity_mean": round(float(np.mean(purities)) if purities else 0.0, 6),
        "eval_purity_p10": round(float(np.quantile(purities, 0.10)) if purities else 0.0, 6),
        "uses_gt_identity_for_features": False,
        "uses_gt_identity_for_evaluation_columns": True,
    }


def _write_features(df: pd.DataFrame, crop_cache: Path, out_path: Path, metadata_extra: dict[str, object]) -> dict[str, object]:
    crop_index, crop_features, crop_meta = _cache_key_index(crop_cache)
    ordered = (
        df.groupby("tracklet_key", sort=False)
        .agg(
            video_key=("video_key", "first"),
            eval_video_key=("eval_video_key", "first"),
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
        records.append(
            {
                "index": int(idx),
                "tracklet_key": key,
                "video_key": str(row.video_key),
                "eval_video_key": str(row.eval_video_key),
                "local_track_id": int(row.local_track_id),
            }
        )
        cache_pos = crop_index.get(key)
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

    np.savez_compressed(
        out_path,
        features_crop=_l2n(np.vstack(crop_rows).astype(np.float32)),
        features_bbox=_l2n(_standardize(np.asarray(bbox_rows, dtype=np.float32))),
        features_trajectory=_standardize(np.asarray(traj_rows, dtype=np.float32)).astype(np.float32),
        metadata=np.array(
            json.dumps(
                {
                    "schema_version": 1,
                    "model": "botsort_crop_cache_plus_bbox_stats",
                    "records": records,
                    "missing_crop_features": int(missing_crop),
                    **crop_meta,
                    **metadata_extra,
                },
                sort_keys=True,
            ),
            dtype=object,
        ),
    )
    return {"feature_npz": str(out_path), "feature_tracklets": int(len(records)), "missing_crop_features": int(missing_crop)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tracklets-root",
        default="/mnt/localssd/vlincs_reid_data/Box/VLINCS_Performer/sample/tracklets/sample/botsort_baseline/tracklets",
    )
    ap.add_argument("--crop-cache", default="/mnt/localssd/vlincs_reid_runs/botsort_crop3_cache_v1.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-iou-thr", type=float, default=0.50)
    ap.add_argument("--eval-min-matches", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "botsort_eval.parquet"
    feature_out = out_dir / "features_botsort_crop_bbox.npz"
    summary_out = out_dir / "prepare_summary.json"

    df = _read_tracklets(Path(args.tracklets_root))
    labeled, eval_meta = _attach_eval_labels(df, iou_thr=float(args.eval_iou_thr), min_matches=int(args.eval_min_matches))
    labeled.to_parquet(parquet_out, index=False)
    feature_meta = _write_features(
        labeled,
        Path(args.crop_cache),
        feature_out,
        {
            "tracklets_root": str(args.tracklets_root),
            "parquet_out": str(parquet_out),
            **eval_meta,
        },
    )
    summary = {
        "parquet": str(parquet_out),
        "feature_npz": str(feature_out),
        "rows": int(len(labeled)),
        "tracklets": int(labeled["tracklet_key"].nunique()),
        "unique_eval_gt_ids": int(labeled["tracklet_majority_gt_id"].astype(str).nunique()),
        **eval_meta,
        **feature_meta,
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
