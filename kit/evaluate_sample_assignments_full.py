#!/usr/bin/env python
"""Evaluate sample/parquet global-ID assignments with the canonical scorer.

The no-anchor sample sweeps report fast tracklet/row proxy metrics.  This
bridge turns their assignment CSV back into a submission-like component and
runs the VLINCS DS1 HOTA/IDF1 scorer.  Ground truth is loaded only for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
KIT_ROOT = Path(__file__).resolve().parent
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from submit import _box_hash
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _load_parquets(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("at least one --tracklet-parquet is required")
    df = pd.concat(frames, ignore_index=True)
    rename = {
        "video_key": "video",
        "frame_idx": "frame",
        "score": "confidence",
    }
    df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
    required = {"video", "frame", "tracklet_key", "x1", "y1", "x2", "y2"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"tracklet parquet is missing columns: {missing}")
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    if "object_type" not in df.columns:
        df["object_type"] = 0
    return df


def _load_assignments(path: str) -> pd.DataFrame:
    assign = pd.read_csv(path)
    required = {"tracklet_key", "predicted_global_id"}
    missing = sorted(required.difference(assign.columns))
    if missing:
        raise ValueError(f"assignment CSV is missing columns: {missing}")
    keep_cols = ["tracklet_key", "predicted_global_id"]
    if "video" in assign.columns:
        keep_cols.insert(0, "video")
    optional = [
        "prediction_confidence",
        "decision_status",
        "resolution_status",
        "component_size",
        "component_label",
    ]
    keep_cols.extend([col for col in optional if col in assign.columns])
    assign = assign[keep_cols].copy()
    assign["tracklet_key"] = assign["tracklet_key"].astype(str)
    if "video" in assign.columns:
        assign["video"] = assign["video"].astype(str)
        assign = assign.drop_duplicates(["video", "tracklet_key"], keep="first")
    else:
        assign = assign.drop_duplicates(["tracklet_key"], keep="first")
    assign["predicted_global_id"] = assign["predicted_global_id"].astype(np.int64)
    return assign


def _merge_predictions(df: pd.DataFrame, assign: pd.DataFrame, fallback: str) -> tuple[pd.DataFrame, dict[str, object]]:
    work = df.copy()
    work["tracklet_key"] = work["tracklet_key"].astype(str)
    work["video"] = work["video"].astype(str)
    merge_cols = ["video", "tracklet_key"] if "video" in assign.columns else ["tracklet_key"]
    work = work.merge(assign, on=merge_cols, how="left", suffixes=("", "_assign"))
    missing_mask = work["predicted_global_id"].isna()
    missing_tracklets = int(work.loc[missing_mask, "tracklet_key"].nunique())
    if missing_mask.any():
        if fallback == "drop":
            work = work.loc[~missing_mask].copy()
        elif fallback == "singleton":
            unique_keys = sorted(work.loc[missing_mask, "tracklet_key"].astype(str).unique())
            fallback_ids = {key: 99_000_000 + idx for idx, key in enumerate(unique_keys)}
            work.loc[missing_mask, "predicted_global_id"] = work.loc[missing_mask, "tracklet_key"].map(fallback_ids)
        else:
            raise ValueError(f"unknown fallback mode: {fallback}")
    work["predicted_global_id"] = work["predicted_global_id"].astype(np.int64)
    info = {
        "fallback": fallback,
        "input_rows": int(len(df)),
        "scored_rows": int(len(work)),
        "input_tracklets": int(df["tracklet_key"].nunique()),
        "assigned_tracklets": int(assign["tracklet_key"].nunique()),
        "missing_assignment_tracklets": missing_tracklets,
        "predicted_ids": int(work["predicted_global_id"].nunique()),
    }
    return work, info


def _build_comp(work: pd.DataFrame) -> dict[str, pd.DataFrame]:
    comp = {}
    for video, group in work.groupby("video", sort=True):
        out = group[["frame", "predicted_global_id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        out = out.rename(columns={"predicted_global_id": "id"})
        comp[str(video)] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]]
    return comp


def _metric_dict(metrics) -> dict[str, object]:
    return {
        "idf1": round(float(metrics.idf1), 6),
        "hota": round(float(metrics.hota), 6),
        "assa": round(float(metrics.assa), 6),
        "deta": round(float(metrics.deta), 6),
        "detre": round(float(metrics.detre), 6),
        "detpr": round(float(metrics.detpr), 6),
        "unmatched_fp": int(metrics.unmatched_fp),
        "per_video": {
            key: {metric: round(float(value), 6) for metric, value in vals.items()}
            for key, vals in sorted(metrics.per_video.items())
        },
    }


def _export_zip(comp: dict[str, pd.DataFrame], out_zip: str) -> dict[str, object]:
    out_path = Path(out_zip)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="vlincs_sample_submit_"))
    written = []
    for video, df in comp.items():
        out = df.copy()
        out["frame"] = out["frame"].astype("uint32")
        out["id"] = out["id"].astype("uint32")
        for col in ("x1", "y1", "x2", "y2"):
            out[col] = out[col].clip(lower=0).astype("uint32")
        out["object_type"] = out["object_type"].astype("uint8")
        out["confidence"] = out["confidence"].astype("float32")
        out["box_hash"] = [_box_hash(r.x1, r.y1, r.x2, r.y2) for r in out.itertuples()]
        for col in ("lat", "long", "alt"):
            out[col] = np.float64("nan")
        path = tmp / f"{video}.parquet"
        out[
            [
                "frame",
                "id",
                "x1",
                "y1",
                "x2",
                "y2",
                "box_hash",
                "object_type",
                "confidence",
                "lat",
                "long",
                "alt",
            ]
        ].to_parquet(path)
        written.append(path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, arcname=path.name)
    return {"zip_out": str(out_path), "zip_files": int(len(written))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklet-parquet", nargs="+", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--fallback", choices=["drop", "singleton"], default="singleton")
    parser.add_argument("--json", required=True)
    parser.add_argument("--zip-out", default="")
    parser.add_argument(
        "--allow-no-gt-export",
        action="store_true",
        help="write merge stats and optional zip even when local DS1 GT is unavailable",
    )
    args = parser.parse_args()

    df = _load_parquets(args.tracklet_parquet)
    assignments = _load_assignments(args.assignments)
    work, merge_info = _merge_predictions(df, assignments, args.fallback)
    comp = _build_comp(work)
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    if not keys:
        if args.allow_no_gt_export:
            out = {
                **merge_info,
                "videos_scored": [],
                "gt_available": False,
                "gt_message": "no overlap between predictions and local DS1 ground truth",
                "uses_anchors": False,
                "uses_gt_for_training_or_anchors": False,
                "uses_gt_for_evaluation_only": False,
                "assignment_status_counts": dict(
                    Counter(
                        assignments.get(
                            "resolution_status",
                            assignments.get("decision_status", pd.Series(dtype=str)),
                        ).astype(str)
                    )
                ),
            }
            if args.zip_out:
                out.update(_export_zip(comp, args.zip_out))
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
            print(json.dumps(out, sort_keys=True))
            return
        raise RuntimeError("no overlap between sample videos and DS1 ground truth")
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    out = {
        **merge_info,
        **_metric_dict(metrics),
        "videos_scored": keys,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "assignment_status_counts": dict(Counter(assignments.get("resolution_status", assignments.get("decision_status", pd.Series(dtype=str))).astype(str))),
    }
    if args.zip_out:
        out.update(_export_zip(comp, args.zip_out))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
