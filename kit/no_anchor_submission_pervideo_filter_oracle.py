#!/usr/bin/env python
"""Per-video detection-filter oracle for a no-anchor submission zip.

This is a research diagnostic, not a production policy learner.  It keeps the
input global IDs fixed, sweeps simple detection-row filters independently per
video, and uses ground truth only for scoring and oracle selection.  The script
checkpoints after every scored row so long Pluto/SSH sessions can be resumed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from kit.evaluate_sample_assignments_full import _export_zip, _metric_dict
    from kit.submission_video_switch import _load_zip
except ModuleNotFoundError:
    from evaluate_sample_assignments_full import _export_zip, _metric_dict
    from submission_video_switch import _load_zip

from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_strings(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def _row_key(video: str, conf_q: float, area_q: float) -> str:
    return f"{video}|conf_q={conf_q:.6g}|area_q={area_q:.6g}"


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text())
    rows = list(data.get("rows", []))
    by_key = {str(row.get("row_key")): row for row in rows if row.get("row_key")}
    return rows, by_key


def _write_checkpoint(
    path: Path,
    *,
    submission_zip: str,
    conf_quantiles: list[float],
    area_quantiles: list[float],
    only_videos: set[str],
    rows: list[dict[str, Any]],
    combined: dict[str, Any] | None = None,
) -> None:
    best_by_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video = str(row["video"])
        if video not in best_by_video or float(row["idf1"]) > float(best_by_video[video]["idf1"]):
            best_by_video[video] = row
    result = {
        "submission_zip": str(submission_zip),
        "conf_quantiles": conf_quantiles,
        "area_quantiles": area_quantiles,
        "only_videos": sorted(only_videos),
        "rows": rows,
        "best_by_video": best_by_video,
        "suggested_video_conf": {
            video: row["conf_thr"] for video, row in sorted(best_by_video.items()) if float(row["conf_q"]) > 0.0
        },
        "suggested_video_area": {
            video: row["area_thr"] for video, row in sorted(best_by_video.items()) if float(row["area_q"]) > 0.0
        },
        "combined": combined,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "selection_uses_gt_metric": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _filter_frame(df: pd.DataFrame, conf_thr: float, area_thr: float) -> pd.DataFrame:
    work = df.copy()
    area = (work["x2"].astype(float) - work["x1"].astype(float)).clip(lower=0.0) * (
        work["y2"].astype(float) - work["y1"].astype(float)
    ).clip(lower=0.0)
    keep = (work["confidence"].astype(float) >= float(conf_thr)) & (area >= float(area_thr))
    return work.loc[keep].copy()


def _score_video(video: str, pred: pd.DataFrame, gt_by_video: dict[str, pd.DataFrame]) -> dict[str, Any]:
    metrics = evaluate({video: gt_by_video[video]}, {video: pred}, dense=False, n_workers=1)
    return {key: value for key, value in _metric_dict(metrics).items() if key != "per_video"}


def _score_full(comp: dict[str, pd.DataFrame], gt_by_video: dict[str, pd.DataFrame]) -> dict[str, Any]:
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {**_metric_dict(metrics), "videos_scored": keys}


def _build_oracle_comp(
    source: dict[str, pd.DataFrame],
    best_by_video: dict[str, dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    comp: dict[str, pd.DataFrame] = {}
    by_video_rows: dict[str, Any] = {}
    for video, df in sorted(source.items()):
        best = best_by_video.get(video, {"conf_thr": 0.0, "area_thr": 0.0, "conf_q": 0.0, "area_q": 0.0})
        filtered = _filter_frame(df, float(best["conf_thr"]), float(best["area_thr"]))
        comp[video] = filtered
        by_video_rows[video] = {
            "conf_q": float(best.get("conf_q", 0.0)),
            "area_q": float(best.get("area_q", 0.0)),
            "conf_thr": round(float(best.get("conf_thr", 0.0)), 6),
            "area_thr": round(float(best.get("area_thr", 0.0)), 3),
            "input_rows": int(len(df)),
            "rows": int(len(filtered)),
            "dropped_rows": int(len(df) - len(filtered)),
            "ids": int(filtered["id"].nunique()) if len(filtered) else 0,
        }
    info = {
        "by_video_rows": by_video_rows,
        "input_rows": int(sum(len(df) for df in source.values())),
        "rows": int(sum(len(df) for df in comp.values())),
        "dropped_rows": int(sum(len(source[video]) - len(comp[video]) for video in comp)),
        "predicted_ids": int(sum(df["id"].nunique() for df in comp.values() if len(df))),
    }
    return comp, info


def _export_comp_zip(comp: dict[str, pd.DataFrame], out_zip: str) -> dict[str, Any]:
    # Reuse the canonical exporter by matching its expected column shape.
    normalized = {
        video: df[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        for video, df in comp.items()
    }
    return _export_zip(normalized, out_zip)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission-zip", required=True)
    ap.add_argument("--conf-quantiles", default="0,0.01,0.02,0.03,0.05,0.08,0.10")
    ap.add_argument("--area-quantiles", default="0")
    ap.add_argument("--only-videos", default="", help="optional comma-separated videos to sweep; others stay unchanged")
    ap.add_argument("--json", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-oracle", action="store_true")
    ap.add_argument("--zip-out", default="")
    args = ap.parse_args()

    out_path = Path(args.json)
    conf_quantiles = _parse_floats(args.conf_quantiles)
    area_quantiles = _parse_floats(args.area_quantiles)
    only_videos = _parse_strings(args.only_videos)
    rows, existing = _load_existing(out_path) if args.resume else ([], {})

    source = _load_zip(Path(args.submission_zip))
    gt_by_video = load_ds1_gt_by_video()
    videos = sorted(set(source).intersection(gt_by_video))
    if only_videos:
        videos = [video for video in videos if video in only_videos]
    if not videos:
        raise RuntimeError("no overlap between submission videos and DS1 ground truth")

    for video in videos:
        df = source[video]
        conf_values = df["confidence"].astype(float)
        area_values = (df["x2"].astype(float) - df["x1"].astype(float)).clip(lower=0.0) * (
            df["y2"].astype(float) - df["y1"].astype(float)
        ).clip(lower=0.0)
        for conf_q in conf_quantiles:
            conf_thr = 0.0 if float(conf_q) <= 0.0 else float(conf_values.quantile(float(conf_q)))
            for area_q in area_quantiles:
                area_thr = 0.0 if float(area_q) <= 0.0 else float(area_values.quantile(float(area_q)))
                key = _row_key(video, float(conf_q), float(area_q))
                if key in existing:
                    continue
                filtered = _filter_frame(df, conf_thr, area_thr)
                scored = _score_video(video, filtered, gt_by_video)
                row = {
                    "row_key": key,
                    "video": video,
                    "conf_q": float(conf_q),
                    "area_q": float(area_q),
                    "conf_thr": round(float(conf_thr), 6),
                    "area_thr": round(float(area_thr), 3),
                    "input_rows": int(len(df)),
                    "rows": int(len(filtered)),
                    "dropped_rows": int(len(df) - len(filtered)),
                    "ids": int(filtered["id"].nunique()) if len(filtered) else 0,
                    **scored,
                }
                rows.append(row)
                existing[key] = row
                print(json.dumps({"stage": "scored", **row}, sort_keys=True), flush=True)
                _write_checkpoint(
                    out_path,
                    submission_zip=str(args.submission_zip),
                    conf_quantiles=conf_quantiles,
                    area_quantiles=area_quantiles,
                    only_videos=only_videos,
                    rows=rows,
                    combined=None,
                )

    combined = None
    if args.full_oracle:
        best_by_video: dict[str, dict[str, Any]] = {}
        for row in rows:
            video = str(row["video"])
            if video not in best_by_video or float(row["idf1"]) > float(best_by_video[video]["idf1"]):
                best_by_video[video] = row
        comp, info = _build_oracle_comp(source, best_by_video)
        scored = _score_full(comp, gt_by_video)
        combined = {
            "policy_name": "per_video_oracle",
            **info,
            **scored,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
            "selection_uses_gt_metric": True,
        }
        if args.zip_out:
            combined.update(_export_comp_zip(comp, args.zip_out))
        print(json.dumps({"stage": "full_oracle", **combined}, sort_keys=True), flush=True)

    _write_checkpoint(
        out_path,
        submission_zip=str(args.submission_zip),
        conf_quantiles=conf_quantiles,
        area_quantiles=area_quantiles,
        only_videos=only_videos,
        rows=rows,
        combined=combined,
    )
    print(json.dumps({"stage": "done", "json": str(out_path), "rows": len(rows), "combined": combined}, sort_keys=True))


if __name__ == "__main__":
    main()
