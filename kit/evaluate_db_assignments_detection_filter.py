#!/usr/bin/env python
"""Evaluate DB assignment CSVs with detection-level no-anchor filters.

The existing assignment maps tracklet ``seq`` to ``predicted_global_id``.  This
script keeps that identity mapping fixed and changes only which detection rows
are delivered to the canonical scorer.  It is meant to test whether e2e IDF1 is
limited by low-confidence detection false positives rather than identity
resolution alone.

Ground truth is used only by the final scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import _export_zip, _metric_dict
from kit.no_anchor_resolve_sweep import _connect, _load_predictions
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _load_assignment_csv(path: str, pred_col: str) -> tuple[dict[int, int], Counter[str]]:
    pred: dict[int, int] = {}
    statuses: Counter[str] = Counter()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"seq", pred_col} - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        status_col = "resolution_status" if "resolution_status" in fields else "decision_status" if "decision_status" in fields else ""
        for row in reader:
            seq = int(float(row["seq"]))
            pred[seq] = int(float(row[pred_col]))
            if status_col:
                statuses[str(row.get(status_col, ""))] += 1
    return pred, statuses


def _parse_kv_float_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.rpartition(":")
        if not sep or not key:
            raise ValueError(f"bad map entry {part!r}; expected key:value")
        out[key] = float(value)
    return out


def _parse_config(text: str) -> dict[str, object]:
    parts = [part.strip() for part in str(text).split(";") if part.strip()]
    if not parts:
        raise ValueError("empty --config")
    cfg: dict[str, object] = {
        "name": parts[0],
        "global_conf": 0.0,
        "global_area": 0.0,
        "video_conf": {},
        "video_area": {},
    }
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"bad config segment {part!r}; expected key=value")
        key = key.strip()
        value = value.strip()
        if key in {"global_conf", "global_area"}:
            cfg[key] = float(value)
        elif key == "video_conf":
            cfg[key] = _parse_kv_float_map(value)
        elif key == "video_area":
            cfg[key] = _parse_kv_float_map(value)
        else:
            raise ValueError(f"unknown config key {key!r}")
    return cfg


def _build_comp(
    pred_by_video: dict[str, pd.DataFrame],
    pred_by_seq: dict[int, int],
    cfg: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    comp: dict[str, pd.DataFrame] = {}
    video_conf = dict(cfg.get("video_conf") or {})
    video_area = dict(cfg.get("video_area") or {})
    global_conf = float(cfg.get("global_conf", 0.0))
    global_area = float(cfg.get("global_area", 0.0))
    by_video = {}
    for video, pred in pred_by_video.items():
        out = pred.copy()
        out = out[out["seq"].map(lambda seq: int(seq) in pred_by_seq)].copy()
        input_rows = int(len(out))
        input_tracklets = int(out["seq"].nunique()) if len(out) else 0
        if out.empty:
            empty = out[["frame", "seq", "x1", "y1", "x2", "y2", "object_type", "confidence"]].drop(columns=["seq"])
            empty["id"] = pd.Series(dtype=np.int64)
            comp[video] = empty[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]]
            by_video[video] = {"input_rows": 0, "rows": 0, "input_tracklets": 0, "tracklets": 0}
            continue
        conf_thr = max(global_conf, float(video_conf.get(video, 0.0)))
        area_thr = max(global_area, float(video_area.get(video, 0.0)))
        area = (out["x2"].astype(float) - out["x1"].astype(float)).clip(lower=0.0) * (
            out["y2"].astype(float) - out["y1"].astype(float)
        ).clip(lower=0.0)
        keep = (out["confidence"].astype(float) >= conf_thr) & (area >= area_thr)
        out = out.loc[keep].copy()
        if out.empty:
            out["id"] = pd.Series(dtype=np.int64)
        else:
            out["id"] = [int(pred_by_seq[int(seq)]) for seq in out["seq"]]
        comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        by_video[video] = {
            "input_rows": input_rows,
            "rows": int(len(out)),
            "dropped_rows": int(input_rows - len(out)),
            "input_tracklets": input_tracklets,
            "tracklets": int(out["seq"].nunique()) if len(out) else 0,
            "conf_thr": round(float(conf_thr), 6),
            "area_thr": round(float(area_thr), 3),
        }
    info = {
        "config_name": str(cfg["name"]),
        "global_conf": round(float(global_conf), 6),
        "global_area": round(float(global_area), 3),
        "video_conf": {str(k): round(float(v), 6) for k, v in sorted(video_conf.items())},
        "video_area": {str(k): round(float(v), 3) for k, v in sorted(video_area.items())},
        "input_rows": int(sum(item["input_rows"] for item in by_video.values())),
        "rows": int(sum(item["rows"] for item in by_video.values())),
        "dropped_rows": int(sum(item.get("dropped_rows", 0) for item in by_video.values())),
        "by_video_rows": by_video,
        "predicted_ids": int(sum(df["id"].nunique() for df in comp.values() if len(df))),
    }
    return comp, info


def _score(comp: dict[str, pd.DataFrame]) -> dict[str, object]:
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {
        **_metric_dict(metrics),
        "videos_scored": keys,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--config", action="append", required=True, help="name;global_conf=0.2;video_conf=video:thr,...")
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip-out", default="", help="optional zip for the best scored config")
    args = ap.parse_args()

    pred_by_seq, statuses = _load_assignment_csv(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    pred_by_video = _load_predictions(con)
    rows = []
    comp_by_name: dict[str, dict[str, pd.DataFrame]] = {}
    for text in args.config:
        cfg = _parse_config(text)
        comp, info = _build_comp(pred_by_video, pred_by_seq, cfg)
        comp_by_name[str(info["config_name"])] = comp
        scored = _score(comp)
        row = {
            **info,
            **scored,
            "assignment_csv": str(args.assignment_csv),
            "assigned_tracklets": int(len(pred_by_seq)),
            "assignment_status_counts": dict(statuses),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
        }
        rows.append(row)
        print(json.dumps({"stage": "scored", **{k: row[k] for k in ["config_name", "idf1", "hota", "assa", "detpr", "detre", "rows", "dropped_rows"]}}, sort_keys=True), flush=True)
    rows.sort(key=lambda row: float(row["idf1"]), reverse=True)
    if args.zip_out and rows:
        rows[0].update(_export_zip(comp_by_name[str(rows[0]["config_name"])], args.zip_out))
    result = {
        "assignment_csv": str(args.assignment_csv),
        "configs": [_parse_config(text) for text in args.config],
        "rows": rows,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "selection_uses_gt_metric": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "done", "json": str(out), "best": {k: rows[0].get(k) for k in ["config_name", "idf1", "hota", "assa", "detpr", "detre", "rows", "dropped_rows"]}}, sort_keys=True))


if __name__ == "__main__":
    main()
