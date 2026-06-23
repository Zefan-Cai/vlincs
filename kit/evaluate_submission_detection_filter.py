#!/usr/bin/env python
"""Evaluate no-anchor submission zips with detection-level filters."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, KIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kit.evaluate_sample_assignments_full import _export_zip, _metric_dict
from kit.submission_video_switch import _load_zip
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


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


def _resolve_config_text(text: str) -> tuple[str, str]:
    raw = str(text)
    if raw.startswith("@"):
        path = Path(raw[1:])
        return path.read_text(encoding="utf-8").strip(), raw
    if raw.startswith("b64:"):
        decoded = base64.b64decode(raw[4:]).decode("utf-8").strip()
        return decoded, "b64:<redacted>"
    return raw, raw


def _filter_comp(comp: dict[str, pd.DataFrame], cfg: dict[str, object]) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    out_comp: dict[str, pd.DataFrame] = {}
    video_conf = dict(cfg.get("video_conf") or {})
    video_area = dict(cfg.get("video_area") or {})
    global_conf = float(cfg.get("global_conf", 0.0))
    global_area = float(cfg.get("global_area", 0.0))
    by_video = {}
    for video, df in sorted(comp.items()):
        work = df.copy()
        input_rows = int(len(work))
        conf_thr = max(global_conf, float(video_conf.get(video, 0.0)))
        area_thr = max(global_area, float(video_area.get(video, 0.0)))
        area = (work["x2"].astype(float) - work["x1"].astype(float)).clip(lower=0.0) * (
            work["y2"].astype(float) - work["y1"].astype(float)
        ).clip(lower=0.0)
        keep = (work["confidence"].astype(float) >= conf_thr) & (area >= area_thr)
        work = work.loc[keep].copy()
        out_comp[video] = work
        by_video[video] = {
            "input_rows": input_rows,
            "rows": int(len(work)),
            "dropped_rows": int(input_rows - len(work)),
            "input_ids": int(df["id"].nunique()) if len(df) else 0,
            "ids": int(work["id"].nunique()) if len(work) else 0,
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
        "dropped_rows": int(sum(item["dropped_rows"] for item in by_video.values())),
        "predicted_ids": int(sum(df["id"].nunique() for df in out_comp.values() if len(df))),
        "by_video_rows": by_video,
    }
    return out_comp, info


def _score(comp: dict[str, pd.DataFrame]) -> dict[str, object]:
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {**_metric_dict(metrics), "videos_scored": keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission-zip", required=True)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip-out", default="")
    args = ap.parse_args()

    source = _load_zip(Path(args.submission_zip))
    resolved_configs = []
    for text in args.config:
        config_text, config_source = _resolve_config_text(text)
        resolved_configs.append({"source": config_source, "config": _parse_config(config_text)})

    rows = []
    comp_by_name = {}
    for item in resolved_configs:
        cfg = item["config"]
        comp, info = _filter_comp(source, cfg)
        comp_by_name[str(info["config_name"])] = comp
        scored = _score(comp)
        row = {
            **info,
            **scored,
            "submission_zip": str(args.submission_zip),
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
        "submission_zip": str(args.submission_zip),
        "configs": [item["config"] for item in resolved_configs],
        "config_sources": [item["source"] for item in resolved_configs],
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
