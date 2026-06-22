#!/usr/bin/env python
"""No-GT per-video detection filter selector for no-anchor assignments.

This script replaces the per-video filter oracle with predeclared policies that
look only at prediction statistics: row density, confidence quantiles, ID count,
and frame coverage.  Ground truth is used only by the final scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kit.evaluate_sample_assignments_full import _export_zip, _metric_dict
from kit.no_anchor_resolve_sweep import _connect, _load_predictions
from kit.submission_video_switch import _load_zip
from vlincs_gallery.eval.score import evaluate, load_ds1_gt_by_video


def _load_assignment_csv(path: str, pred_col: str) -> tuple[dict[int, int], Counter[str]]:
    pred: dict[int, int] = {}
    statuses: Counter[str] = Counter()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"seq", pred_col} - fields
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        status_col = "resolution_status" if "resolution_status" in fields else "decision_status" if "decision_status" in fields else ""
        for row in reader:
            seq = int(float(row["seq"]))
            pred[seq] = int(float(row[pred_col]))
            if status_col:
                statuses[str(row.get(status_col, ""))] += 1
    return pred, statuses


def _video_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "input_rows": 0,
            "input_tracklets": 0,
            "frames": 0,
            "row_density": 0.0,
            "id_count": 0,
            "conf_quantiles": {},
            "area_quantiles": {},
        }
    conf = df["confidence"].astype(float)
    area = (df["x2"].astype(float) - df["x1"].astype(float)).clip(lower=0.0) * (
        df["y2"].astype(float) - df["y1"].astype(float)
    ).clip(lower=0.0)
    frames = int(df["frame"].nunique())
    q_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.25, 0.50]
    return {
        "input_rows": int(len(df)),
        "input_tracklets": int(df["seq"].nunique()) if "seq" in df.columns else 0,
        "frames": frames,
        "row_density": round(float(len(df) / max(frames, 1)), 6),
        "id_count": int(df["id"].nunique()) if "id" in df.columns else 0,
        "conf_quantiles": {f"q{int(q * 100):02d}": round(float(conf.quantile(q)), 6) for q in q_values},
        "area_quantiles": {f"q{int(q * 100):02d}": round(float(area.quantile(q)), 3) for q in q_values},
    }


def _select_conf_q(stats: dict[str, Any], policy: str) -> float:
    rows = int(stats["input_rows"])
    density = float(stats["row_density"])
    q02 = float(stats["conf_quantiles"].get("q02", 1.0))
    q03 = float(stats["conf_quantiles"].get("q03", 1.0))
    q05 = float(stats["conf_quantiles"].get("q05", 1.0))
    tracklets = int(stats.get("input_tracklets", 0))

    if policy == "none":
        return 0.0
    if policy == "density_simple":
        if rows >= 300_000 or density >= 12.0:
            return 0.03
        if rows >= 150_000 or density >= 6.0:
            return 0.01
        return 0.0
    if policy == "density_oracle_lite":
        if rows >= 300_000 or density >= 12.0:
            return 0.03
        if rows >= 150_000 or density >= 6.0:
            return 0.01
        if rows >= 30_000 and 2.0 <= density < 3.0 and q02 <= 0.16:
            return 0.02
        return 0.0
    if policy == "confidence_tail":
        if rows >= 250_000 and q03 <= 0.18:
            return 0.03
        if rows >= 100_000 and q01_or_zero(stats) <= 0.13:
            return 0.01
        if tracklets >= 25 and rows >= 30_000 and q02 <= 0.16 and q05 <= 0.26:
            return 0.02
        return 0.0
    raise ValueError(f"unknown policy {policy!r}")


def q01_or_zero(stats: dict[str, Any]) -> float:
    return float(stats["conf_quantiles"].get("q01", 0.0))


def _build_comp(
    pred_by_video: dict[str, pd.DataFrame],
    pred_by_seq: dict[int, int],
    *,
    policy: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    comp: dict[str, pd.DataFrame] = {}
    by_video: dict[str, Any] = {}
    for video, pred in sorted(pred_by_video.items()):
        out = pred.copy()
        out = out[out["seq"].map(lambda seq: int(seq) in pred_by_seq)].copy()
        if out.empty:
            out["id"] = pd.Series(dtype=np.int64)
            comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
            by_video[video] = {"policy": policy, "conf_q": 0.0, "conf_thr": 0.0, **_video_stats(out)}
            continue
        out["id"] = [int(pred_by_seq[int(seq)]) for seq in out["seq"]]
        stats = _video_stats(out)
        conf_q = float(_select_conf_q(stats, policy))
        conf_thr = 0.0 if conf_q <= 0.0 else float(out["confidence"].astype(float).quantile(conf_q))
        keep = out["confidence"].astype(float) >= conf_thr
        filtered = out.loc[keep].copy()
        comp[video] = filtered[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        by_video[video] = {
            "policy": policy,
            "conf_q": round(float(conf_q), 6),
            "conf_thr": round(float(conf_thr), 6),
            "input_rows": int(len(out)),
            "rows": int(len(filtered)),
            "dropped_rows": int(len(out) - len(filtered)),
            "input_tracklets": int(out["seq"].nunique()),
            "tracklets": int(filtered["seq"].nunique()) if len(filtered) else 0,
            **stats,
        }
    info = {
        "policy_name": policy,
        "by_video_rows": by_video,
        "input_rows": int(sum(item["input_rows"] for item in by_video.values())),
        "rows": int(sum(item["rows"] for item in by_video.values())),
        "dropped_rows": int(sum(item.get("dropped_rows", 0) for item in by_video.values())),
        "predicted_ids": int(sum(df["id"].nunique() for df in comp.values() if len(df))),
    }
    return comp, info


def _build_comp_from_source_zip(
    source: dict[str, pd.DataFrame],
    *,
    policy: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    comp: dict[str, pd.DataFrame] = {}
    by_video: dict[str, Any] = {}
    for video, df in sorted(source.items()):
        out = df.copy()
        if out.empty:
            comp[video] = out[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
            by_video[video] = {"policy": policy, "conf_q": 0.0, "conf_thr": 0.0, **_video_stats(out)}
            continue
        stats = _video_stats(out)
        conf_q = float(_select_conf_q(stats, policy))
        conf_thr = 0.0 if conf_q <= 0.0 else float(out["confidence"].astype(float).quantile(conf_q))
        keep = out["confidence"].astype(float) >= conf_thr
        filtered = out.loc[keep].copy()
        comp[video] = filtered[["frame", "id", "x1", "y1", "x2", "y2", "object_type", "confidence"]].copy()
        by_video[video] = {
            "policy": policy,
            "conf_q": round(float(conf_q), 6),
            "conf_thr": round(float(conf_thr), 6),
            "input_rows": int(len(out)),
            "rows": int(len(filtered)),
            "dropped_rows": int(len(out) - len(filtered)),
            "tracklets": 0,
            **stats,
        }
    info = {
        "policy_name": policy,
        "by_video_rows": by_video,
        "input_rows": int(sum(item["input_rows"] for item in by_video.values())),
        "rows": int(sum(item["rows"] for item in by_video.values())),
        "dropped_rows": int(sum(item.get("dropped_rows", 0) for item in by_video.values())),
        "predicted_ids": int(sum(df["id"].nunique() for df in comp.values() if len(df))),
    }
    return comp, info


def _score(comp: dict[str, pd.DataFrame]) -> dict[str, Any]:
    gt_by_video = load_ds1_gt_by_video()
    keys = sorted(set(gt_by_video).intersection(comp))
    metrics = evaluate({key: gt_by_video[key] for key in keys}, {key: comp[key] for key in keys}, dense=False, n_workers=1)
    return {**_metric_dict(metrics), "videos_scored": keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--assignment-csv", default="")
    ap.add_argument("--source-zip", default="", help="optional existing submission zip to filter without rebuilding IDs")
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--policies", default="density_oracle_lite")
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip-out", default="", help="optional zip for the first policy, not metric-selected")
    args = ap.parse_args()

    if bool(args.source_zip) == bool(args.assignment_csv):
        raise ValueError("provide exactly one of --source-zip or --assignment-csv")
    statuses: Counter[str] = Counter()
    pred_by_seq: dict[int, int] = {}
    pred_by_video: dict[str, pd.DataFrame] | None = None
    source_zip: dict[str, pd.DataFrame] | None = None
    if args.source_zip:
        source_zip = _load_zip(Path(args.source_zip))
    else:
        pred_by_seq, statuses = _load_assignment_csv(args.assignment_csv, args.pred_col)
        con = _connect(args.dbname)
        pred_by_video = _load_predictions(con)
    policies = [part.strip() for part in str(args.policies).split(",") if part.strip()]
    if not policies:
        raise ValueError("--policies is empty")

    rows = []
    comp_by_policy: dict[str, dict[str, pd.DataFrame]] = {}
    for policy in policies:
        if source_zip is not None:
            comp, info = _build_comp_from_source_zip(source_zip, policy=policy)
        else:
            assert pred_by_video is not None
            comp, info = _build_comp(pred_by_video, pred_by_seq, policy=policy)
        comp_by_policy[policy] = comp
        scored = _score(comp)
        row = {
            **info,
            **scored,
            "assignment_csv": str(args.assignment_csv) if args.assignment_csv else "",
            "source_zip": str(args.source_zip) if args.source_zip else "",
            "assigned_tracklets": int(len(pred_by_seq)),
            "assignment_status_counts": dict(statuses),
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_filter_selection": False,
            "uses_gt_for_evaluation_only": True,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "scored",
                    **{key: row[key] for key in ["policy_name", "idf1", "hota", "assa", "detpr", "detre", "rows", "dropped_rows"]},
                },
                sort_keys=True,
            ),
            flush=True,
        )

    zip_info = None
    if args.zip_out:
        first = policies[0]
        zip_info = _export_zip(comp_by_policy[first], args.zip_out)
        rows[0].update(zip_info)

    result = {
        "assignment_csv": str(args.assignment_csv) if args.assignment_csv else "",
        "source_zip": str(args.source_zip) if args.source_zip else "",
        "policies": policies,
        "primary_policy": policies[0],
        "rows": rows,
        "zip_info": zip_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_filter_selection": False,
        "uses_gt_for_evaluation_only": True,
        "selection_uses_gt_metric": False,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "done",
                "json": str(out),
                "primary": {key: rows[0].get(key) for key in ["policy_name", "idf1", "hota", "assa", "detpr", "detre", "rows", "dropped_rows"]},
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
