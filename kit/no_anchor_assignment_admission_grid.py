#!/usr/bin/env python
"""Sweep M3 admission thresholds for an existing no-anchor assignment CSV.

This keeps the predicted global IDs fixed and only changes which tracklets are
delivered.  It is useful when pair metrics are already high but full IDF1 is
limited by detector/admission false positives.  Ground truth is used only for
post-hoc metrics and ranking.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        _connect,
        _load_eval_label_cache,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def _parse_fixed(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in str(text or "").split(","):
        if not part.strip():
            continue
        video, value = part.rsplit(":", 1)
        out[video] = int(value)
    return out


def _parse_grid(text: str) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = []
    for part in str(text or "").split(";"):
        if not part.strip():
            continue
        video, values = part.rsplit(":", 1)
        out.append((video, [int(value) for value in values.split("|") if value.strip()]))
    return out


def _parse_float_grid(text: str) -> list[float]:
    return [float(value) for value in str(text or "").split(",") if value.strip()]


def _load_assignment_csv(path: str, pred_col: str) -> tuple[dict[int, int], list[dict[str, str]]]:
    pred: dict[int, int] = {}
    rows: list[dict[str, str]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "seq" not in fields:
            raise ValueError(f"{path} is missing seq")
        if pred_col not in fields:
            raise ValueError(f"{path} is missing {pred_col}")
        for row in reader:
            seq = int(float(row["seq"]))
            pred[seq] = int(float(row[pred_col]))
            rows.append(dict(row))
    return pred, rows


def _admission_args(args, video_min_area: dict[str, int], min_conf: float, min_quality: float) -> SimpleNamespace:
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(min_quality),
        output_min_area_by_video=",".join(f"{video}:{area}" for video, area in sorted(video_min_area.items())),
        output_drop_area_quantile=float(args.output_drop_area_quantile),
        output_drop_area_quantile_by_video=str(args.output_drop_area_quantile_by_video),
        output_drop_quality_quantile=float(args.output_drop_quality_quantile),
        output_drop_quality_quantile_by_video=str(args.output_drop_quality_quantile_by_video),
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def _sort_tuple(row: dict[str, object], key: str) -> tuple[float, float, float, float, int]:
    return (
        float(row.get(key, 0.0)),
        float(row.get("tracklet_pair_f1", 0.0)),
        float(row.get("tracklet_pair_precision", 0.0)),
        float(row.get("tracklet_pair_recall", 0.0)),
        int(row.get("output_tracklets", 0)),
    )


def _write_filtered_assignments(path: str, rows: list[dict[str, str]], keep: set[int]) -> dict[str, object]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no assignment rows to write")
    fieldnames = list(rows[0].keys())
    written = 0
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if int(float(row["seq"])) not in keep:
                continue
            writer.writerow(row)
            written += 1
    return {"assignments_out": str(out), "assignment_rows": int(written)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf-grid", default="0.0")
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality-grid", default="-1000000000.0")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--fixed-video-min-area", default="")
    ap.add_argument("--video-area-grid", default="", help="semicolon list: video:0|2000|4000;video2:0|4000")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=200)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    pred_all, assignment_rows = _load_assignment_csv(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, _emb = _load_tracklets(con, args.role)
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {
        "cache_version": 1,
        "dbname": args.dbname,
        "role": args.role,
        "iou_thr": 0.5,
        "min_matches": 1,
        "min_purity": 0.0,
        "n_tracklets": len(records),
        "prediction_rows": int(sum(len(value) for value in pred_by_video.values())),
        "gt_rows": int(sum(len(value) for value in gt_by_video.values())),
    }
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached

    seqs = [int(record.seq) for record in records]
    fixed = _parse_fixed(args.fixed_video_min_area)
    grid = _parse_grid(args.video_area_grid)
    if not grid:
        grid = [("__none__", [0])]
    min_conf_values = _parse_float_grid(args.output_min_conf_grid)
    min_quality_values = _parse_float_grid(args.output_min_quality_grid)

    rows: list[dict[str, object]] = []
    keep_by_rank_key: dict[str, set[int]] = {}
    for min_conf in min_conf_values:
        for min_quality in min_quality_values:
            for values in itertools.product(*[item[1] for item in grid]):
                video_min_area = dict(fixed)
                for idx, (video, _choices) in enumerate(grid):
                    if video == "__none__":
                        continue
                    video_min_area[video] = int(values[idx])
                admission = _admission_args(args, video_min_area, min_conf, min_quality)
                keep, output_info = _output_keep_seqs(records, admission)
                pred = {seq: gid for seq, gid in pred_all.items() if int(seq) in keep}
                metrics = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                rank_key = f"r{len(rows)}"
                keep_by_rank_key[rank_key] = keep
                row = {
                    "rank_key": rank_key,
                    "assignment_csv": args.assignment_csv,
                    "pred_col": args.pred_col,
                    "output_min_conf": float(min_conf),
                    "output_min_quality": float(min_quality),
                    "video_min_area": video_min_area,
                    "video_min_area_key": ",".join(f"{video}:{area}" for video, area in sorted(video_min_area.items())),
                    "output_tracklets": int(output_info["output_tracklets"]),
                    "output_filtered_tracklets": int(output_info["output_filtered_tracklets"]),
                    "assigned_after_filter": int(len(pred)),
                    "assignment_input_rows": int(len(assignment_rows)),
                    **metrics,
                    "uses_anchors": False,
                    "uses_gt_for_training_or_anchors": False,
                    "uses_gt_for_evaluation_only": True,
                }
                row["coverage_ratio"] = float(len(pred) / max(len(records), 1))
                row["coverage_pair_score"] = float(row["tracklet_pair_f1"]) * row["coverage_ratio"]
                row["coverage_sqrt_pair_score"] = float(row["tracklet_pair_f1"]) * (row["coverage_ratio"] ** 0.5)
                row["min_precision_recall"] = min(float(row["tracklet_pair_precision"]), float(row["tracklet_pair_recall"]))
                rows.append(row)

    rows.sort(key=lambda row: _sort_tuple(row, str(args.sort_key)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        keep = keep_by_rank_key[str(row["rank_key"])]
        pred = {seq: gid for seq, gid in pred_all.items() if int(seq) in keep}
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        assignment_info = _write_filtered_assignments(
            args.assignments_out,
            assignment_rows,
            keep_by_rank_key[str(rows[0]["rank_key"])],
        )
        rows[0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "assignment_csv": args.assignment_csv,
        "pred_col": args.pred_col,
        "fixed_video_min_area": fixed,
        "video_area_grid": {video: values for video, values in grid if video != "__none__"},
        "min_conf_grid": min_conf_values,
        "min_quality_grid": min_quality_values,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "n_rows": len(rows),
        "sort_key": str(args.sort_key),
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "n_rows": len(rows), "top": rows[:10]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
