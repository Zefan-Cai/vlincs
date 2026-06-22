#!/usr/bin/env python
"""No-anchor multi-video admission grid after one time-agglom resolve.

This is a focused follow-up to `no_anchor_time_admission_grid.py`: keep one
identity clustering fixed, then sweep video-specific M3 admission thresholds.
Ground truth is used only for evaluation / ranking rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _labels_to_seq_map,
        _label_tracklets_for_eval,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _cache_eval_labels,
        _connect,
        _labels_to_seq_map,
        _label_tracklets_for_eval,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_agglom_resolve,
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


def _admission_args(args, video_min_area: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        output_min_dets=1,
        output_min_conf=float(args.base_min_conf),
        output_min_area=float(args.base_min_area),
        output_min_quality=-1.0e9,
        output_min_area_by_video=",".join(f"{video}:{area}" for video, area in sorted(video_min_area.items())),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
        output_auto_anomaly_metric="quality",
        output_auto_anomaly_quantile=0.75,
        output_auto_anomaly_area_ratio=0.60,
        output_auto_anomaly_quality_mad=1.0,
        output_auto_anomaly_min_video_tracklets=20,
        output_auto_anomaly_max_videos=3,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.0165)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--base-min-conf", type=float, default=0.0)
    ap.add_argument("--base-min-area", type=int, default=0)
    ap.add_argument("--fixed-video-min-area", default="")
    ap.add_argument(
        "--video-area-grid",
        required=True,
        help="semicolon list: video:0|2000|4000;video2:0|4000",
    )
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=200)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
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
    cached = _load_eval_label_cache(args.eval_cache, expected) if args.eval_cache else None
    if cached is None:
        gt_by_seq, weight_by_seq, eval_stats = _label_tracklets_for_eval(
            pred_by_video,
            gt_by_video,
            iou_thr=0.5,
            min_matches=1,
            min_purity=0.0,
        )
        eval_stats.update(expected)
        if args.eval_cache:
            _cache_eval_labels(args.eval_cache, gt_by_seq, weight_by_seq, eval_stats)
    else:
        gt_by_seq, weight_by_seq, eval_stats = cached

    cfg = ResolveConfig(
        mode="time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    labels, resolve_info = _time_agglom_resolve(records, emb, cfg)
    seqs = [record.seq for record in records]
    fixed = _parse_fixed(args.fixed_video_min_area)
    grid = _parse_grid(args.video_area_grid)

    rows = []
    for values in itertools.product(*[item[1] for item in grid]):
        video_min_area = dict(fixed)
        video_min_area.update({grid[i][0]: int(values[i]) for i in range(len(grid))})
        admission = _admission_args(args, video_min_area)
        keep, output_info = _output_keep_seqs(records, admission)
        pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep)
        metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
        row = {
            "base_min_conf": float(args.base_min_conf),
            "base_min_area": int(args.base_min_area),
            "video_min_area": video_min_area,
            "video_min_area_key": ",".join(f"{video}:{area}" for video, area in sorted(video_min_area.items())),
            "output_tracklets": int(output_info["output_tracklets"]),
            "output_filtered_tracklets": int(output_info["output_filtered_tracklets"]),
            **metrics,
        }
        row["coverage_pair_score"] = row["tracklet_pair_f1"] * min(1.0, row["output_tracklets"] / max(len(records), 1))
        rows.append(row)

    rows.sort(key=lambda row: (float(row.get(args.sort_key, 0.0)), row["tracklet_pair_recall"]), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        admission = _admission_args(args, row["video_min_area"])
        keep, _output_info = _output_keep_seqs(records, admission)
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=keep))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = rank
        print(json.dumps({"stage": "full", "rank": rank, "row": row}, sort_keys=True), flush=True)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "resolve_config": cfg.__dict__,
        "resolve_info": resolve_info,
        "eval_stats": eval_stats,
        "fixed_video_min_area": fixed,
        "video_area_grid": {video: values for video, values in grid},
        "n_rows": len(rows),
        "sort_key": args.sort_key,
        "top": rows[: max(int(args.json_top_n), int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "json": str(out), "n_rows": len(rows), "top10": rows[:10]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
