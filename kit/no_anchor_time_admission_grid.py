#!/usr/bin/env python
"""Fast no-anchor admission grid for one time-agglom resolve.

This evaluates output admission thresholds after a single no-GT resolve run.
Ground truth is used only through the evaluation cache / scorer.
"""

from __future__ import annotations

import argparse
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


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _admission_args(args, min_conf: float, min_area: int, mcam05_area: int) -> SimpleNamespace:
    video = str(args.mcam05_video)
    return SimpleNamespace(
        output_min_dets=1,
        output_min_conf=float(min_conf),
        output_min_area=float(min_area),
        output_min_quality=-1.0e9,
        output_min_area_by_video=(f"{video}:{mcam05_area}" if int(mcam05_area) > 0 else ""),
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
    ap.add_argument("--feature-npz", default=None, help="optional no-anchor external tracklet features keyed by seq")
    ap.add_argument("--concat-db-embedding", action="store_true", help="concatenate --feature-npz with DB role embedding")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.018)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--min-confs", default="0,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--min-areas", default="0,1000,2000,3000,4000,5000,6000,8000,10000,12000")
    ap.add_argument("--mcam05-areas", default="0,8000,10000,12000,14000,16000,20000,24000")
    ap.add_argument("--mcam05-video", default="vlincs_MS01_MC0001_MCAM05_2024-03-Tc6")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json-top-n", type=int, default=50)
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

    rows = []
    for min_conf in _parse_floats(args.min_confs):
        for min_area in _parse_ints(args.min_areas):
            for mcam05_area in _parse_ints(args.mcam05_areas):
                admission = _admission_args(args, min_conf, min_area, mcam05_area)
                keep, output_info = _output_keep_seqs(records, admission)
                pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep)
                metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
                row = {
                    "min_conf": float(min_conf),
                    "min_area": int(min_area),
                    "mcam05_area": int(mcam05_area),
                    "output_tracklets": int(output_info["output_tracklets"]),
                    "output_filtered_tracklets": int(output_info["output_filtered_tracklets"]),
                    **metrics,
                }
                row["gate_min_pr"] = min(row["tracklet_pair_precision"], row["tracklet_pair_recall"])
                row["gate_min_f1pr"] = min(
                    row["tracklet_pair_f1"],
                    row["tracklet_pair_precision"],
                    row["tracklet_pair_recall"],
                )
                rows.append(row)

    rows.sort(key=lambda row: (row["gate_min_f1pr"], row["tracklet_pair_f1"], row["tracklet_pair_recall"]), reverse=True)
    full_rows = rows[: max(int(args.full_top_n), 0)]
    for rank, row in enumerate(full_rows, start=1):
        admission = _admission_args(args, float(row["min_conf"]), int(row["min_area"]), int(row["mcam05_area"]))
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
        "n_rows": len(rows),
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
