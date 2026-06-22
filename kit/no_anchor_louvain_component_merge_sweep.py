#!/usr/bin/env python
"""Second-stage component merge sweep on top of no-anchor Louvain labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import (
        _candidate_edges,
        _component_members,
        _merge_edges,
        _parse_floats,
        _parse_ints,
        _write_csv,
    )
    from kit.no_anchor_louvain_sweep import _louvain_labels, _write_assignments
    from kit.no_anchor_resolve_sweep import (
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_component_merge_sweep import (
        _candidate_edges,
        _component_members,
        _merge_edges,
        _parse_floats,
        _parse_ints,
        _write_csv,
    )
    from no_anchor_louvain_sweep import _louvain_labels, _write_assignments
    from no_anchor_resolve_sweep import (
        _connect,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _with_detection_endpoints,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--edge-floor", type=float, default=0.035)
    ap.add_argument("--resolution", type=float, default=5.0)
    ap.add_argument("--random-state", type=int, default=17)
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--output-drop-area-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-area-quantile-by-video", default="")
    ap.add_argument("--output-drop-quality-quantile", type=float, default=0.0)
    ap.add_argument("--output-drop-quality-quantile-by-video", default="")
    ap.add_argument("--output-auto-anomaly-admission", action="store_true")
    ap.add_argument("--output-auto-anomaly-metric", default="quality")
    ap.add_argument("--output-auto-anomaly-quantile", type=float, default=0.75)
    ap.add_argument("--output-auto-anomaly-area-ratio", type=float, default=0.60)
    ap.add_argument("--output-auto-anomaly-quality-mad", type=float, default=1.0)
    ap.add_argument("--output-auto-anomaly-min-video-tracklets", type=int, default=20)
    ap.add_argument("--output-auto-anomaly-max-videos", type=int, default=3)
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weights", default="0.2")
    ap.add_argument("--min-source-size", type=int, default=1)
    ap.add_argument("--max-source-size", type=int, default=1000000)
    ap.add_argument("--min-target-size", type=int, default=1)
    ap.add_argument("--max-target-size", type=int, default=1000000)
    ap.add_argument("--forbid-camera-overlap", action="store_true")
    ap.add_argument("--forbid-video-overlap", action="store_true")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--mutual-top-ks", default="0")
    ap.add_argument("--thresholds", default="0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74")
    ap.add_argument("--margins", default="-1.0,0.0,0.02,0.04")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default=None)
    ap.add_argument("--assignment-offset", type=int, default=40_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default=None)
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
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, args)
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    base_labels, base_info = _louvain_labels(
        records,
        emb,
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
        edge_floor=float(args.edge_floor),
        resolution=float(args.resolution),
        random_state=int(args.random_state),
    )
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", **base_info, **base_pair}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    edge_summaries: list[dict[str, object]] = []
    edges_by_centroid_weight: dict[float, list[dict[str, float | int]]] = {}
    reps, members = _component_members(base_labels, keep_indices)
    for centroid_weight in _parse_floats(args.centroid_weights):
        edges, edge_info = _candidate_edges(
            records,
            emb,
            reps,
            members,
            candidate_top_k=int(args.candidate_top_k),
            top_edge_k=int(args.top_edge_k),
            centroid_weight=float(centroid_weight),
            min_source_size=int(args.min_source_size),
            max_source_size=int(args.max_source_size),
            min_target_size=int(args.min_target_size),
            max_target_size=int(args.max_target_size),
            forbid_camera_overlap=bool(args.forbid_camera_overlap),
            forbid_video_overlap=bool(args.forbid_video_overlap),
        )
        edges_by_centroid_weight[float(centroid_weight)] = edges
        edge_summaries.append(edge_info)
        for max_component_size in _parse_ints(args.max_component_sizes):
            args.max_component_size = int(max_component_size)
            for mutual_top_k in _parse_ints(args.mutual_top_ks):
                args.mutual_top_k = int(mutual_top_k)
                for threshold in _parse_floats(args.thresholds):
                    for margin in _parse_floats(args.margins):
                        labels, merge_info = _merge_edges(records, base_labels, edges, args, threshold, margin)
                        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        row = {
                            "mode": "louvain_component_merge",
                            "centroid_weight": float(centroid_weight),
                            **edge_info,
                            **merge_info,
                            **pair,
                            "uses_anchors": False,
                            "uses_gt_for_training_or_anchors": False,
                            "uses_gt_for_evaluation_only": True,
                        }
                        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )

    labels_by_rank: dict[int, object] = {}
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        full_edges = edges_by_centroid_weight[float(row["centroid_weight"])]
        args.max_component_size = int(row["max_component_size"])
        args.mutual_top_k = int(row.get("mutual_top_k", 0))
        labels, _merge_info = _merge_edges(
            records,
            base_labels,
            full_edges,
            args,
            float(row["merge_threshold"]),
            float(row["merge_margin"]),
        )
        labels_by_rank[rank] = labels
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            full_edges = edges_by_centroid_weight[float(rows[0]["centroid_weight"])]
            args.max_component_size = int(rows[0]["max_component_size"])
            args.mutual_top_k = int(rows[0].get("mutual_top_k", 0))
            labels, _merge_info = _merge_edges(
                records,
                base_labels,
                full_edges,
                args,
                float(rows[0]["merge_threshold"]),
                float(rows[0]["merge_margin"]),
            )
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels,
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_resolve_config": {
            "mode": "louvain",
            "top_k": int(args.top_k),
            "edge_floor": float(args.edge_floor),
            "resolution": float(args.resolution),
            "min_dets": int(args.min_dets),
            "exclude_same": str(args.exclude_same),
            "temporal_bonus": float(args.temporal_bonus),
            "time_window_ms": int(args.time_window_ms),
        },
        "base_resolve_info": base_info,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_summaries": edge_summaries,
        "assignment_info": assignment_info,
        "top": rows[: max(50, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"base": base_pair, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
