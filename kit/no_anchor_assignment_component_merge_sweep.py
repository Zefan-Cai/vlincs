#!/usr/bin/env python
"""Merge components from an existing no-anchor assignment CSV.

The input assignment is treated as the current identity-resolution output.
This script uses a separate no-anchor feature view to propose conservative
component-level merges, then reports pair/full metrics.  Ground truth is used
only after prediction for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_merge_sweep import (
        _candidate_edges,
        _merge_edges,
        _parse_floats,
        _parse_ints,
        _row_sort_key,
        _write_csv,
    )
    from kit.no_anchor_louvain_sweep import _write_assignments
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
        _merge_edges,
        _parse_floats,
        _parse_ints,
        _row_sort_key,
        _write_csv,
    )
    from no_anchor_louvain_sweep import _write_assignments
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


def _load_assignment_labels(path: str, pred_col: str) -> dict[int, int]:
    pred: dict[int, int] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "seq" not in fields:
            raise ValueError(f"{path} is missing seq")
        if pred_col not in fields:
            raise ValueError(f"{path} is missing {pred_col}")
        for row in reader:
            pred[int(float(row["seq"]))] = int(float(row[pred_col]))
    return pred


def _labels_from_assignment(records, pred_by_seq: dict[int, int]) -> tuple[np.ndarray, dict[int, int]]:
    raw_to_local: dict[int, int] = {}
    labels = np.full(len(records), -1, dtype=np.int64)
    next_label = 0
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in pred_by_seq:
            continue
        raw = int(pred_by_seq[seq])
        if raw not in raw_to_local:
            raw_to_local[raw] = next_label
            next_label += 1
        labels[idx] = raw_to_local[raw]
    for idx in range(len(records)):
        if labels[idx] < 0:
            labels[idx] = next_label
            next_label += 1
    return labels, raw_to_local


def _component_members(labels: np.ndarray, keep_indices: set[int]) -> tuple[list[int], list[list[int]]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in keep_indices:
            by_label[int(label)].append(int(idx))
    reps: list[int] = []
    members: list[list[int]] = []
    for _label, indices in sorted(by_label.items(), key=lambda item: min(item[1])):
        reps.append(int(indices[0]))
        members.append(indices)
    return reps, members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--merge-feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
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
    ap.add_argument("--centroid-weights", default="0.0,0.5,1.0")
    ap.add_argument("--min-source-size", type=int, default=1)
    ap.add_argument("--max-source-size", type=int, default=1000000)
    ap.add_argument("--min-target-size", type=int, default=1)
    ap.add_argument("--max-target-size", type=int, default=1000000)
    ap.add_argument("--forbid-camera-overlap", action="store_true")
    ap.add_argument("--forbid-video-overlap", action="store_true")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--mutual-top-ks", default="0,1,2")
    ap.add_argument("--accepted-preview-n", type=int, default=20)
    ap.add_argument("--rank-by", default="pair", choices=["pair", "precision", "recall", "mass_proxy", "mass_then_pair"])
    ap.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--margins", default="-1.0,0.0,0.02,0.04,0.06")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=60_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    merge_emb = _load_feature_npz(
        args.merge_feature_npz,
        records,
        db_emb,
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
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    reps, members = _component_members(base_labels, keep_indices)
    rows: list[dict[str, object]] = []
    edge_summaries: list[dict[str, object]] = []
    edges_by_centroid_weight: dict[float, list[dict[str, float | int]]] = {}
    for centroid_weight in _parse_floats(args.centroid_weights):
        edges, edge_info = _candidate_edges(
            records,
            merge_emb,
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
        print(json.dumps({"stage": "candidate_edges", **edge_info}, sort_keys=True), flush=True)
        for max_component_size in _parse_ints(args.max_component_sizes):
            args.max_component_size = int(max_component_size)
            for mutual_top_k in _parse_ints(args.mutual_top_ks):
                args.mutual_top_k = int(mutual_top_k)
                for threshold in _parse_floats(args.thresholds):
                    for margin in _parse_floats(args.margins):
                        labels, merge_info = _merge_edges(records, base_labels, edges, args, threshold, margin)
                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        rows.append(
                            {
                                "mode": "assignment_component_merge",
                                "centroid_weight": float(centroid_weight),
                                **edge_info,
                                **merge_info,
                                **pair,
                                "uses_anchors": False,
                                "uses_gt_for_training_or_anchors": False,
                                "uses_gt_for_evaluation_only": True,
                            }
                        )

    rows.sort(key=lambda row: _row_sort_key(row, str(args.rank_by)), reverse=True)

    labels_by_rank: dict[int, np.ndarray] = {}
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
        full = _score_full(
            pred_by_video,
            gt_by_video,
            _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs),
        )
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
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
        "assignment_csv": args.assignment_csv,
        "merge_feature_npz": args.merge_feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "edge_summaries": edge_summaries,
        "assignment_info": assignment_info,
        "rank_by": str(args.rank_by),
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
