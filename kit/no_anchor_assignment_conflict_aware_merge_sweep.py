#!/usr/bin/env python
"""Conflict-aware component merge over an existing no-anchor assignment.

This is a deterministic verifier-style post-processor.  It starts from a
no-anchor assignment CSV, proposes component merges from visual evidence, then
filters candidate edges by component conflict statistics before unioning them.

The conflict statistics are no-anchor evidence:

- same-video/same-camera temporal overlap inside a delivered component;
- an identity-level NMS drop fraction inside a component.

Ground truth is used only after prediction for pair/full metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _merge_edges, _parse_floats, _parse_ints, _write_csv
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
    from no_anchor_assignment_component_merge_sweep import _component_members, _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _merge_edges, _parse_floats, _parse_ints, _write_csv
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


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int, min_overlap: int) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) + 1 >= int(min_overlap)


def _component_conflict_stats(records, members: list[list[int]], *, min_overlap_frames: int) -> list[dict[str, float | int]]:
    stats: list[dict[str, float | int]] = []
    for indices in members:
        indices = [int(idx) for idx in indices]
        by_stream: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx in indices:
            rec = records[idx]
            by_stream[(str(rec.video), str(rec.camera))].append(idx)

        conflict_edges = 0
        conflict_nodes: set[int] = set()
        keep_nodes: set[int] = set()
        drop_nodes: set[int] = set()
        for group in by_stream.values():
            ordered = sorted(group, key=lambda idx: (int(records[idx].start_frame), int(records[idx].end_frame), int(records[idx].seq)))
            for pos, a_idx in enumerate(ordered):
                a = records[a_idx]
                for b_idx in ordered[pos + 1 :]:
                    b = records[b_idx]
                    if int(b.start_frame) > int(a.end_frame):
                        break
                    if _interval_overlap(int(a.start_frame), int(a.end_frame), int(b.start_frame), int(b.end_frame), int(min_overlap_frames)):
                        conflict_edges += 1
                        conflict_nodes.add(a_idx)
                        conflict_nodes.add(b_idx)

            kept: list[int] = []
            ranked = sorted(
                group,
                key=lambda idx: (float(records[idx].n_dets) * (0.25 + float(records[idx].avg_conf)), int(records[idx].n_dets)),
                reverse=True,
            )
            for idx in ranked:
                rec = records[idx]
                has_overlap = False
                for kept_idx in kept:
                    other = records[kept_idx]
                    if _interval_overlap(
                        int(rec.start_frame),
                        int(rec.end_frame),
                        int(other.start_frame),
                        int(other.end_frame),
                        int(min_overlap_frames),
                    ):
                        has_overlap = True
                        break
                if has_overlap:
                    drop_nodes.add(idx)
                else:
                    kept.append(idx)
                    keep_nodes.add(idx)
        n = max(len(indices), 1)
        pairs = max(n * (n - 1) / 2.0, 1.0)
        stats.append(
            {
                "component_size": int(len(indices)),
                "component_weight": int(sum(max(int(records[idx].n_dets), 1) for idx in indices)),
                "conflict_edges": int(conflict_edges),
                "conflict_nodes": int(len(conflict_nodes)),
                "conflict_node_frac": float(len(conflict_nodes) / n),
                "conflict_edge_density": float(conflict_edges / pairs),
                "nms_drop_nodes": int(len(drop_nodes)),
                "nms_drop_frac": float(len(drop_nodes) / n),
            }
        )
    return stats


def _annotate_edges(edges: list[dict[str, float | int]], stats: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for edge in edges:
        src = int(edge["source"])
        tgt = int(edge["target"])
        ss = stats[src]
        ts = stats[tgt]
        row = dict(edge)
        row["source_conflict_node_frac"] = float(ss["conflict_node_frac"])
        row["target_conflict_node_frac"] = float(ts["conflict_node_frac"])
        row["max_conflict_node_frac"] = max(float(ss["conflict_node_frac"]), float(ts["conflict_node_frac"]))
        row["mean_conflict_node_frac"] = 0.5 * (float(ss["conflict_node_frac"]) + float(ts["conflict_node_frac"]))
        row["source_nms_drop_frac"] = float(ss["nms_drop_frac"])
        row["target_nms_drop_frac"] = float(ts["nms_drop_frac"])
        row["max_nms_drop_frac"] = max(float(ss["nms_drop_frac"]), float(ts["nms_drop_frac"]))
        row["mean_nms_drop_frac"] = 0.5 * (float(ss["nms_drop_frac"]) + float(ts["nms_drop_frac"]))
        row["max_conflict_edge_density"] = max(float(ss["conflict_edge_density"]), float(ts["conflict_edge_density"]))
        out.append(row)
    return out


def _filter_edges(
    edges: list[dict[str, float | int]],
    *,
    max_conflict_node_frac: float,
    max_nms_drop_frac: float,
    max_conflict_edge_density: float,
    min_clean_size: int,
) -> tuple[list[dict[str, float | int]], dict[str, int | float]]:
    filtered: list[dict[str, float | int]] = []
    rejected_conflict = 0
    rejected_nms = 0
    rejected_density = 0
    rejected_clean_size = 0
    for edge in edges:
        if max(int(edge["source_size"]), int(edge["target_size"])) < int(min_clean_size):
            rejected_clean_size += 1
            continue
        if float(edge["max_conflict_node_frac"]) > float(max_conflict_node_frac):
            rejected_conflict += 1
            continue
        if float(edge["max_nms_drop_frac"]) > float(max_nms_drop_frac):
            rejected_nms += 1
            continue
        if float(edge["max_conflict_edge_density"]) > float(max_conflict_edge_density):
            rejected_density += 1
            continue
        filtered.append(edge)
    return filtered, {
        "conflict_edges_in": int(len(edges)),
        "conflict_edges_kept": int(len(filtered)),
        "conflict_rejected_node_frac": int(rejected_conflict),
        "conflict_rejected_nms_drop": int(rejected_nms),
        "conflict_rejected_edge_density": int(rejected_density),
        "conflict_rejected_clean_size": int(rejected_clean_size),
        "max_conflict_node_frac": float(max_conflict_node_frac),
        "max_nms_drop_frac": float(max_nms_drop_frac),
        "max_conflict_edge_density": float(max_conflict_edge_density),
        "min_clean_size": int(min_clean_size),
    }


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
    ap.add_argument("--candidate-top-k", type=int, default=80)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weights", default="0.0,0.2,0.5")
    ap.add_argument("--min-source-size", type=int, default=1)
    ap.add_argument("--max-source-size", type=int, default=1000000)
    ap.add_argument("--min-target-size", type=int, default=1)
    ap.add_argument("--max-target-size", type=int, default=1000000)
    ap.add_argument("--forbid-camera-overlap", action="store_true")
    ap.add_argument("--forbid-video-overlap", action="store_true")
    ap.add_argument("--conflict-min-overlap-frames", type=int, default=1)
    ap.add_argument("--max-conflict-node-fracs", default="1.0,0.8,0.5,0.25")
    ap.add_argument("--max-nms-drop-fracs", default="1.0,0.8,0.5,0.25")
    ap.add_argument("--max-conflict-edge-densities", default="1.0,0.10,0.03")
    ap.add_argument("--min-clean-sizes", default="1,4,8")
    ap.add_argument("--max-component-sizes", default="500")
    ap.add_argument("--mutual-top-ks", default="0,1,2")
    ap.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--margins", default="-1.0,0.0,0.02,0.04")
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
    conflict_stats = _component_conflict_stats(records, members, min_overlap_frames=int(args.conflict_min_overlap_frames))
    conflict_summary = {
        "components": int(len(conflict_stats)),
        "components_with_conflict": int(sum(1 for stat in conflict_stats if int(stat["conflict_edges"]) > 0)),
        "mean_conflict_node_frac": float(np.mean([float(stat["conflict_node_frac"]) for stat in conflict_stats])) if conflict_stats else 0.0,
        "mean_nms_drop_frac": float(np.mean([float(stat["nms_drop_frac"]) for stat in conflict_stats])) if conflict_stats else 0.0,
        "max_conflict_node_frac_observed": float(max((float(stat["conflict_node_frac"]) for stat in conflict_stats), default=0.0)),
        "max_nms_drop_frac_observed": float(max((float(stat["nms_drop_frac"]) for stat in conflict_stats), default=0.0)),
    }
    print(json.dumps({"stage": "conflict_stats", **conflict_summary}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    edge_summaries: list[dict[str, object]] = []
    edges_by_key: dict[tuple[float, float, float, float, int], list[dict[str, float | int]]] = {}
    for centroid_weight in _parse_floats(args.centroid_weights):
        base_edges, edge_info = _candidate_edges(
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
        annotated_edges = _annotate_edges(base_edges, conflict_stats)
        print(json.dumps({"stage": "candidate_edges", **edge_info}, sort_keys=True), flush=True)
        for max_conflict_node_frac in _parse_floats(args.max_conflict_node_fracs):
            for max_nms_drop_frac in _parse_floats(args.max_nms_drop_fracs):
                for max_conflict_edge_density in _parse_floats(args.max_conflict_edge_densities):
                    for min_clean_size in _parse_ints(args.min_clean_sizes):
                        edges, filter_info = _filter_edges(
                            annotated_edges,
                            max_conflict_node_frac=max_conflict_node_frac,
                            max_nms_drop_frac=max_nms_drop_frac,
                            max_conflict_edge_density=max_conflict_edge_density,
                            min_clean_size=min_clean_size,
                        )
                        key = (
                            float(centroid_weight),
                            float(max_conflict_node_frac),
                            float(max_nms_drop_frac),
                            float(max_conflict_edge_density),
                            int(min_clean_size),
                        )
                        edges_by_key[key] = edges
                        edge_summaries.append({**edge_info, **filter_info})
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
                                                "mode": "assignment_conflict_aware_merge",
                                                "centroid_weight": float(centroid_weight),
                                                **edge_info,
                                                **filter_info,
                                                **merge_info,
                                                **pair,
                                                "uses_anchors": False,
                                                "uses_gt_for_training_or_anchors": False,
                                                "uses_gt_for_evaluation_only": True,
                                            }
                                        )

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_recall"]),
            float(row["tracklet_pair_precision"]),
        ),
        reverse=True,
    )

    labels_by_rank: dict[int, np.ndarray] = {}
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (
            float(row["centroid_weight"]),
            float(row["max_conflict_node_frac"]),
            float(row["max_nms_drop_frac"]),
            float(row["max_conflict_edge_density"]),
            int(row["min_clean_size"]),
        )
        args.max_component_size = int(row["max_component_size"])
        args.mutual_top_k = int(row.get("mutual_top_k", 0))
        labels, _merge_info = _merge_edges(
            records,
            base_labels,
            edges_by_key[key],
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
        row.update({f"full_{key_name}": value for key_name, value in full.items() if key_name != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full, "row": row}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        labels = labels_by_rank.get(1)
        if labels is None:
            row = rows[0]
            key = (
                float(row["centroid_weight"]),
                float(row["max_conflict_node_frac"]),
                float(row["max_nms_drop_frac"]),
                float(row["max_conflict_edge_density"]),
                int(row["min_clean_size"]),
            )
            args.max_component_size = int(row["max_component_size"])
            args.mutual_top_k = int(row.get("mutual_top_k", 0))
            labels, _merge_info = _merge_edges(
                records,
                base_labels,
                edges_by_key[key],
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
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "conflict_summary": conflict_summary,
        "edge_summaries": edge_summaries[:200],
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
