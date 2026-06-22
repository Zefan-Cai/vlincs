#!/usr/bin/env python
"""No-anchor split-then-merge sweep over an existing global-ID assignment.

The current no-anchor bottleneck is structural: many useful merge candidates
are blocked by cannot-link conflicts inside oversized components.  This driver
first splits conflicted components with same-stream overlap coloring, then
tries conservative multi-view component merges.  GT is loaded only after
prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from kit.no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from kit.no_anchor_component_verifier_sweep import _edge_feature_table, _load_npz_aligned, _parse_floats, _parse_ints, _parse_view
    from kit.no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
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
    from no_anchor_assignment_component_merge_sweep import _labels_from_assignment, _load_assignment_labels
    from no_anchor_component_merge_sweep import _candidate_edges, _component_members, _unionfind_from_labels
    from no_anchor_component_verifier_sweep import _edge_feature_table, _load_npz_aligned, _parse_floats, _parse_ints, _parse_view
    from no_anchor_resolve_sweep import (
        _build_overlap_forbidden,
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


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _component_indices(labels: np.ndarray, allowed: set[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in allowed:
            out[int(label)].append(int(idx))
    return out


def _conflict_graph(indices: list[int], forbidden: list[set[int]]) -> tuple[dict[int, set[int]], int]:
    idx_set = set(indices)
    graph = {idx: set() for idx in indices}
    edges = 0
    for idx in indices:
        for nbr in forbidden[idx] & idx_set:
            if idx < nbr:
                graph[idx].add(nbr)
                graph[nbr].add(idx)
                edges += 1
    return graph, int(edges)


def _greedy_color(indices: list[int], graph: dict[int, set[int]], records) -> dict[int, int]:
    order = sorted(
        indices,
        key=lambda idx: (
            -len(graph[idx]),
            -int(records[idx].n_dets),
            int(records[idx].start_frame),
            int(idx),
        ),
    )
    color: dict[int, int] = {}
    for idx in order:
        used = {color[nbr] for nbr in graph[idx] if nbr in color}
        value = 0
        while value in used:
            value += 1
        color[int(idx)] = int(value)
    return color


def _split_labels(records, labels: np.ndarray, assignment_indices: set[int], forbidden, args, min_size: int, min_rate: float):
    out = labels.copy()
    next_label = int(labels.max()) + 1
    split_components = 0
    skipped_too_many_parts = 0
    rewritten = 0
    total_edges = 0
    total_nodes = 0
    max_parts_seen = 1
    max_parts = int(args.split_max_parts)
    for label, indices in _component_indices(labels, assignment_indices).items():
        if len(indices) < int(min_size):
            continue
        graph, conflict_edges = _conflict_graph(indices, forbidden)
        if conflict_edges <= 0:
            continue
        possible = len(indices) * (len(indices) - 1) / 2.0
        conflict_rate = float(conflict_edges / max(possible, 1.0))
        if conflict_rate < float(min_rate):
            continue
        color = _greedy_color(indices, graph, records)
        colors = sorted(set(color.values()))
        max_parts_seen = max(max_parts_seen, len(colors))
        total_edges += int(conflict_edges)
        total_nodes += int(sum(1 for idx in indices if graph[idx]))
        if max_parts > 0 and len(colors) > max_parts:
            skipped_too_many_parts += 1
            continue
        color_to_label = {0: int(label)}
        for value in colors:
            if int(value) == 0:
                continue
            color_to_label[int(value)] = next_label
            next_label += 1
        for idx, value in color.items():
            new_label = int(color_to_label[int(value)])
            if int(out[idx]) != new_label:
                out[idx] = new_label
                rewritten += 1
        split_components += 1
    return out, {
        "split_min_component_size": int(min_size),
        "split_min_conflict_rate": float(min_rate),
        "split_max_parts": int(max_parts),
        "split_components": int(split_components),
        "split_skipped_too_many_parts": int(skipped_too_many_parts),
        "split_rewritten_tracklets": int(rewritten),
        "split_conflict_edges": int(total_edges),
        "split_conflict_nodes": int(total_nodes),
        "split_max_parts_seen": int(max_parts_seen),
    }


def _merge_by_rule(records, split_labels, edge_rows, args, threshold: float, min_votes: int, max_size: int):
    uf = _unionfind_from_labels(split_labels)
    forbidden = _build_overlap_forbidden(records)
    accepted = rejected_threshold = rejected_votes = rejected_forbidden = rejected_size = rejected_stale = 0
    order = sorted(range(len(edge_rows)), key=lambda idx: float(edge_rows[idx]["score"]), reverse=True)
    for idx in order:
        row = edge_rows[idx]
        if float(row["score"]) < float(threshold):
            rejected_threshold += 1
            continue
        if int(row["votes_top5"]) < int(min_votes):
            rejected_votes += 1
            continue
        if int(row["is_forbidden"]) > 0:
            rejected_forbidden += 1
            continue
        a = int(row["source_rep"])
        b = int(row["target_rep"])
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            rejected_stale += 1
            continue
        if len(uf.members[ra]) + len(uf.members[rb]) > int(max_size):
            rejected_size += 1
            continue
        if not uf.can_merge(a, b, forbidden, int(max_size)):
            rejected_forbidden += 1
            continue
        uf.merge(a, b)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "merge_score_threshold": float(threshold),
        "merge_min_votes_top5": int(min_votes),
        "merge_max_component_size": int(max_size),
        "merge_accepted": int(accepted),
        "merge_rejected_threshold": int(rejected_threshold),
        "merge_rejected_votes": int(rejected_votes),
        "merge_rejected_forbidden": int(rejected_forbidden),
        "merge_rejected_size": int(rejected_size),
        "merge_rejected_stale": int(rejected_stale),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
    }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k, v in row.items() if not isinstance(v, (dict, list, tuple))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _admission_args(args):
    return SimpleNamespace(
        output_min_dets=int(args.output_min_dets),
        output_min_conf=float(args.output_min_conf),
        output_min_area=float(args.output_min_area),
        output_min_quality=float(args.output_min_quality),
        output_min_area_by_video=str(args.output_min_area_by_video),
        output_drop_area_quantile=0.0,
        output_drop_area_quantile_by_video="",
        output_drop_quality_quantile=0.0,
        output_drop_quality_quantile_by_video="",
        output_auto_anomaly_admission=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--view", action="append", default=[])
    ap.add_argument("--split-min-component-sizes", default="32,64,96,128")
    ap.add_argument("--split-min-conflict-rates", default="0.0,0.0005,0.001,0.003")
    ap.add_argument("--split-max-parts", type=int, default=64)
    ap.add_argument("--candidate-top-k", type=int, default=100)
    ap.add_argument("--top-edge-k", type=int, default=8)
    ap.add_argument("--centroid-weight", type=float, default=0.0)
    ap.add_argument("--merge-score-thresholds", default="0.45,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--merge-min-votes-top5-grid", default="0,1,2,3")
    ap.add_argument("--merge-max-component-sizes", default="300,500,800")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
    con = _connect(args.dbname)
    records, db_emb = _load_tracklets(con, args.role)
    primary_emb = _load_feature_npz(args.feature_npz, records, db_emb, concat_db=bool(args.concat_db_embedding), db_weight=float(args.db_weight), feature_weight=float(args.feature_weight))
    pred_by_video = _load_predictions(con)
    records = _with_detection_endpoints(records, pred_by_video)
    gt_by_video = {key: value for key, value in load_ds1_gt_by_video().items() if key in pred_by_video}
    expected = {"cache_version": 1, "dbname": args.dbname, "role": args.role, "iou_thr": 0.5, "min_matches": 1, "min_purity": 0.0, "n_tracklets": len(records), "prediction_rows": int(sum(len(v) for v in pred_by_video.values())), "gt_rows": int(sum(len(v) for v in gt_by_video.values()))}
    cached = _load_eval_label_cache(args.eval_cache, expected)
    if cached is None:
        raise RuntimeError(f"missing or incompatible eval cache: {args.eval_cache}")
    gt_by_seq, weight_by_seq, eval_stats = cached
    keep_seqs, output_info = _output_keep_seqs(records, _admission_args(args))
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    seq_to_idx = {int(record.seq): idx for idx, record in enumerate(records)}
    keep_indices = {seq_to_idx[int(seq)] for seq in keep_seqs if int(seq) in seq_to_idx}
    assignment_indices = {idx for idx, record in enumerate(records) if int(record.seq) in pred_input}

    base_labels, raw_to_local = _labels_from_assignment(records, pred_input)
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    forbidden = _build_overlap_forbidden(records)
    view_embeddings: dict[str, np.ndarray] = {"primary": primary_emb.astype(np.float32)}
    for spec in args.view:
        name, path, weight = _parse_view(spec)
        view_embeddings[name] = (_l2n(db_emb.astype(np.float32)) * float(weight)) if path.lower() == "db" else _load_npz_aligned(path, records, weight=float(weight))

    rows: list[dict[str, object]] = []
    edge_summaries: list[dict[str, object]] = []
    cache: dict[tuple[int, float], tuple[np.ndarray, list[dict[str, object]], dict[str, object]]] = {}
    for min_size in _parse_ints(args.split_min_component_sizes):
        for min_rate in _parse_floats(args.split_min_conflict_rates):
            split_labels, split_info = _split_labels(records, base_labels, assignment_indices, forbidden, args, min_size, min_rate)
            split_pred = _labels_to_seq_map(records, split_labels, keep_seqs=keep_seqs)
            split_pair = _pair_metrics([record.seq for record in records], split_pred, gt_by_seq, weight_by_seq)
            rows.append({"mode": "split_only", **split_info, **split_pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})
            reps, members = _component_members(split_labels, keep_indices)
            edges, edge_info = _candidate_edges(records, primary_emb, reps, members, candidate_top_k=int(args.candidate_top_k), top_edge_k=int(args.top_edge_k), centroid_weight=float(args.centroid_weight), min_source_size=1, max_source_size=1_000_000, min_target_size=1, max_target_size=1_000_000, forbid_camera_overlap=False, forbid_video_overlap=False)
            edge_rows, _X, feature_names = _edge_feature_table(records, members, edges, view_embeddings)
            edge_summaries.append({**split_info, **edge_info, "feature_count": int(len(feature_names))})
            cache[(int(min_size), float(min_rate))] = (split_labels, edge_rows, split_info)
            for threshold in _parse_floats(args.merge_score_thresholds):
                for min_votes in _parse_ints(args.merge_min_votes_top5_grid):
                    for max_size in _parse_ints(args.merge_max_component_sizes):
                        labels, merge_info = _merge_by_rule(records, split_labels, edge_rows, args, threshold, min_votes, max_size)
                        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                        pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
                        rows.append({"mode": "split_then_merge", **split_info, **edge_info, **merge_info, **pair, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True})

    rows.sort(key=lambda row: (float(row["tracklet_pair_f1"]), float(row["tracklet_pair_recall"]), float(row["tracklet_pair_precision"])), reverse=True)
    full_rows = []
    for row in rows[: max(int(args.full_top_n), 0)]:
        if row["mode"] == "split_only":
            labels = cache[(int(row["split_min_component_size"]), float(row["split_min_conflict_rate"]))][0]
        else:
            split_labels, edge_rows, _ = cache[(int(row["split_min_component_size"]), float(row["split_min_conflict_rate"]))]
            labels, _ = _merge_by_rule(records, split_labels, edge_rows, args, float(row["merge_score_threshold"]), int(row["merge_min_votes_top5"]), int(row["merge_max_component_size"]))
        pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
        full = _score_full(pred_by_video, gt_by_video, pred)
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        full_rows.append(row)

    result = {"assignment_csv": args.assignment_csv, "base_assignment_components": int(len(raw_to_local)), "base_pair_metrics": base_pair, "output_admission": output_info, "eval_stats": eval_stats, "edge_summaries": edge_summaries[:80], "top": rows[:100], "full_rows": full_rows, "uses_anchors": False, "uses_gt_for_training_or_anchors": False, "uses_gt_for_evaluation_only": True}
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
