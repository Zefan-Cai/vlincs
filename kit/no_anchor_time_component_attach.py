#!/usr/bin/env python
"""Attach split fragments on top of the best no-anchor time-agglom resolver.

The base resolver remains label-free.  This script only uses component-level
appearance evidence to attach small/medium components to larger components when
there is a clear no-GT margin.  GT is loaded only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _UnionFind,
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
        _time_agglom_resolve,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _UnionFind,
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
        _time_agglom_resolve,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _unionfind_from_labels(labels: np.ndarray) -> _UnionFind:
    uf = _UnionFind(int(len(labels)))
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        groups[int(label)].append(int(idx))
    for indices in groups.values():
        head = int(indices[0])
        for idx in indices[1:]:
            uf.merge(head, int(idx))
    return uf


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


def _candidate_edges(
    records,
    emb: np.ndarray,
    reps: list[int],
    members: list[list[int]],
    *,
    source_max_size: int,
    target_min_size: int,
    candidate_top_k: int,
    top_edge_k: int,
    centroid_weight: float,
    forbid_same_camera: bool,
) -> list[dict[str, float | int]]:
    x = _l2n(emb.astype(np.float32))
    cents = []
    for indices in members:
        v = x[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        cents.append(v / (np.linalg.norm(v) + 1.0e-9))
    C = np.stack(cents).astype(np.float32)
    sim = C @ C.T
    np.fill_diagonal(sim, -2.0)
    sizes = np.asarray([len(indices) for indices in members], dtype=np.int64)
    if forbid_same_camera:
        cameras = np.asarray([records[rep].camera for rep in reps])
        sim[cameras[:, None] == cameras[None, :]] = -2.0

    edges: list[dict[str, float | int]] = []
    k = min(max(int(candidate_top_k), 1), max(len(members) - 1, 1))
    for src, src_members in enumerate(members):
        if int(sizes[src]) > int(source_max_size):
            continue
        target_mask = sizes >= int(target_min_size)
        target_mask[src] = False
        if not np.any(target_mask):
            continue
        row = np.where(target_mask, sim[src], -2.0)
        top = np.argpartition(-row, k - 1)[:k]
        best: list[tuple[float, int, float, float]] = []
        src_x = x[np.asarray(src_members, dtype=np.int64)]
        for tgt in top.tolist():
            tgt = int(tgt)
            centroid_score = float(sim[src, tgt])
            if centroid_score <= -1.0:
                continue
            tgt_x = x[np.asarray(members[tgt], dtype=np.int64)]
            edge_scores = (src_x @ tgt_x.T).reshape(-1)
            if edge_scores.size == 0:
                continue
            kk = min(max(int(top_edge_k), 1), int(edge_scores.size))
            top_scores = np.partition(edge_scores, -kk)[-kk:]
            top_mean = float(np.mean(top_scores))
            score = float(centroid_weight) * centroid_score + (1.0 - float(centroid_weight)) * top_mean
            best.append((score, tgt, centroid_score, top_mean))
        best.sort(reverse=True, key=lambda item: item[0])
        if not best:
            continue
        second = float(best[1][0]) if len(best) > 1 else -1.0
        score, tgt, centroid_score, top_mean = best[0]
        edges.append(
            {
                "score": float(score),
                "second_score": float(second),
                "margin": float(score - second),
                "source": int(src),
                "target": int(tgt),
                "source_rep": int(reps[src]),
                "target_rep": int(reps[tgt]),
                "source_size": int(sizes[src]),
                "target_size": int(sizes[tgt]),
                "centroid_score": float(centroid_score),
                "top_edge_mean": float(top_mean),
            }
        )
    edges.sort(key=lambda row: float(row["score"]), reverse=True)
    return edges


def _attach(records, emb, base_labels, keep_indices: set[int], args, source_max_size: int, threshold: float, margin: float):
    reps, members = _component_members(base_labels, keep_indices)
    edges = _candidate_edges(
        records,
        emb,
        reps,
        members,
        source_max_size=int(source_max_size),
        target_min_size=int(args.attach_target_min_size),
        candidate_top_k=int(args.attach_candidate_top_k),
        top_edge_k=int(args.attach_top_edge_k),
        centroid_weight=float(args.attach_centroid_weight),
        forbid_same_camera=bool(args.attach_forbid_same_camera),
    )
    uf = _unionfind_from_labels(base_labels)
    forbidden = _build_overlap_forbidden(records)
    accepted = 0
    rejected_threshold = 0
    rejected_margin = 0
    rejected_stale = 0
    rejected_forbidden = 0
    rejected_size = 0
    for edge in edges:
        if float(edge["score"]) < float(threshold):
            rejected_threshold += 1
            continue
        if float(edge["margin"]) < float(margin):
            rejected_margin += 1
            continue
        src = int(edge["source_rep"])
        tgt = int(edge["target_rep"])
        src_root = uf.find(src)
        tgt_root = uf.find(tgt)
        if src_root == tgt_root:
            rejected_stale += 1
            continue
        if len(uf.members[src_root]) > int(source_max_size):
            rejected_stale += 1
            continue
        if len(uf.members[tgt_root]) < int(args.attach_target_min_size):
            rejected_stale += 1
            continue
        if len(uf.members[src_root]) + len(uf.members[tgt_root]) > int(args.max_component_size):
            rejected_size += 1
            continue
        if not uf.can_merge(src, tgt, forbidden, int(args.max_component_size)):
            rejected_forbidden += 1
            continue
        uf.merge(src, tgt)
        accepted += 1
    labels = uf.labels()
    return labels, {
        "base_components": int(len(set(base_labels.tolist()))),
        "base_largest_component": int(max(Counter(base_labels.tolist()).values(), default=0)),
        "attach_candidate_edges": int(len(edges)),
        "attach_accepted": int(accepted),
        "attach_rejected_threshold": int(rejected_threshold),
        "attach_rejected_margin": int(rejected_margin),
        "attach_rejected_stale": int(rejected_stale),
        "attach_rejected_forbidden": int(rejected_forbidden),
        "attach_rejected_size": int(rejected_size),
        "components": int(len(set(labels.tolist()))),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "attach_source_max_size": int(source_max_size),
        "attach_threshold": float(threshold),
        "attach_margin": float(margin),
        "attach_target_min_size": int(args.attach_target_min_size),
        "attach_candidate_top_k": int(args.attach_candidate_top_k),
        "attach_top_edge_k": int(args.attach_top_edge_k),
        "attach_centroid_weight": float(args.attach_centroid_weight),
        "attach_forbid_same_camera": bool(args.attach_forbid_same_camera),
        "uses_ground_truth": False,
    }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    keys = sorted(key for row in rows for key, value in row.items() if not isinstance(value, (dict, list, tuple)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1")
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.014)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--max-component-size", type=int, default=500)
    ap.add_argument("--attach-source-max-sizes", default="4,8,16,32")
    ap.add_argument("--attach-thresholds", default="0.70,0.72,0.74,0.76")
    ap.add_argument("--attach-margins", default="0.02,0.04,0.06")
    ap.add_argument("--attach-target-min-size", type=int, default=12)
    ap.add_argument("--attach-candidate-top-k", type=int, default=10)
    ap.add_argument("--attach-top-edge-k", type=int, default=5)
    ap.add_argument("--attach-centroid-weight", type=float, default=0.45)
    ap.add_argument("--attach-forbid-same-camera", action="store_true")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--sort-key", default="tracklet_pair_f1")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    con = _connect(args.dbname)
    records, emb = _load_tracklets(con, args.role)
    print(json.dumps({"stage": "loaded_tracklets", "n_tracklets": len(records), "emb_dim": int(emb.shape[1])}), flush=True)
    if args.feature_npz:
        emb = _load_feature_npz(
            args.feature_npz,
            records,
            emb,
            concat_db=bool(args.concat_db_embedding),
            db_weight=float(args.db_weight),
            feature_weight=float(args.feature_weight),
        )
        print(json.dumps({"stage": "loaded_feature_npz", "emb_dim": int(emb.shape[1]), "feature_npz": args.feature_npz}), flush=True)
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
    keep_indices = {idx for idx, record in enumerate(records) if int(record.seq) in keep_seqs}
    cfg = ResolveConfig(
        mode="time_agglom",
        theta=float(args.theta),
        top_k=int(args.top_k),
        min_dets=int(args.min_dets),
        exclude_same=str(args.exclude_same),
        temporal_bonus=float(args.temporal_bonus),
        time_window_ms=int(args.time_window_ms),
    )
    base_labels, base_info = _time_agglom_resolve(records, emb, cfg)
    base_pred = _labels_to_seq_map(records, base_labels, keep_seqs=keep_seqs)
    base_metrics = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", **base_info, **base_metrics}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    total = len(_parse_ints(args.attach_source_max_sizes)) * len(_parse_floats(args.attach_thresholds)) * len(_parse_floats(args.attach_margins))
    progress = 0
    for source_max_size in _parse_ints(args.attach_source_max_sizes):
        for threshold in _parse_floats(args.attach_thresholds):
            for margin in _parse_floats(args.attach_margins):
                progress += 1
                labels, info = _attach(records, emb, base_labels, keep_indices, args, source_max_size, threshold, margin)
                pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                metrics = _pair_metrics([record.seq for record in records], pred_by_seq, gt_by_seq, weight_by_seq)
                row = {
                    "progress": int(progress),
                    "total": int(total),
                    **asdict(cfg),
                    **info,
                    **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
                    **metrics,
                }
                rows.append(row)
                print(json.dumps({"stage": "config", **{k: row[k] for k in ["progress", "total", "attach_source_max_size", "attach_threshold", "attach_margin", "attach_accepted", "tracklet_pair_f1", "tracklet_pair_precision", "tracklet_pair_recall"]}}, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row.get(args.sort_key, 0.0)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _attach(
            records,
            emb,
            base_labels,
            keep_indices,
            args,
            int(row["attach_source_max_size"]),
            float(row["attach_threshold"]),
            float(row["attach_margin"]),
        )
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full, "config": {k: row[k] for k in ["attach_source_max_size", "attach_threshold", "attach_margin"]}}, sort_keys=True), flush=True)

    result = {
        "dataset": args.dataset,
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_config": asdict(cfg),
        "base_info": base_info,
        "base_pair_metrics": base_metrics,
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "n_configs": int(total),
        "sort_key": args.sort_key,
        "top": rows[: max(20, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
