#!/usr/bin/env python
"""No-anchor Louvain community detection over tracklet evidence graphs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _group_codes,
        _knn_sparse_affinity,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_support_matrix,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
        _connect,
        _group_codes,
        _knn_sparse_affinity,
        _labels_to_seq_map,
        _load_eval_label_cache,
        _load_feature_npz,
        _load_predictions,
        _load_tracklets,
        _output_keep_seqs,
        _pair_metrics,
        _score_full,
        _time_support_matrix,
        _with_detection_endpoints,
    )


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _louvain_labels(
    records,
    emb: np.ndarray,
    *,
    top_k: int,
    min_dets: int,
    exclude_same: str,
    temporal_bonus: float,
    time_window_ms: int,
    edge_floor: float,
    resolution: float,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    keep = [i for i, record in enumerate(records) if int(record.n_dets) >= int(min_dets)]
    labels = np.full(len(records), -1, dtype=np.int64)
    edge_count = 0
    positive_time_edges = 0
    if len(keep) >= 2:
        group_codes = _group_codes(records, exclude_same, keep)
        x = _l2n(emb[np.asarray(keep)].astype(np.float32))
        S = (x @ x.T).astype(np.float32)
        support = _time_support_matrix(records, keep, int(time_window_ms))
        if exclude_same != "none":
            same = group_codes[:, None] == group_codes[None, :]
            support[same] = 0.0
            S[same] = -2.0
        S += float(temporal_bonus) * support
        np.fill_diagonal(S, -2.0)
        positive_time_edges = int(np.count_nonzero(np.triu(support > 0, k=1)))
        A, n_edges = _knn_sparse_affinity(S, int(top_k))
        edge_count = int(n_edges)
        G = nx.Graph()
        G.add_nodes_from(range(len(keep)))
        rows, cols = np.where(np.triu(A, k=1) > float(edge_floor))
        edge_after_floor = int(len(rows))
        for r, c in zip(rows.tolist(), cols.tolist()):
            score = float(A[r, c])
            if score <= float(edge_floor):
                continue
            G.add_edge(int(r), int(c), weight=max(score - float(edge_floor), 1.0e-6))
        if G.number_of_edges():
            communities = nx.community.louvain_communities(
                G,
                weight="weight",
                resolution=float(resolution),
                seed=int(random_state),
            )
        else:
            communities = [{idx} for idx in range(len(keep))]
        next_label = 0
        for community in communities:
            for local_idx in community:
                labels[keep[int(local_idx)]] = next_label
            next_label += 1
    else:
        next_label = 0
        edge_after_floor = 0
    for i in range(len(records)):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return labels, {
        "candidate_edges": int(edge_count),
        "louvain_edges_after_floor": int(edge_after_floor),
        "time_candidate_pairs": int(positive_time_edges),
        "components": int(next_label),
        "largest_component": int(max(Counter(labels.tolist()).values(), default=0)),
        "n_tracklets": int(len(records)),
        "n_clustered": int(len(keep)),
        "n_singleton": int(len(records) - len(keep)),
        "uses_ground_truth": False,
    }


def _write_assignments(
    path: str,
    records,
    labels: np.ndarray,
    *,
    keep_seqs: set[int],
    offset: int,
) -> dict[str, object]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(int(label) for record, label in zip(records, labels) if int(record.seq) in keep_seqs)
    rows = []
    for record, label in zip(records, labels):
        if int(record.seq) not in keep_seqs:
            continue
        label = int(label)
        size = int(counts[label])
        confidence = 0.15 if size == 1 else min(0.85, 0.30 + 0.02 * min(size, 20))
        rows.append(
            {
                "seq": int(record.seq),
                "tracklet_key": record.tracklet_key,
                "video": record.video,
                "camera": record.camera,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "n_dets": int(record.n_dets),
                "avg_conf": round(float(record.avg_conf), 6),
                "predicted_global_id": int(offset + label),
                "component_label": label,
                "component_size": size,
                "prediction_confidence": round(float(confidence), 6),
                "decision_status": "forced_singleton" if size == 1 else "forced_component",
                "component_internal_edges": 0,
                "component_internal_prob_median": 0.0,
                "component_internal_score_median": 0.0,
                "component_external_prob_max": 0.0,
                "component_margin_prob": 0.0,
            }
        )
    fieldnames = [
        "seq",
        "tracklet_key",
        "video",
        "camera",
        "start_frame",
        "end_frame",
        "n_dets",
        "avg_conf",
        "predicted_global_id",
        "component_label",
        "component_size",
        "prediction_confidence",
        "decision_status",
        "component_internal_edges",
        "component_internal_prob_median",
        "component_internal_score_median",
        "component_external_prob_max",
        "component_margin_prob",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    status_counts = Counter(str(row["decision_status"]) for row in rows)
    return {
        "assignments_out": str(path),
        "assignment_rows": int(len(rows)),
        "assignment_components": int(len(counts)),
        "largest_assignment_component": int(max(counts.values(), default=0)),
        "assignment_status_counts": dict(status_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--nfc-k1", type=int, default=0, help="enable Pose2ID-style no-anchor neighbor feature centralization")
    ap.add_argument("--nfc-k2", type=int, default=2)
    ap.add_argument("--nfc-eta", type=float, default=1.0)
    ap.add_argument("--nfc-exclude-same", default="none", choices=["none", "camera", "stream", "video"])
    ap.add_argument("--top-ks", default="15,25,35")
    ap.add_argument("--min-dets", type=int, default=10)
    ap.add_argument("--exclude-same", default="camera")
    ap.add_argument("--temporal-bonus", type=float, default=0.005)
    ap.add_argument("--time-window-ms", type=int, default=1000)
    ap.add_argument("--edge-floors", default="0.0,0.01,0.02,0.04")
    ap.add_argument("--resolutions", default="0.6,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--output-min-dets", type=int, default=1)
    ap.add_argument("--output-min-conf", type=float, default=0.0)
    ap.add_argument("--output-min-area", type=float, default=0.0)
    ap.add_argument("--output-min-quality", type=float, default=-1.0e9)
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
    ap.add_argument("--output-min-area-by-video", default="")
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default=None, help="optional CSV of best-config seq -> predicted_global_id assignments")
    ap.add_argument("--assignment-offset", type=int, default=40_000_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--random-state", type=int, default=17)
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
    nfc_info = None
    if int(args.nfc_k1) > 0:
        from vlincs_gallery.feature_centralization import neighbor_feature_centralization

        indices = list(range(len(records)))
        group_codes = _group_codes(records, str(args.nfc_exclude_same), indices)
        emb, nfc_info = neighbor_feature_centralization(
            emb,
            k1=int(args.nfc_k1),
            k2=int(args.nfc_k2),
            eta=float(args.nfc_eta),
            group_codes=group_codes,
            exclude_same_group=str(args.nfc_exclude_same) != "none",
        )
        print(json.dumps({"stage": "nfc", **nfc_info.__dict__, "exclude_same": str(args.nfc_exclude_same)}), flush=True)
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

    seqs = [int(record.seq) for record in records]
    rows = []
    labels_cache: dict[tuple[int, float, float], np.ndarray] = {}
    total = len(_parse_ints(args.top_ks)) * len(_parse_floats(args.edge_floors)) * len(_parse_floats(args.resolutions))
    progress = 0
    for top_k in _parse_ints(args.top_ks):
        for edge_floor in _parse_floats(args.edge_floors):
            for resolution in _parse_floats(args.resolutions):
                progress += 1
                labels, info = _louvain_labels(
                    records,
                    emb,
                    top_k=top_k,
                    min_dets=int(args.min_dets),
                    exclude_same=str(args.exclude_same),
                    temporal_bonus=float(args.temporal_bonus),
                    time_window_ms=int(args.time_window_ms),
                    edge_floor=float(edge_floor),
                    resolution=float(resolution),
                    random_state=int(args.random_state),
                )
                key = (int(top_k), float(edge_floor), float(resolution))
                labels_cache[key] = labels
                pred_by_seq = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                metrics = _pair_metrics(seqs, pred_by_seq, gt_by_seq, weight_by_seq)
                row = {
                    "rank_input_order": int(progress),
                    "mode": "louvain",
                    "top_k": int(top_k),
                    "edge_floor": float(edge_floor),
                    "resolution": float(resolution),
                    "min_dets": int(args.min_dets),
                    "exclude_same": str(args.exclude_same),
                    "temporal_bonus": float(args.temporal_bonus),
                    "time_window_ms": int(args.time_window_ms),
                    **info,
                    **{k: v for k, v in output_info.items() if not isinstance(v, dict)},
                    **metrics,
                }
                rows.append(row)
                print(json.dumps({"progress": progress, "total": total, **metrics, "top_k": top_k, "edge_floor": edge_floor, "resolution": resolution}, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row.get("tracklet_pair_f1", 0.0)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        key = (int(row["top_k"]), float(row["edge_floor"]), float(row["resolution"]))
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels_cache[key], keep_seqs=keep_seqs))
        row.update({f"full_{name}": value for name, value in full.items() if name != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"full_rank": rank, "full": full, "config": key}, sort_keys=True), flush=True)

    assignment_info = None
    if args.assignments_out and rows:
        best_key = (int(rows[0]["top_k"]), float(rows[0]["edge_floor"]), float(rows[0]["resolution"]))
        assignment_info = _write_assignments(
            args.assignments_out,
            records,
            labels_cache[best_key],
            keep_seqs=keep_seqs,
            offset=int(args.assignment_offset),
        )
        rows[0].update(assignment_info)

    result = {
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "nfc": nfc_info.__dict__ if nfc_info is not None else None,
        "nfc_exclude_same": str(args.nfc_exclude_same),
        "eval_stats": eval_stats,
        "output_admission": output_info,
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
        "assignment_info": assignment_info,
        "n_configs": int(len(rows)),
        "top": rows[: max(20, int(args.full_top_n))],
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()) if rows else [])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(json.dumps({"json": str(out), "best": rows[0] if rows else None}, sort_keys=True))


if __name__ == "__main__":
    main()
