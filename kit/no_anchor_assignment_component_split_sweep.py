#!/usr/bin/env python
"""Split components from an existing no-anchor assignment CSV.

The input assignment is treated as the current identity-resolution output.  This
script only reclusters large delivered components with stricter no-GT visual
evidence, then evaluates the tradeoff.  GT is used only for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_component_split_sweep import _parse_floats, _parse_ints, _sparse_topk_affinity
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
        _time_support_matrix,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_component_split_sweep import _parse_floats, _parse_ints, _sparse_topk_affinity
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
        _time_support_matrix,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


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


def _labels_from_assignment(records, pred_by_seq: dict[int, int]) -> tuple[np.ndarray, set[int], dict[int, int]]:
    raw_to_local: dict[int, int] = {}
    labels = np.full(len(records), -1, dtype=np.int64)
    keep_indices: set[int] = set()
    for idx, record in enumerate(records):
        seq = int(record.seq)
        if seq not in pred_by_seq:
            continue
        raw = int(pred_by_seq[seq])
        if raw not in raw_to_local:
            raw_to_local[raw] = len(raw_to_local)
        labels[idx] = raw_to_local[raw]
        keep_indices.add(idx)
    next_label = len(raw_to_local)
    for idx in range(len(records)):
        if labels[idx] < 0:
            labels[idx] = next_label
            next_label += 1
    return labels, keep_indices, raw_to_local


def _component_indices(labels: np.ndarray, keep_indices: set[int]) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(keep_indices):
        groups[int(labels[idx])].append(int(idx))
    return list(groups.values())


def _split_labels(
    records,
    emb: np.ndarray,
    base_labels: np.ndarray,
    keep_indices: set[int],
    *,
    split_min_size: int,
    split_max_size: int,
    split_theta: float,
    split_top_k: int,
    split_temporal_bonus: float,
    split_time_window_ms: int,
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.full(len(base_labels), -1, dtype=np.int64)
    x_all = _l2n(emb.astype(np.float32))
    next_label = 0
    split_components = 0
    produced_parts = 0
    singleton_parts = 0
    for indices in _component_indices(base_labels, keep_indices):
        should_split = int(split_min_size) <= len(indices) <= int(split_max_size)
        if not should_split:
            for idx in indices:
                out[idx] = next_label
            next_label += 1
            continue
        x = x_all[np.asarray(indices, dtype=np.int64)]
        S = (x @ x.T).astype(np.float32)
        if float(split_temporal_bonus) != 0.0:
            S += float(split_temporal_bonus) * _time_support_matrix(records, indices, int(split_time_window_ms))
        np.fill_diagonal(S, -2.0)
        if int(split_top_k) > 0:
            A = _sparse_topk_affinity(S, int(split_top_k))
        else:
            A = S.copy()
            np.fill_diagonal(A, 1.0)
        D = 1.0 - A
        np.clip(D, 0.0, None, out=D)
        clustered = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - float(split_theta)),
            metric="precomputed",
            linkage="average",
        ).fit_predict(D)
        _, clustered = np.unique(clustered, return_inverse=True)
        n_parts = int(clustered.max()) + 1 if clustered.size else 0
        if n_parts > 1:
            split_components += 1
            produced_parts += n_parts
            singleton_parts += sum(1 for label in range(n_parts) if int(np.count_nonzero(clustered == label)) == 1)
        for part_label in range(n_parts):
            for local_pos in np.where(clustered == part_label)[0].tolist():
                out[indices[int(local_pos)]] = next_label
            next_label += 1
    for idx in range(len(base_labels)):
        if out[idx] < 0:
            out[idx] = next_label
            next_label += 1
    return out, {
        "split_min_size": int(split_min_size),
        "split_max_size": int(split_max_size),
        "split_theta": float(split_theta),
        "split_top_k": int(split_top_k),
        "split_temporal_bonus": float(split_temporal_bonus),
        "split_time_window_ms": int(split_time_window_ms),
        "split_components": int(split_components),
        "split_produced_parts": int(produced_parts),
        "split_singleton_parts": int(singleton_parts),
        "components": int(len(set(out[i] for i in keep_indices))),
        "largest_component": int(max(Counter(out[i] for i in keep_indices).values(), default=0)),
        "uses_ground_truth": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dbname", default="gallery_ds1")
    ap.add_argument("--role", default="resolve")
    ap.add_argument("--assignment-csv", required=True)
    ap.add_argument("--pred-col", default="predicted_global_id")
    ap.add_argument("--feature-npz", default=None)
    ap.add_argument("--concat-db-embedding", action="store_true")
    ap.add_argument("--db-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--split-min-sizes", default="96,128,160,192")
    ap.add_argument("--split-max-sizes", default="1000000")
    ap.add_argument("--split-thetas", default="0.04,0.06,0.08,0.10,0.12")
    ap.add_argument("--split-top-ks", default="0,15,30")
    ap.add_argument("--split-temporal-bonuses", default="0.0,0.005")
    ap.add_argument("--split-time-window-ms", type=int, default=1000)
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
    ap.add_argument("--eval-cache", default="/mnt/localssd/vlincs_reid_runs/ds1_eval_labels_iou050_gallery_ds1.npz")
    ap.add_argument("--full-top-n", type=int, default=0)
    ap.add_argument("--assignments-out", default="")
    ap.add_argument("--assignment-offset", type=int, default=50_000_000)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    pred_input = _load_assignment_labels(args.assignment_csv, args.pred_col)
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
    keep_seqs = {int(seq) for seq in keep_seqs if int(seq) in pred_input}
    base_labels, keep_indices, raw_to_local = _labels_from_assignment(records, pred_input)
    seqs = [int(record.seq) for record in records]
    base_pred = _labels_to_seq_map(records, base_labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
    base_pair = _pair_metrics(seqs, base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", "components": len(raw_to_local), **base_pair}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    labels_by_rank: dict[int, np.ndarray] = {}
    for split_min_size in _parse_ints(args.split_min_sizes):
        for split_max_size in _parse_ints(args.split_max_sizes):
            for split_theta in _parse_floats(args.split_thetas):
                for split_top_k in _parse_ints(args.split_top_ks):
                    for split_temporal_bonus in _parse_floats(args.split_temporal_bonuses):
                        labels, info = _split_labels(
                            records,
                            emb,
                            base_labels,
                            keep_indices,
                            split_min_size=split_min_size,
                            split_max_size=split_max_size,
                            split_theta=split_theta,
                            split_top_k=split_top_k,
                            split_temporal_bonus=split_temporal_bonus,
                            split_time_window_ms=int(args.split_time_window_ms),
                        )
                        pred = _labels_to_seq_map(records, labels, offset=int(args.assignment_offset), keep_seqs=keep_seqs)
                        pair = _pair_metrics(seqs, pred, gt_by_seq, weight_by_seq)
                        rows.append(
                            {
                                "mode": "assignment_component_split",
                                **info,
                                **pair,
                                "uses_anchors": False,
                                "uses_gt_for_training_or_anchors": False,
                                "uses_gt_for_evaluation_only": True,
                            }
                        )

    rows.sort(
        key=lambda row: (
            float(row["tracklet_pair_f1"]),
            float(row["tracklet_pair_precision"]),
            float(row["tracklet_pair_recall"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _split_labels(
            records,
            emb,
            base_labels,
            keep_indices,
            split_min_size=int(row["split_min_size"]),
            split_max_size=int(row["split_max_size"]),
            split_theta=float(row["split_theta"]),
            split_top_k=int(row["split_top_k"]),
            split_temporal_bonus=float(row["split_temporal_bonus"]),
            split_time_window_ms=int(row["split_time_window_ms"]),
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
            labels, _info = _split_labels(
                records,
                emb,
                base_labels,
                keep_indices,
                split_min_size=int(row["split_min_size"]),
                split_max_size=int(row["split_max_size"]),
                split_theta=float(row["split_theta"]),
                split_top_k=int(row["split_top_k"]),
                split_temporal_bonus=float(row["split_temporal_bonus"]),
                split_time_window_ms=int(row["split_time_window_ms"]),
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
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "assignment_info": assignment_info,
        "top": rows[: max(50, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"base": base_pair, "best": rows[0] if rows else None, "json": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
