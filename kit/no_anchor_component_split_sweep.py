#!/usr/bin/env python
"""Split suspicious large no-anchor components and evaluate the tradeoff.

The base resolver is the current label-free time-aware agglomeration.  This
script then re-clusters only large components with a stricter no-GT visual
threshold.  Ground truth is loaded only after prediction for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vlincs_gallery.eval.score import load_ds1_gt_by_video

try:
    from kit.no_anchor_resolve_sweep import (
        ResolveConfig,
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
        _time_support_matrix,
        _with_detection_endpoints,
    )
except ModuleNotFoundError:
    from no_anchor_resolve_sweep import (
        ResolveConfig,
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
        _time_support_matrix,
        _with_detection_endpoints,
    )


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-9)


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _component_indices(labels: np.ndarray, keep_indices: set[int]) -> list[list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        if idx in keep_indices:
            by_label[int(label)].append(int(idx))
    return [indices for _label, indices in sorted(by_label.items(), key=lambda item: min(item[1]))]


def _sparse_topk_affinity(S: np.ndarray, top_k: int) -> np.ndarray:
    n = int(S.shape[0])
    if n <= 1:
        return np.ones((n, n), dtype=np.float32)
    k = min(max(int(top_k), 1), n - 1)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        top = np.argpartition(-S[i], k - 1)[:k]
        for j in top.tolist():
            if i == int(j):
                continue
            score = float(S[i, int(j)])
            if score > A[i, int(j)]:
                A[i, int(j)] = score
                A[int(j), i] = max(float(A[int(j), i]), score)
    np.fill_diagonal(A, 1.0)
    return A


def _split_labels(
    records,
    emb: np.ndarray,
    base_labels: np.ndarray,
    keep_indices: set[int],
    *,
    split_min_size: int,
    split_theta: float,
    split_top_k: int,
    split_temporal_bonus: float,
    split_time_window_ms: int,
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.full(len(base_labels), -1, dtype=np.int64)
    next_label = 0
    split_components = 0
    produced_parts = 0
    singleton_parts = 0
    x_all = _l2n(emb.astype(np.float32))
    for indices in _component_indices(base_labels, keep_indices):
        if len(indices) < int(split_min_size):
            for idx in indices:
                out[idx] = next_label
            next_label += 1
            continue
        x = x_all[np.asarray(indices, dtype=np.int64)]
        S = (x @ x.T).astype(np.float32)
        if float(split_temporal_bonus) != 0.0:
            support = _time_support_matrix(records, indices, int(split_time_window_ms))
            S += float(split_temporal_bonus) * support
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
        "split_theta": float(split_theta),
        "split_top_k": int(split_top_k),
        "split_temporal_bonus": float(split_temporal_bonus),
        "split_time_window_ms": int(split_time_window_ms),
        "split_components": int(split_components),
        "split_produced_parts": int(produced_parts),
        "split_singleton_parts": int(singleton_parts),
        "components": int(len(set(out.tolist()))),
        "largest_component": int(max(Counter(out.tolist()).values(), default=0)),
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
    ap.add_argument("--split-min-sizes", default="32,48,64,96,128")
    ap.add_argument("--split-thetas", default="0.016,0.018,0.020,0.024,0.030,0.040")
    ap.add_argument("--split-top-ks", default="0,15,30")
    ap.add_argument("--split-temporal-bonuses", default="0.0,0.005")
    ap.add_argument("--split-time-window-ms", type=int, default=1000)
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
    base_pair = _pair_metrics([record.seq for record in records], base_pred, gt_by_seq, weight_by_seq)
    print(json.dumps({"stage": "base", **base_info, **base_pair}, sort_keys=True), flush=True)

    rows: list[dict[str, object]] = []
    total = (
        len(_parse_ints(args.split_min_sizes))
        * len(_parse_floats(args.split_thetas))
        * len(_parse_ints(args.split_top_ks))
        * len(_parse_floats(args.split_temporal_bonuses))
    )
    progress = 0
    for split_min_size in _parse_ints(args.split_min_sizes):
        for split_theta in _parse_floats(args.split_thetas):
            for split_top_k in _parse_ints(args.split_top_ks):
                for split_temporal_bonus in _parse_floats(args.split_temporal_bonuses):
                    progress += 1
                    labels, info = _split_labels(
                        records,
                        emb,
                        base_labels,
                        keep_indices,
                        split_min_size=split_min_size,
                        split_theta=split_theta,
                        split_top_k=split_top_k,
                        split_temporal_bonus=split_temporal_bonus,
                        split_time_window_ms=int(args.split_time_window_ms),
                    )
                    pred = _labels_to_seq_map(records, labels, keep_seqs=keep_seqs)
                    pair = _pair_metrics([record.seq for record in records], pred, gt_by_seq, weight_by_seq)
                    row = {
                        "progress": int(progress),
                        "total": int(total),
                        **asdict(cfg),
                        **info,
                        **{key: value for key, value in output_info.items() if not isinstance(value, dict)},
                        **pair,
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": "config",
                                "progress": progress,
                                "total": total,
                                "split_min_size": split_min_size,
                                "split_theta": split_theta,
                                "split_top_k": split_top_k,
                                "split_components": info["split_components"],
                                "tracklet_pair_f1": pair["tracklet_pair_f1"],
                                "tracklet_pair_precision": pair["tracklet_pair_precision"],
                                "tracklet_pair_recall": pair["tracklet_pair_recall"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    rows.sort(key=lambda row: float(row.get(args.sort_key, 0.0)), reverse=True)
    for rank, row in enumerate(rows[: max(int(args.full_top_n), 0)], start=1):
        labels, _info = _split_labels(
            records,
            emb,
            base_labels,
            keep_indices,
            split_min_size=int(row["split_min_size"]),
            split_theta=float(row["split_theta"]),
            split_top_k=int(row["split_top_k"]),
            split_temporal_bonus=float(row["split_temporal_bonus"]),
            split_time_window_ms=int(row["split_time_window_ms"]),
        )
        full = _score_full(pred_by_video, gt_by_video, _labels_to_seq_map(records, labels, keep_seqs=keep_seqs))
        row.update({f"full_{key}": value for key, value in full.items() if key != "per_video"})
        row["full_per_video"] = full["per_video"]
        row["full_rank"] = int(rank)
        print(json.dumps({"stage": "full", "rank": rank, "full": full}, sort_keys=True), flush=True)

    result = {
        "dbname": args.dbname,
        "role": args.role,
        "feature_npz": args.feature_npz,
        "concat_db_embedding": bool(args.concat_db_embedding),
        "db_weight": float(args.db_weight),
        "feature_weight": float(args.feature_weight),
        "base_config": asdict(cfg),
        "base_info": base_info,
        "base_pair_metrics": base_pair,
        "output_admission": output_info,
        "eval_stats": eval_stats,
        "n_configs": int(total),
        "sort_key": str(args.sort_key),
        "top": rows[: max(50, int(args.full_top_n))],
        "uses_anchors": False,
        "uses_gt_for_training_or_anchors": False,
        "uses_gt_for_evaluation_only": True,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv:
        _write_csv(args.csv, rows)
    print(json.dumps({"json": str(out), "base": base_pair, "best": rows[0] if rows else None}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
